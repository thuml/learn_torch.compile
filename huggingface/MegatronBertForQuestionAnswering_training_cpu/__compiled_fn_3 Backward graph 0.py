from __future__ import annotations



def forward(self, primals_4: "f32[1024]", primals_14: "f32[1024]", primals_20: "f32[1024]", primals_30: "f32[1024]", primals_36: "f32[1024]", primals_46: "f32[1024]", primals_52: "f32[1024]", primals_62: "f32[1024]", primals_68: "f32[1024]", primals_78: "f32[1024]", primals_84: "f32[1024]", primals_94: "f32[1024]", primals_100: "f32[1024]", primals_110: "f32[1024]", primals_116: "f32[1024]", primals_126: "f32[1024]", primals_132: "f32[1024]", primals_142: "f32[1024]", primals_148: "f32[1024]", primals_158: "f32[1024]", primals_164: "f32[1024]", primals_174: "f32[1024]", primals_180: "f32[1024]", primals_190: "f32[1024]", primals_196: "f32[1024]", primals_206: "f32[1024]", primals_212: "f32[1024]", primals_222: "f32[1024]", primals_228: "f32[1024]", primals_238: "f32[1024]", primals_244: "f32[1024]", primals_254: "f32[1024]", primals_260: "f32[1024]", primals_270: "f32[1024]", primals_276: "f32[1024]", primals_286: "f32[1024]", primals_292: "f32[1024]", primals_302: "f32[1024]", primals_308: "f32[1024]", primals_318: "f32[1024]", primals_324: "f32[1024]", primals_334: "f32[1024]", primals_340: "f32[1024]", primals_350: "f32[1024]", primals_356: "f32[1024]", primals_366: "f32[1024]", primals_372: "f32[1024]", primals_382: "f32[1024]", primals_388: "f32[1024]", primals_393: "i64[1, 512]", full_default: "i64[1, 512]", slice_3: "i64[1, 512]", getitem_1: "b8[1, 512, 1024]", mul_1: "f32[1, 512, 1024]", view: "f32[512, 1024]", getitem_293: "b8[1, 16, 512, 512]", permute_default_139: "f32[16, 512, 512]", permute_default_140: "f32[16, 64, 512]", alias_default_47: "f32[1, 16, 512, 512]", permute_default_141: "f32[16, 64, 512]", permute_default_142: "f32[16, 512, 64]", view_16: "f32[512, 1024]", getitem_7: "b8[1, 512, 1024]", mul_3: "f32[1, 512, 1024]", view_18: "f32[512, 1024]", addmm_4: "f32[512, 4096]", view_20: "f32[512, 4096]", getitem_11: "b8[1, 512, 1024]", mul_8: "f32[1, 512, 1024]", view_22: "f32[512, 1024]", getitem_291: "b8[1, 16, 512, 512]", permute_default_133: "f32[16, 512, 512]", permute_default_134: "f32[16, 64, 512]", alias_default_45: "f32[1, 16, 512, 512]", permute_default_135: "f32[16, 64, 512]", permute_default_136: "f32[16, 512, 64]", view_38: "f32[512, 1024]", getitem_17: "b8[1, 512, 1024]", mul_10: "f32[1, 512, 1024]", view_40: "f32[512, 1024]", addmm_10: "f32[512, 4096]", view_42: "f32[512, 4096]", getitem_21: "b8[1, 512, 1024]", mul_15: "f32[1, 512, 1024]", view_44: "f32[512, 1024]", getitem_289: "b8[1, 16, 512, 512]", permute_default_127: "f32[16, 512, 512]", permute_default_128: "f32[16, 64, 512]", alias_default_43: "f32[1, 16, 512, 512]", permute_default_129: "f32[16, 64, 512]", permute_default_130: "f32[16, 512, 64]", view_60: "f32[512, 1024]", getitem_27: "b8[1, 512, 1024]", mul_17: "f32[1, 512, 1024]", view_62: "f32[512, 1024]", addmm_16: "f32[512, 4096]", view_64: "f32[512, 4096]", getitem_31: "b8[1, 512, 1024]", mul_22: "f32[1, 512, 1024]", view_66: "f32[512, 1024]", getitem_287: "b8[1, 16, 512, 512]", permute_default_121: "f32[16, 512, 512]", permute_default_122: "f32[16, 64, 512]", alias_default_41: "f32[1, 16, 512, 512]", permute_default_123: "f32[16, 64, 512]", permute_default_124: "f32[16, 512, 64]", view_82: "f32[512, 1024]", getitem_37: "b8[1, 512, 1024]", mul_24: "f32[1, 512, 1024]", view_84: "f32[512, 1024]", addmm_22: "f32[512, 4096]", view_86: "f32[512, 4096]", getitem_41: "b8[1, 512, 1024]", mul_29: "f32[1, 512, 1024]", view_88: "f32[512, 1024]", getitem_285: "b8[1, 16, 512, 512]", permute_default_115: "f32[16, 512, 512]", permute_default_116: "f32[16, 64, 512]", alias_default_39: "f32[1, 16, 512, 512]", permute_default_117: "f32[16, 64, 512]", permute_default_118: "f32[16, 512, 64]", view_104: "f32[512, 1024]", getitem_47: "b8[1, 512, 1024]", mul_31: "f32[1, 512, 1024]", view_106: "f32[512, 1024]", addmm_28: "f32[512, 4096]", view_108: "f32[512, 4096]", getitem_51: "b8[1, 512, 1024]", mul_36: "f32[1, 512, 1024]", view_110: "f32[512, 1024]", getitem_283: "b8[1, 16, 512, 512]", permute_default_109: "f32[16, 512, 512]", permute_default_110: "f32[16, 64, 512]", alias_default_37: "f32[1, 16, 512, 512]", permute_default_111: "f32[16, 64, 512]", permute_default_112: "f32[16, 512, 64]", view_126: "f32[512, 1024]", getitem_57: "b8[1, 512, 1024]", mul_38: "f32[1, 512, 1024]", view_128: "f32[512, 1024]", addmm_34: "f32[512, 4096]", view_130: "f32[512, 4096]", getitem_61: "b8[1, 512, 1024]", mul_43: "f32[1, 512, 1024]", view_132: "f32[512, 1024]", getitem_281: "b8[1, 16, 512, 512]", permute_default_103: "f32[16, 512, 512]", permute_default_104: "f32[16, 64, 512]", alias_default_35: "f32[1, 16, 512, 512]", permute_default_105: "f32[16, 64, 512]", permute_default_106: "f32[16, 512, 64]", view_148: "f32[512, 1024]", getitem_67: "b8[1, 512, 1024]", mul_45: "f32[1, 512, 1024]", view_150: "f32[512, 1024]", addmm_40: "f32[512, 4096]", view_152: "f32[512, 4096]", getitem_71: "b8[1, 512, 1024]", mul_50: "f32[1, 512, 1024]", view_154: "f32[512, 1024]", getitem_279: "b8[1, 16, 512, 512]", permute_default_97: "f32[16, 512, 512]", permute_default_98: "f32[16, 64, 512]", alias_default_33: "f32[1, 16, 512, 512]", permute_default_99: "f32[16, 64, 512]", permute_default_100: "f32[16, 512, 64]", view_170: "f32[512, 1024]", getitem_77: "b8[1, 512, 1024]", mul_52: "f32[1, 512, 1024]", view_172: "f32[512, 1024]", addmm_46: "f32[512, 4096]", view_174: "f32[512, 4096]", getitem_81: "b8[1, 512, 1024]", mul_57: "f32[1, 512, 1024]", view_176: "f32[512, 1024]", getitem_277: "b8[1, 16, 512, 512]", permute_default_91: "f32[16, 512, 512]", permute_default_92: "f32[16, 64, 512]", alias_default_31: "f32[1, 16, 512, 512]", permute_default_93: "f32[16, 64, 512]", permute_default_94: "f32[16, 512, 64]", view_192: "f32[512, 1024]", getitem_87: "b8[1, 512, 1024]", mul_59: "f32[1, 512, 1024]", view_194: "f32[512, 1024]", addmm_52: "f32[512, 4096]", view_196: "f32[512, 4096]", getitem_91: "b8[1, 512, 1024]", mul_64: "f32[1, 512, 1024]", view_198: "f32[512, 1024]", getitem_275: "b8[1, 16, 512, 512]", permute_default_85: "f32[16, 512, 512]", permute_default_86: "f32[16, 64, 512]", alias_default_29: "f32[1, 16, 512, 512]", permute_default_87: "f32[16, 64, 512]", permute_default_88: "f32[16, 512, 64]", view_214: "f32[512, 1024]", getitem_97: "b8[1, 512, 1024]", mul_66: "f32[1, 512, 1024]", view_216: "f32[512, 1024]", addmm_58: "f32[512, 4096]", view_218: "f32[512, 4096]", getitem_101: "b8[1, 512, 1024]", mul_71: "f32[1, 512, 1024]", view_220: "f32[512, 1024]", getitem_273: "b8[1, 16, 512, 512]", permute_default_79: "f32[16, 512, 512]", permute_default_80: "f32[16, 64, 512]", alias_default_27: "f32[1, 16, 512, 512]", permute_default_81: "f32[16, 64, 512]", permute_default_82: "f32[16, 512, 64]", view_236: "f32[512, 1024]", getitem_107: "b8[1, 512, 1024]", mul_73: "f32[1, 512, 1024]", view_238: "f32[512, 1024]", addmm_64: "f32[512, 4096]", view_240: "f32[512, 4096]", getitem_111: "b8[1, 512, 1024]", mul_78: "f32[1, 512, 1024]", view_242: "f32[512, 1024]", getitem_271: "b8[1, 16, 512, 512]", permute_default_73: "f32[16, 512, 512]", permute_default_74: "f32[16, 64, 512]", alias_default_25: "f32[1, 16, 512, 512]", permute_default_75: "f32[16, 64, 512]", permute_default_76: "f32[16, 512, 64]", view_258: "f32[512, 1024]", getitem_117: "b8[1, 512, 1024]", mul_80: "f32[1, 512, 1024]", view_260: "f32[512, 1024]", addmm_70: "f32[512, 4096]", view_262: "f32[512, 4096]", getitem_121: "b8[1, 512, 1024]", mul_85: "f32[1, 512, 1024]", view_264: "f32[512, 1024]", getitem_269: "b8[1, 16, 512, 512]", permute_default_67: "f32[16, 512, 512]", permute_default_68: "f32[16, 64, 512]", alias_default_23: "f32[1, 16, 512, 512]", permute_default_69: "f32[16, 64, 512]", permute_default_70: "f32[16, 512, 64]", view_280: "f32[512, 1024]", getitem_127: "b8[1, 512, 1024]", mul_87: "f32[1, 512, 1024]", view_282: "f32[512, 1024]", addmm_76: "f32[512, 4096]", view_284: "f32[512, 4096]", getitem_131: "b8[1, 512, 1024]", mul_92: "f32[1, 512, 1024]", view_286: "f32[512, 1024]", getitem_267: "b8[1, 16, 512, 512]", permute_default_61: "f32[16, 512, 512]", permute_default_62: "f32[16, 64, 512]", alias_default_21: "f32[1, 16, 512, 512]", permute_default_63: "f32[16, 64, 512]", permute_default_64: "f32[16, 512, 64]", view_302: "f32[512, 1024]", getitem_137: "b8[1, 512, 1024]", mul_94: "f32[1, 512, 1024]", view_304: "f32[512, 1024]", addmm_82: "f32[512, 4096]", view_306: "f32[512, 4096]", getitem_141: "b8[1, 512, 1024]", mul_99: "f32[1, 512, 1024]", view_308: "f32[512, 1024]", getitem_265: "b8[1, 16, 512, 512]", permute_default_55: "f32[16, 512, 512]", permute_default_56: "f32[16, 64, 512]", alias_default_19: "f32[1, 16, 512, 512]", permute_default_57: "f32[16, 64, 512]", permute_default_58: "f32[16, 512, 64]", view_324: "f32[512, 1024]", getitem_147: "b8[1, 512, 1024]", mul_101: "f32[1, 512, 1024]", view_326: "f32[512, 1024]", addmm_88: "f32[512, 4096]", view_328: "f32[512, 4096]", getitem_151: "b8[1, 512, 1024]", mul_106: "f32[1, 512, 1024]", view_330: "f32[512, 1024]", getitem_263: "b8[1, 16, 512, 512]", permute_default_49: "f32[16, 512, 512]", permute_default_50: "f32[16, 64, 512]", alias_default_17: "f32[1, 16, 512, 512]", permute_default_51: "f32[16, 64, 512]", permute_default_52: "f32[16, 512, 64]", view_346: "f32[512, 1024]", getitem_157: "b8[1, 512, 1024]", mul_108: "f32[1, 512, 1024]", view_348: "f32[512, 1024]", addmm_94: "f32[512, 4096]", view_350: "f32[512, 4096]", getitem_161: "b8[1, 512, 1024]", mul_113: "f32[1, 512, 1024]", view_352: "f32[512, 1024]", getitem_261: "b8[1, 16, 512, 512]", permute_default_43: "f32[16, 512, 512]", permute_default_44: "f32[16, 64, 512]", alias_default_15: "f32[1, 16, 512, 512]", permute_default_45: "f32[16, 64, 512]", permute_default_46: "f32[16, 512, 64]", view_368: "f32[512, 1024]", getitem_167: "b8[1, 512, 1024]", mul_115: "f32[1, 512, 1024]", view_370: "f32[512, 1024]", addmm_100: "f32[512, 4096]", view_372: "f32[512, 4096]", getitem_171: "b8[1, 512, 1024]", mul_120: "f32[1, 512, 1024]", view_374: "f32[512, 1024]", getitem_259: "b8[1, 16, 512, 512]", permute_default_37: "f32[16, 512, 512]", permute_default_38: "f32[16, 64, 512]", alias_default_13: "f32[1, 16, 512, 512]", permute_default_39: "f32[16, 64, 512]", permute_default_40: "f32[16, 512, 64]", view_390: "f32[512, 1024]", getitem_177: "b8[1, 512, 1024]", mul_122: "f32[1, 512, 1024]", view_392: "f32[512, 1024]", addmm_106: "f32[512, 4096]", view_394: "f32[512, 4096]", getitem_181: "b8[1, 512, 1024]", mul_127: "f32[1, 512, 1024]", view_396: "f32[512, 1024]", getitem_257: "b8[1, 16, 512, 512]", permute_default_31: "f32[16, 512, 512]", permute_default_32: "f32[16, 64, 512]", alias_default_11: "f32[1, 16, 512, 512]", permute_default_33: "f32[16, 64, 512]", permute_default_34: "f32[16, 512, 64]", view_412: "f32[512, 1024]", getitem_187: "b8[1, 512, 1024]", mul_129: "f32[1, 512, 1024]", view_414: "f32[512, 1024]", addmm_112: "f32[512, 4096]", view_416: "f32[512, 4096]", getitem_191: "b8[1, 512, 1024]", mul_134: "f32[1, 512, 1024]", view_418: "f32[512, 1024]", getitem_255: "b8[1, 16, 512, 512]", permute_default_25: "f32[16, 512, 512]", permute_default_26: "f32[16, 64, 512]", alias_default_9: "f32[1, 16, 512, 512]", permute_default_27: "f32[16, 64, 512]", permute_default_28: "f32[16, 512, 64]", view_434: "f32[512, 1024]", getitem_197: "b8[1, 512, 1024]", mul_136: "f32[1, 512, 1024]", view_436: "f32[512, 1024]", addmm_118: "f32[512, 4096]", view_438: "f32[512, 4096]", getitem_201: "b8[1, 512, 1024]", mul_141: "f32[1, 512, 1024]", view_440: "f32[512, 1024]", getitem_253: "b8[1, 16, 512, 512]", permute_default_19: "f32[16, 512, 512]", permute_default_20: "f32[16, 64, 512]", alias_default_7: "f32[1, 16, 512, 512]", permute_default_21: "f32[16, 64, 512]", permute_default_22: "f32[16, 512, 64]", view_456: "f32[512, 1024]", getitem_207: "b8[1, 512, 1024]", mul_143: "f32[1, 512, 1024]", view_458: "f32[512, 1024]", addmm_124: "f32[512, 4096]", view_460: "f32[512, 4096]", getitem_211: "b8[1, 512, 1024]", mul_148: "f32[1, 512, 1024]", view_462: "f32[512, 1024]", getitem_251: "b8[1, 16, 512, 512]", permute_default_13: "f32[16, 512, 512]", permute_default_14: "f32[16, 64, 512]", alias_default_5: "f32[1, 16, 512, 512]", permute_default_15: "f32[16, 64, 512]", permute_default_16: "f32[16, 512, 64]", view_478: "f32[512, 1024]", getitem_217: "b8[1, 512, 1024]", mul_150: "f32[1, 512, 1024]", view_480: "f32[512, 1024]", addmm_130: "f32[512, 4096]", view_482: "f32[512, 4096]", getitem_221: "b8[1, 512, 1024]", mul_155: "f32[1, 512, 1024]", view_484: "f32[512, 1024]", getitem_249: "b8[1, 16, 512, 512]", permute_default_7: "f32[16, 512, 512]", permute_default_8: "f32[16, 64, 512]", alias_default_3: "f32[1, 16, 512, 512]", permute_default_9: "f32[16, 64, 512]", permute_default_10: "f32[16, 512, 64]", view_500: "f32[512, 1024]", getitem_227: "b8[1, 512, 1024]", mul_157: "f32[1, 512, 1024]", view_502: "f32[512, 1024]", addmm_136: "f32[512, 4096]", view_504: "f32[512, 4096]", getitem_231: "b8[1, 512, 1024]", mul_162: "f32[1, 512, 1024]", view_506: "f32[512, 1024]", getitem_247: "b8[1, 16, 512, 512]", permute_default_1: "f32[16, 512, 512]", permute_default_2: "f32[16, 64, 512]", alias_default_1: "f32[1, 16, 512, 512]", permute_default_3: "f32[16, 64, 512]", permute_default_4: "f32[16, 512, 64]", view_522: "f32[512, 1024]", getitem_237: "b8[1, 512, 1024]", mul_164: "f32[1, 512, 1024]", view_524: "f32[512, 1024]", addmm_142: "f32[512, 4096]", view_526: "f32[512, 4096]", getitem_241: "b8[1, 512, 1024]", mul_169: "f32[1, 512, 1024]", view_528: "f32[512, 1024]", sub_75: "f32[1, 512]", ne: "b8[1]", sub_77: "f32[1, 512]", ne_3: "b8[1]", ne_6: "b8[1, 1]", where_4: "i64[1, 1]", ne_8: "b8[1, 1]", where_6: "i64[1, 1]", permute_265: "f32[2, 1024]", div_54: "f32[1, 512, 1]", permute_269: "f32[1024, 4096]", permute_273: "f32[4096, 1024]", div_55: "f32[1, 512, 1]", permute_277: "f32[1024, 1024]", permute_289: "f32[1024, 1024]", permute_294: "f32[1024, 1024]", permute_298: "f32[1024, 1024]", div_57: "f32[1, 512, 1]", permute_302: "f32[1024, 4096]", permute_306: "f32[4096, 1024]", div_58: "f32[1, 512, 1]", permute_310: "f32[1024, 1024]", permute_322: "f32[1024, 1024]", permute_327: "f32[1024, 1024]", permute_331: "f32[1024, 1024]", div_60: "f32[1, 512, 1]", permute_335: "f32[1024, 4096]", permute_339: "f32[4096, 1024]", div_61: "f32[1, 512, 1]", permute_343: "f32[1024, 1024]", permute_355: "f32[1024, 1024]", permute_360: "f32[1024, 1024]", permute_364: "f32[1024, 1024]", div_63: "f32[1, 512, 1]", permute_368: "f32[1024, 4096]", permute_372: "f32[4096, 1024]", div_64: "f32[1, 512, 1]", permute_376: "f32[1024, 1024]", permute_388: "f32[1024, 1024]", permute_393: "f32[1024, 1024]", permute_397: "f32[1024, 1024]", div_66: "f32[1, 512, 1]", permute_401: "f32[1024, 4096]", permute_405: "f32[4096, 1024]", div_67: "f32[1, 512, 1]", permute_409: "f32[1024, 1024]", permute_421: "f32[1024, 1024]", permute_426: "f32[1024, 1024]", permute_430: "f32[1024, 1024]", div_69: "f32[1, 512, 1]", permute_434: "f32[1024, 4096]", permute_438: "f32[4096, 1024]", div_70: "f32[1, 512, 1]", permute_442: "f32[1024, 1024]", permute_454: "f32[1024, 1024]", permute_459: "f32[1024, 1024]", permute_463: "f32[1024, 1024]", div_72: "f32[1, 512, 1]", permute_467: "f32[1024, 4096]", permute_471: "f32[4096, 1024]", div_73: "f32[1, 512, 1]", permute_475: "f32[1024, 1024]", permute_487: "f32[1024, 1024]", permute_492: "f32[1024, 1024]", permute_496: "f32[1024, 1024]", div_75: "f32[1, 512, 1]", permute_500: "f32[1024, 4096]", permute_504: "f32[4096, 1024]", div_76: "f32[1, 512, 1]", permute_508: "f32[1024, 1024]", permute_520: "f32[1024, 1024]", permute_525: "f32[1024, 1024]", permute_529: "f32[1024, 1024]", div_78: "f32[1, 512, 1]", permute_533: "f32[1024, 4096]", permute_537: "f32[4096, 1024]", div_79: "f32[1, 512, 1]", permute_541: "f32[1024, 1024]", permute_553: "f32[1024, 1024]", permute_558: "f32[1024, 1024]", permute_562: "f32[1024, 1024]", div_81: "f32[1, 512, 1]", permute_566: "f32[1024, 4096]", permute_570: "f32[4096, 1024]", div_82: "f32[1, 512, 1]", permute_574: "f32[1024, 1024]", permute_586: "f32[1024, 1024]", permute_591: "f32[1024, 1024]", permute_595: "f32[1024, 1024]", div_84: "f32[1, 512, 1]", permute_599: "f32[1024, 4096]", permute_603: "f32[4096, 1024]", div_85: "f32[1, 512, 1]", permute_607: "f32[1024, 1024]", permute_619: "f32[1024, 1024]", permute_624: "f32[1024, 1024]", permute_628: "f32[1024, 1024]", div_87: "f32[1, 512, 1]", permute_632: "f32[1024, 4096]", permute_636: "f32[4096, 1024]", div_88: "f32[1, 512, 1]", permute_640: "f32[1024, 1024]", permute_652: "f32[1024, 1024]", permute_657: "f32[1024, 1024]", permute_661: "f32[1024, 1024]", div_90: "f32[1, 512, 1]", permute_665: "f32[1024, 4096]", permute_669: "f32[4096, 1024]", div_91: "f32[1, 512, 1]", permute_673: "f32[1024, 1024]", permute_685: "f32[1024, 1024]", permute_690: "f32[1024, 1024]", permute_694: "f32[1024, 1024]", div_93: "f32[1, 512, 1]", permute_698: "f32[1024, 4096]", permute_702: "f32[4096, 1024]", div_94: "f32[1, 512, 1]", permute_706: "f32[1024, 1024]", permute_718: "f32[1024, 1024]", permute_723: "f32[1024, 1024]", permute_727: "f32[1024, 1024]", div_96: "f32[1, 512, 1]", permute_731: "f32[1024, 4096]", permute_735: "f32[4096, 1024]", div_97: "f32[1, 512, 1]", permute_739: "f32[1024, 1024]", permute_751: "f32[1024, 1024]", permute_756: "f32[1024, 1024]", permute_760: "f32[1024, 1024]", div_99: "f32[1, 512, 1]", permute_764: "f32[1024, 4096]", permute_768: "f32[4096, 1024]", div_100: "f32[1, 512, 1]", permute_772: "f32[1024, 1024]", permute_784: "f32[1024, 1024]", permute_789: "f32[1024, 1024]", permute_793: "f32[1024, 1024]", div_102: "f32[1, 512, 1]", permute_797: "f32[1024, 4096]", permute_801: "f32[4096, 1024]", div_103: "f32[1, 512, 1]", permute_805: "f32[1024, 1024]", permute_817: "f32[1024, 1024]", permute_822: "f32[1024, 1024]", permute_826: "f32[1024, 1024]", div_105: "f32[1, 512, 1]", permute_830: "f32[1024, 4096]", permute_834: "f32[4096, 1024]", div_106: "f32[1, 512, 1]", permute_838: "f32[1024, 1024]", permute_850: "f32[1024, 1024]", permute_855: "f32[1024, 1024]", permute_859: "f32[1024, 1024]", div_108: "f32[1, 512, 1]", permute_863: "f32[1024, 4096]", permute_867: "f32[4096, 1024]", div_109: "f32[1, 512, 1]", permute_871: "f32[1024, 1024]", permute_883: "f32[1024, 1024]", permute_888: "f32[1024, 1024]", permute_892: "f32[1024, 1024]", div_111: "f32[1, 512, 1]", permute_896: "f32[1024, 4096]", permute_900: "f32[4096, 1024]", div_112: "f32[1, 512, 1]", permute_904: "f32[1024, 1024]", permute_916: "f32[1024, 1024]", permute_921: "f32[1024, 1024]", permute_925: "f32[1024, 1024]", div_114: "f32[1, 512, 1]", permute_929: "f32[1024, 4096]", permute_933: "f32[4096, 1024]", div_115: "f32[1, 512, 1]", permute_937: "f32[1024, 1024]", permute_949: "f32[1024, 1024]", permute_954: "f32[1024, 1024]", permute_958: "f32[1024, 1024]", div_117: "f32[1, 512, 1]", permute_962: "f32[1024, 4096]", permute_966: "f32[4096, 1024]", div_118: "f32[1, 512, 1]", permute_970: "f32[1024, 1024]", permute_982: "f32[1024, 1024]", permute_987: "f32[1024, 1024]", permute_991: "f32[1024, 1024]", div_120: "f32[1, 512, 1]", permute_995: "f32[1024, 4096]", permute_999: "f32[4096, 1024]", div_121: "f32[1, 512, 1]", permute_1003: "f32[1024, 1024]", permute_1015: "f32[1024, 1024]", permute_1020: "f32[1024, 1024]", permute_1024: "f32[1024, 1024]", div_123: "f32[1, 512, 1]", permute_1028: "f32[1024, 4096]", permute_1032: "f32[4096, 1024]", div_124: "f32[1, 512, 1]", permute_1036: "f32[1024, 1024]", permute_1048: "f32[1024, 1024]", permute_1053: "f32[1024, 1024]", permute_1057: "f32[1024, 1024]", div_126: "f32[1, 512, 1]", tangents_1: "f32[]", tangents_2: "f32[1, 512]", tangents_3: "f32[1, 512]"):
    # No stacktrace found for following nodes
    convert_element_type_default_23: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_293, torch.float32);  getitem_293 = None
    mul_tensor_92: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_23, 1.1111111111111112);  convert_element_type_default_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_19: "f32[1, 512, 4096]" = torch.ops.aten.view.default(addmm_4, [1, 512, 4096]);  addmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_6: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_19, 0.7071067811865476)
    erf: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_6);  mul_6 = None
    add_8: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    
    # No stacktrace found for following nodes
    convert_element_type_default_22: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_291, torch.float32);  getitem_291 = None
    mul_tensor_88: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_22, 1.1111111111111112);  convert_element_type_default_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_41: "f32[1, 512, 4096]" = torch.ops.aten.view.default(addmm_10, [1, 512, 4096]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_13: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_41, 0.7071067811865476)
    erf_1: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_13);  mul_13 = None
    add_16: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    
    # No stacktrace found for following nodes
    convert_element_type_default_21: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_289, torch.float32);  getitem_289 = None
    mul_tensor_84: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_21, 1.1111111111111112);  convert_element_type_default_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_63: "f32[1, 512, 4096]" = torch.ops.aten.view.default(addmm_16, [1, 512, 4096]);  addmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_20: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_63, 0.7071067811865476)
    erf_2: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_20);  mul_20 = None
    add_24: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    
    # No stacktrace found for following nodes
    convert_element_type_default_20: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_287, torch.float32);  getitem_287 = None
    mul_tensor_80: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_20, 1.1111111111111112);  convert_element_type_default_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_85: "f32[1, 512, 4096]" = torch.ops.aten.view.default(addmm_22, [1, 512, 4096]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_27: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_85, 0.7071067811865476)
    erf_3: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_27);  mul_27 = None
    add_32: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    
    # No stacktrace found for following nodes
    convert_element_type_default_19: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_285, torch.float32);  getitem_285 = None
    mul_tensor_76: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_19, 1.1111111111111112);  convert_element_type_default_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_107: "f32[1, 512, 4096]" = torch.ops.aten.view.default(addmm_28, [1, 512, 4096]);  addmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_34: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_107, 0.7071067811865476)
    erf_4: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_34);  mul_34 = None
    add_40: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    
    # No stacktrace found for following nodes
    convert_element_type_default_18: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_283, torch.float32);  getitem_283 = None
    mul_tensor_72: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_18, 1.1111111111111112);  convert_element_type_default_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_129: "f32[1, 512, 4096]" = torch.ops.aten.view.default(addmm_34, [1, 512, 4096]);  addmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_41: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_129, 0.7071067811865476)
    erf_5: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_41);  mul_41 = None
    add_48: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    
    # No stacktrace found for following nodes
    convert_element_type_default_17: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_281, torch.float32);  getitem_281 = None
    mul_tensor_68: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_17, 1.1111111111111112);  convert_element_type_default_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_151: "f32[1, 512, 4096]" = torch.ops.aten.view.default(addmm_40, [1, 512, 4096]);  addmm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_48: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_151, 0.7071067811865476)
    erf_6: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_48);  mul_48 = None
    add_56: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    
    # No stacktrace found for following nodes
    convert_element_type_default_16: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_279, torch.float32);  getitem_279 = None
    mul_tensor_64: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_16, 1.1111111111111112);  convert_element_type_default_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_173: "f32[1, 512, 4096]" = torch.ops.aten.view.default(addmm_46, [1, 512, 4096]);  addmm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_55: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_173, 0.7071067811865476)
    erf_7: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_55);  mul_55 = None
    add_64: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    
    # No stacktrace found for following nodes
    convert_element_type_default_15: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_277, torch.float32);  getitem_277 = None
    mul_tensor_60: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_15, 1.1111111111111112);  convert_element_type_default_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_195: "f32[1, 512, 4096]" = torch.ops.aten.view.default(addmm_52, [1, 512, 4096]);  addmm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_62: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_195, 0.7071067811865476)
    erf_8: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_62);  mul_62 = None
    add_72: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    
    # No stacktrace found for following nodes
    convert_element_type_default_14: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_275, torch.float32);  getitem_275 = None
    mul_tensor_56: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_14, 1.1111111111111112);  convert_element_type_default_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_217: "f32[1, 512, 4096]" = torch.ops.aten.view.default(addmm_58, [1, 512, 4096]);  addmm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_69: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_217, 0.7071067811865476)
    erf_9: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_69);  mul_69 = None
    add_80: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    
    # No stacktrace found for following nodes
    convert_element_type_default_13: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_273, torch.float32);  getitem_273 = None
    mul_tensor_52: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_13, 1.1111111111111112);  convert_element_type_default_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_239: "f32[1, 512, 4096]" = torch.ops.aten.view.default(addmm_64, [1, 512, 4096]);  addmm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_76: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_239, 0.7071067811865476)
    erf_10: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_76);  mul_76 = None
    add_88: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    
    # No stacktrace found for following nodes
    convert_element_type_default_12: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_271, torch.float32);  getitem_271 = None
    mul_tensor_48: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_12, 1.1111111111111112);  convert_element_type_default_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_261: "f32[1, 512, 4096]" = torch.ops.aten.view.default(addmm_70, [1, 512, 4096]);  addmm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_83: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_261, 0.7071067811865476)
    erf_11: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_83);  mul_83 = None
    add_96: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    
    # No stacktrace found for following nodes
    convert_element_type_default_11: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_269, torch.float32);  getitem_269 = None
    mul_tensor_44: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_11, 1.1111111111111112);  convert_element_type_default_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_283: "f32[1, 512, 4096]" = torch.ops.aten.view.default(addmm_76, [1, 512, 4096]);  addmm_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_90: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_283, 0.7071067811865476)
    erf_12: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_90);  mul_90 = None
    add_104: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
    
    # No stacktrace found for following nodes
    convert_element_type_default_10: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_267, torch.float32);  getitem_267 = None
    mul_tensor_40: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_10, 1.1111111111111112);  convert_element_type_default_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_305: "f32[1, 512, 4096]" = torch.ops.aten.view.default(addmm_82, [1, 512, 4096]);  addmm_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_97: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_305, 0.7071067811865476)
    erf_13: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_97);  mul_97 = None
    add_112: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
    
    # No stacktrace found for following nodes
    convert_element_type_default_9: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_265, torch.float32);  getitem_265 = None
    mul_tensor_36: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_9, 1.1111111111111112);  convert_element_type_default_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_327: "f32[1, 512, 4096]" = torch.ops.aten.view.default(addmm_88, [1, 512, 4096]);  addmm_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_104: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_327, 0.7071067811865476)
    erf_14: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_104);  mul_104 = None
    add_120: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
    
    # No stacktrace found for following nodes
    convert_element_type_default_8: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_263, torch.float32);  getitem_263 = None
    mul_tensor_32: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_8, 1.1111111111111112);  convert_element_type_default_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_349: "f32[1, 512, 4096]" = torch.ops.aten.view.default(addmm_94, [1, 512, 4096]);  addmm_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_111: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_349, 0.7071067811865476)
    erf_15: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_111);  mul_111 = None
    add_128: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
    
    # No stacktrace found for following nodes
    convert_element_type_default_7: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_261, torch.float32);  getitem_261 = None
    mul_tensor_28: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_7, 1.1111111111111112);  convert_element_type_default_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_371: "f32[1, 512, 4096]" = torch.ops.aten.view.default(addmm_100, [1, 512, 4096]);  addmm_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_118: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_371, 0.7071067811865476)
    erf_16: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_118);  mul_118 = None
    add_136: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
    
    # No stacktrace found for following nodes
    convert_element_type_default_6: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_259, torch.float32);  getitem_259 = None
    mul_tensor_24: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_6, 1.1111111111111112);  convert_element_type_default_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_393: "f32[1, 512, 4096]" = torch.ops.aten.view.default(addmm_106, [1, 512, 4096]);  addmm_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_125: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_393, 0.7071067811865476)
    erf_17: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_125);  mul_125 = None
    add_144: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
    
    # No stacktrace found for following nodes
    convert_element_type_default_5: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_257, torch.float32);  getitem_257 = None
    mul_tensor_20: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_5, 1.1111111111111112);  convert_element_type_default_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_415: "f32[1, 512, 4096]" = torch.ops.aten.view.default(addmm_112, [1, 512, 4096]);  addmm_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_132: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_415, 0.7071067811865476)
    erf_18: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_132);  mul_132 = None
    add_152: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
    
    # No stacktrace found for following nodes
    convert_element_type_default_4: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_255, torch.float32);  getitem_255 = None
    mul_tensor_16: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_4, 1.1111111111111112);  convert_element_type_default_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_437: "f32[1, 512, 4096]" = torch.ops.aten.view.default(addmm_118, [1, 512, 4096]);  addmm_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_139: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_437, 0.7071067811865476)
    erf_19: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_139);  mul_139 = None
    add_160: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
    
    # No stacktrace found for following nodes
    convert_element_type_default_3: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_253, torch.float32);  getitem_253 = None
    mul_tensor_12: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_3, 1.1111111111111112);  convert_element_type_default_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_459: "f32[1, 512, 4096]" = torch.ops.aten.view.default(addmm_124, [1, 512, 4096]);  addmm_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_146: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_459, 0.7071067811865476)
    erf_20: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_146);  mul_146 = None
    add_168: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
    
    # No stacktrace found for following nodes
    convert_element_type_default_2: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_251, torch.float32);  getitem_251 = None
    mul_tensor_8: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_2, 1.1111111111111112);  convert_element_type_default_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_481: "f32[1, 512, 4096]" = torch.ops.aten.view.default(addmm_130, [1, 512, 4096]);  addmm_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_153: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_481, 0.7071067811865476)
    erf_21: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_153);  mul_153 = None
    add_176: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
    
    # No stacktrace found for following nodes
    convert_element_type_default_1: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_249, torch.float32);  getitem_249 = None
    mul_tensor_4: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 1.1111111111111112);  convert_element_type_default_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_503: "f32[1, 512, 4096]" = torch.ops.aten.view.default(addmm_136, [1, 512, 4096]);  addmm_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_160: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_503, 0.7071067811865476)
    erf_22: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_160);  mul_160 = None
    add_184: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
    
    # No stacktrace found for following nodes
    convert_element_type_default: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_247, torch.float32);  getitem_247 = None
    mul_tensor: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_525: "f32[1, 512, 4096]" = torch.ops.aten.view.default(addmm_142, [1, 512, 4096]);  addmm_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_167: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_525, 0.7071067811865476)
    erf_23: "f32[1, 512, 4096]" = torch.ops.aten.erf.default(mul_167);  mul_167 = None
    add_192: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:1822, code: start_loss = loss_fct(start_logits, start_positions)
    alias_24: "f32[1, 512]" = torch.ops.aten.alias.default(sub_75);  sub_75 = None
    full_default_3: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sum_26: "i64[]" = torch.ops.aten.sum.default(ne);  ne = None
    convert_element_type: "f32[]" = torch.ops.prims.convert_element_type.default(sum_26, torch.float32);  sum_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:1823, code: end_loss = loss_fct(end_logits, end_positions)
    alias_25: "f32[1, 512]" = torch.ops.aten.alias.default(sub_77);  sub_77 = None
    sum_29: "i64[]" = torch.ops.aten.sum.default(ne_3);  ne_3 = None
    convert_element_type_1: "f32[]" = torch.ops.prims.convert_element_type.default(sum_29, torch.float32);  sum_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:1824, code: total_loss = (start_loss + end_loss) / 2
    div_51: "f32[]" = torch.ops.aten.div.Tensor(tangents_1, 2);  tangents_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:1823, code: end_loss = loss_fct(end_logits, end_positions)
    div_52: "f32[]" = torch.ops.aten.div.Tensor(div_51, convert_element_type_1);  convert_element_type_1 = None
    full_default_7: "f32[1, 512]" = torch.ops.aten.full.default([1, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    scatter: "f32[1, 512]" = torch.ops.aten.scatter.value(full_default_7, 1, where_4, -1.0);  where_4 = None
    where_5: "f32[1, 1]" = torch.ops.aten.where.self(ne_6, div_52, full_default_3);  ne_6 = div_52 = None
    mul_171: "f32[1, 512]" = torch.ops.aten.mul.Tensor(scatter, where_5);  scatter = where_5 = None
    alias_26: "f32[1, 512]" = torch.ops.aten.alias.default(alias_25);  alias_25 = None
    exp_26: "f32[1, 512]" = torch.ops.aten.exp.default(alias_26);  alias_26 = None
    sum_31: "f32[1, 1]" = torch.ops.aten.sum.dim_IntList(mul_171, [1], True)
    mul_172: "f32[1, 512]" = torch.ops.aten.mul.Tensor(exp_26, sum_31);  exp_26 = sum_31 = None
    sub_78: "f32[1, 512]" = torch.ops.aten.sub.Tensor(mul_171, mul_172);  mul_171 = mul_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:1823, code: end_loss = loss_fct(end_logits, end_positions)
    add_197: "f32[1, 512]" = torch.ops.aten.add.Tensor(tangents_3, sub_78);  tangents_3 = sub_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:1822, code: start_loss = loss_fct(start_logits, start_positions)
    div_53: "f32[]" = torch.ops.aten.div.Tensor(div_51, convert_element_type);  div_51 = convert_element_type = None
    scatter_1: "f32[1, 512]" = torch.ops.aten.scatter.value(full_default_7, 1, where_6, -1.0);  full_default_7 = where_6 = None
    where_7: "f32[1, 1]" = torch.ops.aten.where.self(ne_8, div_53, full_default_3);  ne_8 = div_53 = None
    mul_173: "f32[1, 512]" = torch.ops.aten.mul.Tensor(scatter_1, where_7);  scatter_1 = where_7 = None
    alias_27: "f32[1, 512]" = torch.ops.aten.alias.default(alias_24);  alias_24 = None
    exp_27: "f32[1, 512]" = torch.ops.aten.exp.default(alias_27);  alias_27 = None
    sum_32: "f32[1, 1]" = torch.ops.aten.sum.dim_IntList(mul_173, [1], True)
    mul_174: "f32[1, 512]" = torch.ops.aten.mul.Tensor(exp_27, sum_32);  exp_27 = sum_32 = None
    sub_79: "f32[1, 512]" = torch.ops.aten.sub.Tensor(mul_173, mul_174);  mul_173 = mul_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:1822, code: start_loss = loss_fct(start_logits, start_positions)
    add_198: "f32[1, 512]" = torch.ops.aten.add.Tensor(tangents_2, sub_79);  tangents_2 = sub_79 = None
    
    # No stacktrace found for following nodes
    unsqueeze_6: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(add_197, 2);  add_197 = None
    unsqueeze_7: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(add_198, 2);  add_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:1805, code: start_logits, end_logits = logits.split(1, dim=-1)
    cat: "f32[1, 512, 2]" = torch.ops.aten.cat.default([unsqueeze_7, unsqueeze_6], 2);  unsqueeze_7 = unsqueeze_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:1804, code: logits = self.qa_outputs(sequence_output)
    view_530: "f32[512, 2]" = torch.ops.aten.view.default(cat, [512, 2]);  cat = None
    mm: "f32[512, 1024]" = torch.ops.aten.mm.default(view_530, permute_265);  permute_265 = None
    permute_266: "f32[2, 512]" = torch.ops.aten.permute.default(view_530, [1, 0])
    mm_1: "f32[2, 1024]" = torch.ops.aten.mm.default(permute_266, view_528);  permute_266 = view_528 = None
    permute_267: "f32[1024, 2]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_33: "f32[1, 2]" = torch.ops.aten.sum.dim_IntList(view_530, [0], True);  view_530 = None
    view_531: "f32[2]" = torch.ops.aten.view.default(sum_33, [2]);  sum_33 = None
    permute_268: "f32[2, 1024]" = torch.ops.aten.permute.default(permute_267, [1, 0]);  permute_267 = None
    view_532: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm, [1, 512, 1024]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:592, code: hidden_states = self.ln(hidden_states)
    mul_176: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_532, primals_388);  primals_388 = None
    mul_177: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_176, 1024)
    sum_34: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_176, [2], True)
    mul_178: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_176, mul_169);  mul_176 = None
    sum_35: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_178, [2], True);  mul_178 = None
    mul_179: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_169, sum_35);  sum_35 = None
    sub_81: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_177, sum_34);  mul_177 = sum_34 = None
    sub_82: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_81, mul_179);  sub_81 = mul_179 = None
    mul_180: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_54, sub_82);  div_54 = sub_82 = None
    mul_181: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_532, mul_169);  mul_169 = None
    sum_36: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_181, [0, 1]);  mul_181 = None
    sum_37: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_532, [0, 1]);  view_532 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_2: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_241, torch.float32);  getitem_241 = None
    mul_182: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_2, 1.1111111111111112);  convert_element_type_2 = None
    mul_183: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_180, mul_182);  mul_182 = None
    clone_26: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_183, memory_format = torch.contiguous_format);  mul_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_533: "f32[512, 1024]" = torch.ops.aten.view.default(clone_26, [512, 1024]);  clone_26 = None
    mm_2: "f32[512, 4096]" = torch.ops.aten.mm.default(view_533, permute_269);  permute_269 = None
    permute_270: "f32[1024, 512]" = torch.ops.aten.permute.default(view_533, [1, 0])
    mm_3: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_270, view_526);  permute_270 = view_526 = None
    permute_271: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_3, [1, 0]);  mm_3 = None
    sum_38: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_533, [0], True);  view_533 = None
    view_534: "f32[1024]" = torch.ops.aten.view.default(sum_38, [1024]);  sum_38 = None
    permute_272: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_271, [1, 0]);  permute_271 = None
    view_535: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_2, [1, 512, 4096]);  mm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_185: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_192, 0.5);  add_192 = None
    mul_186: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_525, view_525)
    mul_187: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_186, -0.5);  mul_186 = None
    exp_28: "f32[1, 512, 4096]" = torch.ops.aten.exp.default(mul_187);  mul_187 = None
    mul_188: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_28, 0.3989422804014327);  exp_28 = None
    mul_189: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_525, mul_188);  view_525 = mul_188 = None
    add_200: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_185, mul_189);  mul_185 = mul_189 = None
    mul_190: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_535, add_200);  view_535 = add_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_536: "f32[512, 4096]" = torch.ops.aten.view.default(mul_190, [512, 4096]);  mul_190 = None
    mm_4: "f32[512, 1024]" = torch.ops.aten.mm.default(view_536, permute_273);  permute_273 = None
    permute_274: "f32[4096, 512]" = torch.ops.aten.permute.default(view_536, [1, 0])
    mm_5: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_274, view_524);  permute_274 = view_524 = None
    permute_275: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_5, [1, 0]);  mm_5 = None
    sum_39: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_536, [0], True);  view_536 = None
    view_537: "f32[4096]" = torch.ops.aten.view.default(sum_39, [4096]);  sum_39 = None
    permute_276: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_275, [1, 0]);  permute_275 = None
    view_538: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_4, [1, 512, 1024]);  mm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    mul_192: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_538, primals_382);  primals_382 = None
    mul_193: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_192, 1024)
    sum_40: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_192, [2], True)
    mul_194: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_192, mul_164);  mul_192 = None
    sum_41: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_194, [2], True);  mul_194 = None
    mul_195: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_164, sum_41);  sum_41 = None
    sub_84: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_193, sum_40);  mul_193 = sum_40 = None
    sub_85: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_84, mul_195);  sub_84 = mul_195 = None
    mul_196: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_55, sub_85);  div_55 = sub_85 = None
    mul_197: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_538, mul_164);  mul_164 = None
    sum_42: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_197, [0, 1]);  mul_197 = None
    sum_43: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_538, [0, 1]);  view_538 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    add_201: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_180, mul_196);  mul_180 = mul_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_3: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_237, torch.float32);  getitem_237 = None
    mul_198: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_3, 1.1111111111111112);  convert_element_type_3 = None
    mul_199: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_201, mul_198);  mul_198 = None
    clone_27: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_199, memory_format = torch.contiguous_format);  mul_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_539: "f32[512, 1024]" = torch.ops.aten.view.default(clone_27, [512, 1024]);  clone_27 = None
    mm_6: "f32[512, 1024]" = torch.ops.aten.mm.default(view_539, permute_277);  permute_277 = None
    permute_278: "f32[1024, 512]" = torch.ops.aten.permute.default(view_539, [1, 0])
    mm_7: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_278, view_522);  permute_278 = view_522 = None
    permute_279: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_7, [1, 0]);  mm_7 = None
    sum_44: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_539, [0], True);  view_539 = None
    view_540: "f32[1024]" = torch.ops.aten.view.default(sum_44, [1024]);  sum_44 = None
    permute_280: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_279, [1, 0]);  permute_279 = None
    view_541: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_6, [1, 512, 1024]);  mm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_542: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_541, [1, 512, 16, 64]);  view_541 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_281: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_542, [0, 2, 1, 3]);  view_542 = None
    
    # No stacktrace found for following nodes
    view_default_6: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_281, [16, 512, 64]);  permute_281 = None
    bmm_default_2: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_default_1, view_default_6);  permute_default_1 = None
    view_default_7: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_default_2, [1, 16, 512, 64]);  bmm_default_2 = None
    bmm_default_3: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_default_6, permute_default_2);  view_default_6 = permute_default_2 = None
    view_default_8: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_default_3, [1, 16, 512, 512]);  bmm_default_3 = None
    mul_tensor_1: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_default_8, mul_tensor);  view_default_8 = mul_tensor = None
    clone_default_3: "f32[1, 16, 512, 512]" = torch.ops.aten.clone.default(mul_tensor_1, memory_format = torch.contiguous_format);  mul_tensor_1 = None
    mul_tensor_2: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(clone_default_3, alias_default_1);  clone_default_3 = None
    sum_dim_int_list_1: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [-1], True)
    mul_tensor_3: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_default_1, sum_dim_int_list_1);  alias_default_1 = sum_dim_int_list_1 = None
    sub_tensor_1: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None
    view_default_9: "f32[16, 512, 512]" = torch.ops.aten.view.default(sub_tensor_1, [16, 512, 512]);  sub_tensor_1 = None
    bmm_default_4: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_default_3, view_default_9);  permute_default_3 = None
    view_default_10: "f32[1, 16, 64, 512]" = torch.ops.aten.view.default(bmm_default_4, [1, 16, 64, 512]);  bmm_default_4 = None
    mul_scalar_2: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(view_default_10, 0.3535533905932738);  view_default_10 = None
    permute_default_5: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(mul_scalar_2, [0, 1, 3, 2]);  mul_scalar_2 = None
    bmm_default_5: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_default_9, permute_default_4);  view_default_9 = permute_default_4 = None
    view_default_11: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_default_5, [1, 16, 512, 64]);  bmm_default_5 = None
    mul_scalar_3: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(view_default_11, 0.3535533905932738);  view_default_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_287: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(mul_scalar_3, [0, 2, 1, 3]);  mul_scalar_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_29: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_287, memory_format = torch.contiguous_format);  permute_287 = None
    view_549: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_29, [1, 512, 1024]);  clone_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_288: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_default_7, [0, 2, 1, 3]);  view_default_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_30: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_288, memory_format = torch.contiguous_format);  permute_288 = None
    view_550: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_30, [1, 512, 1024]);  clone_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_551: "f32[512, 1024]" = torch.ops.aten.view.default(view_550, [512, 1024]);  view_550 = None
    mm_8: "f32[512, 1024]" = torch.ops.aten.mm.default(view_551, permute_289);  permute_289 = None
    permute_290: "f32[1024, 512]" = torch.ops.aten.permute.default(view_551, [1, 0])
    mm_9: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_290, view_506);  permute_290 = None
    permute_291: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_9, [1, 0]);  mm_9 = None
    sum_46: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_551, [0], True);  view_551 = None
    view_552: "f32[1024]" = torch.ops.aten.view.default(sum_46, [1024]);  sum_46 = None
    permute_292: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_291, [1, 0]);  permute_291 = None
    view_553: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_8, [1, 512, 1024]);  mm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_293: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(permute_default_5, [0, 2, 1, 3]);  permute_default_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_554: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_293, [1, 512, 1024]);  permute_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_555: "f32[512, 1024]" = torch.ops.aten.view.default(view_554, [512, 1024]);  view_554 = None
    mm_10: "f32[512, 1024]" = torch.ops.aten.mm.default(view_555, permute_294);  permute_294 = None
    permute_295: "f32[1024, 512]" = torch.ops.aten.permute.default(view_555, [1, 0])
    mm_11: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_295, view_506);  permute_295 = None
    permute_296: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_11, [1, 0]);  mm_11 = None
    sum_47: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_555, [0], True);  view_555 = None
    view_556: "f32[1024]" = torch.ops.aten.view.default(sum_47, [1024]);  sum_47 = None
    permute_297: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_296, [1, 0]);  permute_296 = None
    view_557: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_10, [1, 512, 1024]);  mm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_202: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_553, view_557);  view_553 = view_557 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_558: "f32[512, 1024]" = torch.ops.aten.view.default(view_549, [512, 1024]);  view_549 = None
    mm_12: "f32[512, 1024]" = torch.ops.aten.mm.default(view_558, permute_298);  permute_298 = None
    permute_299: "f32[1024, 512]" = torch.ops.aten.permute.default(view_558, [1, 0])
    mm_13: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_299, view_506);  permute_299 = view_506 = None
    permute_300: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_13, [1, 0]);  mm_13 = None
    sum_48: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_558, [0], True);  view_558 = None
    view_559: "f32[1024]" = torch.ops.aten.view.default(sum_48, [1024]);  sum_48 = None
    permute_301: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_300, [1, 0]);  permute_300 = None
    view_560: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_12, [1, 512, 1024]);  mm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    add_203: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_202, view_560);  add_202 = view_560 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    mul_205: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_203, primals_372);  primals_372 = None
    mul_206: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_205, 1024)
    sum_49: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_205, [2], True)
    mul_207: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_205, mul_162);  mul_205 = None
    sum_50: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_207, [2], True);  mul_207 = None
    mul_208: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_162, sum_50);  sum_50 = None
    sub_88: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_206, sum_49);  mul_206 = sum_49 = None
    sub_89: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_88, mul_208);  sub_88 = mul_208 = None
    mul_209: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_57, sub_89);  div_57 = sub_89 = None
    mul_210: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_203, mul_162);  mul_162 = None
    sum_51: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_210, [0, 1]);  mul_210 = None
    sum_52: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_203, [0, 1]);  add_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    add_204: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_201, mul_209);  add_201 = mul_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_5: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_231, torch.float32);  getitem_231 = None
    mul_211: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_5, 1.1111111111111112);  convert_element_type_5 = None
    mul_212: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_204, mul_211);  mul_211 = None
    clone_31: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_212, memory_format = torch.contiguous_format);  mul_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_561: "f32[512, 1024]" = torch.ops.aten.view.default(clone_31, [512, 1024]);  clone_31 = None
    mm_14: "f32[512, 4096]" = torch.ops.aten.mm.default(view_561, permute_302);  permute_302 = None
    permute_303: "f32[1024, 512]" = torch.ops.aten.permute.default(view_561, [1, 0])
    mm_15: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_303, view_504);  permute_303 = view_504 = None
    permute_304: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_15, [1, 0]);  mm_15 = None
    sum_53: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_561, [0], True);  view_561 = None
    view_562: "f32[1024]" = torch.ops.aten.view.default(sum_53, [1024]);  sum_53 = None
    permute_305: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_304, [1, 0]);  permute_304 = None
    view_563: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_14, [1, 512, 4096]);  mm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_214: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_184, 0.5);  add_184 = None
    mul_215: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_503, view_503)
    mul_216: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_215, -0.5);  mul_215 = None
    exp_29: "f32[1, 512, 4096]" = torch.ops.aten.exp.default(mul_216);  mul_216 = None
    mul_217: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_29, 0.3989422804014327);  exp_29 = None
    mul_218: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_503, mul_217);  view_503 = mul_217 = None
    add_206: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_214, mul_218);  mul_214 = mul_218 = None
    mul_219: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_563, add_206);  view_563 = add_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_564: "f32[512, 4096]" = torch.ops.aten.view.default(mul_219, [512, 4096]);  mul_219 = None
    mm_16: "f32[512, 1024]" = torch.ops.aten.mm.default(view_564, permute_306);  permute_306 = None
    permute_307: "f32[4096, 512]" = torch.ops.aten.permute.default(view_564, [1, 0])
    mm_17: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_307, view_502);  permute_307 = view_502 = None
    permute_308: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_17, [1, 0]);  mm_17 = None
    sum_54: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_564, [0], True);  view_564 = None
    view_565: "f32[4096]" = torch.ops.aten.view.default(sum_54, [4096]);  sum_54 = None
    permute_309: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_308, [1, 0]);  permute_308 = None
    view_566: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_16, [1, 512, 1024]);  mm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    mul_221: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_566, primals_366);  primals_366 = None
    mul_222: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_221, 1024)
    sum_55: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_221, [2], True)
    mul_223: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_221, mul_157);  mul_221 = None
    sum_56: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_223, [2], True);  mul_223 = None
    mul_224: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_157, sum_56);  sum_56 = None
    sub_91: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_222, sum_55);  mul_222 = sum_55 = None
    sub_92: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_91, mul_224);  sub_91 = mul_224 = None
    mul_225: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_58, sub_92);  div_58 = sub_92 = None
    mul_226: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_566, mul_157);  mul_157 = None
    sum_57: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_226, [0, 1]);  mul_226 = None
    sum_58: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_566, [0, 1]);  view_566 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    add_207: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_204, mul_225);  add_204 = mul_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_6: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_227, torch.float32);  getitem_227 = None
    mul_227: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_6, 1.1111111111111112);  convert_element_type_6 = None
    mul_228: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_207, mul_227);  mul_227 = None
    clone_32: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_228, memory_format = torch.contiguous_format);  mul_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_567: "f32[512, 1024]" = torch.ops.aten.view.default(clone_32, [512, 1024]);  clone_32 = None
    mm_18: "f32[512, 1024]" = torch.ops.aten.mm.default(view_567, permute_310);  permute_310 = None
    permute_311: "f32[1024, 512]" = torch.ops.aten.permute.default(view_567, [1, 0])
    mm_19: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_311, view_500);  permute_311 = view_500 = None
    permute_312: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_19, [1, 0]);  mm_19 = None
    sum_59: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_567, [0], True);  view_567 = None
    view_568: "f32[1024]" = torch.ops.aten.view.default(sum_59, [1024]);  sum_59 = None
    permute_313: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_312, [1, 0]);  permute_312 = None
    view_569: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_18, [1, 512, 1024]);  mm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_570: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_569, [1, 512, 16, 64]);  view_569 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_314: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_570, [0, 2, 1, 3]);  view_570 = None
    
    # No stacktrace found for following nodes
    view_default_18: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_314, [16, 512, 64]);  permute_314 = None
    bmm_default_8: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_default_7, view_default_18);  permute_default_7 = None
    view_default_19: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_default_8, [1, 16, 512, 64]);  bmm_default_8 = None
    bmm_default_9: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_default_18, permute_default_8);  view_default_18 = permute_default_8 = None
    view_default_20: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_default_9, [1, 16, 512, 512]);  bmm_default_9 = None
    mul_tensor_5: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_default_20, mul_tensor_4);  view_default_20 = mul_tensor_4 = None
    clone_default_7: "f32[1, 16, 512, 512]" = torch.ops.aten.clone.default(mul_tensor_5, memory_format = torch.contiguous_format);  mul_tensor_5 = None
    mul_tensor_6: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(clone_default_7, alias_default_3);  clone_default_7 = None
    sum_dim_int_list_3: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [-1], True)
    mul_tensor_7: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_default_3, sum_dim_int_list_3);  alias_default_3 = sum_dim_int_list_3 = None
    sub_tensor_3: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_tensor_6, mul_tensor_7);  mul_tensor_6 = mul_tensor_7 = None
    view_default_21: "f32[16, 512, 512]" = torch.ops.aten.view.default(sub_tensor_3, [16, 512, 512]);  sub_tensor_3 = None
    bmm_default_10: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_default_9, view_default_21);  permute_default_9 = None
    view_default_22: "f32[1, 16, 64, 512]" = torch.ops.aten.view.default(bmm_default_10, [1, 16, 64, 512]);  bmm_default_10 = None
    mul_scalar_6: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(view_default_22, 0.3535533905932738);  view_default_22 = None
    permute_default_11: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(mul_scalar_6, [0, 1, 3, 2]);  mul_scalar_6 = None
    bmm_default_11: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_default_21, permute_default_10);  view_default_21 = permute_default_10 = None
    view_default_23: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_default_11, [1, 16, 512, 64]);  bmm_default_11 = None
    mul_scalar_7: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(view_default_23, 0.3535533905932738);  view_default_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_320: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(mul_scalar_7, [0, 2, 1, 3]);  mul_scalar_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_34: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_320, memory_format = torch.contiguous_format);  permute_320 = None
    view_577: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_34, [1, 512, 1024]);  clone_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_321: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_default_19, [0, 2, 1, 3]);  view_default_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_35: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_321, memory_format = torch.contiguous_format);  permute_321 = None
    view_578: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_35, [1, 512, 1024]);  clone_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_579: "f32[512, 1024]" = torch.ops.aten.view.default(view_578, [512, 1024]);  view_578 = None
    mm_20: "f32[512, 1024]" = torch.ops.aten.mm.default(view_579, permute_322);  permute_322 = None
    permute_323: "f32[1024, 512]" = torch.ops.aten.permute.default(view_579, [1, 0])
    mm_21: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_323, view_484);  permute_323 = None
    permute_324: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_21, [1, 0]);  mm_21 = None
    sum_61: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_579, [0], True);  view_579 = None
    view_580: "f32[1024]" = torch.ops.aten.view.default(sum_61, [1024]);  sum_61 = None
    permute_325: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_324, [1, 0]);  permute_324 = None
    view_581: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_20, [1, 512, 1024]);  mm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_326: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(permute_default_11, [0, 2, 1, 3]);  permute_default_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_582: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_326, [1, 512, 1024]);  permute_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_583: "f32[512, 1024]" = torch.ops.aten.view.default(view_582, [512, 1024]);  view_582 = None
    mm_22: "f32[512, 1024]" = torch.ops.aten.mm.default(view_583, permute_327);  permute_327 = None
    permute_328: "f32[1024, 512]" = torch.ops.aten.permute.default(view_583, [1, 0])
    mm_23: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_328, view_484);  permute_328 = None
    permute_329: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_23, [1, 0]);  mm_23 = None
    sum_62: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_583, [0], True);  view_583 = None
    view_584: "f32[1024]" = torch.ops.aten.view.default(sum_62, [1024]);  sum_62 = None
    permute_330: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_329, [1, 0]);  permute_329 = None
    view_585: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_22, [1, 512, 1024]);  mm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_208: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_581, view_585);  view_581 = view_585 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_586: "f32[512, 1024]" = torch.ops.aten.view.default(view_577, [512, 1024]);  view_577 = None
    mm_24: "f32[512, 1024]" = torch.ops.aten.mm.default(view_586, permute_331);  permute_331 = None
    permute_332: "f32[1024, 512]" = torch.ops.aten.permute.default(view_586, [1, 0])
    mm_25: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_332, view_484);  permute_332 = view_484 = None
    permute_333: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_25, [1, 0]);  mm_25 = None
    sum_63: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_586, [0], True);  view_586 = None
    view_587: "f32[1024]" = torch.ops.aten.view.default(sum_63, [1024]);  sum_63 = None
    permute_334: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_333, [1, 0]);  permute_333 = None
    view_588: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_24, [1, 512, 1024]);  mm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    add_209: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_208, view_588);  add_208 = view_588 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    mul_234: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_209, primals_356);  primals_356 = None
    mul_235: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_234, 1024)
    sum_64: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_234, [2], True)
    mul_236: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_234, mul_155);  mul_234 = None
    sum_65: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_236, [2], True);  mul_236 = None
    mul_237: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_155, sum_65);  sum_65 = None
    sub_95: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_235, sum_64);  mul_235 = sum_64 = None
    sub_96: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_95, mul_237);  sub_95 = mul_237 = None
    mul_238: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_60, sub_96);  div_60 = sub_96 = None
    mul_239: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_209, mul_155);  mul_155 = None
    sum_66: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_239, [0, 1]);  mul_239 = None
    sum_67: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_209, [0, 1]);  add_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    add_210: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_207, mul_238);  add_207 = mul_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_8: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_221, torch.float32);  getitem_221 = None
    mul_240: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_8, 1.1111111111111112);  convert_element_type_8 = None
    mul_241: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_210, mul_240);  mul_240 = None
    clone_36: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_241, memory_format = torch.contiguous_format);  mul_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_589: "f32[512, 1024]" = torch.ops.aten.view.default(clone_36, [512, 1024]);  clone_36 = None
    mm_26: "f32[512, 4096]" = torch.ops.aten.mm.default(view_589, permute_335);  permute_335 = None
    permute_336: "f32[1024, 512]" = torch.ops.aten.permute.default(view_589, [1, 0])
    mm_27: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_336, view_482);  permute_336 = view_482 = None
    permute_337: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_27, [1, 0]);  mm_27 = None
    sum_68: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_589, [0], True);  view_589 = None
    view_590: "f32[1024]" = torch.ops.aten.view.default(sum_68, [1024]);  sum_68 = None
    permute_338: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_337, [1, 0]);  permute_337 = None
    view_591: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_26, [1, 512, 4096]);  mm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_243: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_176, 0.5);  add_176 = None
    mul_244: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_481, view_481)
    mul_245: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_244, -0.5);  mul_244 = None
    exp_30: "f32[1, 512, 4096]" = torch.ops.aten.exp.default(mul_245);  mul_245 = None
    mul_246: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_30, 0.3989422804014327);  exp_30 = None
    mul_247: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_481, mul_246);  view_481 = mul_246 = None
    add_212: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_243, mul_247);  mul_243 = mul_247 = None
    mul_248: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_591, add_212);  view_591 = add_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_592: "f32[512, 4096]" = torch.ops.aten.view.default(mul_248, [512, 4096]);  mul_248 = None
    mm_28: "f32[512, 1024]" = torch.ops.aten.mm.default(view_592, permute_339);  permute_339 = None
    permute_340: "f32[4096, 512]" = torch.ops.aten.permute.default(view_592, [1, 0])
    mm_29: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_340, view_480);  permute_340 = view_480 = None
    permute_341: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_29, [1, 0]);  mm_29 = None
    sum_69: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_592, [0], True);  view_592 = None
    view_593: "f32[4096]" = torch.ops.aten.view.default(sum_69, [4096]);  sum_69 = None
    permute_342: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_341, [1, 0]);  permute_341 = None
    view_594: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_28, [1, 512, 1024]);  mm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    mul_250: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_594, primals_350);  primals_350 = None
    mul_251: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_250, 1024)
    sum_70: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_250, [2], True)
    mul_252: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_250, mul_150);  mul_250 = None
    sum_71: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_252, [2], True);  mul_252 = None
    mul_253: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_150, sum_71);  sum_71 = None
    sub_98: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_251, sum_70);  mul_251 = sum_70 = None
    sub_99: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_98, mul_253);  sub_98 = mul_253 = None
    mul_254: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_61, sub_99);  div_61 = sub_99 = None
    mul_255: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_594, mul_150);  mul_150 = None
    sum_72: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_255, [0, 1]);  mul_255 = None
    sum_73: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_594, [0, 1]);  view_594 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    add_213: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_210, mul_254);  add_210 = mul_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_9: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_217, torch.float32);  getitem_217 = None
    mul_256: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_9, 1.1111111111111112);  convert_element_type_9 = None
    mul_257: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_213, mul_256);  mul_256 = None
    clone_37: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_257, memory_format = torch.contiguous_format);  mul_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_595: "f32[512, 1024]" = torch.ops.aten.view.default(clone_37, [512, 1024]);  clone_37 = None
    mm_30: "f32[512, 1024]" = torch.ops.aten.mm.default(view_595, permute_343);  permute_343 = None
    permute_344: "f32[1024, 512]" = torch.ops.aten.permute.default(view_595, [1, 0])
    mm_31: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_344, view_478);  permute_344 = view_478 = None
    permute_345: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_31, [1, 0]);  mm_31 = None
    sum_74: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_595, [0], True);  view_595 = None
    view_596: "f32[1024]" = torch.ops.aten.view.default(sum_74, [1024]);  sum_74 = None
    permute_346: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_345, [1, 0]);  permute_345 = None
    view_597: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_30, [1, 512, 1024]);  mm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_598: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_597, [1, 512, 16, 64]);  view_597 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_347: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_598, [0, 2, 1, 3]);  view_598 = None
    
    # No stacktrace found for following nodes
    view_default_30: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_347, [16, 512, 64]);  permute_347 = None
    bmm_default_14: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_default_13, view_default_30);  permute_default_13 = None
    view_default_31: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_default_14, [1, 16, 512, 64]);  bmm_default_14 = None
    bmm_default_15: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_default_30, permute_default_14);  view_default_30 = permute_default_14 = None
    view_default_32: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_default_15, [1, 16, 512, 512]);  bmm_default_15 = None
    mul_tensor_9: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_default_32, mul_tensor_8);  view_default_32 = mul_tensor_8 = None
    clone_default_11: "f32[1, 16, 512, 512]" = torch.ops.aten.clone.default(mul_tensor_9, memory_format = torch.contiguous_format);  mul_tensor_9 = None
    mul_tensor_10: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(clone_default_11, alias_default_5);  clone_default_11 = None
    sum_dim_int_list_5: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_10, [-1], True)
    mul_tensor_11: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_default_5, sum_dim_int_list_5);  alias_default_5 = sum_dim_int_list_5 = None
    sub_tensor_5: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_tensor_10, mul_tensor_11);  mul_tensor_10 = mul_tensor_11 = None
    view_default_33: "f32[16, 512, 512]" = torch.ops.aten.view.default(sub_tensor_5, [16, 512, 512]);  sub_tensor_5 = None
    bmm_default_16: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_default_15, view_default_33);  permute_default_15 = None
    view_default_34: "f32[1, 16, 64, 512]" = torch.ops.aten.view.default(bmm_default_16, [1, 16, 64, 512]);  bmm_default_16 = None
    mul_scalar_10: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(view_default_34, 0.3535533905932738);  view_default_34 = None
    permute_default_17: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(mul_scalar_10, [0, 1, 3, 2]);  mul_scalar_10 = None
    bmm_default_17: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_default_33, permute_default_16);  view_default_33 = permute_default_16 = None
    view_default_35: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_default_17, [1, 16, 512, 64]);  bmm_default_17 = None
    mul_scalar_11: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(view_default_35, 0.3535533905932738);  view_default_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_353: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(mul_scalar_11, [0, 2, 1, 3]);  mul_scalar_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_39: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_353, memory_format = torch.contiguous_format);  permute_353 = None
    view_605: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_39, [1, 512, 1024]);  clone_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_354: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_default_31, [0, 2, 1, 3]);  view_default_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_40: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_354, memory_format = torch.contiguous_format);  permute_354 = None
    view_606: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_40, [1, 512, 1024]);  clone_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_607: "f32[512, 1024]" = torch.ops.aten.view.default(view_606, [512, 1024]);  view_606 = None
    mm_32: "f32[512, 1024]" = torch.ops.aten.mm.default(view_607, permute_355);  permute_355 = None
    permute_356: "f32[1024, 512]" = torch.ops.aten.permute.default(view_607, [1, 0])
    mm_33: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_356, view_462);  permute_356 = None
    permute_357: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_33, [1, 0]);  mm_33 = None
    sum_76: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_607, [0], True);  view_607 = None
    view_608: "f32[1024]" = torch.ops.aten.view.default(sum_76, [1024]);  sum_76 = None
    permute_358: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_357, [1, 0]);  permute_357 = None
    view_609: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_32, [1, 512, 1024]);  mm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_359: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(permute_default_17, [0, 2, 1, 3]);  permute_default_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_610: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_359, [1, 512, 1024]);  permute_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_611: "f32[512, 1024]" = torch.ops.aten.view.default(view_610, [512, 1024]);  view_610 = None
    mm_34: "f32[512, 1024]" = torch.ops.aten.mm.default(view_611, permute_360);  permute_360 = None
    permute_361: "f32[1024, 512]" = torch.ops.aten.permute.default(view_611, [1, 0])
    mm_35: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_361, view_462);  permute_361 = None
    permute_362: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_35, [1, 0]);  mm_35 = None
    sum_77: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_611, [0], True);  view_611 = None
    view_612: "f32[1024]" = torch.ops.aten.view.default(sum_77, [1024]);  sum_77 = None
    permute_363: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_362, [1, 0]);  permute_362 = None
    view_613: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_34, [1, 512, 1024]);  mm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_214: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_609, view_613);  view_609 = view_613 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_614: "f32[512, 1024]" = torch.ops.aten.view.default(view_605, [512, 1024]);  view_605 = None
    mm_36: "f32[512, 1024]" = torch.ops.aten.mm.default(view_614, permute_364);  permute_364 = None
    permute_365: "f32[1024, 512]" = torch.ops.aten.permute.default(view_614, [1, 0])
    mm_37: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_365, view_462);  permute_365 = view_462 = None
    permute_366: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_37, [1, 0]);  mm_37 = None
    sum_78: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_614, [0], True);  view_614 = None
    view_615: "f32[1024]" = torch.ops.aten.view.default(sum_78, [1024]);  sum_78 = None
    permute_367: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_366, [1, 0]);  permute_366 = None
    view_616: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_36, [1, 512, 1024]);  mm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    add_215: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_214, view_616);  add_214 = view_616 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    mul_263: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_215, primals_340);  primals_340 = None
    mul_264: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_263, 1024)
    sum_79: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_263, [2], True)
    mul_265: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_263, mul_148);  mul_263 = None
    sum_80: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_265, [2], True);  mul_265 = None
    mul_266: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_148, sum_80);  sum_80 = None
    sub_102: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_264, sum_79);  mul_264 = sum_79 = None
    sub_103: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_102, mul_266);  sub_102 = mul_266 = None
    mul_267: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_63, sub_103);  div_63 = sub_103 = None
    mul_268: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_215, mul_148);  mul_148 = None
    sum_81: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_268, [0, 1]);  mul_268 = None
    sum_82: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_215, [0, 1]);  add_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    add_216: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_213, mul_267);  add_213 = mul_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_11: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_211, torch.float32);  getitem_211 = None
    mul_269: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_11, 1.1111111111111112);  convert_element_type_11 = None
    mul_270: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_216, mul_269);  mul_269 = None
    clone_41: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_270, memory_format = torch.contiguous_format);  mul_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_617: "f32[512, 1024]" = torch.ops.aten.view.default(clone_41, [512, 1024]);  clone_41 = None
    mm_38: "f32[512, 4096]" = torch.ops.aten.mm.default(view_617, permute_368);  permute_368 = None
    permute_369: "f32[1024, 512]" = torch.ops.aten.permute.default(view_617, [1, 0])
    mm_39: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_369, view_460);  permute_369 = view_460 = None
    permute_370: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_39, [1, 0]);  mm_39 = None
    sum_83: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_617, [0], True);  view_617 = None
    view_618: "f32[1024]" = torch.ops.aten.view.default(sum_83, [1024]);  sum_83 = None
    permute_371: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_370, [1, 0]);  permute_370 = None
    view_619: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_38, [1, 512, 4096]);  mm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_272: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_168, 0.5);  add_168 = None
    mul_273: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_459, view_459)
    mul_274: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_273, -0.5);  mul_273 = None
    exp_31: "f32[1, 512, 4096]" = torch.ops.aten.exp.default(mul_274);  mul_274 = None
    mul_275: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_31, 0.3989422804014327);  exp_31 = None
    mul_276: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_459, mul_275);  view_459 = mul_275 = None
    add_218: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_272, mul_276);  mul_272 = mul_276 = None
    mul_277: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_619, add_218);  view_619 = add_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_620: "f32[512, 4096]" = torch.ops.aten.view.default(mul_277, [512, 4096]);  mul_277 = None
    mm_40: "f32[512, 1024]" = torch.ops.aten.mm.default(view_620, permute_372);  permute_372 = None
    permute_373: "f32[4096, 512]" = torch.ops.aten.permute.default(view_620, [1, 0])
    mm_41: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_373, view_458);  permute_373 = view_458 = None
    permute_374: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_41, [1, 0]);  mm_41 = None
    sum_84: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_620, [0], True);  view_620 = None
    view_621: "f32[4096]" = torch.ops.aten.view.default(sum_84, [4096]);  sum_84 = None
    permute_375: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_374, [1, 0]);  permute_374 = None
    view_622: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_40, [1, 512, 1024]);  mm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    mul_279: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_622, primals_334);  primals_334 = None
    mul_280: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_279, 1024)
    sum_85: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_279, [2], True)
    mul_281: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_279, mul_143);  mul_279 = None
    sum_86: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_281, [2], True);  mul_281 = None
    mul_282: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_143, sum_86);  sum_86 = None
    sub_105: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_280, sum_85);  mul_280 = sum_85 = None
    sub_106: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_105, mul_282);  sub_105 = mul_282 = None
    mul_283: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_64, sub_106);  div_64 = sub_106 = None
    mul_284: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_622, mul_143);  mul_143 = None
    sum_87: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_284, [0, 1]);  mul_284 = None
    sum_88: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_622, [0, 1]);  view_622 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    add_219: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_216, mul_283);  add_216 = mul_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_12: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_207, torch.float32);  getitem_207 = None
    mul_285: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_12, 1.1111111111111112);  convert_element_type_12 = None
    mul_286: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_219, mul_285);  mul_285 = None
    clone_42: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_286, memory_format = torch.contiguous_format);  mul_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_623: "f32[512, 1024]" = torch.ops.aten.view.default(clone_42, [512, 1024]);  clone_42 = None
    mm_42: "f32[512, 1024]" = torch.ops.aten.mm.default(view_623, permute_376);  permute_376 = None
    permute_377: "f32[1024, 512]" = torch.ops.aten.permute.default(view_623, [1, 0])
    mm_43: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_377, view_456);  permute_377 = view_456 = None
    permute_378: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_43, [1, 0]);  mm_43 = None
    sum_89: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_623, [0], True);  view_623 = None
    view_624: "f32[1024]" = torch.ops.aten.view.default(sum_89, [1024]);  sum_89 = None
    permute_379: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_378, [1, 0]);  permute_378 = None
    view_625: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_42, [1, 512, 1024]);  mm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_626: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_625, [1, 512, 16, 64]);  view_625 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_380: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_626, [0, 2, 1, 3]);  view_626 = None
    
    # No stacktrace found for following nodes
    view_default_42: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_380, [16, 512, 64]);  permute_380 = None
    bmm_default_20: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_default_19, view_default_42);  permute_default_19 = None
    view_default_43: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_default_20, [1, 16, 512, 64]);  bmm_default_20 = None
    bmm_default_21: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_default_42, permute_default_20);  view_default_42 = permute_default_20 = None
    view_default_44: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_default_21, [1, 16, 512, 512]);  bmm_default_21 = None
    mul_tensor_13: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_default_44, mul_tensor_12);  view_default_44 = mul_tensor_12 = None
    clone_default_15: "f32[1, 16, 512, 512]" = torch.ops.aten.clone.default(mul_tensor_13, memory_format = torch.contiguous_format);  mul_tensor_13 = None
    mul_tensor_14: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(clone_default_15, alias_default_7);  clone_default_15 = None
    sum_dim_int_list_7: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_14, [-1], True)
    mul_tensor_15: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_default_7, sum_dim_int_list_7);  alias_default_7 = sum_dim_int_list_7 = None
    sub_tensor_7: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_tensor_14, mul_tensor_15);  mul_tensor_14 = mul_tensor_15 = None
    view_default_45: "f32[16, 512, 512]" = torch.ops.aten.view.default(sub_tensor_7, [16, 512, 512]);  sub_tensor_7 = None
    bmm_default_22: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_default_21, view_default_45);  permute_default_21 = None
    view_default_46: "f32[1, 16, 64, 512]" = torch.ops.aten.view.default(bmm_default_22, [1, 16, 64, 512]);  bmm_default_22 = None
    mul_scalar_14: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(view_default_46, 0.3535533905932738);  view_default_46 = None
    permute_default_23: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(mul_scalar_14, [0, 1, 3, 2]);  mul_scalar_14 = None
    bmm_default_23: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_default_45, permute_default_22);  view_default_45 = permute_default_22 = None
    view_default_47: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_default_23, [1, 16, 512, 64]);  bmm_default_23 = None
    mul_scalar_15: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(view_default_47, 0.3535533905932738);  view_default_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_386: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(mul_scalar_15, [0, 2, 1, 3]);  mul_scalar_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_44: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_386, memory_format = torch.contiguous_format);  permute_386 = None
    view_633: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_44, [1, 512, 1024]);  clone_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_387: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_default_43, [0, 2, 1, 3]);  view_default_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_45: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_387, memory_format = torch.contiguous_format);  permute_387 = None
    view_634: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_45, [1, 512, 1024]);  clone_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_635: "f32[512, 1024]" = torch.ops.aten.view.default(view_634, [512, 1024]);  view_634 = None
    mm_44: "f32[512, 1024]" = torch.ops.aten.mm.default(view_635, permute_388);  permute_388 = None
    permute_389: "f32[1024, 512]" = torch.ops.aten.permute.default(view_635, [1, 0])
    mm_45: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_389, view_440);  permute_389 = None
    permute_390: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_45, [1, 0]);  mm_45 = None
    sum_91: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_635, [0], True);  view_635 = None
    view_636: "f32[1024]" = torch.ops.aten.view.default(sum_91, [1024]);  sum_91 = None
    permute_391: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_390, [1, 0]);  permute_390 = None
    view_637: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_44, [1, 512, 1024]);  mm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_392: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(permute_default_23, [0, 2, 1, 3]);  permute_default_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_638: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_392, [1, 512, 1024]);  permute_392 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_639: "f32[512, 1024]" = torch.ops.aten.view.default(view_638, [512, 1024]);  view_638 = None
    mm_46: "f32[512, 1024]" = torch.ops.aten.mm.default(view_639, permute_393);  permute_393 = None
    permute_394: "f32[1024, 512]" = torch.ops.aten.permute.default(view_639, [1, 0])
    mm_47: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_394, view_440);  permute_394 = None
    permute_395: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_47, [1, 0]);  mm_47 = None
    sum_92: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_639, [0], True);  view_639 = None
    view_640: "f32[1024]" = torch.ops.aten.view.default(sum_92, [1024]);  sum_92 = None
    permute_396: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_395, [1, 0]);  permute_395 = None
    view_641: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_46, [1, 512, 1024]);  mm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_220: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_637, view_641);  view_637 = view_641 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_642: "f32[512, 1024]" = torch.ops.aten.view.default(view_633, [512, 1024]);  view_633 = None
    mm_48: "f32[512, 1024]" = torch.ops.aten.mm.default(view_642, permute_397);  permute_397 = None
    permute_398: "f32[1024, 512]" = torch.ops.aten.permute.default(view_642, [1, 0])
    mm_49: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_398, view_440);  permute_398 = view_440 = None
    permute_399: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_49, [1, 0]);  mm_49 = None
    sum_93: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_642, [0], True);  view_642 = None
    view_643: "f32[1024]" = torch.ops.aten.view.default(sum_93, [1024]);  sum_93 = None
    permute_400: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_399, [1, 0]);  permute_399 = None
    view_644: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_48, [1, 512, 1024]);  mm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    add_221: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_220, view_644);  add_220 = view_644 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    mul_292: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_221, primals_324);  primals_324 = None
    mul_293: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_292, 1024)
    sum_94: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_292, [2], True)
    mul_294: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_292, mul_141);  mul_292 = None
    sum_95: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_294, [2], True);  mul_294 = None
    mul_295: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_141, sum_95);  sum_95 = None
    sub_109: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_293, sum_94);  mul_293 = sum_94 = None
    sub_110: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_109, mul_295);  sub_109 = mul_295 = None
    mul_296: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_66, sub_110);  div_66 = sub_110 = None
    mul_297: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_221, mul_141);  mul_141 = None
    sum_96: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_297, [0, 1]);  mul_297 = None
    sum_97: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_221, [0, 1]);  add_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    add_222: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_219, mul_296);  add_219 = mul_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_14: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_201, torch.float32);  getitem_201 = None
    mul_298: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_14, 1.1111111111111112);  convert_element_type_14 = None
    mul_299: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_222, mul_298);  mul_298 = None
    clone_46: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_299, memory_format = torch.contiguous_format);  mul_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_645: "f32[512, 1024]" = torch.ops.aten.view.default(clone_46, [512, 1024]);  clone_46 = None
    mm_50: "f32[512, 4096]" = torch.ops.aten.mm.default(view_645, permute_401);  permute_401 = None
    permute_402: "f32[1024, 512]" = torch.ops.aten.permute.default(view_645, [1, 0])
    mm_51: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_402, view_438);  permute_402 = view_438 = None
    permute_403: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_51, [1, 0]);  mm_51 = None
    sum_98: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_645, [0], True);  view_645 = None
    view_646: "f32[1024]" = torch.ops.aten.view.default(sum_98, [1024]);  sum_98 = None
    permute_404: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_403, [1, 0]);  permute_403 = None
    view_647: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_50, [1, 512, 4096]);  mm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_301: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_160, 0.5);  add_160 = None
    mul_302: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_437, view_437)
    mul_303: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_302, -0.5);  mul_302 = None
    exp_32: "f32[1, 512, 4096]" = torch.ops.aten.exp.default(mul_303);  mul_303 = None
    mul_304: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_32, 0.3989422804014327);  exp_32 = None
    mul_305: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_437, mul_304);  view_437 = mul_304 = None
    add_224: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_301, mul_305);  mul_301 = mul_305 = None
    mul_306: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_647, add_224);  view_647 = add_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_648: "f32[512, 4096]" = torch.ops.aten.view.default(mul_306, [512, 4096]);  mul_306 = None
    mm_52: "f32[512, 1024]" = torch.ops.aten.mm.default(view_648, permute_405);  permute_405 = None
    permute_406: "f32[4096, 512]" = torch.ops.aten.permute.default(view_648, [1, 0])
    mm_53: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_406, view_436);  permute_406 = view_436 = None
    permute_407: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_53, [1, 0]);  mm_53 = None
    sum_99: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_648, [0], True);  view_648 = None
    view_649: "f32[4096]" = torch.ops.aten.view.default(sum_99, [4096]);  sum_99 = None
    permute_408: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_407, [1, 0]);  permute_407 = None
    view_650: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_52, [1, 512, 1024]);  mm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    mul_308: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_650, primals_318);  primals_318 = None
    mul_309: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_308, 1024)
    sum_100: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_308, [2], True)
    mul_310: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_308, mul_136);  mul_308 = None
    sum_101: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_310, [2], True);  mul_310 = None
    mul_311: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_136, sum_101);  sum_101 = None
    sub_112: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_309, sum_100);  mul_309 = sum_100 = None
    sub_113: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_112, mul_311);  sub_112 = mul_311 = None
    mul_312: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_67, sub_113);  div_67 = sub_113 = None
    mul_313: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_650, mul_136);  mul_136 = None
    sum_102: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_313, [0, 1]);  mul_313 = None
    sum_103: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_650, [0, 1]);  view_650 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    add_225: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_222, mul_312);  add_222 = mul_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_15: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_197, torch.float32);  getitem_197 = None
    mul_314: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_15, 1.1111111111111112);  convert_element_type_15 = None
    mul_315: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_225, mul_314);  mul_314 = None
    clone_47: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_315, memory_format = torch.contiguous_format);  mul_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_651: "f32[512, 1024]" = torch.ops.aten.view.default(clone_47, [512, 1024]);  clone_47 = None
    mm_54: "f32[512, 1024]" = torch.ops.aten.mm.default(view_651, permute_409);  permute_409 = None
    permute_410: "f32[1024, 512]" = torch.ops.aten.permute.default(view_651, [1, 0])
    mm_55: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_410, view_434);  permute_410 = view_434 = None
    permute_411: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_55, [1, 0]);  mm_55 = None
    sum_104: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_651, [0], True);  view_651 = None
    view_652: "f32[1024]" = torch.ops.aten.view.default(sum_104, [1024]);  sum_104 = None
    permute_412: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_411, [1, 0]);  permute_411 = None
    view_653: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_54, [1, 512, 1024]);  mm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_654: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_653, [1, 512, 16, 64]);  view_653 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_413: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_654, [0, 2, 1, 3]);  view_654 = None
    
    # No stacktrace found for following nodes
    view_default_54: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_413, [16, 512, 64]);  permute_413 = None
    bmm_default_26: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_default_25, view_default_54);  permute_default_25 = None
    view_default_55: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_default_26, [1, 16, 512, 64]);  bmm_default_26 = None
    bmm_default_27: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_default_54, permute_default_26);  view_default_54 = permute_default_26 = None
    view_default_56: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_default_27, [1, 16, 512, 512]);  bmm_default_27 = None
    mul_tensor_17: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_default_56, mul_tensor_16);  view_default_56 = mul_tensor_16 = None
    clone_default_19: "f32[1, 16, 512, 512]" = torch.ops.aten.clone.default(mul_tensor_17, memory_format = torch.contiguous_format);  mul_tensor_17 = None
    mul_tensor_18: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(clone_default_19, alias_default_9);  clone_default_19 = None
    sum_dim_int_list_9: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_18, [-1], True)
    mul_tensor_19: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_default_9, sum_dim_int_list_9);  alias_default_9 = sum_dim_int_list_9 = None
    sub_tensor_9: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_tensor_18, mul_tensor_19);  mul_tensor_18 = mul_tensor_19 = None
    view_default_57: "f32[16, 512, 512]" = torch.ops.aten.view.default(sub_tensor_9, [16, 512, 512]);  sub_tensor_9 = None
    bmm_default_28: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_default_27, view_default_57);  permute_default_27 = None
    view_default_58: "f32[1, 16, 64, 512]" = torch.ops.aten.view.default(bmm_default_28, [1, 16, 64, 512]);  bmm_default_28 = None
    mul_scalar_18: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(view_default_58, 0.3535533905932738);  view_default_58 = None
    permute_default_29: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(mul_scalar_18, [0, 1, 3, 2]);  mul_scalar_18 = None
    bmm_default_29: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_default_57, permute_default_28);  view_default_57 = permute_default_28 = None
    view_default_59: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_default_29, [1, 16, 512, 64]);  bmm_default_29 = None
    mul_scalar_19: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(view_default_59, 0.3535533905932738);  view_default_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_419: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(mul_scalar_19, [0, 2, 1, 3]);  mul_scalar_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_49: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_419, memory_format = torch.contiguous_format);  permute_419 = None
    view_661: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_49, [1, 512, 1024]);  clone_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_420: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_default_55, [0, 2, 1, 3]);  view_default_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_50: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_420, memory_format = torch.contiguous_format);  permute_420 = None
    view_662: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_50, [1, 512, 1024]);  clone_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_663: "f32[512, 1024]" = torch.ops.aten.view.default(view_662, [512, 1024]);  view_662 = None
    mm_56: "f32[512, 1024]" = torch.ops.aten.mm.default(view_663, permute_421);  permute_421 = None
    permute_422: "f32[1024, 512]" = torch.ops.aten.permute.default(view_663, [1, 0])
    mm_57: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_422, view_418);  permute_422 = None
    permute_423: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_57, [1, 0]);  mm_57 = None
    sum_106: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_663, [0], True);  view_663 = None
    view_664: "f32[1024]" = torch.ops.aten.view.default(sum_106, [1024]);  sum_106 = None
    permute_424: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_423, [1, 0]);  permute_423 = None
    view_665: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_56, [1, 512, 1024]);  mm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_425: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(permute_default_29, [0, 2, 1, 3]);  permute_default_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_666: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_425, [1, 512, 1024]);  permute_425 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_667: "f32[512, 1024]" = torch.ops.aten.view.default(view_666, [512, 1024]);  view_666 = None
    mm_58: "f32[512, 1024]" = torch.ops.aten.mm.default(view_667, permute_426);  permute_426 = None
    permute_427: "f32[1024, 512]" = torch.ops.aten.permute.default(view_667, [1, 0])
    mm_59: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_427, view_418);  permute_427 = None
    permute_428: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_59, [1, 0]);  mm_59 = None
    sum_107: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_667, [0], True);  view_667 = None
    view_668: "f32[1024]" = torch.ops.aten.view.default(sum_107, [1024]);  sum_107 = None
    permute_429: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_428, [1, 0]);  permute_428 = None
    view_669: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_58, [1, 512, 1024]);  mm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_226: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_665, view_669);  view_665 = view_669 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_670: "f32[512, 1024]" = torch.ops.aten.view.default(view_661, [512, 1024]);  view_661 = None
    mm_60: "f32[512, 1024]" = torch.ops.aten.mm.default(view_670, permute_430);  permute_430 = None
    permute_431: "f32[1024, 512]" = torch.ops.aten.permute.default(view_670, [1, 0])
    mm_61: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_431, view_418);  permute_431 = view_418 = None
    permute_432: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_61, [1, 0]);  mm_61 = None
    sum_108: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_670, [0], True);  view_670 = None
    view_671: "f32[1024]" = torch.ops.aten.view.default(sum_108, [1024]);  sum_108 = None
    permute_433: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_432, [1, 0]);  permute_432 = None
    view_672: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_60, [1, 512, 1024]);  mm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    add_227: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_226, view_672);  add_226 = view_672 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    mul_321: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_227, primals_308);  primals_308 = None
    mul_322: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_321, 1024)
    sum_109: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_321, [2], True)
    mul_323: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_321, mul_134);  mul_321 = None
    sum_110: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_323, [2], True);  mul_323 = None
    mul_324: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_134, sum_110);  sum_110 = None
    sub_116: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_322, sum_109);  mul_322 = sum_109 = None
    sub_117: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_116, mul_324);  sub_116 = mul_324 = None
    mul_325: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_69, sub_117);  div_69 = sub_117 = None
    mul_326: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_227, mul_134);  mul_134 = None
    sum_111: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_326, [0, 1]);  mul_326 = None
    sum_112: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_227, [0, 1]);  add_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    add_228: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_225, mul_325);  add_225 = mul_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_17: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_191, torch.float32);  getitem_191 = None
    mul_327: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_17, 1.1111111111111112);  convert_element_type_17 = None
    mul_328: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_228, mul_327);  mul_327 = None
    clone_51: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_328, memory_format = torch.contiguous_format);  mul_328 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_673: "f32[512, 1024]" = torch.ops.aten.view.default(clone_51, [512, 1024]);  clone_51 = None
    mm_62: "f32[512, 4096]" = torch.ops.aten.mm.default(view_673, permute_434);  permute_434 = None
    permute_435: "f32[1024, 512]" = torch.ops.aten.permute.default(view_673, [1, 0])
    mm_63: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_435, view_416);  permute_435 = view_416 = None
    permute_436: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_63, [1, 0]);  mm_63 = None
    sum_113: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_673, [0], True);  view_673 = None
    view_674: "f32[1024]" = torch.ops.aten.view.default(sum_113, [1024]);  sum_113 = None
    permute_437: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_436, [1, 0]);  permute_436 = None
    view_675: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_62, [1, 512, 4096]);  mm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_330: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_152, 0.5);  add_152 = None
    mul_331: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_415, view_415)
    mul_332: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_331, -0.5);  mul_331 = None
    exp_33: "f32[1, 512, 4096]" = torch.ops.aten.exp.default(mul_332);  mul_332 = None
    mul_333: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_33, 0.3989422804014327);  exp_33 = None
    mul_334: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_415, mul_333);  view_415 = mul_333 = None
    add_230: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_330, mul_334);  mul_330 = mul_334 = None
    mul_335: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_675, add_230);  view_675 = add_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_676: "f32[512, 4096]" = torch.ops.aten.view.default(mul_335, [512, 4096]);  mul_335 = None
    mm_64: "f32[512, 1024]" = torch.ops.aten.mm.default(view_676, permute_438);  permute_438 = None
    permute_439: "f32[4096, 512]" = torch.ops.aten.permute.default(view_676, [1, 0])
    mm_65: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_439, view_414);  permute_439 = view_414 = None
    permute_440: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_65, [1, 0]);  mm_65 = None
    sum_114: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_676, [0], True);  view_676 = None
    view_677: "f32[4096]" = torch.ops.aten.view.default(sum_114, [4096]);  sum_114 = None
    permute_441: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_440, [1, 0]);  permute_440 = None
    view_678: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_64, [1, 512, 1024]);  mm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    mul_337: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_678, primals_302);  primals_302 = None
    mul_338: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_337, 1024)
    sum_115: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_337, [2], True)
    mul_339: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_337, mul_129);  mul_337 = None
    sum_116: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_339, [2], True);  mul_339 = None
    mul_340: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_129, sum_116);  sum_116 = None
    sub_119: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_338, sum_115);  mul_338 = sum_115 = None
    sub_120: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_119, mul_340);  sub_119 = mul_340 = None
    mul_341: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_70, sub_120);  div_70 = sub_120 = None
    mul_342: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_678, mul_129);  mul_129 = None
    sum_117: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_342, [0, 1]);  mul_342 = None
    sum_118: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_678, [0, 1]);  view_678 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    add_231: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_228, mul_341);  add_228 = mul_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_18: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_187, torch.float32);  getitem_187 = None
    mul_343: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_18, 1.1111111111111112);  convert_element_type_18 = None
    mul_344: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_231, mul_343);  mul_343 = None
    clone_52: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_344, memory_format = torch.contiguous_format);  mul_344 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_679: "f32[512, 1024]" = torch.ops.aten.view.default(clone_52, [512, 1024]);  clone_52 = None
    mm_66: "f32[512, 1024]" = torch.ops.aten.mm.default(view_679, permute_442);  permute_442 = None
    permute_443: "f32[1024, 512]" = torch.ops.aten.permute.default(view_679, [1, 0])
    mm_67: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_443, view_412);  permute_443 = view_412 = None
    permute_444: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_67, [1, 0]);  mm_67 = None
    sum_119: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_679, [0], True);  view_679 = None
    view_680: "f32[1024]" = torch.ops.aten.view.default(sum_119, [1024]);  sum_119 = None
    permute_445: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_444, [1, 0]);  permute_444 = None
    view_681: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_66, [1, 512, 1024]);  mm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_682: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_681, [1, 512, 16, 64]);  view_681 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_446: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_682, [0, 2, 1, 3]);  view_682 = None
    
    # No stacktrace found for following nodes
    view_default_66: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_446, [16, 512, 64]);  permute_446 = None
    bmm_default_32: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_default_31, view_default_66);  permute_default_31 = None
    view_default_67: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_default_32, [1, 16, 512, 64]);  bmm_default_32 = None
    bmm_default_33: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_default_66, permute_default_32);  view_default_66 = permute_default_32 = None
    view_default_68: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_default_33, [1, 16, 512, 512]);  bmm_default_33 = None
    mul_tensor_21: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_default_68, mul_tensor_20);  view_default_68 = mul_tensor_20 = None
    clone_default_23: "f32[1, 16, 512, 512]" = torch.ops.aten.clone.default(mul_tensor_21, memory_format = torch.contiguous_format);  mul_tensor_21 = None
    mul_tensor_22: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(clone_default_23, alias_default_11);  clone_default_23 = None
    sum_dim_int_list_11: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_22, [-1], True)
    mul_tensor_23: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_default_11, sum_dim_int_list_11);  alias_default_11 = sum_dim_int_list_11 = None
    sub_tensor_11: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_tensor_22, mul_tensor_23);  mul_tensor_22 = mul_tensor_23 = None
    view_default_69: "f32[16, 512, 512]" = torch.ops.aten.view.default(sub_tensor_11, [16, 512, 512]);  sub_tensor_11 = None
    bmm_default_34: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_default_33, view_default_69);  permute_default_33 = None
    view_default_70: "f32[1, 16, 64, 512]" = torch.ops.aten.view.default(bmm_default_34, [1, 16, 64, 512]);  bmm_default_34 = None
    mul_scalar_22: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(view_default_70, 0.3535533905932738);  view_default_70 = None
    permute_default_35: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(mul_scalar_22, [0, 1, 3, 2]);  mul_scalar_22 = None
    bmm_default_35: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_default_69, permute_default_34);  view_default_69 = permute_default_34 = None
    view_default_71: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_default_35, [1, 16, 512, 64]);  bmm_default_35 = None
    mul_scalar_23: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(view_default_71, 0.3535533905932738);  view_default_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_452: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(mul_scalar_23, [0, 2, 1, 3]);  mul_scalar_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_54: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_452, memory_format = torch.contiguous_format);  permute_452 = None
    view_689: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_54, [1, 512, 1024]);  clone_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_453: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_default_67, [0, 2, 1, 3]);  view_default_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_55: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_453, memory_format = torch.contiguous_format);  permute_453 = None
    view_690: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_55, [1, 512, 1024]);  clone_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_691: "f32[512, 1024]" = torch.ops.aten.view.default(view_690, [512, 1024]);  view_690 = None
    mm_68: "f32[512, 1024]" = torch.ops.aten.mm.default(view_691, permute_454);  permute_454 = None
    permute_455: "f32[1024, 512]" = torch.ops.aten.permute.default(view_691, [1, 0])
    mm_69: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_455, view_396);  permute_455 = None
    permute_456: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_69, [1, 0]);  mm_69 = None
    sum_121: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_691, [0], True);  view_691 = None
    view_692: "f32[1024]" = torch.ops.aten.view.default(sum_121, [1024]);  sum_121 = None
    permute_457: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_456, [1, 0]);  permute_456 = None
    view_693: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_68, [1, 512, 1024]);  mm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_458: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(permute_default_35, [0, 2, 1, 3]);  permute_default_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_694: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_458, [1, 512, 1024]);  permute_458 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_695: "f32[512, 1024]" = torch.ops.aten.view.default(view_694, [512, 1024]);  view_694 = None
    mm_70: "f32[512, 1024]" = torch.ops.aten.mm.default(view_695, permute_459);  permute_459 = None
    permute_460: "f32[1024, 512]" = torch.ops.aten.permute.default(view_695, [1, 0])
    mm_71: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_460, view_396);  permute_460 = None
    permute_461: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_71, [1, 0]);  mm_71 = None
    sum_122: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_695, [0], True);  view_695 = None
    view_696: "f32[1024]" = torch.ops.aten.view.default(sum_122, [1024]);  sum_122 = None
    permute_462: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_461, [1, 0]);  permute_461 = None
    view_697: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_70, [1, 512, 1024]);  mm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_232: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_693, view_697);  view_693 = view_697 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_698: "f32[512, 1024]" = torch.ops.aten.view.default(view_689, [512, 1024]);  view_689 = None
    mm_72: "f32[512, 1024]" = torch.ops.aten.mm.default(view_698, permute_463);  permute_463 = None
    permute_464: "f32[1024, 512]" = torch.ops.aten.permute.default(view_698, [1, 0])
    mm_73: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_464, view_396);  permute_464 = view_396 = None
    permute_465: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_73, [1, 0]);  mm_73 = None
    sum_123: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_698, [0], True);  view_698 = None
    view_699: "f32[1024]" = torch.ops.aten.view.default(sum_123, [1024]);  sum_123 = None
    permute_466: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_465, [1, 0]);  permute_465 = None
    view_700: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_72, [1, 512, 1024]);  mm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    add_233: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_232, view_700);  add_232 = view_700 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    mul_350: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_233, primals_292);  primals_292 = None
    mul_351: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_350, 1024)
    sum_124: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_350, [2], True)
    mul_352: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_350, mul_127);  mul_350 = None
    sum_125: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_352, [2], True);  mul_352 = None
    mul_353: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_127, sum_125);  sum_125 = None
    sub_123: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_351, sum_124);  mul_351 = sum_124 = None
    sub_124: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_123, mul_353);  sub_123 = mul_353 = None
    mul_354: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_72, sub_124);  div_72 = sub_124 = None
    mul_355: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_233, mul_127);  mul_127 = None
    sum_126: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_355, [0, 1]);  mul_355 = None
    sum_127: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_233, [0, 1]);  add_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    add_234: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_231, mul_354);  add_231 = mul_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_20: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_181, torch.float32);  getitem_181 = None
    mul_356: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_20, 1.1111111111111112);  convert_element_type_20 = None
    mul_357: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_234, mul_356);  mul_356 = None
    clone_56: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_357, memory_format = torch.contiguous_format);  mul_357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_701: "f32[512, 1024]" = torch.ops.aten.view.default(clone_56, [512, 1024]);  clone_56 = None
    mm_74: "f32[512, 4096]" = torch.ops.aten.mm.default(view_701, permute_467);  permute_467 = None
    permute_468: "f32[1024, 512]" = torch.ops.aten.permute.default(view_701, [1, 0])
    mm_75: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_468, view_394);  permute_468 = view_394 = None
    permute_469: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_75, [1, 0]);  mm_75 = None
    sum_128: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_701, [0], True);  view_701 = None
    view_702: "f32[1024]" = torch.ops.aten.view.default(sum_128, [1024]);  sum_128 = None
    permute_470: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_469, [1, 0]);  permute_469 = None
    view_703: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_74, [1, 512, 4096]);  mm_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_359: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_144, 0.5);  add_144 = None
    mul_360: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_393, view_393)
    mul_361: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_360, -0.5);  mul_360 = None
    exp_34: "f32[1, 512, 4096]" = torch.ops.aten.exp.default(mul_361);  mul_361 = None
    mul_362: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_34, 0.3989422804014327);  exp_34 = None
    mul_363: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_393, mul_362);  view_393 = mul_362 = None
    add_236: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_359, mul_363);  mul_359 = mul_363 = None
    mul_364: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_703, add_236);  view_703 = add_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_704: "f32[512, 4096]" = torch.ops.aten.view.default(mul_364, [512, 4096]);  mul_364 = None
    mm_76: "f32[512, 1024]" = torch.ops.aten.mm.default(view_704, permute_471);  permute_471 = None
    permute_472: "f32[4096, 512]" = torch.ops.aten.permute.default(view_704, [1, 0])
    mm_77: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_472, view_392);  permute_472 = view_392 = None
    permute_473: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_77, [1, 0]);  mm_77 = None
    sum_129: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_704, [0], True);  view_704 = None
    view_705: "f32[4096]" = torch.ops.aten.view.default(sum_129, [4096]);  sum_129 = None
    permute_474: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_473, [1, 0]);  permute_473 = None
    view_706: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_76, [1, 512, 1024]);  mm_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    mul_366: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_706, primals_286);  primals_286 = None
    mul_367: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_366, 1024)
    sum_130: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_366, [2], True)
    mul_368: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_366, mul_122);  mul_366 = None
    sum_131: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_368, [2], True);  mul_368 = None
    mul_369: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_122, sum_131);  sum_131 = None
    sub_126: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_367, sum_130);  mul_367 = sum_130 = None
    sub_127: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_126, mul_369);  sub_126 = mul_369 = None
    mul_370: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_73, sub_127);  div_73 = sub_127 = None
    mul_371: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_706, mul_122);  mul_122 = None
    sum_132: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_371, [0, 1]);  mul_371 = None
    sum_133: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_706, [0, 1]);  view_706 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    add_237: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_234, mul_370);  add_234 = mul_370 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_21: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_177, torch.float32);  getitem_177 = None
    mul_372: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_21, 1.1111111111111112);  convert_element_type_21 = None
    mul_373: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_237, mul_372);  mul_372 = None
    clone_57: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_373, memory_format = torch.contiguous_format);  mul_373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_707: "f32[512, 1024]" = torch.ops.aten.view.default(clone_57, [512, 1024]);  clone_57 = None
    mm_78: "f32[512, 1024]" = torch.ops.aten.mm.default(view_707, permute_475);  permute_475 = None
    permute_476: "f32[1024, 512]" = torch.ops.aten.permute.default(view_707, [1, 0])
    mm_79: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_476, view_390);  permute_476 = view_390 = None
    permute_477: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_79, [1, 0]);  mm_79 = None
    sum_134: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_707, [0], True);  view_707 = None
    view_708: "f32[1024]" = torch.ops.aten.view.default(sum_134, [1024]);  sum_134 = None
    permute_478: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_477, [1, 0]);  permute_477 = None
    view_709: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_78, [1, 512, 1024]);  mm_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_710: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_709, [1, 512, 16, 64]);  view_709 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_479: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_710, [0, 2, 1, 3]);  view_710 = None
    
    # No stacktrace found for following nodes
    view_default_78: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_479, [16, 512, 64]);  permute_479 = None
    bmm_default_38: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_default_37, view_default_78);  permute_default_37 = None
    view_default_79: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_default_38, [1, 16, 512, 64]);  bmm_default_38 = None
    bmm_default_39: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_default_78, permute_default_38);  view_default_78 = permute_default_38 = None
    view_default_80: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_default_39, [1, 16, 512, 512]);  bmm_default_39 = None
    mul_tensor_25: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_default_80, mul_tensor_24);  view_default_80 = mul_tensor_24 = None
    clone_default_27: "f32[1, 16, 512, 512]" = torch.ops.aten.clone.default(mul_tensor_25, memory_format = torch.contiguous_format);  mul_tensor_25 = None
    mul_tensor_26: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(clone_default_27, alias_default_13);  clone_default_27 = None
    sum_dim_int_list_13: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_26, [-1], True)
    mul_tensor_27: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_default_13, sum_dim_int_list_13);  alias_default_13 = sum_dim_int_list_13 = None
    sub_tensor_13: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_tensor_26, mul_tensor_27);  mul_tensor_26 = mul_tensor_27 = None
    view_default_81: "f32[16, 512, 512]" = torch.ops.aten.view.default(sub_tensor_13, [16, 512, 512]);  sub_tensor_13 = None
    bmm_default_40: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_default_39, view_default_81);  permute_default_39 = None
    view_default_82: "f32[1, 16, 64, 512]" = torch.ops.aten.view.default(bmm_default_40, [1, 16, 64, 512]);  bmm_default_40 = None
    mul_scalar_26: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(view_default_82, 0.3535533905932738);  view_default_82 = None
    permute_default_41: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(mul_scalar_26, [0, 1, 3, 2]);  mul_scalar_26 = None
    bmm_default_41: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_default_81, permute_default_40);  view_default_81 = permute_default_40 = None
    view_default_83: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_default_41, [1, 16, 512, 64]);  bmm_default_41 = None
    mul_scalar_27: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(view_default_83, 0.3535533905932738);  view_default_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_485: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(mul_scalar_27, [0, 2, 1, 3]);  mul_scalar_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_59: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_485, memory_format = torch.contiguous_format);  permute_485 = None
    view_717: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_59, [1, 512, 1024]);  clone_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_486: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_default_79, [0, 2, 1, 3]);  view_default_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_60: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_486, memory_format = torch.contiguous_format);  permute_486 = None
    view_718: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_60, [1, 512, 1024]);  clone_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_719: "f32[512, 1024]" = torch.ops.aten.view.default(view_718, [512, 1024]);  view_718 = None
    mm_80: "f32[512, 1024]" = torch.ops.aten.mm.default(view_719, permute_487);  permute_487 = None
    permute_488: "f32[1024, 512]" = torch.ops.aten.permute.default(view_719, [1, 0])
    mm_81: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_488, view_374);  permute_488 = None
    permute_489: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_81, [1, 0]);  mm_81 = None
    sum_136: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_719, [0], True);  view_719 = None
    view_720: "f32[1024]" = torch.ops.aten.view.default(sum_136, [1024]);  sum_136 = None
    permute_490: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_489, [1, 0]);  permute_489 = None
    view_721: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_80, [1, 512, 1024]);  mm_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_491: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(permute_default_41, [0, 2, 1, 3]);  permute_default_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_722: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_491, [1, 512, 1024]);  permute_491 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_723: "f32[512, 1024]" = torch.ops.aten.view.default(view_722, [512, 1024]);  view_722 = None
    mm_82: "f32[512, 1024]" = torch.ops.aten.mm.default(view_723, permute_492);  permute_492 = None
    permute_493: "f32[1024, 512]" = torch.ops.aten.permute.default(view_723, [1, 0])
    mm_83: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_493, view_374);  permute_493 = None
    permute_494: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_83, [1, 0]);  mm_83 = None
    sum_137: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_723, [0], True);  view_723 = None
    view_724: "f32[1024]" = torch.ops.aten.view.default(sum_137, [1024]);  sum_137 = None
    permute_495: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_494, [1, 0]);  permute_494 = None
    view_725: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_82, [1, 512, 1024]);  mm_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_238: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_721, view_725);  view_721 = view_725 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_726: "f32[512, 1024]" = torch.ops.aten.view.default(view_717, [512, 1024]);  view_717 = None
    mm_84: "f32[512, 1024]" = torch.ops.aten.mm.default(view_726, permute_496);  permute_496 = None
    permute_497: "f32[1024, 512]" = torch.ops.aten.permute.default(view_726, [1, 0])
    mm_85: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_497, view_374);  permute_497 = view_374 = None
    permute_498: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_85, [1, 0]);  mm_85 = None
    sum_138: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_726, [0], True);  view_726 = None
    view_727: "f32[1024]" = torch.ops.aten.view.default(sum_138, [1024]);  sum_138 = None
    permute_499: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_498, [1, 0]);  permute_498 = None
    view_728: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_84, [1, 512, 1024]);  mm_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    add_239: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_238, view_728);  add_238 = view_728 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    mul_379: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_239, primals_276);  primals_276 = None
    mul_380: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_379, 1024)
    sum_139: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_379, [2], True)
    mul_381: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_379, mul_120);  mul_379 = None
    sum_140: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_381, [2], True);  mul_381 = None
    mul_382: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_120, sum_140);  sum_140 = None
    sub_130: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_380, sum_139);  mul_380 = sum_139 = None
    sub_131: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_130, mul_382);  sub_130 = mul_382 = None
    mul_383: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_75, sub_131);  div_75 = sub_131 = None
    mul_384: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_239, mul_120);  mul_120 = None
    sum_141: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_384, [0, 1]);  mul_384 = None
    sum_142: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_239, [0, 1]);  add_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    add_240: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_237, mul_383);  add_237 = mul_383 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_23: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_171, torch.float32);  getitem_171 = None
    mul_385: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_23, 1.1111111111111112);  convert_element_type_23 = None
    mul_386: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_240, mul_385);  mul_385 = None
    clone_61: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_386, memory_format = torch.contiguous_format);  mul_386 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_729: "f32[512, 1024]" = torch.ops.aten.view.default(clone_61, [512, 1024]);  clone_61 = None
    mm_86: "f32[512, 4096]" = torch.ops.aten.mm.default(view_729, permute_500);  permute_500 = None
    permute_501: "f32[1024, 512]" = torch.ops.aten.permute.default(view_729, [1, 0])
    mm_87: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_501, view_372);  permute_501 = view_372 = None
    permute_502: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_87, [1, 0]);  mm_87 = None
    sum_143: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_729, [0], True);  view_729 = None
    view_730: "f32[1024]" = torch.ops.aten.view.default(sum_143, [1024]);  sum_143 = None
    permute_503: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_502, [1, 0]);  permute_502 = None
    view_731: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_86, [1, 512, 4096]);  mm_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_388: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_136, 0.5);  add_136 = None
    mul_389: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_371, view_371)
    mul_390: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_389, -0.5);  mul_389 = None
    exp_35: "f32[1, 512, 4096]" = torch.ops.aten.exp.default(mul_390);  mul_390 = None
    mul_391: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_35, 0.3989422804014327);  exp_35 = None
    mul_392: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_371, mul_391);  view_371 = mul_391 = None
    add_242: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_388, mul_392);  mul_388 = mul_392 = None
    mul_393: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_731, add_242);  view_731 = add_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_732: "f32[512, 4096]" = torch.ops.aten.view.default(mul_393, [512, 4096]);  mul_393 = None
    mm_88: "f32[512, 1024]" = torch.ops.aten.mm.default(view_732, permute_504);  permute_504 = None
    permute_505: "f32[4096, 512]" = torch.ops.aten.permute.default(view_732, [1, 0])
    mm_89: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_505, view_370);  permute_505 = view_370 = None
    permute_506: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_89, [1, 0]);  mm_89 = None
    sum_144: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_732, [0], True);  view_732 = None
    view_733: "f32[4096]" = torch.ops.aten.view.default(sum_144, [4096]);  sum_144 = None
    permute_507: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_506, [1, 0]);  permute_506 = None
    view_734: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_88, [1, 512, 1024]);  mm_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    mul_395: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_734, primals_270);  primals_270 = None
    mul_396: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_395, 1024)
    sum_145: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_395, [2], True)
    mul_397: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_395, mul_115);  mul_395 = None
    sum_146: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_397, [2], True);  mul_397 = None
    mul_398: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_115, sum_146);  sum_146 = None
    sub_133: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_396, sum_145);  mul_396 = sum_145 = None
    sub_134: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_133, mul_398);  sub_133 = mul_398 = None
    mul_399: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_76, sub_134);  div_76 = sub_134 = None
    mul_400: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_734, mul_115);  mul_115 = None
    sum_147: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_400, [0, 1]);  mul_400 = None
    sum_148: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_734, [0, 1]);  view_734 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    add_243: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_240, mul_399);  add_240 = mul_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_24: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_167, torch.float32);  getitem_167 = None
    mul_401: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_24, 1.1111111111111112);  convert_element_type_24 = None
    mul_402: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_243, mul_401);  mul_401 = None
    clone_62: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_402, memory_format = torch.contiguous_format);  mul_402 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_735: "f32[512, 1024]" = torch.ops.aten.view.default(clone_62, [512, 1024]);  clone_62 = None
    mm_90: "f32[512, 1024]" = torch.ops.aten.mm.default(view_735, permute_508);  permute_508 = None
    permute_509: "f32[1024, 512]" = torch.ops.aten.permute.default(view_735, [1, 0])
    mm_91: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_509, view_368);  permute_509 = view_368 = None
    permute_510: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_91, [1, 0]);  mm_91 = None
    sum_149: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_735, [0], True);  view_735 = None
    view_736: "f32[1024]" = torch.ops.aten.view.default(sum_149, [1024]);  sum_149 = None
    permute_511: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_510, [1, 0]);  permute_510 = None
    view_737: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_90, [1, 512, 1024]);  mm_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_738: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_737, [1, 512, 16, 64]);  view_737 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_512: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_738, [0, 2, 1, 3]);  view_738 = None
    
    # No stacktrace found for following nodes
    view_default_90: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_512, [16, 512, 64]);  permute_512 = None
    bmm_default_44: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_default_43, view_default_90);  permute_default_43 = None
    view_default_91: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_default_44, [1, 16, 512, 64]);  bmm_default_44 = None
    bmm_default_45: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_default_90, permute_default_44);  view_default_90 = permute_default_44 = None
    view_default_92: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_default_45, [1, 16, 512, 512]);  bmm_default_45 = None
    mul_tensor_29: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_default_92, mul_tensor_28);  view_default_92 = mul_tensor_28 = None
    clone_default_31: "f32[1, 16, 512, 512]" = torch.ops.aten.clone.default(mul_tensor_29, memory_format = torch.contiguous_format);  mul_tensor_29 = None
    mul_tensor_30: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(clone_default_31, alias_default_15);  clone_default_31 = None
    sum_dim_int_list_15: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_30, [-1], True)
    mul_tensor_31: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_default_15, sum_dim_int_list_15);  alias_default_15 = sum_dim_int_list_15 = None
    sub_tensor_15: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_tensor_30, mul_tensor_31);  mul_tensor_30 = mul_tensor_31 = None
    view_default_93: "f32[16, 512, 512]" = torch.ops.aten.view.default(sub_tensor_15, [16, 512, 512]);  sub_tensor_15 = None
    bmm_default_46: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_default_45, view_default_93);  permute_default_45 = None
    view_default_94: "f32[1, 16, 64, 512]" = torch.ops.aten.view.default(bmm_default_46, [1, 16, 64, 512]);  bmm_default_46 = None
    mul_scalar_30: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(view_default_94, 0.3535533905932738);  view_default_94 = None
    permute_default_47: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(mul_scalar_30, [0, 1, 3, 2]);  mul_scalar_30 = None
    bmm_default_47: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_default_93, permute_default_46);  view_default_93 = permute_default_46 = None
    view_default_95: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_default_47, [1, 16, 512, 64]);  bmm_default_47 = None
    mul_scalar_31: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(view_default_95, 0.3535533905932738);  view_default_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_518: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(mul_scalar_31, [0, 2, 1, 3]);  mul_scalar_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_64: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_518, memory_format = torch.contiguous_format);  permute_518 = None
    view_745: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_64, [1, 512, 1024]);  clone_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_519: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_default_91, [0, 2, 1, 3]);  view_default_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_65: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_519, memory_format = torch.contiguous_format);  permute_519 = None
    view_746: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_65, [1, 512, 1024]);  clone_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_747: "f32[512, 1024]" = torch.ops.aten.view.default(view_746, [512, 1024]);  view_746 = None
    mm_92: "f32[512, 1024]" = torch.ops.aten.mm.default(view_747, permute_520);  permute_520 = None
    permute_521: "f32[1024, 512]" = torch.ops.aten.permute.default(view_747, [1, 0])
    mm_93: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_521, view_352);  permute_521 = None
    permute_522: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_93, [1, 0]);  mm_93 = None
    sum_151: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_747, [0], True);  view_747 = None
    view_748: "f32[1024]" = torch.ops.aten.view.default(sum_151, [1024]);  sum_151 = None
    permute_523: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_522, [1, 0]);  permute_522 = None
    view_749: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_92, [1, 512, 1024]);  mm_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_524: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(permute_default_47, [0, 2, 1, 3]);  permute_default_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_750: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_524, [1, 512, 1024]);  permute_524 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_751: "f32[512, 1024]" = torch.ops.aten.view.default(view_750, [512, 1024]);  view_750 = None
    mm_94: "f32[512, 1024]" = torch.ops.aten.mm.default(view_751, permute_525);  permute_525 = None
    permute_526: "f32[1024, 512]" = torch.ops.aten.permute.default(view_751, [1, 0])
    mm_95: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_526, view_352);  permute_526 = None
    permute_527: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_95, [1, 0]);  mm_95 = None
    sum_152: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_751, [0], True);  view_751 = None
    view_752: "f32[1024]" = torch.ops.aten.view.default(sum_152, [1024]);  sum_152 = None
    permute_528: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_527, [1, 0]);  permute_527 = None
    view_753: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_94, [1, 512, 1024]);  mm_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_244: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_749, view_753);  view_749 = view_753 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_754: "f32[512, 1024]" = torch.ops.aten.view.default(view_745, [512, 1024]);  view_745 = None
    mm_96: "f32[512, 1024]" = torch.ops.aten.mm.default(view_754, permute_529);  permute_529 = None
    permute_530: "f32[1024, 512]" = torch.ops.aten.permute.default(view_754, [1, 0])
    mm_97: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_530, view_352);  permute_530 = view_352 = None
    permute_531: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_97, [1, 0]);  mm_97 = None
    sum_153: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_754, [0], True);  view_754 = None
    view_755: "f32[1024]" = torch.ops.aten.view.default(sum_153, [1024]);  sum_153 = None
    permute_532: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_531, [1, 0]);  permute_531 = None
    view_756: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_96, [1, 512, 1024]);  mm_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    add_245: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_244, view_756);  add_244 = view_756 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    mul_408: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_245, primals_260);  primals_260 = None
    mul_409: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_408, 1024)
    sum_154: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_408, [2], True)
    mul_410: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_408, mul_113);  mul_408 = None
    sum_155: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_410, [2], True);  mul_410 = None
    mul_411: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_113, sum_155);  sum_155 = None
    sub_137: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_409, sum_154);  mul_409 = sum_154 = None
    sub_138: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_137, mul_411);  sub_137 = mul_411 = None
    mul_412: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_78, sub_138);  div_78 = sub_138 = None
    mul_413: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_245, mul_113);  mul_113 = None
    sum_156: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_413, [0, 1]);  mul_413 = None
    sum_157: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_245, [0, 1]);  add_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    add_246: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_243, mul_412);  add_243 = mul_412 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_26: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_161, torch.float32);  getitem_161 = None
    mul_414: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_26, 1.1111111111111112);  convert_element_type_26 = None
    mul_415: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_246, mul_414);  mul_414 = None
    clone_66: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_415, memory_format = torch.contiguous_format);  mul_415 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_757: "f32[512, 1024]" = torch.ops.aten.view.default(clone_66, [512, 1024]);  clone_66 = None
    mm_98: "f32[512, 4096]" = torch.ops.aten.mm.default(view_757, permute_533);  permute_533 = None
    permute_534: "f32[1024, 512]" = torch.ops.aten.permute.default(view_757, [1, 0])
    mm_99: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_534, view_350);  permute_534 = view_350 = None
    permute_535: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_99, [1, 0]);  mm_99 = None
    sum_158: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_757, [0], True);  view_757 = None
    view_758: "f32[1024]" = torch.ops.aten.view.default(sum_158, [1024]);  sum_158 = None
    permute_536: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_535, [1, 0]);  permute_535 = None
    view_759: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_98, [1, 512, 4096]);  mm_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_417: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_128, 0.5);  add_128 = None
    mul_418: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_349, view_349)
    mul_419: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_418, -0.5);  mul_418 = None
    exp_36: "f32[1, 512, 4096]" = torch.ops.aten.exp.default(mul_419);  mul_419 = None
    mul_420: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_36, 0.3989422804014327);  exp_36 = None
    mul_421: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_349, mul_420);  view_349 = mul_420 = None
    add_248: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_417, mul_421);  mul_417 = mul_421 = None
    mul_422: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_759, add_248);  view_759 = add_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_760: "f32[512, 4096]" = torch.ops.aten.view.default(mul_422, [512, 4096]);  mul_422 = None
    mm_100: "f32[512, 1024]" = torch.ops.aten.mm.default(view_760, permute_537);  permute_537 = None
    permute_538: "f32[4096, 512]" = torch.ops.aten.permute.default(view_760, [1, 0])
    mm_101: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_538, view_348);  permute_538 = view_348 = None
    permute_539: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_101, [1, 0]);  mm_101 = None
    sum_159: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_760, [0], True);  view_760 = None
    view_761: "f32[4096]" = torch.ops.aten.view.default(sum_159, [4096]);  sum_159 = None
    permute_540: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_539, [1, 0]);  permute_539 = None
    view_762: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_100, [1, 512, 1024]);  mm_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    mul_424: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_762, primals_254);  primals_254 = None
    mul_425: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_424, 1024)
    sum_160: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_424, [2], True)
    mul_426: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_424, mul_108);  mul_424 = None
    sum_161: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_426, [2], True);  mul_426 = None
    mul_427: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_108, sum_161);  sum_161 = None
    sub_140: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_425, sum_160);  mul_425 = sum_160 = None
    sub_141: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_140, mul_427);  sub_140 = mul_427 = None
    mul_428: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_79, sub_141);  div_79 = sub_141 = None
    mul_429: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_762, mul_108);  mul_108 = None
    sum_162: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_429, [0, 1]);  mul_429 = None
    sum_163: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_762, [0, 1]);  view_762 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    add_249: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_246, mul_428);  add_246 = mul_428 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_27: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_157, torch.float32);  getitem_157 = None
    mul_430: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_27, 1.1111111111111112);  convert_element_type_27 = None
    mul_431: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_249, mul_430);  mul_430 = None
    clone_67: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_431, memory_format = torch.contiguous_format);  mul_431 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_763: "f32[512, 1024]" = torch.ops.aten.view.default(clone_67, [512, 1024]);  clone_67 = None
    mm_102: "f32[512, 1024]" = torch.ops.aten.mm.default(view_763, permute_541);  permute_541 = None
    permute_542: "f32[1024, 512]" = torch.ops.aten.permute.default(view_763, [1, 0])
    mm_103: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_542, view_346);  permute_542 = view_346 = None
    permute_543: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_103, [1, 0]);  mm_103 = None
    sum_164: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_763, [0], True);  view_763 = None
    view_764: "f32[1024]" = torch.ops.aten.view.default(sum_164, [1024]);  sum_164 = None
    permute_544: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_543, [1, 0]);  permute_543 = None
    view_765: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_102, [1, 512, 1024]);  mm_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_766: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_765, [1, 512, 16, 64]);  view_765 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_545: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_766, [0, 2, 1, 3]);  view_766 = None
    
    # No stacktrace found for following nodes
    view_default_102: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_545, [16, 512, 64]);  permute_545 = None
    bmm_default_50: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_default_49, view_default_102);  permute_default_49 = None
    view_default_103: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_default_50, [1, 16, 512, 64]);  bmm_default_50 = None
    bmm_default_51: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_default_102, permute_default_50);  view_default_102 = permute_default_50 = None
    view_default_104: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_default_51, [1, 16, 512, 512]);  bmm_default_51 = None
    mul_tensor_33: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_default_104, mul_tensor_32);  view_default_104 = mul_tensor_32 = None
    clone_default_35: "f32[1, 16, 512, 512]" = torch.ops.aten.clone.default(mul_tensor_33, memory_format = torch.contiguous_format);  mul_tensor_33 = None
    mul_tensor_34: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(clone_default_35, alias_default_17);  clone_default_35 = None
    sum_dim_int_list_17: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_34, [-1], True)
    mul_tensor_35: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_default_17, sum_dim_int_list_17);  alias_default_17 = sum_dim_int_list_17 = None
    sub_tensor_17: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_tensor_34, mul_tensor_35);  mul_tensor_34 = mul_tensor_35 = None
    view_default_105: "f32[16, 512, 512]" = torch.ops.aten.view.default(sub_tensor_17, [16, 512, 512]);  sub_tensor_17 = None
    bmm_default_52: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_default_51, view_default_105);  permute_default_51 = None
    view_default_106: "f32[1, 16, 64, 512]" = torch.ops.aten.view.default(bmm_default_52, [1, 16, 64, 512]);  bmm_default_52 = None
    mul_scalar_34: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(view_default_106, 0.3535533905932738);  view_default_106 = None
    permute_default_53: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(mul_scalar_34, [0, 1, 3, 2]);  mul_scalar_34 = None
    bmm_default_53: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_default_105, permute_default_52);  view_default_105 = permute_default_52 = None
    view_default_107: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_default_53, [1, 16, 512, 64]);  bmm_default_53 = None
    mul_scalar_35: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(view_default_107, 0.3535533905932738);  view_default_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_551: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(mul_scalar_35, [0, 2, 1, 3]);  mul_scalar_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_69: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_551, memory_format = torch.contiguous_format);  permute_551 = None
    view_773: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_69, [1, 512, 1024]);  clone_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_552: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_default_103, [0, 2, 1, 3]);  view_default_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_70: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_552, memory_format = torch.contiguous_format);  permute_552 = None
    view_774: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_70, [1, 512, 1024]);  clone_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_775: "f32[512, 1024]" = torch.ops.aten.view.default(view_774, [512, 1024]);  view_774 = None
    mm_104: "f32[512, 1024]" = torch.ops.aten.mm.default(view_775, permute_553);  permute_553 = None
    permute_554: "f32[1024, 512]" = torch.ops.aten.permute.default(view_775, [1, 0])
    mm_105: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_554, view_330);  permute_554 = None
    permute_555: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_105, [1, 0]);  mm_105 = None
    sum_166: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_775, [0], True);  view_775 = None
    view_776: "f32[1024]" = torch.ops.aten.view.default(sum_166, [1024]);  sum_166 = None
    permute_556: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_555, [1, 0]);  permute_555 = None
    view_777: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_104, [1, 512, 1024]);  mm_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_557: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(permute_default_53, [0, 2, 1, 3]);  permute_default_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_778: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_557, [1, 512, 1024]);  permute_557 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_779: "f32[512, 1024]" = torch.ops.aten.view.default(view_778, [512, 1024]);  view_778 = None
    mm_106: "f32[512, 1024]" = torch.ops.aten.mm.default(view_779, permute_558);  permute_558 = None
    permute_559: "f32[1024, 512]" = torch.ops.aten.permute.default(view_779, [1, 0])
    mm_107: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_559, view_330);  permute_559 = None
    permute_560: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_107, [1, 0]);  mm_107 = None
    sum_167: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_779, [0], True);  view_779 = None
    view_780: "f32[1024]" = torch.ops.aten.view.default(sum_167, [1024]);  sum_167 = None
    permute_561: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_560, [1, 0]);  permute_560 = None
    view_781: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_106, [1, 512, 1024]);  mm_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_250: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_777, view_781);  view_777 = view_781 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_782: "f32[512, 1024]" = torch.ops.aten.view.default(view_773, [512, 1024]);  view_773 = None
    mm_108: "f32[512, 1024]" = torch.ops.aten.mm.default(view_782, permute_562);  permute_562 = None
    permute_563: "f32[1024, 512]" = torch.ops.aten.permute.default(view_782, [1, 0])
    mm_109: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_563, view_330);  permute_563 = view_330 = None
    permute_564: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_109, [1, 0]);  mm_109 = None
    sum_168: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_782, [0], True);  view_782 = None
    view_783: "f32[1024]" = torch.ops.aten.view.default(sum_168, [1024]);  sum_168 = None
    permute_565: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_564, [1, 0]);  permute_564 = None
    view_784: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_108, [1, 512, 1024]);  mm_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    add_251: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_250, view_784);  add_250 = view_784 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    mul_437: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_251, primals_244);  primals_244 = None
    mul_438: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_437, 1024)
    sum_169: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_437, [2], True)
    mul_439: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_437, mul_106);  mul_437 = None
    sum_170: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_439, [2], True);  mul_439 = None
    mul_440: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_106, sum_170);  sum_170 = None
    sub_144: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_438, sum_169);  mul_438 = sum_169 = None
    sub_145: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_144, mul_440);  sub_144 = mul_440 = None
    mul_441: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_81, sub_145);  div_81 = sub_145 = None
    mul_442: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_251, mul_106);  mul_106 = None
    sum_171: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_442, [0, 1]);  mul_442 = None
    sum_172: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_251, [0, 1]);  add_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    add_252: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_249, mul_441);  add_249 = mul_441 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_29: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_151, torch.float32);  getitem_151 = None
    mul_443: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_29, 1.1111111111111112);  convert_element_type_29 = None
    mul_444: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_252, mul_443);  mul_443 = None
    clone_71: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_444, memory_format = torch.contiguous_format);  mul_444 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_785: "f32[512, 1024]" = torch.ops.aten.view.default(clone_71, [512, 1024]);  clone_71 = None
    mm_110: "f32[512, 4096]" = torch.ops.aten.mm.default(view_785, permute_566);  permute_566 = None
    permute_567: "f32[1024, 512]" = torch.ops.aten.permute.default(view_785, [1, 0])
    mm_111: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_567, view_328);  permute_567 = view_328 = None
    permute_568: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_111, [1, 0]);  mm_111 = None
    sum_173: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_785, [0], True);  view_785 = None
    view_786: "f32[1024]" = torch.ops.aten.view.default(sum_173, [1024]);  sum_173 = None
    permute_569: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_568, [1, 0]);  permute_568 = None
    view_787: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_110, [1, 512, 4096]);  mm_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_446: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_120, 0.5);  add_120 = None
    mul_447: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_327, view_327)
    mul_448: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_447, -0.5);  mul_447 = None
    exp_37: "f32[1, 512, 4096]" = torch.ops.aten.exp.default(mul_448);  mul_448 = None
    mul_449: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_37, 0.3989422804014327);  exp_37 = None
    mul_450: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_327, mul_449);  view_327 = mul_449 = None
    add_254: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_446, mul_450);  mul_446 = mul_450 = None
    mul_451: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_787, add_254);  view_787 = add_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_788: "f32[512, 4096]" = torch.ops.aten.view.default(mul_451, [512, 4096]);  mul_451 = None
    mm_112: "f32[512, 1024]" = torch.ops.aten.mm.default(view_788, permute_570);  permute_570 = None
    permute_571: "f32[4096, 512]" = torch.ops.aten.permute.default(view_788, [1, 0])
    mm_113: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_571, view_326);  permute_571 = view_326 = None
    permute_572: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_113, [1, 0]);  mm_113 = None
    sum_174: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_788, [0], True);  view_788 = None
    view_789: "f32[4096]" = torch.ops.aten.view.default(sum_174, [4096]);  sum_174 = None
    permute_573: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_572, [1, 0]);  permute_572 = None
    view_790: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_112, [1, 512, 1024]);  mm_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    mul_453: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_790, primals_238);  primals_238 = None
    mul_454: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_453, 1024)
    sum_175: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_453, [2], True)
    mul_455: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_453, mul_101);  mul_453 = None
    sum_176: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_455, [2], True);  mul_455 = None
    mul_456: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_101, sum_176);  sum_176 = None
    sub_147: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_454, sum_175);  mul_454 = sum_175 = None
    sub_148: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_147, mul_456);  sub_147 = mul_456 = None
    mul_457: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_82, sub_148);  div_82 = sub_148 = None
    mul_458: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_790, mul_101);  mul_101 = None
    sum_177: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_458, [0, 1]);  mul_458 = None
    sum_178: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_790, [0, 1]);  view_790 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    add_255: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_252, mul_457);  add_252 = mul_457 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_30: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_147, torch.float32);  getitem_147 = None
    mul_459: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_30, 1.1111111111111112);  convert_element_type_30 = None
    mul_460: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_255, mul_459);  mul_459 = None
    clone_72: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_460, memory_format = torch.contiguous_format);  mul_460 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_791: "f32[512, 1024]" = torch.ops.aten.view.default(clone_72, [512, 1024]);  clone_72 = None
    mm_114: "f32[512, 1024]" = torch.ops.aten.mm.default(view_791, permute_574);  permute_574 = None
    permute_575: "f32[1024, 512]" = torch.ops.aten.permute.default(view_791, [1, 0])
    mm_115: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_575, view_324);  permute_575 = view_324 = None
    permute_576: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_115, [1, 0]);  mm_115 = None
    sum_179: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_791, [0], True);  view_791 = None
    view_792: "f32[1024]" = torch.ops.aten.view.default(sum_179, [1024]);  sum_179 = None
    permute_577: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_576, [1, 0]);  permute_576 = None
    view_793: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_114, [1, 512, 1024]);  mm_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_794: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_793, [1, 512, 16, 64]);  view_793 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_578: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_794, [0, 2, 1, 3]);  view_794 = None
    
    # No stacktrace found for following nodes
    view_default_114: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_578, [16, 512, 64]);  permute_578 = None
    bmm_default_56: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_default_55, view_default_114);  permute_default_55 = None
    view_default_115: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_default_56, [1, 16, 512, 64]);  bmm_default_56 = None
    bmm_default_57: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_default_114, permute_default_56);  view_default_114 = permute_default_56 = None
    view_default_116: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_default_57, [1, 16, 512, 512]);  bmm_default_57 = None
    mul_tensor_37: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_default_116, mul_tensor_36);  view_default_116 = mul_tensor_36 = None
    clone_default_39: "f32[1, 16, 512, 512]" = torch.ops.aten.clone.default(mul_tensor_37, memory_format = torch.contiguous_format);  mul_tensor_37 = None
    mul_tensor_38: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(clone_default_39, alias_default_19);  clone_default_39 = None
    sum_dim_int_list_19: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_38, [-1], True)
    mul_tensor_39: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_default_19, sum_dim_int_list_19);  alias_default_19 = sum_dim_int_list_19 = None
    sub_tensor_19: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_tensor_38, mul_tensor_39);  mul_tensor_38 = mul_tensor_39 = None
    view_default_117: "f32[16, 512, 512]" = torch.ops.aten.view.default(sub_tensor_19, [16, 512, 512]);  sub_tensor_19 = None
    bmm_default_58: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_default_57, view_default_117);  permute_default_57 = None
    view_default_118: "f32[1, 16, 64, 512]" = torch.ops.aten.view.default(bmm_default_58, [1, 16, 64, 512]);  bmm_default_58 = None
    mul_scalar_38: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(view_default_118, 0.3535533905932738);  view_default_118 = None
    permute_default_59: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(mul_scalar_38, [0, 1, 3, 2]);  mul_scalar_38 = None
    bmm_default_59: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_default_117, permute_default_58);  view_default_117 = permute_default_58 = None
    view_default_119: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_default_59, [1, 16, 512, 64]);  bmm_default_59 = None
    mul_scalar_39: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(view_default_119, 0.3535533905932738);  view_default_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_584: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(mul_scalar_39, [0, 2, 1, 3]);  mul_scalar_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_74: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_584, memory_format = torch.contiguous_format);  permute_584 = None
    view_801: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_74, [1, 512, 1024]);  clone_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_585: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_default_115, [0, 2, 1, 3]);  view_default_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_75: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_585, memory_format = torch.contiguous_format);  permute_585 = None
    view_802: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_75, [1, 512, 1024]);  clone_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_803: "f32[512, 1024]" = torch.ops.aten.view.default(view_802, [512, 1024]);  view_802 = None
    mm_116: "f32[512, 1024]" = torch.ops.aten.mm.default(view_803, permute_586);  permute_586 = None
    permute_587: "f32[1024, 512]" = torch.ops.aten.permute.default(view_803, [1, 0])
    mm_117: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_587, view_308);  permute_587 = None
    permute_588: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_117, [1, 0]);  mm_117 = None
    sum_181: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_803, [0], True);  view_803 = None
    view_804: "f32[1024]" = torch.ops.aten.view.default(sum_181, [1024]);  sum_181 = None
    permute_589: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_588, [1, 0]);  permute_588 = None
    view_805: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_116, [1, 512, 1024]);  mm_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_590: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(permute_default_59, [0, 2, 1, 3]);  permute_default_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_806: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_590, [1, 512, 1024]);  permute_590 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_807: "f32[512, 1024]" = torch.ops.aten.view.default(view_806, [512, 1024]);  view_806 = None
    mm_118: "f32[512, 1024]" = torch.ops.aten.mm.default(view_807, permute_591);  permute_591 = None
    permute_592: "f32[1024, 512]" = torch.ops.aten.permute.default(view_807, [1, 0])
    mm_119: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_592, view_308);  permute_592 = None
    permute_593: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_119, [1, 0]);  mm_119 = None
    sum_182: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_807, [0], True);  view_807 = None
    view_808: "f32[1024]" = torch.ops.aten.view.default(sum_182, [1024]);  sum_182 = None
    permute_594: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_593, [1, 0]);  permute_593 = None
    view_809: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_118, [1, 512, 1024]);  mm_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_256: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_805, view_809);  view_805 = view_809 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_810: "f32[512, 1024]" = torch.ops.aten.view.default(view_801, [512, 1024]);  view_801 = None
    mm_120: "f32[512, 1024]" = torch.ops.aten.mm.default(view_810, permute_595);  permute_595 = None
    permute_596: "f32[1024, 512]" = torch.ops.aten.permute.default(view_810, [1, 0])
    mm_121: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_596, view_308);  permute_596 = view_308 = None
    permute_597: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_121, [1, 0]);  mm_121 = None
    sum_183: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_810, [0], True);  view_810 = None
    view_811: "f32[1024]" = torch.ops.aten.view.default(sum_183, [1024]);  sum_183 = None
    permute_598: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_597, [1, 0]);  permute_597 = None
    view_812: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_120, [1, 512, 1024]);  mm_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    add_257: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_256, view_812);  add_256 = view_812 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    mul_466: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_257, primals_228);  primals_228 = None
    mul_467: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_466, 1024)
    sum_184: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_466, [2], True)
    mul_468: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_466, mul_99);  mul_466 = None
    sum_185: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_468, [2], True);  mul_468 = None
    mul_469: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_99, sum_185);  sum_185 = None
    sub_151: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_467, sum_184);  mul_467 = sum_184 = None
    sub_152: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_151, mul_469);  sub_151 = mul_469 = None
    mul_470: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_84, sub_152);  div_84 = sub_152 = None
    mul_471: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_257, mul_99);  mul_99 = None
    sum_186: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_471, [0, 1]);  mul_471 = None
    sum_187: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_257, [0, 1]);  add_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    add_258: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_255, mul_470);  add_255 = mul_470 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_32: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_141, torch.float32);  getitem_141 = None
    mul_472: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_32, 1.1111111111111112);  convert_element_type_32 = None
    mul_473: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_258, mul_472);  mul_472 = None
    clone_76: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_473, memory_format = torch.contiguous_format);  mul_473 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_813: "f32[512, 1024]" = torch.ops.aten.view.default(clone_76, [512, 1024]);  clone_76 = None
    mm_122: "f32[512, 4096]" = torch.ops.aten.mm.default(view_813, permute_599);  permute_599 = None
    permute_600: "f32[1024, 512]" = torch.ops.aten.permute.default(view_813, [1, 0])
    mm_123: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_600, view_306);  permute_600 = view_306 = None
    permute_601: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_123, [1, 0]);  mm_123 = None
    sum_188: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_813, [0], True);  view_813 = None
    view_814: "f32[1024]" = torch.ops.aten.view.default(sum_188, [1024]);  sum_188 = None
    permute_602: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_601, [1, 0]);  permute_601 = None
    view_815: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_122, [1, 512, 4096]);  mm_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_475: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_112, 0.5);  add_112 = None
    mul_476: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_305, view_305)
    mul_477: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_476, -0.5);  mul_476 = None
    exp_38: "f32[1, 512, 4096]" = torch.ops.aten.exp.default(mul_477);  mul_477 = None
    mul_478: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_38, 0.3989422804014327);  exp_38 = None
    mul_479: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_305, mul_478);  view_305 = mul_478 = None
    add_260: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_475, mul_479);  mul_475 = mul_479 = None
    mul_480: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_815, add_260);  view_815 = add_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_816: "f32[512, 4096]" = torch.ops.aten.view.default(mul_480, [512, 4096]);  mul_480 = None
    mm_124: "f32[512, 1024]" = torch.ops.aten.mm.default(view_816, permute_603);  permute_603 = None
    permute_604: "f32[4096, 512]" = torch.ops.aten.permute.default(view_816, [1, 0])
    mm_125: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_604, view_304);  permute_604 = view_304 = None
    permute_605: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_125, [1, 0]);  mm_125 = None
    sum_189: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_816, [0], True);  view_816 = None
    view_817: "f32[4096]" = torch.ops.aten.view.default(sum_189, [4096]);  sum_189 = None
    permute_606: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_605, [1, 0]);  permute_605 = None
    view_818: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_124, [1, 512, 1024]);  mm_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    mul_482: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_818, primals_222);  primals_222 = None
    mul_483: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_482, 1024)
    sum_190: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_482, [2], True)
    mul_484: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_482, mul_94);  mul_482 = None
    sum_191: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_484, [2], True);  mul_484 = None
    mul_485: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_94, sum_191);  sum_191 = None
    sub_154: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_483, sum_190);  mul_483 = sum_190 = None
    sub_155: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_154, mul_485);  sub_154 = mul_485 = None
    mul_486: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_85, sub_155);  div_85 = sub_155 = None
    mul_487: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_818, mul_94);  mul_94 = None
    sum_192: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_487, [0, 1]);  mul_487 = None
    sum_193: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_818, [0, 1]);  view_818 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    add_261: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_258, mul_486);  add_258 = mul_486 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_33: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_137, torch.float32);  getitem_137 = None
    mul_488: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_33, 1.1111111111111112);  convert_element_type_33 = None
    mul_489: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_261, mul_488);  mul_488 = None
    clone_77: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_489, memory_format = torch.contiguous_format);  mul_489 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_819: "f32[512, 1024]" = torch.ops.aten.view.default(clone_77, [512, 1024]);  clone_77 = None
    mm_126: "f32[512, 1024]" = torch.ops.aten.mm.default(view_819, permute_607);  permute_607 = None
    permute_608: "f32[1024, 512]" = torch.ops.aten.permute.default(view_819, [1, 0])
    mm_127: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_608, view_302);  permute_608 = view_302 = None
    permute_609: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_127, [1, 0]);  mm_127 = None
    sum_194: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_819, [0], True);  view_819 = None
    view_820: "f32[1024]" = torch.ops.aten.view.default(sum_194, [1024]);  sum_194 = None
    permute_610: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_609, [1, 0]);  permute_609 = None
    view_821: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_126, [1, 512, 1024]);  mm_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_822: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_821, [1, 512, 16, 64]);  view_821 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_611: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_822, [0, 2, 1, 3]);  view_822 = None
    
    # No stacktrace found for following nodes
    view_default_126: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_611, [16, 512, 64]);  permute_611 = None
    bmm_default_62: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_default_61, view_default_126);  permute_default_61 = None
    view_default_127: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_default_62, [1, 16, 512, 64]);  bmm_default_62 = None
    bmm_default_63: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_default_126, permute_default_62);  view_default_126 = permute_default_62 = None
    view_default_128: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_default_63, [1, 16, 512, 512]);  bmm_default_63 = None
    mul_tensor_41: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_default_128, mul_tensor_40);  view_default_128 = mul_tensor_40 = None
    clone_default_43: "f32[1, 16, 512, 512]" = torch.ops.aten.clone.default(mul_tensor_41, memory_format = torch.contiguous_format);  mul_tensor_41 = None
    mul_tensor_42: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(clone_default_43, alias_default_21);  clone_default_43 = None
    sum_dim_int_list_21: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_42, [-1], True)
    mul_tensor_43: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_default_21, sum_dim_int_list_21);  alias_default_21 = sum_dim_int_list_21 = None
    sub_tensor_21: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_tensor_42, mul_tensor_43);  mul_tensor_42 = mul_tensor_43 = None
    view_default_129: "f32[16, 512, 512]" = torch.ops.aten.view.default(sub_tensor_21, [16, 512, 512]);  sub_tensor_21 = None
    bmm_default_64: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_default_63, view_default_129);  permute_default_63 = None
    view_default_130: "f32[1, 16, 64, 512]" = torch.ops.aten.view.default(bmm_default_64, [1, 16, 64, 512]);  bmm_default_64 = None
    mul_scalar_42: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(view_default_130, 0.3535533905932738);  view_default_130 = None
    permute_default_65: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(mul_scalar_42, [0, 1, 3, 2]);  mul_scalar_42 = None
    bmm_default_65: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_default_129, permute_default_64);  view_default_129 = permute_default_64 = None
    view_default_131: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_default_65, [1, 16, 512, 64]);  bmm_default_65 = None
    mul_scalar_43: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(view_default_131, 0.3535533905932738);  view_default_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_617: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(mul_scalar_43, [0, 2, 1, 3]);  mul_scalar_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_79: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_617, memory_format = torch.contiguous_format);  permute_617 = None
    view_829: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_79, [1, 512, 1024]);  clone_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_618: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_default_127, [0, 2, 1, 3]);  view_default_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_80: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_618, memory_format = torch.contiguous_format);  permute_618 = None
    view_830: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_80, [1, 512, 1024]);  clone_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_831: "f32[512, 1024]" = torch.ops.aten.view.default(view_830, [512, 1024]);  view_830 = None
    mm_128: "f32[512, 1024]" = torch.ops.aten.mm.default(view_831, permute_619);  permute_619 = None
    permute_620: "f32[1024, 512]" = torch.ops.aten.permute.default(view_831, [1, 0])
    mm_129: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_620, view_286);  permute_620 = None
    permute_621: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_129, [1, 0]);  mm_129 = None
    sum_196: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_831, [0], True);  view_831 = None
    view_832: "f32[1024]" = torch.ops.aten.view.default(sum_196, [1024]);  sum_196 = None
    permute_622: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_621, [1, 0]);  permute_621 = None
    view_833: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_128, [1, 512, 1024]);  mm_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_623: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(permute_default_65, [0, 2, 1, 3]);  permute_default_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_834: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_623, [1, 512, 1024]);  permute_623 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_835: "f32[512, 1024]" = torch.ops.aten.view.default(view_834, [512, 1024]);  view_834 = None
    mm_130: "f32[512, 1024]" = torch.ops.aten.mm.default(view_835, permute_624);  permute_624 = None
    permute_625: "f32[1024, 512]" = torch.ops.aten.permute.default(view_835, [1, 0])
    mm_131: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_625, view_286);  permute_625 = None
    permute_626: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_131, [1, 0]);  mm_131 = None
    sum_197: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_835, [0], True);  view_835 = None
    view_836: "f32[1024]" = torch.ops.aten.view.default(sum_197, [1024]);  sum_197 = None
    permute_627: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_626, [1, 0]);  permute_626 = None
    view_837: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_130, [1, 512, 1024]);  mm_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_262: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_833, view_837);  view_833 = view_837 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_838: "f32[512, 1024]" = torch.ops.aten.view.default(view_829, [512, 1024]);  view_829 = None
    mm_132: "f32[512, 1024]" = torch.ops.aten.mm.default(view_838, permute_628);  permute_628 = None
    permute_629: "f32[1024, 512]" = torch.ops.aten.permute.default(view_838, [1, 0])
    mm_133: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_629, view_286);  permute_629 = view_286 = None
    permute_630: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_133, [1, 0]);  mm_133 = None
    sum_198: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_838, [0], True);  view_838 = None
    view_839: "f32[1024]" = torch.ops.aten.view.default(sum_198, [1024]);  sum_198 = None
    permute_631: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_630, [1, 0]);  permute_630 = None
    view_840: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_132, [1, 512, 1024]);  mm_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    add_263: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_262, view_840);  add_262 = view_840 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    mul_495: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_263, primals_212);  primals_212 = None
    mul_496: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_495, 1024)
    sum_199: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_495, [2], True)
    mul_497: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_495, mul_92);  mul_495 = None
    sum_200: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_497, [2], True);  mul_497 = None
    mul_498: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_92, sum_200);  sum_200 = None
    sub_158: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_496, sum_199);  mul_496 = sum_199 = None
    sub_159: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_158, mul_498);  sub_158 = mul_498 = None
    mul_499: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_87, sub_159);  div_87 = sub_159 = None
    mul_500: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_263, mul_92);  mul_92 = None
    sum_201: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_500, [0, 1]);  mul_500 = None
    sum_202: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_263, [0, 1]);  add_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    add_264: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_261, mul_499);  add_261 = mul_499 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_35: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_131, torch.float32);  getitem_131 = None
    mul_501: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_35, 1.1111111111111112);  convert_element_type_35 = None
    mul_502: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_264, mul_501);  mul_501 = None
    clone_81: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_502, memory_format = torch.contiguous_format);  mul_502 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_841: "f32[512, 1024]" = torch.ops.aten.view.default(clone_81, [512, 1024]);  clone_81 = None
    mm_134: "f32[512, 4096]" = torch.ops.aten.mm.default(view_841, permute_632);  permute_632 = None
    permute_633: "f32[1024, 512]" = torch.ops.aten.permute.default(view_841, [1, 0])
    mm_135: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_633, view_284);  permute_633 = view_284 = None
    permute_634: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_135, [1, 0]);  mm_135 = None
    sum_203: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_841, [0], True);  view_841 = None
    view_842: "f32[1024]" = torch.ops.aten.view.default(sum_203, [1024]);  sum_203 = None
    permute_635: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_634, [1, 0]);  permute_634 = None
    view_843: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_134, [1, 512, 4096]);  mm_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_504: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_104, 0.5);  add_104 = None
    mul_505: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_283, view_283)
    mul_506: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_505, -0.5);  mul_505 = None
    exp_39: "f32[1, 512, 4096]" = torch.ops.aten.exp.default(mul_506);  mul_506 = None
    mul_507: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_39, 0.3989422804014327);  exp_39 = None
    mul_508: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_283, mul_507);  view_283 = mul_507 = None
    add_266: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_504, mul_508);  mul_504 = mul_508 = None
    mul_509: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_843, add_266);  view_843 = add_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_844: "f32[512, 4096]" = torch.ops.aten.view.default(mul_509, [512, 4096]);  mul_509 = None
    mm_136: "f32[512, 1024]" = torch.ops.aten.mm.default(view_844, permute_636);  permute_636 = None
    permute_637: "f32[4096, 512]" = torch.ops.aten.permute.default(view_844, [1, 0])
    mm_137: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_637, view_282);  permute_637 = view_282 = None
    permute_638: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_137, [1, 0]);  mm_137 = None
    sum_204: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_844, [0], True);  view_844 = None
    view_845: "f32[4096]" = torch.ops.aten.view.default(sum_204, [4096]);  sum_204 = None
    permute_639: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_638, [1, 0]);  permute_638 = None
    view_846: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_136, [1, 512, 1024]);  mm_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    mul_511: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_846, primals_206);  primals_206 = None
    mul_512: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_511, 1024)
    sum_205: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_511, [2], True)
    mul_513: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_511, mul_87);  mul_511 = None
    sum_206: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_513, [2], True);  mul_513 = None
    mul_514: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_87, sum_206);  sum_206 = None
    sub_161: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_512, sum_205);  mul_512 = sum_205 = None
    sub_162: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_161, mul_514);  sub_161 = mul_514 = None
    mul_515: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_88, sub_162);  div_88 = sub_162 = None
    mul_516: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_846, mul_87);  mul_87 = None
    sum_207: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_516, [0, 1]);  mul_516 = None
    sum_208: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_846, [0, 1]);  view_846 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    add_267: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_264, mul_515);  add_264 = mul_515 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_36: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_127, torch.float32);  getitem_127 = None
    mul_517: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_36, 1.1111111111111112);  convert_element_type_36 = None
    mul_518: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_267, mul_517);  mul_517 = None
    clone_82: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_518, memory_format = torch.contiguous_format);  mul_518 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_847: "f32[512, 1024]" = torch.ops.aten.view.default(clone_82, [512, 1024]);  clone_82 = None
    mm_138: "f32[512, 1024]" = torch.ops.aten.mm.default(view_847, permute_640);  permute_640 = None
    permute_641: "f32[1024, 512]" = torch.ops.aten.permute.default(view_847, [1, 0])
    mm_139: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_641, view_280);  permute_641 = view_280 = None
    permute_642: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_139, [1, 0]);  mm_139 = None
    sum_209: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_847, [0], True);  view_847 = None
    view_848: "f32[1024]" = torch.ops.aten.view.default(sum_209, [1024]);  sum_209 = None
    permute_643: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_642, [1, 0]);  permute_642 = None
    view_849: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_138, [1, 512, 1024]);  mm_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_850: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_849, [1, 512, 16, 64]);  view_849 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_644: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_850, [0, 2, 1, 3]);  view_850 = None
    
    # No stacktrace found for following nodes
    view_default_138: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_644, [16, 512, 64]);  permute_644 = None
    bmm_default_68: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_default_67, view_default_138);  permute_default_67 = None
    view_default_139: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_default_68, [1, 16, 512, 64]);  bmm_default_68 = None
    bmm_default_69: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_default_138, permute_default_68);  view_default_138 = permute_default_68 = None
    view_default_140: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_default_69, [1, 16, 512, 512]);  bmm_default_69 = None
    mul_tensor_45: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_default_140, mul_tensor_44);  view_default_140 = mul_tensor_44 = None
    clone_default_47: "f32[1, 16, 512, 512]" = torch.ops.aten.clone.default(mul_tensor_45, memory_format = torch.contiguous_format);  mul_tensor_45 = None
    mul_tensor_46: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(clone_default_47, alias_default_23);  clone_default_47 = None
    sum_dim_int_list_23: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_46, [-1], True)
    mul_tensor_47: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_default_23, sum_dim_int_list_23);  alias_default_23 = sum_dim_int_list_23 = None
    sub_tensor_23: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_tensor_46, mul_tensor_47);  mul_tensor_46 = mul_tensor_47 = None
    view_default_141: "f32[16, 512, 512]" = torch.ops.aten.view.default(sub_tensor_23, [16, 512, 512]);  sub_tensor_23 = None
    bmm_default_70: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_default_69, view_default_141);  permute_default_69 = None
    view_default_142: "f32[1, 16, 64, 512]" = torch.ops.aten.view.default(bmm_default_70, [1, 16, 64, 512]);  bmm_default_70 = None
    mul_scalar_46: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(view_default_142, 0.3535533905932738);  view_default_142 = None
    permute_default_71: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(mul_scalar_46, [0, 1, 3, 2]);  mul_scalar_46 = None
    bmm_default_71: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_default_141, permute_default_70);  view_default_141 = permute_default_70 = None
    view_default_143: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_default_71, [1, 16, 512, 64]);  bmm_default_71 = None
    mul_scalar_47: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(view_default_143, 0.3535533905932738);  view_default_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_650: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(mul_scalar_47, [0, 2, 1, 3]);  mul_scalar_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_84: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_650, memory_format = torch.contiguous_format);  permute_650 = None
    view_857: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_84, [1, 512, 1024]);  clone_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_651: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_default_139, [0, 2, 1, 3]);  view_default_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_85: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_651, memory_format = torch.contiguous_format);  permute_651 = None
    view_858: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_85, [1, 512, 1024]);  clone_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_859: "f32[512, 1024]" = torch.ops.aten.view.default(view_858, [512, 1024]);  view_858 = None
    mm_140: "f32[512, 1024]" = torch.ops.aten.mm.default(view_859, permute_652);  permute_652 = None
    permute_653: "f32[1024, 512]" = torch.ops.aten.permute.default(view_859, [1, 0])
    mm_141: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_653, view_264);  permute_653 = None
    permute_654: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_141, [1, 0]);  mm_141 = None
    sum_211: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_859, [0], True);  view_859 = None
    view_860: "f32[1024]" = torch.ops.aten.view.default(sum_211, [1024]);  sum_211 = None
    permute_655: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_654, [1, 0]);  permute_654 = None
    view_861: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_140, [1, 512, 1024]);  mm_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_656: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(permute_default_71, [0, 2, 1, 3]);  permute_default_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_862: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_656, [1, 512, 1024]);  permute_656 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_863: "f32[512, 1024]" = torch.ops.aten.view.default(view_862, [512, 1024]);  view_862 = None
    mm_142: "f32[512, 1024]" = torch.ops.aten.mm.default(view_863, permute_657);  permute_657 = None
    permute_658: "f32[1024, 512]" = torch.ops.aten.permute.default(view_863, [1, 0])
    mm_143: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_658, view_264);  permute_658 = None
    permute_659: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_143, [1, 0]);  mm_143 = None
    sum_212: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_863, [0], True);  view_863 = None
    view_864: "f32[1024]" = torch.ops.aten.view.default(sum_212, [1024]);  sum_212 = None
    permute_660: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_659, [1, 0]);  permute_659 = None
    view_865: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_142, [1, 512, 1024]);  mm_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_268: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_861, view_865);  view_861 = view_865 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_866: "f32[512, 1024]" = torch.ops.aten.view.default(view_857, [512, 1024]);  view_857 = None
    mm_144: "f32[512, 1024]" = torch.ops.aten.mm.default(view_866, permute_661);  permute_661 = None
    permute_662: "f32[1024, 512]" = torch.ops.aten.permute.default(view_866, [1, 0])
    mm_145: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_662, view_264);  permute_662 = view_264 = None
    permute_663: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_145, [1, 0]);  mm_145 = None
    sum_213: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_866, [0], True);  view_866 = None
    view_867: "f32[1024]" = torch.ops.aten.view.default(sum_213, [1024]);  sum_213 = None
    permute_664: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_663, [1, 0]);  permute_663 = None
    view_868: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_144, [1, 512, 1024]);  mm_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    add_269: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_268, view_868);  add_268 = view_868 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    mul_524: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_269, primals_196);  primals_196 = None
    mul_525: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_524, 1024)
    sum_214: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_524, [2], True)
    mul_526: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_524, mul_85);  mul_524 = None
    sum_215: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_526, [2], True);  mul_526 = None
    mul_527: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_85, sum_215);  sum_215 = None
    sub_165: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_525, sum_214);  mul_525 = sum_214 = None
    sub_166: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_165, mul_527);  sub_165 = mul_527 = None
    mul_528: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_90, sub_166);  div_90 = sub_166 = None
    mul_529: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_269, mul_85);  mul_85 = None
    sum_216: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_529, [0, 1]);  mul_529 = None
    sum_217: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_269, [0, 1]);  add_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    add_270: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_267, mul_528);  add_267 = mul_528 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_38: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_121, torch.float32);  getitem_121 = None
    mul_530: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_38, 1.1111111111111112);  convert_element_type_38 = None
    mul_531: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_270, mul_530);  mul_530 = None
    clone_86: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_531, memory_format = torch.contiguous_format);  mul_531 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_869: "f32[512, 1024]" = torch.ops.aten.view.default(clone_86, [512, 1024]);  clone_86 = None
    mm_146: "f32[512, 4096]" = torch.ops.aten.mm.default(view_869, permute_665);  permute_665 = None
    permute_666: "f32[1024, 512]" = torch.ops.aten.permute.default(view_869, [1, 0])
    mm_147: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_666, view_262);  permute_666 = view_262 = None
    permute_667: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_147, [1, 0]);  mm_147 = None
    sum_218: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_869, [0], True);  view_869 = None
    view_870: "f32[1024]" = torch.ops.aten.view.default(sum_218, [1024]);  sum_218 = None
    permute_668: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_667, [1, 0]);  permute_667 = None
    view_871: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_146, [1, 512, 4096]);  mm_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_533: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_96, 0.5);  add_96 = None
    mul_534: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_261, view_261)
    mul_535: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_534, -0.5);  mul_534 = None
    exp_40: "f32[1, 512, 4096]" = torch.ops.aten.exp.default(mul_535);  mul_535 = None
    mul_536: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_40, 0.3989422804014327);  exp_40 = None
    mul_537: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_261, mul_536);  view_261 = mul_536 = None
    add_272: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_533, mul_537);  mul_533 = mul_537 = None
    mul_538: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_871, add_272);  view_871 = add_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_872: "f32[512, 4096]" = torch.ops.aten.view.default(mul_538, [512, 4096]);  mul_538 = None
    mm_148: "f32[512, 1024]" = torch.ops.aten.mm.default(view_872, permute_669);  permute_669 = None
    permute_670: "f32[4096, 512]" = torch.ops.aten.permute.default(view_872, [1, 0])
    mm_149: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_670, view_260);  permute_670 = view_260 = None
    permute_671: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_149, [1, 0]);  mm_149 = None
    sum_219: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_872, [0], True);  view_872 = None
    view_873: "f32[4096]" = torch.ops.aten.view.default(sum_219, [4096]);  sum_219 = None
    permute_672: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_671, [1, 0]);  permute_671 = None
    view_874: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_148, [1, 512, 1024]);  mm_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    mul_540: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_874, primals_190);  primals_190 = None
    mul_541: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_540, 1024)
    sum_220: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_540, [2], True)
    mul_542: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_540, mul_80);  mul_540 = None
    sum_221: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_542, [2], True);  mul_542 = None
    mul_543: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_80, sum_221);  sum_221 = None
    sub_168: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_541, sum_220);  mul_541 = sum_220 = None
    sub_169: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_168, mul_543);  sub_168 = mul_543 = None
    mul_544: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_91, sub_169);  div_91 = sub_169 = None
    mul_545: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_874, mul_80);  mul_80 = None
    sum_222: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_545, [0, 1]);  mul_545 = None
    sum_223: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_874, [0, 1]);  view_874 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    add_273: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_270, mul_544);  add_270 = mul_544 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_39: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_117, torch.float32);  getitem_117 = None
    mul_546: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_39, 1.1111111111111112);  convert_element_type_39 = None
    mul_547: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_273, mul_546);  mul_546 = None
    clone_87: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_547, memory_format = torch.contiguous_format);  mul_547 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_875: "f32[512, 1024]" = torch.ops.aten.view.default(clone_87, [512, 1024]);  clone_87 = None
    mm_150: "f32[512, 1024]" = torch.ops.aten.mm.default(view_875, permute_673);  permute_673 = None
    permute_674: "f32[1024, 512]" = torch.ops.aten.permute.default(view_875, [1, 0])
    mm_151: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_674, view_258);  permute_674 = view_258 = None
    permute_675: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_151, [1, 0]);  mm_151 = None
    sum_224: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_875, [0], True);  view_875 = None
    view_876: "f32[1024]" = torch.ops.aten.view.default(sum_224, [1024]);  sum_224 = None
    permute_676: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_675, [1, 0]);  permute_675 = None
    view_877: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_150, [1, 512, 1024]);  mm_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_878: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_877, [1, 512, 16, 64]);  view_877 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_677: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_878, [0, 2, 1, 3]);  view_878 = None
    
    # No stacktrace found for following nodes
    view_default_150: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_677, [16, 512, 64]);  permute_677 = None
    bmm_default_74: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_default_73, view_default_150);  permute_default_73 = None
    view_default_151: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_default_74, [1, 16, 512, 64]);  bmm_default_74 = None
    bmm_default_75: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_default_150, permute_default_74);  view_default_150 = permute_default_74 = None
    view_default_152: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_default_75, [1, 16, 512, 512]);  bmm_default_75 = None
    mul_tensor_49: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_default_152, mul_tensor_48);  view_default_152 = mul_tensor_48 = None
    clone_default_51: "f32[1, 16, 512, 512]" = torch.ops.aten.clone.default(mul_tensor_49, memory_format = torch.contiguous_format);  mul_tensor_49 = None
    mul_tensor_50: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(clone_default_51, alias_default_25);  clone_default_51 = None
    sum_dim_int_list_25: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_50, [-1], True)
    mul_tensor_51: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_default_25, sum_dim_int_list_25);  alias_default_25 = sum_dim_int_list_25 = None
    sub_tensor_25: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_tensor_50, mul_tensor_51);  mul_tensor_50 = mul_tensor_51 = None
    view_default_153: "f32[16, 512, 512]" = torch.ops.aten.view.default(sub_tensor_25, [16, 512, 512]);  sub_tensor_25 = None
    bmm_default_76: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_default_75, view_default_153);  permute_default_75 = None
    view_default_154: "f32[1, 16, 64, 512]" = torch.ops.aten.view.default(bmm_default_76, [1, 16, 64, 512]);  bmm_default_76 = None
    mul_scalar_50: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(view_default_154, 0.3535533905932738);  view_default_154 = None
    permute_default_77: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(mul_scalar_50, [0, 1, 3, 2]);  mul_scalar_50 = None
    bmm_default_77: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_default_153, permute_default_76);  view_default_153 = permute_default_76 = None
    view_default_155: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_default_77, [1, 16, 512, 64]);  bmm_default_77 = None
    mul_scalar_51: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(view_default_155, 0.3535533905932738);  view_default_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_683: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(mul_scalar_51, [0, 2, 1, 3]);  mul_scalar_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_89: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_683, memory_format = torch.contiguous_format);  permute_683 = None
    view_885: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_89, [1, 512, 1024]);  clone_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_684: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_default_151, [0, 2, 1, 3]);  view_default_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_90: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_684, memory_format = torch.contiguous_format);  permute_684 = None
    view_886: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_90, [1, 512, 1024]);  clone_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_887: "f32[512, 1024]" = torch.ops.aten.view.default(view_886, [512, 1024]);  view_886 = None
    mm_152: "f32[512, 1024]" = torch.ops.aten.mm.default(view_887, permute_685);  permute_685 = None
    permute_686: "f32[1024, 512]" = torch.ops.aten.permute.default(view_887, [1, 0])
    mm_153: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_686, view_242);  permute_686 = None
    permute_687: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_153, [1, 0]);  mm_153 = None
    sum_226: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_887, [0], True);  view_887 = None
    view_888: "f32[1024]" = torch.ops.aten.view.default(sum_226, [1024]);  sum_226 = None
    permute_688: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_687, [1, 0]);  permute_687 = None
    view_889: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_152, [1, 512, 1024]);  mm_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_689: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(permute_default_77, [0, 2, 1, 3]);  permute_default_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_890: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_689, [1, 512, 1024]);  permute_689 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_891: "f32[512, 1024]" = torch.ops.aten.view.default(view_890, [512, 1024]);  view_890 = None
    mm_154: "f32[512, 1024]" = torch.ops.aten.mm.default(view_891, permute_690);  permute_690 = None
    permute_691: "f32[1024, 512]" = torch.ops.aten.permute.default(view_891, [1, 0])
    mm_155: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_691, view_242);  permute_691 = None
    permute_692: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_155, [1, 0]);  mm_155 = None
    sum_227: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_891, [0], True);  view_891 = None
    view_892: "f32[1024]" = torch.ops.aten.view.default(sum_227, [1024]);  sum_227 = None
    permute_693: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_692, [1, 0]);  permute_692 = None
    view_893: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_154, [1, 512, 1024]);  mm_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_274: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_889, view_893);  view_889 = view_893 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_894: "f32[512, 1024]" = torch.ops.aten.view.default(view_885, [512, 1024]);  view_885 = None
    mm_156: "f32[512, 1024]" = torch.ops.aten.mm.default(view_894, permute_694);  permute_694 = None
    permute_695: "f32[1024, 512]" = torch.ops.aten.permute.default(view_894, [1, 0])
    mm_157: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_695, view_242);  permute_695 = view_242 = None
    permute_696: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_157, [1, 0]);  mm_157 = None
    sum_228: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_894, [0], True);  view_894 = None
    view_895: "f32[1024]" = torch.ops.aten.view.default(sum_228, [1024]);  sum_228 = None
    permute_697: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_696, [1, 0]);  permute_696 = None
    view_896: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_156, [1, 512, 1024]);  mm_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    add_275: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_274, view_896);  add_274 = view_896 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    mul_553: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_275, primals_180);  primals_180 = None
    mul_554: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_553, 1024)
    sum_229: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_553, [2], True)
    mul_555: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_553, mul_78);  mul_553 = None
    sum_230: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_555, [2], True);  mul_555 = None
    mul_556: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_78, sum_230);  sum_230 = None
    sub_172: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_554, sum_229);  mul_554 = sum_229 = None
    sub_173: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_172, mul_556);  sub_172 = mul_556 = None
    mul_557: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_93, sub_173);  div_93 = sub_173 = None
    mul_558: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_275, mul_78);  mul_78 = None
    sum_231: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_558, [0, 1]);  mul_558 = None
    sum_232: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_275, [0, 1]);  add_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    add_276: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_273, mul_557);  add_273 = mul_557 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_41: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_111, torch.float32);  getitem_111 = None
    mul_559: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_41, 1.1111111111111112);  convert_element_type_41 = None
    mul_560: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_276, mul_559);  mul_559 = None
    clone_91: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_560, memory_format = torch.contiguous_format);  mul_560 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_897: "f32[512, 1024]" = torch.ops.aten.view.default(clone_91, [512, 1024]);  clone_91 = None
    mm_158: "f32[512, 4096]" = torch.ops.aten.mm.default(view_897, permute_698);  permute_698 = None
    permute_699: "f32[1024, 512]" = torch.ops.aten.permute.default(view_897, [1, 0])
    mm_159: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_699, view_240);  permute_699 = view_240 = None
    permute_700: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_159, [1, 0]);  mm_159 = None
    sum_233: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_897, [0], True);  view_897 = None
    view_898: "f32[1024]" = torch.ops.aten.view.default(sum_233, [1024]);  sum_233 = None
    permute_701: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_700, [1, 0]);  permute_700 = None
    view_899: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_158, [1, 512, 4096]);  mm_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_562: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_88, 0.5);  add_88 = None
    mul_563: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_239, view_239)
    mul_564: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_563, -0.5);  mul_563 = None
    exp_41: "f32[1, 512, 4096]" = torch.ops.aten.exp.default(mul_564);  mul_564 = None
    mul_565: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_41, 0.3989422804014327);  exp_41 = None
    mul_566: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_239, mul_565);  view_239 = mul_565 = None
    add_278: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_562, mul_566);  mul_562 = mul_566 = None
    mul_567: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_899, add_278);  view_899 = add_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_900: "f32[512, 4096]" = torch.ops.aten.view.default(mul_567, [512, 4096]);  mul_567 = None
    mm_160: "f32[512, 1024]" = torch.ops.aten.mm.default(view_900, permute_702);  permute_702 = None
    permute_703: "f32[4096, 512]" = torch.ops.aten.permute.default(view_900, [1, 0])
    mm_161: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_703, view_238);  permute_703 = view_238 = None
    permute_704: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_161, [1, 0]);  mm_161 = None
    sum_234: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_900, [0], True);  view_900 = None
    view_901: "f32[4096]" = torch.ops.aten.view.default(sum_234, [4096]);  sum_234 = None
    permute_705: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_704, [1, 0]);  permute_704 = None
    view_902: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_160, [1, 512, 1024]);  mm_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    mul_569: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_902, primals_174);  primals_174 = None
    mul_570: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_569, 1024)
    sum_235: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_569, [2], True)
    mul_571: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_569, mul_73);  mul_569 = None
    sum_236: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_571, [2], True);  mul_571 = None
    mul_572: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_73, sum_236);  sum_236 = None
    sub_175: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_570, sum_235);  mul_570 = sum_235 = None
    sub_176: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_175, mul_572);  sub_175 = mul_572 = None
    mul_573: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_94, sub_176);  div_94 = sub_176 = None
    mul_574: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_902, mul_73);  mul_73 = None
    sum_237: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_574, [0, 1]);  mul_574 = None
    sum_238: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_902, [0, 1]);  view_902 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    add_279: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_276, mul_573);  add_276 = mul_573 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_42: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_107, torch.float32);  getitem_107 = None
    mul_575: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_42, 1.1111111111111112);  convert_element_type_42 = None
    mul_576: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_279, mul_575);  mul_575 = None
    clone_92: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_576, memory_format = torch.contiguous_format);  mul_576 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_903: "f32[512, 1024]" = torch.ops.aten.view.default(clone_92, [512, 1024]);  clone_92 = None
    mm_162: "f32[512, 1024]" = torch.ops.aten.mm.default(view_903, permute_706);  permute_706 = None
    permute_707: "f32[1024, 512]" = torch.ops.aten.permute.default(view_903, [1, 0])
    mm_163: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_707, view_236);  permute_707 = view_236 = None
    permute_708: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_163, [1, 0]);  mm_163 = None
    sum_239: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_903, [0], True);  view_903 = None
    view_904: "f32[1024]" = torch.ops.aten.view.default(sum_239, [1024]);  sum_239 = None
    permute_709: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_708, [1, 0]);  permute_708 = None
    view_905: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_162, [1, 512, 1024]);  mm_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_906: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_905, [1, 512, 16, 64]);  view_905 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_710: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_906, [0, 2, 1, 3]);  view_906 = None
    
    # No stacktrace found for following nodes
    view_default_162: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_710, [16, 512, 64]);  permute_710 = None
    bmm_default_80: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_default_79, view_default_162);  permute_default_79 = None
    view_default_163: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_default_80, [1, 16, 512, 64]);  bmm_default_80 = None
    bmm_default_81: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_default_162, permute_default_80);  view_default_162 = permute_default_80 = None
    view_default_164: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_default_81, [1, 16, 512, 512]);  bmm_default_81 = None
    mul_tensor_53: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_default_164, mul_tensor_52);  view_default_164 = mul_tensor_52 = None
    clone_default_55: "f32[1, 16, 512, 512]" = torch.ops.aten.clone.default(mul_tensor_53, memory_format = torch.contiguous_format);  mul_tensor_53 = None
    mul_tensor_54: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(clone_default_55, alias_default_27);  clone_default_55 = None
    sum_dim_int_list_27: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_54, [-1], True)
    mul_tensor_55: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_default_27, sum_dim_int_list_27);  alias_default_27 = sum_dim_int_list_27 = None
    sub_tensor_27: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_tensor_54, mul_tensor_55);  mul_tensor_54 = mul_tensor_55 = None
    view_default_165: "f32[16, 512, 512]" = torch.ops.aten.view.default(sub_tensor_27, [16, 512, 512]);  sub_tensor_27 = None
    bmm_default_82: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_default_81, view_default_165);  permute_default_81 = None
    view_default_166: "f32[1, 16, 64, 512]" = torch.ops.aten.view.default(bmm_default_82, [1, 16, 64, 512]);  bmm_default_82 = None
    mul_scalar_54: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(view_default_166, 0.3535533905932738);  view_default_166 = None
    permute_default_83: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(mul_scalar_54, [0, 1, 3, 2]);  mul_scalar_54 = None
    bmm_default_83: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_default_165, permute_default_82);  view_default_165 = permute_default_82 = None
    view_default_167: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_default_83, [1, 16, 512, 64]);  bmm_default_83 = None
    mul_scalar_55: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(view_default_167, 0.3535533905932738);  view_default_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_716: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(mul_scalar_55, [0, 2, 1, 3]);  mul_scalar_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_94: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_716, memory_format = torch.contiguous_format);  permute_716 = None
    view_913: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_94, [1, 512, 1024]);  clone_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_717: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_default_163, [0, 2, 1, 3]);  view_default_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_95: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_717, memory_format = torch.contiguous_format);  permute_717 = None
    view_914: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_95, [1, 512, 1024]);  clone_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_915: "f32[512, 1024]" = torch.ops.aten.view.default(view_914, [512, 1024]);  view_914 = None
    mm_164: "f32[512, 1024]" = torch.ops.aten.mm.default(view_915, permute_718);  permute_718 = None
    permute_719: "f32[1024, 512]" = torch.ops.aten.permute.default(view_915, [1, 0])
    mm_165: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_719, view_220);  permute_719 = None
    permute_720: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_165, [1, 0]);  mm_165 = None
    sum_241: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_915, [0], True);  view_915 = None
    view_916: "f32[1024]" = torch.ops.aten.view.default(sum_241, [1024]);  sum_241 = None
    permute_721: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_720, [1, 0]);  permute_720 = None
    view_917: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_164, [1, 512, 1024]);  mm_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_722: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(permute_default_83, [0, 2, 1, 3]);  permute_default_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_918: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_722, [1, 512, 1024]);  permute_722 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_919: "f32[512, 1024]" = torch.ops.aten.view.default(view_918, [512, 1024]);  view_918 = None
    mm_166: "f32[512, 1024]" = torch.ops.aten.mm.default(view_919, permute_723);  permute_723 = None
    permute_724: "f32[1024, 512]" = torch.ops.aten.permute.default(view_919, [1, 0])
    mm_167: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_724, view_220);  permute_724 = None
    permute_725: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_167, [1, 0]);  mm_167 = None
    sum_242: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_919, [0], True);  view_919 = None
    view_920: "f32[1024]" = torch.ops.aten.view.default(sum_242, [1024]);  sum_242 = None
    permute_726: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_725, [1, 0]);  permute_725 = None
    view_921: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_166, [1, 512, 1024]);  mm_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_280: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_917, view_921);  view_917 = view_921 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_922: "f32[512, 1024]" = torch.ops.aten.view.default(view_913, [512, 1024]);  view_913 = None
    mm_168: "f32[512, 1024]" = torch.ops.aten.mm.default(view_922, permute_727);  permute_727 = None
    permute_728: "f32[1024, 512]" = torch.ops.aten.permute.default(view_922, [1, 0])
    mm_169: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_728, view_220);  permute_728 = view_220 = None
    permute_729: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_169, [1, 0]);  mm_169 = None
    sum_243: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_922, [0], True);  view_922 = None
    view_923: "f32[1024]" = torch.ops.aten.view.default(sum_243, [1024]);  sum_243 = None
    permute_730: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_729, [1, 0]);  permute_729 = None
    view_924: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_168, [1, 512, 1024]);  mm_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    add_281: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_280, view_924);  add_280 = view_924 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    mul_582: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_281, primals_164);  primals_164 = None
    mul_583: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_582, 1024)
    sum_244: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_582, [2], True)
    mul_584: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_582, mul_71);  mul_582 = None
    sum_245: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_584, [2], True);  mul_584 = None
    mul_585: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_71, sum_245);  sum_245 = None
    sub_179: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_583, sum_244);  mul_583 = sum_244 = None
    sub_180: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_179, mul_585);  sub_179 = mul_585 = None
    mul_586: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_96, sub_180);  div_96 = sub_180 = None
    mul_587: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_281, mul_71);  mul_71 = None
    sum_246: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_587, [0, 1]);  mul_587 = None
    sum_247: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_281, [0, 1]);  add_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    add_282: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_279, mul_586);  add_279 = mul_586 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_44: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_101, torch.float32);  getitem_101 = None
    mul_588: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_44, 1.1111111111111112);  convert_element_type_44 = None
    mul_589: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_282, mul_588);  mul_588 = None
    clone_96: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_589, memory_format = torch.contiguous_format);  mul_589 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_925: "f32[512, 1024]" = torch.ops.aten.view.default(clone_96, [512, 1024]);  clone_96 = None
    mm_170: "f32[512, 4096]" = torch.ops.aten.mm.default(view_925, permute_731);  permute_731 = None
    permute_732: "f32[1024, 512]" = torch.ops.aten.permute.default(view_925, [1, 0])
    mm_171: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_732, view_218);  permute_732 = view_218 = None
    permute_733: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_171, [1, 0]);  mm_171 = None
    sum_248: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_925, [0], True);  view_925 = None
    view_926: "f32[1024]" = torch.ops.aten.view.default(sum_248, [1024]);  sum_248 = None
    permute_734: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_733, [1, 0]);  permute_733 = None
    view_927: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_170, [1, 512, 4096]);  mm_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_591: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_80, 0.5);  add_80 = None
    mul_592: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_217, view_217)
    mul_593: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_592, -0.5);  mul_592 = None
    exp_42: "f32[1, 512, 4096]" = torch.ops.aten.exp.default(mul_593);  mul_593 = None
    mul_594: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_42, 0.3989422804014327);  exp_42 = None
    mul_595: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_217, mul_594);  view_217 = mul_594 = None
    add_284: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_591, mul_595);  mul_591 = mul_595 = None
    mul_596: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_927, add_284);  view_927 = add_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_928: "f32[512, 4096]" = torch.ops.aten.view.default(mul_596, [512, 4096]);  mul_596 = None
    mm_172: "f32[512, 1024]" = torch.ops.aten.mm.default(view_928, permute_735);  permute_735 = None
    permute_736: "f32[4096, 512]" = torch.ops.aten.permute.default(view_928, [1, 0])
    mm_173: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_736, view_216);  permute_736 = view_216 = None
    permute_737: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_173, [1, 0]);  mm_173 = None
    sum_249: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_928, [0], True);  view_928 = None
    view_929: "f32[4096]" = torch.ops.aten.view.default(sum_249, [4096]);  sum_249 = None
    permute_738: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_737, [1, 0]);  permute_737 = None
    view_930: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_172, [1, 512, 1024]);  mm_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    mul_598: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_930, primals_158);  primals_158 = None
    mul_599: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_598, 1024)
    sum_250: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_598, [2], True)
    mul_600: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_598, mul_66);  mul_598 = None
    sum_251: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_600, [2], True);  mul_600 = None
    mul_601: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_66, sum_251);  sum_251 = None
    sub_182: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_599, sum_250);  mul_599 = sum_250 = None
    sub_183: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_182, mul_601);  sub_182 = mul_601 = None
    mul_602: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_97, sub_183);  div_97 = sub_183 = None
    mul_603: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_930, mul_66);  mul_66 = None
    sum_252: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_603, [0, 1]);  mul_603 = None
    sum_253: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_930, [0, 1]);  view_930 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    add_285: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_282, mul_602);  add_282 = mul_602 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_45: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_97, torch.float32);  getitem_97 = None
    mul_604: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_45, 1.1111111111111112);  convert_element_type_45 = None
    mul_605: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_285, mul_604);  mul_604 = None
    clone_97: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_605, memory_format = torch.contiguous_format);  mul_605 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_931: "f32[512, 1024]" = torch.ops.aten.view.default(clone_97, [512, 1024]);  clone_97 = None
    mm_174: "f32[512, 1024]" = torch.ops.aten.mm.default(view_931, permute_739);  permute_739 = None
    permute_740: "f32[1024, 512]" = torch.ops.aten.permute.default(view_931, [1, 0])
    mm_175: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_740, view_214);  permute_740 = view_214 = None
    permute_741: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_175, [1, 0]);  mm_175 = None
    sum_254: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_931, [0], True);  view_931 = None
    view_932: "f32[1024]" = torch.ops.aten.view.default(sum_254, [1024]);  sum_254 = None
    permute_742: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_741, [1, 0]);  permute_741 = None
    view_933: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_174, [1, 512, 1024]);  mm_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_934: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_933, [1, 512, 16, 64]);  view_933 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_743: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_934, [0, 2, 1, 3]);  view_934 = None
    
    # No stacktrace found for following nodes
    view_default_174: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_743, [16, 512, 64]);  permute_743 = None
    bmm_default_86: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_default_85, view_default_174);  permute_default_85 = None
    view_default_175: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_default_86, [1, 16, 512, 64]);  bmm_default_86 = None
    bmm_default_87: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_default_174, permute_default_86);  view_default_174 = permute_default_86 = None
    view_default_176: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_default_87, [1, 16, 512, 512]);  bmm_default_87 = None
    mul_tensor_57: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_default_176, mul_tensor_56);  view_default_176 = mul_tensor_56 = None
    clone_default_59: "f32[1, 16, 512, 512]" = torch.ops.aten.clone.default(mul_tensor_57, memory_format = torch.contiguous_format);  mul_tensor_57 = None
    mul_tensor_58: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(clone_default_59, alias_default_29);  clone_default_59 = None
    sum_dim_int_list_29: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_58, [-1], True)
    mul_tensor_59: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_default_29, sum_dim_int_list_29);  alias_default_29 = sum_dim_int_list_29 = None
    sub_tensor_29: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_tensor_58, mul_tensor_59);  mul_tensor_58 = mul_tensor_59 = None
    view_default_177: "f32[16, 512, 512]" = torch.ops.aten.view.default(sub_tensor_29, [16, 512, 512]);  sub_tensor_29 = None
    bmm_default_88: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_default_87, view_default_177);  permute_default_87 = None
    view_default_178: "f32[1, 16, 64, 512]" = torch.ops.aten.view.default(bmm_default_88, [1, 16, 64, 512]);  bmm_default_88 = None
    mul_scalar_58: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(view_default_178, 0.3535533905932738);  view_default_178 = None
    permute_default_89: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(mul_scalar_58, [0, 1, 3, 2]);  mul_scalar_58 = None
    bmm_default_89: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_default_177, permute_default_88);  view_default_177 = permute_default_88 = None
    view_default_179: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_default_89, [1, 16, 512, 64]);  bmm_default_89 = None
    mul_scalar_59: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(view_default_179, 0.3535533905932738);  view_default_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_749: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(mul_scalar_59, [0, 2, 1, 3]);  mul_scalar_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_99: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_749, memory_format = torch.contiguous_format);  permute_749 = None
    view_941: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_99, [1, 512, 1024]);  clone_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_750: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_default_175, [0, 2, 1, 3]);  view_default_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_100: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_750, memory_format = torch.contiguous_format);  permute_750 = None
    view_942: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_100, [1, 512, 1024]);  clone_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_943: "f32[512, 1024]" = torch.ops.aten.view.default(view_942, [512, 1024]);  view_942 = None
    mm_176: "f32[512, 1024]" = torch.ops.aten.mm.default(view_943, permute_751);  permute_751 = None
    permute_752: "f32[1024, 512]" = torch.ops.aten.permute.default(view_943, [1, 0])
    mm_177: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_752, view_198);  permute_752 = None
    permute_753: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_177, [1, 0]);  mm_177 = None
    sum_256: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_943, [0], True);  view_943 = None
    view_944: "f32[1024]" = torch.ops.aten.view.default(sum_256, [1024]);  sum_256 = None
    permute_754: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_753, [1, 0]);  permute_753 = None
    view_945: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_176, [1, 512, 1024]);  mm_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_755: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(permute_default_89, [0, 2, 1, 3]);  permute_default_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_946: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_755, [1, 512, 1024]);  permute_755 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_947: "f32[512, 1024]" = torch.ops.aten.view.default(view_946, [512, 1024]);  view_946 = None
    mm_178: "f32[512, 1024]" = torch.ops.aten.mm.default(view_947, permute_756);  permute_756 = None
    permute_757: "f32[1024, 512]" = torch.ops.aten.permute.default(view_947, [1, 0])
    mm_179: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_757, view_198);  permute_757 = None
    permute_758: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_179, [1, 0]);  mm_179 = None
    sum_257: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_947, [0], True);  view_947 = None
    view_948: "f32[1024]" = torch.ops.aten.view.default(sum_257, [1024]);  sum_257 = None
    permute_759: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_758, [1, 0]);  permute_758 = None
    view_949: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_178, [1, 512, 1024]);  mm_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_286: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_945, view_949);  view_945 = view_949 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_950: "f32[512, 1024]" = torch.ops.aten.view.default(view_941, [512, 1024]);  view_941 = None
    mm_180: "f32[512, 1024]" = torch.ops.aten.mm.default(view_950, permute_760);  permute_760 = None
    permute_761: "f32[1024, 512]" = torch.ops.aten.permute.default(view_950, [1, 0])
    mm_181: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_761, view_198);  permute_761 = view_198 = None
    permute_762: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_181, [1, 0]);  mm_181 = None
    sum_258: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_950, [0], True);  view_950 = None
    view_951: "f32[1024]" = torch.ops.aten.view.default(sum_258, [1024]);  sum_258 = None
    permute_763: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_762, [1, 0]);  permute_762 = None
    view_952: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_180, [1, 512, 1024]);  mm_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    add_287: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_286, view_952);  add_286 = view_952 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    mul_611: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_287, primals_148);  primals_148 = None
    mul_612: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_611, 1024)
    sum_259: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_611, [2], True)
    mul_613: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_611, mul_64);  mul_611 = None
    sum_260: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_613, [2], True);  mul_613 = None
    mul_614: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_64, sum_260);  sum_260 = None
    sub_186: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_612, sum_259);  mul_612 = sum_259 = None
    sub_187: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_186, mul_614);  sub_186 = mul_614 = None
    mul_615: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_99, sub_187);  div_99 = sub_187 = None
    mul_616: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_287, mul_64);  mul_64 = None
    sum_261: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_616, [0, 1]);  mul_616 = None
    sum_262: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_287, [0, 1]);  add_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    add_288: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_285, mul_615);  add_285 = mul_615 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_47: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_91, torch.float32);  getitem_91 = None
    mul_617: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_47, 1.1111111111111112);  convert_element_type_47 = None
    mul_618: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_288, mul_617);  mul_617 = None
    clone_101: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_618, memory_format = torch.contiguous_format);  mul_618 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_953: "f32[512, 1024]" = torch.ops.aten.view.default(clone_101, [512, 1024]);  clone_101 = None
    mm_182: "f32[512, 4096]" = torch.ops.aten.mm.default(view_953, permute_764);  permute_764 = None
    permute_765: "f32[1024, 512]" = torch.ops.aten.permute.default(view_953, [1, 0])
    mm_183: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_765, view_196);  permute_765 = view_196 = None
    permute_766: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_183, [1, 0]);  mm_183 = None
    sum_263: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_953, [0], True);  view_953 = None
    view_954: "f32[1024]" = torch.ops.aten.view.default(sum_263, [1024]);  sum_263 = None
    permute_767: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_766, [1, 0]);  permute_766 = None
    view_955: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_182, [1, 512, 4096]);  mm_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_620: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_72, 0.5);  add_72 = None
    mul_621: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_195, view_195)
    mul_622: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_621, -0.5);  mul_621 = None
    exp_43: "f32[1, 512, 4096]" = torch.ops.aten.exp.default(mul_622);  mul_622 = None
    mul_623: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_43, 0.3989422804014327);  exp_43 = None
    mul_624: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_195, mul_623);  view_195 = mul_623 = None
    add_290: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_620, mul_624);  mul_620 = mul_624 = None
    mul_625: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_955, add_290);  view_955 = add_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_956: "f32[512, 4096]" = torch.ops.aten.view.default(mul_625, [512, 4096]);  mul_625 = None
    mm_184: "f32[512, 1024]" = torch.ops.aten.mm.default(view_956, permute_768);  permute_768 = None
    permute_769: "f32[4096, 512]" = torch.ops.aten.permute.default(view_956, [1, 0])
    mm_185: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_769, view_194);  permute_769 = view_194 = None
    permute_770: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_185, [1, 0]);  mm_185 = None
    sum_264: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_956, [0], True);  view_956 = None
    view_957: "f32[4096]" = torch.ops.aten.view.default(sum_264, [4096]);  sum_264 = None
    permute_771: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_770, [1, 0]);  permute_770 = None
    view_958: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_184, [1, 512, 1024]);  mm_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    mul_627: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_958, primals_142);  primals_142 = None
    mul_628: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_627, 1024)
    sum_265: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_627, [2], True)
    mul_629: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_627, mul_59);  mul_627 = None
    sum_266: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_629, [2], True);  mul_629 = None
    mul_630: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_59, sum_266);  sum_266 = None
    sub_189: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_628, sum_265);  mul_628 = sum_265 = None
    sub_190: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_189, mul_630);  sub_189 = mul_630 = None
    mul_631: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_100, sub_190);  div_100 = sub_190 = None
    mul_632: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_958, mul_59);  mul_59 = None
    sum_267: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_632, [0, 1]);  mul_632 = None
    sum_268: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_958, [0, 1]);  view_958 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    add_291: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_288, mul_631);  add_288 = mul_631 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_48: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_87, torch.float32);  getitem_87 = None
    mul_633: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_48, 1.1111111111111112);  convert_element_type_48 = None
    mul_634: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_291, mul_633);  mul_633 = None
    clone_102: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_634, memory_format = torch.contiguous_format);  mul_634 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_959: "f32[512, 1024]" = torch.ops.aten.view.default(clone_102, [512, 1024]);  clone_102 = None
    mm_186: "f32[512, 1024]" = torch.ops.aten.mm.default(view_959, permute_772);  permute_772 = None
    permute_773: "f32[1024, 512]" = torch.ops.aten.permute.default(view_959, [1, 0])
    mm_187: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_773, view_192);  permute_773 = view_192 = None
    permute_774: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_187, [1, 0]);  mm_187 = None
    sum_269: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_959, [0], True);  view_959 = None
    view_960: "f32[1024]" = torch.ops.aten.view.default(sum_269, [1024]);  sum_269 = None
    permute_775: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_774, [1, 0]);  permute_774 = None
    view_961: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_186, [1, 512, 1024]);  mm_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_962: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_961, [1, 512, 16, 64]);  view_961 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_776: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_962, [0, 2, 1, 3]);  view_962 = None
    
    # No stacktrace found for following nodes
    view_default_186: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_776, [16, 512, 64]);  permute_776 = None
    bmm_default_92: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_default_91, view_default_186);  permute_default_91 = None
    view_default_187: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_default_92, [1, 16, 512, 64]);  bmm_default_92 = None
    bmm_default_93: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_default_186, permute_default_92);  view_default_186 = permute_default_92 = None
    view_default_188: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_default_93, [1, 16, 512, 512]);  bmm_default_93 = None
    mul_tensor_61: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_default_188, mul_tensor_60);  view_default_188 = mul_tensor_60 = None
    clone_default_63: "f32[1, 16, 512, 512]" = torch.ops.aten.clone.default(mul_tensor_61, memory_format = torch.contiguous_format);  mul_tensor_61 = None
    mul_tensor_62: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(clone_default_63, alias_default_31);  clone_default_63 = None
    sum_dim_int_list_31: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_62, [-1], True)
    mul_tensor_63: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_default_31, sum_dim_int_list_31);  alias_default_31 = sum_dim_int_list_31 = None
    sub_tensor_31: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_tensor_62, mul_tensor_63);  mul_tensor_62 = mul_tensor_63 = None
    view_default_189: "f32[16, 512, 512]" = torch.ops.aten.view.default(sub_tensor_31, [16, 512, 512]);  sub_tensor_31 = None
    bmm_default_94: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_default_93, view_default_189);  permute_default_93 = None
    view_default_190: "f32[1, 16, 64, 512]" = torch.ops.aten.view.default(bmm_default_94, [1, 16, 64, 512]);  bmm_default_94 = None
    mul_scalar_62: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(view_default_190, 0.3535533905932738);  view_default_190 = None
    permute_default_95: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(mul_scalar_62, [0, 1, 3, 2]);  mul_scalar_62 = None
    bmm_default_95: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_default_189, permute_default_94);  view_default_189 = permute_default_94 = None
    view_default_191: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_default_95, [1, 16, 512, 64]);  bmm_default_95 = None
    mul_scalar_63: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(view_default_191, 0.3535533905932738);  view_default_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_782: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(mul_scalar_63, [0, 2, 1, 3]);  mul_scalar_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_104: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_782, memory_format = torch.contiguous_format);  permute_782 = None
    view_969: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_104, [1, 512, 1024]);  clone_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_783: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_default_187, [0, 2, 1, 3]);  view_default_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_105: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_783, memory_format = torch.contiguous_format);  permute_783 = None
    view_970: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_105, [1, 512, 1024]);  clone_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_971: "f32[512, 1024]" = torch.ops.aten.view.default(view_970, [512, 1024]);  view_970 = None
    mm_188: "f32[512, 1024]" = torch.ops.aten.mm.default(view_971, permute_784);  permute_784 = None
    permute_785: "f32[1024, 512]" = torch.ops.aten.permute.default(view_971, [1, 0])
    mm_189: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_785, view_176);  permute_785 = None
    permute_786: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_189, [1, 0]);  mm_189 = None
    sum_271: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_971, [0], True);  view_971 = None
    view_972: "f32[1024]" = torch.ops.aten.view.default(sum_271, [1024]);  sum_271 = None
    permute_787: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_786, [1, 0]);  permute_786 = None
    view_973: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_188, [1, 512, 1024]);  mm_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_788: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(permute_default_95, [0, 2, 1, 3]);  permute_default_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_974: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_788, [1, 512, 1024]);  permute_788 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_975: "f32[512, 1024]" = torch.ops.aten.view.default(view_974, [512, 1024]);  view_974 = None
    mm_190: "f32[512, 1024]" = torch.ops.aten.mm.default(view_975, permute_789);  permute_789 = None
    permute_790: "f32[1024, 512]" = torch.ops.aten.permute.default(view_975, [1, 0])
    mm_191: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_790, view_176);  permute_790 = None
    permute_791: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_191, [1, 0]);  mm_191 = None
    sum_272: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_975, [0], True);  view_975 = None
    view_976: "f32[1024]" = torch.ops.aten.view.default(sum_272, [1024]);  sum_272 = None
    permute_792: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_791, [1, 0]);  permute_791 = None
    view_977: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_190, [1, 512, 1024]);  mm_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_292: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_973, view_977);  view_973 = view_977 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_978: "f32[512, 1024]" = torch.ops.aten.view.default(view_969, [512, 1024]);  view_969 = None
    mm_192: "f32[512, 1024]" = torch.ops.aten.mm.default(view_978, permute_793);  permute_793 = None
    permute_794: "f32[1024, 512]" = torch.ops.aten.permute.default(view_978, [1, 0])
    mm_193: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_794, view_176);  permute_794 = view_176 = None
    permute_795: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_193, [1, 0]);  mm_193 = None
    sum_273: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_978, [0], True);  view_978 = None
    view_979: "f32[1024]" = torch.ops.aten.view.default(sum_273, [1024]);  sum_273 = None
    permute_796: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_795, [1, 0]);  permute_795 = None
    view_980: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_192, [1, 512, 1024]);  mm_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    add_293: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_292, view_980);  add_292 = view_980 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    mul_640: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_293, primals_132);  primals_132 = None
    mul_641: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_640, 1024)
    sum_274: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_640, [2], True)
    mul_642: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_640, mul_57);  mul_640 = None
    sum_275: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_642, [2], True);  mul_642 = None
    mul_643: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_57, sum_275);  sum_275 = None
    sub_193: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_641, sum_274);  mul_641 = sum_274 = None
    sub_194: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_193, mul_643);  sub_193 = mul_643 = None
    mul_644: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_102, sub_194);  div_102 = sub_194 = None
    mul_645: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_293, mul_57);  mul_57 = None
    sum_276: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_645, [0, 1]);  mul_645 = None
    sum_277: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_293, [0, 1]);  add_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    add_294: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_291, mul_644);  add_291 = mul_644 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_50: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_81, torch.float32);  getitem_81 = None
    mul_646: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_50, 1.1111111111111112);  convert_element_type_50 = None
    mul_647: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_294, mul_646);  mul_646 = None
    clone_106: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_647, memory_format = torch.contiguous_format);  mul_647 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_981: "f32[512, 1024]" = torch.ops.aten.view.default(clone_106, [512, 1024]);  clone_106 = None
    mm_194: "f32[512, 4096]" = torch.ops.aten.mm.default(view_981, permute_797);  permute_797 = None
    permute_798: "f32[1024, 512]" = torch.ops.aten.permute.default(view_981, [1, 0])
    mm_195: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_798, view_174);  permute_798 = view_174 = None
    permute_799: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_195, [1, 0]);  mm_195 = None
    sum_278: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_981, [0], True);  view_981 = None
    view_982: "f32[1024]" = torch.ops.aten.view.default(sum_278, [1024]);  sum_278 = None
    permute_800: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_799, [1, 0]);  permute_799 = None
    view_983: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_194, [1, 512, 4096]);  mm_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_649: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_64, 0.5);  add_64 = None
    mul_650: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_173, view_173)
    mul_651: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_650, -0.5);  mul_650 = None
    exp_44: "f32[1, 512, 4096]" = torch.ops.aten.exp.default(mul_651);  mul_651 = None
    mul_652: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_44, 0.3989422804014327);  exp_44 = None
    mul_653: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_173, mul_652);  view_173 = mul_652 = None
    add_296: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_649, mul_653);  mul_649 = mul_653 = None
    mul_654: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_983, add_296);  view_983 = add_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_984: "f32[512, 4096]" = torch.ops.aten.view.default(mul_654, [512, 4096]);  mul_654 = None
    mm_196: "f32[512, 1024]" = torch.ops.aten.mm.default(view_984, permute_801);  permute_801 = None
    permute_802: "f32[4096, 512]" = torch.ops.aten.permute.default(view_984, [1, 0])
    mm_197: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_802, view_172);  permute_802 = view_172 = None
    permute_803: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_197, [1, 0]);  mm_197 = None
    sum_279: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_984, [0], True);  view_984 = None
    view_985: "f32[4096]" = torch.ops.aten.view.default(sum_279, [4096]);  sum_279 = None
    permute_804: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_803, [1, 0]);  permute_803 = None
    view_986: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_196, [1, 512, 1024]);  mm_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    mul_656: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_986, primals_126);  primals_126 = None
    mul_657: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_656, 1024)
    sum_280: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_656, [2], True)
    mul_658: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_656, mul_52);  mul_656 = None
    sum_281: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_658, [2], True);  mul_658 = None
    mul_659: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_52, sum_281);  sum_281 = None
    sub_196: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_657, sum_280);  mul_657 = sum_280 = None
    sub_197: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_196, mul_659);  sub_196 = mul_659 = None
    mul_660: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_103, sub_197);  div_103 = sub_197 = None
    mul_661: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_986, mul_52);  mul_52 = None
    sum_282: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_661, [0, 1]);  mul_661 = None
    sum_283: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_986, [0, 1]);  view_986 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    add_297: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_294, mul_660);  add_294 = mul_660 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_51: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_77, torch.float32);  getitem_77 = None
    mul_662: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_51, 1.1111111111111112);  convert_element_type_51 = None
    mul_663: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_297, mul_662);  mul_662 = None
    clone_107: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_663, memory_format = torch.contiguous_format);  mul_663 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_987: "f32[512, 1024]" = torch.ops.aten.view.default(clone_107, [512, 1024]);  clone_107 = None
    mm_198: "f32[512, 1024]" = torch.ops.aten.mm.default(view_987, permute_805);  permute_805 = None
    permute_806: "f32[1024, 512]" = torch.ops.aten.permute.default(view_987, [1, 0])
    mm_199: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_806, view_170);  permute_806 = view_170 = None
    permute_807: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_199, [1, 0]);  mm_199 = None
    sum_284: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_987, [0], True);  view_987 = None
    view_988: "f32[1024]" = torch.ops.aten.view.default(sum_284, [1024]);  sum_284 = None
    permute_808: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_807, [1, 0]);  permute_807 = None
    view_989: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_198, [1, 512, 1024]);  mm_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_990: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_989, [1, 512, 16, 64]);  view_989 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_809: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_990, [0, 2, 1, 3]);  view_990 = None
    
    # No stacktrace found for following nodes
    view_default_198: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_809, [16, 512, 64]);  permute_809 = None
    bmm_default_98: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_default_97, view_default_198);  permute_default_97 = None
    view_default_199: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_default_98, [1, 16, 512, 64]);  bmm_default_98 = None
    bmm_default_99: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_default_198, permute_default_98);  view_default_198 = permute_default_98 = None
    view_default_200: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_default_99, [1, 16, 512, 512]);  bmm_default_99 = None
    mul_tensor_65: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_default_200, mul_tensor_64);  view_default_200 = mul_tensor_64 = None
    clone_default_67: "f32[1, 16, 512, 512]" = torch.ops.aten.clone.default(mul_tensor_65, memory_format = torch.contiguous_format);  mul_tensor_65 = None
    mul_tensor_66: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(clone_default_67, alias_default_33);  clone_default_67 = None
    sum_dim_int_list_33: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_66, [-1], True)
    mul_tensor_67: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_default_33, sum_dim_int_list_33);  alias_default_33 = sum_dim_int_list_33 = None
    sub_tensor_33: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_tensor_66, mul_tensor_67);  mul_tensor_66 = mul_tensor_67 = None
    view_default_201: "f32[16, 512, 512]" = torch.ops.aten.view.default(sub_tensor_33, [16, 512, 512]);  sub_tensor_33 = None
    bmm_default_100: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_default_99, view_default_201);  permute_default_99 = None
    view_default_202: "f32[1, 16, 64, 512]" = torch.ops.aten.view.default(bmm_default_100, [1, 16, 64, 512]);  bmm_default_100 = None
    mul_scalar_66: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(view_default_202, 0.3535533905932738);  view_default_202 = None
    permute_default_101: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(mul_scalar_66, [0, 1, 3, 2]);  mul_scalar_66 = None
    bmm_default_101: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_default_201, permute_default_100);  view_default_201 = permute_default_100 = None
    view_default_203: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_default_101, [1, 16, 512, 64]);  bmm_default_101 = None
    mul_scalar_67: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(view_default_203, 0.3535533905932738);  view_default_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_815: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(mul_scalar_67, [0, 2, 1, 3]);  mul_scalar_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_109: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_815, memory_format = torch.contiguous_format);  permute_815 = None
    view_997: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_109, [1, 512, 1024]);  clone_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_816: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_default_199, [0, 2, 1, 3]);  view_default_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_110: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_816, memory_format = torch.contiguous_format);  permute_816 = None
    view_998: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_110, [1, 512, 1024]);  clone_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_999: "f32[512, 1024]" = torch.ops.aten.view.default(view_998, [512, 1024]);  view_998 = None
    mm_200: "f32[512, 1024]" = torch.ops.aten.mm.default(view_999, permute_817);  permute_817 = None
    permute_818: "f32[1024, 512]" = torch.ops.aten.permute.default(view_999, [1, 0])
    mm_201: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_818, view_154);  permute_818 = None
    permute_819: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_201, [1, 0]);  mm_201 = None
    sum_286: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_999, [0], True);  view_999 = None
    view_1000: "f32[1024]" = torch.ops.aten.view.default(sum_286, [1024]);  sum_286 = None
    permute_820: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_819, [1, 0]);  permute_819 = None
    view_1001: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_200, [1, 512, 1024]);  mm_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_821: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(permute_default_101, [0, 2, 1, 3]);  permute_default_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_1002: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_821, [1, 512, 1024]);  permute_821 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_1003: "f32[512, 1024]" = torch.ops.aten.view.default(view_1002, [512, 1024]);  view_1002 = None
    mm_202: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1003, permute_822);  permute_822 = None
    permute_823: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1003, [1, 0])
    mm_203: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_823, view_154);  permute_823 = None
    permute_824: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_203, [1, 0]);  mm_203 = None
    sum_287: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1003, [0], True);  view_1003 = None
    view_1004: "f32[1024]" = torch.ops.aten.view.default(sum_287, [1024]);  sum_287 = None
    permute_825: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_824, [1, 0]);  permute_824 = None
    view_1005: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_202, [1, 512, 1024]);  mm_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_298: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_1001, view_1005);  view_1001 = view_1005 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_1006: "f32[512, 1024]" = torch.ops.aten.view.default(view_997, [512, 1024]);  view_997 = None
    mm_204: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1006, permute_826);  permute_826 = None
    permute_827: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1006, [1, 0])
    mm_205: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_827, view_154);  permute_827 = view_154 = None
    permute_828: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_205, [1, 0]);  mm_205 = None
    sum_288: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1006, [0], True);  view_1006 = None
    view_1007: "f32[1024]" = torch.ops.aten.view.default(sum_288, [1024]);  sum_288 = None
    permute_829: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_828, [1, 0]);  permute_828 = None
    view_1008: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_204, [1, 512, 1024]);  mm_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    add_299: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_298, view_1008);  add_298 = view_1008 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    mul_669: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_299, primals_116);  primals_116 = None
    mul_670: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_669, 1024)
    sum_289: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_669, [2], True)
    mul_671: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_669, mul_50);  mul_669 = None
    sum_290: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_671, [2], True);  mul_671 = None
    mul_672: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_50, sum_290);  sum_290 = None
    sub_200: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_670, sum_289);  mul_670 = sum_289 = None
    sub_201: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_200, mul_672);  sub_200 = mul_672 = None
    mul_673: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_105, sub_201);  div_105 = sub_201 = None
    mul_674: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_299, mul_50);  mul_50 = None
    sum_291: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_674, [0, 1]);  mul_674 = None
    sum_292: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_299, [0, 1]);  add_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    add_300: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_297, mul_673);  add_297 = mul_673 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_53: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_71, torch.float32);  getitem_71 = None
    mul_675: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_53, 1.1111111111111112);  convert_element_type_53 = None
    mul_676: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_300, mul_675);  mul_675 = None
    clone_111: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_676, memory_format = torch.contiguous_format);  mul_676 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_1009: "f32[512, 1024]" = torch.ops.aten.view.default(clone_111, [512, 1024]);  clone_111 = None
    mm_206: "f32[512, 4096]" = torch.ops.aten.mm.default(view_1009, permute_830);  permute_830 = None
    permute_831: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1009, [1, 0])
    mm_207: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_831, view_152);  permute_831 = view_152 = None
    permute_832: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_207, [1, 0]);  mm_207 = None
    sum_293: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1009, [0], True);  view_1009 = None
    view_1010: "f32[1024]" = torch.ops.aten.view.default(sum_293, [1024]);  sum_293 = None
    permute_833: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_832, [1, 0]);  permute_832 = None
    view_1011: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_206, [1, 512, 4096]);  mm_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_678: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_56, 0.5);  add_56 = None
    mul_679: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_151, view_151)
    mul_680: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_679, -0.5);  mul_679 = None
    exp_45: "f32[1, 512, 4096]" = torch.ops.aten.exp.default(mul_680);  mul_680 = None
    mul_681: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_45, 0.3989422804014327);  exp_45 = None
    mul_682: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_151, mul_681);  view_151 = mul_681 = None
    add_302: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_678, mul_682);  mul_678 = mul_682 = None
    mul_683: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_1011, add_302);  view_1011 = add_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_1012: "f32[512, 4096]" = torch.ops.aten.view.default(mul_683, [512, 4096]);  mul_683 = None
    mm_208: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1012, permute_834);  permute_834 = None
    permute_835: "f32[4096, 512]" = torch.ops.aten.permute.default(view_1012, [1, 0])
    mm_209: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_835, view_150);  permute_835 = view_150 = None
    permute_836: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_209, [1, 0]);  mm_209 = None
    sum_294: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1012, [0], True);  view_1012 = None
    view_1013: "f32[4096]" = torch.ops.aten.view.default(sum_294, [4096]);  sum_294 = None
    permute_837: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_836, [1, 0]);  permute_836 = None
    view_1014: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_208, [1, 512, 1024]);  mm_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    mul_685: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_1014, primals_110);  primals_110 = None
    mul_686: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_685, 1024)
    sum_295: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_685, [2], True)
    mul_687: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_685, mul_45);  mul_685 = None
    sum_296: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_687, [2], True);  mul_687 = None
    mul_688: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_45, sum_296);  sum_296 = None
    sub_203: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_686, sum_295);  mul_686 = sum_295 = None
    sub_204: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_203, mul_688);  sub_203 = mul_688 = None
    mul_689: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_106, sub_204);  div_106 = sub_204 = None
    mul_690: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_1014, mul_45);  mul_45 = None
    sum_297: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_690, [0, 1]);  mul_690 = None
    sum_298: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_1014, [0, 1]);  view_1014 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    add_303: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_300, mul_689);  add_300 = mul_689 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_54: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_67, torch.float32);  getitem_67 = None
    mul_691: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_54, 1.1111111111111112);  convert_element_type_54 = None
    mul_692: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_303, mul_691);  mul_691 = None
    clone_112: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_692, memory_format = torch.contiguous_format);  mul_692 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_1015: "f32[512, 1024]" = torch.ops.aten.view.default(clone_112, [512, 1024]);  clone_112 = None
    mm_210: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1015, permute_838);  permute_838 = None
    permute_839: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1015, [1, 0])
    mm_211: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_839, view_148);  permute_839 = view_148 = None
    permute_840: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_211, [1, 0]);  mm_211 = None
    sum_299: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1015, [0], True);  view_1015 = None
    view_1016: "f32[1024]" = torch.ops.aten.view.default(sum_299, [1024]);  sum_299 = None
    permute_841: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_840, [1, 0]);  permute_840 = None
    view_1017: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_210, [1, 512, 1024]);  mm_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_1018: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_1017, [1, 512, 16, 64]);  view_1017 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_842: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_1018, [0, 2, 1, 3]);  view_1018 = None
    
    # No stacktrace found for following nodes
    view_default_210: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_842, [16, 512, 64]);  permute_842 = None
    bmm_default_104: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_default_103, view_default_210);  permute_default_103 = None
    view_default_211: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_default_104, [1, 16, 512, 64]);  bmm_default_104 = None
    bmm_default_105: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_default_210, permute_default_104);  view_default_210 = permute_default_104 = None
    view_default_212: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_default_105, [1, 16, 512, 512]);  bmm_default_105 = None
    mul_tensor_69: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_default_212, mul_tensor_68);  view_default_212 = mul_tensor_68 = None
    clone_default_71: "f32[1, 16, 512, 512]" = torch.ops.aten.clone.default(mul_tensor_69, memory_format = torch.contiguous_format);  mul_tensor_69 = None
    mul_tensor_70: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(clone_default_71, alias_default_35);  clone_default_71 = None
    sum_dim_int_list_35: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_70, [-1], True)
    mul_tensor_71: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_default_35, sum_dim_int_list_35);  alias_default_35 = sum_dim_int_list_35 = None
    sub_tensor_35: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_tensor_70, mul_tensor_71);  mul_tensor_70 = mul_tensor_71 = None
    view_default_213: "f32[16, 512, 512]" = torch.ops.aten.view.default(sub_tensor_35, [16, 512, 512]);  sub_tensor_35 = None
    bmm_default_106: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_default_105, view_default_213);  permute_default_105 = None
    view_default_214: "f32[1, 16, 64, 512]" = torch.ops.aten.view.default(bmm_default_106, [1, 16, 64, 512]);  bmm_default_106 = None
    mul_scalar_70: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(view_default_214, 0.3535533905932738);  view_default_214 = None
    permute_default_107: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(mul_scalar_70, [0, 1, 3, 2]);  mul_scalar_70 = None
    bmm_default_107: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_default_213, permute_default_106);  view_default_213 = permute_default_106 = None
    view_default_215: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_default_107, [1, 16, 512, 64]);  bmm_default_107 = None
    mul_scalar_71: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(view_default_215, 0.3535533905932738);  view_default_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_848: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(mul_scalar_71, [0, 2, 1, 3]);  mul_scalar_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_114: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_848, memory_format = torch.contiguous_format);  permute_848 = None
    view_1025: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_114, [1, 512, 1024]);  clone_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_849: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_default_211, [0, 2, 1, 3]);  view_default_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_115: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_849, memory_format = torch.contiguous_format);  permute_849 = None
    view_1026: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_115, [1, 512, 1024]);  clone_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_1027: "f32[512, 1024]" = torch.ops.aten.view.default(view_1026, [512, 1024]);  view_1026 = None
    mm_212: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1027, permute_850);  permute_850 = None
    permute_851: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1027, [1, 0])
    mm_213: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_851, view_132);  permute_851 = None
    permute_852: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_213, [1, 0]);  mm_213 = None
    sum_301: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1027, [0], True);  view_1027 = None
    view_1028: "f32[1024]" = torch.ops.aten.view.default(sum_301, [1024]);  sum_301 = None
    permute_853: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_852, [1, 0]);  permute_852 = None
    view_1029: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_212, [1, 512, 1024]);  mm_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_854: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(permute_default_107, [0, 2, 1, 3]);  permute_default_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_1030: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_854, [1, 512, 1024]);  permute_854 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_1031: "f32[512, 1024]" = torch.ops.aten.view.default(view_1030, [512, 1024]);  view_1030 = None
    mm_214: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1031, permute_855);  permute_855 = None
    permute_856: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1031, [1, 0])
    mm_215: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_856, view_132);  permute_856 = None
    permute_857: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_215, [1, 0]);  mm_215 = None
    sum_302: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1031, [0], True);  view_1031 = None
    view_1032: "f32[1024]" = torch.ops.aten.view.default(sum_302, [1024]);  sum_302 = None
    permute_858: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_857, [1, 0]);  permute_857 = None
    view_1033: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_214, [1, 512, 1024]);  mm_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_304: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_1029, view_1033);  view_1029 = view_1033 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_1034: "f32[512, 1024]" = torch.ops.aten.view.default(view_1025, [512, 1024]);  view_1025 = None
    mm_216: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1034, permute_859);  permute_859 = None
    permute_860: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1034, [1, 0])
    mm_217: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_860, view_132);  permute_860 = view_132 = None
    permute_861: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_217, [1, 0]);  mm_217 = None
    sum_303: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1034, [0], True);  view_1034 = None
    view_1035: "f32[1024]" = torch.ops.aten.view.default(sum_303, [1024]);  sum_303 = None
    permute_862: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_861, [1, 0]);  permute_861 = None
    view_1036: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_216, [1, 512, 1024]);  mm_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    add_305: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_304, view_1036);  add_304 = view_1036 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    mul_698: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_305, primals_100);  primals_100 = None
    mul_699: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_698, 1024)
    sum_304: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_698, [2], True)
    mul_700: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_698, mul_43);  mul_698 = None
    sum_305: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_700, [2], True);  mul_700 = None
    mul_701: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_43, sum_305);  sum_305 = None
    sub_207: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_699, sum_304);  mul_699 = sum_304 = None
    sub_208: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_207, mul_701);  sub_207 = mul_701 = None
    mul_702: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_108, sub_208);  div_108 = sub_208 = None
    mul_703: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_305, mul_43);  mul_43 = None
    sum_306: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_703, [0, 1]);  mul_703 = None
    sum_307: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_305, [0, 1]);  add_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    add_306: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_303, mul_702);  add_303 = mul_702 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_56: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_61, torch.float32);  getitem_61 = None
    mul_704: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_56, 1.1111111111111112);  convert_element_type_56 = None
    mul_705: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_306, mul_704);  mul_704 = None
    clone_116: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_705, memory_format = torch.contiguous_format);  mul_705 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_1037: "f32[512, 1024]" = torch.ops.aten.view.default(clone_116, [512, 1024]);  clone_116 = None
    mm_218: "f32[512, 4096]" = torch.ops.aten.mm.default(view_1037, permute_863);  permute_863 = None
    permute_864: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1037, [1, 0])
    mm_219: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_864, view_130);  permute_864 = view_130 = None
    permute_865: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_219, [1, 0]);  mm_219 = None
    sum_308: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1037, [0], True);  view_1037 = None
    view_1038: "f32[1024]" = torch.ops.aten.view.default(sum_308, [1024]);  sum_308 = None
    permute_866: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_865, [1, 0]);  permute_865 = None
    view_1039: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_218, [1, 512, 4096]);  mm_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_707: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_48, 0.5);  add_48 = None
    mul_708: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_129, view_129)
    mul_709: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_708, -0.5);  mul_708 = None
    exp_46: "f32[1, 512, 4096]" = torch.ops.aten.exp.default(mul_709);  mul_709 = None
    mul_710: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_46, 0.3989422804014327);  exp_46 = None
    mul_711: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_129, mul_710);  view_129 = mul_710 = None
    add_308: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_707, mul_711);  mul_707 = mul_711 = None
    mul_712: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_1039, add_308);  view_1039 = add_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_1040: "f32[512, 4096]" = torch.ops.aten.view.default(mul_712, [512, 4096]);  mul_712 = None
    mm_220: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1040, permute_867);  permute_867 = None
    permute_868: "f32[4096, 512]" = torch.ops.aten.permute.default(view_1040, [1, 0])
    mm_221: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_868, view_128);  permute_868 = view_128 = None
    permute_869: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_221, [1, 0]);  mm_221 = None
    sum_309: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1040, [0], True);  view_1040 = None
    view_1041: "f32[4096]" = torch.ops.aten.view.default(sum_309, [4096]);  sum_309 = None
    permute_870: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_869, [1, 0]);  permute_869 = None
    view_1042: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_220, [1, 512, 1024]);  mm_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    mul_714: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_1042, primals_94);  primals_94 = None
    mul_715: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_714, 1024)
    sum_310: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_714, [2], True)
    mul_716: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_714, mul_38);  mul_714 = None
    sum_311: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_716, [2], True);  mul_716 = None
    mul_717: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_38, sum_311);  sum_311 = None
    sub_210: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_715, sum_310);  mul_715 = sum_310 = None
    sub_211: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_210, mul_717);  sub_210 = mul_717 = None
    mul_718: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_109, sub_211);  div_109 = sub_211 = None
    mul_719: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_1042, mul_38);  mul_38 = None
    sum_312: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_719, [0, 1]);  mul_719 = None
    sum_313: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_1042, [0, 1]);  view_1042 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    add_309: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_306, mul_718);  add_306 = mul_718 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_57: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_57, torch.float32);  getitem_57 = None
    mul_720: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_57, 1.1111111111111112);  convert_element_type_57 = None
    mul_721: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_309, mul_720);  mul_720 = None
    clone_117: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_721, memory_format = torch.contiguous_format);  mul_721 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_1043: "f32[512, 1024]" = torch.ops.aten.view.default(clone_117, [512, 1024]);  clone_117 = None
    mm_222: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1043, permute_871);  permute_871 = None
    permute_872: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1043, [1, 0])
    mm_223: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_872, view_126);  permute_872 = view_126 = None
    permute_873: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_223, [1, 0]);  mm_223 = None
    sum_314: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1043, [0], True);  view_1043 = None
    view_1044: "f32[1024]" = torch.ops.aten.view.default(sum_314, [1024]);  sum_314 = None
    permute_874: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_873, [1, 0]);  permute_873 = None
    view_1045: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_222, [1, 512, 1024]);  mm_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_1046: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_1045, [1, 512, 16, 64]);  view_1045 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_875: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_1046, [0, 2, 1, 3]);  view_1046 = None
    
    # No stacktrace found for following nodes
    view_default_222: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_875, [16, 512, 64]);  permute_875 = None
    bmm_default_110: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_default_109, view_default_222);  permute_default_109 = None
    view_default_223: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_default_110, [1, 16, 512, 64]);  bmm_default_110 = None
    bmm_default_111: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_default_222, permute_default_110);  view_default_222 = permute_default_110 = None
    view_default_224: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_default_111, [1, 16, 512, 512]);  bmm_default_111 = None
    mul_tensor_73: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_default_224, mul_tensor_72);  view_default_224 = mul_tensor_72 = None
    clone_default_75: "f32[1, 16, 512, 512]" = torch.ops.aten.clone.default(mul_tensor_73, memory_format = torch.contiguous_format);  mul_tensor_73 = None
    mul_tensor_74: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(clone_default_75, alias_default_37);  clone_default_75 = None
    sum_dim_int_list_37: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_74, [-1], True)
    mul_tensor_75: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_default_37, sum_dim_int_list_37);  alias_default_37 = sum_dim_int_list_37 = None
    sub_tensor_37: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_tensor_74, mul_tensor_75);  mul_tensor_74 = mul_tensor_75 = None
    view_default_225: "f32[16, 512, 512]" = torch.ops.aten.view.default(sub_tensor_37, [16, 512, 512]);  sub_tensor_37 = None
    bmm_default_112: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_default_111, view_default_225);  permute_default_111 = None
    view_default_226: "f32[1, 16, 64, 512]" = torch.ops.aten.view.default(bmm_default_112, [1, 16, 64, 512]);  bmm_default_112 = None
    mul_scalar_74: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(view_default_226, 0.3535533905932738);  view_default_226 = None
    permute_default_113: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(mul_scalar_74, [0, 1, 3, 2]);  mul_scalar_74 = None
    bmm_default_113: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_default_225, permute_default_112);  view_default_225 = permute_default_112 = None
    view_default_227: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_default_113, [1, 16, 512, 64]);  bmm_default_113 = None
    mul_scalar_75: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(view_default_227, 0.3535533905932738);  view_default_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_881: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(mul_scalar_75, [0, 2, 1, 3]);  mul_scalar_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_119: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_881, memory_format = torch.contiguous_format);  permute_881 = None
    view_1053: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_119, [1, 512, 1024]);  clone_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_882: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_default_223, [0, 2, 1, 3]);  view_default_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_120: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_882, memory_format = torch.contiguous_format);  permute_882 = None
    view_1054: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_120, [1, 512, 1024]);  clone_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_1055: "f32[512, 1024]" = torch.ops.aten.view.default(view_1054, [512, 1024]);  view_1054 = None
    mm_224: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1055, permute_883);  permute_883 = None
    permute_884: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1055, [1, 0])
    mm_225: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_884, view_110);  permute_884 = None
    permute_885: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_225, [1, 0]);  mm_225 = None
    sum_316: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1055, [0], True);  view_1055 = None
    view_1056: "f32[1024]" = torch.ops.aten.view.default(sum_316, [1024]);  sum_316 = None
    permute_886: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_885, [1, 0]);  permute_885 = None
    view_1057: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_224, [1, 512, 1024]);  mm_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_887: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(permute_default_113, [0, 2, 1, 3]);  permute_default_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_1058: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_887, [1, 512, 1024]);  permute_887 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_1059: "f32[512, 1024]" = torch.ops.aten.view.default(view_1058, [512, 1024]);  view_1058 = None
    mm_226: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1059, permute_888);  permute_888 = None
    permute_889: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1059, [1, 0])
    mm_227: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_889, view_110);  permute_889 = None
    permute_890: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_227, [1, 0]);  mm_227 = None
    sum_317: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1059, [0], True);  view_1059 = None
    view_1060: "f32[1024]" = torch.ops.aten.view.default(sum_317, [1024]);  sum_317 = None
    permute_891: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_890, [1, 0]);  permute_890 = None
    view_1061: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_226, [1, 512, 1024]);  mm_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_310: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_1057, view_1061);  view_1057 = view_1061 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_1062: "f32[512, 1024]" = torch.ops.aten.view.default(view_1053, [512, 1024]);  view_1053 = None
    mm_228: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1062, permute_892);  permute_892 = None
    permute_893: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1062, [1, 0])
    mm_229: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_893, view_110);  permute_893 = view_110 = None
    permute_894: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_229, [1, 0]);  mm_229 = None
    sum_318: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1062, [0], True);  view_1062 = None
    view_1063: "f32[1024]" = torch.ops.aten.view.default(sum_318, [1024]);  sum_318 = None
    permute_895: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_894, [1, 0]);  permute_894 = None
    view_1064: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_228, [1, 512, 1024]);  mm_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    add_311: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_310, view_1064);  add_310 = view_1064 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    mul_727: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_311, primals_84);  primals_84 = None
    mul_728: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_727, 1024)
    sum_319: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_727, [2], True)
    mul_729: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_727, mul_36);  mul_727 = None
    sum_320: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_729, [2], True);  mul_729 = None
    mul_730: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_36, sum_320);  sum_320 = None
    sub_214: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_728, sum_319);  mul_728 = sum_319 = None
    sub_215: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_214, mul_730);  sub_214 = mul_730 = None
    mul_731: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_111, sub_215);  div_111 = sub_215 = None
    mul_732: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_311, mul_36);  mul_36 = None
    sum_321: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_732, [0, 1]);  mul_732 = None
    sum_322: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_311, [0, 1]);  add_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    add_312: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_309, mul_731);  add_309 = mul_731 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_59: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_51, torch.float32);  getitem_51 = None
    mul_733: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_59, 1.1111111111111112);  convert_element_type_59 = None
    mul_734: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_312, mul_733);  mul_733 = None
    clone_121: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_734, memory_format = torch.contiguous_format);  mul_734 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_1065: "f32[512, 1024]" = torch.ops.aten.view.default(clone_121, [512, 1024]);  clone_121 = None
    mm_230: "f32[512, 4096]" = torch.ops.aten.mm.default(view_1065, permute_896);  permute_896 = None
    permute_897: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1065, [1, 0])
    mm_231: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_897, view_108);  permute_897 = view_108 = None
    permute_898: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_231, [1, 0]);  mm_231 = None
    sum_323: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1065, [0], True);  view_1065 = None
    view_1066: "f32[1024]" = torch.ops.aten.view.default(sum_323, [1024]);  sum_323 = None
    permute_899: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_898, [1, 0]);  permute_898 = None
    view_1067: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_230, [1, 512, 4096]);  mm_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_736: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_40, 0.5);  add_40 = None
    mul_737: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_107, view_107)
    mul_738: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_737, -0.5);  mul_737 = None
    exp_47: "f32[1, 512, 4096]" = torch.ops.aten.exp.default(mul_738);  mul_738 = None
    mul_739: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_47, 0.3989422804014327);  exp_47 = None
    mul_740: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_107, mul_739);  view_107 = mul_739 = None
    add_314: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_736, mul_740);  mul_736 = mul_740 = None
    mul_741: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_1067, add_314);  view_1067 = add_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_1068: "f32[512, 4096]" = torch.ops.aten.view.default(mul_741, [512, 4096]);  mul_741 = None
    mm_232: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1068, permute_900);  permute_900 = None
    permute_901: "f32[4096, 512]" = torch.ops.aten.permute.default(view_1068, [1, 0])
    mm_233: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_901, view_106);  permute_901 = view_106 = None
    permute_902: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_233, [1, 0]);  mm_233 = None
    sum_324: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1068, [0], True);  view_1068 = None
    view_1069: "f32[4096]" = torch.ops.aten.view.default(sum_324, [4096]);  sum_324 = None
    permute_903: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_902, [1, 0]);  permute_902 = None
    view_1070: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_232, [1, 512, 1024]);  mm_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    mul_743: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_1070, primals_78);  primals_78 = None
    mul_744: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_743, 1024)
    sum_325: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_743, [2], True)
    mul_745: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_743, mul_31);  mul_743 = None
    sum_326: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_745, [2], True);  mul_745 = None
    mul_746: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_31, sum_326);  sum_326 = None
    sub_217: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_744, sum_325);  mul_744 = sum_325 = None
    sub_218: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_217, mul_746);  sub_217 = mul_746 = None
    mul_747: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_112, sub_218);  div_112 = sub_218 = None
    mul_748: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_1070, mul_31);  mul_31 = None
    sum_327: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_748, [0, 1]);  mul_748 = None
    sum_328: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_1070, [0, 1]);  view_1070 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    add_315: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_312, mul_747);  add_312 = mul_747 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_60: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_47, torch.float32);  getitem_47 = None
    mul_749: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_60, 1.1111111111111112);  convert_element_type_60 = None
    mul_750: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_315, mul_749);  mul_749 = None
    clone_122: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_750, memory_format = torch.contiguous_format);  mul_750 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_1071: "f32[512, 1024]" = torch.ops.aten.view.default(clone_122, [512, 1024]);  clone_122 = None
    mm_234: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1071, permute_904);  permute_904 = None
    permute_905: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1071, [1, 0])
    mm_235: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_905, view_104);  permute_905 = view_104 = None
    permute_906: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_235, [1, 0]);  mm_235 = None
    sum_329: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1071, [0], True);  view_1071 = None
    view_1072: "f32[1024]" = torch.ops.aten.view.default(sum_329, [1024]);  sum_329 = None
    permute_907: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_906, [1, 0]);  permute_906 = None
    view_1073: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_234, [1, 512, 1024]);  mm_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_1074: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_1073, [1, 512, 16, 64]);  view_1073 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_908: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_1074, [0, 2, 1, 3]);  view_1074 = None
    
    # No stacktrace found for following nodes
    view_default_234: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_908, [16, 512, 64]);  permute_908 = None
    bmm_default_116: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_default_115, view_default_234);  permute_default_115 = None
    view_default_235: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_default_116, [1, 16, 512, 64]);  bmm_default_116 = None
    bmm_default_117: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_default_234, permute_default_116);  view_default_234 = permute_default_116 = None
    view_default_236: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_default_117, [1, 16, 512, 512]);  bmm_default_117 = None
    mul_tensor_77: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_default_236, mul_tensor_76);  view_default_236 = mul_tensor_76 = None
    clone_default_79: "f32[1, 16, 512, 512]" = torch.ops.aten.clone.default(mul_tensor_77, memory_format = torch.contiguous_format);  mul_tensor_77 = None
    mul_tensor_78: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(clone_default_79, alias_default_39);  clone_default_79 = None
    sum_dim_int_list_39: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_78, [-1], True)
    mul_tensor_79: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_default_39, sum_dim_int_list_39);  alias_default_39 = sum_dim_int_list_39 = None
    sub_tensor_39: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_tensor_78, mul_tensor_79);  mul_tensor_78 = mul_tensor_79 = None
    view_default_237: "f32[16, 512, 512]" = torch.ops.aten.view.default(sub_tensor_39, [16, 512, 512]);  sub_tensor_39 = None
    bmm_default_118: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_default_117, view_default_237);  permute_default_117 = None
    view_default_238: "f32[1, 16, 64, 512]" = torch.ops.aten.view.default(bmm_default_118, [1, 16, 64, 512]);  bmm_default_118 = None
    mul_scalar_78: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(view_default_238, 0.3535533905932738);  view_default_238 = None
    permute_default_119: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(mul_scalar_78, [0, 1, 3, 2]);  mul_scalar_78 = None
    bmm_default_119: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_default_237, permute_default_118);  view_default_237 = permute_default_118 = None
    view_default_239: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_default_119, [1, 16, 512, 64]);  bmm_default_119 = None
    mul_scalar_79: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(view_default_239, 0.3535533905932738);  view_default_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_914: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(mul_scalar_79, [0, 2, 1, 3]);  mul_scalar_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_124: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_914, memory_format = torch.contiguous_format);  permute_914 = None
    view_1081: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_124, [1, 512, 1024]);  clone_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_915: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_default_235, [0, 2, 1, 3]);  view_default_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_125: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_915, memory_format = torch.contiguous_format);  permute_915 = None
    view_1082: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_125, [1, 512, 1024]);  clone_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_1083: "f32[512, 1024]" = torch.ops.aten.view.default(view_1082, [512, 1024]);  view_1082 = None
    mm_236: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1083, permute_916);  permute_916 = None
    permute_917: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1083, [1, 0])
    mm_237: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_917, view_88);  permute_917 = None
    permute_918: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_237, [1, 0]);  mm_237 = None
    sum_331: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1083, [0], True);  view_1083 = None
    view_1084: "f32[1024]" = torch.ops.aten.view.default(sum_331, [1024]);  sum_331 = None
    permute_919: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_918, [1, 0]);  permute_918 = None
    view_1085: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_236, [1, 512, 1024]);  mm_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_920: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(permute_default_119, [0, 2, 1, 3]);  permute_default_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_1086: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_920, [1, 512, 1024]);  permute_920 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_1087: "f32[512, 1024]" = torch.ops.aten.view.default(view_1086, [512, 1024]);  view_1086 = None
    mm_238: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1087, permute_921);  permute_921 = None
    permute_922: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1087, [1, 0])
    mm_239: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_922, view_88);  permute_922 = None
    permute_923: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_239, [1, 0]);  mm_239 = None
    sum_332: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1087, [0], True);  view_1087 = None
    view_1088: "f32[1024]" = torch.ops.aten.view.default(sum_332, [1024]);  sum_332 = None
    permute_924: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_923, [1, 0]);  permute_923 = None
    view_1089: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_238, [1, 512, 1024]);  mm_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_316: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_1085, view_1089);  view_1085 = view_1089 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_1090: "f32[512, 1024]" = torch.ops.aten.view.default(view_1081, [512, 1024]);  view_1081 = None
    mm_240: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1090, permute_925);  permute_925 = None
    permute_926: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1090, [1, 0])
    mm_241: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_926, view_88);  permute_926 = view_88 = None
    permute_927: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_241, [1, 0]);  mm_241 = None
    sum_333: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1090, [0], True);  view_1090 = None
    view_1091: "f32[1024]" = torch.ops.aten.view.default(sum_333, [1024]);  sum_333 = None
    permute_928: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_927, [1, 0]);  permute_927 = None
    view_1092: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_240, [1, 512, 1024]);  mm_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    add_317: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_316, view_1092);  add_316 = view_1092 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    mul_756: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_317, primals_68);  primals_68 = None
    mul_757: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_756, 1024)
    sum_334: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_756, [2], True)
    mul_758: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_756, mul_29);  mul_756 = None
    sum_335: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_758, [2], True);  mul_758 = None
    mul_759: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_29, sum_335);  sum_335 = None
    sub_221: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_757, sum_334);  mul_757 = sum_334 = None
    sub_222: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_221, mul_759);  sub_221 = mul_759 = None
    mul_760: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_114, sub_222);  div_114 = sub_222 = None
    mul_761: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_317, mul_29);  mul_29 = None
    sum_336: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_761, [0, 1]);  mul_761 = None
    sum_337: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_317, [0, 1]);  add_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    add_318: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_315, mul_760);  add_315 = mul_760 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_62: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_41, torch.float32);  getitem_41 = None
    mul_762: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_62, 1.1111111111111112);  convert_element_type_62 = None
    mul_763: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_318, mul_762);  mul_762 = None
    clone_126: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_763, memory_format = torch.contiguous_format);  mul_763 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_1093: "f32[512, 1024]" = torch.ops.aten.view.default(clone_126, [512, 1024]);  clone_126 = None
    mm_242: "f32[512, 4096]" = torch.ops.aten.mm.default(view_1093, permute_929);  permute_929 = None
    permute_930: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1093, [1, 0])
    mm_243: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_930, view_86);  permute_930 = view_86 = None
    permute_931: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_243, [1, 0]);  mm_243 = None
    sum_338: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1093, [0], True);  view_1093 = None
    view_1094: "f32[1024]" = torch.ops.aten.view.default(sum_338, [1024]);  sum_338 = None
    permute_932: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_931, [1, 0]);  permute_931 = None
    view_1095: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_242, [1, 512, 4096]);  mm_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_765: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_32, 0.5);  add_32 = None
    mul_766: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_85, view_85)
    mul_767: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_766, -0.5);  mul_766 = None
    exp_48: "f32[1, 512, 4096]" = torch.ops.aten.exp.default(mul_767);  mul_767 = None
    mul_768: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_48, 0.3989422804014327);  exp_48 = None
    mul_769: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_85, mul_768);  view_85 = mul_768 = None
    add_320: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_765, mul_769);  mul_765 = mul_769 = None
    mul_770: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_1095, add_320);  view_1095 = add_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_1096: "f32[512, 4096]" = torch.ops.aten.view.default(mul_770, [512, 4096]);  mul_770 = None
    mm_244: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1096, permute_933);  permute_933 = None
    permute_934: "f32[4096, 512]" = torch.ops.aten.permute.default(view_1096, [1, 0])
    mm_245: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_934, view_84);  permute_934 = view_84 = None
    permute_935: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_245, [1, 0]);  mm_245 = None
    sum_339: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1096, [0], True);  view_1096 = None
    view_1097: "f32[4096]" = torch.ops.aten.view.default(sum_339, [4096]);  sum_339 = None
    permute_936: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_935, [1, 0]);  permute_935 = None
    view_1098: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_244, [1, 512, 1024]);  mm_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    mul_772: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_1098, primals_62);  primals_62 = None
    mul_773: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_772, 1024)
    sum_340: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_772, [2], True)
    mul_774: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_772, mul_24);  mul_772 = None
    sum_341: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_774, [2], True);  mul_774 = None
    mul_775: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_24, sum_341);  sum_341 = None
    sub_224: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_773, sum_340);  mul_773 = sum_340 = None
    sub_225: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_224, mul_775);  sub_224 = mul_775 = None
    mul_776: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_115, sub_225);  div_115 = sub_225 = None
    mul_777: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_1098, mul_24);  mul_24 = None
    sum_342: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_777, [0, 1]);  mul_777 = None
    sum_343: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_1098, [0, 1]);  view_1098 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    add_321: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_318, mul_776);  add_318 = mul_776 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_63: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_37, torch.float32);  getitem_37 = None
    mul_778: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_63, 1.1111111111111112);  convert_element_type_63 = None
    mul_779: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_321, mul_778);  mul_778 = None
    clone_127: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_779, memory_format = torch.contiguous_format);  mul_779 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_1099: "f32[512, 1024]" = torch.ops.aten.view.default(clone_127, [512, 1024]);  clone_127 = None
    mm_246: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1099, permute_937);  permute_937 = None
    permute_938: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1099, [1, 0])
    mm_247: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_938, view_82);  permute_938 = view_82 = None
    permute_939: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_247, [1, 0]);  mm_247 = None
    sum_344: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1099, [0], True);  view_1099 = None
    view_1100: "f32[1024]" = torch.ops.aten.view.default(sum_344, [1024]);  sum_344 = None
    permute_940: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_939, [1, 0]);  permute_939 = None
    view_1101: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_246, [1, 512, 1024]);  mm_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_1102: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_1101, [1, 512, 16, 64]);  view_1101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_941: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_1102, [0, 2, 1, 3]);  view_1102 = None
    
    # No stacktrace found for following nodes
    view_default_246: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_941, [16, 512, 64]);  permute_941 = None
    bmm_default_122: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_default_121, view_default_246);  permute_default_121 = None
    view_default_247: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_default_122, [1, 16, 512, 64]);  bmm_default_122 = None
    bmm_default_123: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_default_246, permute_default_122);  view_default_246 = permute_default_122 = None
    view_default_248: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_default_123, [1, 16, 512, 512]);  bmm_default_123 = None
    mul_tensor_81: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_default_248, mul_tensor_80);  view_default_248 = mul_tensor_80 = None
    clone_default_83: "f32[1, 16, 512, 512]" = torch.ops.aten.clone.default(mul_tensor_81, memory_format = torch.contiguous_format);  mul_tensor_81 = None
    mul_tensor_82: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(clone_default_83, alias_default_41);  clone_default_83 = None
    sum_dim_int_list_41: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_82, [-1], True)
    mul_tensor_83: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_default_41, sum_dim_int_list_41);  alias_default_41 = sum_dim_int_list_41 = None
    sub_tensor_41: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_tensor_82, mul_tensor_83);  mul_tensor_82 = mul_tensor_83 = None
    view_default_249: "f32[16, 512, 512]" = torch.ops.aten.view.default(sub_tensor_41, [16, 512, 512]);  sub_tensor_41 = None
    bmm_default_124: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_default_123, view_default_249);  permute_default_123 = None
    view_default_250: "f32[1, 16, 64, 512]" = torch.ops.aten.view.default(bmm_default_124, [1, 16, 64, 512]);  bmm_default_124 = None
    mul_scalar_82: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(view_default_250, 0.3535533905932738);  view_default_250 = None
    permute_default_125: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(mul_scalar_82, [0, 1, 3, 2]);  mul_scalar_82 = None
    bmm_default_125: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_default_249, permute_default_124);  view_default_249 = permute_default_124 = None
    view_default_251: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_default_125, [1, 16, 512, 64]);  bmm_default_125 = None
    mul_scalar_83: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(view_default_251, 0.3535533905932738);  view_default_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_947: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(mul_scalar_83, [0, 2, 1, 3]);  mul_scalar_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_129: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_947, memory_format = torch.contiguous_format);  permute_947 = None
    view_1109: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_129, [1, 512, 1024]);  clone_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_948: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_default_247, [0, 2, 1, 3]);  view_default_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_130: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_948, memory_format = torch.contiguous_format);  permute_948 = None
    view_1110: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_130, [1, 512, 1024]);  clone_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_1111: "f32[512, 1024]" = torch.ops.aten.view.default(view_1110, [512, 1024]);  view_1110 = None
    mm_248: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1111, permute_949);  permute_949 = None
    permute_950: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1111, [1, 0])
    mm_249: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_950, view_66);  permute_950 = None
    permute_951: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_249, [1, 0]);  mm_249 = None
    sum_346: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1111, [0], True);  view_1111 = None
    view_1112: "f32[1024]" = torch.ops.aten.view.default(sum_346, [1024]);  sum_346 = None
    permute_952: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_951, [1, 0]);  permute_951 = None
    view_1113: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_248, [1, 512, 1024]);  mm_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_953: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(permute_default_125, [0, 2, 1, 3]);  permute_default_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_1114: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_953, [1, 512, 1024]);  permute_953 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_1115: "f32[512, 1024]" = torch.ops.aten.view.default(view_1114, [512, 1024]);  view_1114 = None
    mm_250: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1115, permute_954);  permute_954 = None
    permute_955: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1115, [1, 0])
    mm_251: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_955, view_66);  permute_955 = None
    permute_956: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_251, [1, 0]);  mm_251 = None
    sum_347: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1115, [0], True);  view_1115 = None
    view_1116: "f32[1024]" = torch.ops.aten.view.default(sum_347, [1024]);  sum_347 = None
    permute_957: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_956, [1, 0]);  permute_956 = None
    view_1117: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_250, [1, 512, 1024]);  mm_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_322: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_1113, view_1117);  view_1113 = view_1117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_1118: "f32[512, 1024]" = torch.ops.aten.view.default(view_1109, [512, 1024]);  view_1109 = None
    mm_252: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1118, permute_958);  permute_958 = None
    permute_959: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1118, [1, 0])
    mm_253: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_959, view_66);  permute_959 = view_66 = None
    permute_960: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_253, [1, 0]);  mm_253 = None
    sum_348: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1118, [0], True);  view_1118 = None
    view_1119: "f32[1024]" = torch.ops.aten.view.default(sum_348, [1024]);  sum_348 = None
    permute_961: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_960, [1, 0]);  permute_960 = None
    view_1120: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_252, [1, 512, 1024]);  mm_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    add_323: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_322, view_1120);  add_322 = view_1120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    mul_785: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_323, primals_52);  primals_52 = None
    mul_786: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_785, 1024)
    sum_349: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_785, [2], True)
    mul_787: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_785, mul_22);  mul_785 = None
    sum_350: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_787, [2], True);  mul_787 = None
    mul_788: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_22, sum_350);  sum_350 = None
    sub_228: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_786, sum_349);  mul_786 = sum_349 = None
    sub_229: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_228, mul_788);  sub_228 = mul_788 = None
    mul_789: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_117, sub_229);  div_117 = sub_229 = None
    mul_790: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_323, mul_22);  mul_22 = None
    sum_351: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_790, [0, 1]);  mul_790 = None
    sum_352: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_323, [0, 1]);  add_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    add_324: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_321, mul_789);  add_321 = mul_789 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_65: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_31, torch.float32);  getitem_31 = None
    mul_791: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_65, 1.1111111111111112);  convert_element_type_65 = None
    mul_792: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_324, mul_791);  mul_791 = None
    clone_131: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_792, memory_format = torch.contiguous_format);  mul_792 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_1121: "f32[512, 1024]" = torch.ops.aten.view.default(clone_131, [512, 1024]);  clone_131 = None
    mm_254: "f32[512, 4096]" = torch.ops.aten.mm.default(view_1121, permute_962);  permute_962 = None
    permute_963: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1121, [1, 0])
    mm_255: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_963, view_64);  permute_963 = view_64 = None
    permute_964: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_255, [1, 0]);  mm_255 = None
    sum_353: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1121, [0], True);  view_1121 = None
    view_1122: "f32[1024]" = torch.ops.aten.view.default(sum_353, [1024]);  sum_353 = None
    permute_965: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_964, [1, 0]);  permute_964 = None
    view_1123: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_254, [1, 512, 4096]);  mm_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_794: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_24, 0.5);  add_24 = None
    mul_795: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_63, view_63)
    mul_796: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_795, -0.5);  mul_795 = None
    exp_49: "f32[1, 512, 4096]" = torch.ops.aten.exp.default(mul_796);  mul_796 = None
    mul_797: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_49, 0.3989422804014327);  exp_49 = None
    mul_798: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_63, mul_797);  view_63 = mul_797 = None
    add_326: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_794, mul_798);  mul_794 = mul_798 = None
    mul_799: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_1123, add_326);  view_1123 = add_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_1124: "f32[512, 4096]" = torch.ops.aten.view.default(mul_799, [512, 4096]);  mul_799 = None
    mm_256: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1124, permute_966);  permute_966 = None
    permute_967: "f32[4096, 512]" = torch.ops.aten.permute.default(view_1124, [1, 0])
    mm_257: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_967, view_62);  permute_967 = view_62 = None
    permute_968: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_257, [1, 0]);  mm_257 = None
    sum_354: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1124, [0], True);  view_1124 = None
    view_1125: "f32[4096]" = torch.ops.aten.view.default(sum_354, [4096]);  sum_354 = None
    permute_969: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_968, [1, 0]);  permute_968 = None
    view_1126: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_256, [1, 512, 1024]);  mm_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    mul_801: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_1126, primals_46);  primals_46 = None
    mul_802: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_801, 1024)
    sum_355: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_801, [2], True)
    mul_803: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_801, mul_17);  mul_801 = None
    sum_356: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_803, [2], True);  mul_803 = None
    mul_804: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_17, sum_356);  sum_356 = None
    sub_231: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_802, sum_355);  mul_802 = sum_355 = None
    sub_232: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_231, mul_804);  sub_231 = mul_804 = None
    mul_805: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_118, sub_232);  div_118 = sub_232 = None
    mul_806: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_1126, mul_17);  mul_17 = None
    sum_357: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_806, [0, 1]);  mul_806 = None
    sum_358: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_1126, [0, 1]);  view_1126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    add_327: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_324, mul_805);  add_324 = mul_805 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_66: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_27, torch.float32);  getitem_27 = None
    mul_807: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_66, 1.1111111111111112);  convert_element_type_66 = None
    mul_808: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_327, mul_807);  mul_807 = None
    clone_132: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_808, memory_format = torch.contiguous_format);  mul_808 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_1127: "f32[512, 1024]" = torch.ops.aten.view.default(clone_132, [512, 1024]);  clone_132 = None
    mm_258: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1127, permute_970);  permute_970 = None
    permute_971: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1127, [1, 0])
    mm_259: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_971, view_60);  permute_971 = view_60 = None
    permute_972: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_259, [1, 0]);  mm_259 = None
    sum_359: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1127, [0], True);  view_1127 = None
    view_1128: "f32[1024]" = torch.ops.aten.view.default(sum_359, [1024]);  sum_359 = None
    permute_973: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_972, [1, 0]);  permute_972 = None
    view_1129: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_258, [1, 512, 1024]);  mm_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_1130: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_1129, [1, 512, 16, 64]);  view_1129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_974: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_1130, [0, 2, 1, 3]);  view_1130 = None
    
    # No stacktrace found for following nodes
    view_default_258: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_974, [16, 512, 64]);  permute_974 = None
    bmm_default_128: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_default_127, view_default_258);  permute_default_127 = None
    view_default_259: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_default_128, [1, 16, 512, 64]);  bmm_default_128 = None
    bmm_default_129: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_default_258, permute_default_128);  view_default_258 = permute_default_128 = None
    view_default_260: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_default_129, [1, 16, 512, 512]);  bmm_default_129 = None
    mul_tensor_85: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_default_260, mul_tensor_84);  view_default_260 = mul_tensor_84 = None
    clone_default_87: "f32[1, 16, 512, 512]" = torch.ops.aten.clone.default(mul_tensor_85, memory_format = torch.contiguous_format);  mul_tensor_85 = None
    mul_tensor_86: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(clone_default_87, alias_default_43);  clone_default_87 = None
    sum_dim_int_list_43: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_86, [-1], True)
    mul_tensor_87: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_default_43, sum_dim_int_list_43);  alias_default_43 = sum_dim_int_list_43 = None
    sub_tensor_43: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_tensor_86, mul_tensor_87);  mul_tensor_86 = mul_tensor_87 = None
    view_default_261: "f32[16, 512, 512]" = torch.ops.aten.view.default(sub_tensor_43, [16, 512, 512]);  sub_tensor_43 = None
    bmm_default_130: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_default_129, view_default_261);  permute_default_129 = None
    view_default_262: "f32[1, 16, 64, 512]" = torch.ops.aten.view.default(bmm_default_130, [1, 16, 64, 512]);  bmm_default_130 = None
    mul_scalar_86: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(view_default_262, 0.3535533905932738);  view_default_262 = None
    permute_default_131: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(mul_scalar_86, [0, 1, 3, 2]);  mul_scalar_86 = None
    bmm_default_131: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_default_261, permute_default_130);  view_default_261 = permute_default_130 = None
    view_default_263: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_default_131, [1, 16, 512, 64]);  bmm_default_131 = None
    mul_scalar_87: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(view_default_263, 0.3535533905932738);  view_default_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_980: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(mul_scalar_87, [0, 2, 1, 3]);  mul_scalar_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_134: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_980, memory_format = torch.contiguous_format);  permute_980 = None
    view_1137: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_134, [1, 512, 1024]);  clone_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_981: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_default_259, [0, 2, 1, 3]);  view_default_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_135: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_981, memory_format = torch.contiguous_format);  permute_981 = None
    view_1138: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_135, [1, 512, 1024]);  clone_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_1139: "f32[512, 1024]" = torch.ops.aten.view.default(view_1138, [512, 1024]);  view_1138 = None
    mm_260: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1139, permute_982);  permute_982 = None
    permute_983: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1139, [1, 0])
    mm_261: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_983, view_44);  permute_983 = None
    permute_984: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_261, [1, 0]);  mm_261 = None
    sum_361: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1139, [0], True);  view_1139 = None
    view_1140: "f32[1024]" = torch.ops.aten.view.default(sum_361, [1024]);  sum_361 = None
    permute_985: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_984, [1, 0]);  permute_984 = None
    view_1141: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_260, [1, 512, 1024]);  mm_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_986: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(permute_default_131, [0, 2, 1, 3]);  permute_default_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_1142: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_986, [1, 512, 1024]);  permute_986 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_1143: "f32[512, 1024]" = torch.ops.aten.view.default(view_1142, [512, 1024]);  view_1142 = None
    mm_262: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1143, permute_987);  permute_987 = None
    permute_988: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1143, [1, 0])
    mm_263: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_988, view_44);  permute_988 = None
    permute_989: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_263, [1, 0]);  mm_263 = None
    sum_362: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1143, [0], True);  view_1143 = None
    view_1144: "f32[1024]" = torch.ops.aten.view.default(sum_362, [1024]);  sum_362 = None
    permute_990: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_989, [1, 0]);  permute_989 = None
    view_1145: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_262, [1, 512, 1024]);  mm_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_328: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_1141, view_1145);  view_1141 = view_1145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_1146: "f32[512, 1024]" = torch.ops.aten.view.default(view_1137, [512, 1024]);  view_1137 = None
    mm_264: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1146, permute_991);  permute_991 = None
    permute_992: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1146, [1, 0])
    mm_265: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_992, view_44);  permute_992 = view_44 = None
    permute_993: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_265, [1, 0]);  mm_265 = None
    sum_363: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1146, [0], True);  view_1146 = None
    view_1147: "f32[1024]" = torch.ops.aten.view.default(sum_363, [1024]);  sum_363 = None
    permute_994: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_993, [1, 0]);  permute_993 = None
    view_1148: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_264, [1, 512, 1024]);  mm_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    add_329: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_328, view_1148);  add_328 = view_1148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    mul_814: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_329, primals_36);  primals_36 = None
    mul_815: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_814, 1024)
    sum_364: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_814, [2], True)
    mul_816: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_814, mul_15);  mul_814 = None
    sum_365: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_816, [2], True);  mul_816 = None
    mul_817: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_15, sum_365);  sum_365 = None
    sub_235: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_815, sum_364);  mul_815 = sum_364 = None
    sub_236: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_235, mul_817);  sub_235 = mul_817 = None
    mul_818: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_120, sub_236);  div_120 = sub_236 = None
    mul_819: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_329, mul_15);  mul_15 = None
    sum_366: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_819, [0, 1]);  mul_819 = None
    sum_367: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_329, [0, 1]);  add_329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    add_330: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_327, mul_818);  add_327 = mul_818 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_68: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_21, torch.float32);  getitem_21 = None
    mul_820: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_68, 1.1111111111111112);  convert_element_type_68 = None
    mul_821: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_330, mul_820);  mul_820 = None
    clone_136: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_821, memory_format = torch.contiguous_format);  mul_821 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_1149: "f32[512, 1024]" = torch.ops.aten.view.default(clone_136, [512, 1024]);  clone_136 = None
    mm_266: "f32[512, 4096]" = torch.ops.aten.mm.default(view_1149, permute_995);  permute_995 = None
    permute_996: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1149, [1, 0])
    mm_267: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_996, view_42);  permute_996 = view_42 = None
    permute_997: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_267, [1, 0]);  mm_267 = None
    sum_368: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1149, [0], True);  view_1149 = None
    view_1150: "f32[1024]" = torch.ops.aten.view.default(sum_368, [1024]);  sum_368 = None
    permute_998: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_997, [1, 0]);  permute_997 = None
    view_1151: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_266, [1, 512, 4096]);  mm_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_823: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_16, 0.5);  add_16 = None
    mul_824: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_41, view_41)
    mul_825: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_824, -0.5);  mul_824 = None
    exp_50: "f32[1, 512, 4096]" = torch.ops.aten.exp.default(mul_825);  mul_825 = None
    mul_826: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_50, 0.3989422804014327);  exp_50 = None
    mul_827: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_41, mul_826);  view_41 = mul_826 = None
    add_332: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_823, mul_827);  mul_823 = mul_827 = None
    mul_828: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_1151, add_332);  view_1151 = add_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_1152: "f32[512, 4096]" = torch.ops.aten.view.default(mul_828, [512, 4096]);  mul_828 = None
    mm_268: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1152, permute_999);  permute_999 = None
    permute_1000: "f32[4096, 512]" = torch.ops.aten.permute.default(view_1152, [1, 0])
    mm_269: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_1000, view_40);  permute_1000 = view_40 = None
    permute_1001: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_269, [1, 0]);  mm_269 = None
    sum_369: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1152, [0], True);  view_1152 = None
    view_1153: "f32[4096]" = torch.ops.aten.view.default(sum_369, [4096]);  sum_369 = None
    permute_1002: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_1001, [1, 0]);  permute_1001 = None
    view_1154: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_268, [1, 512, 1024]);  mm_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    mul_830: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_1154, primals_30);  primals_30 = None
    mul_831: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_830, 1024)
    sum_370: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_830, [2], True)
    mul_832: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_830, mul_10);  mul_830 = None
    sum_371: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_832, [2], True);  mul_832 = None
    mul_833: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_10, sum_371);  sum_371 = None
    sub_238: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_831, sum_370);  mul_831 = sum_370 = None
    sub_239: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_238, mul_833);  sub_238 = mul_833 = None
    mul_834: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_121, sub_239);  div_121 = sub_239 = None
    mul_835: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_1154, mul_10);  mul_10 = None
    sum_372: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_835, [0, 1]);  mul_835 = None
    sum_373: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_1154, [0, 1]);  view_1154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    add_333: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_330, mul_834);  add_330 = mul_834 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_69: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_17, torch.float32);  getitem_17 = None
    mul_836: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_69, 1.1111111111111112);  convert_element_type_69 = None
    mul_837: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_333, mul_836);  mul_836 = None
    clone_137: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_837, memory_format = torch.contiguous_format);  mul_837 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_1155: "f32[512, 1024]" = torch.ops.aten.view.default(clone_137, [512, 1024]);  clone_137 = None
    mm_270: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1155, permute_1003);  permute_1003 = None
    permute_1004: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1155, [1, 0])
    mm_271: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1004, view_38);  permute_1004 = view_38 = None
    permute_1005: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_271, [1, 0]);  mm_271 = None
    sum_374: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1155, [0], True);  view_1155 = None
    view_1156: "f32[1024]" = torch.ops.aten.view.default(sum_374, [1024]);  sum_374 = None
    permute_1006: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_1005, [1, 0]);  permute_1005 = None
    view_1157: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_270, [1, 512, 1024]);  mm_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_1158: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_1157, [1, 512, 16, 64]);  view_1157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_1007: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_1158, [0, 2, 1, 3]);  view_1158 = None
    
    # No stacktrace found for following nodes
    view_default_270: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_1007, [16, 512, 64]);  permute_1007 = None
    bmm_default_134: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_default_133, view_default_270);  permute_default_133 = None
    view_default_271: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_default_134, [1, 16, 512, 64]);  bmm_default_134 = None
    bmm_default_135: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_default_270, permute_default_134);  view_default_270 = permute_default_134 = None
    view_default_272: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_default_135, [1, 16, 512, 512]);  bmm_default_135 = None
    mul_tensor_89: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_default_272, mul_tensor_88);  view_default_272 = mul_tensor_88 = None
    clone_default_91: "f32[1, 16, 512, 512]" = torch.ops.aten.clone.default(mul_tensor_89, memory_format = torch.contiguous_format);  mul_tensor_89 = None
    mul_tensor_90: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(clone_default_91, alias_default_45);  clone_default_91 = None
    sum_dim_int_list_45: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_90, [-1], True)
    mul_tensor_91: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_default_45, sum_dim_int_list_45);  alias_default_45 = sum_dim_int_list_45 = None
    sub_tensor_45: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_tensor_90, mul_tensor_91);  mul_tensor_90 = mul_tensor_91 = None
    view_default_273: "f32[16, 512, 512]" = torch.ops.aten.view.default(sub_tensor_45, [16, 512, 512]);  sub_tensor_45 = None
    bmm_default_136: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_default_135, view_default_273);  permute_default_135 = None
    view_default_274: "f32[1, 16, 64, 512]" = torch.ops.aten.view.default(bmm_default_136, [1, 16, 64, 512]);  bmm_default_136 = None
    mul_scalar_90: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(view_default_274, 0.3535533905932738);  view_default_274 = None
    permute_default_137: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(mul_scalar_90, [0, 1, 3, 2]);  mul_scalar_90 = None
    bmm_default_137: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_default_273, permute_default_136);  view_default_273 = permute_default_136 = None
    view_default_275: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_default_137, [1, 16, 512, 64]);  bmm_default_137 = None
    mul_scalar_91: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(view_default_275, 0.3535533905932738);  view_default_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_1013: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(mul_scalar_91, [0, 2, 1, 3]);  mul_scalar_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_139: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_1013, memory_format = torch.contiguous_format);  permute_1013 = None
    view_1165: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_139, [1, 512, 1024]);  clone_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_1014: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_default_271, [0, 2, 1, 3]);  view_default_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_140: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_1014, memory_format = torch.contiguous_format);  permute_1014 = None
    view_1166: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_140, [1, 512, 1024]);  clone_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_1167: "f32[512, 1024]" = torch.ops.aten.view.default(view_1166, [512, 1024]);  view_1166 = None
    mm_272: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1167, permute_1015);  permute_1015 = None
    permute_1016: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1167, [1, 0])
    mm_273: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1016, view_22);  permute_1016 = None
    permute_1017: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_273, [1, 0]);  mm_273 = None
    sum_376: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1167, [0], True);  view_1167 = None
    view_1168: "f32[1024]" = torch.ops.aten.view.default(sum_376, [1024]);  sum_376 = None
    permute_1018: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_1017, [1, 0]);  permute_1017 = None
    view_1169: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_272, [1, 512, 1024]);  mm_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_1019: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(permute_default_137, [0, 2, 1, 3]);  permute_default_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_1170: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_1019, [1, 512, 1024]);  permute_1019 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_1171: "f32[512, 1024]" = torch.ops.aten.view.default(view_1170, [512, 1024]);  view_1170 = None
    mm_274: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1171, permute_1020);  permute_1020 = None
    permute_1021: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1171, [1, 0])
    mm_275: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1021, view_22);  permute_1021 = None
    permute_1022: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_275, [1, 0]);  mm_275 = None
    sum_377: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1171, [0], True);  view_1171 = None
    view_1172: "f32[1024]" = torch.ops.aten.view.default(sum_377, [1024]);  sum_377 = None
    permute_1023: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_1022, [1, 0]);  permute_1022 = None
    view_1173: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_274, [1, 512, 1024]);  mm_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_334: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_1169, view_1173);  view_1169 = view_1173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_1174: "f32[512, 1024]" = torch.ops.aten.view.default(view_1165, [512, 1024]);  view_1165 = None
    mm_276: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1174, permute_1024);  permute_1024 = None
    permute_1025: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1174, [1, 0])
    mm_277: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1025, view_22);  permute_1025 = view_22 = None
    permute_1026: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_277, [1, 0]);  mm_277 = None
    sum_378: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1174, [0], True);  view_1174 = None
    view_1175: "f32[1024]" = torch.ops.aten.view.default(sum_378, [1024]);  sum_378 = None
    permute_1027: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_1026, [1, 0]);  permute_1026 = None
    view_1176: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_276, [1, 512, 1024]);  mm_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    add_335: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_334, view_1176);  add_334 = view_1176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    mul_843: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_335, primals_20);  primals_20 = None
    mul_844: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_843, 1024)
    sum_379: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_843, [2], True)
    mul_845: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_843, mul_8);  mul_843 = None
    sum_380: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_845, [2], True);  mul_845 = None
    mul_846: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_8, sum_380);  sum_380 = None
    sub_242: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_844, sum_379);  mul_844 = sum_379 = None
    sub_243: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_242, mul_846);  sub_242 = mul_846 = None
    mul_847: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_123, sub_243);  div_123 = sub_243 = None
    mul_848: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_335, mul_8);  mul_8 = None
    sum_381: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_848, [0, 1]);  mul_848 = None
    sum_382: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_335, [0, 1]);  add_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    add_336: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_333, mul_847);  add_333 = mul_847 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:418, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_71: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_11, torch.float32);  getitem_11 = None
    mul_849: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_71, 1.1111111111111112);  convert_element_type_71 = None
    mul_850: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_336, mul_849);  mul_849 = None
    clone_141: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_850, memory_format = torch.contiguous_format);  mul_850 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:417, code: hidden_states = self.dense(hidden_states)
    view_1177: "f32[512, 1024]" = torch.ops.aten.view.default(clone_141, [512, 1024]);  clone_141 = None
    mm_278: "f32[512, 4096]" = torch.ops.aten.mm.default(view_1177, permute_1028);  permute_1028 = None
    permute_1029: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1177, [1, 0])
    mm_279: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_1029, view_20);  permute_1029 = view_20 = None
    permute_1030: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_279, [1, 0]);  mm_279 = None
    sum_383: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1177, [0], True);  view_1177 = None
    view_1178: "f32[1024]" = torch.ops.aten.view.default(sum_383, [1024]);  sum_383 = None
    permute_1031: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_1030, [1, 0]);  permute_1030 = None
    view_1179: "f32[1, 512, 4096]" = torch.ops.aten.view.default(mm_278, [1, 512, 4096]);  mm_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_852: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(add_8, 0.5);  add_8 = None
    mul_853: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_19, view_19)
    mul_854: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_853, -0.5);  mul_853 = None
    exp_51: "f32[1, 512, 4096]" = torch.ops.aten.exp.default(mul_854);  mul_854 = None
    mul_855: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_51, 0.3989422804014327);  exp_51 = None
    mul_856: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_19, mul_855);  view_19 = mul_855 = None
    add_338: "f32[1, 512, 4096]" = torch.ops.aten.add.Tensor(mul_852, mul_856);  mul_852 = mul_856 = None
    mul_857: "f32[1, 512, 4096]" = torch.ops.aten.mul.Tensor(view_1179, add_338);  view_1179 = add_338 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:404, code: hidden_states = self.dense(hidden_states)
    view_1180: "f32[512, 4096]" = torch.ops.aten.view.default(mul_857, [512, 4096]);  mul_857 = None
    mm_280: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1180, permute_1032);  permute_1032 = None
    permute_1033: "f32[4096, 512]" = torch.ops.aten.permute.default(view_1180, [1, 0])
    mm_281: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_1033, view_18);  permute_1033 = view_18 = None
    permute_1034: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_281, [1, 0]);  mm_281 = None
    sum_384: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1180, [0], True);  view_1180 = None
    view_1181: "f32[4096]" = torch.ops.aten.view.default(sum_384, [4096]);  sum_384 = None
    permute_1035: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_1034, [1, 0]);  permute_1034 = None
    view_1182: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_280, [1, 512, 1024]);  mm_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    mul_859: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_1182, primals_14);  primals_14 = None
    mul_860: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_859, 1024)
    sum_385: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_859, [2], True)
    mul_861: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_859, mul_3);  mul_859 = None
    sum_386: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_861, [2], True);  mul_861 = None
    mul_862: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_3, sum_386);  sum_386 = None
    sub_245: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_860, sum_385);  mul_860 = sum_385 = None
    sub_246: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_245, mul_862);  sub_245 = mul_862 = None
    mul_863: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_124, sub_246);  div_124 = sub_246 = None
    mul_864: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_1182, mul_3);  mul_3 = None
    sum_387: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_864, [0, 1]);  mul_864 = None
    sum_388: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_1182, [0, 1]);  view_1182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:505, code: ln_output = self.ln(attention_output)
    add_339: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_336, mul_863);  add_336 = mul_863 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:337, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_72: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_7, torch.float32);  getitem_7 = None
    mul_865: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_72, 1.1111111111111112);  convert_element_type_72 = None
    mul_866: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_339, mul_865);  mul_865 = None
    clone_142: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_866, memory_format = torch.contiguous_format);  mul_866 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:336, code: hidden_states = self.dense(hidden_states)
    view_1183: "f32[512, 1024]" = torch.ops.aten.view.default(clone_142, [512, 1024]);  clone_142 = None
    mm_282: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1183, permute_1036);  permute_1036 = None
    permute_1037: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1183, [1, 0])
    mm_283: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1037, view_16);  permute_1037 = view_16 = None
    permute_1038: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_283, [1, 0]);  mm_283 = None
    sum_389: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1183, [0], True);  view_1183 = None
    view_1184: "f32[1024]" = torch.ops.aten.view.default(sum_389, [1024]);  sum_389 = None
    permute_1039: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_1038, [1, 0]);  permute_1038 = None
    view_1185: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_282, [1, 512, 1024]);  mm_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:319, code: context_layer = context_layer.view(new_context_layer_shape)
    view_1186: "f32[1, 512, 16, 64]" = torch.ops.aten.view.default(view_1185, [1, 512, 16, 64]);  view_1185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:317, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_1040: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(view_1186, [0, 2, 1, 3]);  view_1186 = None
    
    # No stacktrace found for following nodes
    view_default_282: "f32[16, 512, 64]" = torch.ops.aten.view.default(permute_1040, [16, 512, 64]);  permute_1040 = None
    bmm_default_140: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(permute_default_139, view_default_282);  permute_default_139 = None
    view_default_283: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_default_140, [1, 16, 512, 64]);  bmm_default_140 = None
    bmm_default_141: "f32[16, 512, 512]" = torch.ops.aten.bmm.default(view_default_282, permute_default_140);  view_default_282 = permute_default_140 = None
    view_default_284: "f32[1, 16, 512, 512]" = torch.ops.aten.view.default(bmm_default_141, [1, 16, 512, 512]);  bmm_default_141 = None
    mul_tensor_93: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_default_284, mul_tensor_92);  view_default_284 = mul_tensor_92 = None
    clone_default_95: "f32[1, 16, 512, 512]" = torch.ops.aten.clone.default(mul_tensor_93, memory_format = torch.contiguous_format);  mul_tensor_93 = None
    mul_tensor_94: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(clone_default_95, alias_default_47);  clone_default_95 = None
    sum_dim_int_list_47: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_94, [-1], True)
    mul_tensor_95: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(alias_default_47, sum_dim_int_list_47);  alias_default_47 = sum_dim_int_list_47 = None
    sub_tensor_47: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_tensor_94, mul_tensor_95);  mul_tensor_94 = mul_tensor_95 = None
    view_default_285: "f32[16, 512, 512]" = torch.ops.aten.view.default(sub_tensor_47, [16, 512, 512]);  sub_tensor_47 = None
    bmm_default_142: "f32[16, 64, 512]" = torch.ops.aten.bmm.default(permute_default_141, view_default_285);  permute_default_141 = None
    view_default_286: "f32[1, 16, 64, 512]" = torch.ops.aten.view.default(bmm_default_142, [1, 16, 64, 512]);  bmm_default_142 = None
    mul_scalar_94: "f32[1, 16, 64, 512]" = torch.ops.aten.mul.Scalar(view_default_286, 0.3535533905932738);  view_default_286 = None
    permute_default_143: "f32[1, 16, 512, 64]" = torch.ops.aten.permute.default(mul_scalar_94, [0, 1, 3, 2]);  mul_scalar_94 = None
    bmm_default_143: "f32[16, 512, 64]" = torch.ops.aten.bmm.default(view_default_285, permute_default_142);  view_default_285 = permute_default_142 = None
    view_default_287: "f32[1, 16, 512, 64]" = torch.ops.aten.view.default(bmm_default_143, [1, 16, 512, 64]);  bmm_default_143 = None
    mul_scalar_95: "f32[1, 16, 512, 64]" = torch.ops.aten.mul.Scalar(view_default_287, 0.3535533905932738);  view_default_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_1046: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(mul_scalar_95, [0, 2, 1, 3]);  mul_scalar_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_144: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_1046, memory_format = torch.contiguous_format);  permute_1046 = None
    view_1193: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_144, [1, 512, 1024]);  clone_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_1047: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(view_default_283, [0, 2, 1, 3]);  view_default_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    clone_145: "f32[1, 512, 16, 64]" = torch.ops.aten.clone.default(permute_1047, memory_format = torch.contiguous_format);  permute_1047 = None
    view_1194: "f32[1, 512, 1024]" = torch.ops.aten.view.default(clone_145, [1, 512, 1024]);  clone_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:259, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_1195: "f32[512, 1024]" = torch.ops.aten.view.default(view_1194, [512, 1024]);  view_1194 = None
    mm_284: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1195, permute_1048);  permute_1048 = None
    permute_1049: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1195, [1, 0])
    mm_285: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1049, view);  permute_1049 = None
    permute_1050: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_285, [1, 0]);  mm_285 = None
    sum_391: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1195, [0], True);  view_1195 = None
    view_1196: "f32[1024]" = torch.ops.aten.view.default(sum_391, [1024]);  sum_391 = None
    permute_1051: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_1050, [1, 0]);  permute_1050 = None
    view_1197: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_284, [1, 512, 1024]);  mm_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224, code: return x.permute(0, 2, 1, 3)
    permute_1052: "f32[1, 512, 16, 64]" = torch.ops.aten.permute.default(permute_default_143, [0, 2, 1, 3]);  permute_default_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:223, code: x = x.view(new_x_shape)
    view_1198: "f32[1, 512, 1024]" = torch.ops.aten.view.default(permute_1052, [1, 512, 1024]);  permute_1052 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_1199: "f32[512, 1024]" = torch.ops.aten.view.default(view_1198, [512, 1024]);  view_1198 = None
    mm_286: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1199, permute_1053);  permute_1053 = None
    permute_1054: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1199, [1, 0])
    mm_287: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1054, view);  permute_1054 = None
    permute_1055: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_287, [1, 0]);  mm_287 = None
    sum_392: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1199, [0], True);  view_1199 = None
    view_1200: "f32[1024]" = torch.ops.aten.view.default(sum_392, [1024]);  sum_392 = None
    permute_1056: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_1055, [1, 0]);  permute_1055 = None
    view_1201: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_286, [1, 512, 1024]);  mm_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:258, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_340: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(view_1197, view_1201);  view_1197 = view_1201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    view_1202: "f32[512, 1024]" = torch.ops.aten.view.default(view_1193, [512, 1024]);  view_1193 = None
    mm_288: "f32[512, 1024]" = torch.ops.aten.mm.default(view_1202, permute_1057);  permute_1057 = None
    permute_1058: "f32[1024, 512]" = torch.ops.aten.permute.default(view_1202, [1, 0])
    mm_289: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1058, view);  permute_1058 = view = None
    permute_1059: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_289, [1, 0]);  mm_289 = None
    sum_393: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1202, [0], True);  view_1202 = None
    view_1203: "f32[1024]" = torch.ops.aten.view.default(sum_393, [1024]);  sum_393 = None
    permute_1060: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_1059, [1, 0]);  permute_1059 = None
    view_1204: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_288, [1, 512, 1024]);  mm_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:236, code: mixed_query_layer = self.query(hidden_states)
    add_341: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_340, view_1204);  add_340 = view_1204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    mul_872: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_341, primals_4);  primals_4 = None
    mul_873: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_872, 1024)
    sum_394: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_872, [2], True)
    mul_874: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_872, mul_1);  mul_872 = None
    sum_395: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_874, [2], True);  mul_874 = None
    mul_875: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_1, sum_395);  sum_395 = None
    sub_249: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_873, sum_394);  mul_873 = sum_394 = None
    sub_250: "f32[1, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_249, mul_875);  sub_249 = mul_875 = None
    mul_876: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(div_126, sub_250);  div_126 = sub_250 = None
    mul_877: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_341, mul_1);  mul_1 = None
    sum_396: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_877, [0, 1]);  mul_877 = None
    sum_397: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_341, [0, 1]);  add_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:378, code: ln_outputs = self.ln(hidden_states)
    add_342: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(add_339, mul_876);  add_339 = mul_876 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189, code: embeddings = self.dropout(embeddings)
    convert_element_type_74: "f32[1, 512, 1024]" = torch.ops.prims.convert_element_type.default(getitem_1, torch.float32);  getitem_1 = None
    mul_878: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_74, 1.1111111111111112);  convert_element_type_74 = None
    mul_879: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_342, mul_878);  add_342 = mul_878 = None
    clone_146: "f32[1, 512, 1024]" = torch.ops.aten.clone.default(mul_879, memory_format = torch.contiguous_format);  mul_879 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:184, code: position_embeddings = self.position_embeddings(position_ids)
    eq: "b8[1, 512]" = torch.ops.aten.eq.Scalar(slice_3, -1)
    unsqueeze_8: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq, -1);  eq = None
    where_8: "f32[1, 512, 1024]" = torch.ops.aten.where.self(unsqueeze_8, full_default_3, clone_146);  unsqueeze_8 = None
    full_default_13: "f32[512, 1024]" = torch.ops.aten.full.default([512, 1024], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    _unsafe_index_put: "f32[512, 1024]" = torch.ops.aten._unsafe_index_put.default(full_default_13, [slice_3], where_8, True);  full_default_13 = slice_3 = where_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:180, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
    full_default_14: "b8[1, 512, 1]" = torch.ops.aten.full.default([1, 512, 1], False, dtype = torch.bool, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    where_9: "f32[1, 512, 1024]" = torch.ops.aten.where.self(full_default_14, full_default_3, clone_146);  full_default_14 = None
    full_default_16: "f32[2, 1024]" = torch.ops.aten.full.default([2, 1024], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    _unsafe_index_put_1: "f32[2, 1024]" = torch.ops.aten._unsafe_index_put.default(full_default_16, [full_default], where_9, True);  full_default_16 = full_default = where_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:179, code: inputs_embeds = self.word_embeddings(input_ids)
    eq_2: "b8[1, 512]" = torch.ops.aten.eq.Scalar(primals_393, 0)
    unsqueeze_10: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_2, -1);  eq_2 = None
    where_10: "f32[1, 512, 1024]" = torch.ops.aten.where.self(unsqueeze_10, full_default_3, clone_146);  unsqueeze_10 = full_default_3 = clone_146 = None
    full_default_18: "f32[29056, 1024]" = torch.ops.aten.full.default([29056, 1024], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    _unsafe_index_put_2: "f32[29056, 1024]" = torch.ops.aten._unsafe_index_put.default(full_default_18, [primals_393], where_10, True);  full_default_18 = primals_393 = where_10 = None
    return [_unsafe_index_put_2, _unsafe_index_put_1, _unsafe_index_put, sum_396, sum_397, permute_1060, view_1203, permute_1056, view_1200, permute_1051, view_1196, permute_1039, view_1184, sum_387, sum_388, permute_1035, view_1181, permute_1031, view_1178, sum_381, sum_382, permute_1027, view_1175, permute_1023, view_1172, permute_1018, view_1168, permute_1006, view_1156, sum_372, sum_373, permute_1002, view_1153, permute_998, view_1150, sum_366, sum_367, permute_994, view_1147, permute_990, view_1144, permute_985, view_1140, permute_973, view_1128, sum_357, sum_358, permute_969, view_1125, permute_965, view_1122, sum_351, sum_352, permute_961, view_1119, permute_957, view_1116, permute_952, view_1112, permute_940, view_1100, sum_342, sum_343, permute_936, view_1097, permute_932, view_1094, sum_336, sum_337, permute_928, view_1091, permute_924, view_1088, permute_919, view_1084, permute_907, view_1072, sum_327, sum_328, permute_903, view_1069, permute_899, view_1066, sum_321, sum_322, permute_895, view_1063, permute_891, view_1060, permute_886, view_1056, permute_874, view_1044, sum_312, sum_313, permute_870, view_1041, permute_866, view_1038, sum_306, sum_307, permute_862, view_1035, permute_858, view_1032, permute_853, view_1028, permute_841, view_1016, sum_297, sum_298, permute_837, view_1013, permute_833, view_1010, sum_291, sum_292, permute_829, view_1007, permute_825, view_1004, permute_820, view_1000, permute_808, view_988, sum_282, sum_283, permute_804, view_985, permute_800, view_982, sum_276, sum_277, permute_796, view_979, permute_792, view_976, permute_787, view_972, permute_775, view_960, sum_267, sum_268, permute_771, view_957, permute_767, view_954, sum_261, sum_262, permute_763, view_951, permute_759, view_948, permute_754, view_944, permute_742, view_932, sum_252, sum_253, permute_738, view_929, permute_734, view_926, sum_246, sum_247, permute_730, view_923, permute_726, view_920, permute_721, view_916, permute_709, view_904, sum_237, sum_238, permute_705, view_901, permute_701, view_898, sum_231, sum_232, permute_697, view_895, permute_693, view_892, permute_688, view_888, permute_676, view_876, sum_222, sum_223, permute_672, view_873, permute_668, view_870, sum_216, sum_217, permute_664, view_867, permute_660, view_864, permute_655, view_860, permute_643, view_848, sum_207, sum_208, permute_639, view_845, permute_635, view_842, sum_201, sum_202, permute_631, view_839, permute_627, view_836, permute_622, view_832, permute_610, view_820, sum_192, sum_193, permute_606, view_817, permute_602, view_814, sum_186, sum_187, permute_598, view_811, permute_594, view_808, permute_589, view_804, permute_577, view_792, sum_177, sum_178, permute_573, view_789, permute_569, view_786, sum_171, sum_172, permute_565, view_783, permute_561, view_780, permute_556, view_776, permute_544, view_764, sum_162, sum_163, permute_540, view_761, permute_536, view_758, sum_156, sum_157, permute_532, view_755, permute_528, view_752, permute_523, view_748, permute_511, view_736, sum_147, sum_148, permute_507, view_733, permute_503, view_730, sum_141, sum_142, permute_499, view_727, permute_495, view_724, permute_490, view_720, permute_478, view_708, sum_132, sum_133, permute_474, view_705, permute_470, view_702, sum_126, sum_127, permute_466, view_699, permute_462, view_696, permute_457, view_692, permute_445, view_680, sum_117, sum_118, permute_441, view_677, permute_437, view_674, sum_111, sum_112, permute_433, view_671, permute_429, view_668, permute_424, view_664, permute_412, view_652, sum_102, sum_103, permute_408, view_649, permute_404, view_646, sum_96, sum_97, permute_400, view_643, permute_396, view_640, permute_391, view_636, permute_379, view_624, sum_87, sum_88, permute_375, view_621, permute_371, view_618, sum_81, sum_82, permute_367, view_615, permute_363, view_612, permute_358, view_608, permute_346, view_596, sum_72, sum_73, permute_342, view_593, permute_338, view_590, sum_66, sum_67, permute_334, view_587, permute_330, view_584, permute_325, view_580, permute_313, view_568, sum_57, sum_58, permute_309, view_565, permute_305, view_562, sum_51, sum_52, permute_301, view_559, permute_297, view_556, permute_292, view_552, permute_280, view_540, sum_42, sum_43, permute_276, view_537, permute_272, view_534, sum_36, sum_37, permute_268, view_531, None, None, None, None]
    