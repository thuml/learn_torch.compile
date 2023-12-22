from __future__ import annotations



def forward(self, primals_1: "f32[768]", primals_2: "f32[16, 1, 1]", primals_3: "f32[768]", primals_4: "f32[768]", primals_5: "f32[768]", primals_6: "f32[16, 1, 1]", primals_7: "f32[768]", primals_8: "f32[768]", primals_9: "f32[768]", primals_10: "f32[16, 1, 1]", primals_11: "f32[768]", primals_12: "f32[768]", primals_13: "f32[768]", primals_14: "f32[16, 1, 1]", primals_15: "f32[768]", primals_16: "f32[768]", primals_17: "f32[768]", primals_18: "f32[16, 1, 1]", primals_19: "f32[768]", primals_20: "f32[768]", primals_21: "f32[768]", primals_22: "f32[16, 1, 1]", primals_23: "f32[768]", primals_24: "f32[768]", primals_25: "f32[768]", primals_26: "f32[16, 1, 1]", primals_27: "f32[768]", primals_28: "f32[768]", primals_29: "f32[768]", primals_30: "f32[16, 1, 1]", primals_31: "f32[768]", primals_32: "f32[768]", primals_33: "f32[768]", primals_34: "f32[16, 1, 1]", primals_35: "f32[768]", primals_36: "f32[768]", primals_37: "f32[768]", primals_38: "f32[16, 1, 1]", primals_39: "f32[768]", primals_40: "f32[768]", primals_41: "f32[768]", primals_42: "f32[16, 1, 1]", primals_43: "f32[768]", primals_44: "f32[768]", primals_45: "f32[768]", primals_46: "f32[16, 1, 1]", primals_47: "f32[768]", primals_48: "f32[768]", primals_49: "f32[768]", primals_50: "f32[16, 1, 1]", primals_51: "f32[768]", primals_52: "f32[768]", primals_53: "f32[768]", primals_54: "f32[16, 1, 1]", primals_55: "f32[768]", primals_56: "f32[768]", primals_57: "f32[768]", primals_58: "f32[16, 1, 1]", primals_59: "f32[768]", primals_60: "f32[768]", primals_61: "f32[768]", primals_62: "f32[16, 1, 1]", primals_63: "f32[768]", primals_64: "f32[768]", primals_65: "f32[768]", primals_66: "f32[16, 1, 1]", primals_67: "f32[768]", primals_68: "f32[768]", primals_69: "f32[768]", primals_70: "f32[16, 1, 1]", primals_71: "f32[768]", primals_72: "f32[768]", primals_73: "f32[768]", primals_74: "f32[16, 1, 1]", primals_75: "f32[768]", primals_76: "f32[768]", primals_77: "f32[768]", primals_78: "f32[16, 1, 1]", primals_79: "f32[768]", primals_80: "f32[768]", primals_81: "f32[768]", primals_82: "f32[16, 1, 1]", primals_83: "f32[768]", primals_84: "f32[768]", primals_85: "f32[768]", primals_86: "f32[16, 1, 1]", primals_87: "f32[768]", primals_88: "f32[768]", primals_89: "f32[768]", primals_90: "f32[16, 1, 1]", primals_91: "f32[768]", primals_92: "f32[768]", primals_93: "f32[768]", primals_94: "f32[16, 1, 1]", primals_95: "f32[768]", primals_96: "f32[768]", primals_98: "f32[768]", primals_99: "f32[768]", primals_100: "f32[768]", primals_101: "f32[768]", primals_102: "f32[192, 3, 3, 3]", primals_103: "f32[192]", primals_105: "f32[384, 192, 3, 3]", primals_106: "f32[384]", primals_108: "f32[768, 384, 3, 3]", primals_109: "f32[768]", primals_111: "f32[768, 64, 1, 1]", primals_113: "f32[768]", primals_118: "f32[768]", primals_119: "f32[768]", primals_121: "f32[768, 1, 3, 3]", primals_123: "f32[768]", primals_125: "f32[768, 1, 3, 3]", primals_127: "f32[768]", primals_133: "f32[768]", primals_138: "f32[768]", primals_139: "f32[768]", primals_141: "f32[768, 1, 3, 3]", primals_143: "f32[768]", primals_145: "f32[768, 1, 3, 3]", primals_147: "f32[768]", primals_153: "f32[768]", primals_158: "f32[768]", primals_159: "f32[768]", primals_161: "f32[768, 1, 3, 3]", primals_163: "f32[768]", primals_165: "f32[768, 1, 3, 3]", primals_167: "f32[768]", primals_173: "f32[768]", primals_178: "f32[768]", primals_179: "f32[768]", primals_181: "f32[768, 1, 3, 3]", primals_183: "f32[768]", primals_185: "f32[768, 1, 3, 3]", primals_187: "f32[768]", primals_193: "f32[768]", primals_198: "f32[768]", primals_199: "f32[768]", primals_201: "f32[768, 1, 3, 3]", primals_203: "f32[768]", primals_205: "f32[768, 1, 3, 3]", primals_207: "f32[768]", primals_213: "f32[768]", primals_218: "f32[768]", primals_219: "f32[768]", primals_221: "f32[768, 1, 3, 3]", primals_223: "f32[768]", primals_225: "f32[768, 1, 3, 3]", primals_227: "f32[768]", primals_233: "f32[768]", primals_238: "f32[768]", primals_239: "f32[768]", primals_241: "f32[768, 1, 3, 3]", primals_243: "f32[768]", primals_245: "f32[768, 1, 3, 3]", primals_247: "f32[768]", primals_253: "f32[768]", primals_258: "f32[768]", primals_259: "f32[768]", primals_261: "f32[768, 1, 3, 3]", primals_263: "f32[768]", primals_265: "f32[768, 1, 3, 3]", primals_267: "f32[768]", primals_273: "f32[768]", primals_278: "f32[768]", primals_279: "f32[768]", primals_281: "f32[768, 1, 3, 3]", primals_283: "f32[768]", primals_285: "f32[768, 1, 3, 3]", primals_287: "f32[768]", primals_293: "f32[768]", primals_298: "f32[768]", primals_299: "f32[768]", primals_301: "f32[768, 1, 3, 3]", primals_303: "f32[768]", primals_305: "f32[768, 1, 3, 3]", primals_307: "f32[768]", primals_313: "f32[768]", primals_318: "f32[768]", primals_319: "f32[768]", primals_321: "f32[768, 1, 3, 3]", primals_323: "f32[768]", primals_325: "f32[768, 1, 3, 3]", primals_327: "f32[768]", primals_333: "f32[768]", primals_338: "f32[768]", primals_339: "f32[768]", primals_341: "f32[768, 1, 3, 3]", primals_343: "f32[768]", primals_345: "f32[768, 1, 3, 3]", primals_347: "f32[768]", primals_353: "f32[768]", primals_358: "f32[768]", primals_359: "f32[768]", primals_361: "f32[768, 1, 3, 3]", primals_363: "f32[768]", primals_365: "f32[768, 1, 3, 3]", primals_367: "f32[768]", primals_373: "f32[768]", primals_378: "f32[768]", primals_379: "f32[768]", primals_381: "f32[768, 1, 3, 3]", primals_383: "f32[768]", primals_385: "f32[768, 1, 3, 3]", primals_387: "f32[768]", primals_393: "f32[768]", primals_398: "f32[768]", primals_399: "f32[768]", primals_401: "f32[768, 1, 3, 3]", primals_403: "f32[768]", primals_405: "f32[768, 1, 3, 3]", primals_407: "f32[768]", primals_413: "f32[768]", primals_418: "f32[768]", primals_419: "f32[768]", primals_421: "f32[768, 1, 3, 3]", primals_423: "f32[768]", primals_425: "f32[768, 1, 3, 3]", primals_427: "f32[768]", primals_433: "f32[768]", primals_438: "f32[768]", primals_439: "f32[768]", primals_441: "f32[768, 1, 3, 3]", primals_443: "f32[768]", primals_445: "f32[768, 1, 3, 3]", primals_447: "f32[768]", primals_453: "f32[768]", primals_458: "f32[768]", primals_459: "f32[768]", primals_461: "f32[768, 1, 3, 3]", primals_463: "f32[768]", primals_465: "f32[768, 1, 3, 3]", primals_467: "f32[768]", primals_473: "f32[768]", primals_478: "f32[768]", primals_479: "f32[768]", primals_481: "f32[768, 1, 3, 3]", primals_483: "f32[768]", primals_485: "f32[768, 1, 3, 3]", primals_487: "f32[768]", primals_493: "f32[768]", primals_498: "f32[768]", primals_499: "f32[768]", primals_501: "f32[768, 1, 3, 3]", primals_503: "f32[768]", primals_505: "f32[768, 1, 3, 3]", primals_507: "f32[768]", primals_513: "f32[768]", primals_518: "f32[768]", primals_519: "f32[768]", primals_521: "f32[768, 1, 3, 3]", primals_523: "f32[768]", primals_525: "f32[768, 1, 3, 3]", primals_527: "f32[768]", primals_533: "f32[768]", primals_538: "f32[768]", primals_539: "f32[768]", primals_541: "f32[768, 1, 3, 3]", primals_543: "f32[768]", primals_545: "f32[768, 1, 3, 3]", primals_547: "f32[768]", primals_553: "f32[768]", primals_558: "f32[768]", primals_559: "f32[768]", primals_561: "f32[768, 1, 3, 3]", primals_563: "f32[768]", primals_565: "f32[768, 1, 3, 3]", primals_567: "f32[768]", primals_573: "f32[768]", primals_578: "f32[768]", primals_579: "f32[768]", primals_581: "f32[768, 1, 3, 3]", primals_583: "f32[768]", primals_585: "f32[768, 1, 3, 3]", primals_587: "f32[768]", primals_593: "f32[768]", primals_603: "f32[768]", primals_606: "f32[3072]", primals_609: "f32[768]", primals_619: "f32[768]", primals_622: "f32[3072]", primals_625: "f32[768]", primals_710: "f32[8, 3, 224, 224]", convolution: "f32[8, 192, 112, 112]", squeeze_1: "f32[192]", mul_9: "f32[8, 192, 112, 112]", convolution_1: "f32[8, 384, 56, 56]", squeeze_4: "f32[384]", mul_19: "f32[8, 384, 56, 56]", convolution_2: "f32[8, 768, 28, 28]", squeeze_7: "f32[768]", permute_1: "f32[1, 64, 28, 28]", mul_33: "f32[8, 784, 768]", view_4: "f32[6272, 768]", getitem_8: "f32[8, 16, 48, 784]", getitem_9: "f32[8, 16, 48, 784]", pow_3: "f32[8, 16, 48, 1]", pow_5: "f32[8, 16, 48, 1]", bmm: "f32[128, 48, 48]", view_14: "f32[6272, 768]", mm: "f32[6272, 768]", mul_37: "f32[8, 784, 768]", view_16: "f32[8, 768, 28, 28]", convolution_4: "f32[8, 768, 28, 28]", squeeze_10: "f32[768]", add_34: "f32[8, 768, 28, 28]", convolution_5: "f32[8, 768, 28, 28]", mul_50: "f32[8, 784, 768]", view_18: "f32[6272, 768]", addmm_1: "f32[6272, 3072]", view_20: "f32[6272, 3072]", addmm_2: "f32[6272, 768]", mul_56: "f32[8, 784, 768]", view_22: "f32[6272, 768]", getitem_19: "f32[8, 16, 48, 784]", getitem_20: "f32[8, 16, 48, 784]", pow_7: "f32[8, 16, 48, 1]", pow_9: "f32[8, 16, 48, 1]", bmm_2: "f32[128, 48, 48]", view_32: "f32[6272, 768]", mm_1: "f32[6272, 768]", mul_60: "f32[8, 784, 768]", view_34: "f32[8, 768, 28, 28]", convolution_6: "f32[8, 768, 28, 28]", squeeze_13: "f32[768]", add_51: "f32[8, 768, 28, 28]", convolution_7: "f32[8, 768, 28, 28]", mul_73: "f32[8, 784, 768]", view_36: "f32[6272, 768]", addmm_4: "f32[6272, 3072]", view_38: "f32[6272, 3072]", addmm_5: "f32[6272, 768]", mul_79: "f32[8, 784, 768]", view_40: "f32[6272, 768]", getitem_30: "f32[8, 16, 48, 784]", getitem_31: "f32[8, 16, 48, 784]", pow_11: "f32[8, 16, 48, 1]", pow_13: "f32[8, 16, 48, 1]", bmm_4: "f32[128, 48, 48]", view_50: "f32[6272, 768]", mm_2: "f32[6272, 768]", mul_83: "f32[8, 784, 768]", view_52: "f32[8, 768, 28, 28]", convolution_8: "f32[8, 768, 28, 28]", squeeze_16: "f32[768]", add_68: "f32[8, 768, 28, 28]", convolution_9: "f32[8, 768, 28, 28]", mul_96: "f32[8, 784, 768]", view_54: "f32[6272, 768]", addmm_7: "f32[6272, 3072]", view_56: "f32[6272, 3072]", addmm_8: "f32[6272, 768]", mul_102: "f32[8, 784, 768]", view_58: "f32[6272, 768]", getitem_41: "f32[8, 16, 48, 784]", getitem_42: "f32[8, 16, 48, 784]", pow_15: "f32[8, 16, 48, 1]", pow_17: "f32[8, 16, 48, 1]", bmm_6: "f32[128, 48, 48]", view_68: "f32[6272, 768]", mm_3: "f32[6272, 768]", mul_106: "f32[8, 784, 768]", view_70: "f32[8, 768, 28, 28]", convolution_10: "f32[8, 768, 28, 28]", squeeze_19: "f32[768]", add_85: "f32[8, 768, 28, 28]", convolution_11: "f32[8, 768, 28, 28]", mul_119: "f32[8, 784, 768]", view_72: "f32[6272, 768]", addmm_10: "f32[6272, 3072]", view_74: "f32[6272, 3072]", addmm_11: "f32[6272, 768]", mul_125: "f32[8, 784, 768]", view_76: "f32[6272, 768]", getitem_52: "f32[8, 16, 48, 784]", getitem_53: "f32[8, 16, 48, 784]", pow_19: "f32[8, 16, 48, 1]", pow_21: "f32[8, 16, 48, 1]", bmm_8: "f32[128, 48, 48]", view_86: "f32[6272, 768]", mm_4: "f32[6272, 768]", mul_129: "f32[8, 784, 768]", view_88: "f32[8, 768, 28, 28]", convolution_12: "f32[8, 768, 28, 28]", squeeze_22: "f32[768]", add_102: "f32[8, 768, 28, 28]", convolution_13: "f32[8, 768, 28, 28]", mul_142: "f32[8, 784, 768]", view_90: "f32[6272, 768]", addmm_13: "f32[6272, 3072]", view_92: "f32[6272, 3072]", addmm_14: "f32[6272, 768]", mul_148: "f32[8, 784, 768]", view_94: "f32[6272, 768]", getitem_63: "f32[8, 16, 48, 784]", getitem_64: "f32[8, 16, 48, 784]", pow_23: "f32[8, 16, 48, 1]", pow_25: "f32[8, 16, 48, 1]", bmm_10: "f32[128, 48, 48]", view_104: "f32[6272, 768]", mm_5: "f32[6272, 768]", mul_152: "f32[8, 784, 768]", view_106: "f32[8, 768, 28, 28]", convolution_14: "f32[8, 768, 28, 28]", squeeze_25: "f32[768]", add_119: "f32[8, 768, 28, 28]", convolution_15: "f32[8, 768, 28, 28]", mul_165: "f32[8, 784, 768]", view_108: "f32[6272, 768]", addmm_16: "f32[6272, 3072]", view_110: "f32[6272, 3072]", addmm_17: "f32[6272, 768]", mul_171: "f32[8, 784, 768]", view_112: "f32[6272, 768]", getitem_74: "f32[8, 16, 48, 784]", getitem_75: "f32[8, 16, 48, 784]", pow_27: "f32[8, 16, 48, 1]", pow_29: "f32[8, 16, 48, 1]", bmm_12: "f32[128, 48, 48]", view_122: "f32[6272, 768]", mm_6: "f32[6272, 768]", mul_175: "f32[8, 784, 768]", view_124: "f32[8, 768, 28, 28]", convolution_16: "f32[8, 768, 28, 28]", squeeze_28: "f32[768]", add_136: "f32[8, 768, 28, 28]", convolution_17: "f32[8, 768, 28, 28]", mul_188: "f32[8, 784, 768]", view_126: "f32[6272, 768]", addmm_19: "f32[6272, 3072]", view_128: "f32[6272, 3072]", addmm_20: "f32[6272, 768]", mul_194: "f32[8, 784, 768]", view_130: "f32[6272, 768]", getitem_85: "f32[8, 16, 48, 784]", getitem_86: "f32[8, 16, 48, 784]", pow_31: "f32[8, 16, 48, 1]", pow_33: "f32[8, 16, 48, 1]", bmm_14: "f32[128, 48, 48]", view_140: "f32[6272, 768]", mm_7: "f32[6272, 768]", mul_198: "f32[8, 784, 768]", view_142: "f32[8, 768, 28, 28]", convolution_18: "f32[8, 768, 28, 28]", squeeze_31: "f32[768]", add_153: "f32[8, 768, 28, 28]", convolution_19: "f32[8, 768, 28, 28]", mul_211: "f32[8, 784, 768]", view_144: "f32[6272, 768]", addmm_22: "f32[6272, 3072]", view_146: "f32[6272, 3072]", addmm_23: "f32[6272, 768]", mul_217: "f32[8, 784, 768]", view_148: "f32[6272, 768]", getitem_96: "f32[8, 16, 48, 784]", getitem_97: "f32[8, 16, 48, 784]", pow_35: "f32[8, 16, 48, 1]", pow_37: "f32[8, 16, 48, 1]", bmm_16: "f32[128, 48, 48]", view_158: "f32[6272, 768]", mm_8: "f32[6272, 768]", mul_221: "f32[8, 784, 768]", view_160: "f32[8, 768, 28, 28]", convolution_20: "f32[8, 768, 28, 28]", squeeze_34: "f32[768]", add_170: "f32[8, 768, 28, 28]", convolution_21: "f32[8, 768, 28, 28]", mul_234: "f32[8, 784, 768]", view_162: "f32[6272, 768]", addmm_25: "f32[6272, 3072]", view_164: "f32[6272, 3072]", addmm_26: "f32[6272, 768]", mul_240: "f32[8, 784, 768]", view_166: "f32[6272, 768]", getitem_107: "f32[8, 16, 48, 784]", getitem_108: "f32[8, 16, 48, 784]", pow_39: "f32[8, 16, 48, 1]", pow_41: "f32[8, 16, 48, 1]", bmm_18: "f32[128, 48, 48]", view_176: "f32[6272, 768]", mm_9: "f32[6272, 768]", mul_244: "f32[8, 784, 768]", view_178: "f32[8, 768, 28, 28]", convolution_22: "f32[8, 768, 28, 28]", squeeze_37: "f32[768]", add_187: "f32[8, 768, 28, 28]", convolution_23: "f32[8, 768, 28, 28]", mul_257: "f32[8, 784, 768]", view_180: "f32[6272, 768]", addmm_28: "f32[6272, 3072]", view_182: "f32[6272, 3072]", addmm_29: "f32[6272, 768]", mul_263: "f32[8, 784, 768]", view_184: "f32[6272, 768]", getitem_118: "f32[8, 16, 48, 784]", getitem_119: "f32[8, 16, 48, 784]", pow_43: "f32[8, 16, 48, 1]", pow_45: "f32[8, 16, 48, 1]", bmm_20: "f32[128, 48, 48]", view_194: "f32[6272, 768]", mm_10: "f32[6272, 768]", mul_267: "f32[8, 784, 768]", view_196: "f32[8, 768, 28, 28]", convolution_24: "f32[8, 768, 28, 28]", squeeze_40: "f32[768]", add_204: "f32[8, 768, 28, 28]", convolution_25: "f32[8, 768, 28, 28]", mul_280: "f32[8, 784, 768]", view_198: "f32[6272, 768]", addmm_31: "f32[6272, 3072]", view_200: "f32[6272, 3072]", addmm_32: "f32[6272, 768]", mul_286: "f32[8, 784, 768]", view_202: "f32[6272, 768]", getitem_129: "f32[8, 16, 48, 784]", getitem_130: "f32[8, 16, 48, 784]", pow_47: "f32[8, 16, 48, 1]", pow_49: "f32[8, 16, 48, 1]", bmm_22: "f32[128, 48, 48]", view_212: "f32[6272, 768]", mm_11: "f32[6272, 768]", mul_290: "f32[8, 784, 768]", view_214: "f32[8, 768, 28, 28]", convolution_26: "f32[8, 768, 28, 28]", squeeze_43: "f32[768]", add_221: "f32[8, 768, 28, 28]", convolution_27: "f32[8, 768, 28, 28]", mul_303: "f32[8, 784, 768]", view_216: "f32[6272, 768]", addmm_34: "f32[6272, 3072]", view_218: "f32[6272, 3072]", addmm_35: "f32[6272, 768]", mul_309: "f32[8, 784, 768]", view_220: "f32[6272, 768]", getitem_140: "f32[8, 16, 48, 784]", getitem_141: "f32[8, 16, 48, 784]", pow_51: "f32[8, 16, 48, 1]", pow_53: "f32[8, 16, 48, 1]", bmm_24: "f32[128, 48, 48]", view_230: "f32[6272, 768]", mm_12: "f32[6272, 768]", mul_313: "f32[8, 784, 768]", view_232: "f32[8, 768, 28, 28]", convolution_28: "f32[8, 768, 28, 28]", squeeze_46: "f32[768]", add_238: "f32[8, 768, 28, 28]", convolution_29: "f32[8, 768, 28, 28]", mul_326: "f32[8, 784, 768]", view_234: "f32[6272, 768]", addmm_37: "f32[6272, 3072]", view_236: "f32[6272, 3072]", addmm_38: "f32[6272, 768]", mul_332: "f32[8, 784, 768]", view_238: "f32[6272, 768]", getitem_151: "f32[8, 16, 48, 784]", getitem_152: "f32[8, 16, 48, 784]", pow_55: "f32[8, 16, 48, 1]", pow_57: "f32[8, 16, 48, 1]", bmm_26: "f32[128, 48, 48]", view_248: "f32[6272, 768]", mm_13: "f32[6272, 768]", mul_336: "f32[8, 784, 768]", view_250: "f32[8, 768, 28, 28]", convolution_30: "f32[8, 768, 28, 28]", squeeze_49: "f32[768]", add_255: "f32[8, 768, 28, 28]", convolution_31: "f32[8, 768, 28, 28]", mul_349: "f32[8, 784, 768]", view_252: "f32[6272, 768]", addmm_40: "f32[6272, 3072]", view_254: "f32[6272, 3072]", addmm_41: "f32[6272, 768]", mul_355: "f32[8, 784, 768]", view_256: "f32[6272, 768]", getitem_162: "f32[8, 16, 48, 784]", getitem_163: "f32[8, 16, 48, 784]", pow_59: "f32[8, 16, 48, 1]", pow_61: "f32[8, 16, 48, 1]", bmm_28: "f32[128, 48, 48]", view_266: "f32[6272, 768]", mm_14: "f32[6272, 768]", mul_359: "f32[8, 784, 768]", view_268: "f32[8, 768, 28, 28]", convolution_32: "f32[8, 768, 28, 28]", squeeze_52: "f32[768]", add_272: "f32[8, 768, 28, 28]", convolution_33: "f32[8, 768, 28, 28]", mul_372: "f32[8, 784, 768]", view_270: "f32[6272, 768]", addmm_43: "f32[6272, 3072]", view_272: "f32[6272, 3072]", addmm_44: "f32[6272, 768]", mul_378: "f32[8, 784, 768]", view_274: "f32[6272, 768]", getitem_173: "f32[8, 16, 48, 784]", getitem_174: "f32[8, 16, 48, 784]", pow_63: "f32[8, 16, 48, 1]", pow_65: "f32[8, 16, 48, 1]", bmm_30: "f32[128, 48, 48]", view_284: "f32[6272, 768]", mm_15: "f32[6272, 768]", mul_382: "f32[8, 784, 768]", view_286: "f32[8, 768, 28, 28]", convolution_34: "f32[8, 768, 28, 28]", squeeze_55: "f32[768]", add_289: "f32[8, 768, 28, 28]", convolution_35: "f32[8, 768, 28, 28]", mul_395: "f32[8, 784, 768]", view_288: "f32[6272, 768]", addmm_46: "f32[6272, 3072]", view_290: "f32[6272, 3072]", addmm_47: "f32[6272, 768]", mul_401: "f32[8, 784, 768]", view_292: "f32[6272, 768]", getitem_184: "f32[8, 16, 48, 784]", getitem_185: "f32[8, 16, 48, 784]", pow_67: "f32[8, 16, 48, 1]", pow_69: "f32[8, 16, 48, 1]", bmm_32: "f32[128, 48, 48]", view_302: "f32[6272, 768]", mm_16: "f32[6272, 768]", mul_405: "f32[8, 784, 768]", view_304: "f32[8, 768, 28, 28]", convolution_36: "f32[8, 768, 28, 28]", squeeze_58: "f32[768]", add_306: "f32[8, 768, 28, 28]", convolution_37: "f32[8, 768, 28, 28]", mul_418: "f32[8, 784, 768]", view_306: "f32[6272, 768]", addmm_49: "f32[6272, 3072]", view_308: "f32[6272, 3072]", addmm_50: "f32[6272, 768]", mul_424: "f32[8, 784, 768]", view_310: "f32[6272, 768]", getitem_195: "f32[8, 16, 48, 784]", getitem_196: "f32[8, 16, 48, 784]", pow_71: "f32[8, 16, 48, 1]", pow_73: "f32[8, 16, 48, 1]", bmm_34: "f32[128, 48, 48]", view_320: "f32[6272, 768]", mm_17: "f32[6272, 768]", mul_428: "f32[8, 784, 768]", view_322: "f32[8, 768, 28, 28]", convolution_38: "f32[8, 768, 28, 28]", squeeze_61: "f32[768]", add_323: "f32[8, 768, 28, 28]", convolution_39: "f32[8, 768, 28, 28]", mul_441: "f32[8, 784, 768]", view_324: "f32[6272, 768]", addmm_52: "f32[6272, 3072]", view_326: "f32[6272, 3072]", addmm_53: "f32[6272, 768]", mul_447: "f32[8, 784, 768]", view_328: "f32[6272, 768]", getitem_206: "f32[8, 16, 48, 784]", getitem_207: "f32[8, 16, 48, 784]", pow_75: "f32[8, 16, 48, 1]", pow_77: "f32[8, 16, 48, 1]", bmm_36: "f32[128, 48, 48]", view_338: "f32[6272, 768]", mm_18: "f32[6272, 768]", mul_451: "f32[8, 784, 768]", view_340: "f32[8, 768, 28, 28]", convolution_40: "f32[8, 768, 28, 28]", squeeze_64: "f32[768]", add_340: "f32[8, 768, 28, 28]", convolution_41: "f32[8, 768, 28, 28]", mul_464: "f32[8, 784, 768]", view_342: "f32[6272, 768]", addmm_55: "f32[6272, 3072]", view_344: "f32[6272, 3072]", addmm_56: "f32[6272, 768]", mul_470: "f32[8, 784, 768]", view_346: "f32[6272, 768]", getitem_217: "f32[8, 16, 48, 784]", getitem_218: "f32[8, 16, 48, 784]", pow_79: "f32[8, 16, 48, 1]", pow_81: "f32[8, 16, 48, 1]", bmm_38: "f32[128, 48, 48]", view_356: "f32[6272, 768]", mm_19: "f32[6272, 768]", mul_474: "f32[8, 784, 768]", view_358: "f32[8, 768, 28, 28]", convolution_42: "f32[8, 768, 28, 28]", squeeze_67: "f32[768]", add_357: "f32[8, 768, 28, 28]", convolution_43: "f32[8, 768, 28, 28]", mul_487: "f32[8, 784, 768]", view_360: "f32[6272, 768]", addmm_58: "f32[6272, 3072]", view_362: "f32[6272, 3072]", addmm_59: "f32[6272, 768]", mul_493: "f32[8, 784, 768]", view_364: "f32[6272, 768]", getitem_228: "f32[8, 16, 48, 784]", getitem_229: "f32[8, 16, 48, 784]", pow_83: "f32[8, 16, 48, 1]", pow_85: "f32[8, 16, 48, 1]", bmm_40: "f32[128, 48, 48]", view_374: "f32[6272, 768]", mm_20: "f32[6272, 768]", mul_497: "f32[8, 784, 768]", view_376: "f32[8, 768, 28, 28]", convolution_44: "f32[8, 768, 28, 28]", squeeze_70: "f32[768]", add_374: "f32[8, 768, 28, 28]", convolution_45: "f32[8, 768, 28, 28]", mul_510: "f32[8, 784, 768]", view_378: "f32[6272, 768]", addmm_61: "f32[6272, 3072]", view_380: "f32[6272, 3072]", addmm_62: "f32[6272, 768]", mul_516: "f32[8, 784, 768]", view_382: "f32[6272, 768]", getitem_239: "f32[8, 16, 48, 784]", getitem_240: "f32[8, 16, 48, 784]", pow_87: "f32[8, 16, 48, 1]", pow_89: "f32[8, 16, 48, 1]", bmm_42: "f32[128, 48, 48]", view_392: "f32[6272, 768]", mm_21: "f32[6272, 768]", mul_520: "f32[8, 784, 768]", view_394: "f32[8, 768, 28, 28]", convolution_46: "f32[8, 768, 28, 28]", squeeze_73: "f32[768]", add_391: "f32[8, 768, 28, 28]", convolution_47: "f32[8, 768, 28, 28]", mul_533: "f32[8, 784, 768]", view_396: "f32[6272, 768]", addmm_64: "f32[6272, 3072]", view_398: "f32[6272, 3072]", addmm_65: "f32[6272, 768]", mul_539: "f32[8, 784, 768]", view_400: "f32[6272, 768]", getitem_250: "f32[8, 16, 48, 784]", getitem_251: "f32[8, 16, 48, 784]", pow_91: "f32[8, 16, 48, 1]", pow_93: "f32[8, 16, 48, 1]", bmm_44: "f32[128, 48, 48]", view_410: "f32[6272, 768]", mm_22: "f32[6272, 768]", mul_543: "f32[8, 784, 768]", view_412: "f32[8, 768, 28, 28]", convolution_48: "f32[8, 768, 28, 28]", squeeze_76: "f32[768]", add_408: "f32[8, 768, 28, 28]", convolution_49: "f32[8, 768, 28, 28]", mul_556: "f32[8, 784, 768]", view_414: "f32[6272, 768]", addmm_67: "f32[6272, 3072]", view_416: "f32[6272, 3072]", addmm_68: "f32[6272, 768]", mul_562: "f32[8, 784, 768]", view_418: "f32[6272, 768]", getitem_261: "f32[8, 16, 48, 784]", getitem_262: "f32[8, 16, 48, 784]", pow_95: "f32[8, 16, 48, 1]", pow_97: "f32[8, 16, 48, 1]", bmm_46: "f32[128, 48, 48]", view_428: "f32[6272, 768]", mm_23: "f32[6272, 768]", mul_566: "f32[8, 784, 768]", view_430: "f32[8, 768, 28, 28]", convolution_50: "f32[8, 768, 28, 28]", squeeze_79: "f32[768]", add_425: "f32[8, 768, 28, 28]", convolution_51: "f32[8, 768, 28, 28]", mul_579: "f32[8, 784, 768]", view_432: "f32[6272, 768]", addmm_70: "f32[6272, 3072]", view_434: "f32[6272, 3072]", addmm_71: "f32[6272, 768]", cat_3: "f32[8, 785, 768]", getitem_271: "f32[8, 785, 1]", rsqrt_99: "f32[8, 785, 1]", select: "f32[8, 768]", permute_220: "f32[8, 16, 1, 48]", view_437: "f32[6280, 768]", permute_222: "f32[8, 16, 785, 48]", permute_224: "f32[8, 16, 785, 48]", getitem_273: "f32[8, 16, 32]", getitem_274: "i64[]", getitem_275: "i64[]", view_444: "f32[8, 768]", cat_4: "f32[8, 785, 768]", mul_588: "f32[8, 785, 768]", view_446: "f32[8, 768]", mm_24: "f32[8, 3072]", view_448: "f32[8, 3072]", addmm_76: "f32[8, 768]", mul_594: "f32[8, 785, 768]", select_1: "f32[8, 768]", permute_230: "f32[8, 16, 1, 48]", view_451: "f32[6280, 768]", permute_232: "f32[8, 16, 785, 48]", permute_234: "f32[8, 16, 785, 48]", getitem_281: "f32[8, 16, 32]", getitem_282: "i64[]", getitem_283: "i64[]", view_458: "f32[8, 768]", cat_6: "f32[8, 785, 768]", mul_597: "f32[8, 785, 768]", view_460: "f32[8, 768]", mm_25: "f32[8, 3072]", view_462: "f32[8, 3072]", addmm_81: "f32[8, 768]", mul_603: "f32[8, 785, 768]", clone_271: "f32[8, 768]", permute_240: "f32[1000, 768]", div_78: "f32[8, 785, 1]", permute_244: "f32[768, 3072]", permute_250: "f32[3072, 768]", div_79: "f32[8, 785, 1]", permute_252: "f32[768, 768]", alias_74: "f32[8, 16, 1, 48]", permute_258: "f32[768, 768]", permute_263: "f32[768, 768]", permute_268: "f32[768, 768]", div_80: "f32[8, 785, 1]", permute_272: "f32[768, 3072]", permute_278: "f32[3072, 768]", div_81: "f32[8, 785, 1]", permute_280: "f32[768, 768]", alias_75: "f32[8, 16, 1, 48]", permute_286: "f32[768, 768]", permute_291: "f32[768, 768]", permute_296: "f32[768, 768]", permute_300: "f32[768, 3072]", permute_304: "f32[3072, 768]", div_83: "f32[8, 784, 1]", unsqueeze_119: "f32[1, 768, 1, 1]", div_84: "f32[8, 784, 1]", permute_312: "f32[768, 768]", permute_315: "f32[128, 48, 48]", permute_316: "f32[128, 784, 48]", alias_76: "f32[8, 16, 48, 48]", permute_317: "f32[128, 784, 48]", permute_318: "f32[128, 48, 784]", permute_321: "f32[2304, 768]", div_93: "f32[8, 784, 1]", permute_325: "f32[768, 3072]", permute_329: "f32[3072, 768]", div_94: "f32[8, 784, 1]", unsqueeze_131: "f32[1, 768, 1, 1]", div_95: "f32[8, 784, 1]", permute_337: "f32[768, 768]", permute_340: "f32[128, 48, 48]", permute_341: "f32[128, 784, 48]", alias_79: "f32[8, 16, 48, 48]", permute_342: "f32[128, 784, 48]", permute_343: "f32[128, 48, 784]", permute_346: "f32[2304, 768]", div_104: "f32[8, 784, 1]", permute_350: "f32[768, 3072]", permute_354: "f32[3072, 768]", div_105: "f32[8, 784, 1]", unsqueeze_143: "f32[1, 768, 1, 1]", div_106: "f32[8, 784, 1]", permute_362: "f32[768, 768]", permute_365: "f32[128, 48, 48]", permute_366: "f32[128, 784, 48]", alias_82: "f32[8, 16, 48, 48]", permute_367: "f32[128, 784, 48]", permute_368: "f32[128, 48, 784]", permute_371: "f32[2304, 768]", div_115: "f32[8, 784, 1]", permute_375: "f32[768, 3072]", permute_379: "f32[3072, 768]", div_116: "f32[8, 784, 1]", unsqueeze_155: "f32[1, 768, 1, 1]", div_117: "f32[8, 784, 1]", permute_387: "f32[768, 768]", permute_390: "f32[128, 48, 48]", permute_391: "f32[128, 784, 48]", alias_85: "f32[8, 16, 48, 48]", permute_392: "f32[128, 784, 48]", permute_393: "f32[128, 48, 784]", permute_396: "f32[2304, 768]", div_126: "f32[8, 784, 1]", permute_400: "f32[768, 3072]", permute_404: "f32[3072, 768]", div_127: "f32[8, 784, 1]", unsqueeze_167: "f32[1, 768, 1, 1]", div_128: "f32[8, 784, 1]", permute_412: "f32[768, 768]", permute_415: "f32[128, 48, 48]", permute_416: "f32[128, 784, 48]", alias_88: "f32[8, 16, 48, 48]", permute_417: "f32[128, 784, 48]", permute_418: "f32[128, 48, 784]", permute_421: "f32[2304, 768]", div_137: "f32[8, 784, 1]", permute_425: "f32[768, 3072]", permute_429: "f32[3072, 768]", div_138: "f32[8, 784, 1]", unsqueeze_179: "f32[1, 768, 1, 1]", div_139: "f32[8, 784, 1]", permute_437: "f32[768, 768]", permute_440: "f32[128, 48, 48]", permute_441: "f32[128, 784, 48]", alias_91: "f32[8, 16, 48, 48]", permute_442: "f32[128, 784, 48]", permute_443: "f32[128, 48, 784]", permute_446: "f32[2304, 768]", div_148: "f32[8, 784, 1]", permute_450: "f32[768, 3072]", permute_454: "f32[3072, 768]", div_149: "f32[8, 784, 1]", unsqueeze_191: "f32[1, 768, 1, 1]", div_150: "f32[8, 784, 1]", permute_462: "f32[768, 768]", permute_465: "f32[128, 48, 48]", permute_466: "f32[128, 784, 48]", alias_94: "f32[8, 16, 48, 48]", permute_467: "f32[128, 784, 48]", permute_468: "f32[128, 48, 784]", permute_471: "f32[2304, 768]", div_159: "f32[8, 784, 1]", permute_475: "f32[768, 3072]", permute_479: "f32[3072, 768]", div_160: "f32[8, 784, 1]", unsqueeze_203: "f32[1, 768, 1, 1]", div_161: "f32[8, 784, 1]", permute_487: "f32[768, 768]", permute_490: "f32[128, 48, 48]", permute_491: "f32[128, 784, 48]", alias_97: "f32[8, 16, 48, 48]", permute_492: "f32[128, 784, 48]", permute_493: "f32[128, 48, 784]", permute_496: "f32[2304, 768]", div_170: "f32[8, 784, 1]", permute_500: "f32[768, 3072]", permute_504: "f32[3072, 768]", div_171: "f32[8, 784, 1]", unsqueeze_215: "f32[1, 768, 1, 1]", div_172: "f32[8, 784, 1]", permute_512: "f32[768, 768]", permute_515: "f32[128, 48, 48]", permute_516: "f32[128, 784, 48]", alias_100: "f32[8, 16, 48, 48]", permute_517: "f32[128, 784, 48]", permute_518: "f32[128, 48, 784]", permute_521: "f32[2304, 768]", div_181: "f32[8, 784, 1]", permute_525: "f32[768, 3072]", permute_529: "f32[3072, 768]", div_182: "f32[8, 784, 1]", unsqueeze_227: "f32[1, 768, 1, 1]", div_183: "f32[8, 784, 1]", permute_537: "f32[768, 768]", permute_540: "f32[128, 48, 48]", permute_541: "f32[128, 784, 48]", alias_103: "f32[8, 16, 48, 48]", permute_542: "f32[128, 784, 48]", permute_543: "f32[128, 48, 784]", permute_546: "f32[2304, 768]", div_192: "f32[8, 784, 1]", permute_550: "f32[768, 3072]", permute_554: "f32[3072, 768]", div_193: "f32[8, 784, 1]", unsqueeze_239: "f32[1, 768, 1, 1]", div_194: "f32[8, 784, 1]", permute_562: "f32[768, 768]", permute_565: "f32[128, 48, 48]", permute_566: "f32[128, 784, 48]", alias_106: "f32[8, 16, 48, 48]", permute_567: "f32[128, 784, 48]", permute_568: "f32[128, 48, 784]", permute_571: "f32[2304, 768]", div_203: "f32[8, 784, 1]", permute_575: "f32[768, 3072]", permute_579: "f32[3072, 768]", div_204: "f32[8, 784, 1]", unsqueeze_251: "f32[1, 768, 1, 1]", div_205: "f32[8, 784, 1]", permute_587: "f32[768, 768]", permute_590: "f32[128, 48, 48]", permute_591: "f32[128, 784, 48]", alias_109: "f32[8, 16, 48, 48]", permute_592: "f32[128, 784, 48]", permute_593: "f32[128, 48, 784]", permute_596: "f32[2304, 768]", div_214: "f32[8, 784, 1]", permute_600: "f32[768, 3072]", permute_604: "f32[3072, 768]", div_215: "f32[8, 784, 1]", unsqueeze_263: "f32[1, 768, 1, 1]", div_216: "f32[8, 784, 1]", permute_612: "f32[768, 768]", permute_615: "f32[128, 48, 48]", permute_616: "f32[128, 784, 48]", alias_112: "f32[8, 16, 48, 48]", permute_617: "f32[128, 784, 48]", permute_618: "f32[128, 48, 784]", permute_621: "f32[2304, 768]", div_225: "f32[8, 784, 1]", permute_625: "f32[768, 3072]", permute_629: "f32[3072, 768]", div_226: "f32[8, 784, 1]", unsqueeze_275: "f32[1, 768, 1, 1]", div_227: "f32[8, 784, 1]", permute_637: "f32[768, 768]", permute_640: "f32[128, 48, 48]", permute_641: "f32[128, 784, 48]", alias_115: "f32[8, 16, 48, 48]", permute_642: "f32[128, 784, 48]", permute_643: "f32[128, 48, 784]", permute_646: "f32[2304, 768]", div_236: "f32[8, 784, 1]", permute_650: "f32[768, 3072]", permute_654: "f32[3072, 768]", div_237: "f32[8, 784, 1]", unsqueeze_287: "f32[1, 768, 1, 1]", div_238: "f32[8, 784, 1]", permute_662: "f32[768, 768]", permute_665: "f32[128, 48, 48]", permute_666: "f32[128, 784, 48]", alias_118: "f32[8, 16, 48, 48]", permute_667: "f32[128, 784, 48]", permute_668: "f32[128, 48, 784]", permute_671: "f32[2304, 768]", div_247: "f32[8, 784, 1]", permute_675: "f32[768, 3072]", permute_679: "f32[3072, 768]", div_248: "f32[8, 784, 1]", unsqueeze_299: "f32[1, 768, 1, 1]", div_249: "f32[8, 784, 1]", permute_687: "f32[768, 768]", permute_690: "f32[128, 48, 48]", permute_691: "f32[128, 784, 48]", alias_121: "f32[8, 16, 48, 48]", permute_692: "f32[128, 784, 48]", permute_693: "f32[128, 48, 784]", permute_696: "f32[2304, 768]", div_258: "f32[8, 784, 1]", permute_700: "f32[768, 3072]", permute_704: "f32[3072, 768]", div_259: "f32[8, 784, 1]", unsqueeze_311: "f32[1, 768, 1, 1]", div_260: "f32[8, 784, 1]", permute_712: "f32[768, 768]", permute_715: "f32[128, 48, 48]", permute_716: "f32[128, 784, 48]", alias_124: "f32[8, 16, 48, 48]", permute_717: "f32[128, 784, 48]", permute_718: "f32[128, 48, 784]", permute_721: "f32[2304, 768]", div_269: "f32[8, 784, 1]", permute_725: "f32[768, 3072]", permute_729: "f32[3072, 768]", div_270: "f32[8, 784, 1]", unsqueeze_323: "f32[1, 768, 1, 1]", div_271: "f32[8, 784, 1]", permute_737: "f32[768, 768]", permute_740: "f32[128, 48, 48]", permute_741: "f32[128, 784, 48]", alias_127: "f32[8, 16, 48, 48]", permute_742: "f32[128, 784, 48]", permute_743: "f32[128, 48, 784]", permute_746: "f32[2304, 768]", div_280: "f32[8, 784, 1]", permute_750: "f32[768, 3072]", permute_754: "f32[3072, 768]", div_281: "f32[8, 784, 1]", unsqueeze_335: "f32[1, 768, 1, 1]", div_282: "f32[8, 784, 1]", permute_762: "f32[768, 768]", permute_765: "f32[128, 48, 48]", permute_766: "f32[128, 784, 48]", alias_130: "f32[8, 16, 48, 48]", permute_767: "f32[128, 784, 48]", permute_768: "f32[128, 48, 784]", permute_771: "f32[2304, 768]", div_291: "f32[8, 784, 1]", permute_775: "f32[768, 3072]", permute_779: "f32[3072, 768]", div_292: "f32[8, 784, 1]", unsqueeze_347: "f32[1, 768, 1, 1]", div_293: "f32[8, 784, 1]", permute_787: "f32[768, 768]", permute_790: "f32[128, 48, 48]", permute_791: "f32[128, 784, 48]", alias_133: "f32[8, 16, 48, 48]", permute_792: "f32[128, 784, 48]", permute_793: "f32[128, 48, 784]", permute_796: "f32[2304, 768]", div_302: "f32[8, 784, 1]", permute_800: "f32[768, 3072]", permute_804: "f32[3072, 768]", div_303: "f32[8, 784, 1]", unsqueeze_359: "f32[1, 768, 1, 1]", div_304: "f32[8, 784, 1]", permute_812: "f32[768, 768]", permute_815: "f32[128, 48, 48]", permute_816: "f32[128, 784, 48]", alias_136: "f32[8, 16, 48, 48]", permute_817: "f32[128, 784, 48]", permute_818: "f32[128, 48, 784]", permute_821: "f32[2304, 768]", div_313: "f32[8, 784, 1]", permute_825: "f32[768, 3072]", permute_829: "f32[3072, 768]", div_314: "f32[8, 784, 1]", unsqueeze_371: "f32[1, 768, 1, 1]", div_315: "f32[8, 784, 1]", permute_837: "f32[768, 768]", permute_840: "f32[128, 48, 48]", permute_841: "f32[128, 784, 48]", alias_139: "f32[8, 16, 48, 48]", permute_842: "f32[128, 784, 48]", permute_843: "f32[128, 48, 784]", permute_846: "f32[2304, 768]", div_324: "f32[8, 784, 1]", permute_850: "f32[768, 3072]", permute_854: "f32[3072, 768]", div_325: "f32[8, 784, 1]", unsqueeze_383: "f32[1, 768, 1, 1]", div_326: "f32[8, 784, 1]", permute_862: "f32[768, 768]", permute_865: "f32[128, 48, 48]", permute_866: "f32[128, 784, 48]", alias_142: "f32[8, 16, 48, 48]", permute_867: "f32[128, 784, 48]", permute_868: "f32[128, 48, 784]", permute_871: "f32[2304, 768]", div_335: "f32[8, 784, 1]", permute_875: "f32[768, 3072]", permute_879: "f32[3072, 768]", div_336: "f32[8, 784, 1]", unsqueeze_395: "f32[1, 768, 1, 1]", div_337: "f32[8, 784, 1]", permute_887: "f32[768, 768]", permute_890: "f32[128, 48, 48]", permute_891: "f32[128, 784, 48]", alias_145: "f32[8, 16, 48, 48]", permute_892: "f32[128, 784, 48]", permute_893: "f32[128, 48, 784]", permute_896: "f32[2304, 768]", div_346: "f32[8, 784, 1]", unsqueeze_407: "f32[1, 768, 1, 1]", add_682: "f32[8, 384, 56, 56]", unsqueeze_419: "f32[1, 384, 1, 1]", add_684: "f32[8, 192, 112, 112]", unsqueeze_431: "f32[1, 192, 1, 1]", tangents_1: "f32[8, 1000]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    clamp_min: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_3, 1e-12)
    expand: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min, [8, 16, 48, 784]);  clamp_min = None
    div_6: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_8, expand)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    clamp_min_1: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_5, 1e-12)
    expand_1: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_1, [8, 16, 48, 784]);  clamp_min_1 = None
    div_7: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_9, expand_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    view_9: "f32[8, 16, 48, 48]" = torch.ops.aten.reshape.default(bmm, [8, 16, 48, 48]);  bmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    view_15: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm, [8, 784, 768]);  mm = None
    add_25: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_15, primals_118);  view_15 = primals_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_39: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_4, 0.5)
    mul_40: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_4, 0.7071067811865476)
    erf_2: "f32[8, 768, 28, 28]" = torch.ops.aten.erf.default(mul_40);  mul_40 = None
    add_29: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    mul_41: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_39, add_29);  mul_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_17: "f32[8, 768, 784]" = torch.ops.aten.reshape.default(convolution_5, [8, 768, 784]);  convolution_5 = None
    permute_9: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_17, [0, 2, 1]);  view_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_19: "f32[8, 784, 3072]" = torch.ops.aten.reshape.default(addmm_1, [8, 784, 3072]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_53: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_19, 0.7071067811865476)
    erf_3: "f32[8, 784, 3072]" = torch.ops.aten.erf.default(mul_53);  mul_53 = None
    add_38: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_21: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(addmm_2, [8, 784, 768]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    clamp_min_2: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_7, 1e-12)
    expand_6: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_2, [8, 16, 48, 784]);  clamp_min_2 = None
    div_9: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_19, expand_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    clamp_min_3: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_9, 1e-12)
    expand_7: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_3, [8, 16, 48, 784]);  clamp_min_3 = None
    div_10: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_20, expand_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    view_27: "f32[8, 16, 48, 48]" = torch.ops.aten.reshape.default(bmm_2, [8, 16, 48, 48]);  bmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    view_33: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_1, [8, 784, 768]);  mm_1 = None
    add_42: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_33, primals_138);  view_33 = primals_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_62: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_6, 0.5)
    mul_63: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_6, 0.7071067811865476)
    erf_4: "f32[8, 768, 28, 28]" = torch.ops.aten.erf.default(mul_63);  mul_63 = None
    add_46: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    mul_64: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_62, add_46);  mul_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_35: "f32[8, 768, 784]" = torch.ops.aten.reshape.default(convolution_7, [8, 768, 784]);  convolution_7 = None
    permute_18: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_35, [0, 2, 1]);  view_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_37: "f32[8, 784, 3072]" = torch.ops.aten.reshape.default(addmm_4, [8, 784, 3072]);  addmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_76: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_37, 0.7071067811865476)
    erf_5: "f32[8, 784, 3072]" = torch.ops.aten.erf.default(mul_76);  mul_76 = None
    add_55: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_39: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(addmm_5, [8, 784, 768]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    clamp_min_4: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_11, 1e-12)
    expand_12: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_4, [8, 16, 48, 784]);  clamp_min_4 = None
    div_12: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_30, expand_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    clamp_min_5: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_13, 1e-12)
    expand_13: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_5, [8, 16, 48, 784]);  clamp_min_5 = None
    div_13: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_31, expand_13)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    view_45: "f32[8, 16, 48, 48]" = torch.ops.aten.reshape.default(bmm_4, [8, 16, 48, 48]);  bmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    view_51: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_2, [8, 784, 768]);  mm_2 = None
    add_59: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_51, primals_158);  view_51 = primals_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_85: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_8, 0.5)
    mul_86: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_8, 0.7071067811865476)
    erf_6: "f32[8, 768, 28, 28]" = torch.ops.aten.erf.default(mul_86);  mul_86 = None
    add_63: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    mul_87: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_85, add_63);  mul_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_53: "f32[8, 768, 784]" = torch.ops.aten.reshape.default(convolution_9, [8, 768, 784]);  convolution_9 = None
    permute_27: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_53, [0, 2, 1]);  view_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_55: "f32[8, 784, 3072]" = torch.ops.aten.reshape.default(addmm_7, [8, 784, 3072]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_99: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_55, 0.7071067811865476)
    erf_7: "f32[8, 784, 3072]" = torch.ops.aten.erf.default(mul_99);  mul_99 = None
    add_72: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_57: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(addmm_8, [8, 784, 768]);  addmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    clamp_min_6: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_15, 1e-12)
    expand_18: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_6, [8, 16, 48, 784]);  clamp_min_6 = None
    div_15: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_41, expand_18)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    clamp_min_7: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_17, 1e-12)
    expand_19: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_7, [8, 16, 48, 784]);  clamp_min_7 = None
    div_16: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_42, expand_19)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    view_63: "f32[8, 16, 48, 48]" = torch.ops.aten.reshape.default(bmm_6, [8, 16, 48, 48]);  bmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    view_69: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_3, [8, 784, 768]);  mm_3 = None
    add_76: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_69, primals_178);  view_69 = primals_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_108: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_10, 0.5)
    mul_109: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_10, 0.7071067811865476)
    erf_8: "f32[8, 768, 28, 28]" = torch.ops.aten.erf.default(mul_109);  mul_109 = None
    add_80: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    mul_110: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_108, add_80);  mul_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_71: "f32[8, 768, 784]" = torch.ops.aten.reshape.default(convolution_11, [8, 768, 784]);  convolution_11 = None
    permute_36: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_71, [0, 2, 1]);  view_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_73: "f32[8, 784, 3072]" = torch.ops.aten.reshape.default(addmm_10, [8, 784, 3072]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_122: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_73, 0.7071067811865476)
    erf_9: "f32[8, 784, 3072]" = torch.ops.aten.erf.default(mul_122);  mul_122 = None
    add_89: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_75: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(addmm_11, [8, 784, 768]);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    clamp_min_8: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_19, 1e-12)
    expand_24: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_8, [8, 16, 48, 784]);  clamp_min_8 = None
    div_18: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_52, expand_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    clamp_min_9: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_21, 1e-12)
    expand_25: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_9, [8, 16, 48, 784]);  clamp_min_9 = None
    div_19: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_53, expand_25)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    view_81: "f32[8, 16, 48, 48]" = torch.ops.aten.reshape.default(bmm_8, [8, 16, 48, 48]);  bmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    view_87: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_4, [8, 784, 768]);  mm_4 = None
    add_93: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_87, primals_198);  view_87 = primals_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_131: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_12, 0.5)
    mul_132: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_12, 0.7071067811865476)
    erf_10: "f32[8, 768, 28, 28]" = torch.ops.aten.erf.default(mul_132);  mul_132 = None
    add_97: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    mul_133: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_131, add_97);  mul_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_89: "f32[8, 768, 784]" = torch.ops.aten.reshape.default(convolution_13, [8, 768, 784]);  convolution_13 = None
    permute_45: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_89, [0, 2, 1]);  view_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_91: "f32[8, 784, 3072]" = torch.ops.aten.reshape.default(addmm_13, [8, 784, 3072]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_145: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_91, 0.7071067811865476)
    erf_11: "f32[8, 784, 3072]" = torch.ops.aten.erf.default(mul_145);  mul_145 = None
    add_106: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_93: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(addmm_14, [8, 784, 768]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    clamp_min_10: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_23, 1e-12)
    expand_30: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_10, [8, 16, 48, 784]);  clamp_min_10 = None
    div_21: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_63, expand_30)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    clamp_min_11: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_25, 1e-12)
    expand_31: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_11, [8, 16, 48, 784]);  clamp_min_11 = None
    div_22: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_64, expand_31)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    view_99: "f32[8, 16, 48, 48]" = torch.ops.aten.reshape.default(bmm_10, [8, 16, 48, 48]);  bmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    view_105: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_5, [8, 784, 768]);  mm_5 = None
    add_110: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_105, primals_218);  view_105 = primals_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_154: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_14, 0.5)
    mul_155: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_14, 0.7071067811865476)
    erf_12: "f32[8, 768, 28, 28]" = torch.ops.aten.erf.default(mul_155);  mul_155 = None
    add_114: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
    mul_156: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_154, add_114);  mul_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_107: "f32[8, 768, 784]" = torch.ops.aten.reshape.default(convolution_15, [8, 768, 784]);  convolution_15 = None
    permute_54: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_107, [0, 2, 1]);  view_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_109: "f32[8, 784, 3072]" = torch.ops.aten.reshape.default(addmm_16, [8, 784, 3072]);  addmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_168: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_109, 0.7071067811865476)
    erf_13: "f32[8, 784, 3072]" = torch.ops.aten.erf.default(mul_168);  mul_168 = None
    add_123: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_111: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(addmm_17, [8, 784, 768]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    clamp_min_12: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_27, 1e-12)
    expand_36: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_12, [8, 16, 48, 784]);  clamp_min_12 = None
    div_24: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_74, expand_36)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    clamp_min_13: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_29, 1e-12)
    expand_37: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_13, [8, 16, 48, 784]);  clamp_min_13 = None
    div_25: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_75, expand_37)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    view_117: "f32[8, 16, 48, 48]" = torch.ops.aten.reshape.default(bmm_12, [8, 16, 48, 48]);  bmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    view_123: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_6, [8, 784, 768]);  mm_6 = None
    add_127: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_123, primals_238);  view_123 = primals_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_177: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_16, 0.5)
    mul_178: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_16, 0.7071067811865476)
    erf_14: "f32[8, 768, 28, 28]" = torch.ops.aten.erf.default(mul_178);  mul_178 = None
    add_131: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
    mul_179: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_177, add_131);  mul_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_125: "f32[8, 768, 784]" = torch.ops.aten.reshape.default(convolution_17, [8, 768, 784]);  convolution_17 = None
    permute_63: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_125, [0, 2, 1]);  view_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_127: "f32[8, 784, 3072]" = torch.ops.aten.reshape.default(addmm_19, [8, 784, 3072]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_191: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_127, 0.7071067811865476)
    erf_15: "f32[8, 784, 3072]" = torch.ops.aten.erf.default(mul_191);  mul_191 = None
    add_140: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_129: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(addmm_20, [8, 784, 768]);  addmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    clamp_min_14: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_31, 1e-12)
    expand_42: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_14, [8, 16, 48, 784]);  clamp_min_14 = None
    div_27: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_85, expand_42)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    clamp_min_15: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_33, 1e-12)
    expand_43: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_15, [8, 16, 48, 784]);  clamp_min_15 = None
    div_28: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_86, expand_43)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    view_135: "f32[8, 16, 48, 48]" = torch.ops.aten.reshape.default(bmm_14, [8, 16, 48, 48]);  bmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    view_141: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_7, [8, 784, 768]);  mm_7 = None
    add_144: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_141, primals_258);  view_141 = primals_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_200: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_18, 0.5)
    mul_201: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_18, 0.7071067811865476)
    erf_16: "f32[8, 768, 28, 28]" = torch.ops.aten.erf.default(mul_201);  mul_201 = None
    add_148: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
    mul_202: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_200, add_148);  mul_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_143: "f32[8, 768, 784]" = torch.ops.aten.reshape.default(convolution_19, [8, 768, 784]);  convolution_19 = None
    permute_72: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_143, [0, 2, 1]);  view_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_145: "f32[8, 784, 3072]" = torch.ops.aten.reshape.default(addmm_22, [8, 784, 3072]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_214: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_145, 0.7071067811865476)
    erf_17: "f32[8, 784, 3072]" = torch.ops.aten.erf.default(mul_214);  mul_214 = None
    add_157: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_147: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(addmm_23, [8, 784, 768]);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    clamp_min_16: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_35, 1e-12)
    expand_48: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_16, [8, 16, 48, 784]);  clamp_min_16 = None
    div_30: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_96, expand_48)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    clamp_min_17: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_37, 1e-12)
    expand_49: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_17, [8, 16, 48, 784]);  clamp_min_17 = None
    div_31: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_97, expand_49)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    view_153: "f32[8, 16, 48, 48]" = torch.ops.aten.reshape.default(bmm_16, [8, 16, 48, 48]);  bmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    view_159: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_8, [8, 784, 768]);  mm_8 = None
    add_161: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_159, primals_278);  view_159 = primals_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_223: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_20, 0.5)
    mul_224: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_20, 0.7071067811865476)
    erf_18: "f32[8, 768, 28, 28]" = torch.ops.aten.erf.default(mul_224);  mul_224 = None
    add_165: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
    mul_225: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_223, add_165);  mul_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_161: "f32[8, 768, 784]" = torch.ops.aten.reshape.default(convolution_21, [8, 768, 784]);  convolution_21 = None
    permute_81: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_161, [0, 2, 1]);  view_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_163: "f32[8, 784, 3072]" = torch.ops.aten.reshape.default(addmm_25, [8, 784, 3072]);  addmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_237: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_163, 0.7071067811865476)
    erf_19: "f32[8, 784, 3072]" = torch.ops.aten.erf.default(mul_237);  mul_237 = None
    add_174: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_165: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(addmm_26, [8, 784, 768]);  addmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    clamp_min_18: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_39, 1e-12)
    expand_54: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_18, [8, 16, 48, 784]);  clamp_min_18 = None
    div_33: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_107, expand_54)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    clamp_min_19: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_41, 1e-12)
    expand_55: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_19, [8, 16, 48, 784]);  clamp_min_19 = None
    div_34: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_108, expand_55)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    view_171: "f32[8, 16, 48, 48]" = torch.ops.aten.reshape.default(bmm_18, [8, 16, 48, 48]);  bmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    view_177: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_9, [8, 784, 768]);  mm_9 = None
    add_178: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_177, primals_298);  view_177 = primals_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_246: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_22, 0.5)
    mul_247: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_22, 0.7071067811865476)
    erf_20: "f32[8, 768, 28, 28]" = torch.ops.aten.erf.default(mul_247);  mul_247 = None
    add_182: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
    mul_248: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_246, add_182);  mul_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_179: "f32[8, 768, 784]" = torch.ops.aten.reshape.default(convolution_23, [8, 768, 784]);  convolution_23 = None
    permute_90: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_179, [0, 2, 1]);  view_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_181: "f32[8, 784, 3072]" = torch.ops.aten.reshape.default(addmm_28, [8, 784, 3072]);  addmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_260: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_181, 0.7071067811865476)
    erf_21: "f32[8, 784, 3072]" = torch.ops.aten.erf.default(mul_260);  mul_260 = None
    add_191: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_183: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(addmm_29, [8, 784, 768]);  addmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    clamp_min_20: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_43, 1e-12)
    expand_60: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_20, [8, 16, 48, 784]);  clamp_min_20 = None
    div_36: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_118, expand_60)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    clamp_min_21: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_45, 1e-12)
    expand_61: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_21, [8, 16, 48, 784]);  clamp_min_21 = None
    div_37: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_119, expand_61)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    view_189: "f32[8, 16, 48, 48]" = torch.ops.aten.reshape.default(bmm_20, [8, 16, 48, 48]);  bmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    view_195: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_10, [8, 784, 768]);  mm_10 = None
    add_195: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_195, primals_318);  view_195 = primals_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_269: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_24, 0.5)
    mul_270: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_24, 0.7071067811865476)
    erf_22: "f32[8, 768, 28, 28]" = torch.ops.aten.erf.default(mul_270);  mul_270 = None
    add_199: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
    mul_271: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_269, add_199);  mul_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_197: "f32[8, 768, 784]" = torch.ops.aten.reshape.default(convolution_25, [8, 768, 784]);  convolution_25 = None
    permute_99: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_197, [0, 2, 1]);  view_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_199: "f32[8, 784, 3072]" = torch.ops.aten.reshape.default(addmm_31, [8, 784, 3072]);  addmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_283: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_199, 0.7071067811865476)
    erf_23: "f32[8, 784, 3072]" = torch.ops.aten.erf.default(mul_283);  mul_283 = None
    add_208: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_201: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(addmm_32, [8, 784, 768]);  addmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    clamp_min_22: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_47, 1e-12)
    expand_66: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_22, [8, 16, 48, 784]);  clamp_min_22 = None
    div_39: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_129, expand_66)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    clamp_min_23: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_49, 1e-12)
    expand_67: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_23, [8, 16, 48, 784]);  clamp_min_23 = None
    div_40: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_130, expand_67)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    view_207: "f32[8, 16, 48, 48]" = torch.ops.aten.reshape.default(bmm_22, [8, 16, 48, 48]);  bmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    view_213: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_11, [8, 784, 768]);  mm_11 = None
    add_212: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_213, primals_338);  view_213 = primals_338 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_292: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_26, 0.5)
    mul_293: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_26, 0.7071067811865476)
    erf_24: "f32[8, 768, 28, 28]" = torch.ops.aten.erf.default(mul_293);  mul_293 = None
    add_216: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(erf_24, 1);  erf_24 = None
    mul_294: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_292, add_216);  mul_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_215: "f32[8, 768, 784]" = torch.ops.aten.reshape.default(convolution_27, [8, 768, 784]);  convolution_27 = None
    permute_108: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_215, [0, 2, 1]);  view_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_217: "f32[8, 784, 3072]" = torch.ops.aten.reshape.default(addmm_34, [8, 784, 3072]);  addmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_306: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_217, 0.7071067811865476)
    erf_25: "f32[8, 784, 3072]" = torch.ops.aten.erf.default(mul_306);  mul_306 = None
    add_225: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(erf_25, 1);  erf_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_219: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(addmm_35, [8, 784, 768]);  addmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    clamp_min_24: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_51, 1e-12)
    expand_72: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_24, [8, 16, 48, 784]);  clamp_min_24 = None
    div_42: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_140, expand_72)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    clamp_min_25: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_53, 1e-12)
    expand_73: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_25, [8, 16, 48, 784]);  clamp_min_25 = None
    div_43: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_141, expand_73)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    view_225: "f32[8, 16, 48, 48]" = torch.ops.aten.reshape.default(bmm_24, [8, 16, 48, 48]);  bmm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    view_231: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_12, [8, 784, 768]);  mm_12 = None
    add_229: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_231, primals_358);  view_231 = primals_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_315: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_28, 0.5)
    mul_316: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_28, 0.7071067811865476)
    erf_26: "f32[8, 768, 28, 28]" = torch.ops.aten.erf.default(mul_316);  mul_316 = None
    add_233: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(erf_26, 1);  erf_26 = None
    mul_317: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_315, add_233);  mul_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_233: "f32[8, 768, 784]" = torch.ops.aten.reshape.default(convolution_29, [8, 768, 784]);  convolution_29 = None
    permute_117: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_233, [0, 2, 1]);  view_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_235: "f32[8, 784, 3072]" = torch.ops.aten.reshape.default(addmm_37, [8, 784, 3072]);  addmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_329: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_235, 0.7071067811865476)
    erf_27: "f32[8, 784, 3072]" = torch.ops.aten.erf.default(mul_329);  mul_329 = None
    add_242: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(erf_27, 1);  erf_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_237: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(addmm_38, [8, 784, 768]);  addmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    clamp_min_26: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_55, 1e-12)
    expand_78: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_26, [8, 16, 48, 784]);  clamp_min_26 = None
    div_45: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_151, expand_78)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    clamp_min_27: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_57, 1e-12)
    expand_79: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_27, [8, 16, 48, 784]);  clamp_min_27 = None
    div_46: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_152, expand_79)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    view_243: "f32[8, 16, 48, 48]" = torch.ops.aten.reshape.default(bmm_26, [8, 16, 48, 48]);  bmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    view_249: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_13, [8, 784, 768]);  mm_13 = None
    add_246: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_249, primals_378);  view_249 = primals_378 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_338: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_30, 0.5)
    mul_339: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_30, 0.7071067811865476)
    erf_28: "f32[8, 768, 28, 28]" = torch.ops.aten.erf.default(mul_339);  mul_339 = None
    add_250: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(erf_28, 1);  erf_28 = None
    mul_340: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_338, add_250);  mul_338 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_251: "f32[8, 768, 784]" = torch.ops.aten.reshape.default(convolution_31, [8, 768, 784]);  convolution_31 = None
    permute_126: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_251, [0, 2, 1]);  view_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_253: "f32[8, 784, 3072]" = torch.ops.aten.reshape.default(addmm_40, [8, 784, 3072]);  addmm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_352: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_253, 0.7071067811865476)
    erf_29: "f32[8, 784, 3072]" = torch.ops.aten.erf.default(mul_352);  mul_352 = None
    add_259: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(erf_29, 1);  erf_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_255: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(addmm_41, [8, 784, 768]);  addmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    clamp_min_28: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_59, 1e-12)
    expand_84: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_28, [8, 16, 48, 784]);  clamp_min_28 = None
    div_48: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_162, expand_84)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    clamp_min_29: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_61, 1e-12)
    expand_85: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_29, [8, 16, 48, 784]);  clamp_min_29 = None
    div_49: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_163, expand_85)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    view_261: "f32[8, 16, 48, 48]" = torch.ops.aten.reshape.default(bmm_28, [8, 16, 48, 48]);  bmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    view_267: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_14, [8, 784, 768]);  mm_14 = None
    add_263: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_267, primals_398);  view_267 = primals_398 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_361: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_32, 0.5)
    mul_362: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_32, 0.7071067811865476)
    erf_30: "f32[8, 768, 28, 28]" = torch.ops.aten.erf.default(mul_362);  mul_362 = None
    add_267: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(erf_30, 1);  erf_30 = None
    mul_363: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_361, add_267);  mul_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_269: "f32[8, 768, 784]" = torch.ops.aten.reshape.default(convolution_33, [8, 768, 784]);  convolution_33 = None
    permute_135: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_269, [0, 2, 1]);  view_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_271: "f32[8, 784, 3072]" = torch.ops.aten.reshape.default(addmm_43, [8, 784, 3072]);  addmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_375: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_271, 0.7071067811865476)
    erf_31: "f32[8, 784, 3072]" = torch.ops.aten.erf.default(mul_375);  mul_375 = None
    add_276: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(erf_31, 1);  erf_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_273: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(addmm_44, [8, 784, 768]);  addmm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    clamp_min_30: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_63, 1e-12)
    expand_90: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_30, [8, 16, 48, 784]);  clamp_min_30 = None
    div_51: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_173, expand_90)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    clamp_min_31: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_65, 1e-12)
    expand_91: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_31, [8, 16, 48, 784]);  clamp_min_31 = None
    div_52: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_174, expand_91)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    view_279: "f32[8, 16, 48, 48]" = torch.ops.aten.reshape.default(bmm_30, [8, 16, 48, 48]);  bmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    view_285: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_15, [8, 784, 768]);  mm_15 = None
    add_280: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_285, primals_418);  view_285 = primals_418 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_384: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_34, 0.5)
    mul_385: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_34, 0.7071067811865476)
    erf_32: "f32[8, 768, 28, 28]" = torch.ops.aten.erf.default(mul_385);  mul_385 = None
    add_284: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(erf_32, 1);  erf_32 = None
    mul_386: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_384, add_284);  mul_384 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_287: "f32[8, 768, 784]" = torch.ops.aten.reshape.default(convolution_35, [8, 768, 784]);  convolution_35 = None
    permute_144: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_287, [0, 2, 1]);  view_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_289: "f32[8, 784, 3072]" = torch.ops.aten.reshape.default(addmm_46, [8, 784, 3072]);  addmm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_398: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_289, 0.7071067811865476)
    erf_33: "f32[8, 784, 3072]" = torch.ops.aten.erf.default(mul_398);  mul_398 = None
    add_293: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(erf_33, 1);  erf_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_291: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(addmm_47, [8, 784, 768]);  addmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    clamp_min_32: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_67, 1e-12)
    expand_96: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_32, [8, 16, 48, 784]);  clamp_min_32 = None
    div_54: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_184, expand_96)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    clamp_min_33: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_69, 1e-12)
    expand_97: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_33, [8, 16, 48, 784]);  clamp_min_33 = None
    div_55: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_185, expand_97)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    view_297: "f32[8, 16, 48, 48]" = torch.ops.aten.reshape.default(bmm_32, [8, 16, 48, 48]);  bmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    view_303: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_16, [8, 784, 768]);  mm_16 = None
    add_297: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_303, primals_438);  view_303 = primals_438 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_407: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_36, 0.5)
    mul_408: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_36, 0.7071067811865476)
    erf_34: "f32[8, 768, 28, 28]" = torch.ops.aten.erf.default(mul_408);  mul_408 = None
    add_301: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(erf_34, 1);  erf_34 = None
    mul_409: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_407, add_301);  mul_407 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_305: "f32[8, 768, 784]" = torch.ops.aten.reshape.default(convolution_37, [8, 768, 784]);  convolution_37 = None
    permute_153: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_305, [0, 2, 1]);  view_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_307: "f32[8, 784, 3072]" = torch.ops.aten.reshape.default(addmm_49, [8, 784, 3072]);  addmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_421: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_307, 0.7071067811865476)
    erf_35: "f32[8, 784, 3072]" = torch.ops.aten.erf.default(mul_421);  mul_421 = None
    add_310: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(erf_35, 1);  erf_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_309: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(addmm_50, [8, 784, 768]);  addmm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    clamp_min_34: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_71, 1e-12)
    expand_102: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_34, [8, 16, 48, 784]);  clamp_min_34 = None
    div_57: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_195, expand_102)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    clamp_min_35: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_73, 1e-12)
    expand_103: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_35, [8, 16, 48, 784]);  clamp_min_35 = None
    div_58: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_196, expand_103)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    view_315: "f32[8, 16, 48, 48]" = torch.ops.aten.reshape.default(bmm_34, [8, 16, 48, 48]);  bmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    view_321: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_17, [8, 784, 768]);  mm_17 = None
    add_314: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_321, primals_458);  view_321 = primals_458 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_430: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_38, 0.5)
    mul_431: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_38, 0.7071067811865476)
    erf_36: "f32[8, 768, 28, 28]" = torch.ops.aten.erf.default(mul_431);  mul_431 = None
    add_318: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(erf_36, 1);  erf_36 = None
    mul_432: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_430, add_318);  mul_430 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_323: "f32[8, 768, 784]" = torch.ops.aten.reshape.default(convolution_39, [8, 768, 784]);  convolution_39 = None
    permute_162: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_323, [0, 2, 1]);  view_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_325: "f32[8, 784, 3072]" = torch.ops.aten.reshape.default(addmm_52, [8, 784, 3072]);  addmm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_444: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_325, 0.7071067811865476)
    erf_37: "f32[8, 784, 3072]" = torch.ops.aten.erf.default(mul_444);  mul_444 = None
    add_327: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(erf_37, 1);  erf_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_327: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(addmm_53, [8, 784, 768]);  addmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    clamp_min_36: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_75, 1e-12)
    expand_108: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_36, [8, 16, 48, 784]);  clamp_min_36 = None
    div_60: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_206, expand_108)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    clamp_min_37: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_77, 1e-12)
    expand_109: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_37, [8, 16, 48, 784]);  clamp_min_37 = None
    div_61: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_207, expand_109)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    view_333: "f32[8, 16, 48, 48]" = torch.ops.aten.reshape.default(bmm_36, [8, 16, 48, 48]);  bmm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    view_339: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_18, [8, 784, 768]);  mm_18 = None
    add_331: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_339, primals_478);  view_339 = primals_478 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_453: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_40, 0.5)
    mul_454: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_40, 0.7071067811865476)
    erf_38: "f32[8, 768, 28, 28]" = torch.ops.aten.erf.default(mul_454);  mul_454 = None
    add_335: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(erf_38, 1);  erf_38 = None
    mul_455: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_453, add_335);  mul_453 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_341: "f32[8, 768, 784]" = torch.ops.aten.reshape.default(convolution_41, [8, 768, 784]);  convolution_41 = None
    permute_171: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_341, [0, 2, 1]);  view_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_343: "f32[8, 784, 3072]" = torch.ops.aten.reshape.default(addmm_55, [8, 784, 3072]);  addmm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_467: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_343, 0.7071067811865476)
    erf_39: "f32[8, 784, 3072]" = torch.ops.aten.erf.default(mul_467);  mul_467 = None
    add_344: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(erf_39, 1);  erf_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_345: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(addmm_56, [8, 784, 768]);  addmm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    clamp_min_38: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_79, 1e-12)
    expand_114: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_38, [8, 16, 48, 784]);  clamp_min_38 = None
    div_63: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_217, expand_114)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    clamp_min_39: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_81, 1e-12)
    expand_115: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_39, [8, 16, 48, 784]);  clamp_min_39 = None
    div_64: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_218, expand_115)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    view_351: "f32[8, 16, 48, 48]" = torch.ops.aten.reshape.default(bmm_38, [8, 16, 48, 48]);  bmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    view_357: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_19, [8, 784, 768]);  mm_19 = None
    add_348: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_357, primals_498);  view_357 = primals_498 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_476: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_42, 0.5)
    mul_477: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_42, 0.7071067811865476)
    erf_40: "f32[8, 768, 28, 28]" = torch.ops.aten.erf.default(mul_477);  mul_477 = None
    add_352: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(erf_40, 1);  erf_40 = None
    mul_478: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_476, add_352);  mul_476 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_359: "f32[8, 768, 784]" = torch.ops.aten.reshape.default(convolution_43, [8, 768, 784]);  convolution_43 = None
    permute_180: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_359, [0, 2, 1]);  view_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_361: "f32[8, 784, 3072]" = torch.ops.aten.reshape.default(addmm_58, [8, 784, 3072]);  addmm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_490: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_361, 0.7071067811865476)
    erf_41: "f32[8, 784, 3072]" = torch.ops.aten.erf.default(mul_490);  mul_490 = None
    add_361: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(erf_41, 1);  erf_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_363: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(addmm_59, [8, 784, 768]);  addmm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    clamp_min_40: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_83, 1e-12)
    expand_120: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_40, [8, 16, 48, 784]);  clamp_min_40 = None
    div_66: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_228, expand_120)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    clamp_min_41: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_85, 1e-12)
    expand_121: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_41, [8, 16, 48, 784]);  clamp_min_41 = None
    div_67: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_229, expand_121)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    view_369: "f32[8, 16, 48, 48]" = torch.ops.aten.reshape.default(bmm_40, [8, 16, 48, 48]);  bmm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    view_375: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_20, [8, 784, 768]);  mm_20 = None
    add_365: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_375, primals_518);  view_375 = primals_518 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_499: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_44, 0.5)
    mul_500: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_44, 0.7071067811865476)
    erf_42: "f32[8, 768, 28, 28]" = torch.ops.aten.erf.default(mul_500);  mul_500 = None
    add_369: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(erf_42, 1);  erf_42 = None
    mul_501: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_499, add_369);  mul_499 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_377: "f32[8, 768, 784]" = torch.ops.aten.reshape.default(convolution_45, [8, 768, 784]);  convolution_45 = None
    permute_189: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_377, [0, 2, 1]);  view_377 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_379: "f32[8, 784, 3072]" = torch.ops.aten.reshape.default(addmm_61, [8, 784, 3072]);  addmm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_513: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_379, 0.7071067811865476)
    erf_43: "f32[8, 784, 3072]" = torch.ops.aten.erf.default(mul_513);  mul_513 = None
    add_378: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(erf_43, 1);  erf_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_381: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(addmm_62, [8, 784, 768]);  addmm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    clamp_min_42: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_87, 1e-12)
    expand_126: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_42, [8, 16, 48, 784]);  clamp_min_42 = None
    div_69: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_239, expand_126)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    clamp_min_43: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_89, 1e-12)
    expand_127: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_43, [8, 16, 48, 784]);  clamp_min_43 = None
    div_70: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_240, expand_127)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    view_387: "f32[8, 16, 48, 48]" = torch.ops.aten.reshape.default(bmm_42, [8, 16, 48, 48]);  bmm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    view_393: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_21, [8, 784, 768]);  mm_21 = None
    add_382: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_393, primals_538);  view_393 = primals_538 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_522: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_46, 0.5)
    mul_523: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_46, 0.7071067811865476)
    erf_44: "f32[8, 768, 28, 28]" = torch.ops.aten.erf.default(mul_523);  mul_523 = None
    add_386: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(erf_44, 1);  erf_44 = None
    mul_524: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_522, add_386);  mul_522 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_395: "f32[8, 768, 784]" = torch.ops.aten.reshape.default(convolution_47, [8, 768, 784]);  convolution_47 = None
    permute_198: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_395, [0, 2, 1]);  view_395 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_397: "f32[8, 784, 3072]" = torch.ops.aten.reshape.default(addmm_64, [8, 784, 3072]);  addmm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_536: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_397, 0.7071067811865476)
    erf_45: "f32[8, 784, 3072]" = torch.ops.aten.erf.default(mul_536);  mul_536 = None
    add_395: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(erf_45, 1);  erf_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_399: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(addmm_65, [8, 784, 768]);  addmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    clamp_min_44: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_91, 1e-12)
    expand_132: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_44, [8, 16, 48, 784]);  clamp_min_44 = None
    div_72: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_250, expand_132)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    clamp_min_45: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_93, 1e-12)
    expand_133: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_45, [8, 16, 48, 784]);  clamp_min_45 = None
    div_73: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_251, expand_133)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    view_405: "f32[8, 16, 48, 48]" = torch.ops.aten.reshape.default(bmm_44, [8, 16, 48, 48]);  bmm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    view_411: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_22, [8, 784, 768]);  mm_22 = None
    add_399: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_411, primals_558);  view_411 = primals_558 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_545: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_48, 0.5)
    mul_546: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_48, 0.7071067811865476)
    erf_46: "f32[8, 768, 28, 28]" = torch.ops.aten.erf.default(mul_546);  mul_546 = None
    add_403: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(erf_46, 1);  erf_46 = None
    mul_547: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_545, add_403);  mul_545 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_413: "f32[8, 768, 784]" = torch.ops.aten.reshape.default(convolution_49, [8, 768, 784]);  convolution_49 = None
    permute_207: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_413, [0, 2, 1]);  view_413 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_415: "f32[8, 784, 3072]" = torch.ops.aten.reshape.default(addmm_67, [8, 784, 3072]);  addmm_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_559: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_415, 0.7071067811865476)
    erf_47: "f32[8, 784, 3072]" = torch.ops.aten.erf.default(mul_559);  mul_559 = None
    add_412: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(erf_47, 1);  erf_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_417: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(addmm_68, [8, 784, 768]);  addmm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    clamp_min_46: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_95, 1e-12)
    expand_138: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_46, [8, 16, 48, 784]);  clamp_min_46 = None
    div_75: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_261, expand_138)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    clamp_min_47: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_97, 1e-12)
    expand_139: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_47, [8, 16, 48, 784]);  clamp_min_47 = None
    div_76: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_262, expand_139)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    view_423: "f32[8, 16, 48, 48]" = torch.ops.aten.reshape.default(bmm_46, [8, 16, 48, 48]);  bmm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    view_429: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_23, [8, 784, 768]);  mm_23 = None
    add_416: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_429, primals_578);  view_429 = primals_578 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_568: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_50, 0.5)
    mul_569: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_50, 0.7071067811865476)
    erf_48: "f32[8, 768, 28, 28]" = torch.ops.aten.erf.default(mul_569);  mul_569 = None
    add_420: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(erf_48, 1);  erf_48 = None
    mul_570: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_568, add_420);  mul_568 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_431: "f32[8, 768, 784]" = torch.ops.aten.reshape.default(convolution_51, [8, 768, 784]);  convolution_51 = None
    permute_216: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_431, [0, 2, 1]);  view_431 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_433: "f32[8, 784, 3072]" = torch.ops.aten.reshape.default(addmm_70, [8, 784, 3072]);  addmm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_582: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_433, 0.7071067811865476)
    erf_49: "f32[8, 784, 3072]" = torch.ops.aten.erf.default(mul_582);  mul_582 = None
    add_429: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(erf_49, 1);  erf_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_435: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(addmm_71, [8, 784, 768]);  addmm_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:181, code: x_norm1 = self.norm1(x)
    sub_123: "f32[8, 785, 768]" = torch.ops.aten.sub.Tensor(cat_3, getitem_271);  cat_3 = getitem_271 = None
    mul_585: "f32[8, 785, 768]" = torch.ops.aten.mul.Tensor(sub_123, rsqrt_99);  sub_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_447: "f32[8, 1, 3072]" = torch.ops.aten.reshape.default(mm_24, [8, 1, 3072]);  mm_24 = None
    add_436: "f32[8, 1, 3072]" = torch.ops.aten.add.Tensor(view_447, primals_606);  view_447 = primals_606 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_591: "f32[8, 1, 3072]" = torch.ops.aten.mul.Tensor(add_436, 0.7071067811865476)
    erf_50: "f32[8, 1, 3072]" = torch.ops.aten.erf.default(mul_591);  mul_591 = None
    add_437: "f32[8, 1, 3072]" = torch.ops.aten.add.Tensor(erf_50, 1);  erf_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_449: "f32[8, 1, 768]" = torch.ops.aten.reshape.default(addmm_76, [8, 1, 768]);  addmm_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_461: "f32[8, 1, 3072]" = torch.ops.aten.reshape.default(mm_25, [8, 1, 3072]);  mm_25 = None
    add_444: "f32[8, 1, 3072]" = torch.ops.aten.add.Tensor(view_461, primals_622);  view_461 = primals_622 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_600: "f32[8, 1, 3072]" = torch.ops.aten.mul.Tensor(add_444, 0.7071067811865476)
    erf_51: "f32[8, 1, 3072]" = torch.ops.aten.erf.default(mul_600);  mul_600 = None
    add_445: "f32[8, 1, 3072]" = torch.ops.aten.add.Tensor(erf_51, 1);  erf_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_463: "f32[8, 1, 768]" = torch.ops.aten.reshape.default(addmm_81, [8, 1, 768]);  addmm_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:462, code: return x if pre_logits else self.head(x)
    mm_26: "f32[8, 768]" = torch.ops.aten.mm.default(tangents_1, permute_240);  permute_240 = None
    permute_241: "f32[1000, 8]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
    mm_27: "f32[1000, 768]" = torch.ops.aten.mm.default(permute_241, clone_271);  permute_241 = clone_271 = None
    permute_242: "f32[768, 1000]" = torch.ops.aten.permute.default(mm_27, [1, 0]);  mm_27 = None
    sum_73: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view_464: "f32[1000]" = torch.ops.aten.reshape.default(sum_73, [1000]);  sum_73 = None
    permute_243: "f32[1000, 768]" = torch.ops.aten.permute.default(permute_242, [1, 0]);  permute_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:460, code: x = x[:, 1:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
    full_default_2: "f32[8, 785, 768]" = torch.ops.aten.full.default([8, 785, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    select_scatter: "f32[8, 785, 768]" = torch.ops.aten.select_scatter.default(full_default_2, mm_26, 1, 0);  mm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:455, code: x = self.norm(x)
    mul_606: "f32[8, 785, 768]" = torch.ops.aten.mul.Tensor(select_scatter, primals_625);  primals_625 = None
    mul_607: "f32[8, 785, 768]" = torch.ops.aten.mul.Tensor(mul_606, 768)
    sum_74: "f32[8, 785, 1]" = torch.ops.aten.sum.dim_IntList(mul_606, [2], True)
    mul_608: "f32[8, 785, 768]" = torch.ops.aten.mul.Tensor(mul_606, mul_603);  mul_606 = None
    sum_75: "f32[8, 785, 1]" = torch.ops.aten.sum.dim_IntList(mul_608, [2], True);  mul_608 = None
    mul_609: "f32[8, 785, 768]" = torch.ops.aten.mul.Tensor(mul_603, sum_75);  sum_75 = None
    sub_129: "f32[8, 785, 768]" = torch.ops.aten.sub.Tensor(mul_607, sum_74);  mul_607 = sum_74 = None
    sub_130: "f32[8, 785, 768]" = torch.ops.aten.sub.Tensor(sub_129, mul_609);  sub_129 = mul_609 = None
    mul_610: "f32[8, 785, 768]" = torch.ops.aten.mul.Tensor(div_78, sub_130);  div_78 = sub_130 = None
    mul_611: "f32[8, 785, 768]" = torch.ops.aten.mul.Tensor(select_scatter, mul_603);  mul_603 = None
    sum_76: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_611, [0, 1]);  mul_611 = None
    sum_77: "f32[768]" = torch.ops.aten.sum.dim_IntList(select_scatter, [0, 1]);  select_scatter = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:191, code: x = torch.cat([cls_token, x[:, 1:]], dim=1)
    slice_44: "f32[8, 1, 768]" = torch.ops.aten.slice.Tensor(mul_610, 1, 0, 1)
    slice_45: "f32[8, 784, 768]" = torch.ops.aten.slice.Tensor(mul_610, 1, 1, 785)
    slice_scatter_1: "f32[8, 785, 768]" = torch.ops.aten.slice_scatter.default(full_default_2, slice_45, 1, 1, 9223372036854775807);  slice_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:191, code: x = torch.cat([cls_token, x[:, 1:]], dim=1)
    add_449: "f32[8, 785, 768]" = torch.ops.aten.add.Tensor(mul_610, slice_scatter_1);  mul_610 = slice_scatter_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:190, code: cls_token = self.gamma2 * self.mlp(cls_token)
    mul_612: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(slice_44, primals_101);  primals_101 = None
    mul_613: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(slice_44, view_463);  slice_44 = view_463 = None
    sum_78: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_613, [0, 1], True);  mul_613 = None
    view_465: "f32[768]" = torch.ops.aten.reshape.default(sum_78, [768]);  sum_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_466: "f32[8, 768]" = torch.ops.aten.reshape.default(mul_612, [8, 768]);  mul_612 = None
    mm_28: "f32[8, 3072]" = torch.ops.aten.mm.default(view_466, permute_244);  permute_244 = None
    permute_245: "f32[768, 8]" = torch.ops.aten.permute.default(view_466, [1, 0])
    mm_29: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_245, view_462);  permute_245 = view_462 = None
    permute_246: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_29, [1, 0]);  mm_29 = None
    sum_79: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_466, [0], True);  view_466 = None
    view_467: "f32[768]" = torch.ops.aten.reshape.default(sum_79, [768]);  sum_79 = None
    permute_247: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_246, [1, 0]);  permute_246 = None
    view_468: "f32[8, 1, 3072]" = torch.ops.aten.reshape.default(mm_28, [8, 1, 3072]);  mm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_615: "f32[8, 1, 3072]" = torch.ops.aten.mul.Tensor(add_445, 0.5);  add_445 = None
    mul_616: "f32[8, 1, 3072]" = torch.ops.aten.mul.Tensor(add_444, add_444)
    mul_617: "f32[8, 1, 3072]" = torch.ops.aten.mul.Tensor(mul_616, -0.5);  mul_616 = None
    exp_24: "f32[8, 1, 3072]" = torch.ops.aten.exp.default(mul_617);  mul_617 = None
    mul_618: "f32[8, 1, 3072]" = torch.ops.aten.mul.Tensor(exp_24, 0.3989422804014327);  exp_24 = None
    mul_619: "f32[8, 1, 3072]" = torch.ops.aten.mul.Tensor(add_444, mul_618);  add_444 = mul_618 = None
    add_451: "f32[8, 1, 3072]" = torch.ops.aten.add.Tensor(mul_615, mul_619);  mul_615 = mul_619 = None
    mul_620: "f32[8, 1, 3072]" = torch.ops.aten.mul.Tensor(view_468, add_451);  view_468 = add_451 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    sum_80: "f32[1, 1, 3072]" = torch.ops.aten.sum.dim_IntList(mul_620, [0, 1], True)
    view_469: "f32[3072]" = torch.ops.aten.reshape.default(sum_80, [3072]);  sum_80 = None
    view_470: "f32[8, 3072]" = torch.ops.aten.reshape.default(mul_620, [8, 3072]);  mul_620 = None
    permute_248: "f32[3072, 8]" = torch.ops.aten.permute.default(view_470, [1, 0])
    mm_30: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_248, view_460);  permute_248 = view_460 = None
    permute_249: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_30, [1, 0]);  mm_30 = None
    mm_31: "f32[8, 768]" = torch.ops.aten.mm.default(view_470, permute_250);  view_470 = permute_250 = None
    view_471: "f32[8, 1, 768]" = torch.ops.aten.reshape.default(mm_31, [8, 1, 768]);  mm_31 = None
    permute_251: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_249, [1, 0]);  permute_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:189, code: cls_token = x[:, 0:1]
    slice_scatter_3: "f32[8, 785, 768]" = torch.ops.aten.slice_scatter.default(full_default_2, view_471, 1, 0, 1);  view_471 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:189, code: cls_token = x[:, 0:1]
    add_452: "f32[8, 785, 768]" = torch.ops.aten.add.Tensor(add_449, slice_scatter_3);  add_449 = slice_scatter_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:185, code: x = self.norm2(x)
    mul_622: "f32[8, 785, 768]" = torch.ops.aten.mul.Tensor(add_452, primals_619);  primals_619 = None
    mul_623: "f32[8, 785, 768]" = torch.ops.aten.mul.Tensor(mul_622, 768)
    sum_81: "f32[8, 785, 1]" = torch.ops.aten.sum.dim_IntList(mul_622, [2], True)
    mul_624: "f32[8, 785, 768]" = torch.ops.aten.mul.Tensor(mul_622, mul_597);  mul_622 = None
    sum_82: "f32[8, 785, 1]" = torch.ops.aten.sum.dim_IntList(mul_624, [2], True);  mul_624 = None
    mul_625: "f32[8, 785, 768]" = torch.ops.aten.mul.Tensor(mul_597, sum_82);  sum_82 = None
    sub_132: "f32[8, 785, 768]" = torch.ops.aten.sub.Tensor(mul_623, sum_81);  mul_623 = sum_81 = None
    sub_133: "f32[8, 785, 768]" = torch.ops.aten.sub.Tensor(sub_132, mul_625);  sub_132 = mul_625 = None
    mul_626: "f32[8, 785, 768]" = torch.ops.aten.mul.Tensor(div_79, sub_133);  div_79 = sub_133 = None
    mul_627: "f32[8, 785, 768]" = torch.ops.aten.mul.Tensor(add_452, mul_597);  mul_597 = None
    sum_83: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_627, [0, 1]);  mul_627 = None
    sum_84: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_452, [0, 1]);  add_452 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:183, code: x = x + self.drop_path(self.gamma1 * x_attn)
    mul_628: "f32[8, 785, 768]" = torch.ops.aten.mul.Tensor(mul_626, primals_100);  primals_100 = None
    mul_629: "f32[8, 785, 768]" = torch.ops.aten.mul.Tensor(mul_626, cat_6);  cat_6 = None
    sum_85: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_629, [0, 1], True);  mul_629 = None
    view_472: "f32[768]" = torch.ops.aten.reshape.default(sum_85, [768]);  sum_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:182, code: x_attn = torch.cat([self.attn(x_norm1), x_norm1[:, 1:]], dim=1)
    slice_46: "f32[8, 1, 768]" = torch.ops.aten.slice.Tensor(mul_628, 1, 0, 1)
    slice_47: "f32[8, 784, 768]" = torch.ops.aten.slice.Tensor(mul_628, 1, 1, 785);  mul_628 = None
    slice_scatter_5: "f32[8, 785, 768]" = torch.ops.aten.slice_scatter.default(full_default_2, slice_47, 1, 1, 9223372036854775807);  slice_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:63, code: x_cls = self.proj(x_cls)
    view_473: "f32[8, 768]" = torch.ops.aten.reshape.default(slice_46, [8, 768]);  slice_46 = None
    mm_32: "f32[8, 768]" = torch.ops.aten.mm.default(view_473, permute_252);  permute_252 = None
    permute_253: "f32[768, 8]" = torch.ops.aten.permute.default(view_473, [1, 0])
    mm_33: "f32[768, 768]" = torch.ops.aten.mm.default(permute_253, view_458);  permute_253 = view_458 = None
    permute_254: "f32[768, 768]" = torch.ops.aten.permute.default(mm_33, [1, 0]);  mm_33 = None
    sum_86: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_473, [0], True);  view_473 = None
    view_474: "f32[768]" = torch.ops.aten.reshape.default(sum_86, [768]);  sum_86 = None
    permute_255: "f32[768, 768]" = torch.ops.aten.permute.default(permute_254, [1, 0]);  permute_254 = None
    view_475: "f32[8, 1, 768]" = torch.ops.aten.reshape.default(mm_32, [8, 1, 768]);  mm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:62, code: x_cls = x_cls.transpose(1, 2).reshape(B, 1, C)
    view_476: "f32[8, 1, 16, 48]" = torch.ops.aten.reshape.default(view_475, [8, 1, 16, 48]);  view_475 = None
    permute_256: "f32[8, 16, 1, 48]" = torch.ops.aten.permute.default(view_476, [0, 2, 1, 3]);  view_476 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:51, code: x_cls = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_backward = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_256, permute_230, permute_232, permute_234, None, alias_74, getitem_281, getitem_282, getitem_283, 0.0, [True, True, True, False]);  permute_256 = permute_230 = permute_232 = permute_234 = alias_74 = getitem_281 = getitem_282 = getitem_283 = None
    getitem_288: "f32[8, 16, 1, 48]" = _scaled_dot_product_efficient_attention_backward[0]
    getitem_289: "f32[8, 16, 785, 48]" = _scaled_dot_product_efficient_attention_backward[1]
    getitem_290: "f32[8, 16, 785, 48]" = _scaled_dot_product_efficient_attention_backward[2];  _scaled_dot_product_efficient_attention_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:48, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_257: "f32[8, 785, 16, 48]" = torch.ops.aten.permute.default(getitem_290, [0, 2, 1, 3]);  getitem_290 = None
    view_477: "f32[8, 785, 768]" = torch.ops.aten.reshape.default(permute_257, [8, 785, 768]);  permute_257 = None
    view_478: "f32[6280, 768]" = torch.ops.aten.reshape.default(view_477, [6280, 768]);  view_477 = None
    mm_34: "f32[6280, 768]" = torch.ops.aten.mm.default(view_478, permute_258);  permute_258 = None
    permute_259: "f32[768, 6280]" = torch.ops.aten.permute.default(view_478, [1, 0])
    mm_35: "f32[768, 768]" = torch.ops.aten.mm.default(permute_259, view_451);  permute_259 = None
    permute_260: "f32[768, 768]" = torch.ops.aten.permute.default(mm_35, [1, 0]);  mm_35 = None
    sum_87: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_478, [0], True);  view_478 = None
    view_479: "f32[768]" = torch.ops.aten.reshape.default(sum_87, [768]);  sum_87 = None
    permute_261: "f32[768, 768]" = torch.ops.aten.permute.default(permute_260, [1, 0]);  permute_260 = None
    view_480: "f32[8, 785, 768]" = torch.ops.aten.reshape.default(mm_34, [8, 785, 768]);  mm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:48, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    add_453: "f32[8, 785, 768]" = torch.ops.aten.add.Tensor(slice_scatter_5, view_480);  slice_scatter_5 = view_480 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:47, code: k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_262: "f32[8, 785, 16, 48]" = torch.ops.aten.permute.default(getitem_289, [0, 2, 1, 3]);  getitem_289 = None
    view_481: "f32[8, 785, 768]" = torch.ops.aten.reshape.default(permute_262, [8, 785, 768]);  permute_262 = None
    view_482: "f32[6280, 768]" = torch.ops.aten.reshape.default(view_481, [6280, 768]);  view_481 = None
    mm_36: "f32[6280, 768]" = torch.ops.aten.mm.default(view_482, permute_263);  permute_263 = None
    permute_264: "f32[768, 6280]" = torch.ops.aten.permute.default(view_482, [1, 0])
    mm_37: "f32[768, 768]" = torch.ops.aten.mm.default(permute_264, view_451);  permute_264 = view_451 = None
    permute_265: "f32[768, 768]" = torch.ops.aten.permute.default(mm_37, [1, 0]);  mm_37 = None
    sum_88: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_482, [0], True);  view_482 = None
    view_483: "f32[768]" = torch.ops.aten.reshape.default(sum_88, [768]);  sum_88 = None
    permute_266: "f32[768, 768]" = torch.ops.aten.permute.default(permute_265, [1, 0]);  permute_265 = None
    view_484: "f32[8, 785, 768]" = torch.ops.aten.reshape.default(mm_36, [8, 785, 768]);  mm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:47, code: k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    add_454: "f32[8, 785, 768]" = torch.ops.aten.add.Tensor(add_453, view_484);  add_453 = view_484 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:46, code: q = self.q(x[:, 0]).unsqueeze(1).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_267: "f32[8, 1, 16, 48]" = torch.ops.aten.permute.default(getitem_288, [0, 2, 1, 3]);  getitem_288 = None
    view_485: "f32[8, 1, 768]" = torch.ops.aten.reshape.default(permute_267, [8, 1, 768]);  permute_267 = None
    squeeze_81: "f32[8, 768]" = torch.ops.aten.squeeze.dim(view_485, 1);  view_485 = None
    mm_38: "f32[8, 768]" = torch.ops.aten.mm.default(squeeze_81, permute_268);  permute_268 = None
    permute_269: "f32[768, 8]" = torch.ops.aten.permute.default(squeeze_81, [1, 0])
    mm_39: "f32[768, 768]" = torch.ops.aten.mm.default(permute_269, select_1);  permute_269 = select_1 = None
    permute_270: "f32[768, 768]" = torch.ops.aten.permute.default(mm_39, [1, 0]);  mm_39 = None
    sum_89: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(squeeze_81, [0], True);  squeeze_81 = None
    view_486: "f32[768]" = torch.ops.aten.reshape.default(sum_89, [768]);  sum_89 = None
    permute_271: "f32[768, 768]" = torch.ops.aten.permute.default(permute_270, [1, 0]);  permute_270 = None
    select_scatter_1: "f32[8, 785, 768]" = torch.ops.aten.select_scatter.default(full_default_2, mm_38, 1, 0);  mm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:46, code: q = self.q(x[:, 0]).unsqueeze(1).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    add_455: "f32[8, 785, 768]" = torch.ops.aten.add.Tensor(add_454, select_scatter_1);  add_454 = select_scatter_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:181, code: x_norm1 = self.norm1(x)
    mul_631: "f32[8, 785, 768]" = torch.ops.aten.mul.Tensor(add_455, primals_609);  primals_609 = None
    mul_632: "f32[8, 785, 768]" = torch.ops.aten.mul.Tensor(mul_631, 768)
    sum_90: "f32[8, 785, 1]" = torch.ops.aten.sum.dim_IntList(mul_631, [2], True)
    mul_633: "f32[8, 785, 768]" = torch.ops.aten.mul.Tensor(mul_631, mul_594);  mul_631 = None
    sum_91: "f32[8, 785, 1]" = torch.ops.aten.sum.dim_IntList(mul_633, [2], True);  mul_633 = None
    mul_634: "f32[8, 785, 768]" = torch.ops.aten.mul.Tensor(mul_594, sum_91);  sum_91 = None
    sub_135: "f32[8, 785, 768]" = torch.ops.aten.sub.Tensor(mul_632, sum_90);  mul_632 = sum_90 = None
    sub_136: "f32[8, 785, 768]" = torch.ops.aten.sub.Tensor(sub_135, mul_634);  sub_135 = mul_634 = None
    mul_635: "f32[8, 785, 768]" = torch.ops.aten.mul.Tensor(div_80, sub_136);  div_80 = sub_136 = None
    mul_636: "f32[8, 785, 768]" = torch.ops.aten.mul.Tensor(add_455, mul_594);  mul_594 = None
    sum_92: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_636, [0, 1]);  mul_636 = None
    sum_93: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_455, [0, 1]);  add_455 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:181, code: x_norm1 = self.norm1(x)
    add_456: "f32[8, 785, 768]" = torch.ops.aten.add.Tensor(mul_626, mul_635);  mul_626 = mul_635 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:191, code: x = torch.cat([cls_token, x[:, 1:]], dim=1)
    slice_48: "f32[8, 1, 768]" = torch.ops.aten.slice.Tensor(add_456, 1, 0, 1)
    slice_49: "f32[8, 784, 768]" = torch.ops.aten.slice.Tensor(add_456, 1, 1, 785)
    slice_scatter_8: "f32[8, 785, 768]" = torch.ops.aten.slice_scatter.default(full_default_2, slice_49, 1, 1, 9223372036854775807);  slice_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:191, code: x = torch.cat([cls_token, x[:, 1:]], dim=1)
    add_457: "f32[8, 785, 768]" = torch.ops.aten.add.Tensor(add_456, slice_scatter_8);  add_456 = slice_scatter_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:190, code: cls_token = self.gamma2 * self.mlp(cls_token)
    mul_637: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(slice_48, primals_99);  primals_99 = None
    mul_638: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(slice_48, view_449);  slice_48 = view_449 = None
    sum_94: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_638, [0, 1], True);  mul_638 = None
    view_487: "f32[768]" = torch.ops.aten.reshape.default(sum_94, [768]);  sum_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_488: "f32[8, 768]" = torch.ops.aten.reshape.default(mul_637, [8, 768]);  mul_637 = None
    mm_40: "f32[8, 3072]" = torch.ops.aten.mm.default(view_488, permute_272);  permute_272 = None
    permute_273: "f32[768, 8]" = torch.ops.aten.permute.default(view_488, [1, 0])
    mm_41: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_273, view_448);  permute_273 = view_448 = None
    permute_274: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_41, [1, 0]);  mm_41 = None
    sum_95: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_488, [0], True);  view_488 = None
    view_489: "f32[768]" = torch.ops.aten.reshape.default(sum_95, [768]);  sum_95 = None
    permute_275: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_274, [1, 0]);  permute_274 = None
    view_490: "f32[8, 1, 3072]" = torch.ops.aten.reshape.default(mm_40, [8, 1, 3072]);  mm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_640: "f32[8, 1, 3072]" = torch.ops.aten.mul.Tensor(add_437, 0.5);  add_437 = None
    mul_641: "f32[8, 1, 3072]" = torch.ops.aten.mul.Tensor(add_436, add_436)
    mul_642: "f32[8, 1, 3072]" = torch.ops.aten.mul.Tensor(mul_641, -0.5);  mul_641 = None
    exp_25: "f32[8, 1, 3072]" = torch.ops.aten.exp.default(mul_642);  mul_642 = None
    mul_643: "f32[8, 1, 3072]" = torch.ops.aten.mul.Tensor(exp_25, 0.3989422804014327);  exp_25 = None
    mul_644: "f32[8, 1, 3072]" = torch.ops.aten.mul.Tensor(add_436, mul_643);  add_436 = mul_643 = None
    add_459: "f32[8, 1, 3072]" = torch.ops.aten.add.Tensor(mul_640, mul_644);  mul_640 = mul_644 = None
    mul_645: "f32[8, 1, 3072]" = torch.ops.aten.mul.Tensor(view_490, add_459);  view_490 = add_459 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    sum_96: "f32[1, 1, 3072]" = torch.ops.aten.sum.dim_IntList(mul_645, [0, 1], True)
    view_491: "f32[3072]" = torch.ops.aten.reshape.default(sum_96, [3072]);  sum_96 = None
    view_492: "f32[8, 3072]" = torch.ops.aten.reshape.default(mul_645, [8, 3072]);  mul_645 = None
    permute_276: "f32[3072, 8]" = torch.ops.aten.permute.default(view_492, [1, 0])
    mm_42: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_276, view_446);  permute_276 = view_446 = None
    permute_277: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_42, [1, 0]);  mm_42 = None
    mm_43: "f32[8, 768]" = torch.ops.aten.mm.default(view_492, permute_278);  view_492 = permute_278 = None
    view_493: "f32[8, 1, 768]" = torch.ops.aten.reshape.default(mm_43, [8, 1, 768]);  mm_43 = None
    permute_279: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_277, [1, 0]);  permute_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:189, code: cls_token = x[:, 0:1]
    slice_scatter_10: "f32[8, 785, 768]" = torch.ops.aten.slice_scatter.default(full_default_2, view_493, 1, 0, 1);  view_493 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:189, code: cls_token = x[:, 0:1]
    add_460: "f32[8, 785, 768]" = torch.ops.aten.add.Tensor(add_457, slice_scatter_10);  add_457 = slice_scatter_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:185, code: x = self.norm2(x)
    mul_647: "f32[8, 785, 768]" = torch.ops.aten.mul.Tensor(add_460, primals_603);  primals_603 = None
    mul_648: "f32[8, 785, 768]" = torch.ops.aten.mul.Tensor(mul_647, 768)
    sum_97: "f32[8, 785, 1]" = torch.ops.aten.sum.dim_IntList(mul_647, [2], True)
    mul_649: "f32[8, 785, 768]" = torch.ops.aten.mul.Tensor(mul_647, mul_588);  mul_647 = None
    sum_98: "f32[8, 785, 1]" = torch.ops.aten.sum.dim_IntList(mul_649, [2], True);  mul_649 = None
    mul_650: "f32[8, 785, 768]" = torch.ops.aten.mul.Tensor(mul_588, sum_98);  sum_98 = None
    sub_138: "f32[8, 785, 768]" = torch.ops.aten.sub.Tensor(mul_648, sum_97);  mul_648 = sum_97 = None
    sub_139: "f32[8, 785, 768]" = torch.ops.aten.sub.Tensor(sub_138, mul_650);  sub_138 = mul_650 = None
    mul_651: "f32[8, 785, 768]" = torch.ops.aten.mul.Tensor(div_81, sub_139);  div_81 = sub_139 = None
    mul_652: "f32[8, 785, 768]" = torch.ops.aten.mul.Tensor(add_460, mul_588);  mul_588 = None
    sum_99: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_652, [0, 1]);  mul_652 = None
    sum_100: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_460, [0, 1]);  add_460 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:183, code: x = x + self.drop_path(self.gamma1 * x_attn)
    mul_653: "f32[8, 785, 768]" = torch.ops.aten.mul.Tensor(mul_651, primals_98);  primals_98 = None
    mul_654: "f32[8, 785, 768]" = torch.ops.aten.mul.Tensor(mul_651, cat_4);  cat_4 = None
    sum_101: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_654, [0, 1], True);  mul_654 = None
    view_494: "f32[768]" = torch.ops.aten.reshape.default(sum_101, [768]);  sum_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:182, code: x_attn = torch.cat([self.attn(x_norm1), x_norm1[:, 1:]], dim=1)
    slice_50: "f32[8, 1, 768]" = torch.ops.aten.slice.Tensor(mul_653, 1, 0, 1)
    slice_51: "f32[8, 784, 768]" = torch.ops.aten.slice.Tensor(mul_653, 1, 1, 785);  mul_653 = None
    slice_scatter_12: "f32[8, 785, 768]" = torch.ops.aten.slice_scatter.default(full_default_2, slice_51, 1, 1, 9223372036854775807);  slice_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:63, code: x_cls = self.proj(x_cls)
    view_495: "f32[8, 768]" = torch.ops.aten.reshape.default(slice_50, [8, 768]);  slice_50 = None
    mm_44: "f32[8, 768]" = torch.ops.aten.mm.default(view_495, permute_280);  permute_280 = None
    permute_281: "f32[768, 8]" = torch.ops.aten.permute.default(view_495, [1, 0])
    mm_45: "f32[768, 768]" = torch.ops.aten.mm.default(permute_281, view_444);  permute_281 = view_444 = None
    permute_282: "f32[768, 768]" = torch.ops.aten.permute.default(mm_45, [1, 0]);  mm_45 = None
    sum_102: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_495, [0], True);  view_495 = None
    view_496: "f32[768]" = torch.ops.aten.reshape.default(sum_102, [768]);  sum_102 = None
    permute_283: "f32[768, 768]" = torch.ops.aten.permute.default(permute_282, [1, 0]);  permute_282 = None
    view_497: "f32[8, 1, 768]" = torch.ops.aten.reshape.default(mm_44, [8, 1, 768]);  mm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:62, code: x_cls = x_cls.transpose(1, 2).reshape(B, 1, C)
    view_498: "f32[8, 1, 16, 48]" = torch.ops.aten.reshape.default(view_497, [8, 1, 16, 48]);  view_497 = None
    permute_284: "f32[8, 16, 1, 48]" = torch.ops.aten.permute.default(view_498, [0, 2, 1, 3]);  view_498 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:51, code: x_cls = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_backward_1 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_284, permute_220, permute_222, permute_224, None, alias_75, getitem_273, getitem_274, getitem_275, 0.0, [True, True, True, False]);  permute_284 = permute_220 = permute_222 = permute_224 = alias_75 = getitem_273 = getitem_274 = getitem_275 = None
    getitem_292: "f32[8, 16, 1, 48]" = _scaled_dot_product_efficient_attention_backward_1[0]
    getitem_293: "f32[8, 16, 785, 48]" = _scaled_dot_product_efficient_attention_backward_1[1]
    getitem_294: "f32[8, 16, 785, 48]" = _scaled_dot_product_efficient_attention_backward_1[2];  _scaled_dot_product_efficient_attention_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:48, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_285: "f32[8, 785, 16, 48]" = torch.ops.aten.permute.default(getitem_294, [0, 2, 1, 3]);  getitem_294 = None
    view_499: "f32[8, 785, 768]" = torch.ops.aten.reshape.default(permute_285, [8, 785, 768]);  permute_285 = None
    view_500: "f32[6280, 768]" = torch.ops.aten.reshape.default(view_499, [6280, 768]);  view_499 = None
    mm_46: "f32[6280, 768]" = torch.ops.aten.mm.default(view_500, permute_286);  permute_286 = None
    permute_287: "f32[768, 6280]" = torch.ops.aten.permute.default(view_500, [1, 0])
    mm_47: "f32[768, 768]" = torch.ops.aten.mm.default(permute_287, view_437);  permute_287 = None
    permute_288: "f32[768, 768]" = torch.ops.aten.permute.default(mm_47, [1, 0]);  mm_47 = None
    sum_103: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_500, [0], True);  view_500 = None
    view_501: "f32[768]" = torch.ops.aten.reshape.default(sum_103, [768]);  sum_103 = None
    permute_289: "f32[768, 768]" = torch.ops.aten.permute.default(permute_288, [1, 0]);  permute_288 = None
    view_502: "f32[8, 785, 768]" = torch.ops.aten.reshape.default(mm_46, [8, 785, 768]);  mm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:48, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    add_461: "f32[8, 785, 768]" = torch.ops.aten.add.Tensor(slice_scatter_12, view_502);  slice_scatter_12 = view_502 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:47, code: k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_290: "f32[8, 785, 16, 48]" = torch.ops.aten.permute.default(getitem_293, [0, 2, 1, 3]);  getitem_293 = None
    view_503: "f32[8, 785, 768]" = torch.ops.aten.reshape.default(permute_290, [8, 785, 768]);  permute_290 = None
    view_504: "f32[6280, 768]" = torch.ops.aten.reshape.default(view_503, [6280, 768]);  view_503 = None
    mm_48: "f32[6280, 768]" = torch.ops.aten.mm.default(view_504, permute_291);  permute_291 = None
    permute_292: "f32[768, 6280]" = torch.ops.aten.permute.default(view_504, [1, 0])
    mm_49: "f32[768, 768]" = torch.ops.aten.mm.default(permute_292, view_437);  permute_292 = view_437 = None
    permute_293: "f32[768, 768]" = torch.ops.aten.permute.default(mm_49, [1, 0]);  mm_49 = None
    sum_104: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_504, [0], True);  view_504 = None
    view_505: "f32[768]" = torch.ops.aten.reshape.default(sum_104, [768]);  sum_104 = None
    permute_294: "f32[768, 768]" = torch.ops.aten.permute.default(permute_293, [1, 0]);  permute_293 = None
    view_506: "f32[8, 785, 768]" = torch.ops.aten.reshape.default(mm_48, [8, 785, 768]);  mm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:47, code: k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    add_462: "f32[8, 785, 768]" = torch.ops.aten.add.Tensor(add_461, view_506);  add_461 = view_506 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:46, code: q = self.q(x[:, 0]).unsqueeze(1).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_295: "f32[8, 1, 16, 48]" = torch.ops.aten.permute.default(getitem_292, [0, 2, 1, 3]);  getitem_292 = None
    view_507: "f32[8, 1, 768]" = torch.ops.aten.reshape.default(permute_295, [8, 1, 768]);  permute_295 = None
    squeeze_82: "f32[8, 768]" = torch.ops.aten.squeeze.dim(view_507, 1);  view_507 = None
    mm_50: "f32[8, 768]" = torch.ops.aten.mm.default(squeeze_82, permute_296);  permute_296 = None
    permute_297: "f32[768, 8]" = torch.ops.aten.permute.default(squeeze_82, [1, 0])
    mm_51: "f32[768, 768]" = torch.ops.aten.mm.default(permute_297, select);  permute_297 = select = None
    permute_298: "f32[768, 768]" = torch.ops.aten.permute.default(mm_51, [1, 0]);  mm_51 = None
    sum_105: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(squeeze_82, [0], True);  squeeze_82 = None
    view_508: "f32[768]" = torch.ops.aten.reshape.default(sum_105, [768]);  sum_105 = None
    permute_299: "f32[768, 768]" = torch.ops.aten.permute.default(permute_298, [1, 0]);  permute_298 = None
    select_scatter_2: "f32[8, 785, 768]" = torch.ops.aten.select_scatter.default(full_default_2, mm_50, 1, 0);  full_default_2 = mm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:46, code: q = self.q(x[:, 0]).unsqueeze(1).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    add_463: "f32[8, 785, 768]" = torch.ops.aten.add.Tensor(add_462, select_scatter_2);  add_462 = select_scatter_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:181, code: x_norm1 = self.norm1(x)
    mul_656: "f32[8, 785, 768]" = torch.ops.aten.mul.Tensor(add_463, primals_593);  primals_593 = None
    mul_657: "f32[8, 785, 768]" = torch.ops.aten.mul.Tensor(mul_656, 768)
    sum_106: "f32[8, 785, 1]" = torch.ops.aten.sum.dim_IntList(mul_656, [2], True)
    mul_658: "f32[8, 785, 768]" = torch.ops.aten.mul.Tensor(mul_656, mul_585);  mul_656 = None
    sum_107: "f32[8, 785, 1]" = torch.ops.aten.sum.dim_IntList(mul_658, [2], True);  mul_658 = None
    mul_659: "f32[8, 785, 768]" = torch.ops.aten.mul.Tensor(mul_585, sum_107);  sum_107 = None
    sub_141: "f32[8, 785, 768]" = torch.ops.aten.sub.Tensor(mul_657, sum_106);  mul_657 = sum_106 = None
    sub_142: "f32[8, 785, 768]" = torch.ops.aten.sub.Tensor(sub_141, mul_659);  sub_141 = mul_659 = None
    div_82: "f32[8, 785, 1]" = torch.ops.aten.div.Tensor(rsqrt_99, 768);  rsqrt_99 = None
    mul_660: "f32[8, 785, 768]" = torch.ops.aten.mul.Tensor(div_82, sub_142);  div_82 = sub_142 = None
    mul_661: "f32[8, 785, 768]" = torch.ops.aten.mul.Tensor(add_463, mul_585);  mul_585 = None
    sum_108: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_661, [0, 1]);  mul_661 = None
    sum_109: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_463, [0, 1]);  add_463 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:181, code: x_norm1 = self.norm1(x)
    add_464: "f32[8, 785, 768]" = torch.ops.aten.add.Tensor(mul_651, mul_660);  mul_651 = mul_660 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:447, code: x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1)
    slice_52: "f32[8, 1, 768]" = torch.ops.aten.slice.Tensor(add_464, 1, 0, 1)
    slice_53: "f32[8, 784, 768]" = torch.ops.aten.slice.Tensor(add_464, 1, 1, 785);  add_464 = None
    sum_110: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(slice_52, [0], True);  slice_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_662: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(slice_53, primals_96);  primals_96 = None
    mul_663: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(slice_53, view_435);  view_435 = None
    sum_111: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_663, [0, 1], True);  mul_663 = None
    view_509: "f32[768]" = torch.ops.aten.reshape.default(sum_111, [768]);  sum_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_510: "f32[6272, 768]" = torch.ops.aten.reshape.default(mul_662, [6272, 768]);  mul_662 = None
    mm_52: "f32[6272, 3072]" = torch.ops.aten.mm.default(view_510, permute_300);  permute_300 = None
    permute_301: "f32[768, 6272]" = torch.ops.aten.permute.default(view_510, [1, 0])
    mm_53: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_301, view_434);  permute_301 = view_434 = None
    permute_302: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_53, [1, 0]);  mm_53 = None
    sum_112: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_510, [0], True);  view_510 = None
    view_511: "f32[768]" = torch.ops.aten.reshape.default(sum_112, [768]);  sum_112 = None
    permute_303: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_302, [1, 0]);  permute_302 = None
    view_512: "f32[8, 784, 3072]" = torch.ops.aten.reshape.default(mm_52, [8, 784, 3072]);  mm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_665: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(add_429, 0.5);  add_429 = None
    mul_666: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_433, view_433)
    mul_667: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(mul_666, -0.5);  mul_666 = None
    exp_26: "f32[8, 784, 3072]" = torch.ops.aten.exp.default(mul_667);  mul_667 = None
    mul_668: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(exp_26, 0.3989422804014327);  exp_26 = None
    mul_669: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_433, mul_668);  view_433 = mul_668 = None
    add_466: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(mul_665, mul_669);  mul_665 = mul_669 = None
    mul_670: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_512, add_466);  view_512 = add_466 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_513: "f32[6272, 3072]" = torch.ops.aten.reshape.default(mul_670, [6272, 3072]);  mul_670 = None
    mm_54: "f32[6272, 768]" = torch.ops.aten.mm.default(view_513, permute_304);  permute_304 = None
    permute_305: "f32[3072, 6272]" = torch.ops.aten.permute.default(view_513, [1, 0])
    mm_55: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_305, view_432);  permute_305 = view_432 = None
    permute_306: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_55, [1, 0]);  mm_55 = None
    sum_113: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_513, [0], True);  view_513 = None
    view_514: "f32[3072]" = torch.ops.aten.reshape.default(sum_113, [3072]);  sum_113 = None
    permute_307: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_306, [1, 0]);  permute_306 = None
    view_515: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_54, [8, 784, 768]);  mm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_672: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_515, primals_587);  primals_587 = None
    mul_673: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_672, 768)
    sum_114: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_672, [2], True)
    mul_674: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_672, mul_579);  mul_672 = None
    sum_115: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_674, [2], True);  mul_674 = None
    mul_675: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_579, sum_115);  sum_115 = None
    sub_144: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(mul_673, sum_114);  mul_673 = sum_114 = None
    sub_145: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(sub_144, mul_675);  sub_144 = mul_675 = None
    mul_676: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(div_83, sub_145);  div_83 = sub_145 = None
    mul_677: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_515, mul_579);  mul_579 = None
    sum_116: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_677, [0, 1]);  mul_677 = None
    sum_117: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_515, [0, 1]);  view_515 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    add_467: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(slice_53, mul_676);  slice_53 = mul_676 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_678: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_467, primals_95);  primals_95 = None
    mul_679: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_467, permute_216);  permute_216 = None
    sum_118: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_679, [0, 1], True);  mul_679 = None
    view_516: "f32[768]" = torch.ops.aten.reshape.default(sum_118, [768]);  sum_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    permute_308: "f32[8, 768, 784]" = torch.ops.aten.permute.default(mul_678, [0, 2, 1]);  mul_678 = None
    view_517: "f32[8, 768, 28, 28]" = torch.ops.aten.reshape.default(permute_308, [8, 768, 28, 28]);  permute_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    sum_119: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_517, [0, 2, 3])
    convolution_backward = torch.ops.aten.convolution_backward.default(view_517, add_425, primals_585, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False]);  view_517 = add_425 = primals_585 = None
    getitem_296: "f32[8, 768, 28, 28]" = convolution_backward[0]
    getitem_297: "f32[768, 1, 3, 3]" = convolution_backward[1];  convolution_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    sum_120: "f32[768]" = torch.ops.aten.sum.dim_IntList(getitem_296, [0, 2, 3])
    sub_146: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(mul_570, unsqueeze_119);  mul_570 = unsqueeze_119 = None
    mul_680: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_296, sub_146)
    sum_121: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_680, [0, 2, 3]);  mul_680 = None
    mul_681: "f32[768]" = torch.ops.aten.mul.Tensor(sum_120, 0.00015943877551020407)
    unsqueeze_120: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_681, 0);  mul_681 = None
    unsqueeze_121: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_120, 2);  unsqueeze_120 = None
    unsqueeze_122: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_121, 3);  unsqueeze_121 = None
    mul_682: "f32[768]" = torch.ops.aten.mul.Tensor(sum_121, 0.00015943877551020407)
    mul_683: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_79, squeeze_79)
    mul_684: "f32[768]" = torch.ops.aten.mul.Tensor(mul_682, mul_683);  mul_682 = mul_683 = None
    unsqueeze_123: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_684, 0);  mul_684 = None
    unsqueeze_124: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_123, 2);  unsqueeze_123 = None
    unsqueeze_125: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_124, 3);  unsqueeze_124 = None
    mul_685: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_79, primals_583);  primals_583 = None
    unsqueeze_126: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_685, 0);  mul_685 = None
    unsqueeze_127: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_126, 2);  unsqueeze_126 = None
    unsqueeze_128: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_127, 3);  unsqueeze_127 = None
    mul_686: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_146, unsqueeze_125);  sub_146 = unsqueeze_125 = None
    sub_148: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(getitem_296, mul_686);  getitem_296 = mul_686 = None
    sub_149: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(sub_148, unsqueeze_122);  sub_148 = unsqueeze_122 = None
    mul_687: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_149, unsqueeze_128);  sub_149 = unsqueeze_128 = None
    mul_688: "f32[768]" = torch.ops.aten.mul.Tensor(sum_121, squeeze_79);  sum_121 = squeeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_690: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(add_420, 0.5);  add_420 = None
    mul_691: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_50, convolution_50)
    mul_692: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_691, -0.5);  mul_691 = None
    exp_27: "f32[8, 768, 28, 28]" = torch.ops.aten.exp.default(mul_692);  mul_692 = None
    mul_693: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(exp_27, 0.3989422804014327);  exp_27 = None
    mul_694: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_50, mul_693);  convolution_50 = mul_693 = None
    add_469: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(mul_690, mul_694);  mul_690 = mul_694 = None
    mul_695: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_687, add_469);  mul_687 = add_469 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    sum_122: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_695, [0, 2, 3])
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(mul_695, view_430, primals_581, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False]);  mul_695 = view_430 = primals_581 = None
    getitem_299: "f32[8, 768, 28, 28]" = convolution_backward_1[0]
    getitem_300: "f32[768, 1, 3, 3]" = convolution_backward_1[1];  convolution_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    view_518: "f32[8, 768, 784]" = torch.ops.aten.reshape.default(getitem_299, [8, 768, 784]);  getitem_299 = None
    permute_309: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_518, [0, 2, 1]);  view_518 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    clone_273: "f32[8, 784, 768]" = torch.ops.aten.clone.default(permute_309, memory_format = torch.contiguous_format);  permute_309 = None
    mul_697: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(clone_273, primals_579);  primals_579 = None
    mul_698: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_697, 768)
    sum_123: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_697, [2], True)
    mul_699: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_697, mul_566);  mul_697 = None
    sum_124: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_699, [2], True);  mul_699 = None
    mul_700: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_566, sum_124);  sum_124 = None
    sub_151: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(mul_698, sum_123);  mul_698 = sum_123 = None
    sub_152: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(sub_151, mul_700);  sub_151 = mul_700 = None
    mul_701: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(div_84, sub_152);  div_84 = sub_152 = None
    mul_702: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(clone_273, mul_566);  mul_566 = None
    sum_125: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_702, [0, 1]);  mul_702 = None
    sum_126: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_273, [0, 1]);  clone_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    add_470: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_467, mul_701);  add_467 = mul_701 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_703: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_470, primals_93);  primals_93 = None
    mul_704: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_470, add_416);  add_416 = None
    sum_127: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_704, [0, 1], True);  mul_704 = None
    view_519: "f32[768]" = torch.ops.aten.reshape.default(sum_127, [768]);  sum_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    sum_128: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_703, [0, 1], True)
    view_520: "f32[768]" = torch.ops.aten.reshape.default(sum_128, [768]);  sum_128 = None
    view_521: "f32[6272, 768]" = torch.ops.aten.reshape.default(mul_703, [6272, 768]);  mul_703 = None
    permute_310: "f32[768, 6272]" = torch.ops.aten.permute.default(view_521, [1, 0])
    mm_56: "f32[768, 768]" = torch.ops.aten.mm.default(permute_310, view_428);  permute_310 = view_428 = None
    permute_311: "f32[768, 768]" = torch.ops.aten.permute.default(mm_56, [1, 0]);  mm_56 = None
    mm_57: "f32[6272, 768]" = torch.ops.aten.mm.default(view_521, permute_312);  view_521 = permute_312 = None
    view_522: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_57, [8, 784, 768]);  mm_57 = None
    permute_313: "f32[768, 768]" = torch.ops.aten.permute.default(permute_311, [1, 0]);  permute_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    view_523: "f32[8, 784, 16, 48]" = torch.ops.aten.reshape.default(view_522, [8, 784, 16, 48]);  view_522 = None
    permute_314: "f32[8, 16, 48, 784]" = torch.ops.aten.permute.default(view_523, [0, 2, 3, 1]);  view_523 = None
    clone_275: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(permute_314, memory_format = torch.contiguous_format);  permute_314 = None
    view_524: "f32[128, 48, 784]" = torch.ops.aten.reshape.default(clone_275, [128, 48, 784]);  clone_275 = None
    bmm_48: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(permute_315, view_524);  permute_315 = None
    bmm_49: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(view_524, permute_316);  view_524 = permute_316 = None
    view_525: "f32[8, 16, 48, 784]" = torch.ops.aten.reshape.default(bmm_48, [8, 16, 48, 784]);  bmm_48 = None
    view_526: "f32[8, 16, 48, 48]" = torch.ops.aten.reshape.default(bmm_49, [8, 16, 48, 48]);  bmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    mul_705: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_526, alias_76);  view_526 = None
    sum_129: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_705, [-1], True)
    mul_706: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(alias_76, sum_129);  alias_76 = sum_129 = None
    sub_153: "f32[8, 16, 48, 48]" = torch.ops.aten.sub.Tensor(mul_705, mul_706);  mul_705 = mul_706 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    mul_707: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(sub_153, view_423);  view_423 = None
    mul_708: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(sub_153, primals_94);  sub_153 = primals_94 = None
    sum_130: "f32[1, 16, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_707, [0, 2, 3], True);  mul_707 = None
    view_527: "f32[16, 1, 1]" = torch.ops.aten.reshape.default(sum_130, [16, 1, 1]);  sum_130 = None
    view_528: "f32[128, 48, 48]" = torch.ops.aten.reshape.default(mul_708, [128, 48, 48]);  mul_708 = None
    bmm_50: "f32[128, 784, 48]" = torch.ops.aten.bmm.default(permute_317, view_528);  permute_317 = None
    bmm_51: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_528, permute_318);  view_528 = permute_318 = None
    view_529: "f32[8, 16, 784, 48]" = torch.ops.aten.reshape.default(bmm_50, [8, 16, 784, 48]);  bmm_50 = None
    view_530: "f32[8, 16, 48, 784]" = torch.ops.aten.reshape.default(bmm_51, [8, 16, 48, 784]);  bmm_51 = None
    permute_319: "f32[8, 16, 48, 784]" = torch.ops.aten.permute.default(view_529, [0, 1, 3, 2]);  view_529 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    div_86: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_76, expand_139);  div_76 = None
    neg: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(permute_319)
    mul_709: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg, div_86);  neg = div_86 = None
    div_87: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(permute_319, expand_139);  permute_319 = expand_139 = None
    sum_131: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_709, [3], True);  mul_709 = None
    full_default_20: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    ge: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(pow_97, 1e-12)
    where: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge, sum_131, full_default_20);  ge = sum_131 = None
    div_88: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_262, pow_97);  getitem_262 = None
    eq: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(pow_97, 0);  pow_97 = None
    where_1: "f32[8, 16, 48, 784]" = torch.ops.aten.where.self(eq, full_default_20, div_88);  eq = div_88 = None
    clone_276: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(where_1, memory_format = torch.contiguous_format);  where_1 = None
    mul_710: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where, clone_276);  where = clone_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    add_471: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_87, mul_710);  div_87 = mul_710 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    div_90: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_75, expand_138);  div_75 = None
    neg_1: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(view_530)
    mul_711: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_1, div_90);  neg_1 = div_90 = None
    div_91: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(view_530, expand_138);  view_530 = expand_138 = None
    sum_132: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_711, [3], True);  mul_711 = None
    ge_1: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(pow_95, 1e-12)
    where_2: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_1, sum_132, full_default_20);  ge_1 = sum_132 = None
    div_92: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_261, pow_95);  getitem_261 = None
    eq_1: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(pow_95, 0);  pow_95 = None
    where_3: "f32[8, 16, 48, 784]" = torch.ops.aten.where.self(eq_1, full_default_20, div_92);  eq_1 = div_92 = None
    clone_277: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(where_3, memory_format = torch.contiguous_format);  where_3 = None
    mul_712: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_2, clone_277);  where_2 = clone_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    add_472: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_91, mul_712);  div_91 = mul_712 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    cat_8: "f32[24, 16, 48, 784]" = torch.ops.aten.cat.default([add_472, add_471, view_525]);  add_472 = add_471 = view_525 = None
    view_531: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.reshape.default(cat_8, [3, 8, 16, 48, 784]);  cat_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    permute_320: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.permute.default(view_531, [1, 4, 0, 2, 3]);  view_531 = None
    clone_278: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.clone.default(permute_320, memory_format = torch.contiguous_format);  permute_320 = None
    view_532: "f32[8, 784, 2304]" = torch.ops.aten.reshape.default(clone_278, [8, 784, 2304]);  clone_278 = None
    view_533: "f32[6272, 2304]" = torch.ops.aten.reshape.default(view_532, [6272, 2304]);  view_532 = None
    mm_58: "f32[6272, 768]" = torch.ops.aten.mm.default(view_533, permute_321);  permute_321 = None
    permute_322: "f32[2304, 6272]" = torch.ops.aten.permute.default(view_533, [1, 0])
    mm_59: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_322, view_418);  permute_322 = view_418 = None
    permute_323: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_59, [1, 0]);  mm_59 = None
    sum_133: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_533, [0], True);  view_533 = None
    view_534: "f32[2304]" = torch.ops.aten.reshape.default(sum_133, [2304]);  sum_133 = None
    permute_324: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_323, [1, 0]);  permute_323 = None
    view_535: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_58, [8, 784, 768]);  mm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_714: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_535, primals_573);  primals_573 = None
    mul_715: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_714, 768)
    sum_134: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_714, [2], True)
    mul_716: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_714, mul_562);  mul_714 = None
    sum_135: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_716, [2], True);  mul_716 = None
    mul_717: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_562, sum_135);  sum_135 = None
    sub_155: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(mul_715, sum_134);  mul_715 = sum_134 = None
    sub_156: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(sub_155, mul_717);  sub_155 = mul_717 = None
    mul_718: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(div_93, sub_156);  div_93 = sub_156 = None
    mul_719: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_535, mul_562);  mul_562 = None
    sum_136: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_719, [0, 1]);  mul_719 = None
    sum_137: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_535, [0, 1]);  view_535 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    add_473: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_470, mul_718);  add_470 = mul_718 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_720: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_473, primals_92);  primals_92 = None
    mul_721: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_473, view_417);  view_417 = None
    sum_138: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_721, [0, 1], True);  mul_721 = None
    view_536: "f32[768]" = torch.ops.aten.reshape.default(sum_138, [768]);  sum_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_537: "f32[6272, 768]" = torch.ops.aten.reshape.default(mul_720, [6272, 768]);  mul_720 = None
    mm_60: "f32[6272, 3072]" = torch.ops.aten.mm.default(view_537, permute_325);  permute_325 = None
    permute_326: "f32[768, 6272]" = torch.ops.aten.permute.default(view_537, [1, 0])
    mm_61: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_326, view_416);  permute_326 = view_416 = None
    permute_327: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_61, [1, 0]);  mm_61 = None
    sum_139: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_537, [0], True);  view_537 = None
    view_538: "f32[768]" = torch.ops.aten.reshape.default(sum_139, [768]);  sum_139 = None
    permute_328: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_327, [1, 0]);  permute_327 = None
    view_539: "f32[8, 784, 3072]" = torch.ops.aten.reshape.default(mm_60, [8, 784, 3072]);  mm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_723: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(add_412, 0.5);  add_412 = None
    mul_724: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_415, view_415)
    mul_725: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(mul_724, -0.5);  mul_724 = None
    exp_28: "f32[8, 784, 3072]" = torch.ops.aten.exp.default(mul_725);  mul_725 = None
    mul_726: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(exp_28, 0.3989422804014327);  exp_28 = None
    mul_727: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_415, mul_726);  view_415 = mul_726 = None
    add_475: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(mul_723, mul_727);  mul_723 = mul_727 = None
    mul_728: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_539, add_475);  view_539 = add_475 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_540: "f32[6272, 3072]" = torch.ops.aten.reshape.default(mul_728, [6272, 3072]);  mul_728 = None
    mm_62: "f32[6272, 768]" = torch.ops.aten.mm.default(view_540, permute_329);  permute_329 = None
    permute_330: "f32[3072, 6272]" = torch.ops.aten.permute.default(view_540, [1, 0])
    mm_63: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_330, view_414);  permute_330 = view_414 = None
    permute_331: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_63, [1, 0]);  mm_63 = None
    sum_140: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_540, [0], True);  view_540 = None
    view_541: "f32[3072]" = torch.ops.aten.reshape.default(sum_140, [3072]);  sum_140 = None
    permute_332: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_331, [1, 0]);  permute_331 = None
    view_542: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_62, [8, 784, 768]);  mm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_730: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_542, primals_567);  primals_567 = None
    mul_731: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_730, 768)
    sum_141: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_730, [2], True)
    mul_732: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_730, mul_556);  mul_730 = None
    sum_142: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_732, [2], True);  mul_732 = None
    mul_733: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_556, sum_142);  sum_142 = None
    sub_158: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(mul_731, sum_141);  mul_731 = sum_141 = None
    sub_159: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(sub_158, mul_733);  sub_158 = mul_733 = None
    mul_734: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(div_94, sub_159);  div_94 = sub_159 = None
    mul_735: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_542, mul_556);  mul_556 = None
    sum_143: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_735, [0, 1]);  mul_735 = None
    sum_144: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_542, [0, 1]);  view_542 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    add_476: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_473, mul_734);  add_473 = mul_734 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_736: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_476, primals_91);  primals_91 = None
    mul_737: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_476, permute_207);  permute_207 = None
    sum_145: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_737, [0, 1], True);  mul_737 = None
    view_543: "f32[768]" = torch.ops.aten.reshape.default(sum_145, [768]);  sum_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    permute_333: "f32[8, 768, 784]" = torch.ops.aten.permute.default(mul_736, [0, 2, 1]);  mul_736 = None
    view_544: "f32[8, 768, 28, 28]" = torch.ops.aten.reshape.default(permute_333, [8, 768, 28, 28]);  permute_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    sum_146: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_544, [0, 2, 3])
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(view_544, add_408, primals_565, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False]);  view_544 = add_408 = primals_565 = None
    getitem_302: "f32[8, 768, 28, 28]" = convolution_backward_2[0]
    getitem_303: "f32[768, 1, 3, 3]" = convolution_backward_2[1];  convolution_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    sum_147: "f32[768]" = torch.ops.aten.sum.dim_IntList(getitem_302, [0, 2, 3])
    sub_160: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(mul_547, unsqueeze_131);  mul_547 = unsqueeze_131 = None
    mul_738: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_302, sub_160)
    sum_148: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_738, [0, 2, 3]);  mul_738 = None
    mul_739: "f32[768]" = torch.ops.aten.mul.Tensor(sum_147, 0.00015943877551020407)
    unsqueeze_132: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_739, 0);  mul_739 = None
    unsqueeze_133: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_132, 2);  unsqueeze_132 = None
    unsqueeze_134: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_133, 3);  unsqueeze_133 = None
    mul_740: "f32[768]" = torch.ops.aten.mul.Tensor(sum_148, 0.00015943877551020407)
    mul_741: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_76, squeeze_76)
    mul_742: "f32[768]" = torch.ops.aten.mul.Tensor(mul_740, mul_741);  mul_740 = mul_741 = None
    unsqueeze_135: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_742, 0);  mul_742 = None
    unsqueeze_136: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_135, 2);  unsqueeze_135 = None
    unsqueeze_137: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_136, 3);  unsqueeze_136 = None
    mul_743: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_76, primals_563);  primals_563 = None
    unsqueeze_138: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_743, 0);  mul_743 = None
    unsqueeze_139: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_138, 2);  unsqueeze_138 = None
    unsqueeze_140: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_139, 3);  unsqueeze_139 = None
    mul_744: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_160, unsqueeze_137);  sub_160 = unsqueeze_137 = None
    sub_162: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(getitem_302, mul_744);  getitem_302 = mul_744 = None
    sub_163: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(sub_162, unsqueeze_134);  sub_162 = unsqueeze_134 = None
    mul_745: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_163, unsqueeze_140);  sub_163 = unsqueeze_140 = None
    mul_746: "f32[768]" = torch.ops.aten.mul.Tensor(sum_148, squeeze_76);  sum_148 = squeeze_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_748: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(add_403, 0.5);  add_403 = None
    mul_749: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_48, convolution_48)
    mul_750: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_749, -0.5);  mul_749 = None
    exp_29: "f32[8, 768, 28, 28]" = torch.ops.aten.exp.default(mul_750);  mul_750 = None
    mul_751: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(exp_29, 0.3989422804014327);  exp_29 = None
    mul_752: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_48, mul_751);  convolution_48 = mul_751 = None
    add_478: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(mul_748, mul_752);  mul_748 = mul_752 = None
    mul_753: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_745, add_478);  mul_745 = add_478 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    sum_149: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_753, [0, 2, 3])
    convolution_backward_3 = torch.ops.aten.convolution_backward.default(mul_753, view_412, primals_561, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False]);  mul_753 = view_412 = primals_561 = None
    getitem_305: "f32[8, 768, 28, 28]" = convolution_backward_3[0]
    getitem_306: "f32[768, 1, 3, 3]" = convolution_backward_3[1];  convolution_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    view_545: "f32[8, 768, 784]" = torch.ops.aten.reshape.default(getitem_305, [8, 768, 784]);  getitem_305 = None
    permute_334: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_545, [0, 2, 1]);  view_545 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    clone_281: "f32[8, 784, 768]" = torch.ops.aten.clone.default(permute_334, memory_format = torch.contiguous_format);  permute_334 = None
    mul_755: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(clone_281, primals_559);  primals_559 = None
    mul_756: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_755, 768)
    sum_150: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_755, [2], True)
    mul_757: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_755, mul_543);  mul_755 = None
    sum_151: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_757, [2], True);  mul_757 = None
    mul_758: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_543, sum_151);  sum_151 = None
    sub_165: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(mul_756, sum_150);  mul_756 = sum_150 = None
    sub_166: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(sub_165, mul_758);  sub_165 = mul_758 = None
    mul_759: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(div_95, sub_166);  div_95 = sub_166 = None
    mul_760: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(clone_281, mul_543);  mul_543 = None
    sum_152: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_760, [0, 1]);  mul_760 = None
    sum_153: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_281, [0, 1]);  clone_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    add_479: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_476, mul_759);  add_476 = mul_759 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_761: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_479, primals_89);  primals_89 = None
    mul_762: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_479, add_399);  add_399 = None
    sum_154: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_762, [0, 1], True);  mul_762 = None
    view_546: "f32[768]" = torch.ops.aten.reshape.default(sum_154, [768]);  sum_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    sum_155: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_761, [0, 1], True)
    view_547: "f32[768]" = torch.ops.aten.reshape.default(sum_155, [768]);  sum_155 = None
    view_548: "f32[6272, 768]" = torch.ops.aten.reshape.default(mul_761, [6272, 768]);  mul_761 = None
    permute_335: "f32[768, 6272]" = torch.ops.aten.permute.default(view_548, [1, 0])
    mm_64: "f32[768, 768]" = torch.ops.aten.mm.default(permute_335, view_410);  permute_335 = view_410 = None
    permute_336: "f32[768, 768]" = torch.ops.aten.permute.default(mm_64, [1, 0]);  mm_64 = None
    mm_65: "f32[6272, 768]" = torch.ops.aten.mm.default(view_548, permute_337);  view_548 = permute_337 = None
    view_549: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_65, [8, 784, 768]);  mm_65 = None
    permute_338: "f32[768, 768]" = torch.ops.aten.permute.default(permute_336, [1, 0]);  permute_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    view_550: "f32[8, 784, 16, 48]" = torch.ops.aten.reshape.default(view_549, [8, 784, 16, 48]);  view_549 = None
    permute_339: "f32[8, 16, 48, 784]" = torch.ops.aten.permute.default(view_550, [0, 2, 3, 1]);  view_550 = None
    clone_283: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(permute_339, memory_format = torch.contiguous_format);  permute_339 = None
    view_551: "f32[128, 48, 784]" = torch.ops.aten.reshape.default(clone_283, [128, 48, 784]);  clone_283 = None
    bmm_52: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(permute_340, view_551);  permute_340 = None
    bmm_53: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(view_551, permute_341);  view_551 = permute_341 = None
    view_552: "f32[8, 16, 48, 784]" = torch.ops.aten.reshape.default(bmm_52, [8, 16, 48, 784]);  bmm_52 = None
    view_553: "f32[8, 16, 48, 48]" = torch.ops.aten.reshape.default(bmm_53, [8, 16, 48, 48]);  bmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    mul_763: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_553, alias_79);  view_553 = None
    sum_156: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_763, [-1], True)
    mul_764: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(alias_79, sum_156);  alias_79 = sum_156 = None
    sub_167: "f32[8, 16, 48, 48]" = torch.ops.aten.sub.Tensor(mul_763, mul_764);  mul_763 = mul_764 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    mul_765: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(sub_167, view_405);  view_405 = None
    mul_766: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(sub_167, primals_90);  sub_167 = primals_90 = None
    sum_157: "f32[1, 16, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_765, [0, 2, 3], True);  mul_765 = None
    view_554: "f32[16, 1, 1]" = torch.ops.aten.reshape.default(sum_157, [16, 1, 1]);  sum_157 = None
    view_555: "f32[128, 48, 48]" = torch.ops.aten.reshape.default(mul_766, [128, 48, 48]);  mul_766 = None
    bmm_54: "f32[128, 784, 48]" = torch.ops.aten.bmm.default(permute_342, view_555);  permute_342 = None
    bmm_55: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_555, permute_343);  view_555 = permute_343 = None
    view_556: "f32[8, 16, 784, 48]" = torch.ops.aten.reshape.default(bmm_54, [8, 16, 784, 48]);  bmm_54 = None
    view_557: "f32[8, 16, 48, 784]" = torch.ops.aten.reshape.default(bmm_55, [8, 16, 48, 784]);  bmm_55 = None
    permute_344: "f32[8, 16, 48, 784]" = torch.ops.aten.permute.default(view_556, [0, 1, 3, 2]);  view_556 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    div_97: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_73, expand_133);  div_73 = None
    neg_2: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(permute_344)
    mul_767: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_2, div_97);  neg_2 = div_97 = None
    div_98: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(permute_344, expand_133);  permute_344 = expand_133 = None
    sum_158: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_767, [3], True);  mul_767 = None
    ge_2: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(pow_93, 1e-12)
    where_4: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_2, sum_158, full_default_20);  ge_2 = sum_158 = None
    div_99: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_251, pow_93);  getitem_251 = None
    eq_2: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(pow_93, 0);  pow_93 = None
    where_5: "f32[8, 16, 48, 784]" = torch.ops.aten.where.self(eq_2, full_default_20, div_99);  eq_2 = div_99 = None
    clone_284: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(where_5, memory_format = torch.contiguous_format);  where_5 = None
    mul_768: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_4, clone_284);  where_4 = clone_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    add_480: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_98, mul_768);  div_98 = mul_768 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    div_101: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_72, expand_132);  div_72 = None
    neg_3: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(view_557)
    mul_769: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_3, div_101);  neg_3 = div_101 = None
    div_102: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(view_557, expand_132);  view_557 = expand_132 = None
    sum_159: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_769, [3], True);  mul_769 = None
    ge_3: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(pow_91, 1e-12)
    where_6: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_3, sum_159, full_default_20);  ge_3 = sum_159 = None
    div_103: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_250, pow_91);  getitem_250 = None
    eq_3: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(pow_91, 0);  pow_91 = None
    where_7: "f32[8, 16, 48, 784]" = torch.ops.aten.where.self(eq_3, full_default_20, div_103);  eq_3 = div_103 = None
    clone_285: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(where_7, memory_format = torch.contiguous_format);  where_7 = None
    mul_770: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_6, clone_285);  where_6 = clone_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    add_481: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_102, mul_770);  div_102 = mul_770 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    cat_9: "f32[24, 16, 48, 784]" = torch.ops.aten.cat.default([add_481, add_480, view_552]);  add_481 = add_480 = view_552 = None
    view_558: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.reshape.default(cat_9, [3, 8, 16, 48, 784]);  cat_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    permute_345: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.permute.default(view_558, [1, 4, 0, 2, 3]);  view_558 = None
    clone_286: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.clone.default(permute_345, memory_format = torch.contiguous_format);  permute_345 = None
    view_559: "f32[8, 784, 2304]" = torch.ops.aten.reshape.default(clone_286, [8, 784, 2304]);  clone_286 = None
    view_560: "f32[6272, 2304]" = torch.ops.aten.reshape.default(view_559, [6272, 2304]);  view_559 = None
    mm_66: "f32[6272, 768]" = torch.ops.aten.mm.default(view_560, permute_346);  permute_346 = None
    permute_347: "f32[2304, 6272]" = torch.ops.aten.permute.default(view_560, [1, 0])
    mm_67: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_347, view_400);  permute_347 = view_400 = None
    permute_348: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_67, [1, 0]);  mm_67 = None
    sum_160: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_560, [0], True);  view_560 = None
    view_561: "f32[2304]" = torch.ops.aten.reshape.default(sum_160, [2304]);  sum_160 = None
    permute_349: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_348, [1, 0]);  permute_348 = None
    view_562: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_66, [8, 784, 768]);  mm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_772: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_562, primals_553);  primals_553 = None
    mul_773: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_772, 768)
    sum_161: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_772, [2], True)
    mul_774: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_772, mul_539);  mul_772 = None
    sum_162: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_774, [2], True);  mul_774 = None
    mul_775: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_539, sum_162);  sum_162 = None
    sub_169: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(mul_773, sum_161);  mul_773 = sum_161 = None
    sub_170: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(sub_169, mul_775);  sub_169 = mul_775 = None
    mul_776: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(div_104, sub_170);  div_104 = sub_170 = None
    mul_777: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_562, mul_539);  mul_539 = None
    sum_163: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_777, [0, 1]);  mul_777 = None
    sum_164: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_562, [0, 1]);  view_562 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    add_482: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_479, mul_776);  add_479 = mul_776 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_778: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_482, primals_88);  primals_88 = None
    mul_779: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_482, view_399);  view_399 = None
    sum_165: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_779, [0, 1], True);  mul_779 = None
    view_563: "f32[768]" = torch.ops.aten.reshape.default(sum_165, [768]);  sum_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_564: "f32[6272, 768]" = torch.ops.aten.reshape.default(mul_778, [6272, 768]);  mul_778 = None
    mm_68: "f32[6272, 3072]" = torch.ops.aten.mm.default(view_564, permute_350);  permute_350 = None
    permute_351: "f32[768, 6272]" = torch.ops.aten.permute.default(view_564, [1, 0])
    mm_69: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_351, view_398);  permute_351 = view_398 = None
    permute_352: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_69, [1, 0]);  mm_69 = None
    sum_166: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_564, [0], True);  view_564 = None
    view_565: "f32[768]" = torch.ops.aten.reshape.default(sum_166, [768]);  sum_166 = None
    permute_353: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_352, [1, 0]);  permute_352 = None
    view_566: "f32[8, 784, 3072]" = torch.ops.aten.reshape.default(mm_68, [8, 784, 3072]);  mm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_781: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(add_395, 0.5);  add_395 = None
    mul_782: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_397, view_397)
    mul_783: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(mul_782, -0.5);  mul_782 = None
    exp_30: "f32[8, 784, 3072]" = torch.ops.aten.exp.default(mul_783);  mul_783 = None
    mul_784: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(exp_30, 0.3989422804014327);  exp_30 = None
    mul_785: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_397, mul_784);  view_397 = mul_784 = None
    add_484: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(mul_781, mul_785);  mul_781 = mul_785 = None
    mul_786: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_566, add_484);  view_566 = add_484 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_567: "f32[6272, 3072]" = torch.ops.aten.reshape.default(mul_786, [6272, 3072]);  mul_786 = None
    mm_70: "f32[6272, 768]" = torch.ops.aten.mm.default(view_567, permute_354);  permute_354 = None
    permute_355: "f32[3072, 6272]" = torch.ops.aten.permute.default(view_567, [1, 0])
    mm_71: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_355, view_396);  permute_355 = view_396 = None
    permute_356: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_71, [1, 0]);  mm_71 = None
    sum_167: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_567, [0], True);  view_567 = None
    view_568: "f32[3072]" = torch.ops.aten.reshape.default(sum_167, [3072]);  sum_167 = None
    permute_357: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_356, [1, 0]);  permute_356 = None
    view_569: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_70, [8, 784, 768]);  mm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_788: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_569, primals_547);  primals_547 = None
    mul_789: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_788, 768)
    sum_168: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_788, [2], True)
    mul_790: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_788, mul_533);  mul_788 = None
    sum_169: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_790, [2], True);  mul_790 = None
    mul_791: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_533, sum_169);  sum_169 = None
    sub_172: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(mul_789, sum_168);  mul_789 = sum_168 = None
    sub_173: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(sub_172, mul_791);  sub_172 = mul_791 = None
    mul_792: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(div_105, sub_173);  div_105 = sub_173 = None
    mul_793: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_569, mul_533);  mul_533 = None
    sum_170: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_793, [0, 1]);  mul_793 = None
    sum_171: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_569, [0, 1]);  view_569 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    add_485: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_482, mul_792);  add_482 = mul_792 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_794: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_485, primals_87);  primals_87 = None
    mul_795: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_485, permute_198);  permute_198 = None
    sum_172: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_795, [0, 1], True);  mul_795 = None
    view_570: "f32[768]" = torch.ops.aten.reshape.default(sum_172, [768]);  sum_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    permute_358: "f32[8, 768, 784]" = torch.ops.aten.permute.default(mul_794, [0, 2, 1]);  mul_794 = None
    view_571: "f32[8, 768, 28, 28]" = torch.ops.aten.reshape.default(permute_358, [8, 768, 28, 28]);  permute_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    sum_173: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_571, [0, 2, 3])
    convolution_backward_4 = torch.ops.aten.convolution_backward.default(view_571, add_391, primals_545, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False]);  view_571 = add_391 = primals_545 = None
    getitem_308: "f32[8, 768, 28, 28]" = convolution_backward_4[0]
    getitem_309: "f32[768, 1, 3, 3]" = convolution_backward_4[1];  convolution_backward_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    sum_174: "f32[768]" = torch.ops.aten.sum.dim_IntList(getitem_308, [0, 2, 3])
    sub_174: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(mul_524, unsqueeze_143);  mul_524 = unsqueeze_143 = None
    mul_796: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_308, sub_174)
    sum_175: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_796, [0, 2, 3]);  mul_796 = None
    mul_797: "f32[768]" = torch.ops.aten.mul.Tensor(sum_174, 0.00015943877551020407)
    unsqueeze_144: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_797, 0);  mul_797 = None
    unsqueeze_145: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_144, 2);  unsqueeze_144 = None
    unsqueeze_146: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_145, 3);  unsqueeze_145 = None
    mul_798: "f32[768]" = torch.ops.aten.mul.Tensor(sum_175, 0.00015943877551020407)
    mul_799: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_73, squeeze_73)
    mul_800: "f32[768]" = torch.ops.aten.mul.Tensor(mul_798, mul_799);  mul_798 = mul_799 = None
    unsqueeze_147: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_800, 0);  mul_800 = None
    unsqueeze_148: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_147, 2);  unsqueeze_147 = None
    unsqueeze_149: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_148, 3);  unsqueeze_148 = None
    mul_801: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_73, primals_543);  primals_543 = None
    unsqueeze_150: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_801, 0);  mul_801 = None
    unsqueeze_151: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_150, 2);  unsqueeze_150 = None
    unsqueeze_152: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_151, 3);  unsqueeze_151 = None
    mul_802: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_174, unsqueeze_149);  sub_174 = unsqueeze_149 = None
    sub_176: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(getitem_308, mul_802);  getitem_308 = mul_802 = None
    sub_177: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(sub_176, unsqueeze_146);  sub_176 = unsqueeze_146 = None
    mul_803: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_177, unsqueeze_152);  sub_177 = unsqueeze_152 = None
    mul_804: "f32[768]" = torch.ops.aten.mul.Tensor(sum_175, squeeze_73);  sum_175 = squeeze_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_806: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(add_386, 0.5);  add_386 = None
    mul_807: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_46, convolution_46)
    mul_808: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_807, -0.5);  mul_807 = None
    exp_31: "f32[8, 768, 28, 28]" = torch.ops.aten.exp.default(mul_808);  mul_808 = None
    mul_809: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(exp_31, 0.3989422804014327);  exp_31 = None
    mul_810: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_46, mul_809);  convolution_46 = mul_809 = None
    add_487: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(mul_806, mul_810);  mul_806 = mul_810 = None
    mul_811: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_803, add_487);  mul_803 = add_487 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    sum_176: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_811, [0, 2, 3])
    convolution_backward_5 = torch.ops.aten.convolution_backward.default(mul_811, view_394, primals_541, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False]);  mul_811 = view_394 = primals_541 = None
    getitem_311: "f32[8, 768, 28, 28]" = convolution_backward_5[0]
    getitem_312: "f32[768, 1, 3, 3]" = convolution_backward_5[1];  convolution_backward_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    view_572: "f32[8, 768, 784]" = torch.ops.aten.reshape.default(getitem_311, [8, 768, 784]);  getitem_311 = None
    permute_359: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_572, [0, 2, 1]);  view_572 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    clone_289: "f32[8, 784, 768]" = torch.ops.aten.clone.default(permute_359, memory_format = torch.contiguous_format);  permute_359 = None
    mul_813: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(clone_289, primals_539);  primals_539 = None
    mul_814: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_813, 768)
    sum_177: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_813, [2], True)
    mul_815: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_813, mul_520);  mul_813 = None
    sum_178: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_815, [2], True);  mul_815 = None
    mul_816: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_520, sum_178);  sum_178 = None
    sub_179: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(mul_814, sum_177);  mul_814 = sum_177 = None
    sub_180: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(sub_179, mul_816);  sub_179 = mul_816 = None
    mul_817: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(div_106, sub_180);  div_106 = sub_180 = None
    mul_818: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(clone_289, mul_520);  mul_520 = None
    sum_179: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_818, [0, 1]);  mul_818 = None
    sum_180: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_289, [0, 1]);  clone_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    add_488: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_485, mul_817);  add_485 = mul_817 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_819: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_488, primals_85);  primals_85 = None
    mul_820: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_488, add_382);  add_382 = None
    sum_181: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_820, [0, 1], True);  mul_820 = None
    view_573: "f32[768]" = torch.ops.aten.reshape.default(sum_181, [768]);  sum_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    sum_182: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_819, [0, 1], True)
    view_574: "f32[768]" = torch.ops.aten.reshape.default(sum_182, [768]);  sum_182 = None
    view_575: "f32[6272, 768]" = torch.ops.aten.reshape.default(mul_819, [6272, 768]);  mul_819 = None
    permute_360: "f32[768, 6272]" = torch.ops.aten.permute.default(view_575, [1, 0])
    mm_72: "f32[768, 768]" = torch.ops.aten.mm.default(permute_360, view_392);  permute_360 = view_392 = None
    permute_361: "f32[768, 768]" = torch.ops.aten.permute.default(mm_72, [1, 0]);  mm_72 = None
    mm_73: "f32[6272, 768]" = torch.ops.aten.mm.default(view_575, permute_362);  view_575 = permute_362 = None
    view_576: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_73, [8, 784, 768]);  mm_73 = None
    permute_363: "f32[768, 768]" = torch.ops.aten.permute.default(permute_361, [1, 0]);  permute_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    view_577: "f32[8, 784, 16, 48]" = torch.ops.aten.reshape.default(view_576, [8, 784, 16, 48]);  view_576 = None
    permute_364: "f32[8, 16, 48, 784]" = torch.ops.aten.permute.default(view_577, [0, 2, 3, 1]);  view_577 = None
    clone_291: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(permute_364, memory_format = torch.contiguous_format);  permute_364 = None
    view_578: "f32[128, 48, 784]" = torch.ops.aten.reshape.default(clone_291, [128, 48, 784]);  clone_291 = None
    bmm_56: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(permute_365, view_578);  permute_365 = None
    bmm_57: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(view_578, permute_366);  view_578 = permute_366 = None
    view_579: "f32[8, 16, 48, 784]" = torch.ops.aten.reshape.default(bmm_56, [8, 16, 48, 784]);  bmm_56 = None
    view_580: "f32[8, 16, 48, 48]" = torch.ops.aten.reshape.default(bmm_57, [8, 16, 48, 48]);  bmm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    mul_821: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_580, alias_82);  view_580 = None
    sum_183: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_821, [-1], True)
    mul_822: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(alias_82, sum_183);  alias_82 = sum_183 = None
    sub_181: "f32[8, 16, 48, 48]" = torch.ops.aten.sub.Tensor(mul_821, mul_822);  mul_821 = mul_822 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    mul_823: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(sub_181, view_387);  view_387 = None
    mul_824: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(sub_181, primals_86);  sub_181 = primals_86 = None
    sum_184: "f32[1, 16, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_823, [0, 2, 3], True);  mul_823 = None
    view_581: "f32[16, 1, 1]" = torch.ops.aten.reshape.default(sum_184, [16, 1, 1]);  sum_184 = None
    view_582: "f32[128, 48, 48]" = torch.ops.aten.reshape.default(mul_824, [128, 48, 48]);  mul_824 = None
    bmm_58: "f32[128, 784, 48]" = torch.ops.aten.bmm.default(permute_367, view_582);  permute_367 = None
    bmm_59: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_582, permute_368);  view_582 = permute_368 = None
    view_583: "f32[8, 16, 784, 48]" = torch.ops.aten.reshape.default(bmm_58, [8, 16, 784, 48]);  bmm_58 = None
    view_584: "f32[8, 16, 48, 784]" = torch.ops.aten.reshape.default(bmm_59, [8, 16, 48, 784]);  bmm_59 = None
    permute_369: "f32[8, 16, 48, 784]" = torch.ops.aten.permute.default(view_583, [0, 1, 3, 2]);  view_583 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    div_108: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_70, expand_127);  div_70 = None
    neg_4: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(permute_369)
    mul_825: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_4, div_108);  neg_4 = div_108 = None
    div_109: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(permute_369, expand_127);  permute_369 = expand_127 = None
    sum_185: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_825, [3], True);  mul_825 = None
    ge_4: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(pow_89, 1e-12)
    where_8: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_4, sum_185, full_default_20);  ge_4 = sum_185 = None
    div_110: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_240, pow_89);  getitem_240 = None
    eq_4: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(pow_89, 0);  pow_89 = None
    where_9: "f32[8, 16, 48, 784]" = torch.ops.aten.where.self(eq_4, full_default_20, div_110);  eq_4 = div_110 = None
    clone_292: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(where_9, memory_format = torch.contiguous_format);  where_9 = None
    mul_826: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_8, clone_292);  where_8 = clone_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    add_489: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_109, mul_826);  div_109 = mul_826 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    div_112: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_69, expand_126);  div_69 = None
    neg_5: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(view_584)
    mul_827: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_5, div_112);  neg_5 = div_112 = None
    div_113: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(view_584, expand_126);  view_584 = expand_126 = None
    sum_186: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_827, [3], True);  mul_827 = None
    ge_5: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(pow_87, 1e-12)
    where_10: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_5, sum_186, full_default_20);  ge_5 = sum_186 = None
    div_114: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_239, pow_87);  getitem_239 = None
    eq_5: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(pow_87, 0);  pow_87 = None
    where_11: "f32[8, 16, 48, 784]" = torch.ops.aten.where.self(eq_5, full_default_20, div_114);  eq_5 = div_114 = None
    clone_293: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(where_11, memory_format = torch.contiguous_format);  where_11 = None
    mul_828: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_10, clone_293);  where_10 = clone_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    add_490: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_113, mul_828);  div_113 = mul_828 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    cat_10: "f32[24, 16, 48, 784]" = torch.ops.aten.cat.default([add_490, add_489, view_579]);  add_490 = add_489 = view_579 = None
    view_585: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.reshape.default(cat_10, [3, 8, 16, 48, 784]);  cat_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    permute_370: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.permute.default(view_585, [1, 4, 0, 2, 3]);  view_585 = None
    clone_294: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.clone.default(permute_370, memory_format = torch.contiguous_format);  permute_370 = None
    view_586: "f32[8, 784, 2304]" = torch.ops.aten.reshape.default(clone_294, [8, 784, 2304]);  clone_294 = None
    view_587: "f32[6272, 2304]" = torch.ops.aten.reshape.default(view_586, [6272, 2304]);  view_586 = None
    mm_74: "f32[6272, 768]" = torch.ops.aten.mm.default(view_587, permute_371);  permute_371 = None
    permute_372: "f32[2304, 6272]" = torch.ops.aten.permute.default(view_587, [1, 0])
    mm_75: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_372, view_382);  permute_372 = view_382 = None
    permute_373: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_75, [1, 0]);  mm_75 = None
    sum_187: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_587, [0], True);  view_587 = None
    view_588: "f32[2304]" = torch.ops.aten.reshape.default(sum_187, [2304]);  sum_187 = None
    permute_374: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_373, [1, 0]);  permute_373 = None
    view_589: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_74, [8, 784, 768]);  mm_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_830: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_589, primals_533);  primals_533 = None
    mul_831: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_830, 768)
    sum_188: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_830, [2], True)
    mul_832: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_830, mul_516);  mul_830 = None
    sum_189: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_832, [2], True);  mul_832 = None
    mul_833: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_516, sum_189);  sum_189 = None
    sub_183: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(mul_831, sum_188);  mul_831 = sum_188 = None
    sub_184: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(sub_183, mul_833);  sub_183 = mul_833 = None
    mul_834: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(div_115, sub_184);  div_115 = sub_184 = None
    mul_835: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_589, mul_516);  mul_516 = None
    sum_190: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_835, [0, 1]);  mul_835 = None
    sum_191: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_589, [0, 1]);  view_589 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    add_491: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_488, mul_834);  add_488 = mul_834 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_836: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_491, primals_84);  primals_84 = None
    mul_837: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_491, view_381);  view_381 = None
    sum_192: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_837, [0, 1], True);  mul_837 = None
    view_590: "f32[768]" = torch.ops.aten.reshape.default(sum_192, [768]);  sum_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_591: "f32[6272, 768]" = torch.ops.aten.reshape.default(mul_836, [6272, 768]);  mul_836 = None
    mm_76: "f32[6272, 3072]" = torch.ops.aten.mm.default(view_591, permute_375);  permute_375 = None
    permute_376: "f32[768, 6272]" = torch.ops.aten.permute.default(view_591, [1, 0])
    mm_77: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_376, view_380);  permute_376 = view_380 = None
    permute_377: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_77, [1, 0]);  mm_77 = None
    sum_193: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_591, [0], True);  view_591 = None
    view_592: "f32[768]" = torch.ops.aten.reshape.default(sum_193, [768]);  sum_193 = None
    permute_378: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_377, [1, 0]);  permute_377 = None
    view_593: "f32[8, 784, 3072]" = torch.ops.aten.reshape.default(mm_76, [8, 784, 3072]);  mm_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_839: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(add_378, 0.5);  add_378 = None
    mul_840: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_379, view_379)
    mul_841: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(mul_840, -0.5);  mul_840 = None
    exp_32: "f32[8, 784, 3072]" = torch.ops.aten.exp.default(mul_841);  mul_841 = None
    mul_842: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(exp_32, 0.3989422804014327);  exp_32 = None
    mul_843: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_379, mul_842);  view_379 = mul_842 = None
    add_493: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(mul_839, mul_843);  mul_839 = mul_843 = None
    mul_844: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_593, add_493);  view_593 = add_493 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_594: "f32[6272, 3072]" = torch.ops.aten.reshape.default(mul_844, [6272, 3072]);  mul_844 = None
    mm_78: "f32[6272, 768]" = torch.ops.aten.mm.default(view_594, permute_379);  permute_379 = None
    permute_380: "f32[3072, 6272]" = torch.ops.aten.permute.default(view_594, [1, 0])
    mm_79: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_380, view_378);  permute_380 = view_378 = None
    permute_381: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_79, [1, 0]);  mm_79 = None
    sum_194: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_594, [0], True);  view_594 = None
    view_595: "f32[3072]" = torch.ops.aten.reshape.default(sum_194, [3072]);  sum_194 = None
    permute_382: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_381, [1, 0]);  permute_381 = None
    view_596: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_78, [8, 784, 768]);  mm_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_846: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_596, primals_527);  primals_527 = None
    mul_847: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_846, 768)
    sum_195: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_846, [2], True)
    mul_848: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_846, mul_510);  mul_846 = None
    sum_196: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_848, [2], True);  mul_848 = None
    mul_849: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_510, sum_196);  sum_196 = None
    sub_186: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(mul_847, sum_195);  mul_847 = sum_195 = None
    sub_187: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(sub_186, mul_849);  sub_186 = mul_849 = None
    mul_850: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(div_116, sub_187);  div_116 = sub_187 = None
    mul_851: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_596, mul_510);  mul_510 = None
    sum_197: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_851, [0, 1]);  mul_851 = None
    sum_198: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_596, [0, 1]);  view_596 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    add_494: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_491, mul_850);  add_491 = mul_850 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_852: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_494, primals_83);  primals_83 = None
    mul_853: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_494, permute_189);  permute_189 = None
    sum_199: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_853, [0, 1], True);  mul_853 = None
    view_597: "f32[768]" = torch.ops.aten.reshape.default(sum_199, [768]);  sum_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    permute_383: "f32[8, 768, 784]" = torch.ops.aten.permute.default(mul_852, [0, 2, 1]);  mul_852 = None
    view_598: "f32[8, 768, 28, 28]" = torch.ops.aten.reshape.default(permute_383, [8, 768, 28, 28]);  permute_383 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    sum_200: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_598, [0, 2, 3])
    convolution_backward_6 = torch.ops.aten.convolution_backward.default(view_598, add_374, primals_525, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False]);  view_598 = add_374 = primals_525 = None
    getitem_314: "f32[8, 768, 28, 28]" = convolution_backward_6[0]
    getitem_315: "f32[768, 1, 3, 3]" = convolution_backward_6[1];  convolution_backward_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    sum_201: "f32[768]" = torch.ops.aten.sum.dim_IntList(getitem_314, [0, 2, 3])
    sub_188: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(mul_501, unsqueeze_155);  mul_501 = unsqueeze_155 = None
    mul_854: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_314, sub_188)
    sum_202: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_854, [0, 2, 3]);  mul_854 = None
    mul_855: "f32[768]" = torch.ops.aten.mul.Tensor(sum_201, 0.00015943877551020407)
    unsqueeze_156: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_855, 0);  mul_855 = None
    unsqueeze_157: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_156, 2);  unsqueeze_156 = None
    unsqueeze_158: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_157, 3);  unsqueeze_157 = None
    mul_856: "f32[768]" = torch.ops.aten.mul.Tensor(sum_202, 0.00015943877551020407)
    mul_857: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_70, squeeze_70)
    mul_858: "f32[768]" = torch.ops.aten.mul.Tensor(mul_856, mul_857);  mul_856 = mul_857 = None
    unsqueeze_159: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_858, 0);  mul_858 = None
    unsqueeze_160: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_159, 2);  unsqueeze_159 = None
    unsqueeze_161: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_160, 3);  unsqueeze_160 = None
    mul_859: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_70, primals_523);  primals_523 = None
    unsqueeze_162: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_859, 0);  mul_859 = None
    unsqueeze_163: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_162, 2);  unsqueeze_162 = None
    unsqueeze_164: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_163, 3);  unsqueeze_163 = None
    mul_860: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_188, unsqueeze_161);  sub_188 = unsqueeze_161 = None
    sub_190: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(getitem_314, mul_860);  getitem_314 = mul_860 = None
    sub_191: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(sub_190, unsqueeze_158);  sub_190 = unsqueeze_158 = None
    mul_861: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_191, unsqueeze_164);  sub_191 = unsqueeze_164 = None
    mul_862: "f32[768]" = torch.ops.aten.mul.Tensor(sum_202, squeeze_70);  sum_202 = squeeze_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_864: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(add_369, 0.5);  add_369 = None
    mul_865: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_44, convolution_44)
    mul_866: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_865, -0.5);  mul_865 = None
    exp_33: "f32[8, 768, 28, 28]" = torch.ops.aten.exp.default(mul_866);  mul_866 = None
    mul_867: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(exp_33, 0.3989422804014327);  exp_33 = None
    mul_868: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_44, mul_867);  convolution_44 = mul_867 = None
    add_496: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(mul_864, mul_868);  mul_864 = mul_868 = None
    mul_869: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_861, add_496);  mul_861 = add_496 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    sum_203: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_869, [0, 2, 3])
    convolution_backward_7 = torch.ops.aten.convolution_backward.default(mul_869, view_376, primals_521, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False]);  mul_869 = view_376 = primals_521 = None
    getitem_317: "f32[8, 768, 28, 28]" = convolution_backward_7[0]
    getitem_318: "f32[768, 1, 3, 3]" = convolution_backward_7[1];  convolution_backward_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    view_599: "f32[8, 768, 784]" = torch.ops.aten.reshape.default(getitem_317, [8, 768, 784]);  getitem_317 = None
    permute_384: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_599, [0, 2, 1]);  view_599 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    clone_297: "f32[8, 784, 768]" = torch.ops.aten.clone.default(permute_384, memory_format = torch.contiguous_format);  permute_384 = None
    mul_871: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(clone_297, primals_519);  primals_519 = None
    mul_872: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_871, 768)
    sum_204: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_871, [2], True)
    mul_873: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_871, mul_497);  mul_871 = None
    sum_205: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_873, [2], True);  mul_873 = None
    mul_874: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_497, sum_205);  sum_205 = None
    sub_193: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(mul_872, sum_204);  mul_872 = sum_204 = None
    sub_194: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(sub_193, mul_874);  sub_193 = mul_874 = None
    mul_875: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(div_117, sub_194);  div_117 = sub_194 = None
    mul_876: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(clone_297, mul_497);  mul_497 = None
    sum_206: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_876, [0, 1]);  mul_876 = None
    sum_207: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_297, [0, 1]);  clone_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    add_497: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_494, mul_875);  add_494 = mul_875 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_877: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_497, primals_81);  primals_81 = None
    mul_878: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_497, add_365);  add_365 = None
    sum_208: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_878, [0, 1], True);  mul_878 = None
    view_600: "f32[768]" = torch.ops.aten.reshape.default(sum_208, [768]);  sum_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    sum_209: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_877, [0, 1], True)
    view_601: "f32[768]" = torch.ops.aten.reshape.default(sum_209, [768]);  sum_209 = None
    view_602: "f32[6272, 768]" = torch.ops.aten.reshape.default(mul_877, [6272, 768]);  mul_877 = None
    permute_385: "f32[768, 6272]" = torch.ops.aten.permute.default(view_602, [1, 0])
    mm_80: "f32[768, 768]" = torch.ops.aten.mm.default(permute_385, view_374);  permute_385 = view_374 = None
    permute_386: "f32[768, 768]" = torch.ops.aten.permute.default(mm_80, [1, 0]);  mm_80 = None
    mm_81: "f32[6272, 768]" = torch.ops.aten.mm.default(view_602, permute_387);  view_602 = permute_387 = None
    view_603: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_81, [8, 784, 768]);  mm_81 = None
    permute_388: "f32[768, 768]" = torch.ops.aten.permute.default(permute_386, [1, 0]);  permute_386 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    view_604: "f32[8, 784, 16, 48]" = torch.ops.aten.reshape.default(view_603, [8, 784, 16, 48]);  view_603 = None
    permute_389: "f32[8, 16, 48, 784]" = torch.ops.aten.permute.default(view_604, [0, 2, 3, 1]);  view_604 = None
    clone_299: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(permute_389, memory_format = torch.contiguous_format);  permute_389 = None
    view_605: "f32[128, 48, 784]" = torch.ops.aten.reshape.default(clone_299, [128, 48, 784]);  clone_299 = None
    bmm_60: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(permute_390, view_605);  permute_390 = None
    bmm_61: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(view_605, permute_391);  view_605 = permute_391 = None
    view_606: "f32[8, 16, 48, 784]" = torch.ops.aten.reshape.default(bmm_60, [8, 16, 48, 784]);  bmm_60 = None
    view_607: "f32[8, 16, 48, 48]" = torch.ops.aten.reshape.default(bmm_61, [8, 16, 48, 48]);  bmm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    mul_879: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_607, alias_85);  view_607 = None
    sum_210: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_879, [-1], True)
    mul_880: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(alias_85, sum_210);  alias_85 = sum_210 = None
    sub_195: "f32[8, 16, 48, 48]" = torch.ops.aten.sub.Tensor(mul_879, mul_880);  mul_879 = mul_880 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    mul_881: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(sub_195, view_369);  view_369 = None
    mul_882: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(sub_195, primals_82);  sub_195 = primals_82 = None
    sum_211: "f32[1, 16, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_881, [0, 2, 3], True);  mul_881 = None
    view_608: "f32[16, 1, 1]" = torch.ops.aten.reshape.default(sum_211, [16, 1, 1]);  sum_211 = None
    view_609: "f32[128, 48, 48]" = torch.ops.aten.reshape.default(mul_882, [128, 48, 48]);  mul_882 = None
    bmm_62: "f32[128, 784, 48]" = torch.ops.aten.bmm.default(permute_392, view_609);  permute_392 = None
    bmm_63: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_609, permute_393);  view_609 = permute_393 = None
    view_610: "f32[8, 16, 784, 48]" = torch.ops.aten.reshape.default(bmm_62, [8, 16, 784, 48]);  bmm_62 = None
    view_611: "f32[8, 16, 48, 784]" = torch.ops.aten.reshape.default(bmm_63, [8, 16, 48, 784]);  bmm_63 = None
    permute_394: "f32[8, 16, 48, 784]" = torch.ops.aten.permute.default(view_610, [0, 1, 3, 2]);  view_610 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    div_119: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_67, expand_121);  div_67 = None
    neg_6: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(permute_394)
    mul_883: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_6, div_119);  neg_6 = div_119 = None
    div_120: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(permute_394, expand_121);  permute_394 = expand_121 = None
    sum_212: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_883, [3], True);  mul_883 = None
    ge_6: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(pow_85, 1e-12)
    where_12: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_6, sum_212, full_default_20);  ge_6 = sum_212 = None
    div_121: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_229, pow_85);  getitem_229 = None
    eq_6: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(pow_85, 0);  pow_85 = None
    where_13: "f32[8, 16, 48, 784]" = torch.ops.aten.where.self(eq_6, full_default_20, div_121);  eq_6 = div_121 = None
    clone_300: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(where_13, memory_format = torch.contiguous_format);  where_13 = None
    mul_884: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_12, clone_300);  where_12 = clone_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    add_498: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_120, mul_884);  div_120 = mul_884 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    div_123: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_66, expand_120);  div_66 = None
    neg_7: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(view_611)
    mul_885: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_7, div_123);  neg_7 = div_123 = None
    div_124: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(view_611, expand_120);  view_611 = expand_120 = None
    sum_213: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_885, [3], True);  mul_885 = None
    ge_7: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(pow_83, 1e-12)
    where_14: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_7, sum_213, full_default_20);  ge_7 = sum_213 = None
    div_125: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_228, pow_83);  getitem_228 = None
    eq_7: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(pow_83, 0);  pow_83 = None
    where_15: "f32[8, 16, 48, 784]" = torch.ops.aten.where.self(eq_7, full_default_20, div_125);  eq_7 = div_125 = None
    clone_301: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(where_15, memory_format = torch.contiguous_format);  where_15 = None
    mul_886: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_14, clone_301);  where_14 = clone_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    add_499: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_124, mul_886);  div_124 = mul_886 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    cat_11: "f32[24, 16, 48, 784]" = torch.ops.aten.cat.default([add_499, add_498, view_606]);  add_499 = add_498 = view_606 = None
    view_612: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.reshape.default(cat_11, [3, 8, 16, 48, 784]);  cat_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    permute_395: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.permute.default(view_612, [1, 4, 0, 2, 3]);  view_612 = None
    clone_302: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.clone.default(permute_395, memory_format = torch.contiguous_format);  permute_395 = None
    view_613: "f32[8, 784, 2304]" = torch.ops.aten.reshape.default(clone_302, [8, 784, 2304]);  clone_302 = None
    view_614: "f32[6272, 2304]" = torch.ops.aten.reshape.default(view_613, [6272, 2304]);  view_613 = None
    mm_82: "f32[6272, 768]" = torch.ops.aten.mm.default(view_614, permute_396);  permute_396 = None
    permute_397: "f32[2304, 6272]" = torch.ops.aten.permute.default(view_614, [1, 0])
    mm_83: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_397, view_364);  permute_397 = view_364 = None
    permute_398: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_83, [1, 0]);  mm_83 = None
    sum_214: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_614, [0], True);  view_614 = None
    view_615: "f32[2304]" = torch.ops.aten.reshape.default(sum_214, [2304]);  sum_214 = None
    permute_399: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_398, [1, 0]);  permute_398 = None
    view_616: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_82, [8, 784, 768]);  mm_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_888: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_616, primals_513);  primals_513 = None
    mul_889: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_888, 768)
    sum_215: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_888, [2], True)
    mul_890: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_888, mul_493);  mul_888 = None
    sum_216: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_890, [2], True);  mul_890 = None
    mul_891: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_493, sum_216);  sum_216 = None
    sub_197: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(mul_889, sum_215);  mul_889 = sum_215 = None
    sub_198: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(sub_197, mul_891);  sub_197 = mul_891 = None
    mul_892: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(div_126, sub_198);  div_126 = sub_198 = None
    mul_893: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_616, mul_493);  mul_493 = None
    sum_217: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_893, [0, 1]);  mul_893 = None
    sum_218: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_616, [0, 1]);  view_616 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    add_500: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_497, mul_892);  add_497 = mul_892 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_894: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_500, primals_80);  primals_80 = None
    mul_895: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_500, view_363);  view_363 = None
    sum_219: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_895, [0, 1], True);  mul_895 = None
    view_617: "f32[768]" = torch.ops.aten.reshape.default(sum_219, [768]);  sum_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_618: "f32[6272, 768]" = torch.ops.aten.reshape.default(mul_894, [6272, 768]);  mul_894 = None
    mm_84: "f32[6272, 3072]" = torch.ops.aten.mm.default(view_618, permute_400);  permute_400 = None
    permute_401: "f32[768, 6272]" = torch.ops.aten.permute.default(view_618, [1, 0])
    mm_85: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_401, view_362);  permute_401 = view_362 = None
    permute_402: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_85, [1, 0]);  mm_85 = None
    sum_220: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_618, [0], True);  view_618 = None
    view_619: "f32[768]" = torch.ops.aten.reshape.default(sum_220, [768]);  sum_220 = None
    permute_403: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_402, [1, 0]);  permute_402 = None
    view_620: "f32[8, 784, 3072]" = torch.ops.aten.reshape.default(mm_84, [8, 784, 3072]);  mm_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_897: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(add_361, 0.5);  add_361 = None
    mul_898: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_361, view_361)
    mul_899: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(mul_898, -0.5);  mul_898 = None
    exp_34: "f32[8, 784, 3072]" = torch.ops.aten.exp.default(mul_899);  mul_899 = None
    mul_900: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(exp_34, 0.3989422804014327);  exp_34 = None
    mul_901: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_361, mul_900);  view_361 = mul_900 = None
    add_502: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(mul_897, mul_901);  mul_897 = mul_901 = None
    mul_902: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_620, add_502);  view_620 = add_502 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_621: "f32[6272, 3072]" = torch.ops.aten.reshape.default(mul_902, [6272, 3072]);  mul_902 = None
    mm_86: "f32[6272, 768]" = torch.ops.aten.mm.default(view_621, permute_404);  permute_404 = None
    permute_405: "f32[3072, 6272]" = torch.ops.aten.permute.default(view_621, [1, 0])
    mm_87: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_405, view_360);  permute_405 = view_360 = None
    permute_406: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_87, [1, 0]);  mm_87 = None
    sum_221: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_621, [0], True);  view_621 = None
    view_622: "f32[3072]" = torch.ops.aten.reshape.default(sum_221, [3072]);  sum_221 = None
    permute_407: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_406, [1, 0]);  permute_406 = None
    view_623: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_86, [8, 784, 768]);  mm_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_904: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_623, primals_507);  primals_507 = None
    mul_905: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_904, 768)
    sum_222: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_904, [2], True)
    mul_906: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_904, mul_487);  mul_904 = None
    sum_223: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_906, [2], True);  mul_906 = None
    mul_907: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_487, sum_223);  sum_223 = None
    sub_200: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(mul_905, sum_222);  mul_905 = sum_222 = None
    sub_201: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(sub_200, mul_907);  sub_200 = mul_907 = None
    mul_908: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(div_127, sub_201);  div_127 = sub_201 = None
    mul_909: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_623, mul_487);  mul_487 = None
    sum_224: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_909, [0, 1]);  mul_909 = None
    sum_225: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_623, [0, 1]);  view_623 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    add_503: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_500, mul_908);  add_500 = mul_908 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_910: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_503, primals_79);  primals_79 = None
    mul_911: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_503, permute_180);  permute_180 = None
    sum_226: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_911, [0, 1], True);  mul_911 = None
    view_624: "f32[768]" = torch.ops.aten.reshape.default(sum_226, [768]);  sum_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    permute_408: "f32[8, 768, 784]" = torch.ops.aten.permute.default(mul_910, [0, 2, 1]);  mul_910 = None
    view_625: "f32[8, 768, 28, 28]" = torch.ops.aten.reshape.default(permute_408, [8, 768, 28, 28]);  permute_408 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    sum_227: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_625, [0, 2, 3])
    convolution_backward_8 = torch.ops.aten.convolution_backward.default(view_625, add_357, primals_505, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False]);  view_625 = add_357 = primals_505 = None
    getitem_320: "f32[8, 768, 28, 28]" = convolution_backward_8[0]
    getitem_321: "f32[768, 1, 3, 3]" = convolution_backward_8[1];  convolution_backward_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    sum_228: "f32[768]" = torch.ops.aten.sum.dim_IntList(getitem_320, [0, 2, 3])
    sub_202: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(mul_478, unsqueeze_167);  mul_478 = unsqueeze_167 = None
    mul_912: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_320, sub_202)
    sum_229: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_912, [0, 2, 3]);  mul_912 = None
    mul_913: "f32[768]" = torch.ops.aten.mul.Tensor(sum_228, 0.00015943877551020407)
    unsqueeze_168: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_913, 0);  mul_913 = None
    unsqueeze_169: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_168, 2);  unsqueeze_168 = None
    unsqueeze_170: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_169, 3);  unsqueeze_169 = None
    mul_914: "f32[768]" = torch.ops.aten.mul.Tensor(sum_229, 0.00015943877551020407)
    mul_915: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_67, squeeze_67)
    mul_916: "f32[768]" = torch.ops.aten.mul.Tensor(mul_914, mul_915);  mul_914 = mul_915 = None
    unsqueeze_171: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_916, 0);  mul_916 = None
    unsqueeze_172: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_171, 2);  unsqueeze_171 = None
    unsqueeze_173: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_172, 3);  unsqueeze_172 = None
    mul_917: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_67, primals_503);  primals_503 = None
    unsqueeze_174: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_917, 0);  mul_917 = None
    unsqueeze_175: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_174, 2);  unsqueeze_174 = None
    unsqueeze_176: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_175, 3);  unsqueeze_175 = None
    mul_918: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_202, unsqueeze_173);  sub_202 = unsqueeze_173 = None
    sub_204: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(getitem_320, mul_918);  getitem_320 = mul_918 = None
    sub_205: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(sub_204, unsqueeze_170);  sub_204 = unsqueeze_170 = None
    mul_919: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_205, unsqueeze_176);  sub_205 = unsqueeze_176 = None
    mul_920: "f32[768]" = torch.ops.aten.mul.Tensor(sum_229, squeeze_67);  sum_229 = squeeze_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_922: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(add_352, 0.5);  add_352 = None
    mul_923: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_42, convolution_42)
    mul_924: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_923, -0.5);  mul_923 = None
    exp_35: "f32[8, 768, 28, 28]" = torch.ops.aten.exp.default(mul_924);  mul_924 = None
    mul_925: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(exp_35, 0.3989422804014327);  exp_35 = None
    mul_926: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_42, mul_925);  convolution_42 = mul_925 = None
    add_505: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(mul_922, mul_926);  mul_922 = mul_926 = None
    mul_927: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_919, add_505);  mul_919 = add_505 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    sum_230: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_927, [0, 2, 3])
    convolution_backward_9 = torch.ops.aten.convolution_backward.default(mul_927, view_358, primals_501, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False]);  mul_927 = view_358 = primals_501 = None
    getitem_323: "f32[8, 768, 28, 28]" = convolution_backward_9[0]
    getitem_324: "f32[768, 1, 3, 3]" = convolution_backward_9[1];  convolution_backward_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    view_626: "f32[8, 768, 784]" = torch.ops.aten.reshape.default(getitem_323, [8, 768, 784]);  getitem_323 = None
    permute_409: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_626, [0, 2, 1]);  view_626 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    clone_305: "f32[8, 784, 768]" = torch.ops.aten.clone.default(permute_409, memory_format = torch.contiguous_format);  permute_409 = None
    mul_929: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(clone_305, primals_499);  primals_499 = None
    mul_930: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_929, 768)
    sum_231: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_929, [2], True)
    mul_931: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_929, mul_474);  mul_929 = None
    sum_232: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_931, [2], True);  mul_931 = None
    mul_932: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_474, sum_232);  sum_232 = None
    sub_207: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(mul_930, sum_231);  mul_930 = sum_231 = None
    sub_208: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(sub_207, mul_932);  sub_207 = mul_932 = None
    mul_933: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(div_128, sub_208);  div_128 = sub_208 = None
    mul_934: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(clone_305, mul_474);  mul_474 = None
    sum_233: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_934, [0, 1]);  mul_934 = None
    sum_234: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_305, [0, 1]);  clone_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    add_506: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_503, mul_933);  add_503 = mul_933 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_935: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_506, primals_77);  primals_77 = None
    mul_936: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_506, add_348);  add_348 = None
    sum_235: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_936, [0, 1], True);  mul_936 = None
    view_627: "f32[768]" = torch.ops.aten.reshape.default(sum_235, [768]);  sum_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    sum_236: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_935, [0, 1], True)
    view_628: "f32[768]" = torch.ops.aten.reshape.default(sum_236, [768]);  sum_236 = None
    view_629: "f32[6272, 768]" = torch.ops.aten.reshape.default(mul_935, [6272, 768]);  mul_935 = None
    permute_410: "f32[768, 6272]" = torch.ops.aten.permute.default(view_629, [1, 0])
    mm_88: "f32[768, 768]" = torch.ops.aten.mm.default(permute_410, view_356);  permute_410 = view_356 = None
    permute_411: "f32[768, 768]" = torch.ops.aten.permute.default(mm_88, [1, 0]);  mm_88 = None
    mm_89: "f32[6272, 768]" = torch.ops.aten.mm.default(view_629, permute_412);  view_629 = permute_412 = None
    view_630: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_89, [8, 784, 768]);  mm_89 = None
    permute_413: "f32[768, 768]" = torch.ops.aten.permute.default(permute_411, [1, 0]);  permute_411 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    view_631: "f32[8, 784, 16, 48]" = torch.ops.aten.reshape.default(view_630, [8, 784, 16, 48]);  view_630 = None
    permute_414: "f32[8, 16, 48, 784]" = torch.ops.aten.permute.default(view_631, [0, 2, 3, 1]);  view_631 = None
    clone_307: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(permute_414, memory_format = torch.contiguous_format);  permute_414 = None
    view_632: "f32[128, 48, 784]" = torch.ops.aten.reshape.default(clone_307, [128, 48, 784]);  clone_307 = None
    bmm_64: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(permute_415, view_632);  permute_415 = None
    bmm_65: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(view_632, permute_416);  view_632 = permute_416 = None
    view_633: "f32[8, 16, 48, 784]" = torch.ops.aten.reshape.default(bmm_64, [8, 16, 48, 784]);  bmm_64 = None
    view_634: "f32[8, 16, 48, 48]" = torch.ops.aten.reshape.default(bmm_65, [8, 16, 48, 48]);  bmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    mul_937: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_634, alias_88);  view_634 = None
    sum_237: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_937, [-1], True)
    mul_938: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(alias_88, sum_237);  alias_88 = sum_237 = None
    sub_209: "f32[8, 16, 48, 48]" = torch.ops.aten.sub.Tensor(mul_937, mul_938);  mul_937 = mul_938 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    mul_939: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(sub_209, view_351);  view_351 = None
    mul_940: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(sub_209, primals_78);  sub_209 = primals_78 = None
    sum_238: "f32[1, 16, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_939, [0, 2, 3], True);  mul_939 = None
    view_635: "f32[16, 1, 1]" = torch.ops.aten.reshape.default(sum_238, [16, 1, 1]);  sum_238 = None
    view_636: "f32[128, 48, 48]" = torch.ops.aten.reshape.default(mul_940, [128, 48, 48]);  mul_940 = None
    bmm_66: "f32[128, 784, 48]" = torch.ops.aten.bmm.default(permute_417, view_636);  permute_417 = None
    bmm_67: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_636, permute_418);  view_636 = permute_418 = None
    view_637: "f32[8, 16, 784, 48]" = torch.ops.aten.reshape.default(bmm_66, [8, 16, 784, 48]);  bmm_66 = None
    view_638: "f32[8, 16, 48, 784]" = torch.ops.aten.reshape.default(bmm_67, [8, 16, 48, 784]);  bmm_67 = None
    permute_419: "f32[8, 16, 48, 784]" = torch.ops.aten.permute.default(view_637, [0, 1, 3, 2]);  view_637 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    div_130: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_64, expand_115);  div_64 = None
    neg_8: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(permute_419)
    mul_941: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_8, div_130);  neg_8 = div_130 = None
    div_131: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(permute_419, expand_115);  permute_419 = expand_115 = None
    sum_239: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_941, [3], True);  mul_941 = None
    ge_8: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(pow_81, 1e-12)
    where_16: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_8, sum_239, full_default_20);  ge_8 = sum_239 = None
    div_132: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_218, pow_81);  getitem_218 = None
    eq_8: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(pow_81, 0);  pow_81 = None
    where_17: "f32[8, 16, 48, 784]" = torch.ops.aten.where.self(eq_8, full_default_20, div_132);  eq_8 = div_132 = None
    clone_308: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(where_17, memory_format = torch.contiguous_format);  where_17 = None
    mul_942: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_16, clone_308);  where_16 = clone_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    add_507: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_131, mul_942);  div_131 = mul_942 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    div_134: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_63, expand_114);  div_63 = None
    neg_9: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(view_638)
    mul_943: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_9, div_134);  neg_9 = div_134 = None
    div_135: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(view_638, expand_114);  view_638 = expand_114 = None
    sum_240: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_943, [3], True);  mul_943 = None
    ge_9: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(pow_79, 1e-12)
    where_18: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_9, sum_240, full_default_20);  ge_9 = sum_240 = None
    div_136: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_217, pow_79);  getitem_217 = None
    eq_9: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(pow_79, 0);  pow_79 = None
    where_19: "f32[8, 16, 48, 784]" = torch.ops.aten.where.self(eq_9, full_default_20, div_136);  eq_9 = div_136 = None
    clone_309: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(where_19, memory_format = torch.contiguous_format);  where_19 = None
    mul_944: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_18, clone_309);  where_18 = clone_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    add_508: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_135, mul_944);  div_135 = mul_944 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    cat_12: "f32[24, 16, 48, 784]" = torch.ops.aten.cat.default([add_508, add_507, view_633]);  add_508 = add_507 = view_633 = None
    view_639: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.reshape.default(cat_12, [3, 8, 16, 48, 784]);  cat_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    permute_420: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.permute.default(view_639, [1, 4, 0, 2, 3]);  view_639 = None
    clone_310: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.clone.default(permute_420, memory_format = torch.contiguous_format);  permute_420 = None
    view_640: "f32[8, 784, 2304]" = torch.ops.aten.reshape.default(clone_310, [8, 784, 2304]);  clone_310 = None
    view_641: "f32[6272, 2304]" = torch.ops.aten.reshape.default(view_640, [6272, 2304]);  view_640 = None
    mm_90: "f32[6272, 768]" = torch.ops.aten.mm.default(view_641, permute_421);  permute_421 = None
    permute_422: "f32[2304, 6272]" = torch.ops.aten.permute.default(view_641, [1, 0])
    mm_91: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_422, view_346);  permute_422 = view_346 = None
    permute_423: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_91, [1, 0]);  mm_91 = None
    sum_241: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_641, [0], True);  view_641 = None
    view_642: "f32[2304]" = torch.ops.aten.reshape.default(sum_241, [2304]);  sum_241 = None
    permute_424: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_423, [1, 0]);  permute_423 = None
    view_643: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_90, [8, 784, 768]);  mm_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_946: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_643, primals_493);  primals_493 = None
    mul_947: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_946, 768)
    sum_242: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_946, [2], True)
    mul_948: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_946, mul_470);  mul_946 = None
    sum_243: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_948, [2], True);  mul_948 = None
    mul_949: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_470, sum_243);  sum_243 = None
    sub_211: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(mul_947, sum_242);  mul_947 = sum_242 = None
    sub_212: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(sub_211, mul_949);  sub_211 = mul_949 = None
    mul_950: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(div_137, sub_212);  div_137 = sub_212 = None
    mul_951: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_643, mul_470);  mul_470 = None
    sum_244: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_951, [0, 1]);  mul_951 = None
    sum_245: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_643, [0, 1]);  view_643 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    add_509: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_506, mul_950);  add_506 = mul_950 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_952: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_509, primals_76);  primals_76 = None
    mul_953: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_509, view_345);  view_345 = None
    sum_246: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_953, [0, 1], True);  mul_953 = None
    view_644: "f32[768]" = torch.ops.aten.reshape.default(sum_246, [768]);  sum_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_645: "f32[6272, 768]" = torch.ops.aten.reshape.default(mul_952, [6272, 768]);  mul_952 = None
    mm_92: "f32[6272, 3072]" = torch.ops.aten.mm.default(view_645, permute_425);  permute_425 = None
    permute_426: "f32[768, 6272]" = torch.ops.aten.permute.default(view_645, [1, 0])
    mm_93: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_426, view_344);  permute_426 = view_344 = None
    permute_427: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_93, [1, 0]);  mm_93 = None
    sum_247: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_645, [0], True);  view_645 = None
    view_646: "f32[768]" = torch.ops.aten.reshape.default(sum_247, [768]);  sum_247 = None
    permute_428: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_427, [1, 0]);  permute_427 = None
    view_647: "f32[8, 784, 3072]" = torch.ops.aten.reshape.default(mm_92, [8, 784, 3072]);  mm_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_955: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(add_344, 0.5);  add_344 = None
    mul_956: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_343, view_343)
    mul_957: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(mul_956, -0.5);  mul_956 = None
    exp_36: "f32[8, 784, 3072]" = torch.ops.aten.exp.default(mul_957);  mul_957 = None
    mul_958: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(exp_36, 0.3989422804014327);  exp_36 = None
    mul_959: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_343, mul_958);  view_343 = mul_958 = None
    add_511: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(mul_955, mul_959);  mul_955 = mul_959 = None
    mul_960: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_647, add_511);  view_647 = add_511 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_648: "f32[6272, 3072]" = torch.ops.aten.reshape.default(mul_960, [6272, 3072]);  mul_960 = None
    mm_94: "f32[6272, 768]" = torch.ops.aten.mm.default(view_648, permute_429);  permute_429 = None
    permute_430: "f32[3072, 6272]" = torch.ops.aten.permute.default(view_648, [1, 0])
    mm_95: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_430, view_342);  permute_430 = view_342 = None
    permute_431: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_95, [1, 0]);  mm_95 = None
    sum_248: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_648, [0], True);  view_648 = None
    view_649: "f32[3072]" = torch.ops.aten.reshape.default(sum_248, [3072]);  sum_248 = None
    permute_432: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_431, [1, 0]);  permute_431 = None
    view_650: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_94, [8, 784, 768]);  mm_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_962: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_650, primals_487);  primals_487 = None
    mul_963: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_962, 768)
    sum_249: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_962, [2], True)
    mul_964: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_962, mul_464);  mul_962 = None
    sum_250: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_964, [2], True);  mul_964 = None
    mul_965: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_464, sum_250);  sum_250 = None
    sub_214: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(mul_963, sum_249);  mul_963 = sum_249 = None
    sub_215: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(sub_214, mul_965);  sub_214 = mul_965 = None
    mul_966: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(div_138, sub_215);  div_138 = sub_215 = None
    mul_967: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_650, mul_464);  mul_464 = None
    sum_251: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_967, [0, 1]);  mul_967 = None
    sum_252: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_650, [0, 1]);  view_650 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    add_512: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_509, mul_966);  add_509 = mul_966 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_968: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_512, primals_75);  primals_75 = None
    mul_969: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_512, permute_171);  permute_171 = None
    sum_253: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_969, [0, 1], True);  mul_969 = None
    view_651: "f32[768]" = torch.ops.aten.reshape.default(sum_253, [768]);  sum_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    permute_433: "f32[8, 768, 784]" = torch.ops.aten.permute.default(mul_968, [0, 2, 1]);  mul_968 = None
    view_652: "f32[8, 768, 28, 28]" = torch.ops.aten.reshape.default(permute_433, [8, 768, 28, 28]);  permute_433 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    sum_254: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_652, [0, 2, 3])
    convolution_backward_10 = torch.ops.aten.convolution_backward.default(view_652, add_340, primals_485, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False]);  view_652 = add_340 = primals_485 = None
    getitem_326: "f32[8, 768, 28, 28]" = convolution_backward_10[0]
    getitem_327: "f32[768, 1, 3, 3]" = convolution_backward_10[1];  convolution_backward_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    sum_255: "f32[768]" = torch.ops.aten.sum.dim_IntList(getitem_326, [0, 2, 3])
    sub_216: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(mul_455, unsqueeze_179);  mul_455 = unsqueeze_179 = None
    mul_970: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_326, sub_216)
    sum_256: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_970, [0, 2, 3]);  mul_970 = None
    mul_971: "f32[768]" = torch.ops.aten.mul.Tensor(sum_255, 0.00015943877551020407)
    unsqueeze_180: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_971, 0);  mul_971 = None
    unsqueeze_181: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_180, 2);  unsqueeze_180 = None
    unsqueeze_182: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_181, 3);  unsqueeze_181 = None
    mul_972: "f32[768]" = torch.ops.aten.mul.Tensor(sum_256, 0.00015943877551020407)
    mul_973: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_64, squeeze_64)
    mul_974: "f32[768]" = torch.ops.aten.mul.Tensor(mul_972, mul_973);  mul_972 = mul_973 = None
    unsqueeze_183: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_974, 0);  mul_974 = None
    unsqueeze_184: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_183, 2);  unsqueeze_183 = None
    unsqueeze_185: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_184, 3);  unsqueeze_184 = None
    mul_975: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_64, primals_483);  primals_483 = None
    unsqueeze_186: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_975, 0);  mul_975 = None
    unsqueeze_187: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_186, 2);  unsqueeze_186 = None
    unsqueeze_188: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_187, 3);  unsqueeze_187 = None
    mul_976: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_216, unsqueeze_185);  sub_216 = unsqueeze_185 = None
    sub_218: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(getitem_326, mul_976);  getitem_326 = mul_976 = None
    sub_219: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(sub_218, unsqueeze_182);  sub_218 = unsqueeze_182 = None
    mul_977: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_219, unsqueeze_188);  sub_219 = unsqueeze_188 = None
    mul_978: "f32[768]" = torch.ops.aten.mul.Tensor(sum_256, squeeze_64);  sum_256 = squeeze_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_980: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(add_335, 0.5);  add_335 = None
    mul_981: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_40, convolution_40)
    mul_982: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_981, -0.5);  mul_981 = None
    exp_37: "f32[8, 768, 28, 28]" = torch.ops.aten.exp.default(mul_982);  mul_982 = None
    mul_983: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(exp_37, 0.3989422804014327);  exp_37 = None
    mul_984: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_40, mul_983);  convolution_40 = mul_983 = None
    add_514: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(mul_980, mul_984);  mul_980 = mul_984 = None
    mul_985: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_977, add_514);  mul_977 = add_514 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    sum_257: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_985, [0, 2, 3])
    convolution_backward_11 = torch.ops.aten.convolution_backward.default(mul_985, view_340, primals_481, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False]);  mul_985 = view_340 = primals_481 = None
    getitem_329: "f32[8, 768, 28, 28]" = convolution_backward_11[0]
    getitem_330: "f32[768, 1, 3, 3]" = convolution_backward_11[1];  convolution_backward_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    view_653: "f32[8, 768, 784]" = torch.ops.aten.reshape.default(getitem_329, [8, 768, 784]);  getitem_329 = None
    permute_434: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_653, [0, 2, 1]);  view_653 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    clone_313: "f32[8, 784, 768]" = torch.ops.aten.clone.default(permute_434, memory_format = torch.contiguous_format);  permute_434 = None
    mul_987: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(clone_313, primals_479);  primals_479 = None
    mul_988: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_987, 768)
    sum_258: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_987, [2], True)
    mul_989: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_987, mul_451);  mul_987 = None
    sum_259: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_989, [2], True);  mul_989 = None
    mul_990: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_451, sum_259);  sum_259 = None
    sub_221: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(mul_988, sum_258);  mul_988 = sum_258 = None
    sub_222: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(sub_221, mul_990);  sub_221 = mul_990 = None
    mul_991: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(div_139, sub_222);  div_139 = sub_222 = None
    mul_992: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(clone_313, mul_451);  mul_451 = None
    sum_260: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_992, [0, 1]);  mul_992 = None
    sum_261: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_313, [0, 1]);  clone_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    add_515: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_512, mul_991);  add_512 = mul_991 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_993: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_515, primals_73);  primals_73 = None
    mul_994: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_515, add_331);  add_331 = None
    sum_262: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_994, [0, 1], True);  mul_994 = None
    view_654: "f32[768]" = torch.ops.aten.reshape.default(sum_262, [768]);  sum_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    sum_263: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_993, [0, 1], True)
    view_655: "f32[768]" = torch.ops.aten.reshape.default(sum_263, [768]);  sum_263 = None
    view_656: "f32[6272, 768]" = torch.ops.aten.reshape.default(mul_993, [6272, 768]);  mul_993 = None
    permute_435: "f32[768, 6272]" = torch.ops.aten.permute.default(view_656, [1, 0])
    mm_96: "f32[768, 768]" = torch.ops.aten.mm.default(permute_435, view_338);  permute_435 = view_338 = None
    permute_436: "f32[768, 768]" = torch.ops.aten.permute.default(mm_96, [1, 0]);  mm_96 = None
    mm_97: "f32[6272, 768]" = torch.ops.aten.mm.default(view_656, permute_437);  view_656 = permute_437 = None
    view_657: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_97, [8, 784, 768]);  mm_97 = None
    permute_438: "f32[768, 768]" = torch.ops.aten.permute.default(permute_436, [1, 0]);  permute_436 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    view_658: "f32[8, 784, 16, 48]" = torch.ops.aten.reshape.default(view_657, [8, 784, 16, 48]);  view_657 = None
    permute_439: "f32[8, 16, 48, 784]" = torch.ops.aten.permute.default(view_658, [0, 2, 3, 1]);  view_658 = None
    clone_315: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(permute_439, memory_format = torch.contiguous_format);  permute_439 = None
    view_659: "f32[128, 48, 784]" = torch.ops.aten.reshape.default(clone_315, [128, 48, 784]);  clone_315 = None
    bmm_68: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(permute_440, view_659);  permute_440 = None
    bmm_69: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(view_659, permute_441);  view_659 = permute_441 = None
    view_660: "f32[8, 16, 48, 784]" = torch.ops.aten.reshape.default(bmm_68, [8, 16, 48, 784]);  bmm_68 = None
    view_661: "f32[8, 16, 48, 48]" = torch.ops.aten.reshape.default(bmm_69, [8, 16, 48, 48]);  bmm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    mul_995: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_661, alias_91);  view_661 = None
    sum_264: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_995, [-1], True)
    mul_996: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(alias_91, sum_264);  alias_91 = sum_264 = None
    sub_223: "f32[8, 16, 48, 48]" = torch.ops.aten.sub.Tensor(mul_995, mul_996);  mul_995 = mul_996 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    mul_997: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(sub_223, view_333);  view_333 = None
    mul_998: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(sub_223, primals_74);  sub_223 = primals_74 = None
    sum_265: "f32[1, 16, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_997, [0, 2, 3], True);  mul_997 = None
    view_662: "f32[16, 1, 1]" = torch.ops.aten.reshape.default(sum_265, [16, 1, 1]);  sum_265 = None
    view_663: "f32[128, 48, 48]" = torch.ops.aten.reshape.default(mul_998, [128, 48, 48]);  mul_998 = None
    bmm_70: "f32[128, 784, 48]" = torch.ops.aten.bmm.default(permute_442, view_663);  permute_442 = None
    bmm_71: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_663, permute_443);  view_663 = permute_443 = None
    view_664: "f32[8, 16, 784, 48]" = torch.ops.aten.reshape.default(bmm_70, [8, 16, 784, 48]);  bmm_70 = None
    view_665: "f32[8, 16, 48, 784]" = torch.ops.aten.reshape.default(bmm_71, [8, 16, 48, 784]);  bmm_71 = None
    permute_444: "f32[8, 16, 48, 784]" = torch.ops.aten.permute.default(view_664, [0, 1, 3, 2]);  view_664 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    div_141: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_61, expand_109);  div_61 = None
    neg_10: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(permute_444)
    mul_999: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_10, div_141);  neg_10 = div_141 = None
    div_142: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(permute_444, expand_109);  permute_444 = expand_109 = None
    sum_266: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_999, [3], True);  mul_999 = None
    ge_10: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(pow_77, 1e-12)
    where_20: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_10, sum_266, full_default_20);  ge_10 = sum_266 = None
    div_143: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_207, pow_77);  getitem_207 = None
    eq_10: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(pow_77, 0);  pow_77 = None
    where_21: "f32[8, 16, 48, 784]" = torch.ops.aten.where.self(eq_10, full_default_20, div_143);  eq_10 = div_143 = None
    clone_316: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(where_21, memory_format = torch.contiguous_format);  where_21 = None
    mul_1000: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_20, clone_316);  where_20 = clone_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    add_516: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_142, mul_1000);  div_142 = mul_1000 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    div_145: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_60, expand_108);  div_60 = None
    neg_11: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(view_665)
    mul_1001: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_11, div_145);  neg_11 = div_145 = None
    div_146: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(view_665, expand_108);  view_665 = expand_108 = None
    sum_267: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_1001, [3], True);  mul_1001 = None
    ge_11: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(pow_75, 1e-12)
    where_22: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_11, sum_267, full_default_20);  ge_11 = sum_267 = None
    div_147: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_206, pow_75);  getitem_206 = None
    eq_11: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(pow_75, 0);  pow_75 = None
    where_23: "f32[8, 16, 48, 784]" = torch.ops.aten.where.self(eq_11, full_default_20, div_147);  eq_11 = div_147 = None
    clone_317: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(where_23, memory_format = torch.contiguous_format);  where_23 = None
    mul_1002: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_22, clone_317);  where_22 = clone_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    add_517: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_146, mul_1002);  div_146 = mul_1002 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    cat_13: "f32[24, 16, 48, 784]" = torch.ops.aten.cat.default([add_517, add_516, view_660]);  add_517 = add_516 = view_660 = None
    view_666: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.reshape.default(cat_13, [3, 8, 16, 48, 784]);  cat_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    permute_445: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.permute.default(view_666, [1, 4, 0, 2, 3]);  view_666 = None
    clone_318: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.clone.default(permute_445, memory_format = torch.contiguous_format);  permute_445 = None
    view_667: "f32[8, 784, 2304]" = torch.ops.aten.reshape.default(clone_318, [8, 784, 2304]);  clone_318 = None
    view_668: "f32[6272, 2304]" = torch.ops.aten.reshape.default(view_667, [6272, 2304]);  view_667 = None
    mm_98: "f32[6272, 768]" = torch.ops.aten.mm.default(view_668, permute_446);  permute_446 = None
    permute_447: "f32[2304, 6272]" = torch.ops.aten.permute.default(view_668, [1, 0])
    mm_99: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_447, view_328);  permute_447 = view_328 = None
    permute_448: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_99, [1, 0]);  mm_99 = None
    sum_268: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_668, [0], True);  view_668 = None
    view_669: "f32[2304]" = torch.ops.aten.reshape.default(sum_268, [2304]);  sum_268 = None
    permute_449: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_448, [1, 0]);  permute_448 = None
    view_670: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_98, [8, 784, 768]);  mm_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_1004: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_670, primals_473);  primals_473 = None
    mul_1005: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1004, 768)
    sum_269: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1004, [2], True)
    mul_1006: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1004, mul_447);  mul_1004 = None
    sum_270: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1006, [2], True);  mul_1006 = None
    mul_1007: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_447, sum_270);  sum_270 = None
    sub_225: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(mul_1005, sum_269);  mul_1005 = sum_269 = None
    sub_226: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(sub_225, mul_1007);  sub_225 = mul_1007 = None
    mul_1008: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(div_148, sub_226);  div_148 = sub_226 = None
    mul_1009: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_670, mul_447);  mul_447 = None
    sum_271: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1009, [0, 1]);  mul_1009 = None
    sum_272: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_670, [0, 1]);  view_670 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    add_518: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_515, mul_1008);  add_515 = mul_1008 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_1010: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_518, primals_72);  primals_72 = None
    mul_1011: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_518, view_327);  view_327 = None
    sum_273: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1011, [0, 1], True);  mul_1011 = None
    view_671: "f32[768]" = torch.ops.aten.reshape.default(sum_273, [768]);  sum_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_672: "f32[6272, 768]" = torch.ops.aten.reshape.default(mul_1010, [6272, 768]);  mul_1010 = None
    mm_100: "f32[6272, 3072]" = torch.ops.aten.mm.default(view_672, permute_450);  permute_450 = None
    permute_451: "f32[768, 6272]" = torch.ops.aten.permute.default(view_672, [1, 0])
    mm_101: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_451, view_326);  permute_451 = view_326 = None
    permute_452: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_101, [1, 0]);  mm_101 = None
    sum_274: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_672, [0], True);  view_672 = None
    view_673: "f32[768]" = torch.ops.aten.reshape.default(sum_274, [768]);  sum_274 = None
    permute_453: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_452, [1, 0]);  permute_452 = None
    view_674: "f32[8, 784, 3072]" = torch.ops.aten.reshape.default(mm_100, [8, 784, 3072]);  mm_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_1013: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(add_327, 0.5);  add_327 = None
    mul_1014: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_325, view_325)
    mul_1015: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(mul_1014, -0.5);  mul_1014 = None
    exp_38: "f32[8, 784, 3072]" = torch.ops.aten.exp.default(mul_1015);  mul_1015 = None
    mul_1016: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(exp_38, 0.3989422804014327);  exp_38 = None
    mul_1017: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_325, mul_1016);  view_325 = mul_1016 = None
    add_520: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(mul_1013, mul_1017);  mul_1013 = mul_1017 = None
    mul_1018: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_674, add_520);  view_674 = add_520 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_675: "f32[6272, 3072]" = torch.ops.aten.reshape.default(mul_1018, [6272, 3072]);  mul_1018 = None
    mm_102: "f32[6272, 768]" = torch.ops.aten.mm.default(view_675, permute_454);  permute_454 = None
    permute_455: "f32[3072, 6272]" = torch.ops.aten.permute.default(view_675, [1, 0])
    mm_103: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_455, view_324);  permute_455 = view_324 = None
    permute_456: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_103, [1, 0]);  mm_103 = None
    sum_275: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_675, [0], True);  view_675 = None
    view_676: "f32[3072]" = torch.ops.aten.reshape.default(sum_275, [3072]);  sum_275 = None
    permute_457: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_456, [1, 0]);  permute_456 = None
    view_677: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_102, [8, 784, 768]);  mm_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_1020: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_677, primals_467);  primals_467 = None
    mul_1021: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1020, 768)
    sum_276: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1020, [2], True)
    mul_1022: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1020, mul_441);  mul_1020 = None
    sum_277: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1022, [2], True);  mul_1022 = None
    mul_1023: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_441, sum_277);  sum_277 = None
    sub_228: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(mul_1021, sum_276);  mul_1021 = sum_276 = None
    sub_229: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(sub_228, mul_1023);  sub_228 = mul_1023 = None
    mul_1024: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(div_149, sub_229);  div_149 = sub_229 = None
    mul_1025: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_677, mul_441);  mul_441 = None
    sum_278: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1025, [0, 1]);  mul_1025 = None
    sum_279: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_677, [0, 1]);  view_677 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    add_521: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_518, mul_1024);  add_518 = mul_1024 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_1026: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_521, primals_71);  primals_71 = None
    mul_1027: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_521, permute_162);  permute_162 = None
    sum_280: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1027, [0, 1], True);  mul_1027 = None
    view_678: "f32[768]" = torch.ops.aten.reshape.default(sum_280, [768]);  sum_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    permute_458: "f32[8, 768, 784]" = torch.ops.aten.permute.default(mul_1026, [0, 2, 1]);  mul_1026 = None
    view_679: "f32[8, 768, 28, 28]" = torch.ops.aten.reshape.default(permute_458, [8, 768, 28, 28]);  permute_458 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    sum_281: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_679, [0, 2, 3])
    convolution_backward_12 = torch.ops.aten.convolution_backward.default(view_679, add_323, primals_465, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False]);  view_679 = add_323 = primals_465 = None
    getitem_332: "f32[8, 768, 28, 28]" = convolution_backward_12[0]
    getitem_333: "f32[768, 1, 3, 3]" = convolution_backward_12[1];  convolution_backward_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    sum_282: "f32[768]" = torch.ops.aten.sum.dim_IntList(getitem_332, [0, 2, 3])
    sub_230: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(mul_432, unsqueeze_191);  mul_432 = unsqueeze_191 = None
    mul_1028: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_332, sub_230)
    sum_283: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1028, [0, 2, 3]);  mul_1028 = None
    mul_1029: "f32[768]" = torch.ops.aten.mul.Tensor(sum_282, 0.00015943877551020407)
    unsqueeze_192: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1029, 0);  mul_1029 = None
    unsqueeze_193: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_192, 2);  unsqueeze_192 = None
    unsqueeze_194: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_193, 3);  unsqueeze_193 = None
    mul_1030: "f32[768]" = torch.ops.aten.mul.Tensor(sum_283, 0.00015943877551020407)
    mul_1031: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_61, squeeze_61)
    mul_1032: "f32[768]" = torch.ops.aten.mul.Tensor(mul_1030, mul_1031);  mul_1030 = mul_1031 = None
    unsqueeze_195: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1032, 0);  mul_1032 = None
    unsqueeze_196: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_195, 2);  unsqueeze_195 = None
    unsqueeze_197: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_196, 3);  unsqueeze_196 = None
    mul_1033: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_61, primals_463);  primals_463 = None
    unsqueeze_198: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1033, 0);  mul_1033 = None
    unsqueeze_199: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_198, 2);  unsqueeze_198 = None
    unsqueeze_200: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_199, 3);  unsqueeze_199 = None
    mul_1034: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_230, unsqueeze_197);  sub_230 = unsqueeze_197 = None
    sub_232: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(getitem_332, mul_1034);  getitem_332 = mul_1034 = None
    sub_233: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(sub_232, unsqueeze_194);  sub_232 = unsqueeze_194 = None
    mul_1035: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_233, unsqueeze_200);  sub_233 = unsqueeze_200 = None
    mul_1036: "f32[768]" = torch.ops.aten.mul.Tensor(sum_283, squeeze_61);  sum_283 = squeeze_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_1038: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(add_318, 0.5);  add_318 = None
    mul_1039: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_38, convolution_38)
    mul_1040: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_1039, -0.5);  mul_1039 = None
    exp_39: "f32[8, 768, 28, 28]" = torch.ops.aten.exp.default(mul_1040);  mul_1040 = None
    mul_1041: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(exp_39, 0.3989422804014327);  exp_39 = None
    mul_1042: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_38, mul_1041);  convolution_38 = mul_1041 = None
    add_523: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(mul_1038, mul_1042);  mul_1038 = mul_1042 = None
    mul_1043: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_1035, add_523);  mul_1035 = add_523 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    sum_284: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1043, [0, 2, 3])
    convolution_backward_13 = torch.ops.aten.convolution_backward.default(mul_1043, view_322, primals_461, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False]);  mul_1043 = view_322 = primals_461 = None
    getitem_335: "f32[8, 768, 28, 28]" = convolution_backward_13[0]
    getitem_336: "f32[768, 1, 3, 3]" = convolution_backward_13[1];  convolution_backward_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    view_680: "f32[8, 768, 784]" = torch.ops.aten.reshape.default(getitem_335, [8, 768, 784]);  getitem_335 = None
    permute_459: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_680, [0, 2, 1]);  view_680 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    clone_321: "f32[8, 784, 768]" = torch.ops.aten.clone.default(permute_459, memory_format = torch.contiguous_format);  permute_459 = None
    mul_1045: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(clone_321, primals_459);  primals_459 = None
    mul_1046: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1045, 768)
    sum_285: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1045, [2], True)
    mul_1047: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1045, mul_428);  mul_1045 = None
    sum_286: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1047, [2], True);  mul_1047 = None
    mul_1048: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_428, sum_286);  sum_286 = None
    sub_235: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(mul_1046, sum_285);  mul_1046 = sum_285 = None
    sub_236: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(sub_235, mul_1048);  sub_235 = mul_1048 = None
    mul_1049: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(div_150, sub_236);  div_150 = sub_236 = None
    mul_1050: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(clone_321, mul_428);  mul_428 = None
    sum_287: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1050, [0, 1]);  mul_1050 = None
    sum_288: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_321, [0, 1]);  clone_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    add_524: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_521, mul_1049);  add_521 = mul_1049 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_1051: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_524, primals_69);  primals_69 = None
    mul_1052: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_524, add_314);  add_314 = None
    sum_289: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1052, [0, 1], True);  mul_1052 = None
    view_681: "f32[768]" = torch.ops.aten.reshape.default(sum_289, [768]);  sum_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    sum_290: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1051, [0, 1], True)
    view_682: "f32[768]" = torch.ops.aten.reshape.default(sum_290, [768]);  sum_290 = None
    view_683: "f32[6272, 768]" = torch.ops.aten.reshape.default(mul_1051, [6272, 768]);  mul_1051 = None
    permute_460: "f32[768, 6272]" = torch.ops.aten.permute.default(view_683, [1, 0])
    mm_104: "f32[768, 768]" = torch.ops.aten.mm.default(permute_460, view_320);  permute_460 = view_320 = None
    permute_461: "f32[768, 768]" = torch.ops.aten.permute.default(mm_104, [1, 0]);  mm_104 = None
    mm_105: "f32[6272, 768]" = torch.ops.aten.mm.default(view_683, permute_462);  view_683 = permute_462 = None
    view_684: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_105, [8, 784, 768]);  mm_105 = None
    permute_463: "f32[768, 768]" = torch.ops.aten.permute.default(permute_461, [1, 0]);  permute_461 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    view_685: "f32[8, 784, 16, 48]" = torch.ops.aten.reshape.default(view_684, [8, 784, 16, 48]);  view_684 = None
    permute_464: "f32[8, 16, 48, 784]" = torch.ops.aten.permute.default(view_685, [0, 2, 3, 1]);  view_685 = None
    clone_323: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(permute_464, memory_format = torch.contiguous_format);  permute_464 = None
    view_686: "f32[128, 48, 784]" = torch.ops.aten.reshape.default(clone_323, [128, 48, 784]);  clone_323 = None
    bmm_72: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(permute_465, view_686);  permute_465 = None
    bmm_73: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(view_686, permute_466);  view_686 = permute_466 = None
    view_687: "f32[8, 16, 48, 784]" = torch.ops.aten.reshape.default(bmm_72, [8, 16, 48, 784]);  bmm_72 = None
    view_688: "f32[8, 16, 48, 48]" = torch.ops.aten.reshape.default(bmm_73, [8, 16, 48, 48]);  bmm_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    mul_1053: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_688, alias_94);  view_688 = None
    sum_291: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_1053, [-1], True)
    mul_1054: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(alias_94, sum_291);  alias_94 = sum_291 = None
    sub_237: "f32[8, 16, 48, 48]" = torch.ops.aten.sub.Tensor(mul_1053, mul_1054);  mul_1053 = mul_1054 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    mul_1055: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(sub_237, view_315);  view_315 = None
    mul_1056: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(sub_237, primals_70);  sub_237 = primals_70 = None
    sum_292: "f32[1, 16, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1055, [0, 2, 3], True);  mul_1055 = None
    view_689: "f32[16, 1, 1]" = torch.ops.aten.reshape.default(sum_292, [16, 1, 1]);  sum_292 = None
    view_690: "f32[128, 48, 48]" = torch.ops.aten.reshape.default(mul_1056, [128, 48, 48]);  mul_1056 = None
    bmm_74: "f32[128, 784, 48]" = torch.ops.aten.bmm.default(permute_467, view_690);  permute_467 = None
    bmm_75: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_690, permute_468);  view_690 = permute_468 = None
    view_691: "f32[8, 16, 784, 48]" = torch.ops.aten.reshape.default(bmm_74, [8, 16, 784, 48]);  bmm_74 = None
    view_692: "f32[8, 16, 48, 784]" = torch.ops.aten.reshape.default(bmm_75, [8, 16, 48, 784]);  bmm_75 = None
    permute_469: "f32[8, 16, 48, 784]" = torch.ops.aten.permute.default(view_691, [0, 1, 3, 2]);  view_691 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    div_152: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_58, expand_103);  div_58 = None
    neg_12: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(permute_469)
    mul_1057: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_12, div_152);  neg_12 = div_152 = None
    div_153: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(permute_469, expand_103);  permute_469 = expand_103 = None
    sum_293: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_1057, [3], True);  mul_1057 = None
    ge_12: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(pow_73, 1e-12)
    where_24: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_12, sum_293, full_default_20);  ge_12 = sum_293 = None
    div_154: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_196, pow_73);  getitem_196 = None
    eq_12: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(pow_73, 0);  pow_73 = None
    where_25: "f32[8, 16, 48, 784]" = torch.ops.aten.where.self(eq_12, full_default_20, div_154);  eq_12 = div_154 = None
    clone_324: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(where_25, memory_format = torch.contiguous_format);  where_25 = None
    mul_1058: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_24, clone_324);  where_24 = clone_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    add_525: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_153, mul_1058);  div_153 = mul_1058 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    div_156: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_57, expand_102);  div_57 = None
    neg_13: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(view_692)
    mul_1059: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_13, div_156);  neg_13 = div_156 = None
    div_157: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(view_692, expand_102);  view_692 = expand_102 = None
    sum_294: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_1059, [3], True);  mul_1059 = None
    ge_13: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(pow_71, 1e-12)
    where_26: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_13, sum_294, full_default_20);  ge_13 = sum_294 = None
    div_158: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_195, pow_71);  getitem_195 = None
    eq_13: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(pow_71, 0);  pow_71 = None
    where_27: "f32[8, 16, 48, 784]" = torch.ops.aten.where.self(eq_13, full_default_20, div_158);  eq_13 = div_158 = None
    clone_325: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(where_27, memory_format = torch.contiguous_format);  where_27 = None
    mul_1060: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_26, clone_325);  where_26 = clone_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    add_526: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_157, mul_1060);  div_157 = mul_1060 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    cat_14: "f32[24, 16, 48, 784]" = torch.ops.aten.cat.default([add_526, add_525, view_687]);  add_526 = add_525 = view_687 = None
    view_693: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.reshape.default(cat_14, [3, 8, 16, 48, 784]);  cat_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    permute_470: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.permute.default(view_693, [1, 4, 0, 2, 3]);  view_693 = None
    clone_326: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.clone.default(permute_470, memory_format = torch.contiguous_format);  permute_470 = None
    view_694: "f32[8, 784, 2304]" = torch.ops.aten.reshape.default(clone_326, [8, 784, 2304]);  clone_326 = None
    view_695: "f32[6272, 2304]" = torch.ops.aten.reshape.default(view_694, [6272, 2304]);  view_694 = None
    mm_106: "f32[6272, 768]" = torch.ops.aten.mm.default(view_695, permute_471);  permute_471 = None
    permute_472: "f32[2304, 6272]" = torch.ops.aten.permute.default(view_695, [1, 0])
    mm_107: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_472, view_310);  permute_472 = view_310 = None
    permute_473: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_107, [1, 0]);  mm_107 = None
    sum_295: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_695, [0], True);  view_695 = None
    view_696: "f32[2304]" = torch.ops.aten.reshape.default(sum_295, [2304]);  sum_295 = None
    permute_474: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_473, [1, 0]);  permute_473 = None
    view_697: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_106, [8, 784, 768]);  mm_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_1062: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_697, primals_453);  primals_453 = None
    mul_1063: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1062, 768)
    sum_296: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1062, [2], True)
    mul_1064: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1062, mul_424);  mul_1062 = None
    sum_297: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1064, [2], True);  mul_1064 = None
    mul_1065: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_424, sum_297);  sum_297 = None
    sub_239: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(mul_1063, sum_296);  mul_1063 = sum_296 = None
    sub_240: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(sub_239, mul_1065);  sub_239 = mul_1065 = None
    mul_1066: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(div_159, sub_240);  div_159 = sub_240 = None
    mul_1067: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_697, mul_424);  mul_424 = None
    sum_298: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1067, [0, 1]);  mul_1067 = None
    sum_299: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_697, [0, 1]);  view_697 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    add_527: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_524, mul_1066);  add_524 = mul_1066 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_1068: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_527, primals_68);  primals_68 = None
    mul_1069: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_527, view_309);  view_309 = None
    sum_300: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1069, [0, 1], True);  mul_1069 = None
    view_698: "f32[768]" = torch.ops.aten.reshape.default(sum_300, [768]);  sum_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_699: "f32[6272, 768]" = torch.ops.aten.reshape.default(mul_1068, [6272, 768]);  mul_1068 = None
    mm_108: "f32[6272, 3072]" = torch.ops.aten.mm.default(view_699, permute_475);  permute_475 = None
    permute_476: "f32[768, 6272]" = torch.ops.aten.permute.default(view_699, [1, 0])
    mm_109: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_476, view_308);  permute_476 = view_308 = None
    permute_477: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_109, [1, 0]);  mm_109 = None
    sum_301: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_699, [0], True);  view_699 = None
    view_700: "f32[768]" = torch.ops.aten.reshape.default(sum_301, [768]);  sum_301 = None
    permute_478: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_477, [1, 0]);  permute_477 = None
    view_701: "f32[8, 784, 3072]" = torch.ops.aten.reshape.default(mm_108, [8, 784, 3072]);  mm_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_1071: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(add_310, 0.5);  add_310 = None
    mul_1072: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_307, view_307)
    mul_1073: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(mul_1072, -0.5);  mul_1072 = None
    exp_40: "f32[8, 784, 3072]" = torch.ops.aten.exp.default(mul_1073);  mul_1073 = None
    mul_1074: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(exp_40, 0.3989422804014327);  exp_40 = None
    mul_1075: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_307, mul_1074);  view_307 = mul_1074 = None
    add_529: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(mul_1071, mul_1075);  mul_1071 = mul_1075 = None
    mul_1076: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_701, add_529);  view_701 = add_529 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_702: "f32[6272, 3072]" = torch.ops.aten.reshape.default(mul_1076, [6272, 3072]);  mul_1076 = None
    mm_110: "f32[6272, 768]" = torch.ops.aten.mm.default(view_702, permute_479);  permute_479 = None
    permute_480: "f32[3072, 6272]" = torch.ops.aten.permute.default(view_702, [1, 0])
    mm_111: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_480, view_306);  permute_480 = view_306 = None
    permute_481: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_111, [1, 0]);  mm_111 = None
    sum_302: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_702, [0], True);  view_702 = None
    view_703: "f32[3072]" = torch.ops.aten.reshape.default(sum_302, [3072]);  sum_302 = None
    permute_482: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_481, [1, 0]);  permute_481 = None
    view_704: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_110, [8, 784, 768]);  mm_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_1078: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_704, primals_447);  primals_447 = None
    mul_1079: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1078, 768)
    sum_303: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1078, [2], True)
    mul_1080: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1078, mul_418);  mul_1078 = None
    sum_304: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1080, [2], True);  mul_1080 = None
    mul_1081: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_418, sum_304);  sum_304 = None
    sub_242: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(mul_1079, sum_303);  mul_1079 = sum_303 = None
    sub_243: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(sub_242, mul_1081);  sub_242 = mul_1081 = None
    mul_1082: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(div_160, sub_243);  div_160 = sub_243 = None
    mul_1083: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_704, mul_418);  mul_418 = None
    sum_305: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1083, [0, 1]);  mul_1083 = None
    sum_306: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_704, [0, 1]);  view_704 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    add_530: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_527, mul_1082);  add_527 = mul_1082 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_1084: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_530, primals_67);  primals_67 = None
    mul_1085: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_530, permute_153);  permute_153 = None
    sum_307: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1085, [0, 1], True);  mul_1085 = None
    view_705: "f32[768]" = torch.ops.aten.reshape.default(sum_307, [768]);  sum_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    permute_483: "f32[8, 768, 784]" = torch.ops.aten.permute.default(mul_1084, [0, 2, 1]);  mul_1084 = None
    view_706: "f32[8, 768, 28, 28]" = torch.ops.aten.reshape.default(permute_483, [8, 768, 28, 28]);  permute_483 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    sum_308: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_706, [0, 2, 3])
    convolution_backward_14 = torch.ops.aten.convolution_backward.default(view_706, add_306, primals_445, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False]);  view_706 = add_306 = primals_445 = None
    getitem_338: "f32[8, 768, 28, 28]" = convolution_backward_14[0]
    getitem_339: "f32[768, 1, 3, 3]" = convolution_backward_14[1];  convolution_backward_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    sum_309: "f32[768]" = torch.ops.aten.sum.dim_IntList(getitem_338, [0, 2, 3])
    sub_244: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(mul_409, unsqueeze_203);  mul_409 = unsqueeze_203 = None
    mul_1086: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_338, sub_244)
    sum_310: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1086, [0, 2, 3]);  mul_1086 = None
    mul_1087: "f32[768]" = torch.ops.aten.mul.Tensor(sum_309, 0.00015943877551020407)
    unsqueeze_204: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1087, 0);  mul_1087 = None
    unsqueeze_205: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_204, 2);  unsqueeze_204 = None
    unsqueeze_206: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_205, 3);  unsqueeze_205 = None
    mul_1088: "f32[768]" = torch.ops.aten.mul.Tensor(sum_310, 0.00015943877551020407)
    mul_1089: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_58, squeeze_58)
    mul_1090: "f32[768]" = torch.ops.aten.mul.Tensor(mul_1088, mul_1089);  mul_1088 = mul_1089 = None
    unsqueeze_207: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1090, 0);  mul_1090 = None
    unsqueeze_208: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_207, 2);  unsqueeze_207 = None
    unsqueeze_209: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_208, 3);  unsqueeze_208 = None
    mul_1091: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_58, primals_443);  primals_443 = None
    unsqueeze_210: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1091, 0);  mul_1091 = None
    unsqueeze_211: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_210, 2);  unsqueeze_210 = None
    unsqueeze_212: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_211, 3);  unsqueeze_211 = None
    mul_1092: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_244, unsqueeze_209);  sub_244 = unsqueeze_209 = None
    sub_246: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(getitem_338, mul_1092);  getitem_338 = mul_1092 = None
    sub_247: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(sub_246, unsqueeze_206);  sub_246 = unsqueeze_206 = None
    mul_1093: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_247, unsqueeze_212);  sub_247 = unsqueeze_212 = None
    mul_1094: "f32[768]" = torch.ops.aten.mul.Tensor(sum_310, squeeze_58);  sum_310 = squeeze_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_1096: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(add_301, 0.5);  add_301 = None
    mul_1097: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_36, convolution_36)
    mul_1098: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_1097, -0.5);  mul_1097 = None
    exp_41: "f32[8, 768, 28, 28]" = torch.ops.aten.exp.default(mul_1098);  mul_1098 = None
    mul_1099: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(exp_41, 0.3989422804014327);  exp_41 = None
    mul_1100: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_36, mul_1099);  convolution_36 = mul_1099 = None
    add_532: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(mul_1096, mul_1100);  mul_1096 = mul_1100 = None
    mul_1101: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_1093, add_532);  mul_1093 = add_532 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    sum_311: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1101, [0, 2, 3])
    convolution_backward_15 = torch.ops.aten.convolution_backward.default(mul_1101, view_304, primals_441, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False]);  mul_1101 = view_304 = primals_441 = None
    getitem_341: "f32[8, 768, 28, 28]" = convolution_backward_15[0]
    getitem_342: "f32[768, 1, 3, 3]" = convolution_backward_15[1];  convolution_backward_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    view_707: "f32[8, 768, 784]" = torch.ops.aten.reshape.default(getitem_341, [8, 768, 784]);  getitem_341 = None
    permute_484: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_707, [0, 2, 1]);  view_707 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    clone_329: "f32[8, 784, 768]" = torch.ops.aten.clone.default(permute_484, memory_format = torch.contiguous_format);  permute_484 = None
    mul_1103: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(clone_329, primals_439);  primals_439 = None
    mul_1104: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1103, 768)
    sum_312: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1103, [2], True)
    mul_1105: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1103, mul_405);  mul_1103 = None
    sum_313: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1105, [2], True);  mul_1105 = None
    mul_1106: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_405, sum_313);  sum_313 = None
    sub_249: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(mul_1104, sum_312);  mul_1104 = sum_312 = None
    sub_250: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(sub_249, mul_1106);  sub_249 = mul_1106 = None
    mul_1107: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(div_161, sub_250);  div_161 = sub_250 = None
    mul_1108: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(clone_329, mul_405);  mul_405 = None
    sum_314: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1108, [0, 1]);  mul_1108 = None
    sum_315: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_329, [0, 1]);  clone_329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    add_533: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_530, mul_1107);  add_530 = mul_1107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_1109: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_533, primals_65);  primals_65 = None
    mul_1110: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_533, add_297);  add_297 = None
    sum_316: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1110, [0, 1], True);  mul_1110 = None
    view_708: "f32[768]" = torch.ops.aten.reshape.default(sum_316, [768]);  sum_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    sum_317: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1109, [0, 1], True)
    view_709: "f32[768]" = torch.ops.aten.reshape.default(sum_317, [768]);  sum_317 = None
    view_710: "f32[6272, 768]" = torch.ops.aten.reshape.default(mul_1109, [6272, 768]);  mul_1109 = None
    permute_485: "f32[768, 6272]" = torch.ops.aten.permute.default(view_710, [1, 0])
    mm_112: "f32[768, 768]" = torch.ops.aten.mm.default(permute_485, view_302);  permute_485 = view_302 = None
    permute_486: "f32[768, 768]" = torch.ops.aten.permute.default(mm_112, [1, 0]);  mm_112 = None
    mm_113: "f32[6272, 768]" = torch.ops.aten.mm.default(view_710, permute_487);  view_710 = permute_487 = None
    view_711: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_113, [8, 784, 768]);  mm_113 = None
    permute_488: "f32[768, 768]" = torch.ops.aten.permute.default(permute_486, [1, 0]);  permute_486 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    view_712: "f32[8, 784, 16, 48]" = torch.ops.aten.reshape.default(view_711, [8, 784, 16, 48]);  view_711 = None
    permute_489: "f32[8, 16, 48, 784]" = torch.ops.aten.permute.default(view_712, [0, 2, 3, 1]);  view_712 = None
    clone_331: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(permute_489, memory_format = torch.contiguous_format);  permute_489 = None
    view_713: "f32[128, 48, 784]" = torch.ops.aten.reshape.default(clone_331, [128, 48, 784]);  clone_331 = None
    bmm_76: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(permute_490, view_713);  permute_490 = None
    bmm_77: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(view_713, permute_491);  view_713 = permute_491 = None
    view_714: "f32[8, 16, 48, 784]" = torch.ops.aten.reshape.default(bmm_76, [8, 16, 48, 784]);  bmm_76 = None
    view_715: "f32[8, 16, 48, 48]" = torch.ops.aten.reshape.default(bmm_77, [8, 16, 48, 48]);  bmm_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    mul_1111: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_715, alias_97);  view_715 = None
    sum_318: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_1111, [-1], True)
    mul_1112: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(alias_97, sum_318);  alias_97 = sum_318 = None
    sub_251: "f32[8, 16, 48, 48]" = torch.ops.aten.sub.Tensor(mul_1111, mul_1112);  mul_1111 = mul_1112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    mul_1113: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(sub_251, view_297);  view_297 = None
    mul_1114: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(sub_251, primals_66);  sub_251 = primals_66 = None
    sum_319: "f32[1, 16, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1113, [0, 2, 3], True);  mul_1113 = None
    view_716: "f32[16, 1, 1]" = torch.ops.aten.reshape.default(sum_319, [16, 1, 1]);  sum_319 = None
    view_717: "f32[128, 48, 48]" = torch.ops.aten.reshape.default(mul_1114, [128, 48, 48]);  mul_1114 = None
    bmm_78: "f32[128, 784, 48]" = torch.ops.aten.bmm.default(permute_492, view_717);  permute_492 = None
    bmm_79: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_717, permute_493);  view_717 = permute_493 = None
    view_718: "f32[8, 16, 784, 48]" = torch.ops.aten.reshape.default(bmm_78, [8, 16, 784, 48]);  bmm_78 = None
    view_719: "f32[8, 16, 48, 784]" = torch.ops.aten.reshape.default(bmm_79, [8, 16, 48, 784]);  bmm_79 = None
    permute_494: "f32[8, 16, 48, 784]" = torch.ops.aten.permute.default(view_718, [0, 1, 3, 2]);  view_718 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    div_163: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_55, expand_97);  div_55 = None
    neg_14: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(permute_494)
    mul_1115: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_14, div_163);  neg_14 = div_163 = None
    div_164: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(permute_494, expand_97);  permute_494 = expand_97 = None
    sum_320: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_1115, [3], True);  mul_1115 = None
    ge_14: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(pow_69, 1e-12)
    where_28: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_14, sum_320, full_default_20);  ge_14 = sum_320 = None
    div_165: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_185, pow_69);  getitem_185 = None
    eq_14: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(pow_69, 0);  pow_69 = None
    where_29: "f32[8, 16, 48, 784]" = torch.ops.aten.where.self(eq_14, full_default_20, div_165);  eq_14 = div_165 = None
    clone_332: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(where_29, memory_format = torch.contiguous_format);  where_29 = None
    mul_1116: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_28, clone_332);  where_28 = clone_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    add_534: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_164, mul_1116);  div_164 = mul_1116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    div_167: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_54, expand_96);  div_54 = None
    neg_15: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(view_719)
    mul_1117: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_15, div_167);  neg_15 = div_167 = None
    div_168: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(view_719, expand_96);  view_719 = expand_96 = None
    sum_321: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_1117, [3], True);  mul_1117 = None
    ge_15: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(pow_67, 1e-12)
    where_30: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_15, sum_321, full_default_20);  ge_15 = sum_321 = None
    div_169: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_184, pow_67);  getitem_184 = None
    eq_15: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(pow_67, 0);  pow_67 = None
    where_31: "f32[8, 16, 48, 784]" = torch.ops.aten.where.self(eq_15, full_default_20, div_169);  eq_15 = div_169 = None
    clone_333: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(where_31, memory_format = torch.contiguous_format);  where_31 = None
    mul_1118: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_30, clone_333);  where_30 = clone_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    add_535: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_168, mul_1118);  div_168 = mul_1118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    cat_15: "f32[24, 16, 48, 784]" = torch.ops.aten.cat.default([add_535, add_534, view_714]);  add_535 = add_534 = view_714 = None
    view_720: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.reshape.default(cat_15, [3, 8, 16, 48, 784]);  cat_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    permute_495: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.permute.default(view_720, [1, 4, 0, 2, 3]);  view_720 = None
    clone_334: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.clone.default(permute_495, memory_format = torch.contiguous_format);  permute_495 = None
    view_721: "f32[8, 784, 2304]" = torch.ops.aten.reshape.default(clone_334, [8, 784, 2304]);  clone_334 = None
    view_722: "f32[6272, 2304]" = torch.ops.aten.reshape.default(view_721, [6272, 2304]);  view_721 = None
    mm_114: "f32[6272, 768]" = torch.ops.aten.mm.default(view_722, permute_496);  permute_496 = None
    permute_497: "f32[2304, 6272]" = torch.ops.aten.permute.default(view_722, [1, 0])
    mm_115: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_497, view_292);  permute_497 = view_292 = None
    permute_498: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_115, [1, 0]);  mm_115 = None
    sum_322: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_722, [0], True);  view_722 = None
    view_723: "f32[2304]" = torch.ops.aten.reshape.default(sum_322, [2304]);  sum_322 = None
    permute_499: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_498, [1, 0]);  permute_498 = None
    view_724: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_114, [8, 784, 768]);  mm_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_1120: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_724, primals_433);  primals_433 = None
    mul_1121: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1120, 768)
    sum_323: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1120, [2], True)
    mul_1122: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1120, mul_401);  mul_1120 = None
    sum_324: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1122, [2], True);  mul_1122 = None
    mul_1123: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_401, sum_324);  sum_324 = None
    sub_253: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(mul_1121, sum_323);  mul_1121 = sum_323 = None
    sub_254: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(sub_253, mul_1123);  sub_253 = mul_1123 = None
    mul_1124: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(div_170, sub_254);  div_170 = sub_254 = None
    mul_1125: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_724, mul_401);  mul_401 = None
    sum_325: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1125, [0, 1]);  mul_1125 = None
    sum_326: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_724, [0, 1]);  view_724 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    add_536: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_533, mul_1124);  add_533 = mul_1124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_1126: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_536, primals_64);  primals_64 = None
    mul_1127: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_536, view_291);  view_291 = None
    sum_327: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1127, [0, 1], True);  mul_1127 = None
    view_725: "f32[768]" = torch.ops.aten.reshape.default(sum_327, [768]);  sum_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_726: "f32[6272, 768]" = torch.ops.aten.reshape.default(mul_1126, [6272, 768]);  mul_1126 = None
    mm_116: "f32[6272, 3072]" = torch.ops.aten.mm.default(view_726, permute_500);  permute_500 = None
    permute_501: "f32[768, 6272]" = torch.ops.aten.permute.default(view_726, [1, 0])
    mm_117: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_501, view_290);  permute_501 = view_290 = None
    permute_502: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_117, [1, 0]);  mm_117 = None
    sum_328: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_726, [0], True);  view_726 = None
    view_727: "f32[768]" = torch.ops.aten.reshape.default(sum_328, [768]);  sum_328 = None
    permute_503: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_502, [1, 0]);  permute_502 = None
    view_728: "f32[8, 784, 3072]" = torch.ops.aten.reshape.default(mm_116, [8, 784, 3072]);  mm_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_1129: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(add_293, 0.5);  add_293 = None
    mul_1130: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_289, view_289)
    mul_1131: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(mul_1130, -0.5);  mul_1130 = None
    exp_42: "f32[8, 784, 3072]" = torch.ops.aten.exp.default(mul_1131);  mul_1131 = None
    mul_1132: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(exp_42, 0.3989422804014327);  exp_42 = None
    mul_1133: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_289, mul_1132);  view_289 = mul_1132 = None
    add_538: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(mul_1129, mul_1133);  mul_1129 = mul_1133 = None
    mul_1134: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_728, add_538);  view_728 = add_538 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_729: "f32[6272, 3072]" = torch.ops.aten.reshape.default(mul_1134, [6272, 3072]);  mul_1134 = None
    mm_118: "f32[6272, 768]" = torch.ops.aten.mm.default(view_729, permute_504);  permute_504 = None
    permute_505: "f32[3072, 6272]" = torch.ops.aten.permute.default(view_729, [1, 0])
    mm_119: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_505, view_288);  permute_505 = view_288 = None
    permute_506: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_119, [1, 0]);  mm_119 = None
    sum_329: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_729, [0], True);  view_729 = None
    view_730: "f32[3072]" = torch.ops.aten.reshape.default(sum_329, [3072]);  sum_329 = None
    permute_507: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_506, [1, 0]);  permute_506 = None
    view_731: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_118, [8, 784, 768]);  mm_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_1136: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_731, primals_427);  primals_427 = None
    mul_1137: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1136, 768)
    sum_330: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1136, [2], True)
    mul_1138: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1136, mul_395);  mul_1136 = None
    sum_331: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1138, [2], True);  mul_1138 = None
    mul_1139: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_395, sum_331);  sum_331 = None
    sub_256: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(mul_1137, sum_330);  mul_1137 = sum_330 = None
    sub_257: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(sub_256, mul_1139);  sub_256 = mul_1139 = None
    mul_1140: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(div_171, sub_257);  div_171 = sub_257 = None
    mul_1141: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_731, mul_395);  mul_395 = None
    sum_332: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1141, [0, 1]);  mul_1141 = None
    sum_333: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_731, [0, 1]);  view_731 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    add_539: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_536, mul_1140);  add_536 = mul_1140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_1142: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_539, primals_63);  primals_63 = None
    mul_1143: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_539, permute_144);  permute_144 = None
    sum_334: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1143, [0, 1], True);  mul_1143 = None
    view_732: "f32[768]" = torch.ops.aten.reshape.default(sum_334, [768]);  sum_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    permute_508: "f32[8, 768, 784]" = torch.ops.aten.permute.default(mul_1142, [0, 2, 1]);  mul_1142 = None
    view_733: "f32[8, 768, 28, 28]" = torch.ops.aten.reshape.default(permute_508, [8, 768, 28, 28]);  permute_508 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    sum_335: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_733, [0, 2, 3])
    convolution_backward_16 = torch.ops.aten.convolution_backward.default(view_733, add_289, primals_425, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False]);  view_733 = add_289 = primals_425 = None
    getitem_344: "f32[8, 768, 28, 28]" = convolution_backward_16[0]
    getitem_345: "f32[768, 1, 3, 3]" = convolution_backward_16[1];  convolution_backward_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    sum_336: "f32[768]" = torch.ops.aten.sum.dim_IntList(getitem_344, [0, 2, 3])
    sub_258: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(mul_386, unsqueeze_215);  mul_386 = unsqueeze_215 = None
    mul_1144: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_344, sub_258)
    sum_337: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1144, [0, 2, 3]);  mul_1144 = None
    mul_1145: "f32[768]" = torch.ops.aten.mul.Tensor(sum_336, 0.00015943877551020407)
    unsqueeze_216: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1145, 0);  mul_1145 = None
    unsqueeze_217: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_216, 2);  unsqueeze_216 = None
    unsqueeze_218: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_217, 3);  unsqueeze_217 = None
    mul_1146: "f32[768]" = torch.ops.aten.mul.Tensor(sum_337, 0.00015943877551020407)
    mul_1147: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_55, squeeze_55)
    mul_1148: "f32[768]" = torch.ops.aten.mul.Tensor(mul_1146, mul_1147);  mul_1146 = mul_1147 = None
    unsqueeze_219: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1148, 0);  mul_1148 = None
    unsqueeze_220: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_219, 2);  unsqueeze_219 = None
    unsqueeze_221: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_220, 3);  unsqueeze_220 = None
    mul_1149: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_55, primals_423);  primals_423 = None
    unsqueeze_222: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1149, 0);  mul_1149 = None
    unsqueeze_223: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_222, 2);  unsqueeze_222 = None
    unsqueeze_224: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_223, 3);  unsqueeze_223 = None
    mul_1150: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_258, unsqueeze_221);  sub_258 = unsqueeze_221 = None
    sub_260: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(getitem_344, mul_1150);  getitem_344 = mul_1150 = None
    sub_261: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(sub_260, unsqueeze_218);  sub_260 = unsqueeze_218 = None
    mul_1151: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_261, unsqueeze_224);  sub_261 = unsqueeze_224 = None
    mul_1152: "f32[768]" = torch.ops.aten.mul.Tensor(sum_337, squeeze_55);  sum_337 = squeeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_1154: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(add_284, 0.5);  add_284 = None
    mul_1155: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_34, convolution_34)
    mul_1156: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_1155, -0.5);  mul_1155 = None
    exp_43: "f32[8, 768, 28, 28]" = torch.ops.aten.exp.default(mul_1156);  mul_1156 = None
    mul_1157: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(exp_43, 0.3989422804014327);  exp_43 = None
    mul_1158: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_34, mul_1157);  convolution_34 = mul_1157 = None
    add_541: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(mul_1154, mul_1158);  mul_1154 = mul_1158 = None
    mul_1159: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_1151, add_541);  mul_1151 = add_541 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    sum_338: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1159, [0, 2, 3])
    convolution_backward_17 = torch.ops.aten.convolution_backward.default(mul_1159, view_286, primals_421, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False]);  mul_1159 = view_286 = primals_421 = None
    getitem_347: "f32[8, 768, 28, 28]" = convolution_backward_17[0]
    getitem_348: "f32[768, 1, 3, 3]" = convolution_backward_17[1];  convolution_backward_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    view_734: "f32[8, 768, 784]" = torch.ops.aten.reshape.default(getitem_347, [8, 768, 784]);  getitem_347 = None
    permute_509: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_734, [0, 2, 1]);  view_734 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    clone_337: "f32[8, 784, 768]" = torch.ops.aten.clone.default(permute_509, memory_format = torch.contiguous_format);  permute_509 = None
    mul_1161: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(clone_337, primals_419);  primals_419 = None
    mul_1162: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1161, 768)
    sum_339: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1161, [2], True)
    mul_1163: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1161, mul_382);  mul_1161 = None
    sum_340: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1163, [2], True);  mul_1163 = None
    mul_1164: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_382, sum_340);  sum_340 = None
    sub_263: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(mul_1162, sum_339);  mul_1162 = sum_339 = None
    sub_264: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(sub_263, mul_1164);  sub_263 = mul_1164 = None
    mul_1165: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(div_172, sub_264);  div_172 = sub_264 = None
    mul_1166: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(clone_337, mul_382);  mul_382 = None
    sum_341: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1166, [0, 1]);  mul_1166 = None
    sum_342: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_337, [0, 1]);  clone_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    add_542: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_539, mul_1165);  add_539 = mul_1165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_1167: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_542, primals_61);  primals_61 = None
    mul_1168: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_542, add_280);  add_280 = None
    sum_343: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1168, [0, 1], True);  mul_1168 = None
    view_735: "f32[768]" = torch.ops.aten.reshape.default(sum_343, [768]);  sum_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    sum_344: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1167, [0, 1], True)
    view_736: "f32[768]" = torch.ops.aten.reshape.default(sum_344, [768]);  sum_344 = None
    view_737: "f32[6272, 768]" = torch.ops.aten.reshape.default(mul_1167, [6272, 768]);  mul_1167 = None
    permute_510: "f32[768, 6272]" = torch.ops.aten.permute.default(view_737, [1, 0])
    mm_120: "f32[768, 768]" = torch.ops.aten.mm.default(permute_510, view_284);  permute_510 = view_284 = None
    permute_511: "f32[768, 768]" = torch.ops.aten.permute.default(mm_120, [1, 0]);  mm_120 = None
    mm_121: "f32[6272, 768]" = torch.ops.aten.mm.default(view_737, permute_512);  view_737 = permute_512 = None
    view_738: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_121, [8, 784, 768]);  mm_121 = None
    permute_513: "f32[768, 768]" = torch.ops.aten.permute.default(permute_511, [1, 0]);  permute_511 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    view_739: "f32[8, 784, 16, 48]" = torch.ops.aten.reshape.default(view_738, [8, 784, 16, 48]);  view_738 = None
    permute_514: "f32[8, 16, 48, 784]" = torch.ops.aten.permute.default(view_739, [0, 2, 3, 1]);  view_739 = None
    clone_339: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(permute_514, memory_format = torch.contiguous_format);  permute_514 = None
    view_740: "f32[128, 48, 784]" = torch.ops.aten.reshape.default(clone_339, [128, 48, 784]);  clone_339 = None
    bmm_80: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(permute_515, view_740);  permute_515 = None
    bmm_81: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(view_740, permute_516);  view_740 = permute_516 = None
    view_741: "f32[8, 16, 48, 784]" = torch.ops.aten.reshape.default(bmm_80, [8, 16, 48, 784]);  bmm_80 = None
    view_742: "f32[8, 16, 48, 48]" = torch.ops.aten.reshape.default(bmm_81, [8, 16, 48, 48]);  bmm_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    mul_1169: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_742, alias_100);  view_742 = None
    sum_345: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_1169, [-1], True)
    mul_1170: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(alias_100, sum_345);  alias_100 = sum_345 = None
    sub_265: "f32[8, 16, 48, 48]" = torch.ops.aten.sub.Tensor(mul_1169, mul_1170);  mul_1169 = mul_1170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    mul_1171: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(sub_265, view_279);  view_279 = None
    mul_1172: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(sub_265, primals_62);  sub_265 = primals_62 = None
    sum_346: "f32[1, 16, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1171, [0, 2, 3], True);  mul_1171 = None
    view_743: "f32[16, 1, 1]" = torch.ops.aten.reshape.default(sum_346, [16, 1, 1]);  sum_346 = None
    view_744: "f32[128, 48, 48]" = torch.ops.aten.reshape.default(mul_1172, [128, 48, 48]);  mul_1172 = None
    bmm_82: "f32[128, 784, 48]" = torch.ops.aten.bmm.default(permute_517, view_744);  permute_517 = None
    bmm_83: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_744, permute_518);  view_744 = permute_518 = None
    view_745: "f32[8, 16, 784, 48]" = torch.ops.aten.reshape.default(bmm_82, [8, 16, 784, 48]);  bmm_82 = None
    view_746: "f32[8, 16, 48, 784]" = torch.ops.aten.reshape.default(bmm_83, [8, 16, 48, 784]);  bmm_83 = None
    permute_519: "f32[8, 16, 48, 784]" = torch.ops.aten.permute.default(view_745, [0, 1, 3, 2]);  view_745 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    div_174: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_52, expand_91);  div_52 = None
    neg_16: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(permute_519)
    mul_1173: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_16, div_174);  neg_16 = div_174 = None
    div_175: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(permute_519, expand_91);  permute_519 = expand_91 = None
    sum_347: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_1173, [3], True);  mul_1173 = None
    ge_16: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(pow_65, 1e-12)
    where_32: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_16, sum_347, full_default_20);  ge_16 = sum_347 = None
    div_176: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_174, pow_65);  getitem_174 = None
    eq_16: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(pow_65, 0);  pow_65 = None
    where_33: "f32[8, 16, 48, 784]" = torch.ops.aten.where.self(eq_16, full_default_20, div_176);  eq_16 = div_176 = None
    clone_340: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(where_33, memory_format = torch.contiguous_format);  where_33 = None
    mul_1174: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_32, clone_340);  where_32 = clone_340 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    add_543: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_175, mul_1174);  div_175 = mul_1174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    div_178: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_51, expand_90);  div_51 = None
    neg_17: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(view_746)
    mul_1175: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_17, div_178);  neg_17 = div_178 = None
    div_179: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(view_746, expand_90);  view_746 = expand_90 = None
    sum_348: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_1175, [3], True);  mul_1175 = None
    ge_17: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(pow_63, 1e-12)
    where_34: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_17, sum_348, full_default_20);  ge_17 = sum_348 = None
    div_180: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_173, pow_63);  getitem_173 = None
    eq_17: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(pow_63, 0);  pow_63 = None
    where_35: "f32[8, 16, 48, 784]" = torch.ops.aten.where.self(eq_17, full_default_20, div_180);  eq_17 = div_180 = None
    clone_341: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(where_35, memory_format = torch.contiguous_format);  where_35 = None
    mul_1176: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_34, clone_341);  where_34 = clone_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    add_544: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_179, mul_1176);  div_179 = mul_1176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    cat_16: "f32[24, 16, 48, 784]" = torch.ops.aten.cat.default([add_544, add_543, view_741]);  add_544 = add_543 = view_741 = None
    view_747: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.reshape.default(cat_16, [3, 8, 16, 48, 784]);  cat_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    permute_520: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.permute.default(view_747, [1, 4, 0, 2, 3]);  view_747 = None
    clone_342: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.clone.default(permute_520, memory_format = torch.contiguous_format);  permute_520 = None
    view_748: "f32[8, 784, 2304]" = torch.ops.aten.reshape.default(clone_342, [8, 784, 2304]);  clone_342 = None
    view_749: "f32[6272, 2304]" = torch.ops.aten.reshape.default(view_748, [6272, 2304]);  view_748 = None
    mm_122: "f32[6272, 768]" = torch.ops.aten.mm.default(view_749, permute_521);  permute_521 = None
    permute_522: "f32[2304, 6272]" = torch.ops.aten.permute.default(view_749, [1, 0])
    mm_123: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_522, view_274);  permute_522 = view_274 = None
    permute_523: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_123, [1, 0]);  mm_123 = None
    sum_349: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_749, [0], True);  view_749 = None
    view_750: "f32[2304]" = torch.ops.aten.reshape.default(sum_349, [2304]);  sum_349 = None
    permute_524: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_523, [1, 0]);  permute_523 = None
    view_751: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_122, [8, 784, 768]);  mm_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_1178: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_751, primals_413);  primals_413 = None
    mul_1179: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1178, 768)
    sum_350: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1178, [2], True)
    mul_1180: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1178, mul_378);  mul_1178 = None
    sum_351: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1180, [2], True);  mul_1180 = None
    mul_1181: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_378, sum_351);  sum_351 = None
    sub_267: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(mul_1179, sum_350);  mul_1179 = sum_350 = None
    sub_268: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(sub_267, mul_1181);  sub_267 = mul_1181 = None
    mul_1182: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(div_181, sub_268);  div_181 = sub_268 = None
    mul_1183: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_751, mul_378);  mul_378 = None
    sum_352: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1183, [0, 1]);  mul_1183 = None
    sum_353: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_751, [0, 1]);  view_751 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    add_545: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_542, mul_1182);  add_542 = mul_1182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_1184: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_545, primals_60);  primals_60 = None
    mul_1185: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_545, view_273);  view_273 = None
    sum_354: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1185, [0, 1], True);  mul_1185 = None
    view_752: "f32[768]" = torch.ops.aten.reshape.default(sum_354, [768]);  sum_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_753: "f32[6272, 768]" = torch.ops.aten.reshape.default(mul_1184, [6272, 768]);  mul_1184 = None
    mm_124: "f32[6272, 3072]" = torch.ops.aten.mm.default(view_753, permute_525);  permute_525 = None
    permute_526: "f32[768, 6272]" = torch.ops.aten.permute.default(view_753, [1, 0])
    mm_125: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_526, view_272);  permute_526 = view_272 = None
    permute_527: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_125, [1, 0]);  mm_125 = None
    sum_355: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_753, [0], True);  view_753 = None
    view_754: "f32[768]" = torch.ops.aten.reshape.default(sum_355, [768]);  sum_355 = None
    permute_528: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_527, [1, 0]);  permute_527 = None
    view_755: "f32[8, 784, 3072]" = torch.ops.aten.reshape.default(mm_124, [8, 784, 3072]);  mm_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_1187: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(add_276, 0.5);  add_276 = None
    mul_1188: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_271, view_271)
    mul_1189: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(mul_1188, -0.5);  mul_1188 = None
    exp_44: "f32[8, 784, 3072]" = torch.ops.aten.exp.default(mul_1189);  mul_1189 = None
    mul_1190: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(exp_44, 0.3989422804014327);  exp_44 = None
    mul_1191: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_271, mul_1190);  view_271 = mul_1190 = None
    add_547: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(mul_1187, mul_1191);  mul_1187 = mul_1191 = None
    mul_1192: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_755, add_547);  view_755 = add_547 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_756: "f32[6272, 3072]" = torch.ops.aten.reshape.default(mul_1192, [6272, 3072]);  mul_1192 = None
    mm_126: "f32[6272, 768]" = torch.ops.aten.mm.default(view_756, permute_529);  permute_529 = None
    permute_530: "f32[3072, 6272]" = torch.ops.aten.permute.default(view_756, [1, 0])
    mm_127: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_530, view_270);  permute_530 = view_270 = None
    permute_531: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_127, [1, 0]);  mm_127 = None
    sum_356: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_756, [0], True);  view_756 = None
    view_757: "f32[3072]" = torch.ops.aten.reshape.default(sum_356, [3072]);  sum_356 = None
    permute_532: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_531, [1, 0]);  permute_531 = None
    view_758: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_126, [8, 784, 768]);  mm_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_1194: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_758, primals_407);  primals_407 = None
    mul_1195: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1194, 768)
    sum_357: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1194, [2], True)
    mul_1196: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1194, mul_372);  mul_1194 = None
    sum_358: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1196, [2], True);  mul_1196 = None
    mul_1197: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_372, sum_358);  sum_358 = None
    sub_270: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(mul_1195, sum_357);  mul_1195 = sum_357 = None
    sub_271: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(sub_270, mul_1197);  sub_270 = mul_1197 = None
    mul_1198: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(div_182, sub_271);  div_182 = sub_271 = None
    mul_1199: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_758, mul_372);  mul_372 = None
    sum_359: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1199, [0, 1]);  mul_1199 = None
    sum_360: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_758, [0, 1]);  view_758 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    add_548: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_545, mul_1198);  add_545 = mul_1198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_1200: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_548, primals_59);  primals_59 = None
    mul_1201: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_548, permute_135);  permute_135 = None
    sum_361: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1201, [0, 1], True);  mul_1201 = None
    view_759: "f32[768]" = torch.ops.aten.reshape.default(sum_361, [768]);  sum_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    permute_533: "f32[8, 768, 784]" = torch.ops.aten.permute.default(mul_1200, [0, 2, 1]);  mul_1200 = None
    view_760: "f32[8, 768, 28, 28]" = torch.ops.aten.reshape.default(permute_533, [8, 768, 28, 28]);  permute_533 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    sum_362: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_760, [0, 2, 3])
    convolution_backward_18 = torch.ops.aten.convolution_backward.default(view_760, add_272, primals_405, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False]);  view_760 = add_272 = primals_405 = None
    getitem_350: "f32[8, 768, 28, 28]" = convolution_backward_18[0]
    getitem_351: "f32[768, 1, 3, 3]" = convolution_backward_18[1];  convolution_backward_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    sum_363: "f32[768]" = torch.ops.aten.sum.dim_IntList(getitem_350, [0, 2, 3])
    sub_272: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(mul_363, unsqueeze_227);  mul_363 = unsqueeze_227 = None
    mul_1202: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_350, sub_272)
    sum_364: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1202, [0, 2, 3]);  mul_1202 = None
    mul_1203: "f32[768]" = torch.ops.aten.mul.Tensor(sum_363, 0.00015943877551020407)
    unsqueeze_228: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1203, 0);  mul_1203 = None
    unsqueeze_229: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_228, 2);  unsqueeze_228 = None
    unsqueeze_230: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_229, 3);  unsqueeze_229 = None
    mul_1204: "f32[768]" = torch.ops.aten.mul.Tensor(sum_364, 0.00015943877551020407)
    mul_1205: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_52, squeeze_52)
    mul_1206: "f32[768]" = torch.ops.aten.mul.Tensor(mul_1204, mul_1205);  mul_1204 = mul_1205 = None
    unsqueeze_231: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1206, 0);  mul_1206 = None
    unsqueeze_232: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_231, 2);  unsqueeze_231 = None
    unsqueeze_233: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_232, 3);  unsqueeze_232 = None
    mul_1207: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_52, primals_403);  primals_403 = None
    unsqueeze_234: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1207, 0);  mul_1207 = None
    unsqueeze_235: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_234, 2);  unsqueeze_234 = None
    unsqueeze_236: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_235, 3);  unsqueeze_235 = None
    mul_1208: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_272, unsqueeze_233);  sub_272 = unsqueeze_233 = None
    sub_274: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(getitem_350, mul_1208);  getitem_350 = mul_1208 = None
    sub_275: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(sub_274, unsqueeze_230);  sub_274 = unsqueeze_230 = None
    mul_1209: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_275, unsqueeze_236);  sub_275 = unsqueeze_236 = None
    mul_1210: "f32[768]" = torch.ops.aten.mul.Tensor(sum_364, squeeze_52);  sum_364 = squeeze_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_1212: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(add_267, 0.5);  add_267 = None
    mul_1213: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_32, convolution_32)
    mul_1214: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_1213, -0.5);  mul_1213 = None
    exp_45: "f32[8, 768, 28, 28]" = torch.ops.aten.exp.default(mul_1214);  mul_1214 = None
    mul_1215: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(exp_45, 0.3989422804014327);  exp_45 = None
    mul_1216: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_32, mul_1215);  convolution_32 = mul_1215 = None
    add_550: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(mul_1212, mul_1216);  mul_1212 = mul_1216 = None
    mul_1217: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_1209, add_550);  mul_1209 = add_550 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    sum_365: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1217, [0, 2, 3])
    convolution_backward_19 = torch.ops.aten.convolution_backward.default(mul_1217, view_268, primals_401, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False]);  mul_1217 = view_268 = primals_401 = None
    getitem_353: "f32[8, 768, 28, 28]" = convolution_backward_19[0]
    getitem_354: "f32[768, 1, 3, 3]" = convolution_backward_19[1];  convolution_backward_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    view_761: "f32[8, 768, 784]" = torch.ops.aten.reshape.default(getitem_353, [8, 768, 784]);  getitem_353 = None
    permute_534: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_761, [0, 2, 1]);  view_761 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    clone_345: "f32[8, 784, 768]" = torch.ops.aten.clone.default(permute_534, memory_format = torch.contiguous_format);  permute_534 = None
    mul_1219: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(clone_345, primals_399);  primals_399 = None
    mul_1220: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1219, 768)
    sum_366: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1219, [2], True)
    mul_1221: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1219, mul_359);  mul_1219 = None
    sum_367: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1221, [2], True);  mul_1221 = None
    mul_1222: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_359, sum_367);  sum_367 = None
    sub_277: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(mul_1220, sum_366);  mul_1220 = sum_366 = None
    sub_278: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(sub_277, mul_1222);  sub_277 = mul_1222 = None
    mul_1223: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(div_183, sub_278);  div_183 = sub_278 = None
    mul_1224: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(clone_345, mul_359);  mul_359 = None
    sum_368: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1224, [0, 1]);  mul_1224 = None
    sum_369: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_345, [0, 1]);  clone_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    add_551: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_548, mul_1223);  add_548 = mul_1223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_1225: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_551, primals_57);  primals_57 = None
    mul_1226: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_551, add_263);  add_263 = None
    sum_370: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1226, [0, 1], True);  mul_1226 = None
    view_762: "f32[768]" = torch.ops.aten.reshape.default(sum_370, [768]);  sum_370 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    sum_371: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1225, [0, 1], True)
    view_763: "f32[768]" = torch.ops.aten.reshape.default(sum_371, [768]);  sum_371 = None
    view_764: "f32[6272, 768]" = torch.ops.aten.reshape.default(mul_1225, [6272, 768]);  mul_1225 = None
    permute_535: "f32[768, 6272]" = torch.ops.aten.permute.default(view_764, [1, 0])
    mm_128: "f32[768, 768]" = torch.ops.aten.mm.default(permute_535, view_266);  permute_535 = view_266 = None
    permute_536: "f32[768, 768]" = torch.ops.aten.permute.default(mm_128, [1, 0]);  mm_128 = None
    mm_129: "f32[6272, 768]" = torch.ops.aten.mm.default(view_764, permute_537);  view_764 = permute_537 = None
    view_765: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_129, [8, 784, 768]);  mm_129 = None
    permute_538: "f32[768, 768]" = torch.ops.aten.permute.default(permute_536, [1, 0]);  permute_536 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    view_766: "f32[8, 784, 16, 48]" = torch.ops.aten.reshape.default(view_765, [8, 784, 16, 48]);  view_765 = None
    permute_539: "f32[8, 16, 48, 784]" = torch.ops.aten.permute.default(view_766, [0, 2, 3, 1]);  view_766 = None
    clone_347: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(permute_539, memory_format = torch.contiguous_format);  permute_539 = None
    view_767: "f32[128, 48, 784]" = torch.ops.aten.reshape.default(clone_347, [128, 48, 784]);  clone_347 = None
    bmm_84: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(permute_540, view_767);  permute_540 = None
    bmm_85: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(view_767, permute_541);  view_767 = permute_541 = None
    view_768: "f32[8, 16, 48, 784]" = torch.ops.aten.reshape.default(bmm_84, [8, 16, 48, 784]);  bmm_84 = None
    view_769: "f32[8, 16, 48, 48]" = torch.ops.aten.reshape.default(bmm_85, [8, 16, 48, 48]);  bmm_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    mul_1227: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_769, alias_103);  view_769 = None
    sum_372: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_1227, [-1], True)
    mul_1228: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(alias_103, sum_372);  alias_103 = sum_372 = None
    sub_279: "f32[8, 16, 48, 48]" = torch.ops.aten.sub.Tensor(mul_1227, mul_1228);  mul_1227 = mul_1228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    mul_1229: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(sub_279, view_261);  view_261 = None
    mul_1230: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(sub_279, primals_58);  sub_279 = primals_58 = None
    sum_373: "f32[1, 16, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1229, [0, 2, 3], True);  mul_1229 = None
    view_770: "f32[16, 1, 1]" = torch.ops.aten.reshape.default(sum_373, [16, 1, 1]);  sum_373 = None
    view_771: "f32[128, 48, 48]" = torch.ops.aten.reshape.default(mul_1230, [128, 48, 48]);  mul_1230 = None
    bmm_86: "f32[128, 784, 48]" = torch.ops.aten.bmm.default(permute_542, view_771);  permute_542 = None
    bmm_87: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_771, permute_543);  view_771 = permute_543 = None
    view_772: "f32[8, 16, 784, 48]" = torch.ops.aten.reshape.default(bmm_86, [8, 16, 784, 48]);  bmm_86 = None
    view_773: "f32[8, 16, 48, 784]" = torch.ops.aten.reshape.default(bmm_87, [8, 16, 48, 784]);  bmm_87 = None
    permute_544: "f32[8, 16, 48, 784]" = torch.ops.aten.permute.default(view_772, [0, 1, 3, 2]);  view_772 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    div_185: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_49, expand_85);  div_49 = None
    neg_18: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(permute_544)
    mul_1231: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_18, div_185);  neg_18 = div_185 = None
    div_186: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(permute_544, expand_85);  permute_544 = expand_85 = None
    sum_374: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_1231, [3], True);  mul_1231 = None
    ge_18: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(pow_61, 1e-12)
    where_36: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_18, sum_374, full_default_20);  ge_18 = sum_374 = None
    div_187: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_163, pow_61);  getitem_163 = None
    eq_18: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(pow_61, 0);  pow_61 = None
    where_37: "f32[8, 16, 48, 784]" = torch.ops.aten.where.self(eq_18, full_default_20, div_187);  eq_18 = div_187 = None
    clone_348: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(where_37, memory_format = torch.contiguous_format);  where_37 = None
    mul_1232: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_36, clone_348);  where_36 = clone_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    add_552: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_186, mul_1232);  div_186 = mul_1232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    div_189: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_48, expand_84);  div_48 = None
    neg_19: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(view_773)
    mul_1233: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_19, div_189);  neg_19 = div_189 = None
    div_190: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(view_773, expand_84);  view_773 = expand_84 = None
    sum_375: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_1233, [3], True);  mul_1233 = None
    ge_19: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(pow_59, 1e-12)
    where_38: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_19, sum_375, full_default_20);  ge_19 = sum_375 = None
    div_191: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_162, pow_59);  getitem_162 = None
    eq_19: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(pow_59, 0);  pow_59 = None
    where_39: "f32[8, 16, 48, 784]" = torch.ops.aten.where.self(eq_19, full_default_20, div_191);  eq_19 = div_191 = None
    clone_349: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(where_39, memory_format = torch.contiguous_format);  where_39 = None
    mul_1234: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_38, clone_349);  where_38 = clone_349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    add_553: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_190, mul_1234);  div_190 = mul_1234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    cat_17: "f32[24, 16, 48, 784]" = torch.ops.aten.cat.default([add_553, add_552, view_768]);  add_553 = add_552 = view_768 = None
    view_774: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.reshape.default(cat_17, [3, 8, 16, 48, 784]);  cat_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    permute_545: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.permute.default(view_774, [1, 4, 0, 2, 3]);  view_774 = None
    clone_350: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.clone.default(permute_545, memory_format = torch.contiguous_format);  permute_545 = None
    view_775: "f32[8, 784, 2304]" = torch.ops.aten.reshape.default(clone_350, [8, 784, 2304]);  clone_350 = None
    view_776: "f32[6272, 2304]" = torch.ops.aten.reshape.default(view_775, [6272, 2304]);  view_775 = None
    mm_130: "f32[6272, 768]" = torch.ops.aten.mm.default(view_776, permute_546);  permute_546 = None
    permute_547: "f32[2304, 6272]" = torch.ops.aten.permute.default(view_776, [1, 0])
    mm_131: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_547, view_256);  permute_547 = view_256 = None
    permute_548: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_131, [1, 0]);  mm_131 = None
    sum_376: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_776, [0], True);  view_776 = None
    view_777: "f32[2304]" = torch.ops.aten.reshape.default(sum_376, [2304]);  sum_376 = None
    permute_549: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_548, [1, 0]);  permute_548 = None
    view_778: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_130, [8, 784, 768]);  mm_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_1236: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_778, primals_393);  primals_393 = None
    mul_1237: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1236, 768)
    sum_377: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1236, [2], True)
    mul_1238: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1236, mul_355);  mul_1236 = None
    sum_378: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1238, [2], True);  mul_1238 = None
    mul_1239: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_355, sum_378);  sum_378 = None
    sub_281: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(mul_1237, sum_377);  mul_1237 = sum_377 = None
    sub_282: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(sub_281, mul_1239);  sub_281 = mul_1239 = None
    mul_1240: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(div_192, sub_282);  div_192 = sub_282 = None
    mul_1241: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_778, mul_355);  mul_355 = None
    sum_379: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1241, [0, 1]);  mul_1241 = None
    sum_380: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_778, [0, 1]);  view_778 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    add_554: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_551, mul_1240);  add_551 = mul_1240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_1242: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_554, primals_56);  primals_56 = None
    mul_1243: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_554, view_255);  view_255 = None
    sum_381: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1243, [0, 1], True);  mul_1243 = None
    view_779: "f32[768]" = torch.ops.aten.reshape.default(sum_381, [768]);  sum_381 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_780: "f32[6272, 768]" = torch.ops.aten.reshape.default(mul_1242, [6272, 768]);  mul_1242 = None
    mm_132: "f32[6272, 3072]" = torch.ops.aten.mm.default(view_780, permute_550);  permute_550 = None
    permute_551: "f32[768, 6272]" = torch.ops.aten.permute.default(view_780, [1, 0])
    mm_133: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_551, view_254);  permute_551 = view_254 = None
    permute_552: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_133, [1, 0]);  mm_133 = None
    sum_382: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_780, [0], True);  view_780 = None
    view_781: "f32[768]" = torch.ops.aten.reshape.default(sum_382, [768]);  sum_382 = None
    permute_553: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_552, [1, 0]);  permute_552 = None
    view_782: "f32[8, 784, 3072]" = torch.ops.aten.reshape.default(mm_132, [8, 784, 3072]);  mm_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_1245: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(add_259, 0.5);  add_259 = None
    mul_1246: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_253, view_253)
    mul_1247: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(mul_1246, -0.5);  mul_1246 = None
    exp_46: "f32[8, 784, 3072]" = torch.ops.aten.exp.default(mul_1247);  mul_1247 = None
    mul_1248: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(exp_46, 0.3989422804014327);  exp_46 = None
    mul_1249: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_253, mul_1248);  view_253 = mul_1248 = None
    add_556: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(mul_1245, mul_1249);  mul_1245 = mul_1249 = None
    mul_1250: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_782, add_556);  view_782 = add_556 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_783: "f32[6272, 3072]" = torch.ops.aten.reshape.default(mul_1250, [6272, 3072]);  mul_1250 = None
    mm_134: "f32[6272, 768]" = torch.ops.aten.mm.default(view_783, permute_554);  permute_554 = None
    permute_555: "f32[3072, 6272]" = torch.ops.aten.permute.default(view_783, [1, 0])
    mm_135: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_555, view_252);  permute_555 = view_252 = None
    permute_556: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_135, [1, 0]);  mm_135 = None
    sum_383: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_783, [0], True);  view_783 = None
    view_784: "f32[3072]" = torch.ops.aten.reshape.default(sum_383, [3072]);  sum_383 = None
    permute_557: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_556, [1, 0]);  permute_556 = None
    view_785: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_134, [8, 784, 768]);  mm_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_1252: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_785, primals_387);  primals_387 = None
    mul_1253: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1252, 768)
    sum_384: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1252, [2], True)
    mul_1254: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1252, mul_349);  mul_1252 = None
    sum_385: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1254, [2], True);  mul_1254 = None
    mul_1255: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_349, sum_385);  sum_385 = None
    sub_284: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(mul_1253, sum_384);  mul_1253 = sum_384 = None
    sub_285: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(sub_284, mul_1255);  sub_284 = mul_1255 = None
    mul_1256: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(div_193, sub_285);  div_193 = sub_285 = None
    mul_1257: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_785, mul_349);  mul_349 = None
    sum_386: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1257, [0, 1]);  mul_1257 = None
    sum_387: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_785, [0, 1]);  view_785 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    add_557: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_554, mul_1256);  add_554 = mul_1256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_1258: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_557, primals_55);  primals_55 = None
    mul_1259: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_557, permute_126);  permute_126 = None
    sum_388: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1259, [0, 1], True);  mul_1259 = None
    view_786: "f32[768]" = torch.ops.aten.reshape.default(sum_388, [768]);  sum_388 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    permute_558: "f32[8, 768, 784]" = torch.ops.aten.permute.default(mul_1258, [0, 2, 1]);  mul_1258 = None
    view_787: "f32[8, 768, 28, 28]" = torch.ops.aten.reshape.default(permute_558, [8, 768, 28, 28]);  permute_558 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    sum_389: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_787, [0, 2, 3])
    convolution_backward_20 = torch.ops.aten.convolution_backward.default(view_787, add_255, primals_385, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False]);  view_787 = add_255 = primals_385 = None
    getitem_356: "f32[8, 768, 28, 28]" = convolution_backward_20[0]
    getitem_357: "f32[768, 1, 3, 3]" = convolution_backward_20[1];  convolution_backward_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    sum_390: "f32[768]" = torch.ops.aten.sum.dim_IntList(getitem_356, [0, 2, 3])
    sub_286: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(mul_340, unsqueeze_239);  mul_340 = unsqueeze_239 = None
    mul_1260: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_356, sub_286)
    sum_391: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1260, [0, 2, 3]);  mul_1260 = None
    mul_1261: "f32[768]" = torch.ops.aten.mul.Tensor(sum_390, 0.00015943877551020407)
    unsqueeze_240: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1261, 0);  mul_1261 = None
    unsqueeze_241: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_240, 2);  unsqueeze_240 = None
    unsqueeze_242: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_241, 3);  unsqueeze_241 = None
    mul_1262: "f32[768]" = torch.ops.aten.mul.Tensor(sum_391, 0.00015943877551020407)
    mul_1263: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_49, squeeze_49)
    mul_1264: "f32[768]" = torch.ops.aten.mul.Tensor(mul_1262, mul_1263);  mul_1262 = mul_1263 = None
    unsqueeze_243: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1264, 0);  mul_1264 = None
    unsqueeze_244: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_243, 2);  unsqueeze_243 = None
    unsqueeze_245: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_244, 3);  unsqueeze_244 = None
    mul_1265: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_49, primals_383);  primals_383 = None
    unsqueeze_246: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1265, 0);  mul_1265 = None
    unsqueeze_247: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_246, 2);  unsqueeze_246 = None
    unsqueeze_248: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_247, 3);  unsqueeze_247 = None
    mul_1266: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_286, unsqueeze_245);  sub_286 = unsqueeze_245 = None
    sub_288: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(getitem_356, mul_1266);  getitem_356 = mul_1266 = None
    sub_289: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(sub_288, unsqueeze_242);  sub_288 = unsqueeze_242 = None
    mul_1267: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_289, unsqueeze_248);  sub_289 = unsqueeze_248 = None
    mul_1268: "f32[768]" = torch.ops.aten.mul.Tensor(sum_391, squeeze_49);  sum_391 = squeeze_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_1270: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(add_250, 0.5);  add_250 = None
    mul_1271: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_30, convolution_30)
    mul_1272: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_1271, -0.5);  mul_1271 = None
    exp_47: "f32[8, 768, 28, 28]" = torch.ops.aten.exp.default(mul_1272);  mul_1272 = None
    mul_1273: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(exp_47, 0.3989422804014327);  exp_47 = None
    mul_1274: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_30, mul_1273);  convolution_30 = mul_1273 = None
    add_559: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(mul_1270, mul_1274);  mul_1270 = mul_1274 = None
    mul_1275: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_1267, add_559);  mul_1267 = add_559 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    sum_392: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1275, [0, 2, 3])
    convolution_backward_21 = torch.ops.aten.convolution_backward.default(mul_1275, view_250, primals_381, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False]);  mul_1275 = view_250 = primals_381 = None
    getitem_359: "f32[8, 768, 28, 28]" = convolution_backward_21[0]
    getitem_360: "f32[768, 1, 3, 3]" = convolution_backward_21[1];  convolution_backward_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    view_788: "f32[8, 768, 784]" = torch.ops.aten.reshape.default(getitem_359, [8, 768, 784]);  getitem_359 = None
    permute_559: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_788, [0, 2, 1]);  view_788 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    clone_353: "f32[8, 784, 768]" = torch.ops.aten.clone.default(permute_559, memory_format = torch.contiguous_format);  permute_559 = None
    mul_1277: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(clone_353, primals_379);  primals_379 = None
    mul_1278: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1277, 768)
    sum_393: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1277, [2], True)
    mul_1279: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1277, mul_336);  mul_1277 = None
    sum_394: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1279, [2], True);  mul_1279 = None
    mul_1280: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_336, sum_394);  sum_394 = None
    sub_291: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(mul_1278, sum_393);  mul_1278 = sum_393 = None
    sub_292: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(sub_291, mul_1280);  sub_291 = mul_1280 = None
    mul_1281: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(div_194, sub_292);  div_194 = sub_292 = None
    mul_1282: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(clone_353, mul_336);  mul_336 = None
    sum_395: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1282, [0, 1]);  mul_1282 = None
    sum_396: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_353, [0, 1]);  clone_353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    add_560: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_557, mul_1281);  add_557 = mul_1281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_1283: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_560, primals_53);  primals_53 = None
    mul_1284: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_560, add_246);  add_246 = None
    sum_397: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1284, [0, 1], True);  mul_1284 = None
    view_789: "f32[768]" = torch.ops.aten.reshape.default(sum_397, [768]);  sum_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    sum_398: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1283, [0, 1], True)
    view_790: "f32[768]" = torch.ops.aten.reshape.default(sum_398, [768]);  sum_398 = None
    view_791: "f32[6272, 768]" = torch.ops.aten.reshape.default(mul_1283, [6272, 768]);  mul_1283 = None
    permute_560: "f32[768, 6272]" = torch.ops.aten.permute.default(view_791, [1, 0])
    mm_136: "f32[768, 768]" = torch.ops.aten.mm.default(permute_560, view_248);  permute_560 = view_248 = None
    permute_561: "f32[768, 768]" = torch.ops.aten.permute.default(mm_136, [1, 0]);  mm_136 = None
    mm_137: "f32[6272, 768]" = torch.ops.aten.mm.default(view_791, permute_562);  view_791 = permute_562 = None
    view_792: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_137, [8, 784, 768]);  mm_137 = None
    permute_563: "f32[768, 768]" = torch.ops.aten.permute.default(permute_561, [1, 0]);  permute_561 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    view_793: "f32[8, 784, 16, 48]" = torch.ops.aten.reshape.default(view_792, [8, 784, 16, 48]);  view_792 = None
    permute_564: "f32[8, 16, 48, 784]" = torch.ops.aten.permute.default(view_793, [0, 2, 3, 1]);  view_793 = None
    clone_355: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(permute_564, memory_format = torch.contiguous_format);  permute_564 = None
    view_794: "f32[128, 48, 784]" = torch.ops.aten.reshape.default(clone_355, [128, 48, 784]);  clone_355 = None
    bmm_88: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(permute_565, view_794);  permute_565 = None
    bmm_89: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(view_794, permute_566);  view_794 = permute_566 = None
    view_795: "f32[8, 16, 48, 784]" = torch.ops.aten.reshape.default(bmm_88, [8, 16, 48, 784]);  bmm_88 = None
    view_796: "f32[8, 16, 48, 48]" = torch.ops.aten.reshape.default(bmm_89, [8, 16, 48, 48]);  bmm_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    mul_1285: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_796, alias_106);  view_796 = None
    sum_399: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_1285, [-1], True)
    mul_1286: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(alias_106, sum_399);  alias_106 = sum_399 = None
    sub_293: "f32[8, 16, 48, 48]" = torch.ops.aten.sub.Tensor(mul_1285, mul_1286);  mul_1285 = mul_1286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    mul_1287: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(sub_293, view_243);  view_243 = None
    mul_1288: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(sub_293, primals_54);  sub_293 = primals_54 = None
    sum_400: "f32[1, 16, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1287, [0, 2, 3], True);  mul_1287 = None
    view_797: "f32[16, 1, 1]" = torch.ops.aten.reshape.default(sum_400, [16, 1, 1]);  sum_400 = None
    view_798: "f32[128, 48, 48]" = torch.ops.aten.reshape.default(mul_1288, [128, 48, 48]);  mul_1288 = None
    bmm_90: "f32[128, 784, 48]" = torch.ops.aten.bmm.default(permute_567, view_798);  permute_567 = None
    bmm_91: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_798, permute_568);  view_798 = permute_568 = None
    view_799: "f32[8, 16, 784, 48]" = torch.ops.aten.reshape.default(bmm_90, [8, 16, 784, 48]);  bmm_90 = None
    view_800: "f32[8, 16, 48, 784]" = torch.ops.aten.reshape.default(bmm_91, [8, 16, 48, 784]);  bmm_91 = None
    permute_569: "f32[8, 16, 48, 784]" = torch.ops.aten.permute.default(view_799, [0, 1, 3, 2]);  view_799 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    div_196: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_46, expand_79);  div_46 = None
    neg_20: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(permute_569)
    mul_1289: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_20, div_196);  neg_20 = div_196 = None
    div_197: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(permute_569, expand_79);  permute_569 = expand_79 = None
    sum_401: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_1289, [3], True);  mul_1289 = None
    ge_20: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(pow_57, 1e-12)
    where_40: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_20, sum_401, full_default_20);  ge_20 = sum_401 = None
    div_198: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_152, pow_57);  getitem_152 = None
    eq_20: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(pow_57, 0);  pow_57 = None
    where_41: "f32[8, 16, 48, 784]" = torch.ops.aten.where.self(eq_20, full_default_20, div_198);  eq_20 = div_198 = None
    clone_356: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(where_41, memory_format = torch.contiguous_format);  where_41 = None
    mul_1290: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_40, clone_356);  where_40 = clone_356 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    add_561: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_197, mul_1290);  div_197 = mul_1290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    div_200: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_45, expand_78);  div_45 = None
    neg_21: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(view_800)
    mul_1291: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_21, div_200);  neg_21 = div_200 = None
    div_201: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(view_800, expand_78);  view_800 = expand_78 = None
    sum_402: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_1291, [3], True);  mul_1291 = None
    ge_21: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(pow_55, 1e-12)
    where_42: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_21, sum_402, full_default_20);  ge_21 = sum_402 = None
    div_202: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_151, pow_55);  getitem_151 = None
    eq_21: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(pow_55, 0);  pow_55 = None
    where_43: "f32[8, 16, 48, 784]" = torch.ops.aten.where.self(eq_21, full_default_20, div_202);  eq_21 = div_202 = None
    clone_357: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(where_43, memory_format = torch.contiguous_format);  where_43 = None
    mul_1292: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_42, clone_357);  where_42 = clone_357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    add_562: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_201, mul_1292);  div_201 = mul_1292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    cat_18: "f32[24, 16, 48, 784]" = torch.ops.aten.cat.default([add_562, add_561, view_795]);  add_562 = add_561 = view_795 = None
    view_801: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.reshape.default(cat_18, [3, 8, 16, 48, 784]);  cat_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    permute_570: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.permute.default(view_801, [1, 4, 0, 2, 3]);  view_801 = None
    clone_358: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.clone.default(permute_570, memory_format = torch.contiguous_format);  permute_570 = None
    view_802: "f32[8, 784, 2304]" = torch.ops.aten.reshape.default(clone_358, [8, 784, 2304]);  clone_358 = None
    view_803: "f32[6272, 2304]" = torch.ops.aten.reshape.default(view_802, [6272, 2304]);  view_802 = None
    mm_138: "f32[6272, 768]" = torch.ops.aten.mm.default(view_803, permute_571);  permute_571 = None
    permute_572: "f32[2304, 6272]" = torch.ops.aten.permute.default(view_803, [1, 0])
    mm_139: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_572, view_238);  permute_572 = view_238 = None
    permute_573: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_139, [1, 0]);  mm_139 = None
    sum_403: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_803, [0], True);  view_803 = None
    view_804: "f32[2304]" = torch.ops.aten.reshape.default(sum_403, [2304]);  sum_403 = None
    permute_574: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_573, [1, 0]);  permute_573 = None
    view_805: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_138, [8, 784, 768]);  mm_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_1294: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_805, primals_373);  primals_373 = None
    mul_1295: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1294, 768)
    sum_404: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1294, [2], True)
    mul_1296: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1294, mul_332);  mul_1294 = None
    sum_405: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1296, [2], True);  mul_1296 = None
    mul_1297: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_332, sum_405);  sum_405 = None
    sub_295: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(mul_1295, sum_404);  mul_1295 = sum_404 = None
    sub_296: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(sub_295, mul_1297);  sub_295 = mul_1297 = None
    mul_1298: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(div_203, sub_296);  div_203 = sub_296 = None
    mul_1299: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_805, mul_332);  mul_332 = None
    sum_406: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1299, [0, 1]);  mul_1299 = None
    sum_407: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_805, [0, 1]);  view_805 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    add_563: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_560, mul_1298);  add_560 = mul_1298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_1300: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_563, primals_52);  primals_52 = None
    mul_1301: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_563, view_237);  view_237 = None
    sum_408: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1301, [0, 1], True);  mul_1301 = None
    view_806: "f32[768]" = torch.ops.aten.reshape.default(sum_408, [768]);  sum_408 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_807: "f32[6272, 768]" = torch.ops.aten.reshape.default(mul_1300, [6272, 768]);  mul_1300 = None
    mm_140: "f32[6272, 3072]" = torch.ops.aten.mm.default(view_807, permute_575);  permute_575 = None
    permute_576: "f32[768, 6272]" = torch.ops.aten.permute.default(view_807, [1, 0])
    mm_141: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_576, view_236);  permute_576 = view_236 = None
    permute_577: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_141, [1, 0]);  mm_141 = None
    sum_409: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_807, [0], True);  view_807 = None
    view_808: "f32[768]" = torch.ops.aten.reshape.default(sum_409, [768]);  sum_409 = None
    permute_578: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_577, [1, 0]);  permute_577 = None
    view_809: "f32[8, 784, 3072]" = torch.ops.aten.reshape.default(mm_140, [8, 784, 3072]);  mm_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_1303: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(add_242, 0.5);  add_242 = None
    mul_1304: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_235, view_235)
    mul_1305: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(mul_1304, -0.5);  mul_1304 = None
    exp_48: "f32[8, 784, 3072]" = torch.ops.aten.exp.default(mul_1305);  mul_1305 = None
    mul_1306: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(exp_48, 0.3989422804014327);  exp_48 = None
    mul_1307: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_235, mul_1306);  view_235 = mul_1306 = None
    add_565: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(mul_1303, mul_1307);  mul_1303 = mul_1307 = None
    mul_1308: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_809, add_565);  view_809 = add_565 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_810: "f32[6272, 3072]" = torch.ops.aten.reshape.default(mul_1308, [6272, 3072]);  mul_1308 = None
    mm_142: "f32[6272, 768]" = torch.ops.aten.mm.default(view_810, permute_579);  permute_579 = None
    permute_580: "f32[3072, 6272]" = torch.ops.aten.permute.default(view_810, [1, 0])
    mm_143: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_580, view_234);  permute_580 = view_234 = None
    permute_581: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_143, [1, 0]);  mm_143 = None
    sum_410: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_810, [0], True);  view_810 = None
    view_811: "f32[3072]" = torch.ops.aten.reshape.default(sum_410, [3072]);  sum_410 = None
    permute_582: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_581, [1, 0]);  permute_581 = None
    view_812: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_142, [8, 784, 768]);  mm_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_1310: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_812, primals_367);  primals_367 = None
    mul_1311: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1310, 768)
    sum_411: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1310, [2], True)
    mul_1312: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1310, mul_326);  mul_1310 = None
    sum_412: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1312, [2], True);  mul_1312 = None
    mul_1313: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_326, sum_412);  sum_412 = None
    sub_298: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(mul_1311, sum_411);  mul_1311 = sum_411 = None
    sub_299: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(sub_298, mul_1313);  sub_298 = mul_1313 = None
    mul_1314: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(div_204, sub_299);  div_204 = sub_299 = None
    mul_1315: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_812, mul_326);  mul_326 = None
    sum_413: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1315, [0, 1]);  mul_1315 = None
    sum_414: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_812, [0, 1]);  view_812 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    add_566: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_563, mul_1314);  add_563 = mul_1314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_1316: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_566, primals_51);  primals_51 = None
    mul_1317: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_566, permute_117);  permute_117 = None
    sum_415: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1317, [0, 1], True);  mul_1317 = None
    view_813: "f32[768]" = torch.ops.aten.reshape.default(sum_415, [768]);  sum_415 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    permute_583: "f32[8, 768, 784]" = torch.ops.aten.permute.default(mul_1316, [0, 2, 1]);  mul_1316 = None
    view_814: "f32[8, 768, 28, 28]" = torch.ops.aten.reshape.default(permute_583, [8, 768, 28, 28]);  permute_583 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    sum_416: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_814, [0, 2, 3])
    convolution_backward_22 = torch.ops.aten.convolution_backward.default(view_814, add_238, primals_365, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False]);  view_814 = add_238 = primals_365 = None
    getitem_362: "f32[8, 768, 28, 28]" = convolution_backward_22[0]
    getitem_363: "f32[768, 1, 3, 3]" = convolution_backward_22[1];  convolution_backward_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    sum_417: "f32[768]" = torch.ops.aten.sum.dim_IntList(getitem_362, [0, 2, 3])
    sub_300: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(mul_317, unsqueeze_251);  mul_317 = unsqueeze_251 = None
    mul_1318: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_362, sub_300)
    sum_418: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1318, [0, 2, 3]);  mul_1318 = None
    mul_1319: "f32[768]" = torch.ops.aten.mul.Tensor(sum_417, 0.00015943877551020407)
    unsqueeze_252: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1319, 0);  mul_1319 = None
    unsqueeze_253: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_252, 2);  unsqueeze_252 = None
    unsqueeze_254: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_253, 3);  unsqueeze_253 = None
    mul_1320: "f32[768]" = torch.ops.aten.mul.Tensor(sum_418, 0.00015943877551020407)
    mul_1321: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_46, squeeze_46)
    mul_1322: "f32[768]" = torch.ops.aten.mul.Tensor(mul_1320, mul_1321);  mul_1320 = mul_1321 = None
    unsqueeze_255: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1322, 0);  mul_1322 = None
    unsqueeze_256: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_255, 2);  unsqueeze_255 = None
    unsqueeze_257: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_256, 3);  unsqueeze_256 = None
    mul_1323: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_46, primals_363);  primals_363 = None
    unsqueeze_258: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1323, 0);  mul_1323 = None
    unsqueeze_259: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_258, 2);  unsqueeze_258 = None
    unsqueeze_260: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_259, 3);  unsqueeze_259 = None
    mul_1324: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_300, unsqueeze_257);  sub_300 = unsqueeze_257 = None
    sub_302: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(getitem_362, mul_1324);  getitem_362 = mul_1324 = None
    sub_303: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(sub_302, unsqueeze_254);  sub_302 = unsqueeze_254 = None
    mul_1325: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_303, unsqueeze_260);  sub_303 = unsqueeze_260 = None
    mul_1326: "f32[768]" = torch.ops.aten.mul.Tensor(sum_418, squeeze_46);  sum_418 = squeeze_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_1328: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(add_233, 0.5);  add_233 = None
    mul_1329: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_28, convolution_28)
    mul_1330: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_1329, -0.5);  mul_1329 = None
    exp_49: "f32[8, 768, 28, 28]" = torch.ops.aten.exp.default(mul_1330);  mul_1330 = None
    mul_1331: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(exp_49, 0.3989422804014327);  exp_49 = None
    mul_1332: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_28, mul_1331);  convolution_28 = mul_1331 = None
    add_568: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(mul_1328, mul_1332);  mul_1328 = mul_1332 = None
    mul_1333: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_1325, add_568);  mul_1325 = add_568 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    sum_419: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1333, [0, 2, 3])
    convolution_backward_23 = torch.ops.aten.convolution_backward.default(mul_1333, view_232, primals_361, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False]);  mul_1333 = view_232 = primals_361 = None
    getitem_365: "f32[8, 768, 28, 28]" = convolution_backward_23[0]
    getitem_366: "f32[768, 1, 3, 3]" = convolution_backward_23[1];  convolution_backward_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    view_815: "f32[8, 768, 784]" = torch.ops.aten.reshape.default(getitem_365, [8, 768, 784]);  getitem_365 = None
    permute_584: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_815, [0, 2, 1]);  view_815 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    clone_361: "f32[8, 784, 768]" = torch.ops.aten.clone.default(permute_584, memory_format = torch.contiguous_format);  permute_584 = None
    mul_1335: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(clone_361, primals_359);  primals_359 = None
    mul_1336: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1335, 768)
    sum_420: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1335, [2], True)
    mul_1337: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1335, mul_313);  mul_1335 = None
    sum_421: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1337, [2], True);  mul_1337 = None
    mul_1338: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_313, sum_421);  sum_421 = None
    sub_305: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(mul_1336, sum_420);  mul_1336 = sum_420 = None
    sub_306: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(sub_305, mul_1338);  sub_305 = mul_1338 = None
    mul_1339: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(div_205, sub_306);  div_205 = sub_306 = None
    mul_1340: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(clone_361, mul_313);  mul_313 = None
    sum_422: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1340, [0, 1]);  mul_1340 = None
    sum_423: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_361, [0, 1]);  clone_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    add_569: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_566, mul_1339);  add_566 = mul_1339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_1341: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_569, primals_49);  primals_49 = None
    mul_1342: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_569, add_229);  add_229 = None
    sum_424: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1342, [0, 1], True);  mul_1342 = None
    view_816: "f32[768]" = torch.ops.aten.reshape.default(sum_424, [768]);  sum_424 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    sum_425: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1341, [0, 1], True)
    view_817: "f32[768]" = torch.ops.aten.reshape.default(sum_425, [768]);  sum_425 = None
    view_818: "f32[6272, 768]" = torch.ops.aten.reshape.default(mul_1341, [6272, 768]);  mul_1341 = None
    permute_585: "f32[768, 6272]" = torch.ops.aten.permute.default(view_818, [1, 0])
    mm_144: "f32[768, 768]" = torch.ops.aten.mm.default(permute_585, view_230);  permute_585 = view_230 = None
    permute_586: "f32[768, 768]" = torch.ops.aten.permute.default(mm_144, [1, 0]);  mm_144 = None
    mm_145: "f32[6272, 768]" = torch.ops.aten.mm.default(view_818, permute_587);  view_818 = permute_587 = None
    view_819: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_145, [8, 784, 768]);  mm_145 = None
    permute_588: "f32[768, 768]" = torch.ops.aten.permute.default(permute_586, [1, 0]);  permute_586 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    view_820: "f32[8, 784, 16, 48]" = torch.ops.aten.reshape.default(view_819, [8, 784, 16, 48]);  view_819 = None
    permute_589: "f32[8, 16, 48, 784]" = torch.ops.aten.permute.default(view_820, [0, 2, 3, 1]);  view_820 = None
    clone_363: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(permute_589, memory_format = torch.contiguous_format);  permute_589 = None
    view_821: "f32[128, 48, 784]" = torch.ops.aten.reshape.default(clone_363, [128, 48, 784]);  clone_363 = None
    bmm_92: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(permute_590, view_821);  permute_590 = None
    bmm_93: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(view_821, permute_591);  view_821 = permute_591 = None
    view_822: "f32[8, 16, 48, 784]" = torch.ops.aten.reshape.default(bmm_92, [8, 16, 48, 784]);  bmm_92 = None
    view_823: "f32[8, 16, 48, 48]" = torch.ops.aten.reshape.default(bmm_93, [8, 16, 48, 48]);  bmm_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    mul_1343: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_823, alias_109);  view_823 = None
    sum_426: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_1343, [-1], True)
    mul_1344: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(alias_109, sum_426);  alias_109 = sum_426 = None
    sub_307: "f32[8, 16, 48, 48]" = torch.ops.aten.sub.Tensor(mul_1343, mul_1344);  mul_1343 = mul_1344 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    mul_1345: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(sub_307, view_225);  view_225 = None
    mul_1346: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(sub_307, primals_50);  sub_307 = primals_50 = None
    sum_427: "f32[1, 16, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1345, [0, 2, 3], True);  mul_1345 = None
    view_824: "f32[16, 1, 1]" = torch.ops.aten.reshape.default(sum_427, [16, 1, 1]);  sum_427 = None
    view_825: "f32[128, 48, 48]" = torch.ops.aten.reshape.default(mul_1346, [128, 48, 48]);  mul_1346 = None
    bmm_94: "f32[128, 784, 48]" = torch.ops.aten.bmm.default(permute_592, view_825);  permute_592 = None
    bmm_95: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_825, permute_593);  view_825 = permute_593 = None
    view_826: "f32[8, 16, 784, 48]" = torch.ops.aten.reshape.default(bmm_94, [8, 16, 784, 48]);  bmm_94 = None
    view_827: "f32[8, 16, 48, 784]" = torch.ops.aten.reshape.default(bmm_95, [8, 16, 48, 784]);  bmm_95 = None
    permute_594: "f32[8, 16, 48, 784]" = torch.ops.aten.permute.default(view_826, [0, 1, 3, 2]);  view_826 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    div_207: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_43, expand_73);  div_43 = None
    neg_22: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(permute_594)
    mul_1347: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_22, div_207);  neg_22 = div_207 = None
    div_208: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(permute_594, expand_73);  permute_594 = expand_73 = None
    sum_428: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_1347, [3], True);  mul_1347 = None
    ge_22: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(pow_53, 1e-12)
    where_44: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_22, sum_428, full_default_20);  ge_22 = sum_428 = None
    div_209: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_141, pow_53);  getitem_141 = None
    eq_22: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(pow_53, 0);  pow_53 = None
    where_45: "f32[8, 16, 48, 784]" = torch.ops.aten.where.self(eq_22, full_default_20, div_209);  eq_22 = div_209 = None
    clone_364: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(where_45, memory_format = torch.contiguous_format);  where_45 = None
    mul_1348: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_44, clone_364);  where_44 = clone_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    add_570: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_208, mul_1348);  div_208 = mul_1348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    div_211: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_42, expand_72);  div_42 = None
    neg_23: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(view_827)
    mul_1349: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_23, div_211);  neg_23 = div_211 = None
    div_212: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(view_827, expand_72);  view_827 = expand_72 = None
    sum_429: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_1349, [3], True);  mul_1349 = None
    ge_23: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(pow_51, 1e-12)
    where_46: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_23, sum_429, full_default_20);  ge_23 = sum_429 = None
    div_213: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_140, pow_51);  getitem_140 = None
    eq_23: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(pow_51, 0);  pow_51 = None
    where_47: "f32[8, 16, 48, 784]" = torch.ops.aten.where.self(eq_23, full_default_20, div_213);  eq_23 = div_213 = None
    clone_365: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(where_47, memory_format = torch.contiguous_format);  where_47 = None
    mul_1350: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_46, clone_365);  where_46 = clone_365 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    add_571: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_212, mul_1350);  div_212 = mul_1350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    cat_19: "f32[24, 16, 48, 784]" = torch.ops.aten.cat.default([add_571, add_570, view_822]);  add_571 = add_570 = view_822 = None
    view_828: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.reshape.default(cat_19, [3, 8, 16, 48, 784]);  cat_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    permute_595: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.permute.default(view_828, [1, 4, 0, 2, 3]);  view_828 = None
    clone_366: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.clone.default(permute_595, memory_format = torch.contiguous_format);  permute_595 = None
    view_829: "f32[8, 784, 2304]" = torch.ops.aten.reshape.default(clone_366, [8, 784, 2304]);  clone_366 = None
    view_830: "f32[6272, 2304]" = torch.ops.aten.reshape.default(view_829, [6272, 2304]);  view_829 = None
    mm_146: "f32[6272, 768]" = torch.ops.aten.mm.default(view_830, permute_596);  permute_596 = None
    permute_597: "f32[2304, 6272]" = torch.ops.aten.permute.default(view_830, [1, 0])
    mm_147: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_597, view_220);  permute_597 = view_220 = None
    permute_598: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_147, [1, 0]);  mm_147 = None
    sum_430: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_830, [0], True);  view_830 = None
    view_831: "f32[2304]" = torch.ops.aten.reshape.default(sum_430, [2304]);  sum_430 = None
    permute_599: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_598, [1, 0]);  permute_598 = None
    view_832: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_146, [8, 784, 768]);  mm_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_1352: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_832, primals_353);  primals_353 = None
    mul_1353: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1352, 768)
    sum_431: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1352, [2], True)
    mul_1354: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1352, mul_309);  mul_1352 = None
    sum_432: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1354, [2], True);  mul_1354 = None
    mul_1355: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_309, sum_432);  sum_432 = None
    sub_309: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(mul_1353, sum_431);  mul_1353 = sum_431 = None
    sub_310: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(sub_309, mul_1355);  sub_309 = mul_1355 = None
    mul_1356: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(div_214, sub_310);  div_214 = sub_310 = None
    mul_1357: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_832, mul_309);  mul_309 = None
    sum_433: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1357, [0, 1]);  mul_1357 = None
    sum_434: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_832, [0, 1]);  view_832 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    add_572: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_569, mul_1356);  add_569 = mul_1356 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_1358: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_572, primals_48);  primals_48 = None
    mul_1359: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_572, view_219);  view_219 = None
    sum_435: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1359, [0, 1], True);  mul_1359 = None
    view_833: "f32[768]" = torch.ops.aten.reshape.default(sum_435, [768]);  sum_435 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_834: "f32[6272, 768]" = torch.ops.aten.reshape.default(mul_1358, [6272, 768]);  mul_1358 = None
    mm_148: "f32[6272, 3072]" = torch.ops.aten.mm.default(view_834, permute_600);  permute_600 = None
    permute_601: "f32[768, 6272]" = torch.ops.aten.permute.default(view_834, [1, 0])
    mm_149: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_601, view_218);  permute_601 = view_218 = None
    permute_602: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_149, [1, 0]);  mm_149 = None
    sum_436: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_834, [0], True);  view_834 = None
    view_835: "f32[768]" = torch.ops.aten.reshape.default(sum_436, [768]);  sum_436 = None
    permute_603: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_602, [1, 0]);  permute_602 = None
    view_836: "f32[8, 784, 3072]" = torch.ops.aten.reshape.default(mm_148, [8, 784, 3072]);  mm_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_1361: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(add_225, 0.5);  add_225 = None
    mul_1362: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_217, view_217)
    mul_1363: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(mul_1362, -0.5);  mul_1362 = None
    exp_50: "f32[8, 784, 3072]" = torch.ops.aten.exp.default(mul_1363);  mul_1363 = None
    mul_1364: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(exp_50, 0.3989422804014327);  exp_50 = None
    mul_1365: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_217, mul_1364);  view_217 = mul_1364 = None
    add_574: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(mul_1361, mul_1365);  mul_1361 = mul_1365 = None
    mul_1366: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_836, add_574);  view_836 = add_574 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_837: "f32[6272, 3072]" = torch.ops.aten.reshape.default(mul_1366, [6272, 3072]);  mul_1366 = None
    mm_150: "f32[6272, 768]" = torch.ops.aten.mm.default(view_837, permute_604);  permute_604 = None
    permute_605: "f32[3072, 6272]" = torch.ops.aten.permute.default(view_837, [1, 0])
    mm_151: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_605, view_216);  permute_605 = view_216 = None
    permute_606: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_151, [1, 0]);  mm_151 = None
    sum_437: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_837, [0], True);  view_837 = None
    view_838: "f32[3072]" = torch.ops.aten.reshape.default(sum_437, [3072]);  sum_437 = None
    permute_607: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_606, [1, 0]);  permute_606 = None
    view_839: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_150, [8, 784, 768]);  mm_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_1368: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_839, primals_347);  primals_347 = None
    mul_1369: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1368, 768)
    sum_438: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1368, [2], True)
    mul_1370: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1368, mul_303);  mul_1368 = None
    sum_439: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1370, [2], True);  mul_1370 = None
    mul_1371: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_303, sum_439);  sum_439 = None
    sub_312: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(mul_1369, sum_438);  mul_1369 = sum_438 = None
    sub_313: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(sub_312, mul_1371);  sub_312 = mul_1371 = None
    mul_1372: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(div_215, sub_313);  div_215 = sub_313 = None
    mul_1373: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_839, mul_303);  mul_303 = None
    sum_440: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1373, [0, 1]);  mul_1373 = None
    sum_441: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_839, [0, 1]);  view_839 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    add_575: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_572, mul_1372);  add_572 = mul_1372 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_1374: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_575, primals_47);  primals_47 = None
    mul_1375: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_575, permute_108);  permute_108 = None
    sum_442: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1375, [0, 1], True);  mul_1375 = None
    view_840: "f32[768]" = torch.ops.aten.reshape.default(sum_442, [768]);  sum_442 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    permute_608: "f32[8, 768, 784]" = torch.ops.aten.permute.default(mul_1374, [0, 2, 1]);  mul_1374 = None
    view_841: "f32[8, 768, 28, 28]" = torch.ops.aten.reshape.default(permute_608, [8, 768, 28, 28]);  permute_608 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    sum_443: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_841, [0, 2, 3])
    convolution_backward_24 = torch.ops.aten.convolution_backward.default(view_841, add_221, primals_345, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False]);  view_841 = add_221 = primals_345 = None
    getitem_368: "f32[8, 768, 28, 28]" = convolution_backward_24[0]
    getitem_369: "f32[768, 1, 3, 3]" = convolution_backward_24[1];  convolution_backward_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    sum_444: "f32[768]" = torch.ops.aten.sum.dim_IntList(getitem_368, [0, 2, 3])
    sub_314: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(mul_294, unsqueeze_263);  mul_294 = unsqueeze_263 = None
    mul_1376: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_368, sub_314)
    sum_445: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1376, [0, 2, 3]);  mul_1376 = None
    mul_1377: "f32[768]" = torch.ops.aten.mul.Tensor(sum_444, 0.00015943877551020407)
    unsqueeze_264: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1377, 0);  mul_1377 = None
    unsqueeze_265: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_264, 2);  unsqueeze_264 = None
    unsqueeze_266: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_265, 3);  unsqueeze_265 = None
    mul_1378: "f32[768]" = torch.ops.aten.mul.Tensor(sum_445, 0.00015943877551020407)
    mul_1379: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
    mul_1380: "f32[768]" = torch.ops.aten.mul.Tensor(mul_1378, mul_1379);  mul_1378 = mul_1379 = None
    unsqueeze_267: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1380, 0);  mul_1380 = None
    unsqueeze_268: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_267, 2);  unsqueeze_267 = None
    unsqueeze_269: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_268, 3);  unsqueeze_268 = None
    mul_1381: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_43, primals_343);  primals_343 = None
    unsqueeze_270: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1381, 0);  mul_1381 = None
    unsqueeze_271: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_270, 2);  unsqueeze_270 = None
    unsqueeze_272: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_271, 3);  unsqueeze_271 = None
    mul_1382: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_314, unsqueeze_269);  sub_314 = unsqueeze_269 = None
    sub_316: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(getitem_368, mul_1382);  getitem_368 = mul_1382 = None
    sub_317: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(sub_316, unsqueeze_266);  sub_316 = unsqueeze_266 = None
    mul_1383: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_317, unsqueeze_272);  sub_317 = unsqueeze_272 = None
    mul_1384: "f32[768]" = torch.ops.aten.mul.Tensor(sum_445, squeeze_43);  sum_445 = squeeze_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_1386: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(add_216, 0.5);  add_216 = None
    mul_1387: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_26, convolution_26)
    mul_1388: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_1387, -0.5);  mul_1387 = None
    exp_51: "f32[8, 768, 28, 28]" = torch.ops.aten.exp.default(mul_1388);  mul_1388 = None
    mul_1389: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(exp_51, 0.3989422804014327);  exp_51 = None
    mul_1390: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_26, mul_1389);  convolution_26 = mul_1389 = None
    add_577: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(mul_1386, mul_1390);  mul_1386 = mul_1390 = None
    mul_1391: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_1383, add_577);  mul_1383 = add_577 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    sum_446: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1391, [0, 2, 3])
    convolution_backward_25 = torch.ops.aten.convolution_backward.default(mul_1391, view_214, primals_341, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False]);  mul_1391 = view_214 = primals_341 = None
    getitem_371: "f32[8, 768, 28, 28]" = convolution_backward_25[0]
    getitem_372: "f32[768, 1, 3, 3]" = convolution_backward_25[1];  convolution_backward_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    view_842: "f32[8, 768, 784]" = torch.ops.aten.reshape.default(getitem_371, [8, 768, 784]);  getitem_371 = None
    permute_609: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_842, [0, 2, 1]);  view_842 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    clone_369: "f32[8, 784, 768]" = torch.ops.aten.clone.default(permute_609, memory_format = torch.contiguous_format);  permute_609 = None
    mul_1393: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(clone_369, primals_339);  primals_339 = None
    mul_1394: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1393, 768)
    sum_447: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1393, [2], True)
    mul_1395: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1393, mul_290);  mul_1393 = None
    sum_448: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1395, [2], True);  mul_1395 = None
    mul_1396: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_290, sum_448);  sum_448 = None
    sub_319: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(mul_1394, sum_447);  mul_1394 = sum_447 = None
    sub_320: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(sub_319, mul_1396);  sub_319 = mul_1396 = None
    mul_1397: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(div_216, sub_320);  div_216 = sub_320 = None
    mul_1398: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(clone_369, mul_290);  mul_290 = None
    sum_449: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1398, [0, 1]);  mul_1398 = None
    sum_450: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_369, [0, 1]);  clone_369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    add_578: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_575, mul_1397);  add_575 = mul_1397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_1399: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_578, primals_45);  primals_45 = None
    mul_1400: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_578, add_212);  add_212 = None
    sum_451: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1400, [0, 1], True);  mul_1400 = None
    view_843: "f32[768]" = torch.ops.aten.reshape.default(sum_451, [768]);  sum_451 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    sum_452: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1399, [0, 1], True)
    view_844: "f32[768]" = torch.ops.aten.reshape.default(sum_452, [768]);  sum_452 = None
    view_845: "f32[6272, 768]" = torch.ops.aten.reshape.default(mul_1399, [6272, 768]);  mul_1399 = None
    permute_610: "f32[768, 6272]" = torch.ops.aten.permute.default(view_845, [1, 0])
    mm_152: "f32[768, 768]" = torch.ops.aten.mm.default(permute_610, view_212);  permute_610 = view_212 = None
    permute_611: "f32[768, 768]" = torch.ops.aten.permute.default(mm_152, [1, 0]);  mm_152 = None
    mm_153: "f32[6272, 768]" = torch.ops.aten.mm.default(view_845, permute_612);  view_845 = permute_612 = None
    view_846: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_153, [8, 784, 768]);  mm_153 = None
    permute_613: "f32[768, 768]" = torch.ops.aten.permute.default(permute_611, [1, 0]);  permute_611 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    view_847: "f32[8, 784, 16, 48]" = torch.ops.aten.reshape.default(view_846, [8, 784, 16, 48]);  view_846 = None
    permute_614: "f32[8, 16, 48, 784]" = torch.ops.aten.permute.default(view_847, [0, 2, 3, 1]);  view_847 = None
    clone_371: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(permute_614, memory_format = torch.contiguous_format);  permute_614 = None
    view_848: "f32[128, 48, 784]" = torch.ops.aten.reshape.default(clone_371, [128, 48, 784]);  clone_371 = None
    bmm_96: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(permute_615, view_848);  permute_615 = None
    bmm_97: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(view_848, permute_616);  view_848 = permute_616 = None
    view_849: "f32[8, 16, 48, 784]" = torch.ops.aten.reshape.default(bmm_96, [8, 16, 48, 784]);  bmm_96 = None
    view_850: "f32[8, 16, 48, 48]" = torch.ops.aten.reshape.default(bmm_97, [8, 16, 48, 48]);  bmm_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    mul_1401: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_850, alias_112);  view_850 = None
    sum_453: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_1401, [-1], True)
    mul_1402: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(alias_112, sum_453);  alias_112 = sum_453 = None
    sub_321: "f32[8, 16, 48, 48]" = torch.ops.aten.sub.Tensor(mul_1401, mul_1402);  mul_1401 = mul_1402 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    mul_1403: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(sub_321, view_207);  view_207 = None
    mul_1404: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(sub_321, primals_46);  sub_321 = primals_46 = None
    sum_454: "f32[1, 16, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1403, [0, 2, 3], True);  mul_1403 = None
    view_851: "f32[16, 1, 1]" = torch.ops.aten.reshape.default(sum_454, [16, 1, 1]);  sum_454 = None
    view_852: "f32[128, 48, 48]" = torch.ops.aten.reshape.default(mul_1404, [128, 48, 48]);  mul_1404 = None
    bmm_98: "f32[128, 784, 48]" = torch.ops.aten.bmm.default(permute_617, view_852);  permute_617 = None
    bmm_99: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_852, permute_618);  view_852 = permute_618 = None
    view_853: "f32[8, 16, 784, 48]" = torch.ops.aten.reshape.default(bmm_98, [8, 16, 784, 48]);  bmm_98 = None
    view_854: "f32[8, 16, 48, 784]" = torch.ops.aten.reshape.default(bmm_99, [8, 16, 48, 784]);  bmm_99 = None
    permute_619: "f32[8, 16, 48, 784]" = torch.ops.aten.permute.default(view_853, [0, 1, 3, 2]);  view_853 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    div_218: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_40, expand_67);  div_40 = None
    neg_24: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(permute_619)
    mul_1405: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_24, div_218);  neg_24 = div_218 = None
    div_219: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(permute_619, expand_67);  permute_619 = expand_67 = None
    sum_455: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_1405, [3], True);  mul_1405 = None
    ge_24: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(pow_49, 1e-12)
    where_48: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_24, sum_455, full_default_20);  ge_24 = sum_455 = None
    div_220: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_130, pow_49);  getitem_130 = None
    eq_24: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(pow_49, 0);  pow_49 = None
    where_49: "f32[8, 16, 48, 784]" = torch.ops.aten.where.self(eq_24, full_default_20, div_220);  eq_24 = div_220 = None
    clone_372: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(where_49, memory_format = torch.contiguous_format);  where_49 = None
    mul_1406: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_48, clone_372);  where_48 = clone_372 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    add_579: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_219, mul_1406);  div_219 = mul_1406 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    div_222: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_39, expand_66);  div_39 = None
    neg_25: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(view_854)
    mul_1407: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_25, div_222);  neg_25 = div_222 = None
    div_223: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(view_854, expand_66);  view_854 = expand_66 = None
    sum_456: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_1407, [3], True);  mul_1407 = None
    ge_25: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(pow_47, 1e-12)
    where_50: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_25, sum_456, full_default_20);  ge_25 = sum_456 = None
    div_224: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_129, pow_47);  getitem_129 = None
    eq_25: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(pow_47, 0);  pow_47 = None
    where_51: "f32[8, 16, 48, 784]" = torch.ops.aten.where.self(eq_25, full_default_20, div_224);  eq_25 = div_224 = None
    clone_373: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(where_51, memory_format = torch.contiguous_format);  where_51 = None
    mul_1408: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_50, clone_373);  where_50 = clone_373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    add_580: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_223, mul_1408);  div_223 = mul_1408 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    cat_20: "f32[24, 16, 48, 784]" = torch.ops.aten.cat.default([add_580, add_579, view_849]);  add_580 = add_579 = view_849 = None
    view_855: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.reshape.default(cat_20, [3, 8, 16, 48, 784]);  cat_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    permute_620: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.permute.default(view_855, [1, 4, 0, 2, 3]);  view_855 = None
    clone_374: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.clone.default(permute_620, memory_format = torch.contiguous_format);  permute_620 = None
    view_856: "f32[8, 784, 2304]" = torch.ops.aten.reshape.default(clone_374, [8, 784, 2304]);  clone_374 = None
    view_857: "f32[6272, 2304]" = torch.ops.aten.reshape.default(view_856, [6272, 2304]);  view_856 = None
    mm_154: "f32[6272, 768]" = torch.ops.aten.mm.default(view_857, permute_621);  permute_621 = None
    permute_622: "f32[2304, 6272]" = torch.ops.aten.permute.default(view_857, [1, 0])
    mm_155: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_622, view_202);  permute_622 = view_202 = None
    permute_623: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_155, [1, 0]);  mm_155 = None
    sum_457: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_857, [0], True);  view_857 = None
    view_858: "f32[2304]" = torch.ops.aten.reshape.default(sum_457, [2304]);  sum_457 = None
    permute_624: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_623, [1, 0]);  permute_623 = None
    view_859: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_154, [8, 784, 768]);  mm_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_1410: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_859, primals_333);  primals_333 = None
    mul_1411: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1410, 768)
    sum_458: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1410, [2], True)
    mul_1412: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1410, mul_286);  mul_1410 = None
    sum_459: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1412, [2], True);  mul_1412 = None
    mul_1413: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_286, sum_459);  sum_459 = None
    sub_323: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(mul_1411, sum_458);  mul_1411 = sum_458 = None
    sub_324: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(sub_323, mul_1413);  sub_323 = mul_1413 = None
    mul_1414: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(div_225, sub_324);  div_225 = sub_324 = None
    mul_1415: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_859, mul_286);  mul_286 = None
    sum_460: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1415, [0, 1]);  mul_1415 = None
    sum_461: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_859, [0, 1]);  view_859 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    add_581: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_578, mul_1414);  add_578 = mul_1414 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_1416: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_581, primals_44);  primals_44 = None
    mul_1417: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_581, view_201);  view_201 = None
    sum_462: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1417, [0, 1], True);  mul_1417 = None
    view_860: "f32[768]" = torch.ops.aten.reshape.default(sum_462, [768]);  sum_462 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_861: "f32[6272, 768]" = torch.ops.aten.reshape.default(mul_1416, [6272, 768]);  mul_1416 = None
    mm_156: "f32[6272, 3072]" = torch.ops.aten.mm.default(view_861, permute_625);  permute_625 = None
    permute_626: "f32[768, 6272]" = torch.ops.aten.permute.default(view_861, [1, 0])
    mm_157: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_626, view_200);  permute_626 = view_200 = None
    permute_627: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_157, [1, 0]);  mm_157 = None
    sum_463: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_861, [0], True);  view_861 = None
    view_862: "f32[768]" = torch.ops.aten.reshape.default(sum_463, [768]);  sum_463 = None
    permute_628: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_627, [1, 0]);  permute_627 = None
    view_863: "f32[8, 784, 3072]" = torch.ops.aten.reshape.default(mm_156, [8, 784, 3072]);  mm_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_1419: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(add_208, 0.5);  add_208 = None
    mul_1420: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_199, view_199)
    mul_1421: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(mul_1420, -0.5);  mul_1420 = None
    exp_52: "f32[8, 784, 3072]" = torch.ops.aten.exp.default(mul_1421);  mul_1421 = None
    mul_1422: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(exp_52, 0.3989422804014327);  exp_52 = None
    mul_1423: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_199, mul_1422);  view_199 = mul_1422 = None
    add_583: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(mul_1419, mul_1423);  mul_1419 = mul_1423 = None
    mul_1424: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_863, add_583);  view_863 = add_583 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_864: "f32[6272, 3072]" = torch.ops.aten.reshape.default(mul_1424, [6272, 3072]);  mul_1424 = None
    mm_158: "f32[6272, 768]" = torch.ops.aten.mm.default(view_864, permute_629);  permute_629 = None
    permute_630: "f32[3072, 6272]" = torch.ops.aten.permute.default(view_864, [1, 0])
    mm_159: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_630, view_198);  permute_630 = view_198 = None
    permute_631: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_159, [1, 0]);  mm_159 = None
    sum_464: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_864, [0], True);  view_864 = None
    view_865: "f32[3072]" = torch.ops.aten.reshape.default(sum_464, [3072]);  sum_464 = None
    permute_632: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_631, [1, 0]);  permute_631 = None
    view_866: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_158, [8, 784, 768]);  mm_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_1426: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_866, primals_327);  primals_327 = None
    mul_1427: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1426, 768)
    sum_465: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1426, [2], True)
    mul_1428: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1426, mul_280);  mul_1426 = None
    sum_466: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1428, [2], True);  mul_1428 = None
    mul_1429: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_280, sum_466);  sum_466 = None
    sub_326: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(mul_1427, sum_465);  mul_1427 = sum_465 = None
    sub_327: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(sub_326, mul_1429);  sub_326 = mul_1429 = None
    mul_1430: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(div_226, sub_327);  div_226 = sub_327 = None
    mul_1431: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_866, mul_280);  mul_280 = None
    sum_467: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1431, [0, 1]);  mul_1431 = None
    sum_468: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_866, [0, 1]);  view_866 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    add_584: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_581, mul_1430);  add_581 = mul_1430 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_1432: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_584, primals_43);  primals_43 = None
    mul_1433: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_584, permute_99);  permute_99 = None
    sum_469: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1433, [0, 1], True);  mul_1433 = None
    view_867: "f32[768]" = torch.ops.aten.reshape.default(sum_469, [768]);  sum_469 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    permute_633: "f32[8, 768, 784]" = torch.ops.aten.permute.default(mul_1432, [0, 2, 1]);  mul_1432 = None
    view_868: "f32[8, 768, 28, 28]" = torch.ops.aten.reshape.default(permute_633, [8, 768, 28, 28]);  permute_633 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    sum_470: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_868, [0, 2, 3])
    convolution_backward_26 = torch.ops.aten.convolution_backward.default(view_868, add_204, primals_325, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False]);  view_868 = add_204 = primals_325 = None
    getitem_374: "f32[8, 768, 28, 28]" = convolution_backward_26[0]
    getitem_375: "f32[768, 1, 3, 3]" = convolution_backward_26[1];  convolution_backward_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    sum_471: "f32[768]" = torch.ops.aten.sum.dim_IntList(getitem_374, [0, 2, 3])
    sub_328: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(mul_271, unsqueeze_275);  mul_271 = unsqueeze_275 = None
    mul_1434: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_374, sub_328)
    sum_472: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1434, [0, 2, 3]);  mul_1434 = None
    mul_1435: "f32[768]" = torch.ops.aten.mul.Tensor(sum_471, 0.00015943877551020407)
    unsqueeze_276: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1435, 0);  mul_1435 = None
    unsqueeze_277: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_276, 2);  unsqueeze_276 = None
    unsqueeze_278: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_277, 3);  unsqueeze_277 = None
    mul_1436: "f32[768]" = torch.ops.aten.mul.Tensor(sum_472, 0.00015943877551020407)
    mul_1437: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_40, squeeze_40)
    mul_1438: "f32[768]" = torch.ops.aten.mul.Tensor(mul_1436, mul_1437);  mul_1436 = mul_1437 = None
    unsqueeze_279: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1438, 0);  mul_1438 = None
    unsqueeze_280: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_279, 2);  unsqueeze_279 = None
    unsqueeze_281: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_280, 3);  unsqueeze_280 = None
    mul_1439: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_40, primals_323);  primals_323 = None
    unsqueeze_282: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1439, 0);  mul_1439 = None
    unsqueeze_283: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_282, 2);  unsqueeze_282 = None
    unsqueeze_284: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_283, 3);  unsqueeze_283 = None
    mul_1440: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_328, unsqueeze_281);  sub_328 = unsqueeze_281 = None
    sub_330: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(getitem_374, mul_1440);  getitem_374 = mul_1440 = None
    sub_331: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(sub_330, unsqueeze_278);  sub_330 = unsqueeze_278 = None
    mul_1441: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_331, unsqueeze_284);  sub_331 = unsqueeze_284 = None
    mul_1442: "f32[768]" = torch.ops.aten.mul.Tensor(sum_472, squeeze_40);  sum_472 = squeeze_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_1444: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(add_199, 0.5);  add_199 = None
    mul_1445: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_24, convolution_24)
    mul_1446: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_1445, -0.5);  mul_1445 = None
    exp_53: "f32[8, 768, 28, 28]" = torch.ops.aten.exp.default(mul_1446);  mul_1446 = None
    mul_1447: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(exp_53, 0.3989422804014327);  exp_53 = None
    mul_1448: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_24, mul_1447);  convolution_24 = mul_1447 = None
    add_586: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(mul_1444, mul_1448);  mul_1444 = mul_1448 = None
    mul_1449: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_1441, add_586);  mul_1441 = add_586 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    sum_473: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1449, [0, 2, 3])
    convolution_backward_27 = torch.ops.aten.convolution_backward.default(mul_1449, view_196, primals_321, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False]);  mul_1449 = view_196 = primals_321 = None
    getitem_377: "f32[8, 768, 28, 28]" = convolution_backward_27[0]
    getitem_378: "f32[768, 1, 3, 3]" = convolution_backward_27[1];  convolution_backward_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    view_869: "f32[8, 768, 784]" = torch.ops.aten.reshape.default(getitem_377, [8, 768, 784]);  getitem_377 = None
    permute_634: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_869, [0, 2, 1]);  view_869 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    clone_377: "f32[8, 784, 768]" = torch.ops.aten.clone.default(permute_634, memory_format = torch.contiguous_format);  permute_634 = None
    mul_1451: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(clone_377, primals_319);  primals_319 = None
    mul_1452: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1451, 768)
    sum_474: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1451, [2], True)
    mul_1453: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1451, mul_267);  mul_1451 = None
    sum_475: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1453, [2], True);  mul_1453 = None
    mul_1454: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_267, sum_475);  sum_475 = None
    sub_333: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(mul_1452, sum_474);  mul_1452 = sum_474 = None
    sub_334: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(sub_333, mul_1454);  sub_333 = mul_1454 = None
    mul_1455: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(div_227, sub_334);  div_227 = sub_334 = None
    mul_1456: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(clone_377, mul_267);  mul_267 = None
    sum_476: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1456, [0, 1]);  mul_1456 = None
    sum_477: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_377, [0, 1]);  clone_377 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    add_587: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_584, mul_1455);  add_584 = mul_1455 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_1457: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_587, primals_41);  primals_41 = None
    mul_1458: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_587, add_195);  add_195 = None
    sum_478: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1458, [0, 1], True);  mul_1458 = None
    view_870: "f32[768]" = torch.ops.aten.reshape.default(sum_478, [768]);  sum_478 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    sum_479: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1457, [0, 1], True)
    view_871: "f32[768]" = torch.ops.aten.reshape.default(sum_479, [768]);  sum_479 = None
    view_872: "f32[6272, 768]" = torch.ops.aten.reshape.default(mul_1457, [6272, 768]);  mul_1457 = None
    permute_635: "f32[768, 6272]" = torch.ops.aten.permute.default(view_872, [1, 0])
    mm_160: "f32[768, 768]" = torch.ops.aten.mm.default(permute_635, view_194);  permute_635 = view_194 = None
    permute_636: "f32[768, 768]" = torch.ops.aten.permute.default(mm_160, [1, 0]);  mm_160 = None
    mm_161: "f32[6272, 768]" = torch.ops.aten.mm.default(view_872, permute_637);  view_872 = permute_637 = None
    view_873: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_161, [8, 784, 768]);  mm_161 = None
    permute_638: "f32[768, 768]" = torch.ops.aten.permute.default(permute_636, [1, 0]);  permute_636 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    view_874: "f32[8, 784, 16, 48]" = torch.ops.aten.reshape.default(view_873, [8, 784, 16, 48]);  view_873 = None
    permute_639: "f32[8, 16, 48, 784]" = torch.ops.aten.permute.default(view_874, [0, 2, 3, 1]);  view_874 = None
    clone_379: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(permute_639, memory_format = torch.contiguous_format);  permute_639 = None
    view_875: "f32[128, 48, 784]" = torch.ops.aten.reshape.default(clone_379, [128, 48, 784]);  clone_379 = None
    bmm_100: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(permute_640, view_875);  permute_640 = None
    bmm_101: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(view_875, permute_641);  view_875 = permute_641 = None
    view_876: "f32[8, 16, 48, 784]" = torch.ops.aten.reshape.default(bmm_100, [8, 16, 48, 784]);  bmm_100 = None
    view_877: "f32[8, 16, 48, 48]" = torch.ops.aten.reshape.default(bmm_101, [8, 16, 48, 48]);  bmm_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    mul_1459: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_877, alias_115);  view_877 = None
    sum_480: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_1459, [-1], True)
    mul_1460: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(alias_115, sum_480);  alias_115 = sum_480 = None
    sub_335: "f32[8, 16, 48, 48]" = torch.ops.aten.sub.Tensor(mul_1459, mul_1460);  mul_1459 = mul_1460 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    mul_1461: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(sub_335, view_189);  view_189 = None
    mul_1462: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(sub_335, primals_42);  sub_335 = primals_42 = None
    sum_481: "f32[1, 16, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1461, [0, 2, 3], True);  mul_1461 = None
    view_878: "f32[16, 1, 1]" = torch.ops.aten.reshape.default(sum_481, [16, 1, 1]);  sum_481 = None
    view_879: "f32[128, 48, 48]" = torch.ops.aten.reshape.default(mul_1462, [128, 48, 48]);  mul_1462 = None
    bmm_102: "f32[128, 784, 48]" = torch.ops.aten.bmm.default(permute_642, view_879);  permute_642 = None
    bmm_103: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_879, permute_643);  view_879 = permute_643 = None
    view_880: "f32[8, 16, 784, 48]" = torch.ops.aten.reshape.default(bmm_102, [8, 16, 784, 48]);  bmm_102 = None
    view_881: "f32[8, 16, 48, 784]" = torch.ops.aten.reshape.default(bmm_103, [8, 16, 48, 784]);  bmm_103 = None
    permute_644: "f32[8, 16, 48, 784]" = torch.ops.aten.permute.default(view_880, [0, 1, 3, 2]);  view_880 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    div_229: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_37, expand_61);  div_37 = None
    neg_26: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(permute_644)
    mul_1463: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_26, div_229);  neg_26 = div_229 = None
    div_230: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(permute_644, expand_61);  permute_644 = expand_61 = None
    sum_482: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_1463, [3], True);  mul_1463 = None
    ge_26: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(pow_45, 1e-12)
    where_52: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_26, sum_482, full_default_20);  ge_26 = sum_482 = None
    div_231: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_119, pow_45);  getitem_119 = None
    eq_26: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(pow_45, 0);  pow_45 = None
    where_53: "f32[8, 16, 48, 784]" = torch.ops.aten.where.self(eq_26, full_default_20, div_231);  eq_26 = div_231 = None
    clone_380: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(where_53, memory_format = torch.contiguous_format);  where_53 = None
    mul_1464: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_52, clone_380);  where_52 = clone_380 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    add_588: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_230, mul_1464);  div_230 = mul_1464 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    div_233: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_36, expand_60);  div_36 = None
    neg_27: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(view_881)
    mul_1465: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_27, div_233);  neg_27 = div_233 = None
    div_234: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(view_881, expand_60);  view_881 = expand_60 = None
    sum_483: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_1465, [3], True);  mul_1465 = None
    ge_27: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(pow_43, 1e-12)
    where_54: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_27, sum_483, full_default_20);  ge_27 = sum_483 = None
    div_235: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_118, pow_43);  getitem_118 = None
    eq_27: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(pow_43, 0);  pow_43 = None
    where_55: "f32[8, 16, 48, 784]" = torch.ops.aten.where.self(eq_27, full_default_20, div_235);  eq_27 = div_235 = None
    clone_381: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(where_55, memory_format = torch.contiguous_format);  where_55 = None
    mul_1466: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_54, clone_381);  where_54 = clone_381 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    add_589: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_234, mul_1466);  div_234 = mul_1466 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    cat_21: "f32[24, 16, 48, 784]" = torch.ops.aten.cat.default([add_589, add_588, view_876]);  add_589 = add_588 = view_876 = None
    view_882: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.reshape.default(cat_21, [3, 8, 16, 48, 784]);  cat_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    permute_645: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.permute.default(view_882, [1, 4, 0, 2, 3]);  view_882 = None
    clone_382: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.clone.default(permute_645, memory_format = torch.contiguous_format);  permute_645 = None
    view_883: "f32[8, 784, 2304]" = torch.ops.aten.reshape.default(clone_382, [8, 784, 2304]);  clone_382 = None
    view_884: "f32[6272, 2304]" = torch.ops.aten.reshape.default(view_883, [6272, 2304]);  view_883 = None
    mm_162: "f32[6272, 768]" = torch.ops.aten.mm.default(view_884, permute_646);  permute_646 = None
    permute_647: "f32[2304, 6272]" = torch.ops.aten.permute.default(view_884, [1, 0])
    mm_163: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_647, view_184);  permute_647 = view_184 = None
    permute_648: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_163, [1, 0]);  mm_163 = None
    sum_484: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_884, [0], True);  view_884 = None
    view_885: "f32[2304]" = torch.ops.aten.reshape.default(sum_484, [2304]);  sum_484 = None
    permute_649: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_648, [1, 0]);  permute_648 = None
    view_886: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_162, [8, 784, 768]);  mm_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_1468: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_886, primals_313);  primals_313 = None
    mul_1469: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1468, 768)
    sum_485: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1468, [2], True)
    mul_1470: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1468, mul_263);  mul_1468 = None
    sum_486: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1470, [2], True);  mul_1470 = None
    mul_1471: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_263, sum_486);  sum_486 = None
    sub_337: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(mul_1469, sum_485);  mul_1469 = sum_485 = None
    sub_338: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(sub_337, mul_1471);  sub_337 = mul_1471 = None
    mul_1472: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(div_236, sub_338);  div_236 = sub_338 = None
    mul_1473: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_886, mul_263);  mul_263 = None
    sum_487: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1473, [0, 1]);  mul_1473 = None
    sum_488: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_886, [0, 1]);  view_886 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    add_590: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_587, mul_1472);  add_587 = mul_1472 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_1474: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_590, primals_40);  primals_40 = None
    mul_1475: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_590, view_183);  view_183 = None
    sum_489: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1475, [0, 1], True);  mul_1475 = None
    view_887: "f32[768]" = torch.ops.aten.reshape.default(sum_489, [768]);  sum_489 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_888: "f32[6272, 768]" = torch.ops.aten.reshape.default(mul_1474, [6272, 768]);  mul_1474 = None
    mm_164: "f32[6272, 3072]" = torch.ops.aten.mm.default(view_888, permute_650);  permute_650 = None
    permute_651: "f32[768, 6272]" = torch.ops.aten.permute.default(view_888, [1, 0])
    mm_165: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_651, view_182);  permute_651 = view_182 = None
    permute_652: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_165, [1, 0]);  mm_165 = None
    sum_490: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_888, [0], True);  view_888 = None
    view_889: "f32[768]" = torch.ops.aten.reshape.default(sum_490, [768]);  sum_490 = None
    permute_653: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_652, [1, 0]);  permute_652 = None
    view_890: "f32[8, 784, 3072]" = torch.ops.aten.reshape.default(mm_164, [8, 784, 3072]);  mm_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_1477: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(add_191, 0.5);  add_191 = None
    mul_1478: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_181, view_181)
    mul_1479: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(mul_1478, -0.5);  mul_1478 = None
    exp_54: "f32[8, 784, 3072]" = torch.ops.aten.exp.default(mul_1479);  mul_1479 = None
    mul_1480: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(exp_54, 0.3989422804014327);  exp_54 = None
    mul_1481: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_181, mul_1480);  view_181 = mul_1480 = None
    add_592: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(mul_1477, mul_1481);  mul_1477 = mul_1481 = None
    mul_1482: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_890, add_592);  view_890 = add_592 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_891: "f32[6272, 3072]" = torch.ops.aten.reshape.default(mul_1482, [6272, 3072]);  mul_1482 = None
    mm_166: "f32[6272, 768]" = torch.ops.aten.mm.default(view_891, permute_654);  permute_654 = None
    permute_655: "f32[3072, 6272]" = torch.ops.aten.permute.default(view_891, [1, 0])
    mm_167: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_655, view_180);  permute_655 = view_180 = None
    permute_656: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_167, [1, 0]);  mm_167 = None
    sum_491: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_891, [0], True);  view_891 = None
    view_892: "f32[3072]" = torch.ops.aten.reshape.default(sum_491, [3072]);  sum_491 = None
    permute_657: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_656, [1, 0]);  permute_656 = None
    view_893: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_166, [8, 784, 768]);  mm_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_1484: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_893, primals_307);  primals_307 = None
    mul_1485: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1484, 768)
    sum_492: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1484, [2], True)
    mul_1486: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1484, mul_257);  mul_1484 = None
    sum_493: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1486, [2], True);  mul_1486 = None
    mul_1487: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_257, sum_493);  sum_493 = None
    sub_340: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(mul_1485, sum_492);  mul_1485 = sum_492 = None
    sub_341: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(sub_340, mul_1487);  sub_340 = mul_1487 = None
    mul_1488: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(div_237, sub_341);  div_237 = sub_341 = None
    mul_1489: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_893, mul_257);  mul_257 = None
    sum_494: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1489, [0, 1]);  mul_1489 = None
    sum_495: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_893, [0, 1]);  view_893 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    add_593: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_590, mul_1488);  add_590 = mul_1488 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_1490: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_593, primals_39);  primals_39 = None
    mul_1491: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_593, permute_90);  permute_90 = None
    sum_496: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1491, [0, 1], True);  mul_1491 = None
    view_894: "f32[768]" = torch.ops.aten.reshape.default(sum_496, [768]);  sum_496 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    permute_658: "f32[8, 768, 784]" = torch.ops.aten.permute.default(mul_1490, [0, 2, 1]);  mul_1490 = None
    view_895: "f32[8, 768, 28, 28]" = torch.ops.aten.reshape.default(permute_658, [8, 768, 28, 28]);  permute_658 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    sum_497: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_895, [0, 2, 3])
    convolution_backward_28 = torch.ops.aten.convolution_backward.default(view_895, add_187, primals_305, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False]);  view_895 = add_187 = primals_305 = None
    getitem_380: "f32[8, 768, 28, 28]" = convolution_backward_28[0]
    getitem_381: "f32[768, 1, 3, 3]" = convolution_backward_28[1];  convolution_backward_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    sum_498: "f32[768]" = torch.ops.aten.sum.dim_IntList(getitem_380, [0, 2, 3])
    sub_342: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(mul_248, unsqueeze_287);  mul_248 = unsqueeze_287 = None
    mul_1492: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_380, sub_342)
    sum_499: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1492, [0, 2, 3]);  mul_1492 = None
    mul_1493: "f32[768]" = torch.ops.aten.mul.Tensor(sum_498, 0.00015943877551020407)
    unsqueeze_288: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1493, 0);  mul_1493 = None
    unsqueeze_289: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_288, 2);  unsqueeze_288 = None
    unsqueeze_290: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_289, 3);  unsqueeze_289 = None
    mul_1494: "f32[768]" = torch.ops.aten.mul.Tensor(sum_499, 0.00015943877551020407)
    mul_1495: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
    mul_1496: "f32[768]" = torch.ops.aten.mul.Tensor(mul_1494, mul_1495);  mul_1494 = mul_1495 = None
    unsqueeze_291: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1496, 0);  mul_1496 = None
    unsqueeze_292: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_291, 2);  unsqueeze_291 = None
    unsqueeze_293: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_292, 3);  unsqueeze_292 = None
    mul_1497: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_37, primals_303);  primals_303 = None
    unsqueeze_294: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1497, 0);  mul_1497 = None
    unsqueeze_295: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_294, 2);  unsqueeze_294 = None
    unsqueeze_296: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_295, 3);  unsqueeze_295 = None
    mul_1498: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_342, unsqueeze_293);  sub_342 = unsqueeze_293 = None
    sub_344: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(getitem_380, mul_1498);  getitem_380 = mul_1498 = None
    sub_345: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(sub_344, unsqueeze_290);  sub_344 = unsqueeze_290 = None
    mul_1499: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_345, unsqueeze_296);  sub_345 = unsqueeze_296 = None
    mul_1500: "f32[768]" = torch.ops.aten.mul.Tensor(sum_499, squeeze_37);  sum_499 = squeeze_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_1502: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(add_182, 0.5);  add_182 = None
    mul_1503: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_22, convolution_22)
    mul_1504: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_1503, -0.5);  mul_1503 = None
    exp_55: "f32[8, 768, 28, 28]" = torch.ops.aten.exp.default(mul_1504);  mul_1504 = None
    mul_1505: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(exp_55, 0.3989422804014327);  exp_55 = None
    mul_1506: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_22, mul_1505);  convolution_22 = mul_1505 = None
    add_595: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(mul_1502, mul_1506);  mul_1502 = mul_1506 = None
    mul_1507: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_1499, add_595);  mul_1499 = add_595 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    sum_500: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1507, [0, 2, 3])
    convolution_backward_29 = torch.ops.aten.convolution_backward.default(mul_1507, view_178, primals_301, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False]);  mul_1507 = view_178 = primals_301 = None
    getitem_383: "f32[8, 768, 28, 28]" = convolution_backward_29[0]
    getitem_384: "f32[768, 1, 3, 3]" = convolution_backward_29[1];  convolution_backward_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    view_896: "f32[8, 768, 784]" = torch.ops.aten.reshape.default(getitem_383, [8, 768, 784]);  getitem_383 = None
    permute_659: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_896, [0, 2, 1]);  view_896 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    clone_385: "f32[8, 784, 768]" = torch.ops.aten.clone.default(permute_659, memory_format = torch.contiguous_format);  permute_659 = None
    mul_1509: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(clone_385, primals_299);  primals_299 = None
    mul_1510: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1509, 768)
    sum_501: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1509, [2], True)
    mul_1511: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1509, mul_244);  mul_1509 = None
    sum_502: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1511, [2], True);  mul_1511 = None
    mul_1512: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_244, sum_502);  sum_502 = None
    sub_347: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(mul_1510, sum_501);  mul_1510 = sum_501 = None
    sub_348: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(sub_347, mul_1512);  sub_347 = mul_1512 = None
    mul_1513: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(div_238, sub_348);  div_238 = sub_348 = None
    mul_1514: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(clone_385, mul_244);  mul_244 = None
    sum_503: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1514, [0, 1]);  mul_1514 = None
    sum_504: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_385, [0, 1]);  clone_385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    add_596: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_593, mul_1513);  add_593 = mul_1513 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_1515: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_596, primals_37);  primals_37 = None
    mul_1516: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_596, add_178);  add_178 = None
    sum_505: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1516, [0, 1], True);  mul_1516 = None
    view_897: "f32[768]" = torch.ops.aten.reshape.default(sum_505, [768]);  sum_505 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    sum_506: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1515, [0, 1], True)
    view_898: "f32[768]" = torch.ops.aten.reshape.default(sum_506, [768]);  sum_506 = None
    view_899: "f32[6272, 768]" = torch.ops.aten.reshape.default(mul_1515, [6272, 768]);  mul_1515 = None
    permute_660: "f32[768, 6272]" = torch.ops.aten.permute.default(view_899, [1, 0])
    mm_168: "f32[768, 768]" = torch.ops.aten.mm.default(permute_660, view_176);  permute_660 = view_176 = None
    permute_661: "f32[768, 768]" = torch.ops.aten.permute.default(mm_168, [1, 0]);  mm_168 = None
    mm_169: "f32[6272, 768]" = torch.ops.aten.mm.default(view_899, permute_662);  view_899 = permute_662 = None
    view_900: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_169, [8, 784, 768]);  mm_169 = None
    permute_663: "f32[768, 768]" = torch.ops.aten.permute.default(permute_661, [1, 0]);  permute_661 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    view_901: "f32[8, 784, 16, 48]" = torch.ops.aten.reshape.default(view_900, [8, 784, 16, 48]);  view_900 = None
    permute_664: "f32[8, 16, 48, 784]" = torch.ops.aten.permute.default(view_901, [0, 2, 3, 1]);  view_901 = None
    clone_387: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(permute_664, memory_format = torch.contiguous_format);  permute_664 = None
    view_902: "f32[128, 48, 784]" = torch.ops.aten.reshape.default(clone_387, [128, 48, 784]);  clone_387 = None
    bmm_104: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(permute_665, view_902);  permute_665 = None
    bmm_105: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(view_902, permute_666);  view_902 = permute_666 = None
    view_903: "f32[8, 16, 48, 784]" = torch.ops.aten.reshape.default(bmm_104, [8, 16, 48, 784]);  bmm_104 = None
    view_904: "f32[8, 16, 48, 48]" = torch.ops.aten.reshape.default(bmm_105, [8, 16, 48, 48]);  bmm_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    mul_1517: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_904, alias_118);  view_904 = None
    sum_507: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_1517, [-1], True)
    mul_1518: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(alias_118, sum_507);  alias_118 = sum_507 = None
    sub_349: "f32[8, 16, 48, 48]" = torch.ops.aten.sub.Tensor(mul_1517, mul_1518);  mul_1517 = mul_1518 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    mul_1519: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(sub_349, view_171);  view_171 = None
    mul_1520: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(sub_349, primals_38);  sub_349 = primals_38 = None
    sum_508: "f32[1, 16, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1519, [0, 2, 3], True);  mul_1519 = None
    view_905: "f32[16, 1, 1]" = torch.ops.aten.reshape.default(sum_508, [16, 1, 1]);  sum_508 = None
    view_906: "f32[128, 48, 48]" = torch.ops.aten.reshape.default(mul_1520, [128, 48, 48]);  mul_1520 = None
    bmm_106: "f32[128, 784, 48]" = torch.ops.aten.bmm.default(permute_667, view_906);  permute_667 = None
    bmm_107: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_906, permute_668);  view_906 = permute_668 = None
    view_907: "f32[8, 16, 784, 48]" = torch.ops.aten.reshape.default(bmm_106, [8, 16, 784, 48]);  bmm_106 = None
    view_908: "f32[8, 16, 48, 784]" = torch.ops.aten.reshape.default(bmm_107, [8, 16, 48, 784]);  bmm_107 = None
    permute_669: "f32[8, 16, 48, 784]" = torch.ops.aten.permute.default(view_907, [0, 1, 3, 2]);  view_907 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    div_240: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_34, expand_55);  div_34 = None
    neg_28: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(permute_669)
    mul_1521: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_28, div_240);  neg_28 = div_240 = None
    div_241: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(permute_669, expand_55);  permute_669 = expand_55 = None
    sum_509: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_1521, [3], True);  mul_1521 = None
    ge_28: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(pow_41, 1e-12)
    where_56: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_28, sum_509, full_default_20);  ge_28 = sum_509 = None
    div_242: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_108, pow_41);  getitem_108 = None
    eq_28: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(pow_41, 0);  pow_41 = None
    where_57: "f32[8, 16, 48, 784]" = torch.ops.aten.where.self(eq_28, full_default_20, div_242);  eq_28 = div_242 = None
    clone_388: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(where_57, memory_format = torch.contiguous_format);  where_57 = None
    mul_1522: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_56, clone_388);  where_56 = clone_388 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    add_597: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_241, mul_1522);  div_241 = mul_1522 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    div_244: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_33, expand_54);  div_33 = None
    neg_29: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(view_908)
    mul_1523: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_29, div_244);  neg_29 = div_244 = None
    div_245: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(view_908, expand_54);  view_908 = expand_54 = None
    sum_510: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_1523, [3], True);  mul_1523 = None
    ge_29: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(pow_39, 1e-12)
    where_58: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_29, sum_510, full_default_20);  ge_29 = sum_510 = None
    div_246: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_107, pow_39);  getitem_107 = None
    eq_29: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(pow_39, 0);  pow_39 = None
    where_59: "f32[8, 16, 48, 784]" = torch.ops.aten.where.self(eq_29, full_default_20, div_246);  eq_29 = div_246 = None
    clone_389: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(where_59, memory_format = torch.contiguous_format);  where_59 = None
    mul_1524: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_58, clone_389);  where_58 = clone_389 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    add_598: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_245, mul_1524);  div_245 = mul_1524 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    cat_22: "f32[24, 16, 48, 784]" = torch.ops.aten.cat.default([add_598, add_597, view_903]);  add_598 = add_597 = view_903 = None
    view_909: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.reshape.default(cat_22, [3, 8, 16, 48, 784]);  cat_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    permute_670: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.permute.default(view_909, [1, 4, 0, 2, 3]);  view_909 = None
    clone_390: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.clone.default(permute_670, memory_format = torch.contiguous_format);  permute_670 = None
    view_910: "f32[8, 784, 2304]" = torch.ops.aten.reshape.default(clone_390, [8, 784, 2304]);  clone_390 = None
    view_911: "f32[6272, 2304]" = torch.ops.aten.reshape.default(view_910, [6272, 2304]);  view_910 = None
    mm_170: "f32[6272, 768]" = torch.ops.aten.mm.default(view_911, permute_671);  permute_671 = None
    permute_672: "f32[2304, 6272]" = torch.ops.aten.permute.default(view_911, [1, 0])
    mm_171: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_672, view_166);  permute_672 = view_166 = None
    permute_673: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_171, [1, 0]);  mm_171 = None
    sum_511: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_911, [0], True);  view_911 = None
    view_912: "f32[2304]" = torch.ops.aten.reshape.default(sum_511, [2304]);  sum_511 = None
    permute_674: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_673, [1, 0]);  permute_673 = None
    view_913: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_170, [8, 784, 768]);  mm_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_1526: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_913, primals_293);  primals_293 = None
    mul_1527: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1526, 768)
    sum_512: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1526, [2], True)
    mul_1528: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1526, mul_240);  mul_1526 = None
    sum_513: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1528, [2], True);  mul_1528 = None
    mul_1529: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_240, sum_513);  sum_513 = None
    sub_351: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(mul_1527, sum_512);  mul_1527 = sum_512 = None
    sub_352: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(sub_351, mul_1529);  sub_351 = mul_1529 = None
    mul_1530: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(div_247, sub_352);  div_247 = sub_352 = None
    mul_1531: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_913, mul_240);  mul_240 = None
    sum_514: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1531, [0, 1]);  mul_1531 = None
    sum_515: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_913, [0, 1]);  view_913 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    add_599: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_596, mul_1530);  add_596 = mul_1530 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_1532: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_599, primals_36);  primals_36 = None
    mul_1533: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_599, view_165);  view_165 = None
    sum_516: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1533, [0, 1], True);  mul_1533 = None
    view_914: "f32[768]" = torch.ops.aten.reshape.default(sum_516, [768]);  sum_516 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_915: "f32[6272, 768]" = torch.ops.aten.reshape.default(mul_1532, [6272, 768]);  mul_1532 = None
    mm_172: "f32[6272, 3072]" = torch.ops.aten.mm.default(view_915, permute_675);  permute_675 = None
    permute_676: "f32[768, 6272]" = torch.ops.aten.permute.default(view_915, [1, 0])
    mm_173: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_676, view_164);  permute_676 = view_164 = None
    permute_677: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_173, [1, 0]);  mm_173 = None
    sum_517: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_915, [0], True);  view_915 = None
    view_916: "f32[768]" = torch.ops.aten.reshape.default(sum_517, [768]);  sum_517 = None
    permute_678: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_677, [1, 0]);  permute_677 = None
    view_917: "f32[8, 784, 3072]" = torch.ops.aten.reshape.default(mm_172, [8, 784, 3072]);  mm_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_1535: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(add_174, 0.5);  add_174 = None
    mul_1536: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_163, view_163)
    mul_1537: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(mul_1536, -0.5);  mul_1536 = None
    exp_56: "f32[8, 784, 3072]" = torch.ops.aten.exp.default(mul_1537);  mul_1537 = None
    mul_1538: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(exp_56, 0.3989422804014327);  exp_56 = None
    mul_1539: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_163, mul_1538);  view_163 = mul_1538 = None
    add_601: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(mul_1535, mul_1539);  mul_1535 = mul_1539 = None
    mul_1540: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_917, add_601);  view_917 = add_601 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_918: "f32[6272, 3072]" = torch.ops.aten.reshape.default(mul_1540, [6272, 3072]);  mul_1540 = None
    mm_174: "f32[6272, 768]" = torch.ops.aten.mm.default(view_918, permute_679);  permute_679 = None
    permute_680: "f32[3072, 6272]" = torch.ops.aten.permute.default(view_918, [1, 0])
    mm_175: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_680, view_162);  permute_680 = view_162 = None
    permute_681: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_175, [1, 0]);  mm_175 = None
    sum_518: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_918, [0], True);  view_918 = None
    view_919: "f32[3072]" = torch.ops.aten.reshape.default(sum_518, [3072]);  sum_518 = None
    permute_682: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_681, [1, 0]);  permute_681 = None
    view_920: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_174, [8, 784, 768]);  mm_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_1542: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_920, primals_287);  primals_287 = None
    mul_1543: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1542, 768)
    sum_519: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1542, [2], True)
    mul_1544: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1542, mul_234);  mul_1542 = None
    sum_520: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1544, [2], True);  mul_1544 = None
    mul_1545: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_234, sum_520);  sum_520 = None
    sub_354: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(mul_1543, sum_519);  mul_1543 = sum_519 = None
    sub_355: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(sub_354, mul_1545);  sub_354 = mul_1545 = None
    mul_1546: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(div_248, sub_355);  div_248 = sub_355 = None
    mul_1547: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_920, mul_234);  mul_234 = None
    sum_521: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1547, [0, 1]);  mul_1547 = None
    sum_522: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_920, [0, 1]);  view_920 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    add_602: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_599, mul_1546);  add_599 = mul_1546 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_1548: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_602, primals_35);  primals_35 = None
    mul_1549: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_602, permute_81);  permute_81 = None
    sum_523: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1549, [0, 1], True);  mul_1549 = None
    view_921: "f32[768]" = torch.ops.aten.reshape.default(sum_523, [768]);  sum_523 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    permute_683: "f32[8, 768, 784]" = torch.ops.aten.permute.default(mul_1548, [0, 2, 1]);  mul_1548 = None
    view_922: "f32[8, 768, 28, 28]" = torch.ops.aten.reshape.default(permute_683, [8, 768, 28, 28]);  permute_683 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    sum_524: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_922, [0, 2, 3])
    convolution_backward_30 = torch.ops.aten.convolution_backward.default(view_922, add_170, primals_285, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False]);  view_922 = add_170 = primals_285 = None
    getitem_386: "f32[8, 768, 28, 28]" = convolution_backward_30[0]
    getitem_387: "f32[768, 1, 3, 3]" = convolution_backward_30[1];  convolution_backward_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    sum_525: "f32[768]" = torch.ops.aten.sum.dim_IntList(getitem_386, [0, 2, 3])
    sub_356: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(mul_225, unsqueeze_299);  mul_225 = unsqueeze_299 = None
    mul_1550: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_386, sub_356)
    sum_526: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1550, [0, 2, 3]);  mul_1550 = None
    mul_1551: "f32[768]" = torch.ops.aten.mul.Tensor(sum_525, 0.00015943877551020407)
    unsqueeze_300: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1551, 0);  mul_1551 = None
    unsqueeze_301: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_300, 2);  unsqueeze_300 = None
    unsqueeze_302: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_301, 3);  unsqueeze_301 = None
    mul_1552: "f32[768]" = torch.ops.aten.mul.Tensor(sum_526, 0.00015943877551020407)
    mul_1553: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_34, squeeze_34)
    mul_1554: "f32[768]" = torch.ops.aten.mul.Tensor(mul_1552, mul_1553);  mul_1552 = mul_1553 = None
    unsqueeze_303: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1554, 0);  mul_1554 = None
    unsqueeze_304: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_303, 2);  unsqueeze_303 = None
    unsqueeze_305: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_304, 3);  unsqueeze_304 = None
    mul_1555: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_34, primals_283);  primals_283 = None
    unsqueeze_306: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1555, 0);  mul_1555 = None
    unsqueeze_307: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_306, 2);  unsqueeze_306 = None
    unsqueeze_308: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_307, 3);  unsqueeze_307 = None
    mul_1556: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_356, unsqueeze_305);  sub_356 = unsqueeze_305 = None
    sub_358: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(getitem_386, mul_1556);  getitem_386 = mul_1556 = None
    sub_359: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(sub_358, unsqueeze_302);  sub_358 = unsqueeze_302 = None
    mul_1557: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_359, unsqueeze_308);  sub_359 = unsqueeze_308 = None
    mul_1558: "f32[768]" = torch.ops.aten.mul.Tensor(sum_526, squeeze_34);  sum_526 = squeeze_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_1560: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(add_165, 0.5);  add_165 = None
    mul_1561: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_20, convolution_20)
    mul_1562: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_1561, -0.5);  mul_1561 = None
    exp_57: "f32[8, 768, 28, 28]" = torch.ops.aten.exp.default(mul_1562);  mul_1562 = None
    mul_1563: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(exp_57, 0.3989422804014327);  exp_57 = None
    mul_1564: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_20, mul_1563);  convolution_20 = mul_1563 = None
    add_604: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(mul_1560, mul_1564);  mul_1560 = mul_1564 = None
    mul_1565: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_1557, add_604);  mul_1557 = add_604 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    sum_527: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1565, [0, 2, 3])
    convolution_backward_31 = torch.ops.aten.convolution_backward.default(mul_1565, view_160, primals_281, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False]);  mul_1565 = view_160 = primals_281 = None
    getitem_389: "f32[8, 768, 28, 28]" = convolution_backward_31[0]
    getitem_390: "f32[768, 1, 3, 3]" = convolution_backward_31[1];  convolution_backward_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    view_923: "f32[8, 768, 784]" = torch.ops.aten.reshape.default(getitem_389, [8, 768, 784]);  getitem_389 = None
    permute_684: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_923, [0, 2, 1]);  view_923 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    clone_393: "f32[8, 784, 768]" = torch.ops.aten.clone.default(permute_684, memory_format = torch.contiguous_format);  permute_684 = None
    mul_1567: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(clone_393, primals_279);  primals_279 = None
    mul_1568: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1567, 768)
    sum_528: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1567, [2], True)
    mul_1569: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1567, mul_221);  mul_1567 = None
    sum_529: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1569, [2], True);  mul_1569 = None
    mul_1570: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_221, sum_529);  sum_529 = None
    sub_361: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(mul_1568, sum_528);  mul_1568 = sum_528 = None
    sub_362: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(sub_361, mul_1570);  sub_361 = mul_1570 = None
    mul_1571: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(div_249, sub_362);  div_249 = sub_362 = None
    mul_1572: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(clone_393, mul_221);  mul_221 = None
    sum_530: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1572, [0, 1]);  mul_1572 = None
    sum_531: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_393, [0, 1]);  clone_393 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    add_605: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_602, mul_1571);  add_602 = mul_1571 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_1573: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_605, primals_33);  primals_33 = None
    mul_1574: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_605, add_161);  add_161 = None
    sum_532: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1574, [0, 1], True);  mul_1574 = None
    view_924: "f32[768]" = torch.ops.aten.reshape.default(sum_532, [768]);  sum_532 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    sum_533: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1573, [0, 1], True)
    view_925: "f32[768]" = torch.ops.aten.reshape.default(sum_533, [768]);  sum_533 = None
    view_926: "f32[6272, 768]" = torch.ops.aten.reshape.default(mul_1573, [6272, 768]);  mul_1573 = None
    permute_685: "f32[768, 6272]" = torch.ops.aten.permute.default(view_926, [1, 0])
    mm_176: "f32[768, 768]" = torch.ops.aten.mm.default(permute_685, view_158);  permute_685 = view_158 = None
    permute_686: "f32[768, 768]" = torch.ops.aten.permute.default(mm_176, [1, 0]);  mm_176 = None
    mm_177: "f32[6272, 768]" = torch.ops.aten.mm.default(view_926, permute_687);  view_926 = permute_687 = None
    view_927: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_177, [8, 784, 768]);  mm_177 = None
    permute_688: "f32[768, 768]" = torch.ops.aten.permute.default(permute_686, [1, 0]);  permute_686 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    view_928: "f32[8, 784, 16, 48]" = torch.ops.aten.reshape.default(view_927, [8, 784, 16, 48]);  view_927 = None
    permute_689: "f32[8, 16, 48, 784]" = torch.ops.aten.permute.default(view_928, [0, 2, 3, 1]);  view_928 = None
    clone_395: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(permute_689, memory_format = torch.contiguous_format);  permute_689 = None
    view_929: "f32[128, 48, 784]" = torch.ops.aten.reshape.default(clone_395, [128, 48, 784]);  clone_395 = None
    bmm_108: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(permute_690, view_929);  permute_690 = None
    bmm_109: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(view_929, permute_691);  view_929 = permute_691 = None
    view_930: "f32[8, 16, 48, 784]" = torch.ops.aten.reshape.default(bmm_108, [8, 16, 48, 784]);  bmm_108 = None
    view_931: "f32[8, 16, 48, 48]" = torch.ops.aten.reshape.default(bmm_109, [8, 16, 48, 48]);  bmm_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    mul_1575: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_931, alias_121);  view_931 = None
    sum_534: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_1575, [-1], True)
    mul_1576: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(alias_121, sum_534);  alias_121 = sum_534 = None
    sub_363: "f32[8, 16, 48, 48]" = torch.ops.aten.sub.Tensor(mul_1575, mul_1576);  mul_1575 = mul_1576 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    mul_1577: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(sub_363, view_153);  view_153 = None
    mul_1578: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(sub_363, primals_34);  sub_363 = primals_34 = None
    sum_535: "f32[1, 16, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1577, [0, 2, 3], True);  mul_1577 = None
    view_932: "f32[16, 1, 1]" = torch.ops.aten.reshape.default(sum_535, [16, 1, 1]);  sum_535 = None
    view_933: "f32[128, 48, 48]" = torch.ops.aten.reshape.default(mul_1578, [128, 48, 48]);  mul_1578 = None
    bmm_110: "f32[128, 784, 48]" = torch.ops.aten.bmm.default(permute_692, view_933);  permute_692 = None
    bmm_111: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_933, permute_693);  view_933 = permute_693 = None
    view_934: "f32[8, 16, 784, 48]" = torch.ops.aten.reshape.default(bmm_110, [8, 16, 784, 48]);  bmm_110 = None
    view_935: "f32[8, 16, 48, 784]" = torch.ops.aten.reshape.default(bmm_111, [8, 16, 48, 784]);  bmm_111 = None
    permute_694: "f32[8, 16, 48, 784]" = torch.ops.aten.permute.default(view_934, [0, 1, 3, 2]);  view_934 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    div_251: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_31, expand_49);  div_31 = None
    neg_30: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(permute_694)
    mul_1579: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_30, div_251);  neg_30 = div_251 = None
    div_252: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(permute_694, expand_49);  permute_694 = expand_49 = None
    sum_536: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_1579, [3], True);  mul_1579 = None
    ge_30: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(pow_37, 1e-12)
    where_60: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_30, sum_536, full_default_20);  ge_30 = sum_536 = None
    div_253: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_97, pow_37);  getitem_97 = None
    eq_30: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(pow_37, 0);  pow_37 = None
    where_61: "f32[8, 16, 48, 784]" = torch.ops.aten.where.self(eq_30, full_default_20, div_253);  eq_30 = div_253 = None
    clone_396: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(where_61, memory_format = torch.contiguous_format);  where_61 = None
    mul_1580: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_60, clone_396);  where_60 = clone_396 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    add_606: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_252, mul_1580);  div_252 = mul_1580 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    div_255: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_30, expand_48);  div_30 = None
    neg_31: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(view_935)
    mul_1581: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_31, div_255);  neg_31 = div_255 = None
    div_256: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(view_935, expand_48);  view_935 = expand_48 = None
    sum_537: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_1581, [3], True);  mul_1581 = None
    ge_31: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(pow_35, 1e-12)
    where_62: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_31, sum_537, full_default_20);  ge_31 = sum_537 = None
    div_257: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_96, pow_35);  getitem_96 = None
    eq_31: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(pow_35, 0);  pow_35 = None
    where_63: "f32[8, 16, 48, 784]" = torch.ops.aten.where.self(eq_31, full_default_20, div_257);  eq_31 = div_257 = None
    clone_397: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(where_63, memory_format = torch.contiguous_format);  where_63 = None
    mul_1582: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_62, clone_397);  where_62 = clone_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    add_607: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_256, mul_1582);  div_256 = mul_1582 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    cat_23: "f32[24, 16, 48, 784]" = torch.ops.aten.cat.default([add_607, add_606, view_930]);  add_607 = add_606 = view_930 = None
    view_936: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.reshape.default(cat_23, [3, 8, 16, 48, 784]);  cat_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    permute_695: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.permute.default(view_936, [1, 4, 0, 2, 3]);  view_936 = None
    clone_398: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.clone.default(permute_695, memory_format = torch.contiguous_format);  permute_695 = None
    view_937: "f32[8, 784, 2304]" = torch.ops.aten.reshape.default(clone_398, [8, 784, 2304]);  clone_398 = None
    view_938: "f32[6272, 2304]" = torch.ops.aten.reshape.default(view_937, [6272, 2304]);  view_937 = None
    mm_178: "f32[6272, 768]" = torch.ops.aten.mm.default(view_938, permute_696);  permute_696 = None
    permute_697: "f32[2304, 6272]" = torch.ops.aten.permute.default(view_938, [1, 0])
    mm_179: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_697, view_148);  permute_697 = view_148 = None
    permute_698: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_179, [1, 0]);  mm_179 = None
    sum_538: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_938, [0], True);  view_938 = None
    view_939: "f32[2304]" = torch.ops.aten.reshape.default(sum_538, [2304]);  sum_538 = None
    permute_699: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_698, [1, 0]);  permute_698 = None
    view_940: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_178, [8, 784, 768]);  mm_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_1584: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_940, primals_273);  primals_273 = None
    mul_1585: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1584, 768)
    sum_539: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1584, [2], True)
    mul_1586: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1584, mul_217);  mul_1584 = None
    sum_540: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1586, [2], True);  mul_1586 = None
    mul_1587: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_217, sum_540);  sum_540 = None
    sub_365: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(mul_1585, sum_539);  mul_1585 = sum_539 = None
    sub_366: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(sub_365, mul_1587);  sub_365 = mul_1587 = None
    mul_1588: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(div_258, sub_366);  div_258 = sub_366 = None
    mul_1589: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_940, mul_217);  mul_217 = None
    sum_541: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1589, [0, 1]);  mul_1589 = None
    sum_542: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_940, [0, 1]);  view_940 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    add_608: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_605, mul_1588);  add_605 = mul_1588 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_1590: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_608, primals_32);  primals_32 = None
    mul_1591: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_608, view_147);  view_147 = None
    sum_543: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1591, [0, 1], True);  mul_1591 = None
    view_941: "f32[768]" = torch.ops.aten.reshape.default(sum_543, [768]);  sum_543 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_942: "f32[6272, 768]" = torch.ops.aten.reshape.default(mul_1590, [6272, 768]);  mul_1590 = None
    mm_180: "f32[6272, 3072]" = torch.ops.aten.mm.default(view_942, permute_700);  permute_700 = None
    permute_701: "f32[768, 6272]" = torch.ops.aten.permute.default(view_942, [1, 0])
    mm_181: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_701, view_146);  permute_701 = view_146 = None
    permute_702: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_181, [1, 0]);  mm_181 = None
    sum_544: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_942, [0], True);  view_942 = None
    view_943: "f32[768]" = torch.ops.aten.reshape.default(sum_544, [768]);  sum_544 = None
    permute_703: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_702, [1, 0]);  permute_702 = None
    view_944: "f32[8, 784, 3072]" = torch.ops.aten.reshape.default(mm_180, [8, 784, 3072]);  mm_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_1593: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(add_157, 0.5);  add_157 = None
    mul_1594: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_145, view_145)
    mul_1595: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(mul_1594, -0.5);  mul_1594 = None
    exp_58: "f32[8, 784, 3072]" = torch.ops.aten.exp.default(mul_1595);  mul_1595 = None
    mul_1596: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(exp_58, 0.3989422804014327);  exp_58 = None
    mul_1597: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_145, mul_1596);  view_145 = mul_1596 = None
    add_610: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(mul_1593, mul_1597);  mul_1593 = mul_1597 = None
    mul_1598: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_944, add_610);  view_944 = add_610 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_945: "f32[6272, 3072]" = torch.ops.aten.reshape.default(mul_1598, [6272, 3072]);  mul_1598 = None
    mm_182: "f32[6272, 768]" = torch.ops.aten.mm.default(view_945, permute_704);  permute_704 = None
    permute_705: "f32[3072, 6272]" = torch.ops.aten.permute.default(view_945, [1, 0])
    mm_183: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_705, view_144);  permute_705 = view_144 = None
    permute_706: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_183, [1, 0]);  mm_183 = None
    sum_545: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_945, [0], True);  view_945 = None
    view_946: "f32[3072]" = torch.ops.aten.reshape.default(sum_545, [3072]);  sum_545 = None
    permute_707: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_706, [1, 0]);  permute_706 = None
    view_947: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_182, [8, 784, 768]);  mm_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_1600: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_947, primals_267);  primals_267 = None
    mul_1601: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1600, 768)
    sum_546: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1600, [2], True)
    mul_1602: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1600, mul_211);  mul_1600 = None
    sum_547: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1602, [2], True);  mul_1602 = None
    mul_1603: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_211, sum_547);  sum_547 = None
    sub_368: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(mul_1601, sum_546);  mul_1601 = sum_546 = None
    sub_369: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(sub_368, mul_1603);  sub_368 = mul_1603 = None
    mul_1604: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(div_259, sub_369);  div_259 = sub_369 = None
    mul_1605: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_947, mul_211);  mul_211 = None
    sum_548: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1605, [0, 1]);  mul_1605 = None
    sum_549: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_947, [0, 1]);  view_947 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    add_611: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_608, mul_1604);  add_608 = mul_1604 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_1606: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_611, primals_31);  primals_31 = None
    mul_1607: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_611, permute_72);  permute_72 = None
    sum_550: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1607, [0, 1], True);  mul_1607 = None
    view_948: "f32[768]" = torch.ops.aten.reshape.default(sum_550, [768]);  sum_550 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    permute_708: "f32[8, 768, 784]" = torch.ops.aten.permute.default(mul_1606, [0, 2, 1]);  mul_1606 = None
    view_949: "f32[8, 768, 28, 28]" = torch.ops.aten.reshape.default(permute_708, [8, 768, 28, 28]);  permute_708 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    sum_551: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_949, [0, 2, 3])
    convolution_backward_32 = torch.ops.aten.convolution_backward.default(view_949, add_153, primals_265, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False]);  view_949 = add_153 = primals_265 = None
    getitem_392: "f32[8, 768, 28, 28]" = convolution_backward_32[0]
    getitem_393: "f32[768, 1, 3, 3]" = convolution_backward_32[1];  convolution_backward_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    sum_552: "f32[768]" = torch.ops.aten.sum.dim_IntList(getitem_392, [0, 2, 3])
    sub_370: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(mul_202, unsqueeze_311);  mul_202 = unsqueeze_311 = None
    mul_1608: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_392, sub_370)
    sum_553: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1608, [0, 2, 3]);  mul_1608 = None
    mul_1609: "f32[768]" = torch.ops.aten.mul.Tensor(sum_552, 0.00015943877551020407)
    unsqueeze_312: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1609, 0);  mul_1609 = None
    unsqueeze_313: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_312, 2);  unsqueeze_312 = None
    unsqueeze_314: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_313, 3);  unsqueeze_313 = None
    mul_1610: "f32[768]" = torch.ops.aten.mul.Tensor(sum_553, 0.00015943877551020407)
    mul_1611: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
    mul_1612: "f32[768]" = torch.ops.aten.mul.Tensor(mul_1610, mul_1611);  mul_1610 = mul_1611 = None
    unsqueeze_315: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1612, 0);  mul_1612 = None
    unsqueeze_316: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_315, 2);  unsqueeze_315 = None
    unsqueeze_317: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_316, 3);  unsqueeze_316 = None
    mul_1613: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_31, primals_263);  primals_263 = None
    unsqueeze_318: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1613, 0);  mul_1613 = None
    unsqueeze_319: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_318, 2);  unsqueeze_318 = None
    unsqueeze_320: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_319, 3);  unsqueeze_319 = None
    mul_1614: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_370, unsqueeze_317);  sub_370 = unsqueeze_317 = None
    sub_372: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(getitem_392, mul_1614);  getitem_392 = mul_1614 = None
    sub_373: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(sub_372, unsqueeze_314);  sub_372 = unsqueeze_314 = None
    mul_1615: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_373, unsqueeze_320);  sub_373 = unsqueeze_320 = None
    mul_1616: "f32[768]" = torch.ops.aten.mul.Tensor(sum_553, squeeze_31);  sum_553 = squeeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_1618: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(add_148, 0.5);  add_148 = None
    mul_1619: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_18, convolution_18)
    mul_1620: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_1619, -0.5);  mul_1619 = None
    exp_59: "f32[8, 768, 28, 28]" = torch.ops.aten.exp.default(mul_1620);  mul_1620 = None
    mul_1621: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(exp_59, 0.3989422804014327);  exp_59 = None
    mul_1622: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_18, mul_1621);  convolution_18 = mul_1621 = None
    add_613: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(mul_1618, mul_1622);  mul_1618 = mul_1622 = None
    mul_1623: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_1615, add_613);  mul_1615 = add_613 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    sum_554: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1623, [0, 2, 3])
    convolution_backward_33 = torch.ops.aten.convolution_backward.default(mul_1623, view_142, primals_261, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False]);  mul_1623 = view_142 = primals_261 = None
    getitem_395: "f32[8, 768, 28, 28]" = convolution_backward_33[0]
    getitem_396: "f32[768, 1, 3, 3]" = convolution_backward_33[1];  convolution_backward_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    view_950: "f32[8, 768, 784]" = torch.ops.aten.reshape.default(getitem_395, [8, 768, 784]);  getitem_395 = None
    permute_709: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_950, [0, 2, 1]);  view_950 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    clone_401: "f32[8, 784, 768]" = torch.ops.aten.clone.default(permute_709, memory_format = torch.contiguous_format);  permute_709 = None
    mul_1625: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(clone_401, primals_259);  primals_259 = None
    mul_1626: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1625, 768)
    sum_555: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1625, [2], True)
    mul_1627: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1625, mul_198);  mul_1625 = None
    sum_556: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1627, [2], True);  mul_1627 = None
    mul_1628: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_198, sum_556);  sum_556 = None
    sub_375: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(mul_1626, sum_555);  mul_1626 = sum_555 = None
    sub_376: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(sub_375, mul_1628);  sub_375 = mul_1628 = None
    mul_1629: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(div_260, sub_376);  div_260 = sub_376 = None
    mul_1630: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(clone_401, mul_198);  mul_198 = None
    sum_557: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1630, [0, 1]);  mul_1630 = None
    sum_558: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_401, [0, 1]);  clone_401 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    add_614: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_611, mul_1629);  add_611 = mul_1629 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_1631: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_614, primals_29);  primals_29 = None
    mul_1632: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_614, add_144);  add_144 = None
    sum_559: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1632, [0, 1], True);  mul_1632 = None
    view_951: "f32[768]" = torch.ops.aten.reshape.default(sum_559, [768]);  sum_559 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    sum_560: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1631, [0, 1], True)
    view_952: "f32[768]" = torch.ops.aten.reshape.default(sum_560, [768]);  sum_560 = None
    view_953: "f32[6272, 768]" = torch.ops.aten.reshape.default(mul_1631, [6272, 768]);  mul_1631 = None
    permute_710: "f32[768, 6272]" = torch.ops.aten.permute.default(view_953, [1, 0])
    mm_184: "f32[768, 768]" = torch.ops.aten.mm.default(permute_710, view_140);  permute_710 = view_140 = None
    permute_711: "f32[768, 768]" = torch.ops.aten.permute.default(mm_184, [1, 0]);  mm_184 = None
    mm_185: "f32[6272, 768]" = torch.ops.aten.mm.default(view_953, permute_712);  view_953 = permute_712 = None
    view_954: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_185, [8, 784, 768]);  mm_185 = None
    permute_713: "f32[768, 768]" = torch.ops.aten.permute.default(permute_711, [1, 0]);  permute_711 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    view_955: "f32[8, 784, 16, 48]" = torch.ops.aten.reshape.default(view_954, [8, 784, 16, 48]);  view_954 = None
    permute_714: "f32[8, 16, 48, 784]" = torch.ops.aten.permute.default(view_955, [0, 2, 3, 1]);  view_955 = None
    clone_403: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(permute_714, memory_format = torch.contiguous_format);  permute_714 = None
    view_956: "f32[128, 48, 784]" = torch.ops.aten.reshape.default(clone_403, [128, 48, 784]);  clone_403 = None
    bmm_112: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(permute_715, view_956);  permute_715 = None
    bmm_113: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(view_956, permute_716);  view_956 = permute_716 = None
    view_957: "f32[8, 16, 48, 784]" = torch.ops.aten.reshape.default(bmm_112, [8, 16, 48, 784]);  bmm_112 = None
    view_958: "f32[8, 16, 48, 48]" = torch.ops.aten.reshape.default(bmm_113, [8, 16, 48, 48]);  bmm_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    mul_1633: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_958, alias_124);  view_958 = None
    sum_561: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_1633, [-1], True)
    mul_1634: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(alias_124, sum_561);  alias_124 = sum_561 = None
    sub_377: "f32[8, 16, 48, 48]" = torch.ops.aten.sub.Tensor(mul_1633, mul_1634);  mul_1633 = mul_1634 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    mul_1635: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(sub_377, view_135);  view_135 = None
    mul_1636: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(sub_377, primals_30);  sub_377 = primals_30 = None
    sum_562: "f32[1, 16, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1635, [0, 2, 3], True);  mul_1635 = None
    view_959: "f32[16, 1, 1]" = torch.ops.aten.reshape.default(sum_562, [16, 1, 1]);  sum_562 = None
    view_960: "f32[128, 48, 48]" = torch.ops.aten.reshape.default(mul_1636, [128, 48, 48]);  mul_1636 = None
    bmm_114: "f32[128, 784, 48]" = torch.ops.aten.bmm.default(permute_717, view_960);  permute_717 = None
    bmm_115: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_960, permute_718);  view_960 = permute_718 = None
    view_961: "f32[8, 16, 784, 48]" = torch.ops.aten.reshape.default(bmm_114, [8, 16, 784, 48]);  bmm_114 = None
    view_962: "f32[8, 16, 48, 784]" = torch.ops.aten.reshape.default(bmm_115, [8, 16, 48, 784]);  bmm_115 = None
    permute_719: "f32[8, 16, 48, 784]" = torch.ops.aten.permute.default(view_961, [0, 1, 3, 2]);  view_961 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    div_262: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_28, expand_43);  div_28 = None
    neg_32: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(permute_719)
    mul_1637: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_32, div_262);  neg_32 = div_262 = None
    div_263: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(permute_719, expand_43);  permute_719 = expand_43 = None
    sum_563: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_1637, [3], True);  mul_1637 = None
    ge_32: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(pow_33, 1e-12)
    where_64: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_32, sum_563, full_default_20);  ge_32 = sum_563 = None
    div_264: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_86, pow_33);  getitem_86 = None
    eq_32: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(pow_33, 0);  pow_33 = None
    where_65: "f32[8, 16, 48, 784]" = torch.ops.aten.where.self(eq_32, full_default_20, div_264);  eq_32 = div_264 = None
    clone_404: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(where_65, memory_format = torch.contiguous_format);  where_65 = None
    mul_1638: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_64, clone_404);  where_64 = clone_404 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    add_615: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_263, mul_1638);  div_263 = mul_1638 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    div_266: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_27, expand_42);  div_27 = None
    neg_33: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(view_962)
    mul_1639: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_33, div_266);  neg_33 = div_266 = None
    div_267: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(view_962, expand_42);  view_962 = expand_42 = None
    sum_564: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_1639, [3], True);  mul_1639 = None
    ge_33: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(pow_31, 1e-12)
    where_66: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_33, sum_564, full_default_20);  ge_33 = sum_564 = None
    div_268: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_85, pow_31);  getitem_85 = None
    eq_33: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(pow_31, 0);  pow_31 = None
    where_67: "f32[8, 16, 48, 784]" = torch.ops.aten.where.self(eq_33, full_default_20, div_268);  eq_33 = div_268 = None
    clone_405: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(where_67, memory_format = torch.contiguous_format);  where_67 = None
    mul_1640: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_66, clone_405);  where_66 = clone_405 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    add_616: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_267, mul_1640);  div_267 = mul_1640 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    cat_24: "f32[24, 16, 48, 784]" = torch.ops.aten.cat.default([add_616, add_615, view_957]);  add_616 = add_615 = view_957 = None
    view_963: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.reshape.default(cat_24, [3, 8, 16, 48, 784]);  cat_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    permute_720: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.permute.default(view_963, [1, 4, 0, 2, 3]);  view_963 = None
    clone_406: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.clone.default(permute_720, memory_format = torch.contiguous_format);  permute_720 = None
    view_964: "f32[8, 784, 2304]" = torch.ops.aten.reshape.default(clone_406, [8, 784, 2304]);  clone_406 = None
    view_965: "f32[6272, 2304]" = torch.ops.aten.reshape.default(view_964, [6272, 2304]);  view_964 = None
    mm_186: "f32[6272, 768]" = torch.ops.aten.mm.default(view_965, permute_721);  permute_721 = None
    permute_722: "f32[2304, 6272]" = torch.ops.aten.permute.default(view_965, [1, 0])
    mm_187: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_722, view_130);  permute_722 = view_130 = None
    permute_723: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_187, [1, 0]);  mm_187 = None
    sum_565: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_965, [0], True);  view_965 = None
    view_966: "f32[2304]" = torch.ops.aten.reshape.default(sum_565, [2304]);  sum_565 = None
    permute_724: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_723, [1, 0]);  permute_723 = None
    view_967: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_186, [8, 784, 768]);  mm_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_1642: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_967, primals_253);  primals_253 = None
    mul_1643: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1642, 768)
    sum_566: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1642, [2], True)
    mul_1644: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1642, mul_194);  mul_1642 = None
    sum_567: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1644, [2], True);  mul_1644 = None
    mul_1645: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_194, sum_567);  sum_567 = None
    sub_379: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(mul_1643, sum_566);  mul_1643 = sum_566 = None
    sub_380: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(sub_379, mul_1645);  sub_379 = mul_1645 = None
    mul_1646: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(div_269, sub_380);  div_269 = sub_380 = None
    mul_1647: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_967, mul_194);  mul_194 = None
    sum_568: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1647, [0, 1]);  mul_1647 = None
    sum_569: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_967, [0, 1]);  view_967 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    add_617: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_614, mul_1646);  add_614 = mul_1646 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_1648: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_617, primals_28);  primals_28 = None
    mul_1649: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_617, view_129);  view_129 = None
    sum_570: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1649, [0, 1], True);  mul_1649 = None
    view_968: "f32[768]" = torch.ops.aten.reshape.default(sum_570, [768]);  sum_570 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_969: "f32[6272, 768]" = torch.ops.aten.reshape.default(mul_1648, [6272, 768]);  mul_1648 = None
    mm_188: "f32[6272, 3072]" = torch.ops.aten.mm.default(view_969, permute_725);  permute_725 = None
    permute_726: "f32[768, 6272]" = torch.ops.aten.permute.default(view_969, [1, 0])
    mm_189: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_726, view_128);  permute_726 = view_128 = None
    permute_727: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_189, [1, 0]);  mm_189 = None
    sum_571: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_969, [0], True);  view_969 = None
    view_970: "f32[768]" = torch.ops.aten.reshape.default(sum_571, [768]);  sum_571 = None
    permute_728: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_727, [1, 0]);  permute_727 = None
    view_971: "f32[8, 784, 3072]" = torch.ops.aten.reshape.default(mm_188, [8, 784, 3072]);  mm_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_1651: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(add_140, 0.5);  add_140 = None
    mul_1652: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_127, view_127)
    mul_1653: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(mul_1652, -0.5);  mul_1652 = None
    exp_60: "f32[8, 784, 3072]" = torch.ops.aten.exp.default(mul_1653);  mul_1653 = None
    mul_1654: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(exp_60, 0.3989422804014327);  exp_60 = None
    mul_1655: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_127, mul_1654);  view_127 = mul_1654 = None
    add_619: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(mul_1651, mul_1655);  mul_1651 = mul_1655 = None
    mul_1656: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_971, add_619);  view_971 = add_619 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_972: "f32[6272, 3072]" = torch.ops.aten.reshape.default(mul_1656, [6272, 3072]);  mul_1656 = None
    mm_190: "f32[6272, 768]" = torch.ops.aten.mm.default(view_972, permute_729);  permute_729 = None
    permute_730: "f32[3072, 6272]" = torch.ops.aten.permute.default(view_972, [1, 0])
    mm_191: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_730, view_126);  permute_730 = view_126 = None
    permute_731: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_191, [1, 0]);  mm_191 = None
    sum_572: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_972, [0], True);  view_972 = None
    view_973: "f32[3072]" = torch.ops.aten.reshape.default(sum_572, [3072]);  sum_572 = None
    permute_732: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_731, [1, 0]);  permute_731 = None
    view_974: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_190, [8, 784, 768]);  mm_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_1658: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_974, primals_247);  primals_247 = None
    mul_1659: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1658, 768)
    sum_573: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1658, [2], True)
    mul_1660: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1658, mul_188);  mul_1658 = None
    sum_574: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1660, [2], True);  mul_1660 = None
    mul_1661: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_188, sum_574);  sum_574 = None
    sub_382: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(mul_1659, sum_573);  mul_1659 = sum_573 = None
    sub_383: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(sub_382, mul_1661);  sub_382 = mul_1661 = None
    mul_1662: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(div_270, sub_383);  div_270 = sub_383 = None
    mul_1663: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_974, mul_188);  mul_188 = None
    sum_575: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1663, [0, 1]);  mul_1663 = None
    sum_576: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_974, [0, 1]);  view_974 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    add_620: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_617, mul_1662);  add_617 = mul_1662 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_1664: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_620, primals_27);  primals_27 = None
    mul_1665: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_620, permute_63);  permute_63 = None
    sum_577: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1665, [0, 1], True);  mul_1665 = None
    view_975: "f32[768]" = torch.ops.aten.reshape.default(sum_577, [768]);  sum_577 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    permute_733: "f32[8, 768, 784]" = torch.ops.aten.permute.default(mul_1664, [0, 2, 1]);  mul_1664 = None
    view_976: "f32[8, 768, 28, 28]" = torch.ops.aten.reshape.default(permute_733, [8, 768, 28, 28]);  permute_733 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    sum_578: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_976, [0, 2, 3])
    convolution_backward_34 = torch.ops.aten.convolution_backward.default(view_976, add_136, primals_245, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False]);  view_976 = add_136 = primals_245 = None
    getitem_398: "f32[8, 768, 28, 28]" = convolution_backward_34[0]
    getitem_399: "f32[768, 1, 3, 3]" = convolution_backward_34[1];  convolution_backward_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    sum_579: "f32[768]" = torch.ops.aten.sum.dim_IntList(getitem_398, [0, 2, 3])
    sub_384: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(mul_179, unsqueeze_323);  mul_179 = unsqueeze_323 = None
    mul_1666: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_398, sub_384)
    sum_580: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1666, [0, 2, 3]);  mul_1666 = None
    mul_1667: "f32[768]" = torch.ops.aten.mul.Tensor(sum_579, 0.00015943877551020407)
    unsqueeze_324: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1667, 0);  mul_1667 = None
    unsqueeze_325: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_324, 2);  unsqueeze_324 = None
    unsqueeze_326: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_325, 3);  unsqueeze_325 = None
    mul_1668: "f32[768]" = torch.ops.aten.mul.Tensor(sum_580, 0.00015943877551020407)
    mul_1669: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_28, squeeze_28)
    mul_1670: "f32[768]" = torch.ops.aten.mul.Tensor(mul_1668, mul_1669);  mul_1668 = mul_1669 = None
    unsqueeze_327: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1670, 0);  mul_1670 = None
    unsqueeze_328: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_327, 2);  unsqueeze_327 = None
    unsqueeze_329: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_328, 3);  unsqueeze_328 = None
    mul_1671: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_28, primals_243);  primals_243 = None
    unsqueeze_330: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1671, 0);  mul_1671 = None
    unsqueeze_331: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_330, 2);  unsqueeze_330 = None
    unsqueeze_332: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_331, 3);  unsqueeze_331 = None
    mul_1672: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_384, unsqueeze_329);  sub_384 = unsqueeze_329 = None
    sub_386: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(getitem_398, mul_1672);  getitem_398 = mul_1672 = None
    sub_387: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(sub_386, unsqueeze_326);  sub_386 = unsqueeze_326 = None
    mul_1673: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_387, unsqueeze_332);  sub_387 = unsqueeze_332 = None
    mul_1674: "f32[768]" = torch.ops.aten.mul.Tensor(sum_580, squeeze_28);  sum_580 = squeeze_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_1676: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(add_131, 0.5);  add_131 = None
    mul_1677: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_16, convolution_16)
    mul_1678: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_1677, -0.5);  mul_1677 = None
    exp_61: "f32[8, 768, 28, 28]" = torch.ops.aten.exp.default(mul_1678);  mul_1678 = None
    mul_1679: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(exp_61, 0.3989422804014327);  exp_61 = None
    mul_1680: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_16, mul_1679);  convolution_16 = mul_1679 = None
    add_622: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(mul_1676, mul_1680);  mul_1676 = mul_1680 = None
    mul_1681: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_1673, add_622);  mul_1673 = add_622 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    sum_581: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1681, [0, 2, 3])
    convolution_backward_35 = torch.ops.aten.convolution_backward.default(mul_1681, view_124, primals_241, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False]);  mul_1681 = view_124 = primals_241 = None
    getitem_401: "f32[8, 768, 28, 28]" = convolution_backward_35[0]
    getitem_402: "f32[768, 1, 3, 3]" = convolution_backward_35[1];  convolution_backward_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    view_977: "f32[8, 768, 784]" = torch.ops.aten.reshape.default(getitem_401, [8, 768, 784]);  getitem_401 = None
    permute_734: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_977, [0, 2, 1]);  view_977 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    clone_409: "f32[8, 784, 768]" = torch.ops.aten.clone.default(permute_734, memory_format = torch.contiguous_format);  permute_734 = None
    mul_1683: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(clone_409, primals_239);  primals_239 = None
    mul_1684: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1683, 768)
    sum_582: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1683, [2], True)
    mul_1685: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1683, mul_175);  mul_1683 = None
    sum_583: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1685, [2], True);  mul_1685 = None
    mul_1686: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_175, sum_583);  sum_583 = None
    sub_389: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(mul_1684, sum_582);  mul_1684 = sum_582 = None
    sub_390: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(sub_389, mul_1686);  sub_389 = mul_1686 = None
    mul_1687: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(div_271, sub_390);  div_271 = sub_390 = None
    mul_1688: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(clone_409, mul_175);  mul_175 = None
    sum_584: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1688, [0, 1]);  mul_1688 = None
    sum_585: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_409, [0, 1]);  clone_409 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    add_623: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_620, mul_1687);  add_620 = mul_1687 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_1689: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_623, primals_25);  primals_25 = None
    mul_1690: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_623, add_127);  add_127 = None
    sum_586: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1690, [0, 1], True);  mul_1690 = None
    view_978: "f32[768]" = torch.ops.aten.reshape.default(sum_586, [768]);  sum_586 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    sum_587: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1689, [0, 1], True)
    view_979: "f32[768]" = torch.ops.aten.reshape.default(sum_587, [768]);  sum_587 = None
    view_980: "f32[6272, 768]" = torch.ops.aten.reshape.default(mul_1689, [6272, 768]);  mul_1689 = None
    permute_735: "f32[768, 6272]" = torch.ops.aten.permute.default(view_980, [1, 0])
    mm_192: "f32[768, 768]" = torch.ops.aten.mm.default(permute_735, view_122);  permute_735 = view_122 = None
    permute_736: "f32[768, 768]" = torch.ops.aten.permute.default(mm_192, [1, 0]);  mm_192 = None
    mm_193: "f32[6272, 768]" = torch.ops.aten.mm.default(view_980, permute_737);  view_980 = permute_737 = None
    view_981: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_193, [8, 784, 768]);  mm_193 = None
    permute_738: "f32[768, 768]" = torch.ops.aten.permute.default(permute_736, [1, 0]);  permute_736 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    view_982: "f32[8, 784, 16, 48]" = torch.ops.aten.reshape.default(view_981, [8, 784, 16, 48]);  view_981 = None
    permute_739: "f32[8, 16, 48, 784]" = torch.ops.aten.permute.default(view_982, [0, 2, 3, 1]);  view_982 = None
    clone_411: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(permute_739, memory_format = torch.contiguous_format);  permute_739 = None
    view_983: "f32[128, 48, 784]" = torch.ops.aten.reshape.default(clone_411, [128, 48, 784]);  clone_411 = None
    bmm_116: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(permute_740, view_983);  permute_740 = None
    bmm_117: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(view_983, permute_741);  view_983 = permute_741 = None
    view_984: "f32[8, 16, 48, 784]" = torch.ops.aten.reshape.default(bmm_116, [8, 16, 48, 784]);  bmm_116 = None
    view_985: "f32[8, 16, 48, 48]" = torch.ops.aten.reshape.default(bmm_117, [8, 16, 48, 48]);  bmm_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    mul_1691: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_985, alias_127);  view_985 = None
    sum_588: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_1691, [-1], True)
    mul_1692: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(alias_127, sum_588);  alias_127 = sum_588 = None
    sub_391: "f32[8, 16, 48, 48]" = torch.ops.aten.sub.Tensor(mul_1691, mul_1692);  mul_1691 = mul_1692 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    mul_1693: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(sub_391, view_117);  view_117 = None
    mul_1694: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(sub_391, primals_26);  sub_391 = primals_26 = None
    sum_589: "f32[1, 16, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1693, [0, 2, 3], True);  mul_1693 = None
    view_986: "f32[16, 1, 1]" = torch.ops.aten.reshape.default(sum_589, [16, 1, 1]);  sum_589 = None
    view_987: "f32[128, 48, 48]" = torch.ops.aten.reshape.default(mul_1694, [128, 48, 48]);  mul_1694 = None
    bmm_118: "f32[128, 784, 48]" = torch.ops.aten.bmm.default(permute_742, view_987);  permute_742 = None
    bmm_119: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_987, permute_743);  view_987 = permute_743 = None
    view_988: "f32[8, 16, 784, 48]" = torch.ops.aten.reshape.default(bmm_118, [8, 16, 784, 48]);  bmm_118 = None
    view_989: "f32[8, 16, 48, 784]" = torch.ops.aten.reshape.default(bmm_119, [8, 16, 48, 784]);  bmm_119 = None
    permute_744: "f32[8, 16, 48, 784]" = torch.ops.aten.permute.default(view_988, [0, 1, 3, 2]);  view_988 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    div_273: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_25, expand_37);  div_25 = None
    neg_34: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(permute_744)
    mul_1695: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_34, div_273);  neg_34 = div_273 = None
    div_274: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(permute_744, expand_37);  permute_744 = expand_37 = None
    sum_590: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_1695, [3], True);  mul_1695 = None
    ge_34: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(pow_29, 1e-12)
    where_68: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_34, sum_590, full_default_20);  ge_34 = sum_590 = None
    div_275: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_75, pow_29);  getitem_75 = None
    eq_34: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(pow_29, 0);  pow_29 = None
    where_69: "f32[8, 16, 48, 784]" = torch.ops.aten.where.self(eq_34, full_default_20, div_275);  eq_34 = div_275 = None
    clone_412: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(where_69, memory_format = torch.contiguous_format);  where_69 = None
    mul_1696: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_68, clone_412);  where_68 = clone_412 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    add_624: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_274, mul_1696);  div_274 = mul_1696 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    div_277: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_24, expand_36);  div_24 = None
    neg_35: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(view_989)
    mul_1697: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_35, div_277);  neg_35 = div_277 = None
    div_278: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(view_989, expand_36);  view_989 = expand_36 = None
    sum_591: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_1697, [3], True);  mul_1697 = None
    ge_35: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(pow_27, 1e-12)
    where_70: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_35, sum_591, full_default_20);  ge_35 = sum_591 = None
    div_279: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_74, pow_27);  getitem_74 = None
    eq_35: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(pow_27, 0);  pow_27 = None
    where_71: "f32[8, 16, 48, 784]" = torch.ops.aten.where.self(eq_35, full_default_20, div_279);  eq_35 = div_279 = None
    clone_413: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(where_71, memory_format = torch.contiguous_format);  where_71 = None
    mul_1698: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_70, clone_413);  where_70 = clone_413 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    add_625: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_278, mul_1698);  div_278 = mul_1698 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    cat_25: "f32[24, 16, 48, 784]" = torch.ops.aten.cat.default([add_625, add_624, view_984]);  add_625 = add_624 = view_984 = None
    view_990: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.reshape.default(cat_25, [3, 8, 16, 48, 784]);  cat_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    permute_745: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.permute.default(view_990, [1, 4, 0, 2, 3]);  view_990 = None
    clone_414: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.clone.default(permute_745, memory_format = torch.contiguous_format);  permute_745 = None
    view_991: "f32[8, 784, 2304]" = torch.ops.aten.reshape.default(clone_414, [8, 784, 2304]);  clone_414 = None
    view_992: "f32[6272, 2304]" = torch.ops.aten.reshape.default(view_991, [6272, 2304]);  view_991 = None
    mm_194: "f32[6272, 768]" = torch.ops.aten.mm.default(view_992, permute_746);  permute_746 = None
    permute_747: "f32[2304, 6272]" = torch.ops.aten.permute.default(view_992, [1, 0])
    mm_195: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_747, view_112);  permute_747 = view_112 = None
    permute_748: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_195, [1, 0]);  mm_195 = None
    sum_592: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_992, [0], True);  view_992 = None
    view_993: "f32[2304]" = torch.ops.aten.reshape.default(sum_592, [2304]);  sum_592 = None
    permute_749: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_748, [1, 0]);  permute_748 = None
    view_994: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_194, [8, 784, 768]);  mm_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_1700: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_994, primals_233);  primals_233 = None
    mul_1701: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1700, 768)
    sum_593: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1700, [2], True)
    mul_1702: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1700, mul_171);  mul_1700 = None
    sum_594: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1702, [2], True);  mul_1702 = None
    mul_1703: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_171, sum_594);  sum_594 = None
    sub_393: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(mul_1701, sum_593);  mul_1701 = sum_593 = None
    sub_394: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(sub_393, mul_1703);  sub_393 = mul_1703 = None
    mul_1704: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(div_280, sub_394);  div_280 = sub_394 = None
    mul_1705: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_994, mul_171);  mul_171 = None
    sum_595: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1705, [0, 1]);  mul_1705 = None
    sum_596: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_994, [0, 1]);  view_994 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    add_626: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_623, mul_1704);  add_623 = mul_1704 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_1706: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_626, primals_24);  primals_24 = None
    mul_1707: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_626, view_111);  view_111 = None
    sum_597: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1707, [0, 1], True);  mul_1707 = None
    view_995: "f32[768]" = torch.ops.aten.reshape.default(sum_597, [768]);  sum_597 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_996: "f32[6272, 768]" = torch.ops.aten.reshape.default(mul_1706, [6272, 768]);  mul_1706 = None
    mm_196: "f32[6272, 3072]" = torch.ops.aten.mm.default(view_996, permute_750);  permute_750 = None
    permute_751: "f32[768, 6272]" = torch.ops.aten.permute.default(view_996, [1, 0])
    mm_197: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_751, view_110);  permute_751 = view_110 = None
    permute_752: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_197, [1, 0]);  mm_197 = None
    sum_598: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_996, [0], True);  view_996 = None
    view_997: "f32[768]" = torch.ops.aten.reshape.default(sum_598, [768]);  sum_598 = None
    permute_753: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_752, [1, 0]);  permute_752 = None
    view_998: "f32[8, 784, 3072]" = torch.ops.aten.reshape.default(mm_196, [8, 784, 3072]);  mm_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_1709: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(add_123, 0.5);  add_123 = None
    mul_1710: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_109, view_109)
    mul_1711: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(mul_1710, -0.5);  mul_1710 = None
    exp_62: "f32[8, 784, 3072]" = torch.ops.aten.exp.default(mul_1711);  mul_1711 = None
    mul_1712: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(exp_62, 0.3989422804014327);  exp_62 = None
    mul_1713: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_109, mul_1712);  view_109 = mul_1712 = None
    add_628: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(mul_1709, mul_1713);  mul_1709 = mul_1713 = None
    mul_1714: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_998, add_628);  view_998 = add_628 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_999: "f32[6272, 3072]" = torch.ops.aten.reshape.default(mul_1714, [6272, 3072]);  mul_1714 = None
    mm_198: "f32[6272, 768]" = torch.ops.aten.mm.default(view_999, permute_754);  permute_754 = None
    permute_755: "f32[3072, 6272]" = torch.ops.aten.permute.default(view_999, [1, 0])
    mm_199: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_755, view_108);  permute_755 = view_108 = None
    permute_756: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_199, [1, 0]);  mm_199 = None
    sum_599: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_999, [0], True);  view_999 = None
    view_1000: "f32[3072]" = torch.ops.aten.reshape.default(sum_599, [3072]);  sum_599 = None
    permute_757: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_756, [1, 0]);  permute_756 = None
    view_1001: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_198, [8, 784, 768]);  mm_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_1716: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_1001, primals_227);  primals_227 = None
    mul_1717: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1716, 768)
    sum_600: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1716, [2], True)
    mul_1718: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1716, mul_165);  mul_1716 = None
    sum_601: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1718, [2], True);  mul_1718 = None
    mul_1719: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_165, sum_601);  sum_601 = None
    sub_396: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(mul_1717, sum_600);  mul_1717 = sum_600 = None
    sub_397: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(sub_396, mul_1719);  sub_396 = mul_1719 = None
    mul_1720: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(div_281, sub_397);  div_281 = sub_397 = None
    mul_1721: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_1001, mul_165);  mul_165 = None
    sum_602: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1721, [0, 1]);  mul_1721 = None
    sum_603: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_1001, [0, 1]);  view_1001 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    add_629: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_626, mul_1720);  add_626 = mul_1720 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_1722: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_629, primals_23);  primals_23 = None
    mul_1723: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_629, permute_54);  permute_54 = None
    sum_604: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1723, [0, 1], True);  mul_1723 = None
    view_1002: "f32[768]" = torch.ops.aten.reshape.default(sum_604, [768]);  sum_604 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    permute_758: "f32[8, 768, 784]" = torch.ops.aten.permute.default(mul_1722, [0, 2, 1]);  mul_1722 = None
    view_1003: "f32[8, 768, 28, 28]" = torch.ops.aten.reshape.default(permute_758, [8, 768, 28, 28]);  permute_758 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    sum_605: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_1003, [0, 2, 3])
    convolution_backward_36 = torch.ops.aten.convolution_backward.default(view_1003, add_119, primals_225, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False]);  view_1003 = add_119 = primals_225 = None
    getitem_404: "f32[8, 768, 28, 28]" = convolution_backward_36[0]
    getitem_405: "f32[768, 1, 3, 3]" = convolution_backward_36[1];  convolution_backward_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    sum_606: "f32[768]" = torch.ops.aten.sum.dim_IntList(getitem_404, [0, 2, 3])
    sub_398: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(mul_156, unsqueeze_335);  mul_156 = unsqueeze_335 = None
    mul_1724: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_404, sub_398)
    sum_607: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1724, [0, 2, 3]);  mul_1724 = None
    mul_1725: "f32[768]" = torch.ops.aten.mul.Tensor(sum_606, 0.00015943877551020407)
    unsqueeze_336: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1725, 0);  mul_1725 = None
    unsqueeze_337: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_336, 2);  unsqueeze_336 = None
    unsqueeze_338: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_337, 3);  unsqueeze_337 = None
    mul_1726: "f32[768]" = torch.ops.aten.mul.Tensor(sum_607, 0.00015943877551020407)
    mul_1727: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
    mul_1728: "f32[768]" = torch.ops.aten.mul.Tensor(mul_1726, mul_1727);  mul_1726 = mul_1727 = None
    unsqueeze_339: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1728, 0);  mul_1728 = None
    unsqueeze_340: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_339, 2);  unsqueeze_339 = None
    unsqueeze_341: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_340, 3);  unsqueeze_340 = None
    mul_1729: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_25, primals_223);  primals_223 = None
    unsqueeze_342: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1729, 0);  mul_1729 = None
    unsqueeze_343: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_342, 2);  unsqueeze_342 = None
    unsqueeze_344: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_343, 3);  unsqueeze_343 = None
    mul_1730: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_398, unsqueeze_341);  sub_398 = unsqueeze_341 = None
    sub_400: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(getitem_404, mul_1730);  getitem_404 = mul_1730 = None
    sub_401: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(sub_400, unsqueeze_338);  sub_400 = unsqueeze_338 = None
    mul_1731: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_401, unsqueeze_344);  sub_401 = unsqueeze_344 = None
    mul_1732: "f32[768]" = torch.ops.aten.mul.Tensor(sum_607, squeeze_25);  sum_607 = squeeze_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_1734: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(add_114, 0.5);  add_114 = None
    mul_1735: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_14, convolution_14)
    mul_1736: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_1735, -0.5);  mul_1735 = None
    exp_63: "f32[8, 768, 28, 28]" = torch.ops.aten.exp.default(mul_1736);  mul_1736 = None
    mul_1737: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(exp_63, 0.3989422804014327);  exp_63 = None
    mul_1738: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_14, mul_1737);  convolution_14 = mul_1737 = None
    add_631: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(mul_1734, mul_1738);  mul_1734 = mul_1738 = None
    mul_1739: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_1731, add_631);  mul_1731 = add_631 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    sum_608: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1739, [0, 2, 3])
    convolution_backward_37 = torch.ops.aten.convolution_backward.default(mul_1739, view_106, primals_221, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False]);  mul_1739 = view_106 = primals_221 = None
    getitem_407: "f32[8, 768, 28, 28]" = convolution_backward_37[0]
    getitem_408: "f32[768, 1, 3, 3]" = convolution_backward_37[1];  convolution_backward_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    view_1004: "f32[8, 768, 784]" = torch.ops.aten.reshape.default(getitem_407, [8, 768, 784]);  getitem_407 = None
    permute_759: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_1004, [0, 2, 1]);  view_1004 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    clone_417: "f32[8, 784, 768]" = torch.ops.aten.clone.default(permute_759, memory_format = torch.contiguous_format);  permute_759 = None
    mul_1741: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(clone_417, primals_219);  primals_219 = None
    mul_1742: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1741, 768)
    sum_609: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1741, [2], True)
    mul_1743: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1741, mul_152);  mul_1741 = None
    sum_610: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1743, [2], True);  mul_1743 = None
    mul_1744: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_152, sum_610);  sum_610 = None
    sub_403: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(mul_1742, sum_609);  mul_1742 = sum_609 = None
    sub_404: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(sub_403, mul_1744);  sub_403 = mul_1744 = None
    mul_1745: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(div_282, sub_404);  div_282 = sub_404 = None
    mul_1746: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(clone_417, mul_152);  mul_152 = None
    sum_611: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1746, [0, 1]);  mul_1746 = None
    sum_612: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_417, [0, 1]);  clone_417 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    add_632: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_629, mul_1745);  add_629 = mul_1745 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_1747: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_632, primals_21);  primals_21 = None
    mul_1748: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_632, add_110);  add_110 = None
    sum_613: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1748, [0, 1], True);  mul_1748 = None
    view_1005: "f32[768]" = torch.ops.aten.reshape.default(sum_613, [768]);  sum_613 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    sum_614: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1747, [0, 1], True)
    view_1006: "f32[768]" = torch.ops.aten.reshape.default(sum_614, [768]);  sum_614 = None
    view_1007: "f32[6272, 768]" = torch.ops.aten.reshape.default(mul_1747, [6272, 768]);  mul_1747 = None
    permute_760: "f32[768, 6272]" = torch.ops.aten.permute.default(view_1007, [1, 0])
    mm_200: "f32[768, 768]" = torch.ops.aten.mm.default(permute_760, view_104);  permute_760 = view_104 = None
    permute_761: "f32[768, 768]" = torch.ops.aten.permute.default(mm_200, [1, 0]);  mm_200 = None
    mm_201: "f32[6272, 768]" = torch.ops.aten.mm.default(view_1007, permute_762);  view_1007 = permute_762 = None
    view_1008: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_201, [8, 784, 768]);  mm_201 = None
    permute_763: "f32[768, 768]" = torch.ops.aten.permute.default(permute_761, [1, 0]);  permute_761 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    view_1009: "f32[8, 784, 16, 48]" = torch.ops.aten.reshape.default(view_1008, [8, 784, 16, 48]);  view_1008 = None
    permute_764: "f32[8, 16, 48, 784]" = torch.ops.aten.permute.default(view_1009, [0, 2, 3, 1]);  view_1009 = None
    clone_419: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(permute_764, memory_format = torch.contiguous_format);  permute_764 = None
    view_1010: "f32[128, 48, 784]" = torch.ops.aten.reshape.default(clone_419, [128, 48, 784]);  clone_419 = None
    bmm_120: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(permute_765, view_1010);  permute_765 = None
    bmm_121: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(view_1010, permute_766);  view_1010 = permute_766 = None
    view_1011: "f32[8, 16, 48, 784]" = torch.ops.aten.reshape.default(bmm_120, [8, 16, 48, 784]);  bmm_120 = None
    view_1012: "f32[8, 16, 48, 48]" = torch.ops.aten.reshape.default(bmm_121, [8, 16, 48, 48]);  bmm_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    mul_1749: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_1012, alias_130);  view_1012 = None
    sum_615: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_1749, [-1], True)
    mul_1750: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(alias_130, sum_615);  alias_130 = sum_615 = None
    sub_405: "f32[8, 16, 48, 48]" = torch.ops.aten.sub.Tensor(mul_1749, mul_1750);  mul_1749 = mul_1750 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    mul_1751: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(sub_405, view_99);  view_99 = None
    mul_1752: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(sub_405, primals_22);  sub_405 = primals_22 = None
    sum_616: "f32[1, 16, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1751, [0, 2, 3], True);  mul_1751 = None
    view_1013: "f32[16, 1, 1]" = torch.ops.aten.reshape.default(sum_616, [16, 1, 1]);  sum_616 = None
    view_1014: "f32[128, 48, 48]" = torch.ops.aten.reshape.default(mul_1752, [128, 48, 48]);  mul_1752 = None
    bmm_122: "f32[128, 784, 48]" = torch.ops.aten.bmm.default(permute_767, view_1014);  permute_767 = None
    bmm_123: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_1014, permute_768);  view_1014 = permute_768 = None
    view_1015: "f32[8, 16, 784, 48]" = torch.ops.aten.reshape.default(bmm_122, [8, 16, 784, 48]);  bmm_122 = None
    view_1016: "f32[8, 16, 48, 784]" = torch.ops.aten.reshape.default(bmm_123, [8, 16, 48, 784]);  bmm_123 = None
    permute_769: "f32[8, 16, 48, 784]" = torch.ops.aten.permute.default(view_1015, [0, 1, 3, 2]);  view_1015 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    div_284: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_22, expand_31);  div_22 = None
    neg_36: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(permute_769)
    mul_1753: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_36, div_284);  neg_36 = div_284 = None
    div_285: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(permute_769, expand_31);  permute_769 = expand_31 = None
    sum_617: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_1753, [3], True);  mul_1753 = None
    ge_36: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(pow_25, 1e-12)
    where_72: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_36, sum_617, full_default_20);  ge_36 = sum_617 = None
    div_286: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_64, pow_25);  getitem_64 = None
    eq_36: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(pow_25, 0);  pow_25 = None
    where_73: "f32[8, 16, 48, 784]" = torch.ops.aten.where.self(eq_36, full_default_20, div_286);  eq_36 = div_286 = None
    clone_420: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(where_73, memory_format = torch.contiguous_format);  where_73 = None
    mul_1754: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_72, clone_420);  where_72 = clone_420 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    add_633: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_285, mul_1754);  div_285 = mul_1754 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    div_288: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_21, expand_30);  div_21 = None
    neg_37: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(view_1016)
    mul_1755: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_37, div_288);  neg_37 = div_288 = None
    div_289: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(view_1016, expand_30);  view_1016 = expand_30 = None
    sum_618: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_1755, [3], True);  mul_1755 = None
    ge_37: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(pow_23, 1e-12)
    where_74: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_37, sum_618, full_default_20);  ge_37 = sum_618 = None
    div_290: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_63, pow_23);  getitem_63 = None
    eq_37: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(pow_23, 0);  pow_23 = None
    where_75: "f32[8, 16, 48, 784]" = torch.ops.aten.where.self(eq_37, full_default_20, div_290);  eq_37 = div_290 = None
    clone_421: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(where_75, memory_format = torch.contiguous_format);  where_75 = None
    mul_1756: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_74, clone_421);  where_74 = clone_421 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    add_634: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_289, mul_1756);  div_289 = mul_1756 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    cat_26: "f32[24, 16, 48, 784]" = torch.ops.aten.cat.default([add_634, add_633, view_1011]);  add_634 = add_633 = view_1011 = None
    view_1017: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.reshape.default(cat_26, [3, 8, 16, 48, 784]);  cat_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    permute_770: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.permute.default(view_1017, [1, 4, 0, 2, 3]);  view_1017 = None
    clone_422: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.clone.default(permute_770, memory_format = torch.contiguous_format);  permute_770 = None
    view_1018: "f32[8, 784, 2304]" = torch.ops.aten.reshape.default(clone_422, [8, 784, 2304]);  clone_422 = None
    view_1019: "f32[6272, 2304]" = torch.ops.aten.reshape.default(view_1018, [6272, 2304]);  view_1018 = None
    mm_202: "f32[6272, 768]" = torch.ops.aten.mm.default(view_1019, permute_771);  permute_771 = None
    permute_772: "f32[2304, 6272]" = torch.ops.aten.permute.default(view_1019, [1, 0])
    mm_203: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_772, view_94);  permute_772 = view_94 = None
    permute_773: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_203, [1, 0]);  mm_203 = None
    sum_619: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_1019, [0], True);  view_1019 = None
    view_1020: "f32[2304]" = torch.ops.aten.reshape.default(sum_619, [2304]);  sum_619 = None
    permute_774: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_773, [1, 0]);  permute_773 = None
    view_1021: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_202, [8, 784, 768]);  mm_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_1758: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_1021, primals_213);  primals_213 = None
    mul_1759: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1758, 768)
    sum_620: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1758, [2], True)
    mul_1760: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1758, mul_148);  mul_1758 = None
    sum_621: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1760, [2], True);  mul_1760 = None
    mul_1761: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_148, sum_621);  sum_621 = None
    sub_407: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(mul_1759, sum_620);  mul_1759 = sum_620 = None
    sub_408: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(sub_407, mul_1761);  sub_407 = mul_1761 = None
    mul_1762: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(div_291, sub_408);  div_291 = sub_408 = None
    mul_1763: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_1021, mul_148);  mul_148 = None
    sum_622: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1763, [0, 1]);  mul_1763 = None
    sum_623: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_1021, [0, 1]);  view_1021 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    add_635: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_632, mul_1762);  add_632 = mul_1762 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_1764: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_635, primals_20);  primals_20 = None
    mul_1765: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_635, view_93);  view_93 = None
    sum_624: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1765, [0, 1], True);  mul_1765 = None
    view_1022: "f32[768]" = torch.ops.aten.reshape.default(sum_624, [768]);  sum_624 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_1023: "f32[6272, 768]" = torch.ops.aten.reshape.default(mul_1764, [6272, 768]);  mul_1764 = None
    mm_204: "f32[6272, 3072]" = torch.ops.aten.mm.default(view_1023, permute_775);  permute_775 = None
    permute_776: "f32[768, 6272]" = torch.ops.aten.permute.default(view_1023, [1, 0])
    mm_205: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_776, view_92);  permute_776 = view_92 = None
    permute_777: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_205, [1, 0]);  mm_205 = None
    sum_625: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1023, [0], True);  view_1023 = None
    view_1024: "f32[768]" = torch.ops.aten.reshape.default(sum_625, [768]);  sum_625 = None
    permute_778: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_777, [1, 0]);  permute_777 = None
    view_1025: "f32[8, 784, 3072]" = torch.ops.aten.reshape.default(mm_204, [8, 784, 3072]);  mm_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_1767: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(add_106, 0.5);  add_106 = None
    mul_1768: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_91, view_91)
    mul_1769: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(mul_1768, -0.5);  mul_1768 = None
    exp_64: "f32[8, 784, 3072]" = torch.ops.aten.exp.default(mul_1769);  mul_1769 = None
    mul_1770: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(exp_64, 0.3989422804014327);  exp_64 = None
    mul_1771: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_91, mul_1770);  view_91 = mul_1770 = None
    add_637: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(mul_1767, mul_1771);  mul_1767 = mul_1771 = None
    mul_1772: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_1025, add_637);  view_1025 = add_637 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_1026: "f32[6272, 3072]" = torch.ops.aten.reshape.default(mul_1772, [6272, 3072]);  mul_1772 = None
    mm_206: "f32[6272, 768]" = torch.ops.aten.mm.default(view_1026, permute_779);  permute_779 = None
    permute_780: "f32[3072, 6272]" = torch.ops.aten.permute.default(view_1026, [1, 0])
    mm_207: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_780, view_90);  permute_780 = view_90 = None
    permute_781: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_207, [1, 0]);  mm_207 = None
    sum_626: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_1026, [0], True);  view_1026 = None
    view_1027: "f32[3072]" = torch.ops.aten.reshape.default(sum_626, [3072]);  sum_626 = None
    permute_782: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_781, [1, 0]);  permute_781 = None
    view_1028: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_206, [8, 784, 768]);  mm_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_1774: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_1028, primals_207);  primals_207 = None
    mul_1775: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1774, 768)
    sum_627: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1774, [2], True)
    mul_1776: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1774, mul_142);  mul_1774 = None
    sum_628: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1776, [2], True);  mul_1776 = None
    mul_1777: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_142, sum_628);  sum_628 = None
    sub_410: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(mul_1775, sum_627);  mul_1775 = sum_627 = None
    sub_411: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(sub_410, mul_1777);  sub_410 = mul_1777 = None
    mul_1778: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(div_292, sub_411);  div_292 = sub_411 = None
    mul_1779: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_1028, mul_142);  mul_142 = None
    sum_629: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1779, [0, 1]);  mul_1779 = None
    sum_630: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_1028, [0, 1]);  view_1028 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    add_638: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_635, mul_1778);  add_635 = mul_1778 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_1780: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_638, primals_19);  primals_19 = None
    mul_1781: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_638, permute_45);  permute_45 = None
    sum_631: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1781, [0, 1], True);  mul_1781 = None
    view_1029: "f32[768]" = torch.ops.aten.reshape.default(sum_631, [768]);  sum_631 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    permute_783: "f32[8, 768, 784]" = torch.ops.aten.permute.default(mul_1780, [0, 2, 1]);  mul_1780 = None
    view_1030: "f32[8, 768, 28, 28]" = torch.ops.aten.reshape.default(permute_783, [8, 768, 28, 28]);  permute_783 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    sum_632: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_1030, [0, 2, 3])
    convolution_backward_38 = torch.ops.aten.convolution_backward.default(view_1030, add_102, primals_205, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False]);  view_1030 = add_102 = primals_205 = None
    getitem_410: "f32[8, 768, 28, 28]" = convolution_backward_38[0]
    getitem_411: "f32[768, 1, 3, 3]" = convolution_backward_38[1];  convolution_backward_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    sum_633: "f32[768]" = torch.ops.aten.sum.dim_IntList(getitem_410, [0, 2, 3])
    sub_412: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(mul_133, unsqueeze_347);  mul_133 = unsqueeze_347 = None
    mul_1782: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_410, sub_412)
    sum_634: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1782, [0, 2, 3]);  mul_1782 = None
    mul_1783: "f32[768]" = torch.ops.aten.mul.Tensor(sum_633, 0.00015943877551020407)
    unsqueeze_348: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1783, 0);  mul_1783 = None
    unsqueeze_349: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_348, 2);  unsqueeze_348 = None
    unsqueeze_350: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_349, 3);  unsqueeze_349 = None
    mul_1784: "f32[768]" = torch.ops.aten.mul.Tensor(sum_634, 0.00015943877551020407)
    mul_1785: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_22, squeeze_22)
    mul_1786: "f32[768]" = torch.ops.aten.mul.Tensor(mul_1784, mul_1785);  mul_1784 = mul_1785 = None
    unsqueeze_351: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1786, 0);  mul_1786 = None
    unsqueeze_352: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_351, 2);  unsqueeze_351 = None
    unsqueeze_353: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_352, 3);  unsqueeze_352 = None
    mul_1787: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_22, primals_203);  primals_203 = None
    unsqueeze_354: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1787, 0);  mul_1787 = None
    unsqueeze_355: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_354, 2);  unsqueeze_354 = None
    unsqueeze_356: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_355, 3);  unsqueeze_355 = None
    mul_1788: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_412, unsqueeze_353);  sub_412 = unsqueeze_353 = None
    sub_414: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(getitem_410, mul_1788);  getitem_410 = mul_1788 = None
    sub_415: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(sub_414, unsqueeze_350);  sub_414 = unsqueeze_350 = None
    mul_1789: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_415, unsqueeze_356);  sub_415 = unsqueeze_356 = None
    mul_1790: "f32[768]" = torch.ops.aten.mul.Tensor(sum_634, squeeze_22);  sum_634 = squeeze_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_1792: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(add_97, 0.5);  add_97 = None
    mul_1793: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_12, convolution_12)
    mul_1794: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_1793, -0.5);  mul_1793 = None
    exp_65: "f32[8, 768, 28, 28]" = torch.ops.aten.exp.default(mul_1794);  mul_1794 = None
    mul_1795: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(exp_65, 0.3989422804014327);  exp_65 = None
    mul_1796: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_12, mul_1795);  convolution_12 = mul_1795 = None
    add_640: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(mul_1792, mul_1796);  mul_1792 = mul_1796 = None
    mul_1797: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_1789, add_640);  mul_1789 = add_640 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    sum_635: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1797, [0, 2, 3])
    convolution_backward_39 = torch.ops.aten.convolution_backward.default(mul_1797, view_88, primals_201, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False]);  mul_1797 = view_88 = primals_201 = None
    getitem_413: "f32[8, 768, 28, 28]" = convolution_backward_39[0]
    getitem_414: "f32[768, 1, 3, 3]" = convolution_backward_39[1];  convolution_backward_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    view_1031: "f32[8, 768, 784]" = torch.ops.aten.reshape.default(getitem_413, [8, 768, 784]);  getitem_413 = None
    permute_784: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_1031, [0, 2, 1]);  view_1031 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    clone_425: "f32[8, 784, 768]" = torch.ops.aten.clone.default(permute_784, memory_format = torch.contiguous_format);  permute_784 = None
    mul_1799: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(clone_425, primals_199);  primals_199 = None
    mul_1800: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1799, 768)
    sum_636: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1799, [2], True)
    mul_1801: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1799, mul_129);  mul_1799 = None
    sum_637: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1801, [2], True);  mul_1801 = None
    mul_1802: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_129, sum_637);  sum_637 = None
    sub_417: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(mul_1800, sum_636);  mul_1800 = sum_636 = None
    sub_418: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(sub_417, mul_1802);  sub_417 = mul_1802 = None
    mul_1803: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(div_293, sub_418);  div_293 = sub_418 = None
    mul_1804: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(clone_425, mul_129);  mul_129 = None
    sum_638: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1804, [0, 1]);  mul_1804 = None
    sum_639: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_425, [0, 1]);  clone_425 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    add_641: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_638, mul_1803);  add_638 = mul_1803 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_1805: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_641, primals_17);  primals_17 = None
    mul_1806: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_641, add_93);  add_93 = None
    sum_640: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1806, [0, 1], True);  mul_1806 = None
    view_1032: "f32[768]" = torch.ops.aten.reshape.default(sum_640, [768]);  sum_640 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    sum_641: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1805, [0, 1], True)
    view_1033: "f32[768]" = torch.ops.aten.reshape.default(sum_641, [768]);  sum_641 = None
    view_1034: "f32[6272, 768]" = torch.ops.aten.reshape.default(mul_1805, [6272, 768]);  mul_1805 = None
    permute_785: "f32[768, 6272]" = torch.ops.aten.permute.default(view_1034, [1, 0])
    mm_208: "f32[768, 768]" = torch.ops.aten.mm.default(permute_785, view_86);  permute_785 = view_86 = None
    permute_786: "f32[768, 768]" = torch.ops.aten.permute.default(mm_208, [1, 0]);  mm_208 = None
    mm_209: "f32[6272, 768]" = torch.ops.aten.mm.default(view_1034, permute_787);  view_1034 = permute_787 = None
    view_1035: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_209, [8, 784, 768]);  mm_209 = None
    permute_788: "f32[768, 768]" = torch.ops.aten.permute.default(permute_786, [1, 0]);  permute_786 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    view_1036: "f32[8, 784, 16, 48]" = torch.ops.aten.reshape.default(view_1035, [8, 784, 16, 48]);  view_1035 = None
    permute_789: "f32[8, 16, 48, 784]" = torch.ops.aten.permute.default(view_1036, [0, 2, 3, 1]);  view_1036 = None
    clone_427: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(permute_789, memory_format = torch.contiguous_format);  permute_789 = None
    view_1037: "f32[128, 48, 784]" = torch.ops.aten.reshape.default(clone_427, [128, 48, 784]);  clone_427 = None
    bmm_124: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(permute_790, view_1037);  permute_790 = None
    bmm_125: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(view_1037, permute_791);  view_1037 = permute_791 = None
    view_1038: "f32[8, 16, 48, 784]" = torch.ops.aten.reshape.default(bmm_124, [8, 16, 48, 784]);  bmm_124 = None
    view_1039: "f32[8, 16, 48, 48]" = torch.ops.aten.reshape.default(bmm_125, [8, 16, 48, 48]);  bmm_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    mul_1807: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_1039, alias_133);  view_1039 = None
    sum_642: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_1807, [-1], True)
    mul_1808: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(alias_133, sum_642);  alias_133 = sum_642 = None
    sub_419: "f32[8, 16, 48, 48]" = torch.ops.aten.sub.Tensor(mul_1807, mul_1808);  mul_1807 = mul_1808 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    mul_1809: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(sub_419, view_81);  view_81 = None
    mul_1810: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(sub_419, primals_18);  sub_419 = primals_18 = None
    sum_643: "f32[1, 16, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1809, [0, 2, 3], True);  mul_1809 = None
    view_1040: "f32[16, 1, 1]" = torch.ops.aten.reshape.default(sum_643, [16, 1, 1]);  sum_643 = None
    view_1041: "f32[128, 48, 48]" = torch.ops.aten.reshape.default(mul_1810, [128, 48, 48]);  mul_1810 = None
    bmm_126: "f32[128, 784, 48]" = torch.ops.aten.bmm.default(permute_792, view_1041);  permute_792 = None
    bmm_127: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_1041, permute_793);  view_1041 = permute_793 = None
    view_1042: "f32[8, 16, 784, 48]" = torch.ops.aten.reshape.default(bmm_126, [8, 16, 784, 48]);  bmm_126 = None
    view_1043: "f32[8, 16, 48, 784]" = torch.ops.aten.reshape.default(bmm_127, [8, 16, 48, 784]);  bmm_127 = None
    permute_794: "f32[8, 16, 48, 784]" = torch.ops.aten.permute.default(view_1042, [0, 1, 3, 2]);  view_1042 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    div_295: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_19, expand_25);  div_19 = None
    neg_38: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(permute_794)
    mul_1811: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_38, div_295);  neg_38 = div_295 = None
    div_296: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(permute_794, expand_25);  permute_794 = expand_25 = None
    sum_644: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_1811, [3], True);  mul_1811 = None
    ge_38: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(pow_21, 1e-12)
    where_76: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_38, sum_644, full_default_20);  ge_38 = sum_644 = None
    div_297: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_53, pow_21);  getitem_53 = None
    eq_38: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(pow_21, 0);  pow_21 = None
    where_77: "f32[8, 16, 48, 784]" = torch.ops.aten.where.self(eq_38, full_default_20, div_297);  eq_38 = div_297 = None
    clone_428: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(where_77, memory_format = torch.contiguous_format);  where_77 = None
    mul_1812: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_76, clone_428);  where_76 = clone_428 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    add_642: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_296, mul_1812);  div_296 = mul_1812 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    div_299: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_18, expand_24);  div_18 = None
    neg_39: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(view_1043)
    mul_1813: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_39, div_299);  neg_39 = div_299 = None
    div_300: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(view_1043, expand_24);  view_1043 = expand_24 = None
    sum_645: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_1813, [3], True);  mul_1813 = None
    ge_39: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(pow_19, 1e-12)
    where_78: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_39, sum_645, full_default_20);  ge_39 = sum_645 = None
    div_301: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_52, pow_19);  getitem_52 = None
    eq_39: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(pow_19, 0);  pow_19 = None
    where_79: "f32[8, 16, 48, 784]" = torch.ops.aten.where.self(eq_39, full_default_20, div_301);  eq_39 = div_301 = None
    clone_429: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(where_79, memory_format = torch.contiguous_format);  where_79 = None
    mul_1814: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_78, clone_429);  where_78 = clone_429 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    add_643: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_300, mul_1814);  div_300 = mul_1814 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    cat_27: "f32[24, 16, 48, 784]" = torch.ops.aten.cat.default([add_643, add_642, view_1038]);  add_643 = add_642 = view_1038 = None
    view_1044: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.reshape.default(cat_27, [3, 8, 16, 48, 784]);  cat_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    permute_795: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.permute.default(view_1044, [1, 4, 0, 2, 3]);  view_1044 = None
    clone_430: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.clone.default(permute_795, memory_format = torch.contiguous_format);  permute_795 = None
    view_1045: "f32[8, 784, 2304]" = torch.ops.aten.reshape.default(clone_430, [8, 784, 2304]);  clone_430 = None
    view_1046: "f32[6272, 2304]" = torch.ops.aten.reshape.default(view_1045, [6272, 2304]);  view_1045 = None
    mm_210: "f32[6272, 768]" = torch.ops.aten.mm.default(view_1046, permute_796);  permute_796 = None
    permute_797: "f32[2304, 6272]" = torch.ops.aten.permute.default(view_1046, [1, 0])
    mm_211: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_797, view_76);  permute_797 = view_76 = None
    permute_798: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_211, [1, 0]);  mm_211 = None
    sum_646: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_1046, [0], True);  view_1046 = None
    view_1047: "f32[2304]" = torch.ops.aten.reshape.default(sum_646, [2304]);  sum_646 = None
    permute_799: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_798, [1, 0]);  permute_798 = None
    view_1048: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_210, [8, 784, 768]);  mm_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_1816: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_1048, primals_193);  primals_193 = None
    mul_1817: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1816, 768)
    sum_647: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1816, [2], True)
    mul_1818: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1816, mul_125);  mul_1816 = None
    sum_648: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1818, [2], True);  mul_1818 = None
    mul_1819: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_125, sum_648);  sum_648 = None
    sub_421: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(mul_1817, sum_647);  mul_1817 = sum_647 = None
    sub_422: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(sub_421, mul_1819);  sub_421 = mul_1819 = None
    mul_1820: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(div_302, sub_422);  div_302 = sub_422 = None
    mul_1821: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_1048, mul_125);  mul_125 = None
    sum_649: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1821, [0, 1]);  mul_1821 = None
    sum_650: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_1048, [0, 1]);  view_1048 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    add_644: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_641, mul_1820);  add_641 = mul_1820 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_1822: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_644, primals_16);  primals_16 = None
    mul_1823: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_644, view_75);  view_75 = None
    sum_651: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1823, [0, 1], True);  mul_1823 = None
    view_1049: "f32[768]" = torch.ops.aten.reshape.default(sum_651, [768]);  sum_651 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_1050: "f32[6272, 768]" = torch.ops.aten.reshape.default(mul_1822, [6272, 768]);  mul_1822 = None
    mm_212: "f32[6272, 3072]" = torch.ops.aten.mm.default(view_1050, permute_800);  permute_800 = None
    permute_801: "f32[768, 6272]" = torch.ops.aten.permute.default(view_1050, [1, 0])
    mm_213: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_801, view_74);  permute_801 = view_74 = None
    permute_802: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_213, [1, 0]);  mm_213 = None
    sum_652: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1050, [0], True);  view_1050 = None
    view_1051: "f32[768]" = torch.ops.aten.reshape.default(sum_652, [768]);  sum_652 = None
    permute_803: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_802, [1, 0]);  permute_802 = None
    view_1052: "f32[8, 784, 3072]" = torch.ops.aten.reshape.default(mm_212, [8, 784, 3072]);  mm_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_1825: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(add_89, 0.5);  add_89 = None
    mul_1826: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_73, view_73)
    mul_1827: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(mul_1826, -0.5);  mul_1826 = None
    exp_66: "f32[8, 784, 3072]" = torch.ops.aten.exp.default(mul_1827);  mul_1827 = None
    mul_1828: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(exp_66, 0.3989422804014327);  exp_66 = None
    mul_1829: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_73, mul_1828);  view_73 = mul_1828 = None
    add_646: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(mul_1825, mul_1829);  mul_1825 = mul_1829 = None
    mul_1830: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_1052, add_646);  view_1052 = add_646 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_1053: "f32[6272, 3072]" = torch.ops.aten.reshape.default(mul_1830, [6272, 3072]);  mul_1830 = None
    mm_214: "f32[6272, 768]" = torch.ops.aten.mm.default(view_1053, permute_804);  permute_804 = None
    permute_805: "f32[3072, 6272]" = torch.ops.aten.permute.default(view_1053, [1, 0])
    mm_215: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_805, view_72);  permute_805 = view_72 = None
    permute_806: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_215, [1, 0]);  mm_215 = None
    sum_653: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_1053, [0], True);  view_1053 = None
    view_1054: "f32[3072]" = torch.ops.aten.reshape.default(sum_653, [3072]);  sum_653 = None
    permute_807: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_806, [1, 0]);  permute_806 = None
    view_1055: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_214, [8, 784, 768]);  mm_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_1832: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_1055, primals_187);  primals_187 = None
    mul_1833: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1832, 768)
    sum_654: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1832, [2], True)
    mul_1834: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1832, mul_119);  mul_1832 = None
    sum_655: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1834, [2], True);  mul_1834 = None
    mul_1835: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_119, sum_655);  sum_655 = None
    sub_424: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(mul_1833, sum_654);  mul_1833 = sum_654 = None
    sub_425: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(sub_424, mul_1835);  sub_424 = mul_1835 = None
    mul_1836: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(div_303, sub_425);  div_303 = sub_425 = None
    mul_1837: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_1055, mul_119);  mul_119 = None
    sum_656: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1837, [0, 1]);  mul_1837 = None
    sum_657: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_1055, [0, 1]);  view_1055 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    add_647: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_644, mul_1836);  add_644 = mul_1836 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_1838: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_647, primals_15);  primals_15 = None
    mul_1839: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_647, permute_36);  permute_36 = None
    sum_658: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1839, [0, 1], True);  mul_1839 = None
    view_1056: "f32[768]" = torch.ops.aten.reshape.default(sum_658, [768]);  sum_658 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    permute_808: "f32[8, 768, 784]" = torch.ops.aten.permute.default(mul_1838, [0, 2, 1]);  mul_1838 = None
    view_1057: "f32[8, 768, 28, 28]" = torch.ops.aten.reshape.default(permute_808, [8, 768, 28, 28]);  permute_808 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    sum_659: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_1057, [0, 2, 3])
    convolution_backward_40 = torch.ops.aten.convolution_backward.default(view_1057, add_85, primals_185, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False]);  view_1057 = add_85 = primals_185 = None
    getitem_416: "f32[8, 768, 28, 28]" = convolution_backward_40[0]
    getitem_417: "f32[768, 1, 3, 3]" = convolution_backward_40[1];  convolution_backward_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    sum_660: "f32[768]" = torch.ops.aten.sum.dim_IntList(getitem_416, [0, 2, 3])
    sub_426: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(mul_110, unsqueeze_359);  mul_110 = unsqueeze_359 = None
    mul_1840: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_416, sub_426)
    sum_661: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1840, [0, 2, 3]);  mul_1840 = None
    mul_1841: "f32[768]" = torch.ops.aten.mul.Tensor(sum_660, 0.00015943877551020407)
    unsqueeze_360: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1841, 0);  mul_1841 = None
    unsqueeze_361: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_360, 2);  unsqueeze_360 = None
    unsqueeze_362: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_361, 3);  unsqueeze_361 = None
    mul_1842: "f32[768]" = torch.ops.aten.mul.Tensor(sum_661, 0.00015943877551020407)
    mul_1843: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_19, squeeze_19)
    mul_1844: "f32[768]" = torch.ops.aten.mul.Tensor(mul_1842, mul_1843);  mul_1842 = mul_1843 = None
    unsqueeze_363: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1844, 0);  mul_1844 = None
    unsqueeze_364: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_363, 2);  unsqueeze_363 = None
    unsqueeze_365: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_364, 3);  unsqueeze_364 = None
    mul_1845: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_19, primals_183);  primals_183 = None
    unsqueeze_366: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1845, 0);  mul_1845 = None
    unsqueeze_367: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_366, 2);  unsqueeze_366 = None
    unsqueeze_368: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_367, 3);  unsqueeze_367 = None
    mul_1846: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_426, unsqueeze_365);  sub_426 = unsqueeze_365 = None
    sub_428: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(getitem_416, mul_1846);  getitem_416 = mul_1846 = None
    sub_429: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(sub_428, unsqueeze_362);  sub_428 = unsqueeze_362 = None
    mul_1847: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_429, unsqueeze_368);  sub_429 = unsqueeze_368 = None
    mul_1848: "f32[768]" = torch.ops.aten.mul.Tensor(sum_661, squeeze_19);  sum_661 = squeeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_1850: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(add_80, 0.5);  add_80 = None
    mul_1851: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_10, convolution_10)
    mul_1852: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_1851, -0.5);  mul_1851 = None
    exp_67: "f32[8, 768, 28, 28]" = torch.ops.aten.exp.default(mul_1852);  mul_1852 = None
    mul_1853: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(exp_67, 0.3989422804014327);  exp_67 = None
    mul_1854: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_10, mul_1853);  convolution_10 = mul_1853 = None
    add_649: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(mul_1850, mul_1854);  mul_1850 = mul_1854 = None
    mul_1855: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_1847, add_649);  mul_1847 = add_649 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    sum_662: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1855, [0, 2, 3])
    convolution_backward_41 = torch.ops.aten.convolution_backward.default(mul_1855, view_70, primals_181, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False]);  mul_1855 = view_70 = primals_181 = None
    getitem_419: "f32[8, 768, 28, 28]" = convolution_backward_41[0]
    getitem_420: "f32[768, 1, 3, 3]" = convolution_backward_41[1];  convolution_backward_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    view_1058: "f32[8, 768, 784]" = torch.ops.aten.reshape.default(getitem_419, [8, 768, 784]);  getitem_419 = None
    permute_809: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_1058, [0, 2, 1]);  view_1058 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    clone_433: "f32[8, 784, 768]" = torch.ops.aten.clone.default(permute_809, memory_format = torch.contiguous_format);  permute_809 = None
    mul_1857: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(clone_433, primals_179);  primals_179 = None
    mul_1858: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1857, 768)
    sum_663: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1857, [2], True)
    mul_1859: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1857, mul_106);  mul_1857 = None
    sum_664: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1859, [2], True);  mul_1859 = None
    mul_1860: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_106, sum_664);  sum_664 = None
    sub_431: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(mul_1858, sum_663);  mul_1858 = sum_663 = None
    sub_432: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(sub_431, mul_1860);  sub_431 = mul_1860 = None
    mul_1861: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(div_304, sub_432);  div_304 = sub_432 = None
    mul_1862: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(clone_433, mul_106);  mul_106 = None
    sum_665: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1862, [0, 1]);  mul_1862 = None
    sum_666: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_433, [0, 1]);  clone_433 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    add_650: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_647, mul_1861);  add_647 = mul_1861 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_1863: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_650, primals_13);  primals_13 = None
    mul_1864: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_650, add_76);  add_76 = None
    sum_667: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1864, [0, 1], True);  mul_1864 = None
    view_1059: "f32[768]" = torch.ops.aten.reshape.default(sum_667, [768]);  sum_667 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    sum_668: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1863, [0, 1], True)
    view_1060: "f32[768]" = torch.ops.aten.reshape.default(sum_668, [768]);  sum_668 = None
    view_1061: "f32[6272, 768]" = torch.ops.aten.reshape.default(mul_1863, [6272, 768]);  mul_1863 = None
    permute_810: "f32[768, 6272]" = torch.ops.aten.permute.default(view_1061, [1, 0])
    mm_216: "f32[768, 768]" = torch.ops.aten.mm.default(permute_810, view_68);  permute_810 = view_68 = None
    permute_811: "f32[768, 768]" = torch.ops.aten.permute.default(mm_216, [1, 0]);  mm_216 = None
    mm_217: "f32[6272, 768]" = torch.ops.aten.mm.default(view_1061, permute_812);  view_1061 = permute_812 = None
    view_1062: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_217, [8, 784, 768]);  mm_217 = None
    permute_813: "f32[768, 768]" = torch.ops.aten.permute.default(permute_811, [1, 0]);  permute_811 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    view_1063: "f32[8, 784, 16, 48]" = torch.ops.aten.reshape.default(view_1062, [8, 784, 16, 48]);  view_1062 = None
    permute_814: "f32[8, 16, 48, 784]" = torch.ops.aten.permute.default(view_1063, [0, 2, 3, 1]);  view_1063 = None
    clone_435: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(permute_814, memory_format = torch.contiguous_format);  permute_814 = None
    view_1064: "f32[128, 48, 784]" = torch.ops.aten.reshape.default(clone_435, [128, 48, 784]);  clone_435 = None
    bmm_128: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(permute_815, view_1064);  permute_815 = None
    bmm_129: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(view_1064, permute_816);  view_1064 = permute_816 = None
    view_1065: "f32[8, 16, 48, 784]" = torch.ops.aten.reshape.default(bmm_128, [8, 16, 48, 784]);  bmm_128 = None
    view_1066: "f32[8, 16, 48, 48]" = torch.ops.aten.reshape.default(bmm_129, [8, 16, 48, 48]);  bmm_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    mul_1865: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_1066, alias_136);  view_1066 = None
    sum_669: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_1865, [-1], True)
    mul_1866: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(alias_136, sum_669);  alias_136 = sum_669 = None
    sub_433: "f32[8, 16, 48, 48]" = torch.ops.aten.sub.Tensor(mul_1865, mul_1866);  mul_1865 = mul_1866 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    mul_1867: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(sub_433, view_63);  view_63 = None
    mul_1868: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(sub_433, primals_14);  sub_433 = primals_14 = None
    sum_670: "f32[1, 16, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1867, [0, 2, 3], True);  mul_1867 = None
    view_1067: "f32[16, 1, 1]" = torch.ops.aten.reshape.default(sum_670, [16, 1, 1]);  sum_670 = None
    view_1068: "f32[128, 48, 48]" = torch.ops.aten.reshape.default(mul_1868, [128, 48, 48]);  mul_1868 = None
    bmm_130: "f32[128, 784, 48]" = torch.ops.aten.bmm.default(permute_817, view_1068);  permute_817 = None
    bmm_131: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_1068, permute_818);  view_1068 = permute_818 = None
    view_1069: "f32[8, 16, 784, 48]" = torch.ops.aten.reshape.default(bmm_130, [8, 16, 784, 48]);  bmm_130 = None
    view_1070: "f32[8, 16, 48, 784]" = torch.ops.aten.reshape.default(bmm_131, [8, 16, 48, 784]);  bmm_131 = None
    permute_819: "f32[8, 16, 48, 784]" = torch.ops.aten.permute.default(view_1069, [0, 1, 3, 2]);  view_1069 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    div_306: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_16, expand_19);  div_16 = None
    neg_40: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(permute_819)
    mul_1869: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_40, div_306);  neg_40 = div_306 = None
    div_307: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(permute_819, expand_19);  permute_819 = expand_19 = None
    sum_671: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_1869, [3], True);  mul_1869 = None
    ge_40: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(pow_17, 1e-12)
    where_80: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_40, sum_671, full_default_20);  ge_40 = sum_671 = None
    div_308: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_42, pow_17);  getitem_42 = None
    eq_40: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(pow_17, 0);  pow_17 = None
    where_81: "f32[8, 16, 48, 784]" = torch.ops.aten.where.self(eq_40, full_default_20, div_308);  eq_40 = div_308 = None
    clone_436: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(where_81, memory_format = torch.contiguous_format);  where_81 = None
    mul_1870: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_80, clone_436);  where_80 = clone_436 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    add_651: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_307, mul_1870);  div_307 = mul_1870 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    div_310: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_15, expand_18);  div_15 = None
    neg_41: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(view_1070)
    mul_1871: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_41, div_310);  neg_41 = div_310 = None
    div_311: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(view_1070, expand_18);  view_1070 = expand_18 = None
    sum_672: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_1871, [3], True);  mul_1871 = None
    ge_41: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(pow_15, 1e-12)
    where_82: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_41, sum_672, full_default_20);  ge_41 = sum_672 = None
    div_312: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_41, pow_15);  getitem_41 = None
    eq_41: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(pow_15, 0);  pow_15 = None
    where_83: "f32[8, 16, 48, 784]" = torch.ops.aten.where.self(eq_41, full_default_20, div_312);  eq_41 = div_312 = None
    clone_437: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(where_83, memory_format = torch.contiguous_format);  where_83 = None
    mul_1872: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_82, clone_437);  where_82 = clone_437 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    add_652: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_311, mul_1872);  div_311 = mul_1872 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    cat_28: "f32[24, 16, 48, 784]" = torch.ops.aten.cat.default([add_652, add_651, view_1065]);  add_652 = add_651 = view_1065 = None
    view_1071: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.reshape.default(cat_28, [3, 8, 16, 48, 784]);  cat_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    permute_820: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.permute.default(view_1071, [1, 4, 0, 2, 3]);  view_1071 = None
    clone_438: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.clone.default(permute_820, memory_format = torch.contiguous_format);  permute_820 = None
    view_1072: "f32[8, 784, 2304]" = torch.ops.aten.reshape.default(clone_438, [8, 784, 2304]);  clone_438 = None
    view_1073: "f32[6272, 2304]" = torch.ops.aten.reshape.default(view_1072, [6272, 2304]);  view_1072 = None
    mm_218: "f32[6272, 768]" = torch.ops.aten.mm.default(view_1073, permute_821);  permute_821 = None
    permute_822: "f32[2304, 6272]" = torch.ops.aten.permute.default(view_1073, [1, 0])
    mm_219: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_822, view_58);  permute_822 = view_58 = None
    permute_823: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_219, [1, 0]);  mm_219 = None
    sum_673: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_1073, [0], True);  view_1073 = None
    view_1074: "f32[2304]" = torch.ops.aten.reshape.default(sum_673, [2304]);  sum_673 = None
    permute_824: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_823, [1, 0]);  permute_823 = None
    view_1075: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_218, [8, 784, 768]);  mm_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_1874: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_1075, primals_173);  primals_173 = None
    mul_1875: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1874, 768)
    sum_674: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1874, [2], True)
    mul_1876: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1874, mul_102);  mul_1874 = None
    sum_675: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1876, [2], True);  mul_1876 = None
    mul_1877: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_102, sum_675);  sum_675 = None
    sub_435: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(mul_1875, sum_674);  mul_1875 = sum_674 = None
    sub_436: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(sub_435, mul_1877);  sub_435 = mul_1877 = None
    mul_1878: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(div_313, sub_436);  div_313 = sub_436 = None
    mul_1879: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_1075, mul_102);  mul_102 = None
    sum_676: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1879, [0, 1]);  mul_1879 = None
    sum_677: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_1075, [0, 1]);  view_1075 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    add_653: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_650, mul_1878);  add_650 = mul_1878 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_1880: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_653, primals_12);  primals_12 = None
    mul_1881: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_653, view_57);  view_57 = None
    sum_678: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1881, [0, 1], True);  mul_1881 = None
    view_1076: "f32[768]" = torch.ops.aten.reshape.default(sum_678, [768]);  sum_678 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_1077: "f32[6272, 768]" = torch.ops.aten.reshape.default(mul_1880, [6272, 768]);  mul_1880 = None
    mm_220: "f32[6272, 3072]" = torch.ops.aten.mm.default(view_1077, permute_825);  permute_825 = None
    permute_826: "f32[768, 6272]" = torch.ops.aten.permute.default(view_1077, [1, 0])
    mm_221: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_826, view_56);  permute_826 = view_56 = None
    permute_827: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_221, [1, 0]);  mm_221 = None
    sum_679: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1077, [0], True);  view_1077 = None
    view_1078: "f32[768]" = torch.ops.aten.reshape.default(sum_679, [768]);  sum_679 = None
    permute_828: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_827, [1, 0]);  permute_827 = None
    view_1079: "f32[8, 784, 3072]" = torch.ops.aten.reshape.default(mm_220, [8, 784, 3072]);  mm_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_1883: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(add_72, 0.5);  add_72 = None
    mul_1884: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_55, view_55)
    mul_1885: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(mul_1884, -0.5);  mul_1884 = None
    exp_68: "f32[8, 784, 3072]" = torch.ops.aten.exp.default(mul_1885);  mul_1885 = None
    mul_1886: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(exp_68, 0.3989422804014327);  exp_68 = None
    mul_1887: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_55, mul_1886);  view_55 = mul_1886 = None
    add_655: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(mul_1883, mul_1887);  mul_1883 = mul_1887 = None
    mul_1888: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_1079, add_655);  view_1079 = add_655 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_1080: "f32[6272, 3072]" = torch.ops.aten.reshape.default(mul_1888, [6272, 3072]);  mul_1888 = None
    mm_222: "f32[6272, 768]" = torch.ops.aten.mm.default(view_1080, permute_829);  permute_829 = None
    permute_830: "f32[3072, 6272]" = torch.ops.aten.permute.default(view_1080, [1, 0])
    mm_223: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_830, view_54);  permute_830 = view_54 = None
    permute_831: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_223, [1, 0]);  mm_223 = None
    sum_680: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_1080, [0], True);  view_1080 = None
    view_1081: "f32[3072]" = torch.ops.aten.reshape.default(sum_680, [3072]);  sum_680 = None
    permute_832: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_831, [1, 0]);  permute_831 = None
    view_1082: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_222, [8, 784, 768]);  mm_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_1890: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_1082, primals_167);  primals_167 = None
    mul_1891: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1890, 768)
    sum_681: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1890, [2], True)
    mul_1892: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1890, mul_96);  mul_1890 = None
    sum_682: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1892, [2], True);  mul_1892 = None
    mul_1893: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_96, sum_682);  sum_682 = None
    sub_438: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(mul_1891, sum_681);  mul_1891 = sum_681 = None
    sub_439: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(sub_438, mul_1893);  sub_438 = mul_1893 = None
    mul_1894: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(div_314, sub_439);  div_314 = sub_439 = None
    mul_1895: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_1082, mul_96);  mul_96 = None
    sum_683: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1895, [0, 1]);  mul_1895 = None
    sum_684: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_1082, [0, 1]);  view_1082 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    add_656: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_653, mul_1894);  add_653 = mul_1894 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_1896: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_656, primals_11);  primals_11 = None
    mul_1897: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_656, permute_27);  permute_27 = None
    sum_685: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1897, [0, 1], True);  mul_1897 = None
    view_1083: "f32[768]" = torch.ops.aten.reshape.default(sum_685, [768]);  sum_685 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    permute_833: "f32[8, 768, 784]" = torch.ops.aten.permute.default(mul_1896, [0, 2, 1]);  mul_1896 = None
    view_1084: "f32[8, 768, 28, 28]" = torch.ops.aten.reshape.default(permute_833, [8, 768, 28, 28]);  permute_833 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    sum_686: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_1084, [0, 2, 3])
    convolution_backward_42 = torch.ops.aten.convolution_backward.default(view_1084, add_68, primals_165, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False]);  view_1084 = add_68 = primals_165 = None
    getitem_422: "f32[8, 768, 28, 28]" = convolution_backward_42[0]
    getitem_423: "f32[768, 1, 3, 3]" = convolution_backward_42[1];  convolution_backward_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    sum_687: "f32[768]" = torch.ops.aten.sum.dim_IntList(getitem_422, [0, 2, 3])
    sub_440: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(mul_87, unsqueeze_371);  mul_87 = unsqueeze_371 = None
    mul_1898: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_422, sub_440)
    sum_688: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1898, [0, 2, 3]);  mul_1898 = None
    mul_1899: "f32[768]" = torch.ops.aten.mul.Tensor(sum_687, 0.00015943877551020407)
    unsqueeze_372: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1899, 0);  mul_1899 = None
    unsqueeze_373: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_372, 2);  unsqueeze_372 = None
    unsqueeze_374: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_373, 3);  unsqueeze_373 = None
    mul_1900: "f32[768]" = torch.ops.aten.mul.Tensor(sum_688, 0.00015943877551020407)
    mul_1901: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_16, squeeze_16)
    mul_1902: "f32[768]" = torch.ops.aten.mul.Tensor(mul_1900, mul_1901);  mul_1900 = mul_1901 = None
    unsqueeze_375: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1902, 0);  mul_1902 = None
    unsqueeze_376: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_375, 2);  unsqueeze_375 = None
    unsqueeze_377: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_376, 3);  unsqueeze_376 = None
    mul_1903: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_16, primals_163);  primals_163 = None
    unsqueeze_378: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1903, 0);  mul_1903 = None
    unsqueeze_379: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_378, 2);  unsqueeze_378 = None
    unsqueeze_380: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_379, 3);  unsqueeze_379 = None
    mul_1904: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_440, unsqueeze_377);  sub_440 = unsqueeze_377 = None
    sub_442: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(getitem_422, mul_1904);  getitem_422 = mul_1904 = None
    sub_443: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(sub_442, unsqueeze_374);  sub_442 = unsqueeze_374 = None
    mul_1905: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_443, unsqueeze_380);  sub_443 = unsqueeze_380 = None
    mul_1906: "f32[768]" = torch.ops.aten.mul.Tensor(sum_688, squeeze_16);  sum_688 = squeeze_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_1908: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(add_63, 0.5);  add_63 = None
    mul_1909: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_8, convolution_8)
    mul_1910: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_1909, -0.5);  mul_1909 = None
    exp_69: "f32[8, 768, 28, 28]" = torch.ops.aten.exp.default(mul_1910);  mul_1910 = None
    mul_1911: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(exp_69, 0.3989422804014327);  exp_69 = None
    mul_1912: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_8, mul_1911);  convolution_8 = mul_1911 = None
    add_658: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(mul_1908, mul_1912);  mul_1908 = mul_1912 = None
    mul_1913: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_1905, add_658);  mul_1905 = add_658 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    sum_689: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1913, [0, 2, 3])
    convolution_backward_43 = torch.ops.aten.convolution_backward.default(mul_1913, view_52, primals_161, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False]);  mul_1913 = view_52 = primals_161 = None
    getitem_425: "f32[8, 768, 28, 28]" = convolution_backward_43[0]
    getitem_426: "f32[768, 1, 3, 3]" = convolution_backward_43[1];  convolution_backward_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    view_1085: "f32[8, 768, 784]" = torch.ops.aten.reshape.default(getitem_425, [8, 768, 784]);  getitem_425 = None
    permute_834: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_1085, [0, 2, 1]);  view_1085 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    clone_441: "f32[8, 784, 768]" = torch.ops.aten.clone.default(permute_834, memory_format = torch.contiguous_format);  permute_834 = None
    mul_1915: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(clone_441, primals_159);  primals_159 = None
    mul_1916: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1915, 768)
    sum_690: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1915, [2], True)
    mul_1917: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1915, mul_83);  mul_1915 = None
    sum_691: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1917, [2], True);  mul_1917 = None
    mul_1918: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_83, sum_691);  sum_691 = None
    sub_445: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(mul_1916, sum_690);  mul_1916 = sum_690 = None
    sub_446: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(sub_445, mul_1918);  sub_445 = mul_1918 = None
    mul_1919: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(div_315, sub_446);  div_315 = sub_446 = None
    mul_1920: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(clone_441, mul_83);  mul_83 = None
    sum_692: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1920, [0, 1]);  mul_1920 = None
    sum_693: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_441, [0, 1]);  clone_441 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    add_659: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_656, mul_1919);  add_656 = mul_1919 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_1921: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_659, primals_9);  primals_9 = None
    mul_1922: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_659, add_59);  add_59 = None
    sum_694: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1922, [0, 1], True);  mul_1922 = None
    view_1086: "f32[768]" = torch.ops.aten.reshape.default(sum_694, [768]);  sum_694 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    sum_695: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1921, [0, 1], True)
    view_1087: "f32[768]" = torch.ops.aten.reshape.default(sum_695, [768]);  sum_695 = None
    view_1088: "f32[6272, 768]" = torch.ops.aten.reshape.default(mul_1921, [6272, 768]);  mul_1921 = None
    permute_835: "f32[768, 6272]" = torch.ops.aten.permute.default(view_1088, [1, 0])
    mm_224: "f32[768, 768]" = torch.ops.aten.mm.default(permute_835, view_50);  permute_835 = view_50 = None
    permute_836: "f32[768, 768]" = torch.ops.aten.permute.default(mm_224, [1, 0]);  mm_224 = None
    mm_225: "f32[6272, 768]" = torch.ops.aten.mm.default(view_1088, permute_837);  view_1088 = permute_837 = None
    view_1089: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_225, [8, 784, 768]);  mm_225 = None
    permute_838: "f32[768, 768]" = torch.ops.aten.permute.default(permute_836, [1, 0]);  permute_836 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    view_1090: "f32[8, 784, 16, 48]" = torch.ops.aten.reshape.default(view_1089, [8, 784, 16, 48]);  view_1089 = None
    permute_839: "f32[8, 16, 48, 784]" = torch.ops.aten.permute.default(view_1090, [0, 2, 3, 1]);  view_1090 = None
    clone_443: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(permute_839, memory_format = torch.contiguous_format);  permute_839 = None
    view_1091: "f32[128, 48, 784]" = torch.ops.aten.reshape.default(clone_443, [128, 48, 784]);  clone_443 = None
    bmm_132: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(permute_840, view_1091);  permute_840 = None
    bmm_133: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(view_1091, permute_841);  view_1091 = permute_841 = None
    view_1092: "f32[8, 16, 48, 784]" = torch.ops.aten.reshape.default(bmm_132, [8, 16, 48, 784]);  bmm_132 = None
    view_1093: "f32[8, 16, 48, 48]" = torch.ops.aten.reshape.default(bmm_133, [8, 16, 48, 48]);  bmm_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    mul_1923: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_1093, alias_139);  view_1093 = None
    sum_696: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_1923, [-1], True)
    mul_1924: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(alias_139, sum_696);  alias_139 = sum_696 = None
    sub_447: "f32[8, 16, 48, 48]" = torch.ops.aten.sub.Tensor(mul_1923, mul_1924);  mul_1923 = mul_1924 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    mul_1925: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(sub_447, view_45);  view_45 = None
    mul_1926: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(sub_447, primals_10);  sub_447 = primals_10 = None
    sum_697: "f32[1, 16, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1925, [0, 2, 3], True);  mul_1925 = None
    view_1094: "f32[16, 1, 1]" = torch.ops.aten.reshape.default(sum_697, [16, 1, 1]);  sum_697 = None
    view_1095: "f32[128, 48, 48]" = torch.ops.aten.reshape.default(mul_1926, [128, 48, 48]);  mul_1926 = None
    bmm_134: "f32[128, 784, 48]" = torch.ops.aten.bmm.default(permute_842, view_1095);  permute_842 = None
    bmm_135: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_1095, permute_843);  view_1095 = permute_843 = None
    view_1096: "f32[8, 16, 784, 48]" = torch.ops.aten.reshape.default(bmm_134, [8, 16, 784, 48]);  bmm_134 = None
    view_1097: "f32[8, 16, 48, 784]" = torch.ops.aten.reshape.default(bmm_135, [8, 16, 48, 784]);  bmm_135 = None
    permute_844: "f32[8, 16, 48, 784]" = torch.ops.aten.permute.default(view_1096, [0, 1, 3, 2]);  view_1096 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    div_317: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_13, expand_13);  div_13 = None
    neg_42: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(permute_844)
    mul_1927: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_42, div_317);  neg_42 = div_317 = None
    div_318: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(permute_844, expand_13);  permute_844 = expand_13 = None
    sum_698: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_1927, [3], True);  mul_1927 = None
    ge_42: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(pow_13, 1e-12)
    where_84: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_42, sum_698, full_default_20);  ge_42 = sum_698 = None
    div_319: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_31, pow_13);  getitem_31 = None
    eq_42: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(pow_13, 0);  pow_13 = None
    where_85: "f32[8, 16, 48, 784]" = torch.ops.aten.where.self(eq_42, full_default_20, div_319);  eq_42 = div_319 = None
    clone_444: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(where_85, memory_format = torch.contiguous_format);  where_85 = None
    mul_1928: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_84, clone_444);  where_84 = clone_444 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    add_660: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_318, mul_1928);  div_318 = mul_1928 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    div_321: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_12, expand_12);  div_12 = None
    neg_43: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(view_1097)
    mul_1929: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_43, div_321);  neg_43 = div_321 = None
    div_322: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(view_1097, expand_12);  view_1097 = expand_12 = None
    sum_699: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_1929, [3], True);  mul_1929 = None
    ge_43: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(pow_11, 1e-12)
    where_86: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_43, sum_699, full_default_20);  ge_43 = sum_699 = None
    div_323: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_30, pow_11);  getitem_30 = None
    eq_43: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(pow_11, 0);  pow_11 = None
    where_87: "f32[8, 16, 48, 784]" = torch.ops.aten.where.self(eq_43, full_default_20, div_323);  eq_43 = div_323 = None
    clone_445: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(where_87, memory_format = torch.contiguous_format);  where_87 = None
    mul_1930: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_86, clone_445);  where_86 = clone_445 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    add_661: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_322, mul_1930);  div_322 = mul_1930 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    cat_29: "f32[24, 16, 48, 784]" = torch.ops.aten.cat.default([add_661, add_660, view_1092]);  add_661 = add_660 = view_1092 = None
    view_1098: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.reshape.default(cat_29, [3, 8, 16, 48, 784]);  cat_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    permute_845: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.permute.default(view_1098, [1, 4, 0, 2, 3]);  view_1098 = None
    clone_446: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.clone.default(permute_845, memory_format = torch.contiguous_format);  permute_845 = None
    view_1099: "f32[8, 784, 2304]" = torch.ops.aten.reshape.default(clone_446, [8, 784, 2304]);  clone_446 = None
    view_1100: "f32[6272, 2304]" = torch.ops.aten.reshape.default(view_1099, [6272, 2304]);  view_1099 = None
    mm_226: "f32[6272, 768]" = torch.ops.aten.mm.default(view_1100, permute_846);  permute_846 = None
    permute_847: "f32[2304, 6272]" = torch.ops.aten.permute.default(view_1100, [1, 0])
    mm_227: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_847, view_40);  permute_847 = view_40 = None
    permute_848: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_227, [1, 0]);  mm_227 = None
    sum_700: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_1100, [0], True);  view_1100 = None
    view_1101: "f32[2304]" = torch.ops.aten.reshape.default(sum_700, [2304]);  sum_700 = None
    permute_849: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_848, [1, 0]);  permute_848 = None
    view_1102: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_226, [8, 784, 768]);  mm_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_1932: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_1102, primals_153);  primals_153 = None
    mul_1933: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1932, 768)
    sum_701: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1932, [2], True)
    mul_1934: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1932, mul_79);  mul_1932 = None
    sum_702: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1934, [2], True);  mul_1934 = None
    mul_1935: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_79, sum_702);  sum_702 = None
    sub_449: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(mul_1933, sum_701);  mul_1933 = sum_701 = None
    sub_450: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(sub_449, mul_1935);  sub_449 = mul_1935 = None
    mul_1936: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(div_324, sub_450);  div_324 = sub_450 = None
    mul_1937: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_1102, mul_79);  mul_79 = None
    sum_703: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1937, [0, 1]);  mul_1937 = None
    sum_704: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_1102, [0, 1]);  view_1102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    add_662: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_659, mul_1936);  add_659 = mul_1936 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_1938: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_662, primals_8);  primals_8 = None
    mul_1939: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_662, view_39);  view_39 = None
    sum_705: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1939, [0, 1], True);  mul_1939 = None
    view_1103: "f32[768]" = torch.ops.aten.reshape.default(sum_705, [768]);  sum_705 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_1104: "f32[6272, 768]" = torch.ops.aten.reshape.default(mul_1938, [6272, 768]);  mul_1938 = None
    mm_228: "f32[6272, 3072]" = torch.ops.aten.mm.default(view_1104, permute_850);  permute_850 = None
    permute_851: "f32[768, 6272]" = torch.ops.aten.permute.default(view_1104, [1, 0])
    mm_229: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_851, view_38);  permute_851 = view_38 = None
    permute_852: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_229, [1, 0]);  mm_229 = None
    sum_706: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1104, [0], True);  view_1104 = None
    view_1105: "f32[768]" = torch.ops.aten.reshape.default(sum_706, [768]);  sum_706 = None
    permute_853: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_852, [1, 0]);  permute_852 = None
    view_1106: "f32[8, 784, 3072]" = torch.ops.aten.reshape.default(mm_228, [8, 784, 3072]);  mm_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_1941: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(add_55, 0.5);  add_55 = None
    mul_1942: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_37, view_37)
    mul_1943: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(mul_1942, -0.5);  mul_1942 = None
    exp_70: "f32[8, 784, 3072]" = torch.ops.aten.exp.default(mul_1943);  mul_1943 = None
    mul_1944: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(exp_70, 0.3989422804014327);  exp_70 = None
    mul_1945: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_37, mul_1944);  view_37 = mul_1944 = None
    add_664: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(mul_1941, mul_1945);  mul_1941 = mul_1945 = None
    mul_1946: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_1106, add_664);  view_1106 = add_664 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_1107: "f32[6272, 3072]" = torch.ops.aten.reshape.default(mul_1946, [6272, 3072]);  mul_1946 = None
    mm_230: "f32[6272, 768]" = torch.ops.aten.mm.default(view_1107, permute_854);  permute_854 = None
    permute_855: "f32[3072, 6272]" = torch.ops.aten.permute.default(view_1107, [1, 0])
    mm_231: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_855, view_36);  permute_855 = view_36 = None
    permute_856: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_231, [1, 0]);  mm_231 = None
    sum_707: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_1107, [0], True);  view_1107 = None
    view_1108: "f32[3072]" = torch.ops.aten.reshape.default(sum_707, [3072]);  sum_707 = None
    permute_857: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_856, [1, 0]);  permute_856 = None
    view_1109: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_230, [8, 784, 768]);  mm_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_1948: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_1109, primals_147);  primals_147 = None
    mul_1949: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1948, 768)
    sum_708: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1948, [2], True)
    mul_1950: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1948, mul_73);  mul_1948 = None
    sum_709: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1950, [2], True);  mul_1950 = None
    mul_1951: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_73, sum_709);  sum_709 = None
    sub_452: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(mul_1949, sum_708);  mul_1949 = sum_708 = None
    sub_453: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(sub_452, mul_1951);  sub_452 = mul_1951 = None
    mul_1952: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(div_325, sub_453);  div_325 = sub_453 = None
    mul_1953: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_1109, mul_73);  mul_73 = None
    sum_710: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1953, [0, 1]);  mul_1953 = None
    sum_711: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_1109, [0, 1]);  view_1109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    add_665: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_662, mul_1952);  add_662 = mul_1952 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_1954: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_665, primals_7);  primals_7 = None
    mul_1955: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_665, permute_18);  permute_18 = None
    sum_712: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1955, [0, 1], True);  mul_1955 = None
    view_1110: "f32[768]" = torch.ops.aten.reshape.default(sum_712, [768]);  sum_712 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    permute_858: "f32[8, 768, 784]" = torch.ops.aten.permute.default(mul_1954, [0, 2, 1]);  mul_1954 = None
    view_1111: "f32[8, 768, 28, 28]" = torch.ops.aten.reshape.default(permute_858, [8, 768, 28, 28]);  permute_858 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    sum_713: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_1111, [0, 2, 3])
    convolution_backward_44 = torch.ops.aten.convolution_backward.default(view_1111, add_51, primals_145, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False]);  view_1111 = add_51 = primals_145 = None
    getitem_428: "f32[8, 768, 28, 28]" = convolution_backward_44[0]
    getitem_429: "f32[768, 1, 3, 3]" = convolution_backward_44[1];  convolution_backward_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    sum_714: "f32[768]" = torch.ops.aten.sum.dim_IntList(getitem_428, [0, 2, 3])
    sub_454: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(mul_64, unsqueeze_383);  mul_64 = unsqueeze_383 = None
    mul_1956: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_428, sub_454)
    sum_715: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1956, [0, 2, 3]);  mul_1956 = None
    mul_1957: "f32[768]" = torch.ops.aten.mul.Tensor(sum_714, 0.00015943877551020407)
    unsqueeze_384: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1957, 0);  mul_1957 = None
    unsqueeze_385: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_384, 2);  unsqueeze_384 = None
    unsqueeze_386: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_385, 3);  unsqueeze_385 = None
    mul_1958: "f32[768]" = torch.ops.aten.mul.Tensor(sum_715, 0.00015943877551020407)
    mul_1959: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
    mul_1960: "f32[768]" = torch.ops.aten.mul.Tensor(mul_1958, mul_1959);  mul_1958 = mul_1959 = None
    unsqueeze_387: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1960, 0);  mul_1960 = None
    unsqueeze_388: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_387, 2);  unsqueeze_387 = None
    unsqueeze_389: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_388, 3);  unsqueeze_388 = None
    mul_1961: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_13, primals_143);  primals_143 = None
    unsqueeze_390: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1961, 0);  mul_1961 = None
    unsqueeze_391: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_390, 2);  unsqueeze_390 = None
    unsqueeze_392: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_391, 3);  unsqueeze_391 = None
    mul_1962: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_454, unsqueeze_389);  sub_454 = unsqueeze_389 = None
    sub_456: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(getitem_428, mul_1962);  getitem_428 = mul_1962 = None
    sub_457: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(sub_456, unsqueeze_386);  sub_456 = unsqueeze_386 = None
    mul_1963: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_457, unsqueeze_392);  sub_457 = unsqueeze_392 = None
    mul_1964: "f32[768]" = torch.ops.aten.mul.Tensor(sum_715, squeeze_13);  sum_715 = squeeze_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_1966: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(add_46, 0.5);  add_46 = None
    mul_1967: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_6, convolution_6)
    mul_1968: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_1967, -0.5);  mul_1967 = None
    exp_71: "f32[8, 768, 28, 28]" = torch.ops.aten.exp.default(mul_1968);  mul_1968 = None
    mul_1969: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(exp_71, 0.3989422804014327);  exp_71 = None
    mul_1970: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_6, mul_1969);  convolution_6 = mul_1969 = None
    add_667: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(mul_1966, mul_1970);  mul_1966 = mul_1970 = None
    mul_1971: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_1963, add_667);  mul_1963 = add_667 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    sum_716: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1971, [0, 2, 3])
    convolution_backward_45 = torch.ops.aten.convolution_backward.default(mul_1971, view_34, primals_141, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False]);  mul_1971 = view_34 = primals_141 = None
    getitem_431: "f32[8, 768, 28, 28]" = convolution_backward_45[0]
    getitem_432: "f32[768, 1, 3, 3]" = convolution_backward_45[1];  convolution_backward_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    view_1112: "f32[8, 768, 784]" = torch.ops.aten.reshape.default(getitem_431, [8, 768, 784]);  getitem_431 = None
    permute_859: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_1112, [0, 2, 1]);  view_1112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    clone_449: "f32[8, 784, 768]" = torch.ops.aten.clone.default(permute_859, memory_format = torch.contiguous_format);  permute_859 = None
    mul_1973: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(clone_449, primals_139);  primals_139 = None
    mul_1974: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1973, 768)
    sum_717: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1973, [2], True)
    mul_1975: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1973, mul_60);  mul_1973 = None
    sum_718: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1975, [2], True);  mul_1975 = None
    mul_1976: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_60, sum_718);  sum_718 = None
    sub_459: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(mul_1974, sum_717);  mul_1974 = sum_717 = None
    sub_460: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(sub_459, mul_1976);  sub_459 = mul_1976 = None
    mul_1977: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(div_326, sub_460);  div_326 = sub_460 = None
    mul_1978: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(clone_449, mul_60);  mul_60 = None
    sum_719: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1978, [0, 1]);  mul_1978 = None
    sum_720: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_449, [0, 1]);  clone_449 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    add_668: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_665, mul_1977);  add_665 = mul_1977 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_1979: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_668, primals_5);  primals_5 = None
    mul_1980: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_668, add_42);  add_42 = None
    sum_721: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1980, [0, 1], True);  mul_1980 = None
    view_1113: "f32[768]" = torch.ops.aten.reshape.default(sum_721, [768]);  sum_721 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    sum_722: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1979, [0, 1], True)
    view_1114: "f32[768]" = torch.ops.aten.reshape.default(sum_722, [768]);  sum_722 = None
    view_1115: "f32[6272, 768]" = torch.ops.aten.reshape.default(mul_1979, [6272, 768]);  mul_1979 = None
    permute_860: "f32[768, 6272]" = torch.ops.aten.permute.default(view_1115, [1, 0])
    mm_232: "f32[768, 768]" = torch.ops.aten.mm.default(permute_860, view_32);  permute_860 = view_32 = None
    permute_861: "f32[768, 768]" = torch.ops.aten.permute.default(mm_232, [1, 0]);  mm_232 = None
    mm_233: "f32[6272, 768]" = torch.ops.aten.mm.default(view_1115, permute_862);  view_1115 = permute_862 = None
    view_1116: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_233, [8, 784, 768]);  mm_233 = None
    permute_863: "f32[768, 768]" = torch.ops.aten.permute.default(permute_861, [1, 0]);  permute_861 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    view_1117: "f32[8, 784, 16, 48]" = torch.ops.aten.reshape.default(view_1116, [8, 784, 16, 48]);  view_1116 = None
    permute_864: "f32[8, 16, 48, 784]" = torch.ops.aten.permute.default(view_1117, [0, 2, 3, 1]);  view_1117 = None
    clone_451: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(permute_864, memory_format = torch.contiguous_format);  permute_864 = None
    view_1118: "f32[128, 48, 784]" = torch.ops.aten.reshape.default(clone_451, [128, 48, 784]);  clone_451 = None
    bmm_136: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(permute_865, view_1118);  permute_865 = None
    bmm_137: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(view_1118, permute_866);  view_1118 = permute_866 = None
    view_1119: "f32[8, 16, 48, 784]" = torch.ops.aten.reshape.default(bmm_136, [8, 16, 48, 784]);  bmm_136 = None
    view_1120: "f32[8, 16, 48, 48]" = torch.ops.aten.reshape.default(bmm_137, [8, 16, 48, 48]);  bmm_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    mul_1981: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_1120, alias_142);  view_1120 = None
    sum_723: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_1981, [-1], True)
    mul_1982: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(alias_142, sum_723);  alias_142 = sum_723 = None
    sub_461: "f32[8, 16, 48, 48]" = torch.ops.aten.sub.Tensor(mul_1981, mul_1982);  mul_1981 = mul_1982 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    mul_1983: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(sub_461, view_27);  view_27 = None
    mul_1984: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(sub_461, primals_6);  sub_461 = primals_6 = None
    sum_724: "f32[1, 16, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1983, [0, 2, 3], True);  mul_1983 = None
    view_1121: "f32[16, 1, 1]" = torch.ops.aten.reshape.default(sum_724, [16, 1, 1]);  sum_724 = None
    view_1122: "f32[128, 48, 48]" = torch.ops.aten.reshape.default(mul_1984, [128, 48, 48]);  mul_1984 = None
    bmm_138: "f32[128, 784, 48]" = torch.ops.aten.bmm.default(permute_867, view_1122);  permute_867 = None
    bmm_139: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_1122, permute_868);  view_1122 = permute_868 = None
    view_1123: "f32[8, 16, 784, 48]" = torch.ops.aten.reshape.default(bmm_138, [8, 16, 784, 48]);  bmm_138 = None
    view_1124: "f32[8, 16, 48, 784]" = torch.ops.aten.reshape.default(bmm_139, [8, 16, 48, 784]);  bmm_139 = None
    permute_869: "f32[8, 16, 48, 784]" = torch.ops.aten.permute.default(view_1123, [0, 1, 3, 2]);  view_1123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    div_328: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_10, expand_7);  div_10 = None
    neg_44: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(permute_869)
    mul_1985: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_44, div_328);  neg_44 = div_328 = None
    div_329: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(permute_869, expand_7);  permute_869 = expand_7 = None
    sum_725: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_1985, [3], True);  mul_1985 = None
    ge_44: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(pow_9, 1e-12)
    where_88: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_44, sum_725, full_default_20);  ge_44 = sum_725 = None
    div_330: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_20, pow_9);  getitem_20 = None
    eq_44: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(pow_9, 0);  pow_9 = None
    where_89: "f32[8, 16, 48, 784]" = torch.ops.aten.where.self(eq_44, full_default_20, div_330);  eq_44 = div_330 = None
    clone_452: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(where_89, memory_format = torch.contiguous_format);  where_89 = None
    mul_1986: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_88, clone_452);  where_88 = clone_452 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    add_669: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_329, mul_1986);  div_329 = mul_1986 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    div_332: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_9, expand_6);  div_9 = None
    neg_45: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(view_1124)
    mul_1987: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_45, div_332);  neg_45 = div_332 = None
    div_333: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(view_1124, expand_6);  view_1124 = expand_6 = None
    sum_726: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_1987, [3], True);  mul_1987 = None
    ge_45: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(pow_7, 1e-12)
    where_90: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_45, sum_726, full_default_20);  ge_45 = sum_726 = None
    div_334: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_19, pow_7);  getitem_19 = None
    eq_45: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(pow_7, 0);  pow_7 = None
    where_91: "f32[8, 16, 48, 784]" = torch.ops.aten.where.self(eq_45, full_default_20, div_334);  eq_45 = div_334 = None
    clone_453: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(where_91, memory_format = torch.contiguous_format);  where_91 = None
    mul_1988: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_90, clone_453);  where_90 = clone_453 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    add_670: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_333, mul_1988);  div_333 = mul_1988 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    cat_30: "f32[24, 16, 48, 784]" = torch.ops.aten.cat.default([add_670, add_669, view_1119]);  add_670 = add_669 = view_1119 = None
    view_1125: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.reshape.default(cat_30, [3, 8, 16, 48, 784]);  cat_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    permute_870: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.permute.default(view_1125, [1, 4, 0, 2, 3]);  view_1125 = None
    clone_454: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.clone.default(permute_870, memory_format = torch.contiguous_format);  permute_870 = None
    view_1126: "f32[8, 784, 2304]" = torch.ops.aten.reshape.default(clone_454, [8, 784, 2304]);  clone_454 = None
    view_1127: "f32[6272, 2304]" = torch.ops.aten.reshape.default(view_1126, [6272, 2304]);  view_1126 = None
    mm_234: "f32[6272, 768]" = torch.ops.aten.mm.default(view_1127, permute_871);  permute_871 = None
    permute_872: "f32[2304, 6272]" = torch.ops.aten.permute.default(view_1127, [1, 0])
    mm_235: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_872, view_22);  permute_872 = view_22 = None
    permute_873: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_235, [1, 0]);  mm_235 = None
    sum_727: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_1127, [0], True);  view_1127 = None
    view_1128: "f32[2304]" = torch.ops.aten.reshape.default(sum_727, [2304]);  sum_727 = None
    permute_874: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_873, [1, 0]);  permute_873 = None
    view_1129: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_234, [8, 784, 768]);  mm_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_1990: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_1129, primals_133);  primals_133 = None
    mul_1991: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1990, 768)
    sum_728: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1990, [2], True)
    mul_1992: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_1990, mul_56);  mul_1990 = None
    sum_729: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_1992, [2], True);  mul_1992 = None
    mul_1993: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_56, sum_729);  sum_729 = None
    sub_463: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(mul_1991, sum_728);  mul_1991 = sum_728 = None
    sub_464: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(sub_463, mul_1993);  sub_463 = mul_1993 = None
    mul_1994: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(div_335, sub_464);  div_335 = sub_464 = None
    mul_1995: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_1129, mul_56);  mul_56 = None
    sum_730: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1995, [0, 1]);  mul_1995 = None
    sum_731: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_1129, [0, 1]);  view_1129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    add_671: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_668, mul_1994);  add_668 = mul_1994 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_1996: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_671, primals_4);  primals_4 = None
    mul_1997: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_671, view_21);  view_21 = None
    sum_732: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1997, [0, 1], True);  mul_1997 = None
    view_1130: "f32[768]" = torch.ops.aten.reshape.default(sum_732, [768]);  sum_732 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_1131: "f32[6272, 768]" = torch.ops.aten.reshape.default(mul_1996, [6272, 768]);  mul_1996 = None
    mm_236: "f32[6272, 3072]" = torch.ops.aten.mm.default(view_1131, permute_875);  permute_875 = None
    permute_876: "f32[768, 6272]" = torch.ops.aten.permute.default(view_1131, [1, 0])
    mm_237: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_876, view_20);  permute_876 = view_20 = None
    permute_877: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_237, [1, 0]);  mm_237 = None
    sum_733: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1131, [0], True);  view_1131 = None
    view_1132: "f32[768]" = torch.ops.aten.reshape.default(sum_733, [768]);  sum_733 = None
    permute_878: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_877, [1, 0]);  permute_877 = None
    view_1133: "f32[8, 784, 3072]" = torch.ops.aten.reshape.default(mm_236, [8, 784, 3072]);  mm_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_1999: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(add_38, 0.5);  add_38 = None
    mul_2000: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_19, view_19)
    mul_2001: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(mul_2000, -0.5);  mul_2000 = None
    exp_72: "f32[8, 784, 3072]" = torch.ops.aten.exp.default(mul_2001);  mul_2001 = None
    mul_2002: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(exp_72, 0.3989422804014327);  exp_72 = None
    mul_2003: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_19, mul_2002);  view_19 = mul_2002 = None
    add_673: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(mul_1999, mul_2003);  mul_1999 = mul_2003 = None
    mul_2004: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_1133, add_673);  view_1133 = add_673 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_1134: "f32[6272, 3072]" = torch.ops.aten.reshape.default(mul_2004, [6272, 3072]);  mul_2004 = None
    mm_238: "f32[6272, 768]" = torch.ops.aten.mm.default(view_1134, permute_879);  permute_879 = None
    permute_880: "f32[3072, 6272]" = torch.ops.aten.permute.default(view_1134, [1, 0])
    mm_239: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_880, view_18);  permute_880 = view_18 = None
    permute_881: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_239, [1, 0]);  mm_239 = None
    sum_734: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_1134, [0], True);  view_1134 = None
    view_1135: "f32[3072]" = torch.ops.aten.reshape.default(sum_734, [3072]);  sum_734 = None
    permute_882: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_881, [1, 0]);  permute_881 = None
    view_1136: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_238, [8, 784, 768]);  mm_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_2006: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_1136, primals_127);  primals_127 = None
    mul_2007: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_2006, 768)
    sum_735: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_2006, [2], True)
    mul_2008: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_2006, mul_50);  mul_2006 = None
    sum_736: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_2008, [2], True);  mul_2008 = None
    mul_2009: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_50, sum_736);  sum_736 = None
    sub_466: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(mul_2007, sum_735);  mul_2007 = sum_735 = None
    sub_467: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(sub_466, mul_2009);  sub_466 = mul_2009 = None
    mul_2010: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(div_336, sub_467);  div_336 = sub_467 = None
    mul_2011: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_1136, mul_50);  mul_50 = None
    sum_737: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_2011, [0, 1]);  mul_2011 = None
    sum_738: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_1136, [0, 1]);  view_1136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    add_674: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_671, mul_2010);  add_671 = mul_2010 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_2012: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_674, primals_3);  primals_3 = None
    mul_2013: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_674, permute_9);  permute_9 = None
    sum_739: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_2013, [0, 1], True);  mul_2013 = None
    view_1137: "f32[768]" = torch.ops.aten.reshape.default(sum_739, [768]);  sum_739 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    permute_883: "f32[8, 768, 784]" = torch.ops.aten.permute.default(mul_2012, [0, 2, 1]);  mul_2012 = None
    view_1138: "f32[8, 768, 28, 28]" = torch.ops.aten.reshape.default(permute_883, [8, 768, 28, 28]);  permute_883 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    sum_740: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_1138, [0, 2, 3])
    convolution_backward_46 = torch.ops.aten.convolution_backward.default(view_1138, add_34, primals_125, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False]);  view_1138 = add_34 = primals_125 = None
    getitem_434: "f32[8, 768, 28, 28]" = convolution_backward_46[0]
    getitem_435: "f32[768, 1, 3, 3]" = convolution_backward_46[1];  convolution_backward_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    sum_741: "f32[768]" = torch.ops.aten.sum.dim_IntList(getitem_434, [0, 2, 3])
    sub_468: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(mul_41, unsqueeze_395);  mul_41 = unsqueeze_395 = None
    mul_2014: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_434, sub_468)
    sum_742: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_2014, [0, 2, 3]);  mul_2014 = None
    mul_2015: "f32[768]" = torch.ops.aten.mul.Tensor(sum_741, 0.00015943877551020407)
    unsqueeze_396: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_2015, 0);  mul_2015 = None
    unsqueeze_397: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_396, 2);  unsqueeze_396 = None
    unsqueeze_398: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_397, 3);  unsqueeze_397 = None
    mul_2016: "f32[768]" = torch.ops.aten.mul.Tensor(sum_742, 0.00015943877551020407)
    mul_2017: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_10, squeeze_10)
    mul_2018: "f32[768]" = torch.ops.aten.mul.Tensor(mul_2016, mul_2017);  mul_2016 = mul_2017 = None
    unsqueeze_399: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_2018, 0);  mul_2018 = None
    unsqueeze_400: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_399, 2);  unsqueeze_399 = None
    unsqueeze_401: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_400, 3);  unsqueeze_400 = None
    mul_2019: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_10, primals_123);  primals_123 = None
    unsqueeze_402: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_2019, 0);  mul_2019 = None
    unsqueeze_403: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_402, 2);  unsqueeze_402 = None
    unsqueeze_404: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_403, 3);  unsqueeze_403 = None
    mul_2020: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_468, unsqueeze_401);  sub_468 = unsqueeze_401 = None
    sub_470: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(getitem_434, mul_2020);  getitem_434 = mul_2020 = None
    sub_471: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(sub_470, unsqueeze_398);  sub_470 = unsqueeze_398 = None
    mul_2021: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_471, unsqueeze_404);  sub_471 = unsqueeze_404 = None
    mul_2022: "f32[768]" = torch.ops.aten.mul.Tensor(sum_742, squeeze_10);  sum_742 = squeeze_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_2024: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(add_29, 0.5);  add_29 = None
    mul_2025: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_4, convolution_4)
    mul_2026: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_2025, -0.5);  mul_2025 = None
    exp_73: "f32[8, 768, 28, 28]" = torch.ops.aten.exp.default(mul_2026);  mul_2026 = None
    mul_2027: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(exp_73, 0.3989422804014327);  exp_73 = None
    mul_2028: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_4, mul_2027);  convolution_4 = mul_2027 = None
    add_676: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(mul_2024, mul_2028);  mul_2024 = mul_2028 = None
    mul_2029: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_2021, add_676);  mul_2021 = add_676 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    sum_743: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_2029, [0, 2, 3])
    convolution_backward_47 = torch.ops.aten.convolution_backward.default(mul_2029, view_16, primals_121, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False]);  mul_2029 = view_16 = primals_121 = None
    getitem_437: "f32[8, 768, 28, 28]" = convolution_backward_47[0]
    getitem_438: "f32[768, 1, 3, 3]" = convolution_backward_47[1];  convolution_backward_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    view_1139: "f32[8, 768, 784]" = torch.ops.aten.reshape.default(getitem_437, [8, 768, 784]);  getitem_437 = None
    permute_884: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_1139, [0, 2, 1]);  view_1139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    clone_457: "f32[8, 784, 768]" = torch.ops.aten.clone.default(permute_884, memory_format = torch.contiguous_format);  permute_884 = None
    mul_2031: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(clone_457, primals_119);  primals_119 = None
    mul_2032: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_2031, 768)
    sum_744: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_2031, [2], True)
    mul_2033: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_2031, mul_37);  mul_2031 = None
    sum_745: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_2033, [2], True);  mul_2033 = None
    mul_2034: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_37, sum_745);  sum_745 = None
    sub_473: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(mul_2032, sum_744);  mul_2032 = sum_744 = None
    sub_474: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(sub_473, mul_2034);  sub_473 = mul_2034 = None
    mul_2035: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(div_337, sub_474);  div_337 = sub_474 = None
    mul_2036: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(clone_457, mul_37);  mul_37 = None
    sum_746: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_2036, [0, 1]);  mul_2036 = None
    sum_747: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_457, [0, 1]);  clone_457 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    add_677: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_674, mul_2035);  add_674 = mul_2035 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_2037: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_677, primals_1);  primals_1 = None
    mul_2038: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_677, add_25);  add_25 = None
    sum_748: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_2038, [0, 1], True);  mul_2038 = None
    view_1140: "f32[768]" = torch.ops.aten.reshape.default(sum_748, [768]);  sum_748 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    sum_749: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_2037, [0, 1], True)
    view_1141: "f32[768]" = torch.ops.aten.reshape.default(sum_749, [768]);  sum_749 = None
    view_1142: "f32[6272, 768]" = torch.ops.aten.reshape.default(mul_2037, [6272, 768]);  mul_2037 = None
    permute_885: "f32[768, 6272]" = torch.ops.aten.permute.default(view_1142, [1, 0])
    mm_240: "f32[768, 768]" = torch.ops.aten.mm.default(permute_885, view_14);  permute_885 = view_14 = None
    permute_886: "f32[768, 768]" = torch.ops.aten.permute.default(mm_240, [1, 0]);  mm_240 = None
    mm_241: "f32[6272, 768]" = torch.ops.aten.mm.default(view_1142, permute_887);  view_1142 = permute_887 = None
    view_1143: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_241, [8, 784, 768]);  mm_241 = None
    permute_888: "f32[768, 768]" = torch.ops.aten.permute.default(permute_886, [1, 0]);  permute_886 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    view_1144: "f32[8, 784, 16, 48]" = torch.ops.aten.reshape.default(view_1143, [8, 784, 16, 48]);  view_1143 = None
    permute_889: "f32[8, 16, 48, 784]" = torch.ops.aten.permute.default(view_1144, [0, 2, 3, 1]);  view_1144 = None
    clone_459: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(permute_889, memory_format = torch.contiguous_format);  permute_889 = None
    view_1145: "f32[128, 48, 784]" = torch.ops.aten.reshape.default(clone_459, [128, 48, 784]);  clone_459 = None
    bmm_140: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(permute_890, view_1145);  permute_890 = None
    bmm_141: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(view_1145, permute_891);  view_1145 = permute_891 = None
    view_1146: "f32[8, 16, 48, 784]" = torch.ops.aten.reshape.default(bmm_140, [8, 16, 48, 784]);  bmm_140 = None
    view_1147: "f32[8, 16, 48, 48]" = torch.ops.aten.reshape.default(bmm_141, [8, 16, 48, 48]);  bmm_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    mul_2039: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_1147, alias_145);  view_1147 = None
    sum_750: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_2039, [-1], True)
    mul_2040: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(alias_145, sum_750);  alias_145 = sum_750 = None
    sub_475: "f32[8, 16, 48, 48]" = torch.ops.aten.sub.Tensor(mul_2039, mul_2040);  mul_2039 = mul_2040 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    mul_2041: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(sub_475, view_9);  view_9 = None
    mul_2042: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(sub_475, primals_2);  sub_475 = primals_2 = None
    sum_751: "f32[1, 16, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_2041, [0, 2, 3], True);  mul_2041 = None
    view_1148: "f32[16, 1, 1]" = torch.ops.aten.reshape.default(sum_751, [16, 1, 1]);  sum_751 = None
    view_1149: "f32[128, 48, 48]" = torch.ops.aten.reshape.default(mul_2042, [128, 48, 48]);  mul_2042 = None
    bmm_142: "f32[128, 784, 48]" = torch.ops.aten.bmm.default(permute_892, view_1149);  permute_892 = None
    bmm_143: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_1149, permute_893);  view_1149 = permute_893 = None
    view_1150: "f32[8, 16, 784, 48]" = torch.ops.aten.reshape.default(bmm_142, [8, 16, 784, 48]);  bmm_142 = None
    view_1151: "f32[8, 16, 48, 784]" = torch.ops.aten.reshape.default(bmm_143, [8, 16, 48, 784]);  bmm_143 = None
    permute_894: "f32[8, 16, 48, 784]" = torch.ops.aten.permute.default(view_1150, [0, 1, 3, 2]);  view_1150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    div_339: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_7, expand_1);  div_7 = None
    neg_46: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(permute_894)
    mul_2043: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_46, div_339);  neg_46 = div_339 = None
    div_340: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(permute_894, expand_1);  permute_894 = expand_1 = None
    sum_752: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_2043, [3], True);  mul_2043 = None
    ge_46: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(pow_5, 1e-12)
    where_92: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_46, sum_752, full_default_20);  ge_46 = sum_752 = None
    div_341: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_9, pow_5);  getitem_9 = None
    eq_46: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(pow_5, 0);  pow_5 = None
    where_93: "f32[8, 16, 48, 784]" = torch.ops.aten.where.self(eq_46, full_default_20, div_341);  eq_46 = div_341 = None
    clone_460: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(where_93, memory_format = torch.contiguous_format);  where_93 = None
    mul_2044: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_92, clone_460);  where_92 = clone_460 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    add_678: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_340, mul_2044);  div_340 = mul_2044 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    div_343: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_6, expand);  div_6 = None
    neg_47: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(view_1151)
    mul_2045: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_47, div_343);  neg_47 = div_343 = None
    div_344: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(view_1151, expand);  view_1151 = expand = None
    sum_753: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_2045, [3], True);  mul_2045 = None
    ge_47: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(pow_3, 1e-12)
    where_94: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_47, sum_753, full_default_20);  ge_47 = sum_753 = None
    div_345: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_8, pow_3);  getitem_8 = None
    eq_47: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(pow_3, 0);  pow_3 = None
    where_95: "f32[8, 16, 48, 784]" = torch.ops.aten.where.self(eq_47, full_default_20, div_345);  eq_47 = full_default_20 = div_345 = None
    clone_461: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(where_95, memory_format = torch.contiguous_format);  where_95 = None
    mul_2046: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_94, clone_461);  where_94 = clone_461 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    add_679: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_344, mul_2046);  div_344 = mul_2046 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    cat_31: "f32[24, 16, 48, 784]" = torch.ops.aten.cat.default([add_679, add_678, view_1146]);  add_679 = add_678 = view_1146 = None
    view_1152: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.reshape.default(cat_31, [3, 8, 16, 48, 784]);  cat_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    permute_895: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.permute.default(view_1152, [1, 4, 0, 2, 3]);  view_1152 = None
    clone_462: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.clone.default(permute_895, memory_format = torch.contiguous_format);  permute_895 = None
    view_1153: "f32[8, 784, 2304]" = torch.ops.aten.reshape.default(clone_462, [8, 784, 2304]);  clone_462 = None
    view_1154: "f32[6272, 2304]" = torch.ops.aten.reshape.default(view_1153, [6272, 2304]);  view_1153 = None
    mm_242: "f32[6272, 768]" = torch.ops.aten.mm.default(view_1154, permute_896);  permute_896 = None
    permute_897: "f32[2304, 6272]" = torch.ops.aten.permute.default(view_1154, [1, 0])
    mm_243: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_897, view_4);  permute_897 = view_4 = None
    permute_898: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_243, [1, 0]);  mm_243 = None
    sum_754: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_1154, [0], True);  view_1154 = None
    view_1155: "f32[2304]" = torch.ops.aten.reshape.default(sum_754, [2304]);  sum_754 = None
    permute_899: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_898, [1, 0]);  permute_898 = None
    view_1156: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_242, [8, 784, 768]);  mm_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_2048: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_1156, primals_113);  primals_113 = None
    mul_2049: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_2048, 768)
    sum_755: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_2048, [2], True)
    mul_2050: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_2048, mul_33);  mul_2048 = None
    sum_756: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_2050, [2], True);  mul_2050 = None
    mul_2051: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_33, sum_756);  sum_756 = None
    sub_477: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(mul_2049, sum_755);  mul_2049 = sum_755 = None
    sub_478: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(sub_477, mul_2051);  sub_477 = mul_2051 = None
    mul_2052: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(div_346, sub_478);  div_346 = sub_478 = None
    mul_2053: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(view_1156, mul_33);  mul_33 = None
    sum_757: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_2053, [0, 1]);  mul_2053 = None
    sum_758: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_1156, [0, 1]);  view_1156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    add_680: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_677, mul_2052);  add_677 = mul_2052 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:437, code: pos_encoding = self.pos_embed(B, Hp, Wp).reshape(B, -1, x.shape[1]).permute(0, 2, 1)
    permute_900: "f32[8, 768, 784]" = torch.ops.aten.permute.default(add_680, [0, 2, 1]);  add_680 = None
    view_1157: "f32[8, 768, 28, 28]" = torch.ops.aten.reshape.default(permute_900, [8, 768, 28, 28]);  permute_900 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:63, code: return pos.repeat(B, 1, 1, 1)  # (B, C, H, W)
    view_1158: "f32[8, 1, 768, 28, 28]" = torch.ops.aten.reshape.default(view_1157, [8, 1, 768, 28, 28])
    sum_759: "f32[1, 768, 28, 28]" = torch.ops.aten.sum.dim_IntList(view_1158, [0]);  view_1158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:62, code: pos = self.token_projection(pos)
    sum_760: "f32[768]" = torch.ops.aten.sum.dim_IntList(sum_759, [0, 2, 3])
    convolution_backward_48 = torch.ops.aten.convolution_backward.default(sum_759, permute_1, primals_111, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [False, True, False]);  sum_759 = permute_1 = primals_111 = None
    getitem_441: "f32[768, 64, 1, 1]" = convolution_backward_48[1];  convolution_backward_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:107, code: x = self.proj(x)
    sum_761: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_1157, [0, 2, 3])
    sub_479: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_407);  convolution_2 = unsqueeze_407 = None
    mul_2054: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(view_1157, sub_479)
    sum_762: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_2054, [0, 2, 3]);  mul_2054 = None
    mul_2055: "f32[768]" = torch.ops.aten.mul.Tensor(sum_761, 0.00015943877551020407)
    unsqueeze_408: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_2055, 0);  mul_2055 = None
    unsqueeze_409: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_408, 2);  unsqueeze_408 = None
    unsqueeze_410: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_409, 3);  unsqueeze_409 = None
    mul_2056: "f32[768]" = torch.ops.aten.mul.Tensor(sum_762, 0.00015943877551020407)
    mul_2057: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
    mul_2058: "f32[768]" = torch.ops.aten.mul.Tensor(mul_2056, mul_2057);  mul_2056 = mul_2057 = None
    unsqueeze_411: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_2058, 0);  mul_2058 = None
    unsqueeze_412: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_411, 2);  unsqueeze_411 = None
    unsqueeze_413: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_412, 3);  unsqueeze_412 = None
    mul_2059: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_7, primals_109);  primals_109 = None
    unsqueeze_414: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_2059, 0);  mul_2059 = None
    unsqueeze_415: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_414, 2);  unsqueeze_414 = None
    unsqueeze_416: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_415, 3);  unsqueeze_415 = None
    mul_2060: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_479, unsqueeze_413);  sub_479 = unsqueeze_413 = None
    sub_481: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(view_1157, mul_2060);  view_1157 = mul_2060 = None
    sub_482: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(sub_481, unsqueeze_410);  sub_481 = unsqueeze_410 = None
    mul_2061: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_482, unsqueeze_416);  sub_482 = unsqueeze_416 = None
    mul_2062: "f32[768]" = torch.ops.aten.mul.Tensor(sum_762, squeeze_7);  sum_762 = squeeze_7 = None
    convolution_backward_49 = torch.ops.aten.convolution_backward.default(mul_2061, mul_19, primals_108, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2061 = mul_19 = primals_108 = None
    getitem_443: "f32[8, 384, 56, 56]" = convolution_backward_49[0]
    getitem_444: "f32[768, 384, 3, 3]" = convolution_backward_49[1];  convolution_backward_49 = None
    mul_2069: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_443, add_682);  getitem_443 = add_682 = None
    sum_763: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_2069, [0, 2, 3])
    sub_483: "f32[8, 384, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_419);  convolution_1 = unsqueeze_419 = None
    mul_2070: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(mul_2069, sub_483)
    sum_764: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_2070, [0, 2, 3]);  mul_2070 = None
    mul_2071: "f32[384]" = torch.ops.aten.mul.Tensor(sum_763, 3.985969387755102e-05)
    unsqueeze_420: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_2071, 0);  mul_2071 = None
    unsqueeze_421: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_420, 2);  unsqueeze_420 = None
    unsqueeze_422: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_421, 3);  unsqueeze_421 = None
    mul_2072: "f32[384]" = torch.ops.aten.mul.Tensor(sum_764, 3.985969387755102e-05)
    mul_2073: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
    mul_2074: "f32[384]" = torch.ops.aten.mul.Tensor(mul_2072, mul_2073);  mul_2072 = mul_2073 = None
    unsqueeze_423: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_2074, 0);  mul_2074 = None
    unsqueeze_424: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_423, 2);  unsqueeze_423 = None
    unsqueeze_425: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_424, 3);  unsqueeze_424 = None
    mul_2075: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_4, primals_106);  primals_106 = None
    unsqueeze_426: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_2075, 0);  mul_2075 = None
    unsqueeze_427: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_426, 2);  unsqueeze_426 = None
    unsqueeze_428: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_427, 3);  unsqueeze_427 = None
    mul_2076: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(sub_483, unsqueeze_425);  sub_483 = unsqueeze_425 = None
    sub_485: "f32[8, 384, 56, 56]" = torch.ops.aten.sub.Tensor(mul_2069, mul_2076);  mul_2069 = mul_2076 = None
    sub_486: "f32[8, 384, 56, 56]" = torch.ops.aten.sub.Tensor(sub_485, unsqueeze_422);  sub_485 = unsqueeze_422 = None
    mul_2077: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(sub_486, unsqueeze_428);  sub_486 = unsqueeze_428 = None
    mul_2078: "f32[384]" = torch.ops.aten.mul.Tensor(sum_764, squeeze_4);  sum_764 = squeeze_4 = None
    convolution_backward_50 = torch.ops.aten.convolution_backward.default(mul_2077, mul_9, primals_105, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_2077 = mul_9 = primals_105 = None
    getitem_446: "f32[8, 192, 112, 112]" = convolution_backward_50[0]
    getitem_447: "f32[384, 192, 3, 3]" = convolution_backward_50[1];  convolution_backward_50 = None
    mul_2085: "f32[8, 192, 112, 112]" = torch.ops.aten.mul.Tensor(getitem_446, add_684);  getitem_446 = add_684 = None
    sum_765: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_2085, [0, 2, 3])
    sub_487: "f32[8, 192, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_431);  convolution = unsqueeze_431 = None
    mul_2086: "f32[8, 192, 112, 112]" = torch.ops.aten.mul.Tensor(mul_2085, sub_487)
    sum_766: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_2086, [0, 2, 3]);  mul_2086 = None
    mul_2087: "f32[192]" = torch.ops.aten.mul.Tensor(sum_765, 9.964923469387754e-06)
    unsqueeze_432: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_2087, 0);  mul_2087 = None
    unsqueeze_433: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_432, 2);  unsqueeze_432 = None
    unsqueeze_434: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_433, 3);  unsqueeze_433 = None
    mul_2088: "f32[192]" = torch.ops.aten.mul.Tensor(sum_766, 9.964923469387754e-06)
    mul_2089: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
    mul_2090: "f32[192]" = torch.ops.aten.mul.Tensor(mul_2088, mul_2089);  mul_2088 = mul_2089 = None
    unsqueeze_435: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_2090, 0);  mul_2090 = None
    unsqueeze_436: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_435, 2);  unsqueeze_435 = None
    unsqueeze_437: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_436, 3);  unsqueeze_436 = None
    mul_2091: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_1, primals_103);  primals_103 = None
    unsqueeze_438: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_2091, 0);  mul_2091 = None
    unsqueeze_439: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_438, 2);  unsqueeze_438 = None
    unsqueeze_440: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_439, 3);  unsqueeze_439 = None
    mul_2092: "f32[8, 192, 112, 112]" = torch.ops.aten.mul.Tensor(sub_487, unsqueeze_437);  sub_487 = unsqueeze_437 = None
    sub_489: "f32[8, 192, 112, 112]" = torch.ops.aten.sub.Tensor(mul_2085, mul_2092);  mul_2085 = mul_2092 = None
    sub_490: "f32[8, 192, 112, 112]" = torch.ops.aten.sub.Tensor(sub_489, unsqueeze_434);  sub_489 = unsqueeze_434 = None
    mul_2093: "f32[8, 192, 112, 112]" = torch.ops.aten.mul.Tensor(sub_490, unsqueeze_440);  sub_490 = unsqueeze_440 = None
    mul_2094: "f32[192]" = torch.ops.aten.mul.Tensor(sum_766, squeeze_1);  sum_766 = squeeze_1 = None
    convolution_backward_51 = torch.ops.aten.convolution_backward.default(mul_2093, primals_710, primals_102, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_2093 = primals_710 = primals_102 = None
    getitem_450: "f32[192, 3, 3, 3]" = convolution_backward_51[1];  convolution_backward_51 = None
    return [view_1140, view_1148, view_1137, view_1130, view_1113, view_1121, view_1110, view_1103, view_1086, view_1094, view_1083, view_1076, view_1059, view_1067, view_1056, view_1049, view_1032, view_1040, view_1029, view_1022, view_1005, view_1013, view_1002, view_995, view_978, view_986, view_975, view_968, view_951, view_959, view_948, view_941, view_924, view_932, view_921, view_914, view_897, view_905, view_894, view_887, view_870, view_878, view_867, view_860, view_843, view_851, view_840, view_833, view_816, view_824, view_813, view_806, view_789, view_797, view_786, view_779, view_762, view_770, view_759, view_752, view_735, view_743, view_732, view_725, view_708, view_716, view_705, view_698, view_681, view_689, view_678, view_671, view_654, view_662, view_651, view_644, view_627, view_635, view_624, view_617, view_600, view_608, view_597, view_590, view_573, view_581, view_570, view_563, view_546, view_554, view_543, view_536, view_519, view_527, view_516, view_509, sum_110, view_494, view_487, view_472, view_465, getitem_450, mul_2094, sum_765, getitem_447, mul_2078, sum_763, getitem_444, mul_2062, sum_761, getitem_441, sum_760, sum_757, sum_758, permute_899, view_1155, permute_888, view_1141, sum_746, sum_747, getitem_438, sum_743, mul_2022, sum_741, getitem_435, sum_740, sum_737, sum_738, permute_882, view_1135, permute_878, view_1132, sum_730, sum_731, permute_874, view_1128, permute_863, view_1114, sum_719, sum_720, getitem_432, sum_716, mul_1964, sum_714, getitem_429, sum_713, sum_710, sum_711, permute_857, view_1108, permute_853, view_1105, sum_703, sum_704, permute_849, view_1101, permute_838, view_1087, sum_692, sum_693, getitem_426, sum_689, mul_1906, sum_687, getitem_423, sum_686, sum_683, sum_684, permute_832, view_1081, permute_828, view_1078, sum_676, sum_677, permute_824, view_1074, permute_813, view_1060, sum_665, sum_666, getitem_420, sum_662, mul_1848, sum_660, getitem_417, sum_659, sum_656, sum_657, permute_807, view_1054, permute_803, view_1051, sum_649, sum_650, permute_799, view_1047, permute_788, view_1033, sum_638, sum_639, getitem_414, sum_635, mul_1790, sum_633, getitem_411, sum_632, sum_629, sum_630, permute_782, view_1027, permute_778, view_1024, sum_622, sum_623, permute_774, view_1020, permute_763, view_1006, sum_611, sum_612, getitem_408, sum_608, mul_1732, sum_606, getitem_405, sum_605, sum_602, sum_603, permute_757, view_1000, permute_753, view_997, sum_595, sum_596, permute_749, view_993, permute_738, view_979, sum_584, sum_585, getitem_402, sum_581, mul_1674, sum_579, getitem_399, sum_578, sum_575, sum_576, permute_732, view_973, permute_728, view_970, sum_568, sum_569, permute_724, view_966, permute_713, view_952, sum_557, sum_558, getitem_396, sum_554, mul_1616, sum_552, getitem_393, sum_551, sum_548, sum_549, permute_707, view_946, permute_703, view_943, sum_541, sum_542, permute_699, view_939, permute_688, view_925, sum_530, sum_531, getitem_390, sum_527, mul_1558, sum_525, getitem_387, sum_524, sum_521, sum_522, permute_682, view_919, permute_678, view_916, sum_514, sum_515, permute_674, view_912, permute_663, view_898, sum_503, sum_504, getitem_384, sum_500, mul_1500, sum_498, getitem_381, sum_497, sum_494, sum_495, permute_657, view_892, permute_653, view_889, sum_487, sum_488, permute_649, view_885, permute_638, view_871, sum_476, sum_477, getitem_378, sum_473, mul_1442, sum_471, getitem_375, sum_470, sum_467, sum_468, permute_632, view_865, permute_628, view_862, sum_460, sum_461, permute_624, view_858, permute_613, view_844, sum_449, sum_450, getitem_372, sum_446, mul_1384, sum_444, getitem_369, sum_443, sum_440, sum_441, permute_607, view_838, permute_603, view_835, sum_433, sum_434, permute_599, view_831, permute_588, view_817, sum_422, sum_423, getitem_366, sum_419, mul_1326, sum_417, getitem_363, sum_416, sum_413, sum_414, permute_582, view_811, permute_578, view_808, sum_406, sum_407, permute_574, view_804, permute_563, view_790, sum_395, sum_396, getitem_360, sum_392, mul_1268, sum_390, getitem_357, sum_389, sum_386, sum_387, permute_557, view_784, permute_553, view_781, sum_379, sum_380, permute_549, view_777, permute_538, view_763, sum_368, sum_369, getitem_354, sum_365, mul_1210, sum_363, getitem_351, sum_362, sum_359, sum_360, permute_532, view_757, permute_528, view_754, sum_352, sum_353, permute_524, view_750, permute_513, view_736, sum_341, sum_342, getitem_348, sum_338, mul_1152, sum_336, getitem_345, sum_335, sum_332, sum_333, permute_507, view_730, permute_503, view_727, sum_325, sum_326, permute_499, view_723, permute_488, view_709, sum_314, sum_315, getitem_342, sum_311, mul_1094, sum_309, getitem_339, sum_308, sum_305, sum_306, permute_482, view_703, permute_478, view_700, sum_298, sum_299, permute_474, view_696, permute_463, view_682, sum_287, sum_288, getitem_336, sum_284, mul_1036, sum_282, getitem_333, sum_281, sum_278, sum_279, permute_457, view_676, permute_453, view_673, sum_271, sum_272, permute_449, view_669, permute_438, view_655, sum_260, sum_261, getitem_330, sum_257, mul_978, sum_255, getitem_327, sum_254, sum_251, sum_252, permute_432, view_649, permute_428, view_646, sum_244, sum_245, permute_424, view_642, permute_413, view_628, sum_233, sum_234, getitem_324, sum_230, mul_920, sum_228, getitem_321, sum_227, sum_224, sum_225, permute_407, view_622, permute_403, view_619, sum_217, sum_218, permute_399, view_615, permute_388, view_601, sum_206, sum_207, getitem_318, sum_203, mul_862, sum_201, getitem_315, sum_200, sum_197, sum_198, permute_382, view_595, permute_378, view_592, sum_190, sum_191, permute_374, view_588, permute_363, view_574, sum_179, sum_180, getitem_312, sum_176, mul_804, sum_174, getitem_309, sum_173, sum_170, sum_171, permute_357, view_568, permute_353, view_565, sum_163, sum_164, permute_349, view_561, permute_338, view_547, sum_152, sum_153, getitem_306, sum_149, mul_746, sum_147, getitem_303, sum_146, sum_143, sum_144, permute_332, view_541, permute_328, view_538, sum_136, sum_137, permute_324, view_534, permute_313, view_520, sum_125, sum_126, getitem_300, sum_122, mul_688, sum_120, getitem_297, sum_119, sum_116, sum_117, permute_307, view_514, permute_303, view_511, sum_108, sum_109, permute_299, view_508, permute_294, view_505, permute_289, view_501, permute_283, view_496, sum_99, sum_100, permute_279, view_491, permute_275, view_489, sum_92, sum_93, permute_271, view_486, permute_266, view_483, permute_261, view_479, permute_255, view_474, sum_83, sum_84, permute_251, view_469, permute_247, view_467, sum_76, sum_77, permute_243, view_464, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
    