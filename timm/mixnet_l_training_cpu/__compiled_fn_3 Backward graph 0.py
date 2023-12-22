from __future__ import annotations



def forward(self, primals_1: "f32[32]", primals_3: "f32[32]", primals_5: "f32[32]", primals_7: "f32[192]", primals_9: "f32[192]", primals_11: "f32[40]", primals_13: "f32[120]", primals_15: "f32[120]", primals_17: "f32[40]", primals_19: "f32[240]", primals_21: "f32[240]", primals_23: "f32[56]", primals_25: "f32[336]", primals_27: "f32[336]", primals_29: "f32[56]", primals_31: "f32[336]", primals_33: "f32[336]", primals_35: "f32[56]", primals_37: "f32[336]", primals_39: "f32[336]", primals_41: "f32[56]", primals_43: "f32[336]", primals_45: "f32[336]", primals_47: "f32[104]", primals_49: "f32[624]", primals_51: "f32[624]", primals_53: "f32[104]", primals_55: "f32[624]", primals_57: "f32[624]", primals_59: "f32[104]", primals_61: "f32[624]", primals_63: "f32[624]", primals_65: "f32[104]", primals_67: "f32[624]", primals_69: "f32[624]", primals_71: "f32[160]", primals_73: "f32[480]", primals_75: "f32[480]", primals_77: "f32[160]", primals_79: "f32[480]", primals_81: "f32[480]", primals_83: "f32[160]", primals_85: "f32[480]", primals_87: "f32[480]", primals_89: "f32[160]", primals_91: "f32[960]", primals_93: "f32[960]", primals_95: "f32[264]", primals_97: "f32[1584]", primals_99: "f32[1584]", primals_101: "f32[264]", primals_103: "f32[1584]", primals_105: "f32[1584]", primals_107: "f32[264]", primals_109: "f32[1584]", primals_111: "f32[1584]", primals_113: "f32[264]", primals_115: "f32[1536]", primals_117: "f32[32, 3, 3, 3]", primals_118: "f32[32, 1, 3, 3]", primals_119: "f32[32, 32, 1, 1]", primals_120: "f32[96, 16, 1, 1]", primals_121: "f32[96, 16, 1, 1]", primals_122: "f32[64, 1, 3, 3]", primals_123: "f32[64, 1, 5, 5]", primals_124: "f32[64, 1, 7, 7]", primals_125: "f32[20, 96, 1, 1]", primals_126: "f32[20, 96, 1, 1]", primals_127: "f32[60, 20, 1, 1]", primals_128: "f32[60, 20, 1, 1]", primals_129: "f32[120, 1, 3, 3]", primals_130: "f32[20, 60, 1, 1]", primals_131: "f32[20, 60, 1, 1]", primals_132: "f32[240, 40, 1, 1]", primals_133: "f32[60, 1, 3, 3]", primals_134: "f32[60, 1, 5, 5]", primals_135: "f32[60, 1, 7, 7]", primals_136: "f32[60, 1, 9, 9]", primals_137: "f32[20, 240, 1, 1]", primals_139: "f32[240, 20, 1, 1]", primals_141: "f32[56, 240, 1, 1]", primals_142: "f32[168, 28, 1, 1]", primals_143: "f32[168, 28, 1, 1]", primals_144: "f32[168, 1, 3, 3]", primals_145: "f32[168, 1, 5, 5]", primals_146: "f32[28, 336, 1, 1]", primals_148: "f32[336, 28, 1, 1]", primals_150: "f32[28, 168, 1, 1]", primals_151: "f32[28, 168, 1, 1]", primals_152: "f32[168, 28, 1, 1]", primals_153: "f32[168, 28, 1, 1]", primals_154: "f32[168, 1, 3, 3]", primals_155: "f32[168, 1, 5, 5]", primals_156: "f32[28, 336, 1, 1]", primals_158: "f32[336, 28, 1, 1]", primals_160: "f32[28, 168, 1, 1]", primals_161: "f32[28, 168, 1, 1]", primals_162: "f32[168, 28, 1, 1]", primals_163: "f32[168, 28, 1, 1]", primals_164: "f32[168, 1, 3, 3]", primals_165: "f32[168, 1, 5, 5]", primals_166: "f32[28, 336, 1, 1]", primals_168: "f32[336, 28, 1, 1]", primals_170: "f32[28, 168, 1, 1]", primals_171: "f32[28, 168, 1, 1]", primals_172: "f32[336, 56, 1, 1]", primals_173: "f32[112, 1, 3, 3]", primals_174: "f32[112, 1, 5, 5]", primals_175: "f32[112, 1, 7, 7]", primals_176: "f32[14, 336, 1, 1]", primals_178: "f32[336, 14, 1, 1]", primals_180: "f32[104, 336, 1, 1]", primals_181: "f32[312, 52, 1, 1]", primals_182: "f32[312, 52, 1, 1]", primals_183: "f32[156, 1, 3, 3]", primals_184: "f32[156, 1, 5, 5]", primals_185: "f32[156, 1, 7, 7]", primals_186: "f32[156, 1, 9, 9]", primals_187: "f32[26, 624, 1, 1]", primals_189: "f32[624, 26, 1, 1]", primals_191: "f32[52, 312, 1, 1]", primals_192: "f32[52, 312, 1, 1]", primals_193: "f32[312, 52, 1, 1]", primals_194: "f32[312, 52, 1, 1]", primals_195: "f32[156, 1, 3, 3]", primals_196: "f32[156, 1, 5, 5]", primals_197: "f32[156, 1, 7, 7]", primals_198: "f32[156, 1, 9, 9]", primals_199: "f32[26, 624, 1, 1]", primals_201: "f32[624, 26, 1, 1]", primals_203: "f32[52, 312, 1, 1]", primals_204: "f32[52, 312, 1, 1]", primals_205: "f32[312, 52, 1, 1]", primals_206: "f32[312, 52, 1, 1]", primals_207: "f32[156, 1, 3, 3]", primals_208: "f32[156, 1, 5, 5]", primals_209: "f32[156, 1, 7, 7]", primals_210: "f32[156, 1, 9, 9]", primals_211: "f32[26, 624, 1, 1]", primals_213: "f32[624, 26, 1, 1]", primals_215: "f32[52, 312, 1, 1]", primals_216: "f32[52, 312, 1, 1]", primals_217: "f32[624, 104, 1, 1]", primals_218: "f32[624, 1, 3, 3]", primals_219: "f32[52, 624, 1, 1]", primals_221: "f32[624, 52, 1, 1]", primals_223: "f32[160, 624, 1, 1]", primals_224: "f32[240, 80, 1, 1]", primals_225: "f32[240, 80, 1, 1]", primals_226: "f32[120, 1, 3, 3]", primals_227: "f32[120, 1, 5, 5]", primals_228: "f32[120, 1, 7, 7]", primals_229: "f32[120, 1, 9, 9]", primals_230: "f32[80, 480, 1, 1]", primals_232: "f32[480, 80, 1, 1]", primals_234: "f32[80, 240, 1, 1]", primals_235: "f32[80, 240, 1, 1]", primals_236: "f32[240, 80, 1, 1]", primals_237: "f32[240, 80, 1, 1]", primals_238: "f32[120, 1, 3, 3]", primals_239: "f32[120, 1, 5, 5]", primals_240: "f32[120, 1, 7, 7]", primals_241: "f32[120, 1, 9, 9]", primals_242: "f32[80, 480, 1, 1]", primals_244: "f32[480, 80, 1, 1]", primals_246: "f32[80, 240, 1, 1]", primals_247: "f32[80, 240, 1, 1]", primals_248: "f32[240, 80, 1, 1]", primals_249: "f32[240, 80, 1, 1]", primals_250: "f32[120, 1, 3, 3]", primals_251: "f32[120, 1, 5, 5]", primals_252: "f32[120, 1, 7, 7]", primals_253: "f32[120, 1, 9, 9]", primals_254: "f32[80, 480, 1, 1]", primals_256: "f32[480, 80, 1, 1]", primals_258: "f32[80, 240, 1, 1]", primals_259: "f32[80, 240, 1, 1]", primals_260: "f32[960, 160, 1, 1]", primals_261: "f32[240, 1, 3, 3]", primals_262: "f32[240, 1, 5, 5]", primals_263: "f32[240, 1, 7, 7]", primals_264: "f32[240, 1, 9, 9]", primals_265: "f32[80, 960, 1, 1]", primals_267: "f32[960, 80, 1, 1]", primals_269: "f32[264, 960, 1, 1]", primals_270: "f32[1584, 264, 1, 1]", primals_271: "f32[396, 1, 3, 3]", primals_272: "f32[396, 1, 5, 5]", primals_273: "f32[396, 1, 7, 7]", primals_274: "f32[396, 1, 9, 9]", primals_275: "f32[132, 1584, 1, 1]", primals_277: "f32[1584, 132, 1, 1]", primals_279: "f32[132, 792, 1, 1]", primals_280: "f32[132, 792, 1, 1]", primals_281: "f32[1584, 264, 1, 1]", primals_282: "f32[396, 1, 3, 3]", primals_283: "f32[396, 1, 5, 5]", primals_284: "f32[396, 1, 7, 7]", primals_285: "f32[396, 1, 9, 9]", primals_286: "f32[132, 1584, 1, 1]", primals_288: "f32[1584, 132, 1, 1]", primals_290: "f32[132, 792, 1, 1]", primals_291: "f32[132, 792, 1, 1]", primals_292: "f32[1584, 264, 1, 1]", primals_293: "f32[396, 1, 3, 3]", primals_294: "f32[396, 1, 5, 5]", primals_295: "f32[396, 1, 7, 7]", primals_296: "f32[396, 1, 9, 9]", primals_297: "f32[132, 1584, 1, 1]", primals_299: "f32[1584, 132, 1, 1]", primals_301: "f32[132, 792, 1, 1]", primals_302: "f32[132, 792, 1, 1]", primals_303: "f32[1536, 264, 1, 1]", primals_480: "f32[8, 3, 224, 224]", convolution: "f32[8, 32, 112, 112]", squeeze_1: "f32[32]", relu: "f32[8, 32, 112, 112]", convolution_1: "f32[8, 32, 112, 112]", squeeze_4: "f32[32]", relu_1: "f32[8, 32, 112, 112]", convolution_2: "f32[8, 32, 112, 112]", squeeze_7: "f32[32]", getitem_6: "f32[8, 16, 112, 112]", getitem_7: "f32[8, 16, 112, 112]", cat: "f32[8, 192, 112, 112]", squeeze_10: "f32[192]", getitem_13: "f32[8, 64, 112, 112]", getitem_17: "f32[8, 64, 112, 112]", getitem_21: "f32[8, 64, 112, 112]", cat_1: "f32[8, 192, 56, 56]", squeeze_13: "f32[192]", getitem_26: "f32[8, 96, 56, 56]", getitem_29: "f32[8, 96, 56, 56]", cat_2: "f32[8, 40, 56, 56]", squeeze_16: "f32[40]", getitem_32: "f32[8, 20, 56, 56]", getitem_33: "f32[8, 20, 56, 56]", cat_3: "f32[8, 120, 56, 56]", squeeze_19: "f32[120]", relu_4: "f32[8, 120, 56, 56]", convolution_12: "f32[8, 120, 56, 56]", squeeze_22: "f32[120]", getitem_40: "f32[8, 60, 56, 56]", getitem_43: "f32[8, 60, 56, 56]", cat_4: "f32[8, 40, 56, 56]", squeeze_25: "f32[40]", add_46: "f32[8, 40, 56, 56]", convolution_15: "f32[8, 240, 56, 56]", squeeze_28: "f32[240]", getitem_52: "f32[8, 60, 56, 56]", getitem_57: "f32[8, 60, 56, 56]", getitem_62: "f32[8, 60, 56, 56]", getitem_67: "f32[8, 60, 56, 56]", cat_5: "f32[8, 240, 28, 28]", squeeze_31: "f32[240]", add_56: "f32[8, 240, 28, 28]", mean: "f32[8, 240, 1, 1]", convolution_20: "f32[8, 20, 1, 1]", mul_79: "f32[8, 20, 1, 1]", convolution_21: "f32[8, 240, 1, 1]", mul_80: "f32[8, 240, 28, 28]", convolution_22: "f32[8, 56, 28, 28]", squeeze_34: "f32[56]", getitem_72: "f32[8, 28, 28, 28]", getitem_73: "f32[8, 28, 28, 28]", cat_6: "f32[8, 336, 28, 28]", squeeze_37: "f32[336]", getitem_78: "f32[8, 168, 28, 28]", getitem_81: "f32[8, 168, 28, 28]", cat_7: "f32[8, 336, 28, 28]", squeeze_40: "f32[336]", add_71: "f32[8, 336, 28, 28]", mean_1: "f32[8, 336, 1, 1]", convolution_27: "f32[8, 28, 1, 1]", mul_104: "f32[8, 28, 1, 1]", convolution_28: "f32[8, 336, 1, 1]", getitem_84: "f32[8, 168, 28, 28]", getitem_85: "f32[8, 168, 28, 28]", cat_8: "f32[8, 56, 28, 28]", squeeze_43: "f32[56]", getitem_88: "f32[8, 28, 28, 28]", getitem_89: "f32[8, 28, 28, 28]", cat_9: "f32[8, 336, 28, 28]", squeeze_46: "f32[336]", getitem_94: "f32[8, 168, 28, 28]", getitem_97: "f32[8, 168, 28, 28]", cat_10: "f32[8, 336, 28, 28]", squeeze_49: "f32[336]", add_87: "f32[8, 336, 28, 28]", mean_2: "f32[8, 336, 1, 1]", convolution_35: "f32[8, 28, 1, 1]", mul_129: "f32[8, 28, 1, 1]", convolution_36: "f32[8, 336, 1, 1]", getitem_100: "f32[8, 168, 28, 28]", getitem_101: "f32[8, 168, 28, 28]", cat_11: "f32[8, 56, 28, 28]", squeeze_52: "f32[56]", getitem_104: "f32[8, 28, 28, 28]", getitem_105: "f32[8, 28, 28, 28]", cat_12: "f32[8, 336, 28, 28]", squeeze_55: "f32[336]", getitem_110: "f32[8, 168, 28, 28]", getitem_113: "f32[8, 168, 28, 28]", cat_13: "f32[8, 336, 28, 28]", squeeze_58: "f32[336]", add_103: "f32[8, 336, 28, 28]", mean_3: "f32[8, 336, 1, 1]", convolution_43: "f32[8, 28, 1, 1]", mul_154: "f32[8, 28, 1, 1]", convolution_44: "f32[8, 336, 1, 1]", getitem_116: "f32[8, 168, 28, 28]", getitem_117: "f32[8, 168, 28, 28]", cat_14: "f32[8, 56, 28, 28]", squeeze_61: "f32[56]", add_109: "f32[8, 56, 28, 28]", convolution_47: "f32[8, 336, 28, 28]", squeeze_64: "f32[336]", getitem_125: "f32[8, 112, 28, 28]", getitem_129: "f32[8, 112, 28, 28]", getitem_133: "f32[8, 112, 28, 28]", cat_15: "f32[8, 336, 14, 14]", squeeze_67: "f32[336]", add_119: "f32[8, 336, 14, 14]", mean_4: "f32[8, 336, 1, 1]", convolution_51: "f32[8, 14, 1, 1]", mul_179: "f32[8, 14, 1, 1]", convolution_52: "f32[8, 336, 1, 1]", mul_180: "f32[8, 336, 14, 14]", convolution_53: "f32[8, 104, 14, 14]", squeeze_70: "f32[104]", getitem_138: "f32[8, 52, 14, 14]", getitem_139: "f32[8, 52, 14, 14]", cat_16: "f32[8, 624, 14, 14]", squeeze_73: "f32[624]", getitem_146: "f32[8, 156, 14, 14]", getitem_151: "f32[8, 156, 14, 14]", getitem_156: "f32[8, 156, 14, 14]", getitem_161: "f32[8, 156, 14, 14]", cat_17: "f32[8, 624, 14, 14]", squeeze_76: "f32[624]", add_134: "f32[8, 624, 14, 14]", mean_5: "f32[8, 624, 1, 1]", convolution_60: "f32[8, 26, 1, 1]", mul_204: "f32[8, 26, 1, 1]", convolution_61: "f32[8, 624, 1, 1]", getitem_164: "f32[8, 312, 14, 14]", getitem_165: "f32[8, 312, 14, 14]", cat_18: "f32[8, 104, 14, 14]", squeeze_79: "f32[104]", getitem_168: "f32[8, 52, 14, 14]", getitem_169: "f32[8, 52, 14, 14]", cat_19: "f32[8, 624, 14, 14]", squeeze_82: "f32[624]", getitem_176: "f32[8, 156, 14, 14]", getitem_181: "f32[8, 156, 14, 14]", getitem_186: "f32[8, 156, 14, 14]", getitem_191: "f32[8, 156, 14, 14]", cat_20: "f32[8, 624, 14, 14]", squeeze_85: "f32[624]", add_150: "f32[8, 624, 14, 14]", mean_6: "f32[8, 624, 1, 1]", convolution_70: "f32[8, 26, 1, 1]", mul_229: "f32[8, 26, 1, 1]", convolution_71: "f32[8, 624, 1, 1]", getitem_194: "f32[8, 312, 14, 14]", getitem_195: "f32[8, 312, 14, 14]", cat_21: "f32[8, 104, 14, 14]", squeeze_88: "f32[104]", getitem_198: "f32[8, 52, 14, 14]", getitem_199: "f32[8, 52, 14, 14]", cat_22: "f32[8, 624, 14, 14]", squeeze_91: "f32[624]", getitem_206: "f32[8, 156, 14, 14]", getitem_211: "f32[8, 156, 14, 14]", getitem_216: "f32[8, 156, 14, 14]", getitem_221: "f32[8, 156, 14, 14]", cat_23: "f32[8, 624, 14, 14]", squeeze_94: "f32[624]", add_166: "f32[8, 624, 14, 14]", mean_7: "f32[8, 624, 1, 1]", convolution_80: "f32[8, 26, 1, 1]", mul_254: "f32[8, 26, 1, 1]", convolution_81: "f32[8, 624, 1, 1]", getitem_224: "f32[8, 312, 14, 14]", getitem_225: "f32[8, 312, 14, 14]", cat_24: "f32[8, 104, 14, 14]", squeeze_97: "f32[104]", add_172: "f32[8, 104, 14, 14]", convolution_84: "f32[8, 624, 14, 14]", squeeze_100: "f32[624]", mul_270: "f32[8, 624, 14, 14]", convolution_85: "f32[8, 624, 14, 14]", squeeze_103: "f32[624]", add_182: "f32[8, 624, 14, 14]", mean_8: "f32[8, 624, 1, 1]", convolution_86: "f32[8, 52, 1, 1]", mul_279: "f32[8, 52, 1, 1]", convolution_87: "f32[8, 624, 1, 1]", mul_280: "f32[8, 624, 14, 14]", convolution_88: "f32[8, 160, 14, 14]", squeeze_106: "f32[160]", getitem_234: "f32[8, 80, 14, 14]", getitem_235: "f32[8, 80, 14, 14]", cat_25: "f32[8, 480, 14, 14]", squeeze_109: "f32[480]", getitem_242: "f32[8, 120, 14, 14]", getitem_247: "f32[8, 120, 14, 14]", getitem_252: "f32[8, 120, 14, 14]", getitem_257: "f32[8, 120, 14, 14]", cat_26: "f32[8, 480, 14, 14]", squeeze_112: "f32[480]", add_197: "f32[8, 480, 14, 14]", mean_9: "f32[8, 480, 1, 1]", convolution_95: "f32[8, 80, 1, 1]", mul_304: "f32[8, 80, 1, 1]", convolution_96: "f32[8, 480, 1, 1]", getitem_260: "f32[8, 240, 14, 14]", getitem_261: "f32[8, 240, 14, 14]", cat_27: "f32[8, 160, 14, 14]", squeeze_115: "f32[160]", getitem_264: "f32[8, 80, 14, 14]", getitem_265: "f32[8, 80, 14, 14]", cat_28: "f32[8, 480, 14, 14]", squeeze_118: "f32[480]", getitem_272: "f32[8, 120, 14, 14]", getitem_277: "f32[8, 120, 14, 14]", getitem_282: "f32[8, 120, 14, 14]", getitem_287: "f32[8, 120, 14, 14]", cat_29: "f32[8, 480, 14, 14]", squeeze_121: "f32[480]", add_213: "f32[8, 480, 14, 14]", mean_10: "f32[8, 480, 1, 1]", convolution_105: "f32[8, 80, 1, 1]", mul_329: "f32[8, 80, 1, 1]", convolution_106: "f32[8, 480, 1, 1]", getitem_290: "f32[8, 240, 14, 14]", getitem_291: "f32[8, 240, 14, 14]", cat_30: "f32[8, 160, 14, 14]", squeeze_124: "f32[160]", getitem_294: "f32[8, 80, 14, 14]", getitem_295: "f32[8, 80, 14, 14]", cat_31: "f32[8, 480, 14, 14]", squeeze_127: "f32[480]", getitem_302: "f32[8, 120, 14, 14]", getitem_307: "f32[8, 120, 14, 14]", getitem_312: "f32[8, 120, 14, 14]", getitem_317: "f32[8, 120, 14, 14]", cat_32: "f32[8, 480, 14, 14]", squeeze_130: "f32[480]", add_229: "f32[8, 480, 14, 14]", mean_11: "f32[8, 480, 1, 1]", convolution_115: "f32[8, 80, 1, 1]", mul_354: "f32[8, 80, 1, 1]", convolution_116: "f32[8, 480, 1, 1]", getitem_320: "f32[8, 240, 14, 14]", getitem_321: "f32[8, 240, 14, 14]", cat_33: "f32[8, 160, 14, 14]", squeeze_133: "f32[160]", add_235: "f32[8, 160, 14, 14]", convolution_119: "f32[8, 960, 14, 14]", squeeze_136: "f32[960]", getitem_330: "f32[8, 240, 14, 14]", getitem_335: "f32[8, 240, 14, 14]", getitem_340: "f32[8, 240, 14, 14]", getitem_345: "f32[8, 240, 14, 14]", cat_34: "f32[8, 960, 7, 7]", squeeze_139: "f32[960]", add_245: "f32[8, 960, 7, 7]", mean_12: "f32[8, 960, 1, 1]", convolution_124: "f32[8, 80, 1, 1]", mul_379: "f32[8, 80, 1, 1]", convolution_125: "f32[8, 960, 1, 1]", mul_380: "f32[8, 960, 7, 7]", convolution_126: "f32[8, 264, 7, 7]", squeeze_142: "f32[264]", add_250: "f32[8, 264, 7, 7]", convolution_127: "f32[8, 1584, 7, 7]", squeeze_145: "f32[1584]", getitem_356: "f32[8, 396, 7, 7]", getitem_361: "f32[8, 396, 7, 7]", getitem_366: "f32[8, 396, 7, 7]", getitem_371: "f32[8, 396, 7, 7]", cat_35: "f32[8, 1584, 7, 7]", squeeze_148: "f32[1584]", add_260: "f32[8, 1584, 7, 7]", mean_13: "f32[8, 1584, 1, 1]", convolution_132: "f32[8, 132, 1, 1]", mul_404: "f32[8, 132, 1, 1]", convolution_133: "f32[8, 1584, 1, 1]", getitem_374: "f32[8, 792, 7, 7]", getitem_375: "f32[8, 792, 7, 7]", cat_36: "f32[8, 264, 7, 7]", squeeze_151: "f32[264]", add_266: "f32[8, 264, 7, 7]", convolution_136: "f32[8, 1584, 7, 7]", squeeze_154: "f32[1584]", getitem_384: "f32[8, 396, 7, 7]", getitem_389: "f32[8, 396, 7, 7]", getitem_394: "f32[8, 396, 7, 7]", getitem_399: "f32[8, 396, 7, 7]", cat_37: "f32[8, 1584, 7, 7]", squeeze_157: "f32[1584]", add_276: "f32[8, 1584, 7, 7]", mean_14: "f32[8, 1584, 1, 1]", convolution_141: "f32[8, 132, 1, 1]", mul_429: "f32[8, 132, 1, 1]", convolution_142: "f32[8, 1584, 1, 1]", getitem_402: "f32[8, 792, 7, 7]", getitem_403: "f32[8, 792, 7, 7]", cat_38: "f32[8, 264, 7, 7]", squeeze_160: "f32[264]", add_282: "f32[8, 264, 7, 7]", convolution_145: "f32[8, 1584, 7, 7]", squeeze_163: "f32[1584]", getitem_412: "f32[8, 396, 7, 7]", getitem_417: "f32[8, 396, 7, 7]", getitem_422: "f32[8, 396, 7, 7]", getitem_427: "f32[8, 396, 7, 7]", cat_39: "f32[8, 1584, 7, 7]", squeeze_166: "f32[1584]", add_292: "f32[8, 1584, 7, 7]", mean_15: "f32[8, 1584, 1, 1]", convolution_150: "f32[8, 132, 1, 1]", mul_454: "f32[8, 132, 1, 1]", convolution_151: "f32[8, 1584, 1, 1]", getitem_430: "f32[8, 792, 7, 7]", getitem_431: "f32[8, 792, 7, 7]", cat_40: "f32[8, 264, 7, 7]", squeeze_169: "f32[264]", add_298: "f32[8, 264, 7, 7]", convolution_154: "f32[8, 1536, 7, 7]", squeeze_172: "f32[1536]", view: "f32[8, 1536]", permute_1: "f32[1000, 1536]", le: "b8[8, 1536, 7, 7]", unsqueeze_234: "f32[1, 1536, 1, 1]", unsqueeze_246: "f32[1, 264, 1, 1]", unsqueeze_258: "f32[1, 1584, 1, 1]", mul_508: "f32[8, 1584, 7, 7]", unsqueeze_270: "f32[1, 1584, 1, 1]", unsqueeze_282: "f32[1, 264, 1, 1]", unsqueeze_294: "f32[1, 1584, 1, 1]", mul_548: "f32[8, 1584, 7, 7]", unsqueeze_306: "f32[1, 1584, 1, 1]", unsqueeze_318: "f32[1, 264, 1, 1]", unsqueeze_330: "f32[1, 1584, 1, 1]", mul_588: "f32[8, 1584, 7, 7]", unsqueeze_342: "f32[1, 1584, 1, 1]", unsqueeze_354: "f32[1, 264, 1, 1]", unsqueeze_366: "f32[1, 960, 1, 1]", mul_628: "f32[8, 960, 14, 14]", unsqueeze_378: "f32[1, 960, 1, 1]", unsqueeze_390: "f32[1, 160, 1, 1]", unsqueeze_402: "f32[1, 480, 1, 1]", mul_668: "f32[8, 480, 14, 14]", unsqueeze_414: "f32[1, 480, 1, 1]", unsqueeze_426: "f32[1, 160, 1, 1]", unsqueeze_438: "f32[1, 480, 1, 1]", mul_708: "f32[8, 480, 14, 14]", unsqueeze_450: "f32[1, 480, 1, 1]", unsqueeze_462: "f32[1, 160, 1, 1]", unsqueeze_474: "f32[1, 480, 1, 1]", mul_748: "f32[8, 480, 14, 14]", unsqueeze_486: "f32[1, 480, 1, 1]", unsqueeze_498: "f32[1, 160, 1, 1]", unsqueeze_510: "f32[1, 624, 1, 1]", mul_788: "f32[8, 624, 14, 14]", unsqueeze_522: "f32[1, 624, 1, 1]", unsqueeze_534: "f32[1, 104, 1, 1]", unsqueeze_546: "f32[1, 624, 1, 1]", mul_828: "f32[8, 624, 14, 14]", unsqueeze_558: "f32[1, 624, 1, 1]", unsqueeze_570: "f32[1, 104, 1, 1]", unsqueeze_582: "f32[1, 624, 1, 1]", mul_868: "f32[8, 624, 14, 14]", unsqueeze_594: "f32[1, 624, 1, 1]", unsqueeze_606: "f32[1, 104, 1, 1]", unsqueeze_618: "f32[1, 624, 1, 1]", mul_908: "f32[8, 624, 14, 14]", unsqueeze_630: "f32[1, 624, 1, 1]", unsqueeze_642: "f32[1, 104, 1, 1]", unsqueeze_654: "f32[1, 336, 1, 1]", mul_948: "f32[8, 336, 28, 28]", unsqueeze_666: "f32[1, 336, 1, 1]", unsqueeze_678: "f32[1, 56, 1, 1]", unsqueeze_690: "f32[1, 336, 1, 1]", mul_988: "f32[8, 336, 28, 28]", unsqueeze_702: "f32[1, 336, 1, 1]", unsqueeze_714: "f32[1, 56, 1, 1]", unsqueeze_726: "f32[1, 336, 1, 1]", mul_1028: "f32[8, 336, 28, 28]", unsqueeze_738: "f32[1, 336, 1, 1]", unsqueeze_750: "f32[1, 56, 1, 1]", unsqueeze_762: "f32[1, 336, 1, 1]", mul_1068: "f32[8, 336, 28, 28]", unsqueeze_774: "f32[1, 336, 1, 1]", unsqueeze_786: "f32[1, 56, 1, 1]", unsqueeze_798: "f32[1, 240, 1, 1]", mul_1108: "f32[8, 240, 56, 56]", unsqueeze_810: "f32[1, 240, 1, 1]", unsqueeze_822: "f32[1, 40, 1, 1]", le_1: "b8[8, 120, 56, 56]", unsqueeze_834: "f32[1, 120, 1, 1]", unsqueeze_846: "f32[1, 120, 1, 1]", unsqueeze_858: "f32[1, 40, 1, 1]", le_3: "b8[8, 192, 56, 56]", unsqueeze_870: "f32[1, 192, 1, 1]", le_4: "b8[8, 192, 112, 112]", unsqueeze_882: "f32[1, 192, 1, 1]", unsqueeze_894: "f32[1, 32, 1, 1]", unsqueeze_906: "f32[1, 32, 1, 1]", unsqueeze_918: "f32[1, 32, 1, 1]", tangents_1: "f32[8, 1000]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_1: "f32[8, 240, 28, 28]" = torch.ops.aten.clone.default(add_56)
    sigmoid_1: "f32[8, 240, 28, 28]" = torch.ops.aten.sigmoid.default(add_56)
    mul_78: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(add_56, sigmoid_1);  add_56 = sigmoid_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_2: "f32[8, 20, 1, 1]" = torch.ops.aten.clone.default(convolution_20);  convolution_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_3: "f32[8, 240, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_21);  convolution_21 = None
    alias_6: "f32[8, 240, 1, 1]" = torch.ops.aten.alias.default(sigmoid_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_4: "f32[8, 336, 28, 28]" = torch.ops.aten.clone.default(add_71)
    sigmoid_5: "f32[8, 336, 28, 28]" = torch.ops.aten.sigmoid.default(add_71)
    mul_103: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(add_71, sigmoid_5);  add_71 = sigmoid_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_5: "f32[8, 28, 1, 1]" = torch.ops.aten.clone.default(convolution_27);  convolution_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_7: "f32[8, 336, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_28);  convolution_28 = None
    alias_7: "f32[8, 336, 1, 1]" = torch.ops.aten.alias.default(sigmoid_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_7: "f32[8, 336, 28, 28]" = torch.ops.aten.clone.default(add_87)
    sigmoid_9: "f32[8, 336, 28, 28]" = torch.ops.aten.sigmoid.default(add_87)
    mul_128: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(add_87, sigmoid_9);  add_87 = sigmoid_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_8: "f32[8, 28, 1, 1]" = torch.ops.aten.clone.default(convolution_35);  convolution_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_11: "f32[8, 336, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_36);  convolution_36 = None
    alias_8: "f32[8, 336, 1, 1]" = torch.ops.aten.alias.default(sigmoid_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_10: "f32[8, 336, 28, 28]" = torch.ops.aten.clone.default(add_103)
    sigmoid_13: "f32[8, 336, 28, 28]" = torch.ops.aten.sigmoid.default(add_103)
    mul_153: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(add_103, sigmoid_13);  add_103 = sigmoid_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_11: "f32[8, 28, 1, 1]" = torch.ops.aten.clone.default(convolution_43);  convolution_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_15: "f32[8, 336, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_44);  convolution_44 = None
    alias_9: "f32[8, 336, 1, 1]" = torch.ops.aten.alias.default(sigmoid_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_13: "f32[8, 336, 14, 14]" = torch.ops.aten.clone.default(add_119)
    sigmoid_17: "f32[8, 336, 14, 14]" = torch.ops.aten.sigmoid.default(add_119)
    mul_178: "f32[8, 336, 14, 14]" = torch.ops.aten.mul.Tensor(add_119, sigmoid_17);  add_119 = sigmoid_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_14: "f32[8, 14, 1, 1]" = torch.ops.aten.clone.default(convolution_51);  convolution_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_19: "f32[8, 336, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_52);  convolution_52 = None
    alias_10: "f32[8, 336, 1, 1]" = torch.ops.aten.alias.default(sigmoid_19)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_16: "f32[8, 624, 14, 14]" = torch.ops.aten.clone.default(add_134)
    sigmoid_21: "f32[8, 624, 14, 14]" = torch.ops.aten.sigmoid.default(add_134)
    mul_203: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(add_134, sigmoid_21);  add_134 = sigmoid_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_17: "f32[8, 26, 1, 1]" = torch.ops.aten.clone.default(convolution_60);  convolution_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_23: "f32[8, 624, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_61);  convolution_61 = None
    alias_11: "f32[8, 624, 1, 1]" = torch.ops.aten.alias.default(sigmoid_23)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_19: "f32[8, 624, 14, 14]" = torch.ops.aten.clone.default(add_150)
    sigmoid_25: "f32[8, 624, 14, 14]" = torch.ops.aten.sigmoid.default(add_150)
    mul_228: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(add_150, sigmoid_25);  add_150 = sigmoid_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_20: "f32[8, 26, 1, 1]" = torch.ops.aten.clone.default(convolution_70);  convolution_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_27: "f32[8, 624, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_71);  convolution_71 = None
    alias_12: "f32[8, 624, 1, 1]" = torch.ops.aten.alias.default(sigmoid_27)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_22: "f32[8, 624, 14, 14]" = torch.ops.aten.clone.default(add_166)
    sigmoid_29: "f32[8, 624, 14, 14]" = torch.ops.aten.sigmoid.default(add_166)
    mul_253: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(add_166, sigmoid_29);  add_166 = sigmoid_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_23: "f32[8, 26, 1, 1]" = torch.ops.aten.clone.default(convolution_80);  convolution_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_31: "f32[8, 624, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_81);  convolution_81 = None
    alias_13: "f32[8, 624, 1, 1]" = torch.ops.aten.alias.default(sigmoid_31)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_25: "f32[8, 624, 14, 14]" = torch.ops.aten.clone.default(add_182)
    sigmoid_33: "f32[8, 624, 14, 14]" = torch.ops.aten.sigmoid.default(add_182)
    mul_278: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(add_182, sigmoid_33);  add_182 = sigmoid_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_26: "f32[8, 52, 1, 1]" = torch.ops.aten.clone.default(convolution_86);  convolution_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_35: "f32[8, 624, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_87);  convolution_87 = None
    alias_14: "f32[8, 624, 1, 1]" = torch.ops.aten.alias.default(sigmoid_35)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_28: "f32[8, 480, 14, 14]" = torch.ops.aten.clone.default(add_197)
    sigmoid_37: "f32[8, 480, 14, 14]" = torch.ops.aten.sigmoid.default(add_197)
    mul_303: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_197, sigmoid_37);  add_197 = sigmoid_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_29: "f32[8, 80, 1, 1]" = torch.ops.aten.clone.default(convolution_95);  convolution_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_39: "f32[8, 480, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_96);  convolution_96 = None
    alias_15: "f32[8, 480, 1, 1]" = torch.ops.aten.alias.default(sigmoid_39)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_31: "f32[8, 480, 14, 14]" = torch.ops.aten.clone.default(add_213)
    sigmoid_41: "f32[8, 480, 14, 14]" = torch.ops.aten.sigmoid.default(add_213)
    mul_328: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_213, sigmoid_41);  add_213 = sigmoid_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_32: "f32[8, 80, 1, 1]" = torch.ops.aten.clone.default(convolution_105);  convolution_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_43: "f32[8, 480, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_106);  convolution_106 = None
    alias_16: "f32[8, 480, 1, 1]" = torch.ops.aten.alias.default(sigmoid_43)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_34: "f32[8, 480, 14, 14]" = torch.ops.aten.clone.default(add_229)
    sigmoid_45: "f32[8, 480, 14, 14]" = torch.ops.aten.sigmoid.default(add_229)
    mul_353: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_229, sigmoid_45);  add_229 = sigmoid_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_35: "f32[8, 80, 1, 1]" = torch.ops.aten.clone.default(convolution_115);  convolution_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_47: "f32[8, 480, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_116);  convolution_116 = None
    alias_17: "f32[8, 480, 1, 1]" = torch.ops.aten.alias.default(sigmoid_47)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_37: "f32[8, 960, 7, 7]" = torch.ops.aten.clone.default(add_245)
    sigmoid_49: "f32[8, 960, 7, 7]" = torch.ops.aten.sigmoid.default(add_245)
    mul_378: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(add_245, sigmoid_49);  add_245 = sigmoid_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_38: "f32[8, 80, 1, 1]" = torch.ops.aten.clone.default(convolution_124);  convolution_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_51: "f32[8, 960, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_125);  convolution_125 = None
    alias_18: "f32[8, 960, 1, 1]" = torch.ops.aten.alias.default(sigmoid_51)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_40: "f32[8, 1584, 7, 7]" = torch.ops.aten.clone.default(add_260)
    sigmoid_53: "f32[8, 1584, 7, 7]" = torch.ops.aten.sigmoid.default(add_260)
    mul_403: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(add_260, sigmoid_53);  add_260 = sigmoid_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_41: "f32[8, 132, 1, 1]" = torch.ops.aten.clone.default(convolution_132);  convolution_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_55: "f32[8, 1584, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_133);  convolution_133 = None
    alias_19: "f32[8, 1584, 1, 1]" = torch.ops.aten.alias.default(sigmoid_55)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_43: "f32[8, 1584, 7, 7]" = torch.ops.aten.clone.default(add_276)
    sigmoid_57: "f32[8, 1584, 7, 7]" = torch.ops.aten.sigmoid.default(add_276)
    mul_428: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(add_276, sigmoid_57);  add_276 = sigmoid_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_44: "f32[8, 132, 1, 1]" = torch.ops.aten.clone.default(convolution_141);  convolution_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_59: "f32[8, 1584, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_142);  convolution_142 = None
    alias_20: "f32[8, 1584, 1, 1]" = torch.ops.aten.alias.default(sigmoid_59)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_46: "f32[8, 1584, 7, 7]" = torch.ops.aten.clone.default(add_292)
    sigmoid_61: "f32[8, 1584, 7, 7]" = torch.ops.aten.sigmoid.default(add_292)
    mul_453: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(add_292, sigmoid_61);  add_292 = sigmoid_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_47: "f32[8, 132, 1, 1]" = torch.ops.aten.clone.default(convolution_150);  convolution_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_63: "f32[8, 1584, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_151);  convolution_151 = None
    alias_21: "f32[8, 1584, 1, 1]" = torch.ops.aten.alias.default(sigmoid_63)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/efficientnet.py:176, code: return x if pre_logits else self.classifier(x)
    mm: "f32[8, 1536]" = torch.ops.aten.mm.default(tangents_1, permute_1);  permute_1 = None
    permute_2: "f32[1000, 8]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
    mm_1: "f32[1000, 1536]" = torch.ops.aten.mm.default(permute_2, view);  permute_2 = view = None
    permute_3: "f32[1536, 1000]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_1: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view_1: "f32[1000]" = torch.ops.aten.view.default(sum_1, [1000]);  sum_1 = None
    permute_4: "f32[1000, 1536]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view_2: "f32[8, 1536, 1, 1]" = torch.ops.aten.view.default(mm, [8, 1536, 1, 1]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    expand: "f32[8, 1536, 7, 7]" = torch.ops.aten.expand.default(view_2, [8, 1536, 7, 7]);  view_2 = None
    div: "f32[8, 1536, 7, 7]" = torch.ops.aten.div.Scalar(expand, 49);  expand = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    where: "f32[8, 1536, 7, 7]" = torch.ops.aten.where.self(le, full_default, div);  le = div = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_2: "f32[1536]" = torch.ops.aten.sum.dim_IntList(where, [0, 2, 3])
    sub_58: "f32[8, 1536, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_154, unsqueeze_234);  convolution_154 = unsqueeze_234 = None
    mul_470: "f32[8, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(where, sub_58)
    sum_3: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_470, [0, 2, 3]);  mul_470 = None
    mul_471: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_2, 0.002551020408163265)
    unsqueeze_235: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_471, 0);  mul_471 = None
    unsqueeze_236: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_235, 2);  unsqueeze_235 = None
    unsqueeze_237: "f32[1, 1536, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_236, 3);  unsqueeze_236 = None
    mul_472: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_3, 0.002551020408163265)
    mul_473: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_172, squeeze_172)
    mul_474: "f32[1536]" = torch.ops.aten.mul.Tensor(mul_472, mul_473);  mul_472 = mul_473 = None
    unsqueeze_238: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_474, 0);  mul_474 = None
    unsqueeze_239: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_238, 2);  unsqueeze_238 = None
    unsqueeze_240: "f32[1, 1536, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_239, 3);  unsqueeze_239 = None
    mul_475: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_172, primals_115);  primals_115 = None
    unsqueeze_241: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_475, 0);  mul_475 = None
    unsqueeze_242: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_241, 2);  unsqueeze_241 = None
    unsqueeze_243: "f32[1, 1536, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_242, 3);  unsqueeze_242 = None
    mul_476: "f32[8, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(sub_58, unsqueeze_240);  sub_58 = unsqueeze_240 = None
    sub_60: "f32[8, 1536, 7, 7]" = torch.ops.aten.sub.Tensor(where, mul_476);  where = mul_476 = None
    sub_61: "f32[8, 1536, 7, 7]" = torch.ops.aten.sub.Tensor(sub_60, unsqueeze_237);  sub_60 = unsqueeze_237 = None
    mul_477: "f32[8, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(sub_61, unsqueeze_243);  sub_61 = unsqueeze_243 = None
    mul_478: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_3, squeeze_172);  sum_3 = squeeze_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/efficientnet.py:168, code: x = self.conv_head(x)
    convolution_backward = torch.ops.aten.convolution_backward.default(mul_477, add_298, primals_303, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_477 = add_298 = primals_303 = None
    getitem_436: "f32[8, 264, 7, 7]" = convolution_backward[0]
    getitem_437: "f32[1536, 264, 1, 1]" = convolution_backward[1];  convolution_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_4: "f32[264]" = torch.ops.aten.sum.dim_IntList(getitem_436, [0, 2, 3])
    sub_62: "f32[8, 264, 7, 7]" = torch.ops.aten.sub.Tensor(cat_40, unsqueeze_246);  cat_40 = unsqueeze_246 = None
    mul_479: "f32[8, 264, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_436, sub_62)
    sum_5: "f32[264]" = torch.ops.aten.sum.dim_IntList(mul_479, [0, 2, 3]);  mul_479 = None
    mul_480: "f32[264]" = torch.ops.aten.mul.Tensor(sum_4, 0.002551020408163265)
    unsqueeze_247: "f32[1, 264]" = torch.ops.aten.unsqueeze.default(mul_480, 0);  mul_480 = None
    unsqueeze_248: "f32[1, 264, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_247, 2);  unsqueeze_247 = None
    unsqueeze_249: "f32[1, 264, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_248, 3);  unsqueeze_248 = None
    mul_481: "f32[264]" = torch.ops.aten.mul.Tensor(sum_5, 0.002551020408163265)
    mul_482: "f32[264]" = torch.ops.aten.mul.Tensor(squeeze_169, squeeze_169)
    mul_483: "f32[264]" = torch.ops.aten.mul.Tensor(mul_481, mul_482);  mul_481 = mul_482 = None
    unsqueeze_250: "f32[1, 264]" = torch.ops.aten.unsqueeze.default(mul_483, 0);  mul_483 = None
    unsqueeze_251: "f32[1, 264, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_250, 2);  unsqueeze_250 = None
    unsqueeze_252: "f32[1, 264, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_251, 3);  unsqueeze_251 = None
    mul_484: "f32[264]" = torch.ops.aten.mul.Tensor(squeeze_169, primals_113);  primals_113 = None
    unsqueeze_253: "f32[1, 264]" = torch.ops.aten.unsqueeze.default(mul_484, 0);  mul_484 = None
    unsqueeze_254: "f32[1, 264, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_253, 2);  unsqueeze_253 = None
    unsqueeze_255: "f32[1, 264, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_254, 3);  unsqueeze_254 = None
    mul_485: "f32[8, 264, 7, 7]" = torch.ops.aten.mul.Tensor(sub_62, unsqueeze_252);  sub_62 = unsqueeze_252 = None
    sub_64: "f32[8, 264, 7, 7]" = torch.ops.aten.sub.Tensor(getitem_436, mul_485);  mul_485 = None
    sub_65: "f32[8, 264, 7, 7]" = torch.ops.aten.sub.Tensor(sub_64, unsqueeze_249);  sub_64 = unsqueeze_249 = None
    mul_486: "f32[8, 264, 7, 7]" = torch.ops.aten.mul.Tensor(sub_65, unsqueeze_255);  sub_65 = unsqueeze_255 = None
    mul_487: "f32[264]" = torch.ops.aten.mul.Tensor(sum_5, squeeze_169);  sum_5 = squeeze_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    slice_1: "f32[8, 132, 7, 7]" = torch.ops.aten.slice.Tensor(mul_486, 1, 0, 132)
    slice_2: "f32[8, 132, 7, 7]" = torch.ops.aten.slice.Tensor(mul_486, 1, 132, 264);  mul_486 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(slice_2, getitem_431, primals_302, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_2 = getitem_431 = primals_302 = None
    getitem_439: "f32[8, 792, 7, 7]" = convolution_backward_1[0]
    getitem_440: "f32[132, 792, 1, 1]" = convolution_backward_1[1];  convolution_backward_1 = None
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(slice_1, getitem_430, primals_301, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_1 = getitem_430 = primals_301 = None
    getitem_442: "f32[8, 792, 7, 7]" = convolution_backward_2[0]
    getitem_443: "f32[132, 792, 1, 1]" = convolution_backward_2[1];  convolution_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    cat_41: "f32[8, 1584, 7, 7]" = torch.ops.aten.cat.default([getitem_442, getitem_439], 1);  getitem_442 = getitem_439 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_488: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(cat_41, mul_453);  mul_453 = None
    mul_489: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(cat_41, sigmoid_63);  cat_41 = sigmoid_63 = None
    sum_6: "f32[8, 1584, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_488, [2, 3], True);  mul_488 = None
    alias_26: "f32[8, 1584, 1, 1]" = torch.ops.aten.alias.default(alias_21);  alias_21 = None
    sub_66: "f32[8, 1584, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_26)
    mul_490: "f32[8, 1584, 1, 1]" = torch.ops.aten.mul.Tensor(alias_26, sub_66);  alias_26 = sub_66 = None
    mul_491: "f32[8, 1584, 1, 1]" = torch.ops.aten.mul.Tensor(sum_6, mul_490);  sum_6 = mul_490 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_3 = torch.ops.aten.convolution_backward.default(mul_491, mul_454, primals_299, [1584], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_491 = mul_454 = primals_299 = None
    getitem_445: "f32[8, 132, 1, 1]" = convolution_backward_3[0]
    getitem_446: "f32[1584, 132, 1, 1]" = convolution_backward_3[1]
    getitem_447: "f32[1584]" = convolution_backward_3[2];  convolution_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_64: "f32[8, 132, 1, 1]" = torch.ops.aten.sigmoid.default(clone_47)
    full_default_1: "f32[8, 132, 1, 1]" = torch.ops.aten.full.default([8, 132, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_67: "f32[8, 132, 1, 1]" = torch.ops.aten.sub.Tensor(full_default_1, sigmoid_64)
    mul_492: "f32[8, 132, 1, 1]" = torch.ops.aten.mul.Tensor(clone_47, sub_67);  clone_47 = sub_67 = None
    add_304: "f32[8, 132, 1, 1]" = torch.ops.aten.add.Scalar(mul_492, 1);  mul_492 = None
    mul_493: "f32[8, 132, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_64, add_304);  sigmoid_64 = add_304 = None
    mul_494: "f32[8, 132, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_445, mul_493);  getitem_445 = mul_493 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_4 = torch.ops.aten.convolution_backward.default(mul_494, mean_15, primals_297, [132], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_494 = mean_15 = primals_297 = None
    getitem_448: "f32[8, 1584, 1, 1]" = convolution_backward_4[0]
    getitem_449: "f32[132, 1584, 1, 1]" = convolution_backward_4[1]
    getitem_450: "f32[132]" = convolution_backward_4[2];  convolution_backward_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_1: "f32[8, 1584, 7, 7]" = torch.ops.aten.expand.default(getitem_448, [8, 1584, 7, 7]);  getitem_448 = None
    div_1: "f32[8, 1584, 7, 7]" = torch.ops.aten.div.Scalar(expand_1, 49);  expand_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_305: "f32[8, 1584, 7, 7]" = torch.ops.aten.add.Tensor(mul_489, div_1);  mul_489 = div_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_65: "f32[8, 1584, 7, 7]" = torch.ops.aten.sigmoid.default(clone_46)
    full_default_2: "f32[8, 1584, 7, 7]" = torch.ops.aten.full.default([8, 1584, 7, 7], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_68: "f32[8, 1584, 7, 7]" = torch.ops.aten.sub.Tensor(full_default_2, sigmoid_65)
    mul_495: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(clone_46, sub_68);  clone_46 = sub_68 = None
    add_306: "f32[8, 1584, 7, 7]" = torch.ops.aten.add.Scalar(mul_495, 1);  mul_495 = None
    mul_496: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(sigmoid_65, add_306);  sigmoid_65 = add_306 = None
    mul_497: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(add_305, mul_496);  add_305 = mul_496 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_7: "f32[1584]" = torch.ops.aten.sum.dim_IntList(mul_497, [0, 2, 3])
    sub_69: "f32[8, 1584, 7, 7]" = torch.ops.aten.sub.Tensor(cat_39, unsqueeze_258);  cat_39 = unsqueeze_258 = None
    mul_498: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(mul_497, sub_69)
    sum_8: "f32[1584]" = torch.ops.aten.sum.dim_IntList(mul_498, [0, 2, 3]);  mul_498 = None
    mul_499: "f32[1584]" = torch.ops.aten.mul.Tensor(sum_7, 0.002551020408163265)
    unsqueeze_259: "f32[1, 1584]" = torch.ops.aten.unsqueeze.default(mul_499, 0);  mul_499 = None
    unsqueeze_260: "f32[1, 1584, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_259, 2);  unsqueeze_259 = None
    unsqueeze_261: "f32[1, 1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_260, 3);  unsqueeze_260 = None
    mul_500: "f32[1584]" = torch.ops.aten.mul.Tensor(sum_8, 0.002551020408163265)
    mul_501: "f32[1584]" = torch.ops.aten.mul.Tensor(squeeze_166, squeeze_166)
    mul_502: "f32[1584]" = torch.ops.aten.mul.Tensor(mul_500, mul_501);  mul_500 = mul_501 = None
    unsqueeze_262: "f32[1, 1584]" = torch.ops.aten.unsqueeze.default(mul_502, 0);  mul_502 = None
    unsqueeze_263: "f32[1, 1584, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_262, 2);  unsqueeze_262 = None
    unsqueeze_264: "f32[1, 1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_263, 3);  unsqueeze_263 = None
    mul_503: "f32[1584]" = torch.ops.aten.mul.Tensor(squeeze_166, primals_111);  primals_111 = None
    unsqueeze_265: "f32[1, 1584]" = torch.ops.aten.unsqueeze.default(mul_503, 0);  mul_503 = None
    unsqueeze_266: "f32[1, 1584, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_265, 2);  unsqueeze_265 = None
    unsqueeze_267: "f32[1, 1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_266, 3);  unsqueeze_266 = None
    mul_504: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(sub_69, unsqueeze_264);  sub_69 = unsqueeze_264 = None
    sub_71: "f32[8, 1584, 7, 7]" = torch.ops.aten.sub.Tensor(mul_497, mul_504);  mul_497 = mul_504 = None
    sub_72: "f32[8, 1584, 7, 7]" = torch.ops.aten.sub.Tensor(sub_71, unsqueeze_261);  sub_71 = unsqueeze_261 = None
    mul_505: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(sub_72, unsqueeze_267);  sub_72 = unsqueeze_267 = None
    mul_506: "f32[1584]" = torch.ops.aten.mul.Tensor(sum_8, squeeze_166);  sum_8 = squeeze_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    slice_3: "f32[8, 396, 7, 7]" = torch.ops.aten.slice.Tensor(mul_505, 1, 0, 396)
    slice_4: "f32[8, 396, 7, 7]" = torch.ops.aten.slice.Tensor(mul_505, 1, 396, 792)
    slice_5: "f32[8, 396, 7, 7]" = torch.ops.aten.slice.Tensor(mul_505, 1, 792, 1188)
    slice_6: "f32[8, 396, 7, 7]" = torch.ops.aten.slice.Tensor(mul_505, 1, 1188, 1584);  mul_505 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_backward_5 = torch.ops.aten.convolution_backward.default(slice_6, getitem_427, primals_296, [0], [1, 1], [4, 4], [1, 1], False, [0, 0], 396, [True, True, False]);  slice_6 = getitem_427 = primals_296 = None
    getitem_451: "f32[8, 396, 7, 7]" = convolution_backward_5[0]
    getitem_452: "f32[396, 1, 9, 9]" = convolution_backward_5[1];  convolution_backward_5 = None
    convolution_backward_6 = torch.ops.aten.convolution_backward.default(slice_5, getitem_422, primals_295, [0], [1, 1], [3, 3], [1, 1], False, [0, 0], 396, [True, True, False]);  slice_5 = getitem_422 = primals_295 = None
    getitem_454: "f32[8, 396, 7, 7]" = convolution_backward_6[0]
    getitem_455: "f32[396, 1, 7, 7]" = convolution_backward_6[1];  convolution_backward_6 = None
    convolution_backward_7 = torch.ops.aten.convolution_backward.default(slice_4, getitem_417, primals_294, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 396, [True, True, False]);  slice_4 = getitem_417 = primals_294 = None
    getitem_457: "f32[8, 396, 7, 7]" = convolution_backward_7[0]
    getitem_458: "f32[396, 1, 5, 5]" = convolution_backward_7[1];  convolution_backward_7 = None
    convolution_backward_8 = torch.ops.aten.convolution_backward.default(slice_3, getitem_412, primals_293, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 396, [True, True, False]);  slice_3 = getitem_412 = primals_293 = None
    getitem_460: "f32[8, 396, 7, 7]" = convolution_backward_8[0]
    getitem_461: "f32[396, 1, 3, 3]" = convolution_backward_8[1];  convolution_backward_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    cat_42: "f32[8, 1584, 7, 7]" = torch.ops.aten.cat.default([getitem_460, getitem_457, getitem_454, getitem_451], 1);  getitem_460 = getitem_457 = getitem_454 = getitem_451 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_509: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(cat_42, mul_508);  cat_42 = mul_508 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_9: "f32[1584]" = torch.ops.aten.sum.dim_IntList(mul_509, [0, 2, 3])
    sub_74: "f32[8, 1584, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_145, unsqueeze_270);  convolution_145 = unsqueeze_270 = None
    mul_510: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(mul_509, sub_74)
    sum_10: "f32[1584]" = torch.ops.aten.sum.dim_IntList(mul_510, [0, 2, 3]);  mul_510 = None
    mul_511: "f32[1584]" = torch.ops.aten.mul.Tensor(sum_9, 0.002551020408163265)
    unsqueeze_271: "f32[1, 1584]" = torch.ops.aten.unsqueeze.default(mul_511, 0);  mul_511 = None
    unsqueeze_272: "f32[1, 1584, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_271, 2);  unsqueeze_271 = None
    unsqueeze_273: "f32[1, 1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_272, 3);  unsqueeze_272 = None
    mul_512: "f32[1584]" = torch.ops.aten.mul.Tensor(sum_10, 0.002551020408163265)
    mul_513: "f32[1584]" = torch.ops.aten.mul.Tensor(squeeze_163, squeeze_163)
    mul_514: "f32[1584]" = torch.ops.aten.mul.Tensor(mul_512, mul_513);  mul_512 = mul_513 = None
    unsqueeze_274: "f32[1, 1584]" = torch.ops.aten.unsqueeze.default(mul_514, 0);  mul_514 = None
    unsqueeze_275: "f32[1, 1584, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_274, 2);  unsqueeze_274 = None
    unsqueeze_276: "f32[1, 1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_275, 3);  unsqueeze_275 = None
    mul_515: "f32[1584]" = torch.ops.aten.mul.Tensor(squeeze_163, primals_109);  primals_109 = None
    unsqueeze_277: "f32[1, 1584]" = torch.ops.aten.unsqueeze.default(mul_515, 0);  mul_515 = None
    unsqueeze_278: "f32[1, 1584, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_277, 2);  unsqueeze_277 = None
    unsqueeze_279: "f32[1, 1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_278, 3);  unsqueeze_278 = None
    mul_516: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(sub_74, unsqueeze_276);  sub_74 = unsqueeze_276 = None
    sub_76: "f32[8, 1584, 7, 7]" = torch.ops.aten.sub.Tensor(mul_509, mul_516);  mul_509 = mul_516 = None
    sub_77: "f32[8, 1584, 7, 7]" = torch.ops.aten.sub.Tensor(sub_76, unsqueeze_273);  sub_76 = unsqueeze_273 = None
    mul_517: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(sub_77, unsqueeze_279);  sub_77 = unsqueeze_279 = None
    mul_518: "f32[1584]" = torch.ops.aten.mul.Tensor(sum_10, squeeze_163);  sum_10 = squeeze_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_9 = torch.ops.aten.convolution_backward.default(mul_517, add_282, primals_292, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_517 = add_282 = primals_292 = None
    getitem_463: "f32[8, 264, 7, 7]" = convolution_backward_9[0]
    getitem_464: "f32[1584, 264, 1, 1]" = convolution_backward_9[1];  convolution_backward_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_308: "f32[8, 264, 7, 7]" = torch.ops.aten.add.Tensor(getitem_436, getitem_463);  getitem_436 = getitem_463 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_11: "f32[264]" = torch.ops.aten.sum.dim_IntList(add_308, [0, 2, 3])
    sub_78: "f32[8, 264, 7, 7]" = torch.ops.aten.sub.Tensor(cat_38, unsqueeze_282);  cat_38 = unsqueeze_282 = None
    mul_519: "f32[8, 264, 7, 7]" = torch.ops.aten.mul.Tensor(add_308, sub_78)
    sum_12: "f32[264]" = torch.ops.aten.sum.dim_IntList(mul_519, [0, 2, 3]);  mul_519 = None
    mul_520: "f32[264]" = torch.ops.aten.mul.Tensor(sum_11, 0.002551020408163265)
    unsqueeze_283: "f32[1, 264]" = torch.ops.aten.unsqueeze.default(mul_520, 0);  mul_520 = None
    unsqueeze_284: "f32[1, 264, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_283, 2);  unsqueeze_283 = None
    unsqueeze_285: "f32[1, 264, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_284, 3);  unsqueeze_284 = None
    mul_521: "f32[264]" = torch.ops.aten.mul.Tensor(sum_12, 0.002551020408163265)
    mul_522: "f32[264]" = torch.ops.aten.mul.Tensor(squeeze_160, squeeze_160)
    mul_523: "f32[264]" = torch.ops.aten.mul.Tensor(mul_521, mul_522);  mul_521 = mul_522 = None
    unsqueeze_286: "f32[1, 264]" = torch.ops.aten.unsqueeze.default(mul_523, 0);  mul_523 = None
    unsqueeze_287: "f32[1, 264, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_286, 2);  unsqueeze_286 = None
    unsqueeze_288: "f32[1, 264, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_287, 3);  unsqueeze_287 = None
    mul_524: "f32[264]" = torch.ops.aten.mul.Tensor(squeeze_160, primals_107);  primals_107 = None
    unsqueeze_289: "f32[1, 264]" = torch.ops.aten.unsqueeze.default(mul_524, 0);  mul_524 = None
    unsqueeze_290: "f32[1, 264, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_289, 2);  unsqueeze_289 = None
    unsqueeze_291: "f32[1, 264, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_290, 3);  unsqueeze_290 = None
    mul_525: "f32[8, 264, 7, 7]" = torch.ops.aten.mul.Tensor(sub_78, unsqueeze_288);  sub_78 = unsqueeze_288 = None
    sub_80: "f32[8, 264, 7, 7]" = torch.ops.aten.sub.Tensor(add_308, mul_525);  mul_525 = None
    sub_81: "f32[8, 264, 7, 7]" = torch.ops.aten.sub.Tensor(sub_80, unsqueeze_285);  sub_80 = unsqueeze_285 = None
    mul_526: "f32[8, 264, 7, 7]" = torch.ops.aten.mul.Tensor(sub_81, unsqueeze_291);  sub_81 = unsqueeze_291 = None
    mul_527: "f32[264]" = torch.ops.aten.mul.Tensor(sum_12, squeeze_160);  sum_12 = squeeze_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    slice_7: "f32[8, 132, 7, 7]" = torch.ops.aten.slice.Tensor(mul_526, 1, 0, 132)
    slice_8: "f32[8, 132, 7, 7]" = torch.ops.aten.slice.Tensor(mul_526, 1, 132, 264);  mul_526 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_backward_10 = torch.ops.aten.convolution_backward.default(slice_8, getitem_403, primals_291, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_8 = getitem_403 = primals_291 = None
    getitem_466: "f32[8, 792, 7, 7]" = convolution_backward_10[0]
    getitem_467: "f32[132, 792, 1, 1]" = convolution_backward_10[1];  convolution_backward_10 = None
    convolution_backward_11 = torch.ops.aten.convolution_backward.default(slice_7, getitem_402, primals_290, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_7 = getitem_402 = primals_290 = None
    getitem_469: "f32[8, 792, 7, 7]" = convolution_backward_11[0]
    getitem_470: "f32[132, 792, 1, 1]" = convolution_backward_11[1];  convolution_backward_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    cat_43: "f32[8, 1584, 7, 7]" = torch.ops.aten.cat.default([getitem_469, getitem_466], 1);  getitem_469 = getitem_466 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_528: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(cat_43, mul_428);  mul_428 = None
    mul_529: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(cat_43, sigmoid_59);  cat_43 = sigmoid_59 = None
    sum_13: "f32[8, 1584, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_528, [2, 3], True);  mul_528 = None
    alias_27: "f32[8, 1584, 1, 1]" = torch.ops.aten.alias.default(alias_20);  alias_20 = None
    sub_82: "f32[8, 1584, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_27)
    mul_530: "f32[8, 1584, 1, 1]" = torch.ops.aten.mul.Tensor(alias_27, sub_82);  alias_27 = sub_82 = None
    mul_531: "f32[8, 1584, 1, 1]" = torch.ops.aten.mul.Tensor(sum_13, mul_530);  sum_13 = mul_530 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_12 = torch.ops.aten.convolution_backward.default(mul_531, mul_429, primals_288, [1584], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_531 = mul_429 = primals_288 = None
    getitem_472: "f32[8, 132, 1, 1]" = convolution_backward_12[0]
    getitem_473: "f32[1584, 132, 1, 1]" = convolution_backward_12[1]
    getitem_474: "f32[1584]" = convolution_backward_12[2];  convolution_backward_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_67: "f32[8, 132, 1, 1]" = torch.ops.aten.sigmoid.default(clone_44)
    sub_83: "f32[8, 132, 1, 1]" = torch.ops.aten.sub.Tensor(full_default_1, sigmoid_67)
    mul_532: "f32[8, 132, 1, 1]" = torch.ops.aten.mul.Tensor(clone_44, sub_83);  clone_44 = sub_83 = None
    add_309: "f32[8, 132, 1, 1]" = torch.ops.aten.add.Scalar(mul_532, 1);  mul_532 = None
    mul_533: "f32[8, 132, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_67, add_309);  sigmoid_67 = add_309 = None
    mul_534: "f32[8, 132, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_472, mul_533);  getitem_472 = mul_533 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_13 = torch.ops.aten.convolution_backward.default(mul_534, mean_14, primals_286, [132], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_534 = mean_14 = primals_286 = None
    getitem_475: "f32[8, 1584, 1, 1]" = convolution_backward_13[0]
    getitem_476: "f32[132, 1584, 1, 1]" = convolution_backward_13[1]
    getitem_477: "f32[132]" = convolution_backward_13[2];  convolution_backward_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_2: "f32[8, 1584, 7, 7]" = torch.ops.aten.expand.default(getitem_475, [8, 1584, 7, 7]);  getitem_475 = None
    div_2: "f32[8, 1584, 7, 7]" = torch.ops.aten.div.Scalar(expand_2, 49);  expand_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_310: "f32[8, 1584, 7, 7]" = torch.ops.aten.add.Tensor(mul_529, div_2);  mul_529 = div_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_68: "f32[8, 1584, 7, 7]" = torch.ops.aten.sigmoid.default(clone_43)
    sub_84: "f32[8, 1584, 7, 7]" = torch.ops.aten.sub.Tensor(full_default_2, sigmoid_68)
    mul_535: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(clone_43, sub_84);  clone_43 = sub_84 = None
    add_311: "f32[8, 1584, 7, 7]" = torch.ops.aten.add.Scalar(mul_535, 1);  mul_535 = None
    mul_536: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(sigmoid_68, add_311);  sigmoid_68 = add_311 = None
    mul_537: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(add_310, mul_536);  add_310 = mul_536 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_14: "f32[1584]" = torch.ops.aten.sum.dim_IntList(mul_537, [0, 2, 3])
    sub_85: "f32[8, 1584, 7, 7]" = torch.ops.aten.sub.Tensor(cat_37, unsqueeze_294);  cat_37 = unsqueeze_294 = None
    mul_538: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(mul_537, sub_85)
    sum_15: "f32[1584]" = torch.ops.aten.sum.dim_IntList(mul_538, [0, 2, 3]);  mul_538 = None
    mul_539: "f32[1584]" = torch.ops.aten.mul.Tensor(sum_14, 0.002551020408163265)
    unsqueeze_295: "f32[1, 1584]" = torch.ops.aten.unsqueeze.default(mul_539, 0);  mul_539 = None
    unsqueeze_296: "f32[1, 1584, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_295, 2);  unsqueeze_295 = None
    unsqueeze_297: "f32[1, 1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_296, 3);  unsqueeze_296 = None
    mul_540: "f32[1584]" = torch.ops.aten.mul.Tensor(sum_15, 0.002551020408163265)
    mul_541: "f32[1584]" = torch.ops.aten.mul.Tensor(squeeze_157, squeeze_157)
    mul_542: "f32[1584]" = torch.ops.aten.mul.Tensor(mul_540, mul_541);  mul_540 = mul_541 = None
    unsqueeze_298: "f32[1, 1584]" = torch.ops.aten.unsqueeze.default(mul_542, 0);  mul_542 = None
    unsqueeze_299: "f32[1, 1584, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_298, 2);  unsqueeze_298 = None
    unsqueeze_300: "f32[1, 1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_299, 3);  unsqueeze_299 = None
    mul_543: "f32[1584]" = torch.ops.aten.mul.Tensor(squeeze_157, primals_105);  primals_105 = None
    unsqueeze_301: "f32[1, 1584]" = torch.ops.aten.unsqueeze.default(mul_543, 0);  mul_543 = None
    unsqueeze_302: "f32[1, 1584, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_301, 2);  unsqueeze_301 = None
    unsqueeze_303: "f32[1, 1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_302, 3);  unsqueeze_302 = None
    mul_544: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(sub_85, unsqueeze_300);  sub_85 = unsqueeze_300 = None
    sub_87: "f32[8, 1584, 7, 7]" = torch.ops.aten.sub.Tensor(mul_537, mul_544);  mul_537 = mul_544 = None
    sub_88: "f32[8, 1584, 7, 7]" = torch.ops.aten.sub.Tensor(sub_87, unsqueeze_297);  sub_87 = unsqueeze_297 = None
    mul_545: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(sub_88, unsqueeze_303);  sub_88 = unsqueeze_303 = None
    mul_546: "f32[1584]" = torch.ops.aten.mul.Tensor(sum_15, squeeze_157);  sum_15 = squeeze_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    slice_9: "f32[8, 396, 7, 7]" = torch.ops.aten.slice.Tensor(mul_545, 1, 0, 396)
    slice_10: "f32[8, 396, 7, 7]" = torch.ops.aten.slice.Tensor(mul_545, 1, 396, 792)
    slice_11: "f32[8, 396, 7, 7]" = torch.ops.aten.slice.Tensor(mul_545, 1, 792, 1188)
    slice_12: "f32[8, 396, 7, 7]" = torch.ops.aten.slice.Tensor(mul_545, 1, 1188, 1584);  mul_545 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_backward_14 = torch.ops.aten.convolution_backward.default(slice_12, getitem_399, primals_285, [0], [1, 1], [4, 4], [1, 1], False, [0, 0], 396, [True, True, False]);  slice_12 = getitem_399 = primals_285 = None
    getitem_478: "f32[8, 396, 7, 7]" = convolution_backward_14[0]
    getitem_479: "f32[396, 1, 9, 9]" = convolution_backward_14[1];  convolution_backward_14 = None
    convolution_backward_15 = torch.ops.aten.convolution_backward.default(slice_11, getitem_394, primals_284, [0], [1, 1], [3, 3], [1, 1], False, [0, 0], 396, [True, True, False]);  slice_11 = getitem_394 = primals_284 = None
    getitem_481: "f32[8, 396, 7, 7]" = convolution_backward_15[0]
    getitem_482: "f32[396, 1, 7, 7]" = convolution_backward_15[1];  convolution_backward_15 = None
    convolution_backward_16 = torch.ops.aten.convolution_backward.default(slice_10, getitem_389, primals_283, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 396, [True, True, False]);  slice_10 = getitem_389 = primals_283 = None
    getitem_484: "f32[8, 396, 7, 7]" = convolution_backward_16[0]
    getitem_485: "f32[396, 1, 5, 5]" = convolution_backward_16[1];  convolution_backward_16 = None
    convolution_backward_17 = torch.ops.aten.convolution_backward.default(slice_9, getitem_384, primals_282, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 396, [True, True, False]);  slice_9 = getitem_384 = primals_282 = None
    getitem_487: "f32[8, 396, 7, 7]" = convolution_backward_17[0]
    getitem_488: "f32[396, 1, 3, 3]" = convolution_backward_17[1];  convolution_backward_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    cat_44: "f32[8, 1584, 7, 7]" = torch.ops.aten.cat.default([getitem_487, getitem_484, getitem_481, getitem_478], 1);  getitem_487 = getitem_484 = getitem_481 = getitem_478 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_549: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(cat_44, mul_548);  cat_44 = mul_548 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_16: "f32[1584]" = torch.ops.aten.sum.dim_IntList(mul_549, [0, 2, 3])
    sub_90: "f32[8, 1584, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_136, unsqueeze_306);  convolution_136 = unsqueeze_306 = None
    mul_550: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(mul_549, sub_90)
    sum_17: "f32[1584]" = torch.ops.aten.sum.dim_IntList(mul_550, [0, 2, 3]);  mul_550 = None
    mul_551: "f32[1584]" = torch.ops.aten.mul.Tensor(sum_16, 0.002551020408163265)
    unsqueeze_307: "f32[1, 1584]" = torch.ops.aten.unsqueeze.default(mul_551, 0);  mul_551 = None
    unsqueeze_308: "f32[1, 1584, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_307, 2);  unsqueeze_307 = None
    unsqueeze_309: "f32[1, 1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_308, 3);  unsqueeze_308 = None
    mul_552: "f32[1584]" = torch.ops.aten.mul.Tensor(sum_17, 0.002551020408163265)
    mul_553: "f32[1584]" = torch.ops.aten.mul.Tensor(squeeze_154, squeeze_154)
    mul_554: "f32[1584]" = torch.ops.aten.mul.Tensor(mul_552, mul_553);  mul_552 = mul_553 = None
    unsqueeze_310: "f32[1, 1584]" = torch.ops.aten.unsqueeze.default(mul_554, 0);  mul_554 = None
    unsqueeze_311: "f32[1, 1584, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_310, 2);  unsqueeze_310 = None
    unsqueeze_312: "f32[1, 1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_311, 3);  unsqueeze_311 = None
    mul_555: "f32[1584]" = torch.ops.aten.mul.Tensor(squeeze_154, primals_103);  primals_103 = None
    unsqueeze_313: "f32[1, 1584]" = torch.ops.aten.unsqueeze.default(mul_555, 0);  mul_555 = None
    unsqueeze_314: "f32[1, 1584, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_313, 2);  unsqueeze_313 = None
    unsqueeze_315: "f32[1, 1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_314, 3);  unsqueeze_314 = None
    mul_556: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(sub_90, unsqueeze_312);  sub_90 = unsqueeze_312 = None
    sub_92: "f32[8, 1584, 7, 7]" = torch.ops.aten.sub.Tensor(mul_549, mul_556);  mul_549 = mul_556 = None
    sub_93: "f32[8, 1584, 7, 7]" = torch.ops.aten.sub.Tensor(sub_92, unsqueeze_309);  sub_92 = unsqueeze_309 = None
    mul_557: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(sub_93, unsqueeze_315);  sub_93 = unsqueeze_315 = None
    mul_558: "f32[1584]" = torch.ops.aten.mul.Tensor(sum_17, squeeze_154);  sum_17 = squeeze_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_18 = torch.ops.aten.convolution_backward.default(mul_557, add_266, primals_281, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_557 = add_266 = primals_281 = None
    getitem_490: "f32[8, 264, 7, 7]" = convolution_backward_18[0]
    getitem_491: "f32[1584, 264, 1, 1]" = convolution_backward_18[1];  convolution_backward_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_313: "f32[8, 264, 7, 7]" = torch.ops.aten.add.Tensor(add_308, getitem_490);  add_308 = getitem_490 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_18: "f32[264]" = torch.ops.aten.sum.dim_IntList(add_313, [0, 2, 3])
    sub_94: "f32[8, 264, 7, 7]" = torch.ops.aten.sub.Tensor(cat_36, unsqueeze_318);  cat_36 = unsqueeze_318 = None
    mul_559: "f32[8, 264, 7, 7]" = torch.ops.aten.mul.Tensor(add_313, sub_94)
    sum_19: "f32[264]" = torch.ops.aten.sum.dim_IntList(mul_559, [0, 2, 3]);  mul_559 = None
    mul_560: "f32[264]" = torch.ops.aten.mul.Tensor(sum_18, 0.002551020408163265)
    unsqueeze_319: "f32[1, 264]" = torch.ops.aten.unsqueeze.default(mul_560, 0);  mul_560 = None
    unsqueeze_320: "f32[1, 264, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_319, 2);  unsqueeze_319 = None
    unsqueeze_321: "f32[1, 264, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_320, 3);  unsqueeze_320 = None
    mul_561: "f32[264]" = torch.ops.aten.mul.Tensor(sum_19, 0.002551020408163265)
    mul_562: "f32[264]" = torch.ops.aten.mul.Tensor(squeeze_151, squeeze_151)
    mul_563: "f32[264]" = torch.ops.aten.mul.Tensor(mul_561, mul_562);  mul_561 = mul_562 = None
    unsqueeze_322: "f32[1, 264]" = torch.ops.aten.unsqueeze.default(mul_563, 0);  mul_563 = None
    unsqueeze_323: "f32[1, 264, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_322, 2);  unsqueeze_322 = None
    unsqueeze_324: "f32[1, 264, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_323, 3);  unsqueeze_323 = None
    mul_564: "f32[264]" = torch.ops.aten.mul.Tensor(squeeze_151, primals_101);  primals_101 = None
    unsqueeze_325: "f32[1, 264]" = torch.ops.aten.unsqueeze.default(mul_564, 0);  mul_564 = None
    unsqueeze_326: "f32[1, 264, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_325, 2);  unsqueeze_325 = None
    unsqueeze_327: "f32[1, 264, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_326, 3);  unsqueeze_326 = None
    mul_565: "f32[8, 264, 7, 7]" = torch.ops.aten.mul.Tensor(sub_94, unsqueeze_324);  sub_94 = unsqueeze_324 = None
    sub_96: "f32[8, 264, 7, 7]" = torch.ops.aten.sub.Tensor(add_313, mul_565);  mul_565 = None
    sub_97: "f32[8, 264, 7, 7]" = torch.ops.aten.sub.Tensor(sub_96, unsqueeze_321);  sub_96 = unsqueeze_321 = None
    mul_566: "f32[8, 264, 7, 7]" = torch.ops.aten.mul.Tensor(sub_97, unsqueeze_327);  sub_97 = unsqueeze_327 = None
    mul_567: "f32[264]" = torch.ops.aten.mul.Tensor(sum_19, squeeze_151);  sum_19 = squeeze_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    slice_13: "f32[8, 132, 7, 7]" = torch.ops.aten.slice.Tensor(mul_566, 1, 0, 132)
    slice_14: "f32[8, 132, 7, 7]" = torch.ops.aten.slice.Tensor(mul_566, 1, 132, 264);  mul_566 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_backward_19 = torch.ops.aten.convolution_backward.default(slice_14, getitem_375, primals_280, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_14 = getitem_375 = primals_280 = None
    getitem_493: "f32[8, 792, 7, 7]" = convolution_backward_19[0]
    getitem_494: "f32[132, 792, 1, 1]" = convolution_backward_19[1];  convolution_backward_19 = None
    convolution_backward_20 = torch.ops.aten.convolution_backward.default(slice_13, getitem_374, primals_279, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_13 = getitem_374 = primals_279 = None
    getitem_496: "f32[8, 792, 7, 7]" = convolution_backward_20[0]
    getitem_497: "f32[132, 792, 1, 1]" = convolution_backward_20[1];  convolution_backward_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    cat_45: "f32[8, 1584, 7, 7]" = torch.ops.aten.cat.default([getitem_496, getitem_493], 1);  getitem_496 = getitem_493 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_568: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(cat_45, mul_403);  mul_403 = None
    mul_569: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(cat_45, sigmoid_55);  cat_45 = sigmoid_55 = None
    sum_20: "f32[8, 1584, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_568, [2, 3], True);  mul_568 = None
    alias_28: "f32[8, 1584, 1, 1]" = torch.ops.aten.alias.default(alias_19);  alias_19 = None
    sub_98: "f32[8, 1584, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_28)
    mul_570: "f32[8, 1584, 1, 1]" = torch.ops.aten.mul.Tensor(alias_28, sub_98);  alias_28 = sub_98 = None
    mul_571: "f32[8, 1584, 1, 1]" = torch.ops.aten.mul.Tensor(sum_20, mul_570);  sum_20 = mul_570 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_21 = torch.ops.aten.convolution_backward.default(mul_571, mul_404, primals_277, [1584], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_571 = mul_404 = primals_277 = None
    getitem_499: "f32[8, 132, 1, 1]" = convolution_backward_21[0]
    getitem_500: "f32[1584, 132, 1, 1]" = convolution_backward_21[1]
    getitem_501: "f32[1584]" = convolution_backward_21[2];  convolution_backward_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_70: "f32[8, 132, 1, 1]" = torch.ops.aten.sigmoid.default(clone_41)
    sub_99: "f32[8, 132, 1, 1]" = torch.ops.aten.sub.Tensor(full_default_1, sigmoid_70);  full_default_1 = None
    mul_572: "f32[8, 132, 1, 1]" = torch.ops.aten.mul.Tensor(clone_41, sub_99);  clone_41 = sub_99 = None
    add_314: "f32[8, 132, 1, 1]" = torch.ops.aten.add.Scalar(mul_572, 1);  mul_572 = None
    mul_573: "f32[8, 132, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_70, add_314);  sigmoid_70 = add_314 = None
    mul_574: "f32[8, 132, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_499, mul_573);  getitem_499 = mul_573 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_22 = torch.ops.aten.convolution_backward.default(mul_574, mean_13, primals_275, [132], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_574 = mean_13 = primals_275 = None
    getitem_502: "f32[8, 1584, 1, 1]" = convolution_backward_22[0]
    getitem_503: "f32[132, 1584, 1, 1]" = convolution_backward_22[1]
    getitem_504: "f32[132]" = convolution_backward_22[2];  convolution_backward_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_3: "f32[8, 1584, 7, 7]" = torch.ops.aten.expand.default(getitem_502, [8, 1584, 7, 7]);  getitem_502 = None
    div_3: "f32[8, 1584, 7, 7]" = torch.ops.aten.div.Scalar(expand_3, 49);  expand_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_315: "f32[8, 1584, 7, 7]" = torch.ops.aten.add.Tensor(mul_569, div_3);  mul_569 = div_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_71: "f32[8, 1584, 7, 7]" = torch.ops.aten.sigmoid.default(clone_40)
    sub_100: "f32[8, 1584, 7, 7]" = torch.ops.aten.sub.Tensor(full_default_2, sigmoid_71);  full_default_2 = None
    mul_575: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(clone_40, sub_100);  clone_40 = sub_100 = None
    add_316: "f32[8, 1584, 7, 7]" = torch.ops.aten.add.Scalar(mul_575, 1);  mul_575 = None
    mul_576: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(sigmoid_71, add_316);  sigmoid_71 = add_316 = None
    mul_577: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(add_315, mul_576);  add_315 = mul_576 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_21: "f32[1584]" = torch.ops.aten.sum.dim_IntList(mul_577, [0, 2, 3])
    sub_101: "f32[8, 1584, 7, 7]" = torch.ops.aten.sub.Tensor(cat_35, unsqueeze_330);  cat_35 = unsqueeze_330 = None
    mul_578: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(mul_577, sub_101)
    sum_22: "f32[1584]" = torch.ops.aten.sum.dim_IntList(mul_578, [0, 2, 3]);  mul_578 = None
    mul_579: "f32[1584]" = torch.ops.aten.mul.Tensor(sum_21, 0.002551020408163265)
    unsqueeze_331: "f32[1, 1584]" = torch.ops.aten.unsqueeze.default(mul_579, 0);  mul_579 = None
    unsqueeze_332: "f32[1, 1584, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_331, 2);  unsqueeze_331 = None
    unsqueeze_333: "f32[1, 1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_332, 3);  unsqueeze_332 = None
    mul_580: "f32[1584]" = torch.ops.aten.mul.Tensor(sum_22, 0.002551020408163265)
    mul_581: "f32[1584]" = torch.ops.aten.mul.Tensor(squeeze_148, squeeze_148)
    mul_582: "f32[1584]" = torch.ops.aten.mul.Tensor(mul_580, mul_581);  mul_580 = mul_581 = None
    unsqueeze_334: "f32[1, 1584]" = torch.ops.aten.unsqueeze.default(mul_582, 0);  mul_582 = None
    unsqueeze_335: "f32[1, 1584, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_334, 2);  unsqueeze_334 = None
    unsqueeze_336: "f32[1, 1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_335, 3);  unsqueeze_335 = None
    mul_583: "f32[1584]" = torch.ops.aten.mul.Tensor(squeeze_148, primals_99);  primals_99 = None
    unsqueeze_337: "f32[1, 1584]" = torch.ops.aten.unsqueeze.default(mul_583, 0);  mul_583 = None
    unsqueeze_338: "f32[1, 1584, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_337, 2);  unsqueeze_337 = None
    unsqueeze_339: "f32[1, 1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_338, 3);  unsqueeze_338 = None
    mul_584: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(sub_101, unsqueeze_336);  sub_101 = unsqueeze_336 = None
    sub_103: "f32[8, 1584, 7, 7]" = torch.ops.aten.sub.Tensor(mul_577, mul_584);  mul_577 = mul_584 = None
    sub_104: "f32[8, 1584, 7, 7]" = torch.ops.aten.sub.Tensor(sub_103, unsqueeze_333);  sub_103 = unsqueeze_333 = None
    mul_585: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(sub_104, unsqueeze_339);  sub_104 = unsqueeze_339 = None
    mul_586: "f32[1584]" = torch.ops.aten.mul.Tensor(sum_22, squeeze_148);  sum_22 = squeeze_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    slice_15: "f32[8, 396, 7, 7]" = torch.ops.aten.slice.Tensor(mul_585, 1, 0, 396)
    slice_16: "f32[8, 396, 7, 7]" = torch.ops.aten.slice.Tensor(mul_585, 1, 396, 792)
    slice_17: "f32[8, 396, 7, 7]" = torch.ops.aten.slice.Tensor(mul_585, 1, 792, 1188)
    slice_18: "f32[8, 396, 7, 7]" = torch.ops.aten.slice.Tensor(mul_585, 1, 1188, 1584);  mul_585 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_backward_23 = torch.ops.aten.convolution_backward.default(slice_18, getitem_371, primals_274, [0], [1, 1], [4, 4], [1, 1], False, [0, 0], 396, [True, True, False]);  slice_18 = getitem_371 = primals_274 = None
    getitem_505: "f32[8, 396, 7, 7]" = convolution_backward_23[0]
    getitem_506: "f32[396, 1, 9, 9]" = convolution_backward_23[1];  convolution_backward_23 = None
    convolution_backward_24 = torch.ops.aten.convolution_backward.default(slice_17, getitem_366, primals_273, [0], [1, 1], [3, 3], [1, 1], False, [0, 0], 396, [True, True, False]);  slice_17 = getitem_366 = primals_273 = None
    getitem_508: "f32[8, 396, 7, 7]" = convolution_backward_24[0]
    getitem_509: "f32[396, 1, 7, 7]" = convolution_backward_24[1];  convolution_backward_24 = None
    convolution_backward_25 = torch.ops.aten.convolution_backward.default(slice_16, getitem_361, primals_272, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 396, [True, True, False]);  slice_16 = getitem_361 = primals_272 = None
    getitem_511: "f32[8, 396, 7, 7]" = convolution_backward_25[0]
    getitem_512: "f32[396, 1, 5, 5]" = convolution_backward_25[1];  convolution_backward_25 = None
    convolution_backward_26 = torch.ops.aten.convolution_backward.default(slice_15, getitem_356, primals_271, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 396, [True, True, False]);  slice_15 = getitem_356 = primals_271 = None
    getitem_514: "f32[8, 396, 7, 7]" = convolution_backward_26[0]
    getitem_515: "f32[396, 1, 3, 3]" = convolution_backward_26[1];  convolution_backward_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    cat_46: "f32[8, 1584, 7, 7]" = torch.ops.aten.cat.default([getitem_514, getitem_511, getitem_508, getitem_505], 1);  getitem_514 = getitem_511 = getitem_508 = getitem_505 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_589: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(cat_46, mul_588);  cat_46 = mul_588 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_23: "f32[1584]" = torch.ops.aten.sum.dim_IntList(mul_589, [0, 2, 3])
    sub_106: "f32[8, 1584, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_127, unsqueeze_342);  convolution_127 = unsqueeze_342 = None
    mul_590: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(mul_589, sub_106)
    sum_24: "f32[1584]" = torch.ops.aten.sum.dim_IntList(mul_590, [0, 2, 3]);  mul_590 = None
    mul_591: "f32[1584]" = torch.ops.aten.mul.Tensor(sum_23, 0.002551020408163265)
    unsqueeze_343: "f32[1, 1584]" = torch.ops.aten.unsqueeze.default(mul_591, 0);  mul_591 = None
    unsqueeze_344: "f32[1, 1584, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_343, 2);  unsqueeze_343 = None
    unsqueeze_345: "f32[1, 1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_344, 3);  unsqueeze_344 = None
    mul_592: "f32[1584]" = torch.ops.aten.mul.Tensor(sum_24, 0.002551020408163265)
    mul_593: "f32[1584]" = torch.ops.aten.mul.Tensor(squeeze_145, squeeze_145)
    mul_594: "f32[1584]" = torch.ops.aten.mul.Tensor(mul_592, mul_593);  mul_592 = mul_593 = None
    unsqueeze_346: "f32[1, 1584]" = torch.ops.aten.unsqueeze.default(mul_594, 0);  mul_594 = None
    unsqueeze_347: "f32[1, 1584, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_346, 2);  unsqueeze_346 = None
    unsqueeze_348: "f32[1, 1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_347, 3);  unsqueeze_347 = None
    mul_595: "f32[1584]" = torch.ops.aten.mul.Tensor(squeeze_145, primals_97);  primals_97 = None
    unsqueeze_349: "f32[1, 1584]" = torch.ops.aten.unsqueeze.default(mul_595, 0);  mul_595 = None
    unsqueeze_350: "f32[1, 1584, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_349, 2);  unsqueeze_349 = None
    unsqueeze_351: "f32[1, 1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_350, 3);  unsqueeze_350 = None
    mul_596: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(sub_106, unsqueeze_348);  sub_106 = unsqueeze_348 = None
    sub_108: "f32[8, 1584, 7, 7]" = torch.ops.aten.sub.Tensor(mul_589, mul_596);  mul_589 = mul_596 = None
    sub_109: "f32[8, 1584, 7, 7]" = torch.ops.aten.sub.Tensor(sub_108, unsqueeze_345);  sub_108 = unsqueeze_345 = None
    mul_597: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(sub_109, unsqueeze_351);  sub_109 = unsqueeze_351 = None
    mul_598: "f32[1584]" = torch.ops.aten.mul.Tensor(sum_24, squeeze_145);  sum_24 = squeeze_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_27 = torch.ops.aten.convolution_backward.default(mul_597, add_250, primals_270, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_597 = add_250 = primals_270 = None
    getitem_517: "f32[8, 264, 7, 7]" = convolution_backward_27[0]
    getitem_518: "f32[1584, 264, 1, 1]" = convolution_backward_27[1];  convolution_backward_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_318: "f32[8, 264, 7, 7]" = torch.ops.aten.add.Tensor(add_313, getitem_517);  add_313 = getitem_517 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_25: "f32[264]" = torch.ops.aten.sum.dim_IntList(add_318, [0, 2, 3])
    sub_110: "f32[8, 264, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_126, unsqueeze_354);  convolution_126 = unsqueeze_354 = None
    mul_599: "f32[8, 264, 7, 7]" = torch.ops.aten.mul.Tensor(add_318, sub_110)
    sum_26: "f32[264]" = torch.ops.aten.sum.dim_IntList(mul_599, [0, 2, 3]);  mul_599 = None
    mul_600: "f32[264]" = torch.ops.aten.mul.Tensor(sum_25, 0.002551020408163265)
    unsqueeze_355: "f32[1, 264]" = torch.ops.aten.unsqueeze.default(mul_600, 0);  mul_600 = None
    unsqueeze_356: "f32[1, 264, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_355, 2);  unsqueeze_355 = None
    unsqueeze_357: "f32[1, 264, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_356, 3);  unsqueeze_356 = None
    mul_601: "f32[264]" = torch.ops.aten.mul.Tensor(sum_26, 0.002551020408163265)
    mul_602: "f32[264]" = torch.ops.aten.mul.Tensor(squeeze_142, squeeze_142)
    mul_603: "f32[264]" = torch.ops.aten.mul.Tensor(mul_601, mul_602);  mul_601 = mul_602 = None
    unsqueeze_358: "f32[1, 264]" = torch.ops.aten.unsqueeze.default(mul_603, 0);  mul_603 = None
    unsqueeze_359: "f32[1, 264, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_358, 2);  unsqueeze_358 = None
    unsqueeze_360: "f32[1, 264, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_359, 3);  unsqueeze_359 = None
    mul_604: "f32[264]" = torch.ops.aten.mul.Tensor(squeeze_142, primals_95);  primals_95 = None
    unsqueeze_361: "f32[1, 264]" = torch.ops.aten.unsqueeze.default(mul_604, 0);  mul_604 = None
    unsqueeze_362: "f32[1, 264, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_361, 2);  unsqueeze_361 = None
    unsqueeze_363: "f32[1, 264, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_362, 3);  unsqueeze_362 = None
    mul_605: "f32[8, 264, 7, 7]" = torch.ops.aten.mul.Tensor(sub_110, unsqueeze_360);  sub_110 = unsqueeze_360 = None
    sub_112: "f32[8, 264, 7, 7]" = torch.ops.aten.sub.Tensor(add_318, mul_605);  add_318 = mul_605 = None
    sub_113: "f32[8, 264, 7, 7]" = torch.ops.aten.sub.Tensor(sub_112, unsqueeze_357);  sub_112 = unsqueeze_357 = None
    mul_606: "f32[8, 264, 7, 7]" = torch.ops.aten.mul.Tensor(sub_113, unsqueeze_363);  sub_113 = unsqueeze_363 = None
    mul_607: "f32[264]" = torch.ops.aten.mul.Tensor(sum_26, squeeze_142);  sum_26 = squeeze_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_28 = torch.ops.aten.convolution_backward.default(mul_606, mul_380, primals_269, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_606 = mul_380 = primals_269 = None
    getitem_520: "f32[8, 960, 7, 7]" = convolution_backward_28[0]
    getitem_521: "f32[264, 960, 1, 1]" = convolution_backward_28[1];  convolution_backward_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_608: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_520, mul_378);  mul_378 = None
    mul_609: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_520, sigmoid_51);  getitem_520 = sigmoid_51 = None
    sum_27: "f32[8, 960, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_608, [2, 3], True);  mul_608 = None
    alias_29: "f32[8, 960, 1, 1]" = torch.ops.aten.alias.default(alias_18);  alias_18 = None
    sub_114: "f32[8, 960, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_29)
    mul_610: "f32[8, 960, 1, 1]" = torch.ops.aten.mul.Tensor(alias_29, sub_114);  alias_29 = sub_114 = None
    mul_611: "f32[8, 960, 1, 1]" = torch.ops.aten.mul.Tensor(sum_27, mul_610);  sum_27 = mul_610 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_29 = torch.ops.aten.convolution_backward.default(mul_611, mul_379, primals_267, [960], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_611 = mul_379 = primals_267 = None
    getitem_523: "f32[8, 80, 1, 1]" = convolution_backward_29[0]
    getitem_524: "f32[960, 80, 1, 1]" = convolution_backward_29[1]
    getitem_525: "f32[960]" = convolution_backward_29[2];  convolution_backward_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_73: "f32[8, 80, 1, 1]" = torch.ops.aten.sigmoid.default(clone_38)
    full_default_10: "f32[8, 80, 1, 1]" = torch.ops.aten.full.default([8, 80, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_115: "f32[8, 80, 1, 1]" = torch.ops.aten.sub.Tensor(full_default_10, sigmoid_73)
    mul_612: "f32[8, 80, 1, 1]" = torch.ops.aten.mul.Tensor(clone_38, sub_115);  clone_38 = sub_115 = None
    add_319: "f32[8, 80, 1, 1]" = torch.ops.aten.add.Scalar(mul_612, 1);  mul_612 = None
    mul_613: "f32[8, 80, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_73, add_319);  sigmoid_73 = add_319 = None
    mul_614: "f32[8, 80, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_523, mul_613);  getitem_523 = mul_613 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_30 = torch.ops.aten.convolution_backward.default(mul_614, mean_12, primals_265, [80], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_614 = mean_12 = primals_265 = None
    getitem_526: "f32[8, 960, 1, 1]" = convolution_backward_30[0]
    getitem_527: "f32[80, 960, 1, 1]" = convolution_backward_30[1]
    getitem_528: "f32[80]" = convolution_backward_30[2];  convolution_backward_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_4: "f32[8, 960, 7, 7]" = torch.ops.aten.expand.default(getitem_526, [8, 960, 7, 7]);  getitem_526 = None
    div_4: "f32[8, 960, 7, 7]" = torch.ops.aten.div.Scalar(expand_4, 49);  expand_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_320: "f32[8, 960, 7, 7]" = torch.ops.aten.add.Tensor(mul_609, div_4);  mul_609 = div_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_74: "f32[8, 960, 7, 7]" = torch.ops.aten.sigmoid.default(clone_37)
    full_default_11: "f32[8, 960, 7, 7]" = torch.ops.aten.full.default([8, 960, 7, 7], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_116: "f32[8, 960, 7, 7]" = torch.ops.aten.sub.Tensor(full_default_11, sigmoid_74);  full_default_11 = None
    mul_615: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(clone_37, sub_116);  clone_37 = sub_116 = None
    add_321: "f32[8, 960, 7, 7]" = torch.ops.aten.add.Scalar(mul_615, 1);  mul_615 = None
    mul_616: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(sigmoid_74, add_321);  sigmoid_74 = add_321 = None
    mul_617: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(add_320, mul_616);  add_320 = mul_616 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_28: "f32[960]" = torch.ops.aten.sum.dim_IntList(mul_617, [0, 2, 3])
    sub_117: "f32[8, 960, 7, 7]" = torch.ops.aten.sub.Tensor(cat_34, unsqueeze_366);  cat_34 = unsqueeze_366 = None
    mul_618: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(mul_617, sub_117)
    sum_29: "f32[960]" = torch.ops.aten.sum.dim_IntList(mul_618, [0, 2, 3]);  mul_618 = None
    mul_619: "f32[960]" = torch.ops.aten.mul.Tensor(sum_28, 0.002551020408163265)
    unsqueeze_367: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(mul_619, 0);  mul_619 = None
    unsqueeze_368: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_367, 2);  unsqueeze_367 = None
    unsqueeze_369: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_368, 3);  unsqueeze_368 = None
    mul_620: "f32[960]" = torch.ops.aten.mul.Tensor(sum_29, 0.002551020408163265)
    mul_621: "f32[960]" = torch.ops.aten.mul.Tensor(squeeze_139, squeeze_139)
    mul_622: "f32[960]" = torch.ops.aten.mul.Tensor(mul_620, mul_621);  mul_620 = mul_621 = None
    unsqueeze_370: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(mul_622, 0);  mul_622 = None
    unsqueeze_371: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_370, 2);  unsqueeze_370 = None
    unsqueeze_372: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_371, 3);  unsqueeze_371 = None
    mul_623: "f32[960]" = torch.ops.aten.mul.Tensor(squeeze_139, primals_93);  primals_93 = None
    unsqueeze_373: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(mul_623, 0);  mul_623 = None
    unsqueeze_374: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_373, 2);  unsqueeze_373 = None
    unsqueeze_375: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_374, 3);  unsqueeze_374 = None
    mul_624: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(sub_117, unsqueeze_372);  sub_117 = unsqueeze_372 = None
    sub_119: "f32[8, 960, 7, 7]" = torch.ops.aten.sub.Tensor(mul_617, mul_624);  mul_617 = mul_624 = None
    sub_120: "f32[8, 960, 7, 7]" = torch.ops.aten.sub.Tensor(sub_119, unsqueeze_369);  sub_119 = unsqueeze_369 = None
    mul_625: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(sub_120, unsqueeze_375);  sub_120 = unsqueeze_375 = None
    mul_626: "f32[960]" = torch.ops.aten.mul.Tensor(sum_29, squeeze_139);  sum_29 = squeeze_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    slice_19: "f32[8, 240, 7, 7]" = torch.ops.aten.slice.Tensor(mul_625, 1, 0, 240)
    slice_20: "f32[8, 240, 7, 7]" = torch.ops.aten.slice.Tensor(mul_625, 1, 240, 480)
    slice_21: "f32[8, 240, 7, 7]" = torch.ops.aten.slice.Tensor(mul_625, 1, 480, 720)
    slice_22: "f32[8, 240, 7, 7]" = torch.ops.aten.slice.Tensor(mul_625, 1, 720, 960);  mul_625 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_backward_31 = torch.ops.aten.convolution_backward.default(slice_22, getitem_345, primals_264, [0], [2, 2], [4, 4], [1, 1], False, [0, 0], 240, [True, True, False]);  slice_22 = getitem_345 = primals_264 = None
    getitem_529: "f32[8, 240, 14, 14]" = convolution_backward_31[0]
    getitem_530: "f32[240, 1, 9, 9]" = convolution_backward_31[1];  convolution_backward_31 = None
    convolution_backward_32 = torch.ops.aten.convolution_backward.default(slice_21, getitem_340, primals_263, [0], [2, 2], [3, 3], [1, 1], False, [0, 0], 240, [True, True, False]);  slice_21 = getitem_340 = primals_263 = None
    getitem_532: "f32[8, 240, 14, 14]" = convolution_backward_32[0]
    getitem_533: "f32[240, 1, 7, 7]" = convolution_backward_32[1];  convolution_backward_32 = None
    convolution_backward_33 = torch.ops.aten.convolution_backward.default(slice_20, getitem_335, primals_262, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 240, [True, True, False]);  slice_20 = getitem_335 = primals_262 = None
    getitem_535: "f32[8, 240, 14, 14]" = convolution_backward_33[0]
    getitem_536: "f32[240, 1, 5, 5]" = convolution_backward_33[1];  convolution_backward_33 = None
    convolution_backward_34 = torch.ops.aten.convolution_backward.default(slice_19, getitem_330, primals_261, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 240, [True, True, False]);  slice_19 = getitem_330 = primals_261 = None
    getitem_538: "f32[8, 240, 14, 14]" = convolution_backward_34[0]
    getitem_539: "f32[240, 1, 3, 3]" = convolution_backward_34[1];  convolution_backward_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    cat_47: "f32[8, 960, 14, 14]" = torch.ops.aten.cat.default([getitem_538, getitem_535, getitem_532, getitem_529], 1);  getitem_538 = getitem_535 = getitem_532 = getitem_529 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_629: "f32[8, 960, 14, 14]" = torch.ops.aten.mul.Tensor(cat_47, mul_628);  cat_47 = mul_628 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_30: "f32[960]" = torch.ops.aten.sum.dim_IntList(mul_629, [0, 2, 3])
    sub_122: "f32[8, 960, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_119, unsqueeze_378);  convolution_119 = unsqueeze_378 = None
    mul_630: "f32[8, 960, 14, 14]" = torch.ops.aten.mul.Tensor(mul_629, sub_122)
    sum_31: "f32[960]" = torch.ops.aten.sum.dim_IntList(mul_630, [0, 2, 3]);  mul_630 = None
    mul_631: "f32[960]" = torch.ops.aten.mul.Tensor(sum_30, 0.0006377551020408163)
    unsqueeze_379: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(mul_631, 0);  mul_631 = None
    unsqueeze_380: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_379, 2);  unsqueeze_379 = None
    unsqueeze_381: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_380, 3);  unsqueeze_380 = None
    mul_632: "f32[960]" = torch.ops.aten.mul.Tensor(sum_31, 0.0006377551020408163)
    mul_633: "f32[960]" = torch.ops.aten.mul.Tensor(squeeze_136, squeeze_136)
    mul_634: "f32[960]" = torch.ops.aten.mul.Tensor(mul_632, mul_633);  mul_632 = mul_633 = None
    unsqueeze_382: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(mul_634, 0);  mul_634 = None
    unsqueeze_383: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_382, 2);  unsqueeze_382 = None
    unsqueeze_384: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_383, 3);  unsqueeze_383 = None
    mul_635: "f32[960]" = torch.ops.aten.mul.Tensor(squeeze_136, primals_91);  primals_91 = None
    unsqueeze_385: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(mul_635, 0);  mul_635 = None
    unsqueeze_386: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_385, 2);  unsqueeze_385 = None
    unsqueeze_387: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_386, 3);  unsqueeze_386 = None
    mul_636: "f32[8, 960, 14, 14]" = torch.ops.aten.mul.Tensor(sub_122, unsqueeze_384);  sub_122 = unsqueeze_384 = None
    sub_124: "f32[8, 960, 14, 14]" = torch.ops.aten.sub.Tensor(mul_629, mul_636);  mul_629 = mul_636 = None
    sub_125: "f32[8, 960, 14, 14]" = torch.ops.aten.sub.Tensor(sub_124, unsqueeze_381);  sub_124 = unsqueeze_381 = None
    mul_637: "f32[8, 960, 14, 14]" = torch.ops.aten.mul.Tensor(sub_125, unsqueeze_387);  sub_125 = unsqueeze_387 = None
    mul_638: "f32[960]" = torch.ops.aten.mul.Tensor(sum_31, squeeze_136);  sum_31 = squeeze_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_35 = torch.ops.aten.convolution_backward.default(mul_637, add_235, primals_260, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_637 = add_235 = primals_260 = None
    getitem_541: "f32[8, 160, 14, 14]" = convolution_backward_35[0]
    getitem_542: "f32[960, 160, 1, 1]" = convolution_backward_35[1];  convolution_backward_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_32: "f32[160]" = torch.ops.aten.sum.dim_IntList(getitem_541, [0, 2, 3])
    sub_126: "f32[8, 160, 14, 14]" = torch.ops.aten.sub.Tensor(cat_33, unsqueeze_390);  cat_33 = unsqueeze_390 = None
    mul_639: "f32[8, 160, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_541, sub_126)
    sum_33: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_639, [0, 2, 3]);  mul_639 = None
    mul_640: "f32[160]" = torch.ops.aten.mul.Tensor(sum_32, 0.0006377551020408163)
    unsqueeze_391: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_640, 0);  mul_640 = None
    unsqueeze_392: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_391, 2);  unsqueeze_391 = None
    unsqueeze_393: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_392, 3);  unsqueeze_392 = None
    mul_641: "f32[160]" = torch.ops.aten.mul.Tensor(sum_33, 0.0006377551020408163)
    mul_642: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_133, squeeze_133)
    mul_643: "f32[160]" = torch.ops.aten.mul.Tensor(mul_641, mul_642);  mul_641 = mul_642 = None
    unsqueeze_394: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_643, 0);  mul_643 = None
    unsqueeze_395: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_394, 2);  unsqueeze_394 = None
    unsqueeze_396: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_395, 3);  unsqueeze_395 = None
    mul_644: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_133, primals_89);  primals_89 = None
    unsqueeze_397: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_644, 0);  mul_644 = None
    unsqueeze_398: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_397, 2);  unsqueeze_397 = None
    unsqueeze_399: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_398, 3);  unsqueeze_398 = None
    mul_645: "f32[8, 160, 14, 14]" = torch.ops.aten.mul.Tensor(sub_126, unsqueeze_396);  sub_126 = unsqueeze_396 = None
    sub_128: "f32[8, 160, 14, 14]" = torch.ops.aten.sub.Tensor(getitem_541, mul_645);  mul_645 = None
    sub_129: "f32[8, 160, 14, 14]" = torch.ops.aten.sub.Tensor(sub_128, unsqueeze_393);  sub_128 = unsqueeze_393 = None
    mul_646: "f32[8, 160, 14, 14]" = torch.ops.aten.mul.Tensor(sub_129, unsqueeze_399);  sub_129 = unsqueeze_399 = None
    mul_647: "f32[160]" = torch.ops.aten.mul.Tensor(sum_33, squeeze_133);  sum_33 = squeeze_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    slice_23: "f32[8, 80, 14, 14]" = torch.ops.aten.slice.Tensor(mul_646, 1, 0, 80)
    slice_24: "f32[8, 80, 14, 14]" = torch.ops.aten.slice.Tensor(mul_646, 1, 80, 160);  mul_646 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_backward_36 = torch.ops.aten.convolution_backward.default(slice_24, getitem_321, primals_259, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_24 = getitem_321 = primals_259 = None
    getitem_544: "f32[8, 240, 14, 14]" = convolution_backward_36[0]
    getitem_545: "f32[80, 240, 1, 1]" = convolution_backward_36[1];  convolution_backward_36 = None
    convolution_backward_37 = torch.ops.aten.convolution_backward.default(slice_23, getitem_320, primals_258, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_23 = getitem_320 = primals_258 = None
    getitem_547: "f32[8, 240, 14, 14]" = convolution_backward_37[0]
    getitem_548: "f32[80, 240, 1, 1]" = convolution_backward_37[1];  convolution_backward_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    cat_48: "f32[8, 480, 14, 14]" = torch.ops.aten.cat.default([getitem_547, getitem_544], 1);  getitem_547 = getitem_544 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_648: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(cat_48, mul_353);  mul_353 = None
    mul_649: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(cat_48, sigmoid_47);  cat_48 = sigmoid_47 = None
    sum_34: "f32[8, 480, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_648, [2, 3], True);  mul_648 = None
    alias_30: "f32[8, 480, 1, 1]" = torch.ops.aten.alias.default(alias_17);  alias_17 = None
    sub_130: "f32[8, 480, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_30)
    mul_650: "f32[8, 480, 1, 1]" = torch.ops.aten.mul.Tensor(alias_30, sub_130);  alias_30 = sub_130 = None
    mul_651: "f32[8, 480, 1, 1]" = torch.ops.aten.mul.Tensor(sum_34, mul_650);  sum_34 = mul_650 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_38 = torch.ops.aten.convolution_backward.default(mul_651, mul_354, primals_256, [480], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_651 = mul_354 = primals_256 = None
    getitem_550: "f32[8, 80, 1, 1]" = convolution_backward_38[0]
    getitem_551: "f32[480, 80, 1, 1]" = convolution_backward_38[1]
    getitem_552: "f32[480]" = convolution_backward_38[2];  convolution_backward_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_76: "f32[8, 80, 1, 1]" = torch.ops.aten.sigmoid.default(clone_35)
    sub_131: "f32[8, 80, 1, 1]" = torch.ops.aten.sub.Tensor(full_default_10, sigmoid_76)
    mul_652: "f32[8, 80, 1, 1]" = torch.ops.aten.mul.Tensor(clone_35, sub_131);  clone_35 = sub_131 = None
    add_323: "f32[8, 80, 1, 1]" = torch.ops.aten.add.Scalar(mul_652, 1);  mul_652 = None
    mul_653: "f32[8, 80, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_76, add_323);  sigmoid_76 = add_323 = None
    mul_654: "f32[8, 80, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_550, mul_653);  getitem_550 = mul_653 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_39 = torch.ops.aten.convolution_backward.default(mul_654, mean_11, primals_254, [80], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_654 = mean_11 = primals_254 = None
    getitem_553: "f32[8, 480, 1, 1]" = convolution_backward_39[0]
    getitem_554: "f32[80, 480, 1, 1]" = convolution_backward_39[1]
    getitem_555: "f32[80]" = convolution_backward_39[2];  convolution_backward_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_5: "f32[8, 480, 14, 14]" = torch.ops.aten.expand.default(getitem_553, [8, 480, 14, 14]);  getitem_553 = None
    div_5: "f32[8, 480, 14, 14]" = torch.ops.aten.div.Scalar(expand_5, 196);  expand_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_324: "f32[8, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_649, div_5);  mul_649 = div_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_77: "f32[8, 480, 14, 14]" = torch.ops.aten.sigmoid.default(clone_34)
    full_default_14: "f32[8, 480, 14, 14]" = torch.ops.aten.full.default([8, 480, 14, 14], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_132: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(full_default_14, sigmoid_77)
    mul_655: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(clone_34, sub_132);  clone_34 = sub_132 = None
    add_325: "f32[8, 480, 14, 14]" = torch.ops.aten.add.Scalar(mul_655, 1);  mul_655 = None
    mul_656: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_77, add_325);  sigmoid_77 = add_325 = None
    mul_657: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_324, mul_656);  add_324 = mul_656 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_35: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_657, [0, 2, 3])
    sub_133: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(cat_32, unsqueeze_402);  cat_32 = unsqueeze_402 = None
    mul_658: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_657, sub_133)
    sum_36: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_658, [0, 2, 3]);  mul_658 = None
    mul_659: "f32[480]" = torch.ops.aten.mul.Tensor(sum_35, 0.0006377551020408163)
    unsqueeze_403: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_659, 0);  mul_659 = None
    unsqueeze_404: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_403, 2);  unsqueeze_403 = None
    unsqueeze_405: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_404, 3);  unsqueeze_404 = None
    mul_660: "f32[480]" = torch.ops.aten.mul.Tensor(sum_36, 0.0006377551020408163)
    mul_661: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_130, squeeze_130)
    mul_662: "f32[480]" = torch.ops.aten.mul.Tensor(mul_660, mul_661);  mul_660 = mul_661 = None
    unsqueeze_406: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_662, 0);  mul_662 = None
    unsqueeze_407: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_406, 2);  unsqueeze_406 = None
    unsqueeze_408: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_407, 3);  unsqueeze_407 = None
    mul_663: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_130, primals_87);  primals_87 = None
    unsqueeze_409: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_663, 0);  mul_663 = None
    unsqueeze_410: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_409, 2);  unsqueeze_409 = None
    unsqueeze_411: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_410, 3);  unsqueeze_410 = None
    mul_664: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_133, unsqueeze_408);  sub_133 = unsqueeze_408 = None
    sub_135: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(mul_657, mul_664);  mul_657 = mul_664 = None
    sub_136: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(sub_135, unsqueeze_405);  sub_135 = unsqueeze_405 = None
    mul_665: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_136, unsqueeze_411);  sub_136 = unsqueeze_411 = None
    mul_666: "f32[480]" = torch.ops.aten.mul.Tensor(sum_36, squeeze_130);  sum_36 = squeeze_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    slice_25: "f32[8, 120, 14, 14]" = torch.ops.aten.slice.Tensor(mul_665, 1, 0, 120)
    slice_26: "f32[8, 120, 14, 14]" = torch.ops.aten.slice.Tensor(mul_665, 1, 120, 240)
    slice_27: "f32[8, 120, 14, 14]" = torch.ops.aten.slice.Tensor(mul_665, 1, 240, 360)
    slice_28: "f32[8, 120, 14, 14]" = torch.ops.aten.slice.Tensor(mul_665, 1, 360, 480);  mul_665 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_backward_40 = torch.ops.aten.convolution_backward.default(slice_28, getitem_317, primals_253, [0], [1, 1], [4, 4], [1, 1], False, [0, 0], 120, [True, True, False]);  slice_28 = getitem_317 = primals_253 = None
    getitem_556: "f32[8, 120, 14, 14]" = convolution_backward_40[0]
    getitem_557: "f32[120, 1, 9, 9]" = convolution_backward_40[1];  convolution_backward_40 = None
    convolution_backward_41 = torch.ops.aten.convolution_backward.default(slice_27, getitem_312, primals_252, [0], [1, 1], [3, 3], [1, 1], False, [0, 0], 120, [True, True, False]);  slice_27 = getitem_312 = primals_252 = None
    getitem_559: "f32[8, 120, 14, 14]" = convolution_backward_41[0]
    getitem_560: "f32[120, 1, 7, 7]" = convolution_backward_41[1];  convolution_backward_41 = None
    convolution_backward_42 = torch.ops.aten.convolution_backward.default(slice_26, getitem_307, primals_251, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 120, [True, True, False]);  slice_26 = getitem_307 = primals_251 = None
    getitem_562: "f32[8, 120, 14, 14]" = convolution_backward_42[0]
    getitem_563: "f32[120, 1, 5, 5]" = convolution_backward_42[1];  convolution_backward_42 = None
    convolution_backward_43 = torch.ops.aten.convolution_backward.default(slice_25, getitem_302, primals_250, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 120, [True, True, False]);  slice_25 = getitem_302 = primals_250 = None
    getitem_565: "f32[8, 120, 14, 14]" = convolution_backward_43[0]
    getitem_566: "f32[120, 1, 3, 3]" = convolution_backward_43[1];  convolution_backward_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    cat_49: "f32[8, 480, 14, 14]" = torch.ops.aten.cat.default([getitem_565, getitem_562, getitem_559, getitem_556], 1);  getitem_565 = getitem_562 = getitem_559 = getitem_556 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_669: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(cat_49, mul_668);  cat_49 = mul_668 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_37: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_669, [0, 2, 3])
    sub_138: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(cat_31, unsqueeze_414);  cat_31 = unsqueeze_414 = None
    mul_670: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_669, sub_138)
    sum_38: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_670, [0, 2, 3]);  mul_670 = None
    mul_671: "f32[480]" = torch.ops.aten.mul.Tensor(sum_37, 0.0006377551020408163)
    unsqueeze_415: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_671, 0);  mul_671 = None
    unsqueeze_416: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_415, 2);  unsqueeze_415 = None
    unsqueeze_417: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_416, 3);  unsqueeze_416 = None
    mul_672: "f32[480]" = torch.ops.aten.mul.Tensor(sum_38, 0.0006377551020408163)
    mul_673: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_127, squeeze_127)
    mul_674: "f32[480]" = torch.ops.aten.mul.Tensor(mul_672, mul_673);  mul_672 = mul_673 = None
    unsqueeze_418: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_674, 0);  mul_674 = None
    unsqueeze_419: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_418, 2);  unsqueeze_418 = None
    unsqueeze_420: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_419, 3);  unsqueeze_419 = None
    mul_675: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_127, primals_85);  primals_85 = None
    unsqueeze_421: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_675, 0);  mul_675 = None
    unsqueeze_422: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_421, 2);  unsqueeze_421 = None
    unsqueeze_423: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_422, 3);  unsqueeze_422 = None
    mul_676: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_138, unsqueeze_420);  sub_138 = unsqueeze_420 = None
    sub_140: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(mul_669, mul_676);  mul_669 = mul_676 = None
    sub_141: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(sub_140, unsqueeze_417);  sub_140 = unsqueeze_417 = None
    mul_677: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_141, unsqueeze_423);  sub_141 = unsqueeze_423 = None
    mul_678: "f32[480]" = torch.ops.aten.mul.Tensor(sum_38, squeeze_127);  sum_38 = squeeze_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    slice_29: "f32[8, 240, 14, 14]" = torch.ops.aten.slice.Tensor(mul_677, 1, 0, 240)
    slice_30: "f32[8, 240, 14, 14]" = torch.ops.aten.slice.Tensor(mul_677, 1, 240, 480);  mul_677 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_backward_44 = torch.ops.aten.convolution_backward.default(slice_30, getitem_295, primals_249, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_30 = getitem_295 = primals_249 = None
    getitem_568: "f32[8, 80, 14, 14]" = convolution_backward_44[0]
    getitem_569: "f32[240, 80, 1, 1]" = convolution_backward_44[1];  convolution_backward_44 = None
    convolution_backward_45 = torch.ops.aten.convolution_backward.default(slice_29, getitem_294, primals_248, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_29 = getitem_294 = primals_248 = None
    getitem_571: "f32[8, 80, 14, 14]" = convolution_backward_45[0]
    getitem_572: "f32[240, 80, 1, 1]" = convolution_backward_45[1];  convolution_backward_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    cat_50: "f32[8, 160, 14, 14]" = torch.ops.aten.cat.default([getitem_571, getitem_568], 1);  getitem_571 = getitem_568 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    add_327: "f32[8, 160, 14, 14]" = torch.ops.aten.add.Tensor(getitem_541, cat_50);  getitem_541 = cat_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_39: "f32[160]" = torch.ops.aten.sum.dim_IntList(add_327, [0, 2, 3])
    sub_142: "f32[8, 160, 14, 14]" = torch.ops.aten.sub.Tensor(cat_30, unsqueeze_426);  cat_30 = unsqueeze_426 = None
    mul_679: "f32[8, 160, 14, 14]" = torch.ops.aten.mul.Tensor(add_327, sub_142)
    sum_40: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_679, [0, 2, 3]);  mul_679 = None
    mul_680: "f32[160]" = torch.ops.aten.mul.Tensor(sum_39, 0.0006377551020408163)
    unsqueeze_427: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_680, 0);  mul_680 = None
    unsqueeze_428: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_427, 2);  unsqueeze_427 = None
    unsqueeze_429: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_428, 3);  unsqueeze_428 = None
    mul_681: "f32[160]" = torch.ops.aten.mul.Tensor(sum_40, 0.0006377551020408163)
    mul_682: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_124, squeeze_124)
    mul_683: "f32[160]" = torch.ops.aten.mul.Tensor(mul_681, mul_682);  mul_681 = mul_682 = None
    unsqueeze_430: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_683, 0);  mul_683 = None
    unsqueeze_431: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_430, 2);  unsqueeze_430 = None
    unsqueeze_432: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_431, 3);  unsqueeze_431 = None
    mul_684: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_124, primals_83);  primals_83 = None
    unsqueeze_433: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_684, 0);  mul_684 = None
    unsqueeze_434: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_433, 2);  unsqueeze_433 = None
    unsqueeze_435: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_434, 3);  unsqueeze_434 = None
    mul_685: "f32[8, 160, 14, 14]" = torch.ops.aten.mul.Tensor(sub_142, unsqueeze_432);  sub_142 = unsqueeze_432 = None
    sub_144: "f32[8, 160, 14, 14]" = torch.ops.aten.sub.Tensor(add_327, mul_685);  mul_685 = None
    sub_145: "f32[8, 160, 14, 14]" = torch.ops.aten.sub.Tensor(sub_144, unsqueeze_429);  sub_144 = unsqueeze_429 = None
    mul_686: "f32[8, 160, 14, 14]" = torch.ops.aten.mul.Tensor(sub_145, unsqueeze_435);  sub_145 = unsqueeze_435 = None
    mul_687: "f32[160]" = torch.ops.aten.mul.Tensor(sum_40, squeeze_124);  sum_40 = squeeze_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    slice_31: "f32[8, 80, 14, 14]" = torch.ops.aten.slice.Tensor(mul_686, 1, 0, 80)
    slice_32: "f32[8, 80, 14, 14]" = torch.ops.aten.slice.Tensor(mul_686, 1, 80, 160);  mul_686 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_backward_46 = torch.ops.aten.convolution_backward.default(slice_32, getitem_291, primals_247, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_32 = getitem_291 = primals_247 = None
    getitem_574: "f32[8, 240, 14, 14]" = convolution_backward_46[0]
    getitem_575: "f32[80, 240, 1, 1]" = convolution_backward_46[1];  convolution_backward_46 = None
    convolution_backward_47 = torch.ops.aten.convolution_backward.default(slice_31, getitem_290, primals_246, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_31 = getitem_290 = primals_246 = None
    getitem_577: "f32[8, 240, 14, 14]" = convolution_backward_47[0]
    getitem_578: "f32[80, 240, 1, 1]" = convolution_backward_47[1];  convolution_backward_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    cat_51: "f32[8, 480, 14, 14]" = torch.ops.aten.cat.default([getitem_577, getitem_574], 1);  getitem_577 = getitem_574 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_688: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(cat_51, mul_328);  mul_328 = None
    mul_689: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(cat_51, sigmoid_43);  cat_51 = sigmoid_43 = None
    sum_41: "f32[8, 480, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_688, [2, 3], True);  mul_688 = None
    alias_31: "f32[8, 480, 1, 1]" = torch.ops.aten.alias.default(alias_16);  alias_16 = None
    sub_146: "f32[8, 480, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_31)
    mul_690: "f32[8, 480, 1, 1]" = torch.ops.aten.mul.Tensor(alias_31, sub_146);  alias_31 = sub_146 = None
    mul_691: "f32[8, 480, 1, 1]" = torch.ops.aten.mul.Tensor(sum_41, mul_690);  sum_41 = mul_690 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_48 = torch.ops.aten.convolution_backward.default(mul_691, mul_329, primals_244, [480], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_691 = mul_329 = primals_244 = None
    getitem_580: "f32[8, 80, 1, 1]" = convolution_backward_48[0]
    getitem_581: "f32[480, 80, 1, 1]" = convolution_backward_48[1]
    getitem_582: "f32[480]" = convolution_backward_48[2];  convolution_backward_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_79: "f32[8, 80, 1, 1]" = torch.ops.aten.sigmoid.default(clone_32)
    sub_147: "f32[8, 80, 1, 1]" = torch.ops.aten.sub.Tensor(full_default_10, sigmoid_79)
    mul_692: "f32[8, 80, 1, 1]" = torch.ops.aten.mul.Tensor(clone_32, sub_147);  clone_32 = sub_147 = None
    add_328: "f32[8, 80, 1, 1]" = torch.ops.aten.add.Scalar(mul_692, 1);  mul_692 = None
    mul_693: "f32[8, 80, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_79, add_328);  sigmoid_79 = add_328 = None
    mul_694: "f32[8, 80, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_580, mul_693);  getitem_580 = mul_693 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_49 = torch.ops.aten.convolution_backward.default(mul_694, mean_10, primals_242, [80], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_694 = mean_10 = primals_242 = None
    getitem_583: "f32[8, 480, 1, 1]" = convolution_backward_49[0]
    getitem_584: "f32[80, 480, 1, 1]" = convolution_backward_49[1]
    getitem_585: "f32[80]" = convolution_backward_49[2];  convolution_backward_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_6: "f32[8, 480, 14, 14]" = torch.ops.aten.expand.default(getitem_583, [8, 480, 14, 14]);  getitem_583 = None
    div_6: "f32[8, 480, 14, 14]" = torch.ops.aten.div.Scalar(expand_6, 196);  expand_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_329: "f32[8, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_689, div_6);  mul_689 = div_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_80: "f32[8, 480, 14, 14]" = torch.ops.aten.sigmoid.default(clone_31)
    sub_148: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(full_default_14, sigmoid_80)
    mul_695: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(clone_31, sub_148);  clone_31 = sub_148 = None
    add_330: "f32[8, 480, 14, 14]" = torch.ops.aten.add.Scalar(mul_695, 1);  mul_695 = None
    mul_696: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_80, add_330);  sigmoid_80 = add_330 = None
    mul_697: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_329, mul_696);  add_329 = mul_696 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_42: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_697, [0, 2, 3])
    sub_149: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(cat_29, unsqueeze_438);  cat_29 = unsqueeze_438 = None
    mul_698: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_697, sub_149)
    sum_43: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_698, [0, 2, 3]);  mul_698 = None
    mul_699: "f32[480]" = torch.ops.aten.mul.Tensor(sum_42, 0.0006377551020408163)
    unsqueeze_439: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_699, 0);  mul_699 = None
    unsqueeze_440: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_439, 2);  unsqueeze_439 = None
    unsqueeze_441: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_440, 3);  unsqueeze_440 = None
    mul_700: "f32[480]" = torch.ops.aten.mul.Tensor(sum_43, 0.0006377551020408163)
    mul_701: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_121, squeeze_121)
    mul_702: "f32[480]" = torch.ops.aten.mul.Tensor(mul_700, mul_701);  mul_700 = mul_701 = None
    unsqueeze_442: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_702, 0);  mul_702 = None
    unsqueeze_443: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_442, 2);  unsqueeze_442 = None
    unsqueeze_444: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_443, 3);  unsqueeze_443 = None
    mul_703: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_121, primals_81);  primals_81 = None
    unsqueeze_445: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_703, 0);  mul_703 = None
    unsqueeze_446: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_445, 2);  unsqueeze_445 = None
    unsqueeze_447: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_446, 3);  unsqueeze_446 = None
    mul_704: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_149, unsqueeze_444);  sub_149 = unsqueeze_444 = None
    sub_151: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(mul_697, mul_704);  mul_697 = mul_704 = None
    sub_152: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(sub_151, unsqueeze_441);  sub_151 = unsqueeze_441 = None
    mul_705: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_152, unsqueeze_447);  sub_152 = unsqueeze_447 = None
    mul_706: "f32[480]" = torch.ops.aten.mul.Tensor(sum_43, squeeze_121);  sum_43 = squeeze_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    slice_33: "f32[8, 120, 14, 14]" = torch.ops.aten.slice.Tensor(mul_705, 1, 0, 120)
    slice_34: "f32[8, 120, 14, 14]" = torch.ops.aten.slice.Tensor(mul_705, 1, 120, 240)
    slice_35: "f32[8, 120, 14, 14]" = torch.ops.aten.slice.Tensor(mul_705, 1, 240, 360)
    slice_36: "f32[8, 120, 14, 14]" = torch.ops.aten.slice.Tensor(mul_705, 1, 360, 480);  mul_705 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_backward_50 = torch.ops.aten.convolution_backward.default(slice_36, getitem_287, primals_241, [0], [1, 1], [4, 4], [1, 1], False, [0, 0], 120, [True, True, False]);  slice_36 = getitem_287 = primals_241 = None
    getitem_586: "f32[8, 120, 14, 14]" = convolution_backward_50[0]
    getitem_587: "f32[120, 1, 9, 9]" = convolution_backward_50[1];  convolution_backward_50 = None
    convolution_backward_51 = torch.ops.aten.convolution_backward.default(slice_35, getitem_282, primals_240, [0], [1, 1], [3, 3], [1, 1], False, [0, 0], 120, [True, True, False]);  slice_35 = getitem_282 = primals_240 = None
    getitem_589: "f32[8, 120, 14, 14]" = convolution_backward_51[0]
    getitem_590: "f32[120, 1, 7, 7]" = convolution_backward_51[1];  convolution_backward_51 = None
    convolution_backward_52 = torch.ops.aten.convolution_backward.default(slice_34, getitem_277, primals_239, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 120, [True, True, False]);  slice_34 = getitem_277 = primals_239 = None
    getitem_592: "f32[8, 120, 14, 14]" = convolution_backward_52[0]
    getitem_593: "f32[120, 1, 5, 5]" = convolution_backward_52[1];  convolution_backward_52 = None
    convolution_backward_53 = torch.ops.aten.convolution_backward.default(slice_33, getitem_272, primals_238, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 120, [True, True, False]);  slice_33 = getitem_272 = primals_238 = None
    getitem_595: "f32[8, 120, 14, 14]" = convolution_backward_53[0]
    getitem_596: "f32[120, 1, 3, 3]" = convolution_backward_53[1];  convolution_backward_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    cat_52: "f32[8, 480, 14, 14]" = torch.ops.aten.cat.default([getitem_595, getitem_592, getitem_589, getitem_586], 1);  getitem_595 = getitem_592 = getitem_589 = getitem_586 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_709: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(cat_52, mul_708);  cat_52 = mul_708 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_44: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_709, [0, 2, 3])
    sub_154: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(cat_28, unsqueeze_450);  cat_28 = unsqueeze_450 = None
    mul_710: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_709, sub_154)
    sum_45: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_710, [0, 2, 3]);  mul_710 = None
    mul_711: "f32[480]" = torch.ops.aten.mul.Tensor(sum_44, 0.0006377551020408163)
    unsqueeze_451: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_711, 0);  mul_711 = None
    unsqueeze_452: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_451, 2);  unsqueeze_451 = None
    unsqueeze_453: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_452, 3);  unsqueeze_452 = None
    mul_712: "f32[480]" = torch.ops.aten.mul.Tensor(sum_45, 0.0006377551020408163)
    mul_713: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_118, squeeze_118)
    mul_714: "f32[480]" = torch.ops.aten.mul.Tensor(mul_712, mul_713);  mul_712 = mul_713 = None
    unsqueeze_454: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_714, 0);  mul_714 = None
    unsqueeze_455: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_454, 2);  unsqueeze_454 = None
    unsqueeze_456: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_455, 3);  unsqueeze_455 = None
    mul_715: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_118, primals_79);  primals_79 = None
    unsqueeze_457: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_715, 0);  mul_715 = None
    unsqueeze_458: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_457, 2);  unsqueeze_457 = None
    unsqueeze_459: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_458, 3);  unsqueeze_458 = None
    mul_716: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_154, unsqueeze_456);  sub_154 = unsqueeze_456 = None
    sub_156: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(mul_709, mul_716);  mul_709 = mul_716 = None
    sub_157: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(sub_156, unsqueeze_453);  sub_156 = unsqueeze_453 = None
    mul_717: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_157, unsqueeze_459);  sub_157 = unsqueeze_459 = None
    mul_718: "f32[480]" = torch.ops.aten.mul.Tensor(sum_45, squeeze_118);  sum_45 = squeeze_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    slice_37: "f32[8, 240, 14, 14]" = torch.ops.aten.slice.Tensor(mul_717, 1, 0, 240)
    slice_38: "f32[8, 240, 14, 14]" = torch.ops.aten.slice.Tensor(mul_717, 1, 240, 480);  mul_717 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_backward_54 = torch.ops.aten.convolution_backward.default(slice_38, getitem_265, primals_237, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_38 = getitem_265 = primals_237 = None
    getitem_598: "f32[8, 80, 14, 14]" = convolution_backward_54[0]
    getitem_599: "f32[240, 80, 1, 1]" = convolution_backward_54[1];  convolution_backward_54 = None
    convolution_backward_55 = torch.ops.aten.convolution_backward.default(slice_37, getitem_264, primals_236, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_37 = getitem_264 = primals_236 = None
    getitem_601: "f32[8, 80, 14, 14]" = convolution_backward_55[0]
    getitem_602: "f32[240, 80, 1, 1]" = convolution_backward_55[1];  convolution_backward_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    cat_53: "f32[8, 160, 14, 14]" = torch.ops.aten.cat.default([getitem_601, getitem_598], 1);  getitem_601 = getitem_598 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    add_332: "f32[8, 160, 14, 14]" = torch.ops.aten.add.Tensor(add_327, cat_53);  add_327 = cat_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_46: "f32[160]" = torch.ops.aten.sum.dim_IntList(add_332, [0, 2, 3])
    sub_158: "f32[8, 160, 14, 14]" = torch.ops.aten.sub.Tensor(cat_27, unsqueeze_462);  cat_27 = unsqueeze_462 = None
    mul_719: "f32[8, 160, 14, 14]" = torch.ops.aten.mul.Tensor(add_332, sub_158)
    sum_47: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_719, [0, 2, 3]);  mul_719 = None
    mul_720: "f32[160]" = torch.ops.aten.mul.Tensor(sum_46, 0.0006377551020408163)
    unsqueeze_463: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_720, 0);  mul_720 = None
    unsqueeze_464: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_463, 2);  unsqueeze_463 = None
    unsqueeze_465: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_464, 3);  unsqueeze_464 = None
    mul_721: "f32[160]" = torch.ops.aten.mul.Tensor(sum_47, 0.0006377551020408163)
    mul_722: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_115, squeeze_115)
    mul_723: "f32[160]" = torch.ops.aten.mul.Tensor(mul_721, mul_722);  mul_721 = mul_722 = None
    unsqueeze_466: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_723, 0);  mul_723 = None
    unsqueeze_467: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_466, 2);  unsqueeze_466 = None
    unsqueeze_468: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_467, 3);  unsqueeze_467 = None
    mul_724: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_115, primals_77);  primals_77 = None
    unsqueeze_469: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_724, 0);  mul_724 = None
    unsqueeze_470: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_469, 2);  unsqueeze_469 = None
    unsqueeze_471: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_470, 3);  unsqueeze_470 = None
    mul_725: "f32[8, 160, 14, 14]" = torch.ops.aten.mul.Tensor(sub_158, unsqueeze_468);  sub_158 = unsqueeze_468 = None
    sub_160: "f32[8, 160, 14, 14]" = torch.ops.aten.sub.Tensor(add_332, mul_725);  mul_725 = None
    sub_161: "f32[8, 160, 14, 14]" = torch.ops.aten.sub.Tensor(sub_160, unsqueeze_465);  sub_160 = unsqueeze_465 = None
    mul_726: "f32[8, 160, 14, 14]" = torch.ops.aten.mul.Tensor(sub_161, unsqueeze_471);  sub_161 = unsqueeze_471 = None
    mul_727: "f32[160]" = torch.ops.aten.mul.Tensor(sum_47, squeeze_115);  sum_47 = squeeze_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    slice_39: "f32[8, 80, 14, 14]" = torch.ops.aten.slice.Tensor(mul_726, 1, 0, 80)
    slice_40: "f32[8, 80, 14, 14]" = torch.ops.aten.slice.Tensor(mul_726, 1, 80, 160);  mul_726 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_backward_56 = torch.ops.aten.convolution_backward.default(slice_40, getitem_261, primals_235, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_40 = getitem_261 = primals_235 = None
    getitem_604: "f32[8, 240, 14, 14]" = convolution_backward_56[0]
    getitem_605: "f32[80, 240, 1, 1]" = convolution_backward_56[1];  convolution_backward_56 = None
    convolution_backward_57 = torch.ops.aten.convolution_backward.default(slice_39, getitem_260, primals_234, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_39 = getitem_260 = primals_234 = None
    getitem_607: "f32[8, 240, 14, 14]" = convolution_backward_57[0]
    getitem_608: "f32[80, 240, 1, 1]" = convolution_backward_57[1];  convolution_backward_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    cat_54: "f32[8, 480, 14, 14]" = torch.ops.aten.cat.default([getitem_607, getitem_604], 1);  getitem_607 = getitem_604 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_728: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(cat_54, mul_303);  mul_303 = None
    mul_729: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(cat_54, sigmoid_39);  cat_54 = sigmoid_39 = None
    sum_48: "f32[8, 480, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_728, [2, 3], True);  mul_728 = None
    alias_32: "f32[8, 480, 1, 1]" = torch.ops.aten.alias.default(alias_15);  alias_15 = None
    sub_162: "f32[8, 480, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_32)
    mul_730: "f32[8, 480, 1, 1]" = torch.ops.aten.mul.Tensor(alias_32, sub_162);  alias_32 = sub_162 = None
    mul_731: "f32[8, 480, 1, 1]" = torch.ops.aten.mul.Tensor(sum_48, mul_730);  sum_48 = mul_730 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_58 = torch.ops.aten.convolution_backward.default(mul_731, mul_304, primals_232, [480], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_731 = mul_304 = primals_232 = None
    getitem_610: "f32[8, 80, 1, 1]" = convolution_backward_58[0]
    getitem_611: "f32[480, 80, 1, 1]" = convolution_backward_58[1]
    getitem_612: "f32[480]" = convolution_backward_58[2];  convolution_backward_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_82: "f32[8, 80, 1, 1]" = torch.ops.aten.sigmoid.default(clone_29)
    sub_163: "f32[8, 80, 1, 1]" = torch.ops.aten.sub.Tensor(full_default_10, sigmoid_82);  full_default_10 = None
    mul_732: "f32[8, 80, 1, 1]" = torch.ops.aten.mul.Tensor(clone_29, sub_163);  clone_29 = sub_163 = None
    add_333: "f32[8, 80, 1, 1]" = torch.ops.aten.add.Scalar(mul_732, 1);  mul_732 = None
    mul_733: "f32[8, 80, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_82, add_333);  sigmoid_82 = add_333 = None
    mul_734: "f32[8, 80, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_610, mul_733);  getitem_610 = mul_733 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_59 = torch.ops.aten.convolution_backward.default(mul_734, mean_9, primals_230, [80], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_734 = mean_9 = primals_230 = None
    getitem_613: "f32[8, 480, 1, 1]" = convolution_backward_59[0]
    getitem_614: "f32[80, 480, 1, 1]" = convolution_backward_59[1]
    getitem_615: "f32[80]" = convolution_backward_59[2];  convolution_backward_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_7: "f32[8, 480, 14, 14]" = torch.ops.aten.expand.default(getitem_613, [8, 480, 14, 14]);  getitem_613 = None
    div_7: "f32[8, 480, 14, 14]" = torch.ops.aten.div.Scalar(expand_7, 196);  expand_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_334: "f32[8, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_729, div_7);  mul_729 = div_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_83: "f32[8, 480, 14, 14]" = torch.ops.aten.sigmoid.default(clone_28)
    sub_164: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(full_default_14, sigmoid_83);  full_default_14 = None
    mul_735: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(clone_28, sub_164);  clone_28 = sub_164 = None
    add_335: "f32[8, 480, 14, 14]" = torch.ops.aten.add.Scalar(mul_735, 1);  mul_735 = None
    mul_736: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_83, add_335);  sigmoid_83 = add_335 = None
    mul_737: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_334, mul_736);  add_334 = mul_736 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_49: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_737, [0, 2, 3])
    sub_165: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(cat_26, unsqueeze_474);  cat_26 = unsqueeze_474 = None
    mul_738: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_737, sub_165)
    sum_50: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_738, [0, 2, 3]);  mul_738 = None
    mul_739: "f32[480]" = torch.ops.aten.mul.Tensor(sum_49, 0.0006377551020408163)
    unsqueeze_475: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_739, 0);  mul_739 = None
    unsqueeze_476: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_475, 2);  unsqueeze_475 = None
    unsqueeze_477: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_476, 3);  unsqueeze_476 = None
    mul_740: "f32[480]" = torch.ops.aten.mul.Tensor(sum_50, 0.0006377551020408163)
    mul_741: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_112, squeeze_112)
    mul_742: "f32[480]" = torch.ops.aten.mul.Tensor(mul_740, mul_741);  mul_740 = mul_741 = None
    unsqueeze_478: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_742, 0);  mul_742 = None
    unsqueeze_479: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_478, 2);  unsqueeze_478 = None
    unsqueeze_480: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_479, 3);  unsqueeze_479 = None
    mul_743: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_112, primals_75);  primals_75 = None
    unsqueeze_481: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_743, 0);  mul_743 = None
    unsqueeze_482: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_481, 2);  unsqueeze_481 = None
    unsqueeze_483: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_482, 3);  unsqueeze_482 = None
    mul_744: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_165, unsqueeze_480);  sub_165 = unsqueeze_480 = None
    sub_167: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(mul_737, mul_744);  mul_737 = mul_744 = None
    sub_168: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(sub_167, unsqueeze_477);  sub_167 = unsqueeze_477 = None
    mul_745: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_168, unsqueeze_483);  sub_168 = unsqueeze_483 = None
    mul_746: "f32[480]" = torch.ops.aten.mul.Tensor(sum_50, squeeze_112);  sum_50 = squeeze_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    slice_41: "f32[8, 120, 14, 14]" = torch.ops.aten.slice.Tensor(mul_745, 1, 0, 120)
    slice_42: "f32[8, 120, 14, 14]" = torch.ops.aten.slice.Tensor(mul_745, 1, 120, 240)
    slice_43: "f32[8, 120, 14, 14]" = torch.ops.aten.slice.Tensor(mul_745, 1, 240, 360)
    slice_44: "f32[8, 120, 14, 14]" = torch.ops.aten.slice.Tensor(mul_745, 1, 360, 480);  mul_745 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_backward_60 = torch.ops.aten.convolution_backward.default(slice_44, getitem_257, primals_229, [0], [1, 1], [4, 4], [1, 1], False, [0, 0], 120, [True, True, False]);  slice_44 = getitem_257 = primals_229 = None
    getitem_616: "f32[8, 120, 14, 14]" = convolution_backward_60[0]
    getitem_617: "f32[120, 1, 9, 9]" = convolution_backward_60[1];  convolution_backward_60 = None
    convolution_backward_61 = torch.ops.aten.convolution_backward.default(slice_43, getitem_252, primals_228, [0], [1, 1], [3, 3], [1, 1], False, [0, 0], 120, [True, True, False]);  slice_43 = getitem_252 = primals_228 = None
    getitem_619: "f32[8, 120, 14, 14]" = convolution_backward_61[0]
    getitem_620: "f32[120, 1, 7, 7]" = convolution_backward_61[1];  convolution_backward_61 = None
    convolution_backward_62 = torch.ops.aten.convolution_backward.default(slice_42, getitem_247, primals_227, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 120, [True, True, False]);  slice_42 = getitem_247 = primals_227 = None
    getitem_622: "f32[8, 120, 14, 14]" = convolution_backward_62[0]
    getitem_623: "f32[120, 1, 5, 5]" = convolution_backward_62[1];  convolution_backward_62 = None
    convolution_backward_63 = torch.ops.aten.convolution_backward.default(slice_41, getitem_242, primals_226, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 120, [True, True, False]);  slice_41 = getitem_242 = primals_226 = None
    getitem_625: "f32[8, 120, 14, 14]" = convolution_backward_63[0]
    getitem_626: "f32[120, 1, 3, 3]" = convolution_backward_63[1];  convolution_backward_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    cat_55: "f32[8, 480, 14, 14]" = torch.ops.aten.cat.default([getitem_625, getitem_622, getitem_619, getitem_616], 1);  getitem_625 = getitem_622 = getitem_619 = getitem_616 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_749: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(cat_55, mul_748);  cat_55 = mul_748 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_51: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_749, [0, 2, 3])
    sub_170: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(cat_25, unsqueeze_486);  cat_25 = unsqueeze_486 = None
    mul_750: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_749, sub_170)
    sum_52: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_750, [0, 2, 3]);  mul_750 = None
    mul_751: "f32[480]" = torch.ops.aten.mul.Tensor(sum_51, 0.0006377551020408163)
    unsqueeze_487: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_751, 0);  mul_751 = None
    unsqueeze_488: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_487, 2);  unsqueeze_487 = None
    unsqueeze_489: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_488, 3);  unsqueeze_488 = None
    mul_752: "f32[480]" = torch.ops.aten.mul.Tensor(sum_52, 0.0006377551020408163)
    mul_753: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_109, squeeze_109)
    mul_754: "f32[480]" = torch.ops.aten.mul.Tensor(mul_752, mul_753);  mul_752 = mul_753 = None
    unsqueeze_490: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_754, 0);  mul_754 = None
    unsqueeze_491: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_490, 2);  unsqueeze_490 = None
    unsqueeze_492: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_491, 3);  unsqueeze_491 = None
    mul_755: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_109, primals_73);  primals_73 = None
    unsqueeze_493: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_755, 0);  mul_755 = None
    unsqueeze_494: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_493, 2);  unsqueeze_493 = None
    unsqueeze_495: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_494, 3);  unsqueeze_494 = None
    mul_756: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_170, unsqueeze_492);  sub_170 = unsqueeze_492 = None
    sub_172: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(mul_749, mul_756);  mul_749 = mul_756 = None
    sub_173: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(sub_172, unsqueeze_489);  sub_172 = unsqueeze_489 = None
    mul_757: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_173, unsqueeze_495);  sub_173 = unsqueeze_495 = None
    mul_758: "f32[480]" = torch.ops.aten.mul.Tensor(sum_52, squeeze_109);  sum_52 = squeeze_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    slice_45: "f32[8, 240, 14, 14]" = torch.ops.aten.slice.Tensor(mul_757, 1, 0, 240)
    slice_46: "f32[8, 240, 14, 14]" = torch.ops.aten.slice.Tensor(mul_757, 1, 240, 480);  mul_757 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_backward_64 = torch.ops.aten.convolution_backward.default(slice_46, getitem_235, primals_225, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_46 = getitem_235 = primals_225 = None
    getitem_628: "f32[8, 80, 14, 14]" = convolution_backward_64[0]
    getitem_629: "f32[240, 80, 1, 1]" = convolution_backward_64[1];  convolution_backward_64 = None
    convolution_backward_65 = torch.ops.aten.convolution_backward.default(slice_45, getitem_234, primals_224, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_45 = getitem_234 = primals_224 = None
    getitem_631: "f32[8, 80, 14, 14]" = convolution_backward_65[0]
    getitem_632: "f32[240, 80, 1, 1]" = convolution_backward_65[1];  convolution_backward_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    cat_56: "f32[8, 160, 14, 14]" = torch.ops.aten.cat.default([getitem_631, getitem_628], 1);  getitem_631 = getitem_628 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    add_337: "f32[8, 160, 14, 14]" = torch.ops.aten.add.Tensor(add_332, cat_56);  add_332 = cat_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_53: "f32[160]" = torch.ops.aten.sum.dim_IntList(add_337, [0, 2, 3])
    sub_174: "f32[8, 160, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_88, unsqueeze_498);  convolution_88 = unsqueeze_498 = None
    mul_759: "f32[8, 160, 14, 14]" = torch.ops.aten.mul.Tensor(add_337, sub_174)
    sum_54: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_759, [0, 2, 3]);  mul_759 = None
    mul_760: "f32[160]" = torch.ops.aten.mul.Tensor(sum_53, 0.0006377551020408163)
    unsqueeze_499: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_760, 0);  mul_760 = None
    unsqueeze_500: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_499, 2);  unsqueeze_499 = None
    unsqueeze_501: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_500, 3);  unsqueeze_500 = None
    mul_761: "f32[160]" = torch.ops.aten.mul.Tensor(sum_54, 0.0006377551020408163)
    mul_762: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_106, squeeze_106)
    mul_763: "f32[160]" = torch.ops.aten.mul.Tensor(mul_761, mul_762);  mul_761 = mul_762 = None
    unsqueeze_502: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_763, 0);  mul_763 = None
    unsqueeze_503: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_502, 2);  unsqueeze_502 = None
    unsqueeze_504: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_503, 3);  unsqueeze_503 = None
    mul_764: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_106, primals_71);  primals_71 = None
    unsqueeze_505: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_764, 0);  mul_764 = None
    unsqueeze_506: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_505, 2);  unsqueeze_505 = None
    unsqueeze_507: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_506, 3);  unsqueeze_506 = None
    mul_765: "f32[8, 160, 14, 14]" = torch.ops.aten.mul.Tensor(sub_174, unsqueeze_504);  sub_174 = unsqueeze_504 = None
    sub_176: "f32[8, 160, 14, 14]" = torch.ops.aten.sub.Tensor(add_337, mul_765);  add_337 = mul_765 = None
    sub_177: "f32[8, 160, 14, 14]" = torch.ops.aten.sub.Tensor(sub_176, unsqueeze_501);  sub_176 = unsqueeze_501 = None
    mul_766: "f32[8, 160, 14, 14]" = torch.ops.aten.mul.Tensor(sub_177, unsqueeze_507);  sub_177 = unsqueeze_507 = None
    mul_767: "f32[160]" = torch.ops.aten.mul.Tensor(sum_54, squeeze_106);  sum_54 = squeeze_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_66 = torch.ops.aten.convolution_backward.default(mul_766, mul_280, primals_223, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_766 = mul_280 = primals_223 = None
    getitem_634: "f32[8, 624, 14, 14]" = convolution_backward_66[0]
    getitem_635: "f32[160, 624, 1, 1]" = convolution_backward_66[1];  convolution_backward_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_768: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_634, mul_278);  mul_278 = None
    mul_769: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_634, sigmoid_35);  getitem_634 = sigmoid_35 = None
    sum_55: "f32[8, 624, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_768, [2, 3], True);  mul_768 = None
    alias_33: "f32[8, 624, 1, 1]" = torch.ops.aten.alias.default(alias_14);  alias_14 = None
    sub_178: "f32[8, 624, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_33)
    mul_770: "f32[8, 624, 1, 1]" = torch.ops.aten.mul.Tensor(alias_33, sub_178);  alias_33 = sub_178 = None
    mul_771: "f32[8, 624, 1, 1]" = torch.ops.aten.mul.Tensor(sum_55, mul_770);  sum_55 = mul_770 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_67 = torch.ops.aten.convolution_backward.default(mul_771, mul_279, primals_221, [624], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_771 = mul_279 = primals_221 = None
    getitem_637: "f32[8, 52, 1, 1]" = convolution_backward_67[0]
    getitem_638: "f32[624, 52, 1, 1]" = convolution_backward_67[1]
    getitem_639: "f32[624]" = convolution_backward_67[2];  convolution_backward_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_85: "f32[8, 52, 1, 1]" = torch.ops.aten.sigmoid.default(clone_26)
    full_default_22: "f32[8, 52, 1, 1]" = torch.ops.aten.full.default([8, 52, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_179: "f32[8, 52, 1, 1]" = torch.ops.aten.sub.Tensor(full_default_22, sigmoid_85);  full_default_22 = None
    mul_772: "f32[8, 52, 1, 1]" = torch.ops.aten.mul.Tensor(clone_26, sub_179);  clone_26 = sub_179 = None
    add_338: "f32[8, 52, 1, 1]" = torch.ops.aten.add.Scalar(mul_772, 1);  mul_772 = None
    mul_773: "f32[8, 52, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_85, add_338);  sigmoid_85 = add_338 = None
    mul_774: "f32[8, 52, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_637, mul_773);  getitem_637 = mul_773 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_68 = torch.ops.aten.convolution_backward.default(mul_774, mean_8, primals_219, [52], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_774 = mean_8 = primals_219 = None
    getitem_640: "f32[8, 624, 1, 1]" = convolution_backward_68[0]
    getitem_641: "f32[52, 624, 1, 1]" = convolution_backward_68[1]
    getitem_642: "f32[52]" = convolution_backward_68[2];  convolution_backward_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_8: "f32[8, 624, 14, 14]" = torch.ops.aten.expand.default(getitem_640, [8, 624, 14, 14]);  getitem_640 = None
    div_8: "f32[8, 624, 14, 14]" = torch.ops.aten.div.Scalar(expand_8, 196);  expand_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_339: "f32[8, 624, 14, 14]" = torch.ops.aten.add.Tensor(mul_769, div_8);  mul_769 = div_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_86: "f32[8, 624, 14, 14]" = torch.ops.aten.sigmoid.default(clone_25)
    full_default_23: "f32[8, 624, 14, 14]" = torch.ops.aten.full.default([8, 624, 14, 14], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_180: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(full_default_23, sigmoid_86)
    mul_775: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(clone_25, sub_180);  clone_25 = sub_180 = None
    add_340: "f32[8, 624, 14, 14]" = torch.ops.aten.add.Scalar(mul_775, 1);  mul_775 = None
    mul_776: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_86, add_340);  sigmoid_86 = add_340 = None
    mul_777: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(add_339, mul_776);  add_339 = mul_776 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_56: "f32[624]" = torch.ops.aten.sum.dim_IntList(mul_777, [0, 2, 3])
    sub_181: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_85, unsqueeze_510);  convolution_85 = unsqueeze_510 = None
    mul_778: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(mul_777, sub_181)
    sum_57: "f32[624]" = torch.ops.aten.sum.dim_IntList(mul_778, [0, 2, 3]);  mul_778 = None
    mul_779: "f32[624]" = torch.ops.aten.mul.Tensor(sum_56, 0.0006377551020408163)
    unsqueeze_511: "f32[1, 624]" = torch.ops.aten.unsqueeze.default(mul_779, 0);  mul_779 = None
    unsqueeze_512: "f32[1, 624, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_511, 2);  unsqueeze_511 = None
    unsqueeze_513: "f32[1, 624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_512, 3);  unsqueeze_512 = None
    mul_780: "f32[624]" = torch.ops.aten.mul.Tensor(sum_57, 0.0006377551020408163)
    mul_781: "f32[624]" = torch.ops.aten.mul.Tensor(squeeze_103, squeeze_103)
    mul_782: "f32[624]" = torch.ops.aten.mul.Tensor(mul_780, mul_781);  mul_780 = mul_781 = None
    unsqueeze_514: "f32[1, 624]" = torch.ops.aten.unsqueeze.default(mul_782, 0);  mul_782 = None
    unsqueeze_515: "f32[1, 624, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_514, 2);  unsqueeze_514 = None
    unsqueeze_516: "f32[1, 624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_515, 3);  unsqueeze_515 = None
    mul_783: "f32[624]" = torch.ops.aten.mul.Tensor(squeeze_103, primals_69);  primals_69 = None
    unsqueeze_517: "f32[1, 624]" = torch.ops.aten.unsqueeze.default(mul_783, 0);  mul_783 = None
    unsqueeze_518: "f32[1, 624, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_517, 2);  unsqueeze_517 = None
    unsqueeze_519: "f32[1, 624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_518, 3);  unsqueeze_518 = None
    mul_784: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(sub_181, unsqueeze_516);  sub_181 = unsqueeze_516 = None
    sub_183: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(mul_777, mul_784);  mul_777 = mul_784 = None
    sub_184: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(sub_183, unsqueeze_513);  sub_183 = unsqueeze_513 = None
    mul_785: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(sub_184, unsqueeze_519);  sub_184 = unsqueeze_519 = None
    mul_786: "f32[624]" = torch.ops.aten.mul.Tensor(sum_57, squeeze_103);  sum_57 = squeeze_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_69 = torch.ops.aten.convolution_backward.default(mul_785, mul_270, primals_218, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 624, [True, True, False]);  mul_785 = mul_270 = primals_218 = None
    getitem_643: "f32[8, 624, 14, 14]" = convolution_backward_69[0]
    getitem_644: "f32[624, 1, 3, 3]" = convolution_backward_69[1];  convolution_backward_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_789: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_643, mul_788);  getitem_643 = mul_788 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_58: "f32[624]" = torch.ops.aten.sum.dim_IntList(mul_789, [0, 2, 3])
    sub_186: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_84, unsqueeze_522);  convolution_84 = unsqueeze_522 = None
    mul_790: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(mul_789, sub_186)
    sum_59: "f32[624]" = torch.ops.aten.sum.dim_IntList(mul_790, [0, 2, 3]);  mul_790 = None
    mul_791: "f32[624]" = torch.ops.aten.mul.Tensor(sum_58, 0.0006377551020408163)
    unsqueeze_523: "f32[1, 624]" = torch.ops.aten.unsqueeze.default(mul_791, 0);  mul_791 = None
    unsqueeze_524: "f32[1, 624, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_523, 2);  unsqueeze_523 = None
    unsqueeze_525: "f32[1, 624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_524, 3);  unsqueeze_524 = None
    mul_792: "f32[624]" = torch.ops.aten.mul.Tensor(sum_59, 0.0006377551020408163)
    mul_793: "f32[624]" = torch.ops.aten.mul.Tensor(squeeze_100, squeeze_100)
    mul_794: "f32[624]" = torch.ops.aten.mul.Tensor(mul_792, mul_793);  mul_792 = mul_793 = None
    unsqueeze_526: "f32[1, 624]" = torch.ops.aten.unsqueeze.default(mul_794, 0);  mul_794 = None
    unsqueeze_527: "f32[1, 624, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_526, 2);  unsqueeze_526 = None
    unsqueeze_528: "f32[1, 624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_527, 3);  unsqueeze_527 = None
    mul_795: "f32[624]" = torch.ops.aten.mul.Tensor(squeeze_100, primals_67);  primals_67 = None
    unsqueeze_529: "f32[1, 624]" = torch.ops.aten.unsqueeze.default(mul_795, 0);  mul_795 = None
    unsqueeze_530: "f32[1, 624, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_529, 2);  unsqueeze_529 = None
    unsqueeze_531: "f32[1, 624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_530, 3);  unsqueeze_530 = None
    mul_796: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(sub_186, unsqueeze_528);  sub_186 = unsqueeze_528 = None
    sub_188: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(mul_789, mul_796);  mul_789 = mul_796 = None
    sub_189: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(sub_188, unsqueeze_525);  sub_188 = unsqueeze_525 = None
    mul_797: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(sub_189, unsqueeze_531);  sub_189 = unsqueeze_531 = None
    mul_798: "f32[624]" = torch.ops.aten.mul.Tensor(sum_59, squeeze_100);  sum_59 = squeeze_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_70 = torch.ops.aten.convolution_backward.default(mul_797, add_172, primals_217, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_797 = add_172 = primals_217 = None
    getitem_646: "f32[8, 104, 14, 14]" = convolution_backward_70[0]
    getitem_647: "f32[624, 104, 1, 1]" = convolution_backward_70[1];  convolution_backward_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_60: "f32[104]" = torch.ops.aten.sum.dim_IntList(getitem_646, [0, 2, 3])
    sub_190: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(cat_24, unsqueeze_534);  cat_24 = unsqueeze_534 = None
    mul_799: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_646, sub_190)
    sum_61: "f32[104]" = torch.ops.aten.sum.dim_IntList(mul_799, [0, 2, 3]);  mul_799 = None
    mul_800: "f32[104]" = torch.ops.aten.mul.Tensor(sum_60, 0.0006377551020408163)
    unsqueeze_535: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_800, 0);  mul_800 = None
    unsqueeze_536: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_535, 2);  unsqueeze_535 = None
    unsqueeze_537: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_536, 3);  unsqueeze_536 = None
    mul_801: "f32[104]" = torch.ops.aten.mul.Tensor(sum_61, 0.0006377551020408163)
    mul_802: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_97, squeeze_97)
    mul_803: "f32[104]" = torch.ops.aten.mul.Tensor(mul_801, mul_802);  mul_801 = mul_802 = None
    unsqueeze_538: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_803, 0);  mul_803 = None
    unsqueeze_539: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_538, 2);  unsqueeze_538 = None
    unsqueeze_540: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_539, 3);  unsqueeze_539 = None
    mul_804: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_97, primals_65);  primals_65 = None
    unsqueeze_541: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_804, 0);  mul_804 = None
    unsqueeze_542: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_541, 2);  unsqueeze_541 = None
    unsqueeze_543: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_542, 3);  unsqueeze_542 = None
    mul_805: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_190, unsqueeze_540);  sub_190 = unsqueeze_540 = None
    sub_192: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(getitem_646, mul_805);  mul_805 = None
    sub_193: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(sub_192, unsqueeze_537);  sub_192 = unsqueeze_537 = None
    mul_806: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_193, unsqueeze_543);  sub_193 = unsqueeze_543 = None
    mul_807: "f32[104]" = torch.ops.aten.mul.Tensor(sum_61, squeeze_97);  sum_61 = squeeze_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    slice_47: "f32[8, 52, 14, 14]" = torch.ops.aten.slice.Tensor(mul_806, 1, 0, 52)
    slice_48: "f32[8, 52, 14, 14]" = torch.ops.aten.slice.Tensor(mul_806, 1, 52, 104);  mul_806 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_backward_71 = torch.ops.aten.convolution_backward.default(slice_48, getitem_225, primals_216, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_48 = getitem_225 = primals_216 = None
    getitem_649: "f32[8, 312, 14, 14]" = convolution_backward_71[0]
    getitem_650: "f32[52, 312, 1, 1]" = convolution_backward_71[1];  convolution_backward_71 = None
    convolution_backward_72 = torch.ops.aten.convolution_backward.default(slice_47, getitem_224, primals_215, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_47 = getitem_224 = primals_215 = None
    getitem_652: "f32[8, 312, 14, 14]" = convolution_backward_72[0]
    getitem_653: "f32[52, 312, 1, 1]" = convolution_backward_72[1];  convolution_backward_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    cat_57: "f32[8, 624, 14, 14]" = torch.ops.aten.cat.default([getitem_652, getitem_649], 1);  getitem_652 = getitem_649 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_808: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(cat_57, mul_253);  mul_253 = None
    mul_809: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(cat_57, sigmoid_31);  cat_57 = sigmoid_31 = None
    sum_62: "f32[8, 624, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_808, [2, 3], True);  mul_808 = None
    alias_34: "f32[8, 624, 1, 1]" = torch.ops.aten.alias.default(alias_13);  alias_13 = None
    sub_194: "f32[8, 624, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_34)
    mul_810: "f32[8, 624, 1, 1]" = torch.ops.aten.mul.Tensor(alias_34, sub_194);  alias_34 = sub_194 = None
    mul_811: "f32[8, 624, 1, 1]" = torch.ops.aten.mul.Tensor(sum_62, mul_810);  sum_62 = mul_810 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_73 = torch.ops.aten.convolution_backward.default(mul_811, mul_254, primals_213, [624], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_811 = mul_254 = primals_213 = None
    getitem_655: "f32[8, 26, 1, 1]" = convolution_backward_73[0]
    getitem_656: "f32[624, 26, 1, 1]" = convolution_backward_73[1]
    getitem_657: "f32[624]" = convolution_backward_73[2];  convolution_backward_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_88: "f32[8, 26, 1, 1]" = torch.ops.aten.sigmoid.default(clone_23)
    full_default_25: "f32[8, 26, 1, 1]" = torch.ops.aten.full.default([8, 26, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_195: "f32[8, 26, 1, 1]" = torch.ops.aten.sub.Tensor(full_default_25, sigmoid_88)
    mul_812: "f32[8, 26, 1, 1]" = torch.ops.aten.mul.Tensor(clone_23, sub_195);  clone_23 = sub_195 = None
    add_342: "f32[8, 26, 1, 1]" = torch.ops.aten.add.Scalar(mul_812, 1);  mul_812 = None
    mul_813: "f32[8, 26, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_88, add_342);  sigmoid_88 = add_342 = None
    mul_814: "f32[8, 26, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_655, mul_813);  getitem_655 = mul_813 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_74 = torch.ops.aten.convolution_backward.default(mul_814, mean_7, primals_211, [26], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_814 = mean_7 = primals_211 = None
    getitem_658: "f32[8, 624, 1, 1]" = convolution_backward_74[0]
    getitem_659: "f32[26, 624, 1, 1]" = convolution_backward_74[1]
    getitem_660: "f32[26]" = convolution_backward_74[2];  convolution_backward_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_9: "f32[8, 624, 14, 14]" = torch.ops.aten.expand.default(getitem_658, [8, 624, 14, 14]);  getitem_658 = None
    div_9: "f32[8, 624, 14, 14]" = torch.ops.aten.div.Scalar(expand_9, 196);  expand_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_343: "f32[8, 624, 14, 14]" = torch.ops.aten.add.Tensor(mul_809, div_9);  mul_809 = div_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_89: "f32[8, 624, 14, 14]" = torch.ops.aten.sigmoid.default(clone_22)
    sub_196: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(full_default_23, sigmoid_89)
    mul_815: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(clone_22, sub_196);  clone_22 = sub_196 = None
    add_344: "f32[8, 624, 14, 14]" = torch.ops.aten.add.Scalar(mul_815, 1);  mul_815 = None
    mul_816: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_89, add_344);  sigmoid_89 = add_344 = None
    mul_817: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(add_343, mul_816);  add_343 = mul_816 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_63: "f32[624]" = torch.ops.aten.sum.dim_IntList(mul_817, [0, 2, 3])
    sub_197: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(cat_23, unsqueeze_546);  cat_23 = unsqueeze_546 = None
    mul_818: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(mul_817, sub_197)
    sum_64: "f32[624]" = torch.ops.aten.sum.dim_IntList(mul_818, [0, 2, 3]);  mul_818 = None
    mul_819: "f32[624]" = torch.ops.aten.mul.Tensor(sum_63, 0.0006377551020408163)
    unsqueeze_547: "f32[1, 624]" = torch.ops.aten.unsqueeze.default(mul_819, 0);  mul_819 = None
    unsqueeze_548: "f32[1, 624, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_547, 2);  unsqueeze_547 = None
    unsqueeze_549: "f32[1, 624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_548, 3);  unsqueeze_548 = None
    mul_820: "f32[624]" = torch.ops.aten.mul.Tensor(sum_64, 0.0006377551020408163)
    mul_821: "f32[624]" = torch.ops.aten.mul.Tensor(squeeze_94, squeeze_94)
    mul_822: "f32[624]" = torch.ops.aten.mul.Tensor(mul_820, mul_821);  mul_820 = mul_821 = None
    unsqueeze_550: "f32[1, 624]" = torch.ops.aten.unsqueeze.default(mul_822, 0);  mul_822 = None
    unsqueeze_551: "f32[1, 624, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_550, 2);  unsqueeze_550 = None
    unsqueeze_552: "f32[1, 624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_551, 3);  unsqueeze_551 = None
    mul_823: "f32[624]" = torch.ops.aten.mul.Tensor(squeeze_94, primals_63);  primals_63 = None
    unsqueeze_553: "f32[1, 624]" = torch.ops.aten.unsqueeze.default(mul_823, 0);  mul_823 = None
    unsqueeze_554: "f32[1, 624, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_553, 2);  unsqueeze_553 = None
    unsqueeze_555: "f32[1, 624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_554, 3);  unsqueeze_554 = None
    mul_824: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(sub_197, unsqueeze_552);  sub_197 = unsqueeze_552 = None
    sub_199: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(mul_817, mul_824);  mul_817 = mul_824 = None
    sub_200: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(sub_199, unsqueeze_549);  sub_199 = unsqueeze_549 = None
    mul_825: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(sub_200, unsqueeze_555);  sub_200 = unsqueeze_555 = None
    mul_826: "f32[624]" = torch.ops.aten.mul.Tensor(sum_64, squeeze_94);  sum_64 = squeeze_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    slice_49: "f32[8, 156, 14, 14]" = torch.ops.aten.slice.Tensor(mul_825, 1, 0, 156)
    slice_50: "f32[8, 156, 14, 14]" = torch.ops.aten.slice.Tensor(mul_825, 1, 156, 312)
    slice_51: "f32[8, 156, 14, 14]" = torch.ops.aten.slice.Tensor(mul_825, 1, 312, 468)
    slice_52: "f32[8, 156, 14, 14]" = torch.ops.aten.slice.Tensor(mul_825, 1, 468, 624);  mul_825 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_backward_75 = torch.ops.aten.convolution_backward.default(slice_52, getitem_221, primals_210, [0], [1, 1], [4, 4], [1, 1], False, [0, 0], 156, [True, True, False]);  slice_52 = getitem_221 = primals_210 = None
    getitem_661: "f32[8, 156, 14, 14]" = convolution_backward_75[0]
    getitem_662: "f32[156, 1, 9, 9]" = convolution_backward_75[1];  convolution_backward_75 = None
    convolution_backward_76 = torch.ops.aten.convolution_backward.default(slice_51, getitem_216, primals_209, [0], [1, 1], [3, 3], [1, 1], False, [0, 0], 156, [True, True, False]);  slice_51 = getitem_216 = primals_209 = None
    getitem_664: "f32[8, 156, 14, 14]" = convolution_backward_76[0]
    getitem_665: "f32[156, 1, 7, 7]" = convolution_backward_76[1];  convolution_backward_76 = None
    convolution_backward_77 = torch.ops.aten.convolution_backward.default(slice_50, getitem_211, primals_208, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 156, [True, True, False]);  slice_50 = getitem_211 = primals_208 = None
    getitem_667: "f32[8, 156, 14, 14]" = convolution_backward_77[0]
    getitem_668: "f32[156, 1, 5, 5]" = convolution_backward_77[1];  convolution_backward_77 = None
    convolution_backward_78 = torch.ops.aten.convolution_backward.default(slice_49, getitem_206, primals_207, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 156, [True, True, False]);  slice_49 = getitem_206 = primals_207 = None
    getitem_670: "f32[8, 156, 14, 14]" = convolution_backward_78[0]
    getitem_671: "f32[156, 1, 3, 3]" = convolution_backward_78[1];  convolution_backward_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    cat_58: "f32[8, 624, 14, 14]" = torch.ops.aten.cat.default([getitem_670, getitem_667, getitem_664, getitem_661], 1);  getitem_670 = getitem_667 = getitem_664 = getitem_661 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_829: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(cat_58, mul_828);  cat_58 = mul_828 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_65: "f32[624]" = torch.ops.aten.sum.dim_IntList(mul_829, [0, 2, 3])
    sub_202: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(cat_22, unsqueeze_558);  cat_22 = unsqueeze_558 = None
    mul_830: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(mul_829, sub_202)
    sum_66: "f32[624]" = torch.ops.aten.sum.dim_IntList(mul_830, [0, 2, 3]);  mul_830 = None
    mul_831: "f32[624]" = torch.ops.aten.mul.Tensor(sum_65, 0.0006377551020408163)
    unsqueeze_559: "f32[1, 624]" = torch.ops.aten.unsqueeze.default(mul_831, 0);  mul_831 = None
    unsqueeze_560: "f32[1, 624, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_559, 2);  unsqueeze_559 = None
    unsqueeze_561: "f32[1, 624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_560, 3);  unsqueeze_560 = None
    mul_832: "f32[624]" = torch.ops.aten.mul.Tensor(sum_66, 0.0006377551020408163)
    mul_833: "f32[624]" = torch.ops.aten.mul.Tensor(squeeze_91, squeeze_91)
    mul_834: "f32[624]" = torch.ops.aten.mul.Tensor(mul_832, mul_833);  mul_832 = mul_833 = None
    unsqueeze_562: "f32[1, 624]" = torch.ops.aten.unsqueeze.default(mul_834, 0);  mul_834 = None
    unsqueeze_563: "f32[1, 624, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_562, 2);  unsqueeze_562 = None
    unsqueeze_564: "f32[1, 624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_563, 3);  unsqueeze_563 = None
    mul_835: "f32[624]" = torch.ops.aten.mul.Tensor(squeeze_91, primals_61);  primals_61 = None
    unsqueeze_565: "f32[1, 624]" = torch.ops.aten.unsqueeze.default(mul_835, 0);  mul_835 = None
    unsqueeze_566: "f32[1, 624, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_565, 2);  unsqueeze_565 = None
    unsqueeze_567: "f32[1, 624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_566, 3);  unsqueeze_566 = None
    mul_836: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(sub_202, unsqueeze_564);  sub_202 = unsqueeze_564 = None
    sub_204: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(mul_829, mul_836);  mul_829 = mul_836 = None
    sub_205: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(sub_204, unsqueeze_561);  sub_204 = unsqueeze_561 = None
    mul_837: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(sub_205, unsqueeze_567);  sub_205 = unsqueeze_567 = None
    mul_838: "f32[624]" = torch.ops.aten.mul.Tensor(sum_66, squeeze_91);  sum_66 = squeeze_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    slice_53: "f32[8, 312, 14, 14]" = torch.ops.aten.slice.Tensor(mul_837, 1, 0, 312)
    slice_54: "f32[8, 312, 14, 14]" = torch.ops.aten.slice.Tensor(mul_837, 1, 312, 624);  mul_837 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_backward_79 = torch.ops.aten.convolution_backward.default(slice_54, getitem_199, primals_206, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_54 = getitem_199 = primals_206 = None
    getitem_673: "f32[8, 52, 14, 14]" = convolution_backward_79[0]
    getitem_674: "f32[312, 52, 1, 1]" = convolution_backward_79[1];  convolution_backward_79 = None
    convolution_backward_80 = torch.ops.aten.convolution_backward.default(slice_53, getitem_198, primals_205, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_53 = getitem_198 = primals_205 = None
    getitem_676: "f32[8, 52, 14, 14]" = convolution_backward_80[0]
    getitem_677: "f32[312, 52, 1, 1]" = convolution_backward_80[1];  convolution_backward_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    cat_59: "f32[8, 104, 14, 14]" = torch.ops.aten.cat.default([getitem_676, getitem_673], 1);  getitem_676 = getitem_673 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    add_346: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(getitem_646, cat_59);  getitem_646 = cat_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_67: "f32[104]" = torch.ops.aten.sum.dim_IntList(add_346, [0, 2, 3])
    sub_206: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(cat_21, unsqueeze_570);  cat_21 = unsqueeze_570 = None
    mul_839: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(add_346, sub_206)
    sum_68: "f32[104]" = torch.ops.aten.sum.dim_IntList(mul_839, [0, 2, 3]);  mul_839 = None
    mul_840: "f32[104]" = torch.ops.aten.mul.Tensor(sum_67, 0.0006377551020408163)
    unsqueeze_571: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_840, 0);  mul_840 = None
    unsqueeze_572: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_571, 2);  unsqueeze_571 = None
    unsqueeze_573: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_572, 3);  unsqueeze_572 = None
    mul_841: "f32[104]" = torch.ops.aten.mul.Tensor(sum_68, 0.0006377551020408163)
    mul_842: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_88, squeeze_88)
    mul_843: "f32[104]" = torch.ops.aten.mul.Tensor(mul_841, mul_842);  mul_841 = mul_842 = None
    unsqueeze_574: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_843, 0);  mul_843 = None
    unsqueeze_575: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_574, 2);  unsqueeze_574 = None
    unsqueeze_576: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_575, 3);  unsqueeze_575 = None
    mul_844: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_88, primals_59);  primals_59 = None
    unsqueeze_577: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_844, 0);  mul_844 = None
    unsqueeze_578: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_577, 2);  unsqueeze_577 = None
    unsqueeze_579: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_578, 3);  unsqueeze_578 = None
    mul_845: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_206, unsqueeze_576);  sub_206 = unsqueeze_576 = None
    sub_208: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(add_346, mul_845);  mul_845 = None
    sub_209: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(sub_208, unsqueeze_573);  sub_208 = unsqueeze_573 = None
    mul_846: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_209, unsqueeze_579);  sub_209 = unsqueeze_579 = None
    mul_847: "f32[104]" = torch.ops.aten.mul.Tensor(sum_68, squeeze_88);  sum_68 = squeeze_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    slice_55: "f32[8, 52, 14, 14]" = torch.ops.aten.slice.Tensor(mul_846, 1, 0, 52)
    slice_56: "f32[8, 52, 14, 14]" = torch.ops.aten.slice.Tensor(mul_846, 1, 52, 104);  mul_846 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_backward_81 = torch.ops.aten.convolution_backward.default(slice_56, getitem_195, primals_204, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_56 = getitem_195 = primals_204 = None
    getitem_679: "f32[8, 312, 14, 14]" = convolution_backward_81[0]
    getitem_680: "f32[52, 312, 1, 1]" = convolution_backward_81[1];  convolution_backward_81 = None
    convolution_backward_82 = torch.ops.aten.convolution_backward.default(slice_55, getitem_194, primals_203, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_55 = getitem_194 = primals_203 = None
    getitem_682: "f32[8, 312, 14, 14]" = convolution_backward_82[0]
    getitem_683: "f32[52, 312, 1, 1]" = convolution_backward_82[1];  convolution_backward_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    cat_60: "f32[8, 624, 14, 14]" = torch.ops.aten.cat.default([getitem_682, getitem_679], 1);  getitem_682 = getitem_679 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_848: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(cat_60, mul_228);  mul_228 = None
    mul_849: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(cat_60, sigmoid_27);  cat_60 = sigmoid_27 = None
    sum_69: "f32[8, 624, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_848, [2, 3], True);  mul_848 = None
    alias_35: "f32[8, 624, 1, 1]" = torch.ops.aten.alias.default(alias_12);  alias_12 = None
    sub_210: "f32[8, 624, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_35)
    mul_850: "f32[8, 624, 1, 1]" = torch.ops.aten.mul.Tensor(alias_35, sub_210);  alias_35 = sub_210 = None
    mul_851: "f32[8, 624, 1, 1]" = torch.ops.aten.mul.Tensor(sum_69, mul_850);  sum_69 = mul_850 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_83 = torch.ops.aten.convolution_backward.default(mul_851, mul_229, primals_201, [624], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_851 = mul_229 = primals_201 = None
    getitem_685: "f32[8, 26, 1, 1]" = convolution_backward_83[0]
    getitem_686: "f32[624, 26, 1, 1]" = convolution_backward_83[1]
    getitem_687: "f32[624]" = convolution_backward_83[2];  convolution_backward_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_91: "f32[8, 26, 1, 1]" = torch.ops.aten.sigmoid.default(clone_20)
    sub_211: "f32[8, 26, 1, 1]" = torch.ops.aten.sub.Tensor(full_default_25, sigmoid_91)
    mul_852: "f32[8, 26, 1, 1]" = torch.ops.aten.mul.Tensor(clone_20, sub_211);  clone_20 = sub_211 = None
    add_347: "f32[8, 26, 1, 1]" = torch.ops.aten.add.Scalar(mul_852, 1);  mul_852 = None
    mul_853: "f32[8, 26, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_91, add_347);  sigmoid_91 = add_347 = None
    mul_854: "f32[8, 26, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_685, mul_853);  getitem_685 = mul_853 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_84 = torch.ops.aten.convolution_backward.default(mul_854, mean_6, primals_199, [26], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_854 = mean_6 = primals_199 = None
    getitem_688: "f32[8, 624, 1, 1]" = convolution_backward_84[0]
    getitem_689: "f32[26, 624, 1, 1]" = convolution_backward_84[1]
    getitem_690: "f32[26]" = convolution_backward_84[2];  convolution_backward_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_10: "f32[8, 624, 14, 14]" = torch.ops.aten.expand.default(getitem_688, [8, 624, 14, 14]);  getitem_688 = None
    div_10: "f32[8, 624, 14, 14]" = torch.ops.aten.div.Scalar(expand_10, 196);  expand_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_348: "f32[8, 624, 14, 14]" = torch.ops.aten.add.Tensor(mul_849, div_10);  mul_849 = div_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_92: "f32[8, 624, 14, 14]" = torch.ops.aten.sigmoid.default(clone_19)
    sub_212: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(full_default_23, sigmoid_92)
    mul_855: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(clone_19, sub_212);  clone_19 = sub_212 = None
    add_349: "f32[8, 624, 14, 14]" = torch.ops.aten.add.Scalar(mul_855, 1);  mul_855 = None
    mul_856: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_92, add_349);  sigmoid_92 = add_349 = None
    mul_857: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(add_348, mul_856);  add_348 = mul_856 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_70: "f32[624]" = torch.ops.aten.sum.dim_IntList(mul_857, [0, 2, 3])
    sub_213: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(cat_20, unsqueeze_582);  cat_20 = unsqueeze_582 = None
    mul_858: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(mul_857, sub_213)
    sum_71: "f32[624]" = torch.ops.aten.sum.dim_IntList(mul_858, [0, 2, 3]);  mul_858 = None
    mul_859: "f32[624]" = torch.ops.aten.mul.Tensor(sum_70, 0.0006377551020408163)
    unsqueeze_583: "f32[1, 624]" = torch.ops.aten.unsqueeze.default(mul_859, 0);  mul_859 = None
    unsqueeze_584: "f32[1, 624, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_583, 2);  unsqueeze_583 = None
    unsqueeze_585: "f32[1, 624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_584, 3);  unsqueeze_584 = None
    mul_860: "f32[624]" = torch.ops.aten.mul.Tensor(sum_71, 0.0006377551020408163)
    mul_861: "f32[624]" = torch.ops.aten.mul.Tensor(squeeze_85, squeeze_85)
    mul_862: "f32[624]" = torch.ops.aten.mul.Tensor(mul_860, mul_861);  mul_860 = mul_861 = None
    unsqueeze_586: "f32[1, 624]" = torch.ops.aten.unsqueeze.default(mul_862, 0);  mul_862 = None
    unsqueeze_587: "f32[1, 624, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_586, 2);  unsqueeze_586 = None
    unsqueeze_588: "f32[1, 624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_587, 3);  unsqueeze_587 = None
    mul_863: "f32[624]" = torch.ops.aten.mul.Tensor(squeeze_85, primals_57);  primals_57 = None
    unsqueeze_589: "f32[1, 624]" = torch.ops.aten.unsqueeze.default(mul_863, 0);  mul_863 = None
    unsqueeze_590: "f32[1, 624, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_589, 2);  unsqueeze_589 = None
    unsqueeze_591: "f32[1, 624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_590, 3);  unsqueeze_590 = None
    mul_864: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(sub_213, unsqueeze_588);  sub_213 = unsqueeze_588 = None
    sub_215: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(mul_857, mul_864);  mul_857 = mul_864 = None
    sub_216: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(sub_215, unsqueeze_585);  sub_215 = unsqueeze_585 = None
    mul_865: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(sub_216, unsqueeze_591);  sub_216 = unsqueeze_591 = None
    mul_866: "f32[624]" = torch.ops.aten.mul.Tensor(sum_71, squeeze_85);  sum_71 = squeeze_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    slice_57: "f32[8, 156, 14, 14]" = torch.ops.aten.slice.Tensor(mul_865, 1, 0, 156)
    slice_58: "f32[8, 156, 14, 14]" = torch.ops.aten.slice.Tensor(mul_865, 1, 156, 312)
    slice_59: "f32[8, 156, 14, 14]" = torch.ops.aten.slice.Tensor(mul_865, 1, 312, 468)
    slice_60: "f32[8, 156, 14, 14]" = torch.ops.aten.slice.Tensor(mul_865, 1, 468, 624);  mul_865 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_backward_85 = torch.ops.aten.convolution_backward.default(slice_60, getitem_191, primals_198, [0], [1, 1], [4, 4], [1, 1], False, [0, 0], 156, [True, True, False]);  slice_60 = getitem_191 = primals_198 = None
    getitem_691: "f32[8, 156, 14, 14]" = convolution_backward_85[0]
    getitem_692: "f32[156, 1, 9, 9]" = convolution_backward_85[1];  convolution_backward_85 = None
    convolution_backward_86 = torch.ops.aten.convolution_backward.default(slice_59, getitem_186, primals_197, [0], [1, 1], [3, 3], [1, 1], False, [0, 0], 156, [True, True, False]);  slice_59 = getitem_186 = primals_197 = None
    getitem_694: "f32[8, 156, 14, 14]" = convolution_backward_86[0]
    getitem_695: "f32[156, 1, 7, 7]" = convolution_backward_86[1];  convolution_backward_86 = None
    convolution_backward_87 = torch.ops.aten.convolution_backward.default(slice_58, getitem_181, primals_196, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 156, [True, True, False]);  slice_58 = getitem_181 = primals_196 = None
    getitem_697: "f32[8, 156, 14, 14]" = convolution_backward_87[0]
    getitem_698: "f32[156, 1, 5, 5]" = convolution_backward_87[1];  convolution_backward_87 = None
    convolution_backward_88 = torch.ops.aten.convolution_backward.default(slice_57, getitem_176, primals_195, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 156, [True, True, False]);  slice_57 = getitem_176 = primals_195 = None
    getitem_700: "f32[8, 156, 14, 14]" = convolution_backward_88[0]
    getitem_701: "f32[156, 1, 3, 3]" = convolution_backward_88[1];  convolution_backward_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    cat_61: "f32[8, 624, 14, 14]" = torch.ops.aten.cat.default([getitem_700, getitem_697, getitem_694, getitem_691], 1);  getitem_700 = getitem_697 = getitem_694 = getitem_691 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_869: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(cat_61, mul_868);  cat_61 = mul_868 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_72: "f32[624]" = torch.ops.aten.sum.dim_IntList(mul_869, [0, 2, 3])
    sub_218: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(cat_19, unsqueeze_594);  cat_19 = unsqueeze_594 = None
    mul_870: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(mul_869, sub_218)
    sum_73: "f32[624]" = torch.ops.aten.sum.dim_IntList(mul_870, [0, 2, 3]);  mul_870 = None
    mul_871: "f32[624]" = torch.ops.aten.mul.Tensor(sum_72, 0.0006377551020408163)
    unsqueeze_595: "f32[1, 624]" = torch.ops.aten.unsqueeze.default(mul_871, 0);  mul_871 = None
    unsqueeze_596: "f32[1, 624, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_595, 2);  unsqueeze_595 = None
    unsqueeze_597: "f32[1, 624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_596, 3);  unsqueeze_596 = None
    mul_872: "f32[624]" = torch.ops.aten.mul.Tensor(sum_73, 0.0006377551020408163)
    mul_873: "f32[624]" = torch.ops.aten.mul.Tensor(squeeze_82, squeeze_82)
    mul_874: "f32[624]" = torch.ops.aten.mul.Tensor(mul_872, mul_873);  mul_872 = mul_873 = None
    unsqueeze_598: "f32[1, 624]" = torch.ops.aten.unsqueeze.default(mul_874, 0);  mul_874 = None
    unsqueeze_599: "f32[1, 624, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_598, 2);  unsqueeze_598 = None
    unsqueeze_600: "f32[1, 624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_599, 3);  unsqueeze_599 = None
    mul_875: "f32[624]" = torch.ops.aten.mul.Tensor(squeeze_82, primals_55);  primals_55 = None
    unsqueeze_601: "f32[1, 624]" = torch.ops.aten.unsqueeze.default(mul_875, 0);  mul_875 = None
    unsqueeze_602: "f32[1, 624, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_601, 2);  unsqueeze_601 = None
    unsqueeze_603: "f32[1, 624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_602, 3);  unsqueeze_602 = None
    mul_876: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(sub_218, unsqueeze_600);  sub_218 = unsqueeze_600 = None
    sub_220: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(mul_869, mul_876);  mul_869 = mul_876 = None
    sub_221: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(sub_220, unsqueeze_597);  sub_220 = unsqueeze_597 = None
    mul_877: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(sub_221, unsqueeze_603);  sub_221 = unsqueeze_603 = None
    mul_878: "f32[624]" = torch.ops.aten.mul.Tensor(sum_73, squeeze_82);  sum_73 = squeeze_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    slice_61: "f32[8, 312, 14, 14]" = torch.ops.aten.slice.Tensor(mul_877, 1, 0, 312)
    slice_62: "f32[8, 312, 14, 14]" = torch.ops.aten.slice.Tensor(mul_877, 1, 312, 624);  mul_877 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_backward_89 = torch.ops.aten.convolution_backward.default(slice_62, getitem_169, primals_194, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_62 = getitem_169 = primals_194 = None
    getitem_703: "f32[8, 52, 14, 14]" = convolution_backward_89[0]
    getitem_704: "f32[312, 52, 1, 1]" = convolution_backward_89[1];  convolution_backward_89 = None
    convolution_backward_90 = torch.ops.aten.convolution_backward.default(slice_61, getitem_168, primals_193, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_61 = getitem_168 = primals_193 = None
    getitem_706: "f32[8, 52, 14, 14]" = convolution_backward_90[0]
    getitem_707: "f32[312, 52, 1, 1]" = convolution_backward_90[1];  convolution_backward_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    cat_62: "f32[8, 104, 14, 14]" = torch.ops.aten.cat.default([getitem_706, getitem_703], 1);  getitem_706 = getitem_703 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    add_351: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(add_346, cat_62);  add_346 = cat_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_74: "f32[104]" = torch.ops.aten.sum.dim_IntList(add_351, [0, 2, 3])
    sub_222: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(cat_18, unsqueeze_606);  cat_18 = unsqueeze_606 = None
    mul_879: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(add_351, sub_222)
    sum_75: "f32[104]" = torch.ops.aten.sum.dim_IntList(mul_879, [0, 2, 3]);  mul_879 = None
    mul_880: "f32[104]" = torch.ops.aten.mul.Tensor(sum_74, 0.0006377551020408163)
    unsqueeze_607: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_880, 0);  mul_880 = None
    unsqueeze_608: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_607, 2);  unsqueeze_607 = None
    unsqueeze_609: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_608, 3);  unsqueeze_608 = None
    mul_881: "f32[104]" = torch.ops.aten.mul.Tensor(sum_75, 0.0006377551020408163)
    mul_882: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_79, squeeze_79)
    mul_883: "f32[104]" = torch.ops.aten.mul.Tensor(mul_881, mul_882);  mul_881 = mul_882 = None
    unsqueeze_610: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_883, 0);  mul_883 = None
    unsqueeze_611: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_610, 2);  unsqueeze_610 = None
    unsqueeze_612: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_611, 3);  unsqueeze_611 = None
    mul_884: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_79, primals_53);  primals_53 = None
    unsqueeze_613: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_884, 0);  mul_884 = None
    unsqueeze_614: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_613, 2);  unsqueeze_613 = None
    unsqueeze_615: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_614, 3);  unsqueeze_614 = None
    mul_885: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_222, unsqueeze_612);  sub_222 = unsqueeze_612 = None
    sub_224: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(add_351, mul_885);  mul_885 = None
    sub_225: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(sub_224, unsqueeze_609);  sub_224 = unsqueeze_609 = None
    mul_886: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_225, unsqueeze_615);  sub_225 = unsqueeze_615 = None
    mul_887: "f32[104]" = torch.ops.aten.mul.Tensor(sum_75, squeeze_79);  sum_75 = squeeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    slice_63: "f32[8, 52, 14, 14]" = torch.ops.aten.slice.Tensor(mul_886, 1, 0, 52)
    slice_64: "f32[8, 52, 14, 14]" = torch.ops.aten.slice.Tensor(mul_886, 1, 52, 104);  mul_886 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_backward_91 = torch.ops.aten.convolution_backward.default(slice_64, getitem_165, primals_192, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_64 = getitem_165 = primals_192 = None
    getitem_709: "f32[8, 312, 14, 14]" = convolution_backward_91[0]
    getitem_710: "f32[52, 312, 1, 1]" = convolution_backward_91[1];  convolution_backward_91 = None
    convolution_backward_92 = torch.ops.aten.convolution_backward.default(slice_63, getitem_164, primals_191, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_63 = getitem_164 = primals_191 = None
    getitem_712: "f32[8, 312, 14, 14]" = convolution_backward_92[0]
    getitem_713: "f32[52, 312, 1, 1]" = convolution_backward_92[1];  convolution_backward_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    cat_63: "f32[8, 624, 14, 14]" = torch.ops.aten.cat.default([getitem_712, getitem_709], 1);  getitem_712 = getitem_709 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_888: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(cat_63, mul_203);  mul_203 = None
    mul_889: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(cat_63, sigmoid_23);  cat_63 = sigmoid_23 = None
    sum_76: "f32[8, 624, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_888, [2, 3], True);  mul_888 = None
    alias_36: "f32[8, 624, 1, 1]" = torch.ops.aten.alias.default(alias_11);  alias_11 = None
    sub_226: "f32[8, 624, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_36)
    mul_890: "f32[8, 624, 1, 1]" = torch.ops.aten.mul.Tensor(alias_36, sub_226);  alias_36 = sub_226 = None
    mul_891: "f32[8, 624, 1, 1]" = torch.ops.aten.mul.Tensor(sum_76, mul_890);  sum_76 = mul_890 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_93 = torch.ops.aten.convolution_backward.default(mul_891, mul_204, primals_189, [624], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_891 = mul_204 = primals_189 = None
    getitem_715: "f32[8, 26, 1, 1]" = convolution_backward_93[0]
    getitem_716: "f32[624, 26, 1, 1]" = convolution_backward_93[1]
    getitem_717: "f32[624]" = convolution_backward_93[2];  convolution_backward_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_94: "f32[8, 26, 1, 1]" = torch.ops.aten.sigmoid.default(clone_17)
    sub_227: "f32[8, 26, 1, 1]" = torch.ops.aten.sub.Tensor(full_default_25, sigmoid_94);  full_default_25 = None
    mul_892: "f32[8, 26, 1, 1]" = torch.ops.aten.mul.Tensor(clone_17, sub_227);  clone_17 = sub_227 = None
    add_352: "f32[8, 26, 1, 1]" = torch.ops.aten.add.Scalar(mul_892, 1);  mul_892 = None
    mul_893: "f32[8, 26, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_94, add_352);  sigmoid_94 = add_352 = None
    mul_894: "f32[8, 26, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_715, mul_893);  getitem_715 = mul_893 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_94 = torch.ops.aten.convolution_backward.default(mul_894, mean_5, primals_187, [26], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_894 = mean_5 = primals_187 = None
    getitem_718: "f32[8, 624, 1, 1]" = convolution_backward_94[0]
    getitem_719: "f32[26, 624, 1, 1]" = convolution_backward_94[1]
    getitem_720: "f32[26]" = convolution_backward_94[2];  convolution_backward_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_11: "f32[8, 624, 14, 14]" = torch.ops.aten.expand.default(getitem_718, [8, 624, 14, 14]);  getitem_718 = None
    div_11: "f32[8, 624, 14, 14]" = torch.ops.aten.div.Scalar(expand_11, 196);  expand_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_353: "f32[8, 624, 14, 14]" = torch.ops.aten.add.Tensor(mul_889, div_11);  mul_889 = div_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_95: "f32[8, 624, 14, 14]" = torch.ops.aten.sigmoid.default(clone_16)
    sub_228: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(full_default_23, sigmoid_95);  full_default_23 = None
    mul_895: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(clone_16, sub_228);  clone_16 = sub_228 = None
    add_354: "f32[8, 624, 14, 14]" = torch.ops.aten.add.Scalar(mul_895, 1);  mul_895 = None
    mul_896: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_95, add_354);  sigmoid_95 = add_354 = None
    mul_897: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(add_353, mul_896);  add_353 = mul_896 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_77: "f32[624]" = torch.ops.aten.sum.dim_IntList(mul_897, [0, 2, 3])
    sub_229: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(cat_17, unsqueeze_618);  cat_17 = unsqueeze_618 = None
    mul_898: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(mul_897, sub_229)
    sum_78: "f32[624]" = torch.ops.aten.sum.dim_IntList(mul_898, [0, 2, 3]);  mul_898 = None
    mul_899: "f32[624]" = torch.ops.aten.mul.Tensor(sum_77, 0.0006377551020408163)
    unsqueeze_619: "f32[1, 624]" = torch.ops.aten.unsqueeze.default(mul_899, 0);  mul_899 = None
    unsqueeze_620: "f32[1, 624, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_619, 2);  unsqueeze_619 = None
    unsqueeze_621: "f32[1, 624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_620, 3);  unsqueeze_620 = None
    mul_900: "f32[624]" = torch.ops.aten.mul.Tensor(sum_78, 0.0006377551020408163)
    mul_901: "f32[624]" = torch.ops.aten.mul.Tensor(squeeze_76, squeeze_76)
    mul_902: "f32[624]" = torch.ops.aten.mul.Tensor(mul_900, mul_901);  mul_900 = mul_901 = None
    unsqueeze_622: "f32[1, 624]" = torch.ops.aten.unsqueeze.default(mul_902, 0);  mul_902 = None
    unsqueeze_623: "f32[1, 624, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_622, 2);  unsqueeze_622 = None
    unsqueeze_624: "f32[1, 624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_623, 3);  unsqueeze_623 = None
    mul_903: "f32[624]" = torch.ops.aten.mul.Tensor(squeeze_76, primals_51);  primals_51 = None
    unsqueeze_625: "f32[1, 624]" = torch.ops.aten.unsqueeze.default(mul_903, 0);  mul_903 = None
    unsqueeze_626: "f32[1, 624, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_625, 2);  unsqueeze_625 = None
    unsqueeze_627: "f32[1, 624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_626, 3);  unsqueeze_626 = None
    mul_904: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(sub_229, unsqueeze_624);  sub_229 = unsqueeze_624 = None
    sub_231: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(mul_897, mul_904);  mul_897 = mul_904 = None
    sub_232: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(sub_231, unsqueeze_621);  sub_231 = unsqueeze_621 = None
    mul_905: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(sub_232, unsqueeze_627);  sub_232 = unsqueeze_627 = None
    mul_906: "f32[624]" = torch.ops.aten.mul.Tensor(sum_78, squeeze_76);  sum_78 = squeeze_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    slice_65: "f32[8, 156, 14, 14]" = torch.ops.aten.slice.Tensor(mul_905, 1, 0, 156)
    slice_66: "f32[8, 156, 14, 14]" = torch.ops.aten.slice.Tensor(mul_905, 1, 156, 312)
    slice_67: "f32[8, 156, 14, 14]" = torch.ops.aten.slice.Tensor(mul_905, 1, 312, 468)
    slice_68: "f32[8, 156, 14, 14]" = torch.ops.aten.slice.Tensor(mul_905, 1, 468, 624);  mul_905 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_backward_95 = torch.ops.aten.convolution_backward.default(slice_68, getitem_161, primals_186, [0], [1, 1], [4, 4], [1, 1], False, [0, 0], 156, [True, True, False]);  slice_68 = getitem_161 = primals_186 = None
    getitem_721: "f32[8, 156, 14, 14]" = convolution_backward_95[0]
    getitem_722: "f32[156, 1, 9, 9]" = convolution_backward_95[1];  convolution_backward_95 = None
    convolution_backward_96 = torch.ops.aten.convolution_backward.default(slice_67, getitem_156, primals_185, [0], [1, 1], [3, 3], [1, 1], False, [0, 0], 156, [True, True, False]);  slice_67 = getitem_156 = primals_185 = None
    getitem_724: "f32[8, 156, 14, 14]" = convolution_backward_96[0]
    getitem_725: "f32[156, 1, 7, 7]" = convolution_backward_96[1];  convolution_backward_96 = None
    convolution_backward_97 = torch.ops.aten.convolution_backward.default(slice_66, getitem_151, primals_184, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 156, [True, True, False]);  slice_66 = getitem_151 = primals_184 = None
    getitem_727: "f32[8, 156, 14, 14]" = convolution_backward_97[0]
    getitem_728: "f32[156, 1, 5, 5]" = convolution_backward_97[1];  convolution_backward_97 = None
    convolution_backward_98 = torch.ops.aten.convolution_backward.default(slice_65, getitem_146, primals_183, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 156, [True, True, False]);  slice_65 = getitem_146 = primals_183 = None
    getitem_730: "f32[8, 156, 14, 14]" = convolution_backward_98[0]
    getitem_731: "f32[156, 1, 3, 3]" = convolution_backward_98[1];  convolution_backward_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    cat_64: "f32[8, 624, 14, 14]" = torch.ops.aten.cat.default([getitem_730, getitem_727, getitem_724, getitem_721], 1);  getitem_730 = getitem_727 = getitem_724 = getitem_721 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_909: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(cat_64, mul_908);  cat_64 = mul_908 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_79: "f32[624]" = torch.ops.aten.sum.dim_IntList(mul_909, [0, 2, 3])
    sub_234: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(cat_16, unsqueeze_630);  cat_16 = unsqueeze_630 = None
    mul_910: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(mul_909, sub_234)
    sum_80: "f32[624]" = torch.ops.aten.sum.dim_IntList(mul_910, [0, 2, 3]);  mul_910 = None
    mul_911: "f32[624]" = torch.ops.aten.mul.Tensor(sum_79, 0.0006377551020408163)
    unsqueeze_631: "f32[1, 624]" = torch.ops.aten.unsqueeze.default(mul_911, 0);  mul_911 = None
    unsqueeze_632: "f32[1, 624, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_631, 2);  unsqueeze_631 = None
    unsqueeze_633: "f32[1, 624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_632, 3);  unsqueeze_632 = None
    mul_912: "f32[624]" = torch.ops.aten.mul.Tensor(sum_80, 0.0006377551020408163)
    mul_913: "f32[624]" = torch.ops.aten.mul.Tensor(squeeze_73, squeeze_73)
    mul_914: "f32[624]" = torch.ops.aten.mul.Tensor(mul_912, mul_913);  mul_912 = mul_913 = None
    unsqueeze_634: "f32[1, 624]" = torch.ops.aten.unsqueeze.default(mul_914, 0);  mul_914 = None
    unsqueeze_635: "f32[1, 624, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_634, 2);  unsqueeze_634 = None
    unsqueeze_636: "f32[1, 624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_635, 3);  unsqueeze_635 = None
    mul_915: "f32[624]" = torch.ops.aten.mul.Tensor(squeeze_73, primals_49);  primals_49 = None
    unsqueeze_637: "f32[1, 624]" = torch.ops.aten.unsqueeze.default(mul_915, 0);  mul_915 = None
    unsqueeze_638: "f32[1, 624, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_637, 2);  unsqueeze_637 = None
    unsqueeze_639: "f32[1, 624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_638, 3);  unsqueeze_638 = None
    mul_916: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(sub_234, unsqueeze_636);  sub_234 = unsqueeze_636 = None
    sub_236: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(mul_909, mul_916);  mul_909 = mul_916 = None
    sub_237: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(sub_236, unsqueeze_633);  sub_236 = unsqueeze_633 = None
    mul_917: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(sub_237, unsqueeze_639);  sub_237 = unsqueeze_639 = None
    mul_918: "f32[624]" = torch.ops.aten.mul.Tensor(sum_80, squeeze_73);  sum_80 = squeeze_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    slice_69: "f32[8, 312, 14, 14]" = torch.ops.aten.slice.Tensor(mul_917, 1, 0, 312)
    slice_70: "f32[8, 312, 14, 14]" = torch.ops.aten.slice.Tensor(mul_917, 1, 312, 624);  mul_917 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_backward_99 = torch.ops.aten.convolution_backward.default(slice_70, getitem_139, primals_182, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_70 = getitem_139 = primals_182 = None
    getitem_733: "f32[8, 52, 14, 14]" = convolution_backward_99[0]
    getitem_734: "f32[312, 52, 1, 1]" = convolution_backward_99[1];  convolution_backward_99 = None
    convolution_backward_100 = torch.ops.aten.convolution_backward.default(slice_69, getitem_138, primals_181, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_69 = getitem_138 = primals_181 = None
    getitem_736: "f32[8, 52, 14, 14]" = convolution_backward_100[0]
    getitem_737: "f32[312, 52, 1, 1]" = convolution_backward_100[1];  convolution_backward_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    cat_65: "f32[8, 104, 14, 14]" = torch.ops.aten.cat.default([getitem_736, getitem_733], 1);  getitem_736 = getitem_733 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    add_356: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(add_351, cat_65);  add_351 = cat_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_81: "f32[104]" = torch.ops.aten.sum.dim_IntList(add_356, [0, 2, 3])
    sub_238: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_53, unsqueeze_642);  convolution_53 = unsqueeze_642 = None
    mul_919: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(add_356, sub_238)
    sum_82: "f32[104]" = torch.ops.aten.sum.dim_IntList(mul_919, [0, 2, 3]);  mul_919 = None
    mul_920: "f32[104]" = torch.ops.aten.mul.Tensor(sum_81, 0.0006377551020408163)
    unsqueeze_643: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_920, 0);  mul_920 = None
    unsqueeze_644: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_643, 2);  unsqueeze_643 = None
    unsqueeze_645: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_644, 3);  unsqueeze_644 = None
    mul_921: "f32[104]" = torch.ops.aten.mul.Tensor(sum_82, 0.0006377551020408163)
    mul_922: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_70, squeeze_70)
    mul_923: "f32[104]" = torch.ops.aten.mul.Tensor(mul_921, mul_922);  mul_921 = mul_922 = None
    unsqueeze_646: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_923, 0);  mul_923 = None
    unsqueeze_647: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_646, 2);  unsqueeze_646 = None
    unsqueeze_648: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_647, 3);  unsqueeze_647 = None
    mul_924: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_70, primals_47);  primals_47 = None
    unsqueeze_649: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_924, 0);  mul_924 = None
    unsqueeze_650: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_649, 2);  unsqueeze_649 = None
    unsqueeze_651: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_650, 3);  unsqueeze_650 = None
    mul_925: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_238, unsqueeze_648);  sub_238 = unsqueeze_648 = None
    sub_240: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(add_356, mul_925);  add_356 = mul_925 = None
    sub_241: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(sub_240, unsqueeze_645);  sub_240 = unsqueeze_645 = None
    mul_926: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_241, unsqueeze_651);  sub_241 = unsqueeze_651 = None
    mul_927: "f32[104]" = torch.ops.aten.mul.Tensor(sum_82, squeeze_70);  sum_82 = squeeze_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_101 = torch.ops.aten.convolution_backward.default(mul_926, mul_180, primals_180, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_926 = mul_180 = primals_180 = None
    getitem_739: "f32[8, 336, 14, 14]" = convolution_backward_101[0]
    getitem_740: "f32[104, 336, 1, 1]" = convolution_backward_101[1];  convolution_backward_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_928: "f32[8, 336, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_739, mul_178);  mul_178 = None
    mul_929: "f32[8, 336, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_739, sigmoid_19);  getitem_739 = sigmoid_19 = None
    sum_83: "f32[8, 336, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_928, [2, 3], True);  mul_928 = None
    alias_37: "f32[8, 336, 1, 1]" = torch.ops.aten.alias.default(alias_10);  alias_10 = None
    sub_242: "f32[8, 336, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_37)
    mul_930: "f32[8, 336, 1, 1]" = torch.ops.aten.mul.Tensor(alias_37, sub_242);  alias_37 = sub_242 = None
    mul_931: "f32[8, 336, 1, 1]" = torch.ops.aten.mul.Tensor(sum_83, mul_930);  sum_83 = mul_930 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_102 = torch.ops.aten.convolution_backward.default(mul_931, mul_179, primals_178, [336], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_931 = mul_179 = primals_178 = None
    getitem_742: "f32[8, 14, 1, 1]" = convolution_backward_102[0]
    getitem_743: "f32[336, 14, 1, 1]" = convolution_backward_102[1]
    getitem_744: "f32[336]" = convolution_backward_102[2];  convolution_backward_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_97: "f32[8, 14, 1, 1]" = torch.ops.aten.sigmoid.default(clone_14)
    full_default_34: "f32[8, 14, 1, 1]" = torch.ops.aten.full.default([8, 14, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_243: "f32[8, 14, 1, 1]" = torch.ops.aten.sub.Tensor(full_default_34, sigmoid_97);  full_default_34 = None
    mul_932: "f32[8, 14, 1, 1]" = torch.ops.aten.mul.Tensor(clone_14, sub_243);  clone_14 = sub_243 = None
    add_357: "f32[8, 14, 1, 1]" = torch.ops.aten.add.Scalar(mul_932, 1);  mul_932 = None
    mul_933: "f32[8, 14, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_97, add_357);  sigmoid_97 = add_357 = None
    mul_934: "f32[8, 14, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_742, mul_933);  getitem_742 = mul_933 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_103 = torch.ops.aten.convolution_backward.default(mul_934, mean_4, primals_176, [14], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_934 = mean_4 = primals_176 = None
    getitem_745: "f32[8, 336, 1, 1]" = convolution_backward_103[0]
    getitem_746: "f32[14, 336, 1, 1]" = convolution_backward_103[1]
    getitem_747: "f32[14]" = convolution_backward_103[2];  convolution_backward_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_12: "f32[8, 336, 14, 14]" = torch.ops.aten.expand.default(getitem_745, [8, 336, 14, 14]);  getitem_745 = None
    div_12: "f32[8, 336, 14, 14]" = torch.ops.aten.div.Scalar(expand_12, 196);  expand_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_358: "f32[8, 336, 14, 14]" = torch.ops.aten.add.Tensor(mul_929, div_12);  mul_929 = div_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_98: "f32[8, 336, 14, 14]" = torch.ops.aten.sigmoid.default(clone_13)
    full_default_35: "f32[8, 336, 14, 14]" = torch.ops.aten.full.default([8, 336, 14, 14], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_244: "f32[8, 336, 14, 14]" = torch.ops.aten.sub.Tensor(full_default_35, sigmoid_98);  full_default_35 = None
    mul_935: "f32[8, 336, 14, 14]" = torch.ops.aten.mul.Tensor(clone_13, sub_244);  clone_13 = sub_244 = None
    add_359: "f32[8, 336, 14, 14]" = torch.ops.aten.add.Scalar(mul_935, 1);  mul_935 = None
    mul_936: "f32[8, 336, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_98, add_359);  sigmoid_98 = add_359 = None
    mul_937: "f32[8, 336, 14, 14]" = torch.ops.aten.mul.Tensor(add_358, mul_936);  add_358 = mul_936 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_84: "f32[336]" = torch.ops.aten.sum.dim_IntList(mul_937, [0, 2, 3])
    sub_245: "f32[8, 336, 14, 14]" = torch.ops.aten.sub.Tensor(cat_15, unsqueeze_654);  cat_15 = unsqueeze_654 = None
    mul_938: "f32[8, 336, 14, 14]" = torch.ops.aten.mul.Tensor(mul_937, sub_245)
    sum_85: "f32[336]" = torch.ops.aten.sum.dim_IntList(mul_938, [0, 2, 3]);  mul_938 = None
    mul_939: "f32[336]" = torch.ops.aten.mul.Tensor(sum_84, 0.0006377551020408163)
    unsqueeze_655: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(mul_939, 0);  mul_939 = None
    unsqueeze_656: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_655, 2);  unsqueeze_655 = None
    unsqueeze_657: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_656, 3);  unsqueeze_656 = None
    mul_940: "f32[336]" = torch.ops.aten.mul.Tensor(sum_85, 0.0006377551020408163)
    mul_941: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_67, squeeze_67)
    mul_942: "f32[336]" = torch.ops.aten.mul.Tensor(mul_940, mul_941);  mul_940 = mul_941 = None
    unsqueeze_658: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(mul_942, 0);  mul_942 = None
    unsqueeze_659: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_658, 2);  unsqueeze_658 = None
    unsqueeze_660: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_659, 3);  unsqueeze_659 = None
    mul_943: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_67, primals_45);  primals_45 = None
    unsqueeze_661: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(mul_943, 0);  mul_943 = None
    unsqueeze_662: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_661, 2);  unsqueeze_661 = None
    unsqueeze_663: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_662, 3);  unsqueeze_662 = None
    mul_944: "f32[8, 336, 14, 14]" = torch.ops.aten.mul.Tensor(sub_245, unsqueeze_660);  sub_245 = unsqueeze_660 = None
    sub_247: "f32[8, 336, 14, 14]" = torch.ops.aten.sub.Tensor(mul_937, mul_944);  mul_937 = mul_944 = None
    sub_248: "f32[8, 336, 14, 14]" = torch.ops.aten.sub.Tensor(sub_247, unsqueeze_657);  sub_247 = unsqueeze_657 = None
    mul_945: "f32[8, 336, 14, 14]" = torch.ops.aten.mul.Tensor(sub_248, unsqueeze_663);  sub_248 = unsqueeze_663 = None
    mul_946: "f32[336]" = torch.ops.aten.mul.Tensor(sum_85, squeeze_67);  sum_85 = squeeze_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    slice_71: "f32[8, 112, 14, 14]" = torch.ops.aten.slice.Tensor(mul_945, 1, 0, 112)
    slice_72: "f32[8, 112, 14, 14]" = torch.ops.aten.slice.Tensor(mul_945, 1, 112, 224)
    slice_73: "f32[8, 112, 14, 14]" = torch.ops.aten.slice.Tensor(mul_945, 1, 224, 336);  mul_945 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_backward_104 = torch.ops.aten.convolution_backward.default(slice_73, getitem_133, primals_175, [0], [2, 2], [3, 3], [1, 1], False, [0, 0], 112, [True, True, False]);  slice_73 = getitem_133 = primals_175 = None
    getitem_748: "f32[8, 112, 28, 28]" = convolution_backward_104[0]
    getitem_749: "f32[112, 1, 7, 7]" = convolution_backward_104[1];  convolution_backward_104 = None
    convolution_backward_105 = torch.ops.aten.convolution_backward.default(slice_72, getitem_129, primals_174, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 112, [True, True, False]);  slice_72 = getitem_129 = primals_174 = None
    getitem_751: "f32[8, 112, 28, 28]" = convolution_backward_105[0]
    getitem_752: "f32[112, 1, 5, 5]" = convolution_backward_105[1];  convolution_backward_105 = None
    convolution_backward_106 = torch.ops.aten.convolution_backward.default(slice_71, getitem_125, primals_173, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 112, [True, True, False]);  slice_71 = getitem_125 = primals_173 = None
    getitem_754: "f32[8, 112, 28, 28]" = convolution_backward_106[0]
    getitem_755: "f32[112, 1, 3, 3]" = convolution_backward_106[1];  convolution_backward_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    cat_66: "f32[8, 336, 28, 28]" = torch.ops.aten.cat.default([getitem_754, getitem_751, getitem_748], 1);  getitem_754 = getitem_751 = getitem_748 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    full_default_36: "f32[8, 336, 28, 28]" = torch.ops.aten.full.default([8, 336, 28, 28], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    mul_949: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(cat_66, mul_948);  cat_66 = mul_948 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_86: "f32[336]" = torch.ops.aten.sum.dim_IntList(mul_949, [0, 2, 3])
    sub_250: "f32[8, 336, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_47, unsqueeze_666);  convolution_47 = unsqueeze_666 = None
    mul_950: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(mul_949, sub_250)
    sum_87: "f32[336]" = torch.ops.aten.sum.dim_IntList(mul_950, [0, 2, 3]);  mul_950 = None
    mul_951: "f32[336]" = torch.ops.aten.mul.Tensor(sum_86, 0.00015943877551020407)
    unsqueeze_667: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(mul_951, 0);  mul_951 = None
    unsqueeze_668: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_667, 2);  unsqueeze_667 = None
    unsqueeze_669: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_668, 3);  unsqueeze_668 = None
    mul_952: "f32[336]" = torch.ops.aten.mul.Tensor(sum_87, 0.00015943877551020407)
    mul_953: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_64, squeeze_64)
    mul_954: "f32[336]" = torch.ops.aten.mul.Tensor(mul_952, mul_953);  mul_952 = mul_953 = None
    unsqueeze_670: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(mul_954, 0);  mul_954 = None
    unsqueeze_671: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_670, 2);  unsqueeze_670 = None
    unsqueeze_672: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_671, 3);  unsqueeze_671 = None
    mul_955: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_64, primals_43);  primals_43 = None
    unsqueeze_673: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(mul_955, 0);  mul_955 = None
    unsqueeze_674: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_673, 2);  unsqueeze_673 = None
    unsqueeze_675: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_674, 3);  unsqueeze_674 = None
    mul_956: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(sub_250, unsqueeze_672);  sub_250 = unsqueeze_672 = None
    sub_252: "f32[8, 336, 28, 28]" = torch.ops.aten.sub.Tensor(mul_949, mul_956);  mul_949 = mul_956 = None
    sub_253: "f32[8, 336, 28, 28]" = torch.ops.aten.sub.Tensor(sub_252, unsqueeze_669);  sub_252 = unsqueeze_669 = None
    mul_957: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(sub_253, unsqueeze_675);  sub_253 = unsqueeze_675 = None
    mul_958: "f32[336]" = torch.ops.aten.mul.Tensor(sum_87, squeeze_64);  sum_87 = squeeze_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_107 = torch.ops.aten.convolution_backward.default(mul_957, add_109, primals_172, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_957 = add_109 = primals_172 = None
    getitem_757: "f32[8, 56, 28, 28]" = convolution_backward_107[0]
    getitem_758: "f32[336, 56, 1, 1]" = convolution_backward_107[1];  convolution_backward_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_88: "f32[56]" = torch.ops.aten.sum.dim_IntList(getitem_757, [0, 2, 3])
    sub_254: "f32[8, 56, 28, 28]" = torch.ops.aten.sub.Tensor(cat_14, unsqueeze_678);  cat_14 = unsqueeze_678 = None
    mul_959: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_757, sub_254)
    sum_89: "f32[56]" = torch.ops.aten.sum.dim_IntList(mul_959, [0, 2, 3]);  mul_959 = None
    mul_960: "f32[56]" = torch.ops.aten.mul.Tensor(sum_88, 0.00015943877551020407)
    unsqueeze_679: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_960, 0);  mul_960 = None
    unsqueeze_680: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_679, 2);  unsqueeze_679 = None
    unsqueeze_681: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_680, 3);  unsqueeze_680 = None
    mul_961: "f32[56]" = torch.ops.aten.mul.Tensor(sum_89, 0.00015943877551020407)
    mul_962: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_61, squeeze_61)
    mul_963: "f32[56]" = torch.ops.aten.mul.Tensor(mul_961, mul_962);  mul_961 = mul_962 = None
    unsqueeze_682: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_963, 0);  mul_963 = None
    unsqueeze_683: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_682, 2);  unsqueeze_682 = None
    unsqueeze_684: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_683, 3);  unsqueeze_683 = None
    mul_964: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_61, primals_41);  primals_41 = None
    unsqueeze_685: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_964, 0);  mul_964 = None
    unsqueeze_686: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_685, 2);  unsqueeze_685 = None
    unsqueeze_687: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_686, 3);  unsqueeze_686 = None
    mul_965: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(sub_254, unsqueeze_684);  sub_254 = unsqueeze_684 = None
    sub_256: "f32[8, 56, 28, 28]" = torch.ops.aten.sub.Tensor(getitem_757, mul_965);  mul_965 = None
    sub_257: "f32[8, 56, 28, 28]" = torch.ops.aten.sub.Tensor(sub_256, unsqueeze_681);  sub_256 = unsqueeze_681 = None
    mul_966: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(sub_257, unsqueeze_687);  sub_257 = unsqueeze_687 = None
    mul_967: "f32[56]" = torch.ops.aten.mul.Tensor(sum_89, squeeze_61);  sum_89 = squeeze_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    slice_74: "f32[8, 28, 28, 28]" = torch.ops.aten.slice.Tensor(mul_966, 1, 0, 28)
    slice_75: "f32[8, 28, 28, 28]" = torch.ops.aten.slice.Tensor(mul_966, 1, 28, 56);  mul_966 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_backward_108 = torch.ops.aten.convolution_backward.default(slice_75, getitem_117, primals_171, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_75 = getitem_117 = primals_171 = None
    getitem_760: "f32[8, 168, 28, 28]" = convolution_backward_108[0]
    getitem_761: "f32[28, 168, 1, 1]" = convolution_backward_108[1];  convolution_backward_108 = None
    convolution_backward_109 = torch.ops.aten.convolution_backward.default(slice_74, getitem_116, primals_170, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_74 = getitem_116 = primals_170 = None
    getitem_763: "f32[8, 168, 28, 28]" = convolution_backward_109[0]
    getitem_764: "f32[28, 168, 1, 1]" = convolution_backward_109[1];  convolution_backward_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    cat_67: "f32[8, 336, 28, 28]" = torch.ops.aten.cat.default([getitem_763, getitem_760], 1);  getitem_763 = getitem_760 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_968: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(cat_67, mul_153);  mul_153 = None
    mul_969: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(cat_67, sigmoid_15);  cat_67 = sigmoid_15 = None
    sum_90: "f32[8, 336, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_968, [2, 3], True);  mul_968 = None
    alias_38: "f32[8, 336, 1, 1]" = torch.ops.aten.alias.default(alias_9);  alias_9 = None
    sub_258: "f32[8, 336, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_38)
    mul_970: "f32[8, 336, 1, 1]" = torch.ops.aten.mul.Tensor(alias_38, sub_258);  alias_38 = sub_258 = None
    mul_971: "f32[8, 336, 1, 1]" = torch.ops.aten.mul.Tensor(sum_90, mul_970);  sum_90 = mul_970 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_110 = torch.ops.aten.convolution_backward.default(mul_971, mul_154, primals_168, [336], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_971 = mul_154 = primals_168 = None
    getitem_766: "f32[8, 28, 1, 1]" = convolution_backward_110[0]
    getitem_767: "f32[336, 28, 1, 1]" = convolution_backward_110[1]
    getitem_768: "f32[336]" = convolution_backward_110[2];  convolution_backward_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_100: "f32[8, 28, 1, 1]" = torch.ops.aten.sigmoid.default(clone_11)
    full_default_37: "f32[8, 28, 1, 1]" = torch.ops.aten.full.default([8, 28, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_259: "f32[8, 28, 1, 1]" = torch.ops.aten.sub.Tensor(full_default_37, sigmoid_100)
    mul_972: "f32[8, 28, 1, 1]" = torch.ops.aten.mul.Tensor(clone_11, sub_259);  clone_11 = sub_259 = None
    add_361: "f32[8, 28, 1, 1]" = torch.ops.aten.add.Scalar(mul_972, 1);  mul_972 = None
    mul_973: "f32[8, 28, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_100, add_361);  sigmoid_100 = add_361 = None
    mul_974: "f32[8, 28, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_766, mul_973);  getitem_766 = mul_973 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_111 = torch.ops.aten.convolution_backward.default(mul_974, mean_3, primals_166, [28], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_974 = mean_3 = primals_166 = None
    getitem_769: "f32[8, 336, 1, 1]" = convolution_backward_111[0]
    getitem_770: "f32[28, 336, 1, 1]" = convolution_backward_111[1]
    getitem_771: "f32[28]" = convolution_backward_111[2];  convolution_backward_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_13: "f32[8, 336, 28, 28]" = torch.ops.aten.expand.default(getitem_769, [8, 336, 28, 28]);  getitem_769 = None
    div_13: "f32[8, 336, 28, 28]" = torch.ops.aten.div.Scalar(expand_13, 784);  expand_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_362: "f32[8, 336, 28, 28]" = torch.ops.aten.add.Tensor(mul_969, div_13);  mul_969 = div_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_101: "f32[8, 336, 28, 28]" = torch.ops.aten.sigmoid.default(clone_10)
    sub_260: "f32[8, 336, 28, 28]" = torch.ops.aten.sub.Tensor(full_default_36, sigmoid_101)
    mul_975: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(clone_10, sub_260);  clone_10 = sub_260 = None
    add_363: "f32[8, 336, 28, 28]" = torch.ops.aten.add.Scalar(mul_975, 1);  mul_975 = None
    mul_976: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(sigmoid_101, add_363);  sigmoid_101 = add_363 = None
    mul_977: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(add_362, mul_976);  add_362 = mul_976 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_91: "f32[336]" = torch.ops.aten.sum.dim_IntList(mul_977, [0, 2, 3])
    sub_261: "f32[8, 336, 28, 28]" = torch.ops.aten.sub.Tensor(cat_13, unsqueeze_690);  cat_13 = unsqueeze_690 = None
    mul_978: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(mul_977, sub_261)
    sum_92: "f32[336]" = torch.ops.aten.sum.dim_IntList(mul_978, [0, 2, 3]);  mul_978 = None
    mul_979: "f32[336]" = torch.ops.aten.mul.Tensor(sum_91, 0.00015943877551020407)
    unsqueeze_691: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(mul_979, 0);  mul_979 = None
    unsqueeze_692: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_691, 2);  unsqueeze_691 = None
    unsqueeze_693: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_692, 3);  unsqueeze_692 = None
    mul_980: "f32[336]" = torch.ops.aten.mul.Tensor(sum_92, 0.00015943877551020407)
    mul_981: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_58, squeeze_58)
    mul_982: "f32[336]" = torch.ops.aten.mul.Tensor(mul_980, mul_981);  mul_980 = mul_981 = None
    unsqueeze_694: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(mul_982, 0);  mul_982 = None
    unsqueeze_695: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_694, 2);  unsqueeze_694 = None
    unsqueeze_696: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_695, 3);  unsqueeze_695 = None
    mul_983: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_58, primals_39);  primals_39 = None
    unsqueeze_697: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(mul_983, 0);  mul_983 = None
    unsqueeze_698: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_697, 2);  unsqueeze_697 = None
    unsqueeze_699: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_698, 3);  unsqueeze_698 = None
    mul_984: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(sub_261, unsqueeze_696);  sub_261 = unsqueeze_696 = None
    sub_263: "f32[8, 336, 28, 28]" = torch.ops.aten.sub.Tensor(mul_977, mul_984);  mul_977 = mul_984 = None
    sub_264: "f32[8, 336, 28, 28]" = torch.ops.aten.sub.Tensor(sub_263, unsqueeze_693);  sub_263 = unsqueeze_693 = None
    mul_985: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(sub_264, unsqueeze_699);  sub_264 = unsqueeze_699 = None
    mul_986: "f32[336]" = torch.ops.aten.mul.Tensor(sum_92, squeeze_58);  sum_92 = squeeze_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    slice_76: "f32[8, 168, 28, 28]" = torch.ops.aten.slice.Tensor(mul_985, 1, 0, 168)
    slice_77: "f32[8, 168, 28, 28]" = torch.ops.aten.slice.Tensor(mul_985, 1, 168, 336);  mul_985 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_backward_112 = torch.ops.aten.convolution_backward.default(slice_77, getitem_113, primals_165, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 168, [True, True, False]);  slice_77 = getitem_113 = primals_165 = None
    getitem_772: "f32[8, 168, 28, 28]" = convolution_backward_112[0]
    getitem_773: "f32[168, 1, 5, 5]" = convolution_backward_112[1];  convolution_backward_112 = None
    convolution_backward_113 = torch.ops.aten.convolution_backward.default(slice_76, getitem_110, primals_164, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 168, [True, True, False]);  slice_76 = getitem_110 = primals_164 = None
    getitem_775: "f32[8, 168, 28, 28]" = convolution_backward_113[0]
    getitem_776: "f32[168, 1, 3, 3]" = convolution_backward_113[1];  convolution_backward_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    cat_68: "f32[8, 336, 28, 28]" = torch.ops.aten.cat.default([getitem_775, getitem_772], 1);  getitem_775 = getitem_772 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_989: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(cat_68, mul_988);  cat_68 = mul_988 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_93: "f32[336]" = torch.ops.aten.sum.dim_IntList(mul_989, [0, 2, 3])
    sub_266: "f32[8, 336, 28, 28]" = torch.ops.aten.sub.Tensor(cat_12, unsqueeze_702);  cat_12 = unsqueeze_702 = None
    mul_990: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(mul_989, sub_266)
    sum_94: "f32[336]" = torch.ops.aten.sum.dim_IntList(mul_990, [0, 2, 3]);  mul_990 = None
    mul_991: "f32[336]" = torch.ops.aten.mul.Tensor(sum_93, 0.00015943877551020407)
    unsqueeze_703: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(mul_991, 0);  mul_991 = None
    unsqueeze_704: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_703, 2);  unsqueeze_703 = None
    unsqueeze_705: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_704, 3);  unsqueeze_704 = None
    mul_992: "f32[336]" = torch.ops.aten.mul.Tensor(sum_94, 0.00015943877551020407)
    mul_993: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_55, squeeze_55)
    mul_994: "f32[336]" = torch.ops.aten.mul.Tensor(mul_992, mul_993);  mul_992 = mul_993 = None
    unsqueeze_706: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(mul_994, 0);  mul_994 = None
    unsqueeze_707: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_706, 2);  unsqueeze_706 = None
    unsqueeze_708: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_707, 3);  unsqueeze_707 = None
    mul_995: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_55, primals_37);  primals_37 = None
    unsqueeze_709: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(mul_995, 0);  mul_995 = None
    unsqueeze_710: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_709, 2);  unsqueeze_709 = None
    unsqueeze_711: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_710, 3);  unsqueeze_710 = None
    mul_996: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(sub_266, unsqueeze_708);  sub_266 = unsqueeze_708 = None
    sub_268: "f32[8, 336, 28, 28]" = torch.ops.aten.sub.Tensor(mul_989, mul_996);  mul_989 = mul_996 = None
    sub_269: "f32[8, 336, 28, 28]" = torch.ops.aten.sub.Tensor(sub_268, unsqueeze_705);  sub_268 = unsqueeze_705 = None
    mul_997: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(sub_269, unsqueeze_711);  sub_269 = unsqueeze_711 = None
    mul_998: "f32[336]" = torch.ops.aten.mul.Tensor(sum_94, squeeze_55);  sum_94 = squeeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    slice_78: "f32[8, 168, 28, 28]" = torch.ops.aten.slice.Tensor(mul_997, 1, 0, 168)
    slice_79: "f32[8, 168, 28, 28]" = torch.ops.aten.slice.Tensor(mul_997, 1, 168, 336);  mul_997 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_backward_114 = torch.ops.aten.convolution_backward.default(slice_79, getitem_105, primals_163, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_79 = getitem_105 = primals_163 = None
    getitem_778: "f32[8, 28, 28, 28]" = convolution_backward_114[0]
    getitem_779: "f32[168, 28, 1, 1]" = convolution_backward_114[1];  convolution_backward_114 = None
    convolution_backward_115 = torch.ops.aten.convolution_backward.default(slice_78, getitem_104, primals_162, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_78 = getitem_104 = primals_162 = None
    getitem_781: "f32[8, 28, 28, 28]" = convolution_backward_115[0]
    getitem_782: "f32[168, 28, 1, 1]" = convolution_backward_115[1];  convolution_backward_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    cat_69: "f32[8, 56, 28, 28]" = torch.ops.aten.cat.default([getitem_781, getitem_778], 1);  getitem_781 = getitem_778 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    add_365: "f32[8, 56, 28, 28]" = torch.ops.aten.add.Tensor(getitem_757, cat_69);  getitem_757 = cat_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_95: "f32[56]" = torch.ops.aten.sum.dim_IntList(add_365, [0, 2, 3])
    sub_270: "f32[8, 56, 28, 28]" = torch.ops.aten.sub.Tensor(cat_11, unsqueeze_714);  cat_11 = unsqueeze_714 = None
    mul_999: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(add_365, sub_270)
    sum_96: "f32[56]" = torch.ops.aten.sum.dim_IntList(mul_999, [0, 2, 3]);  mul_999 = None
    mul_1000: "f32[56]" = torch.ops.aten.mul.Tensor(sum_95, 0.00015943877551020407)
    unsqueeze_715: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1000, 0);  mul_1000 = None
    unsqueeze_716: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_715, 2);  unsqueeze_715 = None
    unsqueeze_717: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_716, 3);  unsqueeze_716 = None
    mul_1001: "f32[56]" = torch.ops.aten.mul.Tensor(sum_96, 0.00015943877551020407)
    mul_1002: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_52, squeeze_52)
    mul_1003: "f32[56]" = torch.ops.aten.mul.Tensor(mul_1001, mul_1002);  mul_1001 = mul_1002 = None
    unsqueeze_718: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1003, 0);  mul_1003 = None
    unsqueeze_719: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_718, 2);  unsqueeze_718 = None
    unsqueeze_720: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_719, 3);  unsqueeze_719 = None
    mul_1004: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_52, primals_35);  primals_35 = None
    unsqueeze_721: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1004, 0);  mul_1004 = None
    unsqueeze_722: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_721, 2);  unsqueeze_721 = None
    unsqueeze_723: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_722, 3);  unsqueeze_722 = None
    mul_1005: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(sub_270, unsqueeze_720);  sub_270 = unsqueeze_720 = None
    sub_272: "f32[8, 56, 28, 28]" = torch.ops.aten.sub.Tensor(add_365, mul_1005);  mul_1005 = None
    sub_273: "f32[8, 56, 28, 28]" = torch.ops.aten.sub.Tensor(sub_272, unsqueeze_717);  sub_272 = unsqueeze_717 = None
    mul_1006: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(sub_273, unsqueeze_723);  sub_273 = unsqueeze_723 = None
    mul_1007: "f32[56]" = torch.ops.aten.mul.Tensor(sum_96, squeeze_52);  sum_96 = squeeze_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    slice_80: "f32[8, 28, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1006, 1, 0, 28)
    slice_81: "f32[8, 28, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1006, 1, 28, 56);  mul_1006 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_backward_116 = torch.ops.aten.convolution_backward.default(slice_81, getitem_101, primals_161, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_81 = getitem_101 = primals_161 = None
    getitem_784: "f32[8, 168, 28, 28]" = convolution_backward_116[0]
    getitem_785: "f32[28, 168, 1, 1]" = convolution_backward_116[1];  convolution_backward_116 = None
    convolution_backward_117 = torch.ops.aten.convolution_backward.default(slice_80, getitem_100, primals_160, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_80 = getitem_100 = primals_160 = None
    getitem_787: "f32[8, 168, 28, 28]" = convolution_backward_117[0]
    getitem_788: "f32[28, 168, 1, 1]" = convolution_backward_117[1];  convolution_backward_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    cat_70: "f32[8, 336, 28, 28]" = torch.ops.aten.cat.default([getitem_787, getitem_784], 1);  getitem_787 = getitem_784 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_1008: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(cat_70, mul_128);  mul_128 = None
    mul_1009: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(cat_70, sigmoid_11);  cat_70 = sigmoid_11 = None
    sum_97: "f32[8, 336, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1008, [2, 3], True);  mul_1008 = None
    alias_39: "f32[8, 336, 1, 1]" = torch.ops.aten.alias.default(alias_8);  alias_8 = None
    sub_274: "f32[8, 336, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_39)
    mul_1010: "f32[8, 336, 1, 1]" = torch.ops.aten.mul.Tensor(alias_39, sub_274);  alias_39 = sub_274 = None
    mul_1011: "f32[8, 336, 1, 1]" = torch.ops.aten.mul.Tensor(sum_97, mul_1010);  sum_97 = mul_1010 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_118 = torch.ops.aten.convolution_backward.default(mul_1011, mul_129, primals_158, [336], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_1011 = mul_129 = primals_158 = None
    getitem_790: "f32[8, 28, 1, 1]" = convolution_backward_118[0]
    getitem_791: "f32[336, 28, 1, 1]" = convolution_backward_118[1]
    getitem_792: "f32[336]" = convolution_backward_118[2];  convolution_backward_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_103: "f32[8, 28, 1, 1]" = torch.ops.aten.sigmoid.default(clone_8)
    sub_275: "f32[8, 28, 1, 1]" = torch.ops.aten.sub.Tensor(full_default_37, sigmoid_103)
    mul_1012: "f32[8, 28, 1, 1]" = torch.ops.aten.mul.Tensor(clone_8, sub_275);  clone_8 = sub_275 = None
    add_366: "f32[8, 28, 1, 1]" = torch.ops.aten.add.Scalar(mul_1012, 1);  mul_1012 = None
    mul_1013: "f32[8, 28, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_103, add_366);  sigmoid_103 = add_366 = None
    mul_1014: "f32[8, 28, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_790, mul_1013);  getitem_790 = mul_1013 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_119 = torch.ops.aten.convolution_backward.default(mul_1014, mean_2, primals_156, [28], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_1014 = mean_2 = primals_156 = None
    getitem_793: "f32[8, 336, 1, 1]" = convolution_backward_119[0]
    getitem_794: "f32[28, 336, 1, 1]" = convolution_backward_119[1]
    getitem_795: "f32[28]" = convolution_backward_119[2];  convolution_backward_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_14: "f32[8, 336, 28, 28]" = torch.ops.aten.expand.default(getitem_793, [8, 336, 28, 28]);  getitem_793 = None
    div_14: "f32[8, 336, 28, 28]" = torch.ops.aten.div.Scalar(expand_14, 784);  expand_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_367: "f32[8, 336, 28, 28]" = torch.ops.aten.add.Tensor(mul_1009, div_14);  mul_1009 = div_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_104: "f32[8, 336, 28, 28]" = torch.ops.aten.sigmoid.default(clone_7)
    sub_276: "f32[8, 336, 28, 28]" = torch.ops.aten.sub.Tensor(full_default_36, sigmoid_104)
    mul_1015: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(clone_7, sub_276);  clone_7 = sub_276 = None
    add_368: "f32[8, 336, 28, 28]" = torch.ops.aten.add.Scalar(mul_1015, 1);  mul_1015 = None
    mul_1016: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(sigmoid_104, add_368);  sigmoid_104 = add_368 = None
    mul_1017: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(add_367, mul_1016);  add_367 = mul_1016 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_98: "f32[336]" = torch.ops.aten.sum.dim_IntList(mul_1017, [0, 2, 3])
    sub_277: "f32[8, 336, 28, 28]" = torch.ops.aten.sub.Tensor(cat_10, unsqueeze_726);  cat_10 = unsqueeze_726 = None
    mul_1018: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(mul_1017, sub_277)
    sum_99: "f32[336]" = torch.ops.aten.sum.dim_IntList(mul_1018, [0, 2, 3]);  mul_1018 = None
    mul_1019: "f32[336]" = torch.ops.aten.mul.Tensor(sum_98, 0.00015943877551020407)
    unsqueeze_727: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(mul_1019, 0);  mul_1019 = None
    unsqueeze_728: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_727, 2);  unsqueeze_727 = None
    unsqueeze_729: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_728, 3);  unsqueeze_728 = None
    mul_1020: "f32[336]" = torch.ops.aten.mul.Tensor(sum_99, 0.00015943877551020407)
    mul_1021: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_49, squeeze_49)
    mul_1022: "f32[336]" = torch.ops.aten.mul.Tensor(mul_1020, mul_1021);  mul_1020 = mul_1021 = None
    unsqueeze_730: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(mul_1022, 0);  mul_1022 = None
    unsqueeze_731: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_730, 2);  unsqueeze_730 = None
    unsqueeze_732: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_731, 3);  unsqueeze_731 = None
    mul_1023: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_49, primals_33);  primals_33 = None
    unsqueeze_733: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(mul_1023, 0);  mul_1023 = None
    unsqueeze_734: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_733, 2);  unsqueeze_733 = None
    unsqueeze_735: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_734, 3);  unsqueeze_734 = None
    mul_1024: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(sub_277, unsqueeze_732);  sub_277 = unsqueeze_732 = None
    sub_279: "f32[8, 336, 28, 28]" = torch.ops.aten.sub.Tensor(mul_1017, mul_1024);  mul_1017 = mul_1024 = None
    sub_280: "f32[8, 336, 28, 28]" = torch.ops.aten.sub.Tensor(sub_279, unsqueeze_729);  sub_279 = unsqueeze_729 = None
    mul_1025: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(sub_280, unsqueeze_735);  sub_280 = unsqueeze_735 = None
    mul_1026: "f32[336]" = torch.ops.aten.mul.Tensor(sum_99, squeeze_49);  sum_99 = squeeze_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    slice_82: "f32[8, 168, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1025, 1, 0, 168)
    slice_83: "f32[8, 168, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1025, 1, 168, 336);  mul_1025 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_backward_120 = torch.ops.aten.convolution_backward.default(slice_83, getitem_97, primals_155, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 168, [True, True, False]);  slice_83 = getitem_97 = primals_155 = None
    getitem_796: "f32[8, 168, 28, 28]" = convolution_backward_120[0]
    getitem_797: "f32[168, 1, 5, 5]" = convolution_backward_120[1];  convolution_backward_120 = None
    convolution_backward_121 = torch.ops.aten.convolution_backward.default(slice_82, getitem_94, primals_154, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 168, [True, True, False]);  slice_82 = getitem_94 = primals_154 = None
    getitem_799: "f32[8, 168, 28, 28]" = convolution_backward_121[0]
    getitem_800: "f32[168, 1, 3, 3]" = convolution_backward_121[1];  convolution_backward_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    cat_71: "f32[8, 336, 28, 28]" = torch.ops.aten.cat.default([getitem_799, getitem_796], 1);  getitem_799 = getitem_796 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_1029: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(cat_71, mul_1028);  cat_71 = mul_1028 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_100: "f32[336]" = torch.ops.aten.sum.dim_IntList(mul_1029, [0, 2, 3])
    sub_282: "f32[8, 336, 28, 28]" = torch.ops.aten.sub.Tensor(cat_9, unsqueeze_738);  cat_9 = unsqueeze_738 = None
    mul_1030: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(mul_1029, sub_282)
    sum_101: "f32[336]" = torch.ops.aten.sum.dim_IntList(mul_1030, [0, 2, 3]);  mul_1030 = None
    mul_1031: "f32[336]" = torch.ops.aten.mul.Tensor(sum_100, 0.00015943877551020407)
    unsqueeze_739: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(mul_1031, 0);  mul_1031 = None
    unsqueeze_740: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_739, 2);  unsqueeze_739 = None
    unsqueeze_741: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_740, 3);  unsqueeze_740 = None
    mul_1032: "f32[336]" = torch.ops.aten.mul.Tensor(sum_101, 0.00015943877551020407)
    mul_1033: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_46, squeeze_46)
    mul_1034: "f32[336]" = torch.ops.aten.mul.Tensor(mul_1032, mul_1033);  mul_1032 = mul_1033 = None
    unsqueeze_742: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(mul_1034, 0);  mul_1034 = None
    unsqueeze_743: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_742, 2);  unsqueeze_742 = None
    unsqueeze_744: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_743, 3);  unsqueeze_743 = None
    mul_1035: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_46, primals_31);  primals_31 = None
    unsqueeze_745: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(mul_1035, 0);  mul_1035 = None
    unsqueeze_746: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_745, 2);  unsqueeze_745 = None
    unsqueeze_747: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_746, 3);  unsqueeze_746 = None
    mul_1036: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(sub_282, unsqueeze_744);  sub_282 = unsqueeze_744 = None
    sub_284: "f32[8, 336, 28, 28]" = torch.ops.aten.sub.Tensor(mul_1029, mul_1036);  mul_1029 = mul_1036 = None
    sub_285: "f32[8, 336, 28, 28]" = torch.ops.aten.sub.Tensor(sub_284, unsqueeze_741);  sub_284 = unsqueeze_741 = None
    mul_1037: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(sub_285, unsqueeze_747);  sub_285 = unsqueeze_747 = None
    mul_1038: "f32[336]" = torch.ops.aten.mul.Tensor(sum_101, squeeze_46);  sum_101 = squeeze_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    slice_84: "f32[8, 168, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1037, 1, 0, 168)
    slice_85: "f32[8, 168, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1037, 1, 168, 336);  mul_1037 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_backward_122 = torch.ops.aten.convolution_backward.default(slice_85, getitem_89, primals_153, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_85 = getitem_89 = primals_153 = None
    getitem_802: "f32[8, 28, 28, 28]" = convolution_backward_122[0]
    getitem_803: "f32[168, 28, 1, 1]" = convolution_backward_122[1];  convolution_backward_122 = None
    convolution_backward_123 = torch.ops.aten.convolution_backward.default(slice_84, getitem_88, primals_152, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_84 = getitem_88 = primals_152 = None
    getitem_805: "f32[8, 28, 28, 28]" = convolution_backward_123[0]
    getitem_806: "f32[168, 28, 1, 1]" = convolution_backward_123[1];  convolution_backward_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    cat_72: "f32[8, 56, 28, 28]" = torch.ops.aten.cat.default([getitem_805, getitem_802], 1);  getitem_805 = getitem_802 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    add_370: "f32[8, 56, 28, 28]" = torch.ops.aten.add.Tensor(add_365, cat_72);  add_365 = cat_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_102: "f32[56]" = torch.ops.aten.sum.dim_IntList(add_370, [0, 2, 3])
    sub_286: "f32[8, 56, 28, 28]" = torch.ops.aten.sub.Tensor(cat_8, unsqueeze_750);  cat_8 = unsqueeze_750 = None
    mul_1039: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(add_370, sub_286)
    sum_103: "f32[56]" = torch.ops.aten.sum.dim_IntList(mul_1039, [0, 2, 3]);  mul_1039 = None
    mul_1040: "f32[56]" = torch.ops.aten.mul.Tensor(sum_102, 0.00015943877551020407)
    unsqueeze_751: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1040, 0);  mul_1040 = None
    unsqueeze_752: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_751, 2);  unsqueeze_751 = None
    unsqueeze_753: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_752, 3);  unsqueeze_752 = None
    mul_1041: "f32[56]" = torch.ops.aten.mul.Tensor(sum_103, 0.00015943877551020407)
    mul_1042: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
    mul_1043: "f32[56]" = torch.ops.aten.mul.Tensor(mul_1041, mul_1042);  mul_1041 = mul_1042 = None
    unsqueeze_754: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1043, 0);  mul_1043 = None
    unsqueeze_755: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_754, 2);  unsqueeze_754 = None
    unsqueeze_756: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_755, 3);  unsqueeze_755 = None
    mul_1044: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_43, primals_29);  primals_29 = None
    unsqueeze_757: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1044, 0);  mul_1044 = None
    unsqueeze_758: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_757, 2);  unsqueeze_757 = None
    unsqueeze_759: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_758, 3);  unsqueeze_758 = None
    mul_1045: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(sub_286, unsqueeze_756);  sub_286 = unsqueeze_756 = None
    sub_288: "f32[8, 56, 28, 28]" = torch.ops.aten.sub.Tensor(add_370, mul_1045);  mul_1045 = None
    sub_289: "f32[8, 56, 28, 28]" = torch.ops.aten.sub.Tensor(sub_288, unsqueeze_753);  sub_288 = unsqueeze_753 = None
    mul_1046: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(sub_289, unsqueeze_759);  sub_289 = unsqueeze_759 = None
    mul_1047: "f32[56]" = torch.ops.aten.mul.Tensor(sum_103, squeeze_43);  sum_103 = squeeze_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    slice_86: "f32[8, 28, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1046, 1, 0, 28)
    slice_87: "f32[8, 28, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1046, 1, 28, 56);  mul_1046 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_backward_124 = torch.ops.aten.convolution_backward.default(slice_87, getitem_85, primals_151, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_87 = getitem_85 = primals_151 = None
    getitem_808: "f32[8, 168, 28, 28]" = convolution_backward_124[0]
    getitem_809: "f32[28, 168, 1, 1]" = convolution_backward_124[1];  convolution_backward_124 = None
    convolution_backward_125 = torch.ops.aten.convolution_backward.default(slice_86, getitem_84, primals_150, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_86 = getitem_84 = primals_150 = None
    getitem_811: "f32[8, 168, 28, 28]" = convolution_backward_125[0]
    getitem_812: "f32[28, 168, 1, 1]" = convolution_backward_125[1];  convolution_backward_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    cat_73: "f32[8, 336, 28, 28]" = torch.ops.aten.cat.default([getitem_811, getitem_808], 1);  getitem_811 = getitem_808 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_1048: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(cat_73, mul_103);  mul_103 = None
    mul_1049: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(cat_73, sigmoid_7);  cat_73 = sigmoid_7 = None
    sum_104: "f32[8, 336, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1048, [2, 3], True);  mul_1048 = None
    alias_40: "f32[8, 336, 1, 1]" = torch.ops.aten.alias.default(alias_7);  alias_7 = None
    sub_290: "f32[8, 336, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_40)
    mul_1050: "f32[8, 336, 1, 1]" = torch.ops.aten.mul.Tensor(alias_40, sub_290);  alias_40 = sub_290 = None
    mul_1051: "f32[8, 336, 1, 1]" = torch.ops.aten.mul.Tensor(sum_104, mul_1050);  sum_104 = mul_1050 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_126 = torch.ops.aten.convolution_backward.default(mul_1051, mul_104, primals_148, [336], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_1051 = mul_104 = primals_148 = None
    getitem_814: "f32[8, 28, 1, 1]" = convolution_backward_126[0]
    getitem_815: "f32[336, 28, 1, 1]" = convolution_backward_126[1]
    getitem_816: "f32[336]" = convolution_backward_126[2];  convolution_backward_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_106: "f32[8, 28, 1, 1]" = torch.ops.aten.sigmoid.default(clone_5)
    sub_291: "f32[8, 28, 1, 1]" = torch.ops.aten.sub.Tensor(full_default_37, sigmoid_106);  full_default_37 = None
    mul_1052: "f32[8, 28, 1, 1]" = torch.ops.aten.mul.Tensor(clone_5, sub_291);  clone_5 = sub_291 = None
    add_371: "f32[8, 28, 1, 1]" = torch.ops.aten.add.Scalar(mul_1052, 1);  mul_1052 = None
    mul_1053: "f32[8, 28, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_106, add_371);  sigmoid_106 = add_371 = None
    mul_1054: "f32[8, 28, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_814, mul_1053);  getitem_814 = mul_1053 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_127 = torch.ops.aten.convolution_backward.default(mul_1054, mean_1, primals_146, [28], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_1054 = mean_1 = primals_146 = None
    getitem_817: "f32[8, 336, 1, 1]" = convolution_backward_127[0]
    getitem_818: "f32[28, 336, 1, 1]" = convolution_backward_127[1]
    getitem_819: "f32[28]" = convolution_backward_127[2];  convolution_backward_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_15: "f32[8, 336, 28, 28]" = torch.ops.aten.expand.default(getitem_817, [8, 336, 28, 28]);  getitem_817 = None
    div_15: "f32[8, 336, 28, 28]" = torch.ops.aten.div.Scalar(expand_15, 784);  expand_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_372: "f32[8, 336, 28, 28]" = torch.ops.aten.add.Tensor(mul_1049, div_15);  mul_1049 = div_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_107: "f32[8, 336, 28, 28]" = torch.ops.aten.sigmoid.default(clone_4)
    sub_292: "f32[8, 336, 28, 28]" = torch.ops.aten.sub.Tensor(full_default_36, sigmoid_107);  full_default_36 = None
    mul_1055: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(clone_4, sub_292);  clone_4 = sub_292 = None
    add_373: "f32[8, 336, 28, 28]" = torch.ops.aten.add.Scalar(mul_1055, 1);  mul_1055 = None
    mul_1056: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(sigmoid_107, add_373);  sigmoid_107 = add_373 = None
    mul_1057: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(add_372, mul_1056);  add_372 = mul_1056 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_105: "f32[336]" = torch.ops.aten.sum.dim_IntList(mul_1057, [0, 2, 3])
    sub_293: "f32[8, 336, 28, 28]" = torch.ops.aten.sub.Tensor(cat_7, unsqueeze_762);  cat_7 = unsqueeze_762 = None
    mul_1058: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(mul_1057, sub_293)
    sum_106: "f32[336]" = torch.ops.aten.sum.dim_IntList(mul_1058, [0, 2, 3]);  mul_1058 = None
    mul_1059: "f32[336]" = torch.ops.aten.mul.Tensor(sum_105, 0.00015943877551020407)
    unsqueeze_763: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(mul_1059, 0);  mul_1059 = None
    unsqueeze_764: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_763, 2);  unsqueeze_763 = None
    unsqueeze_765: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_764, 3);  unsqueeze_764 = None
    mul_1060: "f32[336]" = torch.ops.aten.mul.Tensor(sum_106, 0.00015943877551020407)
    mul_1061: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_40, squeeze_40)
    mul_1062: "f32[336]" = torch.ops.aten.mul.Tensor(mul_1060, mul_1061);  mul_1060 = mul_1061 = None
    unsqueeze_766: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(mul_1062, 0);  mul_1062 = None
    unsqueeze_767: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_766, 2);  unsqueeze_766 = None
    unsqueeze_768: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_767, 3);  unsqueeze_767 = None
    mul_1063: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_40, primals_27);  primals_27 = None
    unsqueeze_769: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(mul_1063, 0);  mul_1063 = None
    unsqueeze_770: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_769, 2);  unsqueeze_769 = None
    unsqueeze_771: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_770, 3);  unsqueeze_770 = None
    mul_1064: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(sub_293, unsqueeze_768);  sub_293 = unsqueeze_768 = None
    sub_295: "f32[8, 336, 28, 28]" = torch.ops.aten.sub.Tensor(mul_1057, mul_1064);  mul_1057 = mul_1064 = None
    sub_296: "f32[8, 336, 28, 28]" = torch.ops.aten.sub.Tensor(sub_295, unsqueeze_765);  sub_295 = unsqueeze_765 = None
    mul_1065: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(sub_296, unsqueeze_771);  sub_296 = unsqueeze_771 = None
    mul_1066: "f32[336]" = torch.ops.aten.mul.Tensor(sum_106, squeeze_40);  sum_106 = squeeze_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    slice_88: "f32[8, 168, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1065, 1, 0, 168)
    slice_89: "f32[8, 168, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1065, 1, 168, 336);  mul_1065 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_backward_128 = torch.ops.aten.convolution_backward.default(slice_89, getitem_81, primals_145, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 168, [True, True, False]);  slice_89 = getitem_81 = primals_145 = None
    getitem_820: "f32[8, 168, 28, 28]" = convolution_backward_128[0]
    getitem_821: "f32[168, 1, 5, 5]" = convolution_backward_128[1];  convolution_backward_128 = None
    convolution_backward_129 = torch.ops.aten.convolution_backward.default(slice_88, getitem_78, primals_144, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 168, [True, True, False]);  slice_88 = getitem_78 = primals_144 = None
    getitem_823: "f32[8, 168, 28, 28]" = convolution_backward_129[0]
    getitem_824: "f32[168, 1, 3, 3]" = convolution_backward_129[1];  convolution_backward_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    cat_74: "f32[8, 336, 28, 28]" = torch.ops.aten.cat.default([getitem_823, getitem_820], 1);  getitem_823 = getitem_820 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_1069: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(cat_74, mul_1068);  cat_74 = mul_1068 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_107: "f32[336]" = torch.ops.aten.sum.dim_IntList(mul_1069, [0, 2, 3])
    sub_298: "f32[8, 336, 28, 28]" = torch.ops.aten.sub.Tensor(cat_6, unsqueeze_774);  cat_6 = unsqueeze_774 = None
    mul_1070: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(mul_1069, sub_298)
    sum_108: "f32[336]" = torch.ops.aten.sum.dim_IntList(mul_1070, [0, 2, 3]);  mul_1070 = None
    mul_1071: "f32[336]" = torch.ops.aten.mul.Tensor(sum_107, 0.00015943877551020407)
    unsqueeze_775: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(mul_1071, 0);  mul_1071 = None
    unsqueeze_776: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_775, 2);  unsqueeze_775 = None
    unsqueeze_777: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_776, 3);  unsqueeze_776 = None
    mul_1072: "f32[336]" = torch.ops.aten.mul.Tensor(sum_108, 0.00015943877551020407)
    mul_1073: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
    mul_1074: "f32[336]" = torch.ops.aten.mul.Tensor(mul_1072, mul_1073);  mul_1072 = mul_1073 = None
    unsqueeze_778: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(mul_1074, 0);  mul_1074 = None
    unsqueeze_779: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_778, 2);  unsqueeze_778 = None
    unsqueeze_780: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_779, 3);  unsqueeze_779 = None
    mul_1075: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_37, primals_25);  primals_25 = None
    unsqueeze_781: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(mul_1075, 0);  mul_1075 = None
    unsqueeze_782: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_781, 2);  unsqueeze_781 = None
    unsqueeze_783: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_782, 3);  unsqueeze_782 = None
    mul_1076: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(sub_298, unsqueeze_780);  sub_298 = unsqueeze_780 = None
    sub_300: "f32[8, 336, 28, 28]" = torch.ops.aten.sub.Tensor(mul_1069, mul_1076);  mul_1069 = mul_1076 = None
    sub_301: "f32[8, 336, 28, 28]" = torch.ops.aten.sub.Tensor(sub_300, unsqueeze_777);  sub_300 = unsqueeze_777 = None
    mul_1077: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(sub_301, unsqueeze_783);  sub_301 = unsqueeze_783 = None
    mul_1078: "f32[336]" = torch.ops.aten.mul.Tensor(sum_108, squeeze_37);  sum_108 = squeeze_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    slice_90: "f32[8, 168, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1077, 1, 0, 168)
    slice_91: "f32[8, 168, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1077, 1, 168, 336);  mul_1077 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_backward_130 = torch.ops.aten.convolution_backward.default(slice_91, getitem_73, primals_143, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_91 = getitem_73 = primals_143 = None
    getitem_826: "f32[8, 28, 28, 28]" = convolution_backward_130[0]
    getitem_827: "f32[168, 28, 1, 1]" = convolution_backward_130[1];  convolution_backward_130 = None
    convolution_backward_131 = torch.ops.aten.convolution_backward.default(slice_90, getitem_72, primals_142, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_90 = getitem_72 = primals_142 = None
    getitem_829: "f32[8, 28, 28, 28]" = convolution_backward_131[0]
    getitem_830: "f32[168, 28, 1, 1]" = convolution_backward_131[1];  convolution_backward_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    cat_75: "f32[8, 56, 28, 28]" = torch.ops.aten.cat.default([getitem_829, getitem_826], 1);  getitem_829 = getitem_826 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    add_375: "f32[8, 56, 28, 28]" = torch.ops.aten.add.Tensor(add_370, cat_75);  add_370 = cat_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_109: "f32[56]" = torch.ops.aten.sum.dim_IntList(add_375, [0, 2, 3])
    sub_302: "f32[8, 56, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_786);  convolution_22 = unsqueeze_786 = None
    mul_1079: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(add_375, sub_302)
    sum_110: "f32[56]" = torch.ops.aten.sum.dim_IntList(mul_1079, [0, 2, 3]);  mul_1079 = None
    mul_1080: "f32[56]" = torch.ops.aten.mul.Tensor(sum_109, 0.00015943877551020407)
    unsqueeze_787: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1080, 0);  mul_1080 = None
    unsqueeze_788: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_787, 2);  unsqueeze_787 = None
    unsqueeze_789: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_788, 3);  unsqueeze_788 = None
    mul_1081: "f32[56]" = torch.ops.aten.mul.Tensor(sum_110, 0.00015943877551020407)
    mul_1082: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_34, squeeze_34)
    mul_1083: "f32[56]" = torch.ops.aten.mul.Tensor(mul_1081, mul_1082);  mul_1081 = mul_1082 = None
    unsqueeze_790: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1083, 0);  mul_1083 = None
    unsqueeze_791: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_790, 2);  unsqueeze_790 = None
    unsqueeze_792: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_791, 3);  unsqueeze_791 = None
    mul_1084: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_34, primals_23);  primals_23 = None
    unsqueeze_793: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1084, 0);  mul_1084 = None
    unsqueeze_794: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_793, 2);  unsqueeze_793 = None
    unsqueeze_795: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_794, 3);  unsqueeze_794 = None
    mul_1085: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(sub_302, unsqueeze_792);  sub_302 = unsqueeze_792 = None
    sub_304: "f32[8, 56, 28, 28]" = torch.ops.aten.sub.Tensor(add_375, mul_1085);  add_375 = mul_1085 = None
    sub_305: "f32[8, 56, 28, 28]" = torch.ops.aten.sub.Tensor(sub_304, unsqueeze_789);  sub_304 = unsqueeze_789 = None
    mul_1086: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(sub_305, unsqueeze_795);  sub_305 = unsqueeze_795 = None
    mul_1087: "f32[56]" = torch.ops.aten.mul.Tensor(sum_110, squeeze_34);  sum_110 = squeeze_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_132 = torch.ops.aten.convolution_backward.default(mul_1086, mul_80, primals_141, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1086 = mul_80 = primals_141 = None
    getitem_832: "f32[8, 240, 28, 28]" = convolution_backward_132[0]
    getitem_833: "f32[56, 240, 1, 1]" = convolution_backward_132[1];  convolution_backward_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_1088: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_832, mul_78);  mul_78 = None
    mul_1089: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_832, sigmoid_3);  getitem_832 = sigmoid_3 = None
    sum_111: "f32[8, 240, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1088, [2, 3], True);  mul_1088 = None
    alias_41: "f32[8, 240, 1, 1]" = torch.ops.aten.alias.default(alias_6);  alias_6 = None
    sub_306: "f32[8, 240, 1, 1]" = torch.ops.aten.sub.Tensor(1, alias_41)
    mul_1090: "f32[8, 240, 1, 1]" = torch.ops.aten.mul.Tensor(alias_41, sub_306);  alias_41 = sub_306 = None
    mul_1091: "f32[8, 240, 1, 1]" = torch.ops.aten.mul.Tensor(sum_111, mul_1090);  sum_111 = mul_1090 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_133 = torch.ops.aten.convolution_backward.default(mul_1091, mul_79, primals_139, [240], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_1091 = mul_79 = primals_139 = None
    getitem_835: "f32[8, 20, 1, 1]" = convolution_backward_133[0]
    getitem_836: "f32[240, 20, 1, 1]" = convolution_backward_133[1]
    getitem_837: "f32[240]" = convolution_backward_133[2];  convolution_backward_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_109: "f32[8, 20, 1, 1]" = torch.ops.aten.sigmoid.default(clone_2)
    full_default_46: "f32[8, 20, 1, 1]" = torch.ops.aten.full.default([8, 20, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_307: "f32[8, 20, 1, 1]" = torch.ops.aten.sub.Tensor(full_default_46, sigmoid_109);  full_default_46 = None
    mul_1092: "f32[8, 20, 1, 1]" = torch.ops.aten.mul.Tensor(clone_2, sub_307);  clone_2 = sub_307 = None
    add_376: "f32[8, 20, 1, 1]" = torch.ops.aten.add.Scalar(mul_1092, 1);  mul_1092 = None
    mul_1093: "f32[8, 20, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_109, add_376);  sigmoid_109 = add_376 = None
    mul_1094: "f32[8, 20, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_835, mul_1093);  getitem_835 = mul_1093 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_134 = torch.ops.aten.convolution_backward.default(mul_1094, mean, primals_137, [20], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_1094 = mean = primals_137 = None
    getitem_838: "f32[8, 240, 1, 1]" = convolution_backward_134[0]
    getitem_839: "f32[20, 240, 1, 1]" = convolution_backward_134[1]
    getitem_840: "f32[20]" = convolution_backward_134[2];  convolution_backward_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_16: "f32[8, 240, 28, 28]" = torch.ops.aten.expand.default(getitem_838, [8, 240, 28, 28]);  getitem_838 = None
    div_16: "f32[8, 240, 28, 28]" = torch.ops.aten.div.Scalar(expand_16, 784);  expand_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_377: "f32[8, 240, 28, 28]" = torch.ops.aten.add.Tensor(mul_1089, div_16);  mul_1089 = div_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_110: "f32[8, 240, 28, 28]" = torch.ops.aten.sigmoid.default(clone_1)
    full_default_47: "f32[8, 240, 28, 28]" = torch.ops.aten.full.default([8, 240, 28, 28], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_308: "f32[8, 240, 28, 28]" = torch.ops.aten.sub.Tensor(full_default_47, sigmoid_110);  full_default_47 = None
    mul_1095: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(clone_1, sub_308);  clone_1 = sub_308 = None
    add_378: "f32[8, 240, 28, 28]" = torch.ops.aten.add.Scalar(mul_1095, 1);  mul_1095 = None
    mul_1096: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(sigmoid_110, add_378);  sigmoid_110 = add_378 = None
    mul_1097: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(add_377, mul_1096);  add_377 = mul_1096 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_112: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_1097, [0, 2, 3])
    sub_309: "f32[8, 240, 28, 28]" = torch.ops.aten.sub.Tensor(cat_5, unsqueeze_798);  cat_5 = unsqueeze_798 = None
    mul_1098: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(mul_1097, sub_309)
    sum_113: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_1098, [0, 2, 3]);  mul_1098 = None
    mul_1099: "f32[240]" = torch.ops.aten.mul.Tensor(sum_112, 0.00015943877551020407)
    unsqueeze_799: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_1099, 0);  mul_1099 = None
    unsqueeze_800: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_799, 2);  unsqueeze_799 = None
    unsqueeze_801: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_800, 3);  unsqueeze_800 = None
    mul_1100: "f32[240]" = torch.ops.aten.mul.Tensor(sum_113, 0.00015943877551020407)
    mul_1101: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
    mul_1102: "f32[240]" = torch.ops.aten.mul.Tensor(mul_1100, mul_1101);  mul_1100 = mul_1101 = None
    unsqueeze_802: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_1102, 0);  mul_1102 = None
    unsqueeze_803: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_802, 2);  unsqueeze_802 = None
    unsqueeze_804: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_803, 3);  unsqueeze_803 = None
    mul_1103: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_31, primals_21);  primals_21 = None
    unsqueeze_805: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_1103, 0);  mul_1103 = None
    unsqueeze_806: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_805, 2);  unsqueeze_805 = None
    unsqueeze_807: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_806, 3);  unsqueeze_806 = None
    mul_1104: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(sub_309, unsqueeze_804);  sub_309 = unsqueeze_804 = None
    sub_311: "f32[8, 240, 28, 28]" = torch.ops.aten.sub.Tensor(mul_1097, mul_1104);  mul_1097 = mul_1104 = None
    sub_312: "f32[8, 240, 28, 28]" = torch.ops.aten.sub.Tensor(sub_311, unsqueeze_801);  sub_311 = unsqueeze_801 = None
    mul_1105: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(sub_312, unsqueeze_807);  sub_312 = unsqueeze_807 = None
    mul_1106: "f32[240]" = torch.ops.aten.mul.Tensor(sum_113, squeeze_31);  sum_113 = squeeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    slice_92: "f32[8, 60, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1105, 1, 0, 60)
    slice_93: "f32[8, 60, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1105, 1, 60, 120)
    slice_94: "f32[8, 60, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1105, 1, 120, 180)
    slice_95: "f32[8, 60, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1105, 1, 180, 240);  mul_1105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_backward_135 = torch.ops.aten.convolution_backward.default(slice_95, getitem_67, primals_136, [0], [2, 2], [4, 4], [1, 1], False, [0, 0], 60, [True, True, False]);  slice_95 = getitem_67 = primals_136 = None
    getitem_841: "f32[8, 60, 56, 56]" = convolution_backward_135[0]
    getitem_842: "f32[60, 1, 9, 9]" = convolution_backward_135[1];  convolution_backward_135 = None
    convolution_backward_136 = torch.ops.aten.convolution_backward.default(slice_94, getitem_62, primals_135, [0], [2, 2], [3, 3], [1, 1], False, [0, 0], 60, [True, True, False]);  slice_94 = getitem_62 = primals_135 = None
    getitem_844: "f32[8, 60, 56, 56]" = convolution_backward_136[0]
    getitem_845: "f32[60, 1, 7, 7]" = convolution_backward_136[1];  convolution_backward_136 = None
    convolution_backward_137 = torch.ops.aten.convolution_backward.default(slice_93, getitem_57, primals_134, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 60, [True, True, False]);  slice_93 = getitem_57 = primals_134 = None
    getitem_847: "f32[8, 60, 56, 56]" = convolution_backward_137[0]
    getitem_848: "f32[60, 1, 5, 5]" = convolution_backward_137[1];  convolution_backward_137 = None
    convolution_backward_138 = torch.ops.aten.convolution_backward.default(slice_92, getitem_52, primals_133, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 60, [True, True, False]);  slice_92 = getitem_52 = primals_133 = None
    getitem_850: "f32[8, 60, 56, 56]" = convolution_backward_138[0]
    getitem_851: "f32[60, 1, 3, 3]" = convolution_backward_138[1];  convolution_backward_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    cat_76: "f32[8, 240, 56, 56]" = torch.ops.aten.cat.default([getitem_850, getitem_847, getitem_844, getitem_841], 1);  getitem_850 = getitem_847 = getitem_844 = getitem_841 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_1109: "f32[8, 240, 56, 56]" = torch.ops.aten.mul.Tensor(cat_76, mul_1108);  cat_76 = mul_1108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_114: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_1109, [0, 2, 3])
    sub_314: "f32[8, 240, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_810);  convolution_15 = unsqueeze_810 = None
    mul_1110: "f32[8, 240, 56, 56]" = torch.ops.aten.mul.Tensor(mul_1109, sub_314)
    sum_115: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_1110, [0, 2, 3]);  mul_1110 = None
    mul_1111: "f32[240]" = torch.ops.aten.mul.Tensor(sum_114, 3.985969387755102e-05)
    unsqueeze_811: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_1111, 0);  mul_1111 = None
    unsqueeze_812: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_811, 2);  unsqueeze_811 = None
    unsqueeze_813: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_812, 3);  unsqueeze_812 = None
    mul_1112: "f32[240]" = torch.ops.aten.mul.Tensor(sum_115, 3.985969387755102e-05)
    mul_1113: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_28, squeeze_28)
    mul_1114: "f32[240]" = torch.ops.aten.mul.Tensor(mul_1112, mul_1113);  mul_1112 = mul_1113 = None
    unsqueeze_814: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_1114, 0);  mul_1114 = None
    unsqueeze_815: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_814, 2);  unsqueeze_814 = None
    unsqueeze_816: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_815, 3);  unsqueeze_815 = None
    mul_1115: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_28, primals_19);  primals_19 = None
    unsqueeze_817: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_1115, 0);  mul_1115 = None
    unsqueeze_818: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_817, 2);  unsqueeze_817 = None
    unsqueeze_819: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_818, 3);  unsqueeze_818 = None
    mul_1116: "f32[8, 240, 56, 56]" = torch.ops.aten.mul.Tensor(sub_314, unsqueeze_816);  sub_314 = unsqueeze_816 = None
    sub_316: "f32[8, 240, 56, 56]" = torch.ops.aten.sub.Tensor(mul_1109, mul_1116);  mul_1109 = mul_1116 = None
    sub_317: "f32[8, 240, 56, 56]" = torch.ops.aten.sub.Tensor(sub_316, unsqueeze_813);  sub_316 = unsqueeze_813 = None
    mul_1117: "f32[8, 240, 56, 56]" = torch.ops.aten.mul.Tensor(sub_317, unsqueeze_819);  sub_317 = unsqueeze_819 = None
    mul_1118: "f32[240]" = torch.ops.aten.mul.Tensor(sum_115, squeeze_28);  sum_115 = squeeze_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_139 = torch.ops.aten.convolution_backward.default(mul_1117, add_46, primals_132, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1117 = add_46 = primals_132 = None
    getitem_853: "f32[8, 40, 56, 56]" = convolution_backward_139[0]
    getitem_854: "f32[240, 40, 1, 1]" = convolution_backward_139[1];  convolution_backward_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_116: "f32[40]" = torch.ops.aten.sum.dim_IntList(getitem_853, [0, 2, 3])
    sub_318: "f32[8, 40, 56, 56]" = torch.ops.aten.sub.Tensor(cat_4, unsqueeze_822);  cat_4 = unsqueeze_822 = None
    mul_1119: "f32[8, 40, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_853, sub_318)
    sum_117: "f32[40]" = torch.ops.aten.sum.dim_IntList(mul_1119, [0, 2, 3]);  mul_1119 = None
    mul_1120: "f32[40]" = torch.ops.aten.mul.Tensor(sum_116, 3.985969387755102e-05)
    unsqueeze_823: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_1120, 0);  mul_1120 = None
    unsqueeze_824: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_823, 2);  unsqueeze_823 = None
    unsqueeze_825: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_824, 3);  unsqueeze_824 = None
    mul_1121: "f32[40]" = torch.ops.aten.mul.Tensor(sum_117, 3.985969387755102e-05)
    mul_1122: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
    mul_1123: "f32[40]" = torch.ops.aten.mul.Tensor(mul_1121, mul_1122);  mul_1121 = mul_1122 = None
    unsqueeze_826: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_1123, 0);  mul_1123 = None
    unsqueeze_827: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_826, 2);  unsqueeze_826 = None
    unsqueeze_828: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_827, 3);  unsqueeze_827 = None
    mul_1124: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_25, primals_17);  primals_17 = None
    unsqueeze_829: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_1124, 0);  mul_1124 = None
    unsqueeze_830: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_829, 2);  unsqueeze_829 = None
    unsqueeze_831: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_830, 3);  unsqueeze_830 = None
    mul_1125: "f32[8, 40, 56, 56]" = torch.ops.aten.mul.Tensor(sub_318, unsqueeze_828);  sub_318 = unsqueeze_828 = None
    sub_320: "f32[8, 40, 56, 56]" = torch.ops.aten.sub.Tensor(getitem_853, mul_1125);  mul_1125 = None
    sub_321: "f32[8, 40, 56, 56]" = torch.ops.aten.sub.Tensor(sub_320, unsqueeze_825);  sub_320 = unsqueeze_825 = None
    mul_1126: "f32[8, 40, 56, 56]" = torch.ops.aten.mul.Tensor(sub_321, unsqueeze_831);  sub_321 = unsqueeze_831 = None
    mul_1127: "f32[40]" = torch.ops.aten.mul.Tensor(sum_117, squeeze_25);  sum_117 = squeeze_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    slice_96: "f32[8, 20, 56, 56]" = torch.ops.aten.slice.Tensor(mul_1126, 1, 0, 20)
    slice_97: "f32[8, 20, 56, 56]" = torch.ops.aten.slice.Tensor(mul_1126, 1, 20, 40);  mul_1126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_backward_140 = torch.ops.aten.convolution_backward.default(slice_97, getitem_43, primals_131, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_97 = getitem_43 = primals_131 = None
    getitem_856: "f32[8, 60, 56, 56]" = convolution_backward_140[0]
    getitem_857: "f32[20, 60, 1, 1]" = convolution_backward_140[1];  convolution_backward_140 = None
    convolution_backward_141 = torch.ops.aten.convolution_backward.default(slice_96, getitem_40, primals_130, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_96 = getitem_40 = primals_130 = None
    getitem_859: "f32[8, 60, 56, 56]" = convolution_backward_141[0]
    getitem_860: "f32[20, 60, 1, 1]" = convolution_backward_141[1];  convolution_backward_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    cat_77: "f32[8, 120, 56, 56]" = torch.ops.aten.cat.default([getitem_859, getitem_856], 1);  getitem_859 = getitem_856 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    where_1: "f32[8, 120, 56, 56]" = torch.ops.aten.where.self(le_1, full_default, cat_77);  le_1 = cat_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_118: "f32[120]" = torch.ops.aten.sum.dim_IntList(where_1, [0, 2, 3])
    sub_322: "f32[8, 120, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_834);  convolution_12 = unsqueeze_834 = None
    mul_1128: "f32[8, 120, 56, 56]" = torch.ops.aten.mul.Tensor(where_1, sub_322)
    sum_119: "f32[120]" = torch.ops.aten.sum.dim_IntList(mul_1128, [0, 2, 3]);  mul_1128 = None
    mul_1129: "f32[120]" = torch.ops.aten.mul.Tensor(sum_118, 3.985969387755102e-05)
    unsqueeze_835: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1129, 0);  mul_1129 = None
    unsqueeze_836: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_835, 2);  unsqueeze_835 = None
    unsqueeze_837: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_836, 3);  unsqueeze_836 = None
    mul_1130: "f32[120]" = torch.ops.aten.mul.Tensor(sum_119, 3.985969387755102e-05)
    mul_1131: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_22, squeeze_22)
    mul_1132: "f32[120]" = torch.ops.aten.mul.Tensor(mul_1130, mul_1131);  mul_1130 = mul_1131 = None
    unsqueeze_838: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1132, 0);  mul_1132 = None
    unsqueeze_839: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_838, 2);  unsqueeze_838 = None
    unsqueeze_840: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_839, 3);  unsqueeze_839 = None
    mul_1133: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_22, primals_15);  primals_15 = None
    unsqueeze_841: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1133, 0);  mul_1133 = None
    unsqueeze_842: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_841, 2);  unsqueeze_841 = None
    unsqueeze_843: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_842, 3);  unsqueeze_842 = None
    mul_1134: "f32[8, 120, 56, 56]" = torch.ops.aten.mul.Tensor(sub_322, unsqueeze_840);  sub_322 = unsqueeze_840 = None
    sub_324: "f32[8, 120, 56, 56]" = torch.ops.aten.sub.Tensor(where_1, mul_1134);  where_1 = mul_1134 = None
    sub_325: "f32[8, 120, 56, 56]" = torch.ops.aten.sub.Tensor(sub_324, unsqueeze_837);  sub_324 = unsqueeze_837 = None
    mul_1135: "f32[8, 120, 56, 56]" = torch.ops.aten.mul.Tensor(sub_325, unsqueeze_843);  sub_325 = unsqueeze_843 = None
    mul_1136: "f32[120]" = torch.ops.aten.mul.Tensor(sum_119, squeeze_22);  sum_119 = squeeze_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_142 = torch.ops.aten.convolution_backward.default(mul_1135, relu_4, primals_129, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 120, [True, True, False]);  mul_1135 = primals_129 = None
    getitem_862: "f32[8, 120, 56, 56]" = convolution_backward_142[0]
    getitem_863: "f32[120, 1, 3, 3]" = convolution_backward_142[1];  convolution_backward_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_46: "f32[8, 120, 56, 56]" = torch.ops.aten.alias.default(relu_4);  relu_4 = None
    alias_47: "f32[8, 120, 56, 56]" = torch.ops.aten.alias.default(alias_46);  alias_46 = None
    le_2: "b8[8, 120, 56, 56]" = torch.ops.aten.le.Scalar(alias_47, 0);  alias_47 = None
    where_2: "f32[8, 120, 56, 56]" = torch.ops.aten.where.self(le_2, full_default, getitem_862);  le_2 = getitem_862 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_120: "f32[120]" = torch.ops.aten.sum.dim_IntList(where_2, [0, 2, 3])
    sub_326: "f32[8, 120, 56, 56]" = torch.ops.aten.sub.Tensor(cat_3, unsqueeze_846);  cat_3 = unsqueeze_846 = None
    mul_1137: "f32[8, 120, 56, 56]" = torch.ops.aten.mul.Tensor(where_2, sub_326)
    sum_121: "f32[120]" = torch.ops.aten.sum.dim_IntList(mul_1137, [0, 2, 3]);  mul_1137 = None
    mul_1138: "f32[120]" = torch.ops.aten.mul.Tensor(sum_120, 3.985969387755102e-05)
    unsqueeze_847: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1138, 0);  mul_1138 = None
    unsqueeze_848: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_847, 2);  unsqueeze_847 = None
    unsqueeze_849: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_848, 3);  unsqueeze_848 = None
    mul_1139: "f32[120]" = torch.ops.aten.mul.Tensor(sum_121, 3.985969387755102e-05)
    mul_1140: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_19, squeeze_19)
    mul_1141: "f32[120]" = torch.ops.aten.mul.Tensor(mul_1139, mul_1140);  mul_1139 = mul_1140 = None
    unsqueeze_850: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1141, 0);  mul_1141 = None
    unsqueeze_851: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_850, 2);  unsqueeze_850 = None
    unsqueeze_852: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_851, 3);  unsqueeze_851 = None
    mul_1142: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_19, primals_13);  primals_13 = None
    unsqueeze_853: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1142, 0);  mul_1142 = None
    unsqueeze_854: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_853, 2);  unsqueeze_853 = None
    unsqueeze_855: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_854, 3);  unsqueeze_854 = None
    mul_1143: "f32[8, 120, 56, 56]" = torch.ops.aten.mul.Tensor(sub_326, unsqueeze_852);  sub_326 = unsqueeze_852 = None
    sub_328: "f32[8, 120, 56, 56]" = torch.ops.aten.sub.Tensor(where_2, mul_1143);  where_2 = mul_1143 = None
    sub_329: "f32[8, 120, 56, 56]" = torch.ops.aten.sub.Tensor(sub_328, unsqueeze_849);  sub_328 = unsqueeze_849 = None
    mul_1144: "f32[8, 120, 56, 56]" = torch.ops.aten.mul.Tensor(sub_329, unsqueeze_855);  sub_329 = unsqueeze_855 = None
    mul_1145: "f32[120]" = torch.ops.aten.mul.Tensor(sum_121, squeeze_19);  sum_121 = squeeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    slice_98: "f32[8, 60, 56, 56]" = torch.ops.aten.slice.Tensor(mul_1144, 1, 0, 60)
    slice_99: "f32[8, 60, 56, 56]" = torch.ops.aten.slice.Tensor(mul_1144, 1, 60, 120);  mul_1144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_backward_143 = torch.ops.aten.convolution_backward.default(slice_99, getitem_33, primals_128, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_99 = getitem_33 = primals_128 = None
    getitem_865: "f32[8, 20, 56, 56]" = convolution_backward_143[0]
    getitem_866: "f32[60, 20, 1, 1]" = convolution_backward_143[1];  convolution_backward_143 = None
    convolution_backward_144 = torch.ops.aten.convolution_backward.default(slice_98, getitem_32, primals_127, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_98 = getitem_32 = primals_127 = None
    getitem_868: "f32[8, 20, 56, 56]" = convolution_backward_144[0]
    getitem_869: "f32[60, 20, 1, 1]" = convolution_backward_144[1];  convolution_backward_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    cat_78: "f32[8, 40, 56, 56]" = torch.ops.aten.cat.default([getitem_868, getitem_865], 1);  getitem_868 = getitem_865 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    add_380: "f32[8, 40, 56, 56]" = torch.ops.aten.add.Tensor(getitem_853, cat_78);  getitem_853 = cat_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_122: "f32[40]" = torch.ops.aten.sum.dim_IntList(add_380, [0, 2, 3])
    sub_330: "f32[8, 40, 56, 56]" = torch.ops.aten.sub.Tensor(cat_2, unsqueeze_858);  cat_2 = unsqueeze_858 = None
    mul_1146: "f32[8, 40, 56, 56]" = torch.ops.aten.mul.Tensor(add_380, sub_330)
    sum_123: "f32[40]" = torch.ops.aten.sum.dim_IntList(mul_1146, [0, 2, 3]);  mul_1146 = None
    mul_1147: "f32[40]" = torch.ops.aten.mul.Tensor(sum_122, 3.985969387755102e-05)
    unsqueeze_859: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_1147, 0);  mul_1147 = None
    unsqueeze_860: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_859, 2);  unsqueeze_859 = None
    unsqueeze_861: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_860, 3);  unsqueeze_860 = None
    mul_1148: "f32[40]" = torch.ops.aten.mul.Tensor(sum_123, 3.985969387755102e-05)
    mul_1149: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_16, squeeze_16)
    mul_1150: "f32[40]" = torch.ops.aten.mul.Tensor(mul_1148, mul_1149);  mul_1148 = mul_1149 = None
    unsqueeze_862: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_1150, 0);  mul_1150 = None
    unsqueeze_863: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_862, 2);  unsqueeze_862 = None
    unsqueeze_864: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_863, 3);  unsqueeze_863 = None
    mul_1151: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_16, primals_11);  primals_11 = None
    unsqueeze_865: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_1151, 0);  mul_1151 = None
    unsqueeze_866: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_865, 2);  unsqueeze_865 = None
    unsqueeze_867: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_866, 3);  unsqueeze_866 = None
    mul_1152: "f32[8, 40, 56, 56]" = torch.ops.aten.mul.Tensor(sub_330, unsqueeze_864);  sub_330 = unsqueeze_864 = None
    sub_332: "f32[8, 40, 56, 56]" = torch.ops.aten.sub.Tensor(add_380, mul_1152);  add_380 = mul_1152 = None
    sub_333: "f32[8, 40, 56, 56]" = torch.ops.aten.sub.Tensor(sub_332, unsqueeze_861);  sub_332 = unsqueeze_861 = None
    mul_1153: "f32[8, 40, 56, 56]" = torch.ops.aten.mul.Tensor(sub_333, unsqueeze_867);  sub_333 = unsqueeze_867 = None
    mul_1154: "f32[40]" = torch.ops.aten.mul.Tensor(sum_123, squeeze_16);  sum_123 = squeeze_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    slice_100: "f32[8, 20, 56, 56]" = torch.ops.aten.slice.Tensor(mul_1153, 1, 0, 20)
    slice_101: "f32[8, 20, 56, 56]" = torch.ops.aten.slice.Tensor(mul_1153, 1, 20, 40);  mul_1153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_backward_145 = torch.ops.aten.convolution_backward.default(slice_101, getitem_29, primals_126, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_101 = getitem_29 = primals_126 = None
    getitem_871: "f32[8, 96, 56, 56]" = convolution_backward_145[0]
    getitem_872: "f32[20, 96, 1, 1]" = convolution_backward_145[1];  convolution_backward_145 = None
    convolution_backward_146 = torch.ops.aten.convolution_backward.default(slice_100, getitem_26, primals_125, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_100 = getitem_26 = primals_125 = None
    getitem_874: "f32[8, 96, 56, 56]" = convolution_backward_146[0]
    getitem_875: "f32[20, 96, 1, 1]" = convolution_backward_146[1];  convolution_backward_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    cat_79: "f32[8, 192, 56, 56]" = torch.ops.aten.cat.default([getitem_874, getitem_871], 1);  getitem_874 = getitem_871 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    where_3: "f32[8, 192, 56, 56]" = torch.ops.aten.where.self(le_3, full_default, cat_79);  le_3 = cat_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_124: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_3, [0, 2, 3])
    sub_334: "f32[8, 192, 56, 56]" = torch.ops.aten.sub.Tensor(cat_1, unsqueeze_870);  cat_1 = unsqueeze_870 = None
    mul_1155: "f32[8, 192, 56, 56]" = torch.ops.aten.mul.Tensor(where_3, sub_334)
    sum_125: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_1155, [0, 2, 3]);  mul_1155 = None
    mul_1156: "f32[192]" = torch.ops.aten.mul.Tensor(sum_124, 3.985969387755102e-05)
    unsqueeze_871: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_1156, 0);  mul_1156 = None
    unsqueeze_872: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_871, 2);  unsqueeze_871 = None
    unsqueeze_873: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_872, 3);  unsqueeze_872 = None
    mul_1157: "f32[192]" = torch.ops.aten.mul.Tensor(sum_125, 3.985969387755102e-05)
    mul_1158: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
    mul_1159: "f32[192]" = torch.ops.aten.mul.Tensor(mul_1157, mul_1158);  mul_1157 = mul_1158 = None
    unsqueeze_874: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_1159, 0);  mul_1159 = None
    unsqueeze_875: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_874, 2);  unsqueeze_874 = None
    unsqueeze_876: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_875, 3);  unsqueeze_875 = None
    mul_1160: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_13, primals_9);  primals_9 = None
    unsqueeze_877: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_1160, 0);  mul_1160 = None
    unsqueeze_878: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_877, 2);  unsqueeze_877 = None
    unsqueeze_879: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_878, 3);  unsqueeze_878 = None
    mul_1161: "f32[8, 192, 56, 56]" = torch.ops.aten.mul.Tensor(sub_334, unsqueeze_876);  sub_334 = unsqueeze_876 = None
    sub_336: "f32[8, 192, 56, 56]" = torch.ops.aten.sub.Tensor(where_3, mul_1161);  where_3 = mul_1161 = None
    sub_337: "f32[8, 192, 56, 56]" = torch.ops.aten.sub.Tensor(sub_336, unsqueeze_873);  sub_336 = unsqueeze_873 = None
    mul_1162: "f32[8, 192, 56, 56]" = torch.ops.aten.mul.Tensor(sub_337, unsqueeze_879);  sub_337 = unsqueeze_879 = None
    mul_1163: "f32[192]" = torch.ops.aten.mul.Tensor(sum_125, squeeze_13);  sum_125 = squeeze_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    slice_102: "f32[8, 64, 56, 56]" = torch.ops.aten.slice.Tensor(mul_1162, 1, 0, 64)
    slice_103: "f32[8, 64, 56, 56]" = torch.ops.aten.slice.Tensor(mul_1162, 1, 64, 128)
    slice_104: "f32[8, 64, 56, 56]" = torch.ops.aten.slice.Tensor(mul_1162, 1, 128, 192);  mul_1162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_backward_147 = torch.ops.aten.convolution_backward.default(slice_104, getitem_21, primals_124, [0], [2, 2], [3, 3], [1, 1], False, [0, 0], 64, [True, True, False]);  slice_104 = getitem_21 = primals_124 = None
    getitem_877: "f32[8, 64, 112, 112]" = convolution_backward_147[0]
    getitem_878: "f32[64, 1, 7, 7]" = convolution_backward_147[1];  convolution_backward_147 = None
    convolution_backward_148 = torch.ops.aten.convolution_backward.default(slice_103, getitem_17, primals_123, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 64, [True, True, False]);  slice_103 = getitem_17 = primals_123 = None
    getitem_880: "f32[8, 64, 112, 112]" = convolution_backward_148[0]
    getitem_881: "f32[64, 1, 5, 5]" = convolution_backward_148[1];  convolution_backward_148 = None
    convolution_backward_149 = torch.ops.aten.convolution_backward.default(slice_102, getitem_13, primals_122, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 64, [True, True, False]);  slice_102 = getitem_13 = primals_122 = None
    getitem_883: "f32[8, 64, 112, 112]" = convolution_backward_149[0]
    getitem_884: "f32[64, 1, 3, 3]" = convolution_backward_149[1];  convolution_backward_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    cat_80: "f32[8, 192, 112, 112]" = torch.ops.aten.cat.default([getitem_883, getitem_880, getitem_877], 1);  getitem_883 = getitem_880 = getitem_877 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    where_4: "f32[8, 192, 112, 112]" = torch.ops.aten.where.self(le_4, full_default, cat_80);  le_4 = cat_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_126: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_4, [0, 2, 3])
    sub_338: "f32[8, 192, 112, 112]" = torch.ops.aten.sub.Tensor(cat, unsqueeze_882);  cat = unsqueeze_882 = None
    mul_1164: "f32[8, 192, 112, 112]" = torch.ops.aten.mul.Tensor(where_4, sub_338)
    sum_127: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_1164, [0, 2, 3]);  mul_1164 = None
    mul_1165: "f32[192]" = torch.ops.aten.mul.Tensor(sum_126, 9.964923469387754e-06)
    unsqueeze_883: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_1165, 0);  mul_1165 = None
    unsqueeze_884: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_883, 2);  unsqueeze_883 = None
    unsqueeze_885: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_884, 3);  unsqueeze_884 = None
    mul_1166: "f32[192]" = torch.ops.aten.mul.Tensor(sum_127, 9.964923469387754e-06)
    mul_1167: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_10, squeeze_10)
    mul_1168: "f32[192]" = torch.ops.aten.mul.Tensor(mul_1166, mul_1167);  mul_1166 = mul_1167 = None
    unsqueeze_886: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_1168, 0);  mul_1168 = None
    unsqueeze_887: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_886, 2);  unsqueeze_886 = None
    unsqueeze_888: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_887, 3);  unsqueeze_887 = None
    mul_1169: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_10, primals_7);  primals_7 = None
    unsqueeze_889: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_1169, 0);  mul_1169 = None
    unsqueeze_890: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_889, 2);  unsqueeze_889 = None
    unsqueeze_891: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_890, 3);  unsqueeze_890 = None
    mul_1170: "f32[8, 192, 112, 112]" = torch.ops.aten.mul.Tensor(sub_338, unsqueeze_888);  sub_338 = unsqueeze_888 = None
    sub_340: "f32[8, 192, 112, 112]" = torch.ops.aten.sub.Tensor(where_4, mul_1170);  where_4 = mul_1170 = None
    sub_341: "f32[8, 192, 112, 112]" = torch.ops.aten.sub.Tensor(sub_340, unsqueeze_885);  sub_340 = unsqueeze_885 = None
    mul_1171: "f32[8, 192, 112, 112]" = torch.ops.aten.mul.Tensor(sub_341, unsqueeze_891);  sub_341 = unsqueeze_891 = None
    mul_1172: "f32[192]" = torch.ops.aten.mul.Tensor(sum_127, squeeze_10);  sum_127 = squeeze_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    slice_105: "f32[8, 96, 112, 112]" = torch.ops.aten.slice.Tensor(mul_1171, 1, 0, 96)
    slice_106: "f32[8, 96, 112, 112]" = torch.ops.aten.slice.Tensor(mul_1171, 1, 96, 192);  mul_1171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_backward_150 = torch.ops.aten.convolution_backward.default(slice_106, getitem_7, primals_121, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_106 = getitem_7 = primals_121 = None
    getitem_886: "f32[8, 16, 112, 112]" = convolution_backward_150[0]
    getitem_887: "f32[96, 16, 1, 1]" = convolution_backward_150[1];  convolution_backward_150 = None
    convolution_backward_151 = torch.ops.aten.convolution_backward.default(slice_105, getitem_6, primals_120, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_105 = getitem_6 = primals_120 = None
    getitem_889: "f32[8, 16, 112, 112]" = convolution_backward_151[0]
    getitem_890: "f32[96, 16, 1, 1]" = convolution_backward_151[1];  convolution_backward_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    cat_81: "f32[8, 32, 112, 112]" = torch.ops.aten.cat.default([getitem_889, getitem_886], 1);  getitem_889 = getitem_886 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_128: "f32[32]" = torch.ops.aten.sum.dim_IntList(cat_81, [0, 2, 3])
    sub_342: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_894);  convolution_2 = unsqueeze_894 = None
    mul_1173: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(cat_81, sub_342)
    sum_129: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_1173, [0, 2, 3]);  mul_1173 = None
    mul_1174: "f32[32]" = torch.ops.aten.mul.Tensor(sum_128, 9.964923469387754e-06)
    unsqueeze_895: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1174, 0);  mul_1174 = None
    unsqueeze_896: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_895, 2);  unsqueeze_895 = None
    unsqueeze_897: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_896, 3);  unsqueeze_896 = None
    mul_1175: "f32[32]" = torch.ops.aten.mul.Tensor(sum_129, 9.964923469387754e-06)
    mul_1176: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
    mul_1177: "f32[32]" = torch.ops.aten.mul.Tensor(mul_1175, mul_1176);  mul_1175 = mul_1176 = None
    unsqueeze_898: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1177, 0);  mul_1177 = None
    unsqueeze_899: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_898, 2);  unsqueeze_898 = None
    unsqueeze_900: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_899, 3);  unsqueeze_899 = None
    mul_1178: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_7, primals_5);  primals_5 = None
    unsqueeze_901: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1178, 0);  mul_1178 = None
    unsqueeze_902: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_901, 2);  unsqueeze_901 = None
    unsqueeze_903: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_902, 3);  unsqueeze_902 = None
    mul_1179: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_342, unsqueeze_900);  sub_342 = unsqueeze_900 = None
    sub_344: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(cat_81, mul_1179);  mul_1179 = None
    sub_345: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(sub_344, unsqueeze_897);  sub_344 = unsqueeze_897 = None
    mul_1180: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_345, unsqueeze_903);  sub_345 = unsqueeze_903 = None
    mul_1181: "f32[32]" = torch.ops.aten.mul.Tensor(sum_129, squeeze_7);  sum_129 = squeeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:126, code: x = self.conv_pw(x)
    convolution_backward_152 = torch.ops.aten.convolution_backward.default(mul_1180, relu_1, primals_119, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1180 = primals_119 = None
    getitem_892: "f32[8, 32, 112, 112]" = convolution_backward_152[0]
    getitem_893: "f32[32, 32, 1, 1]" = convolution_backward_152[1];  convolution_backward_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_55: "f32[8, 32, 112, 112]" = torch.ops.aten.alias.default(relu_1);  relu_1 = None
    alias_56: "f32[8, 32, 112, 112]" = torch.ops.aten.alias.default(alias_55);  alias_55 = None
    le_5: "b8[8, 32, 112, 112]" = torch.ops.aten.le.Scalar(alias_56, 0);  alias_56 = None
    where_5: "f32[8, 32, 112, 112]" = torch.ops.aten.where.self(le_5, full_default, getitem_892);  le_5 = getitem_892 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_130: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_5, [0, 2, 3])
    sub_346: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_906);  convolution_1 = unsqueeze_906 = None
    mul_1182: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(where_5, sub_346)
    sum_131: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_1182, [0, 2, 3]);  mul_1182 = None
    mul_1183: "f32[32]" = torch.ops.aten.mul.Tensor(sum_130, 9.964923469387754e-06)
    unsqueeze_907: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1183, 0);  mul_1183 = None
    unsqueeze_908: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_907, 2);  unsqueeze_907 = None
    unsqueeze_909: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_908, 3);  unsqueeze_908 = None
    mul_1184: "f32[32]" = torch.ops.aten.mul.Tensor(sum_131, 9.964923469387754e-06)
    mul_1185: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
    mul_1186: "f32[32]" = torch.ops.aten.mul.Tensor(mul_1184, mul_1185);  mul_1184 = mul_1185 = None
    unsqueeze_910: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1186, 0);  mul_1186 = None
    unsqueeze_911: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_910, 2);  unsqueeze_910 = None
    unsqueeze_912: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_911, 3);  unsqueeze_911 = None
    mul_1187: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_4, primals_3);  primals_3 = None
    unsqueeze_913: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1187, 0);  mul_1187 = None
    unsqueeze_914: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_913, 2);  unsqueeze_913 = None
    unsqueeze_915: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_914, 3);  unsqueeze_914 = None
    mul_1188: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_346, unsqueeze_912);  sub_346 = unsqueeze_912 = None
    sub_348: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(where_5, mul_1188);  where_5 = mul_1188 = None
    sub_349: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(sub_348, unsqueeze_909);  sub_348 = unsqueeze_909 = None
    mul_1189: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_349, unsqueeze_915);  sub_349 = unsqueeze_915 = None
    mul_1190: "f32[32]" = torch.ops.aten.mul.Tensor(sum_131, squeeze_4);  sum_131 = squeeze_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:123, code: x = self.conv_dw(x)
    convolution_backward_153 = torch.ops.aten.convolution_backward.default(mul_1189, relu, primals_118, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_1189 = primals_118 = None
    getitem_895: "f32[8, 32, 112, 112]" = convolution_backward_153[0]
    getitem_896: "f32[32, 1, 3, 3]" = convolution_backward_153[1];  convolution_backward_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:123, code: x = self.conv_dw(x)
    add_381: "f32[8, 32, 112, 112]" = torch.ops.aten.add.Tensor(cat_81, getitem_895);  cat_81 = getitem_895 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_58: "f32[8, 32, 112, 112]" = torch.ops.aten.alias.default(relu);  relu = None
    alias_59: "f32[8, 32, 112, 112]" = torch.ops.aten.alias.default(alias_58);  alias_58 = None
    le_6: "b8[8, 32, 112, 112]" = torch.ops.aten.le.Scalar(alias_59, 0);  alias_59 = None
    where_6: "f32[8, 32, 112, 112]" = torch.ops.aten.where.self(le_6, full_default, add_381);  le_6 = full_default = add_381 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_132: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_6, [0, 2, 3])
    sub_350: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_918);  convolution = unsqueeze_918 = None
    mul_1191: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(where_6, sub_350)
    sum_133: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_1191, [0, 2, 3]);  mul_1191 = None
    mul_1192: "f32[32]" = torch.ops.aten.mul.Tensor(sum_132, 9.964923469387754e-06)
    unsqueeze_919: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1192, 0);  mul_1192 = None
    unsqueeze_920: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_919, 2);  unsqueeze_919 = None
    unsqueeze_921: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_920, 3);  unsqueeze_920 = None
    mul_1193: "f32[32]" = torch.ops.aten.mul.Tensor(sum_133, 9.964923469387754e-06)
    mul_1194: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
    mul_1195: "f32[32]" = torch.ops.aten.mul.Tensor(mul_1193, mul_1194);  mul_1193 = mul_1194 = None
    unsqueeze_922: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1195, 0);  mul_1195 = None
    unsqueeze_923: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_922, 2);  unsqueeze_922 = None
    unsqueeze_924: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_923, 3);  unsqueeze_923 = None
    mul_1196: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_1, primals_1);  primals_1 = None
    unsqueeze_925: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1196, 0);  mul_1196 = None
    unsqueeze_926: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_925, 2);  unsqueeze_925 = None
    unsqueeze_927: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_926, 3);  unsqueeze_926 = None
    mul_1197: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_350, unsqueeze_924);  sub_350 = unsqueeze_924 = None
    sub_352: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(where_6, mul_1197);  where_6 = mul_1197 = None
    sub_353: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(sub_352, unsqueeze_921);  sub_352 = unsqueeze_921 = None
    mul_1198: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_353, unsqueeze_927);  sub_353 = unsqueeze_927 = None
    mul_1199: "f32[32]" = torch.ops.aten.mul.Tensor(sum_133, squeeze_1);  sum_133 = squeeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/efficientnet.py:162, code: x = self.conv_stem(x)
    convolution_backward_154 = torch.ops.aten.convolution_backward.default(mul_1198, primals_480, primals_117, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_1198 = primals_480 = primals_117 = None
    getitem_899: "f32[32, 3, 3, 3]" = convolution_backward_154[1];  convolution_backward_154 = None
    return [mul_1199, sum_132, mul_1190, sum_130, mul_1181, sum_128, mul_1172, sum_126, mul_1163, sum_124, mul_1154, sum_122, mul_1145, sum_120, mul_1136, sum_118, mul_1127, sum_116, mul_1118, sum_114, mul_1106, sum_112, mul_1087, sum_109, mul_1078, sum_107, mul_1066, sum_105, mul_1047, sum_102, mul_1038, sum_100, mul_1026, sum_98, mul_1007, sum_95, mul_998, sum_93, mul_986, sum_91, mul_967, sum_88, mul_958, sum_86, mul_946, sum_84, mul_927, sum_81, mul_918, sum_79, mul_906, sum_77, mul_887, sum_74, mul_878, sum_72, mul_866, sum_70, mul_847, sum_67, mul_838, sum_65, mul_826, sum_63, mul_807, sum_60, mul_798, sum_58, mul_786, sum_56, mul_767, sum_53, mul_758, sum_51, mul_746, sum_49, mul_727, sum_46, mul_718, sum_44, mul_706, sum_42, mul_687, sum_39, mul_678, sum_37, mul_666, sum_35, mul_647, sum_32, mul_638, sum_30, mul_626, sum_28, mul_607, sum_25, mul_598, sum_23, mul_586, sum_21, mul_567, sum_18, mul_558, sum_16, mul_546, sum_14, mul_527, sum_11, mul_518, sum_9, mul_506, sum_7, mul_487, sum_4, mul_478, sum_2, getitem_899, getitem_896, getitem_893, getitem_890, getitem_887, getitem_884, getitem_881, getitem_878, getitem_875, getitem_872, getitem_869, getitem_866, getitem_863, getitem_860, getitem_857, getitem_854, getitem_851, getitem_848, getitem_845, getitem_842, getitem_839, getitem_840, getitem_836, getitem_837, getitem_833, getitem_830, getitem_827, getitem_824, getitem_821, getitem_818, getitem_819, getitem_815, getitem_816, getitem_812, getitem_809, getitem_806, getitem_803, getitem_800, getitem_797, getitem_794, getitem_795, getitem_791, getitem_792, getitem_788, getitem_785, getitem_782, getitem_779, getitem_776, getitem_773, getitem_770, getitem_771, getitem_767, getitem_768, getitem_764, getitem_761, getitem_758, getitem_755, getitem_752, getitem_749, getitem_746, getitem_747, getitem_743, getitem_744, getitem_740, getitem_737, getitem_734, getitem_731, getitem_728, getitem_725, getitem_722, getitem_719, getitem_720, getitem_716, getitem_717, getitem_713, getitem_710, getitem_707, getitem_704, getitem_701, getitem_698, getitem_695, getitem_692, getitem_689, getitem_690, getitem_686, getitem_687, getitem_683, getitem_680, getitem_677, getitem_674, getitem_671, getitem_668, getitem_665, getitem_662, getitem_659, getitem_660, getitem_656, getitem_657, getitem_653, getitem_650, getitem_647, getitem_644, getitem_641, getitem_642, getitem_638, getitem_639, getitem_635, getitem_632, getitem_629, getitem_626, getitem_623, getitem_620, getitem_617, getitem_614, getitem_615, getitem_611, getitem_612, getitem_608, getitem_605, getitem_602, getitem_599, getitem_596, getitem_593, getitem_590, getitem_587, getitem_584, getitem_585, getitem_581, getitem_582, getitem_578, getitem_575, getitem_572, getitem_569, getitem_566, getitem_563, getitem_560, getitem_557, getitem_554, getitem_555, getitem_551, getitem_552, getitem_548, getitem_545, getitem_542, getitem_539, getitem_536, getitem_533, getitem_530, getitem_527, getitem_528, getitem_524, getitem_525, getitem_521, getitem_518, getitem_515, getitem_512, getitem_509, getitem_506, getitem_503, getitem_504, getitem_500, getitem_501, getitem_497, getitem_494, getitem_491, getitem_488, getitem_485, getitem_482, getitem_479, getitem_476, getitem_477, getitem_473, getitem_474, getitem_470, getitem_467, getitem_464, getitem_461, getitem_458, getitem_455, getitem_452, getitem_449, getitem_450, getitem_446, getitem_447, getitem_443, getitem_440, getitem_437, permute_4, view_1, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
    