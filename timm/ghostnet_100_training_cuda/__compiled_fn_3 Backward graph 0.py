from __future__ import annotations



def forward(self, primals_1: "f32[960]", primals_5: "f32[16, 3, 3, 3]", primals_6: "f32[16]", primals_8: "f32[8, 16, 1, 1]", primals_9: "f32[8]", primals_11: "f32[8, 1, 3, 3]", primals_12: "f32[8]", primals_14: "f32[8, 16, 1, 1]", primals_15: "f32[8]", primals_17: "f32[8, 1, 3, 3]", primals_18: "f32[8]", primals_20: "f32[24, 16, 1, 1]", primals_21: "f32[24]", primals_23: "f32[24, 1, 3, 3]", primals_24: "f32[24]", primals_26: "f32[48, 1, 3, 3]", primals_27: "f32[48]", primals_29: "f32[12, 48, 1, 1]", primals_30: "f32[12]", primals_32: "f32[12, 1, 3, 3]", primals_33: "f32[12]", primals_35: "f32[16, 1, 3, 3]", primals_36: "f32[16]", primals_38: "f32[24, 16, 1, 1]", primals_39: "f32[24]", primals_41: "f32[36, 24, 1, 1]", primals_42: "f32[36]", primals_44: "f32[36, 1, 3, 3]", primals_45: "f32[36]", primals_47: "f32[12, 72, 1, 1]", primals_48: "f32[12]", primals_50: "f32[12, 1, 3, 3]", primals_51: "f32[12]", primals_53: "f32[36, 24, 1, 1]", primals_54: "f32[36]", primals_56: "f32[36, 1, 3, 3]", primals_57: "f32[36]", primals_59: "f32[72, 1, 5, 5]", primals_60: "f32[72]", primals_62: "f32[20, 72, 1, 1]", primals_64: "f32[72, 20, 1, 1]", primals_66: "f32[20, 72, 1, 1]", primals_67: "f32[20]", primals_69: "f32[20, 1, 3, 3]", primals_70: "f32[20]", primals_72: "f32[24, 1, 5, 5]", primals_73: "f32[24]", primals_75: "f32[40, 24, 1, 1]", primals_76: "f32[40]", primals_78: "f32[60, 40, 1, 1]", primals_79: "f32[60]", primals_81: "f32[60, 1, 3, 3]", primals_82: "f32[60]", primals_84: "f32[32, 120, 1, 1]", primals_86: "f32[120, 32, 1, 1]", primals_88: "f32[20, 120, 1, 1]", primals_89: "f32[20]", primals_91: "f32[20, 1, 3, 3]", primals_92: "f32[20]", primals_94: "f32[120, 40, 1, 1]", primals_95: "f32[120]", primals_97: "f32[120, 1, 3, 3]", primals_98: "f32[120]", primals_100: "f32[240, 1, 3, 3]", primals_101: "f32[240]", primals_103: "f32[40, 240, 1, 1]", primals_104: "f32[40]", primals_106: "f32[40, 1, 3, 3]", primals_107: "f32[40]", primals_109: "f32[40, 1, 3, 3]", primals_110: "f32[40]", primals_112: "f32[80, 40, 1, 1]", primals_113: "f32[80]", primals_115: "f32[100, 80, 1, 1]", primals_116: "f32[100]", primals_118: "f32[100, 1, 3, 3]", primals_119: "f32[100]", primals_121: "f32[40, 200, 1, 1]", primals_122: "f32[40]", primals_124: "f32[40, 1, 3, 3]", primals_125: "f32[40]", primals_127: "f32[92, 80, 1, 1]", primals_128: "f32[92]", primals_130: "f32[92, 1, 3, 3]", primals_131: "f32[92]", primals_133: "f32[40, 184, 1, 1]", primals_134: "f32[40]", primals_136: "f32[40, 1, 3, 3]", primals_137: "f32[40]", primals_139: "f32[92, 80, 1, 1]", primals_140: "f32[92]", primals_142: "f32[92, 1, 3, 3]", primals_143: "f32[92]", primals_145: "f32[40, 184, 1, 1]", primals_146: "f32[40]", primals_148: "f32[40, 1, 3, 3]", primals_149: "f32[40]", primals_151: "f32[240, 80, 1, 1]", primals_152: "f32[240]", primals_154: "f32[240, 1, 3, 3]", primals_155: "f32[240]", primals_157: "f32[120, 480, 1, 1]", primals_159: "f32[480, 120, 1, 1]", primals_161: "f32[56, 480, 1, 1]", primals_162: "f32[56]", primals_164: "f32[56, 1, 3, 3]", primals_165: "f32[56]", primals_167: "f32[80, 1, 3, 3]", primals_168: "f32[80]", primals_170: "f32[112, 80, 1, 1]", primals_171: "f32[112]", primals_173: "f32[336, 112, 1, 1]", primals_174: "f32[336]", primals_176: "f32[336, 1, 3, 3]", primals_177: "f32[336]", primals_179: "f32[168, 672, 1, 1]", primals_181: "f32[672, 168, 1, 1]", primals_183: "f32[56, 672, 1, 1]", primals_184: "f32[56]", primals_186: "f32[56, 1, 3, 3]", primals_187: "f32[56]", primals_189: "f32[336, 112, 1, 1]", primals_190: "f32[336]", primals_192: "f32[336, 1, 3, 3]", primals_193: "f32[336]", primals_195: "f32[672, 1, 5, 5]", primals_196: "f32[672]", primals_198: "f32[168, 672, 1, 1]", primals_200: "f32[672, 168, 1, 1]", primals_202: "f32[80, 672, 1, 1]", primals_203: "f32[80]", primals_205: "f32[80, 1, 3, 3]", primals_206: "f32[80]", primals_208: "f32[112, 1, 5, 5]", primals_209: "f32[112]", primals_211: "f32[160, 112, 1, 1]", primals_212: "f32[160]", primals_214: "f32[480, 160, 1, 1]", primals_215: "f32[480]", primals_217: "f32[480, 1, 3, 3]", primals_218: "f32[480]", primals_220: "f32[80, 960, 1, 1]", primals_221: "f32[80]", primals_223: "f32[80, 1, 3, 3]", primals_224: "f32[80]", primals_226: "f32[480, 160, 1, 1]", primals_227: "f32[480]", primals_229: "f32[480, 1, 3, 3]", primals_230: "f32[480]", primals_232: "f32[240, 960, 1, 1]", primals_234: "f32[960, 240, 1, 1]", primals_236: "f32[80, 960, 1, 1]", primals_237: "f32[80]", primals_239: "f32[80, 1, 3, 3]", primals_240: "f32[80]", primals_242: "f32[480, 160, 1, 1]", primals_243: "f32[480]", primals_245: "f32[480, 1, 3, 3]", primals_246: "f32[480]", primals_248: "f32[80, 960, 1, 1]", primals_249: "f32[80]", primals_251: "f32[80, 1, 3, 3]", primals_252: "f32[80]", primals_254: "f32[480, 160, 1, 1]", primals_255: "f32[480]", primals_257: "f32[480, 1, 3, 3]", primals_258: "f32[480]", primals_260: "f32[240, 960, 1, 1]", primals_262: "f32[960, 240, 1, 1]", primals_264: "f32[80, 960, 1, 1]", primals_265: "f32[80]", primals_267: "f32[80, 1, 3, 3]", primals_268: "f32[80]", primals_270: "f32[960, 160, 1, 1]", primals_271: "f32[1280, 960, 1, 1]", primals_513: "f32[8, 3, 224, 224]", convolution: "f32[8, 16, 112, 112]", squeeze_1: "f32[16]", relu: "f32[8, 16, 112, 112]", convolution_1: "f32[8, 8, 112, 112]", squeeze_4: "f32[8]", relu_1: "f32[8, 8, 112, 112]", convolution_2: "f32[8, 8, 112, 112]", squeeze_7: "f32[8]", slice_3: "f32[8, 16, 112, 112]", convolution_3: "f32[8, 8, 112, 112]", squeeze_10: "f32[8]", add_19: "f32[8, 8, 112, 112]", convolution_4: "f32[8, 8, 112, 112]", squeeze_13: "f32[8]", slice_11: "f32[8, 16, 112, 112]", convolution_5: "f32[8, 24, 112, 112]", squeeze_16: "f32[24]", relu_3: "f32[8, 24, 112, 112]", convolution_6: "f32[8, 24, 112, 112]", squeeze_19: "f32[24]", slice_14: "f32[8, 48, 112, 112]", convolution_7: "f32[8, 48, 56, 56]", squeeze_22: "f32[48]", add_40: "f32[8, 48, 56, 56]", convolution_8: "f32[8, 12, 56, 56]", squeeze_25: "f32[12]", add_45: "f32[8, 12, 56, 56]", convolution_9: "f32[8, 12, 56, 56]", squeeze_28: "f32[12]", convolution_10: "f32[8, 16, 56, 56]", squeeze_31: "f32[16]", add_55: "f32[8, 16, 56, 56]", convolution_11: "f32[8, 24, 56, 56]", squeeze_34: "f32[24]", slice_22: "f32[8, 24, 56, 56]", convolution_12: "f32[8, 36, 56, 56]", squeeze_37: "f32[36]", relu_5: "f32[8, 36, 56, 56]", convolution_13: "f32[8, 36, 56, 56]", squeeze_40: "f32[36]", slice_25: "f32[8, 72, 56, 56]", convolution_14: "f32[8, 12, 56, 56]", squeeze_43: "f32[12]", add_76: "f32[8, 12, 56, 56]", convolution_15: "f32[8, 12, 56, 56]", squeeze_46: "f32[12]", slice_33: "f32[8, 24, 56, 56]", convolution_16: "f32[8, 36, 56, 56]", squeeze_49: "f32[36]", relu_7: "f32[8, 36, 56, 56]", convolution_17: "f32[8, 36, 56, 56]", squeeze_52: "f32[36]", slice_36: "f32[8, 72, 56, 56]", convolution_18: "f32[8, 72, 28, 28]", squeeze_55: "f32[72]", add_97: "f32[8, 72, 28, 28]", mean: "f32[8, 72, 1, 1]", relu_9: "f32[8, 20, 1, 1]", div: "f32[8, 72, 1, 1]", mul_133: "f32[8, 72, 28, 28]", convolution_21: "f32[8, 20, 28, 28]", squeeze_58: "f32[20]", add_103: "f32[8, 20, 28, 28]", convolution_22: "f32[8, 20, 28, 28]", squeeze_61: "f32[20]", convolution_23: "f32[8, 24, 28, 28]", squeeze_64: "f32[24]", add_113: "f32[8, 24, 28, 28]", convolution_24: "f32[8, 40, 28, 28]", squeeze_67: "f32[40]", slice_44: "f32[8, 40, 28, 28]", convolution_25: "f32[8, 60, 28, 28]", squeeze_70: "f32[60]", relu_10: "f32[8, 60, 28, 28]", convolution_26: "f32[8, 60, 28, 28]", squeeze_73: "f32[60]", cat_8: "f32[8, 120, 28, 28]", mean_1: "f32[8, 120, 1, 1]", relu_12: "f32[8, 32, 1, 1]", div_1: "f32[8, 120, 1, 1]", mul_176: "f32[8, 120, 28, 28]", convolution_29: "f32[8, 20, 28, 28]", squeeze_76: "f32[20]", add_135: "f32[8, 20, 28, 28]", convolution_30: "f32[8, 20, 28, 28]", squeeze_79: "f32[20]", slice_55: "f32[8, 40, 28, 28]", convolution_31: "f32[8, 120, 28, 28]", squeeze_82: "f32[120]", relu_13: "f32[8, 120, 28, 28]", convolution_32: "f32[8, 120, 28, 28]", squeeze_85: "f32[120]", slice_58: "f32[8, 240, 28, 28]", convolution_33: "f32[8, 240, 14, 14]", squeeze_88: "f32[240]", add_156: "f32[8, 240, 14, 14]", convolution_34: "f32[8, 40, 14, 14]", squeeze_91: "f32[40]", add_161: "f32[8, 40, 14, 14]", convolution_35: "f32[8, 40, 14, 14]", squeeze_94: "f32[40]", convolution_36: "f32[8, 40, 14, 14]", squeeze_97: "f32[40]", add_171: "f32[8, 40, 14, 14]", convolution_37: "f32[8, 80, 14, 14]", squeeze_100: "f32[80]", slice_66: "f32[8, 80, 14, 14]", convolution_38: "f32[8, 100, 14, 14]", squeeze_103: "f32[100]", relu_15: "f32[8, 100, 14, 14]", convolution_39: "f32[8, 100, 14, 14]", squeeze_106: "f32[100]", slice_69: "f32[8, 200, 14, 14]", convolution_40: "f32[8, 40, 14, 14]", squeeze_109: "f32[40]", add_192: "f32[8, 40, 14, 14]", convolution_41: "f32[8, 40, 14, 14]", squeeze_112: "f32[40]", slice_77: "f32[8, 80, 14, 14]", convolution_42: "f32[8, 92, 14, 14]", squeeze_115: "f32[92]", relu_17: "f32[8, 92, 14, 14]", convolution_43: "f32[8, 92, 14, 14]", squeeze_118: "f32[92]", slice_80: "f32[8, 184, 14, 14]", convolution_44: "f32[8, 40, 14, 14]", squeeze_121: "f32[40]", add_213: "f32[8, 40, 14, 14]", convolution_45: "f32[8, 40, 14, 14]", squeeze_124: "f32[40]", slice_88: "f32[8, 80, 14, 14]", convolution_46: "f32[8, 92, 14, 14]", squeeze_127: "f32[92]", relu_19: "f32[8, 92, 14, 14]", convolution_47: "f32[8, 92, 14, 14]", squeeze_130: "f32[92]", slice_91: "f32[8, 184, 14, 14]", convolution_48: "f32[8, 40, 14, 14]", squeeze_133: "f32[40]", add_234: "f32[8, 40, 14, 14]", convolution_49: "f32[8, 40, 14, 14]", squeeze_136: "f32[40]", slice_99: "f32[8, 80, 14, 14]", convolution_50: "f32[8, 240, 14, 14]", squeeze_139: "f32[240]", relu_21: "f32[8, 240, 14, 14]", convolution_51: "f32[8, 240, 14, 14]", squeeze_142: "f32[240]", cat_18: "f32[8, 480, 14, 14]", mean_2: "f32[8, 480, 1, 1]", relu_23: "f32[8, 120, 1, 1]", div_2: "f32[8, 480, 1, 1]", mul_338: "f32[8, 480, 14, 14]", convolution_54: "f32[8, 56, 14, 14]", squeeze_145: "f32[56]", add_256: "f32[8, 56, 14, 14]", convolution_55: "f32[8, 56, 14, 14]", squeeze_148: "f32[56]", convolution_56: "f32[8, 80, 14, 14]", squeeze_151: "f32[80]", add_266: "f32[8, 80, 14, 14]", convolution_57: "f32[8, 112, 14, 14]", squeeze_154: "f32[112]", slice_110: "f32[8, 112, 14, 14]", convolution_58: "f32[8, 336, 14, 14]", squeeze_157: "f32[336]", relu_24: "f32[8, 336, 14, 14]", convolution_59: "f32[8, 336, 14, 14]", squeeze_160: "f32[336]", cat_20: "f32[8, 672, 14, 14]", mean_3: "f32[8, 672, 1, 1]", relu_26: "f32[8, 168, 1, 1]", div_3: "f32[8, 672, 1, 1]", mul_381: "f32[8, 672, 14, 14]", convolution_62: "f32[8, 56, 14, 14]", squeeze_163: "f32[56]", add_288: "f32[8, 56, 14, 14]", convolution_63: "f32[8, 56, 14, 14]", squeeze_166: "f32[56]", slice_121: "f32[8, 112, 14, 14]", convolution_64: "f32[8, 336, 14, 14]", squeeze_169: "f32[336]", relu_27: "f32[8, 336, 14, 14]", convolution_65: "f32[8, 336, 14, 14]", squeeze_172: "f32[336]", slice_124: "f32[8, 672, 14, 14]", convolution_66: "f32[8, 672, 7, 7]", squeeze_175: "f32[672]", add_309: "f32[8, 672, 7, 7]", mean_4: "f32[8, 672, 1, 1]", relu_29: "f32[8, 168, 1, 1]", div_4: "f32[8, 672, 1, 1]", mul_417: "f32[8, 672, 7, 7]", convolution_69: "f32[8, 80, 7, 7]", squeeze_178: "f32[80]", add_315: "f32[8, 80, 7, 7]", convolution_70: "f32[8, 80, 7, 7]", squeeze_181: "f32[80]", convolution_71: "f32[8, 112, 7, 7]", squeeze_184: "f32[112]", add_325: "f32[8, 112, 7, 7]", convolution_72: "f32[8, 160, 7, 7]", squeeze_187: "f32[160]", slice_132: "f32[8, 160, 7, 7]", convolution_73: "f32[8, 480, 7, 7]", squeeze_190: "f32[480]", relu_30: "f32[8, 480, 7, 7]", convolution_74: "f32[8, 480, 7, 7]", squeeze_193: "f32[480]", slice_135: "f32[8, 960, 7, 7]", convolution_75: "f32[8, 80, 7, 7]", squeeze_196: "f32[80]", add_346: "f32[8, 80, 7, 7]", convolution_76: "f32[8, 80, 7, 7]", squeeze_199: "f32[80]", slice_143: "f32[8, 160, 7, 7]", convolution_77: "f32[8, 480, 7, 7]", squeeze_202: "f32[480]", relu_32: "f32[8, 480, 7, 7]", convolution_78: "f32[8, 480, 7, 7]", squeeze_205: "f32[480]", cat_26: "f32[8, 960, 7, 7]", mean_5: "f32[8, 960, 1, 1]", relu_34: "f32[8, 240, 1, 1]", div_5: "f32[8, 960, 1, 1]", mul_488: "f32[8, 960, 7, 7]", convolution_81: "f32[8, 80, 7, 7]", squeeze_208: "f32[80]", add_368: "f32[8, 80, 7, 7]", convolution_82: "f32[8, 80, 7, 7]", squeeze_211: "f32[80]", slice_154: "f32[8, 160, 7, 7]", convolution_83: "f32[8, 480, 7, 7]", squeeze_214: "f32[480]", relu_35: "f32[8, 480, 7, 7]", convolution_84: "f32[8, 480, 7, 7]", squeeze_217: "f32[480]", slice_157: "f32[8, 960, 7, 7]", convolution_85: "f32[8, 80, 7, 7]", squeeze_220: "f32[80]", add_389: "f32[8, 80, 7, 7]", convolution_86: "f32[8, 80, 7, 7]", squeeze_223: "f32[80]", slice_165: "f32[8, 160, 7, 7]", convolution_87: "f32[8, 480, 7, 7]", squeeze_226: "f32[480]", relu_37: "f32[8, 480, 7, 7]", convolution_88: "f32[8, 480, 7, 7]", squeeze_229: "f32[480]", cat_30: "f32[8, 960, 7, 7]", mean_6: "f32[8, 960, 1, 1]", relu_39: "f32[8, 240, 1, 1]", div_6: "f32[8, 960, 1, 1]", mul_545: "f32[8, 960, 7, 7]", convolution_91: "f32[8, 80, 7, 7]", squeeze_232: "f32[80]", add_411: "f32[8, 80, 7, 7]", convolution_92: "f32[8, 80, 7, 7]", squeeze_235: "f32[80]", slice_176: "f32[8, 160, 7, 7]", convolution_93: "f32[8, 960, 7, 7]", squeeze_238: "f32[960]", mean_7: "f32[8, 960, 1, 1]", view_1: "f32[8, 1280]", permute_1: "f32[1000, 1280]", le: "b8[8, 1280, 1, 1]", le_1: "b8[8, 960, 7, 7]", unsqueeze_322: "f32[1, 960, 1, 1]", unsqueeze_334: "f32[1, 80, 1, 1]", unsqueeze_346: "f32[1, 80, 1, 1]", bitwise_and: "b8[8, 960, 1, 1]", le_3: "b8[8, 480, 7, 7]", unsqueeze_358: "f32[1, 480, 1, 1]", unsqueeze_370: "f32[1, 480, 1, 1]", unsqueeze_382: "f32[1, 80, 1, 1]", unsqueeze_394: "f32[1, 80, 1, 1]", le_5: "b8[8, 480, 7, 7]", unsqueeze_406: "f32[1, 480, 1, 1]", unsqueeze_418: "f32[1, 480, 1, 1]", unsqueeze_430: "f32[1, 80, 1, 1]", unsqueeze_442: "f32[1, 80, 1, 1]", bitwise_and_1: "b8[8, 960, 1, 1]", le_8: "b8[8, 480, 7, 7]", unsqueeze_454: "f32[1, 480, 1, 1]", unsqueeze_466: "f32[1, 480, 1, 1]", unsqueeze_478: "f32[1, 80, 1, 1]", unsqueeze_490: "f32[1, 80, 1, 1]", le_10: "b8[8, 480, 7, 7]", unsqueeze_502: "f32[1, 480, 1, 1]", unsqueeze_514: "f32[1, 480, 1, 1]", unsqueeze_526: "f32[1, 160, 1, 1]", unsqueeze_538: "f32[1, 112, 1, 1]", unsqueeze_550: "f32[1, 80, 1, 1]", unsqueeze_562: "f32[1, 80, 1, 1]", bitwise_and_2: "b8[8, 672, 1, 1]", unsqueeze_574: "f32[1, 672, 1, 1]", le_13: "b8[8, 336, 14, 14]", unsqueeze_586: "f32[1, 336, 1, 1]", unsqueeze_598: "f32[1, 336, 1, 1]", unsqueeze_610: "f32[1, 56, 1, 1]", unsqueeze_622: "f32[1, 56, 1, 1]", bitwise_and_3: "b8[8, 672, 1, 1]", le_16: "b8[8, 336, 14, 14]", unsqueeze_634: "f32[1, 336, 1, 1]", unsqueeze_646: "f32[1, 336, 1, 1]", unsqueeze_658: "f32[1, 112, 1, 1]", unsqueeze_670: "f32[1, 80, 1, 1]", unsqueeze_682: "f32[1, 56, 1, 1]", unsqueeze_694: "f32[1, 56, 1, 1]", bitwise_and_4: "b8[8, 480, 1, 1]", le_19: "b8[8, 240, 14, 14]", unsqueeze_706: "f32[1, 240, 1, 1]", unsqueeze_718: "f32[1, 240, 1, 1]", unsqueeze_730: "f32[1, 40, 1, 1]", unsqueeze_742: "f32[1, 40, 1, 1]", le_21: "b8[8, 92, 14, 14]", unsqueeze_754: "f32[1, 92, 1, 1]", unsqueeze_766: "f32[1, 92, 1, 1]", unsqueeze_778: "f32[1, 40, 1, 1]", unsqueeze_790: "f32[1, 40, 1, 1]", le_23: "b8[8, 92, 14, 14]", unsqueeze_802: "f32[1, 92, 1, 1]", unsqueeze_814: "f32[1, 92, 1, 1]", unsqueeze_826: "f32[1, 40, 1, 1]", unsqueeze_838: "f32[1, 40, 1, 1]", le_25: "b8[8, 100, 14, 14]", unsqueeze_850: "f32[1, 100, 1, 1]", unsqueeze_862: "f32[1, 100, 1, 1]", unsqueeze_874: "f32[1, 80, 1, 1]", unsqueeze_886: "f32[1, 40, 1, 1]", unsqueeze_898: "f32[1, 40, 1, 1]", unsqueeze_910: "f32[1, 40, 1, 1]", unsqueeze_922: "f32[1, 240, 1, 1]", le_27: "b8[8, 120, 28, 28]", unsqueeze_934: "f32[1, 120, 1, 1]", unsqueeze_946: "f32[1, 120, 1, 1]", unsqueeze_958: "f32[1, 20, 1, 1]", unsqueeze_970: "f32[1, 20, 1, 1]", bitwise_and_5: "b8[8, 120, 1, 1]", le_30: "b8[8, 60, 28, 28]", unsqueeze_982: "f32[1, 60, 1, 1]", unsqueeze_994: "f32[1, 60, 1, 1]", unsqueeze_1006: "f32[1, 40, 1, 1]", unsqueeze_1018: "f32[1, 24, 1, 1]", unsqueeze_1030: "f32[1, 20, 1, 1]", unsqueeze_1042: "f32[1, 20, 1, 1]", bitwise_and_6: "b8[8, 72, 1, 1]", unsqueeze_1054: "f32[1, 72, 1, 1]", le_33: "b8[8, 36, 56, 56]", unsqueeze_1066: "f32[1, 36, 1, 1]", unsqueeze_1078: "f32[1, 36, 1, 1]", unsqueeze_1090: "f32[1, 12, 1, 1]", unsqueeze_1102: "f32[1, 12, 1, 1]", le_35: "b8[8, 36, 56, 56]", unsqueeze_1114: "f32[1, 36, 1, 1]", unsqueeze_1126: "f32[1, 36, 1, 1]", unsqueeze_1138: "f32[1, 24, 1, 1]", unsqueeze_1150: "f32[1, 16, 1, 1]", unsqueeze_1162: "f32[1, 12, 1, 1]", unsqueeze_1174: "f32[1, 12, 1, 1]", unsqueeze_1186: "f32[1, 48, 1, 1]", le_37: "b8[8, 24, 112, 112]", unsqueeze_1198: "f32[1, 24, 1, 1]", unsqueeze_1210: "f32[1, 24, 1, 1]", unsqueeze_1222: "f32[1, 8, 1, 1]", unsqueeze_1234: "f32[1, 8, 1, 1]", le_39: "b8[8, 8, 112, 112]", unsqueeze_1246: "f32[1, 8, 1, 1]", unsqueeze_1258: "f32[1, 8, 1, 1]", unsqueeze_1270: "f32[1, 16, 1, 1]", tangents_1: "f32[8, 1000]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:63, code: return out[:, :self.out_chs, :, :]
    slice_45: "f32[8, 120, 28, 28]" = torch.ops.aten.slice.Tensor(cat_8, 0, 0, 9223372036854775807);  cat_8 = None
    slice_46: "f32[8, 120, 28, 28]" = torch.ops.aten.slice.Tensor(slice_45, 2, 0, 9223372036854775807);  slice_45 = None
    slice_47: "f32[8, 120, 28, 28]" = torch.ops.aten.slice.Tensor(slice_46, 3, 0, 9223372036854775807);  slice_46 = None
    slice_100: "f32[8, 480, 14, 14]" = torch.ops.aten.slice.Tensor(cat_18, 0, 0, 9223372036854775807);  cat_18 = None
    slice_101: "f32[8, 480, 14, 14]" = torch.ops.aten.slice.Tensor(slice_100, 2, 0, 9223372036854775807);  slice_100 = None
    slice_102: "f32[8, 480, 14, 14]" = torch.ops.aten.slice.Tensor(slice_101, 3, 0, 9223372036854775807);  slice_101 = None
    slice_111: "f32[8, 672, 14, 14]" = torch.ops.aten.slice.Tensor(cat_20, 0, 0, 9223372036854775807);  cat_20 = None
    slice_112: "f32[8, 672, 14, 14]" = torch.ops.aten.slice.Tensor(slice_111, 2, 0, 9223372036854775807);  slice_111 = None
    slice_113: "f32[8, 672, 14, 14]" = torch.ops.aten.slice.Tensor(slice_112, 3, 0, 9223372036854775807);  slice_112 = None
    slice_144: "f32[8, 960, 7, 7]" = torch.ops.aten.slice.Tensor(cat_26, 0, 0, 9223372036854775807);  cat_26 = None
    slice_145: "f32[8, 960, 7, 7]" = torch.ops.aten.slice.Tensor(slice_144, 2, 0, 9223372036854775807);  slice_144 = None
    slice_146: "f32[8, 960, 7, 7]" = torch.ops.aten.slice.Tensor(slice_145, 3, 0, 9223372036854775807);  slice_145 = None
    slice_166: "f32[8, 960, 7, 7]" = torch.ops.aten.slice.Tensor(cat_30, 0, 0, 9223372036854775807);  cat_30 = None
    slice_167: "f32[8, 960, 7, 7]" = torch.ops.aten.slice.Tensor(slice_166, 2, 0, 9223372036854775807);  slice_166 = None
    slice_168: "f32[8, 960, 7, 7]" = torch.ops.aten.slice.Tensor(slice_167, 3, 0, 9223372036854775807);  slice_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/linear.py:19, code: return F.linear(input, self.weight, self.bias)
    mm: "f32[8, 1280]" = torch.ops.aten.mm.default(tangents_1, permute_1);  permute_1 = None
    permute_2: "f32[1000, 8]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
    mm_1: "f32[1000, 1280]" = torch.ops.aten.mm.default(permute_2, view_1);  permute_2 = view_1 = None
    permute_3: "f32[1280, 1000]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_1: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view_2: "f32[1000]" = torch.ops.aten.view.default(sum_1, [1000]);  sum_1 = None
    permute_4: "f32[1000, 1280]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:295, code: x = self.flatten(x)
    view_3: "f32[8, 1280, 1, 1]" = torch.ops.aten.view.default(mm, [8, 1280, 1, 1]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:294, code: x = self.act2(x)
    full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    where: "f32[8, 1280, 1, 1]" = torch.ops.aten.where.self(le, full_default, view_3);  le = view_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:293, code: x = self.conv_head(x)
    sum_2: "f32[1280]" = torch.ops.aten.sum.dim_IntList(where, [0, 2, 3])
    convolution_backward = torch.ops.aten.convolution_backward.default(where, mean_7, primals_271, [1280], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where = mean_7 = primals_271 = None
    getitem_160: "f32[8, 960, 1, 1]" = convolution_backward[0]
    getitem_161: "f32[1280, 960, 1, 1]" = convolution_backward[1];  convolution_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    expand: "f32[8, 960, 7, 7]" = torch.ops.aten.expand.default(getitem_160, [8, 960, 7, 7]);  getitem_160 = None
    div_7: "f32[8, 960, 7, 7]" = torch.ops.aten.div.Scalar(expand, 49);  expand = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    where_1: "f32[8, 960, 7, 7]" = torch.ops.aten.where.self(le_1, full_default, div_7);  le_1 = div_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_3: "f32[960]" = torch.ops.aten.sum.dim_IntList(where_1, [0, 2, 3])
    sub_80: "f32[8, 960, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_93, unsqueeze_322);  convolution_93 = unsqueeze_322 = None
    mul_567: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(where_1, sub_80)
    sum_4: "f32[960]" = torch.ops.aten.sum.dim_IntList(mul_567, [0, 2, 3]);  mul_567 = None
    mul_568: "f32[960]" = torch.ops.aten.mul.Tensor(sum_3, 0.002551020408163265)
    unsqueeze_323: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(mul_568, 0);  mul_568 = None
    unsqueeze_324: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_323, 2);  unsqueeze_323 = None
    unsqueeze_325: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_324, 3);  unsqueeze_324 = None
    mul_569: "f32[960]" = torch.ops.aten.mul.Tensor(sum_4, 0.002551020408163265)
    mul_570: "f32[960]" = torch.ops.aten.mul.Tensor(squeeze_238, squeeze_238)
    mul_571: "f32[960]" = torch.ops.aten.mul.Tensor(mul_569, mul_570);  mul_569 = mul_570 = None
    unsqueeze_326: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(mul_571, 0);  mul_571 = None
    unsqueeze_327: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_326, 2);  unsqueeze_326 = None
    unsqueeze_328: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_327, 3);  unsqueeze_327 = None
    mul_572: "f32[960]" = torch.ops.aten.mul.Tensor(squeeze_238, primals_1);  primals_1 = None
    unsqueeze_329: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(mul_572, 0);  mul_572 = None
    unsqueeze_330: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_329, 2);  unsqueeze_329 = None
    unsqueeze_331: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_330, 3);  unsqueeze_330 = None
    mul_573: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(sub_80, unsqueeze_328);  sub_80 = unsqueeze_328 = None
    sub_82: "f32[8, 960, 7, 7]" = torch.ops.aten.sub.Tensor(where_1, mul_573);  where_1 = mul_573 = None
    sub_83: "f32[8, 960, 7, 7]" = torch.ops.aten.sub.Tensor(sub_82, unsqueeze_325);  sub_82 = unsqueeze_325 = None
    mul_574: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(sub_83, unsqueeze_331);  sub_83 = unsqueeze_331 = None
    mul_575: "f32[960]" = torch.ops.aten.mul.Tensor(sum_4, squeeze_238);  sum_4 = squeeze_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:82, code: x = self.conv(x)
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(mul_574, slice_176, primals_270, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_574 = slice_176 = primals_270 = None
    getitem_163: "f32[8, 160, 7, 7]" = convolution_backward_1[0]
    getitem_164: "f32[960, 160, 1, 1]" = convolution_backward_1[1];  convolution_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:182, code: x += self.shortcut(shortcut)
    full: "f32[62720]" = torch.ops.aten.full.default([62720], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    as_strided: "f32[8, 160, 7, 7]" = torch.ops.aten.as_strided.default(full, [8, 160, 7, 7], [7840, 49, 7, 1], 0)
    copy: "f32[8, 160, 7, 7]" = torch.ops.aten.copy.default(as_strided, getitem_163);  getitem_163 = None
    as_strided_scatter: "f32[62720]" = torch.ops.aten.as_strided_scatter.default(full, copy, [8, 160, 7, 7], [7840, 49, 7, 1], 0);  copy = None
    as_strided_3: "f32[8, 160, 7, 7]" = torch.ops.aten.as_strided.default(as_strided_scatter, [8, 160, 7, 7], [7840, 49, 7, 1], 0);  as_strided_scatter = None
    new_empty_strided: "f32[8, 160, 7, 7]" = torch.ops.aten.new_empty_strided.default(as_strided_3, [8, 160, 7, 7], [7840, 49, 7, 1])
    copy_1: "f32[8, 160, 7, 7]" = torch.ops.aten.copy.default(new_empty_strided, as_strided_3);  new_empty_strided = as_strided_3 = None
    as_strided_5: "f32[8, 160, 7, 7]" = torch.ops.aten.as_strided.default(copy_1, [8, 160, 7, 7], [7840, 49, 7, 1], 0)
    clone: "f32[8, 160, 7, 7]" = torch.ops.aten.clone.default(as_strided_5, memory_format = torch.contiguous_format)
    copy_2: "f32[8, 160, 7, 7]" = torch.ops.aten.copy.default(as_strided_5, clone);  as_strided_5 = None
    as_strided_scatter_1: "f32[8, 160, 7, 7]" = torch.ops.aten.as_strided_scatter.default(copy_1, copy_2, [8, 160, 7, 7], [7840, 49, 7, 1], 0);  copy_1 = copy_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    slice_179: "f32[8, 80, 7, 7]" = torch.ops.aten.slice.Tensor(as_strided_scatter_1, 1, 80, 160)
    sum_5: "f32[80]" = torch.ops.aten.sum.dim_IntList(slice_179, [0, 2, 3])
    sub_84: "f32[8, 80, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_92, unsqueeze_334);  convolution_92 = unsqueeze_334 = None
    mul_576: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(slice_179, sub_84)
    sum_6: "f32[80]" = torch.ops.aten.sum.dim_IntList(mul_576, [0, 2, 3]);  mul_576 = None
    mul_577: "f32[80]" = torch.ops.aten.mul.Tensor(sum_5, 0.002551020408163265)
    unsqueeze_335: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_577, 0);  mul_577 = None
    unsqueeze_336: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_335, 2);  unsqueeze_335 = None
    unsqueeze_337: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_336, 3);  unsqueeze_336 = None
    mul_578: "f32[80]" = torch.ops.aten.mul.Tensor(sum_6, 0.002551020408163265)
    mul_579: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_235, squeeze_235)
    mul_580: "f32[80]" = torch.ops.aten.mul.Tensor(mul_578, mul_579);  mul_578 = mul_579 = None
    unsqueeze_338: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_580, 0);  mul_580 = None
    unsqueeze_339: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_338, 2);  unsqueeze_338 = None
    unsqueeze_340: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_339, 3);  unsqueeze_339 = None
    mul_581: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_235, primals_268);  primals_268 = None
    unsqueeze_341: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_581, 0);  mul_581 = None
    unsqueeze_342: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_341, 2);  unsqueeze_341 = None
    unsqueeze_343: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_342, 3);  unsqueeze_342 = None
    mul_582: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(sub_84, unsqueeze_340);  sub_84 = unsqueeze_340 = None
    sub_86: "f32[8, 80, 7, 7]" = torch.ops.aten.sub.Tensor(slice_179, mul_582);  slice_179 = mul_582 = None
    sub_87: "f32[8, 80, 7, 7]" = torch.ops.aten.sub.Tensor(sub_86, unsqueeze_337);  sub_86 = unsqueeze_337 = None
    mul_583: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(sub_87, unsqueeze_343);  sub_87 = unsqueeze_343 = None
    mul_584: "f32[80]" = torch.ops.aten.mul.Tensor(sum_6, squeeze_235);  sum_6 = squeeze_235 = None
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(mul_583, add_411, primals_267, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 80, [True, True, False]);  mul_583 = add_411 = primals_267 = None
    getitem_166: "f32[8, 80, 7, 7]" = convolution_backward_2[0]
    getitem_167: "f32[80, 1, 3, 3]" = convolution_backward_2[1];  convolution_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    slice_180: "f32[8, 80, 7, 7]" = torch.ops.aten.slice.Tensor(as_strided_scatter_1, 1, 0, 80);  as_strided_scatter_1 = None
    add_423: "f32[8, 80, 7, 7]" = torch.ops.aten.add.Tensor(slice_180, getitem_166);  slice_180 = getitem_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    sum_7: "f32[80]" = torch.ops.aten.sum.dim_IntList(add_423, [0, 2, 3])
    sub_88: "f32[8, 80, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_91, unsqueeze_346);  convolution_91 = unsqueeze_346 = None
    mul_585: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(add_423, sub_88)
    sum_8: "f32[80]" = torch.ops.aten.sum.dim_IntList(mul_585, [0, 2, 3]);  mul_585 = None
    mul_586: "f32[80]" = torch.ops.aten.mul.Tensor(sum_7, 0.002551020408163265)
    unsqueeze_347: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_586, 0);  mul_586 = None
    unsqueeze_348: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_347, 2);  unsqueeze_347 = None
    unsqueeze_349: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_348, 3);  unsqueeze_348 = None
    mul_587: "f32[80]" = torch.ops.aten.mul.Tensor(sum_8, 0.002551020408163265)
    mul_588: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_232, squeeze_232)
    mul_589: "f32[80]" = torch.ops.aten.mul.Tensor(mul_587, mul_588);  mul_587 = mul_588 = None
    unsqueeze_350: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_589, 0);  mul_589 = None
    unsqueeze_351: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_350, 2);  unsqueeze_350 = None
    unsqueeze_352: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_351, 3);  unsqueeze_351 = None
    mul_590: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_232, primals_265);  primals_265 = None
    unsqueeze_353: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_590, 0);  mul_590 = None
    unsqueeze_354: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_353, 2);  unsqueeze_353 = None
    unsqueeze_355: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_354, 3);  unsqueeze_354 = None
    mul_591: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(sub_88, unsqueeze_352);  sub_88 = unsqueeze_352 = None
    sub_90: "f32[8, 80, 7, 7]" = torch.ops.aten.sub.Tensor(add_423, mul_591);  add_423 = mul_591 = None
    sub_91: "f32[8, 80, 7, 7]" = torch.ops.aten.sub.Tensor(sub_90, unsqueeze_349);  sub_90 = unsqueeze_349 = None
    mul_592: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(sub_91, unsqueeze_355);  sub_91 = unsqueeze_355 = None
    mul_593: "f32[80]" = torch.ops.aten.mul.Tensor(sum_8, squeeze_232);  sum_8 = squeeze_232 = None
    convolution_backward_3 = torch.ops.aten.convolution_backward.default(mul_592, mul_545, primals_264, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_592 = mul_545 = primals_264 = None
    getitem_169: "f32[8, 960, 7, 7]" = convolution_backward_3[0]
    getitem_170: "f32[80, 960, 1, 1]" = convolution_backward_3[1];  convolution_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_594: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_169, slice_168);  slice_168 = None
    mul_595: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_169, div_6);  getitem_169 = div_6 = None
    sum_9: "f32[8, 960, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_594, [2, 3], True);  mul_594 = None
    mul_596: "f32[8, 960, 1, 1]" = torch.ops.aten.mul.Tensor(sum_9, 0.16666666666666666);  sum_9 = None
    where_2: "f32[8, 960, 1, 1]" = torch.ops.aten.where.self(bitwise_and, mul_596, full_default);  bitwise_and = mul_596 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    sum_10: "f32[960]" = torch.ops.aten.sum.dim_IntList(where_2, [0, 2, 3])
    convolution_backward_4 = torch.ops.aten.convolution_backward.default(where_2, relu_39, primals_262, [960], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_2 = primals_262 = None
    getitem_172: "f32[8, 240, 1, 1]" = convolution_backward_4[0]
    getitem_173: "f32[960, 240, 1, 1]" = convolution_backward_4[1];  convolution_backward_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    alias_49: "f32[8, 240, 1, 1]" = torch.ops.aten.alias.default(relu_39);  relu_39 = None
    alias_50: "f32[8, 240, 1, 1]" = torch.ops.aten.alias.default(alias_49);  alias_49 = None
    le_2: "b8[8, 240, 1, 1]" = torch.ops.aten.le.Scalar(alias_50, 0);  alias_50 = None
    where_3: "f32[8, 240, 1, 1]" = torch.ops.aten.where.self(le_2, full_default, getitem_172);  le_2 = getitem_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    sum_11: "f32[240]" = torch.ops.aten.sum.dim_IntList(where_3, [0, 2, 3])
    convolution_backward_5 = torch.ops.aten.convolution_backward.default(where_3, mean_6, primals_260, [240], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_3 = mean_6 = primals_260 = None
    getitem_175: "f32[8, 960, 1, 1]" = convolution_backward_5[0]
    getitem_176: "f32[240, 960, 1, 1]" = convolution_backward_5[1];  convolution_backward_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_1: "f32[8, 960, 7, 7]" = torch.ops.aten.expand.default(getitem_175, [8, 960, 7, 7]);  getitem_175 = None
    div_8: "f32[8, 960, 7, 7]" = torch.ops.aten.div.Scalar(expand_1, 49);  expand_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_424: "f32[8, 960, 7, 7]" = torch.ops.aten.add.Tensor(mul_595, div_8);  mul_595 = div_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:63, code: return out[:, :self.out_chs, :, :]
    full_default_4: "f32[8, 960, 7, 7]" = torch.ops.aten.full.default([8, 960, 7, 7], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_48: "f32[8, 960, 7, 7]" = torch.ops.aten.slice_scatter.default(full_default_4, add_424, 3, 0, 9223372036854775807);  add_424 = None
    slice_scatter_49: "f32[8, 960, 7, 7]" = torch.ops.aten.slice_scatter.default(full_default_4, slice_scatter_48, 2, 0, 9223372036854775807);  slice_scatter_48 = None
    slice_scatter_50: "f32[8, 960, 7, 7]" = torch.ops.aten.slice_scatter.default(full_default_4, slice_scatter_49, 0, 0, 9223372036854775807);  slice_scatter_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    slice_181: "f32[8, 480, 7, 7]" = torch.ops.aten.slice.Tensor(slice_scatter_50, 1, 0, 480)
    slice_182: "f32[8, 480, 7, 7]" = torch.ops.aten.slice.Tensor(slice_scatter_50, 1, 480, 960);  slice_scatter_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    where_4: "f32[8, 480, 7, 7]" = torch.ops.aten.where.self(le_3, full_default, slice_182);  le_3 = slice_182 = None
    sum_12: "f32[480]" = torch.ops.aten.sum.dim_IntList(where_4, [0, 2, 3])
    sub_92: "f32[8, 480, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_88, unsqueeze_358);  convolution_88 = unsqueeze_358 = None
    mul_597: "f32[8, 480, 7, 7]" = torch.ops.aten.mul.Tensor(where_4, sub_92)
    sum_13: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_597, [0, 2, 3]);  mul_597 = None
    mul_598: "f32[480]" = torch.ops.aten.mul.Tensor(sum_12, 0.002551020408163265)
    unsqueeze_359: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_598, 0);  mul_598 = None
    unsqueeze_360: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_359, 2);  unsqueeze_359 = None
    unsqueeze_361: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_360, 3);  unsqueeze_360 = None
    mul_599: "f32[480]" = torch.ops.aten.mul.Tensor(sum_13, 0.002551020408163265)
    mul_600: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_229, squeeze_229)
    mul_601: "f32[480]" = torch.ops.aten.mul.Tensor(mul_599, mul_600);  mul_599 = mul_600 = None
    unsqueeze_362: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_601, 0);  mul_601 = None
    unsqueeze_363: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_362, 2);  unsqueeze_362 = None
    unsqueeze_364: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_363, 3);  unsqueeze_363 = None
    mul_602: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_229, primals_258);  primals_258 = None
    unsqueeze_365: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_602, 0);  mul_602 = None
    unsqueeze_366: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_365, 2);  unsqueeze_365 = None
    unsqueeze_367: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_366, 3);  unsqueeze_366 = None
    mul_603: "f32[8, 480, 7, 7]" = torch.ops.aten.mul.Tensor(sub_92, unsqueeze_364);  sub_92 = unsqueeze_364 = None
    sub_94: "f32[8, 480, 7, 7]" = torch.ops.aten.sub.Tensor(where_4, mul_603);  where_4 = mul_603 = None
    sub_95: "f32[8, 480, 7, 7]" = torch.ops.aten.sub.Tensor(sub_94, unsqueeze_361);  sub_94 = unsqueeze_361 = None
    mul_604: "f32[8, 480, 7, 7]" = torch.ops.aten.mul.Tensor(sub_95, unsqueeze_367);  sub_95 = unsqueeze_367 = None
    mul_605: "f32[480]" = torch.ops.aten.mul.Tensor(sum_13, squeeze_229);  sum_13 = squeeze_229 = None
    convolution_backward_6 = torch.ops.aten.convolution_backward.default(mul_604, relu_37, primals_257, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 480, [True, True, False]);  mul_604 = primals_257 = None
    getitem_178: "f32[8, 480, 7, 7]" = convolution_backward_6[0]
    getitem_179: "f32[480, 1, 3, 3]" = convolution_backward_6[1];  convolution_backward_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    add_425: "f32[8, 480, 7, 7]" = torch.ops.aten.add.Tensor(slice_181, getitem_178);  slice_181 = getitem_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    alias_55: "f32[8, 480, 7, 7]" = torch.ops.aten.alias.default(relu_37);  relu_37 = None
    alias_56: "f32[8, 480, 7, 7]" = torch.ops.aten.alias.default(alias_55);  alias_55 = None
    le_4: "b8[8, 480, 7, 7]" = torch.ops.aten.le.Scalar(alias_56, 0);  alias_56 = None
    where_5: "f32[8, 480, 7, 7]" = torch.ops.aten.where.self(le_4, full_default, add_425);  le_4 = add_425 = None
    sum_14: "f32[480]" = torch.ops.aten.sum.dim_IntList(where_5, [0, 2, 3])
    sub_96: "f32[8, 480, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_87, unsqueeze_370);  convolution_87 = unsqueeze_370 = None
    mul_606: "f32[8, 480, 7, 7]" = torch.ops.aten.mul.Tensor(where_5, sub_96)
    sum_15: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_606, [0, 2, 3]);  mul_606 = None
    mul_607: "f32[480]" = torch.ops.aten.mul.Tensor(sum_14, 0.002551020408163265)
    unsqueeze_371: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_607, 0);  mul_607 = None
    unsqueeze_372: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_371, 2);  unsqueeze_371 = None
    unsqueeze_373: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_372, 3);  unsqueeze_372 = None
    mul_608: "f32[480]" = torch.ops.aten.mul.Tensor(sum_15, 0.002551020408163265)
    mul_609: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_226, squeeze_226)
    mul_610: "f32[480]" = torch.ops.aten.mul.Tensor(mul_608, mul_609);  mul_608 = mul_609 = None
    unsqueeze_374: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_610, 0);  mul_610 = None
    unsqueeze_375: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_374, 2);  unsqueeze_374 = None
    unsqueeze_376: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_375, 3);  unsqueeze_375 = None
    mul_611: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_226, primals_255);  primals_255 = None
    unsqueeze_377: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_611, 0);  mul_611 = None
    unsqueeze_378: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_377, 2);  unsqueeze_377 = None
    unsqueeze_379: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_378, 3);  unsqueeze_378 = None
    mul_612: "f32[8, 480, 7, 7]" = torch.ops.aten.mul.Tensor(sub_96, unsqueeze_376);  sub_96 = unsqueeze_376 = None
    sub_98: "f32[8, 480, 7, 7]" = torch.ops.aten.sub.Tensor(where_5, mul_612);  where_5 = mul_612 = None
    sub_99: "f32[8, 480, 7, 7]" = torch.ops.aten.sub.Tensor(sub_98, unsqueeze_373);  sub_98 = unsqueeze_373 = None
    mul_613: "f32[8, 480, 7, 7]" = torch.ops.aten.mul.Tensor(sub_99, unsqueeze_379);  sub_99 = unsqueeze_379 = None
    mul_614: "f32[480]" = torch.ops.aten.mul.Tensor(sum_15, squeeze_226);  sum_15 = squeeze_226 = None
    convolution_backward_7 = torch.ops.aten.convolution_backward.default(mul_613, slice_165, primals_254, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_613 = slice_165 = primals_254 = None
    getitem_181: "f32[8, 160, 7, 7]" = convolution_backward_7[0]
    getitem_182: "f32[480, 160, 1, 1]" = convolution_backward_7[1];  convolution_backward_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    add_426: "f32[8, 160, 7, 7]" = torch.ops.aten.add.Tensor(clone, getitem_181);  clone = getitem_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:182, code: x += self.shortcut(shortcut)
    copy_3: "f32[8, 160, 7, 7]" = torch.ops.aten.copy.default(as_strided, add_426);  add_426 = None
    as_strided_scatter_2: "f32[62720]" = torch.ops.aten.as_strided_scatter.default(full, copy_3, [8, 160, 7, 7], [7840, 49, 7, 1], 0);  copy_3 = None
    as_strided_10: "f32[8, 160, 7, 7]" = torch.ops.aten.as_strided.default(as_strided_scatter_2, [8, 160, 7, 7], [7840, 49, 7, 1], 0);  as_strided_scatter_2 = None
    new_empty_strided_1: "f32[8, 160, 7, 7]" = torch.ops.aten.new_empty_strided.default(as_strided_10, [8, 160, 7, 7], [7840, 49, 7, 1])
    copy_4: "f32[8, 160, 7, 7]" = torch.ops.aten.copy.default(new_empty_strided_1, as_strided_10);  new_empty_strided_1 = as_strided_10 = None
    as_strided_12: "f32[8, 160, 7, 7]" = torch.ops.aten.as_strided.default(copy_4, [8, 160, 7, 7], [7840, 49, 7, 1], 0)
    clone_1: "f32[8, 160, 7, 7]" = torch.ops.aten.clone.default(as_strided_12, memory_format = torch.contiguous_format)
    copy_5: "f32[8, 160, 7, 7]" = torch.ops.aten.copy.default(as_strided_12, clone_1);  as_strided_12 = None
    as_strided_scatter_3: "f32[8, 160, 7, 7]" = torch.ops.aten.as_strided_scatter.default(copy_4, copy_5, [8, 160, 7, 7], [7840, 49, 7, 1], 0);  copy_4 = copy_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    slice_185: "f32[8, 80, 7, 7]" = torch.ops.aten.slice.Tensor(as_strided_scatter_3, 1, 80, 160)
    sum_16: "f32[80]" = torch.ops.aten.sum.dim_IntList(slice_185, [0, 2, 3])
    sub_100: "f32[8, 80, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_86, unsqueeze_382);  convolution_86 = unsqueeze_382 = None
    mul_615: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(slice_185, sub_100)
    sum_17: "f32[80]" = torch.ops.aten.sum.dim_IntList(mul_615, [0, 2, 3]);  mul_615 = None
    mul_616: "f32[80]" = torch.ops.aten.mul.Tensor(sum_16, 0.002551020408163265)
    unsqueeze_383: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_616, 0);  mul_616 = None
    unsqueeze_384: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_383, 2);  unsqueeze_383 = None
    unsqueeze_385: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_384, 3);  unsqueeze_384 = None
    mul_617: "f32[80]" = torch.ops.aten.mul.Tensor(sum_17, 0.002551020408163265)
    mul_618: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_223, squeeze_223)
    mul_619: "f32[80]" = torch.ops.aten.mul.Tensor(mul_617, mul_618);  mul_617 = mul_618 = None
    unsqueeze_386: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_619, 0);  mul_619 = None
    unsqueeze_387: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_386, 2);  unsqueeze_386 = None
    unsqueeze_388: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_387, 3);  unsqueeze_387 = None
    mul_620: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_223, primals_252);  primals_252 = None
    unsqueeze_389: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_620, 0);  mul_620 = None
    unsqueeze_390: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_389, 2);  unsqueeze_389 = None
    unsqueeze_391: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_390, 3);  unsqueeze_390 = None
    mul_621: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(sub_100, unsqueeze_388);  sub_100 = unsqueeze_388 = None
    sub_102: "f32[8, 80, 7, 7]" = torch.ops.aten.sub.Tensor(slice_185, mul_621);  slice_185 = mul_621 = None
    sub_103: "f32[8, 80, 7, 7]" = torch.ops.aten.sub.Tensor(sub_102, unsqueeze_385);  sub_102 = unsqueeze_385 = None
    mul_622: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(sub_103, unsqueeze_391);  sub_103 = unsqueeze_391 = None
    mul_623: "f32[80]" = torch.ops.aten.mul.Tensor(sum_17, squeeze_223);  sum_17 = squeeze_223 = None
    convolution_backward_8 = torch.ops.aten.convolution_backward.default(mul_622, add_389, primals_251, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 80, [True, True, False]);  mul_622 = add_389 = primals_251 = None
    getitem_184: "f32[8, 80, 7, 7]" = convolution_backward_8[0]
    getitem_185: "f32[80, 1, 3, 3]" = convolution_backward_8[1];  convolution_backward_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    slice_186: "f32[8, 80, 7, 7]" = torch.ops.aten.slice.Tensor(as_strided_scatter_3, 1, 0, 80);  as_strided_scatter_3 = None
    add_427: "f32[8, 80, 7, 7]" = torch.ops.aten.add.Tensor(slice_186, getitem_184);  slice_186 = getitem_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    sum_18: "f32[80]" = torch.ops.aten.sum.dim_IntList(add_427, [0, 2, 3])
    sub_104: "f32[8, 80, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_85, unsqueeze_394);  convolution_85 = unsqueeze_394 = None
    mul_624: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(add_427, sub_104)
    sum_19: "f32[80]" = torch.ops.aten.sum.dim_IntList(mul_624, [0, 2, 3]);  mul_624 = None
    mul_625: "f32[80]" = torch.ops.aten.mul.Tensor(sum_18, 0.002551020408163265)
    unsqueeze_395: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_625, 0);  mul_625 = None
    unsqueeze_396: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_395, 2);  unsqueeze_395 = None
    unsqueeze_397: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_396, 3);  unsqueeze_396 = None
    mul_626: "f32[80]" = torch.ops.aten.mul.Tensor(sum_19, 0.002551020408163265)
    mul_627: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_220, squeeze_220)
    mul_628: "f32[80]" = torch.ops.aten.mul.Tensor(mul_626, mul_627);  mul_626 = mul_627 = None
    unsqueeze_398: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_628, 0);  mul_628 = None
    unsqueeze_399: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_398, 2);  unsqueeze_398 = None
    unsqueeze_400: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_399, 3);  unsqueeze_399 = None
    mul_629: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_220, primals_249);  primals_249 = None
    unsqueeze_401: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_629, 0);  mul_629 = None
    unsqueeze_402: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_401, 2);  unsqueeze_401 = None
    unsqueeze_403: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_402, 3);  unsqueeze_402 = None
    mul_630: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(sub_104, unsqueeze_400);  sub_104 = unsqueeze_400 = None
    sub_106: "f32[8, 80, 7, 7]" = torch.ops.aten.sub.Tensor(add_427, mul_630);  add_427 = mul_630 = None
    sub_107: "f32[8, 80, 7, 7]" = torch.ops.aten.sub.Tensor(sub_106, unsqueeze_397);  sub_106 = unsqueeze_397 = None
    mul_631: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(sub_107, unsqueeze_403);  sub_107 = unsqueeze_403 = None
    mul_632: "f32[80]" = torch.ops.aten.mul.Tensor(sum_19, squeeze_220);  sum_19 = squeeze_220 = None
    convolution_backward_9 = torch.ops.aten.convolution_backward.default(mul_631, slice_157, primals_248, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_631 = slice_157 = primals_248 = None
    getitem_187: "f32[8, 960, 7, 7]" = convolution_backward_9[0]
    getitem_188: "f32[80, 960, 1, 1]" = convolution_backward_9[1];  convolution_backward_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:63, code: return out[:, :self.out_chs, :, :]
    slice_scatter_51: "f32[8, 960, 7, 7]" = torch.ops.aten.slice_scatter.default(full_default_4, getitem_187, 3, 0, 9223372036854775807);  getitem_187 = None
    slice_scatter_52: "f32[8, 960, 7, 7]" = torch.ops.aten.slice_scatter.default(full_default_4, slice_scatter_51, 2, 0, 9223372036854775807);  slice_scatter_51 = None
    slice_scatter_53: "f32[8, 960, 7, 7]" = torch.ops.aten.slice_scatter.default(full_default_4, slice_scatter_52, 0, 0, 9223372036854775807);  slice_scatter_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    slice_187: "f32[8, 480, 7, 7]" = torch.ops.aten.slice.Tensor(slice_scatter_53, 1, 0, 480)
    slice_188: "f32[8, 480, 7, 7]" = torch.ops.aten.slice.Tensor(slice_scatter_53, 1, 480, 960);  slice_scatter_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    where_6: "f32[8, 480, 7, 7]" = torch.ops.aten.where.self(le_5, full_default, slice_188);  le_5 = slice_188 = None
    sum_20: "f32[480]" = torch.ops.aten.sum.dim_IntList(where_6, [0, 2, 3])
    sub_108: "f32[8, 480, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_84, unsqueeze_406);  convolution_84 = unsqueeze_406 = None
    mul_633: "f32[8, 480, 7, 7]" = torch.ops.aten.mul.Tensor(where_6, sub_108)
    sum_21: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_633, [0, 2, 3]);  mul_633 = None
    mul_634: "f32[480]" = torch.ops.aten.mul.Tensor(sum_20, 0.002551020408163265)
    unsqueeze_407: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_634, 0);  mul_634 = None
    unsqueeze_408: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_407, 2);  unsqueeze_407 = None
    unsqueeze_409: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_408, 3);  unsqueeze_408 = None
    mul_635: "f32[480]" = torch.ops.aten.mul.Tensor(sum_21, 0.002551020408163265)
    mul_636: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_217, squeeze_217)
    mul_637: "f32[480]" = torch.ops.aten.mul.Tensor(mul_635, mul_636);  mul_635 = mul_636 = None
    unsqueeze_410: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_637, 0);  mul_637 = None
    unsqueeze_411: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_410, 2);  unsqueeze_410 = None
    unsqueeze_412: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_411, 3);  unsqueeze_411 = None
    mul_638: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_217, primals_246);  primals_246 = None
    unsqueeze_413: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_638, 0);  mul_638 = None
    unsqueeze_414: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_413, 2);  unsqueeze_413 = None
    unsqueeze_415: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_414, 3);  unsqueeze_414 = None
    mul_639: "f32[8, 480, 7, 7]" = torch.ops.aten.mul.Tensor(sub_108, unsqueeze_412);  sub_108 = unsqueeze_412 = None
    sub_110: "f32[8, 480, 7, 7]" = torch.ops.aten.sub.Tensor(where_6, mul_639);  where_6 = mul_639 = None
    sub_111: "f32[8, 480, 7, 7]" = torch.ops.aten.sub.Tensor(sub_110, unsqueeze_409);  sub_110 = unsqueeze_409 = None
    mul_640: "f32[8, 480, 7, 7]" = torch.ops.aten.mul.Tensor(sub_111, unsqueeze_415);  sub_111 = unsqueeze_415 = None
    mul_641: "f32[480]" = torch.ops.aten.mul.Tensor(sum_21, squeeze_217);  sum_21 = squeeze_217 = None
    convolution_backward_10 = torch.ops.aten.convolution_backward.default(mul_640, relu_35, primals_245, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 480, [True, True, False]);  mul_640 = primals_245 = None
    getitem_190: "f32[8, 480, 7, 7]" = convolution_backward_10[0]
    getitem_191: "f32[480, 1, 3, 3]" = convolution_backward_10[1];  convolution_backward_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    add_428: "f32[8, 480, 7, 7]" = torch.ops.aten.add.Tensor(slice_187, getitem_190);  slice_187 = getitem_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    alias_61: "f32[8, 480, 7, 7]" = torch.ops.aten.alias.default(relu_35);  relu_35 = None
    alias_62: "f32[8, 480, 7, 7]" = torch.ops.aten.alias.default(alias_61);  alias_61 = None
    le_6: "b8[8, 480, 7, 7]" = torch.ops.aten.le.Scalar(alias_62, 0);  alias_62 = None
    where_7: "f32[8, 480, 7, 7]" = torch.ops.aten.where.self(le_6, full_default, add_428);  le_6 = add_428 = None
    sum_22: "f32[480]" = torch.ops.aten.sum.dim_IntList(where_7, [0, 2, 3])
    sub_112: "f32[8, 480, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_83, unsqueeze_418);  convolution_83 = unsqueeze_418 = None
    mul_642: "f32[8, 480, 7, 7]" = torch.ops.aten.mul.Tensor(where_7, sub_112)
    sum_23: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_642, [0, 2, 3]);  mul_642 = None
    mul_643: "f32[480]" = torch.ops.aten.mul.Tensor(sum_22, 0.002551020408163265)
    unsqueeze_419: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_643, 0);  mul_643 = None
    unsqueeze_420: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_419, 2);  unsqueeze_419 = None
    unsqueeze_421: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_420, 3);  unsqueeze_420 = None
    mul_644: "f32[480]" = torch.ops.aten.mul.Tensor(sum_23, 0.002551020408163265)
    mul_645: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_214, squeeze_214)
    mul_646: "f32[480]" = torch.ops.aten.mul.Tensor(mul_644, mul_645);  mul_644 = mul_645 = None
    unsqueeze_422: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_646, 0);  mul_646 = None
    unsqueeze_423: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_422, 2);  unsqueeze_422 = None
    unsqueeze_424: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_423, 3);  unsqueeze_423 = None
    mul_647: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_214, primals_243);  primals_243 = None
    unsqueeze_425: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_647, 0);  mul_647 = None
    unsqueeze_426: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_425, 2);  unsqueeze_425 = None
    unsqueeze_427: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_426, 3);  unsqueeze_426 = None
    mul_648: "f32[8, 480, 7, 7]" = torch.ops.aten.mul.Tensor(sub_112, unsqueeze_424);  sub_112 = unsqueeze_424 = None
    sub_114: "f32[8, 480, 7, 7]" = torch.ops.aten.sub.Tensor(where_7, mul_648);  where_7 = mul_648 = None
    sub_115: "f32[8, 480, 7, 7]" = torch.ops.aten.sub.Tensor(sub_114, unsqueeze_421);  sub_114 = unsqueeze_421 = None
    mul_649: "f32[8, 480, 7, 7]" = torch.ops.aten.mul.Tensor(sub_115, unsqueeze_427);  sub_115 = unsqueeze_427 = None
    mul_650: "f32[480]" = torch.ops.aten.mul.Tensor(sum_23, squeeze_214);  sum_23 = squeeze_214 = None
    convolution_backward_11 = torch.ops.aten.convolution_backward.default(mul_649, slice_154, primals_242, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_649 = slice_154 = primals_242 = None
    getitem_193: "f32[8, 160, 7, 7]" = convolution_backward_11[0]
    getitem_194: "f32[480, 160, 1, 1]" = convolution_backward_11[1];  convolution_backward_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    add_429: "f32[8, 160, 7, 7]" = torch.ops.aten.add.Tensor(clone_1, getitem_193);  clone_1 = getitem_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:182, code: x += self.shortcut(shortcut)
    copy_6: "f32[8, 160, 7, 7]" = torch.ops.aten.copy.default(as_strided, add_429);  add_429 = None
    as_strided_scatter_4: "f32[62720]" = torch.ops.aten.as_strided_scatter.default(full, copy_6, [8, 160, 7, 7], [7840, 49, 7, 1], 0);  copy_6 = None
    as_strided_17: "f32[8, 160, 7, 7]" = torch.ops.aten.as_strided.default(as_strided_scatter_4, [8, 160, 7, 7], [7840, 49, 7, 1], 0);  as_strided_scatter_4 = None
    new_empty_strided_2: "f32[8, 160, 7, 7]" = torch.ops.aten.new_empty_strided.default(as_strided_17, [8, 160, 7, 7], [7840, 49, 7, 1])
    copy_7: "f32[8, 160, 7, 7]" = torch.ops.aten.copy.default(new_empty_strided_2, as_strided_17);  new_empty_strided_2 = as_strided_17 = None
    as_strided_19: "f32[8, 160, 7, 7]" = torch.ops.aten.as_strided.default(copy_7, [8, 160, 7, 7], [7840, 49, 7, 1], 0)
    clone_2: "f32[8, 160, 7, 7]" = torch.ops.aten.clone.default(as_strided_19, memory_format = torch.contiguous_format)
    copy_8: "f32[8, 160, 7, 7]" = torch.ops.aten.copy.default(as_strided_19, clone_2);  as_strided_19 = None
    as_strided_scatter_5: "f32[8, 160, 7, 7]" = torch.ops.aten.as_strided_scatter.default(copy_7, copy_8, [8, 160, 7, 7], [7840, 49, 7, 1], 0);  copy_7 = copy_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    slice_191: "f32[8, 80, 7, 7]" = torch.ops.aten.slice.Tensor(as_strided_scatter_5, 1, 80, 160)
    sum_24: "f32[80]" = torch.ops.aten.sum.dim_IntList(slice_191, [0, 2, 3])
    sub_116: "f32[8, 80, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_82, unsqueeze_430);  convolution_82 = unsqueeze_430 = None
    mul_651: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(slice_191, sub_116)
    sum_25: "f32[80]" = torch.ops.aten.sum.dim_IntList(mul_651, [0, 2, 3]);  mul_651 = None
    mul_652: "f32[80]" = torch.ops.aten.mul.Tensor(sum_24, 0.002551020408163265)
    unsqueeze_431: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_652, 0);  mul_652 = None
    unsqueeze_432: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_431, 2);  unsqueeze_431 = None
    unsqueeze_433: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_432, 3);  unsqueeze_432 = None
    mul_653: "f32[80]" = torch.ops.aten.mul.Tensor(sum_25, 0.002551020408163265)
    mul_654: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_211, squeeze_211)
    mul_655: "f32[80]" = torch.ops.aten.mul.Tensor(mul_653, mul_654);  mul_653 = mul_654 = None
    unsqueeze_434: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_655, 0);  mul_655 = None
    unsqueeze_435: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_434, 2);  unsqueeze_434 = None
    unsqueeze_436: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_435, 3);  unsqueeze_435 = None
    mul_656: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_211, primals_240);  primals_240 = None
    unsqueeze_437: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_656, 0);  mul_656 = None
    unsqueeze_438: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_437, 2);  unsqueeze_437 = None
    unsqueeze_439: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_438, 3);  unsqueeze_438 = None
    mul_657: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(sub_116, unsqueeze_436);  sub_116 = unsqueeze_436 = None
    sub_118: "f32[8, 80, 7, 7]" = torch.ops.aten.sub.Tensor(slice_191, mul_657);  slice_191 = mul_657 = None
    sub_119: "f32[8, 80, 7, 7]" = torch.ops.aten.sub.Tensor(sub_118, unsqueeze_433);  sub_118 = unsqueeze_433 = None
    mul_658: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(sub_119, unsqueeze_439);  sub_119 = unsqueeze_439 = None
    mul_659: "f32[80]" = torch.ops.aten.mul.Tensor(sum_25, squeeze_211);  sum_25 = squeeze_211 = None
    convolution_backward_12 = torch.ops.aten.convolution_backward.default(mul_658, add_368, primals_239, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 80, [True, True, False]);  mul_658 = add_368 = primals_239 = None
    getitem_196: "f32[8, 80, 7, 7]" = convolution_backward_12[0]
    getitem_197: "f32[80, 1, 3, 3]" = convolution_backward_12[1];  convolution_backward_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    slice_192: "f32[8, 80, 7, 7]" = torch.ops.aten.slice.Tensor(as_strided_scatter_5, 1, 0, 80);  as_strided_scatter_5 = None
    add_430: "f32[8, 80, 7, 7]" = torch.ops.aten.add.Tensor(slice_192, getitem_196);  slice_192 = getitem_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    sum_26: "f32[80]" = torch.ops.aten.sum.dim_IntList(add_430, [0, 2, 3])
    sub_120: "f32[8, 80, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_81, unsqueeze_442);  convolution_81 = unsqueeze_442 = None
    mul_660: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(add_430, sub_120)
    sum_27: "f32[80]" = torch.ops.aten.sum.dim_IntList(mul_660, [0, 2, 3]);  mul_660 = None
    mul_661: "f32[80]" = torch.ops.aten.mul.Tensor(sum_26, 0.002551020408163265)
    unsqueeze_443: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_661, 0);  mul_661 = None
    unsqueeze_444: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_443, 2);  unsqueeze_443 = None
    unsqueeze_445: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_444, 3);  unsqueeze_444 = None
    mul_662: "f32[80]" = torch.ops.aten.mul.Tensor(sum_27, 0.002551020408163265)
    mul_663: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_208, squeeze_208)
    mul_664: "f32[80]" = torch.ops.aten.mul.Tensor(mul_662, mul_663);  mul_662 = mul_663 = None
    unsqueeze_446: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_664, 0);  mul_664 = None
    unsqueeze_447: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_446, 2);  unsqueeze_446 = None
    unsqueeze_448: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_447, 3);  unsqueeze_447 = None
    mul_665: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_208, primals_237);  primals_237 = None
    unsqueeze_449: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_665, 0);  mul_665 = None
    unsqueeze_450: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_449, 2);  unsqueeze_449 = None
    unsqueeze_451: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_450, 3);  unsqueeze_450 = None
    mul_666: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(sub_120, unsqueeze_448);  sub_120 = unsqueeze_448 = None
    sub_122: "f32[8, 80, 7, 7]" = torch.ops.aten.sub.Tensor(add_430, mul_666);  add_430 = mul_666 = None
    sub_123: "f32[8, 80, 7, 7]" = torch.ops.aten.sub.Tensor(sub_122, unsqueeze_445);  sub_122 = unsqueeze_445 = None
    mul_667: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(sub_123, unsqueeze_451);  sub_123 = unsqueeze_451 = None
    mul_668: "f32[80]" = torch.ops.aten.mul.Tensor(sum_27, squeeze_208);  sum_27 = squeeze_208 = None
    convolution_backward_13 = torch.ops.aten.convolution_backward.default(mul_667, mul_488, primals_236, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_667 = mul_488 = primals_236 = None
    getitem_199: "f32[8, 960, 7, 7]" = convolution_backward_13[0]
    getitem_200: "f32[80, 960, 1, 1]" = convolution_backward_13[1];  convolution_backward_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_669: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_199, slice_146);  slice_146 = None
    mul_670: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_199, div_5);  getitem_199 = div_5 = None
    sum_28: "f32[8, 960, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_669, [2, 3], True);  mul_669 = None
    mul_671: "f32[8, 960, 1, 1]" = torch.ops.aten.mul.Tensor(sum_28, 0.16666666666666666);  sum_28 = None
    where_8: "f32[8, 960, 1, 1]" = torch.ops.aten.where.self(bitwise_and_1, mul_671, full_default);  bitwise_and_1 = mul_671 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    sum_29: "f32[960]" = torch.ops.aten.sum.dim_IntList(where_8, [0, 2, 3])
    convolution_backward_14 = torch.ops.aten.convolution_backward.default(where_8, relu_34, primals_234, [960], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_8 = primals_234 = None
    getitem_202: "f32[8, 240, 1, 1]" = convolution_backward_14[0]
    getitem_203: "f32[960, 240, 1, 1]" = convolution_backward_14[1];  convolution_backward_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    alias_64: "f32[8, 240, 1, 1]" = torch.ops.aten.alias.default(relu_34);  relu_34 = None
    alias_65: "f32[8, 240, 1, 1]" = torch.ops.aten.alias.default(alias_64);  alias_64 = None
    le_7: "b8[8, 240, 1, 1]" = torch.ops.aten.le.Scalar(alias_65, 0);  alias_65 = None
    where_9: "f32[8, 240, 1, 1]" = torch.ops.aten.where.self(le_7, full_default, getitem_202);  le_7 = getitem_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    sum_30: "f32[240]" = torch.ops.aten.sum.dim_IntList(where_9, [0, 2, 3])
    convolution_backward_15 = torch.ops.aten.convolution_backward.default(where_9, mean_5, primals_232, [240], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_9 = mean_5 = primals_232 = None
    getitem_205: "f32[8, 960, 1, 1]" = convolution_backward_15[0]
    getitem_206: "f32[240, 960, 1, 1]" = convolution_backward_15[1];  convolution_backward_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_2: "f32[8, 960, 7, 7]" = torch.ops.aten.expand.default(getitem_205, [8, 960, 7, 7]);  getitem_205 = None
    div_9: "f32[8, 960, 7, 7]" = torch.ops.aten.div.Scalar(expand_2, 49);  expand_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_431: "f32[8, 960, 7, 7]" = torch.ops.aten.add.Tensor(mul_670, div_9);  mul_670 = div_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:63, code: return out[:, :self.out_chs, :, :]
    slice_scatter_54: "f32[8, 960, 7, 7]" = torch.ops.aten.slice_scatter.default(full_default_4, add_431, 3, 0, 9223372036854775807);  add_431 = None
    slice_scatter_55: "f32[8, 960, 7, 7]" = torch.ops.aten.slice_scatter.default(full_default_4, slice_scatter_54, 2, 0, 9223372036854775807);  slice_scatter_54 = None
    slice_scatter_56: "f32[8, 960, 7, 7]" = torch.ops.aten.slice_scatter.default(full_default_4, slice_scatter_55, 0, 0, 9223372036854775807);  slice_scatter_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    slice_193: "f32[8, 480, 7, 7]" = torch.ops.aten.slice.Tensor(slice_scatter_56, 1, 0, 480)
    slice_194: "f32[8, 480, 7, 7]" = torch.ops.aten.slice.Tensor(slice_scatter_56, 1, 480, 960);  slice_scatter_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    where_10: "f32[8, 480, 7, 7]" = torch.ops.aten.where.self(le_8, full_default, slice_194);  le_8 = slice_194 = None
    sum_31: "f32[480]" = torch.ops.aten.sum.dim_IntList(where_10, [0, 2, 3])
    sub_124: "f32[8, 480, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_78, unsqueeze_454);  convolution_78 = unsqueeze_454 = None
    mul_672: "f32[8, 480, 7, 7]" = torch.ops.aten.mul.Tensor(where_10, sub_124)
    sum_32: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_672, [0, 2, 3]);  mul_672 = None
    mul_673: "f32[480]" = torch.ops.aten.mul.Tensor(sum_31, 0.002551020408163265)
    unsqueeze_455: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_673, 0);  mul_673 = None
    unsqueeze_456: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_455, 2);  unsqueeze_455 = None
    unsqueeze_457: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_456, 3);  unsqueeze_456 = None
    mul_674: "f32[480]" = torch.ops.aten.mul.Tensor(sum_32, 0.002551020408163265)
    mul_675: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_205, squeeze_205)
    mul_676: "f32[480]" = torch.ops.aten.mul.Tensor(mul_674, mul_675);  mul_674 = mul_675 = None
    unsqueeze_458: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_676, 0);  mul_676 = None
    unsqueeze_459: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_458, 2);  unsqueeze_458 = None
    unsqueeze_460: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_459, 3);  unsqueeze_459 = None
    mul_677: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_205, primals_230);  primals_230 = None
    unsqueeze_461: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_677, 0);  mul_677 = None
    unsqueeze_462: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_461, 2);  unsqueeze_461 = None
    unsqueeze_463: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_462, 3);  unsqueeze_462 = None
    mul_678: "f32[8, 480, 7, 7]" = torch.ops.aten.mul.Tensor(sub_124, unsqueeze_460);  sub_124 = unsqueeze_460 = None
    sub_126: "f32[8, 480, 7, 7]" = torch.ops.aten.sub.Tensor(where_10, mul_678);  where_10 = mul_678 = None
    sub_127: "f32[8, 480, 7, 7]" = torch.ops.aten.sub.Tensor(sub_126, unsqueeze_457);  sub_126 = unsqueeze_457 = None
    mul_679: "f32[8, 480, 7, 7]" = torch.ops.aten.mul.Tensor(sub_127, unsqueeze_463);  sub_127 = unsqueeze_463 = None
    mul_680: "f32[480]" = torch.ops.aten.mul.Tensor(sum_32, squeeze_205);  sum_32 = squeeze_205 = None
    convolution_backward_16 = torch.ops.aten.convolution_backward.default(mul_679, relu_32, primals_229, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 480, [True, True, False]);  mul_679 = primals_229 = None
    getitem_208: "f32[8, 480, 7, 7]" = convolution_backward_16[0]
    getitem_209: "f32[480, 1, 3, 3]" = convolution_backward_16[1];  convolution_backward_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    add_432: "f32[8, 480, 7, 7]" = torch.ops.aten.add.Tensor(slice_193, getitem_208);  slice_193 = getitem_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    alias_70: "f32[8, 480, 7, 7]" = torch.ops.aten.alias.default(relu_32);  relu_32 = None
    alias_71: "f32[8, 480, 7, 7]" = torch.ops.aten.alias.default(alias_70);  alias_70 = None
    le_9: "b8[8, 480, 7, 7]" = torch.ops.aten.le.Scalar(alias_71, 0);  alias_71 = None
    where_11: "f32[8, 480, 7, 7]" = torch.ops.aten.where.self(le_9, full_default, add_432);  le_9 = add_432 = None
    sum_33: "f32[480]" = torch.ops.aten.sum.dim_IntList(where_11, [0, 2, 3])
    sub_128: "f32[8, 480, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_77, unsqueeze_466);  convolution_77 = unsqueeze_466 = None
    mul_681: "f32[8, 480, 7, 7]" = torch.ops.aten.mul.Tensor(where_11, sub_128)
    sum_34: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_681, [0, 2, 3]);  mul_681 = None
    mul_682: "f32[480]" = torch.ops.aten.mul.Tensor(sum_33, 0.002551020408163265)
    unsqueeze_467: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_682, 0);  mul_682 = None
    unsqueeze_468: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_467, 2);  unsqueeze_467 = None
    unsqueeze_469: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_468, 3);  unsqueeze_468 = None
    mul_683: "f32[480]" = torch.ops.aten.mul.Tensor(sum_34, 0.002551020408163265)
    mul_684: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_202, squeeze_202)
    mul_685: "f32[480]" = torch.ops.aten.mul.Tensor(mul_683, mul_684);  mul_683 = mul_684 = None
    unsqueeze_470: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_685, 0);  mul_685 = None
    unsqueeze_471: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_470, 2);  unsqueeze_470 = None
    unsqueeze_472: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_471, 3);  unsqueeze_471 = None
    mul_686: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_202, primals_227);  primals_227 = None
    unsqueeze_473: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_686, 0);  mul_686 = None
    unsqueeze_474: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_473, 2);  unsqueeze_473 = None
    unsqueeze_475: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_474, 3);  unsqueeze_474 = None
    mul_687: "f32[8, 480, 7, 7]" = torch.ops.aten.mul.Tensor(sub_128, unsqueeze_472);  sub_128 = unsqueeze_472 = None
    sub_130: "f32[8, 480, 7, 7]" = torch.ops.aten.sub.Tensor(where_11, mul_687);  where_11 = mul_687 = None
    sub_131: "f32[8, 480, 7, 7]" = torch.ops.aten.sub.Tensor(sub_130, unsqueeze_469);  sub_130 = unsqueeze_469 = None
    mul_688: "f32[8, 480, 7, 7]" = torch.ops.aten.mul.Tensor(sub_131, unsqueeze_475);  sub_131 = unsqueeze_475 = None
    mul_689: "f32[480]" = torch.ops.aten.mul.Tensor(sum_34, squeeze_202);  sum_34 = squeeze_202 = None
    convolution_backward_17 = torch.ops.aten.convolution_backward.default(mul_688, slice_143, primals_226, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_688 = slice_143 = primals_226 = None
    getitem_211: "f32[8, 160, 7, 7]" = convolution_backward_17[0]
    getitem_212: "f32[480, 160, 1, 1]" = convolution_backward_17[1];  convolution_backward_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    add_433: "f32[8, 160, 7, 7]" = torch.ops.aten.add.Tensor(clone_2, getitem_211);  clone_2 = getitem_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:182, code: x += self.shortcut(shortcut)
    copy_9: "f32[8, 160, 7, 7]" = torch.ops.aten.copy.default(as_strided, add_433);  add_433 = None
    as_strided_scatter_6: "f32[62720]" = torch.ops.aten.as_strided_scatter.default(full, copy_9, [8, 160, 7, 7], [7840, 49, 7, 1], 0);  copy_9 = None
    as_strided_24: "f32[8, 160, 7, 7]" = torch.ops.aten.as_strided.default(as_strided_scatter_6, [8, 160, 7, 7], [7840, 49, 7, 1], 0);  as_strided_scatter_6 = None
    new_empty_strided_3: "f32[8, 160, 7, 7]" = torch.ops.aten.new_empty_strided.default(as_strided_24, [8, 160, 7, 7], [7840, 49, 7, 1])
    copy_10: "f32[8, 160, 7, 7]" = torch.ops.aten.copy.default(new_empty_strided_3, as_strided_24);  new_empty_strided_3 = as_strided_24 = None
    as_strided_26: "f32[8, 160, 7, 7]" = torch.ops.aten.as_strided.default(copy_10, [8, 160, 7, 7], [7840, 49, 7, 1], 0)
    clone_3: "f32[8, 160, 7, 7]" = torch.ops.aten.clone.default(as_strided_26, memory_format = torch.contiguous_format)
    copy_11: "f32[8, 160, 7, 7]" = torch.ops.aten.copy.default(as_strided_26, clone_3);  as_strided_26 = None
    as_strided_scatter_7: "f32[8, 160, 7, 7]" = torch.ops.aten.as_strided_scatter.default(copy_10, copy_11, [8, 160, 7, 7], [7840, 49, 7, 1], 0);  copy_10 = copy_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    slice_197: "f32[8, 80, 7, 7]" = torch.ops.aten.slice.Tensor(as_strided_scatter_7, 1, 80, 160)
    sum_35: "f32[80]" = torch.ops.aten.sum.dim_IntList(slice_197, [0, 2, 3])
    sub_132: "f32[8, 80, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_76, unsqueeze_478);  convolution_76 = unsqueeze_478 = None
    mul_690: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(slice_197, sub_132)
    sum_36: "f32[80]" = torch.ops.aten.sum.dim_IntList(mul_690, [0, 2, 3]);  mul_690 = None
    mul_691: "f32[80]" = torch.ops.aten.mul.Tensor(sum_35, 0.002551020408163265)
    unsqueeze_479: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_691, 0);  mul_691 = None
    unsqueeze_480: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_479, 2);  unsqueeze_479 = None
    unsqueeze_481: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_480, 3);  unsqueeze_480 = None
    mul_692: "f32[80]" = torch.ops.aten.mul.Tensor(sum_36, 0.002551020408163265)
    mul_693: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_199, squeeze_199)
    mul_694: "f32[80]" = torch.ops.aten.mul.Tensor(mul_692, mul_693);  mul_692 = mul_693 = None
    unsqueeze_482: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_694, 0);  mul_694 = None
    unsqueeze_483: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_482, 2);  unsqueeze_482 = None
    unsqueeze_484: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_483, 3);  unsqueeze_483 = None
    mul_695: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_199, primals_224);  primals_224 = None
    unsqueeze_485: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_695, 0);  mul_695 = None
    unsqueeze_486: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_485, 2);  unsqueeze_485 = None
    unsqueeze_487: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_486, 3);  unsqueeze_486 = None
    mul_696: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(sub_132, unsqueeze_484);  sub_132 = unsqueeze_484 = None
    sub_134: "f32[8, 80, 7, 7]" = torch.ops.aten.sub.Tensor(slice_197, mul_696);  slice_197 = mul_696 = None
    sub_135: "f32[8, 80, 7, 7]" = torch.ops.aten.sub.Tensor(sub_134, unsqueeze_481);  sub_134 = unsqueeze_481 = None
    mul_697: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(sub_135, unsqueeze_487);  sub_135 = unsqueeze_487 = None
    mul_698: "f32[80]" = torch.ops.aten.mul.Tensor(sum_36, squeeze_199);  sum_36 = squeeze_199 = None
    convolution_backward_18 = torch.ops.aten.convolution_backward.default(mul_697, add_346, primals_223, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 80, [True, True, False]);  mul_697 = add_346 = primals_223 = None
    getitem_214: "f32[8, 80, 7, 7]" = convolution_backward_18[0]
    getitem_215: "f32[80, 1, 3, 3]" = convolution_backward_18[1];  convolution_backward_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    slice_198: "f32[8, 80, 7, 7]" = torch.ops.aten.slice.Tensor(as_strided_scatter_7, 1, 0, 80);  as_strided_scatter_7 = None
    add_434: "f32[8, 80, 7, 7]" = torch.ops.aten.add.Tensor(slice_198, getitem_214);  slice_198 = getitem_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    sum_37: "f32[80]" = torch.ops.aten.sum.dim_IntList(add_434, [0, 2, 3])
    sub_136: "f32[8, 80, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_75, unsqueeze_490);  convolution_75 = unsqueeze_490 = None
    mul_699: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(add_434, sub_136)
    sum_38: "f32[80]" = torch.ops.aten.sum.dim_IntList(mul_699, [0, 2, 3]);  mul_699 = None
    mul_700: "f32[80]" = torch.ops.aten.mul.Tensor(sum_37, 0.002551020408163265)
    unsqueeze_491: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_700, 0);  mul_700 = None
    unsqueeze_492: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_491, 2);  unsqueeze_491 = None
    unsqueeze_493: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_492, 3);  unsqueeze_492 = None
    mul_701: "f32[80]" = torch.ops.aten.mul.Tensor(sum_38, 0.002551020408163265)
    mul_702: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_196, squeeze_196)
    mul_703: "f32[80]" = torch.ops.aten.mul.Tensor(mul_701, mul_702);  mul_701 = mul_702 = None
    unsqueeze_494: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_703, 0);  mul_703 = None
    unsqueeze_495: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_494, 2);  unsqueeze_494 = None
    unsqueeze_496: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_495, 3);  unsqueeze_495 = None
    mul_704: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_196, primals_221);  primals_221 = None
    unsqueeze_497: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_704, 0);  mul_704 = None
    unsqueeze_498: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_497, 2);  unsqueeze_497 = None
    unsqueeze_499: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_498, 3);  unsqueeze_498 = None
    mul_705: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(sub_136, unsqueeze_496);  sub_136 = unsqueeze_496 = None
    sub_138: "f32[8, 80, 7, 7]" = torch.ops.aten.sub.Tensor(add_434, mul_705);  add_434 = mul_705 = None
    sub_139: "f32[8, 80, 7, 7]" = torch.ops.aten.sub.Tensor(sub_138, unsqueeze_493);  sub_138 = unsqueeze_493 = None
    mul_706: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(sub_139, unsqueeze_499);  sub_139 = unsqueeze_499 = None
    mul_707: "f32[80]" = torch.ops.aten.mul.Tensor(sum_38, squeeze_196);  sum_38 = squeeze_196 = None
    convolution_backward_19 = torch.ops.aten.convolution_backward.default(mul_706, slice_135, primals_220, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_706 = slice_135 = primals_220 = None
    getitem_217: "f32[8, 960, 7, 7]" = convolution_backward_19[0]
    getitem_218: "f32[80, 960, 1, 1]" = convolution_backward_19[1];  convolution_backward_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:63, code: return out[:, :self.out_chs, :, :]
    slice_scatter_57: "f32[8, 960, 7, 7]" = torch.ops.aten.slice_scatter.default(full_default_4, getitem_217, 3, 0, 9223372036854775807);  getitem_217 = None
    slice_scatter_58: "f32[8, 960, 7, 7]" = torch.ops.aten.slice_scatter.default(full_default_4, slice_scatter_57, 2, 0, 9223372036854775807);  slice_scatter_57 = None
    slice_scatter_59: "f32[8, 960, 7, 7]" = torch.ops.aten.slice_scatter.default(full_default_4, slice_scatter_58, 0, 0, 9223372036854775807);  full_default_4 = slice_scatter_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    slice_199: "f32[8, 480, 7, 7]" = torch.ops.aten.slice.Tensor(slice_scatter_59, 1, 0, 480)
    slice_200: "f32[8, 480, 7, 7]" = torch.ops.aten.slice.Tensor(slice_scatter_59, 1, 480, 960);  slice_scatter_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    where_12: "f32[8, 480, 7, 7]" = torch.ops.aten.where.self(le_10, full_default, slice_200);  le_10 = slice_200 = None
    sum_39: "f32[480]" = torch.ops.aten.sum.dim_IntList(where_12, [0, 2, 3])
    sub_140: "f32[8, 480, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_74, unsqueeze_502);  convolution_74 = unsqueeze_502 = None
    mul_708: "f32[8, 480, 7, 7]" = torch.ops.aten.mul.Tensor(where_12, sub_140)
    sum_40: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_708, [0, 2, 3]);  mul_708 = None
    mul_709: "f32[480]" = torch.ops.aten.mul.Tensor(sum_39, 0.002551020408163265)
    unsqueeze_503: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_709, 0);  mul_709 = None
    unsqueeze_504: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_503, 2);  unsqueeze_503 = None
    unsqueeze_505: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_504, 3);  unsqueeze_504 = None
    mul_710: "f32[480]" = torch.ops.aten.mul.Tensor(sum_40, 0.002551020408163265)
    mul_711: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_193, squeeze_193)
    mul_712: "f32[480]" = torch.ops.aten.mul.Tensor(mul_710, mul_711);  mul_710 = mul_711 = None
    unsqueeze_506: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_712, 0);  mul_712 = None
    unsqueeze_507: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_506, 2);  unsqueeze_506 = None
    unsqueeze_508: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_507, 3);  unsqueeze_507 = None
    mul_713: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_193, primals_218);  primals_218 = None
    unsqueeze_509: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_713, 0);  mul_713 = None
    unsqueeze_510: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_509, 2);  unsqueeze_509 = None
    unsqueeze_511: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_510, 3);  unsqueeze_510 = None
    mul_714: "f32[8, 480, 7, 7]" = torch.ops.aten.mul.Tensor(sub_140, unsqueeze_508);  sub_140 = unsqueeze_508 = None
    sub_142: "f32[8, 480, 7, 7]" = torch.ops.aten.sub.Tensor(where_12, mul_714);  where_12 = mul_714 = None
    sub_143: "f32[8, 480, 7, 7]" = torch.ops.aten.sub.Tensor(sub_142, unsqueeze_505);  sub_142 = unsqueeze_505 = None
    mul_715: "f32[8, 480, 7, 7]" = torch.ops.aten.mul.Tensor(sub_143, unsqueeze_511);  sub_143 = unsqueeze_511 = None
    mul_716: "f32[480]" = torch.ops.aten.mul.Tensor(sum_40, squeeze_193);  sum_40 = squeeze_193 = None
    convolution_backward_20 = torch.ops.aten.convolution_backward.default(mul_715, relu_30, primals_217, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 480, [True, True, False]);  mul_715 = primals_217 = None
    getitem_220: "f32[8, 480, 7, 7]" = convolution_backward_20[0]
    getitem_221: "f32[480, 1, 3, 3]" = convolution_backward_20[1];  convolution_backward_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    add_435: "f32[8, 480, 7, 7]" = torch.ops.aten.add.Tensor(slice_199, getitem_220);  slice_199 = getitem_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    alias_76: "f32[8, 480, 7, 7]" = torch.ops.aten.alias.default(relu_30);  relu_30 = None
    alias_77: "f32[8, 480, 7, 7]" = torch.ops.aten.alias.default(alias_76);  alias_76 = None
    le_11: "b8[8, 480, 7, 7]" = torch.ops.aten.le.Scalar(alias_77, 0);  alias_77 = None
    where_13: "f32[8, 480, 7, 7]" = torch.ops.aten.where.self(le_11, full_default, add_435);  le_11 = add_435 = None
    sum_41: "f32[480]" = torch.ops.aten.sum.dim_IntList(where_13, [0, 2, 3])
    sub_144: "f32[8, 480, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_73, unsqueeze_514);  convolution_73 = unsqueeze_514 = None
    mul_717: "f32[8, 480, 7, 7]" = torch.ops.aten.mul.Tensor(where_13, sub_144)
    sum_42: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_717, [0, 2, 3]);  mul_717 = None
    mul_718: "f32[480]" = torch.ops.aten.mul.Tensor(sum_41, 0.002551020408163265)
    unsqueeze_515: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_718, 0);  mul_718 = None
    unsqueeze_516: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_515, 2);  unsqueeze_515 = None
    unsqueeze_517: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_516, 3);  unsqueeze_516 = None
    mul_719: "f32[480]" = torch.ops.aten.mul.Tensor(sum_42, 0.002551020408163265)
    mul_720: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_190, squeeze_190)
    mul_721: "f32[480]" = torch.ops.aten.mul.Tensor(mul_719, mul_720);  mul_719 = mul_720 = None
    unsqueeze_518: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_721, 0);  mul_721 = None
    unsqueeze_519: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_518, 2);  unsqueeze_518 = None
    unsqueeze_520: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_519, 3);  unsqueeze_519 = None
    mul_722: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_190, primals_215);  primals_215 = None
    unsqueeze_521: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_722, 0);  mul_722 = None
    unsqueeze_522: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_521, 2);  unsqueeze_521 = None
    unsqueeze_523: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_522, 3);  unsqueeze_522 = None
    mul_723: "f32[8, 480, 7, 7]" = torch.ops.aten.mul.Tensor(sub_144, unsqueeze_520);  sub_144 = unsqueeze_520 = None
    sub_146: "f32[8, 480, 7, 7]" = torch.ops.aten.sub.Tensor(where_13, mul_723);  where_13 = mul_723 = None
    sub_147: "f32[8, 480, 7, 7]" = torch.ops.aten.sub.Tensor(sub_146, unsqueeze_517);  sub_146 = unsqueeze_517 = None
    mul_724: "f32[8, 480, 7, 7]" = torch.ops.aten.mul.Tensor(sub_147, unsqueeze_523);  sub_147 = unsqueeze_523 = None
    mul_725: "f32[480]" = torch.ops.aten.mul.Tensor(sum_42, squeeze_190);  sum_42 = squeeze_190 = None
    convolution_backward_21 = torch.ops.aten.convolution_backward.default(mul_724, slice_132, primals_214, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_724 = slice_132 = primals_214 = None
    getitem_223: "f32[8, 160, 7, 7]" = convolution_backward_21[0]
    getitem_224: "f32[480, 160, 1, 1]" = convolution_backward_21[1];  convolution_backward_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    add_436: "f32[8, 160, 7, 7]" = torch.ops.aten.add.Tensor(clone_3, getitem_223);  clone_3 = getitem_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:182, code: x += self.shortcut(shortcut)
    copy_12: "f32[8, 160, 7, 7]" = torch.ops.aten.copy.default(as_strided, add_436);  as_strided = add_436 = None
    as_strided_scatter_8: "f32[62720]" = torch.ops.aten.as_strided_scatter.default(full, copy_12, [8, 160, 7, 7], [7840, 49, 7, 1], 0);  full = copy_12 = None
    as_strided_31: "f32[8, 160, 7, 7]" = torch.ops.aten.as_strided.default(as_strided_scatter_8, [8, 160, 7, 7], [7840, 49, 7, 1], 0);  as_strided_scatter_8 = None
    new_empty_strided_4: "f32[8, 160, 7, 7]" = torch.ops.aten.new_empty_strided.default(as_strided_31, [8, 160, 7, 7], [7840, 49, 7, 1])
    copy_13: "f32[8, 160, 7, 7]" = torch.ops.aten.copy.default(new_empty_strided_4, as_strided_31);  new_empty_strided_4 = as_strided_31 = None
    as_strided_33: "f32[8, 160, 7, 7]" = torch.ops.aten.as_strided.default(copy_13, [8, 160, 7, 7], [7840, 49, 7, 1], 0)
    clone_4: "f32[8, 160, 7, 7]" = torch.ops.aten.clone.default(as_strided_33, memory_format = torch.contiguous_format)
    copy_14: "f32[8, 160, 7, 7]" = torch.ops.aten.copy.default(as_strided_33, clone_4);  as_strided_33 = None
    as_strided_scatter_9: "f32[8, 160, 7, 7]" = torch.ops.aten.as_strided_scatter.default(copy_13, copy_14, [8, 160, 7, 7], [7840, 49, 7, 1], 0);  copy_13 = copy_14 = None
    sum_43: "f32[160]" = torch.ops.aten.sum.dim_IntList(clone_4, [0, 2, 3])
    sub_148: "f32[8, 160, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_72, unsqueeze_526);  convolution_72 = unsqueeze_526 = None
    mul_726: "f32[8, 160, 7, 7]" = torch.ops.aten.mul.Tensor(clone_4, sub_148)
    sum_44: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_726, [0, 2, 3]);  mul_726 = None
    mul_727: "f32[160]" = torch.ops.aten.mul.Tensor(sum_43, 0.002551020408163265)
    unsqueeze_527: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_727, 0);  mul_727 = None
    unsqueeze_528: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_527, 2);  unsqueeze_527 = None
    unsqueeze_529: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_528, 3);  unsqueeze_528 = None
    mul_728: "f32[160]" = torch.ops.aten.mul.Tensor(sum_44, 0.002551020408163265)
    mul_729: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_187, squeeze_187)
    mul_730: "f32[160]" = torch.ops.aten.mul.Tensor(mul_728, mul_729);  mul_728 = mul_729 = None
    unsqueeze_530: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_730, 0);  mul_730 = None
    unsqueeze_531: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_530, 2);  unsqueeze_530 = None
    unsqueeze_532: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_531, 3);  unsqueeze_531 = None
    mul_731: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_187, primals_212);  primals_212 = None
    unsqueeze_533: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_731, 0);  mul_731 = None
    unsqueeze_534: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_533, 2);  unsqueeze_533 = None
    unsqueeze_535: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_534, 3);  unsqueeze_534 = None
    mul_732: "f32[8, 160, 7, 7]" = torch.ops.aten.mul.Tensor(sub_148, unsqueeze_532);  sub_148 = unsqueeze_532 = None
    sub_150: "f32[8, 160, 7, 7]" = torch.ops.aten.sub.Tensor(clone_4, mul_732);  clone_4 = mul_732 = None
    sub_151: "f32[8, 160, 7, 7]" = torch.ops.aten.sub.Tensor(sub_150, unsqueeze_529);  sub_150 = unsqueeze_529 = None
    mul_733: "f32[8, 160, 7, 7]" = torch.ops.aten.mul.Tensor(sub_151, unsqueeze_535);  sub_151 = unsqueeze_535 = None
    mul_734: "f32[160]" = torch.ops.aten.mul.Tensor(sum_44, squeeze_187);  sum_44 = squeeze_187 = None
    convolution_backward_22 = torch.ops.aten.convolution_backward.default(mul_733, add_325, primals_211, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_733 = add_325 = primals_211 = None
    getitem_226: "f32[8, 112, 7, 7]" = convolution_backward_22[0]
    getitem_227: "f32[160, 112, 1, 1]" = convolution_backward_22[1];  convolution_backward_22 = None
    sum_45: "f32[112]" = torch.ops.aten.sum.dim_IntList(getitem_226, [0, 2, 3])
    sub_152: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_71, unsqueeze_538);  convolution_71 = unsqueeze_538 = None
    mul_735: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_226, sub_152)
    sum_46: "f32[112]" = torch.ops.aten.sum.dim_IntList(mul_735, [0, 2, 3]);  mul_735 = None
    mul_736: "f32[112]" = torch.ops.aten.mul.Tensor(sum_45, 0.002551020408163265)
    unsqueeze_539: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_736, 0);  mul_736 = None
    unsqueeze_540: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_539, 2);  unsqueeze_539 = None
    unsqueeze_541: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_540, 3);  unsqueeze_540 = None
    mul_737: "f32[112]" = torch.ops.aten.mul.Tensor(sum_46, 0.002551020408163265)
    mul_738: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_184, squeeze_184)
    mul_739: "f32[112]" = torch.ops.aten.mul.Tensor(mul_737, mul_738);  mul_737 = mul_738 = None
    unsqueeze_542: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_739, 0);  mul_739 = None
    unsqueeze_543: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_542, 2);  unsqueeze_542 = None
    unsqueeze_544: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_543, 3);  unsqueeze_543 = None
    mul_740: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_184, primals_209);  primals_209 = None
    unsqueeze_545: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_740, 0);  mul_740 = None
    unsqueeze_546: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_545, 2);  unsqueeze_545 = None
    unsqueeze_547: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_546, 3);  unsqueeze_546 = None
    mul_741: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_152, unsqueeze_544);  sub_152 = unsqueeze_544 = None
    sub_154: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(getitem_226, mul_741);  getitem_226 = mul_741 = None
    sub_155: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(sub_154, unsqueeze_541);  sub_154 = unsqueeze_541 = None
    mul_742: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_155, unsqueeze_547);  sub_155 = unsqueeze_547 = None
    mul_743: "f32[112]" = torch.ops.aten.mul.Tensor(sum_46, squeeze_184);  sum_46 = squeeze_184 = None
    convolution_backward_23 = torch.ops.aten.convolution_backward.default(mul_742, slice_121, primals_208, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 112, [True, True, False]);  mul_742 = primals_208 = None
    getitem_229: "f32[8, 112, 14, 14]" = convolution_backward_23[0]
    getitem_230: "f32[112, 1, 5, 5]" = convolution_backward_23[1];  convolution_backward_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    slice_203: "f32[8, 80, 7, 7]" = torch.ops.aten.slice.Tensor(as_strided_scatter_9, 1, 80, 160)
    sum_47: "f32[80]" = torch.ops.aten.sum.dim_IntList(slice_203, [0, 2, 3])
    sub_156: "f32[8, 80, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_70, unsqueeze_550);  convolution_70 = unsqueeze_550 = None
    mul_744: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(slice_203, sub_156)
    sum_48: "f32[80]" = torch.ops.aten.sum.dim_IntList(mul_744, [0, 2, 3]);  mul_744 = None
    mul_745: "f32[80]" = torch.ops.aten.mul.Tensor(sum_47, 0.002551020408163265)
    unsqueeze_551: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_745, 0);  mul_745 = None
    unsqueeze_552: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_551, 2);  unsqueeze_551 = None
    unsqueeze_553: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_552, 3);  unsqueeze_552 = None
    mul_746: "f32[80]" = torch.ops.aten.mul.Tensor(sum_48, 0.002551020408163265)
    mul_747: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_181, squeeze_181)
    mul_748: "f32[80]" = torch.ops.aten.mul.Tensor(mul_746, mul_747);  mul_746 = mul_747 = None
    unsqueeze_554: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_748, 0);  mul_748 = None
    unsqueeze_555: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_554, 2);  unsqueeze_554 = None
    unsqueeze_556: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_555, 3);  unsqueeze_555 = None
    mul_749: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_181, primals_206);  primals_206 = None
    unsqueeze_557: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_749, 0);  mul_749 = None
    unsqueeze_558: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_557, 2);  unsqueeze_557 = None
    unsqueeze_559: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_558, 3);  unsqueeze_558 = None
    mul_750: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(sub_156, unsqueeze_556);  sub_156 = unsqueeze_556 = None
    sub_158: "f32[8, 80, 7, 7]" = torch.ops.aten.sub.Tensor(slice_203, mul_750);  slice_203 = mul_750 = None
    sub_159: "f32[8, 80, 7, 7]" = torch.ops.aten.sub.Tensor(sub_158, unsqueeze_553);  sub_158 = unsqueeze_553 = None
    mul_751: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(sub_159, unsqueeze_559);  sub_159 = unsqueeze_559 = None
    mul_752: "f32[80]" = torch.ops.aten.mul.Tensor(sum_48, squeeze_181);  sum_48 = squeeze_181 = None
    convolution_backward_24 = torch.ops.aten.convolution_backward.default(mul_751, add_315, primals_205, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 80, [True, True, False]);  mul_751 = add_315 = primals_205 = None
    getitem_232: "f32[8, 80, 7, 7]" = convolution_backward_24[0]
    getitem_233: "f32[80, 1, 3, 3]" = convolution_backward_24[1];  convolution_backward_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    slice_204: "f32[8, 80, 7, 7]" = torch.ops.aten.slice.Tensor(as_strided_scatter_9, 1, 0, 80);  as_strided_scatter_9 = None
    add_437: "f32[8, 80, 7, 7]" = torch.ops.aten.add.Tensor(slice_204, getitem_232);  slice_204 = getitem_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    sum_49: "f32[80]" = torch.ops.aten.sum.dim_IntList(add_437, [0, 2, 3])
    sub_160: "f32[8, 80, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_69, unsqueeze_562);  convolution_69 = unsqueeze_562 = None
    mul_753: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(add_437, sub_160)
    sum_50: "f32[80]" = torch.ops.aten.sum.dim_IntList(mul_753, [0, 2, 3]);  mul_753 = None
    mul_754: "f32[80]" = torch.ops.aten.mul.Tensor(sum_49, 0.002551020408163265)
    unsqueeze_563: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_754, 0);  mul_754 = None
    unsqueeze_564: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_563, 2);  unsqueeze_563 = None
    unsqueeze_565: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_564, 3);  unsqueeze_564 = None
    mul_755: "f32[80]" = torch.ops.aten.mul.Tensor(sum_50, 0.002551020408163265)
    mul_756: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_178, squeeze_178)
    mul_757: "f32[80]" = torch.ops.aten.mul.Tensor(mul_755, mul_756);  mul_755 = mul_756 = None
    unsqueeze_566: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_757, 0);  mul_757 = None
    unsqueeze_567: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_566, 2);  unsqueeze_566 = None
    unsqueeze_568: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_567, 3);  unsqueeze_567 = None
    mul_758: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_178, primals_203);  primals_203 = None
    unsqueeze_569: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_758, 0);  mul_758 = None
    unsqueeze_570: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_569, 2);  unsqueeze_569 = None
    unsqueeze_571: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_570, 3);  unsqueeze_570 = None
    mul_759: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(sub_160, unsqueeze_568);  sub_160 = unsqueeze_568 = None
    sub_162: "f32[8, 80, 7, 7]" = torch.ops.aten.sub.Tensor(add_437, mul_759);  add_437 = mul_759 = None
    sub_163: "f32[8, 80, 7, 7]" = torch.ops.aten.sub.Tensor(sub_162, unsqueeze_565);  sub_162 = unsqueeze_565 = None
    mul_760: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(sub_163, unsqueeze_571);  sub_163 = unsqueeze_571 = None
    mul_761: "f32[80]" = torch.ops.aten.mul.Tensor(sum_50, squeeze_178);  sum_50 = squeeze_178 = None
    convolution_backward_25 = torch.ops.aten.convolution_backward.default(mul_760, mul_417, primals_202, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_760 = mul_417 = primals_202 = None
    getitem_235: "f32[8, 672, 7, 7]" = convolution_backward_25[0]
    getitem_236: "f32[80, 672, 1, 1]" = convolution_backward_25[1];  convolution_backward_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_762: "f32[8, 672, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_235, add_309);  add_309 = None
    mul_763: "f32[8, 672, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_235, div_4);  getitem_235 = div_4 = None
    sum_51: "f32[8, 672, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_762, [2, 3], True);  mul_762 = None
    mul_764: "f32[8, 672, 1, 1]" = torch.ops.aten.mul.Tensor(sum_51, 0.16666666666666666);  sum_51 = None
    where_14: "f32[8, 672, 1, 1]" = torch.ops.aten.where.self(bitwise_and_2, mul_764, full_default);  bitwise_and_2 = mul_764 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    sum_52: "f32[672]" = torch.ops.aten.sum.dim_IntList(where_14, [0, 2, 3])
    convolution_backward_26 = torch.ops.aten.convolution_backward.default(where_14, relu_29, primals_200, [672], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_14 = primals_200 = None
    getitem_238: "f32[8, 168, 1, 1]" = convolution_backward_26[0]
    getitem_239: "f32[672, 168, 1, 1]" = convolution_backward_26[1];  convolution_backward_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    alias_79: "f32[8, 168, 1, 1]" = torch.ops.aten.alias.default(relu_29);  relu_29 = None
    alias_80: "f32[8, 168, 1, 1]" = torch.ops.aten.alias.default(alias_79);  alias_79 = None
    le_12: "b8[8, 168, 1, 1]" = torch.ops.aten.le.Scalar(alias_80, 0);  alias_80 = None
    where_15: "f32[8, 168, 1, 1]" = torch.ops.aten.where.self(le_12, full_default, getitem_238);  le_12 = getitem_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    sum_53: "f32[168]" = torch.ops.aten.sum.dim_IntList(where_15, [0, 2, 3])
    convolution_backward_27 = torch.ops.aten.convolution_backward.default(where_15, mean_4, primals_198, [168], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_15 = mean_4 = primals_198 = None
    getitem_241: "f32[8, 672, 1, 1]" = convolution_backward_27[0]
    getitem_242: "f32[168, 672, 1, 1]" = convolution_backward_27[1];  convolution_backward_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_3: "f32[8, 672, 7, 7]" = torch.ops.aten.expand.default(getitem_241, [8, 672, 7, 7]);  getitem_241 = None
    div_10: "f32[8, 672, 7, 7]" = torch.ops.aten.div.Scalar(expand_3, 49);  expand_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_438: "f32[8, 672, 7, 7]" = torch.ops.aten.add.Tensor(mul_763, div_10);  mul_763 = div_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:173, code: x = self.bn_dw(x)
    sum_54: "f32[672]" = torch.ops.aten.sum.dim_IntList(add_438, [0, 2, 3])
    sub_164: "f32[8, 672, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_66, unsqueeze_574);  convolution_66 = unsqueeze_574 = None
    mul_765: "f32[8, 672, 7, 7]" = torch.ops.aten.mul.Tensor(add_438, sub_164)
    sum_55: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_765, [0, 2, 3]);  mul_765 = None
    mul_766: "f32[672]" = torch.ops.aten.mul.Tensor(sum_54, 0.002551020408163265)
    unsqueeze_575: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_766, 0);  mul_766 = None
    unsqueeze_576: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_575, 2);  unsqueeze_575 = None
    unsqueeze_577: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_576, 3);  unsqueeze_576 = None
    mul_767: "f32[672]" = torch.ops.aten.mul.Tensor(sum_55, 0.002551020408163265)
    mul_768: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_175, squeeze_175)
    mul_769: "f32[672]" = torch.ops.aten.mul.Tensor(mul_767, mul_768);  mul_767 = mul_768 = None
    unsqueeze_578: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_769, 0);  mul_769 = None
    unsqueeze_579: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_578, 2);  unsqueeze_578 = None
    unsqueeze_580: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_579, 3);  unsqueeze_579 = None
    mul_770: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_175, primals_196);  primals_196 = None
    unsqueeze_581: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_770, 0);  mul_770 = None
    unsqueeze_582: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_581, 2);  unsqueeze_581 = None
    unsqueeze_583: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_582, 3);  unsqueeze_582 = None
    mul_771: "f32[8, 672, 7, 7]" = torch.ops.aten.mul.Tensor(sub_164, unsqueeze_580);  sub_164 = unsqueeze_580 = None
    sub_166: "f32[8, 672, 7, 7]" = torch.ops.aten.sub.Tensor(add_438, mul_771);  add_438 = mul_771 = None
    sub_167: "f32[8, 672, 7, 7]" = torch.ops.aten.sub.Tensor(sub_166, unsqueeze_577);  sub_166 = unsqueeze_577 = None
    mul_772: "f32[8, 672, 7, 7]" = torch.ops.aten.mul.Tensor(sub_167, unsqueeze_583);  sub_167 = unsqueeze_583 = None
    mul_773: "f32[672]" = torch.ops.aten.mul.Tensor(sum_55, squeeze_175);  sum_55 = squeeze_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:172, code: x = self.conv_dw(x)
    convolution_backward_28 = torch.ops.aten.convolution_backward.default(mul_772, slice_124, primals_195, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 672, [True, True, False]);  mul_772 = slice_124 = primals_195 = None
    getitem_244: "f32[8, 672, 14, 14]" = convolution_backward_28[0]
    getitem_245: "f32[672, 1, 5, 5]" = convolution_backward_28[1];  convolution_backward_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:63, code: return out[:, :self.out_chs, :, :]
    full_default_28: "f32[8, 672, 14, 14]" = torch.ops.aten.full.default([8, 672, 14, 14], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_60: "f32[8, 672, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_28, getitem_244, 3, 0, 9223372036854775807);  getitem_244 = None
    slice_scatter_61: "f32[8, 672, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_28, slice_scatter_60, 2, 0, 9223372036854775807);  slice_scatter_60 = None
    slice_scatter_62: "f32[8, 672, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_28, slice_scatter_61, 0, 0, 9223372036854775807);  slice_scatter_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    slice_205: "f32[8, 336, 14, 14]" = torch.ops.aten.slice.Tensor(slice_scatter_62, 1, 0, 336)
    slice_206: "f32[8, 336, 14, 14]" = torch.ops.aten.slice.Tensor(slice_scatter_62, 1, 336, 672);  slice_scatter_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    where_16: "f32[8, 336, 14, 14]" = torch.ops.aten.where.self(le_13, full_default, slice_206);  le_13 = slice_206 = None
    sum_56: "f32[336]" = torch.ops.aten.sum.dim_IntList(where_16, [0, 2, 3])
    sub_168: "f32[8, 336, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_65, unsqueeze_586);  convolution_65 = unsqueeze_586 = None
    mul_774: "f32[8, 336, 14, 14]" = torch.ops.aten.mul.Tensor(where_16, sub_168)
    sum_57: "f32[336]" = torch.ops.aten.sum.dim_IntList(mul_774, [0, 2, 3]);  mul_774 = None
    mul_775: "f32[336]" = torch.ops.aten.mul.Tensor(sum_56, 0.0006377551020408163)
    unsqueeze_587: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(mul_775, 0);  mul_775 = None
    unsqueeze_588: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_587, 2);  unsqueeze_587 = None
    unsqueeze_589: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_588, 3);  unsqueeze_588 = None
    mul_776: "f32[336]" = torch.ops.aten.mul.Tensor(sum_57, 0.0006377551020408163)
    mul_777: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_172, squeeze_172)
    mul_778: "f32[336]" = torch.ops.aten.mul.Tensor(mul_776, mul_777);  mul_776 = mul_777 = None
    unsqueeze_590: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(mul_778, 0);  mul_778 = None
    unsqueeze_591: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_590, 2);  unsqueeze_590 = None
    unsqueeze_592: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_591, 3);  unsqueeze_591 = None
    mul_779: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_172, primals_193);  primals_193 = None
    unsqueeze_593: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(mul_779, 0);  mul_779 = None
    unsqueeze_594: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_593, 2);  unsqueeze_593 = None
    unsqueeze_595: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_594, 3);  unsqueeze_594 = None
    mul_780: "f32[8, 336, 14, 14]" = torch.ops.aten.mul.Tensor(sub_168, unsqueeze_592);  sub_168 = unsqueeze_592 = None
    sub_170: "f32[8, 336, 14, 14]" = torch.ops.aten.sub.Tensor(where_16, mul_780);  where_16 = mul_780 = None
    sub_171: "f32[8, 336, 14, 14]" = torch.ops.aten.sub.Tensor(sub_170, unsqueeze_589);  sub_170 = unsqueeze_589 = None
    mul_781: "f32[8, 336, 14, 14]" = torch.ops.aten.mul.Tensor(sub_171, unsqueeze_595);  sub_171 = unsqueeze_595 = None
    mul_782: "f32[336]" = torch.ops.aten.mul.Tensor(sum_57, squeeze_172);  sum_57 = squeeze_172 = None
    convolution_backward_29 = torch.ops.aten.convolution_backward.default(mul_781, relu_27, primals_192, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 336, [True, True, False]);  mul_781 = primals_192 = None
    getitem_247: "f32[8, 336, 14, 14]" = convolution_backward_29[0]
    getitem_248: "f32[336, 1, 3, 3]" = convolution_backward_29[1];  convolution_backward_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    add_439: "f32[8, 336, 14, 14]" = torch.ops.aten.add.Tensor(slice_205, getitem_247);  slice_205 = getitem_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    alias_85: "f32[8, 336, 14, 14]" = torch.ops.aten.alias.default(relu_27);  relu_27 = None
    alias_86: "f32[8, 336, 14, 14]" = torch.ops.aten.alias.default(alias_85);  alias_85 = None
    le_14: "b8[8, 336, 14, 14]" = torch.ops.aten.le.Scalar(alias_86, 0);  alias_86 = None
    where_17: "f32[8, 336, 14, 14]" = torch.ops.aten.where.self(le_14, full_default, add_439);  le_14 = add_439 = None
    sum_58: "f32[336]" = torch.ops.aten.sum.dim_IntList(where_17, [0, 2, 3])
    sub_172: "f32[8, 336, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_64, unsqueeze_598);  convolution_64 = unsqueeze_598 = None
    mul_783: "f32[8, 336, 14, 14]" = torch.ops.aten.mul.Tensor(where_17, sub_172)
    sum_59: "f32[336]" = torch.ops.aten.sum.dim_IntList(mul_783, [0, 2, 3]);  mul_783 = None
    mul_784: "f32[336]" = torch.ops.aten.mul.Tensor(sum_58, 0.0006377551020408163)
    unsqueeze_599: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(mul_784, 0);  mul_784 = None
    unsqueeze_600: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_599, 2);  unsqueeze_599 = None
    unsqueeze_601: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_600, 3);  unsqueeze_600 = None
    mul_785: "f32[336]" = torch.ops.aten.mul.Tensor(sum_59, 0.0006377551020408163)
    mul_786: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_169, squeeze_169)
    mul_787: "f32[336]" = torch.ops.aten.mul.Tensor(mul_785, mul_786);  mul_785 = mul_786 = None
    unsqueeze_602: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(mul_787, 0);  mul_787 = None
    unsqueeze_603: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_602, 2);  unsqueeze_602 = None
    unsqueeze_604: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_603, 3);  unsqueeze_603 = None
    mul_788: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_169, primals_190);  primals_190 = None
    unsqueeze_605: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(mul_788, 0);  mul_788 = None
    unsqueeze_606: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_605, 2);  unsqueeze_605 = None
    unsqueeze_607: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_606, 3);  unsqueeze_606 = None
    mul_789: "f32[8, 336, 14, 14]" = torch.ops.aten.mul.Tensor(sub_172, unsqueeze_604);  sub_172 = unsqueeze_604 = None
    sub_174: "f32[8, 336, 14, 14]" = torch.ops.aten.sub.Tensor(where_17, mul_789);  where_17 = mul_789 = None
    sub_175: "f32[8, 336, 14, 14]" = torch.ops.aten.sub.Tensor(sub_174, unsqueeze_601);  sub_174 = unsqueeze_601 = None
    mul_790: "f32[8, 336, 14, 14]" = torch.ops.aten.mul.Tensor(sub_175, unsqueeze_607);  sub_175 = unsqueeze_607 = None
    mul_791: "f32[336]" = torch.ops.aten.mul.Tensor(sum_59, squeeze_169);  sum_59 = squeeze_169 = None
    convolution_backward_30 = torch.ops.aten.convolution_backward.default(mul_790, slice_121, primals_189, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_790 = slice_121 = primals_189 = None
    getitem_250: "f32[8, 112, 14, 14]" = convolution_backward_30[0]
    getitem_251: "f32[336, 112, 1, 1]" = convolution_backward_30[1];  convolution_backward_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    add_440: "f32[8, 112, 14, 14]" = torch.ops.aten.add.Tensor(getitem_229, getitem_250);  getitem_229 = getitem_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:182, code: x += self.shortcut(shortcut)
    full_20: "f32[175616]" = torch.ops.aten.full.default([175616], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    as_strided_35: "f32[8, 112, 14, 14]" = torch.ops.aten.as_strided.default(full_20, [8, 112, 14, 14], [21952, 196, 14, 1], 0)
    copy_15: "f32[8, 112, 14, 14]" = torch.ops.aten.copy.default(as_strided_35, add_440);  add_440 = None
    as_strided_scatter_10: "f32[175616]" = torch.ops.aten.as_strided_scatter.default(full_20, copy_15, [8, 112, 14, 14], [21952, 196, 14, 1], 0);  copy_15 = None
    as_strided_38: "f32[8, 112, 14, 14]" = torch.ops.aten.as_strided.default(as_strided_scatter_10, [8, 112, 14, 14], [21952, 196, 14, 1], 0);  as_strided_scatter_10 = None
    new_empty_strided_5: "f32[8, 112, 14, 14]" = torch.ops.aten.new_empty_strided.default(as_strided_38, [8, 112, 14, 14], [21952, 196, 14, 1])
    copy_16: "f32[8, 112, 14, 14]" = torch.ops.aten.copy.default(new_empty_strided_5, as_strided_38);  new_empty_strided_5 = as_strided_38 = None
    as_strided_40: "f32[8, 112, 14, 14]" = torch.ops.aten.as_strided.default(copy_16, [8, 112, 14, 14], [21952, 196, 14, 1], 0)
    clone_5: "f32[8, 112, 14, 14]" = torch.ops.aten.clone.default(as_strided_40, memory_format = torch.contiguous_format)
    copy_17: "f32[8, 112, 14, 14]" = torch.ops.aten.copy.default(as_strided_40, clone_5);  as_strided_40 = None
    as_strided_scatter_11: "f32[8, 112, 14, 14]" = torch.ops.aten.as_strided_scatter.default(copy_16, copy_17, [8, 112, 14, 14], [21952, 196, 14, 1], 0);  copy_16 = copy_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    slice_209: "f32[8, 56, 14, 14]" = torch.ops.aten.slice.Tensor(as_strided_scatter_11, 1, 56, 112)
    sum_60: "f32[56]" = torch.ops.aten.sum.dim_IntList(slice_209, [0, 2, 3])
    sub_176: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_63, unsqueeze_610);  convolution_63 = unsqueeze_610 = None
    mul_792: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(slice_209, sub_176)
    sum_61: "f32[56]" = torch.ops.aten.sum.dim_IntList(mul_792, [0, 2, 3]);  mul_792 = None
    mul_793: "f32[56]" = torch.ops.aten.mul.Tensor(sum_60, 0.0006377551020408163)
    unsqueeze_611: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_793, 0);  mul_793 = None
    unsqueeze_612: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_611, 2);  unsqueeze_611 = None
    unsqueeze_613: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_612, 3);  unsqueeze_612 = None
    mul_794: "f32[56]" = torch.ops.aten.mul.Tensor(sum_61, 0.0006377551020408163)
    mul_795: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_166, squeeze_166)
    mul_796: "f32[56]" = torch.ops.aten.mul.Tensor(mul_794, mul_795);  mul_794 = mul_795 = None
    unsqueeze_614: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_796, 0);  mul_796 = None
    unsqueeze_615: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_614, 2);  unsqueeze_614 = None
    unsqueeze_616: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_615, 3);  unsqueeze_615 = None
    mul_797: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_166, primals_187);  primals_187 = None
    unsqueeze_617: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_797, 0);  mul_797 = None
    unsqueeze_618: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_617, 2);  unsqueeze_617 = None
    unsqueeze_619: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_618, 3);  unsqueeze_618 = None
    mul_798: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_176, unsqueeze_616);  sub_176 = unsqueeze_616 = None
    sub_178: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(slice_209, mul_798);  slice_209 = mul_798 = None
    sub_179: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(sub_178, unsqueeze_613);  sub_178 = unsqueeze_613 = None
    mul_799: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_179, unsqueeze_619);  sub_179 = unsqueeze_619 = None
    mul_800: "f32[56]" = torch.ops.aten.mul.Tensor(sum_61, squeeze_166);  sum_61 = squeeze_166 = None
    convolution_backward_31 = torch.ops.aten.convolution_backward.default(mul_799, add_288, primals_186, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 56, [True, True, False]);  mul_799 = add_288 = primals_186 = None
    getitem_253: "f32[8, 56, 14, 14]" = convolution_backward_31[0]
    getitem_254: "f32[56, 1, 3, 3]" = convolution_backward_31[1];  convolution_backward_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    slice_210: "f32[8, 56, 14, 14]" = torch.ops.aten.slice.Tensor(as_strided_scatter_11, 1, 0, 56);  as_strided_scatter_11 = None
    add_441: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(slice_210, getitem_253);  slice_210 = getitem_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    sum_62: "f32[56]" = torch.ops.aten.sum.dim_IntList(add_441, [0, 2, 3])
    sub_180: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_62, unsqueeze_622);  convolution_62 = unsqueeze_622 = None
    mul_801: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(add_441, sub_180)
    sum_63: "f32[56]" = torch.ops.aten.sum.dim_IntList(mul_801, [0, 2, 3]);  mul_801 = None
    mul_802: "f32[56]" = torch.ops.aten.mul.Tensor(sum_62, 0.0006377551020408163)
    unsqueeze_623: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_802, 0);  mul_802 = None
    unsqueeze_624: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_623, 2);  unsqueeze_623 = None
    unsqueeze_625: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_624, 3);  unsqueeze_624 = None
    mul_803: "f32[56]" = torch.ops.aten.mul.Tensor(sum_63, 0.0006377551020408163)
    mul_804: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_163, squeeze_163)
    mul_805: "f32[56]" = torch.ops.aten.mul.Tensor(mul_803, mul_804);  mul_803 = mul_804 = None
    unsqueeze_626: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_805, 0);  mul_805 = None
    unsqueeze_627: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_626, 2);  unsqueeze_626 = None
    unsqueeze_628: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_627, 3);  unsqueeze_627 = None
    mul_806: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_163, primals_184);  primals_184 = None
    unsqueeze_629: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_806, 0);  mul_806 = None
    unsqueeze_630: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_629, 2);  unsqueeze_629 = None
    unsqueeze_631: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_630, 3);  unsqueeze_630 = None
    mul_807: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_180, unsqueeze_628);  sub_180 = unsqueeze_628 = None
    sub_182: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(add_441, mul_807);  add_441 = mul_807 = None
    sub_183: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(sub_182, unsqueeze_625);  sub_182 = unsqueeze_625 = None
    mul_808: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_183, unsqueeze_631);  sub_183 = unsqueeze_631 = None
    mul_809: "f32[56]" = torch.ops.aten.mul.Tensor(sum_63, squeeze_163);  sum_63 = squeeze_163 = None
    convolution_backward_32 = torch.ops.aten.convolution_backward.default(mul_808, mul_381, primals_183, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_808 = mul_381 = primals_183 = None
    getitem_256: "f32[8, 672, 14, 14]" = convolution_backward_32[0]
    getitem_257: "f32[56, 672, 1, 1]" = convolution_backward_32[1];  convolution_backward_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_810: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_256, slice_113);  slice_113 = None
    mul_811: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_256, div_3);  getitem_256 = div_3 = None
    sum_64: "f32[8, 672, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_810, [2, 3], True);  mul_810 = None
    mul_812: "f32[8, 672, 1, 1]" = torch.ops.aten.mul.Tensor(sum_64, 0.16666666666666666);  sum_64 = None
    where_18: "f32[8, 672, 1, 1]" = torch.ops.aten.where.self(bitwise_and_3, mul_812, full_default);  bitwise_and_3 = mul_812 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    sum_65: "f32[672]" = torch.ops.aten.sum.dim_IntList(where_18, [0, 2, 3])
    convolution_backward_33 = torch.ops.aten.convolution_backward.default(where_18, relu_26, primals_181, [672], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_18 = primals_181 = None
    getitem_259: "f32[8, 168, 1, 1]" = convolution_backward_33[0]
    getitem_260: "f32[672, 168, 1, 1]" = convolution_backward_33[1];  convolution_backward_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    alias_88: "f32[8, 168, 1, 1]" = torch.ops.aten.alias.default(relu_26);  relu_26 = None
    alias_89: "f32[8, 168, 1, 1]" = torch.ops.aten.alias.default(alias_88);  alias_88 = None
    le_15: "b8[8, 168, 1, 1]" = torch.ops.aten.le.Scalar(alias_89, 0);  alias_89 = None
    where_19: "f32[8, 168, 1, 1]" = torch.ops.aten.where.self(le_15, full_default, getitem_259);  le_15 = getitem_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    sum_66: "f32[168]" = torch.ops.aten.sum.dim_IntList(where_19, [0, 2, 3])
    convolution_backward_34 = torch.ops.aten.convolution_backward.default(where_19, mean_3, primals_179, [168], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_19 = mean_3 = primals_179 = None
    getitem_262: "f32[8, 672, 1, 1]" = convolution_backward_34[0]
    getitem_263: "f32[168, 672, 1, 1]" = convolution_backward_34[1];  convolution_backward_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_4: "f32[8, 672, 14, 14]" = torch.ops.aten.expand.default(getitem_262, [8, 672, 14, 14]);  getitem_262 = None
    div_11: "f32[8, 672, 14, 14]" = torch.ops.aten.div.Scalar(expand_4, 196);  expand_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_442: "f32[8, 672, 14, 14]" = torch.ops.aten.add.Tensor(mul_811, div_11);  mul_811 = div_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:63, code: return out[:, :self.out_chs, :, :]
    slice_scatter_63: "f32[8, 672, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_28, add_442, 3, 0, 9223372036854775807);  add_442 = None
    slice_scatter_64: "f32[8, 672, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_28, slice_scatter_63, 2, 0, 9223372036854775807);  slice_scatter_63 = None
    slice_scatter_65: "f32[8, 672, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_28, slice_scatter_64, 0, 0, 9223372036854775807);  full_default_28 = slice_scatter_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    slice_211: "f32[8, 336, 14, 14]" = torch.ops.aten.slice.Tensor(slice_scatter_65, 1, 0, 336)
    slice_212: "f32[8, 336, 14, 14]" = torch.ops.aten.slice.Tensor(slice_scatter_65, 1, 336, 672);  slice_scatter_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    where_20: "f32[8, 336, 14, 14]" = torch.ops.aten.where.self(le_16, full_default, slice_212);  le_16 = slice_212 = None
    sum_67: "f32[336]" = torch.ops.aten.sum.dim_IntList(where_20, [0, 2, 3])
    sub_184: "f32[8, 336, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_59, unsqueeze_634);  convolution_59 = unsqueeze_634 = None
    mul_813: "f32[8, 336, 14, 14]" = torch.ops.aten.mul.Tensor(where_20, sub_184)
    sum_68: "f32[336]" = torch.ops.aten.sum.dim_IntList(mul_813, [0, 2, 3]);  mul_813 = None
    mul_814: "f32[336]" = torch.ops.aten.mul.Tensor(sum_67, 0.0006377551020408163)
    unsqueeze_635: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(mul_814, 0);  mul_814 = None
    unsqueeze_636: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_635, 2);  unsqueeze_635 = None
    unsqueeze_637: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_636, 3);  unsqueeze_636 = None
    mul_815: "f32[336]" = torch.ops.aten.mul.Tensor(sum_68, 0.0006377551020408163)
    mul_816: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_160, squeeze_160)
    mul_817: "f32[336]" = torch.ops.aten.mul.Tensor(mul_815, mul_816);  mul_815 = mul_816 = None
    unsqueeze_638: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(mul_817, 0);  mul_817 = None
    unsqueeze_639: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_638, 2);  unsqueeze_638 = None
    unsqueeze_640: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_639, 3);  unsqueeze_639 = None
    mul_818: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_160, primals_177);  primals_177 = None
    unsqueeze_641: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(mul_818, 0);  mul_818 = None
    unsqueeze_642: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_641, 2);  unsqueeze_641 = None
    unsqueeze_643: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_642, 3);  unsqueeze_642 = None
    mul_819: "f32[8, 336, 14, 14]" = torch.ops.aten.mul.Tensor(sub_184, unsqueeze_640);  sub_184 = unsqueeze_640 = None
    sub_186: "f32[8, 336, 14, 14]" = torch.ops.aten.sub.Tensor(where_20, mul_819);  where_20 = mul_819 = None
    sub_187: "f32[8, 336, 14, 14]" = torch.ops.aten.sub.Tensor(sub_186, unsqueeze_637);  sub_186 = unsqueeze_637 = None
    mul_820: "f32[8, 336, 14, 14]" = torch.ops.aten.mul.Tensor(sub_187, unsqueeze_643);  sub_187 = unsqueeze_643 = None
    mul_821: "f32[336]" = torch.ops.aten.mul.Tensor(sum_68, squeeze_160);  sum_68 = squeeze_160 = None
    convolution_backward_35 = torch.ops.aten.convolution_backward.default(mul_820, relu_24, primals_176, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 336, [True, True, False]);  mul_820 = primals_176 = None
    getitem_265: "f32[8, 336, 14, 14]" = convolution_backward_35[0]
    getitem_266: "f32[336, 1, 3, 3]" = convolution_backward_35[1];  convolution_backward_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    add_443: "f32[8, 336, 14, 14]" = torch.ops.aten.add.Tensor(slice_211, getitem_265);  slice_211 = getitem_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    alias_94: "f32[8, 336, 14, 14]" = torch.ops.aten.alias.default(relu_24);  relu_24 = None
    alias_95: "f32[8, 336, 14, 14]" = torch.ops.aten.alias.default(alias_94);  alias_94 = None
    le_17: "b8[8, 336, 14, 14]" = torch.ops.aten.le.Scalar(alias_95, 0);  alias_95 = None
    where_21: "f32[8, 336, 14, 14]" = torch.ops.aten.where.self(le_17, full_default, add_443);  le_17 = add_443 = None
    sum_69: "f32[336]" = torch.ops.aten.sum.dim_IntList(where_21, [0, 2, 3])
    sub_188: "f32[8, 336, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_58, unsqueeze_646);  convolution_58 = unsqueeze_646 = None
    mul_822: "f32[8, 336, 14, 14]" = torch.ops.aten.mul.Tensor(where_21, sub_188)
    sum_70: "f32[336]" = torch.ops.aten.sum.dim_IntList(mul_822, [0, 2, 3]);  mul_822 = None
    mul_823: "f32[336]" = torch.ops.aten.mul.Tensor(sum_69, 0.0006377551020408163)
    unsqueeze_647: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(mul_823, 0);  mul_823 = None
    unsqueeze_648: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_647, 2);  unsqueeze_647 = None
    unsqueeze_649: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_648, 3);  unsqueeze_648 = None
    mul_824: "f32[336]" = torch.ops.aten.mul.Tensor(sum_70, 0.0006377551020408163)
    mul_825: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_157, squeeze_157)
    mul_826: "f32[336]" = torch.ops.aten.mul.Tensor(mul_824, mul_825);  mul_824 = mul_825 = None
    unsqueeze_650: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(mul_826, 0);  mul_826 = None
    unsqueeze_651: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_650, 2);  unsqueeze_650 = None
    unsqueeze_652: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_651, 3);  unsqueeze_651 = None
    mul_827: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_157, primals_174);  primals_174 = None
    unsqueeze_653: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(mul_827, 0);  mul_827 = None
    unsqueeze_654: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_653, 2);  unsqueeze_653 = None
    unsqueeze_655: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_654, 3);  unsqueeze_654 = None
    mul_828: "f32[8, 336, 14, 14]" = torch.ops.aten.mul.Tensor(sub_188, unsqueeze_652);  sub_188 = unsqueeze_652 = None
    sub_190: "f32[8, 336, 14, 14]" = torch.ops.aten.sub.Tensor(where_21, mul_828);  where_21 = mul_828 = None
    sub_191: "f32[8, 336, 14, 14]" = torch.ops.aten.sub.Tensor(sub_190, unsqueeze_649);  sub_190 = unsqueeze_649 = None
    mul_829: "f32[8, 336, 14, 14]" = torch.ops.aten.mul.Tensor(sub_191, unsqueeze_655);  sub_191 = unsqueeze_655 = None
    mul_830: "f32[336]" = torch.ops.aten.mul.Tensor(sum_70, squeeze_157);  sum_70 = squeeze_157 = None
    convolution_backward_36 = torch.ops.aten.convolution_backward.default(mul_829, slice_110, primals_173, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_829 = slice_110 = primals_173 = None
    getitem_268: "f32[8, 112, 14, 14]" = convolution_backward_36[0]
    getitem_269: "f32[336, 112, 1, 1]" = convolution_backward_36[1];  convolution_backward_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    add_444: "f32[8, 112, 14, 14]" = torch.ops.aten.add.Tensor(clone_5, getitem_268);  clone_5 = getitem_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:182, code: x += self.shortcut(shortcut)
    copy_18: "f32[8, 112, 14, 14]" = torch.ops.aten.copy.default(as_strided_35, add_444);  as_strided_35 = add_444 = None
    as_strided_scatter_12: "f32[175616]" = torch.ops.aten.as_strided_scatter.default(full_20, copy_18, [8, 112, 14, 14], [21952, 196, 14, 1], 0);  full_20 = copy_18 = None
    as_strided_45: "f32[8, 112, 14, 14]" = torch.ops.aten.as_strided.default(as_strided_scatter_12, [8, 112, 14, 14], [21952, 196, 14, 1], 0);  as_strided_scatter_12 = None
    new_empty_strided_6: "f32[8, 112, 14, 14]" = torch.ops.aten.new_empty_strided.default(as_strided_45, [8, 112, 14, 14], [21952, 196, 14, 1])
    copy_19: "f32[8, 112, 14, 14]" = torch.ops.aten.copy.default(new_empty_strided_6, as_strided_45);  new_empty_strided_6 = as_strided_45 = None
    as_strided_47: "f32[8, 112, 14, 14]" = torch.ops.aten.as_strided.default(copy_19, [8, 112, 14, 14], [21952, 196, 14, 1], 0)
    clone_6: "f32[8, 112, 14, 14]" = torch.ops.aten.clone.default(as_strided_47, memory_format = torch.contiguous_format)
    copy_20: "f32[8, 112, 14, 14]" = torch.ops.aten.copy.default(as_strided_47, clone_6);  as_strided_47 = None
    as_strided_scatter_13: "f32[8, 112, 14, 14]" = torch.ops.aten.as_strided_scatter.default(copy_19, copy_20, [8, 112, 14, 14], [21952, 196, 14, 1], 0);  copy_19 = copy_20 = None
    sum_71: "f32[112]" = torch.ops.aten.sum.dim_IntList(clone_6, [0, 2, 3])
    sub_192: "f32[8, 112, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_57, unsqueeze_658);  convolution_57 = unsqueeze_658 = None
    mul_831: "f32[8, 112, 14, 14]" = torch.ops.aten.mul.Tensor(clone_6, sub_192)
    sum_72: "f32[112]" = torch.ops.aten.sum.dim_IntList(mul_831, [0, 2, 3]);  mul_831 = None
    mul_832: "f32[112]" = torch.ops.aten.mul.Tensor(sum_71, 0.0006377551020408163)
    unsqueeze_659: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_832, 0);  mul_832 = None
    unsqueeze_660: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_659, 2);  unsqueeze_659 = None
    unsqueeze_661: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_660, 3);  unsqueeze_660 = None
    mul_833: "f32[112]" = torch.ops.aten.mul.Tensor(sum_72, 0.0006377551020408163)
    mul_834: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_154, squeeze_154)
    mul_835: "f32[112]" = torch.ops.aten.mul.Tensor(mul_833, mul_834);  mul_833 = mul_834 = None
    unsqueeze_662: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_835, 0);  mul_835 = None
    unsqueeze_663: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_662, 2);  unsqueeze_662 = None
    unsqueeze_664: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_663, 3);  unsqueeze_663 = None
    mul_836: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_154, primals_171);  primals_171 = None
    unsqueeze_665: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_836, 0);  mul_836 = None
    unsqueeze_666: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_665, 2);  unsqueeze_665 = None
    unsqueeze_667: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_666, 3);  unsqueeze_666 = None
    mul_837: "f32[8, 112, 14, 14]" = torch.ops.aten.mul.Tensor(sub_192, unsqueeze_664);  sub_192 = unsqueeze_664 = None
    sub_194: "f32[8, 112, 14, 14]" = torch.ops.aten.sub.Tensor(clone_6, mul_837);  clone_6 = mul_837 = None
    sub_195: "f32[8, 112, 14, 14]" = torch.ops.aten.sub.Tensor(sub_194, unsqueeze_661);  sub_194 = unsqueeze_661 = None
    mul_838: "f32[8, 112, 14, 14]" = torch.ops.aten.mul.Tensor(sub_195, unsqueeze_667);  sub_195 = unsqueeze_667 = None
    mul_839: "f32[112]" = torch.ops.aten.mul.Tensor(sum_72, squeeze_154);  sum_72 = squeeze_154 = None
    convolution_backward_37 = torch.ops.aten.convolution_backward.default(mul_838, add_266, primals_170, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_838 = add_266 = primals_170 = None
    getitem_271: "f32[8, 80, 14, 14]" = convolution_backward_37[0]
    getitem_272: "f32[112, 80, 1, 1]" = convolution_backward_37[1];  convolution_backward_37 = None
    sum_73: "f32[80]" = torch.ops.aten.sum.dim_IntList(getitem_271, [0, 2, 3])
    sub_196: "f32[8, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_56, unsqueeze_670);  convolution_56 = unsqueeze_670 = None
    mul_840: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_271, sub_196)
    sum_74: "f32[80]" = torch.ops.aten.sum.dim_IntList(mul_840, [0, 2, 3]);  mul_840 = None
    mul_841: "f32[80]" = torch.ops.aten.mul.Tensor(sum_73, 0.0006377551020408163)
    unsqueeze_671: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_841, 0);  mul_841 = None
    unsqueeze_672: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_671, 2);  unsqueeze_671 = None
    unsqueeze_673: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_672, 3);  unsqueeze_672 = None
    mul_842: "f32[80]" = torch.ops.aten.mul.Tensor(sum_74, 0.0006377551020408163)
    mul_843: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_151, squeeze_151)
    mul_844: "f32[80]" = torch.ops.aten.mul.Tensor(mul_842, mul_843);  mul_842 = mul_843 = None
    unsqueeze_674: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_844, 0);  mul_844 = None
    unsqueeze_675: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_674, 2);  unsqueeze_674 = None
    unsqueeze_676: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_675, 3);  unsqueeze_675 = None
    mul_845: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_151, primals_168);  primals_168 = None
    unsqueeze_677: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_845, 0);  mul_845 = None
    unsqueeze_678: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_677, 2);  unsqueeze_677 = None
    unsqueeze_679: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_678, 3);  unsqueeze_678 = None
    mul_846: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_196, unsqueeze_676);  sub_196 = unsqueeze_676 = None
    sub_198: "f32[8, 80, 14, 14]" = torch.ops.aten.sub.Tensor(getitem_271, mul_846);  getitem_271 = mul_846 = None
    sub_199: "f32[8, 80, 14, 14]" = torch.ops.aten.sub.Tensor(sub_198, unsqueeze_673);  sub_198 = unsqueeze_673 = None
    mul_847: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_199, unsqueeze_679);  sub_199 = unsqueeze_679 = None
    mul_848: "f32[80]" = torch.ops.aten.mul.Tensor(sum_74, squeeze_151);  sum_74 = squeeze_151 = None
    convolution_backward_38 = torch.ops.aten.convolution_backward.default(mul_847, slice_99, primals_167, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 80, [True, True, False]);  mul_847 = primals_167 = None
    getitem_274: "f32[8, 80, 14, 14]" = convolution_backward_38[0]
    getitem_275: "f32[80, 1, 3, 3]" = convolution_backward_38[1];  convolution_backward_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    slice_215: "f32[8, 56, 14, 14]" = torch.ops.aten.slice.Tensor(as_strided_scatter_13, 1, 56, 112)
    sum_75: "f32[56]" = torch.ops.aten.sum.dim_IntList(slice_215, [0, 2, 3])
    sub_200: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_55, unsqueeze_682);  convolution_55 = unsqueeze_682 = None
    mul_849: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(slice_215, sub_200)
    sum_76: "f32[56]" = torch.ops.aten.sum.dim_IntList(mul_849, [0, 2, 3]);  mul_849 = None
    mul_850: "f32[56]" = torch.ops.aten.mul.Tensor(sum_75, 0.0006377551020408163)
    unsqueeze_683: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_850, 0);  mul_850 = None
    unsqueeze_684: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_683, 2);  unsqueeze_683 = None
    unsqueeze_685: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_684, 3);  unsqueeze_684 = None
    mul_851: "f32[56]" = torch.ops.aten.mul.Tensor(sum_76, 0.0006377551020408163)
    mul_852: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_148, squeeze_148)
    mul_853: "f32[56]" = torch.ops.aten.mul.Tensor(mul_851, mul_852);  mul_851 = mul_852 = None
    unsqueeze_686: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_853, 0);  mul_853 = None
    unsqueeze_687: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_686, 2);  unsqueeze_686 = None
    unsqueeze_688: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_687, 3);  unsqueeze_687 = None
    mul_854: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_148, primals_165);  primals_165 = None
    unsqueeze_689: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_854, 0);  mul_854 = None
    unsqueeze_690: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_689, 2);  unsqueeze_689 = None
    unsqueeze_691: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_690, 3);  unsqueeze_690 = None
    mul_855: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_200, unsqueeze_688);  sub_200 = unsqueeze_688 = None
    sub_202: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(slice_215, mul_855);  slice_215 = mul_855 = None
    sub_203: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(sub_202, unsqueeze_685);  sub_202 = unsqueeze_685 = None
    mul_856: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_203, unsqueeze_691);  sub_203 = unsqueeze_691 = None
    mul_857: "f32[56]" = torch.ops.aten.mul.Tensor(sum_76, squeeze_148);  sum_76 = squeeze_148 = None
    convolution_backward_39 = torch.ops.aten.convolution_backward.default(mul_856, add_256, primals_164, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 56, [True, True, False]);  mul_856 = add_256 = primals_164 = None
    getitem_277: "f32[8, 56, 14, 14]" = convolution_backward_39[0]
    getitem_278: "f32[56, 1, 3, 3]" = convolution_backward_39[1];  convolution_backward_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    slice_216: "f32[8, 56, 14, 14]" = torch.ops.aten.slice.Tensor(as_strided_scatter_13, 1, 0, 56);  as_strided_scatter_13 = None
    add_445: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(slice_216, getitem_277);  slice_216 = getitem_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    sum_77: "f32[56]" = torch.ops.aten.sum.dim_IntList(add_445, [0, 2, 3])
    sub_204: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_54, unsqueeze_694);  convolution_54 = unsqueeze_694 = None
    mul_858: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(add_445, sub_204)
    sum_78: "f32[56]" = torch.ops.aten.sum.dim_IntList(mul_858, [0, 2, 3]);  mul_858 = None
    mul_859: "f32[56]" = torch.ops.aten.mul.Tensor(sum_77, 0.0006377551020408163)
    unsqueeze_695: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_859, 0);  mul_859 = None
    unsqueeze_696: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_695, 2);  unsqueeze_695 = None
    unsqueeze_697: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_696, 3);  unsqueeze_696 = None
    mul_860: "f32[56]" = torch.ops.aten.mul.Tensor(sum_78, 0.0006377551020408163)
    mul_861: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_145, squeeze_145)
    mul_862: "f32[56]" = torch.ops.aten.mul.Tensor(mul_860, mul_861);  mul_860 = mul_861 = None
    unsqueeze_698: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_862, 0);  mul_862 = None
    unsqueeze_699: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_698, 2);  unsqueeze_698 = None
    unsqueeze_700: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_699, 3);  unsqueeze_699 = None
    mul_863: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_145, primals_162);  primals_162 = None
    unsqueeze_701: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_863, 0);  mul_863 = None
    unsqueeze_702: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_701, 2);  unsqueeze_701 = None
    unsqueeze_703: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_702, 3);  unsqueeze_702 = None
    mul_864: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_204, unsqueeze_700);  sub_204 = unsqueeze_700 = None
    sub_206: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(add_445, mul_864);  add_445 = mul_864 = None
    sub_207: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(sub_206, unsqueeze_697);  sub_206 = unsqueeze_697 = None
    mul_865: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_207, unsqueeze_703);  sub_207 = unsqueeze_703 = None
    mul_866: "f32[56]" = torch.ops.aten.mul.Tensor(sum_78, squeeze_145);  sum_78 = squeeze_145 = None
    convolution_backward_40 = torch.ops.aten.convolution_backward.default(mul_865, mul_338, primals_161, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_865 = mul_338 = primals_161 = None
    getitem_280: "f32[8, 480, 14, 14]" = convolution_backward_40[0]
    getitem_281: "f32[56, 480, 1, 1]" = convolution_backward_40[1];  convolution_backward_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_867: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_280, slice_102);  slice_102 = None
    mul_868: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_280, div_2);  getitem_280 = div_2 = None
    sum_79: "f32[8, 480, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_867, [2, 3], True);  mul_867 = None
    mul_869: "f32[8, 480, 1, 1]" = torch.ops.aten.mul.Tensor(sum_79, 0.16666666666666666);  sum_79 = None
    where_22: "f32[8, 480, 1, 1]" = torch.ops.aten.where.self(bitwise_and_4, mul_869, full_default);  bitwise_and_4 = mul_869 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    sum_80: "f32[480]" = torch.ops.aten.sum.dim_IntList(where_22, [0, 2, 3])
    convolution_backward_41 = torch.ops.aten.convolution_backward.default(where_22, relu_23, primals_159, [480], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_22 = primals_159 = None
    getitem_283: "f32[8, 120, 1, 1]" = convolution_backward_41[0]
    getitem_284: "f32[480, 120, 1, 1]" = convolution_backward_41[1];  convolution_backward_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    alias_97: "f32[8, 120, 1, 1]" = torch.ops.aten.alias.default(relu_23);  relu_23 = None
    alias_98: "f32[8, 120, 1, 1]" = torch.ops.aten.alias.default(alias_97);  alias_97 = None
    le_18: "b8[8, 120, 1, 1]" = torch.ops.aten.le.Scalar(alias_98, 0);  alias_98 = None
    where_23: "f32[8, 120, 1, 1]" = torch.ops.aten.where.self(le_18, full_default, getitem_283);  le_18 = getitem_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    sum_81: "f32[120]" = torch.ops.aten.sum.dim_IntList(where_23, [0, 2, 3])
    convolution_backward_42 = torch.ops.aten.convolution_backward.default(where_23, mean_2, primals_157, [120], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_23 = mean_2 = primals_157 = None
    getitem_286: "f32[8, 480, 1, 1]" = convolution_backward_42[0]
    getitem_287: "f32[120, 480, 1, 1]" = convolution_backward_42[1];  convolution_backward_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_5: "f32[8, 480, 14, 14]" = torch.ops.aten.expand.default(getitem_286, [8, 480, 14, 14]);  getitem_286 = None
    div_12: "f32[8, 480, 14, 14]" = torch.ops.aten.div.Scalar(expand_5, 196);  expand_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_446: "f32[8, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_868, div_12);  mul_868 = div_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:63, code: return out[:, :self.out_chs, :, :]
    full_default_42: "f32[8, 480, 14, 14]" = torch.ops.aten.full.default([8, 480, 14, 14], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_66: "f32[8, 480, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_42, add_446, 3, 0, 9223372036854775807);  add_446 = None
    slice_scatter_67: "f32[8, 480, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_42, slice_scatter_66, 2, 0, 9223372036854775807);  slice_scatter_66 = None
    slice_scatter_68: "f32[8, 480, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_42, slice_scatter_67, 0, 0, 9223372036854775807);  full_default_42 = slice_scatter_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    slice_217: "f32[8, 240, 14, 14]" = torch.ops.aten.slice.Tensor(slice_scatter_68, 1, 0, 240)
    slice_218: "f32[8, 240, 14, 14]" = torch.ops.aten.slice.Tensor(slice_scatter_68, 1, 240, 480);  slice_scatter_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    where_24: "f32[8, 240, 14, 14]" = torch.ops.aten.where.self(le_19, full_default, slice_218);  le_19 = slice_218 = None
    sum_82: "f32[240]" = torch.ops.aten.sum.dim_IntList(where_24, [0, 2, 3])
    sub_208: "f32[8, 240, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_706);  convolution_51 = unsqueeze_706 = None
    mul_870: "f32[8, 240, 14, 14]" = torch.ops.aten.mul.Tensor(where_24, sub_208)
    sum_83: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_870, [0, 2, 3]);  mul_870 = None
    mul_871: "f32[240]" = torch.ops.aten.mul.Tensor(sum_82, 0.0006377551020408163)
    unsqueeze_707: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_871, 0);  mul_871 = None
    unsqueeze_708: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_707, 2);  unsqueeze_707 = None
    unsqueeze_709: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_708, 3);  unsqueeze_708 = None
    mul_872: "f32[240]" = torch.ops.aten.mul.Tensor(sum_83, 0.0006377551020408163)
    mul_873: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_142, squeeze_142)
    mul_874: "f32[240]" = torch.ops.aten.mul.Tensor(mul_872, mul_873);  mul_872 = mul_873 = None
    unsqueeze_710: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_874, 0);  mul_874 = None
    unsqueeze_711: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_710, 2);  unsqueeze_710 = None
    unsqueeze_712: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_711, 3);  unsqueeze_711 = None
    mul_875: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_142, primals_155);  primals_155 = None
    unsqueeze_713: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_875, 0);  mul_875 = None
    unsqueeze_714: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_713, 2);  unsqueeze_713 = None
    unsqueeze_715: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_714, 3);  unsqueeze_714 = None
    mul_876: "f32[8, 240, 14, 14]" = torch.ops.aten.mul.Tensor(sub_208, unsqueeze_712);  sub_208 = unsqueeze_712 = None
    sub_210: "f32[8, 240, 14, 14]" = torch.ops.aten.sub.Tensor(where_24, mul_876);  where_24 = mul_876 = None
    sub_211: "f32[8, 240, 14, 14]" = torch.ops.aten.sub.Tensor(sub_210, unsqueeze_709);  sub_210 = unsqueeze_709 = None
    mul_877: "f32[8, 240, 14, 14]" = torch.ops.aten.mul.Tensor(sub_211, unsqueeze_715);  sub_211 = unsqueeze_715 = None
    mul_878: "f32[240]" = torch.ops.aten.mul.Tensor(sum_83, squeeze_142);  sum_83 = squeeze_142 = None
    convolution_backward_43 = torch.ops.aten.convolution_backward.default(mul_877, relu_21, primals_154, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 240, [True, True, False]);  mul_877 = primals_154 = None
    getitem_289: "f32[8, 240, 14, 14]" = convolution_backward_43[0]
    getitem_290: "f32[240, 1, 3, 3]" = convolution_backward_43[1];  convolution_backward_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    add_447: "f32[8, 240, 14, 14]" = torch.ops.aten.add.Tensor(slice_217, getitem_289);  slice_217 = getitem_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    alias_103: "f32[8, 240, 14, 14]" = torch.ops.aten.alias.default(relu_21);  relu_21 = None
    alias_104: "f32[8, 240, 14, 14]" = torch.ops.aten.alias.default(alias_103);  alias_103 = None
    le_20: "b8[8, 240, 14, 14]" = torch.ops.aten.le.Scalar(alias_104, 0);  alias_104 = None
    where_25: "f32[8, 240, 14, 14]" = torch.ops.aten.where.self(le_20, full_default, add_447);  le_20 = add_447 = None
    sum_84: "f32[240]" = torch.ops.aten.sum.dim_IntList(where_25, [0, 2, 3])
    sub_212: "f32[8, 240, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_718);  convolution_50 = unsqueeze_718 = None
    mul_879: "f32[8, 240, 14, 14]" = torch.ops.aten.mul.Tensor(where_25, sub_212)
    sum_85: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_879, [0, 2, 3]);  mul_879 = None
    mul_880: "f32[240]" = torch.ops.aten.mul.Tensor(sum_84, 0.0006377551020408163)
    unsqueeze_719: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_880, 0);  mul_880 = None
    unsqueeze_720: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_719, 2);  unsqueeze_719 = None
    unsqueeze_721: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_720, 3);  unsqueeze_720 = None
    mul_881: "f32[240]" = torch.ops.aten.mul.Tensor(sum_85, 0.0006377551020408163)
    mul_882: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_139, squeeze_139)
    mul_883: "f32[240]" = torch.ops.aten.mul.Tensor(mul_881, mul_882);  mul_881 = mul_882 = None
    unsqueeze_722: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_883, 0);  mul_883 = None
    unsqueeze_723: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_722, 2);  unsqueeze_722 = None
    unsqueeze_724: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_723, 3);  unsqueeze_723 = None
    mul_884: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_139, primals_152);  primals_152 = None
    unsqueeze_725: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_884, 0);  mul_884 = None
    unsqueeze_726: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_725, 2);  unsqueeze_725 = None
    unsqueeze_727: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_726, 3);  unsqueeze_726 = None
    mul_885: "f32[8, 240, 14, 14]" = torch.ops.aten.mul.Tensor(sub_212, unsqueeze_724);  sub_212 = unsqueeze_724 = None
    sub_214: "f32[8, 240, 14, 14]" = torch.ops.aten.sub.Tensor(where_25, mul_885);  where_25 = mul_885 = None
    sub_215: "f32[8, 240, 14, 14]" = torch.ops.aten.sub.Tensor(sub_214, unsqueeze_721);  sub_214 = unsqueeze_721 = None
    mul_886: "f32[8, 240, 14, 14]" = torch.ops.aten.mul.Tensor(sub_215, unsqueeze_727);  sub_215 = unsqueeze_727 = None
    mul_887: "f32[240]" = torch.ops.aten.mul.Tensor(sum_85, squeeze_139);  sum_85 = squeeze_139 = None
    convolution_backward_44 = torch.ops.aten.convolution_backward.default(mul_886, slice_99, primals_151, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_886 = slice_99 = primals_151 = None
    getitem_292: "f32[8, 80, 14, 14]" = convolution_backward_44[0]
    getitem_293: "f32[240, 80, 1, 1]" = convolution_backward_44[1];  convolution_backward_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    add_448: "f32[8, 80, 14, 14]" = torch.ops.aten.add.Tensor(getitem_274, getitem_292);  getitem_274 = getitem_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:182, code: x += self.shortcut(shortcut)
    full_28: "f32[125440]" = torch.ops.aten.full.default([125440], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    as_strided_49: "f32[8, 80, 14, 14]" = torch.ops.aten.as_strided.default(full_28, [8, 80, 14, 14], [15680, 196, 14, 1], 0)
    copy_21: "f32[8, 80, 14, 14]" = torch.ops.aten.copy.default(as_strided_49, add_448);  add_448 = None
    as_strided_scatter_14: "f32[125440]" = torch.ops.aten.as_strided_scatter.default(full_28, copy_21, [8, 80, 14, 14], [15680, 196, 14, 1], 0);  copy_21 = None
    as_strided_52: "f32[8, 80, 14, 14]" = torch.ops.aten.as_strided.default(as_strided_scatter_14, [8, 80, 14, 14], [15680, 196, 14, 1], 0);  as_strided_scatter_14 = None
    new_empty_strided_7: "f32[8, 80, 14, 14]" = torch.ops.aten.new_empty_strided.default(as_strided_52, [8, 80, 14, 14], [15680, 196, 14, 1])
    copy_22: "f32[8, 80, 14, 14]" = torch.ops.aten.copy.default(new_empty_strided_7, as_strided_52);  new_empty_strided_7 = as_strided_52 = None
    as_strided_54: "f32[8, 80, 14, 14]" = torch.ops.aten.as_strided.default(copy_22, [8, 80, 14, 14], [15680, 196, 14, 1], 0)
    clone_7: "f32[8, 80, 14, 14]" = torch.ops.aten.clone.default(as_strided_54, memory_format = torch.contiguous_format)
    copy_23: "f32[8, 80, 14, 14]" = torch.ops.aten.copy.default(as_strided_54, clone_7);  as_strided_54 = None
    as_strided_scatter_15: "f32[8, 80, 14, 14]" = torch.ops.aten.as_strided_scatter.default(copy_22, copy_23, [8, 80, 14, 14], [15680, 196, 14, 1], 0);  copy_22 = copy_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    slice_221: "f32[8, 40, 14, 14]" = torch.ops.aten.slice.Tensor(as_strided_scatter_15, 1, 40, 80)
    sum_86: "f32[40]" = torch.ops.aten.sum.dim_IntList(slice_221, [0, 2, 3])
    sub_216: "f32[8, 40, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_49, unsqueeze_730);  convolution_49 = unsqueeze_730 = None
    mul_888: "f32[8, 40, 14, 14]" = torch.ops.aten.mul.Tensor(slice_221, sub_216)
    sum_87: "f32[40]" = torch.ops.aten.sum.dim_IntList(mul_888, [0, 2, 3]);  mul_888 = None
    mul_889: "f32[40]" = torch.ops.aten.mul.Tensor(sum_86, 0.0006377551020408163)
    unsqueeze_731: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_889, 0);  mul_889 = None
    unsqueeze_732: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_731, 2);  unsqueeze_731 = None
    unsqueeze_733: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_732, 3);  unsqueeze_732 = None
    mul_890: "f32[40]" = torch.ops.aten.mul.Tensor(sum_87, 0.0006377551020408163)
    mul_891: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_136, squeeze_136)
    mul_892: "f32[40]" = torch.ops.aten.mul.Tensor(mul_890, mul_891);  mul_890 = mul_891 = None
    unsqueeze_734: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_892, 0);  mul_892 = None
    unsqueeze_735: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_734, 2);  unsqueeze_734 = None
    unsqueeze_736: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_735, 3);  unsqueeze_735 = None
    mul_893: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_136, primals_149);  primals_149 = None
    unsqueeze_737: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_893, 0);  mul_893 = None
    unsqueeze_738: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_737, 2);  unsqueeze_737 = None
    unsqueeze_739: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_738, 3);  unsqueeze_738 = None
    mul_894: "f32[8, 40, 14, 14]" = torch.ops.aten.mul.Tensor(sub_216, unsqueeze_736);  sub_216 = unsqueeze_736 = None
    sub_218: "f32[8, 40, 14, 14]" = torch.ops.aten.sub.Tensor(slice_221, mul_894);  slice_221 = mul_894 = None
    sub_219: "f32[8, 40, 14, 14]" = torch.ops.aten.sub.Tensor(sub_218, unsqueeze_733);  sub_218 = unsqueeze_733 = None
    mul_895: "f32[8, 40, 14, 14]" = torch.ops.aten.mul.Tensor(sub_219, unsqueeze_739);  sub_219 = unsqueeze_739 = None
    mul_896: "f32[40]" = torch.ops.aten.mul.Tensor(sum_87, squeeze_136);  sum_87 = squeeze_136 = None
    convolution_backward_45 = torch.ops.aten.convolution_backward.default(mul_895, add_234, primals_148, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 40, [True, True, False]);  mul_895 = add_234 = primals_148 = None
    getitem_295: "f32[8, 40, 14, 14]" = convolution_backward_45[0]
    getitem_296: "f32[40, 1, 3, 3]" = convolution_backward_45[1];  convolution_backward_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    slice_222: "f32[8, 40, 14, 14]" = torch.ops.aten.slice.Tensor(as_strided_scatter_15, 1, 0, 40);  as_strided_scatter_15 = None
    add_449: "f32[8, 40, 14, 14]" = torch.ops.aten.add.Tensor(slice_222, getitem_295);  slice_222 = getitem_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    sum_88: "f32[40]" = torch.ops.aten.sum.dim_IntList(add_449, [0, 2, 3])
    sub_220: "f32[8, 40, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_48, unsqueeze_742);  convolution_48 = unsqueeze_742 = None
    mul_897: "f32[8, 40, 14, 14]" = torch.ops.aten.mul.Tensor(add_449, sub_220)
    sum_89: "f32[40]" = torch.ops.aten.sum.dim_IntList(mul_897, [0, 2, 3]);  mul_897 = None
    mul_898: "f32[40]" = torch.ops.aten.mul.Tensor(sum_88, 0.0006377551020408163)
    unsqueeze_743: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_898, 0);  mul_898 = None
    unsqueeze_744: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_743, 2);  unsqueeze_743 = None
    unsqueeze_745: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_744, 3);  unsqueeze_744 = None
    mul_899: "f32[40]" = torch.ops.aten.mul.Tensor(sum_89, 0.0006377551020408163)
    mul_900: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_133, squeeze_133)
    mul_901: "f32[40]" = torch.ops.aten.mul.Tensor(mul_899, mul_900);  mul_899 = mul_900 = None
    unsqueeze_746: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_901, 0);  mul_901 = None
    unsqueeze_747: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_746, 2);  unsqueeze_746 = None
    unsqueeze_748: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_747, 3);  unsqueeze_747 = None
    mul_902: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_133, primals_146);  primals_146 = None
    unsqueeze_749: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_902, 0);  mul_902 = None
    unsqueeze_750: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_749, 2);  unsqueeze_749 = None
    unsqueeze_751: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_750, 3);  unsqueeze_750 = None
    mul_903: "f32[8, 40, 14, 14]" = torch.ops.aten.mul.Tensor(sub_220, unsqueeze_748);  sub_220 = unsqueeze_748 = None
    sub_222: "f32[8, 40, 14, 14]" = torch.ops.aten.sub.Tensor(add_449, mul_903);  add_449 = mul_903 = None
    sub_223: "f32[8, 40, 14, 14]" = torch.ops.aten.sub.Tensor(sub_222, unsqueeze_745);  sub_222 = unsqueeze_745 = None
    mul_904: "f32[8, 40, 14, 14]" = torch.ops.aten.mul.Tensor(sub_223, unsqueeze_751);  sub_223 = unsqueeze_751 = None
    mul_905: "f32[40]" = torch.ops.aten.mul.Tensor(sum_89, squeeze_133);  sum_89 = squeeze_133 = None
    convolution_backward_46 = torch.ops.aten.convolution_backward.default(mul_904, slice_91, primals_145, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_904 = slice_91 = primals_145 = None
    getitem_298: "f32[8, 184, 14, 14]" = convolution_backward_46[0]
    getitem_299: "f32[40, 184, 1, 1]" = convolution_backward_46[1];  convolution_backward_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:63, code: return out[:, :self.out_chs, :, :]
    full_default_47: "f32[8, 184, 14, 14]" = torch.ops.aten.full.default([8, 184, 14, 14], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_69: "f32[8, 184, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_47, getitem_298, 3, 0, 9223372036854775807);  getitem_298 = None
    slice_scatter_70: "f32[8, 184, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_47, slice_scatter_69, 2, 0, 9223372036854775807);  slice_scatter_69 = None
    slice_scatter_71: "f32[8, 184, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_47, slice_scatter_70, 0, 0, 9223372036854775807);  slice_scatter_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    slice_223: "f32[8, 92, 14, 14]" = torch.ops.aten.slice.Tensor(slice_scatter_71, 1, 0, 92)
    slice_224: "f32[8, 92, 14, 14]" = torch.ops.aten.slice.Tensor(slice_scatter_71, 1, 92, 184);  slice_scatter_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    where_26: "f32[8, 92, 14, 14]" = torch.ops.aten.where.self(le_21, full_default, slice_224);  le_21 = slice_224 = None
    sum_90: "f32[92]" = torch.ops.aten.sum.dim_IntList(where_26, [0, 2, 3])
    sub_224: "f32[8, 92, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_47, unsqueeze_754);  convolution_47 = unsqueeze_754 = None
    mul_906: "f32[8, 92, 14, 14]" = torch.ops.aten.mul.Tensor(where_26, sub_224)
    sum_91: "f32[92]" = torch.ops.aten.sum.dim_IntList(mul_906, [0, 2, 3]);  mul_906 = None
    mul_907: "f32[92]" = torch.ops.aten.mul.Tensor(sum_90, 0.0006377551020408163)
    unsqueeze_755: "f32[1, 92]" = torch.ops.aten.unsqueeze.default(mul_907, 0);  mul_907 = None
    unsqueeze_756: "f32[1, 92, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_755, 2);  unsqueeze_755 = None
    unsqueeze_757: "f32[1, 92, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_756, 3);  unsqueeze_756 = None
    mul_908: "f32[92]" = torch.ops.aten.mul.Tensor(sum_91, 0.0006377551020408163)
    mul_909: "f32[92]" = torch.ops.aten.mul.Tensor(squeeze_130, squeeze_130)
    mul_910: "f32[92]" = torch.ops.aten.mul.Tensor(mul_908, mul_909);  mul_908 = mul_909 = None
    unsqueeze_758: "f32[1, 92]" = torch.ops.aten.unsqueeze.default(mul_910, 0);  mul_910 = None
    unsqueeze_759: "f32[1, 92, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_758, 2);  unsqueeze_758 = None
    unsqueeze_760: "f32[1, 92, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_759, 3);  unsqueeze_759 = None
    mul_911: "f32[92]" = torch.ops.aten.mul.Tensor(squeeze_130, primals_143);  primals_143 = None
    unsqueeze_761: "f32[1, 92]" = torch.ops.aten.unsqueeze.default(mul_911, 0);  mul_911 = None
    unsqueeze_762: "f32[1, 92, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_761, 2);  unsqueeze_761 = None
    unsqueeze_763: "f32[1, 92, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_762, 3);  unsqueeze_762 = None
    mul_912: "f32[8, 92, 14, 14]" = torch.ops.aten.mul.Tensor(sub_224, unsqueeze_760);  sub_224 = unsqueeze_760 = None
    sub_226: "f32[8, 92, 14, 14]" = torch.ops.aten.sub.Tensor(where_26, mul_912);  where_26 = mul_912 = None
    sub_227: "f32[8, 92, 14, 14]" = torch.ops.aten.sub.Tensor(sub_226, unsqueeze_757);  sub_226 = unsqueeze_757 = None
    mul_913: "f32[8, 92, 14, 14]" = torch.ops.aten.mul.Tensor(sub_227, unsqueeze_763);  sub_227 = unsqueeze_763 = None
    mul_914: "f32[92]" = torch.ops.aten.mul.Tensor(sum_91, squeeze_130);  sum_91 = squeeze_130 = None
    convolution_backward_47 = torch.ops.aten.convolution_backward.default(mul_913, relu_19, primals_142, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 92, [True, True, False]);  mul_913 = primals_142 = None
    getitem_301: "f32[8, 92, 14, 14]" = convolution_backward_47[0]
    getitem_302: "f32[92, 1, 3, 3]" = convolution_backward_47[1];  convolution_backward_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    add_450: "f32[8, 92, 14, 14]" = torch.ops.aten.add.Tensor(slice_223, getitem_301);  slice_223 = getitem_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    alias_109: "f32[8, 92, 14, 14]" = torch.ops.aten.alias.default(relu_19);  relu_19 = None
    alias_110: "f32[8, 92, 14, 14]" = torch.ops.aten.alias.default(alias_109);  alias_109 = None
    le_22: "b8[8, 92, 14, 14]" = torch.ops.aten.le.Scalar(alias_110, 0);  alias_110 = None
    where_27: "f32[8, 92, 14, 14]" = torch.ops.aten.where.self(le_22, full_default, add_450);  le_22 = add_450 = None
    sum_92: "f32[92]" = torch.ops.aten.sum.dim_IntList(where_27, [0, 2, 3])
    sub_228: "f32[8, 92, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_766);  convolution_46 = unsqueeze_766 = None
    mul_915: "f32[8, 92, 14, 14]" = torch.ops.aten.mul.Tensor(where_27, sub_228)
    sum_93: "f32[92]" = torch.ops.aten.sum.dim_IntList(mul_915, [0, 2, 3]);  mul_915 = None
    mul_916: "f32[92]" = torch.ops.aten.mul.Tensor(sum_92, 0.0006377551020408163)
    unsqueeze_767: "f32[1, 92]" = torch.ops.aten.unsqueeze.default(mul_916, 0);  mul_916 = None
    unsqueeze_768: "f32[1, 92, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_767, 2);  unsqueeze_767 = None
    unsqueeze_769: "f32[1, 92, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_768, 3);  unsqueeze_768 = None
    mul_917: "f32[92]" = torch.ops.aten.mul.Tensor(sum_93, 0.0006377551020408163)
    mul_918: "f32[92]" = torch.ops.aten.mul.Tensor(squeeze_127, squeeze_127)
    mul_919: "f32[92]" = torch.ops.aten.mul.Tensor(mul_917, mul_918);  mul_917 = mul_918 = None
    unsqueeze_770: "f32[1, 92]" = torch.ops.aten.unsqueeze.default(mul_919, 0);  mul_919 = None
    unsqueeze_771: "f32[1, 92, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_770, 2);  unsqueeze_770 = None
    unsqueeze_772: "f32[1, 92, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_771, 3);  unsqueeze_771 = None
    mul_920: "f32[92]" = torch.ops.aten.mul.Tensor(squeeze_127, primals_140);  primals_140 = None
    unsqueeze_773: "f32[1, 92]" = torch.ops.aten.unsqueeze.default(mul_920, 0);  mul_920 = None
    unsqueeze_774: "f32[1, 92, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_773, 2);  unsqueeze_773 = None
    unsqueeze_775: "f32[1, 92, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_774, 3);  unsqueeze_774 = None
    mul_921: "f32[8, 92, 14, 14]" = torch.ops.aten.mul.Tensor(sub_228, unsqueeze_772);  sub_228 = unsqueeze_772 = None
    sub_230: "f32[8, 92, 14, 14]" = torch.ops.aten.sub.Tensor(where_27, mul_921);  where_27 = mul_921 = None
    sub_231: "f32[8, 92, 14, 14]" = torch.ops.aten.sub.Tensor(sub_230, unsqueeze_769);  sub_230 = unsqueeze_769 = None
    mul_922: "f32[8, 92, 14, 14]" = torch.ops.aten.mul.Tensor(sub_231, unsqueeze_775);  sub_231 = unsqueeze_775 = None
    mul_923: "f32[92]" = torch.ops.aten.mul.Tensor(sum_93, squeeze_127);  sum_93 = squeeze_127 = None
    convolution_backward_48 = torch.ops.aten.convolution_backward.default(mul_922, slice_88, primals_139, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_922 = slice_88 = primals_139 = None
    getitem_304: "f32[8, 80, 14, 14]" = convolution_backward_48[0]
    getitem_305: "f32[92, 80, 1, 1]" = convolution_backward_48[1];  convolution_backward_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    add_451: "f32[8, 80, 14, 14]" = torch.ops.aten.add.Tensor(clone_7, getitem_304);  clone_7 = getitem_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:182, code: x += self.shortcut(shortcut)
    copy_24: "f32[8, 80, 14, 14]" = torch.ops.aten.copy.default(as_strided_49, add_451);  add_451 = None
    as_strided_scatter_16: "f32[125440]" = torch.ops.aten.as_strided_scatter.default(full_28, copy_24, [8, 80, 14, 14], [15680, 196, 14, 1], 0);  copy_24 = None
    as_strided_59: "f32[8, 80, 14, 14]" = torch.ops.aten.as_strided.default(as_strided_scatter_16, [8, 80, 14, 14], [15680, 196, 14, 1], 0);  as_strided_scatter_16 = None
    new_empty_strided_8: "f32[8, 80, 14, 14]" = torch.ops.aten.new_empty_strided.default(as_strided_59, [8, 80, 14, 14], [15680, 196, 14, 1])
    copy_25: "f32[8, 80, 14, 14]" = torch.ops.aten.copy.default(new_empty_strided_8, as_strided_59);  new_empty_strided_8 = as_strided_59 = None
    as_strided_61: "f32[8, 80, 14, 14]" = torch.ops.aten.as_strided.default(copy_25, [8, 80, 14, 14], [15680, 196, 14, 1], 0)
    clone_8: "f32[8, 80, 14, 14]" = torch.ops.aten.clone.default(as_strided_61, memory_format = torch.contiguous_format)
    copy_26: "f32[8, 80, 14, 14]" = torch.ops.aten.copy.default(as_strided_61, clone_8);  as_strided_61 = None
    as_strided_scatter_17: "f32[8, 80, 14, 14]" = torch.ops.aten.as_strided_scatter.default(copy_25, copy_26, [8, 80, 14, 14], [15680, 196, 14, 1], 0);  copy_25 = copy_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    slice_227: "f32[8, 40, 14, 14]" = torch.ops.aten.slice.Tensor(as_strided_scatter_17, 1, 40, 80)
    sum_94: "f32[40]" = torch.ops.aten.sum.dim_IntList(slice_227, [0, 2, 3])
    sub_232: "f32[8, 40, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_778);  convolution_45 = unsqueeze_778 = None
    mul_924: "f32[8, 40, 14, 14]" = torch.ops.aten.mul.Tensor(slice_227, sub_232)
    sum_95: "f32[40]" = torch.ops.aten.sum.dim_IntList(mul_924, [0, 2, 3]);  mul_924 = None
    mul_925: "f32[40]" = torch.ops.aten.mul.Tensor(sum_94, 0.0006377551020408163)
    unsqueeze_779: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_925, 0);  mul_925 = None
    unsqueeze_780: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_779, 2);  unsqueeze_779 = None
    unsqueeze_781: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_780, 3);  unsqueeze_780 = None
    mul_926: "f32[40]" = torch.ops.aten.mul.Tensor(sum_95, 0.0006377551020408163)
    mul_927: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_124, squeeze_124)
    mul_928: "f32[40]" = torch.ops.aten.mul.Tensor(mul_926, mul_927);  mul_926 = mul_927 = None
    unsqueeze_782: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_928, 0);  mul_928 = None
    unsqueeze_783: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_782, 2);  unsqueeze_782 = None
    unsqueeze_784: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_783, 3);  unsqueeze_783 = None
    mul_929: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_124, primals_137);  primals_137 = None
    unsqueeze_785: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_929, 0);  mul_929 = None
    unsqueeze_786: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_785, 2);  unsqueeze_785 = None
    unsqueeze_787: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_786, 3);  unsqueeze_786 = None
    mul_930: "f32[8, 40, 14, 14]" = torch.ops.aten.mul.Tensor(sub_232, unsqueeze_784);  sub_232 = unsqueeze_784 = None
    sub_234: "f32[8, 40, 14, 14]" = torch.ops.aten.sub.Tensor(slice_227, mul_930);  slice_227 = mul_930 = None
    sub_235: "f32[8, 40, 14, 14]" = torch.ops.aten.sub.Tensor(sub_234, unsqueeze_781);  sub_234 = unsqueeze_781 = None
    mul_931: "f32[8, 40, 14, 14]" = torch.ops.aten.mul.Tensor(sub_235, unsqueeze_787);  sub_235 = unsqueeze_787 = None
    mul_932: "f32[40]" = torch.ops.aten.mul.Tensor(sum_95, squeeze_124);  sum_95 = squeeze_124 = None
    convolution_backward_49 = torch.ops.aten.convolution_backward.default(mul_931, add_213, primals_136, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 40, [True, True, False]);  mul_931 = add_213 = primals_136 = None
    getitem_307: "f32[8, 40, 14, 14]" = convolution_backward_49[0]
    getitem_308: "f32[40, 1, 3, 3]" = convolution_backward_49[1];  convolution_backward_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    slice_228: "f32[8, 40, 14, 14]" = torch.ops.aten.slice.Tensor(as_strided_scatter_17, 1, 0, 40);  as_strided_scatter_17 = None
    add_452: "f32[8, 40, 14, 14]" = torch.ops.aten.add.Tensor(slice_228, getitem_307);  slice_228 = getitem_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    sum_96: "f32[40]" = torch.ops.aten.sum.dim_IntList(add_452, [0, 2, 3])
    sub_236: "f32[8, 40, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_44, unsqueeze_790);  convolution_44 = unsqueeze_790 = None
    mul_933: "f32[8, 40, 14, 14]" = torch.ops.aten.mul.Tensor(add_452, sub_236)
    sum_97: "f32[40]" = torch.ops.aten.sum.dim_IntList(mul_933, [0, 2, 3]);  mul_933 = None
    mul_934: "f32[40]" = torch.ops.aten.mul.Tensor(sum_96, 0.0006377551020408163)
    unsqueeze_791: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_934, 0);  mul_934 = None
    unsqueeze_792: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_791, 2);  unsqueeze_791 = None
    unsqueeze_793: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_792, 3);  unsqueeze_792 = None
    mul_935: "f32[40]" = torch.ops.aten.mul.Tensor(sum_97, 0.0006377551020408163)
    mul_936: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_121, squeeze_121)
    mul_937: "f32[40]" = torch.ops.aten.mul.Tensor(mul_935, mul_936);  mul_935 = mul_936 = None
    unsqueeze_794: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_937, 0);  mul_937 = None
    unsqueeze_795: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_794, 2);  unsqueeze_794 = None
    unsqueeze_796: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_795, 3);  unsqueeze_795 = None
    mul_938: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_121, primals_134);  primals_134 = None
    unsqueeze_797: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_938, 0);  mul_938 = None
    unsqueeze_798: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_797, 2);  unsqueeze_797 = None
    unsqueeze_799: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_798, 3);  unsqueeze_798 = None
    mul_939: "f32[8, 40, 14, 14]" = torch.ops.aten.mul.Tensor(sub_236, unsqueeze_796);  sub_236 = unsqueeze_796 = None
    sub_238: "f32[8, 40, 14, 14]" = torch.ops.aten.sub.Tensor(add_452, mul_939);  add_452 = mul_939 = None
    sub_239: "f32[8, 40, 14, 14]" = torch.ops.aten.sub.Tensor(sub_238, unsqueeze_793);  sub_238 = unsqueeze_793 = None
    mul_940: "f32[8, 40, 14, 14]" = torch.ops.aten.mul.Tensor(sub_239, unsqueeze_799);  sub_239 = unsqueeze_799 = None
    mul_941: "f32[40]" = torch.ops.aten.mul.Tensor(sum_97, squeeze_121);  sum_97 = squeeze_121 = None
    convolution_backward_50 = torch.ops.aten.convolution_backward.default(mul_940, slice_80, primals_133, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_940 = slice_80 = primals_133 = None
    getitem_310: "f32[8, 184, 14, 14]" = convolution_backward_50[0]
    getitem_311: "f32[40, 184, 1, 1]" = convolution_backward_50[1];  convolution_backward_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:63, code: return out[:, :self.out_chs, :, :]
    slice_scatter_72: "f32[8, 184, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_47, getitem_310, 3, 0, 9223372036854775807);  getitem_310 = None
    slice_scatter_73: "f32[8, 184, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_47, slice_scatter_72, 2, 0, 9223372036854775807);  slice_scatter_72 = None
    slice_scatter_74: "f32[8, 184, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_47, slice_scatter_73, 0, 0, 9223372036854775807);  full_default_47 = slice_scatter_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    slice_229: "f32[8, 92, 14, 14]" = torch.ops.aten.slice.Tensor(slice_scatter_74, 1, 0, 92)
    slice_230: "f32[8, 92, 14, 14]" = torch.ops.aten.slice.Tensor(slice_scatter_74, 1, 92, 184);  slice_scatter_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    where_28: "f32[8, 92, 14, 14]" = torch.ops.aten.where.self(le_23, full_default, slice_230);  le_23 = slice_230 = None
    sum_98: "f32[92]" = torch.ops.aten.sum.dim_IntList(where_28, [0, 2, 3])
    sub_240: "f32[8, 92, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_43, unsqueeze_802);  convolution_43 = unsqueeze_802 = None
    mul_942: "f32[8, 92, 14, 14]" = torch.ops.aten.mul.Tensor(where_28, sub_240)
    sum_99: "f32[92]" = torch.ops.aten.sum.dim_IntList(mul_942, [0, 2, 3]);  mul_942 = None
    mul_943: "f32[92]" = torch.ops.aten.mul.Tensor(sum_98, 0.0006377551020408163)
    unsqueeze_803: "f32[1, 92]" = torch.ops.aten.unsqueeze.default(mul_943, 0);  mul_943 = None
    unsqueeze_804: "f32[1, 92, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_803, 2);  unsqueeze_803 = None
    unsqueeze_805: "f32[1, 92, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_804, 3);  unsqueeze_804 = None
    mul_944: "f32[92]" = torch.ops.aten.mul.Tensor(sum_99, 0.0006377551020408163)
    mul_945: "f32[92]" = torch.ops.aten.mul.Tensor(squeeze_118, squeeze_118)
    mul_946: "f32[92]" = torch.ops.aten.mul.Tensor(mul_944, mul_945);  mul_944 = mul_945 = None
    unsqueeze_806: "f32[1, 92]" = torch.ops.aten.unsqueeze.default(mul_946, 0);  mul_946 = None
    unsqueeze_807: "f32[1, 92, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_806, 2);  unsqueeze_806 = None
    unsqueeze_808: "f32[1, 92, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_807, 3);  unsqueeze_807 = None
    mul_947: "f32[92]" = torch.ops.aten.mul.Tensor(squeeze_118, primals_131);  primals_131 = None
    unsqueeze_809: "f32[1, 92]" = torch.ops.aten.unsqueeze.default(mul_947, 0);  mul_947 = None
    unsqueeze_810: "f32[1, 92, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_809, 2);  unsqueeze_809 = None
    unsqueeze_811: "f32[1, 92, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_810, 3);  unsqueeze_810 = None
    mul_948: "f32[8, 92, 14, 14]" = torch.ops.aten.mul.Tensor(sub_240, unsqueeze_808);  sub_240 = unsqueeze_808 = None
    sub_242: "f32[8, 92, 14, 14]" = torch.ops.aten.sub.Tensor(where_28, mul_948);  where_28 = mul_948 = None
    sub_243: "f32[8, 92, 14, 14]" = torch.ops.aten.sub.Tensor(sub_242, unsqueeze_805);  sub_242 = unsqueeze_805 = None
    mul_949: "f32[8, 92, 14, 14]" = torch.ops.aten.mul.Tensor(sub_243, unsqueeze_811);  sub_243 = unsqueeze_811 = None
    mul_950: "f32[92]" = torch.ops.aten.mul.Tensor(sum_99, squeeze_118);  sum_99 = squeeze_118 = None
    convolution_backward_51 = torch.ops.aten.convolution_backward.default(mul_949, relu_17, primals_130, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 92, [True, True, False]);  mul_949 = primals_130 = None
    getitem_313: "f32[8, 92, 14, 14]" = convolution_backward_51[0]
    getitem_314: "f32[92, 1, 3, 3]" = convolution_backward_51[1];  convolution_backward_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    add_453: "f32[8, 92, 14, 14]" = torch.ops.aten.add.Tensor(slice_229, getitem_313);  slice_229 = getitem_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    alias_115: "f32[8, 92, 14, 14]" = torch.ops.aten.alias.default(relu_17);  relu_17 = None
    alias_116: "f32[8, 92, 14, 14]" = torch.ops.aten.alias.default(alias_115);  alias_115 = None
    le_24: "b8[8, 92, 14, 14]" = torch.ops.aten.le.Scalar(alias_116, 0);  alias_116 = None
    where_29: "f32[8, 92, 14, 14]" = torch.ops.aten.where.self(le_24, full_default, add_453);  le_24 = add_453 = None
    sum_100: "f32[92]" = torch.ops.aten.sum.dim_IntList(where_29, [0, 2, 3])
    sub_244: "f32[8, 92, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_42, unsqueeze_814);  convolution_42 = unsqueeze_814 = None
    mul_951: "f32[8, 92, 14, 14]" = torch.ops.aten.mul.Tensor(where_29, sub_244)
    sum_101: "f32[92]" = torch.ops.aten.sum.dim_IntList(mul_951, [0, 2, 3]);  mul_951 = None
    mul_952: "f32[92]" = torch.ops.aten.mul.Tensor(sum_100, 0.0006377551020408163)
    unsqueeze_815: "f32[1, 92]" = torch.ops.aten.unsqueeze.default(mul_952, 0);  mul_952 = None
    unsqueeze_816: "f32[1, 92, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_815, 2);  unsqueeze_815 = None
    unsqueeze_817: "f32[1, 92, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_816, 3);  unsqueeze_816 = None
    mul_953: "f32[92]" = torch.ops.aten.mul.Tensor(sum_101, 0.0006377551020408163)
    mul_954: "f32[92]" = torch.ops.aten.mul.Tensor(squeeze_115, squeeze_115)
    mul_955: "f32[92]" = torch.ops.aten.mul.Tensor(mul_953, mul_954);  mul_953 = mul_954 = None
    unsqueeze_818: "f32[1, 92]" = torch.ops.aten.unsqueeze.default(mul_955, 0);  mul_955 = None
    unsqueeze_819: "f32[1, 92, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_818, 2);  unsqueeze_818 = None
    unsqueeze_820: "f32[1, 92, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_819, 3);  unsqueeze_819 = None
    mul_956: "f32[92]" = torch.ops.aten.mul.Tensor(squeeze_115, primals_128);  primals_128 = None
    unsqueeze_821: "f32[1, 92]" = torch.ops.aten.unsqueeze.default(mul_956, 0);  mul_956 = None
    unsqueeze_822: "f32[1, 92, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_821, 2);  unsqueeze_821 = None
    unsqueeze_823: "f32[1, 92, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_822, 3);  unsqueeze_822 = None
    mul_957: "f32[8, 92, 14, 14]" = torch.ops.aten.mul.Tensor(sub_244, unsqueeze_820);  sub_244 = unsqueeze_820 = None
    sub_246: "f32[8, 92, 14, 14]" = torch.ops.aten.sub.Tensor(where_29, mul_957);  where_29 = mul_957 = None
    sub_247: "f32[8, 92, 14, 14]" = torch.ops.aten.sub.Tensor(sub_246, unsqueeze_817);  sub_246 = unsqueeze_817 = None
    mul_958: "f32[8, 92, 14, 14]" = torch.ops.aten.mul.Tensor(sub_247, unsqueeze_823);  sub_247 = unsqueeze_823 = None
    mul_959: "f32[92]" = torch.ops.aten.mul.Tensor(sum_101, squeeze_115);  sum_101 = squeeze_115 = None
    convolution_backward_52 = torch.ops.aten.convolution_backward.default(mul_958, slice_77, primals_127, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_958 = slice_77 = primals_127 = None
    getitem_316: "f32[8, 80, 14, 14]" = convolution_backward_52[0]
    getitem_317: "f32[92, 80, 1, 1]" = convolution_backward_52[1];  convolution_backward_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    add_454: "f32[8, 80, 14, 14]" = torch.ops.aten.add.Tensor(clone_8, getitem_316);  clone_8 = getitem_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:182, code: x += self.shortcut(shortcut)
    copy_27: "f32[8, 80, 14, 14]" = torch.ops.aten.copy.default(as_strided_49, add_454);  add_454 = None
    as_strided_scatter_18: "f32[125440]" = torch.ops.aten.as_strided_scatter.default(full_28, copy_27, [8, 80, 14, 14], [15680, 196, 14, 1], 0);  copy_27 = None
    as_strided_66: "f32[8, 80, 14, 14]" = torch.ops.aten.as_strided.default(as_strided_scatter_18, [8, 80, 14, 14], [15680, 196, 14, 1], 0);  as_strided_scatter_18 = None
    new_empty_strided_9: "f32[8, 80, 14, 14]" = torch.ops.aten.new_empty_strided.default(as_strided_66, [8, 80, 14, 14], [15680, 196, 14, 1])
    copy_28: "f32[8, 80, 14, 14]" = torch.ops.aten.copy.default(new_empty_strided_9, as_strided_66);  new_empty_strided_9 = as_strided_66 = None
    as_strided_68: "f32[8, 80, 14, 14]" = torch.ops.aten.as_strided.default(copy_28, [8, 80, 14, 14], [15680, 196, 14, 1], 0)
    clone_9: "f32[8, 80, 14, 14]" = torch.ops.aten.clone.default(as_strided_68, memory_format = torch.contiguous_format)
    copy_29: "f32[8, 80, 14, 14]" = torch.ops.aten.copy.default(as_strided_68, clone_9);  as_strided_68 = None
    as_strided_scatter_19: "f32[8, 80, 14, 14]" = torch.ops.aten.as_strided_scatter.default(copy_28, copy_29, [8, 80, 14, 14], [15680, 196, 14, 1], 0);  copy_28 = copy_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    slice_233: "f32[8, 40, 14, 14]" = torch.ops.aten.slice.Tensor(as_strided_scatter_19, 1, 40, 80)
    sum_102: "f32[40]" = torch.ops.aten.sum.dim_IntList(slice_233, [0, 2, 3])
    sub_248: "f32[8, 40, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_826);  convolution_41 = unsqueeze_826 = None
    mul_960: "f32[8, 40, 14, 14]" = torch.ops.aten.mul.Tensor(slice_233, sub_248)
    sum_103: "f32[40]" = torch.ops.aten.sum.dim_IntList(mul_960, [0, 2, 3]);  mul_960 = None
    mul_961: "f32[40]" = torch.ops.aten.mul.Tensor(sum_102, 0.0006377551020408163)
    unsqueeze_827: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_961, 0);  mul_961 = None
    unsqueeze_828: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_827, 2);  unsqueeze_827 = None
    unsqueeze_829: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_828, 3);  unsqueeze_828 = None
    mul_962: "f32[40]" = torch.ops.aten.mul.Tensor(sum_103, 0.0006377551020408163)
    mul_963: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_112, squeeze_112)
    mul_964: "f32[40]" = torch.ops.aten.mul.Tensor(mul_962, mul_963);  mul_962 = mul_963 = None
    unsqueeze_830: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_964, 0);  mul_964 = None
    unsqueeze_831: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_830, 2);  unsqueeze_830 = None
    unsqueeze_832: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_831, 3);  unsqueeze_831 = None
    mul_965: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_112, primals_125);  primals_125 = None
    unsqueeze_833: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_965, 0);  mul_965 = None
    unsqueeze_834: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_833, 2);  unsqueeze_833 = None
    unsqueeze_835: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_834, 3);  unsqueeze_834 = None
    mul_966: "f32[8, 40, 14, 14]" = torch.ops.aten.mul.Tensor(sub_248, unsqueeze_832);  sub_248 = unsqueeze_832 = None
    sub_250: "f32[8, 40, 14, 14]" = torch.ops.aten.sub.Tensor(slice_233, mul_966);  slice_233 = mul_966 = None
    sub_251: "f32[8, 40, 14, 14]" = torch.ops.aten.sub.Tensor(sub_250, unsqueeze_829);  sub_250 = unsqueeze_829 = None
    mul_967: "f32[8, 40, 14, 14]" = torch.ops.aten.mul.Tensor(sub_251, unsqueeze_835);  sub_251 = unsqueeze_835 = None
    mul_968: "f32[40]" = torch.ops.aten.mul.Tensor(sum_103, squeeze_112);  sum_103 = squeeze_112 = None
    convolution_backward_53 = torch.ops.aten.convolution_backward.default(mul_967, add_192, primals_124, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 40, [True, True, False]);  mul_967 = add_192 = primals_124 = None
    getitem_319: "f32[8, 40, 14, 14]" = convolution_backward_53[0]
    getitem_320: "f32[40, 1, 3, 3]" = convolution_backward_53[1];  convolution_backward_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    slice_234: "f32[8, 40, 14, 14]" = torch.ops.aten.slice.Tensor(as_strided_scatter_19, 1, 0, 40);  as_strided_scatter_19 = None
    add_455: "f32[8, 40, 14, 14]" = torch.ops.aten.add.Tensor(slice_234, getitem_319);  slice_234 = getitem_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    sum_104: "f32[40]" = torch.ops.aten.sum.dim_IntList(add_455, [0, 2, 3])
    sub_252: "f32[8, 40, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_838);  convolution_40 = unsqueeze_838 = None
    mul_969: "f32[8, 40, 14, 14]" = torch.ops.aten.mul.Tensor(add_455, sub_252)
    sum_105: "f32[40]" = torch.ops.aten.sum.dim_IntList(mul_969, [0, 2, 3]);  mul_969 = None
    mul_970: "f32[40]" = torch.ops.aten.mul.Tensor(sum_104, 0.0006377551020408163)
    unsqueeze_839: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_970, 0);  mul_970 = None
    unsqueeze_840: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_839, 2);  unsqueeze_839 = None
    unsqueeze_841: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_840, 3);  unsqueeze_840 = None
    mul_971: "f32[40]" = torch.ops.aten.mul.Tensor(sum_105, 0.0006377551020408163)
    mul_972: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_109, squeeze_109)
    mul_973: "f32[40]" = torch.ops.aten.mul.Tensor(mul_971, mul_972);  mul_971 = mul_972 = None
    unsqueeze_842: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_973, 0);  mul_973 = None
    unsqueeze_843: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_842, 2);  unsqueeze_842 = None
    unsqueeze_844: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_843, 3);  unsqueeze_843 = None
    mul_974: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_109, primals_122);  primals_122 = None
    unsqueeze_845: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_974, 0);  mul_974 = None
    unsqueeze_846: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_845, 2);  unsqueeze_845 = None
    unsqueeze_847: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_846, 3);  unsqueeze_846 = None
    mul_975: "f32[8, 40, 14, 14]" = torch.ops.aten.mul.Tensor(sub_252, unsqueeze_844);  sub_252 = unsqueeze_844 = None
    sub_254: "f32[8, 40, 14, 14]" = torch.ops.aten.sub.Tensor(add_455, mul_975);  add_455 = mul_975 = None
    sub_255: "f32[8, 40, 14, 14]" = torch.ops.aten.sub.Tensor(sub_254, unsqueeze_841);  sub_254 = unsqueeze_841 = None
    mul_976: "f32[8, 40, 14, 14]" = torch.ops.aten.mul.Tensor(sub_255, unsqueeze_847);  sub_255 = unsqueeze_847 = None
    mul_977: "f32[40]" = torch.ops.aten.mul.Tensor(sum_105, squeeze_109);  sum_105 = squeeze_109 = None
    convolution_backward_54 = torch.ops.aten.convolution_backward.default(mul_976, slice_69, primals_121, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_976 = slice_69 = primals_121 = None
    getitem_322: "f32[8, 200, 14, 14]" = convolution_backward_54[0]
    getitem_323: "f32[40, 200, 1, 1]" = convolution_backward_54[1];  convolution_backward_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:63, code: return out[:, :self.out_chs, :, :]
    full_default_57: "f32[8, 200, 14, 14]" = torch.ops.aten.full.default([8, 200, 14, 14], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_75: "f32[8, 200, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_57, getitem_322, 3, 0, 9223372036854775807);  getitem_322 = None
    slice_scatter_76: "f32[8, 200, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_57, slice_scatter_75, 2, 0, 9223372036854775807);  slice_scatter_75 = None
    slice_scatter_77: "f32[8, 200, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_57, slice_scatter_76, 0, 0, 9223372036854775807);  full_default_57 = slice_scatter_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    slice_235: "f32[8, 100, 14, 14]" = torch.ops.aten.slice.Tensor(slice_scatter_77, 1, 0, 100)
    slice_236: "f32[8, 100, 14, 14]" = torch.ops.aten.slice.Tensor(slice_scatter_77, 1, 100, 200);  slice_scatter_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    where_30: "f32[8, 100, 14, 14]" = torch.ops.aten.where.self(le_25, full_default, slice_236);  le_25 = slice_236 = None
    sum_106: "f32[100]" = torch.ops.aten.sum.dim_IntList(where_30, [0, 2, 3])
    sub_256: "f32[8, 100, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_39, unsqueeze_850);  convolution_39 = unsqueeze_850 = None
    mul_978: "f32[8, 100, 14, 14]" = torch.ops.aten.mul.Tensor(where_30, sub_256)
    sum_107: "f32[100]" = torch.ops.aten.sum.dim_IntList(mul_978, [0, 2, 3]);  mul_978 = None
    mul_979: "f32[100]" = torch.ops.aten.mul.Tensor(sum_106, 0.0006377551020408163)
    unsqueeze_851: "f32[1, 100]" = torch.ops.aten.unsqueeze.default(mul_979, 0);  mul_979 = None
    unsqueeze_852: "f32[1, 100, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_851, 2);  unsqueeze_851 = None
    unsqueeze_853: "f32[1, 100, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_852, 3);  unsqueeze_852 = None
    mul_980: "f32[100]" = torch.ops.aten.mul.Tensor(sum_107, 0.0006377551020408163)
    mul_981: "f32[100]" = torch.ops.aten.mul.Tensor(squeeze_106, squeeze_106)
    mul_982: "f32[100]" = torch.ops.aten.mul.Tensor(mul_980, mul_981);  mul_980 = mul_981 = None
    unsqueeze_854: "f32[1, 100]" = torch.ops.aten.unsqueeze.default(mul_982, 0);  mul_982 = None
    unsqueeze_855: "f32[1, 100, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_854, 2);  unsqueeze_854 = None
    unsqueeze_856: "f32[1, 100, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_855, 3);  unsqueeze_855 = None
    mul_983: "f32[100]" = torch.ops.aten.mul.Tensor(squeeze_106, primals_119);  primals_119 = None
    unsqueeze_857: "f32[1, 100]" = torch.ops.aten.unsqueeze.default(mul_983, 0);  mul_983 = None
    unsqueeze_858: "f32[1, 100, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_857, 2);  unsqueeze_857 = None
    unsqueeze_859: "f32[1, 100, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_858, 3);  unsqueeze_858 = None
    mul_984: "f32[8, 100, 14, 14]" = torch.ops.aten.mul.Tensor(sub_256, unsqueeze_856);  sub_256 = unsqueeze_856 = None
    sub_258: "f32[8, 100, 14, 14]" = torch.ops.aten.sub.Tensor(where_30, mul_984);  where_30 = mul_984 = None
    sub_259: "f32[8, 100, 14, 14]" = torch.ops.aten.sub.Tensor(sub_258, unsqueeze_853);  sub_258 = unsqueeze_853 = None
    mul_985: "f32[8, 100, 14, 14]" = torch.ops.aten.mul.Tensor(sub_259, unsqueeze_859);  sub_259 = unsqueeze_859 = None
    mul_986: "f32[100]" = torch.ops.aten.mul.Tensor(sum_107, squeeze_106);  sum_107 = squeeze_106 = None
    convolution_backward_55 = torch.ops.aten.convolution_backward.default(mul_985, relu_15, primals_118, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 100, [True, True, False]);  mul_985 = primals_118 = None
    getitem_325: "f32[8, 100, 14, 14]" = convolution_backward_55[0]
    getitem_326: "f32[100, 1, 3, 3]" = convolution_backward_55[1];  convolution_backward_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    add_456: "f32[8, 100, 14, 14]" = torch.ops.aten.add.Tensor(slice_235, getitem_325);  slice_235 = getitem_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    alias_121: "f32[8, 100, 14, 14]" = torch.ops.aten.alias.default(relu_15);  relu_15 = None
    alias_122: "f32[8, 100, 14, 14]" = torch.ops.aten.alias.default(alias_121);  alias_121 = None
    le_26: "b8[8, 100, 14, 14]" = torch.ops.aten.le.Scalar(alias_122, 0);  alias_122 = None
    where_31: "f32[8, 100, 14, 14]" = torch.ops.aten.where.self(le_26, full_default, add_456);  le_26 = add_456 = None
    sum_108: "f32[100]" = torch.ops.aten.sum.dim_IntList(where_31, [0, 2, 3])
    sub_260: "f32[8, 100, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_38, unsqueeze_862);  convolution_38 = unsqueeze_862 = None
    mul_987: "f32[8, 100, 14, 14]" = torch.ops.aten.mul.Tensor(where_31, sub_260)
    sum_109: "f32[100]" = torch.ops.aten.sum.dim_IntList(mul_987, [0, 2, 3]);  mul_987 = None
    mul_988: "f32[100]" = torch.ops.aten.mul.Tensor(sum_108, 0.0006377551020408163)
    unsqueeze_863: "f32[1, 100]" = torch.ops.aten.unsqueeze.default(mul_988, 0);  mul_988 = None
    unsqueeze_864: "f32[1, 100, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_863, 2);  unsqueeze_863 = None
    unsqueeze_865: "f32[1, 100, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_864, 3);  unsqueeze_864 = None
    mul_989: "f32[100]" = torch.ops.aten.mul.Tensor(sum_109, 0.0006377551020408163)
    mul_990: "f32[100]" = torch.ops.aten.mul.Tensor(squeeze_103, squeeze_103)
    mul_991: "f32[100]" = torch.ops.aten.mul.Tensor(mul_989, mul_990);  mul_989 = mul_990 = None
    unsqueeze_866: "f32[1, 100]" = torch.ops.aten.unsqueeze.default(mul_991, 0);  mul_991 = None
    unsqueeze_867: "f32[1, 100, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_866, 2);  unsqueeze_866 = None
    unsqueeze_868: "f32[1, 100, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_867, 3);  unsqueeze_867 = None
    mul_992: "f32[100]" = torch.ops.aten.mul.Tensor(squeeze_103, primals_116);  primals_116 = None
    unsqueeze_869: "f32[1, 100]" = torch.ops.aten.unsqueeze.default(mul_992, 0);  mul_992 = None
    unsqueeze_870: "f32[1, 100, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_869, 2);  unsqueeze_869 = None
    unsqueeze_871: "f32[1, 100, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_870, 3);  unsqueeze_870 = None
    mul_993: "f32[8, 100, 14, 14]" = torch.ops.aten.mul.Tensor(sub_260, unsqueeze_868);  sub_260 = unsqueeze_868 = None
    sub_262: "f32[8, 100, 14, 14]" = torch.ops.aten.sub.Tensor(where_31, mul_993);  where_31 = mul_993 = None
    sub_263: "f32[8, 100, 14, 14]" = torch.ops.aten.sub.Tensor(sub_262, unsqueeze_865);  sub_262 = unsqueeze_865 = None
    mul_994: "f32[8, 100, 14, 14]" = torch.ops.aten.mul.Tensor(sub_263, unsqueeze_871);  sub_263 = unsqueeze_871 = None
    mul_995: "f32[100]" = torch.ops.aten.mul.Tensor(sum_109, squeeze_103);  sum_109 = squeeze_103 = None
    convolution_backward_56 = torch.ops.aten.convolution_backward.default(mul_994, slice_66, primals_115, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_994 = slice_66 = primals_115 = None
    getitem_328: "f32[8, 80, 14, 14]" = convolution_backward_56[0]
    getitem_329: "f32[100, 80, 1, 1]" = convolution_backward_56[1];  convolution_backward_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    add_457: "f32[8, 80, 14, 14]" = torch.ops.aten.add.Tensor(clone_9, getitem_328);  clone_9 = getitem_328 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:182, code: x += self.shortcut(shortcut)
    copy_30: "f32[8, 80, 14, 14]" = torch.ops.aten.copy.default(as_strided_49, add_457);  as_strided_49 = add_457 = None
    as_strided_scatter_20: "f32[125440]" = torch.ops.aten.as_strided_scatter.default(full_28, copy_30, [8, 80, 14, 14], [15680, 196, 14, 1], 0);  full_28 = copy_30 = None
    as_strided_73: "f32[8, 80, 14, 14]" = torch.ops.aten.as_strided.default(as_strided_scatter_20, [8, 80, 14, 14], [15680, 196, 14, 1], 0);  as_strided_scatter_20 = None
    new_empty_strided_10: "f32[8, 80, 14, 14]" = torch.ops.aten.new_empty_strided.default(as_strided_73, [8, 80, 14, 14], [15680, 196, 14, 1])
    copy_31: "f32[8, 80, 14, 14]" = torch.ops.aten.copy.default(new_empty_strided_10, as_strided_73);  new_empty_strided_10 = as_strided_73 = None
    as_strided_75: "f32[8, 80, 14, 14]" = torch.ops.aten.as_strided.default(copy_31, [8, 80, 14, 14], [15680, 196, 14, 1], 0)
    clone_10: "f32[8, 80, 14, 14]" = torch.ops.aten.clone.default(as_strided_75, memory_format = torch.contiguous_format)
    copy_32: "f32[8, 80, 14, 14]" = torch.ops.aten.copy.default(as_strided_75, clone_10);  as_strided_75 = None
    as_strided_scatter_21: "f32[8, 80, 14, 14]" = torch.ops.aten.as_strided_scatter.default(copy_31, copy_32, [8, 80, 14, 14], [15680, 196, 14, 1], 0);  copy_31 = copy_32 = None
    sum_110: "f32[80]" = torch.ops.aten.sum.dim_IntList(clone_10, [0, 2, 3])
    sub_264: "f32[8, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_37, unsqueeze_874);  convolution_37 = unsqueeze_874 = None
    mul_996: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(clone_10, sub_264)
    sum_111: "f32[80]" = torch.ops.aten.sum.dim_IntList(mul_996, [0, 2, 3]);  mul_996 = None
    mul_997: "f32[80]" = torch.ops.aten.mul.Tensor(sum_110, 0.0006377551020408163)
    unsqueeze_875: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_997, 0);  mul_997 = None
    unsqueeze_876: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_875, 2);  unsqueeze_875 = None
    unsqueeze_877: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_876, 3);  unsqueeze_876 = None
    mul_998: "f32[80]" = torch.ops.aten.mul.Tensor(sum_111, 0.0006377551020408163)
    mul_999: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_100, squeeze_100)
    mul_1000: "f32[80]" = torch.ops.aten.mul.Tensor(mul_998, mul_999);  mul_998 = mul_999 = None
    unsqueeze_878: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_1000, 0);  mul_1000 = None
    unsqueeze_879: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_878, 2);  unsqueeze_878 = None
    unsqueeze_880: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_879, 3);  unsqueeze_879 = None
    mul_1001: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_100, primals_113);  primals_113 = None
    unsqueeze_881: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_1001, 0);  mul_1001 = None
    unsqueeze_882: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_881, 2);  unsqueeze_881 = None
    unsqueeze_883: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_882, 3);  unsqueeze_882 = None
    mul_1002: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_264, unsqueeze_880);  sub_264 = unsqueeze_880 = None
    sub_266: "f32[8, 80, 14, 14]" = torch.ops.aten.sub.Tensor(clone_10, mul_1002);  clone_10 = mul_1002 = None
    sub_267: "f32[8, 80, 14, 14]" = torch.ops.aten.sub.Tensor(sub_266, unsqueeze_877);  sub_266 = unsqueeze_877 = None
    mul_1003: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_267, unsqueeze_883);  sub_267 = unsqueeze_883 = None
    mul_1004: "f32[80]" = torch.ops.aten.mul.Tensor(sum_111, squeeze_100);  sum_111 = squeeze_100 = None
    convolution_backward_57 = torch.ops.aten.convolution_backward.default(mul_1003, add_171, primals_112, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1003 = add_171 = primals_112 = None
    getitem_331: "f32[8, 40, 14, 14]" = convolution_backward_57[0]
    getitem_332: "f32[80, 40, 1, 1]" = convolution_backward_57[1];  convolution_backward_57 = None
    sum_112: "f32[40]" = torch.ops.aten.sum.dim_IntList(getitem_331, [0, 2, 3])
    sub_268: "f32[8, 40, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_886);  convolution_36 = unsqueeze_886 = None
    mul_1005: "f32[8, 40, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_331, sub_268)
    sum_113: "f32[40]" = torch.ops.aten.sum.dim_IntList(mul_1005, [0, 2, 3]);  mul_1005 = None
    mul_1006: "f32[40]" = torch.ops.aten.mul.Tensor(sum_112, 0.0006377551020408163)
    unsqueeze_887: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_1006, 0);  mul_1006 = None
    unsqueeze_888: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_887, 2);  unsqueeze_887 = None
    unsqueeze_889: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_888, 3);  unsqueeze_888 = None
    mul_1007: "f32[40]" = torch.ops.aten.mul.Tensor(sum_113, 0.0006377551020408163)
    mul_1008: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_97, squeeze_97)
    mul_1009: "f32[40]" = torch.ops.aten.mul.Tensor(mul_1007, mul_1008);  mul_1007 = mul_1008 = None
    unsqueeze_890: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_1009, 0);  mul_1009 = None
    unsqueeze_891: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_890, 2);  unsqueeze_890 = None
    unsqueeze_892: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_891, 3);  unsqueeze_891 = None
    mul_1010: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_97, primals_110);  primals_110 = None
    unsqueeze_893: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_1010, 0);  mul_1010 = None
    unsqueeze_894: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_893, 2);  unsqueeze_893 = None
    unsqueeze_895: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_894, 3);  unsqueeze_894 = None
    mul_1011: "f32[8, 40, 14, 14]" = torch.ops.aten.mul.Tensor(sub_268, unsqueeze_892);  sub_268 = unsqueeze_892 = None
    sub_270: "f32[8, 40, 14, 14]" = torch.ops.aten.sub.Tensor(getitem_331, mul_1011);  getitem_331 = mul_1011 = None
    sub_271: "f32[8, 40, 14, 14]" = torch.ops.aten.sub.Tensor(sub_270, unsqueeze_889);  sub_270 = unsqueeze_889 = None
    mul_1012: "f32[8, 40, 14, 14]" = torch.ops.aten.mul.Tensor(sub_271, unsqueeze_895);  sub_271 = unsqueeze_895 = None
    mul_1013: "f32[40]" = torch.ops.aten.mul.Tensor(sum_113, squeeze_97);  sum_113 = squeeze_97 = None
    convolution_backward_58 = torch.ops.aten.convolution_backward.default(mul_1012, slice_55, primals_109, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 40, [True, True, False]);  mul_1012 = primals_109 = None
    getitem_334: "f32[8, 40, 28, 28]" = convolution_backward_58[0]
    getitem_335: "f32[40, 1, 3, 3]" = convolution_backward_58[1];  convolution_backward_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    slice_239: "f32[8, 40, 14, 14]" = torch.ops.aten.slice.Tensor(as_strided_scatter_21, 1, 40, 80)
    sum_114: "f32[40]" = torch.ops.aten.sum.dim_IntList(slice_239, [0, 2, 3])
    sub_272: "f32[8, 40, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_898);  convolution_35 = unsqueeze_898 = None
    mul_1014: "f32[8, 40, 14, 14]" = torch.ops.aten.mul.Tensor(slice_239, sub_272)
    sum_115: "f32[40]" = torch.ops.aten.sum.dim_IntList(mul_1014, [0, 2, 3]);  mul_1014 = None
    mul_1015: "f32[40]" = torch.ops.aten.mul.Tensor(sum_114, 0.0006377551020408163)
    unsqueeze_899: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_1015, 0);  mul_1015 = None
    unsqueeze_900: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_899, 2);  unsqueeze_899 = None
    unsqueeze_901: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_900, 3);  unsqueeze_900 = None
    mul_1016: "f32[40]" = torch.ops.aten.mul.Tensor(sum_115, 0.0006377551020408163)
    mul_1017: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_94, squeeze_94)
    mul_1018: "f32[40]" = torch.ops.aten.mul.Tensor(mul_1016, mul_1017);  mul_1016 = mul_1017 = None
    unsqueeze_902: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_1018, 0);  mul_1018 = None
    unsqueeze_903: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_902, 2);  unsqueeze_902 = None
    unsqueeze_904: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_903, 3);  unsqueeze_903 = None
    mul_1019: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_94, primals_107);  primals_107 = None
    unsqueeze_905: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_1019, 0);  mul_1019 = None
    unsqueeze_906: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_905, 2);  unsqueeze_905 = None
    unsqueeze_907: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_906, 3);  unsqueeze_906 = None
    mul_1020: "f32[8, 40, 14, 14]" = torch.ops.aten.mul.Tensor(sub_272, unsqueeze_904);  sub_272 = unsqueeze_904 = None
    sub_274: "f32[8, 40, 14, 14]" = torch.ops.aten.sub.Tensor(slice_239, mul_1020);  slice_239 = mul_1020 = None
    sub_275: "f32[8, 40, 14, 14]" = torch.ops.aten.sub.Tensor(sub_274, unsqueeze_901);  sub_274 = unsqueeze_901 = None
    mul_1021: "f32[8, 40, 14, 14]" = torch.ops.aten.mul.Tensor(sub_275, unsqueeze_907);  sub_275 = unsqueeze_907 = None
    mul_1022: "f32[40]" = torch.ops.aten.mul.Tensor(sum_115, squeeze_94);  sum_115 = squeeze_94 = None
    convolution_backward_59 = torch.ops.aten.convolution_backward.default(mul_1021, add_161, primals_106, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 40, [True, True, False]);  mul_1021 = add_161 = primals_106 = None
    getitem_337: "f32[8, 40, 14, 14]" = convolution_backward_59[0]
    getitem_338: "f32[40, 1, 3, 3]" = convolution_backward_59[1];  convolution_backward_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    slice_240: "f32[8, 40, 14, 14]" = torch.ops.aten.slice.Tensor(as_strided_scatter_21, 1, 0, 40);  as_strided_scatter_21 = None
    add_458: "f32[8, 40, 14, 14]" = torch.ops.aten.add.Tensor(slice_240, getitem_337);  slice_240 = getitem_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    sum_116: "f32[40]" = torch.ops.aten.sum.dim_IntList(add_458, [0, 2, 3])
    sub_276: "f32[8, 40, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_910);  convolution_34 = unsqueeze_910 = None
    mul_1023: "f32[8, 40, 14, 14]" = torch.ops.aten.mul.Tensor(add_458, sub_276)
    sum_117: "f32[40]" = torch.ops.aten.sum.dim_IntList(mul_1023, [0, 2, 3]);  mul_1023 = None
    mul_1024: "f32[40]" = torch.ops.aten.mul.Tensor(sum_116, 0.0006377551020408163)
    unsqueeze_911: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_1024, 0);  mul_1024 = None
    unsqueeze_912: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_911, 2);  unsqueeze_911 = None
    unsqueeze_913: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_912, 3);  unsqueeze_912 = None
    mul_1025: "f32[40]" = torch.ops.aten.mul.Tensor(sum_117, 0.0006377551020408163)
    mul_1026: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_91, squeeze_91)
    mul_1027: "f32[40]" = torch.ops.aten.mul.Tensor(mul_1025, mul_1026);  mul_1025 = mul_1026 = None
    unsqueeze_914: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_1027, 0);  mul_1027 = None
    unsqueeze_915: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_914, 2);  unsqueeze_914 = None
    unsqueeze_916: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_915, 3);  unsqueeze_915 = None
    mul_1028: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_91, primals_104);  primals_104 = None
    unsqueeze_917: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_1028, 0);  mul_1028 = None
    unsqueeze_918: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_917, 2);  unsqueeze_917 = None
    unsqueeze_919: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_918, 3);  unsqueeze_918 = None
    mul_1029: "f32[8, 40, 14, 14]" = torch.ops.aten.mul.Tensor(sub_276, unsqueeze_916);  sub_276 = unsqueeze_916 = None
    sub_278: "f32[8, 40, 14, 14]" = torch.ops.aten.sub.Tensor(add_458, mul_1029);  add_458 = mul_1029 = None
    sub_279: "f32[8, 40, 14, 14]" = torch.ops.aten.sub.Tensor(sub_278, unsqueeze_913);  sub_278 = unsqueeze_913 = None
    mul_1030: "f32[8, 40, 14, 14]" = torch.ops.aten.mul.Tensor(sub_279, unsqueeze_919);  sub_279 = unsqueeze_919 = None
    mul_1031: "f32[40]" = torch.ops.aten.mul.Tensor(sum_117, squeeze_91);  sum_117 = squeeze_91 = None
    convolution_backward_60 = torch.ops.aten.convolution_backward.default(mul_1030, add_156, primals_103, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1030 = add_156 = primals_103 = None
    getitem_340: "f32[8, 240, 14, 14]" = convolution_backward_60[0]
    getitem_341: "f32[40, 240, 1, 1]" = convolution_backward_60[1];  convolution_backward_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:173, code: x = self.bn_dw(x)
    sum_118: "f32[240]" = torch.ops.aten.sum.dim_IntList(getitem_340, [0, 2, 3])
    sub_280: "f32[8, 240, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_922);  convolution_33 = unsqueeze_922 = None
    mul_1032: "f32[8, 240, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_340, sub_280)
    sum_119: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_1032, [0, 2, 3]);  mul_1032 = None
    mul_1033: "f32[240]" = torch.ops.aten.mul.Tensor(sum_118, 0.0006377551020408163)
    unsqueeze_923: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_1033, 0);  mul_1033 = None
    unsqueeze_924: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_923, 2);  unsqueeze_923 = None
    unsqueeze_925: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_924, 3);  unsqueeze_924 = None
    mul_1034: "f32[240]" = torch.ops.aten.mul.Tensor(sum_119, 0.0006377551020408163)
    mul_1035: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_88, squeeze_88)
    mul_1036: "f32[240]" = torch.ops.aten.mul.Tensor(mul_1034, mul_1035);  mul_1034 = mul_1035 = None
    unsqueeze_926: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_1036, 0);  mul_1036 = None
    unsqueeze_927: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_926, 2);  unsqueeze_926 = None
    unsqueeze_928: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_927, 3);  unsqueeze_927 = None
    mul_1037: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_88, primals_101);  primals_101 = None
    unsqueeze_929: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_1037, 0);  mul_1037 = None
    unsqueeze_930: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_929, 2);  unsqueeze_929 = None
    unsqueeze_931: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_930, 3);  unsqueeze_930 = None
    mul_1038: "f32[8, 240, 14, 14]" = torch.ops.aten.mul.Tensor(sub_280, unsqueeze_928);  sub_280 = unsqueeze_928 = None
    sub_282: "f32[8, 240, 14, 14]" = torch.ops.aten.sub.Tensor(getitem_340, mul_1038);  getitem_340 = mul_1038 = None
    sub_283: "f32[8, 240, 14, 14]" = torch.ops.aten.sub.Tensor(sub_282, unsqueeze_925);  sub_282 = unsqueeze_925 = None
    mul_1039: "f32[8, 240, 14, 14]" = torch.ops.aten.mul.Tensor(sub_283, unsqueeze_931);  sub_283 = unsqueeze_931 = None
    mul_1040: "f32[240]" = torch.ops.aten.mul.Tensor(sum_119, squeeze_88);  sum_119 = squeeze_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:172, code: x = self.conv_dw(x)
    convolution_backward_61 = torch.ops.aten.convolution_backward.default(mul_1039, slice_58, primals_100, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 240, [True, True, False]);  mul_1039 = slice_58 = primals_100 = None
    getitem_343: "f32[8, 240, 28, 28]" = convolution_backward_61[0]
    getitem_344: "f32[240, 1, 3, 3]" = convolution_backward_61[1];  convolution_backward_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:63, code: return out[:, :self.out_chs, :, :]
    full_default_62: "f32[8, 240, 28, 28]" = torch.ops.aten.full.default([8, 240, 28, 28], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_78: "f32[8, 240, 28, 28]" = torch.ops.aten.slice_scatter.default(full_default_62, getitem_343, 3, 0, 9223372036854775807);  getitem_343 = None
    slice_scatter_79: "f32[8, 240, 28, 28]" = torch.ops.aten.slice_scatter.default(full_default_62, slice_scatter_78, 2, 0, 9223372036854775807);  slice_scatter_78 = None
    slice_scatter_80: "f32[8, 240, 28, 28]" = torch.ops.aten.slice_scatter.default(full_default_62, slice_scatter_79, 0, 0, 9223372036854775807);  full_default_62 = slice_scatter_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    slice_241: "f32[8, 120, 28, 28]" = torch.ops.aten.slice.Tensor(slice_scatter_80, 1, 0, 120)
    slice_242: "f32[8, 120, 28, 28]" = torch.ops.aten.slice.Tensor(slice_scatter_80, 1, 120, 240);  slice_scatter_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    where_32: "f32[8, 120, 28, 28]" = torch.ops.aten.where.self(le_27, full_default, slice_242);  le_27 = slice_242 = None
    sum_120: "f32[120]" = torch.ops.aten.sum.dim_IntList(where_32, [0, 2, 3])
    sub_284: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_934);  convolution_32 = unsqueeze_934 = None
    mul_1041: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(where_32, sub_284)
    sum_121: "f32[120]" = torch.ops.aten.sum.dim_IntList(mul_1041, [0, 2, 3]);  mul_1041 = None
    mul_1042: "f32[120]" = torch.ops.aten.mul.Tensor(sum_120, 0.00015943877551020407)
    unsqueeze_935: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1042, 0);  mul_1042 = None
    unsqueeze_936: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_935, 2);  unsqueeze_935 = None
    unsqueeze_937: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_936, 3);  unsqueeze_936 = None
    mul_1043: "f32[120]" = torch.ops.aten.mul.Tensor(sum_121, 0.00015943877551020407)
    mul_1044: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_85, squeeze_85)
    mul_1045: "f32[120]" = torch.ops.aten.mul.Tensor(mul_1043, mul_1044);  mul_1043 = mul_1044 = None
    unsqueeze_938: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1045, 0);  mul_1045 = None
    unsqueeze_939: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_938, 2);  unsqueeze_938 = None
    unsqueeze_940: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_939, 3);  unsqueeze_939 = None
    mul_1046: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_85, primals_98);  primals_98 = None
    unsqueeze_941: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1046, 0);  mul_1046 = None
    unsqueeze_942: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_941, 2);  unsqueeze_941 = None
    unsqueeze_943: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_942, 3);  unsqueeze_942 = None
    mul_1047: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_284, unsqueeze_940);  sub_284 = unsqueeze_940 = None
    sub_286: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(where_32, mul_1047);  where_32 = mul_1047 = None
    sub_287: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(sub_286, unsqueeze_937);  sub_286 = unsqueeze_937 = None
    mul_1048: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_287, unsqueeze_943);  sub_287 = unsqueeze_943 = None
    mul_1049: "f32[120]" = torch.ops.aten.mul.Tensor(sum_121, squeeze_85);  sum_121 = squeeze_85 = None
    convolution_backward_62 = torch.ops.aten.convolution_backward.default(mul_1048, relu_13, primals_97, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 120, [True, True, False]);  mul_1048 = primals_97 = None
    getitem_346: "f32[8, 120, 28, 28]" = convolution_backward_62[0]
    getitem_347: "f32[120, 1, 3, 3]" = convolution_backward_62[1];  convolution_backward_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    add_459: "f32[8, 120, 28, 28]" = torch.ops.aten.add.Tensor(slice_241, getitem_346);  slice_241 = getitem_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    alias_127: "f32[8, 120, 28, 28]" = torch.ops.aten.alias.default(relu_13);  relu_13 = None
    alias_128: "f32[8, 120, 28, 28]" = torch.ops.aten.alias.default(alias_127);  alias_127 = None
    le_28: "b8[8, 120, 28, 28]" = torch.ops.aten.le.Scalar(alias_128, 0);  alias_128 = None
    where_33: "f32[8, 120, 28, 28]" = torch.ops.aten.where.self(le_28, full_default, add_459);  le_28 = add_459 = None
    sum_122: "f32[120]" = torch.ops.aten.sum.dim_IntList(where_33, [0, 2, 3])
    sub_288: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_31, unsqueeze_946);  convolution_31 = unsqueeze_946 = None
    mul_1050: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(where_33, sub_288)
    sum_123: "f32[120]" = torch.ops.aten.sum.dim_IntList(mul_1050, [0, 2, 3]);  mul_1050 = None
    mul_1051: "f32[120]" = torch.ops.aten.mul.Tensor(sum_122, 0.00015943877551020407)
    unsqueeze_947: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1051, 0);  mul_1051 = None
    unsqueeze_948: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_947, 2);  unsqueeze_947 = None
    unsqueeze_949: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_948, 3);  unsqueeze_948 = None
    mul_1052: "f32[120]" = torch.ops.aten.mul.Tensor(sum_123, 0.00015943877551020407)
    mul_1053: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_82, squeeze_82)
    mul_1054: "f32[120]" = torch.ops.aten.mul.Tensor(mul_1052, mul_1053);  mul_1052 = mul_1053 = None
    unsqueeze_950: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1054, 0);  mul_1054 = None
    unsqueeze_951: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_950, 2);  unsqueeze_950 = None
    unsqueeze_952: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_951, 3);  unsqueeze_951 = None
    mul_1055: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_82, primals_95);  primals_95 = None
    unsqueeze_953: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1055, 0);  mul_1055 = None
    unsqueeze_954: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_953, 2);  unsqueeze_953 = None
    unsqueeze_955: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_954, 3);  unsqueeze_954 = None
    mul_1056: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_288, unsqueeze_952);  sub_288 = unsqueeze_952 = None
    sub_290: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(where_33, mul_1056);  where_33 = mul_1056 = None
    sub_291: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(sub_290, unsqueeze_949);  sub_290 = unsqueeze_949 = None
    mul_1057: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_291, unsqueeze_955);  sub_291 = unsqueeze_955 = None
    mul_1058: "f32[120]" = torch.ops.aten.mul.Tensor(sum_123, squeeze_82);  sum_123 = squeeze_82 = None
    convolution_backward_63 = torch.ops.aten.convolution_backward.default(mul_1057, slice_55, primals_94, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1057 = slice_55 = primals_94 = None
    getitem_349: "f32[8, 40, 28, 28]" = convolution_backward_63[0]
    getitem_350: "f32[120, 40, 1, 1]" = convolution_backward_63[1];  convolution_backward_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    add_460: "f32[8, 40, 28, 28]" = torch.ops.aten.add.Tensor(getitem_334, getitem_349);  getitem_334 = getitem_349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:182, code: x += self.shortcut(shortcut)
    full_44: "f32[250880]" = torch.ops.aten.full.default([250880], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    as_strided_77: "f32[8, 40, 28, 28]" = torch.ops.aten.as_strided.default(full_44, [8, 40, 28, 28], [31360, 784, 28, 1], 0)
    copy_33: "f32[8, 40, 28, 28]" = torch.ops.aten.copy.default(as_strided_77, add_460);  add_460 = None
    as_strided_scatter_22: "f32[250880]" = torch.ops.aten.as_strided_scatter.default(full_44, copy_33, [8, 40, 28, 28], [31360, 784, 28, 1], 0);  copy_33 = None
    as_strided_80: "f32[8, 40, 28, 28]" = torch.ops.aten.as_strided.default(as_strided_scatter_22, [8, 40, 28, 28], [31360, 784, 28, 1], 0);  as_strided_scatter_22 = None
    new_empty_strided_11: "f32[8, 40, 28, 28]" = torch.ops.aten.new_empty_strided.default(as_strided_80, [8, 40, 28, 28], [31360, 784, 28, 1])
    copy_34: "f32[8, 40, 28, 28]" = torch.ops.aten.copy.default(new_empty_strided_11, as_strided_80);  new_empty_strided_11 = as_strided_80 = None
    as_strided_82: "f32[8, 40, 28, 28]" = torch.ops.aten.as_strided.default(copy_34, [8, 40, 28, 28], [31360, 784, 28, 1], 0)
    clone_11: "f32[8, 40, 28, 28]" = torch.ops.aten.clone.default(as_strided_82, memory_format = torch.contiguous_format)
    copy_35: "f32[8, 40, 28, 28]" = torch.ops.aten.copy.default(as_strided_82, clone_11);  as_strided_82 = None
    as_strided_scatter_23: "f32[8, 40, 28, 28]" = torch.ops.aten.as_strided_scatter.default(copy_34, copy_35, [8, 40, 28, 28], [31360, 784, 28, 1], 0);  copy_34 = copy_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    slice_245: "f32[8, 20, 28, 28]" = torch.ops.aten.slice.Tensor(as_strided_scatter_23, 1, 20, 40)
    sum_124: "f32[20]" = torch.ops.aten.sum.dim_IntList(slice_245, [0, 2, 3])
    sub_292: "f32[8, 20, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_958);  convolution_30 = unsqueeze_958 = None
    mul_1059: "f32[8, 20, 28, 28]" = torch.ops.aten.mul.Tensor(slice_245, sub_292)
    sum_125: "f32[20]" = torch.ops.aten.sum.dim_IntList(mul_1059, [0, 2, 3]);  mul_1059 = None
    mul_1060: "f32[20]" = torch.ops.aten.mul.Tensor(sum_124, 0.00015943877551020407)
    unsqueeze_959: "f32[1, 20]" = torch.ops.aten.unsqueeze.default(mul_1060, 0);  mul_1060 = None
    unsqueeze_960: "f32[1, 20, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_959, 2);  unsqueeze_959 = None
    unsqueeze_961: "f32[1, 20, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_960, 3);  unsqueeze_960 = None
    mul_1061: "f32[20]" = torch.ops.aten.mul.Tensor(sum_125, 0.00015943877551020407)
    mul_1062: "f32[20]" = torch.ops.aten.mul.Tensor(squeeze_79, squeeze_79)
    mul_1063: "f32[20]" = torch.ops.aten.mul.Tensor(mul_1061, mul_1062);  mul_1061 = mul_1062 = None
    unsqueeze_962: "f32[1, 20]" = torch.ops.aten.unsqueeze.default(mul_1063, 0);  mul_1063 = None
    unsqueeze_963: "f32[1, 20, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_962, 2);  unsqueeze_962 = None
    unsqueeze_964: "f32[1, 20, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_963, 3);  unsqueeze_963 = None
    mul_1064: "f32[20]" = torch.ops.aten.mul.Tensor(squeeze_79, primals_92);  primals_92 = None
    unsqueeze_965: "f32[1, 20]" = torch.ops.aten.unsqueeze.default(mul_1064, 0);  mul_1064 = None
    unsqueeze_966: "f32[1, 20, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_965, 2);  unsqueeze_965 = None
    unsqueeze_967: "f32[1, 20, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_966, 3);  unsqueeze_966 = None
    mul_1065: "f32[8, 20, 28, 28]" = torch.ops.aten.mul.Tensor(sub_292, unsqueeze_964);  sub_292 = unsqueeze_964 = None
    sub_294: "f32[8, 20, 28, 28]" = torch.ops.aten.sub.Tensor(slice_245, mul_1065);  slice_245 = mul_1065 = None
    sub_295: "f32[8, 20, 28, 28]" = torch.ops.aten.sub.Tensor(sub_294, unsqueeze_961);  sub_294 = unsqueeze_961 = None
    mul_1066: "f32[8, 20, 28, 28]" = torch.ops.aten.mul.Tensor(sub_295, unsqueeze_967);  sub_295 = unsqueeze_967 = None
    mul_1067: "f32[20]" = torch.ops.aten.mul.Tensor(sum_125, squeeze_79);  sum_125 = squeeze_79 = None
    convolution_backward_64 = torch.ops.aten.convolution_backward.default(mul_1066, add_135, primals_91, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 20, [True, True, False]);  mul_1066 = add_135 = primals_91 = None
    getitem_352: "f32[8, 20, 28, 28]" = convolution_backward_64[0]
    getitem_353: "f32[20, 1, 3, 3]" = convolution_backward_64[1];  convolution_backward_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    slice_246: "f32[8, 20, 28, 28]" = torch.ops.aten.slice.Tensor(as_strided_scatter_23, 1, 0, 20);  as_strided_scatter_23 = None
    add_461: "f32[8, 20, 28, 28]" = torch.ops.aten.add.Tensor(slice_246, getitem_352);  slice_246 = getitem_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    sum_126: "f32[20]" = torch.ops.aten.sum.dim_IntList(add_461, [0, 2, 3])
    sub_296: "f32[8, 20, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_970);  convolution_29 = unsqueeze_970 = None
    mul_1068: "f32[8, 20, 28, 28]" = torch.ops.aten.mul.Tensor(add_461, sub_296)
    sum_127: "f32[20]" = torch.ops.aten.sum.dim_IntList(mul_1068, [0, 2, 3]);  mul_1068 = None
    mul_1069: "f32[20]" = torch.ops.aten.mul.Tensor(sum_126, 0.00015943877551020407)
    unsqueeze_971: "f32[1, 20]" = torch.ops.aten.unsqueeze.default(mul_1069, 0);  mul_1069 = None
    unsqueeze_972: "f32[1, 20, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_971, 2);  unsqueeze_971 = None
    unsqueeze_973: "f32[1, 20, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_972, 3);  unsqueeze_972 = None
    mul_1070: "f32[20]" = torch.ops.aten.mul.Tensor(sum_127, 0.00015943877551020407)
    mul_1071: "f32[20]" = torch.ops.aten.mul.Tensor(squeeze_76, squeeze_76)
    mul_1072: "f32[20]" = torch.ops.aten.mul.Tensor(mul_1070, mul_1071);  mul_1070 = mul_1071 = None
    unsqueeze_974: "f32[1, 20]" = torch.ops.aten.unsqueeze.default(mul_1072, 0);  mul_1072 = None
    unsqueeze_975: "f32[1, 20, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_974, 2);  unsqueeze_974 = None
    unsqueeze_976: "f32[1, 20, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_975, 3);  unsqueeze_975 = None
    mul_1073: "f32[20]" = torch.ops.aten.mul.Tensor(squeeze_76, primals_89);  primals_89 = None
    unsqueeze_977: "f32[1, 20]" = torch.ops.aten.unsqueeze.default(mul_1073, 0);  mul_1073 = None
    unsqueeze_978: "f32[1, 20, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_977, 2);  unsqueeze_977 = None
    unsqueeze_979: "f32[1, 20, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_978, 3);  unsqueeze_978 = None
    mul_1074: "f32[8, 20, 28, 28]" = torch.ops.aten.mul.Tensor(sub_296, unsqueeze_976);  sub_296 = unsqueeze_976 = None
    sub_298: "f32[8, 20, 28, 28]" = torch.ops.aten.sub.Tensor(add_461, mul_1074);  add_461 = mul_1074 = None
    sub_299: "f32[8, 20, 28, 28]" = torch.ops.aten.sub.Tensor(sub_298, unsqueeze_973);  sub_298 = unsqueeze_973 = None
    mul_1075: "f32[8, 20, 28, 28]" = torch.ops.aten.mul.Tensor(sub_299, unsqueeze_979);  sub_299 = unsqueeze_979 = None
    mul_1076: "f32[20]" = torch.ops.aten.mul.Tensor(sum_127, squeeze_76);  sum_127 = squeeze_76 = None
    convolution_backward_65 = torch.ops.aten.convolution_backward.default(mul_1075, mul_176, primals_88, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1075 = mul_176 = primals_88 = None
    getitem_355: "f32[8, 120, 28, 28]" = convolution_backward_65[0]
    getitem_356: "f32[20, 120, 1, 1]" = convolution_backward_65[1];  convolution_backward_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_1077: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_355, slice_47);  slice_47 = None
    mul_1078: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_355, div_1);  getitem_355 = div_1 = None
    sum_128: "f32[8, 120, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1077, [2, 3], True);  mul_1077 = None
    mul_1079: "f32[8, 120, 1, 1]" = torch.ops.aten.mul.Tensor(sum_128, 0.16666666666666666);  sum_128 = None
    where_34: "f32[8, 120, 1, 1]" = torch.ops.aten.where.self(bitwise_and_5, mul_1079, full_default);  bitwise_and_5 = mul_1079 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    sum_129: "f32[120]" = torch.ops.aten.sum.dim_IntList(where_34, [0, 2, 3])
    convolution_backward_66 = torch.ops.aten.convolution_backward.default(where_34, relu_12, primals_86, [120], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_34 = primals_86 = None
    getitem_358: "f32[8, 32, 1, 1]" = convolution_backward_66[0]
    getitem_359: "f32[120, 32, 1, 1]" = convolution_backward_66[1];  convolution_backward_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    alias_130: "f32[8, 32, 1, 1]" = torch.ops.aten.alias.default(relu_12);  relu_12 = None
    alias_131: "f32[8, 32, 1, 1]" = torch.ops.aten.alias.default(alias_130);  alias_130 = None
    le_29: "b8[8, 32, 1, 1]" = torch.ops.aten.le.Scalar(alias_131, 0);  alias_131 = None
    where_35: "f32[8, 32, 1, 1]" = torch.ops.aten.where.self(le_29, full_default, getitem_358);  le_29 = getitem_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    sum_130: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_35, [0, 2, 3])
    convolution_backward_67 = torch.ops.aten.convolution_backward.default(where_35, mean_1, primals_84, [32], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_35 = mean_1 = primals_84 = None
    getitem_361: "f32[8, 120, 1, 1]" = convolution_backward_67[0]
    getitem_362: "f32[32, 120, 1, 1]" = convolution_backward_67[1];  convolution_backward_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_6: "f32[8, 120, 28, 28]" = torch.ops.aten.expand.default(getitem_361, [8, 120, 28, 28]);  getitem_361 = None
    div_13: "f32[8, 120, 28, 28]" = torch.ops.aten.div.Scalar(expand_6, 784);  expand_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_462: "f32[8, 120, 28, 28]" = torch.ops.aten.add.Tensor(mul_1078, div_13);  mul_1078 = div_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:63, code: return out[:, :self.out_chs, :, :]
    full_default_69: "f32[8, 120, 28, 28]" = torch.ops.aten.full.default([8, 120, 28, 28], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_81: "f32[8, 120, 28, 28]" = torch.ops.aten.slice_scatter.default(full_default_69, add_462, 3, 0, 9223372036854775807);  add_462 = None
    slice_scatter_82: "f32[8, 120, 28, 28]" = torch.ops.aten.slice_scatter.default(full_default_69, slice_scatter_81, 2, 0, 9223372036854775807);  slice_scatter_81 = None
    slice_scatter_83: "f32[8, 120, 28, 28]" = torch.ops.aten.slice_scatter.default(full_default_69, slice_scatter_82, 0, 0, 9223372036854775807);  full_default_69 = slice_scatter_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    slice_247: "f32[8, 60, 28, 28]" = torch.ops.aten.slice.Tensor(slice_scatter_83, 1, 0, 60)
    slice_248: "f32[8, 60, 28, 28]" = torch.ops.aten.slice.Tensor(slice_scatter_83, 1, 60, 120);  slice_scatter_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    where_36: "f32[8, 60, 28, 28]" = torch.ops.aten.where.self(le_30, full_default, slice_248);  le_30 = slice_248 = None
    sum_131: "f32[60]" = torch.ops.aten.sum.dim_IntList(where_36, [0, 2, 3])
    sub_300: "f32[8, 60, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_982);  convolution_26 = unsqueeze_982 = None
    mul_1080: "f32[8, 60, 28, 28]" = torch.ops.aten.mul.Tensor(where_36, sub_300)
    sum_132: "f32[60]" = torch.ops.aten.sum.dim_IntList(mul_1080, [0, 2, 3]);  mul_1080 = None
    mul_1081: "f32[60]" = torch.ops.aten.mul.Tensor(sum_131, 0.00015943877551020407)
    unsqueeze_983: "f32[1, 60]" = torch.ops.aten.unsqueeze.default(mul_1081, 0);  mul_1081 = None
    unsqueeze_984: "f32[1, 60, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_983, 2);  unsqueeze_983 = None
    unsqueeze_985: "f32[1, 60, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_984, 3);  unsqueeze_984 = None
    mul_1082: "f32[60]" = torch.ops.aten.mul.Tensor(sum_132, 0.00015943877551020407)
    mul_1083: "f32[60]" = torch.ops.aten.mul.Tensor(squeeze_73, squeeze_73)
    mul_1084: "f32[60]" = torch.ops.aten.mul.Tensor(mul_1082, mul_1083);  mul_1082 = mul_1083 = None
    unsqueeze_986: "f32[1, 60]" = torch.ops.aten.unsqueeze.default(mul_1084, 0);  mul_1084 = None
    unsqueeze_987: "f32[1, 60, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_986, 2);  unsqueeze_986 = None
    unsqueeze_988: "f32[1, 60, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_987, 3);  unsqueeze_987 = None
    mul_1085: "f32[60]" = torch.ops.aten.mul.Tensor(squeeze_73, primals_82);  primals_82 = None
    unsqueeze_989: "f32[1, 60]" = torch.ops.aten.unsqueeze.default(mul_1085, 0);  mul_1085 = None
    unsqueeze_990: "f32[1, 60, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_989, 2);  unsqueeze_989 = None
    unsqueeze_991: "f32[1, 60, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_990, 3);  unsqueeze_990 = None
    mul_1086: "f32[8, 60, 28, 28]" = torch.ops.aten.mul.Tensor(sub_300, unsqueeze_988);  sub_300 = unsqueeze_988 = None
    sub_302: "f32[8, 60, 28, 28]" = torch.ops.aten.sub.Tensor(where_36, mul_1086);  where_36 = mul_1086 = None
    sub_303: "f32[8, 60, 28, 28]" = torch.ops.aten.sub.Tensor(sub_302, unsqueeze_985);  sub_302 = unsqueeze_985 = None
    mul_1087: "f32[8, 60, 28, 28]" = torch.ops.aten.mul.Tensor(sub_303, unsqueeze_991);  sub_303 = unsqueeze_991 = None
    mul_1088: "f32[60]" = torch.ops.aten.mul.Tensor(sum_132, squeeze_73);  sum_132 = squeeze_73 = None
    convolution_backward_68 = torch.ops.aten.convolution_backward.default(mul_1087, relu_10, primals_81, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 60, [True, True, False]);  mul_1087 = primals_81 = None
    getitem_364: "f32[8, 60, 28, 28]" = convolution_backward_68[0]
    getitem_365: "f32[60, 1, 3, 3]" = convolution_backward_68[1];  convolution_backward_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    add_463: "f32[8, 60, 28, 28]" = torch.ops.aten.add.Tensor(slice_247, getitem_364);  slice_247 = getitem_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    alias_136: "f32[8, 60, 28, 28]" = torch.ops.aten.alias.default(relu_10);  relu_10 = None
    alias_137: "f32[8, 60, 28, 28]" = torch.ops.aten.alias.default(alias_136);  alias_136 = None
    le_31: "b8[8, 60, 28, 28]" = torch.ops.aten.le.Scalar(alias_137, 0);  alias_137 = None
    where_37: "f32[8, 60, 28, 28]" = torch.ops.aten.where.self(le_31, full_default, add_463);  le_31 = add_463 = None
    sum_133: "f32[60]" = torch.ops.aten.sum.dim_IntList(where_37, [0, 2, 3])
    sub_304: "f32[8, 60, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_994);  convolution_25 = unsqueeze_994 = None
    mul_1089: "f32[8, 60, 28, 28]" = torch.ops.aten.mul.Tensor(where_37, sub_304)
    sum_134: "f32[60]" = torch.ops.aten.sum.dim_IntList(mul_1089, [0, 2, 3]);  mul_1089 = None
    mul_1090: "f32[60]" = torch.ops.aten.mul.Tensor(sum_133, 0.00015943877551020407)
    unsqueeze_995: "f32[1, 60]" = torch.ops.aten.unsqueeze.default(mul_1090, 0);  mul_1090 = None
    unsqueeze_996: "f32[1, 60, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_995, 2);  unsqueeze_995 = None
    unsqueeze_997: "f32[1, 60, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_996, 3);  unsqueeze_996 = None
    mul_1091: "f32[60]" = torch.ops.aten.mul.Tensor(sum_134, 0.00015943877551020407)
    mul_1092: "f32[60]" = torch.ops.aten.mul.Tensor(squeeze_70, squeeze_70)
    mul_1093: "f32[60]" = torch.ops.aten.mul.Tensor(mul_1091, mul_1092);  mul_1091 = mul_1092 = None
    unsqueeze_998: "f32[1, 60]" = torch.ops.aten.unsqueeze.default(mul_1093, 0);  mul_1093 = None
    unsqueeze_999: "f32[1, 60, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_998, 2);  unsqueeze_998 = None
    unsqueeze_1000: "f32[1, 60, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_999, 3);  unsqueeze_999 = None
    mul_1094: "f32[60]" = torch.ops.aten.mul.Tensor(squeeze_70, primals_79);  primals_79 = None
    unsqueeze_1001: "f32[1, 60]" = torch.ops.aten.unsqueeze.default(mul_1094, 0);  mul_1094 = None
    unsqueeze_1002: "f32[1, 60, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1001, 2);  unsqueeze_1001 = None
    unsqueeze_1003: "f32[1, 60, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1002, 3);  unsqueeze_1002 = None
    mul_1095: "f32[8, 60, 28, 28]" = torch.ops.aten.mul.Tensor(sub_304, unsqueeze_1000);  sub_304 = unsqueeze_1000 = None
    sub_306: "f32[8, 60, 28, 28]" = torch.ops.aten.sub.Tensor(where_37, mul_1095);  where_37 = mul_1095 = None
    sub_307: "f32[8, 60, 28, 28]" = torch.ops.aten.sub.Tensor(sub_306, unsqueeze_997);  sub_306 = unsqueeze_997 = None
    mul_1096: "f32[8, 60, 28, 28]" = torch.ops.aten.mul.Tensor(sub_307, unsqueeze_1003);  sub_307 = unsqueeze_1003 = None
    mul_1097: "f32[60]" = torch.ops.aten.mul.Tensor(sum_134, squeeze_70);  sum_134 = squeeze_70 = None
    convolution_backward_69 = torch.ops.aten.convolution_backward.default(mul_1096, slice_44, primals_78, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1096 = slice_44 = primals_78 = None
    getitem_367: "f32[8, 40, 28, 28]" = convolution_backward_69[0]
    getitem_368: "f32[60, 40, 1, 1]" = convolution_backward_69[1];  convolution_backward_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    add_464: "f32[8, 40, 28, 28]" = torch.ops.aten.add.Tensor(clone_11, getitem_367);  clone_11 = getitem_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:182, code: x += self.shortcut(shortcut)
    copy_36: "f32[8, 40, 28, 28]" = torch.ops.aten.copy.default(as_strided_77, add_464);  as_strided_77 = add_464 = None
    as_strided_scatter_24: "f32[250880]" = torch.ops.aten.as_strided_scatter.default(full_44, copy_36, [8, 40, 28, 28], [31360, 784, 28, 1], 0);  full_44 = copy_36 = None
    as_strided_87: "f32[8, 40, 28, 28]" = torch.ops.aten.as_strided.default(as_strided_scatter_24, [8, 40, 28, 28], [31360, 784, 28, 1], 0);  as_strided_scatter_24 = None
    new_empty_strided_12: "f32[8, 40, 28, 28]" = torch.ops.aten.new_empty_strided.default(as_strided_87, [8, 40, 28, 28], [31360, 784, 28, 1])
    copy_37: "f32[8, 40, 28, 28]" = torch.ops.aten.copy.default(new_empty_strided_12, as_strided_87);  new_empty_strided_12 = as_strided_87 = None
    as_strided_89: "f32[8, 40, 28, 28]" = torch.ops.aten.as_strided.default(copy_37, [8, 40, 28, 28], [31360, 784, 28, 1], 0)
    clone_12: "f32[8, 40, 28, 28]" = torch.ops.aten.clone.default(as_strided_89, memory_format = torch.contiguous_format)
    copy_38: "f32[8, 40, 28, 28]" = torch.ops.aten.copy.default(as_strided_89, clone_12);  as_strided_89 = None
    as_strided_scatter_25: "f32[8, 40, 28, 28]" = torch.ops.aten.as_strided_scatter.default(copy_37, copy_38, [8, 40, 28, 28], [31360, 784, 28, 1], 0);  copy_37 = copy_38 = None
    sum_135: "f32[40]" = torch.ops.aten.sum.dim_IntList(clone_12, [0, 2, 3])
    sub_308: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_1006);  convolution_24 = unsqueeze_1006 = None
    mul_1098: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(clone_12, sub_308)
    sum_136: "f32[40]" = torch.ops.aten.sum.dim_IntList(mul_1098, [0, 2, 3]);  mul_1098 = None
    mul_1099: "f32[40]" = torch.ops.aten.mul.Tensor(sum_135, 0.00015943877551020407)
    unsqueeze_1007: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_1099, 0);  mul_1099 = None
    unsqueeze_1008: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1007, 2);  unsqueeze_1007 = None
    unsqueeze_1009: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1008, 3);  unsqueeze_1008 = None
    mul_1100: "f32[40]" = torch.ops.aten.mul.Tensor(sum_136, 0.00015943877551020407)
    mul_1101: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_67, squeeze_67)
    mul_1102: "f32[40]" = torch.ops.aten.mul.Tensor(mul_1100, mul_1101);  mul_1100 = mul_1101 = None
    unsqueeze_1010: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_1102, 0);  mul_1102 = None
    unsqueeze_1011: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1010, 2);  unsqueeze_1010 = None
    unsqueeze_1012: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1011, 3);  unsqueeze_1011 = None
    mul_1103: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_67, primals_76);  primals_76 = None
    unsqueeze_1013: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_1103, 0);  mul_1103 = None
    unsqueeze_1014: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1013, 2);  unsqueeze_1013 = None
    unsqueeze_1015: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1014, 3);  unsqueeze_1014 = None
    mul_1104: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_308, unsqueeze_1012);  sub_308 = unsqueeze_1012 = None
    sub_310: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(clone_12, mul_1104);  clone_12 = mul_1104 = None
    sub_311: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(sub_310, unsqueeze_1009);  sub_310 = unsqueeze_1009 = None
    mul_1105: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_311, unsqueeze_1015);  sub_311 = unsqueeze_1015 = None
    mul_1106: "f32[40]" = torch.ops.aten.mul.Tensor(sum_136, squeeze_67);  sum_136 = squeeze_67 = None
    convolution_backward_70 = torch.ops.aten.convolution_backward.default(mul_1105, add_113, primals_75, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1105 = add_113 = primals_75 = None
    getitem_370: "f32[8, 24, 28, 28]" = convolution_backward_70[0]
    getitem_371: "f32[40, 24, 1, 1]" = convolution_backward_70[1];  convolution_backward_70 = None
    sum_137: "f32[24]" = torch.ops.aten.sum.dim_IntList(getitem_370, [0, 2, 3])
    sub_312: "f32[8, 24, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_1018);  convolution_23 = unsqueeze_1018 = None
    mul_1107: "f32[8, 24, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_370, sub_312)
    sum_138: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_1107, [0, 2, 3]);  mul_1107 = None
    mul_1108: "f32[24]" = torch.ops.aten.mul.Tensor(sum_137, 0.00015943877551020407)
    unsqueeze_1019: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_1108, 0);  mul_1108 = None
    unsqueeze_1020: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1019, 2);  unsqueeze_1019 = None
    unsqueeze_1021: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1020, 3);  unsqueeze_1020 = None
    mul_1109: "f32[24]" = torch.ops.aten.mul.Tensor(sum_138, 0.00015943877551020407)
    mul_1110: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_64, squeeze_64)
    mul_1111: "f32[24]" = torch.ops.aten.mul.Tensor(mul_1109, mul_1110);  mul_1109 = mul_1110 = None
    unsqueeze_1022: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_1111, 0);  mul_1111 = None
    unsqueeze_1023: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1022, 2);  unsqueeze_1022 = None
    unsqueeze_1024: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1023, 3);  unsqueeze_1023 = None
    mul_1112: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_64, primals_73);  primals_73 = None
    unsqueeze_1025: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_1112, 0);  mul_1112 = None
    unsqueeze_1026: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1025, 2);  unsqueeze_1025 = None
    unsqueeze_1027: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1026, 3);  unsqueeze_1026 = None
    mul_1113: "f32[8, 24, 28, 28]" = torch.ops.aten.mul.Tensor(sub_312, unsqueeze_1024);  sub_312 = unsqueeze_1024 = None
    sub_314: "f32[8, 24, 28, 28]" = torch.ops.aten.sub.Tensor(getitem_370, mul_1113);  getitem_370 = mul_1113 = None
    sub_315: "f32[8, 24, 28, 28]" = torch.ops.aten.sub.Tensor(sub_314, unsqueeze_1021);  sub_314 = unsqueeze_1021 = None
    mul_1114: "f32[8, 24, 28, 28]" = torch.ops.aten.mul.Tensor(sub_315, unsqueeze_1027);  sub_315 = unsqueeze_1027 = None
    mul_1115: "f32[24]" = torch.ops.aten.mul.Tensor(sum_138, squeeze_64);  sum_138 = squeeze_64 = None
    convolution_backward_71 = torch.ops.aten.convolution_backward.default(mul_1114, slice_33, primals_72, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 24, [True, True, False]);  mul_1114 = primals_72 = None
    getitem_373: "f32[8, 24, 56, 56]" = convolution_backward_71[0]
    getitem_374: "f32[24, 1, 5, 5]" = convolution_backward_71[1];  convolution_backward_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    slice_251: "f32[8, 20, 28, 28]" = torch.ops.aten.slice.Tensor(as_strided_scatter_25, 1, 20, 40)
    sum_139: "f32[20]" = torch.ops.aten.sum.dim_IntList(slice_251, [0, 2, 3])
    sub_316: "f32[8, 20, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_1030);  convolution_22 = unsqueeze_1030 = None
    mul_1116: "f32[8, 20, 28, 28]" = torch.ops.aten.mul.Tensor(slice_251, sub_316)
    sum_140: "f32[20]" = torch.ops.aten.sum.dim_IntList(mul_1116, [0, 2, 3]);  mul_1116 = None
    mul_1117: "f32[20]" = torch.ops.aten.mul.Tensor(sum_139, 0.00015943877551020407)
    unsqueeze_1031: "f32[1, 20]" = torch.ops.aten.unsqueeze.default(mul_1117, 0);  mul_1117 = None
    unsqueeze_1032: "f32[1, 20, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1031, 2);  unsqueeze_1031 = None
    unsqueeze_1033: "f32[1, 20, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1032, 3);  unsqueeze_1032 = None
    mul_1118: "f32[20]" = torch.ops.aten.mul.Tensor(sum_140, 0.00015943877551020407)
    mul_1119: "f32[20]" = torch.ops.aten.mul.Tensor(squeeze_61, squeeze_61)
    mul_1120: "f32[20]" = torch.ops.aten.mul.Tensor(mul_1118, mul_1119);  mul_1118 = mul_1119 = None
    unsqueeze_1034: "f32[1, 20]" = torch.ops.aten.unsqueeze.default(mul_1120, 0);  mul_1120 = None
    unsqueeze_1035: "f32[1, 20, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1034, 2);  unsqueeze_1034 = None
    unsqueeze_1036: "f32[1, 20, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1035, 3);  unsqueeze_1035 = None
    mul_1121: "f32[20]" = torch.ops.aten.mul.Tensor(squeeze_61, primals_70);  primals_70 = None
    unsqueeze_1037: "f32[1, 20]" = torch.ops.aten.unsqueeze.default(mul_1121, 0);  mul_1121 = None
    unsqueeze_1038: "f32[1, 20, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1037, 2);  unsqueeze_1037 = None
    unsqueeze_1039: "f32[1, 20, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1038, 3);  unsqueeze_1038 = None
    mul_1122: "f32[8, 20, 28, 28]" = torch.ops.aten.mul.Tensor(sub_316, unsqueeze_1036);  sub_316 = unsqueeze_1036 = None
    sub_318: "f32[8, 20, 28, 28]" = torch.ops.aten.sub.Tensor(slice_251, mul_1122);  slice_251 = mul_1122 = None
    sub_319: "f32[8, 20, 28, 28]" = torch.ops.aten.sub.Tensor(sub_318, unsqueeze_1033);  sub_318 = unsqueeze_1033 = None
    mul_1123: "f32[8, 20, 28, 28]" = torch.ops.aten.mul.Tensor(sub_319, unsqueeze_1039);  sub_319 = unsqueeze_1039 = None
    mul_1124: "f32[20]" = torch.ops.aten.mul.Tensor(sum_140, squeeze_61);  sum_140 = squeeze_61 = None
    convolution_backward_72 = torch.ops.aten.convolution_backward.default(mul_1123, add_103, primals_69, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 20, [True, True, False]);  mul_1123 = add_103 = primals_69 = None
    getitem_376: "f32[8, 20, 28, 28]" = convolution_backward_72[0]
    getitem_377: "f32[20, 1, 3, 3]" = convolution_backward_72[1];  convolution_backward_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    slice_252: "f32[8, 20, 28, 28]" = torch.ops.aten.slice.Tensor(as_strided_scatter_25, 1, 0, 20);  as_strided_scatter_25 = None
    add_465: "f32[8, 20, 28, 28]" = torch.ops.aten.add.Tensor(slice_252, getitem_376);  slice_252 = getitem_376 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    sum_141: "f32[20]" = torch.ops.aten.sum.dim_IntList(add_465, [0, 2, 3])
    sub_320: "f32[8, 20, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_1042);  convolution_21 = unsqueeze_1042 = None
    mul_1125: "f32[8, 20, 28, 28]" = torch.ops.aten.mul.Tensor(add_465, sub_320)
    sum_142: "f32[20]" = torch.ops.aten.sum.dim_IntList(mul_1125, [0, 2, 3]);  mul_1125 = None
    mul_1126: "f32[20]" = torch.ops.aten.mul.Tensor(sum_141, 0.00015943877551020407)
    unsqueeze_1043: "f32[1, 20]" = torch.ops.aten.unsqueeze.default(mul_1126, 0);  mul_1126 = None
    unsqueeze_1044: "f32[1, 20, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1043, 2);  unsqueeze_1043 = None
    unsqueeze_1045: "f32[1, 20, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1044, 3);  unsqueeze_1044 = None
    mul_1127: "f32[20]" = torch.ops.aten.mul.Tensor(sum_142, 0.00015943877551020407)
    mul_1128: "f32[20]" = torch.ops.aten.mul.Tensor(squeeze_58, squeeze_58)
    mul_1129: "f32[20]" = torch.ops.aten.mul.Tensor(mul_1127, mul_1128);  mul_1127 = mul_1128 = None
    unsqueeze_1046: "f32[1, 20]" = torch.ops.aten.unsqueeze.default(mul_1129, 0);  mul_1129 = None
    unsqueeze_1047: "f32[1, 20, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1046, 2);  unsqueeze_1046 = None
    unsqueeze_1048: "f32[1, 20, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1047, 3);  unsqueeze_1047 = None
    mul_1130: "f32[20]" = torch.ops.aten.mul.Tensor(squeeze_58, primals_67);  primals_67 = None
    unsqueeze_1049: "f32[1, 20]" = torch.ops.aten.unsqueeze.default(mul_1130, 0);  mul_1130 = None
    unsqueeze_1050: "f32[1, 20, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1049, 2);  unsqueeze_1049 = None
    unsqueeze_1051: "f32[1, 20, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1050, 3);  unsqueeze_1050 = None
    mul_1131: "f32[8, 20, 28, 28]" = torch.ops.aten.mul.Tensor(sub_320, unsqueeze_1048);  sub_320 = unsqueeze_1048 = None
    sub_322: "f32[8, 20, 28, 28]" = torch.ops.aten.sub.Tensor(add_465, mul_1131);  add_465 = mul_1131 = None
    sub_323: "f32[8, 20, 28, 28]" = torch.ops.aten.sub.Tensor(sub_322, unsqueeze_1045);  sub_322 = unsqueeze_1045 = None
    mul_1132: "f32[8, 20, 28, 28]" = torch.ops.aten.mul.Tensor(sub_323, unsqueeze_1051);  sub_323 = unsqueeze_1051 = None
    mul_1133: "f32[20]" = torch.ops.aten.mul.Tensor(sum_142, squeeze_58);  sum_142 = squeeze_58 = None
    convolution_backward_73 = torch.ops.aten.convolution_backward.default(mul_1132, mul_133, primals_66, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1132 = mul_133 = primals_66 = None
    getitem_379: "f32[8, 72, 28, 28]" = convolution_backward_73[0]
    getitem_380: "f32[20, 72, 1, 1]" = convolution_backward_73[1];  convolution_backward_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_1134: "f32[8, 72, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_379, add_97);  add_97 = None
    mul_1135: "f32[8, 72, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_379, div);  getitem_379 = div = None
    sum_143: "f32[8, 72, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1134, [2, 3], True);  mul_1134 = None
    mul_1136: "f32[8, 72, 1, 1]" = torch.ops.aten.mul.Tensor(sum_143, 0.16666666666666666);  sum_143 = None
    where_38: "f32[8, 72, 1, 1]" = torch.ops.aten.where.self(bitwise_and_6, mul_1136, full_default);  bitwise_and_6 = mul_1136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    sum_144: "f32[72]" = torch.ops.aten.sum.dim_IntList(where_38, [0, 2, 3])
    convolution_backward_74 = torch.ops.aten.convolution_backward.default(where_38, relu_9, primals_64, [72], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_38 = primals_64 = None
    getitem_382: "f32[8, 20, 1, 1]" = convolution_backward_74[0]
    getitem_383: "f32[72, 20, 1, 1]" = convolution_backward_74[1];  convolution_backward_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    alias_139: "f32[8, 20, 1, 1]" = torch.ops.aten.alias.default(relu_9);  relu_9 = None
    alias_140: "f32[8, 20, 1, 1]" = torch.ops.aten.alias.default(alias_139);  alias_139 = None
    le_32: "b8[8, 20, 1, 1]" = torch.ops.aten.le.Scalar(alias_140, 0);  alias_140 = None
    where_39: "f32[8, 20, 1, 1]" = torch.ops.aten.where.self(le_32, full_default, getitem_382);  le_32 = getitem_382 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    sum_145: "f32[20]" = torch.ops.aten.sum.dim_IntList(where_39, [0, 2, 3])
    convolution_backward_75 = torch.ops.aten.convolution_backward.default(where_39, mean, primals_62, [20], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_39 = mean = primals_62 = None
    getitem_385: "f32[8, 72, 1, 1]" = convolution_backward_75[0]
    getitem_386: "f32[20, 72, 1, 1]" = convolution_backward_75[1];  convolution_backward_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_7: "f32[8, 72, 28, 28]" = torch.ops.aten.expand.default(getitem_385, [8, 72, 28, 28]);  getitem_385 = None
    div_14: "f32[8, 72, 28, 28]" = torch.ops.aten.div.Scalar(expand_7, 784);  expand_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_466: "f32[8, 72, 28, 28]" = torch.ops.aten.add.Tensor(mul_1135, div_14);  mul_1135 = div_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:173, code: x = self.bn_dw(x)
    sum_146: "f32[72]" = torch.ops.aten.sum.dim_IntList(add_466, [0, 2, 3])
    sub_324: "f32[8, 72, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_1054);  convolution_18 = unsqueeze_1054 = None
    mul_1137: "f32[8, 72, 28, 28]" = torch.ops.aten.mul.Tensor(add_466, sub_324)
    sum_147: "f32[72]" = torch.ops.aten.sum.dim_IntList(mul_1137, [0, 2, 3]);  mul_1137 = None
    mul_1138: "f32[72]" = torch.ops.aten.mul.Tensor(sum_146, 0.00015943877551020407)
    unsqueeze_1055: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_1138, 0);  mul_1138 = None
    unsqueeze_1056: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1055, 2);  unsqueeze_1055 = None
    unsqueeze_1057: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1056, 3);  unsqueeze_1056 = None
    mul_1139: "f32[72]" = torch.ops.aten.mul.Tensor(sum_147, 0.00015943877551020407)
    mul_1140: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_55, squeeze_55)
    mul_1141: "f32[72]" = torch.ops.aten.mul.Tensor(mul_1139, mul_1140);  mul_1139 = mul_1140 = None
    unsqueeze_1058: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_1141, 0);  mul_1141 = None
    unsqueeze_1059: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1058, 2);  unsqueeze_1058 = None
    unsqueeze_1060: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1059, 3);  unsqueeze_1059 = None
    mul_1142: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_55, primals_60);  primals_60 = None
    unsqueeze_1061: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_1142, 0);  mul_1142 = None
    unsqueeze_1062: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1061, 2);  unsqueeze_1061 = None
    unsqueeze_1063: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1062, 3);  unsqueeze_1062 = None
    mul_1143: "f32[8, 72, 28, 28]" = torch.ops.aten.mul.Tensor(sub_324, unsqueeze_1060);  sub_324 = unsqueeze_1060 = None
    sub_326: "f32[8, 72, 28, 28]" = torch.ops.aten.sub.Tensor(add_466, mul_1143);  add_466 = mul_1143 = None
    sub_327: "f32[8, 72, 28, 28]" = torch.ops.aten.sub.Tensor(sub_326, unsqueeze_1057);  sub_326 = unsqueeze_1057 = None
    mul_1144: "f32[8, 72, 28, 28]" = torch.ops.aten.mul.Tensor(sub_327, unsqueeze_1063);  sub_327 = unsqueeze_1063 = None
    mul_1145: "f32[72]" = torch.ops.aten.mul.Tensor(sum_147, squeeze_55);  sum_147 = squeeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:172, code: x = self.conv_dw(x)
    convolution_backward_76 = torch.ops.aten.convolution_backward.default(mul_1144, slice_36, primals_59, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 72, [True, True, False]);  mul_1144 = slice_36 = primals_59 = None
    getitem_388: "f32[8, 72, 56, 56]" = convolution_backward_76[0]
    getitem_389: "f32[72, 1, 5, 5]" = convolution_backward_76[1];  convolution_backward_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:63, code: return out[:, :self.out_chs, :, :]
    full_default_76: "f32[8, 72, 56, 56]" = torch.ops.aten.full.default([8, 72, 56, 56], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_84: "f32[8, 72, 56, 56]" = torch.ops.aten.slice_scatter.default(full_default_76, getitem_388, 3, 0, 9223372036854775807);  getitem_388 = None
    slice_scatter_85: "f32[8, 72, 56, 56]" = torch.ops.aten.slice_scatter.default(full_default_76, slice_scatter_84, 2, 0, 9223372036854775807);  slice_scatter_84 = None
    slice_scatter_86: "f32[8, 72, 56, 56]" = torch.ops.aten.slice_scatter.default(full_default_76, slice_scatter_85, 0, 0, 9223372036854775807);  slice_scatter_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    slice_253: "f32[8, 36, 56, 56]" = torch.ops.aten.slice.Tensor(slice_scatter_86, 1, 0, 36)
    slice_254: "f32[8, 36, 56, 56]" = torch.ops.aten.slice.Tensor(slice_scatter_86, 1, 36, 72);  slice_scatter_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    where_40: "f32[8, 36, 56, 56]" = torch.ops.aten.where.self(le_33, full_default, slice_254);  le_33 = slice_254 = None
    sum_148: "f32[36]" = torch.ops.aten.sum.dim_IntList(where_40, [0, 2, 3])
    sub_328: "f32[8, 36, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_1066);  convolution_17 = unsqueeze_1066 = None
    mul_1146: "f32[8, 36, 56, 56]" = torch.ops.aten.mul.Tensor(where_40, sub_328)
    sum_149: "f32[36]" = torch.ops.aten.sum.dim_IntList(mul_1146, [0, 2, 3]);  mul_1146 = None
    mul_1147: "f32[36]" = torch.ops.aten.mul.Tensor(sum_148, 3.985969387755102e-05)
    unsqueeze_1067: "f32[1, 36]" = torch.ops.aten.unsqueeze.default(mul_1147, 0);  mul_1147 = None
    unsqueeze_1068: "f32[1, 36, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1067, 2);  unsqueeze_1067 = None
    unsqueeze_1069: "f32[1, 36, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1068, 3);  unsqueeze_1068 = None
    mul_1148: "f32[36]" = torch.ops.aten.mul.Tensor(sum_149, 3.985969387755102e-05)
    mul_1149: "f32[36]" = torch.ops.aten.mul.Tensor(squeeze_52, squeeze_52)
    mul_1150: "f32[36]" = torch.ops.aten.mul.Tensor(mul_1148, mul_1149);  mul_1148 = mul_1149 = None
    unsqueeze_1070: "f32[1, 36]" = torch.ops.aten.unsqueeze.default(mul_1150, 0);  mul_1150 = None
    unsqueeze_1071: "f32[1, 36, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1070, 2);  unsqueeze_1070 = None
    unsqueeze_1072: "f32[1, 36, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1071, 3);  unsqueeze_1071 = None
    mul_1151: "f32[36]" = torch.ops.aten.mul.Tensor(squeeze_52, primals_57);  primals_57 = None
    unsqueeze_1073: "f32[1, 36]" = torch.ops.aten.unsqueeze.default(mul_1151, 0);  mul_1151 = None
    unsqueeze_1074: "f32[1, 36, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1073, 2);  unsqueeze_1073 = None
    unsqueeze_1075: "f32[1, 36, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1074, 3);  unsqueeze_1074 = None
    mul_1152: "f32[8, 36, 56, 56]" = torch.ops.aten.mul.Tensor(sub_328, unsqueeze_1072);  sub_328 = unsqueeze_1072 = None
    sub_330: "f32[8, 36, 56, 56]" = torch.ops.aten.sub.Tensor(where_40, mul_1152);  where_40 = mul_1152 = None
    sub_331: "f32[8, 36, 56, 56]" = torch.ops.aten.sub.Tensor(sub_330, unsqueeze_1069);  sub_330 = unsqueeze_1069 = None
    mul_1153: "f32[8, 36, 56, 56]" = torch.ops.aten.mul.Tensor(sub_331, unsqueeze_1075);  sub_331 = unsqueeze_1075 = None
    mul_1154: "f32[36]" = torch.ops.aten.mul.Tensor(sum_149, squeeze_52);  sum_149 = squeeze_52 = None
    convolution_backward_77 = torch.ops.aten.convolution_backward.default(mul_1153, relu_7, primals_56, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 36, [True, True, False]);  mul_1153 = primals_56 = None
    getitem_391: "f32[8, 36, 56, 56]" = convolution_backward_77[0]
    getitem_392: "f32[36, 1, 3, 3]" = convolution_backward_77[1];  convolution_backward_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    add_467: "f32[8, 36, 56, 56]" = torch.ops.aten.add.Tensor(slice_253, getitem_391);  slice_253 = getitem_391 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    alias_145: "f32[8, 36, 56, 56]" = torch.ops.aten.alias.default(relu_7);  relu_7 = None
    alias_146: "f32[8, 36, 56, 56]" = torch.ops.aten.alias.default(alias_145);  alias_145 = None
    le_34: "b8[8, 36, 56, 56]" = torch.ops.aten.le.Scalar(alias_146, 0);  alias_146 = None
    where_41: "f32[8, 36, 56, 56]" = torch.ops.aten.where.self(le_34, full_default, add_467);  le_34 = add_467 = None
    sum_150: "f32[36]" = torch.ops.aten.sum.dim_IntList(where_41, [0, 2, 3])
    sub_332: "f32[8, 36, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_1078);  convolution_16 = unsqueeze_1078 = None
    mul_1155: "f32[8, 36, 56, 56]" = torch.ops.aten.mul.Tensor(where_41, sub_332)
    sum_151: "f32[36]" = torch.ops.aten.sum.dim_IntList(mul_1155, [0, 2, 3]);  mul_1155 = None
    mul_1156: "f32[36]" = torch.ops.aten.mul.Tensor(sum_150, 3.985969387755102e-05)
    unsqueeze_1079: "f32[1, 36]" = torch.ops.aten.unsqueeze.default(mul_1156, 0);  mul_1156 = None
    unsqueeze_1080: "f32[1, 36, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1079, 2);  unsqueeze_1079 = None
    unsqueeze_1081: "f32[1, 36, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1080, 3);  unsqueeze_1080 = None
    mul_1157: "f32[36]" = torch.ops.aten.mul.Tensor(sum_151, 3.985969387755102e-05)
    mul_1158: "f32[36]" = torch.ops.aten.mul.Tensor(squeeze_49, squeeze_49)
    mul_1159: "f32[36]" = torch.ops.aten.mul.Tensor(mul_1157, mul_1158);  mul_1157 = mul_1158 = None
    unsqueeze_1082: "f32[1, 36]" = torch.ops.aten.unsqueeze.default(mul_1159, 0);  mul_1159 = None
    unsqueeze_1083: "f32[1, 36, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1082, 2);  unsqueeze_1082 = None
    unsqueeze_1084: "f32[1, 36, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1083, 3);  unsqueeze_1083 = None
    mul_1160: "f32[36]" = torch.ops.aten.mul.Tensor(squeeze_49, primals_54);  primals_54 = None
    unsqueeze_1085: "f32[1, 36]" = torch.ops.aten.unsqueeze.default(mul_1160, 0);  mul_1160 = None
    unsqueeze_1086: "f32[1, 36, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1085, 2);  unsqueeze_1085 = None
    unsqueeze_1087: "f32[1, 36, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1086, 3);  unsqueeze_1086 = None
    mul_1161: "f32[8, 36, 56, 56]" = torch.ops.aten.mul.Tensor(sub_332, unsqueeze_1084);  sub_332 = unsqueeze_1084 = None
    sub_334: "f32[8, 36, 56, 56]" = torch.ops.aten.sub.Tensor(where_41, mul_1161);  where_41 = mul_1161 = None
    sub_335: "f32[8, 36, 56, 56]" = torch.ops.aten.sub.Tensor(sub_334, unsqueeze_1081);  sub_334 = unsqueeze_1081 = None
    mul_1162: "f32[8, 36, 56, 56]" = torch.ops.aten.mul.Tensor(sub_335, unsqueeze_1087);  sub_335 = unsqueeze_1087 = None
    mul_1163: "f32[36]" = torch.ops.aten.mul.Tensor(sum_151, squeeze_49);  sum_151 = squeeze_49 = None
    convolution_backward_78 = torch.ops.aten.convolution_backward.default(mul_1162, slice_33, primals_53, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1162 = slice_33 = primals_53 = None
    getitem_394: "f32[8, 24, 56, 56]" = convolution_backward_78[0]
    getitem_395: "f32[36, 24, 1, 1]" = convolution_backward_78[1];  convolution_backward_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    add_468: "f32[8, 24, 56, 56]" = torch.ops.aten.add.Tensor(getitem_373, getitem_394);  getitem_373 = getitem_394 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:182, code: x += self.shortcut(shortcut)
    full_52: "f32[602112]" = torch.ops.aten.full.default([602112], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    as_strided_91: "f32[8, 24, 56, 56]" = torch.ops.aten.as_strided.default(full_52, [8, 24, 56, 56], [75264, 3136, 56, 1], 0)
    copy_39: "f32[8, 24, 56, 56]" = torch.ops.aten.copy.default(as_strided_91, add_468);  add_468 = None
    as_strided_scatter_26: "f32[602112]" = torch.ops.aten.as_strided_scatter.default(full_52, copy_39, [8, 24, 56, 56], [75264, 3136, 56, 1], 0);  copy_39 = None
    as_strided_94: "f32[8, 24, 56, 56]" = torch.ops.aten.as_strided.default(as_strided_scatter_26, [8, 24, 56, 56], [75264, 3136, 56, 1], 0);  as_strided_scatter_26 = None
    new_empty_strided_13: "f32[8, 24, 56, 56]" = torch.ops.aten.new_empty_strided.default(as_strided_94, [8, 24, 56, 56], [75264, 3136, 56, 1])
    copy_40: "f32[8, 24, 56, 56]" = torch.ops.aten.copy.default(new_empty_strided_13, as_strided_94);  new_empty_strided_13 = as_strided_94 = None
    as_strided_96: "f32[8, 24, 56, 56]" = torch.ops.aten.as_strided.default(copy_40, [8, 24, 56, 56], [75264, 3136, 56, 1], 0)
    clone_13: "f32[8, 24, 56, 56]" = torch.ops.aten.clone.default(as_strided_96, memory_format = torch.contiguous_format)
    copy_41: "f32[8, 24, 56, 56]" = torch.ops.aten.copy.default(as_strided_96, clone_13);  as_strided_96 = None
    as_strided_scatter_27: "f32[8, 24, 56, 56]" = torch.ops.aten.as_strided_scatter.default(copy_40, copy_41, [8, 24, 56, 56], [75264, 3136, 56, 1], 0);  copy_40 = copy_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    slice_257: "f32[8, 12, 56, 56]" = torch.ops.aten.slice.Tensor(as_strided_scatter_27, 1, 12, 24)
    sum_152: "f32[12]" = torch.ops.aten.sum.dim_IntList(slice_257, [0, 2, 3])
    sub_336: "f32[8, 12, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_1090);  convolution_15 = unsqueeze_1090 = None
    mul_1164: "f32[8, 12, 56, 56]" = torch.ops.aten.mul.Tensor(slice_257, sub_336)
    sum_153: "f32[12]" = torch.ops.aten.sum.dim_IntList(mul_1164, [0, 2, 3]);  mul_1164 = None
    mul_1165: "f32[12]" = torch.ops.aten.mul.Tensor(sum_152, 3.985969387755102e-05)
    unsqueeze_1091: "f32[1, 12]" = torch.ops.aten.unsqueeze.default(mul_1165, 0);  mul_1165 = None
    unsqueeze_1092: "f32[1, 12, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1091, 2);  unsqueeze_1091 = None
    unsqueeze_1093: "f32[1, 12, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1092, 3);  unsqueeze_1092 = None
    mul_1166: "f32[12]" = torch.ops.aten.mul.Tensor(sum_153, 3.985969387755102e-05)
    mul_1167: "f32[12]" = torch.ops.aten.mul.Tensor(squeeze_46, squeeze_46)
    mul_1168: "f32[12]" = torch.ops.aten.mul.Tensor(mul_1166, mul_1167);  mul_1166 = mul_1167 = None
    unsqueeze_1094: "f32[1, 12]" = torch.ops.aten.unsqueeze.default(mul_1168, 0);  mul_1168 = None
    unsqueeze_1095: "f32[1, 12, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1094, 2);  unsqueeze_1094 = None
    unsqueeze_1096: "f32[1, 12, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1095, 3);  unsqueeze_1095 = None
    mul_1169: "f32[12]" = torch.ops.aten.mul.Tensor(squeeze_46, primals_51);  primals_51 = None
    unsqueeze_1097: "f32[1, 12]" = torch.ops.aten.unsqueeze.default(mul_1169, 0);  mul_1169 = None
    unsqueeze_1098: "f32[1, 12, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1097, 2);  unsqueeze_1097 = None
    unsqueeze_1099: "f32[1, 12, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1098, 3);  unsqueeze_1098 = None
    mul_1170: "f32[8, 12, 56, 56]" = torch.ops.aten.mul.Tensor(sub_336, unsqueeze_1096);  sub_336 = unsqueeze_1096 = None
    sub_338: "f32[8, 12, 56, 56]" = torch.ops.aten.sub.Tensor(slice_257, mul_1170);  slice_257 = mul_1170 = None
    sub_339: "f32[8, 12, 56, 56]" = torch.ops.aten.sub.Tensor(sub_338, unsqueeze_1093);  sub_338 = unsqueeze_1093 = None
    mul_1171: "f32[8, 12, 56, 56]" = torch.ops.aten.mul.Tensor(sub_339, unsqueeze_1099);  sub_339 = unsqueeze_1099 = None
    mul_1172: "f32[12]" = torch.ops.aten.mul.Tensor(sum_153, squeeze_46);  sum_153 = squeeze_46 = None
    convolution_backward_79 = torch.ops.aten.convolution_backward.default(mul_1171, add_76, primals_50, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 12, [True, True, False]);  mul_1171 = add_76 = primals_50 = None
    getitem_397: "f32[8, 12, 56, 56]" = convolution_backward_79[0]
    getitem_398: "f32[12, 1, 3, 3]" = convolution_backward_79[1];  convolution_backward_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    slice_258: "f32[8, 12, 56, 56]" = torch.ops.aten.slice.Tensor(as_strided_scatter_27, 1, 0, 12);  as_strided_scatter_27 = None
    add_469: "f32[8, 12, 56, 56]" = torch.ops.aten.add.Tensor(slice_258, getitem_397);  slice_258 = getitem_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    sum_154: "f32[12]" = torch.ops.aten.sum.dim_IntList(add_469, [0, 2, 3])
    sub_340: "f32[8, 12, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_1102);  convolution_14 = unsqueeze_1102 = None
    mul_1173: "f32[8, 12, 56, 56]" = torch.ops.aten.mul.Tensor(add_469, sub_340)
    sum_155: "f32[12]" = torch.ops.aten.sum.dim_IntList(mul_1173, [0, 2, 3]);  mul_1173 = None
    mul_1174: "f32[12]" = torch.ops.aten.mul.Tensor(sum_154, 3.985969387755102e-05)
    unsqueeze_1103: "f32[1, 12]" = torch.ops.aten.unsqueeze.default(mul_1174, 0);  mul_1174 = None
    unsqueeze_1104: "f32[1, 12, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1103, 2);  unsqueeze_1103 = None
    unsqueeze_1105: "f32[1, 12, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1104, 3);  unsqueeze_1104 = None
    mul_1175: "f32[12]" = torch.ops.aten.mul.Tensor(sum_155, 3.985969387755102e-05)
    mul_1176: "f32[12]" = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
    mul_1177: "f32[12]" = torch.ops.aten.mul.Tensor(mul_1175, mul_1176);  mul_1175 = mul_1176 = None
    unsqueeze_1106: "f32[1, 12]" = torch.ops.aten.unsqueeze.default(mul_1177, 0);  mul_1177 = None
    unsqueeze_1107: "f32[1, 12, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1106, 2);  unsqueeze_1106 = None
    unsqueeze_1108: "f32[1, 12, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1107, 3);  unsqueeze_1107 = None
    mul_1178: "f32[12]" = torch.ops.aten.mul.Tensor(squeeze_43, primals_48);  primals_48 = None
    unsqueeze_1109: "f32[1, 12]" = torch.ops.aten.unsqueeze.default(mul_1178, 0);  mul_1178 = None
    unsqueeze_1110: "f32[1, 12, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1109, 2);  unsqueeze_1109 = None
    unsqueeze_1111: "f32[1, 12, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1110, 3);  unsqueeze_1110 = None
    mul_1179: "f32[8, 12, 56, 56]" = torch.ops.aten.mul.Tensor(sub_340, unsqueeze_1108);  sub_340 = unsqueeze_1108 = None
    sub_342: "f32[8, 12, 56, 56]" = torch.ops.aten.sub.Tensor(add_469, mul_1179);  add_469 = mul_1179 = None
    sub_343: "f32[8, 12, 56, 56]" = torch.ops.aten.sub.Tensor(sub_342, unsqueeze_1105);  sub_342 = unsqueeze_1105 = None
    mul_1180: "f32[8, 12, 56, 56]" = torch.ops.aten.mul.Tensor(sub_343, unsqueeze_1111);  sub_343 = unsqueeze_1111 = None
    mul_1181: "f32[12]" = torch.ops.aten.mul.Tensor(sum_155, squeeze_43);  sum_155 = squeeze_43 = None
    convolution_backward_80 = torch.ops.aten.convolution_backward.default(mul_1180, slice_25, primals_47, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1180 = slice_25 = primals_47 = None
    getitem_400: "f32[8, 72, 56, 56]" = convolution_backward_80[0]
    getitem_401: "f32[12, 72, 1, 1]" = convolution_backward_80[1];  convolution_backward_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:63, code: return out[:, :self.out_chs, :, :]
    slice_scatter_87: "f32[8, 72, 56, 56]" = torch.ops.aten.slice_scatter.default(full_default_76, getitem_400, 3, 0, 9223372036854775807);  getitem_400 = None
    slice_scatter_88: "f32[8, 72, 56, 56]" = torch.ops.aten.slice_scatter.default(full_default_76, slice_scatter_87, 2, 0, 9223372036854775807);  slice_scatter_87 = None
    slice_scatter_89: "f32[8, 72, 56, 56]" = torch.ops.aten.slice_scatter.default(full_default_76, slice_scatter_88, 0, 0, 9223372036854775807);  full_default_76 = slice_scatter_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    slice_259: "f32[8, 36, 56, 56]" = torch.ops.aten.slice.Tensor(slice_scatter_89, 1, 0, 36)
    slice_260: "f32[8, 36, 56, 56]" = torch.ops.aten.slice.Tensor(slice_scatter_89, 1, 36, 72);  slice_scatter_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    where_42: "f32[8, 36, 56, 56]" = torch.ops.aten.where.self(le_35, full_default, slice_260);  le_35 = slice_260 = None
    sum_156: "f32[36]" = torch.ops.aten.sum.dim_IntList(where_42, [0, 2, 3])
    sub_344: "f32[8, 36, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_1114);  convolution_13 = unsqueeze_1114 = None
    mul_1182: "f32[8, 36, 56, 56]" = torch.ops.aten.mul.Tensor(where_42, sub_344)
    sum_157: "f32[36]" = torch.ops.aten.sum.dim_IntList(mul_1182, [0, 2, 3]);  mul_1182 = None
    mul_1183: "f32[36]" = torch.ops.aten.mul.Tensor(sum_156, 3.985969387755102e-05)
    unsqueeze_1115: "f32[1, 36]" = torch.ops.aten.unsqueeze.default(mul_1183, 0);  mul_1183 = None
    unsqueeze_1116: "f32[1, 36, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1115, 2);  unsqueeze_1115 = None
    unsqueeze_1117: "f32[1, 36, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1116, 3);  unsqueeze_1116 = None
    mul_1184: "f32[36]" = torch.ops.aten.mul.Tensor(sum_157, 3.985969387755102e-05)
    mul_1185: "f32[36]" = torch.ops.aten.mul.Tensor(squeeze_40, squeeze_40)
    mul_1186: "f32[36]" = torch.ops.aten.mul.Tensor(mul_1184, mul_1185);  mul_1184 = mul_1185 = None
    unsqueeze_1118: "f32[1, 36]" = torch.ops.aten.unsqueeze.default(mul_1186, 0);  mul_1186 = None
    unsqueeze_1119: "f32[1, 36, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1118, 2);  unsqueeze_1118 = None
    unsqueeze_1120: "f32[1, 36, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1119, 3);  unsqueeze_1119 = None
    mul_1187: "f32[36]" = torch.ops.aten.mul.Tensor(squeeze_40, primals_45);  primals_45 = None
    unsqueeze_1121: "f32[1, 36]" = torch.ops.aten.unsqueeze.default(mul_1187, 0);  mul_1187 = None
    unsqueeze_1122: "f32[1, 36, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1121, 2);  unsqueeze_1121 = None
    unsqueeze_1123: "f32[1, 36, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1122, 3);  unsqueeze_1122 = None
    mul_1188: "f32[8, 36, 56, 56]" = torch.ops.aten.mul.Tensor(sub_344, unsqueeze_1120);  sub_344 = unsqueeze_1120 = None
    sub_346: "f32[8, 36, 56, 56]" = torch.ops.aten.sub.Tensor(where_42, mul_1188);  where_42 = mul_1188 = None
    sub_347: "f32[8, 36, 56, 56]" = torch.ops.aten.sub.Tensor(sub_346, unsqueeze_1117);  sub_346 = unsqueeze_1117 = None
    mul_1189: "f32[8, 36, 56, 56]" = torch.ops.aten.mul.Tensor(sub_347, unsqueeze_1123);  sub_347 = unsqueeze_1123 = None
    mul_1190: "f32[36]" = torch.ops.aten.mul.Tensor(sum_157, squeeze_40);  sum_157 = squeeze_40 = None
    convolution_backward_81 = torch.ops.aten.convolution_backward.default(mul_1189, relu_5, primals_44, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 36, [True, True, False]);  mul_1189 = primals_44 = None
    getitem_403: "f32[8, 36, 56, 56]" = convolution_backward_81[0]
    getitem_404: "f32[36, 1, 3, 3]" = convolution_backward_81[1];  convolution_backward_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    add_470: "f32[8, 36, 56, 56]" = torch.ops.aten.add.Tensor(slice_259, getitem_403);  slice_259 = getitem_403 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    alias_151: "f32[8, 36, 56, 56]" = torch.ops.aten.alias.default(relu_5);  relu_5 = None
    alias_152: "f32[8, 36, 56, 56]" = torch.ops.aten.alias.default(alias_151);  alias_151 = None
    le_36: "b8[8, 36, 56, 56]" = torch.ops.aten.le.Scalar(alias_152, 0);  alias_152 = None
    where_43: "f32[8, 36, 56, 56]" = torch.ops.aten.where.self(le_36, full_default, add_470);  le_36 = add_470 = None
    sum_158: "f32[36]" = torch.ops.aten.sum.dim_IntList(where_43, [0, 2, 3])
    sub_348: "f32[8, 36, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_1126);  convolution_12 = unsqueeze_1126 = None
    mul_1191: "f32[8, 36, 56, 56]" = torch.ops.aten.mul.Tensor(where_43, sub_348)
    sum_159: "f32[36]" = torch.ops.aten.sum.dim_IntList(mul_1191, [0, 2, 3]);  mul_1191 = None
    mul_1192: "f32[36]" = torch.ops.aten.mul.Tensor(sum_158, 3.985969387755102e-05)
    unsqueeze_1127: "f32[1, 36]" = torch.ops.aten.unsqueeze.default(mul_1192, 0);  mul_1192 = None
    unsqueeze_1128: "f32[1, 36, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1127, 2);  unsqueeze_1127 = None
    unsqueeze_1129: "f32[1, 36, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1128, 3);  unsqueeze_1128 = None
    mul_1193: "f32[36]" = torch.ops.aten.mul.Tensor(sum_159, 3.985969387755102e-05)
    mul_1194: "f32[36]" = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
    mul_1195: "f32[36]" = torch.ops.aten.mul.Tensor(mul_1193, mul_1194);  mul_1193 = mul_1194 = None
    unsqueeze_1130: "f32[1, 36]" = torch.ops.aten.unsqueeze.default(mul_1195, 0);  mul_1195 = None
    unsqueeze_1131: "f32[1, 36, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1130, 2);  unsqueeze_1130 = None
    unsqueeze_1132: "f32[1, 36, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1131, 3);  unsqueeze_1131 = None
    mul_1196: "f32[36]" = torch.ops.aten.mul.Tensor(squeeze_37, primals_42);  primals_42 = None
    unsqueeze_1133: "f32[1, 36]" = torch.ops.aten.unsqueeze.default(mul_1196, 0);  mul_1196 = None
    unsqueeze_1134: "f32[1, 36, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1133, 2);  unsqueeze_1133 = None
    unsqueeze_1135: "f32[1, 36, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1134, 3);  unsqueeze_1134 = None
    mul_1197: "f32[8, 36, 56, 56]" = torch.ops.aten.mul.Tensor(sub_348, unsqueeze_1132);  sub_348 = unsqueeze_1132 = None
    sub_350: "f32[8, 36, 56, 56]" = torch.ops.aten.sub.Tensor(where_43, mul_1197);  where_43 = mul_1197 = None
    sub_351: "f32[8, 36, 56, 56]" = torch.ops.aten.sub.Tensor(sub_350, unsqueeze_1129);  sub_350 = unsqueeze_1129 = None
    mul_1198: "f32[8, 36, 56, 56]" = torch.ops.aten.mul.Tensor(sub_351, unsqueeze_1135);  sub_351 = unsqueeze_1135 = None
    mul_1199: "f32[36]" = torch.ops.aten.mul.Tensor(sum_159, squeeze_37);  sum_159 = squeeze_37 = None
    convolution_backward_82 = torch.ops.aten.convolution_backward.default(mul_1198, slice_22, primals_41, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1198 = slice_22 = primals_41 = None
    getitem_406: "f32[8, 24, 56, 56]" = convolution_backward_82[0]
    getitem_407: "f32[36, 24, 1, 1]" = convolution_backward_82[1];  convolution_backward_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    add_471: "f32[8, 24, 56, 56]" = torch.ops.aten.add.Tensor(clone_13, getitem_406);  clone_13 = getitem_406 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:182, code: x += self.shortcut(shortcut)
    copy_42: "f32[8, 24, 56, 56]" = torch.ops.aten.copy.default(as_strided_91, add_471);  as_strided_91 = add_471 = None
    as_strided_scatter_28: "f32[602112]" = torch.ops.aten.as_strided_scatter.default(full_52, copy_42, [8, 24, 56, 56], [75264, 3136, 56, 1], 0);  full_52 = copy_42 = None
    as_strided_101: "f32[8, 24, 56, 56]" = torch.ops.aten.as_strided.default(as_strided_scatter_28, [8, 24, 56, 56], [75264, 3136, 56, 1], 0);  as_strided_scatter_28 = None
    new_empty_strided_14: "f32[8, 24, 56, 56]" = torch.ops.aten.new_empty_strided.default(as_strided_101, [8, 24, 56, 56], [75264, 3136, 56, 1])
    copy_43: "f32[8, 24, 56, 56]" = torch.ops.aten.copy.default(new_empty_strided_14, as_strided_101);  new_empty_strided_14 = as_strided_101 = None
    as_strided_103: "f32[8, 24, 56, 56]" = torch.ops.aten.as_strided.default(copy_43, [8, 24, 56, 56], [75264, 3136, 56, 1], 0)
    clone_14: "f32[8, 24, 56, 56]" = torch.ops.aten.clone.default(as_strided_103, memory_format = torch.contiguous_format)
    copy_44: "f32[8, 24, 56, 56]" = torch.ops.aten.copy.default(as_strided_103, clone_14);  as_strided_103 = None
    as_strided_scatter_29: "f32[8, 24, 56, 56]" = torch.ops.aten.as_strided_scatter.default(copy_43, copy_44, [8, 24, 56, 56], [75264, 3136, 56, 1], 0);  copy_43 = copy_44 = None
    sum_160: "f32[24]" = torch.ops.aten.sum.dim_IntList(clone_14, [0, 2, 3])
    sub_352: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_1138);  convolution_11 = unsqueeze_1138 = None
    mul_1200: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(clone_14, sub_352)
    sum_161: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_1200, [0, 2, 3]);  mul_1200 = None
    mul_1201: "f32[24]" = torch.ops.aten.mul.Tensor(sum_160, 3.985969387755102e-05)
    unsqueeze_1139: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_1201, 0);  mul_1201 = None
    unsqueeze_1140: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1139, 2);  unsqueeze_1139 = None
    unsqueeze_1141: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1140, 3);  unsqueeze_1140 = None
    mul_1202: "f32[24]" = torch.ops.aten.mul.Tensor(sum_161, 3.985969387755102e-05)
    mul_1203: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_34, squeeze_34)
    mul_1204: "f32[24]" = torch.ops.aten.mul.Tensor(mul_1202, mul_1203);  mul_1202 = mul_1203 = None
    unsqueeze_1142: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_1204, 0);  mul_1204 = None
    unsqueeze_1143: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1142, 2);  unsqueeze_1142 = None
    unsqueeze_1144: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1143, 3);  unsqueeze_1143 = None
    mul_1205: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_34, primals_39);  primals_39 = None
    unsqueeze_1145: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_1205, 0);  mul_1205 = None
    unsqueeze_1146: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1145, 2);  unsqueeze_1145 = None
    unsqueeze_1147: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1146, 3);  unsqueeze_1146 = None
    mul_1206: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_352, unsqueeze_1144);  sub_352 = unsqueeze_1144 = None
    sub_354: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(clone_14, mul_1206);  clone_14 = mul_1206 = None
    sub_355: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(sub_354, unsqueeze_1141);  sub_354 = unsqueeze_1141 = None
    mul_1207: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_355, unsqueeze_1147);  sub_355 = unsqueeze_1147 = None
    mul_1208: "f32[24]" = torch.ops.aten.mul.Tensor(sum_161, squeeze_34);  sum_161 = squeeze_34 = None
    convolution_backward_83 = torch.ops.aten.convolution_backward.default(mul_1207, add_55, primals_38, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1207 = add_55 = primals_38 = None
    getitem_409: "f32[8, 16, 56, 56]" = convolution_backward_83[0]
    getitem_410: "f32[24, 16, 1, 1]" = convolution_backward_83[1];  convolution_backward_83 = None
    sum_162: "f32[16]" = torch.ops.aten.sum.dim_IntList(getitem_409, [0, 2, 3])
    sub_356: "f32[8, 16, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_1150);  convolution_10 = unsqueeze_1150 = None
    mul_1209: "f32[8, 16, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_409, sub_356)
    sum_163: "f32[16]" = torch.ops.aten.sum.dim_IntList(mul_1209, [0, 2, 3]);  mul_1209 = None
    mul_1210: "f32[16]" = torch.ops.aten.mul.Tensor(sum_162, 3.985969387755102e-05)
    unsqueeze_1151: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1210, 0);  mul_1210 = None
    unsqueeze_1152: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1151, 2);  unsqueeze_1151 = None
    unsqueeze_1153: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1152, 3);  unsqueeze_1152 = None
    mul_1211: "f32[16]" = torch.ops.aten.mul.Tensor(sum_163, 3.985969387755102e-05)
    mul_1212: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
    mul_1213: "f32[16]" = torch.ops.aten.mul.Tensor(mul_1211, mul_1212);  mul_1211 = mul_1212 = None
    unsqueeze_1154: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1213, 0);  mul_1213 = None
    unsqueeze_1155: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1154, 2);  unsqueeze_1154 = None
    unsqueeze_1156: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1155, 3);  unsqueeze_1155 = None
    mul_1214: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_31, primals_36);  primals_36 = None
    unsqueeze_1157: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1214, 0);  mul_1214 = None
    unsqueeze_1158: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1157, 2);  unsqueeze_1157 = None
    unsqueeze_1159: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1158, 3);  unsqueeze_1158 = None
    mul_1215: "f32[8, 16, 56, 56]" = torch.ops.aten.mul.Tensor(sub_356, unsqueeze_1156);  sub_356 = unsqueeze_1156 = None
    sub_358: "f32[8, 16, 56, 56]" = torch.ops.aten.sub.Tensor(getitem_409, mul_1215);  getitem_409 = mul_1215 = None
    sub_359: "f32[8, 16, 56, 56]" = torch.ops.aten.sub.Tensor(sub_358, unsqueeze_1153);  sub_358 = unsqueeze_1153 = None
    mul_1216: "f32[8, 16, 56, 56]" = torch.ops.aten.mul.Tensor(sub_359, unsqueeze_1159);  sub_359 = unsqueeze_1159 = None
    mul_1217: "f32[16]" = torch.ops.aten.mul.Tensor(sum_163, squeeze_31);  sum_163 = squeeze_31 = None
    convolution_backward_84 = torch.ops.aten.convolution_backward.default(mul_1216, slice_11, primals_35, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 16, [True, True, False]);  mul_1216 = primals_35 = None
    getitem_412: "f32[8, 16, 112, 112]" = convolution_backward_84[0]
    getitem_413: "f32[16, 1, 3, 3]" = convolution_backward_84[1];  convolution_backward_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    slice_263: "f32[8, 12, 56, 56]" = torch.ops.aten.slice.Tensor(as_strided_scatter_29, 1, 12, 24)
    sum_164: "f32[12]" = torch.ops.aten.sum.dim_IntList(slice_263, [0, 2, 3])
    sub_360: "f32[8, 12, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_1162);  convolution_9 = unsqueeze_1162 = None
    mul_1218: "f32[8, 12, 56, 56]" = torch.ops.aten.mul.Tensor(slice_263, sub_360)
    sum_165: "f32[12]" = torch.ops.aten.sum.dim_IntList(mul_1218, [0, 2, 3]);  mul_1218 = None
    mul_1219: "f32[12]" = torch.ops.aten.mul.Tensor(sum_164, 3.985969387755102e-05)
    unsqueeze_1163: "f32[1, 12]" = torch.ops.aten.unsqueeze.default(mul_1219, 0);  mul_1219 = None
    unsqueeze_1164: "f32[1, 12, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1163, 2);  unsqueeze_1163 = None
    unsqueeze_1165: "f32[1, 12, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1164, 3);  unsqueeze_1164 = None
    mul_1220: "f32[12]" = torch.ops.aten.mul.Tensor(sum_165, 3.985969387755102e-05)
    mul_1221: "f32[12]" = torch.ops.aten.mul.Tensor(squeeze_28, squeeze_28)
    mul_1222: "f32[12]" = torch.ops.aten.mul.Tensor(mul_1220, mul_1221);  mul_1220 = mul_1221 = None
    unsqueeze_1166: "f32[1, 12]" = torch.ops.aten.unsqueeze.default(mul_1222, 0);  mul_1222 = None
    unsqueeze_1167: "f32[1, 12, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1166, 2);  unsqueeze_1166 = None
    unsqueeze_1168: "f32[1, 12, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1167, 3);  unsqueeze_1167 = None
    mul_1223: "f32[12]" = torch.ops.aten.mul.Tensor(squeeze_28, primals_33);  primals_33 = None
    unsqueeze_1169: "f32[1, 12]" = torch.ops.aten.unsqueeze.default(mul_1223, 0);  mul_1223 = None
    unsqueeze_1170: "f32[1, 12, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1169, 2);  unsqueeze_1169 = None
    unsqueeze_1171: "f32[1, 12, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1170, 3);  unsqueeze_1170 = None
    mul_1224: "f32[8, 12, 56, 56]" = torch.ops.aten.mul.Tensor(sub_360, unsqueeze_1168);  sub_360 = unsqueeze_1168 = None
    sub_362: "f32[8, 12, 56, 56]" = torch.ops.aten.sub.Tensor(slice_263, mul_1224);  slice_263 = mul_1224 = None
    sub_363: "f32[8, 12, 56, 56]" = torch.ops.aten.sub.Tensor(sub_362, unsqueeze_1165);  sub_362 = unsqueeze_1165 = None
    mul_1225: "f32[8, 12, 56, 56]" = torch.ops.aten.mul.Tensor(sub_363, unsqueeze_1171);  sub_363 = unsqueeze_1171 = None
    mul_1226: "f32[12]" = torch.ops.aten.mul.Tensor(sum_165, squeeze_28);  sum_165 = squeeze_28 = None
    convolution_backward_85 = torch.ops.aten.convolution_backward.default(mul_1225, add_45, primals_32, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 12, [True, True, False]);  mul_1225 = add_45 = primals_32 = None
    getitem_415: "f32[8, 12, 56, 56]" = convolution_backward_85[0]
    getitem_416: "f32[12, 1, 3, 3]" = convolution_backward_85[1];  convolution_backward_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    slice_264: "f32[8, 12, 56, 56]" = torch.ops.aten.slice.Tensor(as_strided_scatter_29, 1, 0, 12);  as_strided_scatter_29 = None
    add_472: "f32[8, 12, 56, 56]" = torch.ops.aten.add.Tensor(slice_264, getitem_415);  slice_264 = getitem_415 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    sum_166: "f32[12]" = torch.ops.aten.sum.dim_IntList(add_472, [0, 2, 3])
    sub_364: "f32[8, 12, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_1174);  convolution_8 = unsqueeze_1174 = None
    mul_1227: "f32[8, 12, 56, 56]" = torch.ops.aten.mul.Tensor(add_472, sub_364)
    sum_167: "f32[12]" = torch.ops.aten.sum.dim_IntList(mul_1227, [0, 2, 3]);  mul_1227 = None
    mul_1228: "f32[12]" = torch.ops.aten.mul.Tensor(sum_166, 3.985969387755102e-05)
    unsqueeze_1175: "f32[1, 12]" = torch.ops.aten.unsqueeze.default(mul_1228, 0);  mul_1228 = None
    unsqueeze_1176: "f32[1, 12, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1175, 2);  unsqueeze_1175 = None
    unsqueeze_1177: "f32[1, 12, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1176, 3);  unsqueeze_1176 = None
    mul_1229: "f32[12]" = torch.ops.aten.mul.Tensor(sum_167, 3.985969387755102e-05)
    mul_1230: "f32[12]" = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
    mul_1231: "f32[12]" = torch.ops.aten.mul.Tensor(mul_1229, mul_1230);  mul_1229 = mul_1230 = None
    unsqueeze_1178: "f32[1, 12]" = torch.ops.aten.unsqueeze.default(mul_1231, 0);  mul_1231 = None
    unsqueeze_1179: "f32[1, 12, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1178, 2);  unsqueeze_1178 = None
    unsqueeze_1180: "f32[1, 12, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1179, 3);  unsqueeze_1179 = None
    mul_1232: "f32[12]" = torch.ops.aten.mul.Tensor(squeeze_25, primals_30);  primals_30 = None
    unsqueeze_1181: "f32[1, 12]" = torch.ops.aten.unsqueeze.default(mul_1232, 0);  mul_1232 = None
    unsqueeze_1182: "f32[1, 12, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1181, 2);  unsqueeze_1181 = None
    unsqueeze_1183: "f32[1, 12, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1182, 3);  unsqueeze_1182 = None
    mul_1233: "f32[8, 12, 56, 56]" = torch.ops.aten.mul.Tensor(sub_364, unsqueeze_1180);  sub_364 = unsqueeze_1180 = None
    sub_366: "f32[8, 12, 56, 56]" = torch.ops.aten.sub.Tensor(add_472, mul_1233);  add_472 = mul_1233 = None
    sub_367: "f32[8, 12, 56, 56]" = torch.ops.aten.sub.Tensor(sub_366, unsqueeze_1177);  sub_366 = unsqueeze_1177 = None
    mul_1234: "f32[8, 12, 56, 56]" = torch.ops.aten.mul.Tensor(sub_367, unsqueeze_1183);  sub_367 = unsqueeze_1183 = None
    mul_1235: "f32[12]" = torch.ops.aten.mul.Tensor(sum_167, squeeze_25);  sum_167 = squeeze_25 = None
    convolution_backward_86 = torch.ops.aten.convolution_backward.default(mul_1234, add_40, primals_29, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1234 = add_40 = primals_29 = None
    getitem_418: "f32[8, 48, 56, 56]" = convolution_backward_86[0]
    getitem_419: "f32[12, 48, 1, 1]" = convolution_backward_86[1];  convolution_backward_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:173, code: x = self.bn_dw(x)
    sum_168: "f32[48]" = torch.ops.aten.sum.dim_IntList(getitem_418, [0, 2, 3])
    sub_368: "f32[8, 48, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_1186);  convolution_7 = unsqueeze_1186 = None
    mul_1236: "f32[8, 48, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_418, sub_368)
    sum_169: "f32[48]" = torch.ops.aten.sum.dim_IntList(mul_1236, [0, 2, 3]);  mul_1236 = None
    mul_1237: "f32[48]" = torch.ops.aten.mul.Tensor(sum_168, 3.985969387755102e-05)
    unsqueeze_1187: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_1237, 0);  mul_1237 = None
    unsqueeze_1188: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1187, 2);  unsqueeze_1187 = None
    unsqueeze_1189: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1188, 3);  unsqueeze_1188 = None
    mul_1238: "f32[48]" = torch.ops.aten.mul.Tensor(sum_169, 3.985969387755102e-05)
    mul_1239: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_22, squeeze_22)
    mul_1240: "f32[48]" = torch.ops.aten.mul.Tensor(mul_1238, mul_1239);  mul_1238 = mul_1239 = None
    unsqueeze_1190: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_1240, 0);  mul_1240 = None
    unsqueeze_1191: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1190, 2);  unsqueeze_1190 = None
    unsqueeze_1192: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1191, 3);  unsqueeze_1191 = None
    mul_1241: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_22, primals_27);  primals_27 = None
    unsqueeze_1193: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_1241, 0);  mul_1241 = None
    unsqueeze_1194: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1193, 2);  unsqueeze_1193 = None
    unsqueeze_1195: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1194, 3);  unsqueeze_1194 = None
    mul_1242: "f32[8, 48, 56, 56]" = torch.ops.aten.mul.Tensor(sub_368, unsqueeze_1192);  sub_368 = unsqueeze_1192 = None
    sub_370: "f32[8, 48, 56, 56]" = torch.ops.aten.sub.Tensor(getitem_418, mul_1242);  getitem_418 = mul_1242 = None
    sub_371: "f32[8, 48, 56, 56]" = torch.ops.aten.sub.Tensor(sub_370, unsqueeze_1189);  sub_370 = unsqueeze_1189 = None
    mul_1243: "f32[8, 48, 56, 56]" = torch.ops.aten.mul.Tensor(sub_371, unsqueeze_1195);  sub_371 = unsqueeze_1195 = None
    mul_1244: "f32[48]" = torch.ops.aten.mul.Tensor(sum_169, squeeze_22);  sum_169 = squeeze_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:172, code: x = self.conv_dw(x)
    convolution_backward_87 = torch.ops.aten.convolution_backward.default(mul_1243, slice_14, primals_26, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 48, [True, True, False]);  mul_1243 = slice_14 = primals_26 = None
    getitem_421: "f32[8, 48, 112, 112]" = convolution_backward_87[0]
    getitem_422: "f32[48, 1, 3, 3]" = convolution_backward_87[1];  convolution_backward_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:63, code: return out[:, :self.out_chs, :, :]
    full_default_86: "f32[8, 48, 112, 112]" = torch.ops.aten.full.default([8, 48, 112, 112], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_90: "f32[8, 48, 112, 112]" = torch.ops.aten.slice_scatter.default(full_default_86, getitem_421, 3, 0, 9223372036854775807);  getitem_421 = None
    slice_scatter_91: "f32[8, 48, 112, 112]" = torch.ops.aten.slice_scatter.default(full_default_86, slice_scatter_90, 2, 0, 9223372036854775807);  slice_scatter_90 = None
    slice_scatter_92: "f32[8, 48, 112, 112]" = torch.ops.aten.slice_scatter.default(full_default_86, slice_scatter_91, 0, 0, 9223372036854775807);  full_default_86 = slice_scatter_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    slice_265: "f32[8, 24, 112, 112]" = torch.ops.aten.slice.Tensor(slice_scatter_92, 1, 0, 24)
    slice_266: "f32[8, 24, 112, 112]" = torch.ops.aten.slice.Tensor(slice_scatter_92, 1, 24, 48);  slice_scatter_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    where_44: "f32[8, 24, 112, 112]" = torch.ops.aten.where.self(le_37, full_default, slice_266);  le_37 = slice_266 = None
    sum_170: "f32[24]" = torch.ops.aten.sum.dim_IntList(where_44, [0, 2, 3])
    sub_372: "f32[8, 24, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_1198);  convolution_6 = unsqueeze_1198 = None
    mul_1245: "f32[8, 24, 112, 112]" = torch.ops.aten.mul.Tensor(where_44, sub_372)
    sum_171: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_1245, [0, 2, 3]);  mul_1245 = None
    mul_1246: "f32[24]" = torch.ops.aten.mul.Tensor(sum_170, 9.964923469387754e-06)
    unsqueeze_1199: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_1246, 0);  mul_1246 = None
    unsqueeze_1200: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1199, 2);  unsqueeze_1199 = None
    unsqueeze_1201: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1200, 3);  unsqueeze_1200 = None
    mul_1247: "f32[24]" = torch.ops.aten.mul.Tensor(sum_171, 9.964923469387754e-06)
    mul_1248: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_19, squeeze_19)
    mul_1249: "f32[24]" = torch.ops.aten.mul.Tensor(mul_1247, mul_1248);  mul_1247 = mul_1248 = None
    unsqueeze_1202: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_1249, 0);  mul_1249 = None
    unsqueeze_1203: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1202, 2);  unsqueeze_1202 = None
    unsqueeze_1204: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1203, 3);  unsqueeze_1203 = None
    mul_1250: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_19, primals_24);  primals_24 = None
    unsqueeze_1205: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_1250, 0);  mul_1250 = None
    unsqueeze_1206: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1205, 2);  unsqueeze_1205 = None
    unsqueeze_1207: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1206, 3);  unsqueeze_1206 = None
    mul_1251: "f32[8, 24, 112, 112]" = torch.ops.aten.mul.Tensor(sub_372, unsqueeze_1204);  sub_372 = unsqueeze_1204 = None
    sub_374: "f32[8, 24, 112, 112]" = torch.ops.aten.sub.Tensor(where_44, mul_1251);  where_44 = mul_1251 = None
    sub_375: "f32[8, 24, 112, 112]" = torch.ops.aten.sub.Tensor(sub_374, unsqueeze_1201);  sub_374 = unsqueeze_1201 = None
    mul_1252: "f32[8, 24, 112, 112]" = torch.ops.aten.mul.Tensor(sub_375, unsqueeze_1207);  sub_375 = unsqueeze_1207 = None
    mul_1253: "f32[24]" = torch.ops.aten.mul.Tensor(sum_171, squeeze_19);  sum_171 = squeeze_19 = None
    convolution_backward_88 = torch.ops.aten.convolution_backward.default(mul_1252, relu_3, primals_23, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 24, [True, True, False]);  mul_1252 = primals_23 = None
    getitem_424: "f32[8, 24, 112, 112]" = convolution_backward_88[0]
    getitem_425: "f32[24, 1, 3, 3]" = convolution_backward_88[1];  convolution_backward_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    add_473: "f32[8, 24, 112, 112]" = torch.ops.aten.add.Tensor(slice_265, getitem_424);  slice_265 = getitem_424 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    alias_157: "f32[8, 24, 112, 112]" = torch.ops.aten.alias.default(relu_3);  relu_3 = None
    alias_158: "f32[8, 24, 112, 112]" = torch.ops.aten.alias.default(alias_157);  alias_157 = None
    le_38: "b8[8, 24, 112, 112]" = torch.ops.aten.le.Scalar(alias_158, 0);  alias_158 = None
    where_45: "f32[8, 24, 112, 112]" = torch.ops.aten.where.self(le_38, full_default, add_473);  le_38 = add_473 = None
    sum_172: "f32[24]" = torch.ops.aten.sum.dim_IntList(where_45, [0, 2, 3])
    sub_376: "f32[8, 24, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_1210);  convolution_5 = unsqueeze_1210 = None
    mul_1254: "f32[8, 24, 112, 112]" = torch.ops.aten.mul.Tensor(where_45, sub_376)
    sum_173: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_1254, [0, 2, 3]);  mul_1254 = None
    mul_1255: "f32[24]" = torch.ops.aten.mul.Tensor(sum_172, 9.964923469387754e-06)
    unsqueeze_1211: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_1255, 0);  mul_1255 = None
    unsqueeze_1212: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1211, 2);  unsqueeze_1211 = None
    unsqueeze_1213: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1212, 3);  unsqueeze_1212 = None
    mul_1256: "f32[24]" = torch.ops.aten.mul.Tensor(sum_173, 9.964923469387754e-06)
    mul_1257: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_16, squeeze_16)
    mul_1258: "f32[24]" = torch.ops.aten.mul.Tensor(mul_1256, mul_1257);  mul_1256 = mul_1257 = None
    unsqueeze_1214: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_1258, 0);  mul_1258 = None
    unsqueeze_1215: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1214, 2);  unsqueeze_1214 = None
    unsqueeze_1216: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1215, 3);  unsqueeze_1215 = None
    mul_1259: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_16, primals_21);  primals_21 = None
    unsqueeze_1217: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_1259, 0);  mul_1259 = None
    unsqueeze_1218: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1217, 2);  unsqueeze_1217 = None
    unsqueeze_1219: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1218, 3);  unsqueeze_1218 = None
    mul_1260: "f32[8, 24, 112, 112]" = torch.ops.aten.mul.Tensor(sub_376, unsqueeze_1216);  sub_376 = unsqueeze_1216 = None
    sub_378: "f32[8, 24, 112, 112]" = torch.ops.aten.sub.Tensor(where_45, mul_1260);  where_45 = mul_1260 = None
    sub_379: "f32[8, 24, 112, 112]" = torch.ops.aten.sub.Tensor(sub_378, unsqueeze_1213);  sub_378 = unsqueeze_1213 = None
    mul_1261: "f32[8, 24, 112, 112]" = torch.ops.aten.mul.Tensor(sub_379, unsqueeze_1219);  sub_379 = unsqueeze_1219 = None
    mul_1262: "f32[24]" = torch.ops.aten.mul.Tensor(sum_173, squeeze_16);  sum_173 = squeeze_16 = None
    convolution_backward_89 = torch.ops.aten.convolution_backward.default(mul_1261, slice_11, primals_20, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1261 = slice_11 = primals_20 = None
    getitem_427: "f32[8, 16, 112, 112]" = convolution_backward_89[0]
    getitem_428: "f32[24, 16, 1, 1]" = convolution_backward_89[1];  convolution_backward_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    add_474: "f32[8, 16, 112, 112]" = torch.ops.aten.add.Tensor(getitem_412, getitem_427);  getitem_412 = getitem_427 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:182, code: x += self.shortcut(shortcut)
    full_60: "f32[1605632]" = torch.ops.aten.full.default([1605632], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    as_strided_105: "f32[8, 16, 112, 112]" = torch.ops.aten.as_strided.default(full_60, [8, 16, 112, 112], [200704, 12544, 112, 1], 0)
    copy_45: "f32[8, 16, 112, 112]" = torch.ops.aten.copy.default(as_strided_105, add_474);  as_strided_105 = add_474 = None
    as_strided_scatter_30: "f32[1605632]" = torch.ops.aten.as_strided_scatter.default(full_60, copy_45, [8, 16, 112, 112], [200704, 12544, 112, 1], 0);  full_60 = copy_45 = None
    as_strided_108: "f32[8, 16, 112, 112]" = torch.ops.aten.as_strided.default(as_strided_scatter_30, [8, 16, 112, 112], [200704, 12544, 112, 1], 0);  as_strided_scatter_30 = None
    new_empty_strided_15: "f32[8, 16, 112, 112]" = torch.ops.aten.new_empty_strided.default(as_strided_108, [8, 16, 112, 112], [200704, 12544, 112, 1])
    copy_46: "f32[8, 16, 112, 112]" = torch.ops.aten.copy.default(new_empty_strided_15, as_strided_108);  new_empty_strided_15 = as_strided_108 = None
    as_strided_110: "f32[8, 16, 112, 112]" = torch.ops.aten.as_strided.default(copy_46, [8, 16, 112, 112], [200704, 12544, 112, 1], 0)
    clone_15: "f32[8, 16, 112, 112]" = torch.ops.aten.clone.default(as_strided_110, memory_format = torch.contiguous_format)
    copy_47: "f32[8, 16, 112, 112]" = torch.ops.aten.copy.default(as_strided_110, clone_15);  as_strided_110 = None
    as_strided_scatter_31: "f32[8, 16, 112, 112]" = torch.ops.aten.as_strided_scatter.default(copy_46, copy_47, [8, 16, 112, 112], [200704, 12544, 112, 1], 0);  copy_46 = copy_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    slice_269: "f32[8, 8, 112, 112]" = torch.ops.aten.slice.Tensor(as_strided_scatter_31, 1, 8, 16)
    sum_174: "f32[8]" = torch.ops.aten.sum.dim_IntList(slice_269, [0, 2, 3])
    sub_380: "f32[8, 8, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_1222);  convolution_4 = unsqueeze_1222 = None
    mul_1263: "f32[8, 8, 112, 112]" = torch.ops.aten.mul.Tensor(slice_269, sub_380)
    sum_175: "f32[8]" = torch.ops.aten.sum.dim_IntList(mul_1263, [0, 2, 3]);  mul_1263 = None
    mul_1264: "f32[8]" = torch.ops.aten.mul.Tensor(sum_174, 9.964923469387754e-06)
    unsqueeze_1223: "f32[1, 8]" = torch.ops.aten.unsqueeze.default(mul_1264, 0);  mul_1264 = None
    unsqueeze_1224: "f32[1, 8, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1223, 2);  unsqueeze_1223 = None
    unsqueeze_1225: "f32[1, 8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1224, 3);  unsqueeze_1224 = None
    mul_1265: "f32[8]" = torch.ops.aten.mul.Tensor(sum_175, 9.964923469387754e-06)
    mul_1266: "f32[8]" = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
    mul_1267: "f32[8]" = torch.ops.aten.mul.Tensor(mul_1265, mul_1266);  mul_1265 = mul_1266 = None
    unsqueeze_1226: "f32[1, 8]" = torch.ops.aten.unsqueeze.default(mul_1267, 0);  mul_1267 = None
    unsqueeze_1227: "f32[1, 8, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1226, 2);  unsqueeze_1226 = None
    unsqueeze_1228: "f32[1, 8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1227, 3);  unsqueeze_1227 = None
    mul_1268: "f32[8]" = torch.ops.aten.mul.Tensor(squeeze_13, primals_18);  primals_18 = None
    unsqueeze_1229: "f32[1, 8]" = torch.ops.aten.unsqueeze.default(mul_1268, 0);  mul_1268 = None
    unsqueeze_1230: "f32[1, 8, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1229, 2);  unsqueeze_1229 = None
    unsqueeze_1231: "f32[1, 8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1230, 3);  unsqueeze_1230 = None
    mul_1269: "f32[8, 8, 112, 112]" = torch.ops.aten.mul.Tensor(sub_380, unsqueeze_1228);  sub_380 = unsqueeze_1228 = None
    sub_382: "f32[8, 8, 112, 112]" = torch.ops.aten.sub.Tensor(slice_269, mul_1269);  slice_269 = mul_1269 = None
    sub_383: "f32[8, 8, 112, 112]" = torch.ops.aten.sub.Tensor(sub_382, unsqueeze_1225);  sub_382 = unsqueeze_1225 = None
    mul_1270: "f32[8, 8, 112, 112]" = torch.ops.aten.mul.Tensor(sub_383, unsqueeze_1231);  sub_383 = unsqueeze_1231 = None
    mul_1271: "f32[8]" = torch.ops.aten.mul.Tensor(sum_175, squeeze_13);  sum_175 = squeeze_13 = None
    convolution_backward_90 = torch.ops.aten.convolution_backward.default(mul_1270, add_19, primals_17, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_1270 = add_19 = primals_17 = None
    getitem_430: "f32[8, 8, 112, 112]" = convolution_backward_90[0]
    getitem_431: "f32[8, 1, 3, 3]" = convolution_backward_90[1];  convolution_backward_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    slice_270: "f32[8, 8, 112, 112]" = torch.ops.aten.slice.Tensor(as_strided_scatter_31, 1, 0, 8);  as_strided_scatter_31 = None
    add_475: "f32[8, 8, 112, 112]" = torch.ops.aten.add.Tensor(slice_270, getitem_430);  slice_270 = getitem_430 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    sum_176: "f32[8]" = torch.ops.aten.sum.dim_IntList(add_475, [0, 2, 3])
    sub_384: "f32[8, 8, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_1234);  convolution_3 = unsqueeze_1234 = None
    mul_1272: "f32[8, 8, 112, 112]" = torch.ops.aten.mul.Tensor(add_475, sub_384)
    sum_177: "f32[8]" = torch.ops.aten.sum.dim_IntList(mul_1272, [0, 2, 3]);  mul_1272 = None
    mul_1273: "f32[8]" = torch.ops.aten.mul.Tensor(sum_176, 9.964923469387754e-06)
    unsqueeze_1235: "f32[1, 8]" = torch.ops.aten.unsqueeze.default(mul_1273, 0);  mul_1273 = None
    unsqueeze_1236: "f32[1, 8, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1235, 2);  unsqueeze_1235 = None
    unsqueeze_1237: "f32[1, 8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1236, 3);  unsqueeze_1236 = None
    mul_1274: "f32[8]" = torch.ops.aten.mul.Tensor(sum_177, 9.964923469387754e-06)
    mul_1275: "f32[8]" = torch.ops.aten.mul.Tensor(squeeze_10, squeeze_10)
    mul_1276: "f32[8]" = torch.ops.aten.mul.Tensor(mul_1274, mul_1275);  mul_1274 = mul_1275 = None
    unsqueeze_1238: "f32[1, 8]" = torch.ops.aten.unsqueeze.default(mul_1276, 0);  mul_1276 = None
    unsqueeze_1239: "f32[1, 8, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1238, 2);  unsqueeze_1238 = None
    unsqueeze_1240: "f32[1, 8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1239, 3);  unsqueeze_1239 = None
    mul_1277: "f32[8]" = torch.ops.aten.mul.Tensor(squeeze_10, primals_15);  primals_15 = None
    unsqueeze_1241: "f32[1, 8]" = torch.ops.aten.unsqueeze.default(mul_1277, 0);  mul_1277 = None
    unsqueeze_1242: "f32[1, 8, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1241, 2);  unsqueeze_1241 = None
    unsqueeze_1243: "f32[1, 8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1242, 3);  unsqueeze_1242 = None
    mul_1278: "f32[8, 8, 112, 112]" = torch.ops.aten.mul.Tensor(sub_384, unsqueeze_1240);  sub_384 = unsqueeze_1240 = None
    sub_386: "f32[8, 8, 112, 112]" = torch.ops.aten.sub.Tensor(add_475, mul_1278);  add_475 = mul_1278 = None
    sub_387: "f32[8, 8, 112, 112]" = torch.ops.aten.sub.Tensor(sub_386, unsqueeze_1237);  sub_386 = unsqueeze_1237 = None
    mul_1279: "f32[8, 8, 112, 112]" = torch.ops.aten.mul.Tensor(sub_387, unsqueeze_1243);  sub_387 = unsqueeze_1243 = None
    mul_1280: "f32[8]" = torch.ops.aten.mul.Tensor(sum_177, squeeze_10);  sum_177 = squeeze_10 = None
    convolution_backward_91 = torch.ops.aten.convolution_backward.default(mul_1279, slice_3, primals_14, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1279 = slice_3 = primals_14 = None
    getitem_433: "f32[8, 16, 112, 112]" = convolution_backward_91[0]
    getitem_434: "f32[8, 16, 1, 1]" = convolution_backward_91[1];  convolution_backward_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:63, code: return out[:, :self.out_chs, :, :]
    full_default_91: "f32[8, 16, 112, 112]" = torch.ops.aten.full.default([8, 16, 112, 112], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_93: "f32[8, 16, 112, 112]" = torch.ops.aten.slice_scatter.default(full_default_91, getitem_433, 3, 0, 9223372036854775807);  getitem_433 = None
    slice_scatter_94: "f32[8, 16, 112, 112]" = torch.ops.aten.slice_scatter.default(full_default_91, slice_scatter_93, 2, 0, 9223372036854775807);  slice_scatter_93 = None
    slice_scatter_95: "f32[8, 16, 112, 112]" = torch.ops.aten.slice_scatter.default(full_default_91, slice_scatter_94, 0, 0, 9223372036854775807);  full_default_91 = slice_scatter_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    slice_271: "f32[8, 8, 112, 112]" = torch.ops.aten.slice.Tensor(slice_scatter_95, 1, 0, 8)
    slice_272: "f32[8, 8, 112, 112]" = torch.ops.aten.slice.Tensor(slice_scatter_95, 1, 8, 16);  slice_scatter_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    where_46: "f32[8, 8, 112, 112]" = torch.ops.aten.where.self(le_39, full_default, slice_272);  le_39 = slice_272 = None
    sum_178: "f32[8]" = torch.ops.aten.sum.dim_IntList(where_46, [0, 2, 3])
    sub_388: "f32[8, 8, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_1246);  convolution_2 = unsqueeze_1246 = None
    mul_1281: "f32[8, 8, 112, 112]" = torch.ops.aten.mul.Tensor(where_46, sub_388)
    sum_179: "f32[8]" = torch.ops.aten.sum.dim_IntList(mul_1281, [0, 2, 3]);  mul_1281 = None
    mul_1282: "f32[8]" = torch.ops.aten.mul.Tensor(sum_178, 9.964923469387754e-06)
    unsqueeze_1247: "f32[1, 8]" = torch.ops.aten.unsqueeze.default(mul_1282, 0);  mul_1282 = None
    unsqueeze_1248: "f32[1, 8, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1247, 2);  unsqueeze_1247 = None
    unsqueeze_1249: "f32[1, 8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1248, 3);  unsqueeze_1248 = None
    mul_1283: "f32[8]" = torch.ops.aten.mul.Tensor(sum_179, 9.964923469387754e-06)
    mul_1284: "f32[8]" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
    mul_1285: "f32[8]" = torch.ops.aten.mul.Tensor(mul_1283, mul_1284);  mul_1283 = mul_1284 = None
    unsqueeze_1250: "f32[1, 8]" = torch.ops.aten.unsqueeze.default(mul_1285, 0);  mul_1285 = None
    unsqueeze_1251: "f32[1, 8, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1250, 2);  unsqueeze_1250 = None
    unsqueeze_1252: "f32[1, 8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1251, 3);  unsqueeze_1251 = None
    mul_1286: "f32[8]" = torch.ops.aten.mul.Tensor(squeeze_7, primals_12);  primals_12 = None
    unsqueeze_1253: "f32[1, 8]" = torch.ops.aten.unsqueeze.default(mul_1286, 0);  mul_1286 = None
    unsqueeze_1254: "f32[1, 8, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1253, 2);  unsqueeze_1253 = None
    unsqueeze_1255: "f32[1, 8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1254, 3);  unsqueeze_1254 = None
    mul_1287: "f32[8, 8, 112, 112]" = torch.ops.aten.mul.Tensor(sub_388, unsqueeze_1252);  sub_388 = unsqueeze_1252 = None
    sub_390: "f32[8, 8, 112, 112]" = torch.ops.aten.sub.Tensor(where_46, mul_1287);  where_46 = mul_1287 = None
    sub_391: "f32[8, 8, 112, 112]" = torch.ops.aten.sub.Tensor(sub_390, unsqueeze_1249);  sub_390 = unsqueeze_1249 = None
    mul_1288: "f32[8, 8, 112, 112]" = torch.ops.aten.mul.Tensor(sub_391, unsqueeze_1255);  sub_391 = unsqueeze_1255 = None
    mul_1289: "f32[8]" = torch.ops.aten.mul.Tensor(sum_179, squeeze_7);  sum_179 = squeeze_7 = None
    convolution_backward_92 = torch.ops.aten.convolution_backward.default(mul_1288, relu_1, primals_11, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_1288 = primals_11 = None
    getitem_436: "f32[8, 8, 112, 112]" = convolution_backward_92[0]
    getitem_437: "f32[8, 1, 3, 3]" = convolution_backward_92[1];  convolution_backward_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    add_476: "f32[8, 8, 112, 112]" = torch.ops.aten.add.Tensor(slice_271, getitem_436);  slice_271 = getitem_436 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    alias_163: "f32[8, 8, 112, 112]" = torch.ops.aten.alias.default(relu_1);  relu_1 = None
    alias_164: "f32[8, 8, 112, 112]" = torch.ops.aten.alias.default(alias_163);  alias_163 = None
    le_40: "b8[8, 8, 112, 112]" = torch.ops.aten.le.Scalar(alias_164, 0);  alias_164 = None
    where_47: "f32[8, 8, 112, 112]" = torch.ops.aten.where.self(le_40, full_default, add_476);  le_40 = add_476 = None
    sum_180: "f32[8]" = torch.ops.aten.sum.dim_IntList(where_47, [0, 2, 3])
    sub_392: "f32[8, 8, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_1258);  convolution_1 = unsqueeze_1258 = None
    mul_1290: "f32[8, 8, 112, 112]" = torch.ops.aten.mul.Tensor(where_47, sub_392)
    sum_181: "f32[8]" = torch.ops.aten.sum.dim_IntList(mul_1290, [0, 2, 3]);  mul_1290 = None
    mul_1291: "f32[8]" = torch.ops.aten.mul.Tensor(sum_180, 9.964923469387754e-06)
    unsqueeze_1259: "f32[1, 8]" = torch.ops.aten.unsqueeze.default(mul_1291, 0);  mul_1291 = None
    unsqueeze_1260: "f32[1, 8, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1259, 2);  unsqueeze_1259 = None
    unsqueeze_1261: "f32[1, 8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1260, 3);  unsqueeze_1260 = None
    mul_1292: "f32[8]" = torch.ops.aten.mul.Tensor(sum_181, 9.964923469387754e-06)
    mul_1293: "f32[8]" = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
    mul_1294: "f32[8]" = torch.ops.aten.mul.Tensor(mul_1292, mul_1293);  mul_1292 = mul_1293 = None
    unsqueeze_1262: "f32[1, 8]" = torch.ops.aten.unsqueeze.default(mul_1294, 0);  mul_1294 = None
    unsqueeze_1263: "f32[1, 8, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1262, 2);  unsqueeze_1262 = None
    unsqueeze_1264: "f32[1, 8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1263, 3);  unsqueeze_1263 = None
    mul_1295: "f32[8]" = torch.ops.aten.mul.Tensor(squeeze_4, primals_9);  primals_9 = None
    unsqueeze_1265: "f32[1, 8]" = torch.ops.aten.unsqueeze.default(mul_1295, 0);  mul_1295 = None
    unsqueeze_1266: "f32[1, 8, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1265, 2);  unsqueeze_1265 = None
    unsqueeze_1267: "f32[1, 8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1266, 3);  unsqueeze_1266 = None
    mul_1296: "f32[8, 8, 112, 112]" = torch.ops.aten.mul.Tensor(sub_392, unsqueeze_1264);  sub_392 = unsqueeze_1264 = None
    sub_394: "f32[8, 8, 112, 112]" = torch.ops.aten.sub.Tensor(where_47, mul_1296);  where_47 = mul_1296 = None
    sub_395: "f32[8, 8, 112, 112]" = torch.ops.aten.sub.Tensor(sub_394, unsqueeze_1261);  sub_394 = unsqueeze_1261 = None
    mul_1297: "f32[8, 8, 112, 112]" = torch.ops.aten.mul.Tensor(sub_395, unsqueeze_1267);  sub_395 = unsqueeze_1267 = None
    mul_1298: "f32[8]" = torch.ops.aten.mul.Tensor(sum_181, squeeze_4);  sum_181 = squeeze_4 = None
    convolution_backward_93 = torch.ops.aten.convolution_backward.default(mul_1297, relu, primals_8, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1297 = primals_8 = None
    getitem_439: "f32[8, 16, 112, 112]" = convolution_backward_93[0]
    getitem_440: "f32[8, 16, 1, 1]" = convolution_backward_93[1];  convolution_backward_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    add_477: "f32[8, 16, 112, 112]" = torch.ops.aten.add.Tensor(clone_15, getitem_439);  clone_15 = getitem_439 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:284, code: x = self.act1(x)
    alias_166: "f32[8, 16, 112, 112]" = torch.ops.aten.alias.default(relu);  relu = None
    alias_167: "f32[8, 16, 112, 112]" = torch.ops.aten.alias.default(alias_166);  alias_166 = None
    le_41: "b8[8, 16, 112, 112]" = torch.ops.aten.le.Scalar(alias_167, 0);  alias_167 = None
    where_48: "f32[8, 16, 112, 112]" = torch.ops.aten.where.self(le_41, full_default, add_477);  le_41 = full_default = add_477 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:283, code: x = self.bn1(x)
    sum_182: "f32[16]" = torch.ops.aten.sum.dim_IntList(where_48, [0, 2, 3])
    sub_396: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_1270);  convolution = unsqueeze_1270 = None
    mul_1299: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(where_48, sub_396)
    sum_183: "f32[16]" = torch.ops.aten.sum.dim_IntList(mul_1299, [0, 2, 3]);  mul_1299 = None
    mul_1300: "f32[16]" = torch.ops.aten.mul.Tensor(sum_182, 9.964923469387754e-06)
    unsqueeze_1271: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1300, 0);  mul_1300 = None
    unsqueeze_1272: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1271, 2);  unsqueeze_1271 = None
    unsqueeze_1273: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1272, 3);  unsqueeze_1272 = None
    mul_1301: "f32[16]" = torch.ops.aten.mul.Tensor(sum_183, 9.964923469387754e-06)
    mul_1302: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
    mul_1303: "f32[16]" = torch.ops.aten.mul.Tensor(mul_1301, mul_1302);  mul_1301 = mul_1302 = None
    unsqueeze_1274: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1303, 0);  mul_1303 = None
    unsqueeze_1275: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1274, 2);  unsqueeze_1274 = None
    unsqueeze_1276: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1275, 3);  unsqueeze_1275 = None
    mul_1304: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_1, primals_6);  primals_6 = None
    unsqueeze_1277: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1304, 0);  mul_1304 = None
    unsqueeze_1278: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1277, 2);  unsqueeze_1277 = None
    unsqueeze_1279: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1278, 3);  unsqueeze_1278 = None
    mul_1305: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_396, unsqueeze_1276);  sub_396 = unsqueeze_1276 = None
    sub_398: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(where_48, mul_1305);  where_48 = mul_1305 = None
    sub_399: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(sub_398, unsqueeze_1273);  sub_398 = unsqueeze_1273 = None
    mul_1306: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_399, unsqueeze_1279);  sub_399 = unsqueeze_1279 = None
    mul_1307: "f32[16]" = torch.ops.aten.mul.Tensor(sum_183, squeeze_1);  sum_183 = squeeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:282, code: x = self.conv_stem(x)
    convolution_backward_94 = torch.ops.aten.convolution_backward.default(mul_1306, primals_513, primals_5, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_1306 = primals_513 = primals_5 = None
    getitem_443: "f32[16, 3, 3, 3]" = convolution_backward_94[1];  convolution_backward_94 = None
    return [mul_575, sum_3, permute_4, view_2, getitem_443, mul_1307, sum_182, getitem_440, mul_1298, sum_180, getitem_437, mul_1289, sum_178, getitem_434, mul_1280, sum_176, getitem_431, mul_1271, sum_174, getitem_428, mul_1262, sum_172, getitem_425, mul_1253, sum_170, getitem_422, mul_1244, sum_168, getitem_419, mul_1235, sum_166, getitem_416, mul_1226, sum_164, getitem_413, mul_1217, sum_162, getitem_410, mul_1208, sum_160, getitem_407, mul_1199, sum_158, getitem_404, mul_1190, sum_156, getitem_401, mul_1181, sum_154, getitem_398, mul_1172, sum_152, getitem_395, mul_1163, sum_150, getitem_392, mul_1154, sum_148, getitem_389, mul_1145, sum_146, getitem_386, sum_145, getitem_383, sum_144, getitem_380, mul_1133, sum_141, getitem_377, mul_1124, sum_139, getitem_374, mul_1115, sum_137, getitem_371, mul_1106, sum_135, getitem_368, mul_1097, sum_133, getitem_365, mul_1088, sum_131, getitem_362, sum_130, getitem_359, sum_129, getitem_356, mul_1076, sum_126, getitem_353, mul_1067, sum_124, getitem_350, mul_1058, sum_122, getitem_347, mul_1049, sum_120, getitem_344, mul_1040, sum_118, getitem_341, mul_1031, sum_116, getitem_338, mul_1022, sum_114, getitem_335, mul_1013, sum_112, getitem_332, mul_1004, sum_110, getitem_329, mul_995, sum_108, getitem_326, mul_986, sum_106, getitem_323, mul_977, sum_104, getitem_320, mul_968, sum_102, getitem_317, mul_959, sum_100, getitem_314, mul_950, sum_98, getitem_311, mul_941, sum_96, getitem_308, mul_932, sum_94, getitem_305, mul_923, sum_92, getitem_302, mul_914, sum_90, getitem_299, mul_905, sum_88, getitem_296, mul_896, sum_86, getitem_293, mul_887, sum_84, getitem_290, mul_878, sum_82, getitem_287, sum_81, getitem_284, sum_80, getitem_281, mul_866, sum_77, getitem_278, mul_857, sum_75, getitem_275, mul_848, sum_73, getitem_272, mul_839, sum_71, getitem_269, mul_830, sum_69, getitem_266, mul_821, sum_67, getitem_263, sum_66, getitem_260, sum_65, getitem_257, mul_809, sum_62, getitem_254, mul_800, sum_60, getitem_251, mul_791, sum_58, getitem_248, mul_782, sum_56, getitem_245, mul_773, sum_54, getitem_242, sum_53, getitem_239, sum_52, getitem_236, mul_761, sum_49, getitem_233, mul_752, sum_47, getitem_230, mul_743, sum_45, getitem_227, mul_734, sum_43, getitem_224, mul_725, sum_41, getitem_221, mul_716, sum_39, getitem_218, mul_707, sum_37, getitem_215, mul_698, sum_35, getitem_212, mul_689, sum_33, getitem_209, mul_680, sum_31, getitem_206, sum_30, getitem_203, sum_29, getitem_200, mul_668, sum_26, getitem_197, mul_659, sum_24, getitem_194, mul_650, sum_22, getitem_191, mul_641, sum_20, getitem_188, mul_632, sum_18, getitem_185, mul_623, sum_16, getitem_182, mul_614, sum_14, getitem_179, mul_605, sum_12, getitem_176, sum_11, getitem_173, sum_10, getitem_170, mul_593, sum_7, getitem_167, mul_584, sum_5, getitem_164, getitem_161, sum_2, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
    