from __future__ import annotations



def forward(self, primals_1: "f32[64, 3, 7, 7]", primals_2: "f32[64]", primals_4: "f32[512, 64, 1, 1]", primals_5: "f32[512]", primals_7: "f32[512, 16, 3, 3]", primals_8: "f32[512]", primals_10: "f32[256, 512, 1, 1]", primals_11: "f32[256]", primals_13: "f32[256, 64, 1, 1]", primals_14: "f32[256]", primals_16: "f32[512, 256, 1, 1]", primals_17: "f32[512]", primals_19: "f32[512, 16, 3, 3]", primals_20: "f32[512]", primals_22: "f32[256, 512, 1, 1]", primals_23: "f32[256]", primals_25: "f32[512, 256, 1, 1]", primals_26: "f32[512]", primals_28: "f32[512, 16, 3, 3]", primals_29: "f32[512]", primals_31: "f32[256, 512, 1, 1]", primals_32: "f32[256]", primals_34: "f32[1024, 256, 1, 1]", primals_35: "f32[1024]", primals_37: "f32[1024, 32, 3, 3]", primals_38: "f32[1024]", primals_40: "f32[512, 1024, 1, 1]", primals_41: "f32[512]", primals_43: "f32[512, 256, 1, 1]", primals_44: "f32[512]", primals_46: "f32[1024, 512, 1, 1]", primals_47: "f32[1024]", primals_49: "f32[1024, 32, 3, 3]", primals_50: "f32[1024]", primals_52: "f32[512, 1024, 1, 1]", primals_53: "f32[512]", primals_55: "f32[1024, 512, 1, 1]", primals_56: "f32[1024]", primals_58: "f32[1024, 32, 3, 3]", primals_59: "f32[1024]", primals_61: "f32[512, 1024, 1, 1]", primals_62: "f32[512]", primals_64: "f32[1024, 512, 1, 1]", primals_65: "f32[1024]", primals_67: "f32[1024, 32, 3, 3]", primals_68: "f32[1024]", primals_70: "f32[512, 1024, 1, 1]", primals_71: "f32[512]", primals_73: "f32[2048, 512, 1, 1]", primals_74: "f32[2048]", primals_76: "f32[2048, 64, 3, 3]", primals_77: "f32[2048]", primals_79: "f32[1024, 2048, 1, 1]", primals_80: "f32[1024]", primals_82: "f32[1024, 512, 1, 1]", primals_83: "f32[1024]", primals_85: "f32[2048, 1024, 1, 1]", primals_86: "f32[2048]", primals_88: "f32[2048, 64, 3, 3]", primals_89: "f32[2048]", primals_91: "f32[1024, 2048, 1, 1]", primals_92: "f32[1024]", primals_94: "f32[2048, 1024, 1, 1]", primals_95: "f32[2048]", primals_97: "f32[2048, 64, 3, 3]", primals_98: "f32[2048]", primals_100: "f32[1024, 2048, 1, 1]", primals_101: "f32[1024]", primals_103: "f32[2048, 1024, 1, 1]", primals_104: "f32[2048]", primals_106: "f32[2048, 64, 3, 3]", primals_107: "f32[2048]", primals_109: "f32[1024, 2048, 1, 1]", primals_110: "f32[1024]", primals_112: "f32[2048, 1024, 1, 1]", primals_113: "f32[2048]", primals_115: "f32[2048, 64, 3, 3]", primals_116: "f32[2048]", primals_118: "f32[1024, 2048, 1, 1]", primals_119: "f32[1024]", primals_121: "f32[2048, 1024, 1, 1]", primals_122: "f32[2048]", primals_124: "f32[2048, 64, 3, 3]", primals_125: "f32[2048]", primals_127: "f32[1024, 2048, 1, 1]", primals_128: "f32[1024]", primals_130: "f32[2048, 1024, 1, 1]", primals_131: "f32[2048]", primals_133: "f32[2048, 64, 3, 3]", primals_134: "f32[2048]", primals_136: "f32[1024, 2048, 1, 1]", primals_137: "f32[1024]", primals_139: "f32[2048, 1024, 1, 1]", primals_140: "f32[2048]", primals_142: "f32[2048, 64, 3, 3]", primals_143: "f32[2048]", primals_145: "f32[1024, 2048, 1, 1]", primals_146: "f32[1024]", primals_148: "f32[2048, 1024, 1, 1]", primals_149: "f32[2048]", primals_151: "f32[2048, 64, 3, 3]", primals_152: "f32[2048]", primals_154: "f32[1024, 2048, 1, 1]", primals_155: "f32[1024]", primals_157: "f32[2048, 1024, 1, 1]", primals_158: "f32[2048]", primals_160: "f32[2048, 64, 3, 3]", primals_161: "f32[2048]", primals_163: "f32[1024, 2048, 1, 1]", primals_164: "f32[1024]", primals_166: "f32[2048, 1024, 1, 1]", primals_167: "f32[2048]", primals_169: "f32[2048, 64, 3, 3]", primals_170: "f32[2048]", primals_172: "f32[1024, 2048, 1, 1]", primals_173: "f32[1024]", primals_175: "f32[2048, 1024, 1, 1]", primals_176: "f32[2048]", primals_178: "f32[2048, 64, 3, 3]", primals_179: "f32[2048]", primals_181: "f32[1024, 2048, 1, 1]", primals_182: "f32[1024]", primals_184: "f32[2048, 1024, 1, 1]", primals_185: "f32[2048]", primals_187: "f32[2048, 64, 3, 3]", primals_188: "f32[2048]", primals_190: "f32[1024, 2048, 1, 1]", primals_191: "f32[1024]", primals_193: "f32[2048, 1024, 1, 1]", primals_194: "f32[2048]", primals_196: "f32[2048, 64, 3, 3]", primals_197: "f32[2048]", primals_199: "f32[1024, 2048, 1, 1]", primals_200: "f32[1024]", primals_202: "f32[2048, 1024, 1, 1]", primals_203: "f32[2048]", primals_205: "f32[2048, 64, 3, 3]", primals_206: "f32[2048]", primals_208: "f32[1024, 2048, 1, 1]", primals_209: "f32[1024]", primals_211: "f32[2048, 1024, 1, 1]", primals_212: "f32[2048]", primals_214: "f32[2048, 64, 3, 3]", primals_215: "f32[2048]", primals_217: "f32[1024, 2048, 1, 1]", primals_218: "f32[1024]", primals_220: "f32[2048, 1024, 1, 1]", primals_221: "f32[2048]", primals_223: "f32[2048, 64, 3, 3]", primals_224: "f32[2048]", primals_226: "f32[1024, 2048, 1, 1]", primals_227: "f32[1024]", primals_229: "f32[2048, 1024, 1, 1]", primals_230: "f32[2048]", primals_232: "f32[2048, 64, 3, 3]", primals_233: "f32[2048]", primals_235: "f32[1024, 2048, 1, 1]", primals_236: "f32[1024]", primals_238: "f32[2048, 1024, 1, 1]", primals_239: "f32[2048]", primals_241: "f32[2048, 64, 3, 3]", primals_242: "f32[2048]", primals_244: "f32[1024, 2048, 1, 1]", primals_245: "f32[1024]", primals_247: "f32[2048, 1024, 1, 1]", primals_248: "f32[2048]", primals_250: "f32[2048, 64, 3, 3]", primals_251: "f32[2048]", primals_253: "f32[1024, 2048, 1, 1]", primals_254: "f32[1024]", primals_256: "f32[2048, 1024, 1, 1]", primals_257: "f32[2048]", primals_259: "f32[2048, 64, 3, 3]", primals_260: "f32[2048]", primals_262: "f32[1024, 2048, 1, 1]", primals_263: "f32[1024]", primals_265: "f32[2048, 1024, 1, 1]", primals_266: "f32[2048]", primals_268: "f32[2048, 64, 3, 3]", primals_269: "f32[2048]", primals_271: "f32[1024, 2048, 1, 1]", primals_272: "f32[1024]", primals_274: "f32[2048, 1024, 1, 1]", primals_275: "f32[2048]", primals_277: "f32[2048, 64, 3, 3]", primals_278: "f32[2048]", primals_280: "f32[1024, 2048, 1, 1]", primals_281: "f32[1024]", primals_283: "f32[4096, 1024, 1, 1]", primals_284: "f32[4096]", primals_286: "f32[4096, 128, 3, 3]", primals_287: "f32[4096]", primals_289: "f32[2048, 4096, 1, 1]", primals_290: "f32[2048]", primals_292: "f32[2048, 1024, 1, 1]", primals_293: "f32[2048]", primals_295: "f32[4096, 2048, 1, 1]", primals_296: "f32[4096]", primals_298: "f32[4096, 128, 3, 3]", primals_299: "f32[4096]", primals_301: "f32[2048, 4096, 1, 1]", primals_302: "f32[2048]", primals_304: "f32[4096, 2048, 1, 1]", primals_305: "f32[4096]", primals_307: "f32[4096, 128, 3, 3]", primals_308: "f32[4096]", primals_310: "f32[2048, 4096, 1, 1]", primals_311: "f32[2048]", primals_627: "f32[8, 3, 224, 224]", convolution: "f32[8, 64, 112, 112]", squeeze_1: "f32[64]", relu: "f32[8, 64, 112, 112]", getitem_2: "f32[8, 64, 56, 56]", getitem_3: "i64[8, 64, 56, 56]", convolution_1: "f32[8, 512, 56, 56]", squeeze_4: "f32[512]", relu_1: "f32[8, 512, 56, 56]", convolution_2: "f32[8, 512, 56, 56]", squeeze_7: "f32[512]", relu_2: "f32[8, 512, 56, 56]", convolution_3: "f32[8, 256, 56, 56]", squeeze_10: "f32[256]", convolution_4: "f32[8, 256, 56, 56]", squeeze_13: "f32[256]", relu_3: "f32[8, 256, 56, 56]", convolution_5: "f32[8, 512, 56, 56]", squeeze_16: "f32[512]", relu_4: "f32[8, 512, 56, 56]", convolution_6: "f32[8, 512, 56, 56]", squeeze_19: "f32[512]", relu_5: "f32[8, 512, 56, 56]", convolution_7: "f32[8, 256, 56, 56]", squeeze_22: "f32[256]", relu_6: "f32[8, 256, 56, 56]", convolution_8: "f32[8, 512, 56, 56]", squeeze_25: "f32[512]", relu_7: "f32[8, 512, 56, 56]", convolution_9: "f32[8, 512, 56, 56]", squeeze_28: "f32[512]", relu_8: "f32[8, 512, 56, 56]", convolution_10: "f32[8, 256, 56, 56]", squeeze_31: "f32[256]", relu_9: "f32[8, 256, 56, 56]", convolution_11: "f32[8, 1024, 56, 56]", squeeze_34: "f32[1024]", relu_10: "f32[8, 1024, 56, 56]", convolution_12: "f32[8, 1024, 28, 28]", squeeze_37: "f32[1024]", relu_11: "f32[8, 1024, 28, 28]", convolution_13: "f32[8, 512, 28, 28]", squeeze_40: "f32[512]", convolution_14: "f32[8, 512, 28, 28]", squeeze_43: "f32[512]", relu_12: "f32[8, 512, 28, 28]", convolution_15: "f32[8, 1024, 28, 28]", squeeze_46: "f32[1024]", relu_13: "f32[8, 1024, 28, 28]", convolution_16: "f32[8, 1024, 28, 28]", squeeze_49: "f32[1024]", relu_14: "f32[8, 1024, 28, 28]", convolution_17: "f32[8, 512, 28, 28]", squeeze_52: "f32[512]", relu_15: "f32[8, 512, 28, 28]", convolution_18: "f32[8, 1024, 28, 28]", squeeze_55: "f32[1024]", relu_16: "f32[8, 1024, 28, 28]", convolution_19: "f32[8, 1024, 28, 28]", squeeze_58: "f32[1024]", relu_17: "f32[8, 1024, 28, 28]", convolution_20: "f32[8, 512, 28, 28]", squeeze_61: "f32[512]", relu_18: "f32[8, 512, 28, 28]", convolution_21: "f32[8, 1024, 28, 28]", squeeze_64: "f32[1024]", relu_19: "f32[8, 1024, 28, 28]", convolution_22: "f32[8, 1024, 28, 28]", squeeze_67: "f32[1024]", relu_20: "f32[8, 1024, 28, 28]", convolution_23: "f32[8, 512, 28, 28]", squeeze_70: "f32[512]", relu_21: "f32[8, 512, 28, 28]", convolution_24: "f32[8, 2048, 28, 28]", squeeze_73: "f32[2048]", relu_22: "f32[8, 2048, 28, 28]", convolution_25: "f32[8, 2048, 14, 14]", squeeze_76: "f32[2048]", relu_23: "f32[8, 2048, 14, 14]", convolution_26: "f32[8, 1024, 14, 14]", squeeze_79: "f32[1024]", convolution_27: "f32[8, 1024, 14, 14]", squeeze_82: "f32[1024]", relu_24: "f32[8, 1024, 14, 14]", convolution_28: "f32[8, 2048, 14, 14]", squeeze_85: "f32[2048]", relu_25: "f32[8, 2048, 14, 14]", convolution_29: "f32[8, 2048, 14, 14]", squeeze_88: "f32[2048]", relu_26: "f32[8, 2048, 14, 14]", convolution_30: "f32[8, 1024, 14, 14]", squeeze_91: "f32[1024]", relu_27: "f32[8, 1024, 14, 14]", convolution_31: "f32[8, 2048, 14, 14]", squeeze_94: "f32[2048]", relu_28: "f32[8, 2048, 14, 14]", convolution_32: "f32[8, 2048, 14, 14]", squeeze_97: "f32[2048]", relu_29: "f32[8, 2048, 14, 14]", convolution_33: "f32[8, 1024, 14, 14]", squeeze_100: "f32[1024]", relu_30: "f32[8, 1024, 14, 14]", convolution_34: "f32[8, 2048, 14, 14]", squeeze_103: "f32[2048]", relu_31: "f32[8, 2048, 14, 14]", convolution_35: "f32[8, 2048, 14, 14]", squeeze_106: "f32[2048]", relu_32: "f32[8, 2048, 14, 14]", convolution_36: "f32[8, 1024, 14, 14]", squeeze_109: "f32[1024]", relu_33: "f32[8, 1024, 14, 14]", convolution_37: "f32[8, 2048, 14, 14]", squeeze_112: "f32[2048]", relu_34: "f32[8, 2048, 14, 14]", convolution_38: "f32[8, 2048, 14, 14]", squeeze_115: "f32[2048]", relu_35: "f32[8, 2048, 14, 14]", convolution_39: "f32[8, 1024, 14, 14]", squeeze_118: "f32[1024]", relu_36: "f32[8, 1024, 14, 14]", convolution_40: "f32[8, 2048, 14, 14]", squeeze_121: "f32[2048]", relu_37: "f32[8, 2048, 14, 14]", convolution_41: "f32[8, 2048, 14, 14]", squeeze_124: "f32[2048]", relu_38: "f32[8, 2048, 14, 14]", convolution_42: "f32[8, 1024, 14, 14]", squeeze_127: "f32[1024]", relu_39: "f32[8, 1024, 14, 14]", convolution_43: "f32[8, 2048, 14, 14]", squeeze_130: "f32[2048]", relu_40: "f32[8, 2048, 14, 14]", convolution_44: "f32[8, 2048, 14, 14]", squeeze_133: "f32[2048]", relu_41: "f32[8, 2048, 14, 14]", convolution_45: "f32[8, 1024, 14, 14]", squeeze_136: "f32[1024]", relu_42: "f32[8, 1024, 14, 14]", convolution_46: "f32[8, 2048, 14, 14]", squeeze_139: "f32[2048]", relu_43: "f32[8, 2048, 14, 14]", convolution_47: "f32[8, 2048, 14, 14]", squeeze_142: "f32[2048]", relu_44: "f32[8, 2048, 14, 14]", convolution_48: "f32[8, 1024, 14, 14]", squeeze_145: "f32[1024]", relu_45: "f32[8, 1024, 14, 14]", convolution_49: "f32[8, 2048, 14, 14]", squeeze_148: "f32[2048]", relu_46: "f32[8, 2048, 14, 14]", convolution_50: "f32[8, 2048, 14, 14]", squeeze_151: "f32[2048]", relu_47: "f32[8, 2048, 14, 14]", convolution_51: "f32[8, 1024, 14, 14]", squeeze_154: "f32[1024]", relu_48: "f32[8, 1024, 14, 14]", convolution_52: "f32[8, 2048, 14, 14]", squeeze_157: "f32[2048]", relu_49: "f32[8, 2048, 14, 14]", convolution_53: "f32[8, 2048, 14, 14]", squeeze_160: "f32[2048]", relu_50: "f32[8, 2048, 14, 14]", convolution_54: "f32[8, 1024, 14, 14]", squeeze_163: "f32[1024]", relu_51: "f32[8, 1024, 14, 14]", convolution_55: "f32[8, 2048, 14, 14]", squeeze_166: "f32[2048]", relu_52: "f32[8, 2048, 14, 14]", convolution_56: "f32[8, 2048, 14, 14]", squeeze_169: "f32[2048]", relu_53: "f32[8, 2048, 14, 14]", convolution_57: "f32[8, 1024, 14, 14]", squeeze_172: "f32[1024]", relu_54: "f32[8, 1024, 14, 14]", convolution_58: "f32[8, 2048, 14, 14]", squeeze_175: "f32[2048]", relu_55: "f32[8, 2048, 14, 14]", convolution_59: "f32[8, 2048, 14, 14]", squeeze_178: "f32[2048]", relu_56: "f32[8, 2048, 14, 14]", convolution_60: "f32[8, 1024, 14, 14]", squeeze_181: "f32[1024]", relu_57: "f32[8, 1024, 14, 14]", convolution_61: "f32[8, 2048, 14, 14]", squeeze_184: "f32[2048]", relu_58: "f32[8, 2048, 14, 14]", convolution_62: "f32[8, 2048, 14, 14]", squeeze_187: "f32[2048]", relu_59: "f32[8, 2048, 14, 14]", convolution_63: "f32[8, 1024, 14, 14]", squeeze_190: "f32[1024]", relu_60: "f32[8, 1024, 14, 14]", convolution_64: "f32[8, 2048, 14, 14]", squeeze_193: "f32[2048]", relu_61: "f32[8, 2048, 14, 14]", convolution_65: "f32[8, 2048, 14, 14]", squeeze_196: "f32[2048]", relu_62: "f32[8, 2048, 14, 14]", convolution_66: "f32[8, 1024, 14, 14]", squeeze_199: "f32[1024]", relu_63: "f32[8, 1024, 14, 14]", convolution_67: "f32[8, 2048, 14, 14]", squeeze_202: "f32[2048]", relu_64: "f32[8, 2048, 14, 14]", convolution_68: "f32[8, 2048, 14, 14]", squeeze_205: "f32[2048]", relu_65: "f32[8, 2048, 14, 14]", convolution_69: "f32[8, 1024, 14, 14]", squeeze_208: "f32[1024]", relu_66: "f32[8, 1024, 14, 14]", convolution_70: "f32[8, 2048, 14, 14]", squeeze_211: "f32[2048]", relu_67: "f32[8, 2048, 14, 14]", convolution_71: "f32[8, 2048, 14, 14]", squeeze_214: "f32[2048]", relu_68: "f32[8, 2048, 14, 14]", convolution_72: "f32[8, 1024, 14, 14]", squeeze_217: "f32[1024]", relu_69: "f32[8, 1024, 14, 14]", convolution_73: "f32[8, 2048, 14, 14]", squeeze_220: "f32[2048]", relu_70: "f32[8, 2048, 14, 14]", convolution_74: "f32[8, 2048, 14, 14]", squeeze_223: "f32[2048]", relu_71: "f32[8, 2048, 14, 14]", convolution_75: "f32[8, 1024, 14, 14]", squeeze_226: "f32[1024]", relu_72: "f32[8, 1024, 14, 14]", convolution_76: "f32[8, 2048, 14, 14]", squeeze_229: "f32[2048]", relu_73: "f32[8, 2048, 14, 14]", convolution_77: "f32[8, 2048, 14, 14]", squeeze_232: "f32[2048]", relu_74: "f32[8, 2048, 14, 14]", convolution_78: "f32[8, 1024, 14, 14]", squeeze_235: "f32[1024]", relu_75: "f32[8, 1024, 14, 14]", convolution_79: "f32[8, 2048, 14, 14]", squeeze_238: "f32[2048]", relu_76: "f32[8, 2048, 14, 14]", convolution_80: "f32[8, 2048, 14, 14]", squeeze_241: "f32[2048]", relu_77: "f32[8, 2048, 14, 14]", convolution_81: "f32[8, 1024, 14, 14]", squeeze_244: "f32[1024]", relu_78: "f32[8, 1024, 14, 14]", convolution_82: "f32[8, 2048, 14, 14]", squeeze_247: "f32[2048]", relu_79: "f32[8, 2048, 14, 14]", convolution_83: "f32[8, 2048, 14, 14]", squeeze_250: "f32[2048]", relu_80: "f32[8, 2048, 14, 14]", convolution_84: "f32[8, 1024, 14, 14]", squeeze_253: "f32[1024]", relu_81: "f32[8, 1024, 14, 14]", convolution_85: "f32[8, 2048, 14, 14]", squeeze_256: "f32[2048]", relu_82: "f32[8, 2048, 14, 14]", convolution_86: "f32[8, 2048, 14, 14]", squeeze_259: "f32[2048]", relu_83: "f32[8, 2048, 14, 14]", convolution_87: "f32[8, 1024, 14, 14]", squeeze_262: "f32[1024]", relu_84: "f32[8, 1024, 14, 14]", convolution_88: "f32[8, 2048, 14, 14]", squeeze_265: "f32[2048]", relu_85: "f32[8, 2048, 14, 14]", convolution_89: "f32[8, 2048, 14, 14]", squeeze_268: "f32[2048]", relu_86: "f32[8, 2048, 14, 14]", convolution_90: "f32[8, 1024, 14, 14]", squeeze_271: "f32[1024]", relu_87: "f32[8, 1024, 14, 14]", convolution_91: "f32[8, 2048, 14, 14]", squeeze_274: "f32[2048]", relu_88: "f32[8, 2048, 14, 14]", convolution_92: "f32[8, 2048, 14, 14]", squeeze_277: "f32[2048]", relu_89: "f32[8, 2048, 14, 14]", convolution_93: "f32[8, 1024, 14, 14]", squeeze_280: "f32[1024]", relu_90: "f32[8, 1024, 14, 14]", convolution_94: "f32[8, 4096, 14, 14]", squeeze_283: "f32[4096]", relu_91: "f32[8, 4096, 14, 14]", convolution_95: "f32[8, 4096, 7, 7]", squeeze_286: "f32[4096]", relu_92: "f32[8, 4096, 7, 7]", convolution_96: "f32[8, 2048, 7, 7]", squeeze_289: "f32[2048]", convolution_97: "f32[8, 2048, 7, 7]", squeeze_292: "f32[2048]", relu_93: "f32[8, 2048, 7, 7]", convolution_98: "f32[8, 4096, 7, 7]", squeeze_295: "f32[4096]", relu_94: "f32[8, 4096, 7, 7]", convolution_99: "f32[8, 4096, 7, 7]", squeeze_298: "f32[4096]", relu_95: "f32[8, 4096, 7, 7]", convolution_100: "f32[8, 2048, 7, 7]", squeeze_301: "f32[2048]", relu_96: "f32[8, 2048, 7, 7]", convolution_101: "f32[8, 4096, 7, 7]", squeeze_304: "f32[4096]", relu_97: "f32[8, 4096, 7, 7]", convolution_102: "f32[8, 4096, 7, 7]", squeeze_307: "f32[4096]", relu_98: "f32[8, 4096, 7, 7]", convolution_103: "f32[8, 2048, 7, 7]", squeeze_310: "f32[2048]", view: "f32[8, 2048]", permute_1: "f32[1000, 2048]", le: "b8[8, 2048, 7, 7]", unsqueeze_418: "f32[1, 2048, 1, 1]", unsqueeze_430: "f32[1, 4096, 1, 1]", unsqueeze_442: "f32[1, 4096, 1, 1]", unsqueeze_454: "f32[1, 2048, 1, 1]", unsqueeze_466: "f32[1, 4096, 1, 1]", unsqueeze_478: "f32[1, 4096, 1, 1]", unsqueeze_490: "f32[1, 2048, 1, 1]", unsqueeze_502: "f32[1, 2048, 1, 1]", unsqueeze_514: "f32[1, 4096, 1, 1]", unsqueeze_526: "f32[1, 4096, 1, 1]", unsqueeze_538: "f32[1, 1024, 1, 1]", unsqueeze_550: "f32[1, 2048, 1, 1]", unsqueeze_562: "f32[1, 2048, 1, 1]", unsqueeze_574: "f32[1, 1024, 1, 1]", unsqueeze_586: "f32[1, 2048, 1, 1]", unsqueeze_598: "f32[1, 2048, 1, 1]", unsqueeze_610: "f32[1, 1024, 1, 1]", unsqueeze_622: "f32[1, 2048, 1, 1]", unsqueeze_634: "f32[1, 2048, 1, 1]", unsqueeze_646: "f32[1, 1024, 1, 1]", unsqueeze_658: "f32[1, 2048, 1, 1]", unsqueeze_670: "f32[1, 2048, 1, 1]", unsqueeze_682: "f32[1, 1024, 1, 1]", unsqueeze_694: "f32[1, 2048, 1, 1]", unsqueeze_706: "f32[1, 2048, 1, 1]", unsqueeze_718: "f32[1, 1024, 1, 1]", unsqueeze_730: "f32[1, 2048, 1, 1]", unsqueeze_742: "f32[1, 2048, 1, 1]", unsqueeze_754: "f32[1, 1024, 1, 1]", unsqueeze_766: "f32[1, 2048, 1, 1]", unsqueeze_778: "f32[1, 2048, 1, 1]", unsqueeze_790: "f32[1, 1024, 1, 1]", unsqueeze_802: "f32[1, 2048, 1, 1]", unsqueeze_814: "f32[1, 2048, 1, 1]", unsqueeze_826: "f32[1, 1024, 1, 1]", unsqueeze_838: "f32[1, 2048, 1, 1]", unsqueeze_850: "f32[1, 2048, 1, 1]", unsqueeze_862: "f32[1, 1024, 1, 1]", unsqueeze_874: "f32[1, 2048, 1, 1]", unsqueeze_886: "f32[1, 2048, 1, 1]", unsqueeze_898: "f32[1, 1024, 1, 1]", unsqueeze_910: "f32[1, 2048, 1, 1]", unsqueeze_922: "f32[1, 2048, 1, 1]", unsqueeze_934: "f32[1, 1024, 1, 1]", unsqueeze_946: "f32[1, 2048, 1, 1]", unsqueeze_958: "f32[1, 2048, 1, 1]", unsqueeze_970: "f32[1, 1024, 1, 1]", unsqueeze_982: "f32[1, 2048, 1, 1]", unsqueeze_994: "f32[1, 2048, 1, 1]", unsqueeze_1006: "f32[1, 1024, 1, 1]", unsqueeze_1018: "f32[1, 2048, 1, 1]", unsqueeze_1030: "f32[1, 2048, 1, 1]", unsqueeze_1042: "f32[1, 1024, 1, 1]", unsqueeze_1054: "f32[1, 2048, 1, 1]", unsqueeze_1066: "f32[1, 2048, 1, 1]", unsqueeze_1078: "f32[1, 1024, 1, 1]", unsqueeze_1090: "f32[1, 2048, 1, 1]", unsqueeze_1102: "f32[1, 2048, 1, 1]", unsqueeze_1114: "f32[1, 1024, 1, 1]", unsqueeze_1126: "f32[1, 2048, 1, 1]", unsqueeze_1138: "f32[1, 2048, 1, 1]", unsqueeze_1150: "f32[1, 1024, 1, 1]", unsqueeze_1162: "f32[1, 2048, 1, 1]", unsqueeze_1174: "f32[1, 2048, 1, 1]", unsqueeze_1186: "f32[1, 1024, 1, 1]", unsqueeze_1198: "f32[1, 2048, 1, 1]", unsqueeze_1210: "f32[1, 2048, 1, 1]", unsqueeze_1222: "f32[1, 1024, 1, 1]", unsqueeze_1234: "f32[1, 2048, 1, 1]", unsqueeze_1246: "f32[1, 2048, 1, 1]", unsqueeze_1258: "f32[1, 1024, 1, 1]", unsqueeze_1270: "f32[1, 2048, 1, 1]", unsqueeze_1282: "f32[1, 2048, 1, 1]", unsqueeze_1294: "f32[1, 1024, 1, 1]", unsqueeze_1306: "f32[1, 2048, 1, 1]", unsqueeze_1318: "f32[1, 2048, 1, 1]", unsqueeze_1330: "f32[1, 1024, 1, 1]", unsqueeze_1342: "f32[1, 1024, 1, 1]", unsqueeze_1354: "f32[1, 2048, 1, 1]", unsqueeze_1366: "f32[1, 2048, 1, 1]", unsqueeze_1378: "f32[1, 512, 1, 1]", unsqueeze_1390: "f32[1, 1024, 1, 1]", unsqueeze_1402: "f32[1, 1024, 1, 1]", unsqueeze_1414: "f32[1, 512, 1, 1]", unsqueeze_1426: "f32[1, 1024, 1, 1]", unsqueeze_1438: "f32[1, 1024, 1, 1]", unsqueeze_1450: "f32[1, 512, 1, 1]", unsqueeze_1462: "f32[1, 1024, 1, 1]", unsqueeze_1474: "f32[1, 1024, 1, 1]", unsqueeze_1486: "f32[1, 512, 1, 1]", unsqueeze_1498: "f32[1, 512, 1, 1]", unsqueeze_1510: "f32[1, 1024, 1, 1]", unsqueeze_1522: "f32[1, 1024, 1, 1]", unsqueeze_1534: "f32[1, 256, 1, 1]", unsqueeze_1546: "f32[1, 512, 1, 1]", unsqueeze_1558: "f32[1, 512, 1, 1]", unsqueeze_1570: "f32[1, 256, 1, 1]", unsqueeze_1582: "f32[1, 512, 1, 1]", unsqueeze_1594: "f32[1, 512, 1, 1]", unsqueeze_1606: "f32[1, 256, 1, 1]", unsqueeze_1618: "f32[1, 256, 1, 1]", unsqueeze_1630: "f32[1, 512, 1, 1]", unsqueeze_1642: "f32[1, 512, 1, 1]", unsqueeze_1654: "f32[1, 64, 1, 1]", tangents_1: "f32[8, 1000]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:538, code: return x if pre_logits else self.fc(x)
    mm: "f32[8, 2048]" = torch.ops.aten.mm.default(tangents_1, permute_1);  permute_1 = None
    permute_2: "f32[1000, 8]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
    mm_1: "f32[1000, 2048]" = torch.ops.aten.mm.default(permute_2, view);  permute_2 = view = None
    permute_3: "f32[2048, 1000]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_1: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view_1: "f32[1000]" = torch.ops.aten.view.default(sum_1, [1000]);  sum_1 = None
    permute_4: "f32[1000, 2048]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view_2: "f32[8, 2048, 1, 1]" = torch.ops.aten.view.default(mm, [8, 2048, 1, 1]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    expand: "f32[8, 2048, 7, 7]" = torch.ops.aten.expand.default(view_2, [8, 2048, 7, 7]);  view_2 = None
    div: "f32[8, 2048, 7, 7]" = torch.ops.aten.div.Scalar(expand, 49);  expand = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    where: "f32[8, 2048, 7, 7]" = torch.ops.aten.where.self(le, full_default, div);  le = div = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    sum_2: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where, [0, 2, 3])
    sub_104: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_103, unsqueeze_418);  convolution_103 = unsqueeze_418 = None
    mul_728: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(where, sub_104)
    sum_3: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_728, [0, 2, 3]);  mul_728 = None
    mul_729: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_2, 0.002551020408163265)
    unsqueeze_419: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_729, 0);  mul_729 = None
    unsqueeze_420: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_419, 2);  unsqueeze_419 = None
    unsqueeze_421: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_420, 3);  unsqueeze_420 = None
    mul_730: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_3, 0.002551020408163265)
    mul_731: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_310, squeeze_310)
    mul_732: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_730, mul_731);  mul_730 = mul_731 = None
    unsqueeze_422: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_732, 0);  mul_732 = None
    unsqueeze_423: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_422, 2);  unsqueeze_422 = None
    unsqueeze_424: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_423, 3);  unsqueeze_423 = None
    mul_733: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_310, primals_311);  primals_311 = None
    unsqueeze_425: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_733, 0);  mul_733 = None
    unsqueeze_426: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_425, 2);  unsqueeze_425 = None
    unsqueeze_427: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_426, 3);  unsqueeze_426 = None
    mul_734: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_104, unsqueeze_424);  sub_104 = unsqueeze_424 = None
    sub_106: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(where, mul_734);  mul_734 = None
    sub_107: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(sub_106, unsqueeze_421);  sub_106 = unsqueeze_421 = None
    mul_735: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_107, unsqueeze_427);  sub_107 = unsqueeze_427 = None
    mul_736: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_3, squeeze_310);  sum_3 = squeeze_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_backward = torch.ops.aten.convolution_backward.default(mul_735, relu_98, primals_310, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_735 = primals_310 = None
    getitem_210: "f32[8, 4096, 7, 7]" = convolution_backward[0]
    getitem_211: "f32[2048, 4096, 1, 1]" = convolution_backward[1];  convolution_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    alias_104: "f32[8, 4096, 7, 7]" = torch.ops.aten.alias.default(relu_98);  relu_98 = None
    alias_105: "f32[8, 4096, 7, 7]" = torch.ops.aten.alias.default(alias_104);  alias_104 = None
    le_1: "b8[8, 4096, 7, 7]" = torch.ops.aten.le.Scalar(alias_105, 0);  alias_105 = None
    where_1: "f32[8, 4096, 7, 7]" = torch.ops.aten.where.self(le_1, full_default, getitem_210);  le_1 = getitem_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    sum_4: "f32[4096]" = torch.ops.aten.sum.dim_IntList(where_1, [0, 2, 3])
    sub_108: "f32[8, 4096, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_102, unsqueeze_430);  convolution_102 = unsqueeze_430 = None
    mul_737: "f32[8, 4096, 7, 7]" = torch.ops.aten.mul.Tensor(where_1, sub_108)
    sum_5: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_737, [0, 2, 3]);  mul_737 = None
    mul_738: "f32[4096]" = torch.ops.aten.mul.Tensor(sum_4, 0.002551020408163265)
    unsqueeze_431: "f32[1, 4096]" = torch.ops.aten.unsqueeze.default(mul_738, 0);  mul_738 = None
    unsqueeze_432: "f32[1, 4096, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_431, 2);  unsqueeze_431 = None
    unsqueeze_433: "f32[1, 4096, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_432, 3);  unsqueeze_432 = None
    mul_739: "f32[4096]" = torch.ops.aten.mul.Tensor(sum_5, 0.002551020408163265)
    mul_740: "f32[4096]" = torch.ops.aten.mul.Tensor(squeeze_307, squeeze_307)
    mul_741: "f32[4096]" = torch.ops.aten.mul.Tensor(mul_739, mul_740);  mul_739 = mul_740 = None
    unsqueeze_434: "f32[1, 4096]" = torch.ops.aten.unsqueeze.default(mul_741, 0);  mul_741 = None
    unsqueeze_435: "f32[1, 4096, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_434, 2);  unsqueeze_434 = None
    unsqueeze_436: "f32[1, 4096, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_435, 3);  unsqueeze_435 = None
    mul_742: "f32[4096]" = torch.ops.aten.mul.Tensor(squeeze_307, primals_308);  primals_308 = None
    unsqueeze_437: "f32[1, 4096]" = torch.ops.aten.unsqueeze.default(mul_742, 0);  mul_742 = None
    unsqueeze_438: "f32[1, 4096, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_437, 2);  unsqueeze_437 = None
    unsqueeze_439: "f32[1, 4096, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_438, 3);  unsqueeze_438 = None
    mul_743: "f32[8, 4096, 7, 7]" = torch.ops.aten.mul.Tensor(sub_108, unsqueeze_436);  sub_108 = unsqueeze_436 = None
    sub_110: "f32[8, 4096, 7, 7]" = torch.ops.aten.sub.Tensor(where_1, mul_743);  where_1 = mul_743 = None
    sub_111: "f32[8, 4096, 7, 7]" = torch.ops.aten.sub.Tensor(sub_110, unsqueeze_433);  sub_110 = unsqueeze_433 = None
    mul_744: "f32[8, 4096, 7, 7]" = torch.ops.aten.mul.Tensor(sub_111, unsqueeze_439);  sub_111 = unsqueeze_439 = None
    mul_745: "f32[4096]" = torch.ops.aten.mul.Tensor(sum_5, squeeze_307);  sum_5 = squeeze_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(mul_744, relu_97, primals_307, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_744 = primals_307 = None
    getitem_213: "f32[8, 4096, 7, 7]" = convolution_backward_1[0]
    getitem_214: "f32[4096, 128, 3, 3]" = convolution_backward_1[1];  convolution_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    alias_107: "f32[8, 4096, 7, 7]" = torch.ops.aten.alias.default(relu_97);  relu_97 = None
    alias_108: "f32[8, 4096, 7, 7]" = torch.ops.aten.alias.default(alias_107);  alias_107 = None
    le_2: "b8[8, 4096, 7, 7]" = torch.ops.aten.le.Scalar(alias_108, 0);  alias_108 = None
    where_2: "f32[8, 4096, 7, 7]" = torch.ops.aten.where.self(le_2, full_default, getitem_213);  le_2 = getitem_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    sum_6: "f32[4096]" = torch.ops.aten.sum.dim_IntList(where_2, [0, 2, 3])
    sub_112: "f32[8, 4096, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_101, unsqueeze_442);  convolution_101 = unsqueeze_442 = None
    mul_746: "f32[8, 4096, 7, 7]" = torch.ops.aten.mul.Tensor(where_2, sub_112)
    sum_7: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_746, [0, 2, 3]);  mul_746 = None
    mul_747: "f32[4096]" = torch.ops.aten.mul.Tensor(sum_6, 0.002551020408163265)
    unsqueeze_443: "f32[1, 4096]" = torch.ops.aten.unsqueeze.default(mul_747, 0);  mul_747 = None
    unsqueeze_444: "f32[1, 4096, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_443, 2);  unsqueeze_443 = None
    unsqueeze_445: "f32[1, 4096, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_444, 3);  unsqueeze_444 = None
    mul_748: "f32[4096]" = torch.ops.aten.mul.Tensor(sum_7, 0.002551020408163265)
    mul_749: "f32[4096]" = torch.ops.aten.mul.Tensor(squeeze_304, squeeze_304)
    mul_750: "f32[4096]" = torch.ops.aten.mul.Tensor(mul_748, mul_749);  mul_748 = mul_749 = None
    unsqueeze_446: "f32[1, 4096]" = torch.ops.aten.unsqueeze.default(mul_750, 0);  mul_750 = None
    unsqueeze_447: "f32[1, 4096, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_446, 2);  unsqueeze_446 = None
    unsqueeze_448: "f32[1, 4096, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_447, 3);  unsqueeze_447 = None
    mul_751: "f32[4096]" = torch.ops.aten.mul.Tensor(squeeze_304, primals_305);  primals_305 = None
    unsqueeze_449: "f32[1, 4096]" = torch.ops.aten.unsqueeze.default(mul_751, 0);  mul_751 = None
    unsqueeze_450: "f32[1, 4096, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_449, 2);  unsqueeze_449 = None
    unsqueeze_451: "f32[1, 4096, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_450, 3);  unsqueeze_450 = None
    mul_752: "f32[8, 4096, 7, 7]" = torch.ops.aten.mul.Tensor(sub_112, unsqueeze_448);  sub_112 = unsqueeze_448 = None
    sub_114: "f32[8, 4096, 7, 7]" = torch.ops.aten.sub.Tensor(where_2, mul_752);  where_2 = mul_752 = None
    sub_115: "f32[8, 4096, 7, 7]" = torch.ops.aten.sub.Tensor(sub_114, unsqueeze_445);  sub_114 = unsqueeze_445 = None
    mul_753: "f32[8, 4096, 7, 7]" = torch.ops.aten.mul.Tensor(sub_115, unsqueeze_451);  sub_115 = unsqueeze_451 = None
    mul_754: "f32[4096]" = torch.ops.aten.mul.Tensor(sum_7, squeeze_304);  sum_7 = squeeze_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(mul_753, relu_96, primals_304, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_753 = primals_304 = None
    getitem_216: "f32[8, 2048, 7, 7]" = convolution_backward_2[0]
    getitem_217: "f32[4096, 2048, 1, 1]" = convolution_backward_2[1];  convolution_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    add_553: "f32[8, 2048, 7, 7]" = torch.ops.aten.add.Tensor(where, getitem_216);  where = getitem_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    alias_110: "f32[8, 2048, 7, 7]" = torch.ops.aten.alias.default(relu_96);  relu_96 = None
    alias_111: "f32[8, 2048, 7, 7]" = torch.ops.aten.alias.default(alias_110);  alias_110 = None
    le_3: "b8[8, 2048, 7, 7]" = torch.ops.aten.le.Scalar(alias_111, 0);  alias_111 = None
    where_3: "f32[8, 2048, 7, 7]" = torch.ops.aten.where.self(le_3, full_default, add_553);  le_3 = add_553 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    sum_8: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_3, [0, 2, 3])
    sub_116: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_100, unsqueeze_454);  convolution_100 = unsqueeze_454 = None
    mul_755: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(where_3, sub_116)
    sum_9: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_755, [0, 2, 3]);  mul_755 = None
    mul_756: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_8, 0.002551020408163265)
    unsqueeze_455: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_756, 0);  mul_756 = None
    unsqueeze_456: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_455, 2);  unsqueeze_455 = None
    unsqueeze_457: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_456, 3);  unsqueeze_456 = None
    mul_757: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_9, 0.002551020408163265)
    mul_758: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_301, squeeze_301)
    mul_759: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_757, mul_758);  mul_757 = mul_758 = None
    unsqueeze_458: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_759, 0);  mul_759 = None
    unsqueeze_459: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_458, 2);  unsqueeze_458 = None
    unsqueeze_460: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_459, 3);  unsqueeze_459 = None
    mul_760: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_301, primals_302);  primals_302 = None
    unsqueeze_461: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_760, 0);  mul_760 = None
    unsqueeze_462: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_461, 2);  unsqueeze_461 = None
    unsqueeze_463: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_462, 3);  unsqueeze_462 = None
    mul_761: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_116, unsqueeze_460);  sub_116 = unsqueeze_460 = None
    sub_118: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(where_3, mul_761);  mul_761 = None
    sub_119: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(sub_118, unsqueeze_457);  sub_118 = unsqueeze_457 = None
    mul_762: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_119, unsqueeze_463);  sub_119 = unsqueeze_463 = None
    mul_763: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_9, squeeze_301);  sum_9 = squeeze_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_backward_3 = torch.ops.aten.convolution_backward.default(mul_762, relu_95, primals_301, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_762 = primals_301 = None
    getitem_219: "f32[8, 4096, 7, 7]" = convolution_backward_3[0]
    getitem_220: "f32[2048, 4096, 1, 1]" = convolution_backward_3[1];  convolution_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    alias_113: "f32[8, 4096, 7, 7]" = torch.ops.aten.alias.default(relu_95);  relu_95 = None
    alias_114: "f32[8, 4096, 7, 7]" = torch.ops.aten.alias.default(alias_113);  alias_113 = None
    le_4: "b8[8, 4096, 7, 7]" = torch.ops.aten.le.Scalar(alias_114, 0);  alias_114 = None
    where_4: "f32[8, 4096, 7, 7]" = torch.ops.aten.where.self(le_4, full_default, getitem_219);  le_4 = getitem_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    sum_10: "f32[4096]" = torch.ops.aten.sum.dim_IntList(where_4, [0, 2, 3])
    sub_120: "f32[8, 4096, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_99, unsqueeze_466);  convolution_99 = unsqueeze_466 = None
    mul_764: "f32[8, 4096, 7, 7]" = torch.ops.aten.mul.Tensor(where_4, sub_120)
    sum_11: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_764, [0, 2, 3]);  mul_764 = None
    mul_765: "f32[4096]" = torch.ops.aten.mul.Tensor(sum_10, 0.002551020408163265)
    unsqueeze_467: "f32[1, 4096]" = torch.ops.aten.unsqueeze.default(mul_765, 0);  mul_765 = None
    unsqueeze_468: "f32[1, 4096, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_467, 2);  unsqueeze_467 = None
    unsqueeze_469: "f32[1, 4096, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_468, 3);  unsqueeze_468 = None
    mul_766: "f32[4096]" = torch.ops.aten.mul.Tensor(sum_11, 0.002551020408163265)
    mul_767: "f32[4096]" = torch.ops.aten.mul.Tensor(squeeze_298, squeeze_298)
    mul_768: "f32[4096]" = torch.ops.aten.mul.Tensor(mul_766, mul_767);  mul_766 = mul_767 = None
    unsqueeze_470: "f32[1, 4096]" = torch.ops.aten.unsqueeze.default(mul_768, 0);  mul_768 = None
    unsqueeze_471: "f32[1, 4096, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_470, 2);  unsqueeze_470 = None
    unsqueeze_472: "f32[1, 4096, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_471, 3);  unsqueeze_471 = None
    mul_769: "f32[4096]" = torch.ops.aten.mul.Tensor(squeeze_298, primals_299);  primals_299 = None
    unsqueeze_473: "f32[1, 4096]" = torch.ops.aten.unsqueeze.default(mul_769, 0);  mul_769 = None
    unsqueeze_474: "f32[1, 4096, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_473, 2);  unsqueeze_473 = None
    unsqueeze_475: "f32[1, 4096, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_474, 3);  unsqueeze_474 = None
    mul_770: "f32[8, 4096, 7, 7]" = torch.ops.aten.mul.Tensor(sub_120, unsqueeze_472);  sub_120 = unsqueeze_472 = None
    sub_122: "f32[8, 4096, 7, 7]" = torch.ops.aten.sub.Tensor(where_4, mul_770);  where_4 = mul_770 = None
    sub_123: "f32[8, 4096, 7, 7]" = torch.ops.aten.sub.Tensor(sub_122, unsqueeze_469);  sub_122 = unsqueeze_469 = None
    mul_771: "f32[8, 4096, 7, 7]" = torch.ops.aten.mul.Tensor(sub_123, unsqueeze_475);  sub_123 = unsqueeze_475 = None
    mul_772: "f32[4096]" = torch.ops.aten.mul.Tensor(sum_11, squeeze_298);  sum_11 = squeeze_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_backward_4 = torch.ops.aten.convolution_backward.default(mul_771, relu_94, primals_298, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_771 = primals_298 = None
    getitem_222: "f32[8, 4096, 7, 7]" = convolution_backward_4[0]
    getitem_223: "f32[4096, 128, 3, 3]" = convolution_backward_4[1];  convolution_backward_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    alias_116: "f32[8, 4096, 7, 7]" = torch.ops.aten.alias.default(relu_94);  relu_94 = None
    alias_117: "f32[8, 4096, 7, 7]" = torch.ops.aten.alias.default(alias_116);  alias_116 = None
    le_5: "b8[8, 4096, 7, 7]" = torch.ops.aten.le.Scalar(alias_117, 0);  alias_117 = None
    where_5: "f32[8, 4096, 7, 7]" = torch.ops.aten.where.self(le_5, full_default, getitem_222);  le_5 = getitem_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    sum_12: "f32[4096]" = torch.ops.aten.sum.dim_IntList(where_5, [0, 2, 3])
    sub_124: "f32[8, 4096, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_98, unsqueeze_478);  convolution_98 = unsqueeze_478 = None
    mul_773: "f32[8, 4096, 7, 7]" = torch.ops.aten.mul.Tensor(where_5, sub_124)
    sum_13: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_773, [0, 2, 3]);  mul_773 = None
    mul_774: "f32[4096]" = torch.ops.aten.mul.Tensor(sum_12, 0.002551020408163265)
    unsqueeze_479: "f32[1, 4096]" = torch.ops.aten.unsqueeze.default(mul_774, 0);  mul_774 = None
    unsqueeze_480: "f32[1, 4096, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_479, 2);  unsqueeze_479 = None
    unsqueeze_481: "f32[1, 4096, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_480, 3);  unsqueeze_480 = None
    mul_775: "f32[4096]" = torch.ops.aten.mul.Tensor(sum_13, 0.002551020408163265)
    mul_776: "f32[4096]" = torch.ops.aten.mul.Tensor(squeeze_295, squeeze_295)
    mul_777: "f32[4096]" = torch.ops.aten.mul.Tensor(mul_775, mul_776);  mul_775 = mul_776 = None
    unsqueeze_482: "f32[1, 4096]" = torch.ops.aten.unsqueeze.default(mul_777, 0);  mul_777 = None
    unsqueeze_483: "f32[1, 4096, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_482, 2);  unsqueeze_482 = None
    unsqueeze_484: "f32[1, 4096, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_483, 3);  unsqueeze_483 = None
    mul_778: "f32[4096]" = torch.ops.aten.mul.Tensor(squeeze_295, primals_296);  primals_296 = None
    unsqueeze_485: "f32[1, 4096]" = torch.ops.aten.unsqueeze.default(mul_778, 0);  mul_778 = None
    unsqueeze_486: "f32[1, 4096, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_485, 2);  unsqueeze_485 = None
    unsqueeze_487: "f32[1, 4096, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_486, 3);  unsqueeze_486 = None
    mul_779: "f32[8, 4096, 7, 7]" = torch.ops.aten.mul.Tensor(sub_124, unsqueeze_484);  sub_124 = unsqueeze_484 = None
    sub_126: "f32[8, 4096, 7, 7]" = torch.ops.aten.sub.Tensor(where_5, mul_779);  where_5 = mul_779 = None
    sub_127: "f32[8, 4096, 7, 7]" = torch.ops.aten.sub.Tensor(sub_126, unsqueeze_481);  sub_126 = unsqueeze_481 = None
    mul_780: "f32[8, 4096, 7, 7]" = torch.ops.aten.mul.Tensor(sub_127, unsqueeze_487);  sub_127 = unsqueeze_487 = None
    mul_781: "f32[4096]" = torch.ops.aten.mul.Tensor(sum_13, squeeze_295);  sum_13 = squeeze_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_backward_5 = torch.ops.aten.convolution_backward.default(mul_780, relu_93, primals_295, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_780 = primals_295 = None
    getitem_225: "f32[8, 2048, 7, 7]" = convolution_backward_5[0]
    getitem_226: "f32[4096, 2048, 1, 1]" = convolution_backward_5[1];  convolution_backward_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    add_554: "f32[8, 2048, 7, 7]" = torch.ops.aten.add.Tensor(where_3, getitem_225);  where_3 = getitem_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    alias_119: "f32[8, 2048, 7, 7]" = torch.ops.aten.alias.default(relu_93);  relu_93 = None
    alias_120: "f32[8, 2048, 7, 7]" = torch.ops.aten.alias.default(alias_119);  alias_119 = None
    le_6: "b8[8, 2048, 7, 7]" = torch.ops.aten.le.Scalar(alias_120, 0);  alias_120 = None
    where_6: "f32[8, 2048, 7, 7]" = torch.ops.aten.where.self(le_6, full_default, add_554);  le_6 = add_554 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:200, code: shortcut = self.downsample(shortcut)
    sum_14: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_6, [0, 2, 3])
    sub_128: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_97, unsqueeze_490);  convolution_97 = unsqueeze_490 = None
    mul_782: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(where_6, sub_128)
    sum_15: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_782, [0, 2, 3]);  mul_782 = None
    mul_783: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_14, 0.002551020408163265)
    unsqueeze_491: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_783, 0);  mul_783 = None
    unsqueeze_492: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_491, 2);  unsqueeze_491 = None
    unsqueeze_493: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_492, 3);  unsqueeze_492 = None
    mul_784: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_15, 0.002551020408163265)
    mul_785: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_292, squeeze_292)
    mul_786: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_784, mul_785);  mul_784 = mul_785 = None
    unsqueeze_494: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_786, 0);  mul_786 = None
    unsqueeze_495: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_494, 2);  unsqueeze_494 = None
    unsqueeze_496: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_495, 3);  unsqueeze_495 = None
    mul_787: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_292, primals_293);  primals_293 = None
    unsqueeze_497: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_787, 0);  mul_787 = None
    unsqueeze_498: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_497, 2);  unsqueeze_497 = None
    unsqueeze_499: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_498, 3);  unsqueeze_498 = None
    mul_788: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_128, unsqueeze_496);  sub_128 = unsqueeze_496 = None
    sub_130: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(where_6, mul_788);  mul_788 = None
    sub_131: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(sub_130, unsqueeze_493);  sub_130 = None
    mul_789: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_131, unsqueeze_499);  sub_131 = unsqueeze_499 = None
    mul_790: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_15, squeeze_292);  sum_15 = squeeze_292 = None
    convolution_backward_6 = torch.ops.aten.convolution_backward.default(mul_789, relu_90, primals_292, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_789 = primals_292 = None
    getitem_228: "f32[8, 1024, 14, 14]" = convolution_backward_6[0]
    getitem_229: "f32[2048, 1024, 1, 1]" = convolution_backward_6[1];  convolution_backward_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    sub_132: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_96, unsqueeze_502);  convolution_96 = unsqueeze_502 = None
    mul_791: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(where_6, sub_132)
    sum_17: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_791, [0, 2, 3]);  mul_791 = None
    mul_793: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_17, 0.002551020408163265)
    mul_794: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_289, squeeze_289)
    mul_795: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_793, mul_794);  mul_793 = mul_794 = None
    unsqueeze_506: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_795, 0);  mul_795 = None
    unsqueeze_507: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_506, 2);  unsqueeze_506 = None
    unsqueeze_508: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_507, 3);  unsqueeze_507 = None
    mul_796: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_289, primals_290);  primals_290 = None
    unsqueeze_509: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_796, 0);  mul_796 = None
    unsqueeze_510: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_509, 2);  unsqueeze_509 = None
    unsqueeze_511: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_510, 3);  unsqueeze_510 = None
    mul_797: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_132, unsqueeze_508);  sub_132 = unsqueeze_508 = None
    sub_134: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(where_6, mul_797);  where_6 = mul_797 = None
    sub_135: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(sub_134, unsqueeze_493);  sub_134 = unsqueeze_493 = None
    mul_798: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_135, unsqueeze_511);  sub_135 = unsqueeze_511 = None
    mul_799: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_17, squeeze_289);  sum_17 = squeeze_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_backward_7 = torch.ops.aten.convolution_backward.default(mul_798, relu_92, primals_289, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_798 = primals_289 = None
    getitem_231: "f32[8, 4096, 7, 7]" = convolution_backward_7[0]
    getitem_232: "f32[2048, 4096, 1, 1]" = convolution_backward_7[1];  convolution_backward_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    alias_122: "f32[8, 4096, 7, 7]" = torch.ops.aten.alias.default(relu_92);  relu_92 = None
    alias_123: "f32[8, 4096, 7, 7]" = torch.ops.aten.alias.default(alias_122);  alias_122 = None
    le_7: "b8[8, 4096, 7, 7]" = torch.ops.aten.le.Scalar(alias_123, 0);  alias_123 = None
    where_7: "f32[8, 4096, 7, 7]" = torch.ops.aten.where.self(le_7, full_default, getitem_231);  le_7 = getitem_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    sum_18: "f32[4096]" = torch.ops.aten.sum.dim_IntList(where_7, [0, 2, 3])
    sub_136: "f32[8, 4096, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_95, unsqueeze_514);  convolution_95 = unsqueeze_514 = None
    mul_800: "f32[8, 4096, 7, 7]" = torch.ops.aten.mul.Tensor(where_7, sub_136)
    sum_19: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_800, [0, 2, 3]);  mul_800 = None
    mul_801: "f32[4096]" = torch.ops.aten.mul.Tensor(sum_18, 0.002551020408163265)
    unsqueeze_515: "f32[1, 4096]" = torch.ops.aten.unsqueeze.default(mul_801, 0);  mul_801 = None
    unsqueeze_516: "f32[1, 4096, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_515, 2);  unsqueeze_515 = None
    unsqueeze_517: "f32[1, 4096, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_516, 3);  unsqueeze_516 = None
    mul_802: "f32[4096]" = torch.ops.aten.mul.Tensor(sum_19, 0.002551020408163265)
    mul_803: "f32[4096]" = torch.ops.aten.mul.Tensor(squeeze_286, squeeze_286)
    mul_804: "f32[4096]" = torch.ops.aten.mul.Tensor(mul_802, mul_803);  mul_802 = mul_803 = None
    unsqueeze_518: "f32[1, 4096]" = torch.ops.aten.unsqueeze.default(mul_804, 0);  mul_804 = None
    unsqueeze_519: "f32[1, 4096, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_518, 2);  unsqueeze_518 = None
    unsqueeze_520: "f32[1, 4096, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_519, 3);  unsqueeze_519 = None
    mul_805: "f32[4096]" = torch.ops.aten.mul.Tensor(squeeze_286, primals_287);  primals_287 = None
    unsqueeze_521: "f32[1, 4096]" = torch.ops.aten.unsqueeze.default(mul_805, 0);  mul_805 = None
    unsqueeze_522: "f32[1, 4096, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_521, 2);  unsqueeze_521 = None
    unsqueeze_523: "f32[1, 4096, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_522, 3);  unsqueeze_522 = None
    mul_806: "f32[8, 4096, 7, 7]" = torch.ops.aten.mul.Tensor(sub_136, unsqueeze_520);  sub_136 = unsqueeze_520 = None
    sub_138: "f32[8, 4096, 7, 7]" = torch.ops.aten.sub.Tensor(where_7, mul_806);  where_7 = mul_806 = None
    sub_139: "f32[8, 4096, 7, 7]" = torch.ops.aten.sub.Tensor(sub_138, unsqueeze_517);  sub_138 = unsqueeze_517 = None
    mul_807: "f32[8, 4096, 7, 7]" = torch.ops.aten.mul.Tensor(sub_139, unsqueeze_523);  sub_139 = unsqueeze_523 = None
    mul_808: "f32[4096]" = torch.ops.aten.mul.Tensor(sum_19, squeeze_286);  sum_19 = squeeze_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_backward_8 = torch.ops.aten.convolution_backward.default(mul_807, relu_91, primals_286, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_807 = primals_286 = None
    getitem_234: "f32[8, 4096, 14, 14]" = convolution_backward_8[0]
    getitem_235: "f32[4096, 128, 3, 3]" = convolution_backward_8[1];  convolution_backward_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    alias_125: "f32[8, 4096, 14, 14]" = torch.ops.aten.alias.default(relu_91);  relu_91 = None
    alias_126: "f32[8, 4096, 14, 14]" = torch.ops.aten.alias.default(alias_125);  alias_125 = None
    le_8: "b8[8, 4096, 14, 14]" = torch.ops.aten.le.Scalar(alias_126, 0);  alias_126 = None
    where_8: "f32[8, 4096, 14, 14]" = torch.ops.aten.where.self(le_8, full_default, getitem_234);  le_8 = getitem_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    sum_20: "f32[4096]" = torch.ops.aten.sum.dim_IntList(where_8, [0, 2, 3])
    sub_140: "f32[8, 4096, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_94, unsqueeze_526);  convolution_94 = unsqueeze_526 = None
    mul_809: "f32[8, 4096, 14, 14]" = torch.ops.aten.mul.Tensor(where_8, sub_140)
    sum_21: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_809, [0, 2, 3]);  mul_809 = None
    mul_810: "f32[4096]" = torch.ops.aten.mul.Tensor(sum_20, 0.0006377551020408163)
    unsqueeze_527: "f32[1, 4096]" = torch.ops.aten.unsqueeze.default(mul_810, 0);  mul_810 = None
    unsqueeze_528: "f32[1, 4096, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_527, 2);  unsqueeze_527 = None
    unsqueeze_529: "f32[1, 4096, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_528, 3);  unsqueeze_528 = None
    mul_811: "f32[4096]" = torch.ops.aten.mul.Tensor(sum_21, 0.0006377551020408163)
    mul_812: "f32[4096]" = torch.ops.aten.mul.Tensor(squeeze_283, squeeze_283)
    mul_813: "f32[4096]" = torch.ops.aten.mul.Tensor(mul_811, mul_812);  mul_811 = mul_812 = None
    unsqueeze_530: "f32[1, 4096]" = torch.ops.aten.unsqueeze.default(mul_813, 0);  mul_813 = None
    unsqueeze_531: "f32[1, 4096, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_530, 2);  unsqueeze_530 = None
    unsqueeze_532: "f32[1, 4096, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_531, 3);  unsqueeze_531 = None
    mul_814: "f32[4096]" = torch.ops.aten.mul.Tensor(squeeze_283, primals_284);  primals_284 = None
    unsqueeze_533: "f32[1, 4096]" = torch.ops.aten.unsqueeze.default(mul_814, 0);  mul_814 = None
    unsqueeze_534: "f32[1, 4096, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_533, 2);  unsqueeze_533 = None
    unsqueeze_535: "f32[1, 4096, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_534, 3);  unsqueeze_534 = None
    mul_815: "f32[8, 4096, 14, 14]" = torch.ops.aten.mul.Tensor(sub_140, unsqueeze_532);  sub_140 = unsqueeze_532 = None
    sub_142: "f32[8, 4096, 14, 14]" = torch.ops.aten.sub.Tensor(where_8, mul_815);  where_8 = mul_815 = None
    sub_143: "f32[8, 4096, 14, 14]" = torch.ops.aten.sub.Tensor(sub_142, unsqueeze_529);  sub_142 = unsqueeze_529 = None
    mul_816: "f32[8, 4096, 14, 14]" = torch.ops.aten.mul.Tensor(sub_143, unsqueeze_535);  sub_143 = unsqueeze_535 = None
    mul_817: "f32[4096]" = torch.ops.aten.mul.Tensor(sum_21, squeeze_283);  sum_21 = squeeze_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_backward_9 = torch.ops.aten.convolution_backward.default(mul_816, relu_90, primals_283, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_816 = primals_283 = None
    getitem_237: "f32[8, 1024, 14, 14]" = convolution_backward_9[0]
    getitem_238: "f32[4096, 1024, 1, 1]" = convolution_backward_9[1];  convolution_backward_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    add_555: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(getitem_228, getitem_237);  getitem_228 = getitem_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    alias_128: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(relu_90);  relu_90 = None
    alias_129: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(alias_128);  alias_128 = None
    le_9: "b8[8, 1024, 14, 14]" = torch.ops.aten.le.Scalar(alias_129, 0);  alias_129 = None
    where_9: "f32[8, 1024, 14, 14]" = torch.ops.aten.where.self(le_9, full_default, add_555);  le_9 = add_555 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    sum_22: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_9, [0, 2, 3])
    sub_144: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_93, unsqueeze_538);  convolution_93 = unsqueeze_538 = None
    mul_818: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_9, sub_144)
    sum_23: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_818, [0, 2, 3]);  mul_818 = None
    mul_819: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_22, 0.0006377551020408163)
    unsqueeze_539: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_819, 0);  mul_819 = None
    unsqueeze_540: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_539, 2);  unsqueeze_539 = None
    unsqueeze_541: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_540, 3);  unsqueeze_540 = None
    mul_820: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_23, 0.0006377551020408163)
    mul_821: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_280, squeeze_280)
    mul_822: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_820, mul_821);  mul_820 = mul_821 = None
    unsqueeze_542: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_822, 0);  mul_822 = None
    unsqueeze_543: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_542, 2);  unsqueeze_542 = None
    unsqueeze_544: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_543, 3);  unsqueeze_543 = None
    mul_823: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_280, primals_281);  primals_281 = None
    unsqueeze_545: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_823, 0);  mul_823 = None
    unsqueeze_546: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_545, 2);  unsqueeze_545 = None
    unsqueeze_547: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_546, 3);  unsqueeze_546 = None
    mul_824: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_144, unsqueeze_544);  sub_144 = unsqueeze_544 = None
    sub_146: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_9, mul_824);  mul_824 = None
    sub_147: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_146, unsqueeze_541);  sub_146 = unsqueeze_541 = None
    mul_825: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_147, unsqueeze_547);  sub_147 = unsqueeze_547 = None
    mul_826: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_23, squeeze_280);  sum_23 = squeeze_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_backward_10 = torch.ops.aten.convolution_backward.default(mul_825, relu_89, primals_280, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_825 = primals_280 = None
    getitem_240: "f32[8, 2048, 14, 14]" = convolution_backward_10[0]
    getitem_241: "f32[1024, 2048, 1, 1]" = convolution_backward_10[1];  convolution_backward_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    alias_131: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(relu_89);  relu_89 = None
    alias_132: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(alias_131);  alias_131 = None
    le_10: "b8[8, 2048, 14, 14]" = torch.ops.aten.le.Scalar(alias_132, 0);  alias_132 = None
    where_10: "f32[8, 2048, 14, 14]" = torch.ops.aten.where.self(le_10, full_default, getitem_240);  le_10 = getitem_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    sum_24: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_10, [0, 2, 3])
    sub_148: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_92, unsqueeze_550);  convolution_92 = unsqueeze_550 = None
    mul_827: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(where_10, sub_148)
    sum_25: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_827, [0, 2, 3]);  mul_827 = None
    mul_828: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_24, 0.0006377551020408163)
    unsqueeze_551: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_828, 0);  mul_828 = None
    unsqueeze_552: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_551, 2);  unsqueeze_551 = None
    unsqueeze_553: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_552, 3);  unsqueeze_552 = None
    mul_829: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_25, 0.0006377551020408163)
    mul_830: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_277, squeeze_277)
    mul_831: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_829, mul_830);  mul_829 = mul_830 = None
    unsqueeze_554: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_831, 0);  mul_831 = None
    unsqueeze_555: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_554, 2);  unsqueeze_554 = None
    unsqueeze_556: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_555, 3);  unsqueeze_555 = None
    mul_832: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_277, primals_278);  primals_278 = None
    unsqueeze_557: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_832, 0);  mul_832 = None
    unsqueeze_558: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_557, 2);  unsqueeze_557 = None
    unsqueeze_559: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_558, 3);  unsqueeze_558 = None
    mul_833: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_148, unsqueeze_556);  sub_148 = unsqueeze_556 = None
    sub_150: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(where_10, mul_833);  where_10 = mul_833 = None
    sub_151: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(sub_150, unsqueeze_553);  sub_150 = unsqueeze_553 = None
    mul_834: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_151, unsqueeze_559);  sub_151 = unsqueeze_559 = None
    mul_835: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_25, squeeze_277);  sum_25 = squeeze_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_backward_11 = torch.ops.aten.convolution_backward.default(mul_834, relu_88, primals_277, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_834 = primals_277 = None
    getitem_243: "f32[8, 2048, 14, 14]" = convolution_backward_11[0]
    getitem_244: "f32[2048, 64, 3, 3]" = convolution_backward_11[1];  convolution_backward_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    alias_134: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(relu_88);  relu_88 = None
    alias_135: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(alias_134);  alias_134 = None
    le_11: "b8[8, 2048, 14, 14]" = torch.ops.aten.le.Scalar(alias_135, 0);  alias_135 = None
    where_11: "f32[8, 2048, 14, 14]" = torch.ops.aten.where.self(le_11, full_default, getitem_243);  le_11 = getitem_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    sum_26: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_11, [0, 2, 3])
    sub_152: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_91, unsqueeze_562);  convolution_91 = unsqueeze_562 = None
    mul_836: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(where_11, sub_152)
    sum_27: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_836, [0, 2, 3]);  mul_836 = None
    mul_837: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_26, 0.0006377551020408163)
    unsqueeze_563: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_837, 0);  mul_837 = None
    unsqueeze_564: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_563, 2);  unsqueeze_563 = None
    unsqueeze_565: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_564, 3);  unsqueeze_564 = None
    mul_838: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_27, 0.0006377551020408163)
    mul_839: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_274, squeeze_274)
    mul_840: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_838, mul_839);  mul_838 = mul_839 = None
    unsqueeze_566: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_840, 0);  mul_840 = None
    unsqueeze_567: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_566, 2);  unsqueeze_566 = None
    unsqueeze_568: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_567, 3);  unsqueeze_567 = None
    mul_841: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_274, primals_275);  primals_275 = None
    unsqueeze_569: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_841, 0);  mul_841 = None
    unsqueeze_570: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_569, 2);  unsqueeze_569 = None
    unsqueeze_571: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_570, 3);  unsqueeze_570 = None
    mul_842: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_152, unsqueeze_568);  sub_152 = unsqueeze_568 = None
    sub_154: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(where_11, mul_842);  where_11 = mul_842 = None
    sub_155: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(sub_154, unsqueeze_565);  sub_154 = unsqueeze_565 = None
    mul_843: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_155, unsqueeze_571);  sub_155 = unsqueeze_571 = None
    mul_844: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_27, squeeze_274);  sum_27 = squeeze_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_backward_12 = torch.ops.aten.convolution_backward.default(mul_843, relu_87, primals_274, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_843 = primals_274 = None
    getitem_246: "f32[8, 1024, 14, 14]" = convolution_backward_12[0]
    getitem_247: "f32[2048, 1024, 1, 1]" = convolution_backward_12[1];  convolution_backward_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    add_556: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(where_9, getitem_246);  where_9 = getitem_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    alias_137: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(relu_87);  relu_87 = None
    alias_138: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(alias_137);  alias_137 = None
    le_12: "b8[8, 1024, 14, 14]" = torch.ops.aten.le.Scalar(alias_138, 0);  alias_138 = None
    where_12: "f32[8, 1024, 14, 14]" = torch.ops.aten.where.self(le_12, full_default, add_556);  le_12 = add_556 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    sum_28: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_12, [0, 2, 3])
    sub_156: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_90, unsqueeze_574);  convolution_90 = unsqueeze_574 = None
    mul_845: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_12, sub_156)
    sum_29: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_845, [0, 2, 3]);  mul_845 = None
    mul_846: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_28, 0.0006377551020408163)
    unsqueeze_575: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_846, 0);  mul_846 = None
    unsqueeze_576: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_575, 2);  unsqueeze_575 = None
    unsqueeze_577: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_576, 3);  unsqueeze_576 = None
    mul_847: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_29, 0.0006377551020408163)
    mul_848: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_271, squeeze_271)
    mul_849: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_847, mul_848);  mul_847 = mul_848 = None
    unsqueeze_578: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_849, 0);  mul_849 = None
    unsqueeze_579: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_578, 2);  unsqueeze_578 = None
    unsqueeze_580: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_579, 3);  unsqueeze_579 = None
    mul_850: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_271, primals_272);  primals_272 = None
    unsqueeze_581: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_850, 0);  mul_850 = None
    unsqueeze_582: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_581, 2);  unsqueeze_581 = None
    unsqueeze_583: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_582, 3);  unsqueeze_582 = None
    mul_851: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_156, unsqueeze_580);  sub_156 = unsqueeze_580 = None
    sub_158: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_12, mul_851);  mul_851 = None
    sub_159: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_158, unsqueeze_577);  sub_158 = unsqueeze_577 = None
    mul_852: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_159, unsqueeze_583);  sub_159 = unsqueeze_583 = None
    mul_853: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_29, squeeze_271);  sum_29 = squeeze_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_backward_13 = torch.ops.aten.convolution_backward.default(mul_852, relu_86, primals_271, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_852 = primals_271 = None
    getitem_249: "f32[8, 2048, 14, 14]" = convolution_backward_13[0]
    getitem_250: "f32[1024, 2048, 1, 1]" = convolution_backward_13[1];  convolution_backward_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    alias_140: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(relu_86);  relu_86 = None
    alias_141: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(alias_140);  alias_140 = None
    le_13: "b8[8, 2048, 14, 14]" = torch.ops.aten.le.Scalar(alias_141, 0);  alias_141 = None
    where_13: "f32[8, 2048, 14, 14]" = torch.ops.aten.where.self(le_13, full_default, getitem_249);  le_13 = getitem_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    sum_30: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_13, [0, 2, 3])
    sub_160: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_89, unsqueeze_586);  convolution_89 = unsqueeze_586 = None
    mul_854: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(where_13, sub_160)
    sum_31: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_854, [0, 2, 3]);  mul_854 = None
    mul_855: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_30, 0.0006377551020408163)
    unsqueeze_587: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_855, 0);  mul_855 = None
    unsqueeze_588: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_587, 2);  unsqueeze_587 = None
    unsqueeze_589: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_588, 3);  unsqueeze_588 = None
    mul_856: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_31, 0.0006377551020408163)
    mul_857: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_268, squeeze_268)
    mul_858: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_856, mul_857);  mul_856 = mul_857 = None
    unsqueeze_590: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_858, 0);  mul_858 = None
    unsqueeze_591: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_590, 2);  unsqueeze_590 = None
    unsqueeze_592: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_591, 3);  unsqueeze_591 = None
    mul_859: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_268, primals_269);  primals_269 = None
    unsqueeze_593: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_859, 0);  mul_859 = None
    unsqueeze_594: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_593, 2);  unsqueeze_593 = None
    unsqueeze_595: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_594, 3);  unsqueeze_594 = None
    mul_860: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_160, unsqueeze_592);  sub_160 = unsqueeze_592 = None
    sub_162: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(where_13, mul_860);  where_13 = mul_860 = None
    sub_163: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(sub_162, unsqueeze_589);  sub_162 = unsqueeze_589 = None
    mul_861: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_163, unsqueeze_595);  sub_163 = unsqueeze_595 = None
    mul_862: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_31, squeeze_268);  sum_31 = squeeze_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_backward_14 = torch.ops.aten.convolution_backward.default(mul_861, relu_85, primals_268, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_861 = primals_268 = None
    getitem_252: "f32[8, 2048, 14, 14]" = convolution_backward_14[0]
    getitem_253: "f32[2048, 64, 3, 3]" = convolution_backward_14[1];  convolution_backward_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    alias_143: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(relu_85);  relu_85 = None
    alias_144: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(alias_143);  alias_143 = None
    le_14: "b8[8, 2048, 14, 14]" = torch.ops.aten.le.Scalar(alias_144, 0);  alias_144 = None
    where_14: "f32[8, 2048, 14, 14]" = torch.ops.aten.where.self(le_14, full_default, getitem_252);  le_14 = getitem_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    sum_32: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_14, [0, 2, 3])
    sub_164: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_88, unsqueeze_598);  convolution_88 = unsqueeze_598 = None
    mul_863: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(where_14, sub_164)
    sum_33: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_863, [0, 2, 3]);  mul_863 = None
    mul_864: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_32, 0.0006377551020408163)
    unsqueeze_599: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_864, 0);  mul_864 = None
    unsqueeze_600: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_599, 2);  unsqueeze_599 = None
    unsqueeze_601: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_600, 3);  unsqueeze_600 = None
    mul_865: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_33, 0.0006377551020408163)
    mul_866: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_265, squeeze_265)
    mul_867: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_865, mul_866);  mul_865 = mul_866 = None
    unsqueeze_602: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_867, 0);  mul_867 = None
    unsqueeze_603: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_602, 2);  unsqueeze_602 = None
    unsqueeze_604: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_603, 3);  unsqueeze_603 = None
    mul_868: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_265, primals_266);  primals_266 = None
    unsqueeze_605: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_868, 0);  mul_868 = None
    unsqueeze_606: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_605, 2);  unsqueeze_605 = None
    unsqueeze_607: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_606, 3);  unsqueeze_606 = None
    mul_869: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_164, unsqueeze_604);  sub_164 = unsqueeze_604 = None
    sub_166: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(where_14, mul_869);  where_14 = mul_869 = None
    sub_167: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(sub_166, unsqueeze_601);  sub_166 = unsqueeze_601 = None
    mul_870: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_167, unsqueeze_607);  sub_167 = unsqueeze_607 = None
    mul_871: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_33, squeeze_265);  sum_33 = squeeze_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_backward_15 = torch.ops.aten.convolution_backward.default(mul_870, relu_84, primals_265, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_870 = primals_265 = None
    getitem_255: "f32[8, 1024, 14, 14]" = convolution_backward_15[0]
    getitem_256: "f32[2048, 1024, 1, 1]" = convolution_backward_15[1];  convolution_backward_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    add_557: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(where_12, getitem_255);  where_12 = getitem_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    alias_146: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(relu_84);  relu_84 = None
    alias_147: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(alias_146);  alias_146 = None
    le_15: "b8[8, 1024, 14, 14]" = torch.ops.aten.le.Scalar(alias_147, 0);  alias_147 = None
    where_15: "f32[8, 1024, 14, 14]" = torch.ops.aten.where.self(le_15, full_default, add_557);  le_15 = add_557 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    sum_34: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_15, [0, 2, 3])
    sub_168: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_87, unsqueeze_610);  convolution_87 = unsqueeze_610 = None
    mul_872: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_15, sub_168)
    sum_35: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_872, [0, 2, 3]);  mul_872 = None
    mul_873: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_34, 0.0006377551020408163)
    unsqueeze_611: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_873, 0);  mul_873 = None
    unsqueeze_612: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_611, 2);  unsqueeze_611 = None
    unsqueeze_613: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_612, 3);  unsqueeze_612 = None
    mul_874: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_35, 0.0006377551020408163)
    mul_875: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_262, squeeze_262)
    mul_876: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_874, mul_875);  mul_874 = mul_875 = None
    unsqueeze_614: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_876, 0);  mul_876 = None
    unsqueeze_615: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_614, 2);  unsqueeze_614 = None
    unsqueeze_616: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_615, 3);  unsqueeze_615 = None
    mul_877: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_262, primals_263);  primals_263 = None
    unsqueeze_617: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_877, 0);  mul_877 = None
    unsqueeze_618: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_617, 2);  unsqueeze_617 = None
    unsqueeze_619: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_618, 3);  unsqueeze_618 = None
    mul_878: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_168, unsqueeze_616);  sub_168 = unsqueeze_616 = None
    sub_170: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_15, mul_878);  mul_878 = None
    sub_171: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_170, unsqueeze_613);  sub_170 = unsqueeze_613 = None
    mul_879: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_171, unsqueeze_619);  sub_171 = unsqueeze_619 = None
    mul_880: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_35, squeeze_262);  sum_35 = squeeze_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_backward_16 = torch.ops.aten.convolution_backward.default(mul_879, relu_83, primals_262, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_879 = primals_262 = None
    getitem_258: "f32[8, 2048, 14, 14]" = convolution_backward_16[0]
    getitem_259: "f32[1024, 2048, 1, 1]" = convolution_backward_16[1];  convolution_backward_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    alias_149: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(relu_83);  relu_83 = None
    alias_150: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(alias_149);  alias_149 = None
    le_16: "b8[8, 2048, 14, 14]" = torch.ops.aten.le.Scalar(alias_150, 0);  alias_150 = None
    where_16: "f32[8, 2048, 14, 14]" = torch.ops.aten.where.self(le_16, full_default, getitem_258);  le_16 = getitem_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    sum_36: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_16, [0, 2, 3])
    sub_172: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_86, unsqueeze_622);  convolution_86 = unsqueeze_622 = None
    mul_881: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(where_16, sub_172)
    sum_37: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_881, [0, 2, 3]);  mul_881 = None
    mul_882: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_36, 0.0006377551020408163)
    unsqueeze_623: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_882, 0);  mul_882 = None
    unsqueeze_624: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_623, 2);  unsqueeze_623 = None
    unsqueeze_625: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_624, 3);  unsqueeze_624 = None
    mul_883: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_37, 0.0006377551020408163)
    mul_884: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_259, squeeze_259)
    mul_885: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_883, mul_884);  mul_883 = mul_884 = None
    unsqueeze_626: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_885, 0);  mul_885 = None
    unsqueeze_627: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_626, 2);  unsqueeze_626 = None
    unsqueeze_628: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_627, 3);  unsqueeze_627 = None
    mul_886: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_259, primals_260);  primals_260 = None
    unsqueeze_629: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_886, 0);  mul_886 = None
    unsqueeze_630: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_629, 2);  unsqueeze_629 = None
    unsqueeze_631: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_630, 3);  unsqueeze_630 = None
    mul_887: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_172, unsqueeze_628);  sub_172 = unsqueeze_628 = None
    sub_174: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(where_16, mul_887);  where_16 = mul_887 = None
    sub_175: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(sub_174, unsqueeze_625);  sub_174 = unsqueeze_625 = None
    mul_888: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_175, unsqueeze_631);  sub_175 = unsqueeze_631 = None
    mul_889: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_37, squeeze_259);  sum_37 = squeeze_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_backward_17 = torch.ops.aten.convolution_backward.default(mul_888, relu_82, primals_259, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_888 = primals_259 = None
    getitem_261: "f32[8, 2048, 14, 14]" = convolution_backward_17[0]
    getitem_262: "f32[2048, 64, 3, 3]" = convolution_backward_17[1];  convolution_backward_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    alias_152: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(relu_82);  relu_82 = None
    alias_153: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(alias_152);  alias_152 = None
    le_17: "b8[8, 2048, 14, 14]" = torch.ops.aten.le.Scalar(alias_153, 0);  alias_153 = None
    where_17: "f32[8, 2048, 14, 14]" = torch.ops.aten.where.self(le_17, full_default, getitem_261);  le_17 = getitem_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    sum_38: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_17, [0, 2, 3])
    sub_176: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_85, unsqueeze_634);  convolution_85 = unsqueeze_634 = None
    mul_890: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(where_17, sub_176)
    sum_39: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_890, [0, 2, 3]);  mul_890 = None
    mul_891: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_38, 0.0006377551020408163)
    unsqueeze_635: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_891, 0);  mul_891 = None
    unsqueeze_636: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_635, 2);  unsqueeze_635 = None
    unsqueeze_637: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_636, 3);  unsqueeze_636 = None
    mul_892: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_39, 0.0006377551020408163)
    mul_893: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_256, squeeze_256)
    mul_894: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_892, mul_893);  mul_892 = mul_893 = None
    unsqueeze_638: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_894, 0);  mul_894 = None
    unsqueeze_639: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_638, 2);  unsqueeze_638 = None
    unsqueeze_640: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_639, 3);  unsqueeze_639 = None
    mul_895: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_256, primals_257);  primals_257 = None
    unsqueeze_641: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_895, 0);  mul_895 = None
    unsqueeze_642: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_641, 2);  unsqueeze_641 = None
    unsqueeze_643: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_642, 3);  unsqueeze_642 = None
    mul_896: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_176, unsqueeze_640);  sub_176 = unsqueeze_640 = None
    sub_178: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(where_17, mul_896);  where_17 = mul_896 = None
    sub_179: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(sub_178, unsqueeze_637);  sub_178 = unsqueeze_637 = None
    mul_897: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_179, unsqueeze_643);  sub_179 = unsqueeze_643 = None
    mul_898: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_39, squeeze_256);  sum_39 = squeeze_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_backward_18 = torch.ops.aten.convolution_backward.default(mul_897, relu_81, primals_256, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_897 = primals_256 = None
    getitem_264: "f32[8, 1024, 14, 14]" = convolution_backward_18[0]
    getitem_265: "f32[2048, 1024, 1, 1]" = convolution_backward_18[1];  convolution_backward_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    add_558: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(where_15, getitem_264);  where_15 = getitem_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    alias_155: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(relu_81);  relu_81 = None
    alias_156: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(alias_155);  alias_155 = None
    le_18: "b8[8, 1024, 14, 14]" = torch.ops.aten.le.Scalar(alias_156, 0);  alias_156 = None
    where_18: "f32[8, 1024, 14, 14]" = torch.ops.aten.where.self(le_18, full_default, add_558);  le_18 = add_558 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    sum_40: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_18, [0, 2, 3])
    sub_180: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_84, unsqueeze_646);  convolution_84 = unsqueeze_646 = None
    mul_899: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_18, sub_180)
    sum_41: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_899, [0, 2, 3]);  mul_899 = None
    mul_900: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_40, 0.0006377551020408163)
    unsqueeze_647: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_900, 0);  mul_900 = None
    unsqueeze_648: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_647, 2);  unsqueeze_647 = None
    unsqueeze_649: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_648, 3);  unsqueeze_648 = None
    mul_901: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_41, 0.0006377551020408163)
    mul_902: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_253, squeeze_253)
    mul_903: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_901, mul_902);  mul_901 = mul_902 = None
    unsqueeze_650: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_903, 0);  mul_903 = None
    unsqueeze_651: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_650, 2);  unsqueeze_650 = None
    unsqueeze_652: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_651, 3);  unsqueeze_651 = None
    mul_904: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_253, primals_254);  primals_254 = None
    unsqueeze_653: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_904, 0);  mul_904 = None
    unsqueeze_654: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_653, 2);  unsqueeze_653 = None
    unsqueeze_655: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_654, 3);  unsqueeze_654 = None
    mul_905: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_180, unsqueeze_652);  sub_180 = unsqueeze_652 = None
    sub_182: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_18, mul_905);  mul_905 = None
    sub_183: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_182, unsqueeze_649);  sub_182 = unsqueeze_649 = None
    mul_906: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_183, unsqueeze_655);  sub_183 = unsqueeze_655 = None
    mul_907: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_41, squeeze_253);  sum_41 = squeeze_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_backward_19 = torch.ops.aten.convolution_backward.default(mul_906, relu_80, primals_253, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_906 = primals_253 = None
    getitem_267: "f32[8, 2048, 14, 14]" = convolution_backward_19[0]
    getitem_268: "f32[1024, 2048, 1, 1]" = convolution_backward_19[1];  convolution_backward_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    alias_158: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(relu_80);  relu_80 = None
    alias_159: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(alias_158);  alias_158 = None
    le_19: "b8[8, 2048, 14, 14]" = torch.ops.aten.le.Scalar(alias_159, 0);  alias_159 = None
    where_19: "f32[8, 2048, 14, 14]" = torch.ops.aten.where.self(le_19, full_default, getitem_267);  le_19 = getitem_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    sum_42: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_19, [0, 2, 3])
    sub_184: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_83, unsqueeze_658);  convolution_83 = unsqueeze_658 = None
    mul_908: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(where_19, sub_184)
    sum_43: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_908, [0, 2, 3]);  mul_908 = None
    mul_909: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_42, 0.0006377551020408163)
    unsqueeze_659: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_909, 0);  mul_909 = None
    unsqueeze_660: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_659, 2);  unsqueeze_659 = None
    unsqueeze_661: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_660, 3);  unsqueeze_660 = None
    mul_910: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_43, 0.0006377551020408163)
    mul_911: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_250, squeeze_250)
    mul_912: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_910, mul_911);  mul_910 = mul_911 = None
    unsqueeze_662: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_912, 0);  mul_912 = None
    unsqueeze_663: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_662, 2);  unsqueeze_662 = None
    unsqueeze_664: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_663, 3);  unsqueeze_663 = None
    mul_913: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_250, primals_251);  primals_251 = None
    unsqueeze_665: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_913, 0);  mul_913 = None
    unsqueeze_666: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_665, 2);  unsqueeze_665 = None
    unsqueeze_667: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_666, 3);  unsqueeze_666 = None
    mul_914: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_184, unsqueeze_664);  sub_184 = unsqueeze_664 = None
    sub_186: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(where_19, mul_914);  where_19 = mul_914 = None
    sub_187: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(sub_186, unsqueeze_661);  sub_186 = unsqueeze_661 = None
    mul_915: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_187, unsqueeze_667);  sub_187 = unsqueeze_667 = None
    mul_916: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_43, squeeze_250);  sum_43 = squeeze_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_backward_20 = torch.ops.aten.convolution_backward.default(mul_915, relu_79, primals_250, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_915 = primals_250 = None
    getitem_270: "f32[8, 2048, 14, 14]" = convolution_backward_20[0]
    getitem_271: "f32[2048, 64, 3, 3]" = convolution_backward_20[1];  convolution_backward_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    alias_161: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(relu_79);  relu_79 = None
    alias_162: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(alias_161);  alias_161 = None
    le_20: "b8[8, 2048, 14, 14]" = torch.ops.aten.le.Scalar(alias_162, 0);  alias_162 = None
    where_20: "f32[8, 2048, 14, 14]" = torch.ops.aten.where.self(le_20, full_default, getitem_270);  le_20 = getitem_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    sum_44: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_20, [0, 2, 3])
    sub_188: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_82, unsqueeze_670);  convolution_82 = unsqueeze_670 = None
    mul_917: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(where_20, sub_188)
    sum_45: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_917, [0, 2, 3]);  mul_917 = None
    mul_918: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_44, 0.0006377551020408163)
    unsqueeze_671: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_918, 0);  mul_918 = None
    unsqueeze_672: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_671, 2);  unsqueeze_671 = None
    unsqueeze_673: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_672, 3);  unsqueeze_672 = None
    mul_919: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_45, 0.0006377551020408163)
    mul_920: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_247, squeeze_247)
    mul_921: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_919, mul_920);  mul_919 = mul_920 = None
    unsqueeze_674: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_921, 0);  mul_921 = None
    unsqueeze_675: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_674, 2);  unsqueeze_674 = None
    unsqueeze_676: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_675, 3);  unsqueeze_675 = None
    mul_922: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_247, primals_248);  primals_248 = None
    unsqueeze_677: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_922, 0);  mul_922 = None
    unsqueeze_678: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_677, 2);  unsqueeze_677 = None
    unsqueeze_679: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_678, 3);  unsqueeze_678 = None
    mul_923: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_188, unsqueeze_676);  sub_188 = unsqueeze_676 = None
    sub_190: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(where_20, mul_923);  where_20 = mul_923 = None
    sub_191: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(sub_190, unsqueeze_673);  sub_190 = unsqueeze_673 = None
    mul_924: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_191, unsqueeze_679);  sub_191 = unsqueeze_679 = None
    mul_925: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_45, squeeze_247);  sum_45 = squeeze_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_backward_21 = torch.ops.aten.convolution_backward.default(mul_924, relu_78, primals_247, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_924 = primals_247 = None
    getitem_273: "f32[8, 1024, 14, 14]" = convolution_backward_21[0]
    getitem_274: "f32[2048, 1024, 1, 1]" = convolution_backward_21[1];  convolution_backward_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    add_559: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(where_18, getitem_273);  where_18 = getitem_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    alias_164: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(relu_78);  relu_78 = None
    alias_165: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(alias_164);  alias_164 = None
    le_21: "b8[8, 1024, 14, 14]" = torch.ops.aten.le.Scalar(alias_165, 0);  alias_165 = None
    where_21: "f32[8, 1024, 14, 14]" = torch.ops.aten.where.self(le_21, full_default, add_559);  le_21 = add_559 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    sum_46: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_21, [0, 2, 3])
    sub_192: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_81, unsqueeze_682);  convolution_81 = unsqueeze_682 = None
    mul_926: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_21, sub_192)
    sum_47: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_926, [0, 2, 3]);  mul_926 = None
    mul_927: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_46, 0.0006377551020408163)
    unsqueeze_683: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_927, 0);  mul_927 = None
    unsqueeze_684: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_683, 2);  unsqueeze_683 = None
    unsqueeze_685: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_684, 3);  unsqueeze_684 = None
    mul_928: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_47, 0.0006377551020408163)
    mul_929: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_244, squeeze_244)
    mul_930: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_928, mul_929);  mul_928 = mul_929 = None
    unsqueeze_686: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_930, 0);  mul_930 = None
    unsqueeze_687: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_686, 2);  unsqueeze_686 = None
    unsqueeze_688: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_687, 3);  unsqueeze_687 = None
    mul_931: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_244, primals_245);  primals_245 = None
    unsqueeze_689: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_931, 0);  mul_931 = None
    unsqueeze_690: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_689, 2);  unsqueeze_689 = None
    unsqueeze_691: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_690, 3);  unsqueeze_690 = None
    mul_932: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_192, unsqueeze_688);  sub_192 = unsqueeze_688 = None
    sub_194: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_21, mul_932);  mul_932 = None
    sub_195: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_194, unsqueeze_685);  sub_194 = unsqueeze_685 = None
    mul_933: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_195, unsqueeze_691);  sub_195 = unsqueeze_691 = None
    mul_934: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_47, squeeze_244);  sum_47 = squeeze_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_backward_22 = torch.ops.aten.convolution_backward.default(mul_933, relu_77, primals_244, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_933 = primals_244 = None
    getitem_276: "f32[8, 2048, 14, 14]" = convolution_backward_22[0]
    getitem_277: "f32[1024, 2048, 1, 1]" = convolution_backward_22[1];  convolution_backward_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    alias_167: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(relu_77);  relu_77 = None
    alias_168: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(alias_167);  alias_167 = None
    le_22: "b8[8, 2048, 14, 14]" = torch.ops.aten.le.Scalar(alias_168, 0);  alias_168 = None
    where_22: "f32[8, 2048, 14, 14]" = torch.ops.aten.where.self(le_22, full_default, getitem_276);  le_22 = getitem_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    sum_48: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_22, [0, 2, 3])
    sub_196: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_80, unsqueeze_694);  convolution_80 = unsqueeze_694 = None
    mul_935: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(where_22, sub_196)
    sum_49: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_935, [0, 2, 3]);  mul_935 = None
    mul_936: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_48, 0.0006377551020408163)
    unsqueeze_695: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_936, 0);  mul_936 = None
    unsqueeze_696: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_695, 2);  unsqueeze_695 = None
    unsqueeze_697: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_696, 3);  unsqueeze_696 = None
    mul_937: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_49, 0.0006377551020408163)
    mul_938: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_241, squeeze_241)
    mul_939: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_937, mul_938);  mul_937 = mul_938 = None
    unsqueeze_698: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_939, 0);  mul_939 = None
    unsqueeze_699: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_698, 2);  unsqueeze_698 = None
    unsqueeze_700: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_699, 3);  unsqueeze_699 = None
    mul_940: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_241, primals_242);  primals_242 = None
    unsqueeze_701: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_940, 0);  mul_940 = None
    unsqueeze_702: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_701, 2);  unsqueeze_701 = None
    unsqueeze_703: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_702, 3);  unsqueeze_702 = None
    mul_941: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_196, unsqueeze_700);  sub_196 = unsqueeze_700 = None
    sub_198: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(where_22, mul_941);  where_22 = mul_941 = None
    sub_199: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(sub_198, unsqueeze_697);  sub_198 = unsqueeze_697 = None
    mul_942: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_199, unsqueeze_703);  sub_199 = unsqueeze_703 = None
    mul_943: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_49, squeeze_241);  sum_49 = squeeze_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_backward_23 = torch.ops.aten.convolution_backward.default(mul_942, relu_76, primals_241, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_942 = primals_241 = None
    getitem_279: "f32[8, 2048, 14, 14]" = convolution_backward_23[0]
    getitem_280: "f32[2048, 64, 3, 3]" = convolution_backward_23[1];  convolution_backward_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    alias_170: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(relu_76);  relu_76 = None
    alias_171: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(alias_170);  alias_170 = None
    le_23: "b8[8, 2048, 14, 14]" = torch.ops.aten.le.Scalar(alias_171, 0);  alias_171 = None
    where_23: "f32[8, 2048, 14, 14]" = torch.ops.aten.where.self(le_23, full_default, getitem_279);  le_23 = getitem_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    sum_50: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_23, [0, 2, 3])
    sub_200: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_79, unsqueeze_706);  convolution_79 = unsqueeze_706 = None
    mul_944: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(where_23, sub_200)
    sum_51: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_944, [0, 2, 3]);  mul_944 = None
    mul_945: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_50, 0.0006377551020408163)
    unsqueeze_707: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_945, 0);  mul_945 = None
    unsqueeze_708: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_707, 2);  unsqueeze_707 = None
    unsqueeze_709: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_708, 3);  unsqueeze_708 = None
    mul_946: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_51, 0.0006377551020408163)
    mul_947: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_238, squeeze_238)
    mul_948: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_946, mul_947);  mul_946 = mul_947 = None
    unsqueeze_710: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_948, 0);  mul_948 = None
    unsqueeze_711: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_710, 2);  unsqueeze_710 = None
    unsqueeze_712: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_711, 3);  unsqueeze_711 = None
    mul_949: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_238, primals_239);  primals_239 = None
    unsqueeze_713: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_949, 0);  mul_949 = None
    unsqueeze_714: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_713, 2);  unsqueeze_713 = None
    unsqueeze_715: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_714, 3);  unsqueeze_714 = None
    mul_950: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_200, unsqueeze_712);  sub_200 = unsqueeze_712 = None
    sub_202: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(where_23, mul_950);  where_23 = mul_950 = None
    sub_203: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(sub_202, unsqueeze_709);  sub_202 = unsqueeze_709 = None
    mul_951: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_203, unsqueeze_715);  sub_203 = unsqueeze_715 = None
    mul_952: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_51, squeeze_238);  sum_51 = squeeze_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_backward_24 = torch.ops.aten.convolution_backward.default(mul_951, relu_75, primals_238, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_951 = primals_238 = None
    getitem_282: "f32[8, 1024, 14, 14]" = convolution_backward_24[0]
    getitem_283: "f32[2048, 1024, 1, 1]" = convolution_backward_24[1];  convolution_backward_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    add_560: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(where_21, getitem_282);  where_21 = getitem_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    alias_173: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(relu_75);  relu_75 = None
    alias_174: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(alias_173);  alias_173 = None
    le_24: "b8[8, 1024, 14, 14]" = torch.ops.aten.le.Scalar(alias_174, 0);  alias_174 = None
    where_24: "f32[8, 1024, 14, 14]" = torch.ops.aten.where.self(le_24, full_default, add_560);  le_24 = add_560 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    sum_52: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_24, [0, 2, 3])
    sub_204: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_78, unsqueeze_718);  convolution_78 = unsqueeze_718 = None
    mul_953: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_24, sub_204)
    sum_53: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_953, [0, 2, 3]);  mul_953 = None
    mul_954: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_52, 0.0006377551020408163)
    unsqueeze_719: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_954, 0);  mul_954 = None
    unsqueeze_720: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_719, 2);  unsqueeze_719 = None
    unsqueeze_721: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_720, 3);  unsqueeze_720 = None
    mul_955: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_53, 0.0006377551020408163)
    mul_956: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_235, squeeze_235)
    mul_957: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_955, mul_956);  mul_955 = mul_956 = None
    unsqueeze_722: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_957, 0);  mul_957 = None
    unsqueeze_723: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_722, 2);  unsqueeze_722 = None
    unsqueeze_724: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_723, 3);  unsqueeze_723 = None
    mul_958: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_235, primals_236);  primals_236 = None
    unsqueeze_725: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_958, 0);  mul_958 = None
    unsqueeze_726: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_725, 2);  unsqueeze_725 = None
    unsqueeze_727: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_726, 3);  unsqueeze_726 = None
    mul_959: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_204, unsqueeze_724);  sub_204 = unsqueeze_724 = None
    sub_206: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_24, mul_959);  mul_959 = None
    sub_207: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_206, unsqueeze_721);  sub_206 = unsqueeze_721 = None
    mul_960: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_207, unsqueeze_727);  sub_207 = unsqueeze_727 = None
    mul_961: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_53, squeeze_235);  sum_53 = squeeze_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_backward_25 = torch.ops.aten.convolution_backward.default(mul_960, relu_74, primals_235, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_960 = primals_235 = None
    getitem_285: "f32[8, 2048, 14, 14]" = convolution_backward_25[0]
    getitem_286: "f32[1024, 2048, 1, 1]" = convolution_backward_25[1];  convolution_backward_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    alias_176: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(relu_74);  relu_74 = None
    alias_177: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(alias_176);  alias_176 = None
    le_25: "b8[8, 2048, 14, 14]" = torch.ops.aten.le.Scalar(alias_177, 0);  alias_177 = None
    where_25: "f32[8, 2048, 14, 14]" = torch.ops.aten.where.self(le_25, full_default, getitem_285);  le_25 = getitem_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    sum_54: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_25, [0, 2, 3])
    sub_208: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_77, unsqueeze_730);  convolution_77 = unsqueeze_730 = None
    mul_962: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(where_25, sub_208)
    sum_55: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_962, [0, 2, 3]);  mul_962 = None
    mul_963: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_54, 0.0006377551020408163)
    unsqueeze_731: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_963, 0);  mul_963 = None
    unsqueeze_732: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_731, 2);  unsqueeze_731 = None
    unsqueeze_733: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_732, 3);  unsqueeze_732 = None
    mul_964: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_55, 0.0006377551020408163)
    mul_965: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_232, squeeze_232)
    mul_966: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_964, mul_965);  mul_964 = mul_965 = None
    unsqueeze_734: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_966, 0);  mul_966 = None
    unsqueeze_735: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_734, 2);  unsqueeze_734 = None
    unsqueeze_736: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_735, 3);  unsqueeze_735 = None
    mul_967: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_232, primals_233);  primals_233 = None
    unsqueeze_737: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_967, 0);  mul_967 = None
    unsqueeze_738: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_737, 2);  unsqueeze_737 = None
    unsqueeze_739: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_738, 3);  unsqueeze_738 = None
    mul_968: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_208, unsqueeze_736);  sub_208 = unsqueeze_736 = None
    sub_210: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(where_25, mul_968);  where_25 = mul_968 = None
    sub_211: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(sub_210, unsqueeze_733);  sub_210 = unsqueeze_733 = None
    mul_969: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_211, unsqueeze_739);  sub_211 = unsqueeze_739 = None
    mul_970: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_55, squeeze_232);  sum_55 = squeeze_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_backward_26 = torch.ops.aten.convolution_backward.default(mul_969, relu_73, primals_232, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_969 = primals_232 = None
    getitem_288: "f32[8, 2048, 14, 14]" = convolution_backward_26[0]
    getitem_289: "f32[2048, 64, 3, 3]" = convolution_backward_26[1];  convolution_backward_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    alias_179: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(relu_73);  relu_73 = None
    alias_180: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(alias_179);  alias_179 = None
    le_26: "b8[8, 2048, 14, 14]" = torch.ops.aten.le.Scalar(alias_180, 0);  alias_180 = None
    where_26: "f32[8, 2048, 14, 14]" = torch.ops.aten.where.self(le_26, full_default, getitem_288);  le_26 = getitem_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    sum_56: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_26, [0, 2, 3])
    sub_212: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_76, unsqueeze_742);  convolution_76 = unsqueeze_742 = None
    mul_971: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(where_26, sub_212)
    sum_57: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_971, [0, 2, 3]);  mul_971 = None
    mul_972: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_56, 0.0006377551020408163)
    unsqueeze_743: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_972, 0);  mul_972 = None
    unsqueeze_744: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_743, 2);  unsqueeze_743 = None
    unsqueeze_745: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_744, 3);  unsqueeze_744 = None
    mul_973: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_57, 0.0006377551020408163)
    mul_974: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_229, squeeze_229)
    mul_975: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_973, mul_974);  mul_973 = mul_974 = None
    unsqueeze_746: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_975, 0);  mul_975 = None
    unsqueeze_747: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_746, 2);  unsqueeze_746 = None
    unsqueeze_748: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_747, 3);  unsqueeze_747 = None
    mul_976: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_229, primals_230);  primals_230 = None
    unsqueeze_749: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_976, 0);  mul_976 = None
    unsqueeze_750: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_749, 2);  unsqueeze_749 = None
    unsqueeze_751: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_750, 3);  unsqueeze_750 = None
    mul_977: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_212, unsqueeze_748);  sub_212 = unsqueeze_748 = None
    sub_214: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(where_26, mul_977);  where_26 = mul_977 = None
    sub_215: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(sub_214, unsqueeze_745);  sub_214 = unsqueeze_745 = None
    mul_978: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_215, unsqueeze_751);  sub_215 = unsqueeze_751 = None
    mul_979: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_57, squeeze_229);  sum_57 = squeeze_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_backward_27 = torch.ops.aten.convolution_backward.default(mul_978, relu_72, primals_229, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_978 = primals_229 = None
    getitem_291: "f32[8, 1024, 14, 14]" = convolution_backward_27[0]
    getitem_292: "f32[2048, 1024, 1, 1]" = convolution_backward_27[1];  convolution_backward_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    add_561: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(where_24, getitem_291);  where_24 = getitem_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    alias_182: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(relu_72);  relu_72 = None
    alias_183: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(alias_182);  alias_182 = None
    le_27: "b8[8, 1024, 14, 14]" = torch.ops.aten.le.Scalar(alias_183, 0);  alias_183 = None
    where_27: "f32[8, 1024, 14, 14]" = torch.ops.aten.where.self(le_27, full_default, add_561);  le_27 = add_561 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    sum_58: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_27, [0, 2, 3])
    sub_216: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_75, unsqueeze_754);  convolution_75 = unsqueeze_754 = None
    mul_980: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_27, sub_216)
    sum_59: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_980, [0, 2, 3]);  mul_980 = None
    mul_981: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_58, 0.0006377551020408163)
    unsqueeze_755: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_981, 0);  mul_981 = None
    unsqueeze_756: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_755, 2);  unsqueeze_755 = None
    unsqueeze_757: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_756, 3);  unsqueeze_756 = None
    mul_982: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_59, 0.0006377551020408163)
    mul_983: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_226, squeeze_226)
    mul_984: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_982, mul_983);  mul_982 = mul_983 = None
    unsqueeze_758: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_984, 0);  mul_984 = None
    unsqueeze_759: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_758, 2);  unsqueeze_758 = None
    unsqueeze_760: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_759, 3);  unsqueeze_759 = None
    mul_985: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_226, primals_227);  primals_227 = None
    unsqueeze_761: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_985, 0);  mul_985 = None
    unsqueeze_762: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_761, 2);  unsqueeze_761 = None
    unsqueeze_763: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_762, 3);  unsqueeze_762 = None
    mul_986: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_216, unsqueeze_760);  sub_216 = unsqueeze_760 = None
    sub_218: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_27, mul_986);  mul_986 = None
    sub_219: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_218, unsqueeze_757);  sub_218 = unsqueeze_757 = None
    mul_987: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_219, unsqueeze_763);  sub_219 = unsqueeze_763 = None
    mul_988: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_59, squeeze_226);  sum_59 = squeeze_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_backward_28 = torch.ops.aten.convolution_backward.default(mul_987, relu_71, primals_226, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_987 = primals_226 = None
    getitem_294: "f32[8, 2048, 14, 14]" = convolution_backward_28[0]
    getitem_295: "f32[1024, 2048, 1, 1]" = convolution_backward_28[1];  convolution_backward_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    alias_185: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(relu_71);  relu_71 = None
    alias_186: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(alias_185);  alias_185 = None
    le_28: "b8[8, 2048, 14, 14]" = torch.ops.aten.le.Scalar(alias_186, 0);  alias_186 = None
    where_28: "f32[8, 2048, 14, 14]" = torch.ops.aten.where.self(le_28, full_default, getitem_294);  le_28 = getitem_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    sum_60: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_28, [0, 2, 3])
    sub_220: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_74, unsqueeze_766);  convolution_74 = unsqueeze_766 = None
    mul_989: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(where_28, sub_220)
    sum_61: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_989, [0, 2, 3]);  mul_989 = None
    mul_990: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_60, 0.0006377551020408163)
    unsqueeze_767: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_990, 0);  mul_990 = None
    unsqueeze_768: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_767, 2);  unsqueeze_767 = None
    unsqueeze_769: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_768, 3);  unsqueeze_768 = None
    mul_991: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_61, 0.0006377551020408163)
    mul_992: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_223, squeeze_223)
    mul_993: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_991, mul_992);  mul_991 = mul_992 = None
    unsqueeze_770: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_993, 0);  mul_993 = None
    unsqueeze_771: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_770, 2);  unsqueeze_770 = None
    unsqueeze_772: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_771, 3);  unsqueeze_771 = None
    mul_994: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_223, primals_224);  primals_224 = None
    unsqueeze_773: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_994, 0);  mul_994 = None
    unsqueeze_774: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_773, 2);  unsqueeze_773 = None
    unsqueeze_775: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_774, 3);  unsqueeze_774 = None
    mul_995: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_220, unsqueeze_772);  sub_220 = unsqueeze_772 = None
    sub_222: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(where_28, mul_995);  where_28 = mul_995 = None
    sub_223: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(sub_222, unsqueeze_769);  sub_222 = unsqueeze_769 = None
    mul_996: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_223, unsqueeze_775);  sub_223 = unsqueeze_775 = None
    mul_997: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_61, squeeze_223);  sum_61 = squeeze_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_backward_29 = torch.ops.aten.convolution_backward.default(mul_996, relu_70, primals_223, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_996 = primals_223 = None
    getitem_297: "f32[8, 2048, 14, 14]" = convolution_backward_29[0]
    getitem_298: "f32[2048, 64, 3, 3]" = convolution_backward_29[1];  convolution_backward_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    alias_188: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(relu_70);  relu_70 = None
    alias_189: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(alias_188);  alias_188 = None
    le_29: "b8[8, 2048, 14, 14]" = torch.ops.aten.le.Scalar(alias_189, 0);  alias_189 = None
    where_29: "f32[8, 2048, 14, 14]" = torch.ops.aten.where.self(le_29, full_default, getitem_297);  le_29 = getitem_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    sum_62: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_29, [0, 2, 3])
    sub_224: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_73, unsqueeze_778);  convolution_73 = unsqueeze_778 = None
    mul_998: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(where_29, sub_224)
    sum_63: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_998, [0, 2, 3]);  mul_998 = None
    mul_999: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_62, 0.0006377551020408163)
    unsqueeze_779: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_999, 0);  mul_999 = None
    unsqueeze_780: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_779, 2);  unsqueeze_779 = None
    unsqueeze_781: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_780, 3);  unsqueeze_780 = None
    mul_1000: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_63, 0.0006377551020408163)
    mul_1001: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_220, squeeze_220)
    mul_1002: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_1000, mul_1001);  mul_1000 = mul_1001 = None
    unsqueeze_782: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1002, 0);  mul_1002 = None
    unsqueeze_783: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_782, 2);  unsqueeze_782 = None
    unsqueeze_784: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_783, 3);  unsqueeze_783 = None
    mul_1003: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_220, primals_221);  primals_221 = None
    unsqueeze_785: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1003, 0);  mul_1003 = None
    unsqueeze_786: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_785, 2);  unsqueeze_785 = None
    unsqueeze_787: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_786, 3);  unsqueeze_786 = None
    mul_1004: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_224, unsqueeze_784);  sub_224 = unsqueeze_784 = None
    sub_226: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(where_29, mul_1004);  where_29 = mul_1004 = None
    sub_227: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(sub_226, unsqueeze_781);  sub_226 = unsqueeze_781 = None
    mul_1005: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_227, unsqueeze_787);  sub_227 = unsqueeze_787 = None
    mul_1006: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_63, squeeze_220);  sum_63 = squeeze_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_backward_30 = torch.ops.aten.convolution_backward.default(mul_1005, relu_69, primals_220, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1005 = primals_220 = None
    getitem_300: "f32[8, 1024, 14, 14]" = convolution_backward_30[0]
    getitem_301: "f32[2048, 1024, 1, 1]" = convolution_backward_30[1];  convolution_backward_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    add_562: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(where_27, getitem_300);  where_27 = getitem_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    alias_191: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(relu_69);  relu_69 = None
    alias_192: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(alias_191);  alias_191 = None
    le_30: "b8[8, 1024, 14, 14]" = torch.ops.aten.le.Scalar(alias_192, 0);  alias_192 = None
    where_30: "f32[8, 1024, 14, 14]" = torch.ops.aten.where.self(le_30, full_default, add_562);  le_30 = add_562 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    sum_64: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_30, [0, 2, 3])
    sub_228: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_72, unsqueeze_790);  convolution_72 = unsqueeze_790 = None
    mul_1007: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_30, sub_228)
    sum_65: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1007, [0, 2, 3]);  mul_1007 = None
    mul_1008: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_64, 0.0006377551020408163)
    unsqueeze_791: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1008, 0);  mul_1008 = None
    unsqueeze_792: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_791, 2);  unsqueeze_791 = None
    unsqueeze_793: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_792, 3);  unsqueeze_792 = None
    mul_1009: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_65, 0.0006377551020408163)
    mul_1010: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_217, squeeze_217)
    mul_1011: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1009, mul_1010);  mul_1009 = mul_1010 = None
    unsqueeze_794: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1011, 0);  mul_1011 = None
    unsqueeze_795: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_794, 2);  unsqueeze_794 = None
    unsqueeze_796: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_795, 3);  unsqueeze_795 = None
    mul_1012: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_217, primals_218);  primals_218 = None
    unsqueeze_797: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1012, 0);  mul_1012 = None
    unsqueeze_798: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_797, 2);  unsqueeze_797 = None
    unsqueeze_799: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_798, 3);  unsqueeze_798 = None
    mul_1013: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_228, unsqueeze_796);  sub_228 = unsqueeze_796 = None
    sub_230: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_30, mul_1013);  mul_1013 = None
    sub_231: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_230, unsqueeze_793);  sub_230 = unsqueeze_793 = None
    mul_1014: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_231, unsqueeze_799);  sub_231 = unsqueeze_799 = None
    mul_1015: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_65, squeeze_217);  sum_65 = squeeze_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_backward_31 = torch.ops.aten.convolution_backward.default(mul_1014, relu_68, primals_217, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1014 = primals_217 = None
    getitem_303: "f32[8, 2048, 14, 14]" = convolution_backward_31[0]
    getitem_304: "f32[1024, 2048, 1, 1]" = convolution_backward_31[1];  convolution_backward_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    alias_194: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(relu_68);  relu_68 = None
    alias_195: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(alias_194);  alias_194 = None
    le_31: "b8[8, 2048, 14, 14]" = torch.ops.aten.le.Scalar(alias_195, 0);  alias_195 = None
    where_31: "f32[8, 2048, 14, 14]" = torch.ops.aten.where.self(le_31, full_default, getitem_303);  le_31 = getitem_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    sum_66: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_31, [0, 2, 3])
    sub_232: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_71, unsqueeze_802);  convolution_71 = unsqueeze_802 = None
    mul_1016: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(where_31, sub_232)
    sum_67: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_1016, [0, 2, 3]);  mul_1016 = None
    mul_1017: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_66, 0.0006377551020408163)
    unsqueeze_803: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1017, 0);  mul_1017 = None
    unsqueeze_804: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_803, 2);  unsqueeze_803 = None
    unsqueeze_805: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_804, 3);  unsqueeze_804 = None
    mul_1018: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_67, 0.0006377551020408163)
    mul_1019: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_214, squeeze_214)
    mul_1020: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_1018, mul_1019);  mul_1018 = mul_1019 = None
    unsqueeze_806: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1020, 0);  mul_1020 = None
    unsqueeze_807: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_806, 2);  unsqueeze_806 = None
    unsqueeze_808: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_807, 3);  unsqueeze_807 = None
    mul_1021: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_214, primals_215);  primals_215 = None
    unsqueeze_809: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1021, 0);  mul_1021 = None
    unsqueeze_810: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_809, 2);  unsqueeze_809 = None
    unsqueeze_811: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_810, 3);  unsqueeze_810 = None
    mul_1022: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_232, unsqueeze_808);  sub_232 = unsqueeze_808 = None
    sub_234: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(where_31, mul_1022);  where_31 = mul_1022 = None
    sub_235: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(sub_234, unsqueeze_805);  sub_234 = unsqueeze_805 = None
    mul_1023: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_235, unsqueeze_811);  sub_235 = unsqueeze_811 = None
    mul_1024: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_67, squeeze_214);  sum_67 = squeeze_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_backward_32 = torch.ops.aten.convolution_backward.default(mul_1023, relu_67, primals_214, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_1023 = primals_214 = None
    getitem_306: "f32[8, 2048, 14, 14]" = convolution_backward_32[0]
    getitem_307: "f32[2048, 64, 3, 3]" = convolution_backward_32[1];  convolution_backward_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    alias_197: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(relu_67);  relu_67 = None
    alias_198: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(alias_197);  alias_197 = None
    le_32: "b8[8, 2048, 14, 14]" = torch.ops.aten.le.Scalar(alias_198, 0);  alias_198 = None
    where_32: "f32[8, 2048, 14, 14]" = torch.ops.aten.where.self(le_32, full_default, getitem_306);  le_32 = getitem_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    sum_68: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_32, [0, 2, 3])
    sub_236: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_70, unsqueeze_814);  convolution_70 = unsqueeze_814 = None
    mul_1025: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(where_32, sub_236)
    sum_69: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_1025, [0, 2, 3]);  mul_1025 = None
    mul_1026: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_68, 0.0006377551020408163)
    unsqueeze_815: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1026, 0);  mul_1026 = None
    unsqueeze_816: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_815, 2);  unsqueeze_815 = None
    unsqueeze_817: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_816, 3);  unsqueeze_816 = None
    mul_1027: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_69, 0.0006377551020408163)
    mul_1028: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_211, squeeze_211)
    mul_1029: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_1027, mul_1028);  mul_1027 = mul_1028 = None
    unsqueeze_818: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1029, 0);  mul_1029 = None
    unsqueeze_819: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_818, 2);  unsqueeze_818 = None
    unsqueeze_820: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_819, 3);  unsqueeze_819 = None
    mul_1030: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_211, primals_212);  primals_212 = None
    unsqueeze_821: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1030, 0);  mul_1030 = None
    unsqueeze_822: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_821, 2);  unsqueeze_821 = None
    unsqueeze_823: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_822, 3);  unsqueeze_822 = None
    mul_1031: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_236, unsqueeze_820);  sub_236 = unsqueeze_820 = None
    sub_238: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(where_32, mul_1031);  where_32 = mul_1031 = None
    sub_239: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(sub_238, unsqueeze_817);  sub_238 = unsqueeze_817 = None
    mul_1032: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_239, unsqueeze_823);  sub_239 = unsqueeze_823 = None
    mul_1033: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_69, squeeze_211);  sum_69 = squeeze_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_backward_33 = torch.ops.aten.convolution_backward.default(mul_1032, relu_66, primals_211, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1032 = primals_211 = None
    getitem_309: "f32[8, 1024, 14, 14]" = convolution_backward_33[0]
    getitem_310: "f32[2048, 1024, 1, 1]" = convolution_backward_33[1];  convolution_backward_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    add_563: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(where_30, getitem_309);  where_30 = getitem_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    alias_200: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(relu_66);  relu_66 = None
    alias_201: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(alias_200);  alias_200 = None
    le_33: "b8[8, 1024, 14, 14]" = torch.ops.aten.le.Scalar(alias_201, 0);  alias_201 = None
    where_33: "f32[8, 1024, 14, 14]" = torch.ops.aten.where.self(le_33, full_default, add_563);  le_33 = add_563 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    sum_70: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_33, [0, 2, 3])
    sub_240: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_69, unsqueeze_826);  convolution_69 = unsqueeze_826 = None
    mul_1034: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_33, sub_240)
    sum_71: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1034, [0, 2, 3]);  mul_1034 = None
    mul_1035: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_70, 0.0006377551020408163)
    unsqueeze_827: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1035, 0);  mul_1035 = None
    unsqueeze_828: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_827, 2);  unsqueeze_827 = None
    unsqueeze_829: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_828, 3);  unsqueeze_828 = None
    mul_1036: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_71, 0.0006377551020408163)
    mul_1037: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_208, squeeze_208)
    mul_1038: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1036, mul_1037);  mul_1036 = mul_1037 = None
    unsqueeze_830: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1038, 0);  mul_1038 = None
    unsqueeze_831: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_830, 2);  unsqueeze_830 = None
    unsqueeze_832: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_831, 3);  unsqueeze_831 = None
    mul_1039: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_208, primals_209);  primals_209 = None
    unsqueeze_833: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1039, 0);  mul_1039 = None
    unsqueeze_834: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_833, 2);  unsqueeze_833 = None
    unsqueeze_835: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_834, 3);  unsqueeze_834 = None
    mul_1040: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_240, unsqueeze_832);  sub_240 = unsqueeze_832 = None
    sub_242: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_33, mul_1040);  mul_1040 = None
    sub_243: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_242, unsqueeze_829);  sub_242 = unsqueeze_829 = None
    mul_1041: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_243, unsqueeze_835);  sub_243 = unsqueeze_835 = None
    mul_1042: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_71, squeeze_208);  sum_71 = squeeze_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_backward_34 = torch.ops.aten.convolution_backward.default(mul_1041, relu_65, primals_208, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1041 = primals_208 = None
    getitem_312: "f32[8, 2048, 14, 14]" = convolution_backward_34[0]
    getitem_313: "f32[1024, 2048, 1, 1]" = convolution_backward_34[1];  convolution_backward_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    alias_203: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(relu_65);  relu_65 = None
    alias_204: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(alias_203);  alias_203 = None
    le_34: "b8[8, 2048, 14, 14]" = torch.ops.aten.le.Scalar(alias_204, 0);  alias_204 = None
    where_34: "f32[8, 2048, 14, 14]" = torch.ops.aten.where.self(le_34, full_default, getitem_312);  le_34 = getitem_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    sum_72: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_34, [0, 2, 3])
    sub_244: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_68, unsqueeze_838);  convolution_68 = unsqueeze_838 = None
    mul_1043: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(where_34, sub_244)
    sum_73: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_1043, [0, 2, 3]);  mul_1043 = None
    mul_1044: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_72, 0.0006377551020408163)
    unsqueeze_839: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1044, 0);  mul_1044 = None
    unsqueeze_840: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_839, 2);  unsqueeze_839 = None
    unsqueeze_841: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_840, 3);  unsqueeze_840 = None
    mul_1045: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_73, 0.0006377551020408163)
    mul_1046: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_205, squeeze_205)
    mul_1047: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_1045, mul_1046);  mul_1045 = mul_1046 = None
    unsqueeze_842: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1047, 0);  mul_1047 = None
    unsqueeze_843: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_842, 2);  unsqueeze_842 = None
    unsqueeze_844: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_843, 3);  unsqueeze_843 = None
    mul_1048: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_205, primals_206);  primals_206 = None
    unsqueeze_845: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1048, 0);  mul_1048 = None
    unsqueeze_846: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_845, 2);  unsqueeze_845 = None
    unsqueeze_847: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_846, 3);  unsqueeze_846 = None
    mul_1049: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_244, unsqueeze_844);  sub_244 = unsqueeze_844 = None
    sub_246: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(where_34, mul_1049);  where_34 = mul_1049 = None
    sub_247: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(sub_246, unsqueeze_841);  sub_246 = unsqueeze_841 = None
    mul_1050: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_247, unsqueeze_847);  sub_247 = unsqueeze_847 = None
    mul_1051: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_73, squeeze_205);  sum_73 = squeeze_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_backward_35 = torch.ops.aten.convolution_backward.default(mul_1050, relu_64, primals_205, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_1050 = primals_205 = None
    getitem_315: "f32[8, 2048, 14, 14]" = convolution_backward_35[0]
    getitem_316: "f32[2048, 64, 3, 3]" = convolution_backward_35[1];  convolution_backward_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    alias_206: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(relu_64);  relu_64 = None
    alias_207: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(alias_206);  alias_206 = None
    le_35: "b8[8, 2048, 14, 14]" = torch.ops.aten.le.Scalar(alias_207, 0);  alias_207 = None
    where_35: "f32[8, 2048, 14, 14]" = torch.ops.aten.where.self(le_35, full_default, getitem_315);  le_35 = getitem_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    sum_74: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_35, [0, 2, 3])
    sub_248: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_67, unsqueeze_850);  convolution_67 = unsqueeze_850 = None
    mul_1052: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(where_35, sub_248)
    sum_75: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_1052, [0, 2, 3]);  mul_1052 = None
    mul_1053: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_74, 0.0006377551020408163)
    unsqueeze_851: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1053, 0);  mul_1053 = None
    unsqueeze_852: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_851, 2);  unsqueeze_851 = None
    unsqueeze_853: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_852, 3);  unsqueeze_852 = None
    mul_1054: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_75, 0.0006377551020408163)
    mul_1055: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_202, squeeze_202)
    mul_1056: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_1054, mul_1055);  mul_1054 = mul_1055 = None
    unsqueeze_854: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1056, 0);  mul_1056 = None
    unsqueeze_855: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_854, 2);  unsqueeze_854 = None
    unsqueeze_856: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_855, 3);  unsqueeze_855 = None
    mul_1057: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_202, primals_203);  primals_203 = None
    unsqueeze_857: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1057, 0);  mul_1057 = None
    unsqueeze_858: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_857, 2);  unsqueeze_857 = None
    unsqueeze_859: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_858, 3);  unsqueeze_858 = None
    mul_1058: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_248, unsqueeze_856);  sub_248 = unsqueeze_856 = None
    sub_250: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(where_35, mul_1058);  where_35 = mul_1058 = None
    sub_251: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(sub_250, unsqueeze_853);  sub_250 = unsqueeze_853 = None
    mul_1059: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_251, unsqueeze_859);  sub_251 = unsqueeze_859 = None
    mul_1060: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_75, squeeze_202);  sum_75 = squeeze_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_backward_36 = torch.ops.aten.convolution_backward.default(mul_1059, relu_63, primals_202, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1059 = primals_202 = None
    getitem_318: "f32[8, 1024, 14, 14]" = convolution_backward_36[0]
    getitem_319: "f32[2048, 1024, 1, 1]" = convolution_backward_36[1];  convolution_backward_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    add_564: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(where_33, getitem_318);  where_33 = getitem_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    alias_209: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(relu_63);  relu_63 = None
    alias_210: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(alias_209);  alias_209 = None
    le_36: "b8[8, 1024, 14, 14]" = torch.ops.aten.le.Scalar(alias_210, 0);  alias_210 = None
    where_36: "f32[8, 1024, 14, 14]" = torch.ops.aten.where.self(le_36, full_default, add_564);  le_36 = add_564 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    sum_76: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_36, [0, 2, 3])
    sub_252: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_66, unsqueeze_862);  convolution_66 = unsqueeze_862 = None
    mul_1061: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_36, sub_252)
    sum_77: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1061, [0, 2, 3]);  mul_1061 = None
    mul_1062: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_76, 0.0006377551020408163)
    unsqueeze_863: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1062, 0);  mul_1062 = None
    unsqueeze_864: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_863, 2);  unsqueeze_863 = None
    unsqueeze_865: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_864, 3);  unsqueeze_864 = None
    mul_1063: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_77, 0.0006377551020408163)
    mul_1064: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_199, squeeze_199)
    mul_1065: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1063, mul_1064);  mul_1063 = mul_1064 = None
    unsqueeze_866: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1065, 0);  mul_1065 = None
    unsqueeze_867: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_866, 2);  unsqueeze_866 = None
    unsqueeze_868: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_867, 3);  unsqueeze_867 = None
    mul_1066: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_199, primals_200);  primals_200 = None
    unsqueeze_869: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1066, 0);  mul_1066 = None
    unsqueeze_870: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_869, 2);  unsqueeze_869 = None
    unsqueeze_871: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_870, 3);  unsqueeze_870 = None
    mul_1067: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_252, unsqueeze_868);  sub_252 = unsqueeze_868 = None
    sub_254: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_36, mul_1067);  mul_1067 = None
    sub_255: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_254, unsqueeze_865);  sub_254 = unsqueeze_865 = None
    mul_1068: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_255, unsqueeze_871);  sub_255 = unsqueeze_871 = None
    mul_1069: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_77, squeeze_199);  sum_77 = squeeze_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_backward_37 = torch.ops.aten.convolution_backward.default(mul_1068, relu_62, primals_199, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1068 = primals_199 = None
    getitem_321: "f32[8, 2048, 14, 14]" = convolution_backward_37[0]
    getitem_322: "f32[1024, 2048, 1, 1]" = convolution_backward_37[1];  convolution_backward_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    alias_212: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(relu_62);  relu_62 = None
    alias_213: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(alias_212);  alias_212 = None
    le_37: "b8[8, 2048, 14, 14]" = torch.ops.aten.le.Scalar(alias_213, 0);  alias_213 = None
    where_37: "f32[8, 2048, 14, 14]" = torch.ops.aten.where.self(le_37, full_default, getitem_321);  le_37 = getitem_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    sum_78: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_37, [0, 2, 3])
    sub_256: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_65, unsqueeze_874);  convolution_65 = unsqueeze_874 = None
    mul_1070: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(where_37, sub_256)
    sum_79: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_1070, [0, 2, 3]);  mul_1070 = None
    mul_1071: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_78, 0.0006377551020408163)
    unsqueeze_875: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1071, 0);  mul_1071 = None
    unsqueeze_876: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_875, 2);  unsqueeze_875 = None
    unsqueeze_877: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_876, 3);  unsqueeze_876 = None
    mul_1072: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_79, 0.0006377551020408163)
    mul_1073: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_196, squeeze_196)
    mul_1074: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_1072, mul_1073);  mul_1072 = mul_1073 = None
    unsqueeze_878: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1074, 0);  mul_1074 = None
    unsqueeze_879: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_878, 2);  unsqueeze_878 = None
    unsqueeze_880: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_879, 3);  unsqueeze_879 = None
    mul_1075: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_196, primals_197);  primals_197 = None
    unsqueeze_881: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1075, 0);  mul_1075 = None
    unsqueeze_882: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_881, 2);  unsqueeze_881 = None
    unsqueeze_883: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_882, 3);  unsqueeze_882 = None
    mul_1076: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_256, unsqueeze_880);  sub_256 = unsqueeze_880 = None
    sub_258: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(where_37, mul_1076);  where_37 = mul_1076 = None
    sub_259: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(sub_258, unsqueeze_877);  sub_258 = unsqueeze_877 = None
    mul_1077: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_259, unsqueeze_883);  sub_259 = unsqueeze_883 = None
    mul_1078: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_79, squeeze_196);  sum_79 = squeeze_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_backward_38 = torch.ops.aten.convolution_backward.default(mul_1077, relu_61, primals_196, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_1077 = primals_196 = None
    getitem_324: "f32[8, 2048, 14, 14]" = convolution_backward_38[0]
    getitem_325: "f32[2048, 64, 3, 3]" = convolution_backward_38[1];  convolution_backward_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    alias_215: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(relu_61);  relu_61 = None
    alias_216: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(alias_215);  alias_215 = None
    le_38: "b8[8, 2048, 14, 14]" = torch.ops.aten.le.Scalar(alias_216, 0);  alias_216 = None
    where_38: "f32[8, 2048, 14, 14]" = torch.ops.aten.where.self(le_38, full_default, getitem_324);  le_38 = getitem_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    sum_80: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_38, [0, 2, 3])
    sub_260: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_64, unsqueeze_886);  convolution_64 = unsqueeze_886 = None
    mul_1079: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(where_38, sub_260)
    sum_81: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_1079, [0, 2, 3]);  mul_1079 = None
    mul_1080: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_80, 0.0006377551020408163)
    unsqueeze_887: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1080, 0);  mul_1080 = None
    unsqueeze_888: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_887, 2);  unsqueeze_887 = None
    unsqueeze_889: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_888, 3);  unsqueeze_888 = None
    mul_1081: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_81, 0.0006377551020408163)
    mul_1082: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_193, squeeze_193)
    mul_1083: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_1081, mul_1082);  mul_1081 = mul_1082 = None
    unsqueeze_890: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1083, 0);  mul_1083 = None
    unsqueeze_891: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_890, 2);  unsqueeze_890 = None
    unsqueeze_892: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_891, 3);  unsqueeze_891 = None
    mul_1084: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_193, primals_194);  primals_194 = None
    unsqueeze_893: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1084, 0);  mul_1084 = None
    unsqueeze_894: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_893, 2);  unsqueeze_893 = None
    unsqueeze_895: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_894, 3);  unsqueeze_894 = None
    mul_1085: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_260, unsqueeze_892);  sub_260 = unsqueeze_892 = None
    sub_262: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(where_38, mul_1085);  where_38 = mul_1085 = None
    sub_263: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(sub_262, unsqueeze_889);  sub_262 = unsqueeze_889 = None
    mul_1086: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_263, unsqueeze_895);  sub_263 = unsqueeze_895 = None
    mul_1087: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_81, squeeze_193);  sum_81 = squeeze_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_backward_39 = torch.ops.aten.convolution_backward.default(mul_1086, relu_60, primals_193, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1086 = primals_193 = None
    getitem_327: "f32[8, 1024, 14, 14]" = convolution_backward_39[0]
    getitem_328: "f32[2048, 1024, 1, 1]" = convolution_backward_39[1];  convolution_backward_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    add_565: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(where_36, getitem_327);  where_36 = getitem_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    alias_218: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(relu_60);  relu_60 = None
    alias_219: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(alias_218);  alias_218 = None
    le_39: "b8[8, 1024, 14, 14]" = torch.ops.aten.le.Scalar(alias_219, 0);  alias_219 = None
    where_39: "f32[8, 1024, 14, 14]" = torch.ops.aten.where.self(le_39, full_default, add_565);  le_39 = add_565 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    sum_82: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_39, [0, 2, 3])
    sub_264: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_63, unsqueeze_898);  convolution_63 = unsqueeze_898 = None
    mul_1088: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_39, sub_264)
    sum_83: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1088, [0, 2, 3]);  mul_1088 = None
    mul_1089: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_82, 0.0006377551020408163)
    unsqueeze_899: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1089, 0);  mul_1089 = None
    unsqueeze_900: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_899, 2);  unsqueeze_899 = None
    unsqueeze_901: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_900, 3);  unsqueeze_900 = None
    mul_1090: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_83, 0.0006377551020408163)
    mul_1091: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_190, squeeze_190)
    mul_1092: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1090, mul_1091);  mul_1090 = mul_1091 = None
    unsqueeze_902: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1092, 0);  mul_1092 = None
    unsqueeze_903: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_902, 2);  unsqueeze_902 = None
    unsqueeze_904: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_903, 3);  unsqueeze_903 = None
    mul_1093: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_190, primals_191);  primals_191 = None
    unsqueeze_905: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1093, 0);  mul_1093 = None
    unsqueeze_906: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_905, 2);  unsqueeze_905 = None
    unsqueeze_907: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_906, 3);  unsqueeze_906 = None
    mul_1094: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_264, unsqueeze_904);  sub_264 = unsqueeze_904 = None
    sub_266: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_39, mul_1094);  mul_1094 = None
    sub_267: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_266, unsqueeze_901);  sub_266 = unsqueeze_901 = None
    mul_1095: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_267, unsqueeze_907);  sub_267 = unsqueeze_907 = None
    mul_1096: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_83, squeeze_190);  sum_83 = squeeze_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_backward_40 = torch.ops.aten.convolution_backward.default(mul_1095, relu_59, primals_190, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1095 = primals_190 = None
    getitem_330: "f32[8, 2048, 14, 14]" = convolution_backward_40[0]
    getitem_331: "f32[1024, 2048, 1, 1]" = convolution_backward_40[1];  convolution_backward_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    alias_221: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(relu_59);  relu_59 = None
    alias_222: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(alias_221);  alias_221 = None
    le_40: "b8[8, 2048, 14, 14]" = torch.ops.aten.le.Scalar(alias_222, 0);  alias_222 = None
    where_40: "f32[8, 2048, 14, 14]" = torch.ops.aten.where.self(le_40, full_default, getitem_330);  le_40 = getitem_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    sum_84: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_40, [0, 2, 3])
    sub_268: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_62, unsqueeze_910);  convolution_62 = unsqueeze_910 = None
    mul_1097: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(where_40, sub_268)
    sum_85: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_1097, [0, 2, 3]);  mul_1097 = None
    mul_1098: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_84, 0.0006377551020408163)
    unsqueeze_911: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1098, 0);  mul_1098 = None
    unsqueeze_912: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_911, 2);  unsqueeze_911 = None
    unsqueeze_913: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_912, 3);  unsqueeze_912 = None
    mul_1099: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_85, 0.0006377551020408163)
    mul_1100: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_187, squeeze_187)
    mul_1101: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_1099, mul_1100);  mul_1099 = mul_1100 = None
    unsqueeze_914: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1101, 0);  mul_1101 = None
    unsqueeze_915: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_914, 2);  unsqueeze_914 = None
    unsqueeze_916: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_915, 3);  unsqueeze_915 = None
    mul_1102: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_187, primals_188);  primals_188 = None
    unsqueeze_917: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1102, 0);  mul_1102 = None
    unsqueeze_918: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_917, 2);  unsqueeze_917 = None
    unsqueeze_919: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_918, 3);  unsqueeze_918 = None
    mul_1103: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_268, unsqueeze_916);  sub_268 = unsqueeze_916 = None
    sub_270: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(where_40, mul_1103);  where_40 = mul_1103 = None
    sub_271: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(sub_270, unsqueeze_913);  sub_270 = unsqueeze_913 = None
    mul_1104: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_271, unsqueeze_919);  sub_271 = unsqueeze_919 = None
    mul_1105: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_85, squeeze_187);  sum_85 = squeeze_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_backward_41 = torch.ops.aten.convolution_backward.default(mul_1104, relu_58, primals_187, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_1104 = primals_187 = None
    getitem_333: "f32[8, 2048, 14, 14]" = convolution_backward_41[0]
    getitem_334: "f32[2048, 64, 3, 3]" = convolution_backward_41[1];  convolution_backward_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    alias_224: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(relu_58);  relu_58 = None
    alias_225: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(alias_224);  alias_224 = None
    le_41: "b8[8, 2048, 14, 14]" = torch.ops.aten.le.Scalar(alias_225, 0);  alias_225 = None
    where_41: "f32[8, 2048, 14, 14]" = torch.ops.aten.where.self(le_41, full_default, getitem_333);  le_41 = getitem_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    sum_86: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_41, [0, 2, 3])
    sub_272: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_61, unsqueeze_922);  convolution_61 = unsqueeze_922 = None
    mul_1106: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(where_41, sub_272)
    sum_87: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_1106, [0, 2, 3]);  mul_1106 = None
    mul_1107: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_86, 0.0006377551020408163)
    unsqueeze_923: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1107, 0);  mul_1107 = None
    unsqueeze_924: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_923, 2);  unsqueeze_923 = None
    unsqueeze_925: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_924, 3);  unsqueeze_924 = None
    mul_1108: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_87, 0.0006377551020408163)
    mul_1109: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_184, squeeze_184)
    mul_1110: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_1108, mul_1109);  mul_1108 = mul_1109 = None
    unsqueeze_926: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1110, 0);  mul_1110 = None
    unsqueeze_927: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_926, 2);  unsqueeze_926 = None
    unsqueeze_928: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_927, 3);  unsqueeze_927 = None
    mul_1111: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_184, primals_185);  primals_185 = None
    unsqueeze_929: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1111, 0);  mul_1111 = None
    unsqueeze_930: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_929, 2);  unsqueeze_929 = None
    unsqueeze_931: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_930, 3);  unsqueeze_930 = None
    mul_1112: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_272, unsqueeze_928);  sub_272 = unsqueeze_928 = None
    sub_274: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(where_41, mul_1112);  where_41 = mul_1112 = None
    sub_275: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(sub_274, unsqueeze_925);  sub_274 = unsqueeze_925 = None
    mul_1113: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_275, unsqueeze_931);  sub_275 = unsqueeze_931 = None
    mul_1114: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_87, squeeze_184);  sum_87 = squeeze_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_backward_42 = torch.ops.aten.convolution_backward.default(mul_1113, relu_57, primals_184, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1113 = primals_184 = None
    getitem_336: "f32[8, 1024, 14, 14]" = convolution_backward_42[0]
    getitem_337: "f32[2048, 1024, 1, 1]" = convolution_backward_42[1];  convolution_backward_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    add_566: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(where_39, getitem_336);  where_39 = getitem_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    alias_227: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(relu_57);  relu_57 = None
    alias_228: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(alias_227);  alias_227 = None
    le_42: "b8[8, 1024, 14, 14]" = torch.ops.aten.le.Scalar(alias_228, 0);  alias_228 = None
    where_42: "f32[8, 1024, 14, 14]" = torch.ops.aten.where.self(le_42, full_default, add_566);  le_42 = add_566 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    sum_88: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_42, [0, 2, 3])
    sub_276: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_60, unsqueeze_934);  convolution_60 = unsqueeze_934 = None
    mul_1115: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_42, sub_276)
    sum_89: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1115, [0, 2, 3]);  mul_1115 = None
    mul_1116: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_88, 0.0006377551020408163)
    unsqueeze_935: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1116, 0);  mul_1116 = None
    unsqueeze_936: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_935, 2);  unsqueeze_935 = None
    unsqueeze_937: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_936, 3);  unsqueeze_936 = None
    mul_1117: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_89, 0.0006377551020408163)
    mul_1118: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_181, squeeze_181)
    mul_1119: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1117, mul_1118);  mul_1117 = mul_1118 = None
    unsqueeze_938: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1119, 0);  mul_1119 = None
    unsqueeze_939: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_938, 2);  unsqueeze_938 = None
    unsqueeze_940: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_939, 3);  unsqueeze_939 = None
    mul_1120: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_181, primals_182);  primals_182 = None
    unsqueeze_941: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1120, 0);  mul_1120 = None
    unsqueeze_942: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_941, 2);  unsqueeze_941 = None
    unsqueeze_943: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_942, 3);  unsqueeze_942 = None
    mul_1121: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_276, unsqueeze_940);  sub_276 = unsqueeze_940 = None
    sub_278: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_42, mul_1121);  mul_1121 = None
    sub_279: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_278, unsqueeze_937);  sub_278 = unsqueeze_937 = None
    mul_1122: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_279, unsqueeze_943);  sub_279 = unsqueeze_943 = None
    mul_1123: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_89, squeeze_181);  sum_89 = squeeze_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_backward_43 = torch.ops.aten.convolution_backward.default(mul_1122, relu_56, primals_181, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1122 = primals_181 = None
    getitem_339: "f32[8, 2048, 14, 14]" = convolution_backward_43[0]
    getitem_340: "f32[1024, 2048, 1, 1]" = convolution_backward_43[1];  convolution_backward_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    alias_230: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(relu_56);  relu_56 = None
    alias_231: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(alias_230);  alias_230 = None
    le_43: "b8[8, 2048, 14, 14]" = torch.ops.aten.le.Scalar(alias_231, 0);  alias_231 = None
    where_43: "f32[8, 2048, 14, 14]" = torch.ops.aten.where.self(le_43, full_default, getitem_339);  le_43 = getitem_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    sum_90: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_43, [0, 2, 3])
    sub_280: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_59, unsqueeze_946);  convolution_59 = unsqueeze_946 = None
    mul_1124: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(where_43, sub_280)
    sum_91: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_1124, [0, 2, 3]);  mul_1124 = None
    mul_1125: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_90, 0.0006377551020408163)
    unsqueeze_947: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1125, 0);  mul_1125 = None
    unsqueeze_948: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_947, 2);  unsqueeze_947 = None
    unsqueeze_949: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_948, 3);  unsqueeze_948 = None
    mul_1126: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_91, 0.0006377551020408163)
    mul_1127: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_178, squeeze_178)
    mul_1128: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_1126, mul_1127);  mul_1126 = mul_1127 = None
    unsqueeze_950: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1128, 0);  mul_1128 = None
    unsqueeze_951: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_950, 2);  unsqueeze_950 = None
    unsqueeze_952: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_951, 3);  unsqueeze_951 = None
    mul_1129: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_178, primals_179);  primals_179 = None
    unsqueeze_953: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1129, 0);  mul_1129 = None
    unsqueeze_954: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_953, 2);  unsqueeze_953 = None
    unsqueeze_955: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_954, 3);  unsqueeze_954 = None
    mul_1130: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_280, unsqueeze_952);  sub_280 = unsqueeze_952 = None
    sub_282: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(where_43, mul_1130);  where_43 = mul_1130 = None
    sub_283: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(sub_282, unsqueeze_949);  sub_282 = unsqueeze_949 = None
    mul_1131: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_283, unsqueeze_955);  sub_283 = unsqueeze_955 = None
    mul_1132: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_91, squeeze_178);  sum_91 = squeeze_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_backward_44 = torch.ops.aten.convolution_backward.default(mul_1131, relu_55, primals_178, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_1131 = primals_178 = None
    getitem_342: "f32[8, 2048, 14, 14]" = convolution_backward_44[0]
    getitem_343: "f32[2048, 64, 3, 3]" = convolution_backward_44[1];  convolution_backward_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    alias_233: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(relu_55);  relu_55 = None
    alias_234: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(alias_233);  alias_233 = None
    le_44: "b8[8, 2048, 14, 14]" = torch.ops.aten.le.Scalar(alias_234, 0);  alias_234 = None
    where_44: "f32[8, 2048, 14, 14]" = torch.ops.aten.where.self(le_44, full_default, getitem_342);  le_44 = getitem_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    sum_92: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_44, [0, 2, 3])
    sub_284: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_58, unsqueeze_958);  convolution_58 = unsqueeze_958 = None
    mul_1133: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(where_44, sub_284)
    sum_93: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_1133, [0, 2, 3]);  mul_1133 = None
    mul_1134: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_92, 0.0006377551020408163)
    unsqueeze_959: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1134, 0);  mul_1134 = None
    unsqueeze_960: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_959, 2);  unsqueeze_959 = None
    unsqueeze_961: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_960, 3);  unsqueeze_960 = None
    mul_1135: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_93, 0.0006377551020408163)
    mul_1136: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_175, squeeze_175)
    mul_1137: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_1135, mul_1136);  mul_1135 = mul_1136 = None
    unsqueeze_962: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1137, 0);  mul_1137 = None
    unsqueeze_963: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_962, 2);  unsqueeze_962 = None
    unsqueeze_964: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_963, 3);  unsqueeze_963 = None
    mul_1138: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_175, primals_176);  primals_176 = None
    unsqueeze_965: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1138, 0);  mul_1138 = None
    unsqueeze_966: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_965, 2);  unsqueeze_965 = None
    unsqueeze_967: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_966, 3);  unsqueeze_966 = None
    mul_1139: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_284, unsqueeze_964);  sub_284 = unsqueeze_964 = None
    sub_286: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(where_44, mul_1139);  where_44 = mul_1139 = None
    sub_287: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(sub_286, unsqueeze_961);  sub_286 = unsqueeze_961 = None
    mul_1140: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_287, unsqueeze_967);  sub_287 = unsqueeze_967 = None
    mul_1141: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_93, squeeze_175);  sum_93 = squeeze_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_backward_45 = torch.ops.aten.convolution_backward.default(mul_1140, relu_54, primals_175, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1140 = primals_175 = None
    getitem_345: "f32[8, 1024, 14, 14]" = convolution_backward_45[0]
    getitem_346: "f32[2048, 1024, 1, 1]" = convolution_backward_45[1];  convolution_backward_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    add_567: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(where_42, getitem_345);  where_42 = getitem_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    alias_236: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(relu_54);  relu_54 = None
    alias_237: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(alias_236);  alias_236 = None
    le_45: "b8[8, 1024, 14, 14]" = torch.ops.aten.le.Scalar(alias_237, 0);  alias_237 = None
    where_45: "f32[8, 1024, 14, 14]" = torch.ops.aten.where.self(le_45, full_default, add_567);  le_45 = add_567 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    sum_94: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_45, [0, 2, 3])
    sub_288: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_57, unsqueeze_970);  convolution_57 = unsqueeze_970 = None
    mul_1142: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_45, sub_288)
    sum_95: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1142, [0, 2, 3]);  mul_1142 = None
    mul_1143: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_94, 0.0006377551020408163)
    unsqueeze_971: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1143, 0);  mul_1143 = None
    unsqueeze_972: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_971, 2);  unsqueeze_971 = None
    unsqueeze_973: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_972, 3);  unsqueeze_972 = None
    mul_1144: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_95, 0.0006377551020408163)
    mul_1145: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_172, squeeze_172)
    mul_1146: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1144, mul_1145);  mul_1144 = mul_1145 = None
    unsqueeze_974: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1146, 0);  mul_1146 = None
    unsqueeze_975: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_974, 2);  unsqueeze_974 = None
    unsqueeze_976: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_975, 3);  unsqueeze_975 = None
    mul_1147: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_172, primals_173);  primals_173 = None
    unsqueeze_977: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1147, 0);  mul_1147 = None
    unsqueeze_978: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_977, 2);  unsqueeze_977 = None
    unsqueeze_979: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_978, 3);  unsqueeze_978 = None
    mul_1148: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_288, unsqueeze_976);  sub_288 = unsqueeze_976 = None
    sub_290: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_45, mul_1148);  mul_1148 = None
    sub_291: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_290, unsqueeze_973);  sub_290 = unsqueeze_973 = None
    mul_1149: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_291, unsqueeze_979);  sub_291 = unsqueeze_979 = None
    mul_1150: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_95, squeeze_172);  sum_95 = squeeze_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_backward_46 = torch.ops.aten.convolution_backward.default(mul_1149, relu_53, primals_172, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1149 = primals_172 = None
    getitem_348: "f32[8, 2048, 14, 14]" = convolution_backward_46[0]
    getitem_349: "f32[1024, 2048, 1, 1]" = convolution_backward_46[1];  convolution_backward_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    alias_239: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(relu_53);  relu_53 = None
    alias_240: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(alias_239);  alias_239 = None
    le_46: "b8[8, 2048, 14, 14]" = torch.ops.aten.le.Scalar(alias_240, 0);  alias_240 = None
    where_46: "f32[8, 2048, 14, 14]" = torch.ops.aten.where.self(le_46, full_default, getitem_348);  le_46 = getitem_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    sum_96: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_46, [0, 2, 3])
    sub_292: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_56, unsqueeze_982);  convolution_56 = unsqueeze_982 = None
    mul_1151: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(where_46, sub_292)
    sum_97: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_1151, [0, 2, 3]);  mul_1151 = None
    mul_1152: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_96, 0.0006377551020408163)
    unsqueeze_983: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1152, 0);  mul_1152 = None
    unsqueeze_984: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_983, 2);  unsqueeze_983 = None
    unsqueeze_985: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_984, 3);  unsqueeze_984 = None
    mul_1153: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_97, 0.0006377551020408163)
    mul_1154: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_169, squeeze_169)
    mul_1155: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_1153, mul_1154);  mul_1153 = mul_1154 = None
    unsqueeze_986: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1155, 0);  mul_1155 = None
    unsqueeze_987: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_986, 2);  unsqueeze_986 = None
    unsqueeze_988: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_987, 3);  unsqueeze_987 = None
    mul_1156: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_169, primals_170);  primals_170 = None
    unsqueeze_989: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1156, 0);  mul_1156 = None
    unsqueeze_990: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_989, 2);  unsqueeze_989 = None
    unsqueeze_991: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_990, 3);  unsqueeze_990 = None
    mul_1157: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_292, unsqueeze_988);  sub_292 = unsqueeze_988 = None
    sub_294: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(where_46, mul_1157);  where_46 = mul_1157 = None
    sub_295: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(sub_294, unsqueeze_985);  sub_294 = unsqueeze_985 = None
    mul_1158: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_295, unsqueeze_991);  sub_295 = unsqueeze_991 = None
    mul_1159: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_97, squeeze_169);  sum_97 = squeeze_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_backward_47 = torch.ops.aten.convolution_backward.default(mul_1158, relu_52, primals_169, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_1158 = primals_169 = None
    getitem_351: "f32[8, 2048, 14, 14]" = convolution_backward_47[0]
    getitem_352: "f32[2048, 64, 3, 3]" = convolution_backward_47[1];  convolution_backward_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    alias_242: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(relu_52);  relu_52 = None
    alias_243: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(alias_242);  alias_242 = None
    le_47: "b8[8, 2048, 14, 14]" = torch.ops.aten.le.Scalar(alias_243, 0);  alias_243 = None
    where_47: "f32[8, 2048, 14, 14]" = torch.ops.aten.where.self(le_47, full_default, getitem_351);  le_47 = getitem_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    sum_98: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_47, [0, 2, 3])
    sub_296: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_55, unsqueeze_994);  convolution_55 = unsqueeze_994 = None
    mul_1160: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(where_47, sub_296)
    sum_99: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_1160, [0, 2, 3]);  mul_1160 = None
    mul_1161: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_98, 0.0006377551020408163)
    unsqueeze_995: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1161, 0);  mul_1161 = None
    unsqueeze_996: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_995, 2);  unsqueeze_995 = None
    unsqueeze_997: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_996, 3);  unsqueeze_996 = None
    mul_1162: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_99, 0.0006377551020408163)
    mul_1163: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_166, squeeze_166)
    mul_1164: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_1162, mul_1163);  mul_1162 = mul_1163 = None
    unsqueeze_998: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1164, 0);  mul_1164 = None
    unsqueeze_999: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_998, 2);  unsqueeze_998 = None
    unsqueeze_1000: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_999, 3);  unsqueeze_999 = None
    mul_1165: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_166, primals_167);  primals_167 = None
    unsqueeze_1001: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1165, 0);  mul_1165 = None
    unsqueeze_1002: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1001, 2);  unsqueeze_1001 = None
    unsqueeze_1003: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1002, 3);  unsqueeze_1002 = None
    mul_1166: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_296, unsqueeze_1000);  sub_296 = unsqueeze_1000 = None
    sub_298: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(where_47, mul_1166);  where_47 = mul_1166 = None
    sub_299: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(sub_298, unsqueeze_997);  sub_298 = unsqueeze_997 = None
    mul_1167: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_299, unsqueeze_1003);  sub_299 = unsqueeze_1003 = None
    mul_1168: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_99, squeeze_166);  sum_99 = squeeze_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_backward_48 = torch.ops.aten.convolution_backward.default(mul_1167, relu_51, primals_166, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1167 = primals_166 = None
    getitem_354: "f32[8, 1024, 14, 14]" = convolution_backward_48[0]
    getitem_355: "f32[2048, 1024, 1, 1]" = convolution_backward_48[1];  convolution_backward_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    add_568: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(where_45, getitem_354);  where_45 = getitem_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    alias_245: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(relu_51);  relu_51 = None
    alias_246: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(alias_245);  alias_245 = None
    le_48: "b8[8, 1024, 14, 14]" = torch.ops.aten.le.Scalar(alias_246, 0);  alias_246 = None
    where_48: "f32[8, 1024, 14, 14]" = torch.ops.aten.where.self(le_48, full_default, add_568);  le_48 = add_568 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    sum_100: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_48, [0, 2, 3])
    sub_300: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_54, unsqueeze_1006);  convolution_54 = unsqueeze_1006 = None
    mul_1169: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_48, sub_300)
    sum_101: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1169, [0, 2, 3]);  mul_1169 = None
    mul_1170: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_100, 0.0006377551020408163)
    unsqueeze_1007: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1170, 0);  mul_1170 = None
    unsqueeze_1008: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1007, 2);  unsqueeze_1007 = None
    unsqueeze_1009: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1008, 3);  unsqueeze_1008 = None
    mul_1171: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_101, 0.0006377551020408163)
    mul_1172: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_163, squeeze_163)
    mul_1173: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1171, mul_1172);  mul_1171 = mul_1172 = None
    unsqueeze_1010: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1173, 0);  mul_1173 = None
    unsqueeze_1011: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1010, 2);  unsqueeze_1010 = None
    unsqueeze_1012: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1011, 3);  unsqueeze_1011 = None
    mul_1174: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_163, primals_164);  primals_164 = None
    unsqueeze_1013: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1174, 0);  mul_1174 = None
    unsqueeze_1014: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1013, 2);  unsqueeze_1013 = None
    unsqueeze_1015: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1014, 3);  unsqueeze_1014 = None
    mul_1175: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_300, unsqueeze_1012);  sub_300 = unsqueeze_1012 = None
    sub_302: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_48, mul_1175);  mul_1175 = None
    sub_303: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_302, unsqueeze_1009);  sub_302 = unsqueeze_1009 = None
    mul_1176: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_303, unsqueeze_1015);  sub_303 = unsqueeze_1015 = None
    mul_1177: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_101, squeeze_163);  sum_101 = squeeze_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_backward_49 = torch.ops.aten.convolution_backward.default(mul_1176, relu_50, primals_163, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1176 = primals_163 = None
    getitem_357: "f32[8, 2048, 14, 14]" = convolution_backward_49[0]
    getitem_358: "f32[1024, 2048, 1, 1]" = convolution_backward_49[1];  convolution_backward_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    alias_248: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(relu_50);  relu_50 = None
    alias_249: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(alias_248);  alias_248 = None
    le_49: "b8[8, 2048, 14, 14]" = torch.ops.aten.le.Scalar(alias_249, 0);  alias_249 = None
    where_49: "f32[8, 2048, 14, 14]" = torch.ops.aten.where.self(le_49, full_default, getitem_357);  le_49 = getitem_357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    sum_102: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_49, [0, 2, 3])
    sub_304: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_53, unsqueeze_1018);  convolution_53 = unsqueeze_1018 = None
    mul_1178: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(where_49, sub_304)
    sum_103: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_1178, [0, 2, 3]);  mul_1178 = None
    mul_1179: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_102, 0.0006377551020408163)
    unsqueeze_1019: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1179, 0);  mul_1179 = None
    unsqueeze_1020: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1019, 2);  unsqueeze_1019 = None
    unsqueeze_1021: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1020, 3);  unsqueeze_1020 = None
    mul_1180: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_103, 0.0006377551020408163)
    mul_1181: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_160, squeeze_160)
    mul_1182: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_1180, mul_1181);  mul_1180 = mul_1181 = None
    unsqueeze_1022: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1182, 0);  mul_1182 = None
    unsqueeze_1023: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1022, 2);  unsqueeze_1022 = None
    unsqueeze_1024: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1023, 3);  unsqueeze_1023 = None
    mul_1183: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_160, primals_161);  primals_161 = None
    unsqueeze_1025: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1183, 0);  mul_1183 = None
    unsqueeze_1026: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1025, 2);  unsqueeze_1025 = None
    unsqueeze_1027: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1026, 3);  unsqueeze_1026 = None
    mul_1184: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_304, unsqueeze_1024);  sub_304 = unsqueeze_1024 = None
    sub_306: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(where_49, mul_1184);  where_49 = mul_1184 = None
    sub_307: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(sub_306, unsqueeze_1021);  sub_306 = unsqueeze_1021 = None
    mul_1185: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_307, unsqueeze_1027);  sub_307 = unsqueeze_1027 = None
    mul_1186: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_103, squeeze_160);  sum_103 = squeeze_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_backward_50 = torch.ops.aten.convolution_backward.default(mul_1185, relu_49, primals_160, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_1185 = primals_160 = None
    getitem_360: "f32[8, 2048, 14, 14]" = convolution_backward_50[0]
    getitem_361: "f32[2048, 64, 3, 3]" = convolution_backward_50[1];  convolution_backward_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    alias_251: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(relu_49);  relu_49 = None
    alias_252: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(alias_251);  alias_251 = None
    le_50: "b8[8, 2048, 14, 14]" = torch.ops.aten.le.Scalar(alias_252, 0);  alias_252 = None
    where_50: "f32[8, 2048, 14, 14]" = torch.ops.aten.where.self(le_50, full_default, getitem_360);  le_50 = getitem_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    sum_104: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_50, [0, 2, 3])
    sub_308: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_52, unsqueeze_1030);  convolution_52 = unsqueeze_1030 = None
    mul_1187: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(where_50, sub_308)
    sum_105: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_1187, [0, 2, 3]);  mul_1187 = None
    mul_1188: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_104, 0.0006377551020408163)
    unsqueeze_1031: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1188, 0);  mul_1188 = None
    unsqueeze_1032: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1031, 2);  unsqueeze_1031 = None
    unsqueeze_1033: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1032, 3);  unsqueeze_1032 = None
    mul_1189: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_105, 0.0006377551020408163)
    mul_1190: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_157, squeeze_157)
    mul_1191: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_1189, mul_1190);  mul_1189 = mul_1190 = None
    unsqueeze_1034: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1191, 0);  mul_1191 = None
    unsqueeze_1035: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1034, 2);  unsqueeze_1034 = None
    unsqueeze_1036: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1035, 3);  unsqueeze_1035 = None
    mul_1192: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_157, primals_158);  primals_158 = None
    unsqueeze_1037: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1192, 0);  mul_1192 = None
    unsqueeze_1038: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1037, 2);  unsqueeze_1037 = None
    unsqueeze_1039: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1038, 3);  unsqueeze_1038 = None
    mul_1193: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_308, unsqueeze_1036);  sub_308 = unsqueeze_1036 = None
    sub_310: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(where_50, mul_1193);  where_50 = mul_1193 = None
    sub_311: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(sub_310, unsqueeze_1033);  sub_310 = unsqueeze_1033 = None
    mul_1194: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_311, unsqueeze_1039);  sub_311 = unsqueeze_1039 = None
    mul_1195: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_105, squeeze_157);  sum_105 = squeeze_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_backward_51 = torch.ops.aten.convolution_backward.default(mul_1194, relu_48, primals_157, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1194 = primals_157 = None
    getitem_363: "f32[8, 1024, 14, 14]" = convolution_backward_51[0]
    getitem_364: "f32[2048, 1024, 1, 1]" = convolution_backward_51[1];  convolution_backward_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    add_569: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(where_48, getitem_363);  where_48 = getitem_363 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    alias_254: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(relu_48);  relu_48 = None
    alias_255: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(alias_254);  alias_254 = None
    le_51: "b8[8, 1024, 14, 14]" = torch.ops.aten.le.Scalar(alias_255, 0);  alias_255 = None
    where_51: "f32[8, 1024, 14, 14]" = torch.ops.aten.where.self(le_51, full_default, add_569);  le_51 = add_569 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    sum_106: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_51, [0, 2, 3])
    sub_312: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_1042);  convolution_51 = unsqueeze_1042 = None
    mul_1196: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_51, sub_312)
    sum_107: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1196, [0, 2, 3]);  mul_1196 = None
    mul_1197: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_106, 0.0006377551020408163)
    unsqueeze_1043: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1197, 0);  mul_1197 = None
    unsqueeze_1044: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1043, 2);  unsqueeze_1043 = None
    unsqueeze_1045: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1044, 3);  unsqueeze_1044 = None
    mul_1198: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_107, 0.0006377551020408163)
    mul_1199: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_154, squeeze_154)
    mul_1200: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1198, mul_1199);  mul_1198 = mul_1199 = None
    unsqueeze_1046: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1200, 0);  mul_1200 = None
    unsqueeze_1047: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1046, 2);  unsqueeze_1046 = None
    unsqueeze_1048: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1047, 3);  unsqueeze_1047 = None
    mul_1201: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_154, primals_155);  primals_155 = None
    unsqueeze_1049: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1201, 0);  mul_1201 = None
    unsqueeze_1050: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1049, 2);  unsqueeze_1049 = None
    unsqueeze_1051: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1050, 3);  unsqueeze_1050 = None
    mul_1202: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_312, unsqueeze_1048);  sub_312 = unsqueeze_1048 = None
    sub_314: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_51, mul_1202);  mul_1202 = None
    sub_315: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_314, unsqueeze_1045);  sub_314 = unsqueeze_1045 = None
    mul_1203: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_315, unsqueeze_1051);  sub_315 = unsqueeze_1051 = None
    mul_1204: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_107, squeeze_154);  sum_107 = squeeze_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_backward_52 = torch.ops.aten.convolution_backward.default(mul_1203, relu_47, primals_154, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1203 = primals_154 = None
    getitem_366: "f32[8, 2048, 14, 14]" = convolution_backward_52[0]
    getitem_367: "f32[1024, 2048, 1, 1]" = convolution_backward_52[1];  convolution_backward_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    alias_257: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(relu_47);  relu_47 = None
    alias_258: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(alias_257);  alias_257 = None
    le_52: "b8[8, 2048, 14, 14]" = torch.ops.aten.le.Scalar(alias_258, 0);  alias_258 = None
    where_52: "f32[8, 2048, 14, 14]" = torch.ops.aten.where.self(le_52, full_default, getitem_366);  le_52 = getitem_366 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    sum_108: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_52, [0, 2, 3])
    sub_316: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_1054);  convolution_50 = unsqueeze_1054 = None
    mul_1205: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(where_52, sub_316)
    sum_109: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_1205, [0, 2, 3]);  mul_1205 = None
    mul_1206: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_108, 0.0006377551020408163)
    unsqueeze_1055: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1206, 0);  mul_1206 = None
    unsqueeze_1056: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1055, 2);  unsqueeze_1055 = None
    unsqueeze_1057: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1056, 3);  unsqueeze_1056 = None
    mul_1207: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_109, 0.0006377551020408163)
    mul_1208: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_151, squeeze_151)
    mul_1209: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_1207, mul_1208);  mul_1207 = mul_1208 = None
    unsqueeze_1058: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1209, 0);  mul_1209 = None
    unsqueeze_1059: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1058, 2);  unsqueeze_1058 = None
    unsqueeze_1060: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1059, 3);  unsqueeze_1059 = None
    mul_1210: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_151, primals_152);  primals_152 = None
    unsqueeze_1061: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1210, 0);  mul_1210 = None
    unsqueeze_1062: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1061, 2);  unsqueeze_1061 = None
    unsqueeze_1063: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1062, 3);  unsqueeze_1062 = None
    mul_1211: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_316, unsqueeze_1060);  sub_316 = unsqueeze_1060 = None
    sub_318: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(where_52, mul_1211);  where_52 = mul_1211 = None
    sub_319: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(sub_318, unsqueeze_1057);  sub_318 = unsqueeze_1057 = None
    mul_1212: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_319, unsqueeze_1063);  sub_319 = unsqueeze_1063 = None
    mul_1213: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_109, squeeze_151);  sum_109 = squeeze_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_backward_53 = torch.ops.aten.convolution_backward.default(mul_1212, relu_46, primals_151, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_1212 = primals_151 = None
    getitem_369: "f32[8, 2048, 14, 14]" = convolution_backward_53[0]
    getitem_370: "f32[2048, 64, 3, 3]" = convolution_backward_53[1];  convolution_backward_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    alias_260: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(relu_46);  relu_46 = None
    alias_261: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(alias_260);  alias_260 = None
    le_53: "b8[8, 2048, 14, 14]" = torch.ops.aten.le.Scalar(alias_261, 0);  alias_261 = None
    where_53: "f32[8, 2048, 14, 14]" = torch.ops.aten.where.self(le_53, full_default, getitem_369);  le_53 = getitem_369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    sum_110: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_53, [0, 2, 3])
    sub_320: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_49, unsqueeze_1066);  convolution_49 = unsqueeze_1066 = None
    mul_1214: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(where_53, sub_320)
    sum_111: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_1214, [0, 2, 3]);  mul_1214 = None
    mul_1215: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_110, 0.0006377551020408163)
    unsqueeze_1067: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1215, 0);  mul_1215 = None
    unsqueeze_1068: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1067, 2);  unsqueeze_1067 = None
    unsqueeze_1069: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1068, 3);  unsqueeze_1068 = None
    mul_1216: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_111, 0.0006377551020408163)
    mul_1217: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_148, squeeze_148)
    mul_1218: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_1216, mul_1217);  mul_1216 = mul_1217 = None
    unsqueeze_1070: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1218, 0);  mul_1218 = None
    unsqueeze_1071: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1070, 2);  unsqueeze_1070 = None
    unsqueeze_1072: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1071, 3);  unsqueeze_1071 = None
    mul_1219: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_148, primals_149);  primals_149 = None
    unsqueeze_1073: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1219, 0);  mul_1219 = None
    unsqueeze_1074: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1073, 2);  unsqueeze_1073 = None
    unsqueeze_1075: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1074, 3);  unsqueeze_1074 = None
    mul_1220: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_320, unsqueeze_1072);  sub_320 = unsqueeze_1072 = None
    sub_322: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(where_53, mul_1220);  where_53 = mul_1220 = None
    sub_323: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(sub_322, unsqueeze_1069);  sub_322 = unsqueeze_1069 = None
    mul_1221: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_323, unsqueeze_1075);  sub_323 = unsqueeze_1075 = None
    mul_1222: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_111, squeeze_148);  sum_111 = squeeze_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_backward_54 = torch.ops.aten.convolution_backward.default(mul_1221, relu_45, primals_148, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1221 = primals_148 = None
    getitem_372: "f32[8, 1024, 14, 14]" = convolution_backward_54[0]
    getitem_373: "f32[2048, 1024, 1, 1]" = convolution_backward_54[1];  convolution_backward_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    add_570: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(where_51, getitem_372);  where_51 = getitem_372 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    alias_263: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(relu_45);  relu_45 = None
    alias_264: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(alias_263);  alias_263 = None
    le_54: "b8[8, 1024, 14, 14]" = torch.ops.aten.le.Scalar(alias_264, 0);  alias_264 = None
    where_54: "f32[8, 1024, 14, 14]" = torch.ops.aten.where.self(le_54, full_default, add_570);  le_54 = add_570 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    sum_112: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_54, [0, 2, 3])
    sub_324: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_48, unsqueeze_1078);  convolution_48 = unsqueeze_1078 = None
    mul_1223: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_54, sub_324)
    sum_113: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1223, [0, 2, 3]);  mul_1223 = None
    mul_1224: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_112, 0.0006377551020408163)
    unsqueeze_1079: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1224, 0);  mul_1224 = None
    unsqueeze_1080: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1079, 2);  unsqueeze_1079 = None
    unsqueeze_1081: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1080, 3);  unsqueeze_1080 = None
    mul_1225: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_113, 0.0006377551020408163)
    mul_1226: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_145, squeeze_145)
    mul_1227: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1225, mul_1226);  mul_1225 = mul_1226 = None
    unsqueeze_1082: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1227, 0);  mul_1227 = None
    unsqueeze_1083: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1082, 2);  unsqueeze_1082 = None
    unsqueeze_1084: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1083, 3);  unsqueeze_1083 = None
    mul_1228: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_145, primals_146);  primals_146 = None
    unsqueeze_1085: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1228, 0);  mul_1228 = None
    unsqueeze_1086: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1085, 2);  unsqueeze_1085 = None
    unsqueeze_1087: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1086, 3);  unsqueeze_1086 = None
    mul_1229: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_324, unsqueeze_1084);  sub_324 = unsqueeze_1084 = None
    sub_326: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_54, mul_1229);  mul_1229 = None
    sub_327: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_326, unsqueeze_1081);  sub_326 = unsqueeze_1081 = None
    mul_1230: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_327, unsqueeze_1087);  sub_327 = unsqueeze_1087 = None
    mul_1231: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_113, squeeze_145);  sum_113 = squeeze_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_backward_55 = torch.ops.aten.convolution_backward.default(mul_1230, relu_44, primals_145, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1230 = primals_145 = None
    getitem_375: "f32[8, 2048, 14, 14]" = convolution_backward_55[0]
    getitem_376: "f32[1024, 2048, 1, 1]" = convolution_backward_55[1];  convolution_backward_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    alias_266: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(relu_44);  relu_44 = None
    alias_267: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(alias_266);  alias_266 = None
    le_55: "b8[8, 2048, 14, 14]" = torch.ops.aten.le.Scalar(alias_267, 0);  alias_267 = None
    where_55: "f32[8, 2048, 14, 14]" = torch.ops.aten.where.self(le_55, full_default, getitem_375);  le_55 = getitem_375 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    sum_114: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_55, [0, 2, 3])
    sub_328: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_47, unsqueeze_1090);  convolution_47 = unsqueeze_1090 = None
    mul_1232: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(where_55, sub_328)
    sum_115: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_1232, [0, 2, 3]);  mul_1232 = None
    mul_1233: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_114, 0.0006377551020408163)
    unsqueeze_1091: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1233, 0);  mul_1233 = None
    unsqueeze_1092: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1091, 2);  unsqueeze_1091 = None
    unsqueeze_1093: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1092, 3);  unsqueeze_1092 = None
    mul_1234: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_115, 0.0006377551020408163)
    mul_1235: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_142, squeeze_142)
    mul_1236: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_1234, mul_1235);  mul_1234 = mul_1235 = None
    unsqueeze_1094: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1236, 0);  mul_1236 = None
    unsqueeze_1095: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1094, 2);  unsqueeze_1094 = None
    unsqueeze_1096: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1095, 3);  unsqueeze_1095 = None
    mul_1237: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_142, primals_143);  primals_143 = None
    unsqueeze_1097: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1237, 0);  mul_1237 = None
    unsqueeze_1098: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1097, 2);  unsqueeze_1097 = None
    unsqueeze_1099: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1098, 3);  unsqueeze_1098 = None
    mul_1238: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_328, unsqueeze_1096);  sub_328 = unsqueeze_1096 = None
    sub_330: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(where_55, mul_1238);  where_55 = mul_1238 = None
    sub_331: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(sub_330, unsqueeze_1093);  sub_330 = unsqueeze_1093 = None
    mul_1239: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_331, unsqueeze_1099);  sub_331 = unsqueeze_1099 = None
    mul_1240: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_115, squeeze_142);  sum_115 = squeeze_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_backward_56 = torch.ops.aten.convolution_backward.default(mul_1239, relu_43, primals_142, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_1239 = primals_142 = None
    getitem_378: "f32[8, 2048, 14, 14]" = convolution_backward_56[0]
    getitem_379: "f32[2048, 64, 3, 3]" = convolution_backward_56[1];  convolution_backward_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    alias_269: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(relu_43);  relu_43 = None
    alias_270: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(alias_269);  alias_269 = None
    le_56: "b8[8, 2048, 14, 14]" = torch.ops.aten.le.Scalar(alias_270, 0);  alias_270 = None
    where_56: "f32[8, 2048, 14, 14]" = torch.ops.aten.where.self(le_56, full_default, getitem_378);  le_56 = getitem_378 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    sum_116: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_56, [0, 2, 3])
    sub_332: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_1102);  convolution_46 = unsqueeze_1102 = None
    mul_1241: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(where_56, sub_332)
    sum_117: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_1241, [0, 2, 3]);  mul_1241 = None
    mul_1242: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_116, 0.0006377551020408163)
    unsqueeze_1103: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1242, 0);  mul_1242 = None
    unsqueeze_1104: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1103, 2);  unsqueeze_1103 = None
    unsqueeze_1105: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1104, 3);  unsqueeze_1104 = None
    mul_1243: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_117, 0.0006377551020408163)
    mul_1244: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_139, squeeze_139)
    mul_1245: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_1243, mul_1244);  mul_1243 = mul_1244 = None
    unsqueeze_1106: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1245, 0);  mul_1245 = None
    unsqueeze_1107: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1106, 2);  unsqueeze_1106 = None
    unsqueeze_1108: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1107, 3);  unsqueeze_1107 = None
    mul_1246: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_139, primals_140);  primals_140 = None
    unsqueeze_1109: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1246, 0);  mul_1246 = None
    unsqueeze_1110: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1109, 2);  unsqueeze_1109 = None
    unsqueeze_1111: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1110, 3);  unsqueeze_1110 = None
    mul_1247: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_332, unsqueeze_1108);  sub_332 = unsqueeze_1108 = None
    sub_334: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(where_56, mul_1247);  where_56 = mul_1247 = None
    sub_335: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(sub_334, unsqueeze_1105);  sub_334 = unsqueeze_1105 = None
    mul_1248: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_335, unsqueeze_1111);  sub_335 = unsqueeze_1111 = None
    mul_1249: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_117, squeeze_139);  sum_117 = squeeze_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_backward_57 = torch.ops.aten.convolution_backward.default(mul_1248, relu_42, primals_139, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1248 = primals_139 = None
    getitem_381: "f32[8, 1024, 14, 14]" = convolution_backward_57[0]
    getitem_382: "f32[2048, 1024, 1, 1]" = convolution_backward_57[1];  convolution_backward_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    add_571: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(where_54, getitem_381);  where_54 = getitem_381 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    alias_272: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(relu_42);  relu_42 = None
    alias_273: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(alias_272);  alias_272 = None
    le_57: "b8[8, 1024, 14, 14]" = torch.ops.aten.le.Scalar(alias_273, 0);  alias_273 = None
    where_57: "f32[8, 1024, 14, 14]" = torch.ops.aten.where.self(le_57, full_default, add_571);  le_57 = add_571 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    sum_118: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_57, [0, 2, 3])
    sub_336: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_1114);  convolution_45 = unsqueeze_1114 = None
    mul_1250: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_57, sub_336)
    sum_119: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1250, [0, 2, 3]);  mul_1250 = None
    mul_1251: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_118, 0.0006377551020408163)
    unsqueeze_1115: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1251, 0);  mul_1251 = None
    unsqueeze_1116: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1115, 2);  unsqueeze_1115 = None
    unsqueeze_1117: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1116, 3);  unsqueeze_1116 = None
    mul_1252: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_119, 0.0006377551020408163)
    mul_1253: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_136, squeeze_136)
    mul_1254: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1252, mul_1253);  mul_1252 = mul_1253 = None
    unsqueeze_1118: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1254, 0);  mul_1254 = None
    unsqueeze_1119: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1118, 2);  unsqueeze_1118 = None
    unsqueeze_1120: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1119, 3);  unsqueeze_1119 = None
    mul_1255: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_136, primals_137);  primals_137 = None
    unsqueeze_1121: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1255, 0);  mul_1255 = None
    unsqueeze_1122: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1121, 2);  unsqueeze_1121 = None
    unsqueeze_1123: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1122, 3);  unsqueeze_1122 = None
    mul_1256: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_336, unsqueeze_1120);  sub_336 = unsqueeze_1120 = None
    sub_338: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_57, mul_1256);  mul_1256 = None
    sub_339: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_338, unsqueeze_1117);  sub_338 = unsqueeze_1117 = None
    mul_1257: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_339, unsqueeze_1123);  sub_339 = unsqueeze_1123 = None
    mul_1258: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_119, squeeze_136);  sum_119 = squeeze_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_backward_58 = torch.ops.aten.convolution_backward.default(mul_1257, relu_41, primals_136, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1257 = primals_136 = None
    getitem_384: "f32[8, 2048, 14, 14]" = convolution_backward_58[0]
    getitem_385: "f32[1024, 2048, 1, 1]" = convolution_backward_58[1];  convolution_backward_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    alias_275: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(relu_41);  relu_41 = None
    alias_276: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(alias_275);  alias_275 = None
    le_58: "b8[8, 2048, 14, 14]" = torch.ops.aten.le.Scalar(alias_276, 0);  alias_276 = None
    where_58: "f32[8, 2048, 14, 14]" = torch.ops.aten.where.self(le_58, full_default, getitem_384);  le_58 = getitem_384 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    sum_120: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_58, [0, 2, 3])
    sub_340: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_44, unsqueeze_1126);  convolution_44 = unsqueeze_1126 = None
    mul_1259: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(where_58, sub_340)
    sum_121: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_1259, [0, 2, 3]);  mul_1259 = None
    mul_1260: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_120, 0.0006377551020408163)
    unsqueeze_1127: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1260, 0);  mul_1260 = None
    unsqueeze_1128: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1127, 2);  unsqueeze_1127 = None
    unsqueeze_1129: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1128, 3);  unsqueeze_1128 = None
    mul_1261: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_121, 0.0006377551020408163)
    mul_1262: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_133, squeeze_133)
    mul_1263: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_1261, mul_1262);  mul_1261 = mul_1262 = None
    unsqueeze_1130: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1263, 0);  mul_1263 = None
    unsqueeze_1131: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1130, 2);  unsqueeze_1130 = None
    unsqueeze_1132: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1131, 3);  unsqueeze_1131 = None
    mul_1264: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_133, primals_134);  primals_134 = None
    unsqueeze_1133: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1264, 0);  mul_1264 = None
    unsqueeze_1134: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1133, 2);  unsqueeze_1133 = None
    unsqueeze_1135: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1134, 3);  unsqueeze_1134 = None
    mul_1265: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_340, unsqueeze_1132);  sub_340 = unsqueeze_1132 = None
    sub_342: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(where_58, mul_1265);  where_58 = mul_1265 = None
    sub_343: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(sub_342, unsqueeze_1129);  sub_342 = unsqueeze_1129 = None
    mul_1266: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_343, unsqueeze_1135);  sub_343 = unsqueeze_1135 = None
    mul_1267: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_121, squeeze_133);  sum_121 = squeeze_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_backward_59 = torch.ops.aten.convolution_backward.default(mul_1266, relu_40, primals_133, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_1266 = primals_133 = None
    getitem_387: "f32[8, 2048, 14, 14]" = convolution_backward_59[0]
    getitem_388: "f32[2048, 64, 3, 3]" = convolution_backward_59[1];  convolution_backward_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    alias_278: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(relu_40);  relu_40 = None
    alias_279: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(alias_278);  alias_278 = None
    le_59: "b8[8, 2048, 14, 14]" = torch.ops.aten.le.Scalar(alias_279, 0);  alias_279 = None
    where_59: "f32[8, 2048, 14, 14]" = torch.ops.aten.where.self(le_59, full_default, getitem_387);  le_59 = getitem_387 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    sum_122: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_59, [0, 2, 3])
    sub_344: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_43, unsqueeze_1138);  convolution_43 = unsqueeze_1138 = None
    mul_1268: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(where_59, sub_344)
    sum_123: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_1268, [0, 2, 3]);  mul_1268 = None
    mul_1269: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_122, 0.0006377551020408163)
    unsqueeze_1139: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1269, 0);  mul_1269 = None
    unsqueeze_1140: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1139, 2);  unsqueeze_1139 = None
    unsqueeze_1141: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1140, 3);  unsqueeze_1140 = None
    mul_1270: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_123, 0.0006377551020408163)
    mul_1271: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_130, squeeze_130)
    mul_1272: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_1270, mul_1271);  mul_1270 = mul_1271 = None
    unsqueeze_1142: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1272, 0);  mul_1272 = None
    unsqueeze_1143: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1142, 2);  unsqueeze_1142 = None
    unsqueeze_1144: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1143, 3);  unsqueeze_1143 = None
    mul_1273: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_130, primals_131);  primals_131 = None
    unsqueeze_1145: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1273, 0);  mul_1273 = None
    unsqueeze_1146: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1145, 2);  unsqueeze_1145 = None
    unsqueeze_1147: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1146, 3);  unsqueeze_1146 = None
    mul_1274: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_344, unsqueeze_1144);  sub_344 = unsqueeze_1144 = None
    sub_346: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(where_59, mul_1274);  where_59 = mul_1274 = None
    sub_347: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(sub_346, unsqueeze_1141);  sub_346 = unsqueeze_1141 = None
    mul_1275: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_347, unsqueeze_1147);  sub_347 = unsqueeze_1147 = None
    mul_1276: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_123, squeeze_130);  sum_123 = squeeze_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_backward_60 = torch.ops.aten.convolution_backward.default(mul_1275, relu_39, primals_130, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1275 = primals_130 = None
    getitem_390: "f32[8, 1024, 14, 14]" = convolution_backward_60[0]
    getitem_391: "f32[2048, 1024, 1, 1]" = convolution_backward_60[1];  convolution_backward_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    add_572: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(where_57, getitem_390);  where_57 = getitem_390 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    alias_281: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(relu_39);  relu_39 = None
    alias_282: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(alias_281);  alias_281 = None
    le_60: "b8[8, 1024, 14, 14]" = torch.ops.aten.le.Scalar(alias_282, 0);  alias_282 = None
    where_60: "f32[8, 1024, 14, 14]" = torch.ops.aten.where.self(le_60, full_default, add_572);  le_60 = add_572 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    sum_124: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_60, [0, 2, 3])
    sub_348: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_42, unsqueeze_1150);  convolution_42 = unsqueeze_1150 = None
    mul_1277: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_60, sub_348)
    sum_125: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1277, [0, 2, 3]);  mul_1277 = None
    mul_1278: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_124, 0.0006377551020408163)
    unsqueeze_1151: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1278, 0);  mul_1278 = None
    unsqueeze_1152: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1151, 2);  unsqueeze_1151 = None
    unsqueeze_1153: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1152, 3);  unsqueeze_1152 = None
    mul_1279: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_125, 0.0006377551020408163)
    mul_1280: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_127, squeeze_127)
    mul_1281: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1279, mul_1280);  mul_1279 = mul_1280 = None
    unsqueeze_1154: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1281, 0);  mul_1281 = None
    unsqueeze_1155: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1154, 2);  unsqueeze_1154 = None
    unsqueeze_1156: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1155, 3);  unsqueeze_1155 = None
    mul_1282: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_127, primals_128);  primals_128 = None
    unsqueeze_1157: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1282, 0);  mul_1282 = None
    unsqueeze_1158: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1157, 2);  unsqueeze_1157 = None
    unsqueeze_1159: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1158, 3);  unsqueeze_1158 = None
    mul_1283: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_348, unsqueeze_1156);  sub_348 = unsqueeze_1156 = None
    sub_350: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_60, mul_1283);  mul_1283 = None
    sub_351: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_350, unsqueeze_1153);  sub_350 = unsqueeze_1153 = None
    mul_1284: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_351, unsqueeze_1159);  sub_351 = unsqueeze_1159 = None
    mul_1285: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_125, squeeze_127);  sum_125 = squeeze_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_backward_61 = torch.ops.aten.convolution_backward.default(mul_1284, relu_38, primals_127, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1284 = primals_127 = None
    getitem_393: "f32[8, 2048, 14, 14]" = convolution_backward_61[0]
    getitem_394: "f32[1024, 2048, 1, 1]" = convolution_backward_61[1];  convolution_backward_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    alias_284: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(relu_38);  relu_38 = None
    alias_285: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(alias_284);  alias_284 = None
    le_61: "b8[8, 2048, 14, 14]" = torch.ops.aten.le.Scalar(alias_285, 0);  alias_285 = None
    where_61: "f32[8, 2048, 14, 14]" = torch.ops.aten.where.self(le_61, full_default, getitem_393);  le_61 = getitem_393 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    sum_126: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_61, [0, 2, 3])
    sub_352: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_1162);  convolution_41 = unsqueeze_1162 = None
    mul_1286: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(where_61, sub_352)
    sum_127: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_1286, [0, 2, 3]);  mul_1286 = None
    mul_1287: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_126, 0.0006377551020408163)
    unsqueeze_1163: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1287, 0);  mul_1287 = None
    unsqueeze_1164: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1163, 2);  unsqueeze_1163 = None
    unsqueeze_1165: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1164, 3);  unsqueeze_1164 = None
    mul_1288: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_127, 0.0006377551020408163)
    mul_1289: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_124, squeeze_124)
    mul_1290: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_1288, mul_1289);  mul_1288 = mul_1289 = None
    unsqueeze_1166: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1290, 0);  mul_1290 = None
    unsqueeze_1167: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1166, 2);  unsqueeze_1166 = None
    unsqueeze_1168: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1167, 3);  unsqueeze_1167 = None
    mul_1291: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_124, primals_125);  primals_125 = None
    unsqueeze_1169: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1291, 0);  mul_1291 = None
    unsqueeze_1170: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1169, 2);  unsqueeze_1169 = None
    unsqueeze_1171: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1170, 3);  unsqueeze_1170 = None
    mul_1292: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_352, unsqueeze_1168);  sub_352 = unsqueeze_1168 = None
    sub_354: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(where_61, mul_1292);  where_61 = mul_1292 = None
    sub_355: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(sub_354, unsqueeze_1165);  sub_354 = unsqueeze_1165 = None
    mul_1293: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_355, unsqueeze_1171);  sub_355 = unsqueeze_1171 = None
    mul_1294: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_127, squeeze_124);  sum_127 = squeeze_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_backward_62 = torch.ops.aten.convolution_backward.default(mul_1293, relu_37, primals_124, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_1293 = primals_124 = None
    getitem_396: "f32[8, 2048, 14, 14]" = convolution_backward_62[0]
    getitem_397: "f32[2048, 64, 3, 3]" = convolution_backward_62[1];  convolution_backward_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    alias_287: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(relu_37);  relu_37 = None
    alias_288: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(alias_287);  alias_287 = None
    le_62: "b8[8, 2048, 14, 14]" = torch.ops.aten.le.Scalar(alias_288, 0);  alias_288 = None
    where_62: "f32[8, 2048, 14, 14]" = torch.ops.aten.where.self(le_62, full_default, getitem_396);  le_62 = getitem_396 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    sum_128: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_62, [0, 2, 3])
    sub_356: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_1174);  convolution_40 = unsqueeze_1174 = None
    mul_1295: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(where_62, sub_356)
    sum_129: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_1295, [0, 2, 3]);  mul_1295 = None
    mul_1296: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_128, 0.0006377551020408163)
    unsqueeze_1175: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1296, 0);  mul_1296 = None
    unsqueeze_1176: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1175, 2);  unsqueeze_1175 = None
    unsqueeze_1177: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1176, 3);  unsqueeze_1176 = None
    mul_1297: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_129, 0.0006377551020408163)
    mul_1298: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_121, squeeze_121)
    mul_1299: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_1297, mul_1298);  mul_1297 = mul_1298 = None
    unsqueeze_1178: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1299, 0);  mul_1299 = None
    unsqueeze_1179: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1178, 2);  unsqueeze_1178 = None
    unsqueeze_1180: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1179, 3);  unsqueeze_1179 = None
    mul_1300: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_121, primals_122);  primals_122 = None
    unsqueeze_1181: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1300, 0);  mul_1300 = None
    unsqueeze_1182: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1181, 2);  unsqueeze_1181 = None
    unsqueeze_1183: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1182, 3);  unsqueeze_1182 = None
    mul_1301: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_356, unsqueeze_1180);  sub_356 = unsqueeze_1180 = None
    sub_358: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(where_62, mul_1301);  where_62 = mul_1301 = None
    sub_359: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(sub_358, unsqueeze_1177);  sub_358 = unsqueeze_1177 = None
    mul_1302: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_359, unsqueeze_1183);  sub_359 = unsqueeze_1183 = None
    mul_1303: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_129, squeeze_121);  sum_129 = squeeze_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_backward_63 = torch.ops.aten.convolution_backward.default(mul_1302, relu_36, primals_121, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1302 = primals_121 = None
    getitem_399: "f32[8, 1024, 14, 14]" = convolution_backward_63[0]
    getitem_400: "f32[2048, 1024, 1, 1]" = convolution_backward_63[1];  convolution_backward_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    add_573: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(where_60, getitem_399);  where_60 = getitem_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    alias_290: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(relu_36);  relu_36 = None
    alias_291: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(alias_290);  alias_290 = None
    le_63: "b8[8, 1024, 14, 14]" = torch.ops.aten.le.Scalar(alias_291, 0);  alias_291 = None
    where_63: "f32[8, 1024, 14, 14]" = torch.ops.aten.where.self(le_63, full_default, add_573);  le_63 = add_573 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    sum_130: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_63, [0, 2, 3])
    sub_360: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_39, unsqueeze_1186);  convolution_39 = unsqueeze_1186 = None
    mul_1304: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_63, sub_360)
    sum_131: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1304, [0, 2, 3]);  mul_1304 = None
    mul_1305: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_130, 0.0006377551020408163)
    unsqueeze_1187: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1305, 0);  mul_1305 = None
    unsqueeze_1188: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1187, 2);  unsqueeze_1187 = None
    unsqueeze_1189: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1188, 3);  unsqueeze_1188 = None
    mul_1306: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_131, 0.0006377551020408163)
    mul_1307: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_118, squeeze_118)
    mul_1308: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1306, mul_1307);  mul_1306 = mul_1307 = None
    unsqueeze_1190: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1308, 0);  mul_1308 = None
    unsqueeze_1191: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1190, 2);  unsqueeze_1190 = None
    unsqueeze_1192: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1191, 3);  unsqueeze_1191 = None
    mul_1309: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_118, primals_119);  primals_119 = None
    unsqueeze_1193: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1309, 0);  mul_1309 = None
    unsqueeze_1194: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1193, 2);  unsqueeze_1193 = None
    unsqueeze_1195: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1194, 3);  unsqueeze_1194 = None
    mul_1310: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_360, unsqueeze_1192);  sub_360 = unsqueeze_1192 = None
    sub_362: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_63, mul_1310);  mul_1310 = None
    sub_363: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_362, unsqueeze_1189);  sub_362 = unsqueeze_1189 = None
    mul_1311: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_363, unsqueeze_1195);  sub_363 = unsqueeze_1195 = None
    mul_1312: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_131, squeeze_118);  sum_131 = squeeze_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_backward_64 = torch.ops.aten.convolution_backward.default(mul_1311, relu_35, primals_118, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1311 = primals_118 = None
    getitem_402: "f32[8, 2048, 14, 14]" = convolution_backward_64[0]
    getitem_403: "f32[1024, 2048, 1, 1]" = convolution_backward_64[1];  convolution_backward_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    alias_293: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(relu_35);  relu_35 = None
    alias_294: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(alias_293);  alias_293 = None
    le_64: "b8[8, 2048, 14, 14]" = torch.ops.aten.le.Scalar(alias_294, 0);  alias_294 = None
    where_64: "f32[8, 2048, 14, 14]" = torch.ops.aten.where.self(le_64, full_default, getitem_402);  le_64 = getitem_402 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    sum_132: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_64, [0, 2, 3])
    sub_364: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_38, unsqueeze_1198);  convolution_38 = unsqueeze_1198 = None
    mul_1313: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(where_64, sub_364)
    sum_133: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_1313, [0, 2, 3]);  mul_1313 = None
    mul_1314: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_132, 0.0006377551020408163)
    unsqueeze_1199: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1314, 0);  mul_1314 = None
    unsqueeze_1200: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1199, 2);  unsqueeze_1199 = None
    unsqueeze_1201: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1200, 3);  unsqueeze_1200 = None
    mul_1315: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_133, 0.0006377551020408163)
    mul_1316: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_115, squeeze_115)
    mul_1317: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_1315, mul_1316);  mul_1315 = mul_1316 = None
    unsqueeze_1202: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1317, 0);  mul_1317 = None
    unsqueeze_1203: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1202, 2);  unsqueeze_1202 = None
    unsqueeze_1204: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1203, 3);  unsqueeze_1203 = None
    mul_1318: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_115, primals_116);  primals_116 = None
    unsqueeze_1205: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1318, 0);  mul_1318 = None
    unsqueeze_1206: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1205, 2);  unsqueeze_1205 = None
    unsqueeze_1207: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1206, 3);  unsqueeze_1206 = None
    mul_1319: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_364, unsqueeze_1204);  sub_364 = unsqueeze_1204 = None
    sub_366: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(where_64, mul_1319);  where_64 = mul_1319 = None
    sub_367: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(sub_366, unsqueeze_1201);  sub_366 = unsqueeze_1201 = None
    mul_1320: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_367, unsqueeze_1207);  sub_367 = unsqueeze_1207 = None
    mul_1321: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_133, squeeze_115);  sum_133 = squeeze_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_backward_65 = torch.ops.aten.convolution_backward.default(mul_1320, relu_34, primals_115, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_1320 = primals_115 = None
    getitem_405: "f32[8, 2048, 14, 14]" = convolution_backward_65[0]
    getitem_406: "f32[2048, 64, 3, 3]" = convolution_backward_65[1];  convolution_backward_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    alias_296: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(relu_34);  relu_34 = None
    alias_297: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(alias_296);  alias_296 = None
    le_65: "b8[8, 2048, 14, 14]" = torch.ops.aten.le.Scalar(alias_297, 0);  alias_297 = None
    where_65: "f32[8, 2048, 14, 14]" = torch.ops.aten.where.self(le_65, full_default, getitem_405);  le_65 = getitem_405 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    sum_134: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_65, [0, 2, 3])
    sub_368: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_37, unsqueeze_1210);  convolution_37 = unsqueeze_1210 = None
    mul_1322: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(where_65, sub_368)
    sum_135: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_1322, [0, 2, 3]);  mul_1322 = None
    mul_1323: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_134, 0.0006377551020408163)
    unsqueeze_1211: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1323, 0);  mul_1323 = None
    unsqueeze_1212: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1211, 2);  unsqueeze_1211 = None
    unsqueeze_1213: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1212, 3);  unsqueeze_1212 = None
    mul_1324: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_135, 0.0006377551020408163)
    mul_1325: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_112, squeeze_112)
    mul_1326: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_1324, mul_1325);  mul_1324 = mul_1325 = None
    unsqueeze_1214: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1326, 0);  mul_1326 = None
    unsqueeze_1215: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1214, 2);  unsqueeze_1214 = None
    unsqueeze_1216: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1215, 3);  unsqueeze_1215 = None
    mul_1327: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_112, primals_113);  primals_113 = None
    unsqueeze_1217: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1327, 0);  mul_1327 = None
    unsqueeze_1218: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1217, 2);  unsqueeze_1217 = None
    unsqueeze_1219: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1218, 3);  unsqueeze_1218 = None
    mul_1328: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_368, unsqueeze_1216);  sub_368 = unsqueeze_1216 = None
    sub_370: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(where_65, mul_1328);  where_65 = mul_1328 = None
    sub_371: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(sub_370, unsqueeze_1213);  sub_370 = unsqueeze_1213 = None
    mul_1329: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_371, unsqueeze_1219);  sub_371 = unsqueeze_1219 = None
    mul_1330: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_135, squeeze_112);  sum_135 = squeeze_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_backward_66 = torch.ops.aten.convolution_backward.default(mul_1329, relu_33, primals_112, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1329 = primals_112 = None
    getitem_408: "f32[8, 1024, 14, 14]" = convolution_backward_66[0]
    getitem_409: "f32[2048, 1024, 1, 1]" = convolution_backward_66[1];  convolution_backward_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    add_574: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(where_63, getitem_408);  where_63 = getitem_408 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    alias_299: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(relu_33);  relu_33 = None
    alias_300: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(alias_299);  alias_299 = None
    le_66: "b8[8, 1024, 14, 14]" = torch.ops.aten.le.Scalar(alias_300, 0);  alias_300 = None
    where_66: "f32[8, 1024, 14, 14]" = torch.ops.aten.where.self(le_66, full_default, add_574);  le_66 = add_574 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    sum_136: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_66, [0, 2, 3])
    sub_372: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_1222);  convolution_36 = unsqueeze_1222 = None
    mul_1331: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_66, sub_372)
    sum_137: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1331, [0, 2, 3]);  mul_1331 = None
    mul_1332: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_136, 0.0006377551020408163)
    unsqueeze_1223: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1332, 0);  mul_1332 = None
    unsqueeze_1224: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1223, 2);  unsqueeze_1223 = None
    unsqueeze_1225: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1224, 3);  unsqueeze_1224 = None
    mul_1333: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_137, 0.0006377551020408163)
    mul_1334: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_109, squeeze_109)
    mul_1335: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1333, mul_1334);  mul_1333 = mul_1334 = None
    unsqueeze_1226: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1335, 0);  mul_1335 = None
    unsqueeze_1227: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1226, 2);  unsqueeze_1226 = None
    unsqueeze_1228: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1227, 3);  unsqueeze_1227 = None
    mul_1336: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_109, primals_110);  primals_110 = None
    unsqueeze_1229: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1336, 0);  mul_1336 = None
    unsqueeze_1230: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1229, 2);  unsqueeze_1229 = None
    unsqueeze_1231: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1230, 3);  unsqueeze_1230 = None
    mul_1337: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_372, unsqueeze_1228);  sub_372 = unsqueeze_1228 = None
    sub_374: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_66, mul_1337);  mul_1337 = None
    sub_375: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_374, unsqueeze_1225);  sub_374 = unsqueeze_1225 = None
    mul_1338: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_375, unsqueeze_1231);  sub_375 = unsqueeze_1231 = None
    mul_1339: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_137, squeeze_109);  sum_137 = squeeze_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_backward_67 = torch.ops.aten.convolution_backward.default(mul_1338, relu_32, primals_109, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1338 = primals_109 = None
    getitem_411: "f32[8, 2048, 14, 14]" = convolution_backward_67[0]
    getitem_412: "f32[1024, 2048, 1, 1]" = convolution_backward_67[1];  convolution_backward_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    alias_302: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(relu_32);  relu_32 = None
    alias_303: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(alias_302);  alias_302 = None
    le_67: "b8[8, 2048, 14, 14]" = torch.ops.aten.le.Scalar(alias_303, 0);  alias_303 = None
    where_67: "f32[8, 2048, 14, 14]" = torch.ops.aten.where.self(le_67, full_default, getitem_411);  le_67 = getitem_411 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    sum_138: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_67, [0, 2, 3])
    sub_376: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_1234);  convolution_35 = unsqueeze_1234 = None
    mul_1340: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(where_67, sub_376)
    sum_139: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_1340, [0, 2, 3]);  mul_1340 = None
    mul_1341: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_138, 0.0006377551020408163)
    unsqueeze_1235: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1341, 0);  mul_1341 = None
    unsqueeze_1236: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1235, 2);  unsqueeze_1235 = None
    unsqueeze_1237: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1236, 3);  unsqueeze_1236 = None
    mul_1342: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_139, 0.0006377551020408163)
    mul_1343: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_106, squeeze_106)
    mul_1344: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_1342, mul_1343);  mul_1342 = mul_1343 = None
    unsqueeze_1238: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1344, 0);  mul_1344 = None
    unsqueeze_1239: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1238, 2);  unsqueeze_1238 = None
    unsqueeze_1240: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1239, 3);  unsqueeze_1239 = None
    mul_1345: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_106, primals_107);  primals_107 = None
    unsqueeze_1241: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1345, 0);  mul_1345 = None
    unsqueeze_1242: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1241, 2);  unsqueeze_1241 = None
    unsqueeze_1243: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1242, 3);  unsqueeze_1242 = None
    mul_1346: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_376, unsqueeze_1240);  sub_376 = unsqueeze_1240 = None
    sub_378: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(where_67, mul_1346);  where_67 = mul_1346 = None
    sub_379: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(sub_378, unsqueeze_1237);  sub_378 = unsqueeze_1237 = None
    mul_1347: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_379, unsqueeze_1243);  sub_379 = unsqueeze_1243 = None
    mul_1348: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_139, squeeze_106);  sum_139 = squeeze_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_backward_68 = torch.ops.aten.convolution_backward.default(mul_1347, relu_31, primals_106, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_1347 = primals_106 = None
    getitem_414: "f32[8, 2048, 14, 14]" = convolution_backward_68[0]
    getitem_415: "f32[2048, 64, 3, 3]" = convolution_backward_68[1];  convolution_backward_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    alias_305: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(relu_31);  relu_31 = None
    alias_306: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(alias_305);  alias_305 = None
    le_68: "b8[8, 2048, 14, 14]" = torch.ops.aten.le.Scalar(alias_306, 0);  alias_306 = None
    where_68: "f32[8, 2048, 14, 14]" = torch.ops.aten.where.self(le_68, full_default, getitem_414);  le_68 = getitem_414 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    sum_140: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_68, [0, 2, 3])
    sub_380: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_1246);  convolution_34 = unsqueeze_1246 = None
    mul_1349: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(where_68, sub_380)
    sum_141: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_1349, [0, 2, 3]);  mul_1349 = None
    mul_1350: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_140, 0.0006377551020408163)
    unsqueeze_1247: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1350, 0);  mul_1350 = None
    unsqueeze_1248: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1247, 2);  unsqueeze_1247 = None
    unsqueeze_1249: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1248, 3);  unsqueeze_1248 = None
    mul_1351: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_141, 0.0006377551020408163)
    mul_1352: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_103, squeeze_103)
    mul_1353: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_1351, mul_1352);  mul_1351 = mul_1352 = None
    unsqueeze_1250: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1353, 0);  mul_1353 = None
    unsqueeze_1251: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1250, 2);  unsqueeze_1250 = None
    unsqueeze_1252: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1251, 3);  unsqueeze_1251 = None
    mul_1354: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_103, primals_104);  primals_104 = None
    unsqueeze_1253: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1354, 0);  mul_1354 = None
    unsqueeze_1254: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1253, 2);  unsqueeze_1253 = None
    unsqueeze_1255: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1254, 3);  unsqueeze_1254 = None
    mul_1355: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_380, unsqueeze_1252);  sub_380 = unsqueeze_1252 = None
    sub_382: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(where_68, mul_1355);  where_68 = mul_1355 = None
    sub_383: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(sub_382, unsqueeze_1249);  sub_382 = unsqueeze_1249 = None
    mul_1356: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_383, unsqueeze_1255);  sub_383 = unsqueeze_1255 = None
    mul_1357: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_141, squeeze_103);  sum_141 = squeeze_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_backward_69 = torch.ops.aten.convolution_backward.default(mul_1356, relu_30, primals_103, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1356 = primals_103 = None
    getitem_417: "f32[8, 1024, 14, 14]" = convolution_backward_69[0]
    getitem_418: "f32[2048, 1024, 1, 1]" = convolution_backward_69[1];  convolution_backward_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    add_575: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(where_66, getitem_417);  where_66 = getitem_417 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    alias_308: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(relu_30);  relu_30 = None
    alias_309: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(alias_308);  alias_308 = None
    le_69: "b8[8, 1024, 14, 14]" = torch.ops.aten.le.Scalar(alias_309, 0);  alias_309 = None
    where_69: "f32[8, 1024, 14, 14]" = torch.ops.aten.where.self(le_69, full_default, add_575);  le_69 = add_575 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    sum_142: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_69, [0, 2, 3])
    sub_384: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_1258);  convolution_33 = unsqueeze_1258 = None
    mul_1358: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_69, sub_384)
    sum_143: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1358, [0, 2, 3]);  mul_1358 = None
    mul_1359: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_142, 0.0006377551020408163)
    unsqueeze_1259: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1359, 0);  mul_1359 = None
    unsqueeze_1260: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1259, 2);  unsqueeze_1259 = None
    unsqueeze_1261: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1260, 3);  unsqueeze_1260 = None
    mul_1360: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_143, 0.0006377551020408163)
    mul_1361: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_100, squeeze_100)
    mul_1362: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1360, mul_1361);  mul_1360 = mul_1361 = None
    unsqueeze_1262: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1362, 0);  mul_1362 = None
    unsqueeze_1263: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1262, 2);  unsqueeze_1262 = None
    unsqueeze_1264: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1263, 3);  unsqueeze_1263 = None
    mul_1363: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_100, primals_101);  primals_101 = None
    unsqueeze_1265: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1363, 0);  mul_1363 = None
    unsqueeze_1266: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1265, 2);  unsqueeze_1265 = None
    unsqueeze_1267: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1266, 3);  unsqueeze_1266 = None
    mul_1364: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_384, unsqueeze_1264);  sub_384 = unsqueeze_1264 = None
    sub_386: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_69, mul_1364);  mul_1364 = None
    sub_387: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_386, unsqueeze_1261);  sub_386 = unsqueeze_1261 = None
    mul_1365: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_387, unsqueeze_1267);  sub_387 = unsqueeze_1267 = None
    mul_1366: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_143, squeeze_100);  sum_143 = squeeze_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_backward_70 = torch.ops.aten.convolution_backward.default(mul_1365, relu_29, primals_100, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1365 = primals_100 = None
    getitem_420: "f32[8, 2048, 14, 14]" = convolution_backward_70[0]
    getitem_421: "f32[1024, 2048, 1, 1]" = convolution_backward_70[1];  convolution_backward_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    alias_311: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(relu_29);  relu_29 = None
    alias_312: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(alias_311);  alias_311 = None
    le_70: "b8[8, 2048, 14, 14]" = torch.ops.aten.le.Scalar(alias_312, 0);  alias_312 = None
    where_70: "f32[8, 2048, 14, 14]" = torch.ops.aten.where.self(le_70, full_default, getitem_420);  le_70 = getitem_420 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    sum_144: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_70, [0, 2, 3])
    sub_388: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_1270);  convolution_32 = unsqueeze_1270 = None
    mul_1367: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(where_70, sub_388)
    sum_145: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_1367, [0, 2, 3]);  mul_1367 = None
    mul_1368: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_144, 0.0006377551020408163)
    unsqueeze_1271: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1368, 0);  mul_1368 = None
    unsqueeze_1272: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1271, 2);  unsqueeze_1271 = None
    unsqueeze_1273: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1272, 3);  unsqueeze_1272 = None
    mul_1369: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_145, 0.0006377551020408163)
    mul_1370: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_97, squeeze_97)
    mul_1371: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_1369, mul_1370);  mul_1369 = mul_1370 = None
    unsqueeze_1274: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1371, 0);  mul_1371 = None
    unsqueeze_1275: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1274, 2);  unsqueeze_1274 = None
    unsqueeze_1276: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1275, 3);  unsqueeze_1275 = None
    mul_1372: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_97, primals_98);  primals_98 = None
    unsqueeze_1277: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1372, 0);  mul_1372 = None
    unsqueeze_1278: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1277, 2);  unsqueeze_1277 = None
    unsqueeze_1279: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1278, 3);  unsqueeze_1278 = None
    mul_1373: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_388, unsqueeze_1276);  sub_388 = unsqueeze_1276 = None
    sub_390: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(where_70, mul_1373);  where_70 = mul_1373 = None
    sub_391: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(sub_390, unsqueeze_1273);  sub_390 = unsqueeze_1273 = None
    mul_1374: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_391, unsqueeze_1279);  sub_391 = unsqueeze_1279 = None
    mul_1375: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_145, squeeze_97);  sum_145 = squeeze_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_backward_71 = torch.ops.aten.convolution_backward.default(mul_1374, relu_28, primals_97, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_1374 = primals_97 = None
    getitem_423: "f32[8, 2048, 14, 14]" = convolution_backward_71[0]
    getitem_424: "f32[2048, 64, 3, 3]" = convolution_backward_71[1];  convolution_backward_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    alias_314: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(relu_28);  relu_28 = None
    alias_315: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(alias_314);  alias_314 = None
    le_71: "b8[8, 2048, 14, 14]" = torch.ops.aten.le.Scalar(alias_315, 0);  alias_315 = None
    where_71: "f32[8, 2048, 14, 14]" = torch.ops.aten.where.self(le_71, full_default, getitem_423);  le_71 = getitem_423 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    sum_146: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_71, [0, 2, 3])
    sub_392: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_31, unsqueeze_1282);  convolution_31 = unsqueeze_1282 = None
    mul_1376: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(where_71, sub_392)
    sum_147: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_1376, [0, 2, 3]);  mul_1376 = None
    mul_1377: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_146, 0.0006377551020408163)
    unsqueeze_1283: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1377, 0);  mul_1377 = None
    unsqueeze_1284: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1283, 2);  unsqueeze_1283 = None
    unsqueeze_1285: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1284, 3);  unsqueeze_1284 = None
    mul_1378: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_147, 0.0006377551020408163)
    mul_1379: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_94, squeeze_94)
    mul_1380: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_1378, mul_1379);  mul_1378 = mul_1379 = None
    unsqueeze_1286: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1380, 0);  mul_1380 = None
    unsqueeze_1287: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1286, 2);  unsqueeze_1286 = None
    unsqueeze_1288: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1287, 3);  unsqueeze_1287 = None
    mul_1381: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_94, primals_95);  primals_95 = None
    unsqueeze_1289: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1381, 0);  mul_1381 = None
    unsqueeze_1290: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1289, 2);  unsqueeze_1289 = None
    unsqueeze_1291: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1290, 3);  unsqueeze_1290 = None
    mul_1382: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_392, unsqueeze_1288);  sub_392 = unsqueeze_1288 = None
    sub_394: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(where_71, mul_1382);  where_71 = mul_1382 = None
    sub_395: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(sub_394, unsqueeze_1285);  sub_394 = unsqueeze_1285 = None
    mul_1383: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_395, unsqueeze_1291);  sub_395 = unsqueeze_1291 = None
    mul_1384: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_147, squeeze_94);  sum_147 = squeeze_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_backward_72 = torch.ops.aten.convolution_backward.default(mul_1383, relu_27, primals_94, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1383 = primals_94 = None
    getitem_426: "f32[8, 1024, 14, 14]" = convolution_backward_72[0]
    getitem_427: "f32[2048, 1024, 1, 1]" = convolution_backward_72[1];  convolution_backward_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    add_576: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(where_69, getitem_426);  where_69 = getitem_426 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    alias_317: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(relu_27);  relu_27 = None
    alias_318: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(alias_317);  alias_317 = None
    le_72: "b8[8, 1024, 14, 14]" = torch.ops.aten.le.Scalar(alias_318, 0);  alias_318 = None
    where_72: "f32[8, 1024, 14, 14]" = torch.ops.aten.where.self(le_72, full_default, add_576);  le_72 = add_576 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    sum_148: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_72, [0, 2, 3])
    sub_396: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_1294);  convolution_30 = unsqueeze_1294 = None
    mul_1385: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_72, sub_396)
    sum_149: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1385, [0, 2, 3]);  mul_1385 = None
    mul_1386: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_148, 0.0006377551020408163)
    unsqueeze_1295: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1386, 0);  mul_1386 = None
    unsqueeze_1296: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1295, 2);  unsqueeze_1295 = None
    unsqueeze_1297: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1296, 3);  unsqueeze_1296 = None
    mul_1387: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_149, 0.0006377551020408163)
    mul_1388: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_91, squeeze_91)
    mul_1389: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1387, mul_1388);  mul_1387 = mul_1388 = None
    unsqueeze_1298: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1389, 0);  mul_1389 = None
    unsqueeze_1299: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1298, 2);  unsqueeze_1298 = None
    unsqueeze_1300: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1299, 3);  unsqueeze_1299 = None
    mul_1390: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_91, primals_92);  primals_92 = None
    unsqueeze_1301: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1390, 0);  mul_1390 = None
    unsqueeze_1302: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1301, 2);  unsqueeze_1301 = None
    unsqueeze_1303: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1302, 3);  unsqueeze_1302 = None
    mul_1391: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_396, unsqueeze_1300);  sub_396 = unsqueeze_1300 = None
    sub_398: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_72, mul_1391);  mul_1391 = None
    sub_399: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_398, unsqueeze_1297);  sub_398 = unsqueeze_1297 = None
    mul_1392: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_399, unsqueeze_1303);  sub_399 = unsqueeze_1303 = None
    mul_1393: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_149, squeeze_91);  sum_149 = squeeze_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_backward_73 = torch.ops.aten.convolution_backward.default(mul_1392, relu_26, primals_91, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1392 = primals_91 = None
    getitem_429: "f32[8, 2048, 14, 14]" = convolution_backward_73[0]
    getitem_430: "f32[1024, 2048, 1, 1]" = convolution_backward_73[1];  convolution_backward_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    alias_320: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(relu_26);  relu_26 = None
    alias_321: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(alias_320);  alias_320 = None
    le_73: "b8[8, 2048, 14, 14]" = torch.ops.aten.le.Scalar(alias_321, 0);  alias_321 = None
    where_73: "f32[8, 2048, 14, 14]" = torch.ops.aten.where.self(le_73, full_default, getitem_429);  le_73 = getitem_429 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    sum_150: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_73, [0, 2, 3])
    sub_400: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_1306);  convolution_29 = unsqueeze_1306 = None
    mul_1394: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(where_73, sub_400)
    sum_151: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_1394, [0, 2, 3]);  mul_1394 = None
    mul_1395: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_150, 0.0006377551020408163)
    unsqueeze_1307: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1395, 0);  mul_1395 = None
    unsqueeze_1308: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1307, 2);  unsqueeze_1307 = None
    unsqueeze_1309: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1308, 3);  unsqueeze_1308 = None
    mul_1396: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_151, 0.0006377551020408163)
    mul_1397: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_88, squeeze_88)
    mul_1398: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_1396, mul_1397);  mul_1396 = mul_1397 = None
    unsqueeze_1310: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1398, 0);  mul_1398 = None
    unsqueeze_1311: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1310, 2);  unsqueeze_1310 = None
    unsqueeze_1312: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1311, 3);  unsqueeze_1311 = None
    mul_1399: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_88, primals_89);  primals_89 = None
    unsqueeze_1313: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1399, 0);  mul_1399 = None
    unsqueeze_1314: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1313, 2);  unsqueeze_1313 = None
    unsqueeze_1315: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1314, 3);  unsqueeze_1314 = None
    mul_1400: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_400, unsqueeze_1312);  sub_400 = unsqueeze_1312 = None
    sub_402: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(where_73, mul_1400);  where_73 = mul_1400 = None
    sub_403: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(sub_402, unsqueeze_1309);  sub_402 = unsqueeze_1309 = None
    mul_1401: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_403, unsqueeze_1315);  sub_403 = unsqueeze_1315 = None
    mul_1402: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_151, squeeze_88);  sum_151 = squeeze_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_backward_74 = torch.ops.aten.convolution_backward.default(mul_1401, relu_25, primals_88, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_1401 = primals_88 = None
    getitem_432: "f32[8, 2048, 14, 14]" = convolution_backward_74[0]
    getitem_433: "f32[2048, 64, 3, 3]" = convolution_backward_74[1];  convolution_backward_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    alias_323: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(relu_25);  relu_25 = None
    alias_324: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(alias_323);  alias_323 = None
    le_74: "b8[8, 2048, 14, 14]" = torch.ops.aten.le.Scalar(alias_324, 0);  alias_324 = None
    where_74: "f32[8, 2048, 14, 14]" = torch.ops.aten.where.self(le_74, full_default, getitem_432);  le_74 = getitem_432 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    sum_152: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_74, [0, 2, 3])
    sub_404: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_1318);  convolution_28 = unsqueeze_1318 = None
    mul_1403: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(where_74, sub_404)
    sum_153: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_1403, [0, 2, 3]);  mul_1403 = None
    mul_1404: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_152, 0.0006377551020408163)
    unsqueeze_1319: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1404, 0);  mul_1404 = None
    unsqueeze_1320: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1319, 2);  unsqueeze_1319 = None
    unsqueeze_1321: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1320, 3);  unsqueeze_1320 = None
    mul_1405: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_153, 0.0006377551020408163)
    mul_1406: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_85, squeeze_85)
    mul_1407: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_1405, mul_1406);  mul_1405 = mul_1406 = None
    unsqueeze_1322: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1407, 0);  mul_1407 = None
    unsqueeze_1323: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1322, 2);  unsqueeze_1322 = None
    unsqueeze_1324: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1323, 3);  unsqueeze_1323 = None
    mul_1408: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_85, primals_86);  primals_86 = None
    unsqueeze_1325: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1408, 0);  mul_1408 = None
    unsqueeze_1326: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1325, 2);  unsqueeze_1325 = None
    unsqueeze_1327: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1326, 3);  unsqueeze_1326 = None
    mul_1409: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_404, unsqueeze_1324);  sub_404 = unsqueeze_1324 = None
    sub_406: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(where_74, mul_1409);  where_74 = mul_1409 = None
    sub_407: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(sub_406, unsqueeze_1321);  sub_406 = unsqueeze_1321 = None
    mul_1410: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_407, unsqueeze_1327);  sub_407 = unsqueeze_1327 = None
    mul_1411: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_153, squeeze_85);  sum_153 = squeeze_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_backward_75 = torch.ops.aten.convolution_backward.default(mul_1410, relu_24, primals_85, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1410 = primals_85 = None
    getitem_435: "f32[8, 1024, 14, 14]" = convolution_backward_75[0]
    getitem_436: "f32[2048, 1024, 1, 1]" = convolution_backward_75[1];  convolution_backward_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    add_577: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(where_72, getitem_435);  where_72 = getitem_435 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    alias_326: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(relu_24);  relu_24 = None
    alias_327: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(alias_326);  alias_326 = None
    le_75: "b8[8, 1024, 14, 14]" = torch.ops.aten.le.Scalar(alias_327, 0);  alias_327 = None
    where_75: "f32[8, 1024, 14, 14]" = torch.ops.aten.where.self(le_75, full_default, add_577);  le_75 = add_577 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:200, code: shortcut = self.downsample(shortcut)
    sum_154: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_75, [0, 2, 3])
    sub_408: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_1330);  convolution_27 = unsqueeze_1330 = None
    mul_1412: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_75, sub_408)
    sum_155: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1412, [0, 2, 3]);  mul_1412 = None
    mul_1413: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_154, 0.0006377551020408163)
    unsqueeze_1331: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1413, 0);  mul_1413 = None
    unsqueeze_1332: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1331, 2);  unsqueeze_1331 = None
    unsqueeze_1333: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1332, 3);  unsqueeze_1332 = None
    mul_1414: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_155, 0.0006377551020408163)
    mul_1415: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_82, squeeze_82)
    mul_1416: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1414, mul_1415);  mul_1414 = mul_1415 = None
    unsqueeze_1334: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1416, 0);  mul_1416 = None
    unsqueeze_1335: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1334, 2);  unsqueeze_1334 = None
    unsqueeze_1336: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1335, 3);  unsqueeze_1335 = None
    mul_1417: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_82, primals_83);  primals_83 = None
    unsqueeze_1337: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1417, 0);  mul_1417 = None
    unsqueeze_1338: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1337, 2);  unsqueeze_1337 = None
    unsqueeze_1339: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1338, 3);  unsqueeze_1338 = None
    mul_1418: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_408, unsqueeze_1336);  sub_408 = unsqueeze_1336 = None
    sub_410: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_75, mul_1418);  mul_1418 = None
    sub_411: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_410, unsqueeze_1333);  sub_410 = None
    mul_1419: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_411, unsqueeze_1339);  sub_411 = unsqueeze_1339 = None
    mul_1420: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_155, squeeze_82);  sum_155 = squeeze_82 = None
    convolution_backward_76 = torch.ops.aten.convolution_backward.default(mul_1419, relu_21, primals_82, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1419 = primals_82 = None
    getitem_438: "f32[8, 512, 28, 28]" = convolution_backward_76[0]
    getitem_439: "f32[1024, 512, 1, 1]" = convolution_backward_76[1];  convolution_backward_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    sub_412: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_1342);  convolution_26 = unsqueeze_1342 = None
    mul_1421: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_75, sub_412)
    sum_157: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1421, [0, 2, 3]);  mul_1421 = None
    mul_1423: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_157, 0.0006377551020408163)
    mul_1424: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_79, squeeze_79)
    mul_1425: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1423, mul_1424);  mul_1423 = mul_1424 = None
    unsqueeze_1346: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1425, 0);  mul_1425 = None
    unsqueeze_1347: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1346, 2);  unsqueeze_1346 = None
    unsqueeze_1348: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1347, 3);  unsqueeze_1347 = None
    mul_1426: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_79, primals_80);  primals_80 = None
    unsqueeze_1349: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1426, 0);  mul_1426 = None
    unsqueeze_1350: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1349, 2);  unsqueeze_1349 = None
    unsqueeze_1351: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1350, 3);  unsqueeze_1350 = None
    mul_1427: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_412, unsqueeze_1348);  sub_412 = unsqueeze_1348 = None
    sub_414: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_75, mul_1427);  where_75 = mul_1427 = None
    sub_415: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_414, unsqueeze_1333);  sub_414 = unsqueeze_1333 = None
    mul_1428: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_415, unsqueeze_1351);  sub_415 = unsqueeze_1351 = None
    mul_1429: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_157, squeeze_79);  sum_157 = squeeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_backward_77 = torch.ops.aten.convolution_backward.default(mul_1428, relu_23, primals_79, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1428 = primals_79 = None
    getitem_441: "f32[8, 2048, 14, 14]" = convolution_backward_77[0]
    getitem_442: "f32[1024, 2048, 1, 1]" = convolution_backward_77[1];  convolution_backward_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    alias_329: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(relu_23);  relu_23 = None
    alias_330: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(alias_329);  alias_329 = None
    le_76: "b8[8, 2048, 14, 14]" = torch.ops.aten.le.Scalar(alias_330, 0);  alias_330 = None
    where_76: "f32[8, 2048, 14, 14]" = torch.ops.aten.where.self(le_76, full_default, getitem_441);  le_76 = getitem_441 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    sum_158: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_76, [0, 2, 3])
    sub_416: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_1354);  convolution_25 = unsqueeze_1354 = None
    mul_1430: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(where_76, sub_416)
    sum_159: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_1430, [0, 2, 3]);  mul_1430 = None
    mul_1431: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_158, 0.0006377551020408163)
    unsqueeze_1355: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1431, 0);  mul_1431 = None
    unsqueeze_1356: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1355, 2);  unsqueeze_1355 = None
    unsqueeze_1357: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1356, 3);  unsqueeze_1356 = None
    mul_1432: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_159, 0.0006377551020408163)
    mul_1433: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_76, squeeze_76)
    mul_1434: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_1432, mul_1433);  mul_1432 = mul_1433 = None
    unsqueeze_1358: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1434, 0);  mul_1434 = None
    unsqueeze_1359: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1358, 2);  unsqueeze_1358 = None
    unsqueeze_1360: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1359, 3);  unsqueeze_1359 = None
    mul_1435: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_76, primals_77);  primals_77 = None
    unsqueeze_1361: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1435, 0);  mul_1435 = None
    unsqueeze_1362: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1361, 2);  unsqueeze_1361 = None
    unsqueeze_1363: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1362, 3);  unsqueeze_1362 = None
    mul_1436: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_416, unsqueeze_1360);  sub_416 = unsqueeze_1360 = None
    sub_418: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(where_76, mul_1436);  where_76 = mul_1436 = None
    sub_419: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(sub_418, unsqueeze_1357);  sub_418 = unsqueeze_1357 = None
    mul_1437: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_419, unsqueeze_1363);  sub_419 = unsqueeze_1363 = None
    mul_1438: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_159, squeeze_76);  sum_159 = squeeze_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_backward_78 = torch.ops.aten.convolution_backward.default(mul_1437, relu_22, primals_76, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_1437 = primals_76 = None
    getitem_444: "f32[8, 2048, 28, 28]" = convolution_backward_78[0]
    getitem_445: "f32[2048, 64, 3, 3]" = convolution_backward_78[1];  convolution_backward_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    alias_332: "f32[8, 2048, 28, 28]" = torch.ops.aten.alias.default(relu_22);  relu_22 = None
    alias_333: "f32[8, 2048, 28, 28]" = torch.ops.aten.alias.default(alias_332);  alias_332 = None
    le_77: "b8[8, 2048, 28, 28]" = torch.ops.aten.le.Scalar(alias_333, 0);  alias_333 = None
    where_77: "f32[8, 2048, 28, 28]" = torch.ops.aten.where.self(le_77, full_default, getitem_444);  le_77 = getitem_444 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    sum_160: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_77, [0, 2, 3])
    sub_420: "f32[8, 2048, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_1366);  convolution_24 = unsqueeze_1366 = None
    mul_1439: "f32[8, 2048, 28, 28]" = torch.ops.aten.mul.Tensor(where_77, sub_420)
    sum_161: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_1439, [0, 2, 3]);  mul_1439 = None
    mul_1440: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_160, 0.00015943877551020407)
    unsqueeze_1367: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1440, 0);  mul_1440 = None
    unsqueeze_1368: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1367, 2);  unsqueeze_1367 = None
    unsqueeze_1369: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1368, 3);  unsqueeze_1368 = None
    mul_1441: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_161, 0.00015943877551020407)
    mul_1442: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_73, squeeze_73)
    mul_1443: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_1441, mul_1442);  mul_1441 = mul_1442 = None
    unsqueeze_1370: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1443, 0);  mul_1443 = None
    unsqueeze_1371: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1370, 2);  unsqueeze_1370 = None
    unsqueeze_1372: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1371, 3);  unsqueeze_1371 = None
    mul_1444: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_73, primals_74);  primals_74 = None
    unsqueeze_1373: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1444, 0);  mul_1444 = None
    unsqueeze_1374: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1373, 2);  unsqueeze_1373 = None
    unsqueeze_1375: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1374, 3);  unsqueeze_1374 = None
    mul_1445: "f32[8, 2048, 28, 28]" = torch.ops.aten.mul.Tensor(sub_420, unsqueeze_1372);  sub_420 = unsqueeze_1372 = None
    sub_422: "f32[8, 2048, 28, 28]" = torch.ops.aten.sub.Tensor(where_77, mul_1445);  where_77 = mul_1445 = None
    sub_423: "f32[8, 2048, 28, 28]" = torch.ops.aten.sub.Tensor(sub_422, unsqueeze_1369);  sub_422 = unsqueeze_1369 = None
    mul_1446: "f32[8, 2048, 28, 28]" = torch.ops.aten.mul.Tensor(sub_423, unsqueeze_1375);  sub_423 = unsqueeze_1375 = None
    mul_1447: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_161, squeeze_73);  sum_161 = squeeze_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_backward_79 = torch.ops.aten.convolution_backward.default(mul_1446, relu_21, primals_73, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1446 = primals_73 = None
    getitem_447: "f32[8, 512, 28, 28]" = convolution_backward_79[0]
    getitem_448: "f32[2048, 512, 1, 1]" = convolution_backward_79[1];  convolution_backward_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    add_578: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(getitem_438, getitem_447);  getitem_438 = getitem_447 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    alias_335: "f32[8, 512, 28, 28]" = torch.ops.aten.alias.default(relu_21);  relu_21 = None
    alias_336: "f32[8, 512, 28, 28]" = torch.ops.aten.alias.default(alias_335);  alias_335 = None
    le_78: "b8[8, 512, 28, 28]" = torch.ops.aten.le.Scalar(alias_336, 0);  alias_336 = None
    where_78: "f32[8, 512, 28, 28]" = torch.ops.aten.where.self(le_78, full_default, add_578);  le_78 = add_578 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    sum_162: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_78, [0, 2, 3])
    sub_424: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_1378);  convolution_23 = unsqueeze_1378 = None
    mul_1448: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(where_78, sub_424)
    sum_163: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1448, [0, 2, 3]);  mul_1448 = None
    mul_1449: "f32[512]" = torch.ops.aten.mul.Tensor(sum_162, 0.00015943877551020407)
    unsqueeze_1379: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1449, 0);  mul_1449 = None
    unsqueeze_1380: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1379, 2);  unsqueeze_1379 = None
    unsqueeze_1381: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1380, 3);  unsqueeze_1380 = None
    mul_1450: "f32[512]" = torch.ops.aten.mul.Tensor(sum_163, 0.00015943877551020407)
    mul_1451: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_70, squeeze_70)
    mul_1452: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1450, mul_1451);  mul_1450 = mul_1451 = None
    unsqueeze_1382: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1452, 0);  mul_1452 = None
    unsqueeze_1383: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1382, 2);  unsqueeze_1382 = None
    unsqueeze_1384: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1383, 3);  unsqueeze_1383 = None
    mul_1453: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_70, primals_71);  primals_71 = None
    unsqueeze_1385: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1453, 0);  mul_1453 = None
    unsqueeze_1386: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1385, 2);  unsqueeze_1385 = None
    unsqueeze_1387: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1386, 3);  unsqueeze_1386 = None
    mul_1454: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_424, unsqueeze_1384);  sub_424 = unsqueeze_1384 = None
    sub_426: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(where_78, mul_1454);  mul_1454 = None
    sub_427: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(sub_426, unsqueeze_1381);  sub_426 = unsqueeze_1381 = None
    mul_1455: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_427, unsqueeze_1387);  sub_427 = unsqueeze_1387 = None
    mul_1456: "f32[512]" = torch.ops.aten.mul.Tensor(sum_163, squeeze_70);  sum_163 = squeeze_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_backward_80 = torch.ops.aten.convolution_backward.default(mul_1455, relu_20, primals_70, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1455 = primals_70 = None
    getitem_450: "f32[8, 1024, 28, 28]" = convolution_backward_80[0]
    getitem_451: "f32[512, 1024, 1, 1]" = convolution_backward_80[1];  convolution_backward_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    alias_338: "f32[8, 1024, 28, 28]" = torch.ops.aten.alias.default(relu_20);  relu_20 = None
    alias_339: "f32[8, 1024, 28, 28]" = torch.ops.aten.alias.default(alias_338);  alias_338 = None
    le_79: "b8[8, 1024, 28, 28]" = torch.ops.aten.le.Scalar(alias_339, 0);  alias_339 = None
    where_79: "f32[8, 1024, 28, 28]" = torch.ops.aten.where.self(le_79, full_default, getitem_450);  le_79 = getitem_450 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    sum_164: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_79, [0, 2, 3])
    sub_428: "f32[8, 1024, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_1390);  convolution_22 = unsqueeze_1390 = None
    mul_1457: "f32[8, 1024, 28, 28]" = torch.ops.aten.mul.Tensor(where_79, sub_428)
    sum_165: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1457, [0, 2, 3]);  mul_1457 = None
    mul_1458: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_164, 0.00015943877551020407)
    unsqueeze_1391: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1458, 0);  mul_1458 = None
    unsqueeze_1392: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1391, 2);  unsqueeze_1391 = None
    unsqueeze_1393: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1392, 3);  unsqueeze_1392 = None
    mul_1459: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_165, 0.00015943877551020407)
    mul_1460: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_67, squeeze_67)
    mul_1461: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1459, mul_1460);  mul_1459 = mul_1460 = None
    unsqueeze_1394: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1461, 0);  mul_1461 = None
    unsqueeze_1395: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1394, 2);  unsqueeze_1394 = None
    unsqueeze_1396: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1395, 3);  unsqueeze_1395 = None
    mul_1462: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_67, primals_68);  primals_68 = None
    unsqueeze_1397: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1462, 0);  mul_1462 = None
    unsqueeze_1398: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1397, 2);  unsqueeze_1397 = None
    unsqueeze_1399: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1398, 3);  unsqueeze_1398 = None
    mul_1463: "f32[8, 1024, 28, 28]" = torch.ops.aten.mul.Tensor(sub_428, unsqueeze_1396);  sub_428 = unsqueeze_1396 = None
    sub_430: "f32[8, 1024, 28, 28]" = torch.ops.aten.sub.Tensor(where_79, mul_1463);  where_79 = mul_1463 = None
    sub_431: "f32[8, 1024, 28, 28]" = torch.ops.aten.sub.Tensor(sub_430, unsqueeze_1393);  sub_430 = unsqueeze_1393 = None
    mul_1464: "f32[8, 1024, 28, 28]" = torch.ops.aten.mul.Tensor(sub_431, unsqueeze_1399);  sub_431 = unsqueeze_1399 = None
    mul_1465: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_165, squeeze_67);  sum_165 = squeeze_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_backward_81 = torch.ops.aten.convolution_backward.default(mul_1464, relu_19, primals_67, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_1464 = primals_67 = None
    getitem_453: "f32[8, 1024, 28, 28]" = convolution_backward_81[0]
    getitem_454: "f32[1024, 32, 3, 3]" = convolution_backward_81[1];  convolution_backward_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    alias_341: "f32[8, 1024, 28, 28]" = torch.ops.aten.alias.default(relu_19);  relu_19 = None
    alias_342: "f32[8, 1024, 28, 28]" = torch.ops.aten.alias.default(alias_341);  alias_341 = None
    le_80: "b8[8, 1024, 28, 28]" = torch.ops.aten.le.Scalar(alias_342, 0);  alias_342 = None
    where_80: "f32[8, 1024, 28, 28]" = torch.ops.aten.where.self(le_80, full_default, getitem_453);  le_80 = getitem_453 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    sum_166: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_80, [0, 2, 3])
    sub_432: "f32[8, 1024, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_1402);  convolution_21 = unsqueeze_1402 = None
    mul_1466: "f32[8, 1024, 28, 28]" = torch.ops.aten.mul.Tensor(where_80, sub_432)
    sum_167: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1466, [0, 2, 3]);  mul_1466 = None
    mul_1467: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_166, 0.00015943877551020407)
    unsqueeze_1403: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1467, 0);  mul_1467 = None
    unsqueeze_1404: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1403, 2);  unsqueeze_1403 = None
    unsqueeze_1405: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1404, 3);  unsqueeze_1404 = None
    mul_1468: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_167, 0.00015943877551020407)
    mul_1469: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_64, squeeze_64)
    mul_1470: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1468, mul_1469);  mul_1468 = mul_1469 = None
    unsqueeze_1406: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1470, 0);  mul_1470 = None
    unsqueeze_1407: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1406, 2);  unsqueeze_1406 = None
    unsqueeze_1408: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1407, 3);  unsqueeze_1407 = None
    mul_1471: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_64, primals_65);  primals_65 = None
    unsqueeze_1409: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1471, 0);  mul_1471 = None
    unsqueeze_1410: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1409, 2);  unsqueeze_1409 = None
    unsqueeze_1411: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1410, 3);  unsqueeze_1410 = None
    mul_1472: "f32[8, 1024, 28, 28]" = torch.ops.aten.mul.Tensor(sub_432, unsqueeze_1408);  sub_432 = unsqueeze_1408 = None
    sub_434: "f32[8, 1024, 28, 28]" = torch.ops.aten.sub.Tensor(where_80, mul_1472);  where_80 = mul_1472 = None
    sub_435: "f32[8, 1024, 28, 28]" = torch.ops.aten.sub.Tensor(sub_434, unsqueeze_1405);  sub_434 = unsqueeze_1405 = None
    mul_1473: "f32[8, 1024, 28, 28]" = torch.ops.aten.mul.Tensor(sub_435, unsqueeze_1411);  sub_435 = unsqueeze_1411 = None
    mul_1474: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_167, squeeze_64);  sum_167 = squeeze_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_backward_82 = torch.ops.aten.convolution_backward.default(mul_1473, relu_18, primals_64, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1473 = primals_64 = None
    getitem_456: "f32[8, 512, 28, 28]" = convolution_backward_82[0]
    getitem_457: "f32[1024, 512, 1, 1]" = convolution_backward_82[1];  convolution_backward_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    add_579: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(where_78, getitem_456);  where_78 = getitem_456 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    alias_344: "f32[8, 512, 28, 28]" = torch.ops.aten.alias.default(relu_18);  relu_18 = None
    alias_345: "f32[8, 512, 28, 28]" = torch.ops.aten.alias.default(alias_344);  alias_344 = None
    le_81: "b8[8, 512, 28, 28]" = torch.ops.aten.le.Scalar(alias_345, 0);  alias_345 = None
    where_81: "f32[8, 512, 28, 28]" = torch.ops.aten.where.self(le_81, full_default, add_579);  le_81 = add_579 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    sum_168: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_81, [0, 2, 3])
    sub_436: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_1414);  convolution_20 = unsqueeze_1414 = None
    mul_1475: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(where_81, sub_436)
    sum_169: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1475, [0, 2, 3]);  mul_1475 = None
    mul_1476: "f32[512]" = torch.ops.aten.mul.Tensor(sum_168, 0.00015943877551020407)
    unsqueeze_1415: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1476, 0);  mul_1476 = None
    unsqueeze_1416: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1415, 2);  unsqueeze_1415 = None
    unsqueeze_1417: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1416, 3);  unsqueeze_1416 = None
    mul_1477: "f32[512]" = torch.ops.aten.mul.Tensor(sum_169, 0.00015943877551020407)
    mul_1478: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_61, squeeze_61)
    mul_1479: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1477, mul_1478);  mul_1477 = mul_1478 = None
    unsqueeze_1418: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1479, 0);  mul_1479 = None
    unsqueeze_1419: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1418, 2);  unsqueeze_1418 = None
    unsqueeze_1420: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1419, 3);  unsqueeze_1419 = None
    mul_1480: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_61, primals_62);  primals_62 = None
    unsqueeze_1421: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1480, 0);  mul_1480 = None
    unsqueeze_1422: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1421, 2);  unsqueeze_1421 = None
    unsqueeze_1423: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1422, 3);  unsqueeze_1422 = None
    mul_1481: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_436, unsqueeze_1420);  sub_436 = unsqueeze_1420 = None
    sub_438: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(where_81, mul_1481);  mul_1481 = None
    sub_439: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(sub_438, unsqueeze_1417);  sub_438 = unsqueeze_1417 = None
    mul_1482: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_439, unsqueeze_1423);  sub_439 = unsqueeze_1423 = None
    mul_1483: "f32[512]" = torch.ops.aten.mul.Tensor(sum_169, squeeze_61);  sum_169 = squeeze_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_backward_83 = torch.ops.aten.convolution_backward.default(mul_1482, relu_17, primals_61, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1482 = primals_61 = None
    getitem_459: "f32[8, 1024, 28, 28]" = convolution_backward_83[0]
    getitem_460: "f32[512, 1024, 1, 1]" = convolution_backward_83[1];  convolution_backward_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    alias_347: "f32[8, 1024, 28, 28]" = torch.ops.aten.alias.default(relu_17);  relu_17 = None
    alias_348: "f32[8, 1024, 28, 28]" = torch.ops.aten.alias.default(alias_347);  alias_347 = None
    le_82: "b8[8, 1024, 28, 28]" = torch.ops.aten.le.Scalar(alias_348, 0);  alias_348 = None
    where_82: "f32[8, 1024, 28, 28]" = torch.ops.aten.where.self(le_82, full_default, getitem_459);  le_82 = getitem_459 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    sum_170: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_82, [0, 2, 3])
    sub_440: "f32[8, 1024, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_1426);  convolution_19 = unsqueeze_1426 = None
    mul_1484: "f32[8, 1024, 28, 28]" = torch.ops.aten.mul.Tensor(where_82, sub_440)
    sum_171: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1484, [0, 2, 3]);  mul_1484 = None
    mul_1485: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_170, 0.00015943877551020407)
    unsqueeze_1427: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1485, 0);  mul_1485 = None
    unsqueeze_1428: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1427, 2);  unsqueeze_1427 = None
    unsqueeze_1429: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1428, 3);  unsqueeze_1428 = None
    mul_1486: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_171, 0.00015943877551020407)
    mul_1487: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_58, squeeze_58)
    mul_1488: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1486, mul_1487);  mul_1486 = mul_1487 = None
    unsqueeze_1430: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1488, 0);  mul_1488 = None
    unsqueeze_1431: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1430, 2);  unsqueeze_1430 = None
    unsqueeze_1432: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1431, 3);  unsqueeze_1431 = None
    mul_1489: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_58, primals_59);  primals_59 = None
    unsqueeze_1433: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1489, 0);  mul_1489 = None
    unsqueeze_1434: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1433, 2);  unsqueeze_1433 = None
    unsqueeze_1435: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1434, 3);  unsqueeze_1434 = None
    mul_1490: "f32[8, 1024, 28, 28]" = torch.ops.aten.mul.Tensor(sub_440, unsqueeze_1432);  sub_440 = unsqueeze_1432 = None
    sub_442: "f32[8, 1024, 28, 28]" = torch.ops.aten.sub.Tensor(where_82, mul_1490);  where_82 = mul_1490 = None
    sub_443: "f32[8, 1024, 28, 28]" = torch.ops.aten.sub.Tensor(sub_442, unsqueeze_1429);  sub_442 = unsqueeze_1429 = None
    mul_1491: "f32[8, 1024, 28, 28]" = torch.ops.aten.mul.Tensor(sub_443, unsqueeze_1435);  sub_443 = unsqueeze_1435 = None
    mul_1492: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_171, squeeze_58);  sum_171 = squeeze_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_backward_84 = torch.ops.aten.convolution_backward.default(mul_1491, relu_16, primals_58, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_1491 = primals_58 = None
    getitem_462: "f32[8, 1024, 28, 28]" = convolution_backward_84[0]
    getitem_463: "f32[1024, 32, 3, 3]" = convolution_backward_84[1];  convolution_backward_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    alias_350: "f32[8, 1024, 28, 28]" = torch.ops.aten.alias.default(relu_16);  relu_16 = None
    alias_351: "f32[8, 1024, 28, 28]" = torch.ops.aten.alias.default(alias_350);  alias_350 = None
    le_83: "b8[8, 1024, 28, 28]" = torch.ops.aten.le.Scalar(alias_351, 0);  alias_351 = None
    where_83: "f32[8, 1024, 28, 28]" = torch.ops.aten.where.self(le_83, full_default, getitem_462);  le_83 = getitem_462 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    sum_172: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_83, [0, 2, 3])
    sub_444: "f32[8, 1024, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_1438);  convolution_18 = unsqueeze_1438 = None
    mul_1493: "f32[8, 1024, 28, 28]" = torch.ops.aten.mul.Tensor(where_83, sub_444)
    sum_173: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1493, [0, 2, 3]);  mul_1493 = None
    mul_1494: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_172, 0.00015943877551020407)
    unsqueeze_1439: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1494, 0);  mul_1494 = None
    unsqueeze_1440: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1439, 2);  unsqueeze_1439 = None
    unsqueeze_1441: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1440, 3);  unsqueeze_1440 = None
    mul_1495: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_173, 0.00015943877551020407)
    mul_1496: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_55, squeeze_55)
    mul_1497: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1495, mul_1496);  mul_1495 = mul_1496 = None
    unsqueeze_1442: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1497, 0);  mul_1497 = None
    unsqueeze_1443: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1442, 2);  unsqueeze_1442 = None
    unsqueeze_1444: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1443, 3);  unsqueeze_1443 = None
    mul_1498: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_55, primals_56);  primals_56 = None
    unsqueeze_1445: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1498, 0);  mul_1498 = None
    unsqueeze_1446: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1445, 2);  unsqueeze_1445 = None
    unsqueeze_1447: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1446, 3);  unsqueeze_1446 = None
    mul_1499: "f32[8, 1024, 28, 28]" = torch.ops.aten.mul.Tensor(sub_444, unsqueeze_1444);  sub_444 = unsqueeze_1444 = None
    sub_446: "f32[8, 1024, 28, 28]" = torch.ops.aten.sub.Tensor(where_83, mul_1499);  where_83 = mul_1499 = None
    sub_447: "f32[8, 1024, 28, 28]" = torch.ops.aten.sub.Tensor(sub_446, unsqueeze_1441);  sub_446 = unsqueeze_1441 = None
    mul_1500: "f32[8, 1024, 28, 28]" = torch.ops.aten.mul.Tensor(sub_447, unsqueeze_1447);  sub_447 = unsqueeze_1447 = None
    mul_1501: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_173, squeeze_55);  sum_173 = squeeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_backward_85 = torch.ops.aten.convolution_backward.default(mul_1500, relu_15, primals_55, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1500 = primals_55 = None
    getitem_465: "f32[8, 512, 28, 28]" = convolution_backward_85[0]
    getitem_466: "f32[1024, 512, 1, 1]" = convolution_backward_85[1];  convolution_backward_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    add_580: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(where_81, getitem_465);  where_81 = getitem_465 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    alias_353: "f32[8, 512, 28, 28]" = torch.ops.aten.alias.default(relu_15);  relu_15 = None
    alias_354: "f32[8, 512, 28, 28]" = torch.ops.aten.alias.default(alias_353);  alias_353 = None
    le_84: "b8[8, 512, 28, 28]" = torch.ops.aten.le.Scalar(alias_354, 0);  alias_354 = None
    where_84: "f32[8, 512, 28, 28]" = torch.ops.aten.where.self(le_84, full_default, add_580);  le_84 = add_580 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    sum_174: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_84, [0, 2, 3])
    sub_448: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_1450);  convolution_17 = unsqueeze_1450 = None
    mul_1502: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(where_84, sub_448)
    sum_175: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1502, [0, 2, 3]);  mul_1502 = None
    mul_1503: "f32[512]" = torch.ops.aten.mul.Tensor(sum_174, 0.00015943877551020407)
    unsqueeze_1451: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1503, 0);  mul_1503 = None
    unsqueeze_1452: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1451, 2);  unsqueeze_1451 = None
    unsqueeze_1453: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1452, 3);  unsqueeze_1452 = None
    mul_1504: "f32[512]" = torch.ops.aten.mul.Tensor(sum_175, 0.00015943877551020407)
    mul_1505: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_52, squeeze_52)
    mul_1506: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1504, mul_1505);  mul_1504 = mul_1505 = None
    unsqueeze_1454: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1506, 0);  mul_1506 = None
    unsqueeze_1455: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1454, 2);  unsqueeze_1454 = None
    unsqueeze_1456: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1455, 3);  unsqueeze_1455 = None
    mul_1507: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_52, primals_53);  primals_53 = None
    unsqueeze_1457: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1507, 0);  mul_1507 = None
    unsqueeze_1458: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1457, 2);  unsqueeze_1457 = None
    unsqueeze_1459: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1458, 3);  unsqueeze_1458 = None
    mul_1508: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_448, unsqueeze_1456);  sub_448 = unsqueeze_1456 = None
    sub_450: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(where_84, mul_1508);  mul_1508 = None
    sub_451: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(sub_450, unsqueeze_1453);  sub_450 = unsqueeze_1453 = None
    mul_1509: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_451, unsqueeze_1459);  sub_451 = unsqueeze_1459 = None
    mul_1510: "f32[512]" = torch.ops.aten.mul.Tensor(sum_175, squeeze_52);  sum_175 = squeeze_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_backward_86 = torch.ops.aten.convolution_backward.default(mul_1509, relu_14, primals_52, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1509 = primals_52 = None
    getitem_468: "f32[8, 1024, 28, 28]" = convolution_backward_86[0]
    getitem_469: "f32[512, 1024, 1, 1]" = convolution_backward_86[1];  convolution_backward_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    alias_356: "f32[8, 1024, 28, 28]" = torch.ops.aten.alias.default(relu_14);  relu_14 = None
    alias_357: "f32[8, 1024, 28, 28]" = torch.ops.aten.alias.default(alias_356);  alias_356 = None
    le_85: "b8[8, 1024, 28, 28]" = torch.ops.aten.le.Scalar(alias_357, 0);  alias_357 = None
    where_85: "f32[8, 1024, 28, 28]" = torch.ops.aten.where.self(le_85, full_default, getitem_468);  le_85 = getitem_468 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    sum_176: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_85, [0, 2, 3])
    sub_452: "f32[8, 1024, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_1462);  convolution_16 = unsqueeze_1462 = None
    mul_1511: "f32[8, 1024, 28, 28]" = torch.ops.aten.mul.Tensor(where_85, sub_452)
    sum_177: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1511, [0, 2, 3]);  mul_1511 = None
    mul_1512: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_176, 0.00015943877551020407)
    unsqueeze_1463: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1512, 0);  mul_1512 = None
    unsqueeze_1464: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1463, 2);  unsqueeze_1463 = None
    unsqueeze_1465: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1464, 3);  unsqueeze_1464 = None
    mul_1513: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_177, 0.00015943877551020407)
    mul_1514: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_49, squeeze_49)
    mul_1515: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1513, mul_1514);  mul_1513 = mul_1514 = None
    unsqueeze_1466: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1515, 0);  mul_1515 = None
    unsqueeze_1467: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1466, 2);  unsqueeze_1466 = None
    unsqueeze_1468: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1467, 3);  unsqueeze_1467 = None
    mul_1516: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_49, primals_50);  primals_50 = None
    unsqueeze_1469: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1516, 0);  mul_1516 = None
    unsqueeze_1470: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1469, 2);  unsqueeze_1469 = None
    unsqueeze_1471: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1470, 3);  unsqueeze_1470 = None
    mul_1517: "f32[8, 1024, 28, 28]" = torch.ops.aten.mul.Tensor(sub_452, unsqueeze_1468);  sub_452 = unsqueeze_1468 = None
    sub_454: "f32[8, 1024, 28, 28]" = torch.ops.aten.sub.Tensor(where_85, mul_1517);  where_85 = mul_1517 = None
    sub_455: "f32[8, 1024, 28, 28]" = torch.ops.aten.sub.Tensor(sub_454, unsqueeze_1465);  sub_454 = unsqueeze_1465 = None
    mul_1518: "f32[8, 1024, 28, 28]" = torch.ops.aten.mul.Tensor(sub_455, unsqueeze_1471);  sub_455 = unsqueeze_1471 = None
    mul_1519: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_177, squeeze_49);  sum_177 = squeeze_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_backward_87 = torch.ops.aten.convolution_backward.default(mul_1518, relu_13, primals_49, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_1518 = primals_49 = None
    getitem_471: "f32[8, 1024, 28, 28]" = convolution_backward_87[0]
    getitem_472: "f32[1024, 32, 3, 3]" = convolution_backward_87[1];  convolution_backward_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    alias_359: "f32[8, 1024, 28, 28]" = torch.ops.aten.alias.default(relu_13);  relu_13 = None
    alias_360: "f32[8, 1024, 28, 28]" = torch.ops.aten.alias.default(alias_359);  alias_359 = None
    le_86: "b8[8, 1024, 28, 28]" = torch.ops.aten.le.Scalar(alias_360, 0);  alias_360 = None
    where_86: "f32[8, 1024, 28, 28]" = torch.ops.aten.where.self(le_86, full_default, getitem_471);  le_86 = getitem_471 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    sum_178: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_86, [0, 2, 3])
    sub_456: "f32[8, 1024, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_1474);  convolution_15 = unsqueeze_1474 = None
    mul_1520: "f32[8, 1024, 28, 28]" = torch.ops.aten.mul.Tensor(where_86, sub_456)
    sum_179: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1520, [0, 2, 3]);  mul_1520 = None
    mul_1521: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_178, 0.00015943877551020407)
    unsqueeze_1475: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1521, 0);  mul_1521 = None
    unsqueeze_1476: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1475, 2);  unsqueeze_1475 = None
    unsqueeze_1477: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1476, 3);  unsqueeze_1476 = None
    mul_1522: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_179, 0.00015943877551020407)
    mul_1523: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_46, squeeze_46)
    mul_1524: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1522, mul_1523);  mul_1522 = mul_1523 = None
    unsqueeze_1478: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1524, 0);  mul_1524 = None
    unsqueeze_1479: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1478, 2);  unsqueeze_1478 = None
    unsqueeze_1480: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1479, 3);  unsqueeze_1479 = None
    mul_1525: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_46, primals_47);  primals_47 = None
    unsqueeze_1481: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1525, 0);  mul_1525 = None
    unsqueeze_1482: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1481, 2);  unsqueeze_1481 = None
    unsqueeze_1483: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1482, 3);  unsqueeze_1482 = None
    mul_1526: "f32[8, 1024, 28, 28]" = torch.ops.aten.mul.Tensor(sub_456, unsqueeze_1480);  sub_456 = unsqueeze_1480 = None
    sub_458: "f32[8, 1024, 28, 28]" = torch.ops.aten.sub.Tensor(where_86, mul_1526);  where_86 = mul_1526 = None
    sub_459: "f32[8, 1024, 28, 28]" = torch.ops.aten.sub.Tensor(sub_458, unsqueeze_1477);  sub_458 = unsqueeze_1477 = None
    mul_1527: "f32[8, 1024, 28, 28]" = torch.ops.aten.mul.Tensor(sub_459, unsqueeze_1483);  sub_459 = unsqueeze_1483 = None
    mul_1528: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_179, squeeze_46);  sum_179 = squeeze_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_backward_88 = torch.ops.aten.convolution_backward.default(mul_1527, relu_12, primals_46, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1527 = primals_46 = None
    getitem_474: "f32[8, 512, 28, 28]" = convolution_backward_88[0]
    getitem_475: "f32[1024, 512, 1, 1]" = convolution_backward_88[1];  convolution_backward_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    add_581: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(where_84, getitem_474);  where_84 = getitem_474 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    alias_362: "f32[8, 512, 28, 28]" = torch.ops.aten.alias.default(relu_12);  relu_12 = None
    alias_363: "f32[8, 512, 28, 28]" = torch.ops.aten.alias.default(alias_362);  alias_362 = None
    le_87: "b8[8, 512, 28, 28]" = torch.ops.aten.le.Scalar(alias_363, 0);  alias_363 = None
    where_87: "f32[8, 512, 28, 28]" = torch.ops.aten.where.self(le_87, full_default, add_581);  le_87 = add_581 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:200, code: shortcut = self.downsample(shortcut)
    sum_180: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_87, [0, 2, 3])
    sub_460: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_1486);  convolution_14 = unsqueeze_1486 = None
    mul_1529: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(where_87, sub_460)
    sum_181: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1529, [0, 2, 3]);  mul_1529 = None
    mul_1530: "f32[512]" = torch.ops.aten.mul.Tensor(sum_180, 0.00015943877551020407)
    unsqueeze_1487: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1530, 0);  mul_1530 = None
    unsqueeze_1488: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1487, 2);  unsqueeze_1487 = None
    unsqueeze_1489: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1488, 3);  unsqueeze_1488 = None
    mul_1531: "f32[512]" = torch.ops.aten.mul.Tensor(sum_181, 0.00015943877551020407)
    mul_1532: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
    mul_1533: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1531, mul_1532);  mul_1531 = mul_1532 = None
    unsqueeze_1490: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1533, 0);  mul_1533 = None
    unsqueeze_1491: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1490, 2);  unsqueeze_1490 = None
    unsqueeze_1492: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1491, 3);  unsqueeze_1491 = None
    mul_1534: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_43, primals_44);  primals_44 = None
    unsqueeze_1493: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1534, 0);  mul_1534 = None
    unsqueeze_1494: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1493, 2);  unsqueeze_1493 = None
    unsqueeze_1495: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1494, 3);  unsqueeze_1494 = None
    mul_1535: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_460, unsqueeze_1492);  sub_460 = unsqueeze_1492 = None
    sub_462: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(where_87, mul_1535);  mul_1535 = None
    sub_463: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(sub_462, unsqueeze_1489);  sub_462 = None
    mul_1536: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_463, unsqueeze_1495);  sub_463 = unsqueeze_1495 = None
    mul_1537: "f32[512]" = torch.ops.aten.mul.Tensor(sum_181, squeeze_43);  sum_181 = squeeze_43 = None
    convolution_backward_89 = torch.ops.aten.convolution_backward.default(mul_1536, relu_9, primals_43, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1536 = primals_43 = None
    getitem_477: "f32[8, 256, 56, 56]" = convolution_backward_89[0]
    getitem_478: "f32[512, 256, 1, 1]" = convolution_backward_89[1];  convolution_backward_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    sub_464: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_1498);  convolution_13 = unsqueeze_1498 = None
    mul_1538: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(where_87, sub_464)
    sum_183: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1538, [0, 2, 3]);  mul_1538 = None
    mul_1540: "f32[512]" = torch.ops.aten.mul.Tensor(sum_183, 0.00015943877551020407)
    mul_1541: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_40, squeeze_40)
    mul_1542: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1540, mul_1541);  mul_1540 = mul_1541 = None
    unsqueeze_1502: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1542, 0);  mul_1542 = None
    unsqueeze_1503: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1502, 2);  unsqueeze_1502 = None
    unsqueeze_1504: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1503, 3);  unsqueeze_1503 = None
    mul_1543: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_40, primals_41);  primals_41 = None
    unsqueeze_1505: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1543, 0);  mul_1543 = None
    unsqueeze_1506: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1505, 2);  unsqueeze_1505 = None
    unsqueeze_1507: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1506, 3);  unsqueeze_1506 = None
    mul_1544: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_464, unsqueeze_1504);  sub_464 = unsqueeze_1504 = None
    sub_466: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(where_87, mul_1544);  where_87 = mul_1544 = None
    sub_467: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(sub_466, unsqueeze_1489);  sub_466 = unsqueeze_1489 = None
    mul_1545: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_467, unsqueeze_1507);  sub_467 = unsqueeze_1507 = None
    mul_1546: "f32[512]" = torch.ops.aten.mul.Tensor(sum_183, squeeze_40);  sum_183 = squeeze_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_backward_90 = torch.ops.aten.convolution_backward.default(mul_1545, relu_11, primals_40, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1545 = primals_40 = None
    getitem_480: "f32[8, 1024, 28, 28]" = convolution_backward_90[0]
    getitem_481: "f32[512, 1024, 1, 1]" = convolution_backward_90[1];  convolution_backward_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    alias_365: "f32[8, 1024, 28, 28]" = torch.ops.aten.alias.default(relu_11);  relu_11 = None
    alias_366: "f32[8, 1024, 28, 28]" = torch.ops.aten.alias.default(alias_365);  alias_365 = None
    le_88: "b8[8, 1024, 28, 28]" = torch.ops.aten.le.Scalar(alias_366, 0);  alias_366 = None
    where_88: "f32[8, 1024, 28, 28]" = torch.ops.aten.where.self(le_88, full_default, getitem_480);  le_88 = getitem_480 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    sum_184: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_88, [0, 2, 3])
    sub_468: "f32[8, 1024, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_1510);  convolution_12 = unsqueeze_1510 = None
    mul_1547: "f32[8, 1024, 28, 28]" = torch.ops.aten.mul.Tensor(where_88, sub_468)
    sum_185: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1547, [0, 2, 3]);  mul_1547 = None
    mul_1548: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_184, 0.00015943877551020407)
    unsqueeze_1511: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1548, 0);  mul_1548 = None
    unsqueeze_1512: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1511, 2);  unsqueeze_1511 = None
    unsqueeze_1513: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1512, 3);  unsqueeze_1512 = None
    mul_1549: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_185, 0.00015943877551020407)
    mul_1550: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
    mul_1551: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1549, mul_1550);  mul_1549 = mul_1550 = None
    unsqueeze_1514: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1551, 0);  mul_1551 = None
    unsqueeze_1515: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1514, 2);  unsqueeze_1514 = None
    unsqueeze_1516: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1515, 3);  unsqueeze_1515 = None
    mul_1552: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_37, primals_38);  primals_38 = None
    unsqueeze_1517: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1552, 0);  mul_1552 = None
    unsqueeze_1518: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1517, 2);  unsqueeze_1517 = None
    unsqueeze_1519: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1518, 3);  unsqueeze_1518 = None
    mul_1553: "f32[8, 1024, 28, 28]" = torch.ops.aten.mul.Tensor(sub_468, unsqueeze_1516);  sub_468 = unsqueeze_1516 = None
    sub_470: "f32[8, 1024, 28, 28]" = torch.ops.aten.sub.Tensor(where_88, mul_1553);  where_88 = mul_1553 = None
    sub_471: "f32[8, 1024, 28, 28]" = torch.ops.aten.sub.Tensor(sub_470, unsqueeze_1513);  sub_470 = unsqueeze_1513 = None
    mul_1554: "f32[8, 1024, 28, 28]" = torch.ops.aten.mul.Tensor(sub_471, unsqueeze_1519);  sub_471 = unsqueeze_1519 = None
    mul_1555: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_185, squeeze_37);  sum_185 = squeeze_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_backward_91 = torch.ops.aten.convolution_backward.default(mul_1554, relu_10, primals_37, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_1554 = primals_37 = None
    getitem_483: "f32[8, 1024, 56, 56]" = convolution_backward_91[0]
    getitem_484: "f32[1024, 32, 3, 3]" = convolution_backward_91[1];  convolution_backward_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    alias_368: "f32[8, 1024, 56, 56]" = torch.ops.aten.alias.default(relu_10);  relu_10 = None
    alias_369: "f32[8, 1024, 56, 56]" = torch.ops.aten.alias.default(alias_368);  alias_368 = None
    le_89: "b8[8, 1024, 56, 56]" = torch.ops.aten.le.Scalar(alias_369, 0);  alias_369 = None
    where_89: "f32[8, 1024, 56, 56]" = torch.ops.aten.where.self(le_89, full_default, getitem_483);  le_89 = getitem_483 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    sum_186: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_89, [0, 2, 3])
    sub_472: "f32[8, 1024, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_1522);  convolution_11 = unsqueeze_1522 = None
    mul_1556: "f32[8, 1024, 56, 56]" = torch.ops.aten.mul.Tensor(where_89, sub_472)
    sum_187: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1556, [0, 2, 3]);  mul_1556 = None
    mul_1557: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_186, 3.985969387755102e-05)
    unsqueeze_1523: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1557, 0);  mul_1557 = None
    unsqueeze_1524: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1523, 2);  unsqueeze_1523 = None
    unsqueeze_1525: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1524, 3);  unsqueeze_1524 = None
    mul_1558: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_187, 3.985969387755102e-05)
    mul_1559: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_34, squeeze_34)
    mul_1560: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1558, mul_1559);  mul_1558 = mul_1559 = None
    unsqueeze_1526: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1560, 0);  mul_1560 = None
    unsqueeze_1527: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1526, 2);  unsqueeze_1526 = None
    unsqueeze_1528: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1527, 3);  unsqueeze_1527 = None
    mul_1561: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_34, primals_35);  primals_35 = None
    unsqueeze_1529: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1561, 0);  mul_1561 = None
    unsqueeze_1530: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1529, 2);  unsqueeze_1529 = None
    unsqueeze_1531: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1530, 3);  unsqueeze_1530 = None
    mul_1562: "f32[8, 1024, 56, 56]" = torch.ops.aten.mul.Tensor(sub_472, unsqueeze_1528);  sub_472 = unsqueeze_1528 = None
    sub_474: "f32[8, 1024, 56, 56]" = torch.ops.aten.sub.Tensor(where_89, mul_1562);  where_89 = mul_1562 = None
    sub_475: "f32[8, 1024, 56, 56]" = torch.ops.aten.sub.Tensor(sub_474, unsqueeze_1525);  sub_474 = unsqueeze_1525 = None
    mul_1563: "f32[8, 1024, 56, 56]" = torch.ops.aten.mul.Tensor(sub_475, unsqueeze_1531);  sub_475 = unsqueeze_1531 = None
    mul_1564: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_187, squeeze_34);  sum_187 = squeeze_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_backward_92 = torch.ops.aten.convolution_backward.default(mul_1563, relu_9, primals_34, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1563 = primals_34 = None
    getitem_486: "f32[8, 256, 56, 56]" = convolution_backward_92[0]
    getitem_487: "f32[1024, 256, 1, 1]" = convolution_backward_92[1];  convolution_backward_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    add_582: "f32[8, 256, 56, 56]" = torch.ops.aten.add.Tensor(getitem_477, getitem_486);  getitem_477 = getitem_486 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    alias_371: "f32[8, 256, 56, 56]" = torch.ops.aten.alias.default(relu_9);  relu_9 = None
    alias_372: "f32[8, 256, 56, 56]" = torch.ops.aten.alias.default(alias_371);  alias_371 = None
    le_90: "b8[8, 256, 56, 56]" = torch.ops.aten.le.Scalar(alias_372, 0);  alias_372 = None
    where_90: "f32[8, 256, 56, 56]" = torch.ops.aten.where.self(le_90, full_default, add_582);  le_90 = add_582 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    sum_188: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_90, [0, 2, 3])
    sub_476: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_1534);  convolution_10 = unsqueeze_1534 = None
    mul_1565: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(where_90, sub_476)
    sum_189: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1565, [0, 2, 3]);  mul_1565 = None
    mul_1566: "f32[256]" = torch.ops.aten.mul.Tensor(sum_188, 3.985969387755102e-05)
    unsqueeze_1535: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1566, 0);  mul_1566 = None
    unsqueeze_1536: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1535, 2);  unsqueeze_1535 = None
    unsqueeze_1537: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1536, 3);  unsqueeze_1536 = None
    mul_1567: "f32[256]" = torch.ops.aten.mul.Tensor(sum_189, 3.985969387755102e-05)
    mul_1568: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
    mul_1569: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1567, mul_1568);  mul_1567 = mul_1568 = None
    unsqueeze_1538: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1569, 0);  mul_1569 = None
    unsqueeze_1539: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1538, 2);  unsqueeze_1538 = None
    unsqueeze_1540: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1539, 3);  unsqueeze_1539 = None
    mul_1570: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_31, primals_32);  primals_32 = None
    unsqueeze_1541: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1570, 0);  mul_1570 = None
    unsqueeze_1542: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1541, 2);  unsqueeze_1541 = None
    unsqueeze_1543: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1542, 3);  unsqueeze_1542 = None
    mul_1571: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_476, unsqueeze_1540);  sub_476 = unsqueeze_1540 = None
    sub_478: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(where_90, mul_1571);  mul_1571 = None
    sub_479: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(sub_478, unsqueeze_1537);  sub_478 = unsqueeze_1537 = None
    mul_1572: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_479, unsqueeze_1543);  sub_479 = unsqueeze_1543 = None
    mul_1573: "f32[256]" = torch.ops.aten.mul.Tensor(sum_189, squeeze_31);  sum_189 = squeeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_backward_93 = torch.ops.aten.convolution_backward.default(mul_1572, relu_8, primals_31, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1572 = primals_31 = None
    getitem_489: "f32[8, 512, 56, 56]" = convolution_backward_93[0]
    getitem_490: "f32[256, 512, 1, 1]" = convolution_backward_93[1];  convolution_backward_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    alias_374: "f32[8, 512, 56, 56]" = torch.ops.aten.alias.default(relu_8);  relu_8 = None
    alias_375: "f32[8, 512, 56, 56]" = torch.ops.aten.alias.default(alias_374);  alias_374 = None
    le_91: "b8[8, 512, 56, 56]" = torch.ops.aten.le.Scalar(alias_375, 0);  alias_375 = None
    where_91: "f32[8, 512, 56, 56]" = torch.ops.aten.where.self(le_91, full_default, getitem_489);  le_91 = getitem_489 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    sum_190: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_91, [0, 2, 3])
    sub_480: "f32[8, 512, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_1546);  convolution_9 = unsqueeze_1546 = None
    mul_1574: "f32[8, 512, 56, 56]" = torch.ops.aten.mul.Tensor(where_91, sub_480)
    sum_191: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1574, [0, 2, 3]);  mul_1574 = None
    mul_1575: "f32[512]" = torch.ops.aten.mul.Tensor(sum_190, 3.985969387755102e-05)
    unsqueeze_1547: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1575, 0);  mul_1575 = None
    unsqueeze_1548: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1547, 2);  unsqueeze_1547 = None
    unsqueeze_1549: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1548, 3);  unsqueeze_1548 = None
    mul_1576: "f32[512]" = torch.ops.aten.mul.Tensor(sum_191, 3.985969387755102e-05)
    mul_1577: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_28, squeeze_28)
    mul_1578: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1576, mul_1577);  mul_1576 = mul_1577 = None
    unsqueeze_1550: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1578, 0);  mul_1578 = None
    unsqueeze_1551: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1550, 2);  unsqueeze_1550 = None
    unsqueeze_1552: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1551, 3);  unsqueeze_1551 = None
    mul_1579: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_28, primals_29);  primals_29 = None
    unsqueeze_1553: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1579, 0);  mul_1579 = None
    unsqueeze_1554: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1553, 2);  unsqueeze_1553 = None
    unsqueeze_1555: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1554, 3);  unsqueeze_1554 = None
    mul_1580: "f32[8, 512, 56, 56]" = torch.ops.aten.mul.Tensor(sub_480, unsqueeze_1552);  sub_480 = unsqueeze_1552 = None
    sub_482: "f32[8, 512, 56, 56]" = torch.ops.aten.sub.Tensor(where_91, mul_1580);  where_91 = mul_1580 = None
    sub_483: "f32[8, 512, 56, 56]" = torch.ops.aten.sub.Tensor(sub_482, unsqueeze_1549);  sub_482 = unsqueeze_1549 = None
    mul_1581: "f32[8, 512, 56, 56]" = torch.ops.aten.mul.Tensor(sub_483, unsqueeze_1555);  sub_483 = unsqueeze_1555 = None
    mul_1582: "f32[512]" = torch.ops.aten.mul.Tensor(sum_191, squeeze_28);  sum_191 = squeeze_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_backward_94 = torch.ops.aten.convolution_backward.default(mul_1581, relu_7, primals_28, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_1581 = primals_28 = None
    getitem_492: "f32[8, 512, 56, 56]" = convolution_backward_94[0]
    getitem_493: "f32[512, 16, 3, 3]" = convolution_backward_94[1];  convolution_backward_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    alias_377: "f32[8, 512, 56, 56]" = torch.ops.aten.alias.default(relu_7);  relu_7 = None
    alias_378: "f32[8, 512, 56, 56]" = torch.ops.aten.alias.default(alias_377);  alias_377 = None
    le_92: "b8[8, 512, 56, 56]" = torch.ops.aten.le.Scalar(alias_378, 0);  alias_378 = None
    where_92: "f32[8, 512, 56, 56]" = torch.ops.aten.where.self(le_92, full_default, getitem_492);  le_92 = getitem_492 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    sum_192: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_92, [0, 2, 3])
    sub_484: "f32[8, 512, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_1558);  convolution_8 = unsqueeze_1558 = None
    mul_1583: "f32[8, 512, 56, 56]" = torch.ops.aten.mul.Tensor(where_92, sub_484)
    sum_193: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1583, [0, 2, 3]);  mul_1583 = None
    mul_1584: "f32[512]" = torch.ops.aten.mul.Tensor(sum_192, 3.985969387755102e-05)
    unsqueeze_1559: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1584, 0);  mul_1584 = None
    unsqueeze_1560: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1559, 2);  unsqueeze_1559 = None
    unsqueeze_1561: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1560, 3);  unsqueeze_1560 = None
    mul_1585: "f32[512]" = torch.ops.aten.mul.Tensor(sum_193, 3.985969387755102e-05)
    mul_1586: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
    mul_1587: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1585, mul_1586);  mul_1585 = mul_1586 = None
    unsqueeze_1562: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1587, 0);  mul_1587 = None
    unsqueeze_1563: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1562, 2);  unsqueeze_1562 = None
    unsqueeze_1564: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1563, 3);  unsqueeze_1563 = None
    mul_1588: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_25, primals_26);  primals_26 = None
    unsqueeze_1565: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1588, 0);  mul_1588 = None
    unsqueeze_1566: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1565, 2);  unsqueeze_1565 = None
    unsqueeze_1567: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1566, 3);  unsqueeze_1566 = None
    mul_1589: "f32[8, 512, 56, 56]" = torch.ops.aten.mul.Tensor(sub_484, unsqueeze_1564);  sub_484 = unsqueeze_1564 = None
    sub_486: "f32[8, 512, 56, 56]" = torch.ops.aten.sub.Tensor(where_92, mul_1589);  where_92 = mul_1589 = None
    sub_487: "f32[8, 512, 56, 56]" = torch.ops.aten.sub.Tensor(sub_486, unsqueeze_1561);  sub_486 = unsqueeze_1561 = None
    mul_1590: "f32[8, 512, 56, 56]" = torch.ops.aten.mul.Tensor(sub_487, unsqueeze_1567);  sub_487 = unsqueeze_1567 = None
    mul_1591: "f32[512]" = torch.ops.aten.mul.Tensor(sum_193, squeeze_25);  sum_193 = squeeze_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_backward_95 = torch.ops.aten.convolution_backward.default(mul_1590, relu_6, primals_25, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1590 = primals_25 = None
    getitem_495: "f32[8, 256, 56, 56]" = convolution_backward_95[0]
    getitem_496: "f32[512, 256, 1, 1]" = convolution_backward_95[1];  convolution_backward_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    add_583: "f32[8, 256, 56, 56]" = torch.ops.aten.add.Tensor(where_90, getitem_495);  where_90 = getitem_495 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    alias_380: "f32[8, 256, 56, 56]" = torch.ops.aten.alias.default(relu_6);  relu_6 = None
    alias_381: "f32[8, 256, 56, 56]" = torch.ops.aten.alias.default(alias_380);  alias_380 = None
    le_93: "b8[8, 256, 56, 56]" = torch.ops.aten.le.Scalar(alias_381, 0);  alias_381 = None
    where_93: "f32[8, 256, 56, 56]" = torch.ops.aten.where.self(le_93, full_default, add_583);  le_93 = add_583 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    sum_194: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_93, [0, 2, 3])
    sub_488: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_1570);  convolution_7 = unsqueeze_1570 = None
    mul_1592: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(where_93, sub_488)
    sum_195: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1592, [0, 2, 3]);  mul_1592 = None
    mul_1593: "f32[256]" = torch.ops.aten.mul.Tensor(sum_194, 3.985969387755102e-05)
    unsqueeze_1571: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1593, 0);  mul_1593 = None
    unsqueeze_1572: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1571, 2);  unsqueeze_1571 = None
    unsqueeze_1573: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1572, 3);  unsqueeze_1572 = None
    mul_1594: "f32[256]" = torch.ops.aten.mul.Tensor(sum_195, 3.985969387755102e-05)
    mul_1595: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_22, squeeze_22)
    mul_1596: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1594, mul_1595);  mul_1594 = mul_1595 = None
    unsqueeze_1574: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1596, 0);  mul_1596 = None
    unsqueeze_1575: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1574, 2);  unsqueeze_1574 = None
    unsqueeze_1576: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1575, 3);  unsqueeze_1575 = None
    mul_1597: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_22, primals_23);  primals_23 = None
    unsqueeze_1577: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1597, 0);  mul_1597 = None
    unsqueeze_1578: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1577, 2);  unsqueeze_1577 = None
    unsqueeze_1579: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1578, 3);  unsqueeze_1578 = None
    mul_1598: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_488, unsqueeze_1576);  sub_488 = unsqueeze_1576 = None
    sub_490: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(where_93, mul_1598);  mul_1598 = None
    sub_491: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(sub_490, unsqueeze_1573);  sub_490 = unsqueeze_1573 = None
    mul_1599: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_491, unsqueeze_1579);  sub_491 = unsqueeze_1579 = None
    mul_1600: "f32[256]" = torch.ops.aten.mul.Tensor(sum_195, squeeze_22);  sum_195 = squeeze_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_backward_96 = torch.ops.aten.convolution_backward.default(mul_1599, relu_5, primals_22, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1599 = primals_22 = None
    getitem_498: "f32[8, 512, 56, 56]" = convolution_backward_96[0]
    getitem_499: "f32[256, 512, 1, 1]" = convolution_backward_96[1];  convolution_backward_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    alias_383: "f32[8, 512, 56, 56]" = torch.ops.aten.alias.default(relu_5);  relu_5 = None
    alias_384: "f32[8, 512, 56, 56]" = torch.ops.aten.alias.default(alias_383);  alias_383 = None
    le_94: "b8[8, 512, 56, 56]" = torch.ops.aten.le.Scalar(alias_384, 0);  alias_384 = None
    where_94: "f32[8, 512, 56, 56]" = torch.ops.aten.where.self(le_94, full_default, getitem_498);  le_94 = getitem_498 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    sum_196: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_94, [0, 2, 3])
    sub_492: "f32[8, 512, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_1582);  convolution_6 = unsqueeze_1582 = None
    mul_1601: "f32[8, 512, 56, 56]" = torch.ops.aten.mul.Tensor(where_94, sub_492)
    sum_197: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1601, [0, 2, 3]);  mul_1601 = None
    mul_1602: "f32[512]" = torch.ops.aten.mul.Tensor(sum_196, 3.985969387755102e-05)
    unsqueeze_1583: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1602, 0);  mul_1602 = None
    unsqueeze_1584: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1583, 2);  unsqueeze_1583 = None
    unsqueeze_1585: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1584, 3);  unsqueeze_1584 = None
    mul_1603: "f32[512]" = torch.ops.aten.mul.Tensor(sum_197, 3.985969387755102e-05)
    mul_1604: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_19, squeeze_19)
    mul_1605: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1603, mul_1604);  mul_1603 = mul_1604 = None
    unsqueeze_1586: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1605, 0);  mul_1605 = None
    unsqueeze_1587: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1586, 2);  unsqueeze_1586 = None
    unsqueeze_1588: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1587, 3);  unsqueeze_1587 = None
    mul_1606: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_19, primals_20);  primals_20 = None
    unsqueeze_1589: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1606, 0);  mul_1606 = None
    unsqueeze_1590: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1589, 2);  unsqueeze_1589 = None
    unsqueeze_1591: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1590, 3);  unsqueeze_1590 = None
    mul_1607: "f32[8, 512, 56, 56]" = torch.ops.aten.mul.Tensor(sub_492, unsqueeze_1588);  sub_492 = unsqueeze_1588 = None
    sub_494: "f32[8, 512, 56, 56]" = torch.ops.aten.sub.Tensor(where_94, mul_1607);  where_94 = mul_1607 = None
    sub_495: "f32[8, 512, 56, 56]" = torch.ops.aten.sub.Tensor(sub_494, unsqueeze_1585);  sub_494 = unsqueeze_1585 = None
    mul_1608: "f32[8, 512, 56, 56]" = torch.ops.aten.mul.Tensor(sub_495, unsqueeze_1591);  sub_495 = unsqueeze_1591 = None
    mul_1609: "f32[512]" = torch.ops.aten.mul.Tensor(sum_197, squeeze_19);  sum_197 = squeeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_backward_97 = torch.ops.aten.convolution_backward.default(mul_1608, relu_4, primals_19, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_1608 = primals_19 = None
    getitem_501: "f32[8, 512, 56, 56]" = convolution_backward_97[0]
    getitem_502: "f32[512, 16, 3, 3]" = convolution_backward_97[1];  convolution_backward_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    alias_386: "f32[8, 512, 56, 56]" = torch.ops.aten.alias.default(relu_4);  relu_4 = None
    alias_387: "f32[8, 512, 56, 56]" = torch.ops.aten.alias.default(alias_386);  alias_386 = None
    le_95: "b8[8, 512, 56, 56]" = torch.ops.aten.le.Scalar(alias_387, 0);  alias_387 = None
    where_95: "f32[8, 512, 56, 56]" = torch.ops.aten.where.self(le_95, full_default, getitem_501);  le_95 = getitem_501 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    sum_198: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_95, [0, 2, 3])
    sub_496: "f32[8, 512, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_1594);  convolution_5 = unsqueeze_1594 = None
    mul_1610: "f32[8, 512, 56, 56]" = torch.ops.aten.mul.Tensor(where_95, sub_496)
    sum_199: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1610, [0, 2, 3]);  mul_1610 = None
    mul_1611: "f32[512]" = torch.ops.aten.mul.Tensor(sum_198, 3.985969387755102e-05)
    unsqueeze_1595: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1611, 0);  mul_1611 = None
    unsqueeze_1596: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1595, 2);  unsqueeze_1595 = None
    unsqueeze_1597: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1596, 3);  unsqueeze_1596 = None
    mul_1612: "f32[512]" = torch.ops.aten.mul.Tensor(sum_199, 3.985969387755102e-05)
    mul_1613: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_16, squeeze_16)
    mul_1614: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1612, mul_1613);  mul_1612 = mul_1613 = None
    unsqueeze_1598: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1614, 0);  mul_1614 = None
    unsqueeze_1599: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1598, 2);  unsqueeze_1598 = None
    unsqueeze_1600: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1599, 3);  unsqueeze_1599 = None
    mul_1615: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_16, primals_17);  primals_17 = None
    unsqueeze_1601: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1615, 0);  mul_1615 = None
    unsqueeze_1602: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1601, 2);  unsqueeze_1601 = None
    unsqueeze_1603: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1602, 3);  unsqueeze_1602 = None
    mul_1616: "f32[8, 512, 56, 56]" = torch.ops.aten.mul.Tensor(sub_496, unsqueeze_1600);  sub_496 = unsqueeze_1600 = None
    sub_498: "f32[8, 512, 56, 56]" = torch.ops.aten.sub.Tensor(where_95, mul_1616);  where_95 = mul_1616 = None
    sub_499: "f32[8, 512, 56, 56]" = torch.ops.aten.sub.Tensor(sub_498, unsqueeze_1597);  sub_498 = unsqueeze_1597 = None
    mul_1617: "f32[8, 512, 56, 56]" = torch.ops.aten.mul.Tensor(sub_499, unsqueeze_1603);  sub_499 = unsqueeze_1603 = None
    mul_1618: "f32[512]" = torch.ops.aten.mul.Tensor(sum_199, squeeze_16);  sum_199 = squeeze_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_backward_98 = torch.ops.aten.convolution_backward.default(mul_1617, relu_3, primals_16, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1617 = primals_16 = None
    getitem_504: "f32[8, 256, 56, 56]" = convolution_backward_98[0]
    getitem_505: "f32[512, 256, 1, 1]" = convolution_backward_98[1];  convolution_backward_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    add_584: "f32[8, 256, 56, 56]" = torch.ops.aten.add.Tensor(where_93, getitem_504);  where_93 = getitem_504 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    alias_389: "f32[8, 256, 56, 56]" = torch.ops.aten.alias.default(relu_3);  relu_3 = None
    alias_390: "f32[8, 256, 56, 56]" = torch.ops.aten.alias.default(alias_389);  alias_389 = None
    le_96: "b8[8, 256, 56, 56]" = torch.ops.aten.le.Scalar(alias_390, 0);  alias_390 = None
    where_96: "f32[8, 256, 56, 56]" = torch.ops.aten.where.self(le_96, full_default, add_584);  le_96 = add_584 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:200, code: shortcut = self.downsample(shortcut)
    sum_200: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_96, [0, 2, 3])
    sub_500: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_1606);  convolution_4 = unsqueeze_1606 = None
    mul_1619: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(where_96, sub_500)
    sum_201: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1619, [0, 2, 3]);  mul_1619 = None
    mul_1620: "f32[256]" = torch.ops.aten.mul.Tensor(sum_200, 3.985969387755102e-05)
    unsqueeze_1607: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1620, 0);  mul_1620 = None
    unsqueeze_1608: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1607, 2);  unsqueeze_1607 = None
    unsqueeze_1609: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1608, 3);  unsqueeze_1608 = None
    mul_1621: "f32[256]" = torch.ops.aten.mul.Tensor(sum_201, 3.985969387755102e-05)
    mul_1622: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
    mul_1623: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1621, mul_1622);  mul_1621 = mul_1622 = None
    unsqueeze_1610: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1623, 0);  mul_1623 = None
    unsqueeze_1611: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1610, 2);  unsqueeze_1610 = None
    unsqueeze_1612: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1611, 3);  unsqueeze_1611 = None
    mul_1624: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_13, primals_14);  primals_14 = None
    unsqueeze_1613: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1624, 0);  mul_1624 = None
    unsqueeze_1614: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1613, 2);  unsqueeze_1613 = None
    unsqueeze_1615: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1614, 3);  unsqueeze_1614 = None
    mul_1625: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_500, unsqueeze_1612);  sub_500 = unsqueeze_1612 = None
    sub_502: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(where_96, mul_1625);  mul_1625 = None
    sub_503: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(sub_502, unsqueeze_1609);  sub_502 = None
    mul_1626: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_503, unsqueeze_1615);  sub_503 = unsqueeze_1615 = None
    mul_1627: "f32[256]" = torch.ops.aten.mul.Tensor(sum_201, squeeze_13);  sum_201 = squeeze_13 = None
    convolution_backward_99 = torch.ops.aten.convolution_backward.default(mul_1626, getitem_2, primals_13, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1626 = primals_13 = None
    getitem_507: "f32[8, 64, 56, 56]" = convolution_backward_99[0]
    getitem_508: "f32[256, 64, 1, 1]" = convolution_backward_99[1];  convolution_backward_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    sub_504: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_1618);  convolution_3 = unsqueeze_1618 = None
    mul_1628: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(where_96, sub_504)
    sum_203: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1628, [0, 2, 3]);  mul_1628 = None
    mul_1630: "f32[256]" = torch.ops.aten.mul.Tensor(sum_203, 3.985969387755102e-05)
    mul_1631: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_10, squeeze_10)
    mul_1632: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1630, mul_1631);  mul_1630 = mul_1631 = None
    unsqueeze_1622: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1632, 0);  mul_1632 = None
    unsqueeze_1623: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1622, 2);  unsqueeze_1622 = None
    unsqueeze_1624: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1623, 3);  unsqueeze_1623 = None
    mul_1633: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_10, primals_11);  primals_11 = None
    unsqueeze_1625: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1633, 0);  mul_1633 = None
    unsqueeze_1626: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1625, 2);  unsqueeze_1625 = None
    unsqueeze_1627: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1626, 3);  unsqueeze_1626 = None
    mul_1634: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_504, unsqueeze_1624);  sub_504 = unsqueeze_1624 = None
    sub_506: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(where_96, mul_1634);  where_96 = mul_1634 = None
    sub_507: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(sub_506, unsqueeze_1609);  sub_506 = unsqueeze_1609 = None
    mul_1635: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_507, unsqueeze_1627);  sub_507 = unsqueeze_1627 = None
    mul_1636: "f32[256]" = torch.ops.aten.mul.Tensor(sum_203, squeeze_10);  sum_203 = squeeze_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_backward_100 = torch.ops.aten.convolution_backward.default(mul_1635, relu_2, primals_10, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1635 = primals_10 = None
    getitem_510: "f32[8, 512, 56, 56]" = convolution_backward_100[0]
    getitem_511: "f32[256, 512, 1, 1]" = convolution_backward_100[1];  convolution_backward_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    alias_392: "f32[8, 512, 56, 56]" = torch.ops.aten.alias.default(relu_2);  relu_2 = None
    alias_393: "f32[8, 512, 56, 56]" = torch.ops.aten.alias.default(alias_392);  alias_392 = None
    le_97: "b8[8, 512, 56, 56]" = torch.ops.aten.le.Scalar(alias_393, 0);  alias_393 = None
    where_97: "f32[8, 512, 56, 56]" = torch.ops.aten.where.self(le_97, full_default, getitem_510);  le_97 = getitem_510 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    sum_204: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_97, [0, 2, 3])
    sub_508: "f32[8, 512, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_1630);  convolution_2 = unsqueeze_1630 = None
    mul_1637: "f32[8, 512, 56, 56]" = torch.ops.aten.mul.Tensor(where_97, sub_508)
    sum_205: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1637, [0, 2, 3]);  mul_1637 = None
    mul_1638: "f32[512]" = torch.ops.aten.mul.Tensor(sum_204, 3.985969387755102e-05)
    unsqueeze_1631: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1638, 0);  mul_1638 = None
    unsqueeze_1632: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1631, 2);  unsqueeze_1631 = None
    unsqueeze_1633: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1632, 3);  unsqueeze_1632 = None
    mul_1639: "f32[512]" = torch.ops.aten.mul.Tensor(sum_205, 3.985969387755102e-05)
    mul_1640: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
    mul_1641: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1639, mul_1640);  mul_1639 = mul_1640 = None
    unsqueeze_1634: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1641, 0);  mul_1641 = None
    unsqueeze_1635: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1634, 2);  unsqueeze_1634 = None
    unsqueeze_1636: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1635, 3);  unsqueeze_1635 = None
    mul_1642: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_7, primals_8);  primals_8 = None
    unsqueeze_1637: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1642, 0);  mul_1642 = None
    unsqueeze_1638: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1637, 2);  unsqueeze_1637 = None
    unsqueeze_1639: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1638, 3);  unsqueeze_1638 = None
    mul_1643: "f32[8, 512, 56, 56]" = torch.ops.aten.mul.Tensor(sub_508, unsqueeze_1636);  sub_508 = unsqueeze_1636 = None
    sub_510: "f32[8, 512, 56, 56]" = torch.ops.aten.sub.Tensor(where_97, mul_1643);  where_97 = mul_1643 = None
    sub_511: "f32[8, 512, 56, 56]" = torch.ops.aten.sub.Tensor(sub_510, unsqueeze_1633);  sub_510 = unsqueeze_1633 = None
    mul_1644: "f32[8, 512, 56, 56]" = torch.ops.aten.mul.Tensor(sub_511, unsqueeze_1639);  sub_511 = unsqueeze_1639 = None
    mul_1645: "f32[512]" = torch.ops.aten.mul.Tensor(sum_205, squeeze_7);  sum_205 = squeeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_backward_101 = torch.ops.aten.convolution_backward.default(mul_1644, relu_1, primals_7, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_1644 = primals_7 = None
    getitem_513: "f32[8, 512, 56, 56]" = convolution_backward_101[0]
    getitem_514: "f32[512, 16, 3, 3]" = convolution_backward_101[1];  convolution_backward_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    alias_395: "f32[8, 512, 56, 56]" = torch.ops.aten.alias.default(relu_1);  relu_1 = None
    alias_396: "f32[8, 512, 56, 56]" = torch.ops.aten.alias.default(alias_395);  alias_395 = None
    le_98: "b8[8, 512, 56, 56]" = torch.ops.aten.le.Scalar(alias_396, 0);  alias_396 = None
    where_98: "f32[8, 512, 56, 56]" = torch.ops.aten.where.self(le_98, full_default, getitem_513);  le_98 = getitem_513 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    sum_206: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_98, [0, 2, 3])
    sub_512: "f32[8, 512, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_1642);  convolution_1 = unsqueeze_1642 = None
    mul_1646: "f32[8, 512, 56, 56]" = torch.ops.aten.mul.Tensor(where_98, sub_512)
    sum_207: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1646, [0, 2, 3]);  mul_1646 = None
    mul_1647: "f32[512]" = torch.ops.aten.mul.Tensor(sum_206, 3.985969387755102e-05)
    unsqueeze_1643: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1647, 0);  mul_1647 = None
    unsqueeze_1644: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1643, 2);  unsqueeze_1643 = None
    unsqueeze_1645: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1644, 3);  unsqueeze_1644 = None
    mul_1648: "f32[512]" = torch.ops.aten.mul.Tensor(sum_207, 3.985969387755102e-05)
    mul_1649: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
    mul_1650: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1648, mul_1649);  mul_1648 = mul_1649 = None
    unsqueeze_1646: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1650, 0);  mul_1650 = None
    unsqueeze_1647: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1646, 2);  unsqueeze_1646 = None
    unsqueeze_1648: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1647, 3);  unsqueeze_1647 = None
    mul_1651: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_4, primals_5);  primals_5 = None
    unsqueeze_1649: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1651, 0);  mul_1651 = None
    unsqueeze_1650: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1649, 2);  unsqueeze_1649 = None
    unsqueeze_1651: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1650, 3);  unsqueeze_1650 = None
    mul_1652: "f32[8, 512, 56, 56]" = torch.ops.aten.mul.Tensor(sub_512, unsqueeze_1648);  sub_512 = unsqueeze_1648 = None
    sub_514: "f32[8, 512, 56, 56]" = torch.ops.aten.sub.Tensor(where_98, mul_1652);  where_98 = mul_1652 = None
    sub_515: "f32[8, 512, 56, 56]" = torch.ops.aten.sub.Tensor(sub_514, unsqueeze_1645);  sub_514 = unsqueeze_1645 = None
    mul_1653: "f32[8, 512, 56, 56]" = torch.ops.aten.mul.Tensor(sub_515, unsqueeze_1651);  sub_515 = unsqueeze_1651 = None
    mul_1654: "f32[512]" = torch.ops.aten.mul.Tensor(sum_207, squeeze_4);  sum_207 = squeeze_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_backward_102 = torch.ops.aten.convolution_backward.default(mul_1653, getitem_2, primals_4, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1653 = getitem_2 = primals_4 = None
    getitem_516: "f32[8, 64, 56, 56]" = convolution_backward_102[0]
    getitem_517: "f32[512, 64, 1, 1]" = convolution_backward_102[1];  convolution_backward_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    add_585: "f32[8, 64, 56, 56]" = torch.ops.aten.add.Tensor(getitem_507, getitem_516);  getitem_507 = getitem_516 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:523, code: x = self.maxpool(x)
    max_pool2d_with_indices_backward: "f32[8, 64, 112, 112]" = torch.ops.aten.max_pool2d_with_indices_backward.default(add_585, relu, [3, 3], [2, 2], [1, 1], [1, 1], False, getitem_3);  add_585 = getitem_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:522, code: x = self.act1(x)
    alias_398: "f32[8, 64, 112, 112]" = torch.ops.aten.alias.default(relu);  relu = None
    alias_399: "f32[8, 64, 112, 112]" = torch.ops.aten.alias.default(alias_398);  alias_398 = None
    le_99: "b8[8, 64, 112, 112]" = torch.ops.aten.le.Scalar(alias_399, 0);  alias_399 = None
    where_99: "f32[8, 64, 112, 112]" = torch.ops.aten.where.self(le_99, full_default, max_pool2d_with_indices_backward);  le_99 = full_default = max_pool2d_with_indices_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:521, code: x = self.bn1(x)
    sum_208: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_99, [0, 2, 3])
    sub_516: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_1654);  convolution = unsqueeze_1654 = None
    mul_1655: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(where_99, sub_516)
    sum_209: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1655, [0, 2, 3]);  mul_1655 = None
    mul_1656: "f32[64]" = torch.ops.aten.mul.Tensor(sum_208, 9.964923469387754e-06)
    unsqueeze_1655: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1656, 0);  mul_1656 = None
    unsqueeze_1656: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1655, 2);  unsqueeze_1655 = None
    unsqueeze_1657: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1656, 3);  unsqueeze_1656 = None
    mul_1657: "f32[64]" = torch.ops.aten.mul.Tensor(sum_209, 9.964923469387754e-06)
    mul_1658: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
    mul_1659: "f32[64]" = torch.ops.aten.mul.Tensor(mul_1657, mul_1658);  mul_1657 = mul_1658 = None
    unsqueeze_1658: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1659, 0);  mul_1659 = None
    unsqueeze_1659: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1658, 2);  unsqueeze_1658 = None
    unsqueeze_1660: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1659, 3);  unsqueeze_1659 = None
    mul_1660: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_1, primals_2);  primals_2 = None
    unsqueeze_1661: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1660, 0);  mul_1660 = None
    unsqueeze_1662: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1661, 2);  unsqueeze_1661 = None
    unsqueeze_1663: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1662, 3);  unsqueeze_1662 = None
    mul_1661: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_516, unsqueeze_1660);  sub_516 = unsqueeze_1660 = None
    sub_518: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(where_99, mul_1661);  where_99 = mul_1661 = None
    sub_519: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(sub_518, unsqueeze_1657);  sub_518 = unsqueeze_1657 = None
    mul_1662: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_519, unsqueeze_1663);  sub_519 = unsqueeze_1663 = None
    mul_1663: "f32[64]" = torch.ops.aten.mul.Tensor(sum_209, squeeze_1);  sum_209 = squeeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:520, code: x = self.conv1(x)
    convolution_backward_103 = torch.ops.aten.convolution_backward.default(mul_1662, primals_627, primals_1, [0], [2, 2], [3, 3], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_1662 = primals_627 = primals_1 = None
    getitem_520: "f32[64, 3, 7, 7]" = convolution_backward_103[1];  convolution_backward_103 = None
    return [getitem_520, mul_1663, sum_208, getitem_517, mul_1654, sum_206, getitem_514, mul_1645, sum_204, getitem_511, mul_1636, sum_200, getitem_508, mul_1627, sum_200, getitem_505, mul_1618, sum_198, getitem_502, mul_1609, sum_196, getitem_499, mul_1600, sum_194, getitem_496, mul_1591, sum_192, getitem_493, mul_1582, sum_190, getitem_490, mul_1573, sum_188, getitem_487, mul_1564, sum_186, getitem_484, mul_1555, sum_184, getitem_481, mul_1546, sum_180, getitem_478, mul_1537, sum_180, getitem_475, mul_1528, sum_178, getitem_472, mul_1519, sum_176, getitem_469, mul_1510, sum_174, getitem_466, mul_1501, sum_172, getitem_463, mul_1492, sum_170, getitem_460, mul_1483, sum_168, getitem_457, mul_1474, sum_166, getitem_454, mul_1465, sum_164, getitem_451, mul_1456, sum_162, getitem_448, mul_1447, sum_160, getitem_445, mul_1438, sum_158, getitem_442, mul_1429, sum_154, getitem_439, mul_1420, sum_154, getitem_436, mul_1411, sum_152, getitem_433, mul_1402, sum_150, getitem_430, mul_1393, sum_148, getitem_427, mul_1384, sum_146, getitem_424, mul_1375, sum_144, getitem_421, mul_1366, sum_142, getitem_418, mul_1357, sum_140, getitem_415, mul_1348, sum_138, getitem_412, mul_1339, sum_136, getitem_409, mul_1330, sum_134, getitem_406, mul_1321, sum_132, getitem_403, mul_1312, sum_130, getitem_400, mul_1303, sum_128, getitem_397, mul_1294, sum_126, getitem_394, mul_1285, sum_124, getitem_391, mul_1276, sum_122, getitem_388, mul_1267, sum_120, getitem_385, mul_1258, sum_118, getitem_382, mul_1249, sum_116, getitem_379, mul_1240, sum_114, getitem_376, mul_1231, sum_112, getitem_373, mul_1222, sum_110, getitem_370, mul_1213, sum_108, getitem_367, mul_1204, sum_106, getitem_364, mul_1195, sum_104, getitem_361, mul_1186, sum_102, getitem_358, mul_1177, sum_100, getitem_355, mul_1168, sum_98, getitem_352, mul_1159, sum_96, getitem_349, mul_1150, sum_94, getitem_346, mul_1141, sum_92, getitem_343, mul_1132, sum_90, getitem_340, mul_1123, sum_88, getitem_337, mul_1114, sum_86, getitem_334, mul_1105, sum_84, getitem_331, mul_1096, sum_82, getitem_328, mul_1087, sum_80, getitem_325, mul_1078, sum_78, getitem_322, mul_1069, sum_76, getitem_319, mul_1060, sum_74, getitem_316, mul_1051, sum_72, getitem_313, mul_1042, sum_70, getitem_310, mul_1033, sum_68, getitem_307, mul_1024, sum_66, getitem_304, mul_1015, sum_64, getitem_301, mul_1006, sum_62, getitem_298, mul_997, sum_60, getitem_295, mul_988, sum_58, getitem_292, mul_979, sum_56, getitem_289, mul_970, sum_54, getitem_286, mul_961, sum_52, getitem_283, mul_952, sum_50, getitem_280, mul_943, sum_48, getitem_277, mul_934, sum_46, getitem_274, mul_925, sum_44, getitem_271, mul_916, sum_42, getitem_268, mul_907, sum_40, getitem_265, mul_898, sum_38, getitem_262, mul_889, sum_36, getitem_259, mul_880, sum_34, getitem_256, mul_871, sum_32, getitem_253, mul_862, sum_30, getitem_250, mul_853, sum_28, getitem_247, mul_844, sum_26, getitem_244, mul_835, sum_24, getitem_241, mul_826, sum_22, getitem_238, mul_817, sum_20, getitem_235, mul_808, sum_18, getitem_232, mul_799, sum_14, getitem_229, mul_790, sum_14, getitem_226, mul_781, sum_12, getitem_223, mul_772, sum_10, getitem_220, mul_763, sum_8, getitem_217, mul_754, sum_6, getitem_214, mul_745, sum_4, getitem_211, mul_736, sum_2, permute_4, view_1, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
    