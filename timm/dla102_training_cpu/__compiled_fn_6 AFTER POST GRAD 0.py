from __future__ import annotations



def forward(self, primals_1: "f32[16, 3, 7, 7]", primals_2: "f32[16]", primals_4: "f32[16, 16, 3, 3]", primals_5: "f32[16]", primals_7: "f32[32, 16, 3, 3]", primals_8: "f32[32]", primals_10: "f32[128, 32, 1, 1]", primals_11: "f32[128]", primals_13: "f32[64, 32, 1, 1]", primals_14: "f32[64]", primals_16: "f32[64, 64, 3, 3]", primals_17: "f32[64]", primals_19: "f32[128, 64, 1, 1]", primals_20: "f32[128]", primals_22: "f32[64, 128, 1, 1]", primals_23: "f32[64]", primals_25: "f32[64, 64, 3, 3]", primals_26: "f32[64]", primals_28: "f32[128, 64, 1, 1]", primals_29: "f32[128]", primals_31: "f32[128, 256, 1, 1]", primals_32: "f32[128]", primals_34: "f32[256, 128, 1, 1]", primals_35: "f32[256]", primals_37: "f32[128, 128, 1, 1]", primals_38: "f32[128]", primals_40: "f32[128, 128, 3, 3]", primals_41: "f32[128]", primals_43: "f32[256, 128, 1, 1]", primals_44: "f32[256]", primals_46: "f32[128, 256, 1, 1]", primals_47: "f32[128]", primals_49: "f32[128, 128, 3, 3]", primals_50: "f32[128]", primals_52: "f32[256, 128, 1, 1]", primals_53: "f32[256]", primals_55: "f32[256, 512, 1, 1]", primals_56: "f32[256]", primals_58: "f32[128, 256, 1, 1]", primals_59: "f32[128]", primals_61: "f32[128, 128, 3, 3]", primals_62: "f32[128]", primals_64: "f32[256, 128, 1, 1]", primals_65: "f32[256]", primals_67: "f32[128, 256, 1, 1]", primals_68: "f32[128]", primals_70: "f32[128, 128, 3, 3]", primals_71: "f32[128]", primals_73: "f32[256, 128, 1, 1]", primals_74: "f32[256]", primals_76: "f32[256, 768, 1, 1]", primals_77: "f32[256]", primals_79: "f32[128, 256, 1, 1]", primals_80: "f32[128]", primals_82: "f32[128, 128, 3, 3]", primals_83: "f32[128]", primals_85: "f32[256, 128, 1, 1]", primals_86: "f32[256]", primals_88: "f32[128, 256, 1, 1]", primals_89: "f32[128]", primals_91: "f32[128, 128, 3, 3]", primals_92: "f32[128]", primals_94: "f32[256, 128, 1, 1]", primals_95: "f32[256]", primals_97: "f32[256, 512, 1, 1]", primals_98: "f32[256]", primals_100: "f32[128, 256, 1, 1]", primals_101: "f32[128]", primals_103: "f32[128, 128, 3, 3]", primals_104: "f32[128]", primals_106: "f32[256, 128, 1, 1]", primals_107: "f32[256]", primals_109: "f32[128, 256, 1, 1]", primals_110: "f32[128]", primals_112: "f32[128, 128, 3, 3]", primals_113: "f32[128]", primals_115: "f32[256, 128, 1, 1]", primals_116: "f32[256]", primals_118: "f32[256, 1152, 1, 1]", primals_119: "f32[256]", primals_121: "f32[512, 256, 1, 1]", primals_122: "f32[512]", primals_124: "f32[256, 256, 1, 1]", primals_125: "f32[256]", primals_127: "f32[256, 256, 3, 3]", primals_128: "f32[256]", primals_130: "f32[512, 256, 1, 1]", primals_131: "f32[512]", primals_133: "f32[256, 512, 1, 1]", primals_134: "f32[256]", primals_136: "f32[256, 256, 3, 3]", primals_137: "f32[256]", primals_139: "f32[512, 256, 1, 1]", primals_140: "f32[512]", primals_142: "f32[512, 1024, 1, 1]", primals_143: "f32[512]", primals_145: "f32[256, 512, 1, 1]", primals_146: "f32[256]", primals_148: "f32[256, 256, 3, 3]", primals_149: "f32[256]", primals_151: "f32[512, 256, 1, 1]", primals_152: "f32[512]", primals_154: "f32[256, 512, 1, 1]", primals_155: "f32[256]", primals_157: "f32[256, 256, 3, 3]", primals_158: "f32[256]", primals_160: "f32[512, 256, 1, 1]", primals_161: "f32[512]", primals_163: "f32[512, 1536, 1, 1]", primals_164: "f32[512]", primals_166: "f32[256, 512, 1, 1]", primals_167: "f32[256]", primals_169: "f32[256, 256, 3, 3]", primals_170: "f32[256]", primals_172: "f32[512, 256, 1, 1]", primals_173: "f32[512]", primals_175: "f32[256, 512, 1, 1]", primals_176: "f32[256]", primals_178: "f32[256, 256, 3, 3]", primals_179: "f32[256]", primals_181: "f32[512, 256, 1, 1]", primals_182: "f32[512]", primals_184: "f32[512, 1024, 1, 1]", primals_185: "f32[512]", primals_187: "f32[256, 512, 1, 1]", primals_188: "f32[256]", primals_190: "f32[256, 256, 3, 3]", primals_191: "f32[256]", primals_193: "f32[512, 256, 1, 1]", primals_194: "f32[512]", primals_196: "f32[256, 512, 1, 1]", primals_197: "f32[256]", primals_199: "f32[256, 256, 3, 3]", primals_200: "f32[256]", primals_202: "f32[512, 256, 1, 1]", primals_203: "f32[512]", primals_205: "f32[512, 2048, 1, 1]", primals_206: "f32[512]", primals_208: "f32[256, 512, 1, 1]", primals_209: "f32[256]", primals_211: "f32[256, 256, 3, 3]", primals_212: "f32[256]", primals_214: "f32[512, 256, 1, 1]", primals_215: "f32[512]", primals_217: "f32[256, 512, 1, 1]", primals_218: "f32[256]", primals_220: "f32[256, 256, 3, 3]", primals_221: "f32[256]", primals_223: "f32[512, 256, 1, 1]", primals_224: "f32[512]", primals_226: "f32[512, 1024, 1, 1]", primals_227: "f32[512]", primals_229: "f32[256, 512, 1, 1]", primals_230: "f32[256]", primals_232: "f32[256, 256, 3, 3]", primals_233: "f32[256]", primals_235: "f32[512, 256, 1, 1]", primals_236: "f32[512]", primals_238: "f32[256, 512, 1, 1]", primals_239: "f32[256]", primals_241: "f32[256, 256, 3, 3]", primals_242: "f32[256]", primals_244: "f32[512, 256, 1, 1]", primals_245: "f32[512]", primals_247: "f32[512, 1536, 1, 1]", primals_248: "f32[512]", primals_250: "f32[256, 512, 1, 1]", primals_251: "f32[256]", primals_253: "f32[256, 256, 3, 3]", primals_254: "f32[256]", primals_256: "f32[512, 256, 1, 1]", primals_257: "f32[512]", primals_259: "f32[256, 512, 1, 1]", primals_260: "f32[256]", primals_262: "f32[256, 256, 3, 3]", primals_263: "f32[256]", primals_265: "f32[512, 256, 1, 1]", primals_266: "f32[512]", primals_268: "f32[512, 1024, 1, 1]", primals_269: "f32[512]", primals_271: "f32[256, 512, 1, 1]", primals_272: "f32[256]", primals_274: "f32[256, 256, 3, 3]", primals_275: "f32[256]", primals_277: "f32[512, 256, 1, 1]", primals_278: "f32[512]", primals_280: "f32[256, 512, 1, 1]", primals_281: "f32[256]", primals_283: "f32[256, 256, 3, 3]", primals_284: "f32[256]", primals_286: "f32[512, 256, 1, 1]", primals_287: "f32[512]", primals_289: "f32[512, 2816, 1, 1]", primals_290: "f32[512]", primals_292: "f32[1024, 512, 1, 1]", primals_293: "f32[1024]", primals_295: "f32[512, 512, 1, 1]", primals_296: "f32[512]", primals_298: "f32[512, 512, 3, 3]", primals_299: "f32[512]", primals_301: "f32[1024, 512, 1, 1]", primals_302: "f32[1024]", primals_304: "f32[512, 1024, 1, 1]", primals_305: "f32[512]", primals_307: "f32[512, 512, 3, 3]", primals_308: "f32[512]", primals_310: "f32[1024, 512, 1, 1]", primals_311: "f32[1024]", primals_313: "f32[1024, 2560, 1, 1]", primals_314: "f32[1024]", primals_316: "f32[1000, 1024, 1, 1]", primals_633: "f32[8, 3, 224, 224]", convolution: "f32[8, 16, 224, 224]", squeeze_1: "f32[16]", relu: "f32[8, 16, 224, 224]", convolution_1: "f32[8, 16, 224, 224]", squeeze_4: "f32[16]", relu_1: "f32[8, 16, 224, 224]", convolution_2: "f32[8, 32, 112, 112]", squeeze_7: "f32[32]", relu_2: "f32[8, 32, 112, 112]", getitem_6: "f32[8, 32, 56, 56]", getitem_7: "i64[8, 32, 56, 56]", convolution_3: "f32[8, 128, 56, 56]", squeeze_10: "f32[128]", convolution_4: "f32[8, 64, 112, 112]", squeeze_13: "f32[64]", relu_3: "f32[8, 64, 112, 112]", convolution_5: "f32[8, 64, 56, 56]", squeeze_16: "f32[64]", relu_4: "f32[8, 64, 56, 56]", convolution_6: "f32[8, 128, 56, 56]", squeeze_19: "f32[128]", relu_5: "f32[8, 128, 56, 56]", convolution_7: "f32[8, 64, 56, 56]", squeeze_22: "f32[64]", relu_6: "f32[8, 64, 56, 56]", convolution_8: "f32[8, 64, 56, 56]", squeeze_25: "f32[64]", relu_7: "f32[8, 64, 56, 56]", convolution_9: "f32[8, 128, 56, 56]", squeeze_28: "f32[128]", cat: "f32[8, 256, 56, 56]", convolution_10: "f32[8, 128, 56, 56]", squeeze_31: "f32[128]", relu_9: "f32[8, 128, 56, 56]", getitem_24: "f32[8, 128, 28, 28]", getitem_25: "i64[8, 128, 28, 28]", convolution_11: "f32[8, 256, 28, 28]", squeeze_34: "f32[256]", convolution_12: "f32[8, 128, 56, 56]", squeeze_37: "f32[128]", relu_10: "f32[8, 128, 56, 56]", convolution_13: "f32[8, 128, 28, 28]", squeeze_40: "f32[128]", relu_11: "f32[8, 128, 28, 28]", convolution_14: "f32[8, 256, 28, 28]", squeeze_43: "f32[256]", relu_12: "f32[8, 256, 28, 28]", convolution_15: "f32[8, 128, 28, 28]", squeeze_46: "f32[128]", relu_13: "f32[8, 128, 28, 28]", convolution_16: "f32[8, 128, 28, 28]", squeeze_49: "f32[128]", relu_14: "f32[8, 128, 28, 28]", convolution_17: "f32[8, 256, 28, 28]", squeeze_52: "f32[256]", cat_1: "f32[8, 512, 28, 28]", convolution_18: "f32[8, 256, 28, 28]", squeeze_55: "f32[256]", relu_16: "f32[8, 256, 28, 28]", convolution_19: "f32[8, 128, 28, 28]", squeeze_58: "f32[128]", relu_17: "f32[8, 128, 28, 28]", convolution_20: "f32[8, 128, 28, 28]", squeeze_61: "f32[128]", relu_18: "f32[8, 128, 28, 28]", convolution_21: "f32[8, 256, 28, 28]", squeeze_64: "f32[256]", relu_19: "f32[8, 256, 28, 28]", convolution_22: "f32[8, 128, 28, 28]", squeeze_67: "f32[128]", relu_20: "f32[8, 128, 28, 28]", convolution_23: "f32[8, 128, 28, 28]", squeeze_70: "f32[128]", relu_21: "f32[8, 128, 28, 28]", convolution_24: "f32[8, 256, 28, 28]", squeeze_73: "f32[256]", cat_2: "f32[8, 768, 28, 28]", convolution_25: "f32[8, 256, 28, 28]", squeeze_76: "f32[256]", relu_23: "f32[8, 256, 28, 28]", convolution_26: "f32[8, 128, 28, 28]", squeeze_79: "f32[128]", relu_24: "f32[8, 128, 28, 28]", convolution_27: "f32[8, 128, 28, 28]", squeeze_82: "f32[128]", relu_25: "f32[8, 128, 28, 28]", convolution_28: "f32[8, 256, 28, 28]", squeeze_85: "f32[256]", relu_26: "f32[8, 256, 28, 28]", convolution_29: "f32[8, 128, 28, 28]", squeeze_88: "f32[128]", relu_27: "f32[8, 128, 28, 28]", convolution_30: "f32[8, 128, 28, 28]", squeeze_91: "f32[128]", relu_28: "f32[8, 128, 28, 28]", convolution_31: "f32[8, 256, 28, 28]", squeeze_94: "f32[256]", cat_3: "f32[8, 512, 28, 28]", convolution_32: "f32[8, 256, 28, 28]", squeeze_97: "f32[256]", relu_30: "f32[8, 256, 28, 28]", convolution_33: "f32[8, 128, 28, 28]", squeeze_100: "f32[128]", relu_31: "f32[8, 128, 28, 28]", convolution_34: "f32[8, 128, 28, 28]", squeeze_103: "f32[128]", relu_32: "f32[8, 128, 28, 28]", convolution_35: "f32[8, 256, 28, 28]", squeeze_106: "f32[256]", relu_33: "f32[8, 256, 28, 28]", convolution_36: "f32[8, 128, 28, 28]", squeeze_109: "f32[128]", relu_34: "f32[8, 128, 28, 28]", convolution_37: "f32[8, 128, 28, 28]", squeeze_112: "f32[128]", relu_35: "f32[8, 128, 28, 28]", convolution_38: "f32[8, 256, 28, 28]", squeeze_115: "f32[256]", cat_4: "f32[8, 1152, 28, 28]", convolution_39: "f32[8, 256, 28, 28]", squeeze_118: "f32[256]", relu_37: "f32[8, 256, 28, 28]", getitem_88: "f32[8, 256, 14, 14]", getitem_89: "i64[8, 256, 14, 14]", convolution_40: "f32[8, 512, 14, 14]", squeeze_121: "f32[512]", convolution_41: "f32[8, 256, 28, 28]", squeeze_124: "f32[256]", relu_38: "f32[8, 256, 28, 28]", convolution_42: "f32[8, 256, 14, 14]", squeeze_127: "f32[256]", relu_39: "f32[8, 256, 14, 14]", convolution_43: "f32[8, 512, 14, 14]", squeeze_130: "f32[512]", relu_40: "f32[8, 512, 14, 14]", convolution_44: "f32[8, 256, 14, 14]", squeeze_133: "f32[256]", relu_41: "f32[8, 256, 14, 14]", convolution_45: "f32[8, 256, 14, 14]", squeeze_136: "f32[256]", relu_42: "f32[8, 256, 14, 14]", convolution_46: "f32[8, 512, 14, 14]", squeeze_139: "f32[512]", cat_5: "f32[8, 1024, 14, 14]", convolution_47: "f32[8, 512, 14, 14]", squeeze_142: "f32[512]", relu_44: "f32[8, 512, 14, 14]", convolution_48: "f32[8, 256, 14, 14]", squeeze_145: "f32[256]", relu_45: "f32[8, 256, 14, 14]", convolution_49: "f32[8, 256, 14, 14]", squeeze_148: "f32[256]", relu_46: "f32[8, 256, 14, 14]", convolution_50: "f32[8, 512, 14, 14]", squeeze_151: "f32[512]", relu_47: "f32[8, 512, 14, 14]", convolution_51: "f32[8, 256, 14, 14]", squeeze_154: "f32[256]", relu_48: "f32[8, 256, 14, 14]", convolution_52: "f32[8, 256, 14, 14]", squeeze_157: "f32[256]", relu_49: "f32[8, 256, 14, 14]", convolution_53: "f32[8, 512, 14, 14]", squeeze_160: "f32[512]", cat_6: "f32[8, 1536, 14, 14]", convolution_54: "f32[8, 512, 14, 14]", squeeze_163: "f32[512]", relu_51: "f32[8, 512, 14, 14]", convolution_55: "f32[8, 256, 14, 14]", squeeze_166: "f32[256]", relu_52: "f32[8, 256, 14, 14]", convolution_56: "f32[8, 256, 14, 14]", squeeze_169: "f32[256]", relu_53: "f32[8, 256, 14, 14]", convolution_57: "f32[8, 512, 14, 14]", squeeze_172: "f32[512]", relu_54: "f32[8, 512, 14, 14]", convolution_58: "f32[8, 256, 14, 14]", squeeze_175: "f32[256]", relu_55: "f32[8, 256, 14, 14]", convolution_59: "f32[8, 256, 14, 14]", squeeze_178: "f32[256]", relu_56: "f32[8, 256, 14, 14]", convolution_60: "f32[8, 512, 14, 14]", squeeze_181: "f32[512]", cat_7: "f32[8, 1024, 14, 14]", convolution_61: "f32[8, 512, 14, 14]", squeeze_184: "f32[512]", relu_58: "f32[8, 512, 14, 14]", convolution_62: "f32[8, 256, 14, 14]", squeeze_187: "f32[256]", relu_59: "f32[8, 256, 14, 14]", convolution_63: "f32[8, 256, 14, 14]", squeeze_190: "f32[256]", relu_60: "f32[8, 256, 14, 14]", convolution_64: "f32[8, 512, 14, 14]", squeeze_193: "f32[512]", relu_61: "f32[8, 512, 14, 14]", convolution_65: "f32[8, 256, 14, 14]", squeeze_196: "f32[256]", relu_62: "f32[8, 256, 14, 14]", convolution_66: "f32[8, 256, 14, 14]", squeeze_199: "f32[256]", relu_63: "f32[8, 256, 14, 14]", convolution_67: "f32[8, 512, 14, 14]", squeeze_202: "f32[512]", cat_8: "f32[8, 2048, 14, 14]", convolution_68: "f32[8, 512, 14, 14]", squeeze_205: "f32[512]", relu_65: "f32[8, 512, 14, 14]", convolution_69: "f32[8, 256, 14, 14]", squeeze_208: "f32[256]", relu_66: "f32[8, 256, 14, 14]", convolution_70: "f32[8, 256, 14, 14]", squeeze_211: "f32[256]", relu_67: "f32[8, 256, 14, 14]", convolution_71: "f32[8, 512, 14, 14]", squeeze_214: "f32[512]", relu_68: "f32[8, 512, 14, 14]", convolution_72: "f32[8, 256, 14, 14]", squeeze_217: "f32[256]", relu_69: "f32[8, 256, 14, 14]", convolution_73: "f32[8, 256, 14, 14]", squeeze_220: "f32[256]", relu_70: "f32[8, 256, 14, 14]", convolution_74: "f32[8, 512, 14, 14]", squeeze_223: "f32[512]", cat_9: "f32[8, 1024, 14, 14]", convolution_75: "f32[8, 512, 14, 14]", squeeze_226: "f32[512]", relu_72: "f32[8, 512, 14, 14]", convolution_76: "f32[8, 256, 14, 14]", squeeze_229: "f32[256]", relu_73: "f32[8, 256, 14, 14]", convolution_77: "f32[8, 256, 14, 14]", squeeze_232: "f32[256]", relu_74: "f32[8, 256, 14, 14]", convolution_78: "f32[8, 512, 14, 14]", squeeze_235: "f32[512]", relu_75: "f32[8, 512, 14, 14]", convolution_79: "f32[8, 256, 14, 14]", squeeze_238: "f32[256]", relu_76: "f32[8, 256, 14, 14]", convolution_80: "f32[8, 256, 14, 14]", squeeze_241: "f32[256]", relu_77: "f32[8, 256, 14, 14]", convolution_81: "f32[8, 512, 14, 14]", squeeze_244: "f32[512]", cat_10: "f32[8, 1536, 14, 14]", convolution_82: "f32[8, 512, 14, 14]", squeeze_247: "f32[512]", relu_79: "f32[8, 512, 14, 14]", convolution_83: "f32[8, 256, 14, 14]", squeeze_250: "f32[256]", relu_80: "f32[8, 256, 14, 14]", convolution_84: "f32[8, 256, 14, 14]", squeeze_253: "f32[256]", relu_81: "f32[8, 256, 14, 14]", convolution_85: "f32[8, 512, 14, 14]", squeeze_256: "f32[512]", relu_82: "f32[8, 512, 14, 14]", convolution_86: "f32[8, 256, 14, 14]", squeeze_259: "f32[256]", relu_83: "f32[8, 256, 14, 14]", convolution_87: "f32[8, 256, 14, 14]", squeeze_262: "f32[256]", relu_84: "f32[8, 256, 14, 14]", convolution_88: "f32[8, 512, 14, 14]", squeeze_265: "f32[512]", cat_11: "f32[8, 1024, 14, 14]", convolution_89: "f32[8, 512, 14, 14]", squeeze_268: "f32[512]", relu_86: "f32[8, 512, 14, 14]", convolution_90: "f32[8, 256, 14, 14]", squeeze_271: "f32[256]", relu_87: "f32[8, 256, 14, 14]", convolution_91: "f32[8, 256, 14, 14]", squeeze_274: "f32[256]", relu_88: "f32[8, 256, 14, 14]", convolution_92: "f32[8, 512, 14, 14]", squeeze_277: "f32[512]", relu_89: "f32[8, 512, 14, 14]", convolution_93: "f32[8, 256, 14, 14]", squeeze_280: "f32[256]", relu_90: "f32[8, 256, 14, 14]", convolution_94: "f32[8, 256, 14, 14]", squeeze_283: "f32[256]", relu_91: "f32[8, 256, 14, 14]", convolution_95: "f32[8, 512, 14, 14]", squeeze_286: "f32[512]", cat_12: "f32[8, 2816, 14, 14]", convolution_96: "f32[8, 512, 14, 14]", squeeze_289: "f32[512]", relu_93: "f32[8, 512, 14, 14]", getitem_210: "f32[8, 512, 7, 7]", getitem_211: "i64[8, 512, 7, 7]", convolution_97: "f32[8, 1024, 7, 7]", squeeze_292: "f32[1024]", convolution_98: "f32[8, 512, 14, 14]", squeeze_295: "f32[512]", relu_94: "f32[8, 512, 14, 14]", convolution_99: "f32[8, 512, 7, 7]", squeeze_298: "f32[512]", relu_95: "f32[8, 512, 7, 7]", convolution_100: "f32[8, 1024, 7, 7]", squeeze_301: "f32[1024]", relu_96: "f32[8, 1024, 7, 7]", convolution_101: "f32[8, 512, 7, 7]", squeeze_304: "f32[512]", relu_97: "f32[8, 512, 7, 7]", convolution_102: "f32[8, 512, 7, 7]", squeeze_307: "f32[512]", relu_98: "f32[8, 512, 7, 7]", convolution_103: "f32[8, 1024, 7, 7]", squeeze_310: "f32[1024]", cat_13: "f32[8, 2560, 7, 7]", convolution_104: "f32[8, 1024, 7, 7]", squeeze_313: "f32[1024]", clone: "f32[8, 1024, 1, 1]", le: "b8[8, 1024, 7, 7]", unsqueeze_422: "f32[1, 1024, 1, 1]", le_1: "b8[8, 1024, 7, 7]", unsqueeze_434: "f32[1, 1024, 1, 1]", unsqueeze_446: "f32[1, 512, 1, 1]", unsqueeze_458: "f32[1, 512, 1, 1]", unsqueeze_470: "f32[1, 1024, 1, 1]", unsqueeze_482: "f32[1, 512, 1, 1]", unsqueeze_494: "f32[1, 512, 1, 1]", unsqueeze_506: "f32[1, 1024, 1, 1]", unsqueeze_518: "f32[1, 512, 1, 1]", le_8: "b8[8, 512, 14, 14]", unsqueeze_530: "f32[1, 512, 1, 1]", unsqueeze_542: "f32[1, 256, 1, 1]", unsqueeze_554: "f32[1, 256, 1, 1]", unsqueeze_566: "f32[1, 512, 1, 1]", unsqueeze_578: "f32[1, 256, 1, 1]", unsqueeze_590: "f32[1, 256, 1, 1]", unsqueeze_602: "f32[1, 512, 1, 1]", le_15: "b8[8, 512, 14, 14]", unsqueeze_614: "f32[1, 512, 1, 1]", unsqueeze_626: "f32[1, 256, 1, 1]", unsqueeze_638: "f32[1, 256, 1, 1]", unsqueeze_650: "f32[1, 512, 1, 1]", unsqueeze_662: "f32[1, 256, 1, 1]", unsqueeze_674: "f32[1, 256, 1, 1]", unsqueeze_686: "f32[1, 512, 1, 1]", le_22: "b8[8, 512, 14, 14]", unsqueeze_698: "f32[1, 512, 1, 1]", unsqueeze_710: "f32[1, 256, 1, 1]", unsqueeze_722: "f32[1, 256, 1, 1]", unsqueeze_734: "f32[1, 512, 1, 1]", unsqueeze_746: "f32[1, 256, 1, 1]", unsqueeze_758: "f32[1, 256, 1, 1]", unsqueeze_770: "f32[1, 512, 1, 1]", le_29: "b8[8, 512, 14, 14]", unsqueeze_782: "f32[1, 512, 1, 1]", unsqueeze_794: "f32[1, 256, 1, 1]", unsqueeze_806: "f32[1, 256, 1, 1]", unsqueeze_818: "f32[1, 512, 1, 1]", unsqueeze_830: "f32[1, 256, 1, 1]", unsqueeze_842: "f32[1, 256, 1, 1]", unsqueeze_854: "f32[1, 512, 1, 1]", le_36: "b8[8, 512, 14, 14]", unsqueeze_866: "f32[1, 512, 1, 1]", unsqueeze_878: "f32[1, 256, 1, 1]", unsqueeze_890: "f32[1, 256, 1, 1]", unsqueeze_902: "f32[1, 512, 1, 1]", unsqueeze_914: "f32[1, 256, 1, 1]", unsqueeze_926: "f32[1, 256, 1, 1]", unsqueeze_938: "f32[1, 512, 1, 1]", le_43: "b8[8, 512, 14, 14]", unsqueeze_950: "f32[1, 512, 1, 1]", unsqueeze_962: "f32[1, 256, 1, 1]", unsqueeze_974: "f32[1, 256, 1, 1]", unsqueeze_986: "f32[1, 512, 1, 1]", unsqueeze_998: "f32[1, 256, 1, 1]", unsqueeze_1010: "f32[1, 256, 1, 1]", unsqueeze_1022: "f32[1, 512, 1, 1]", le_50: "b8[8, 512, 14, 14]", unsqueeze_1034: "f32[1, 512, 1, 1]", unsqueeze_1046: "f32[1, 256, 1, 1]", unsqueeze_1058: "f32[1, 256, 1, 1]", unsqueeze_1070: "f32[1, 512, 1, 1]", unsqueeze_1082: "f32[1, 256, 1, 1]", unsqueeze_1094: "f32[1, 256, 1, 1]", unsqueeze_1106: "f32[1, 512, 1, 1]", le_57: "b8[8, 512, 14, 14]", unsqueeze_1118: "f32[1, 512, 1, 1]", unsqueeze_1130: "f32[1, 256, 1, 1]", unsqueeze_1142: "f32[1, 256, 1, 1]", unsqueeze_1154: "f32[1, 512, 1, 1]", unsqueeze_1166: "f32[1, 256, 1, 1]", unsqueeze_1178: "f32[1, 256, 1, 1]", unsqueeze_1190: "f32[1, 512, 1, 1]", unsqueeze_1202: "f32[1, 256, 1, 1]", le_64: "b8[8, 256, 28, 28]", unsqueeze_1214: "f32[1, 256, 1, 1]", unsqueeze_1226: "f32[1, 128, 1, 1]", unsqueeze_1238: "f32[1, 128, 1, 1]", unsqueeze_1250: "f32[1, 256, 1, 1]", unsqueeze_1262: "f32[1, 128, 1, 1]", unsqueeze_1274: "f32[1, 128, 1, 1]", unsqueeze_1286: "f32[1, 256, 1, 1]", le_71: "b8[8, 256, 28, 28]", unsqueeze_1298: "f32[1, 256, 1, 1]", unsqueeze_1310: "f32[1, 128, 1, 1]", unsqueeze_1322: "f32[1, 128, 1, 1]", unsqueeze_1334: "f32[1, 256, 1, 1]", unsqueeze_1346: "f32[1, 128, 1, 1]", unsqueeze_1358: "f32[1, 128, 1, 1]", unsqueeze_1370: "f32[1, 256, 1, 1]", le_78: "b8[8, 256, 28, 28]", unsqueeze_1382: "f32[1, 256, 1, 1]", unsqueeze_1394: "f32[1, 128, 1, 1]", unsqueeze_1406: "f32[1, 128, 1, 1]", unsqueeze_1418: "f32[1, 256, 1, 1]", unsqueeze_1430: "f32[1, 128, 1, 1]", unsqueeze_1442: "f32[1, 128, 1, 1]", unsqueeze_1454: "f32[1, 256, 1, 1]", le_85: "b8[8, 256, 28, 28]", unsqueeze_1466: "f32[1, 256, 1, 1]", unsqueeze_1478: "f32[1, 128, 1, 1]", unsqueeze_1490: "f32[1, 128, 1, 1]", unsqueeze_1502: "f32[1, 256, 1, 1]", unsqueeze_1514: "f32[1, 128, 1, 1]", unsqueeze_1526: "f32[1, 128, 1, 1]", unsqueeze_1538: "f32[1, 256, 1, 1]", unsqueeze_1550: "f32[1, 128, 1, 1]", le_92: "b8[8, 128, 56, 56]", unsqueeze_1562: "f32[1, 128, 1, 1]", unsqueeze_1574: "f32[1, 64, 1, 1]", unsqueeze_1586: "f32[1, 64, 1, 1]", unsqueeze_1598: "f32[1, 128, 1, 1]", unsqueeze_1610: "f32[1, 64, 1, 1]", unsqueeze_1622: "f32[1, 64, 1, 1]", unsqueeze_1634: "f32[1, 128, 1, 1]", unsqueeze_1646: "f32[1, 32, 1, 1]", unsqueeze_1658: "f32[1, 16, 1, 1]", unsqueeze_1670: "f32[1, 16, 1, 1]", tangents_1: "f32[8, 1000]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:378, code: return self.flatten(x)
    view_1: "f32[8, 1000, 1, 1]" = torch.ops.aten.reshape.default(tangents_1, [8, 1000, 1, 1]);  tangents_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:377, code: x = self.fc(x)
    convolution_backward = torch.ops.aten.convolution_backward.default(view_1, clone, primals_316, [1000], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  view_1 = clone = primals_316 = None
    getitem_228: "f32[8, 1024, 1, 1]" = convolution_backward[0]
    getitem_229: "f32[1000, 1024, 1, 1]" = convolution_backward[1]
    getitem_230: "f32[1000]" = convolution_backward[2];  convolution_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    expand: "f32[8, 1024, 7, 7]" = torch.ops.aten.expand.default(getitem_228, [8, 1024, 7, 7]);  getitem_228 = None
    div: "f32[8, 1024, 7, 7]" = torch.ops.aten.div.Scalar(expand, 49);  expand = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:180, code: x = self.relu(x)
    full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    where: "f32[8, 1024, 7, 7]" = torch.ops.aten.where.self(le, full_default, div);  le = div = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:177, code: x = self.bn(x)
    sum_1: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where, [0, 2, 3])
    sub_105: "f32[8, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_104, unsqueeze_422);  convolution_104 = unsqueeze_422 = None
    mul_735: "f32[8, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(where, sub_105)
    sum_2: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_735, [0, 2, 3]);  mul_735 = None
    mul_736: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_1, 0.002551020408163265)
    unsqueeze_423: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_736, 0);  mul_736 = None
    unsqueeze_424: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_423, 2);  unsqueeze_423 = None
    unsqueeze_425: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_424, 3);  unsqueeze_424 = None
    mul_737: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_2, 0.002551020408163265)
    mul_738: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_313, squeeze_313)
    mul_739: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_737, mul_738);  mul_737 = mul_738 = None
    unsqueeze_426: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_739, 0);  mul_739 = None
    unsqueeze_427: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_426, 2);  unsqueeze_426 = None
    unsqueeze_428: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_427, 3);  unsqueeze_427 = None
    mul_740: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_313, primals_314);  primals_314 = None
    unsqueeze_429: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_740, 0);  mul_740 = None
    unsqueeze_430: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_429, 2);  unsqueeze_429 = None
    unsqueeze_431: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_430, 3);  unsqueeze_430 = None
    mul_741: "f32[8, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(sub_105, unsqueeze_428);  sub_105 = unsqueeze_428 = None
    sub_107: "f32[8, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(where, mul_741);  mul_741 = None
    sub_108: "f32[8, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(sub_107, unsqueeze_425);  sub_107 = unsqueeze_425 = None
    mul_742: "f32[8, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(sub_108, unsqueeze_431);  sub_108 = unsqueeze_431 = None
    mul_743: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_2, squeeze_313);  sum_2 = squeeze_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:176, code: x = self.conv(torch.cat(x_children, 1))
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(mul_742, cat_13, primals_313, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_742 = cat_13 = primals_313 = None
    getitem_231: "f32[8, 2560, 7, 7]" = convolution_backward_1[0]
    getitem_232: "f32[1024, 2560, 1, 1]" = convolution_backward_1[1];  convolution_backward_1 = None
    slice_1: "f32[8, 1024, 7, 7]" = torch.ops.aten.slice.Tensor(getitem_231, 1, 0, 1024)
    slice_2: "f32[8, 1024, 7, 7]" = torch.ops.aten.slice.Tensor(getitem_231, 1, 1024, 2048)
    slice_3: "f32[8, 512, 7, 7]" = torch.ops.aten.slice.Tensor(getitem_231, 1, 2048, 2560);  getitem_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:176, code: x = self.conv(torch.cat(x_children, 1))
    add_567: "f32[8, 1024, 7, 7]" = torch.ops.aten.add.Tensor(where, slice_1);  where = slice_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    where_1: "f32[8, 1024, 7, 7]" = torch.ops.aten.where.self(le_1, full_default, add_567);  le_1 = add_567 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    add_568: "f32[8, 1024, 7, 7]" = torch.ops.aten.add.Tensor(slice_2, where_1);  slice_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    sum_3: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_1, [0, 2, 3])
    sub_109: "f32[8, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_103, unsqueeze_434);  convolution_103 = unsqueeze_434 = None
    mul_744: "f32[8, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(where_1, sub_109)
    sum_4: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_744, [0, 2, 3]);  mul_744 = None
    mul_745: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_3, 0.002551020408163265)
    unsqueeze_435: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_745, 0);  mul_745 = None
    unsqueeze_436: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_435, 2);  unsqueeze_435 = None
    unsqueeze_437: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_436, 3);  unsqueeze_436 = None
    mul_746: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_4, 0.002551020408163265)
    mul_747: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_310, squeeze_310)
    mul_748: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_746, mul_747);  mul_746 = mul_747 = None
    unsqueeze_438: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_748, 0);  mul_748 = None
    unsqueeze_439: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_438, 2);  unsqueeze_438 = None
    unsqueeze_440: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_439, 3);  unsqueeze_439 = None
    mul_749: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_310, primals_311);  primals_311 = None
    unsqueeze_441: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_749, 0);  mul_749 = None
    unsqueeze_442: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_441, 2);  unsqueeze_441 = None
    unsqueeze_443: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_442, 3);  unsqueeze_442 = None
    mul_750: "f32[8, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(sub_109, unsqueeze_440);  sub_109 = unsqueeze_440 = None
    sub_111: "f32[8, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(where_1, mul_750);  where_1 = mul_750 = None
    sub_112: "f32[8, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(sub_111, unsqueeze_437);  sub_111 = unsqueeze_437 = None
    mul_751: "f32[8, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(sub_112, unsqueeze_443);  sub_112 = unsqueeze_443 = None
    mul_752: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_4, squeeze_310);  sum_4 = squeeze_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(mul_751, relu_98, primals_310, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_751 = primals_310 = None
    getitem_234: "f32[8, 512, 7, 7]" = convolution_backward_2[0]
    getitem_235: "f32[1024, 512, 1, 1]" = convolution_backward_2[1];  convolution_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    le_2: "b8[8, 512, 7, 7]" = torch.ops.aten.le.Scalar(relu_98, 0);  relu_98 = None
    where_2: "f32[8, 512, 7, 7]" = torch.ops.aten.where.self(le_2, full_default, getitem_234);  le_2 = getitem_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    sum_5: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_2, [0, 2, 3])
    sub_113: "f32[8, 512, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_102, unsqueeze_446);  convolution_102 = unsqueeze_446 = None
    mul_753: "f32[8, 512, 7, 7]" = torch.ops.aten.mul.Tensor(where_2, sub_113)
    sum_6: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_753, [0, 2, 3]);  mul_753 = None
    mul_754: "f32[512]" = torch.ops.aten.mul.Tensor(sum_5, 0.002551020408163265)
    unsqueeze_447: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_754, 0);  mul_754 = None
    unsqueeze_448: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_447, 2);  unsqueeze_447 = None
    unsqueeze_449: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_448, 3);  unsqueeze_448 = None
    mul_755: "f32[512]" = torch.ops.aten.mul.Tensor(sum_6, 0.002551020408163265)
    mul_756: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_307, squeeze_307)
    mul_757: "f32[512]" = torch.ops.aten.mul.Tensor(mul_755, mul_756);  mul_755 = mul_756 = None
    unsqueeze_450: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_757, 0);  mul_757 = None
    unsqueeze_451: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_450, 2);  unsqueeze_450 = None
    unsqueeze_452: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_451, 3);  unsqueeze_451 = None
    mul_758: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_307, primals_308);  primals_308 = None
    unsqueeze_453: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_758, 0);  mul_758 = None
    unsqueeze_454: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_453, 2);  unsqueeze_453 = None
    unsqueeze_455: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_454, 3);  unsqueeze_454 = None
    mul_759: "f32[8, 512, 7, 7]" = torch.ops.aten.mul.Tensor(sub_113, unsqueeze_452);  sub_113 = unsqueeze_452 = None
    sub_115: "f32[8, 512, 7, 7]" = torch.ops.aten.sub.Tensor(where_2, mul_759);  where_2 = mul_759 = None
    sub_116: "f32[8, 512, 7, 7]" = torch.ops.aten.sub.Tensor(sub_115, unsqueeze_449);  sub_115 = unsqueeze_449 = None
    mul_760: "f32[8, 512, 7, 7]" = torch.ops.aten.mul.Tensor(sub_116, unsqueeze_455);  sub_116 = unsqueeze_455 = None
    mul_761: "f32[512]" = torch.ops.aten.mul.Tensor(sum_6, squeeze_307);  sum_6 = squeeze_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    convolution_backward_3 = torch.ops.aten.convolution_backward.default(mul_760, relu_97, primals_307, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_760 = primals_307 = None
    getitem_237: "f32[8, 512, 7, 7]" = convolution_backward_3[0]
    getitem_238: "f32[512, 512, 3, 3]" = convolution_backward_3[1];  convolution_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    le_3: "b8[8, 512, 7, 7]" = torch.ops.aten.le.Scalar(relu_97, 0);  relu_97 = None
    where_3: "f32[8, 512, 7, 7]" = torch.ops.aten.where.self(le_3, full_default, getitem_237);  le_3 = getitem_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    sum_7: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_3, [0, 2, 3])
    sub_117: "f32[8, 512, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_101, unsqueeze_458);  convolution_101 = unsqueeze_458 = None
    mul_762: "f32[8, 512, 7, 7]" = torch.ops.aten.mul.Tensor(where_3, sub_117)
    sum_8: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_762, [0, 2, 3]);  mul_762 = None
    mul_763: "f32[512]" = torch.ops.aten.mul.Tensor(sum_7, 0.002551020408163265)
    unsqueeze_459: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_763, 0);  mul_763 = None
    unsqueeze_460: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_459, 2);  unsqueeze_459 = None
    unsqueeze_461: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_460, 3);  unsqueeze_460 = None
    mul_764: "f32[512]" = torch.ops.aten.mul.Tensor(sum_8, 0.002551020408163265)
    mul_765: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_304, squeeze_304)
    mul_766: "f32[512]" = torch.ops.aten.mul.Tensor(mul_764, mul_765);  mul_764 = mul_765 = None
    unsqueeze_462: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_766, 0);  mul_766 = None
    unsqueeze_463: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_462, 2);  unsqueeze_462 = None
    unsqueeze_464: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_463, 3);  unsqueeze_463 = None
    mul_767: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_304, primals_305);  primals_305 = None
    unsqueeze_465: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_767, 0);  mul_767 = None
    unsqueeze_466: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_465, 2);  unsqueeze_465 = None
    unsqueeze_467: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_466, 3);  unsqueeze_466 = None
    mul_768: "f32[8, 512, 7, 7]" = torch.ops.aten.mul.Tensor(sub_117, unsqueeze_464);  sub_117 = unsqueeze_464 = None
    sub_119: "f32[8, 512, 7, 7]" = torch.ops.aten.sub.Tensor(where_3, mul_768);  where_3 = mul_768 = None
    sub_120: "f32[8, 512, 7, 7]" = torch.ops.aten.sub.Tensor(sub_119, unsqueeze_461);  sub_119 = unsqueeze_461 = None
    mul_769: "f32[8, 512, 7, 7]" = torch.ops.aten.mul.Tensor(sub_120, unsqueeze_467);  sub_120 = unsqueeze_467 = None
    mul_770: "f32[512]" = torch.ops.aten.mul.Tensor(sum_8, squeeze_304);  sum_8 = squeeze_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    convolution_backward_4 = torch.ops.aten.convolution_backward.default(mul_769, relu_96, primals_304, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_769 = primals_304 = None
    getitem_240: "f32[8, 1024, 7, 7]" = convolution_backward_4[0]
    getitem_241: "f32[512, 1024, 1, 1]" = convolution_backward_4[1];  convolution_backward_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    add_569: "f32[8, 1024, 7, 7]" = torch.ops.aten.add.Tensor(add_568, getitem_240);  add_568 = getitem_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    le_4: "b8[8, 1024, 7, 7]" = torch.ops.aten.le.Scalar(relu_96, 0);  relu_96 = None
    where_4: "f32[8, 1024, 7, 7]" = torch.ops.aten.where.self(le_4, full_default, add_569);  le_4 = add_569 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    sum_9: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_4, [0, 2, 3])
    sub_121: "f32[8, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_100, unsqueeze_470);  convolution_100 = unsqueeze_470 = None
    mul_771: "f32[8, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(where_4, sub_121)
    sum_10: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_771, [0, 2, 3]);  mul_771 = None
    mul_772: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_9, 0.002551020408163265)
    unsqueeze_471: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_772, 0);  mul_772 = None
    unsqueeze_472: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_471, 2);  unsqueeze_471 = None
    unsqueeze_473: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_472, 3);  unsqueeze_472 = None
    mul_773: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_10, 0.002551020408163265)
    mul_774: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_301, squeeze_301)
    mul_775: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_773, mul_774);  mul_773 = mul_774 = None
    unsqueeze_474: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_775, 0);  mul_775 = None
    unsqueeze_475: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_474, 2);  unsqueeze_474 = None
    unsqueeze_476: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_475, 3);  unsqueeze_475 = None
    mul_776: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_301, primals_302);  primals_302 = None
    unsqueeze_477: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_776, 0);  mul_776 = None
    unsqueeze_478: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_477, 2);  unsqueeze_477 = None
    unsqueeze_479: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_478, 3);  unsqueeze_478 = None
    mul_777: "f32[8, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(sub_121, unsqueeze_476);  sub_121 = unsqueeze_476 = None
    sub_123: "f32[8, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(where_4, mul_777);  mul_777 = None
    sub_124: "f32[8, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(sub_123, unsqueeze_473);  sub_123 = None
    mul_778: "f32[8, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(sub_124, unsqueeze_479);  sub_124 = unsqueeze_479 = None
    mul_779: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_10, squeeze_301);  sum_10 = squeeze_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    convolution_backward_5 = torch.ops.aten.convolution_backward.default(mul_778, relu_95, primals_301, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_778 = primals_301 = None
    getitem_243: "f32[8, 512, 7, 7]" = convolution_backward_5[0]
    getitem_244: "f32[1024, 512, 1, 1]" = convolution_backward_5[1];  convolution_backward_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    le_5: "b8[8, 512, 7, 7]" = torch.ops.aten.le.Scalar(relu_95, 0);  relu_95 = None
    where_5: "f32[8, 512, 7, 7]" = torch.ops.aten.where.self(le_5, full_default, getitem_243);  le_5 = getitem_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    sum_11: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_5, [0, 2, 3])
    sub_125: "f32[8, 512, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_99, unsqueeze_482);  convolution_99 = unsqueeze_482 = None
    mul_780: "f32[8, 512, 7, 7]" = torch.ops.aten.mul.Tensor(where_5, sub_125)
    sum_12: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_780, [0, 2, 3]);  mul_780 = None
    mul_781: "f32[512]" = torch.ops.aten.mul.Tensor(sum_11, 0.002551020408163265)
    unsqueeze_483: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_781, 0);  mul_781 = None
    unsqueeze_484: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_483, 2);  unsqueeze_483 = None
    unsqueeze_485: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_484, 3);  unsqueeze_484 = None
    mul_782: "f32[512]" = torch.ops.aten.mul.Tensor(sum_12, 0.002551020408163265)
    mul_783: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_298, squeeze_298)
    mul_784: "f32[512]" = torch.ops.aten.mul.Tensor(mul_782, mul_783);  mul_782 = mul_783 = None
    unsqueeze_486: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_784, 0);  mul_784 = None
    unsqueeze_487: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_486, 2);  unsqueeze_486 = None
    unsqueeze_488: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_487, 3);  unsqueeze_487 = None
    mul_785: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_298, primals_299);  primals_299 = None
    unsqueeze_489: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_785, 0);  mul_785 = None
    unsqueeze_490: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_489, 2);  unsqueeze_489 = None
    unsqueeze_491: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_490, 3);  unsqueeze_490 = None
    mul_786: "f32[8, 512, 7, 7]" = torch.ops.aten.mul.Tensor(sub_125, unsqueeze_488);  sub_125 = unsqueeze_488 = None
    sub_127: "f32[8, 512, 7, 7]" = torch.ops.aten.sub.Tensor(where_5, mul_786);  where_5 = mul_786 = None
    sub_128: "f32[8, 512, 7, 7]" = torch.ops.aten.sub.Tensor(sub_127, unsqueeze_485);  sub_127 = unsqueeze_485 = None
    mul_787: "f32[8, 512, 7, 7]" = torch.ops.aten.mul.Tensor(sub_128, unsqueeze_491);  sub_128 = unsqueeze_491 = None
    mul_788: "f32[512]" = torch.ops.aten.mul.Tensor(sum_12, squeeze_298);  sum_12 = squeeze_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    convolution_backward_6 = torch.ops.aten.convolution_backward.default(mul_787, relu_94, primals_298, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_787 = primals_298 = None
    getitem_246: "f32[8, 512, 14, 14]" = convolution_backward_6[0]
    getitem_247: "f32[512, 512, 3, 3]" = convolution_backward_6[1];  convolution_backward_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    le_6: "b8[8, 512, 14, 14]" = torch.ops.aten.le.Scalar(relu_94, 0);  relu_94 = None
    where_6: "f32[8, 512, 14, 14]" = torch.ops.aten.where.self(le_6, full_default, getitem_246);  le_6 = getitem_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    sum_13: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_6, [0, 2, 3])
    sub_129: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_98, unsqueeze_494);  convolution_98 = unsqueeze_494 = None
    mul_789: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(where_6, sub_129)
    sum_14: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_789, [0, 2, 3]);  mul_789 = None
    mul_790: "f32[512]" = torch.ops.aten.mul.Tensor(sum_13, 0.0006377551020408163)
    unsqueeze_495: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_790, 0);  mul_790 = None
    unsqueeze_496: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_495, 2);  unsqueeze_495 = None
    unsqueeze_497: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_496, 3);  unsqueeze_496 = None
    mul_791: "f32[512]" = torch.ops.aten.mul.Tensor(sum_14, 0.0006377551020408163)
    mul_792: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_295, squeeze_295)
    mul_793: "f32[512]" = torch.ops.aten.mul.Tensor(mul_791, mul_792);  mul_791 = mul_792 = None
    unsqueeze_498: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_793, 0);  mul_793 = None
    unsqueeze_499: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_498, 2);  unsqueeze_498 = None
    unsqueeze_500: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_499, 3);  unsqueeze_499 = None
    mul_794: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_295, primals_296);  primals_296 = None
    unsqueeze_501: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_794, 0);  mul_794 = None
    unsqueeze_502: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_501, 2);  unsqueeze_501 = None
    unsqueeze_503: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_502, 3);  unsqueeze_502 = None
    mul_795: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_129, unsqueeze_500);  sub_129 = unsqueeze_500 = None
    sub_131: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(where_6, mul_795);  where_6 = mul_795 = None
    sub_132: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(sub_131, unsqueeze_497);  sub_131 = unsqueeze_497 = None
    mul_796: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_132, unsqueeze_503);  sub_132 = unsqueeze_503 = None
    mul_797: "f32[512]" = torch.ops.aten.mul.Tensor(sum_14, squeeze_295);  sum_14 = squeeze_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    convolution_backward_7 = torch.ops.aten.convolution_backward.default(mul_796, relu_93, primals_295, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_796 = primals_295 = None
    getitem_249: "f32[8, 512, 14, 14]" = convolution_backward_7[0]
    getitem_250: "f32[512, 512, 1, 1]" = convolution_backward_7[1];  convolution_backward_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:248, code: shortcut = self.project(bottom)
    sub_133: "f32[8, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_97, unsqueeze_506);  convolution_97 = unsqueeze_506 = None
    mul_798: "f32[8, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(where_4, sub_133)
    sum_16: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_798, [0, 2, 3]);  mul_798 = None
    mul_800: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_16, 0.002551020408163265)
    mul_801: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_292, squeeze_292)
    mul_802: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_800, mul_801);  mul_800 = mul_801 = None
    unsqueeze_510: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_802, 0);  mul_802 = None
    unsqueeze_511: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_510, 2);  unsqueeze_510 = None
    unsqueeze_512: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_511, 3);  unsqueeze_511 = None
    mul_803: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_292, primals_293);  primals_293 = None
    unsqueeze_513: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_803, 0);  mul_803 = None
    unsqueeze_514: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_513, 2);  unsqueeze_513 = None
    unsqueeze_515: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_514, 3);  unsqueeze_514 = None
    mul_804: "f32[8, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(sub_133, unsqueeze_512);  sub_133 = unsqueeze_512 = None
    sub_135: "f32[8, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(where_4, mul_804);  where_4 = mul_804 = None
    sub_136: "f32[8, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(sub_135, unsqueeze_473);  sub_135 = unsqueeze_473 = None
    mul_805: "f32[8, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(sub_136, unsqueeze_515);  sub_136 = unsqueeze_515 = None
    mul_806: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_16, squeeze_292);  sum_16 = squeeze_292 = None
    convolution_backward_8 = torch.ops.aten.convolution_backward.default(mul_805, getitem_210, primals_292, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_805 = getitem_210 = primals_292 = None
    getitem_252: "f32[8, 512, 7, 7]" = convolution_backward_8[0]
    getitem_253: "f32[1024, 512, 1, 1]" = convolution_backward_8[1];  convolution_backward_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:248, code: shortcut = self.project(bottom)
    add_570: "f32[8, 512, 7, 7]" = torch.ops.aten.add.Tensor(slice_3, getitem_252);  slice_3 = getitem_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:247, code: bottom = self.downsample(x)
    max_pool2d_with_indices_backward: "f32[8, 512, 14, 14]" = torch.ops.aten.max_pool2d_with_indices_backward.default(add_570, relu_93, [2, 2], [2, 2], [0, 0], [1, 1], False, getitem_211);  add_570 = getitem_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:247, code: bottom = self.downsample(x)
    add_571: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(getitem_249, max_pool2d_with_indices_backward);  getitem_249 = max_pool2d_with_indices_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:180, code: x = self.relu(x)
    le_7: "b8[8, 512, 14, 14]" = torch.ops.aten.le.Scalar(relu_93, 0);  relu_93 = None
    where_7: "f32[8, 512, 14, 14]" = torch.ops.aten.where.self(le_7, full_default, add_571);  le_7 = add_571 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:177, code: x = self.bn(x)
    sum_17: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_7, [0, 2, 3])
    sub_137: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_96, unsqueeze_518);  convolution_96 = unsqueeze_518 = None
    mul_807: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(where_7, sub_137)
    sum_18: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_807, [0, 2, 3]);  mul_807 = None
    mul_808: "f32[512]" = torch.ops.aten.mul.Tensor(sum_17, 0.0006377551020408163)
    unsqueeze_519: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_808, 0);  mul_808 = None
    unsqueeze_520: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_519, 2);  unsqueeze_519 = None
    unsqueeze_521: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_520, 3);  unsqueeze_520 = None
    mul_809: "f32[512]" = torch.ops.aten.mul.Tensor(sum_18, 0.0006377551020408163)
    mul_810: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_289, squeeze_289)
    mul_811: "f32[512]" = torch.ops.aten.mul.Tensor(mul_809, mul_810);  mul_809 = mul_810 = None
    unsqueeze_522: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_811, 0);  mul_811 = None
    unsqueeze_523: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_522, 2);  unsqueeze_522 = None
    unsqueeze_524: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_523, 3);  unsqueeze_523 = None
    mul_812: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_289, primals_290);  primals_290 = None
    unsqueeze_525: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_812, 0);  mul_812 = None
    unsqueeze_526: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_525, 2);  unsqueeze_525 = None
    unsqueeze_527: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_526, 3);  unsqueeze_526 = None
    mul_813: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_137, unsqueeze_524);  sub_137 = unsqueeze_524 = None
    sub_139: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(where_7, mul_813);  mul_813 = None
    sub_140: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(sub_139, unsqueeze_521);  sub_139 = unsqueeze_521 = None
    mul_814: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_140, unsqueeze_527);  sub_140 = unsqueeze_527 = None
    mul_815: "f32[512]" = torch.ops.aten.mul.Tensor(sum_18, squeeze_289);  sum_18 = squeeze_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:176, code: x = self.conv(torch.cat(x_children, 1))
    convolution_backward_9 = torch.ops.aten.convolution_backward.default(mul_814, cat_12, primals_289, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_814 = cat_12 = primals_289 = None
    getitem_255: "f32[8, 2816, 14, 14]" = convolution_backward_9[0]
    getitem_256: "f32[512, 2816, 1, 1]" = convolution_backward_9[1];  convolution_backward_9 = None
    slice_4: "f32[8, 512, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_255, 1, 0, 512)
    slice_5: "f32[8, 512, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_255, 1, 512, 1024)
    slice_6: "f32[8, 256, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_255, 1, 1024, 1280)
    slice_7: "f32[8, 512, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_255, 1, 1280, 1792)
    slice_8: "f32[8, 512, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_255, 1, 1792, 2304)
    slice_9: "f32[8, 512, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_255, 1, 2304, 2816);  getitem_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:176, code: x = self.conv(torch.cat(x_children, 1))
    add_572: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(where_7, slice_4);  where_7 = slice_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    where_8: "f32[8, 512, 14, 14]" = torch.ops.aten.where.self(le_8, full_default, add_572);  le_8 = add_572 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    add_573: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(slice_5, where_8);  slice_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    sum_19: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_8, [0, 2, 3])
    sub_141: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_95, unsqueeze_530);  convolution_95 = unsqueeze_530 = None
    mul_816: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(where_8, sub_141)
    sum_20: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_816, [0, 2, 3]);  mul_816 = None
    mul_817: "f32[512]" = torch.ops.aten.mul.Tensor(sum_19, 0.0006377551020408163)
    unsqueeze_531: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_817, 0);  mul_817 = None
    unsqueeze_532: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_531, 2);  unsqueeze_531 = None
    unsqueeze_533: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_532, 3);  unsqueeze_532 = None
    mul_818: "f32[512]" = torch.ops.aten.mul.Tensor(sum_20, 0.0006377551020408163)
    mul_819: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_286, squeeze_286)
    mul_820: "f32[512]" = torch.ops.aten.mul.Tensor(mul_818, mul_819);  mul_818 = mul_819 = None
    unsqueeze_534: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_820, 0);  mul_820 = None
    unsqueeze_535: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_534, 2);  unsqueeze_534 = None
    unsqueeze_536: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_535, 3);  unsqueeze_535 = None
    mul_821: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_286, primals_287);  primals_287 = None
    unsqueeze_537: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_821, 0);  mul_821 = None
    unsqueeze_538: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_537, 2);  unsqueeze_537 = None
    unsqueeze_539: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_538, 3);  unsqueeze_538 = None
    mul_822: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_141, unsqueeze_536);  sub_141 = unsqueeze_536 = None
    sub_143: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(where_8, mul_822);  where_8 = mul_822 = None
    sub_144: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(sub_143, unsqueeze_533);  sub_143 = unsqueeze_533 = None
    mul_823: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_144, unsqueeze_539);  sub_144 = unsqueeze_539 = None
    mul_824: "f32[512]" = torch.ops.aten.mul.Tensor(sum_20, squeeze_286);  sum_20 = squeeze_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    convolution_backward_10 = torch.ops.aten.convolution_backward.default(mul_823, relu_91, primals_286, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_823 = primals_286 = None
    getitem_258: "f32[8, 256, 14, 14]" = convolution_backward_10[0]
    getitem_259: "f32[512, 256, 1, 1]" = convolution_backward_10[1];  convolution_backward_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    le_9: "b8[8, 256, 14, 14]" = torch.ops.aten.le.Scalar(relu_91, 0);  relu_91 = None
    where_9: "f32[8, 256, 14, 14]" = torch.ops.aten.where.self(le_9, full_default, getitem_258);  le_9 = getitem_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    sum_21: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_9, [0, 2, 3])
    sub_145: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_94, unsqueeze_542);  convolution_94 = unsqueeze_542 = None
    mul_825: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_9, sub_145)
    sum_22: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_825, [0, 2, 3]);  mul_825 = None
    mul_826: "f32[256]" = torch.ops.aten.mul.Tensor(sum_21, 0.0006377551020408163)
    unsqueeze_543: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_826, 0);  mul_826 = None
    unsqueeze_544: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_543, 2);  unsqueeze_543 = None
    unsqueeze_545: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_544, 3);  unsqueeze_544 = None
    mul_827: "f32[256]" = torch.ops.aten.mul.Tensor(sum_22, 0.0006377551020408163)
    mul_828: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_283, squeeze_283)
    mul_829: "f32[256]" = torch.ops.aten.mul.Tensor(mul_827, mul_828);  mul_827 = mul_828 = None
    unsqueeze_546: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_829, 0);  mul_829 = None
    unsqueeze_547: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_546, 2);  unsqueeze_546 = None
    unsqueeze_548: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_547, 3);  unsqueeze_547 = None
    mul_830: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_283, primals_284);  primals_284 = None
    unsqueeze_549: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_830, 0);  mul_830 = None
    unsqueeze_550: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_549, 2);  unsqueeze_549 = None
    unsqueeze_551: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_550, 3);  unsqueeze_550 = None
    mul_831: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_145, unsqueeze_548);  sub_145 = unsqueeze_548 = None
    sub_147: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(where_9, mul_831);  where_9 = mul_831 = None
    sub_148: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(sub_147, unsqueeze_545);  sub_147 = unsqueeze_545 = None
    mul_832: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_148, unsqueeze_551);  sub_148 = unsqueeze_551 = None
    mul_833: "f32[256]" = torch.ops.aten.mul.Tensor(sum_22, squeeze_283);  sum_22 = squeeze_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    convolution_backward_11 = torch.ops.aten.convolution_backward.default(mul_832, relu_90, primals_283, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_832 = primals_283 = None
    getitem_261: "f32[8, 256, 14, 14]" = convolution_backward_11[0]
    getitem_262: "f32[256, 256, 3, 3]" = convolution_backward_11[1];  convolution_backward_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    le_10: "b8[8, 256, 14, 14]" = torch.ops.aten.le.Scalar(relu_90, 0);  relu_90 = None
    where_10: "f32[8, 256, 14, 14]" = torch.ops.aten.where.self(le_10, full_default, getitem_261);  le_10 = getitem_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    sum_23: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_10, [0, 2, 3])
    sub_149: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_93, unsqueeze_554);  convolution_93 = unsqueeze_554 = None
    mul_834: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_10, sub_149)
    sum_24: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_834, [0, 2, 3]);  mul_834 = None
    mul_835: "f32[256]" = torch.ops.aten.mul.Tensor(sum_23, 0.0006377551020408163)
    unsqueeze_555: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_835, 0);  mul_835 = None
    unsqueeze_556: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_555, 2);  unsqueeze_555 = None
    unsqueeze_557: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_556, 3);  unsqueeze_556 = None
    mul_836: "f32[256]" = torch.ops.aten.mul.Tensor(sum_24, 0.0006377551020408163)
    mul_837: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_280, squeeze_280)
    mul_838: "f32[256]" = torch.ops.aten.mul.Tensor(mul_836, mul_837);  mul_836 = mul_837 = None
    unsqueeze_558: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_838, 0);  mul_838 = None
    unsqueeze_559: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_558, 2);  unsqueeze_558 = None
    unsqueeze_560: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_559, 3);  unsqueeze_559 = None
    mul_839: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_280, primals_281);  primals_281 = None
    unsqueeze_561: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_839, 0);  mul_839 = None
    unsqueeze_562: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_561, 2);  unsqueeze_561 = None
    unsqueeze_563: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_562, 3);  unsqueeze_562 = None
    mul_840: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_149, unsqueeze_560);  sub_149 = unsqueeze_560 = None
    sub_151: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(where_10, mul_840);  where_10 = mul_840 = None
    sub_152: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(sub_151, unsqueeze_557);  sub_151 = unsqueeze_557 = None
    mul_841: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_152, unsqueeze_563);  sub_152 = unsqueeze_563 = None
    mul_842: "f32[256]" = torch.ops.aten.mul.Tensor(sum_24, squeeze_280);  sum_24 = squeeze_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    convolution_backward_12 = torch.ops.aten.convolution_backward.default(mul_841, relu_89, primals_280, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_841 = primals_280 = None
    getitem_264: "f32[8, 512, 14, 14]" = convolution_backward_12[0]
    getitem_265: "f32[256, 512, 1, 1]" = convolution_backward_12[1];  convolution_backward_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    add_574: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_573, getitem_264);  add_573 = getitem_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    le_11: "b8[8, 512, 14, 14]" = torch.ops.aten.le.Scalar(relu_89, 0);  relu_89 = None
    where_11: "f32[8, 512, 14, 14]" = torch.ops.aten.where.self(le_11, full_default, add_574);  le_11 = add_574 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    add_575: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(slice_9, where_11);  slice_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    sum_25: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_11, [0, 2, 3])
    sub_153: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_92, unsqueeze_566);  convolution_92 = unsqueeze_566 = None
    mul_843: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(where_11, sub_153)
    sum_26: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_843, [0, 2, 3]);  mul_843 = None
    mul_844: "f32[512]" = torch.ops.aten.mul.Tensor(sum_25, 0.0006377551020408163)
    unsqueeze_567: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_844, 0);  mul_844 = None
    unsqueeze_568: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_567, 2);  unsqueeze_567 = None
    unsqueeze_569: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_568, 3);  unsqueeze_568 = None
    mul_845: "f32[512]" = torch.ops.aten.mul.Tensor(sum_26, 0.0006377551020408163)
    mul_846: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_277, squeeze_277)
    mul_847: "f32[512]" = torch.ops.aten.mul.Tensor(mul_845, mul_846);  mul_845 = mul_846 = None
    unsqueeze_570: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_847, 0);  mul_847 = None
    unsqueeze_571: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_570, 2);  unsqueeze_570 = None
    unsqueeze_572: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_571, 3);  unsqueeze_571 = None
    mul_848: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_277, primals_278);  primals_278 = None
    unsqueeze_573: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_848, 0);  mul_848 = None
    unsqueeze_574: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_573, 2);  unsqueeze_573 = None
    unsqueeze_575: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_574, 3);  unsqueeze_574 = None
    mul_849: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_153, unsqueeze_572);  sub_153 = unsqueeze_572 = None
    sub_155: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(where_11, mul_849);  where_11 = mul_849 = None
    sub_156: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(sub_155, unsqueeze_569);  sub_155 = unsqueeze_569 = None
    mul_850: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_156, unsqueeze_575);  sub_156 = unsqueeze_575 = None
    mul_851: "f32[512]" = torch.ops.aten.mul.Tensor(sum_26, squeeze_277);  sum_26 = squeeze_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    convolution_backward_13 = torch.ops.aten.convolution_backward.default(mul_850, relu_88, primals_277, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_850 = primals_277 = None
    getitem_267: "f32[8, 256, 14, 14]" = convolution_backward_13[0]
    getitem_268: "f32[512, 256, 1, 1]" = convolution_backward_13[1];  convolution_backward_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    le_12: "b8[8, 256, 14, 14]" = torch.ops.aten.le.Scalar(relu_88, 0);  relu_88 = None
    where_12: "f32[8, 256, 14, 14]" = torch.ops.aten.where.self(le_12, full_default, getitem_267);  le_12 = getitem_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    sum_27: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_12, [0, 2, 3])
    sub_157: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_91, unsqueeze_578);  convolution_91 = unsqueeze_578 = None
    mul_852: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_12, sub_157)
    sum_28: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_852, [0, 2, 3]);  mul_852 = None
    mul_853: "f32[256]" = torch.ops.aten.mul.Tensor(sum_27, 0.0006377551020408163)
    unsqueeze_579: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_853, 0);  mul_853 = None
    unsqueeze_580: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_579, 2);  unsqueeze_579 = None
    unsqueeze_581: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_580, 3);  unsqueeze_580 = None
    mul_854: "f32[256]" = torch.ops.aten.mul.Tensor(sum_28, 0.0006377551020408163)
    mul_855: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_274, squeeze_274)
    mul_856: "f32[256]" = torch.ops.aten.mul.Tensor(mul_854, mul_855);  mul_854 = mul_855 = None
    unsqueeze_582: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_856, 0);  mul_856 = None
    unsqueeze_583: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_582, 2);  unsqueeze_582 = None
    unsqueeze_584: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_583, 3);  unsqueeze_583 = None
    mul_857: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_274, primals_275);  primals_275 = None
    unsqueeze_585: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_857, 0);  mul_857 = None
    unsqueeze_586: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_585, 2);  unsqueeze_585 = None
    unsqueeze_587: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_586, 3);  unsqueeze_586 = None
    mul_858: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_157, unsqueeze_584);  sub_157 = unsqueeze_584 = None
    sub_159: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(where_12, mul_858);  where_12 = mul_858 = None
    sub_160: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(sub_159, unsqueeze_581);  sub_159 = unsqueeze_581 = None
    mul_859: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_160, unsqueeze_587);  sub_160 = unsqueeze_587 = None
    mul_860: "f32[256]" = torch.ops.aten.mul.Tensor(sum_28, squeeze_274);  sum_28 = squeeze_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    convolution_backward_14 = torch.ops.aten.convolution_backward.default(mul_859, relu_87, primals_274, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_859 = primals_274 = None
    getitem_270: "f32[8, 256, 14, 14]" = convolution_backward_14[0]
    getitem_271: "f32[256, 256, 3, 3]" = convolution_backward_14[1];  convolution_backward_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    le_13: "b8[8, 256, 14, 14]" = torch.ops.aten.le.Scalar(relu_87, 0);  relu_87 = None
    where_13: "f32[8, 256, 14, 14]" = torch.ops.aten.where.self(le_13, full_default, getitem_270);  le_13 = getitem_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    sum_29: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_13, [0, 2, 3])
    sub_161: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_90, unsqueeze_590);  convolution_90 = unsqueeze_590 = None
    mul_861: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_13, sub_161)
    sum_30: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_861, [0, 2, 3]);  mul_861 = None
    mul_862: "f32[256]" = torch.ops.aten.mul.Tensor(sum_29, 0.0006377551020408163)
    unsqueeze_591: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_862, 0);  mul_862 = None
    unsqueeze_592: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_591, 2);  unsqueeze_591 = None
    unsqueeze_593: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_592, 3);  unsqueeze_592 = None
    mul_863: "f32[256]" = torch.ops.aten.mul.Tensor(sum_30, 0.0006377551020408163)
    mul_864: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_271, squeeze_271)
    mul_865: "f32[256]" = torch.ops.aten.mul.Tensor(mul_863, mul_864);  mul_863 = mul_864 = None
    unsqueeze_594: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_865, 0);  mul_865 = None
    unsqueeze_595: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_594, 2);  unsqueeze_594 = None
    unsqueeze_596: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_595, 3);  unsqueeze_595 = None
    mul_866: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_271, primals_272);  primals_272 = None
    unsqueeze_597: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_866, 0);  mul_866 = None
    unsqueeze_598: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_597, 2);  unsqueeze_597 = None
    unsqueeze_599: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_598, 3);  unsqueeze_598 = None
    mul_867: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_161, unsqueeze_596);  sub_161 = unsqueeze_596 = None
    sub_163: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(where_13, mul_867);  where_13 = mul_867 = None
    sub_164: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(sub_163, unsqueeze_593);  sub_163 = unsqueeze_593 = None
    mul_868: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_164, unsqueeze_599);  sub_164 = unsqueeze_599 = None
    mul_869: "f32[256]" = torch.ops.aten.mul.Tensor(sum_30, squeeze_271);  sum_30 = squeeze_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    convolution_backward_15 = torch.ops.aten.convolution_backward.default(mul_868, relu_86, primals_271, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_868 = primals_271 = None
    getitem_273: "f32[8, 512, 14, 14]" = convolution_backward_15[0]
    getitem_274: "f32[256, 512, 1, 1]" = convolution_backward_15[1];  convolution_backward_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    add_576: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_575, getitem_273);  add_575 = getitem_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:180, code: x = self.relu(x)
    le_14: "b8[8, 512, 14, 14]" = torch.ops.aten.le.Scalar(relu_86, 0);  relu_86 = None
    where_14: "f32[8, 512, 14, 14]" = torch.ops.aten.where.self(le_14, full_default, add_576);  le_14 = add_576 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:177, code: x = self.bn(x)
    sum_31: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_14, [0, 2, 3])
    sub_165: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_89, unsqueeze_602);  convolution_89 = unsqueeze_602 = None
    mul_870: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(where_14, sub_165)
    sum_32: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_870, [0, 2, 3]);  mul_870 = None
    mul_871: "f32[512]" = torch.ops.aten.mul.Tensor(sum_31, 0.0006377551020408163)
    unsqueeze_603: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_871, 0);  mul_871 = None
    unsqueeze_604: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_603, 2);  unsqueeze_603 = None
    unsqueeze_605: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_604, 3);  unsqueeze_604 = None
    mul_872: "f32[512]" = torch.ops.aten.mul.Tensor(sum_32, 0.0006377551020408163)
    mul_873: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_268, squeeze_268)
    mul_874: "f32[512]" = torch.ops.aten.mul.Tensor(mul_872, mul_873);  mul_872 = mul_873 = None
    unsqueeze_606: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_874, 0);  mul_874 = None
    unsqueeze_607: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_606, 2);  unsqueeze_606 = None
    unsqueeze_608: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_607, 3);  unsqueeze_607 = None
    mul_875: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_268, primals_269);  primals_269 = None
    unsqueeze_609: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_875, 0);  mul_875 = None
    unsqueeze_610: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_609, 2);  unsqueeze_609 = None
    unsqueeze_611: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_610, 3);  unsqueeze_610 = None
    mul_876: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_165, unsqueeze_608);  sub_165 = unsqueeze_608 = None
    sub_167: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(where_14, mul_876);  mul_876 = None
    sub_168: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(sub_167, unsqueeze_605);  sub_167 = unsqueeze_605 = None
    mul_877: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_168, unsqueeze_611);  sub_168 = unsqueeze_611 = None
    mul_878: "f32[512]" = torch.ops.aten.mul.Tensor(sum_32, squeeze_268);  sum_32 = squeeze_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:176, code: x = self.conv(torch.cat(x_children, 1))
    convolution_backward_16 = torch.ops.aten.convolution_backward.default(mul_877, cat_11, primals_268, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_877 = cat_11 = primals_268 = None
    getitem_276: "f32[8, 1024, 14, 14]" = convolution_backward_16[0]
    getitem_277: "f32[512, 1024, 1, 1]" = convolution_backward_16[1];  convolution_backward_16 = None
    slice_10: "f32[8, 512, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_276, 1, 0, 512)
    slice_11: "f32[8, 512, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_276, 1, 512, 1024);  getitem_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:176, code: x = self.conv(torch.cat(x_children, 1))
    add_577: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(where_14, slice_10);  where_14 = slice_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    where_15: "f32[8, 512, 14, 14]" = torch.ops.aten.where.self(le_15, full_default, add_577);  le_15 = add_577 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    add_578: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(slice_11, where_15);  slice_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    sum_33: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_15, [0, 2, 3])
    sub_169: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_88, unsqueeze_614);  convolution_88 = unsqueeze_614 = None
    mul_879: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(where_15, sub_169)
    sum_34: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_879, [0, 2, 3]);  mul_879 = None
    mul_880: "f32[512]" = torch.ops.aten.mul.Tensor(sum_33, 0.0006377551020408163)
    unsqueeze_615: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_880, 0);  mul_880 = None
    unsqueeze_616: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_615, 2);  unsqueeze_615 = None
    unsqueeze_617: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_616, 3);  unsqueeze_616 = None
    mul_881: "f32[512]" = torch.ops.aten.mul.Tensor(sum_34, 0.0006377551020408163)
    mul_882: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_265, squeeze_265)
    mul_883: "f32[512]" = torch.ops.aten.mul.Tensor(mul_881, mul_882);  mul_881 = mul_882 = None
    unsqueeze_618: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_883, 0);  mul_883 = None
    unsqueeze_619: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_618, 2);  unsqueeze_618 = None
    unsqueeze_620: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_619, 3);  unsqueeze_619 = None
    mul_884: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_265, primals_266);  primals_266 = None
    unsqueeze_621: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_884, 0);  mul_884 = None
    unsqueeze_622: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_621, 2);  unsqueeze_621 = None
    unsqueeze_623: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_622, 3);  unsqueeze_622 = None
    mul_885: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_169, unsqueeze_620);  sub_169 = unsqueeze_620 = None
    sub_171: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(where_15, mul_885);  where_15 = mul_885 = None
    sub_172: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(sub_171, unsqueeze_617);  sub_171 = unsqueeze_617 = None
    mul_886: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_172, unsqueeze_623);  sub_172 = unsqueeze_623 = None
    mul_887: "f32[512]" = torch.ops.aten.mul.Tensor(sum_34, squeeze_265);  sum_34 = squeeze_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    convolution_backward_17 = torch.ops.aten.convolution_backward.default(mul_886, relu_84, primals_265, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_886 = primals_265 = None
    getitem_279: "f32[8, 256, 14, 14]" = convolution_backward_17[0]
    getitem_280: "f32[512, 256, 1, 1]" = convolution_backward_17[1];  convolution_backward_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    le_16: "b8[8, 256, 14, 14]" = torch.ops.aten.le.Scalar(relu_84, 0);  relu_84 = None
    where_16: "f32[8, 256, 14, 14]" = torch.ops.aten.where.self(le_16, full_default, getitem_279);  le_16 = getitem_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    sum_35: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_16, [0, 2, 3])
    sub_173: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_87, unsqueeze_626);  convolution_87 = unsqueeze_626 = None
    mul_888: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_16, sub_173)
    sum_36: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_888, [0, 2, 3]);  mul_888 = None
    mul_889: "f32[256]" = torch.ops.aten.mul.Tensor(sum_35, 0.0006377551020408163)
    unsqueeze_627: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_889, 0);  mul_889 = None
    unsqueeze_628: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_627, 2);  unsqueeze_627 = None
    unsqueeze_629: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_628, 3);  unsqueeze_628 = None
    mul_890: "f32[256]" = torch.ops.aten.mul.Tensor(sum_36, 0.0006377551020408163)
    mul_891: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_262, squeeze_262)
    mul_892: "f32[256]" = torch.ops.aten.mul.Tensor(mul_890, mul_891);  mul_890 = mul_891 = None
    unsqueeze_630: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_892, 0);  mul_892 = None
    unsqueeze_631: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_630, 2);  unsqueeze_630 = None
    unsqueeze_632: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_631, 3);  unsqueeze_631 = None
    mul_893: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_262, primals_263);  primals_263 = None
    unsqueeze_633: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_893, 0);  mul_893 = None
    unsqueeze_634: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_633, 2);  unsqueeze_633 = None
    unsqueeze_635: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_634, 3);  unsqueeze_634 = None
    mul_894: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_173, unsqueeze_632);  sub_173 = unsqueeze_632 = None
    sub_175: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(where_16, mul_894);  where_16 = mul_894 = None
    sub_176: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(sub_175, unsqueeze_629);  sub_175 = unsqueeze_629 = None
    mul_895: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_176, unsqueeze_635);  sub_176 = unsqueeze_635 = None
    mul_896: "f32[256]" = torch.ops.aten.mul.Tensor(sum_36, squeeze_262);  sum_36 = squeeze_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    convolution_backward_18 = torch.ops.aten.convolution_backward.default(mul_895, relu_83, primals_262, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_895 = primals_262 = None
    getitem_282: "f32[8, 256, 14, 14]" = convolution_backward_18[0]
    getitem_283: "f32[256, 256, 3, 3]" = convolution_backward_18[1];  convolution_backward_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    le_17: "b8[8, 256, 14, 14]" = torch.ops.aten.le.Scalar(relu_83, 0);  relu_83 = None
    where_17: "f32[8, 256, 14, 14]" = torch.ops.aten.where.self(le_17, full_default, getitem_282);  le_17 = getitem_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    sum_37: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_17, [0, 2, 3])
    sub_177: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_86, unsqueeze_638);  convolution_86 = unsqueeze_638 = None
    mul_897: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_17, sub_177)
    sum_38: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_897, [0, 2, 3]);  mul_897 = None
    mul_898: "f32[256]" = torch.ops.aten.mul.Tensor(sum_37, 0.0006377551020408163)
    unsqueeze_639: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_898, 0);  mul_898 = None
    unsqueeze_640: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_639, 2);  unsqueeze_639 = None
    unsqueeze_641: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_640, 3);  unsqueeze_640 = None
    mul_899: "f32[256]" = torch.ops.aten.mul.Tensor(sum_38, 0.0006377551020408163)
    mul_900: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_259, squeeze_259)
    mul_901: "f32[256]" = torch.ops.aten.mul.Tensor(mul_899, mul_900);  mul_899 = mul_900 = None
    unsqueeze_642: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_901, 0);  mul_901 = None
    unsqueeze_643: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_642, 2);  unsqueeze_642 = None
    unsqueeze_644: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_643, 3);  unsqueeze_643 = None
    mul_902: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_259, primals_260);  primals_260 = None
    unsqueeze_645: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_902, 0);  mul_902 = None
    unsqueeze_646: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_645, 2);  unsqueeze_645 = None
    unsqueeze_647: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_646, 3);  unsqueeze_646 = None
    mul_903: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_177, unsqueeze_644);  sub_177 = unsqueeze_644 = None
    sub_179: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(where_17, mul_903);  where_17 = mul_903 = None
    sub_180: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(sub_179, unsqueeze_641);  sub_179 = unsqueeze_641 = None
    mul_904: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_180, unsqueeze_647);  sub_180 = unsqueeze_647 = None
    mul_905: "f32[256]" = torch.ops.aten.mul.Tensor(sum_38, squeeze_259);  sum_38 = squeeze_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    convolution_backward_19 = torch.ops.aten.convolution_backward.default(mul_904, relu_82, primals_259, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_904 = primals_259 = None
    getitem_285: "f32[8, 512, 14, 14]" = convolution_backward_19[0]
    getitem_286: "f32[256, 512, 1, 1]" = convolution_backward_19[1];  convolution_backward_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    add_579: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_578, getitem_285);  add_578 = getitem_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    le_18: "b8[8, 512, 14, 14]" = torch.ops.aten.le.Scalar(relu_82, 0);  relu_82 = None
    where_18: "f32[8, 512, 14, 14]" = torch.ops.aten.where.self(le_18, full_default, add_579);  le_18 = add_579 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    add_580: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(slice_8, where_18);  slice_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    sum_39: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_18, [0, 2, 3])
    sub_181: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_85, unsqueeze_650);  convolution_85 = unsqueeze_650 = None
    mul_906: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(where_18, sub_181)
    sum_40: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_906, [0, 2, 3]);  mul_906 = None
    mul_907: "f32[512]" = torch.ops.aten.mul.Tensor(sum_39, 0.0006377551020408163)
    unsqueeze_651: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_907, 0);  mul_907 = None
    unsqueeze_652: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_651, 2);  unsqueeze_651 = None
    unsqueeze_653: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_652, 3);  unsqueeze_652 = None
    mul_908: "f32[512]" = torch.ops.aten.mul.Tensor(sum_40, 0.0006377551020408163)
    mul_909: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_256, squeeze_256)
    mul_910: "f32[512]" = torch.ops.aten.mul.Tensor(mul_908, mul_909);  mul_908 = mul_909 = None
    unsqueeze_654: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_910, 0);  mul_910 = None
    unsqueeze_655: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_654, 2);  unsqueeze_654 = None
    unsqueeze_656: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_655, 3);  unsqueeze_655 = None
    mul_911: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_256, primals_257);  primals_257 = None
    unsqueeze_657: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_911, 0);  mul_911 = None
    unsqueeze_658: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_657, 2);  unsqueeze_657 = None
    unsqueeze_659: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_658, 3);  unsqueeze_658 = None
    mul_912: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_181, unsqueeze_656);  sub_181 = unsqueeze_656 = None
    sub_183: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(where_18, mul_912);  where_18 = mul_912 = None
    sub_184: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(sub_183, unsqueeze_653);  sub_183 = unsqueeze_653 = None
    mul_913: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_184, unsqueeze_659);  sub_184 = unsqueeze_659 = None
    mul_914: "f32[512]" = torch.ops.aten.mul.Tensor(sum_40, squeeze_256);  sum_40 = squeeze_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    convolution_backward_20 = torch.ops.aten.convolution_backward.default(mul_913, relu_81, primals_256, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_913 = primals_256 = None
    getitem_288: "f32[8, 256, 14, 14]" = convolution_backward_20[0]
    getitem_289: "f32[512, 256, 1, 1]" = convolution_backward_20[1];  convolution_backward_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    le_19: "b8[8, 256, 14, 14]" = torch.ops.aten.le.Scalar(relu_81, 0);  relu_81 = None
    where_19: "f32[8, 256, 14, 14]" = torch.ops.aten.where.self(le_19, full_default, getitem_288);  le_19 = getitem_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    sum_41: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_19, [0, 2, 3])
    sub_185: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_84, unsqueeze_662);  convolution_84 = unsqueeze_662 = None
    mul_915: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_19, sub_185)
    sum_42: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_915, [0, 2, 3]);  mul_915 = None
    mul_916: "f32[256]" = torch.ops.aten.mul.Tensor(sum_41, 0.0006377551020408163)
    unsqueeze_663: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_916, 0);  mul_916 = None
    unsqueeze_664: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_663, 2);  unsqueeze_663 = None
    unsqueeze_665: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_664, 3);  unsqueeze_664 = None
    mul_917: "f32[256]" = torch.ops.aten.mul.Tensor(sum_42, 0.0006377551020408163)
    mul_918: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_253, squeeze_253)
    mul_919: "f32[256]" = torch.ops.aten.mul.Tensor(mul_917, mul_918);  mul_917 = mul_918 = None
    unsqueeze_666: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_919, 0);  mul_919 = None
    unsqueeze_667: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_666, 2);  unsqueeze_666 = None
    unsqueeze_668: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_667, 3);  unsqueeze_667 = None
    mul_920: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_253, primals_254);  primals_254 = None
    unsqueeze_669: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_920, 0);  mul_920 = None
    unsqueeze_670: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_669, 2);  unsqueeze_669 = None
    unsqueeze_671: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_670, 3);  unsqueeze_670 = None
    mul_921: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_185, unsqueeze_668);  sub_185 = unsqueeze_668 = None
    sub_187: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(where_19, mul_921);  where_19 = mul_921 = None
    sub_188: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(sub_187, unsqueeze_665);  sub_187 = unsqueeze_665 = None
    mul_922: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_188, unsqueeze_671);  sub_188 = unsqueeze_671 = None
    mul_923: "f32[256]" = torch.ops.aten.mul.Tensor(sum_42, squeeze_253);  sum_42 = squeeze_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    convolution_backward_21 = torch.ops.aten.convolution_backward.default(mul_922, relu_80, primals_253, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_922 = primals_253 = None
    getitem_291: "f32[8, 256, 14, 14]" = convolution_backward_21[0]
    getitem_292: "f32[256, 256, 3, 3]" = convolution_backward_21[1];  convolution_backward_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    le_20: "b8[8, 256, 14, 14]" = torch.ops.aten.le.Scalar(relu_80, 0);  relu_80 = None
    where_20: "f32[8, 256, 14, 14]" = torch.ops.aten.where.self(le_20, full_default, getitem_291);  le_20 = getitem_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    sum_43: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_20, [0, 2, 3])
    sub_189: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_83, unsqueeze_674);  convolution_83 = unsqueeze_674 = None
    mul_924: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_20, sub_189)
    sum_44: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_924, [0, 2, 3]);  mul_924 = None
    mul_925: "f32[256]" = torch.ops.aten.mul.Tensor(sum_43, 0.0006377551020408163)
    unsqueeze_675: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_925, 0);  mul_925 = None
    unsqueeze_676: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_675, 2);  unsqueeze_675 = None
    unsqueeze_677: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_676, 3);  unsqueeze_676 = None
    mul_926: "f32[256]" = torch.ops.aten.mul.Tensor(sum_44, 0.0006377551020408163)
    mul_927: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_250, squeeze_250)
    mul_928: "f32[256]" = torch.ops.aten.mul.Tensor(mul_926, mul_927);  mul_926 = mul_927 = None
    unsqueeze_678: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_928, 0);  mul_928 = None
    unsqueeze_679: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_678, 2);  unsqueeze_678 = None
    unsqueeze_680: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_679, 3);  unsqueeze_679 = None
    mul_929: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_250, primals_251);  primals_251 = None
    unsqueeze_681: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_929, 0);  mul_929 = None
    unsqueeze_682: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_681, 2);  unsqueeze_681 = None
    unsqueeze_683: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_682, 3);  unsqueeze_682 = None
    mul_930: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_189, unsqueeze_680);  sub_189 = unsqueeze_680 = None
    sub_191: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(where_20, mul_930);  where_20 = mul_930 = None
    sub_192: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(sub_191, unsqueeze_677);  sub_191 = unsqueeze_677 = None
    mul_931: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_192, unsqueeze_683);  sub_192 = unsqueeze_683 = None
    mul_932: "f32[256]" = torch.ops.aten.mul.Tensor(sum_44, squeeze_250);  sum_44 = squeeze_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    convolution_backward_22 = torch.ops.aten.convolution_backward.default(mul_931, relu_79, primals_250, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_931 = primals_250 = None
    getitem_294: "f32[8, 512, 14, 14]" = convolution_backward_22[0]
    getitem_295: "f32[256, 512, 1, 1]" = convolution_backward_22[1];  convolution_backward_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    add_581: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_580, getitem_294);  add_580 = getitem_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:180, code: x = self.relu(x)
    le_21: "b8[8, 512, 14, 14]" = torch.ops.aten.le.Scalar(relu_79, 0);  relu_79 = None
    where_21: "f32[8, 512, 14, 14]" = torch.ops.aten.where.self(le_21, full_default, add_581);  le_21 = add_581 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:177, code: x = self.bn(x)
    sum_45: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_21, [0, 2, 3])
    sub_193: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_82, unsqueeze_686);  convolution_82 = unsqueeze_686 = None
    mul_933: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(where_21, sub_193)
    sum_46: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_933, [0, 2, 3]);  mul_933 = None
    mul_934: "f32[512]" = torch.ops.aten.mul.Tensor(sum_45, 0.0006377551020408163)
    unsqueeze_687: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_934, 0);  mul_934 = None
    unsqueeze_688: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_687, 2);  unsqueeze_687 = None
    unsqueeze_689: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_688, 3);  unsqueeze_688 = None
    mul_935: "f32[512]" = torch.ops.aten.mul.Tensor(sum_46, 0.0006377551020408163)
    mul_936: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_247, squeeze_247)
    mul_937: "f32[512]" = torch.ops.aten.mul.Tensor(mul_935, mul_936);  mul_935 = mul_936 = None
    unsqueeze_690: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_937, 0);  mul_937 = None
    unsqueeze_691: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_690, 2);  unsqueeze_690 = None
    unsqueeze_692: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_691, 3);  unsqueeze_691 = None
    mul_938: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_247, primals_248);  primals_248 = None
    unsqueeze_693: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_938, 0);  mul_938 = None
    unsqueeze_694: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_693, 2);  unsqueeze_693 = None
    unsqueeze_695: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_694, 3);  unsqueeze_694 = None
    mul_939: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_193, unsqueeze_692);  sub_193 = unsqueeze_692 = None
    sub_195: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(where_21, mul_939);  mul_939 = None
    sub_196: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(sub_195, unsqueeze_689);  sub_195 = unsqueeze_689 = None
    mul_940: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_196, unsqueeze_695);  sub_196 = unsqueeze_695 = None
    mul_941: "f32[512]" = torch.ops.aten.mul.Tensor(sum_46, squeeze_247);  sum_46 = squeeze_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:176, code: x = self.conv(torch.cat(x_children, 1))
    convolution_backward_23 = torch.ops.aten.convolution_backward.default(mul_940, cat_10, primals_247, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_940 = cat_10 = primals_247 = None
    getitem_297: "f32[8, 1536, 14, 14]" = convolution_backward_23[0]
    getitem_298: "f32[512, 1536, 1, 1]" = convolution_backward_23[1];  convolution_backward_23 = None
    slice_12: "f32[8, 512, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_297, 1, 0, 512)
    slice_13: "f32[8, 512, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_297, 1, 512, 1024)
    slice_14: "f32[8, 512, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_297, 1, 1024, 1536);  getitem_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:176, code: x = self.conv(torch.cat(x_children, 1))
    add_582: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(where_21, slice_12);  where_21 = slice_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    where_22: "f32[8, 512, 14, 14]" = torch.ops.aten.where.self(le_22, full_default, add_582);  le_22 = add_582 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    add_583: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(slice_13, where_22);  slice_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    sum_47: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_22, [0, 2, 3])
    sub_197: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_81, unsqueeze_698);  convolution_81 = unsqueeze_698 = None
    mul_942: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(where_22, sub_197)
    sum_48: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_942, [0, 2, 3]);  mul_942 = None
    mul_943: "f32[512]" = torch.ops.aten.mul.Tensor(sum_47, 0.0006377551020408163)
    unsqueeze_699: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_943, 0);  mul_943 = None
    unsqueeze_700: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_699, 2);  unsqueeze_699 = None
    unsqueeze_701: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_700, 3);  unsqueeze_700 = None
    mul_944: "f32[512]" = torch.ops.aten.mul.Tensor(sum_48, 0.0006377551020408163)
    mul_945: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_244, squeeze_244)
    mul_946: "f32[512]" = torch.ops.aten.mul.Tensor(mul_944, mul_945);  mul_944 = mul_945 = None
    unsqueeze_702: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_946, 0);  mul_946 = None
    unsqueeze_703: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_702, 2);  unsqueeze_702 = None
    unsqueeze_704: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_703, 3);  unsqueeze_703 = None
    mul_947: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_244, primals_245);  primals_245 = None
    unsqueeze_705: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_947, 0);  mul_947 = None
    unsqueeze_706: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_705, 2);  unsqueeze_705 = None
    unsqueeze_707: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_706, 3);  unsqueeze_706 = None
    mul_948: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_197, unsqueeze_704);  sub_197 = unsqueeze_704 = None
    sub_199: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(where_22, mul_948);  where_22 = mul_948 = None
    sub_200: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(sub_199, unsqueeze_701);  sub_199 = unsqueeze_701 = None
    mul_949: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_200, unsqueeze_707);  sub_200 = unsqueeze_707 = None
    mul_950: "f32[512]" = torch.ops.aten.mul.Tensor(sum_48, squeeze_244);  sum_48 = squeeze_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    convolution_backward_24 = torch.ops.aten.convolution_backward.default(mul_949, relu_77, primals_244, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_949 = primals_244 = None
    getitem_300: "f32[8, 256, 14, 14]" = convolution_backward_24[0]
    getitem_301: "f32[512, 256, 1, 1]" = convolution_backward_24[1];  convolution_backward_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    le_23: "b8[8, 256, 14, 14]" = torch.ops.aten.le.Scalar(relu_77, 0);  relu_77 = None
    where_23: "f32[8, 256, 14, 14]" = torch.ops.aten.where.self(le_23, full_default, getitem_300);  le_23 = getitem_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    sum_49: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_23, [0, 2, 3])
    sub_201: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_80, unsqueeze_710);  convolution_80 = unsqueeze_710 = None
    mul_951: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_23, sub_201)
    sum_50: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_951, [0, 2, 3]);  mul_951 = None
    mul_952: "f32[256]" = torch.ops.aten.mul.Tensor(sum_49, 0.0006377551020408163)
    unsqueeze_711: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_952, 0);  mul_952 = None
    unsqueeze_712: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_711, 2);  unsqueeze_711 = None
    unsqueeze_713: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_712, 3);  unsqueeze_712 = None
    mul_953: "f32[256]" = torch.ops.aten.mul.Tensor(sum_50, 0.0006377551020408163)
    mul_954: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_241, squeeze_241)
    mul_955: "f32[256]" = torch.ops.aten.mul.Tensor(mul_953, mul_954);  mul_953 = mul_954 = None
    unsqueeze_714: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_955, 0);  mul_955 = None
    unsqueeze_715: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_714, 2);  unsqueeze_714 = None
    unsqueeze_716: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_715, 3);  unsqueeze_715 = None
    mul_956: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_241, primals_242);  primals_242 = None
    unsqueeze_717: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_956, 0);  mul_956 = None
    unsqueeze_718: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_717, 2);  unsqueeze_717 = None
    unsqueeze_719: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_718, 3);  unsqueeze_718 = None
    mul_957: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_201, unsqueeze_716);  sub_201 = unsqueeze_716 = None
    sub_203: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(where_23, mul_957);  where_23 = mul_957 = None
    sub_204: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(sub_203, unsqueeze_713);  sub_203 = unsqueeze_713 = None
    mul_958: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_204, unsqueeze_719);  sub_204 = unsqueeze_719 = None
    mul_959: "f32[256]" = torch.ops.aten.mul.Tensor(sum_50, squeeze_241);  sum_50 = squeeze_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    convolution_backward_25 = torch.ops.aten.convolution_backward.default(mul_958, relu_76, primals_241, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_958 = primals_241 = None
    getitem_303: "f32[8, 256, 14, 14]" = convolution_backward_25[0]
    getitem_304: "f32[256, 256, 3, 3]" = convolution_backward_25[1];  convolution_backward_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    le_24: "b8[8, 256, 14, 14]" = torch.ops.aten.le.Scalar(relu_76, 0);  relu_76 = None
    where_24: "f32[8, 256, 14, 14]" = torch.ops.aten.where.self(le_24, full_default, getitem_303);  le_24 = getitem_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    sum_51: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_24, [0, 2, 3])
    sub_205: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_79, unsqueeze_722);  convolution_79 = unsqueeze_722 = None
    mul_960: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_24, sub_205)
    sum_52: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_960, [0, 2, 3]);  mul_960 = None
    mul_961: "f32[256]" = torch.ops.aten.mul.Tensor(sum_51, 0.0006377551020408163)
    unsqueeze_723: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_961, 0);  mul_961 = None
    unsqueeze_724: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_723, 2);  unsqueeze_723 = None
    unsqueeze_725: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_724, 3);  unsqueeze_724 = None
    mul_962: "f32[256]" = torch.ops.aten.mul.Tensor(sum_52, 0.0006377551020408163)
    mul_963: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_238, squeeze_238)
    mul_964: "f32[256]" = torch.ops.aten.mul.Tensor(mul_962, mul_963);  mul_962 = mul_963 = None
    unsqueeze_726: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_964, 0);  mul_964 = None
    unsqueeze_727: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_726, 2);  unsqueeze_726 = None
    unsqueeze_728: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_727, 3);  unsqueeze_727 = None
    mul_965: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_238, primals_239);  primals_239 = None
    unsqueeze_729: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_965, 0);  mul_965 = None
    unsqueeze_730: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_729, 2);  unsqueeze_729 = None
    unsqueeze_731: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_730, 3);  unsqueeze_730 = None
    mul_966: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_205, unsqueeze_728);  sub_205 = unsqueeze_728 = None
    sub_207: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(where_24, mul_966);  where_24 = mul_966 = None
    sub_208: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(sub_207, unsqueeze_725);  sub_207 = unsqueeze_725 = None
    mul_967: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_208, unsqueeze_731);  sub_208 = unsqueeze_731 = None
    mul_968: "f32[256]" = torch.ops.aten.mul.Tensor(sum_52, squeeze_238);  sum_52 = squeeze_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    convolution_backward_26 = torch.ops.aten.convolution_backward.default(mul_967, relu_75, primals_238, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_967 = primals_238 = None
    getitem_306: "f32[8, 512, 14, 14]" = convolution_backward_26[0]
    getitem_307: "f32[256, 512, 1, 1]" = convolution_backward_26[1];  convolution_backward_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    add_584: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_583, getitem_306);  add_583 = getitem_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    le_25: "b8[8, 512, 14, 14]" = torch.ops.aten.le.Scalar(relu_75, 0);  relu_75 = None
    where_25: "f32[8, 512, 14, 14]" = torch.ops.aten.where.self(le_25, full_default, add_584);  le_25 = add_584 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    add_585: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(slice_14, where_25);  slice_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    sum_53: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_25, [0, 2, 3])
    sub_209: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_78, unsqueeze_734);  convolution_78 = unsqueeze_734 = None
    mul_969: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(where_25, sub_209)
    sum_54: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_969, [0, 2, 3]);  mul_969 = None
    mul_970: "f32[512]" = torch.ops.aten.mul.Tensor(sum_53, 0.0006377551020408163)
    unsqueeze_735: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_970, 0);  mul_970 = None
    unsqueeze_736: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_735, 2);  unsqueeze_735 = None
    unsqueeze_737: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_736, 3);  unsqueeze_736 = None
    mul_971: "f32[512]" = torch.ops.aten.mul.Tensor(sum_54, 0.0006377551020408163)
    mul_972: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_235, squeeze_235)
    mul_973: "f32[512]" = torch.ops.aten.mul.Tensor(mul_971, mul_972);  mul_971 = mul_972 = None
    unsqueeze_738: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_973, 0);  mul_973 = None
    unsqueeze_739: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_738, 2);  unsqueeze_738 = None
    unsqueeze_740: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_739, 3);  unsqueeze_739 = None
    mul_974: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_235, primals_236);  primals_236 = None
    unsqueeze_741: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_974, 0);  mul_974 = None
    unsqueeze_742: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_741, 2);  unsqueeze_741 = None
    unsqueeze_743: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_742, 3);  unsqueeze_742 = None
    mul_975: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_209, unsqueeze_740);  sub_209 = unsqueeze_740 = None
    sub_211: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(where_25, mul_975);  where_25 = mul_975 = None
    sub_212: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(sub_211, unsqueeze_737);  sub_211 = unsqueeze_737 = None
    mul_976: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_212, unsqueeze_743);  sub_212 = unsqueeze_743 = None
    mul_977: "f32[512]" = torch.ops.aten.mul.Tensor(sum_54, squeeze_235);  sum_54 = squeeze_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    convolution_backward_27 = torch.ops.aten.convolution_backward.default(mul_976, relu_74, primals_235, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_976 = primals_235 = None
    getitem_309: "f32[8, 256, 14, 14]" = convolution_backward_27[0]
    getitem_310: "f32[512, 256, 1, 1]" = convolution_backward_27[1];  convolution_backward_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    le_26: "b8[8, 256, 14, 14]" = torch.ops.aten.le.Scalar(relu_74, 0);  relu_74 = None
    where_26: "f32[8, 256, 14, 14]" = torch.ops.aten.where.self(le_26, full_default, getitem_309);  le_26 = getitem_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    sum_55: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_26, [0, 2, 3])
    sub_213: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_77, unsqueeze_746);  convolution_77 = unsqueeze_746 = None
    mul_978: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_26, sub_213)
    sum_56: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_978, [0, 2, 3]);  mul_978 = None
    mul_979: "f32[256]" = torch.ops.aten.mul.Tensor(sum_55, 0.0006377551020408163)
    unsqueeze_747: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_979, 0);  mul_979 = None
    unsqueeze_748: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_747, 2);  unsqueeze_747 = None
    unsqueeze_749: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_748, 3);  unsqueeze_748 = None
    mul_980: "f32[256]" = torch.ops.aten.mul.Tensor(sum_56, 0.0006377551020408163)
    mul_981: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_232, squeeze_232)
    mul_982: "f32[256]" = torch.ops.aten.mul.Tensor(mul_980, mul_981);  mul_980 = mul_981 = None
    unsqueeze_750: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_982, 0);  mul_982 = None
    unsqueeze_751: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_750, 2);  unsqueeze_750 = None
    unsqueeze_752: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_751, 3);  unsqueeze_751 = None
    mul_983: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_232, primals_233);  primals_233 = None
    unsqueeze_753: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_983, 0);  mul_983 = None
    unsqueeze_754: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_753, 2);  unsqueeze_753 = None
    unsqueeze_755: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_754, 3);  unsqueeze_754 = None
    mul_984: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_213, unsqueeze_752);  sub_213 = unsqueeze_752 = None
    sub_215: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(where_26, mul_984);  where_26 = mul_984 = None
    sub_216: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(sub_215, unsqueeze_749);  sub_215 = unsqueeze_749 = None
    mul_985: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_216, unsqueeze_755);  sub_216 = unsqueeze_755 = None
    mul_986: "f32[256]" = torch.ops.aten.mul.Tensor(sum_56, squeeze_232);  sum_56 = squeeze_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    convolution_backward_28 = torch.ops.aten.convolution_backward.default(mul_985, relu_73, primals_232, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_985 = primals_232 = None
    getitem_312: "f32[8, 256, 14, 14]" = convolution_backward_28[0]
    getitem_313: "f32[256, 256, 3, 3]" = convolution_backward_28[1];  convolution_backward_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    le_27: "b8[8, 256, 14, 14]" = torch.ops.aten.le.Scalar(relu_73, 0);  relu_73 = None
    where_27: "f32[8, 256, 14, 14]" = torch.ops.aten.where.self(le_27, full_default, getitem_312);  le_27 = getitem_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    sum_57: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_27, [0, 2, 3])
    sub_217: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_76, unsqueeze_758);  convolution_76 = unsqueeze_758 = None
    mul_987: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_27, sub_217)
    sum_58: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_987, [0, 2, 3]);  mul_987 = None
    mul_988: "f32[256]" = torch.ops.aten.mul.Tensor(sum_57, 0.0006377551020408163)
    unsqueeze_759: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_988, 0);  mul_988 = None
    unsqueeze_760: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_759, 2);  unsqueeze_759 = None
    unsqueeze_761: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_760, 3);  unsqueeze_760 = None
    mul_989: "f32[256]" = torch.ops.aten.mul.Tensor(sum_58, 0.0006377551020408163)
    mul_990: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_229, squeeze_229)
    mul_991: "f32[256]" = torch.ops.aten.mul.Tensor(mul_989, mul_990);  mul_989 = mul_990 = None
    unsqueeze_762: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_991, 0);  mul_991 = None
    unsqueeze_763: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_762, 2);  unsqueeze_762 = None
    unsqueeze_764: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_763, 3);  unsqueeze_763 = None
    mul_992: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_229, primals_230);  primals_230 = None
    unsqueeze_765: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_992, 0);  mul_992 = None
    unsqueeze_766: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_765, 2);  unsqueeze_765 = None
    unsqueeze_767: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_766, 3);  unsqueeze_766 = None
    mul_993: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_217, unsqueeze_764);  sub_217 = unsqueeze_764 = None
    sub_219: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(where_27, mul_993);  where_27 = mul_993 = None
    sub_220: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(sub_219, unsqueeze_761);  sub_219 = unsqueeze_761 = None
    mul_994: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_220, unsqueeze_767);  sub_220 = unsqueeze_767 = None
    mul_995: "f32[256]" = torch.ops.aten.mul.Tensor(sum_58, squeeze_229);  sum_58 = squeeze_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    convolution_backward_29 = torch.ops.aten.convolution_backward.default(mul_994, relu_72, primals_229, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_994 = primals_229 = None
    getitem_315: "f32[8, 512, 14, 14]" = convolution_backward_29[0]
    getitem_316: "f32[256, 512, 1, 1]" = convolution_backward_29[1];  convolution_backward_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    add_586: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_585, getitem_315);  add_585 = getitem_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:180, code: x = self.relu(x)
    le_28: "b8[8, 512, 14, 14]" = torch.ops.aten.le.Scalar(relu_72, 0);  relu_72 = None
    where_28: "f32[8, 512, 14, 14]" = torch.ops.aten.where.self(le_28, full_default, add_586);  le_28 = add_586 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:177, code: x = self.bn(x)
    sum_59: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_28, [0, 2, 3])
    sub_221: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_75, unsqueeze_770);  convolution_75 = unsqueeze_770 = None
    mul_996: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(where_28, sub_221)
    sum_60: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_996, [0, 2, 3]);  mul_996 = None
    mul_997: "f32[512]" = torch.ops.aten.mul.Tensor(sum_59, 0.0006377551020408163)
    unsqueeze_771: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_997, 0);  mul_997 = None
    unsqueeze_772: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_771, 2);  unsqueeze_771 = None
    unsqueeze_773: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_772, 3);  unsqueeze_772 = None
    mul_998: "f32[512]" = torch.ops.aten.mul.Tensor(sum_60, 0.0006377551020408163)
    mul_999: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_226, squeeze_226)
    mul_1000: "f32[512]" = torch.ops.aten.mul.Tensor(mul_998, mul_999);  mul_998 = mul_999 = None
    unsqueeze_774: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1000, 0);  mul_1000 = None
    unsqueeze_775: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_774, 2);  unsqueeze_774 = None
    unsqueeze_776: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_775, 3);  unsqueeze_775 = None
    mul_1001: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_226, primals_227);  primals_227 = None
    unsqueeze_777: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1001, 0);  mul_1001 = None
    unsqueeze_778: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_777, 2);  unsqueeze_777 = None
    unsqueeze_779: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_778, 3);  unsqueeze_778 = None
    mul_1002: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_221, unsqueeze_776);  sub_221 = unsqueeze_776 = None
    sub_223: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(where_28, mul_1002);  mul_1002 = None
    sub_224: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(sub_223, unsqueeze_773);  sub_223 = unsqueeze_773 = None
    mul_1003: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_224, unsqueeze_779);  sub_224 = unsqueeze_779 = None
    mul_1004: "f32[512]" = torch.ops.aten.mul.Tensor(sum_60, squeeze_226);  sum_60 = squeeze_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:176, code: x = self.conv(torch.cat(x_children, 1))
    convolution_backward_30 = torch.ops.aten.convolution_backward.default(mul_1003, cat_9, primals_226, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1003 = cat_9 = primals_226 = None
    getitem_318: "f32[8, 1024, 14, 14]" = convolution_backward_30[0]
    getitem_319: "f32[512, 1024, 1, 1]" = convolution_backward_30[1];  convolution_backward_30 = None
    slice_15: "f32[8, 512, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_318, 1, 0, 512)
    slice_16: "f32[8, 512, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_318, 1, 512, 1024);  getitem_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:176, code: x = self.conv(torch.cat(x_children, 1))
    add_587: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(where_28, slice_15);  where_28 = slice_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    where_29: "f32[8, 512, 14, 14]" = torch.ops.aten.where.self(le_29, full_default, add_587);  le_29 = add_587 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    add_588: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(slice_16, where_29);  slice_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    sum_61: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_29, [0, 2, 3])
    sub_225: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_74, unsqueeze_782);  convolution_74 = unsqueeze_782 = None
    mul_1005: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(where_29, sub_225)
    sum_62: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1005, [0, 2, 3]);  mul_1005 = None
    mul_1006: "f32[512]" = torch.ops.aten.mul.Tensor(sum_61, 0.0006377551020408163)
    unsqueeze_783: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1006, 0);  mul_1006 = None
    unsqueeze_784: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_783, 2);  unsqueeze_783 = None
    unsqueeze_785: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_784, 3);  unsqueeze_784 = None
    mul_1007: "f32[512]" = torch.ops.aten.mul.Tensor(sum_62, 0.0006377551020408163)
    mul_1008: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_223, squeeze_223)
    mul_1009: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1007, mul_1008);  mul_1007 = mul_1008 = None
    unsqueeze_786: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1009, 0);  mul_1009 = None
    unsqueeze_787: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_786, 2);  unsqueeze_786 = None
    unsqueeze_788: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_787, 3);  unsqueeze_787 = None
    mul_1010: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_223, primals_224);  primals_224 = None
    unsqueeze_789: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1010, 0);  mul_1010 = None
    unsqueeze_790: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_789, 2);  unsqueeze_789 = None
    unsqueeze_791: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_790, 3);  unsqueeze_790 = None
    mul_1011: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_225, unsqueeze_788);  sub_225 = unsqueeze_788 = None
    sub_227: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(where_29, mul_1011);  where_29 = mul_1011 = None
    sub_228: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(sub_227, unsqueeze_785);  sub_227 = unsqueeze_785 = None
    mul_1012: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_228, unsqueeze_791);  sub_228 = unsqueeze_791 = None
    mul_1013: "f32[512]" = torch.ops.aten.mul.Tensor(sum_62, squeeze_223);  sum_62 = squeeze_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    convolution_backward_31 = torch.ops.aten.convolution_backward.default(mul_1012, relu_70, primals_223, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1012 = primals_223 = None
    getitem_321: "f32[8, 256, 14, 14]" = convolution_backward_31[0]
    getitem_322: "f32[512, 256, 1, 1]" = convolution_backward_31[1];  convolution_backward_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    le_30: "b8[8, 256, 14, 14]" = torch.ops.aten.le.Scalar(relu_70, 0);  relu_70 = None
    where_30: "f32[8, 256, 14, 14]" = torch.ops.aten.where.self(le_30, full_default, getitem_321);  le_30 = getitem_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    sum_63: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_30, [0, 2, 3])
    sub_229: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_73, unsqueeze_794);  convolution_73 = unsqueeze_794 = None
    mul_1014: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_30, sub_229)
    sum_64: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1014, [0, 2, 3]);  mul_1014 = None
    mul_1015: "f32[256]" = torch.ops.aten.mul.Tensor(sum_63, 0.0006377551020408163)
    unsqueeze_795: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1015, 0);  mul_1015 = None
    unsqueeze_796: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_795, 2);  unsqueeze_795 = None
    unsqueeze_797: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_796, 3);  unsqueeze_796 = None
    mul_1016: "f32[256]" = torch.ops.aten.mul.Tensor(sum_64, 0.0006377551020408163)
    mul_1017: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_220, squeeze_220)
    mul_1018: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1016, mul_1017);  mul_1016 = mul_1017 = None
    unsqueeze_798: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1018, 0);  mul_1018 = None
    unsqueeze_799: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_798, 2);  unsqueeze_798 = None
    unsqueeze_800: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_799, 3);  unsqueeze_799 = None
    mul_1019: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_220, primals_221);  primals_221 = None
    unsqueeze_801: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1019, 0);  mul_1019 = None
    unsqueeze_802: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_801, 2);  unsqueeze_801 = None
    unsqueeze_803: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_802, 3);  unsqueeze_802 = None
    mul_1020: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_229, unsqueeze_800);  sub_229 = unsqueeze_800 = None
    sub_231: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(where_30, mul_1020);  where_30 = mul_1020 = None
    sub_232: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(sub_231, unsqueeze_797);  sub_231 = unsqueeze_797 = None
    mul_1021: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_232, unsqueeze_803);  sub_232 = unsqueeze_803 = None
    mul_1022: "f32[256]" = torch.ops.aten.mul.Tensor(sum_64, squeeze_220);  sum_64 = squeeze_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    convolution_backward_32 = torch.ops.aten.convolution_backward.default(mul_1021, relu_69, primals_220, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1021 = primals_220 = None
    getitem_324: "f32[8, 256, 14, 14]" = convolution_backward_32[0]
    getitem_325: "f32[256, 256, 3, 3]" = convolution_backward_32[1];  convolution_backward_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    le_31: "b8[8, 256, 14, 14]" = torch.ops.aten.le.Scalar(relu_69, 0);  relu_69 = None
    where_31: "f32[8, 256, 14, 14]" = torch.ops.aten.where.self(le_31, full_default, getitem_324);  le_31 = getitem_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    sum_65: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_31, [0, 2, 3])
    sub_233: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_72, unsqueeze_806);  convolution_72 = unsqueeze_806 = None
    mul_1023: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_31, sub_233)
    sum_66: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1023, [0, 2, 3]);  mul_1023 = None
    mul_1024: "f32[256]" = torch.ops.aten.mul.Tensor(sum_65, 0.0006377551020408163)
    unsqueeze_807: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1024, 0);  mul_1024 = None
    unsqueeze_808: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_807, 2);  unsqueeze_807 = None
    unsqueeze_809: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_808, 3);  unsqueeze_808 = None
    mul_1025: "f32[256]" = torch.ops.aten.mul.Tensor(sum_66, 0.0006377551020408163)
    mul_1026: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_217, squeeze_217)
    mul_1027: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1025, mul_1026);  mul_1025 = mul_1026 = None
    unsqueeze_810: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1027, 0);  mul_1027 = None
    unsqueeze_811: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_810, 2);  unsqueeze_810 = None
    unsqueeze_812: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_811, 3);  unsqueeze_811 = None
    mul_1028: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_217, primals_218);  primals_218 = None
    unsqueeze_813: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1028, 0);  mul_1028 = None
    unsqueeze_814: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_813, 2);  unsqueeze_813 = None
    unsqueeze_815: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_814, 3);  unsqueeze_814 = None
    mul_1029: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_233, unsqueeze_812);  sub_233 = unsqueeze_812 = None
    sub_235: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(where_31, mul_1029);  where_31 = mul_1029 = None
    sub_236: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(sub_235, unsqueeze_809);  sub_235 = unsqueeze_809 = None
    mul_1030: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_236, unsqueeze_815);  sub_236 = unsqueeze_815 = None
    mul_1031: "f32[256]" = torch.ops.aten.mul.Tensor(sum_66, squeeze_217);  sum_66 = squeeze_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    convolution_backward_33 = torch.ops.aten.convolution_backward.default(mul_1030, relu_68, primals_217, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1030 = primals_217 = None
    getitem_327: "f32[8, 512, 14, 14]" = convolution_backward_33[0]
    getitem_328: "f32[256, 512, 1, 1]" = convolution_backward_33[1];  convolution_backward_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    add_589: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_588, getitem_327);  add_588 = getitem_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    le_32: "b8[8, 512, 14, 14]" = torch.ops.aten.le.Scalar(relu_68, 0);  relu_68 = None
    where_32: "f32[8, 512, 14, 14]" = torch.ops.aten.where.self(le_32, full_default, add_589);  le_32 = add_589 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    add_590: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(slice_7, where_32);  slice_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    sum_67: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_32, [0, 2, 3])
    sub_237: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_71, unsqueeze_818);  convolution_71 = unsqueeze_818 = None
    mul_1032: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(where_32, sub_237)
    sum_68: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1032, [0, 2, 3]);  mul_1032 = None
    mul_1033: "f32[512]" = torch.ops.aten.mul.Tensor(sum_67, 0.0006377551020408163)
    unsqueeze_819: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1033, 0);  mul_1033 = None
    unsqueeze_820: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_819, 2);  unsqueeze_819 = None
    unsqueeze_821: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_820, 3);  unsqueeze_820 = None
    mul_1034: "f32[512]" = torch.ops.aten.mul.Tensor(sum_68, 0.0006377551020408163)
    mul_1035: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_214, squeeze_214)
    mul_1036: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1034, mul_1035);  mul_1034 = mul_1035 = None
    unsqueeze_822: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1036, 0);  mul_1036 = None
    unsqueeze_823: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_822, 2);  unsqueeze_822 = None
    unsqueeze_824: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_823, 3);  unsqueeze_823 = None
    mul_1037: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_214, primals_215);  primals_215 = None
    unsqueeze_825: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1037, 0);  mul_1037 = None
    unsqueeze_826: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_825, 2);  unsqueeze_825 = None
    unsqueeze_827: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_826, 3);  unsqueeze_826 = None
    mul_1038: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_237, unsqueeze_824);  sub_237 = unsqueeze_824 = None
    sub_239: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(where_32, mul_1038);  where_32 = mul_1038 = None
    sub_240: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(sub_239, unsqueeze_821);  sub_239 = unsqueeze_821 = None
    mul_1039: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_240, unsqueeze_827);  sub_240 = unsqueeze_827 = None
    mul_1040: "f32[512]" = torch.ops.aten.mul.Tensor(sum_68, squeeze_214);  sum_68 = squeeze_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    convolution_backward_34 = torch.ops.aten.convolution_backward.default(mul_1039, relu_67, primals_214, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1039 = primals_214 = None
    getitem_330: "f32[8, 256, 14, 14]" = convolution_backward_34[0]
    getitem_331: "f32[512, 256, 1, 1]" = convolution_backward_34[1];  convolution_backward_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    le_33: "b8[8, 256, 14, 14]" = torch.ops.aten.le.Scalar(relu_67, 0);  relu_67 = None
    where_33: "f32[8, 256, 14, 14]" = torch.ops.aten.where.self(le_33, full_default, getitem_330);  le_33 = getitem_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    sum_69: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_33, [0, 2, 3])
    sub_241: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_70, unsqueeze_830);  convolution_70 = unsqueeze_830 = None
    mul_1041: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_33, sub_241)
    sum_70: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1041, [0, 2, 3]);  mul_1041 = None
    mul_1042: "f32[256]" = torch.ops.aten.mul.Tensor(sum_69, 0.0006377551020408163)
    unsqueeze_831: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1042, 0);  mul_1042 = None
    unsqueeze_832: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_831, 2);  unsqueeze_831 = None
    unsqueeze_833: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_832, 3);  unsqueeze_832 = None
    mul_1043: "f32[256]" = torch.ops.aten.mul.Tensor(sum_70, 0.0006377551020408163)
    mul_1044: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_211, squeeze_211)
    mul_1045: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1043, mul_1044);  mul_1043 = mul_1044 = None
    unsqueeze_834: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1045, 0);  mul_1045 = None
    unsqueeze_835: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_834, 2);  unsqueeze_834 = None
    unsqueeze_836: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_835, 3);  unsqueeze_835 = None
    mul_1046: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_211, primals_212);  primals_212 = None
    unsqueeze_837: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1046, 0);  mul_1046 = None
    unsqueeze_838: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_837, 2);  unsqueeze_837 = None
    unsqueeze_839: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_838, 3);  unsqueeze_838 = None
    mul_1047: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_241, unsqueeze_836);  sub_241 = unsqueeze_836 = None
    sub_243: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(where_33, mul_1047);  where_33 = mul_1047 = None
    sub_244: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(sub_243, unsqueeze_833);  sub_243 = unsqueeze_833 = None
    mul_1048: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_244, unsqueeze_839);  sub_244 = unsqueeze_839 = None
    mul_1049: "f32[256]" = torch.ops.aten.mul.Tensor(sum_70, squeeze_211);  sum_70 = squeeze_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    convolution_backward_35 = torch.ops.aten.convolution_backward.default(mul_1048, relu_66, primals_211, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1048 = primals_211 = None
    getitem_333: "f32[8, 256, 14, 14]" = convolution_backward_35[0]
    getitem_334: "f32[256, 256, 3, 3]" = convolution_backward_35[1];  convolution_backward_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    le_34: "b8[8, 256, 14, 14]" = torch.ops.aten.le.Scalar(relu_66, 0);  relu_66 = None
    where_34: "f32[8, 256, 14, 14]" = torch.ops.aten.where.self(le_34, full_default, getitem_333);  le_34 = getitem_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    sum_71: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_34, [0, 2, 3])
    sub_245: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_69, unsqueeze_842);  convolution_69 = unsqueeze_842 = None
    mul_1050: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_34, sub_245)
    sum_72: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1050, [0, 2, 3]);  mul_1050 = None
    mul_1051: "f32[256]" = torch.ops.aten.mul.Tensor(sum_71, 0.0006377551020408163)
    unsqueeze_843: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1051, 0);  mul_1051 = None
    unsqueeze_844: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_843, 2);  unsqueeze_843 = None
    unsqueeze_845: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_844, 3);  unsqueeze_844 = None
    mul_1052: "f32[256]" = torch.ops.aten.mul.Tensor(sum_72, 0.0006377551020408163)
    mul_1053: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_208, squeeze_208)
    mul_1054: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1052, mul_1053);  mul_1052 = mul_1053 = None
    unsqueeze_846: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1054, 0);  mul_1054 = None
    unsqueeze_847: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_846, 2);  unsqueeze_846 = None
    unsqueeze_848: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_847, 3);  unsqueeze_847 = None
    mul_1055: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_208, primals_209);  primals_209 = None
    unsqueeze_849: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1055, 0);  mul_1055 = None
    unsqueeze_850: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_849, 2);  unsqueeze_849 = None
    unsqueeze_851: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_850, 3);  unsqueeze_850 = None
    mul_1056: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_245, unsqueeze_848);  sub_245 = unsqueeze_848 = None
    sub_247: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(where_34, mul_1056);  where_34 = mul_1056 = None
    sub_248: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(sub_247, unsqueeze_845);  sub_247 = unsqueeze_845 = None
    mul_1057: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_248, unsqueeze_851);  sub_248 = unsqueeze_851 = None
    mul_1058: "f32[256]" = torch.ops.aten.mul.Tensor(sum_72, squeeze_208);  sum_72 = squeeze_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    convolution_backward_36 = torch.ops.aten.convolution_backward.default(mul_1057, relu_65, primals_208, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1057 = primals_208 = None
    getitem_336: "f32[8, 512, 14, 14]" = convolution_backward_36[0]
    getitem_337: "f32[256, 512, 1, 1]" = convolution_backward_36[1];  convolution_backward_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    add_591: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_590, getitem_336);  add_590 = getitem_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:180, code: x = self.relu(x)
    le_35: "b8[8, 512, 14, 14]" = torch.ops.aten.le.Scalar(relu_65, 0);  relu_65 = None
    where_35: "f32[8, 512, 14, 14]" = torch.ops.aten.where.self(le_35, full_default, add_591);  le_35 = add_591 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:177, code: x = self.bn(x)
    sum_73: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_35, [0, 2, 3])
    sub_249: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_68, unsqueeze_854);  convolution_68 = unsqueeze_854 = None
    mul_1059: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(where_35, sub_249)
    sum_74: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1059, [0, 2, 3]);  mul_1059 = None
    mul_1060: "f32[512]" = torch.ops.aten.mul.Tensor(sum_73, 0.0006377551020408163)
    unsqueeze_855: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1060, 0);  mul_1060 = None
    unsqueeze_856: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_855, 2);  unsqueeze_855 = None
    unsqueeze_857: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_856, 3);  unsqueeze_856 = None
    mul_1061: "f32[512]" = torch.ops.aten.mul.Tensor(sum_74, 0.0006377551020408163)
    mul_1062: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_205, squeeze_205)
    mul_1063: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1061, mul_1062);  mul_1061 = mul_1062 = None
    unsqueeze_858: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1063, 0);  mul_1063 = None
    unsqueeze_859: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_858, 2);  unsqueeze_858 = None
    unsqueeze_860: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_859, 3);  unsqueeze_859 = None
    mul_1064: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_205, primals_206);  primals_206 = None
    unsqueeze_861: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1064, 0);  mul_1064 = None
    unsqueeze_862: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_861, 2);  unsqueeze_861 = None
    unsqueeze_863: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_862, 3);  unsqueeze_862 = None
    mul_1065: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_249, unsqueeze_860);  sub_249 = unsqueeze_860 = None
    sub_251: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(where_35, mul_1065);  mul_1065 = None
    sub_252: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(sub_251, unsqueeze_857);  sub_251 = unsqueeze_857 = None
    mul_1066: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_252, unsqueeze_863);  sub_252 = unsqueeze_863 = None
    mul_1067: "f32[512]" = torch.ops.aten.mul.Tensor(sum_74, squeeze_205);  sum_74 = squeeze_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:176, code: x = self.conv(torch.cat(x_children, 1))
    convolution_backward_37 = torch.ops.aten.convolution_backward.default(mul_1066, cat_8, primals_205, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1066 = cat_8 = primals_205 = None
    getitem_339: "f32[8, 2048, 14, 14]" = convolution_backward_37[0]
    getitem_340: "f32[512, 2048, 1, 1]" = convolution_backward_37[1];  convolution_backward_37 = None
    slice_17: "f32[8, 512, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_339, 1, 0, 512)
    slice_18: "f32[8, 512, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_339, 1, 512, 1024)
    slice_19: "f32[8, 512, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_339, 1, 1024, 1536)
    slice_20: "f32[8, 512, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_339, 1, 1536, 2048);  getitem_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:176, code: x = self.conv(torch.cat(x_children, 1))
    add_592: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(where_35, slice_17);  where_35 = slice_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    where_36: "f32[8, 512, 14, 14]" = torch.ops.aten.where.self(le_36, full_default, add_592);  le_36 = add_592 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    add_593: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(slice_18, where_36);  slice_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    sum_75: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_36, [0, 2, 3])
    sub_253: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_67, unsqueeze_866);  convolution_67 = unsqueeze_866 = None
    mul_1068: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(where_36, sub_253)
    sum_76: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1068, [0, 2, 3]);  mul_1068 = None
    mul_1069: "f32[512]" = torch.ops.aten.mul.Tensor(sum_75, 0.0006377551020408163)
    unsqueeze_867: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1069, 0);  mul_1069 = None
    unsqueeze_868: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_867, 2);  unsqueeze_867 = None
    unsqueeze_869: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_868, 3);  unsqueeze_868 = None
    mul_1070: "f32[512]" = torch.ops.aten.mul.Tensor(sum_76, 0.0006377551020408163)
    mul_1071: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_202, squeeze_202)
    mul_1072: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1070, mul_1071);  mul_1070 = mul_1071 = None
    unsqueeze_870: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1072, 0);  mul_1072 = None
    unsqueeze_871: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_870, 2);  unsqueeze_870 = None
    unsqueeze_872: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_871, 3);  unsqueeze_871 = None
    mul_1073: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_202, primals_203);  primals_203 = None
    unsqueeze_873: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1073, 0);  mul_1073 = None
    unsqueeze_874: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_873, 2);  unsqueeze_873 = None
    unsqueeze_875: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_874, 3);  unsqueeze_874 = None
    mul_1074: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_253, unsqueeze_872);  sub_253 = unsqueeze_872 = None
    sub_255: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(where_36, mul_1074);  where_36 = mul_1074 = None
    sub_256: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(sub_255, unsqueeze_869);  sub_255 = unsqueeze_869 = None
    mul_1075: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_256, unsqueeze_875);  sub_256 = unsqueeze_875 = None
    mul_1076: "f32[512]" = torch.ops.aten.mul.Tensor(sum_76, squeeze_202);  sum_76 = squeeze_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    convolution_backward_38 = torch.ops.aten.convolution_backward.default(mul_1075, relu_63, primals_202, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1075 = primals_202 = None
    getitem_342: "f32[8, 256, 14, 14]" = convolution_backward_38[0]
    getitem_343: "f32[512, 256, 1, 1]" = convolution_backward_38[1];  convolution_backward_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    le_37: "b8[8, 256, 14, 14]" = torch.ops.aten.le.Scalar(relu_63, 0);  relu_63 = None
    where_37: "f32[8, 256, 14, 14]" = torch.ops.aten.where.self(le_37, full_default, getitem_342);  le_37 = getitem_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    sum_77: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_37, [0, 2, 3])
    sub_257: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_66, unsqueeze_878);  convolution_66 = unsqueeze_878 = None
    mul_1077: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_37, sub_257)
    sum_78: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1077, [0, 2, 3]);  mul_1077 = None
    mul_1078: "f32[256]" = torch.ops.aten.mul.Tensor(sum_77, 0.0006377551020408163)
    unsqueeze_879: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1078, 0);  mul_1078 = None
    unsqueeze_880: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_879, 2);  unsqueeze_879 = None
    unsqueeze_881: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_880, 3);  unsqueeze_880 = None
    mul_1079: "f32[256]" = torch.ops.aten.mul.Tensor(sum_78, 0.0006377551020408163)
    mul_1080: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_199, squeeze_199)
    mul_1081: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1079, mul_1080);  mul_1079 = mul_1080 = None
    unsqueeze_882: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1081, 0);  mul_1081 = None
    unsqueeze_883: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_882, 2);  unsqueeze_882 = None
    unsqueeze_884: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_883, 3);  unsqueeze_883 = None
    mul_1082: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_199, primals_200);  primals_200 = None
    unsqueeze_885: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1082, 0);  mul_1082 = None
    unsqueeze_886: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_885, 2);  unsqueeze_885 = None
    unsqueeze_887: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_886, 3);  unsqueeze_886 = None
    mul_1083: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_257, unsqueeze_884);  sub_257 = unsqueeze_884 = None
    sub_259: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(where_37, mul_1083);  where_37 = mul_1083 = None
    sub_260: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(sub_259, unsqueeze_881);  sub_259 = unsqueeze_881 = None
    mul_1084: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_260, unsqueeze_887);  sub_260 = unsqueeze_887 = None
    mul_1085: "f32[256]" = torch.ops.aten.mul.Tensor(sum_78, squeeze_199);  sum_78 = squeeze_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    convolution_backward_39 = torch.ops.aten.convolution_backward.default(mul_1084, relu_62, primals_199, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1084 = primals_199 = None
    getitem_345: "f32[8, 256, 14, 14]" = convolution_backward_39[0]
    getitem_346: "f32[256, 256, 3, 3]" = convolution_backward_39[1];  convolution_backward_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    le_38: "b8[8, 256, 14, 14]" = torch.ops.aten.le.Scalar(relu_62, 0);  relu_62 = None
    where_38: "f32[8, 256, 14, 14]" = torch.ops.aten.where.self(le_38, full_default, getitem_345);  le_38 = getitem_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    sum_79: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_38, [0, 2, 3])
    sub_261: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_65, unsqueeze_890);  convolution_65 = unsqueeze_890 = None
    mul_1086: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_38, sub_261)
    sum_80: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1086, [0, 2, 3]);  mul_1086 = None
    mul_1087: "f32[256]" = torch.ops.aten.mul.Tensor(sum_79, 0.0006377551020408163)
    unsqueeze_891: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1087, 0);  mul_1087 = None
    unsqueeze_892: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_891, 2);  unsqueeze_891 = None
    unsqueeze_893: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_892, 3);  unsqueeze_892 = None
    mul_1088: "f32[256]" = torch.ops.aten.mul.Tensor(sum_80, 0.0006377551020408163)
    mul_1089: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_196, squeeze_196)
    mul_1090: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1088, mul_1089);  mul_1088 = mul_1089 = None
    unsqueeze_894: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1090, 0);  mul_1090 = None
    unsqueeze_895: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_894, 2);  unsqueeze_894 = None
    unsqueeze_896: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_895, 3);  unsqueeze_895 = None
    mul_1091: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_196, primals_197);  primals_197 = None
    unsqueeze_897: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1091, 0);  mul_1091 = None
    unsqueeze_898: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_897, 2);  unsqueeze_897 = None
    unsqueeze_899: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_898, 3);  unsqueeze_898 = None
    mul_1092: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_261, unsqueeze_896);  sub_261 = unsqueeze_896 = None
    sub_263: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(where_38, mul_1092);  where_38 = mul_1092 = None
    sub_264: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(sub_263, unsqueeze_893);  sub_263 = unsqueeze_893 = None
    mul_1093: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_264, unsqueeze_899);  sub_264 = unsqueeze_899 = None
    mul_1094: "f32[256]" = torch.ops.aten.mul.Tensor(sum_80, squeeze_196);  sum_80 = squeeze_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    convolution_backward_40 = torch.ops.aten.convolution_backward.default(mul_1093, relu_61, primals_196, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1093 = primals_196 = None
    getitem_348: "f32[8, 512, 14, 14]" = convolution_backward_40[0]
    getitem_349: "f32[256, 512, 1, 1]" = convolution_backward_40[1];  convolution_backward_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    add_594: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_593, getitem_348);  add_593 = getitem_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    le_39: "b8[8, 512, 14, 14]" = torch.ops.aten.le.Scalar(relu_61, 0);  relu_61 = None
    where_39: "f32[8, 512, 14, 14]" = torch.ops.aten.where.self(le_39, full_default, add_594);  le_39 = add_594 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    add_595: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(slice_20, where_39);  slice_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    sum_81: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_39, [0, 2, 3])
    sub_265: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_64, unsqueeze_902);  convolution_64 = unsqueeze_902 = None
    mul_1095: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(where_39, sub_265)
    sum_82: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1095, [0, 2, 3]);  mul_1095 = None
    mul_1096: "f32[512]" = torch.ops.aten.mul.Tensor(sum_81, 0.0006377551020408163)
    unsqueeze_903: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1096, 0);  mul_1096 = None
    unsqueeze_904: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_903, 2);  unsqueeze_903 = None
    unsqueeze_905: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_904, 3);  unsqueeze_904 = None
    mul_1097: "f32[512]" = torch.ops.aten.mul.Tensor(sum_82, 0.0006377551020408163)
    mul_1098: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_193, squeeze_193)
    mul_1099: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1097, mul_1098);  mul_1097 = mul_1098 = None
    unsqueeze_906: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1099, 0);  mul_1099 = None
    unsqueeze_907: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_906, 2);  unsqueeze_906 = None
    unsqueeze_908: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_907, 3);  unsqueeze_907 = None
    mul_1100: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_193, primals_194);  primals_194 = None
    unsqueeze_909: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1100, 0);  mul_1100 = None
    unsqueeze_910: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_909, 2);  unsqueeze_909 = None
    unsqueeze_911: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_910, 3);  unsqueeze_910 = None
    mul_1101: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_265, unsqueeze_908);  sub_265 = unsqueeze_908 = None
    sub_267: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(where_39, mul_1101);  where_39 = mul_1101 = None
    sub_268: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(sub_267, unsqueeze_905);  sub_267 = unsqueeze_905 = None
    mul_1102: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_268, unsqueeze_911);  sub_268 = unsqueeze_911 = None
    mul_1103: "f32[512]" = torch.ops.aten.mul.Tensor(sum_82, squeeze_193);  sum_82 = squeeze_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    convolution_backward_41 = torch.ops.aten.convolution_backward.default(mul_1102, relu_60, primals_193, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1102 = primals_193 = None
    getitem_351: "f32[8, 256, 14, 14]" = convolution_backward_41[0]
    getitem_352: "f32[512, 256, 1, 1]" = convolution_backward_41[1];  convolution_backward_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    le_40: "b8[8, 256, 14, 14]" = torch.ops.aten.le.Scalar(relu_60, 0);  relu_60 = None
    where_40: "f32[8, 256, 14, 14]" = torch.ops.aten.where.self(le_40, full_default, getitem_351);  le_40 = getitem_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    sum_83: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_40, [0, 2, 3])
    sub_269: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_63, unsqueeze_914);  convolution_63 = unsqueeze_914 = None
    mul_1104: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_40, sub_269)
    sum_84: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1104, [0, 2, 3]);  mul_1104 = None
    mul_1105: "f32[256]" = torch.ops.aten.mul.Tensor(sum_83, 0.0006377551020408163)
    unsqueeze_915: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1105, 0);  mul_1105 = None
    unsqueeze_916: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_915, 2);  unsqueeze_915 = None
    unsqueeze_917: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_916, 3);  unsqueeze_916 = None
    mul_1106: "f32[256]" = torch.ops.aten.mul.Tensor(sum_84, 0.0006377551020408163)
    mul_1107: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_190, squeeze_190)
    mul_1108: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1106, mul_1107);  mul_1106 = mul_1107 = None
    unsqueeze_918: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1108, 0);  mul_1108 = None
    unsqueeze_919: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_918, 2);  unsqueeze_918 = None
    unsqueeze_920: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_919, 3);  unsqueeze_919 = None
    mul_1109: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_190, primals_191);  primals_191 = None
    unsqueeze_921: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1109, 0);  mul_1109 = None
    unsqueeze_922: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_921, 2);  unsqueeze_921 = None
    unsqueeze_923: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_922, 3);  unsqueeze_922 = None
    mul_1110: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_269, unsqueeze_920);  sub_269 = unsqueeze_920 = None
    sub_271: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(where_40, mul_1110);  where_40 = mul_1110 = None
    sub_272: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(sub_271, unsqueeze_917);  sub_271 = unsqueeze_917 = None
    mul_1111: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_272, unsqueeze_923);  sub_272 = unsqueeze_923 = None
    mul_1112: "f32[256]" = torch.ops.aten.mul.Tensor(sum_84, squeeze_190);  sum_84 = squeeze_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    convolution_backward_42 = torch.ops.aten.convolution_backward.default(mul_1111, relu_59, primals_190, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1111 = primals_190 = None
    getitem_354: "f32[8, 256, 14, 14]" = convolution_backward_42[0]
    getitem_355: "f32[256, 256, 3, 3]" = convolution_backward_42[1];  convolution_backward_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    le_41: "b8[8, 256, 14, 14]" = torch.ops.aten.le.Scalar(relu_59, 0);  relu_59 = None
    where_41: "f32[8, 256, 14, 14]" = torch.ops.aten.where.self(le_41, full_default, getitem_354);  le_41 = getitem_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    sum_85: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_41, [0, 2, 3])
    sub_273: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_62, unsqueeze_926);  convolution_62 = unsqueeze_926 = None
    mul_1113: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_41, sub_273)
    sum_86: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1113, [0, 2, 3]);  mul_1113 = None
    mul_1114: "f32[256]" = torch.ops.aten.mul.Tensor(sum_85, 0.0006377551020408163)
    unsqueeze_927: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1114, 0);  mul_1114 = None
    unsqueeze_928: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_927, 2);  unsqueeze_927 = None
    unsqueeze_929: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_928, 3);  unsqueeze_928 = None
    mul_1115: "f32[256]" = torch.ops.aten.mul.Tensor(sum_86, 0.0006377551020408163)
    mul_1116: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_187, squeeze_187)
    mul_1117: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1115, mul_1116);  mul_1115 = mul_1116 = None
    unsqueeze_930: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1117, 0);  mul_1117 = None
    unsqueeze_931: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_930, 2);  unsqueeze_930 = None
    unsqueeze_932: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_931, 3);  unsqueeze_931 = None
    mul_1118: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_187, primals_188);  primals_188 = None
    unsqueeze_933: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1118, 0);  mul_1118 = None
    unsqueeze_934: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_933, 2);  unsqueeze_933 = None
    unsqueeze_935: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_934, 3);  unsqueeze_934 = None
    mul_1119: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_273, unsqueeze_932);  sub_273 = unsqueeze_932 = None
    sub_275: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(where_41, mul_1119);  where_41 = mul_1119 = None
    sub_276: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(sub_275, unsqueeze_929);  sub_275 = unsqueeze_929 = None
    mul_1120: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_276, unsqueeze_935);  sub_276 = unsqueeze_935 = None
    mul_1121: "f32[256]" = torch.ops.aten.mul.Tensor(sum_86, squeeze_187);  sum_86 = squeeze_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    convolution_backward_43 = torch.ops.aten.convolution_backward.default(mul_1120, relu_58, primals_187, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1120 = primals_187 = None
    getitem_357: "f32[8, 512, 14, 14]" = convolution_backward_43[0]
    getitem_358: "f32[256, 512, 1, 1]" = convolution_backward_43[1];  convolution_backward_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    add_596: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_595, getitem_357);  add_595 = getitem_357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:180, code: x = self.relu(x)
    le_42: "b8[8, 512, 14, 14]" = torch.ops.aten.le.Scalar(relu_58, 0);  relu_58 = None
    where_42: "f32[8, 512, 14, 14]" = torch.ops.aten.where.self(le_42, full_default, add_596);  le_42 = add_596 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:177, code: x = self.bn(x)
    sum_87: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_42, [0, 2, 3])
    sub_277: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_61, unsqueeze_938);  convolution_61 = unsqueeze_938 = None
    mul_1122: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(where_42, sub_277)
    sum_88: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1122, [0, 2, 3]);  mul_1122 = None
    mul_1123: "f32[512]" = torch.ops.aten.mul.Tensor(sum_87, 0.0006377551020408163)
    unsqueeze_939: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1123, 0);  mul_1123 = None
    unsqueeze_940: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_939, 2);  unsqueeze_939 = None
    unsqueeze_941: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_940, 3);  unsqueeze_940 = None
    mul_1124: "f32[512]" = torch.ops.aten.mul.Tensor(sum_88, 0.0006377551020408163)
    mul_1125: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_184, squeeze_184)
    mul_1126: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1124, mul_1125);  mul_1124 = mul_1125 = None
    unsqueeze_942: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1126, 0);  mul_1126 = None
    unsqueeze_943: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_942, 2);  unsqueeze_942 = None
    unsqueeze_944: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_943, 3);  unsqueeze_943 = None
    mul_1127: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_184, primals_185);  primals_185 = None
    unsqueeze_945: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1127, 0);  mul_1127 = None
    unsqueeze_946: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_945, 2);  unsqueeze_945 = None
    unsqueeze_947: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_946, 3);  unsqueeze_946 = None
    mul_1128: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_277, unsqueeze_944);  sub_277 = unsqueeze_944 = None
    sub_279: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(where_42, mul_1128);  mul_1128 = None
    sub_280: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(sub_279, unsqueeze_941);  sub_279 = unsqueeze_941 = None
    mul_1129: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_280, unsqueeze_947);  sub_280 = unsqueeze_947 = None
    mul_1130: "f32[512]" = torch.ops.aten.mul.Tensor(sum_88, squeeze_184);  sum_88 = squeeze_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:176, code: x = self.conv(torch.cat(x_children, 1))
    convolution_backward_44 = torch.ops.aten.convolution_backward.default(mul_1129, cat_7, primals_184, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1129 = cat_7 = primals_184 = None
    getitem_360: "f32[8, 1024, 14, 14]" = convolution_backward_44[0]
    getitem_361: "f32[512, 1024, 1, 1]" = convolution_backward_44[1];  convolution_backward_44 = None
    slice_21: "f32[8, 512, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_360, 1, 0, 512)
    slice_22: "f32[8, 512, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_360, 1, 512, 1024);  getitem_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:176, code: x = self.conv(torch.cat(x_children, 1))
    add_597: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(where_42, slice_21);  where_42 = slice_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    where_43: "f32[8, 512, 14, 14]" = torch.ops.aten.where.self(le_43, full_default, add_597);  le_43 = add_597 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    add_598: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(slice_22, where_43);  slice_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    sum_89: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_43, [0, 2, 3])
    sub_281: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_60, unsqueeze_950);  convolution_60 = unsqueeze_950 = None
    mul_1131: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(where_43, sub_281)
    sum_90: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1131, [0, 2, 3]);  mul_1131 = None
    mul_1132: "f32[512]" = torch.ops.aten.mul.Tensor(sum_89, 0.0006377551020408163)
    unsqueeze_951: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1132, 0);  mul_1132 = None
    unsqueeze_952: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_951, 2);  unsqueeze_951 = None
    unsqueeze_953: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_952, 3);  unsqueeze_952 = None
    mul_1133: "f32[512]" = torch.ops.aten.mul.Tensor(sum_90, 0.0006377551020408163)
    mul_1134: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_181, squeeze_181)
    mul_1135: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1133, mul_1134);  mul_1133 = mul_1134 = None
    unsqueeze_954: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1135, 0);  mul_1135 = None
    unsqueeze_955: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_954, 2);  unsqueeze_954 = None
    unsqueeze_956: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_955, 3);  unsqueeze_955 = None
    mul_1136: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_181, primals_182);  primals_182 = None
    unsqueeze_957: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1136, 0);  mul_1136 = None
    unsqueeze_958: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_957, 2);  unsqueeze_957 = None
    unsqueeze_959: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_958, 3);  unsqueeze_958 = None
    mul_1137: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_281, unsqueeze_956);  sub_281 = unsqueeze_956 = None
    sub_283: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(where_43, mul_1137);  where_43 = mul_1137 = None
    sub_284: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(sub_283, unsqueeze_953);  sub_283 = unsqueeze_953 = None
    mul_1138: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_284, unsqueeze_959);  sub_284 = unsqueeze_959 = None
    mul_1139: "f32[512]" = torch.ops.aten.mul.Tensor(sum_90, squeeze_181);  sum_90 = squeeze_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    convolution_backward_45 = torch.ops.aten.convolution_backward.default(mul_1138, relu_56, primals_181, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1138 = primals_181 = None
    getitem_363: "f32[8, 256, 14, 14]" = convolution_backward_45[0]
    getitem_364: "f32[512, 256, 1, 1]" = convolution_backward_45[1];  convolution_backward_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    le_44: "b8[8, 256, 14, 14]" = torch.ops.aten.le.Scalar(relu_56, 0);  relu_56 = None
    where_44: "f32[8, 256, 14, 14]" = torch.ops.aten.where.self(le_44, full_default, getitem_363);  le_44 = getitem_363 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    sum_91: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_44, [0, 2, 3])
    sub_285: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_59, unsqueeze_962);  convolution_59 = unsqueeze_962 = None
    mul_1140: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_44, sub_285)
    sum_92: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1140, [0, 2, 3]);  mul_1140 = None
    mul_1141: "f32[256]" = torch.ops.aten.mul.Tensor(sum_91, 0.0006377551020408163)
    unsqueeze_963: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1141, 0);  mul_1141 = None
    unsqueeze_964: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_963, 2);  unsqueeze_963 = None
    unsqueeze_965: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_964, 3);  unsqueeze_964 = None
    mul_1142: "f32[256]" = torch.ops.aten.mul.Tensor(sum_92, 0.0006377551020408163)
    mul_1143: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_178, squeeze_178)
    mul_1144: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1142, mul_1143);  mul_1142 = mul_1143 = None
    unsqueeze_966: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1144, 0);  mul_1144 = None
    unsqueeze_967: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_966, 2);  unsqueeze_966 = None
    unsqueeze_968: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_967, 3);  unsqueeze_967 = None
    mul_1145: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_178, primals_179);  primals_179 = None
    unsqueeze_969: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1145, 0);  mul_1145 = None
    unsqueeze_970: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_969, 2);  unsqueeze_969 = None
    unsqueeze_971: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_970, 3);  unsqueeze_970 = None
    mul_1146: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_285, unsqueeze_968);  sub_285 = unsqueeze_968 = None
    sub_287: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(where_44, mul_1146);  where_44 = mul_1146 = None
    sub_288: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(sub_287, unsqueeze_965);  sub_287 = unsqueeze_965 = None
    mul_1147: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_288, unsqueeze_971);  sub_288 = unsqueeze_971 = None
    mul_1148: "f32[256]" = torch.ops.aten.mul.Tensor(sum_92, squeeze_178);  sum_92 = squeeze_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    convolution_backward_46 = torch.ops.aten.convolution_backward.default(mul_1147, relu_55, primals_178, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1147 = primals_178 = None
    getitem_366: "f32[8, 256, 14, 14]" = convolution_backward_46[0]
    getitem_367: "f32[256, 256, 3, 3]" = convolution_backward_46[1];  convolution_backward_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    le_45: "b8[8, 256, 14, 14]" = torch.ops.aten.le.Scalar(relu_55, 0);  relu_55 = None
    where_45: "f32[8, 256, 14, 14]" = torch.ops.aten.where.self(le_45, full_default, getitem_366);  le_45 = getitem_366 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    sum_93: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_45, [0, 2, 3])
    sub_289: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_58, unsqueeze_974);  convolution_58 = unsqueeze_974 = None
    mul_1149: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_45, sub_289)
    sum_94: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1149, [0, 2, 3]);  mul_1149 = None
    mul_1150: "f32[256]" = torch.ops.aten.mul.Tensor(sum_93, 0.0006377551020408163)
    unsqueeze_975: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1150, 0);  mul_1150 = None
    unsqueeze_976: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_975, 2);  unsqueeze_975 = None
    unsqueeze_977: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_976, 3);  unsqueeze_976 = None
    mul_1151: "f32[256]" = torch.ops.aten.mul.Tensor(sum_94, 0.0006377551020408163)
    mul_1152: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_175, squeeze_175)
    mul_1153: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1151, mul_1152);  mul_1151 = mul_1152 = None
    unsqueeze_978: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1153, 0);  mul_1153 = None
    unsqueeze_979: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_978, 2);  unsqueeze_978 = None
    unsqueeze_980: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_979, 3);  unsqueeze_979 = None
    mul_1154: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_175, primals_176);  primals_176 = None
    unsqueeze_981: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1154, 0);  mul_1154 = None
    unsqueeze_982: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_981, 2);  unsqueeze_981 = None
    unsqueeze_983: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_982, 3);  unsqueeze_982 = None
    mul_1155: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_289, unsqueeze_980);  sub_289 = unsqueeze_980 = None
    sub_291: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(where_45, mul_1155);  where_45 = mul_1155 = None
    sub_292: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(sub_291, unsqueeze_977);  sub_291 = unsqueeze_977 = None
    mul_1156: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_292, unsqueeze_983);  sub_292 = unsqueeze_983 = None
    mul_1157: "f32[256]" = torch.ops.aten.mul.Tensor(sum_94, squeeze_175);  sum_94 = squeeze_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    convolution_backward_47 = torch.ops.aten.convolution_backward.default(mul_1156, relu_54, primals_175, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1156 = primals_175 = None
    getitem_369: "f32[8, 512, 14, 14]" = convolution_backward_47[0]
    getitem_370: "f32[256, 512, 1, 1]" = convolution_backward_47[1];  convolution_backward_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    add_599: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_598, getitem_369);  add_598 = getitem_369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    le_46: "b8[8, 512, 14, 14]" = torch.ops.aten.le.Scalar(relu_54, 0);  relu_54 = None
    where_46: "f32[8, 512, 14, 14]" = torch.ops.aten.where.self(le_46, full_default, add_599);  le_46 = add_599 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    add_600: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(slice_19, where_46);  slice_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    sum_95: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_46, [0, 2, 3])
    sub_293: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_57, unsqueeze_986);  convolution_57 = unsqueeze_986 = None
    mul_1158: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(where_46, sub_293)
    sum_96: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1158, [0, 2, 3]);  mul_1158 = None
    mul_1159: "f32[512]" = torch.ops.aten.mul.Tensor(sum_95, 0.0006377551020408163)
    unsqueeze_987: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1159, 0);  mul_1159 = None
    unsqueeze_988: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_987, 2);  unsqueeze_987 = None
    unsqueeze_989: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_988, 3);  unsqueeze_988 = None
    mul_1160: "f32[512]" = torch.ops.aten.mul.Tensor(sum_96, 0.0006377551020408163)
    mul_1161: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_172, squeeze_172)
    mul_1162: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1160, mul_1161);  mul_1160 = mul_1161 = None
    unsqueeze_990: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1162, 0);  mul_1162 = None
    unsqueeze_991: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_990, 2);  unsqueeze_990 = None
    unsqueeze_992: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_991, 3);  unsqueeze_991 = None
    mul_1163: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_172, primals_173);  primals_173 = None
    unsqueeze_993: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1163, 0);  mul_1163 = None
    unsqueeze_994: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_993, 2);  unsqueeze_993 = None
    unsqueeze_995: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_994, 3);  unsqueeze_994 = None
    mul_1164: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_293, unsqueeze_992);  sub_293 = unsqueeze_992 = None
    sub_295: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(where_46, mul_1164);  where_46 = mul_1164 = None
    sub_296: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(sub_295, unsqueeze_989);  sub_295 = unsqueeze_989 = None
    mul_1165: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_296, unsqueeze_995);  sub_296 = unsqueeze_995 = None
    mul_1166: "f32[512]" = torch.ops.aten.mul.Tensor(sum_96, squeeze_172);  sum_96 = squeeze_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    convolution_backward_48 = torch.ops.aten.convolution_backward.default(mul_1165, relu_53, primals_172, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1165 = primals_172 = None
    getitem_372: "f32[8, 256, 14, 14]" = convolution_backward_48[0]
    getitem_373: "f32[512, 256, 1, 1]" = convolution_backward_48[1];  convolution_backward_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    le_47: "b8[8, 256, 14, 14]" = torch.ops.aten.le.Scalar(relu_53, 0);  relu_53 = None
    where_47: "f32[8, 256, 14, 14]" = torch.ops.aten.where.self(le_47, full_default, getitem_372);  le_47 = getitem_372 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    sum_97: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_47, [0, 2, 3])
    sub_297: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_56, unsqueeze_998);  convolution_56 = unsqueeze_998 = None
    mul_1167: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_47, sub_297)
    sum_98: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1167, [0, 2, 3]);  mul_1167 = None
    mul_1168: "f32[256]" = torch.ops.aten.mul.Tensor(sum_97, 0.0006377551020408163)
    unsqueeze_999: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1168, 0);  mul_1168 = None
    unsqueeze_1000: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_999, 2);  unsqueeze_999 = None
    unsqueeze_1001: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1000, 3);  unsqueeze_1000 = None
    mul_1169: "f32[256]" = torch.ops.aten.mul.Tensor(sum_98, 0.0006377551020408163)
    mul_1170: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_169, squeeze_169)
    mul_1171: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1169, mul_1170);  mul_1169 = mul_1170 = None
    unsqueeze_1002: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1171, 0);  mul_1171 = None
    unsqueeze_1003: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1002, 2);  unsqueeze_1002 = None
    unsqueeze_1004: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1003, 3);  unsqueeze_1003 = None
    mul_1172: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_169, primals_170);  primals_170 = None
    unsqueeze_1005: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1172, 0);  mul_1172 = None
    unsqueeze_1006: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1005, 2);  unsqueeze_1005 = None
    unsqueeze_1007: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1006, 3);  unsqueeze_1006 = None
    mul_1173: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_297, unsqueeze_1004);  sub_297 = unsqueeze_1004 = None
    sub_299: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(where_47, mul_1173);  where_47 = mul_1173 = None
    sub_300: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(sub_299, unsqueeze_1001);  sub_299 = unsqueeze_1001 = None
    mul_1174: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_300, unsqueeze_1007);  sub_300 = unsqueeze_1007 = None
    mul_1175: "f32[256]" = torch.ops.aten.mul.Tensor(sum_98, squeeze_169);  sum_98 = squeeze_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    convolution_backward_49 = torch.ops.aten.convolution_backward.default(mul_1174, relu_52, primals_169, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1174 = primals_169 = None
    getitem_375: "f32[8, 256, 14, 14]" = convolution_backward_49[0]
    getitem_376: "f32[256, 256, 3, 3]" = convolution_backward_49[1];  convolution_backward_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    le_48: "b8[8, 256, 14, 14]" = torch.ops.aten.le.Scalar(relu_52, 0);  relu_52 = None
    where_48: "f32[8, 256, 14, 14]" = torch.ops.aten.where.self(le_48, full_default, getitem_375);  le_48 = getitem_375 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    sum_99: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_48, [0, 2, 3])
    sub_301: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_55, unsqueeze_1010);  convolution_55 = unsqueeze_1010 = None
    mul_1176: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_48, sub_301)
    sum_100: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1176, [0, 2, 3]);  mul_1176 = None
    mul_1177: "f32[256]" = torch.ops.aten.mul.Tensor(sum_99, 0.0006377551020408163)
    unsqueeze_1011: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1177, 0);  mul_1177 = None
    unsqueeze_1012: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1011, 2);  unsqueeze_1011 = None
    unsqueeze_1013: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1012, 3);  unsqueeze_1012 = None
    mul_1178: "f32[256]" = torch.ops.aten.mul.Tensor(sum_100, 0.0006377551020408163)
    mul_1179: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_166, squeeze_166)
    mul_1180: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1178, mul_1179);  mul_1178 = mul_1179 = None
    unsqueeze_1014: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1180, 0);  mul_1180 = None
    unsqueeze_1015: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1014, 2);  unsqueeze_1014 = None
    unsqueeze_1016: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1015, 3);  unsqueeze_1015 = None
    mul_1181: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_166, primals_167);  primals_167 = None
    unsqueeze_1017: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1181, 0);  mul_1181 = None
    unsqueeze_1018: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1017, 2);  unsqueeze_1017 = None
    unsqueeze_1019: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1018, 3);  unsqueeze_1018 = None
    mul_1182: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_301, unsqueeze_1016);  sub_301 = unsqueeze_1016 = None
    sub_303: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(where_48, mul_1182);  where_48 = mul_1182 = None
    sub_304: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(sub_303, unsqueeze_1013);  sub_303 = unsqueeze_1013 = None
    mul_1183: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_304, unsqueeze_1019);  sub_304 = unsqueeze_1019 = None
    mul_1184: "f32[256]" = torch.ops.aten.mul.Tensor(sum_100, squeeze_166);  sum_100 = squeeze_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    convolution_backward_50 = torch.ops.aten.convolution_backward.default(mul_1183, relu_51, primals_166, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1183 = primals_166 = None
    getitem_378: "f32[8, 512, 14, 14]" = convolution_backward_50[0]
    getitem_379: "f32[256, 512, 1, 1]" = convolution_backward_50[1];  convolution_backward_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    add_601: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_600, getitem_378);  add_600 = getitem_378 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:180, code: x = self.relu(x)
    le_49: "b8[8, 512, 14, 14]" = torch.ops.aten.le.Scalar(relu_51, 0);  relu_51 = None
    where_49: "f32[8, 512, 14, 14]" = torch.ops.aten.where.self(le_49, full_default, add_601);  le_49 = add_601 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:177, code: x = self.bn(x)
    sum_101: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_49, [0, 2, 3])
    sub_305: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_54, unsqueeze_1022);  convolution_54 = unsqueeze_1022 = None
    mul_1185: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(where_49, sub_305)
    sum_102: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1185, [0, 2, 3]);  mul_1185 = None
    mul_1186: "f32[512]" = torch.ops.aten.mul.Tensor(sum_101, 0.0006377551020408163)
    unsqueeze_1023: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1186, 0);  mul_1186 = None
    unsqueeze_1024: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1023, 2);  unsqueeze_1023 = None
    unsqueeze_1025: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1024, 3);  unsqueeze_1024 = None
    mul_1187: "f32[512]" = torch.ops.aten.mul.Tensor(sum_102, 0.0006377551020408163)
    mul_1188: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_163, squeeze_163)
    mul_1189: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1187, mul_1188);  mul_1187 = mul_1188 = None
    unsqueeze_1026: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1189, 0);  mul_1189 = None
    unsqueeze_1027: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1026, 2);  unsqueeze_1026 = None
    unsqueeze_1028: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1027, 3);  unsqueeze_1027 = None
    mul_1190: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_163, primals_164);  primals_164 = None
    unsqueeze_1029: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1190, 0);  mul_1190 = None
    unsqueeze_1030: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1029, 2);  unsqueeze_1029 = None
    unsqueeze_1031: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1030, 3);  unsqueeze_1030 = None
    mul_1191: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_305, unsqueeze_1028);  sub_305 = unsqueeze_1028 = None
    sub_307: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(where_49, mul_1191);  mul_1191 = None
    sub_308: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(sub_307, unsqueeze_1025);  sub_307 = unsqueeze_1025 = None
    mul_1192: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_308, unsqueeze_1031);  sub_308 = unsqueeze_1031 = None
    mul_1193: "f32[512]" = torch.ops.aten.mul.Tensor(sum_102, squeeze_163);  sum_102 = squeeze_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:176, code: x = self.conv(torch.cat(x_children, 1))
    convolution_backward_51 = torch.ops.aten.convolution_backward.default(mul_1192, cat_6, primals_163, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1192 = cat_6 = primals_163 = None
    getitem_381: "f32[8, 1536, 14, 14]" = convolution_backward_51[0]
    getitem_382: "f32[512, 1536, 1, 1]" = convolution_backward_51[1];  convolution_backward_51 = None
    slice_23: "f32[8, 512, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_381, 1, 0, 512)
    slice_24: "f32[8, 512, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_381, 1, 512, 1024)
    slice_25: "f32[8, 512, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_381, 1, 1024, 1536);  getitem_381 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:176, code: x = self.conv(torch.cat(x_children, 1))
    add_602: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(where_49, slice_23);  where_49 = slice_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    where_50: "f32[8, 512, 14, 14]" = torch.ops.aten.where.self(le_50, full_default, add_602);  le_50 = add_602 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    add_603: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(slice_24, where_50);  slice_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    sum_103: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_50, [0, 2, 3])
    sub_309: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_53, unsqueeze_1034);  convolution_53 = unsqueeze_1034 = None
    mul_1194: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(where_50, sub_309)
    sum_104: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1194, [0, 2, 3]);  mul_1194 = None
    mul_1195: "f32[512]" = torch.ops.aten.mul.Tensor(sum_103, 0.0006377551020408163)
    unsqueeze_1035: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1195, 0);  mul_1195 = None
    unsqueeze_1036: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1035, 2);  unsqueeze_1035 = None
    unsqueeze_1037: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1036, 3);  unsqueeze_1036 = None
    mul_1196: "f32[512]" = torch.ops.aten.mul.Tensor(sum_104, 0.0006377551020408163)
    mul_1197: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_160, squeeze_160)
    mul_1198: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1196, mul_1197);  mul_1196 = mul_1197 = None
    unsqueeze_1038: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1198, 0);  mul_1198 = None
    unsqueeze_1039: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1038, 2);  unsqueeze_1038 = None
    unsqueeze_1040: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1039, 3);  unsqueeze_1039 = None
    mul_1199: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_160, primals_161);  primals_161 = None
    unsqueeze_1041: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1199, 0);  mul_1199 = None
    unsqueeze_1042: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1041, 2);  unsqueeze_1041 = None
    unsqueeze_1043: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1042, 3);  unsqueeze_1042 = None
    mul_1200: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_309, unsqueeze_1040);  sub_309 = unsqueeze_1040 = None
    sub_311: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(where_50, mul_1200);  where_50 = mul_1200 = None
    sub_312: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(sub_311, unsqueeze_1037);  sub_311 = unsqueeze_1037 = None
    mul_1201: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_312, unsqueeze_1043);  sub_312 = unsqueeze_1043 = None
    mul_1202: "f32[512]" = torch.ops.aten.mul.Tensor(sum_104, squeeze_160);  sum_104 = squeeze_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    convolution_backward_52 = torch.ops.aten.convolution_backward.default(mul_1201, relu_49, primals_160, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1201 = primals_160 = None
    getitem_384: "f32[8, 256, 14, 14]" = convolution_backward_52[0]
    getitem_385: "f32[512, 256, 1, 1]" = convolution_backward_52[1];  convolution_backward_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    le_51: "b8[8, 256, 14, 14]" = torch.ops.aten.le.Scalar(relu_49, 0);  relu_49 = None
    where_51: "f32[8, 256, 14, 14]" = torch.ops.aten.where.self(le_51, full_default, getitem_384);  le_51 = getitem_384 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    sum_105: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_51, [0, 2, 3])
    sub_313: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_52, unsqueeze_1046);  convolution_52 = unsqueeze_1046 = None
    mul_1203: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_51, sub_313)
    sum_106: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1203, [0, 2, 3]);  mul_1203 = None
    mul_1204: "f32[256]" = torch.ops.aten.mul.Tensor(sum_105, 0.0006377551020408163)
    unsqueeze_1047: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1204, 0);  mul_1204 = None
    unsqueeze_1048: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1047, 2);  unsqueeze_1047 = None
    unsqueeze_1049: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1048, 3);  unsqueeze_1048 = None
    mul_1205: "f32[256]" = torch.ops.aten.mul.Tensor(sum_106, 0.0006377551020408163)
    mul_1206: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_157, squeeze_157)
    mul_1207: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1205, mul_1206);  mul_1205 = mul_1206 = None
    unsqueeze_1050: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1207, 0);  mul_1207 = None
    unsqueeze_1051: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1050, 2);  unsqueeze_1050 = None
    unsqueeze_1052: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1051, 3);  unsqueeze_1051 = None
    mul_1208: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_157, primals_158);  primals_158 = None
    unsqueeze_1053: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1208, 0);  mul_1208 = None
    unsqueeze_1054: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1053, 2);  unsqueeze_1053 = None
    unsqueeze_1055: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1054, 3);  unsqueeze_1054 = None
    mul_1209: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_313, unsqueeze_1052);  sub_313 = unsqueeze_1052 = None
    sub_315: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(where_51, mul_1209);  where_51 = mul_1209 = None
    sub_316: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(sub_315, unsqueeze_1049);  sub_315 = unsqueeze_1049 = None
    mul_1210: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_316, unsqueeze_1055);  sub_316 = unsqueeze_1055 = None
    mul_1211: "f32[256]" = torch.ops.aten.mul.Tensor(sum_106, squeeze_157);  sum_106 = squeeze_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    convolution_backward_53 = torch.ops.aten.convolution_backward.default(mul_1210, relu_48, primals_157, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1210 = primals_157 = None
    getitem_387: "f32[8, 256, 14, 14]" = convolution_backward_53[0]
    getitem_388: "f32[256, 256, 3, 3]" = convolution_backward_53[1];  convolution_backward_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    le_52: "b8[8, 256, 14, 14]" = torch.ops.aten.le.Scalar(relu_48, 0);  relu_48 = None
    where_52: "f32[8, 256, 14, 14]" = torch.ops.aten.where.self(le_52, full_default, getitem_387);  le_52 = getitem_387 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    sum_107: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_52, [0, 2, 3])
    sub_317: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_1058);  convolution_51 = unsqueeze_1058 = None
    mul_1212: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_52, sub_317)
    sum_108: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1212, [0, 2, 3]);  mul_1212 = None
    mul_1213: "f32[256]" = torch.ops.aten.mul.Tensor(sum_107, 0.0006377551020408163)
    unsqueeze_1059: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1213, 0);  mul_1213 = None
    unsqueeze_1060: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1059, 2);  unsqueeze_1059 = None
    unsqueeze_1061: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1060, 3);  unsqueeze_1060 = None
    mul_1214: "f32[256]" = torch.ops.aten.mul.Tensor(sum_108, 0.0006377551020408163)
    mul_1215: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_154, squeeze_154)
    mul_1216: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1214, mul_1215);  mul_1214 = mul_1215 = None
    unsqueeze_1062: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1216, 0);  mul_1216 = None
    unsqueeze_1063: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1062, 2);  unsqueeze_1062 = None
    unsqueeze_1064: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1063, 3);  unsqueeze_1063 = None
    mul_1217: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_154, primals_155);  primals_155 = None
    unsqueeze_1065: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1217, 0);  mul_1217 = None
    unsqueeze_1066: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1065, 2);  unsqueeze_1065 = None
    unsqueeze_1067: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1066, 3);  unsqueeze_1066 = None
    mul_1218: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_317, unsqueeze_1064);  sub_317 = unsqueeze_1064 = None
    sub_319: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(where_52, mul_1218);  where_52 = mul_1218 = None
    sub_320: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(sub_319, unsqueeze_1061);  sub_319 = unsqueeze_1061 = None
    mul_1219: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_320, unsqueeze_1067);  sub_320 = unsqueeze_1067 = None
    mul_1220: "f32[256]" = torch.ops.aten.mul.Tensor(sum_108, squeeze_154);  sum_108 = squeeze_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    convolution_backward_54 = torch.ops.aten.convolution_backward.default(mul_1219, relu_47, primals_154, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1219 = primals_154 = None
    getitem_390: "f32[8, 512, 14, 14]" = convolution_backward_54[0]
    getitem_391: "f32[256, 512, 1, 1]" = convolution_backward_54[1];  convolution_backward_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    add_604: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_603, getitem_390);  add_603 = getitem_390 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    le_53: "b8[8, 512, 14, 14]" = torch.ops.aten.le.Scalar(relu_47, 0);  relu_47 = None
    where_53: "f32[8, 512, 14, 14]" = torch.ops.aten.where.self(le_53, full_default, add_604);  le_53 = add_604 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    add_605: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(slice_25, where_53);  slice_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    sum_109: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_53, [0, 2, 3])
    sub_321: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_1070);  convolution_50 = unsqueeze_1070 = None
    mul_1221: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(where_53, sub_321)
    sum_110: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1221, [0, 2, 3]);  mul_1221 = None
    mul_1222: "f32[512]" = torch.ops.aten.mul.Tensor(sum_109, 0.0006377551020408163)
    unsqueeze_1071: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1222, 0);  mul_1222 = None
    unsqueeze_1072: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1071, 2);  unsqueeze_1071 = None
    unsqueeze_1073: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1072, 3);  unsqueeze_1072 = None
    mul_1223: "f32[512]" = torch.ops.aten.mul.Tensor(sum_110, 0.0006377551020408163)
    mul_1224: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_151, squeeze_151)
    mul_1225: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1223, mul_1224);  mul_1223 = mul_1224 = None
    unsqueeze_1074: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1225, 0);  mul_1225 = None
    unsqueeze_1075: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1074, 2);  unsqueeze_1074 = None
    unsqueeze_1076: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1075, 3);  unsqueeze_1075 = None
    mul_1226: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_151, primals_152);  primals_152 = None
    unsqueeze_1077: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1226, 0);  mul_1226 = None
    unsqueeze_1078: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1077, 2);  unsqueeze_1077 = None
    unsqueeze_1079: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1078, 3);  unsqueeze_1078 = None
    mul_1227: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_321, unsqueeze_1076);  sub_321 = unsqueeze_1076 = None
    sub_323: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(where_53, mul_1227);  where_53 = mul_1227 = None
    sub_324: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(sub_323, unsqueeze_1073);  sub_323 = unsqueeze_1073 = None
    mul_1228: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_324, unsqueeze_1079);  sub_324 = unsqueeze_1079 = None
    mul_1229: "f32[512]" = torch.ops.aten.mul.Tensor(sum_110, squeeze_151);  sum_110 = squeeze_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    convolution_backward_55 = torch.ops.aten.convolution_backward.default(mul_1228, relu_46, primals_151, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1228 = primals_151 = None
    getitem_393: "f32[8, 256, 14, 14]" = convolution_backward_55[0]
    getitem_394: "f32[512, 256, 1, 1]" = convolution_backward_55[1];  convolution_backward_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    le_54: "b8[8, 256, 14, 14]" = torch.ops.aten.le.Scalar(relu_46, 0);  relu_46 = None
    where_54: "f32[8, 256, 14, 14]" = torch.ops.aten.where.self(le_54, full_default, getitem_393);  le_54 = getitem_393 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    sum_111: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_54, [0, 2, 3])
    sub_325: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_49, unsqueeze_1082);  convolution_49 = unsqueeze_1082 = None
    mul_1230: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_54, sub_325)
    sum_112: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1230, [0, 2, 3]);  mul_1230 = None
    mul_1231: "f32[256]" = torch.ops.aten.mul.Tensor(sum_111, 0.0006377551020408163)
    unsqueeze_1083: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1231, 0);  mul_1231 = None
    unsqueeze_1084: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1083, 2);  unsqueeze_1083 = None
    unsqueeze_1085: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1084, 3);  unsqueeze_1084 = None
    mul_1232: "f32[256]" = torch.ops.aten.mul.Tensor(sum_112, 0.0006377551020408163)
    mul_1233: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_148, squeeze_148)
    mul_1234: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1232, mul_1233);  mul_1232 = mul_1233 = None
    unsqueeze_1086: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1234, 0);  mul_1234 = None
    unsqueeze_1087: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1086, 2);  unsqueeze_1086 = None
    unsqueeze_1088: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1087, 3);  unsqueeze_1087 = None
    mul_1235: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_148, primals_149);  primals_149 = None
    unsqueeze_1089: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1235, 0);  mul_1235 = None
    unsqueeze_1090: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1089, 2);  unsqueeze_1089 = None
    unsqueeze_1091: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1090, 3);  unsqueeze_1090 = None
    mul_1236: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_325, unsqueeze_1088);  sub_325 = unsqueeze_1088 = None
    sub_327: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(where_54, mul_1236);  where_54 = mul_1236 = None
    sub_328: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(sub_327, unsqueeze_1085);  sub_327 = unsqueeze_1085 = None
    mul_1237: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_328, unsqueeze_1091);  sub_328 = unsqueeze_1091 = None
    mul_1238: "f32[256]" = torch.ops.aten.mul.Tensor(sum_112, squeeze_148);  sum_112 = squeeze_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    convolution_backward_56 = torch.ops.aten.convolution_backward.default(mul_1237, relu_45, primals_148, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1237 = primals_148 = None
    getitem_396: "f32[8, 256, 14, 14]" = convolution_backward_56[0]
    getitem_397: "f32[256, 256, 3, 3]" = convolution_backward_56[1];  convolution_backward_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    le_55: "b8[8, 256, 14, 14]" = torch.ops.aten.le.Scalar(relu_45, 0);  relu_45 = None
    where_55: "f32[8, 256, 14, 14]" = torch.ops.aten.where.self(le_55, full_default, getitem_396);  le_55 = getitem_396 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    sum_113: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_55, [0, 2, 3])
    sub_329: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_48, unsqueeze_1094);  convolution_48 = unsqueeze_1094 = None
    mul_1239: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_55, sub_329)
    sum_114: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1239, [0, 2, 3]);  mul_1239 = None
    mul_1240: "f32[256]" = torch.ops.aten.mul.Tensor(sum_113, 0.0006377551020408163)
    unsqueeze_1095: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1240, 0);  mul_1240 = None
    unsqueeze_1096: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1095, 2);  unsqueeze_1095 = None
    unsqueeze_1097: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1096, 3);  unsqueeze_1096 = None
    mul_1241: "f32[256]" = torch.ops.aten.mul.Tensor(sum_114, 0.0006377551020408163)
    mul_1242: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_145, squeeze_145)
    mul_1243: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1241, mul_1242);  mul_1241 = mul_1242 = None
    unsqueeze_1098: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1243, 0);  mul_1243 = None
    unsqueeze_1099: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1098, 2);  unsqueeze_1098 = None
    unsqueeze_1100: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1099, 3);  unsqueeze_1099 = None
    mul_1244: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_145, primals_146);  primals_146 = None
    unsqueeze_1101: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1244, 0);  mul_1244 = None
    unsqueeze_1102: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1101, 2);  unsqueeze_1101 = None
    unsqueeze_1103: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1102, 3);  unsqueeze_1102 = None
    mul_1245: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_329, unsqueeze_1100);  sub_329 = unsqueeze_1100 = None
    sub_331: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(where_55, mul_1245);  where_55 = mul_1245 = None
    sub_332: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(sub_331, unsqueeze_1097);  sub_331 = unsqueeze_1097 = None
    mul_1246: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_332, unsqueeze_1103);  sub_332 = unsqueeze_1103 = None
    mul_1247: "f32[256]" = torch.ops.aten.mul.Tensor(sum_114, squeeze_145);  sum_114 = squeeze_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    convolution_backward_57 = torch.ops.aten.convolution_backward.default(mul_1246, relu_44, primals_145, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1246 = primals_145 = None
    getitem_399: "f32[8, 512, 14, 14]" = convolution_backward_57[0]
    getitem_400: "f32[256, 512, 1, 1]" = convolution_backward_57[1];  convolution_backward_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    add_606: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_605, getitem_399);  add_605 = getitem_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:180, code: x = self.relu(x)
    le_56: "b8[8, 512, 14, 14]" = torch.ops.aten.le.Scalar(relu_44, 0);  relu_44 = None
    where_56: "f32[8, 512, 14, 14]" = torch.ops.aten.where.self(le_56, full_default, add_606);  le_56 = add_606 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:177, code: x = self.bn(x)
    sum_115: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_56, [0, 2, 3])
    sub_333: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_47, unsqueeze_1106);  convolution_47 = unsqueeze_1106 = None
    mul_1248: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(where_56, sub_333)
    sum_116: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1248, [0, 2, 3]);  mul_1248 = None
    mul_1249: "f32[512]" = torch.ops.aten.mul.Tensor(sum_115, 0.0006377551020408163)
    unsqueeze_1107: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1249, 0);  mul_1249 = None
    unsqueeze_1108: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1107, 2);  unsqueeze_1107 = None
    unsqueeze_1109: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1108, 3);  unsqueeze_1108 = None
    mul_1250: "f32[512]" = torch.ops.aten.mul.Tensor(sum_116, 0.0006377551020408163)
    mul_1251: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_142, squeeze_142)
    mul_1252: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1250, mul_1251);  mul_1250 = mul_1251 = None
    unsqueeze_1110: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1252, 0);  mul_1252 = None
    unsqueeze_1111: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1110, 2);  unsqueeze_1110 = None
    unsqueeze_1112: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1111, 3);  unsqueeze_1111 = None
    mul_1253: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_142, primals_143);  primals_143 = None
    unsqueeze_1113: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1253, 0);  mul_1253 = None
    unsqueeze_1114: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1113, 2);  unsqueeze_1113 = None
    unsqueeze_1115: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1114, 3);  unsqueeze_1114 = None
    mul_1254: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_333, unsqueeze_1112);  sub_333 = unsqueeze_1112 = None
    sub_335: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(where_56, mul_1254);  mul_1254 = None
    sub_336: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(sub_335, unsqueeze_1109);  sub_335 = unsqueeze_1109 = None
    mul_1255: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_336, unsqueeze_1115);  sub_336 = unsqueeze_1115 = None
    mul_1256: "f32[512]" = torch.ops.aten.mul.Tensor(sum_116, squeeze_142);  sum_116 = squeeze_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:176, code: x = self.conv(torch.cat(x_children, 1))
    convolution_backward_58 = torch.ops.aten.convolution_backward.default(mul_1255, cat_5, primals_142, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1255 = cat_5 = primals_142 = None
    getitem_402: "f32[8, 1024, 14, 14]" = convolution_backward_58[0]
    getitem_403: "f32[512, 1024, 1, 1]" = convolution_backward_58[1];  convolution_backward_58 = None
    slice_26: "f32[8, 512, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_402, 1, 0, 512)
    slice_27: "f32[8, 512, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_402, 1, 512, 1024);  getitem_402 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:176, code: x = self.conv(torch.cat(x_children, 1))
    add_607: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(where_56, slice_26);  where_56 = slice_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    where_57: "f32[8, 512, 14, 14]" = torch.ops.aten.where.self(le_57, full_default, add_607);  le_57 = add_607 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    add_608: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(slice_27, where_57);  slice_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    sum_117: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_57, [0, 2, 3])
    sub_337: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_1118);  convolution_46 = unsqueeze_1118 = None
    mul_1257: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(where_57, sub_337)
    sum_118: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1257, [0, 2, 3]);  mul_1257 = None
    mul_1258: "f32[512]" = torch.ops.aten.mul.Tensor(sum_117, 0.0006377551020408163)
    unsqueeze_1119: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1258, 0);  mul_1258 = None
    unsqueeze_1120: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1119, 2);  unsqueeze_1119 = None
    unsqueeze_1121: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1120, 3);  unsqueeze_1120 = None
    mul_1259: "f32[512]" = torch.ops.aten.mul.Tensor(sum_118, 0.0006377551020408163)
    mul_1260: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_139, squeeze_139)
    mul_1261: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1259, mul_1260);  mul_1259 = mul_1260 = None
    unsqueeze_1122: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1261, 0);  mul_1261 = None
    unsqueeze_1123: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1122, 2);  unsqueeze_1122 = None
    unsqueeze_1124: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1123, 3);  unsqueeze_1123 = None
    mul_1262: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_139, primals_140);  primals_140 = None
    unsqueeze_1125: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1262, 0);  mul_1262 = None
    unsqueeze_1126: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1125, 2);  unsqueeze_1125 = None
    unsqueeze_1127: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1126, 3);  unsqueeze_1126 = None
    mul_1263: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_337, unsqueeze_1124);  sub_337 = unsqueeze_1124 = None
    sub_339: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(where_57, mul_1263);  where_57 = mul_1263 = None
    sub_340: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(sub_339, unsqueeze_1121);  sub_339 = unsqueeze_1121 = None
    mul_1264: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_340, unsqueeze_1127);  sub_340 = unsqueeze_1127 = None
    mul_1265: "f32[512]" = torch.ops.aten.mul.Tensor(sum_118, squeeze_139);  sum_118 = squeeze_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    convolution_backward_59 = torch.ops.aten.convolution_backward.default(mul_1264, relu_42, primals_139, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1264 = primals_139 = None
    getitem_405: "f32[8, 256, 14, 14]" = convolution_backward_59[0]
    getitem_406: "f32[512, 256, 1, 1]" = convolution_backward_59[1];  convolution_backward_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    le_58: "b8[8, 256, 14, 14]" = torch.ops.aten.le.Scalar(relu_42, 0);  relu_42 = None
    where_58: "f32[8, 256, 14, 14]" = torch.ops.aten.where.self(le_58, full_default, getitem_405);  le_58 = getitem_405 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    sum_119: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_58, [0, 2, 3])
    sub_341: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_1130);  convolution_45 = unsqueeze_1130 = None
    mul_1266: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_58, sub_341)
    sum_120: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1266, [0, 2, 3]);  mul_1266 = None
    mul_1267: "f32[256]" = torch.ops.aten.mul.Tensor(sum_119, 0.0006377551020408163)
    unsqueeze_1131: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1267, 0);  mul_1267 = None
    unsqueeze_1132: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1131, 2);  unsqueeze_1131 = None
    unsqueeze_1133: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1132, 3);  unsqueeze_1132 = None
    mul_1268: "f32[256]" = torch.ops.aten.mul.Tensor(sum_120, 0.0006377551020408163)
    mul_1269: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_136, squeeze_136)
    mul_1270: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1268, mul_1269);  mul_1268 = mul_1269 = None
    unsqueeze_1134: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1270, 0);  mul_1270 = None
    unsqueeze_1135: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1134, 2);  unsqueeze_1134 = None
    unsqueeze_1136: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1135, 3);  unsqueeze_1135 = None
    mul_1271: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_136, primals_137);  primals_137 = None
    unsqueeze_1137: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1271, 0);  mul_1271 = None
    unsqueeze_1138: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1137, 2);  unsqueeze_1137 = None
    unsqueeze_1139: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1138, 3);  unsqueeze_1138 = None
    mul_1272: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_341, unsqueeze_1136);  sub_341 = unsqueeze_1136 = None
    sub_343: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(where_58, mul_1272);  where_58 = mul_1272 = None
    sub_344: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(sub_343, unsqueeze_1133);  sub_343 = unsqueeze_1133 = None
    mul_1273: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_344, unsqueeze_1139);  sub_344 = unsqueeze_1139 = None
    mul_1274: "f32[256]" = torch.ops.aten.mul.Tensor(sum_120, squeeze_136);  sum_120 = squeeze_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    convolution_backward_60 = torch.ops.aten.convolution_backward.default(mul_1273, relu_41, primals_136, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1273 = primals_136 = None
    getitem_408: "f32[8, 256, 14, 14]" = convolution_backward_60[0]
    getitem_409: "f32[256, 256, 3, 3]" = convolution_backward_60[1];  convolution_backward_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    le_59: "b8[8, 256, 14, 14]" = torch.ops.aten.le.Scalar(relu_41, 0);  relu_41 = None
    where_59: "f32[8, 256, 14, 14]" = torch.ops.aten.where.self(le_59, full_default, getitem_408);  le_59 = getitem_408 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    sum_121: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_59, [0, 2, 3])
    sub_345: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_44, unsqueeze_1142);  convolution_44 = unsqueeze_1142 = None
    mul_1275: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_59, sub_345)
    sum_122: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1275, [0, 2, 3]);  mul_1275 = None
    mul_1276: "f32[256]" = torch.ops.aten.mul.Tensor(sum_121, 0.0006377551020408163)
    unsqueeze_1143: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1276, 0);  mul_1276 = None
    unsqueeze_1144: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1143, 2);  unsqueeze_1143 = None
    unsqueeze_1145: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1144, 3);  unsqueeze_1144 = None
    mul_1277: "f32[256]" = torch.ops.aten.mul.Tensor(sum_122, 0.0006377551020408163)
    mul_1278: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_133, squeeze_133)
    mul_1279: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1277, mul_1278);  mul_1277 = mul_1278 = None
    unsqueeze_1146: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1279, 0);  mul_1279 = None
    unsqueeze_1147: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1146, 2);  unsqueeze_1146 = None
    unsqueeze_1148: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1147, 3);  unsqueeze_1147 = None
    mul_1280: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_133, primals_134);  primals_134 = None
    unsqueeze_1149: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1280, 0);  mul_1280 = None
    unsqueeze_1150: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1149, 2);  unsqueeze_1149 = None
    unsqueeze_1151: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1150, 3);  unsqueeze_1150 = None
    mul_1281: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_345, unsqueeze_1148);  sub_345 = unsqueeze_1148 = None
    sub_347: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(where_59, mul_1281);  where_59 = mul_1281 = None
    sub_348: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(sub_347, unsqueeze_1145);  sub_347 = unsqueeze_1145 = None
    mul_1282: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_348, unsqueeze_1151);  sub_348 = unsqueeze_1151 = None
    mul_1283: "f32[256]" = torch.ops.aten.mul.Tensor(sum_122, squeeze_133);  sum_122 = squeeze_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    convolution_backward_61 = torch.ops.aten.convolution_backward.default(mul_1282, relu_40, primals_133, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1282 = primals_133 = None
    getitem_411: "f32[8, 512, 14, 14]" = convolution_backward_61[0]
    getitem_412: "f32[256, 512, 1, 1]" = convolution_backward_61[1];  convolution_backward_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    add_609: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_608, getitem_411);  add_608 = getitem_411 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    le_60: "b8[8, 512, 14, 14]" = torch.ops.aten.le.Scalar(relu_40, 0);  relu_40 = None
    where_60: "f32[8, 512, 14, 14]" = torch.ops.aten.where.self(le_60, full_default, add_609);  le_60 = add_609 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    sum_123: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_60, [0, 2, 3])
    sub_349: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_43, unsqueeze_1154);  convolution_43 = unsqueeze_1154 = None
    mul_1284: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(where_60, sub_349)
    sum_124: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1284, [0, 2, 3]);  mul_1284 = None
    mul_1285: "f32[512]" = torch.ops.aten.mul.Tensor(sum_123, 0.0006377551020408163)
    unsqueeze_1155: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1285, 0);  mul_1285 = None
    unsqueeze_1156: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1155, 2);  unsqueeze_1155 = None
    unsqueeze_1157: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1156, 3);  unsqueeze_1156 = None
    mul_1286: "f32[512]" = torch.ops.aten.mul.Tensor(sum_124, 0.0006377551020408163)
    mul_1287: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_130, squeeze_130)
    mul_1288: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1286, mul_1287);  mul_1286 = mul_1287 = None
    unsqueeze_1158: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1288, 0);  mul_1288 = None
    unsqueeze_1159: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1158, 2);  unsqueeze_1158 = None
    unsqueeze_1160: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1159, 3);  unsqueeze_1159 = None
    mul_1289: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_130, primals_131);  primals_131 = None
    unsqueeze_1161: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1289, 0);  mul_1289 = None
    unsqueeze_1162: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1161, 2);  unsqueeze_1161 = None
    unsqueeze_1163: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1162, 3);  unsqueeze_1162 = None
    mul_1290: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_349, unsqueeze_1160);  sub_349 = unsqueeze_1160 = None
    sub_351: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(where_60, mul_1290);  mul_1290 = None
    sub_352: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(sub_351, unsqueeze_1157);  sub_351 = None
    mul_1291: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_352, unsqueeze_1163);  sub_352 = unsqueeze_1163 = None
    mul_1292: "f32[512]" = torch.ops.aten.mul.Tensor(sum_124, squeeze_130);  sum_124 = squeeze_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    convolution_backward_62 = torch.ops.aten.convolution_backward.default(mul_1291, relu_39, primals_130, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1291 = primals_130 = None
    getitem_414: "f32[8, 256, 14, 14]" = convolution_backward_62[0]
    getitem_415: "f32[512, 256, 1, 1]" = convolution_backward_62[1];  convolution_backward_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    le_61: "b8[8, 256, 14, 14]" = torch.ops.aten.le.Scalar(relu_39, 0);  relu_39 = None
    where_61: "f32[8, 256, 14, 14]" = torch.ops.aten.where.self(le_61, full_default, getitem_414);  le_61 = getitem_414 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    sum_125: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_61, [0, 2, 3])
    sub_353: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_42, unsqueeze_1166);  convolution_42 = unsqueeze_1166 = None
    mul_1293: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_61, sub_353)
    sum_126: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1293, [0, 2, 3]);  mul_1293 = None
    mul_1294: "f32[256]" = torch.ops.aten.mul.Tensor(sum_125, 0.0006377551020408163)
    unsqueeze_1167: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1294, 0);  mul_1294 = None
    unsqueeze_1168: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1167, 2);  unsqueeze_1167 = None
    unsqueeze_1169: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1168, 3);  unsqueeze_1168 = None
    mul_1295: "f32[256]" = torch.ops.aten.mul.Tensor(sum_126, 0.0006377551020408163)
    mul_1296: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_127, squeeze_127)
    mul_1297: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1295, mul_1296);  mul_1295 = mul_1296 = None
    unsqueeze_1170: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1297, 0);  mul_1297 = None
    unsqueeze_1171: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1170, 2);  unsqueeze_1170 = None
    unsqueeze_1172: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1171, 3);  unsqueeze_1171 = None
    mul_1298: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_127, primals_128);  primals_128 = None
    unsqueeze_1173: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1298, 0);  mul_1298 = None
    unsqueeze_1174: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1173, 2);  unsqueeze_1173 = None
    unsqueeze_1175: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1174, 3);  unsqueeze_1174 = None
    mul_1299: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_353, unsqueeze_1172);  sub_353 = unsqueeze_1172 = None
    sub_355: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(where_61, mul_1299);  where_61 = mul_1299 = None
    sub_356: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(sub_355, unsqueeze_1169);  sub_355 = unsqueeze_1169 = None
    mul_1300: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_356, unsqueeze_1175);  sub_356 = unsqueeze_1175 = None
    mul_1301: "f32[256]" = torch.ops.aten.mul.Tensor(sum_126, squeeze_127);  sum_126 = squeeze_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    convolution_backward_63 = torch.ops.aten.convolution_backward.default(mul_1300, relu_38, primals_127, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1300 = primals_127 = None
    getitem_417: "f32[8, 256, 28, 28]" = convolution_backward_63[0]
    getitem_418: "f32[256, 256, 3, 3]" = convolution_backward_63[1];  convolution_backward_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    le_62: "b8[8, 256, 28, 28]" = torch.ops.aten.le.Scalar(relu_38, 0);  relu_38 = None
    where_62: "f32[8, 256, 28, 28]" = torch.ops.aten.where.self(le_62, full_default, getitem_417);  le_62 = getitem_417 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    sum_127: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_62, [0, 2, 3])
    sub_357: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_1178);  convolution_41 = unsqueeze_1178 = None
    mul_1302: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(where_62, sub_357)
    sum_128: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1302, [0, 2, 3]);  mul_1302 = None
    mul_1303: "f32[256]" = torch.ops.aten.mul.Tensor(sum_127, 0.00015943877551020407)
    unsqueeze_1179: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1303, 0);  mul_1303 = None
    unsqueeze_1180: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1179, 2);  unsqueeze_1179 = None
    unsqueeze_1181: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1180, 3);  unsqueeze_1180 = None
    mul_1304: "f32[256]" = torch.ops.aten.mul.Tensor(sum_128, 0.00015943877551020407)
    mul_1305: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_124, squeeze_124)
    mul_1306: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1304, mul_1305);  mul_1304 = mul_1305 = None
    unsqueeze_1182: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1306, 0);  mul_1306 = None
    unsqueeze_1183: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1182, 2);  unsqueeze_1182 = None
    unsqueeze_1184: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1183, 3);  unsqueeze_1183 = None
    mul_1307: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_124, primals_125);  primals_125 = None
    unsqueeze_1185: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1307, 0);  mul_1307 = None
    unsqueeze_1186: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1185, 2);  unsqueeze_1185 = None
    unsqueeze_1187: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1186, 3);  unsqueeze_1186 = None
    mul_1308: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_357, unsqueeze_1184);  sub_357 = unsqueeze_1184 = None
    sub_359: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(where_62, mul_1308);  where_62 = mul_1308 = None
    sub_360: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(sub_359, unsqueeze_1181);  sub_359 = unsqueeze_1181 = None
    mul_1309: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_360, unsqueeze_1187);  sub_360 = unsqueeze_1187 = None
    mul_1310: "f32[256]" = torch.ops.aten.mul.Tensor(sum_128, squeeze_124);  sum_128 = squeeze_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    convolution_backward_64 = torch.ops.aten.convolution_backward.default(mul_1309, relu_37, primals_124, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1309 = primals_124 = None
    getitem_420: "f32[8, 256, 28, 28]" = convolution_backward_64[0]
    getitem_421: "f32[256, 256, 1, 1]" = convolution_backward_64[1];  convolution_backward_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:248, code: shortcut = self.project(bottom)
    sub_361: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_1190);  convolution_40 = unsqueeze_1190 = None
    mul_1311: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(where_60, sub_361)
    sum_130: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1311, [0, 2, 3]);  mul_1311 = None
    mul_1313: "f32[512]" = torch.ops.aten.mul.Tensor(sum_130, 0.0006377551020408163)
    mul_1314: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_121, squeeze_121)
    mul_1315: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1313, mul_1314);  mul_1313 = mul_1314 = None
    unsqueeze_1194: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1315, 0);  mul_1315 = None
    unsqueeze_1195: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1194, 2);  unsqueeze_1194 = None
    unsqueeze_1196: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1195, 3);  unsqueeze_1195 = None
    mul_1316: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_121, primals_122);  primals_122 = None
    unsqueeze_1197: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1316, 0);  mul_1316 = None
    unsqueeze_1198: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1197, 2);  unsqueeze_1197 = None
    unsqueeze_1199: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1198, 3);  unsqueeze_1198 = None
    mul_1317: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_361, unsqueeze_1196);  sub_361 = unsqueeze_1196 = None
    sub_363: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(where_60, mul_1317);  where_60 = mul_1317 = None
    sub_364: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(sub_363, unsqueeze_1157);  sub_363 = unsqueeze_1157 = None
    mul_1318: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_364, unsqueeze_1199);  sub_364 = unsqueeze_1199 = None
    mul_1319: "f32[512]" = torch.ops.aten.mul.Tensor(sum_130, squeeze_121);  sum_130 = squeeze_121 = None
    convolution_backward_65 = torch.ops.aten.convolution_backward.default(mul_1318, getitem_88, primals_121, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1318 = getitem_88 = primals_121 = None
    getitem_423: "f32[8, 256, 14, 14]" = convolution_backward_65[0]
    getitem_424: "f32[512, 256, 1, 1]" = convolution_backward_65[1];  convolution_backward_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:247, code: bottom = self.downsample(x)
    max_pool2d_with_indices_backward_1: "f32[8, 256, 28, 28]" = torch.ops.aten.max_pool2d_with_indices_backward.default(getitem_423, relu_37, [2, 2], [2, 2], [0, 0], [1, 1], False, getitem_89);  getitem_423 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:247, code: bottom = self.downsample(x)
    add_610: "f32[8, 256, 28, 28]" = torch.ops.aten.add.Tensor(getitem_420, max_pool2d_with_indices_backward_1);  getitem_420 = max_pool2d_with_indices_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:247, code: bottom = self.downsample(x)
    max_pool2d_with_indices_backward_2: "f32[8, 256, 28, 28]" = torch.ops.aten.max_pool2d_with_indices_backward.default(slice_6, relu_37, [2, 2], [2, 2], [0, 0], [1, 1], False, getitem_89);  slice_6 = getitem_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:247, code: bottom = self.downsample(x)
    add_611: "f32[8, 256, 28, 28]" = torch.ops.aten.add.Tensor(add_610, max_pool2d_with_indices_backward_2);  add_610 = max_pool2d_with_indices_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:180, code: x = self.relu(x)
    le_63: "b8[8, 256, 28, 28]" = torch.ops.aten.le.Scalar(relu_37, 0);  relu_37 = None
    where_63: "f32[8, 256, 28, 28]" = torch.ops.aten.where.self(le_63, full_default, add_611);  le_63 = add_611 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:177, code: x = self.bn(x)
    sum_131: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_63, [0, 2, 3])
    sub_365: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_39, unsqueeze_1202);  convolution_39 = unsqueeze_1202 = None
    mul_1320: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(where_63, sub_365)
    sum_132: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1320, [0, 2, 3]);  mul_1320 = None
    mul_1321: "f32[256]" = torch.ops.aten.mul.Tensor(sum_131, 0.00015943877551020407)
    unsqueeze_1203: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1321, 0);  mul_1321 = None
    unsqueeze_1204: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1203, 2);  unsqueeze_1203 = None
    unsqueeze_1205: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1204, 3);  unsqueeze_1204 = None
    mul_1322: "f32[256]" = torch.ops.aten.mul.Tensor(sum_132, 0.00015943877551020407)
    mul_1323: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_118, squeeze_118)
    mul_1324: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1322, mul_1323);  mul_1322 = mul_1323 = None
    unsqueeze_1206: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1324, 0);  mul_1324 = None
    unsqueeze_1207: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1206, 2);  unsqueeze_1206 = None
    unsqueeze_1208: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1207, 3);  unsqueeze_1207 = None
    mul_1325: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_118, primals_119);  primals_119 = None
    unsqueeze_1209: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1325, 0);  mul_1325 = None
    unsqueeze_1210: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1209, 2);  unsqueeze_1209 = None
    unsqueeze_1211: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1210, 3);  unsqueeze_1210 = None
    mul_1326: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_365, unsqueeze_1208);  sub_365 = unsqueeze_1208 = None
    sub_367: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(where_63, mul_1326);  mul_1326 = None
    sub_368: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(sub_367, unsqueeze_1205);  sub_367 = unsqueeze_1205 = None
    mul_1327: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_368, unsqueeze_1211);  sub_368 = unsqueeze_1211 = None
    mul_1328: "f32[256]" = torch.ops.aten.mul.Tensor(sum_132, squeeze_118);  sum_132 = squeeze_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:176, code: x = self.conv(torch.cat(x_children, 1))
    convolution_backward_66 = torch.ops.aten.convolution_backward.default(mul_1327, cat_4, primals_118, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1327 = cat_4 = primals_118 = None
    getitem_426: "f32[8, 1152, 28, 28]" = convolution_backward_66[0]
    getitem_427: "f32[256, 1152, 1, 1]" = convolution_backward_66[1];  convolution_backward_66 = None
    slice_28: "f32[8, 256, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_426, 1, 0, 256)
    slice_29: "f32[8, 256, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_426, 1, 256, 512)
    slice_30: "f32[8, 128, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_426, 1, 512, 640)
    slice_31: "f32[8, 256, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_426, 1, 640, 896)
    slice_32: "f32[8, 256, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_426, 1, 896, 1152);  getitem_426 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:176, code: x = self.conv(torch.cat(x_children, 1))
    add_612: "f32[8, 256, 28, 28]" = torch.ops.aten.add.Tensor(where_63, slice_28);  where_63 = slice_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    where_64: "f32[8, 256, 28, 28]" = torch.ops.aten.where.self(le_64, full_default, add_612);  le_64 = add_612 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    add_613: "f32[8, 256, 28, 28]" = torch.ops.aten.add.Tensor(slice_29, where_64);  slice_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    sum_133: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_64, [0, 2, 3])
    sub_369: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_38, unsqueeze_1214);  convolution_38 = unsqueeze_1214 = None
    mul_1329: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(where_64, sub_369)
    sum_134: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1329, [0, 2, 3]);  mul_1329 = None
    mul_1330: "f32[256]" = torch.ops.aten.mul.Tensor(sum_133, 0.00015943877551020407)
    unsqueeze_1215: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1330, 0);  mul_1330 = None
    unsqueeze_1216: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1215, 2);  unsqueeze_1215 = None
    unsqueeze_1217: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1216, 3);  unsqueeze_1216 = None
    mul_1331: "f32[256]" = torch.ops.aten.mul.Tensor(sum_134, 0.00015943877551020407)
    mul_1332: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_115, squeeze_115)
    mul_1333: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1331, mul_1332);  mul_1331 = mul_1332 = None
    unsqueeze_1218: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1333, 0);  mul_1333 = None
    unsqueeze_1219: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1218, 2);  unsqueeze_1218 = None
    unsqueeze_1220: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1219, 3);  unsqueeze_1219 = None
    mul_1334: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_115, primals_116);  primals_116 = None
    unsqueeze_1221: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1334, 0);  mul_1334 = None
    unsqueeze_1222: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1221, 2);  unsqueeze_1221 = None
    unsqueeze_1223: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1222, 3);  unsqueeze_1222 = None
    mul_1335: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_369, unsqueeze_1220);  sub_369 = unsqueeze_1220 = None
    sub_371: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(where_64, mul_1335);  where_64 = mul_1335 = None
    sub_372: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(sub_371, unsqueeze_1217);  sub_371 = unsqueeze_1217 = None
    mul_1336: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_372, unsqueeze_1223);  sub_372 = unsqueeze_1223 = None
    mul_1337: "f32[256]" = torch.ops.aten.mul.Tensor(sum_134, squeeze_115);  sum_134 = squeeze_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    convolution_backward_67 = torch.ops.aten.convolution_backward.default(mul_1336, relu_35, primals_115, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1336 = primals_115 = None
    getitem_429: "f32[8, 128, 28, 28]" = convolution_backward_67[0]
    getitem_430: "f32[256, 128, 1, 1]" = convolution_backward_67[1];  convolution_backward_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    le_65: "b8[8, 128, 28, 28]" = torch.ops.aten.le.Scalar(relu_35, 0);  relu_35 = None
    where_65: "f32[8, 128, 28, 28]" = torch.ops.aten.where.self(le_65, full_default, getitem_429);  le_65 = getitem_429 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    sum_135: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_65, [0, 2, 3])
    sub_373: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_37, unsqueeze_1226);  convolution_37 = unsqueeze_1226 = None
    mul_1338: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_65, sub_373)
    sum_136: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1338, [0, 2, 3]);  mul_1338 = None
    mul_1339: "f32[128]" = torch.ops.aten.mul.Tensor(sum_135, 0.00015943877551020407)
    unsqueeze_1227: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1339, 0);  mul_1339 = None
    unsqueeze_1228: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1227, 2);  unsqueeze_1227 = None
    unsqueeze_1229: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1228, 3);  unsqueeze_1228 = None
    mul_1340: "f32[128]" = torch.ops.aten.mul.Tensor(sum_136, 0.00015943877551020407)
    mul_1341: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_112, squeeze_112)
    mul_1342: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1340, mul_1341);  mul_1340 = mul_1341 = None
    unsqueeze_1230: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1342, 0);  mul_1342 = None
    unsqueeze_1231: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1230, 2);  unsqueeze_1230 = None
    unsqueeze_1232: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1231, 3);  unsqueeze_1231 = None
    mul_1343: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_112, primals_113);  primals_113 = None
    unsqueeze_1233: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1343, 0);  mul_1343 = None
    unsqueeze_1234: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1233, 2);  unsqueeze_1233 = None
    unsqueeze_1235: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1234, 3);  unsqueeze_1234 = None
    mul_1344: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_373, unsqueeze_1232);  sub_373 = unsqueeze_1232 = None
    sub_375: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(where_65, mul_1344);  where_65 = mul_1344 = None
    sub_376: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(sub_375, unsqueeze_1229);  sub_375 = unsqueeze_1229 = None
    mul_1345: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_376, unsqueeze_1235);  sub_376 = unsqueeze_1235 = None
    mul_1346: "f32[128]" = torch.ops.aten.mul.Tensor(sum_136, squeeze_112);  sum_136 = squeeze_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    convolution_backward_68 = torch.ops.aten.convolution_backward.default(mul_1345, relu_34, primals_112, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1345 = primals_112 = None
    getitem_432: "f32[8, 128, 28, 28]" = convolution_backward_68[0]
    getitem_433: "f32[128, 128, 3, 3]" = convolution_backward_68[1];  convolution_backward_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    le_66: "b8[8, 128, 28, 28]" = torch.ops.aten.le.Scalar(relu_34, 0);  relu_34 = None
    where_66: "f32[8, 128, 28, 28]" = torch.ops.aten.where.self(le_66, full_default, getitem_432);  le_66 = getitem_432 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    sum_137: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_66, [0, 2, 3])
    sub_377: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_1238);  convolution_36 = unsqueeze_1238 = None
    mul_1347: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_66, sub_377)
    sum_138: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1347, [0, 2, 3]);  mul_1347 = None
    mul_1348: "f32[128]" = torch.ops.aten.mul.Tensor(sum_137, 0.00015943877551020407)
    unsqueeze_1239: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1348, 0);  mul_1348 = None
    unsqueeze_1240: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1239, 2);  unsqueeze_1239 = None
    unsqueeze_1241: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1240, 3);  unsqueeze_1240 = None
    mul_1349: "f32[128]" = torch.ops.aten.mul.Tensor(sum_138, 0.00015943877551020407)
    mul_1350: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_109, squeeze_109)
    mul_1351: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1349, mul_1350);  mul_1349 = mul_1350 = None
    unsqueeze_1242: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1351, 0);  mul_1351 = None
    unsqueeze_1243: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1242, 2);  unsqueeze_1242 = None
    unsqueeze_1244: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1243, 3);  unsqueeze_1243 = None
    mul_1352: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_109, primals_110);  primals_110 = None
    unsqueeze_1245: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1352, 0);  mul_1352 = None
    unsqueeze_1246: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1245, 2);  unsqueeze_1245 = None
    unsqueeze_1247: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1246, 3);  unsqueeze_1246 = None
    mul_1353: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_377, unsqueeze_1244);  sub_377 = unsqueeze_1244 = None
    sub_379: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(where_66, mul_1353);  where_66 = mul_1353 = None
    sub_380: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(sub_379, unsqueeze_1241);  sub_379 = unsqueeze_1241 = None
    mul_1354: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_380, unsqueeze_1247);  sub_380 = unsqueeze_1247 = None
    mul_1355: "f32[128]" = torch.ops.aten.mul.Tensor(sum_138, squeeze_109);  sum_138 = squeeze_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    convolution_backward_69 = torch.ops.aten.convolution_backward.default(mul_1354, relu_33, primals_109, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1354 = primals_109 = None
    getitem_435: "f32[8, 256, 28, 28]" = convolution_backward_69[0]
    getitem_436: "f32[128, 256, 1, 1]" = convolution_backward_69[1];  convolution_backward_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    add_614: "f32[8, 256, 28, 28]" = torch.ops.aten.add.Tensor(add_613, getitem_435);  add_613 = getitem_435 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    le_67: "b8[8, 256, 28, 28]" = torch.ops.aten.le.Scalar(relu_33, 0);  relu_33 = None
    where_67: "f32[8, 256, 28, 28]" = torch.ops.aten.where.self(le_67, full_default, add_614);  le_67 = add_614 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    add_615: "f32[8, 256, 28, 28]" = torch.ops.aten.add.Tensor(slice_32, where_67);  slice_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    sum_139: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_67, [0, 2, 3])
    sub_381: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_1250);  convolution_35 = unsqueeze_1250 = None
    mul_1356: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(where_67, sub_381)
    sum_140: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1356, [0, 2, 3]);  mul_1356 = None
    mul_1357: "f32[256]" = torch.ops.aten.mul.Tensor(sum_139, 0.00015943877551020407)
    unsqueeze_1251: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1357, 0);  mul_1357 = None
    unsqueeze_1252: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1251, 2);  unsqueeze_1251 = None
    unsqueeze_1253: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1252, 3);  unsqueeze_1252 = None
    mul_1358: "f32[256]" = torch.ops.aten.mul.Tensor(sum_140, 0.00015943877551020407)
    mul_1359: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_106, squeeze_106)
    mul_1360: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1358, mul_1359);  mul_1358 = mul_1359 = None
    unsqueeze_1254: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1360, 0);  mul_1360 = None
    unsqueeze_1255: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1254, 2);  unsqueeze_1254 = None
    unsqueeze_1256: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1255, 3);  unsqueeze_1255 = None
    mul_1361: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_106, primals_107);  primals_107 = None
    unsqueeze_1257: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1361, 0);  mul_1361 = None
    unsqueeze_1258: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1257, 2);  unsqueeze_1257 = None
    unsqueeze_1259: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1258, 3);  unsqueeze_1258 = None
    mul_1362: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_381, unsqueeze_1256);  sub_381 = unsqueeze_1256 = None
    sub_383: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(where_67, mul_1362);  where_67 = mul_1362 = None
    sub_384: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(sub_383, unsqueeze_1253);  sub_383 = unsqueeze_1253 = None
    mul_1363: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_384, unsqueeze_1259);  sub_384 = unsqueeze_1259 = None
    mul_1364: "f32[256]" = torch.ops.aten.mul.Tensor(sum_140, squeeze_106);  sum_140 = squeeze_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    convolution_backward_70 = torch.ops.aten.convolution_backward.default(mul_1363, relu_32, primals_106, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1363 = primals_106 = None
    getitem_438: "f32[8, 128, 28, 28]" = convolution_backward_70[0]
    getitem_439: "f32[256, 128, 1, 1]" = convolution_backward_70[1];  convolution_backward_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    le_68: "b8[8, 128, 28, 28]" = torch.ops.aten.le.Scalar(relu_32, 0);  relu_32 = None
    where_68: "f32[8, 128, 28, 28]" = torch.ops.aten.where.self(le_68, full_default, getitem_438);  le_68 = getitem_438 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    sum_141: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_68, [0, 2, 3])
    sub_385: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_1262);  convolution_34 = unsqueeze_1262 = None
    mul_1365: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_68, sub_385)
    sum_142: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1365, [0, 2, 3]);  mul_1365 = None
    mul_1366: "f32[128]" = torch.ops.aten.mul.Tensor(sum_141, 0.00015943877551020407)
    unsqueeze_1263: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1366, 0);  mul_1366 = None
    unsqueeze_1264: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1263, 2);  unsqueeze_1263 = None
    unsqueeze_1265: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1264, 3);  unsqueeze_1264 = None
    mul_1367: "f32[128]" = torch.ops.aten.mul.Tensor(sum_142, 0.00015943877551020407)
    mul_1368: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_103, squeeze_103)
    mul_1369: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1367, mul_1368);  mul_1367 = mul_1368 = None
    unsqueeze_1266: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1369, 0);  mul_1369 = None
    unsqueeze_1267: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1266, 2);  unsqueeze_1266 = None
    unsqueeze_1268: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1267, 3);  unsqueeze_1267 = None
    mul_1370: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_103, primals_104);  primals_104 = None
    unsqueeze_1269: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1370, 0);  mul_1370 = None
    unsqueeze_1270: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1269, 2);  unsqueeze_1269 = None
    unsqueeze_1271: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1270, 3);  unsqueeze_1270 = None
    mul_1371: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_385, unsqueeze_1268);  sub_385 = unsqueeze_1268 = None
    sub_387: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(where_68, mul_1371);  where_68 = mul_1371 = None
    sub_388: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(sub_387, unsqueeze_1265);  sub_387 = unsqueeze_1265 = None
    mul_1372: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_388, unsqueeze_1271);  sub_388 = unsqueeze_1271 = None
    mul_1373: "f32[128]" = torch.ops.aten.mul.Tensor(sum_142, squeeze_103);  sum_142 = squeeze_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    convolution_backward_71 = torch.ops.aten.convolution_backward.default(mul_1372, relu_31, primals_103, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1372 = primals_103 = None
    getitem_441: "f32[8, 128, 28, 28]" = convolution_backward_71[0]
    getitem_442: "f32[128, 128, 3, 3]" = convolution_backward_71[1];  convolution_backward_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    le_69: "b8[8, 128, 28, 28]" = torch.ops.aten.le.Scalar(relu_31, 0);  relu_31 = None
    where_69: "f32[8, 128, 28, 28]" = torch.ops.aten.where.self(le_69, full_default, getitem_441);  le_69 = getitem_441 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    sum_143: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_69, [0, 2, 3])
    sub_389: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_1274);  convolution_33 = unsqueeze_1274 = None
    mul_1374: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_69, sub_389)
    sum_144: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1374, [0, 2, 3]);  mul_1374 = None
    mul_1375: "f32[128]" = torch.ops.aten.mul.Tensor(sum_143, 0.00015943877551020407)
    unsqueeze_1275: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1375, 0);  mul_1375 = None
    unsqueeze_1276: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1275, 2);  unsqueeze_1275 = None
    unsqueeze_1277: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1276, 3);  unsqueeze_1276 = None
    mul_1376: "f32[128]" = torch.ops.aten.mul.Tensor(sum_144, 0.00015943877551020407)
    mul_1377: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_100, squeeze_100)
    mul_1378: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1376, mul_1377);  mul_1376 = mul_1377 = None
    unsqueeze_1278: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1378, 0);  mul_1378 = None
    unsqueeze_1279: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1278, 2);  unsqueeze_1278 = None
    unsqueeze_1280: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1279, 3);  unsqueeze_1279 = None
    mul_1379: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_100, primals_101);  primals_101 = None
    unsqueeze_1281: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1379, 0);  mul_1379 = None
    unsqueeze_1282: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1281, 2);  unsqueeze_1281 = None
    unsqueeze_1283: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1282, 3);  unsqueeze_1282 = None
    mul_1380: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_389, unsqueeze_1280);  sub_389 = unsqueeze_1280 = None
    sub_391: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(where_69, mul_1380);  where_69 = mul_1380 = None
    sub_392: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(sub_391, unsqueeze_1277);  sub_391 = unsqueeze_1277 = None
    mul_1381: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_392, unsqueeze_1283);  sub_392 = unsqueeze_1283 = None
    mul_1382: "f32[128]" = torch.ops.aten.mul.Tensor(sum_144, squeeze_100);  sum_144 = squeeze_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    convolution_backward_72 = torch.ops.aten.convolution_backward.default(mul_1381, relu_30, primals_100, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1381 = primals_100 = None
    getitem_444: "f32[8, 256, 28, 28]" = convolution_backward_72[0]
    getitem_445: "f32[128, 256, 1, 1]" = convolution_backward_72[1];  convolution_backward_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    add_616: "f32[8, 256, 28, 28]" = torch.ops.aten.add.Tensor(add_615, getitem_444);  add_615 = getitem_444 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:180, code: x = self.relu(x)
    le_70: "b8[8, 256, 28, 28]" = torch.ops.aten.le.Scalar(relu_30, 0);  relu_30 = None
    where_70: "f32[8, 256, 28, 28]" = torch.ops.aten.where.self(le_70, full_default, add_616);  le_70 = add_616 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:177, code: x = self.bn(x)
    sum_145: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_70, [0, 2, 3])
    sub_393: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_1286);  convolution_32 = unsqueeze_1286 = None
    mul_1383: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(where_70, sub_393)
    sum_146: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1383, [0, 2, 3]);  mul_1383 = None
    mul_1384: "f32[256]" = torch.ops.aten.mul.Tensor(sum_145, 0.00015943877551020407)
    unsqueeze_1287: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1384, 0);  mul_1384 = None
    unsqueeze_1288: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1287, 2);  unsqueeze_1287 = None
    unsqueeze_1289: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1288, 3);  unsqueeze_1288 = None
    mul_1385: "f32[256]" = torch.ops.aten.mul.Tensor(sum_146, 0.00015943877551020407)
    mul_1386: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_97, squeeze_97)
    mul_1387: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1385, mul_1386);  mul_1385 = mul_1386 = None
    unsqueeze_1290: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1387, 0);  mul_1387 = None
    unsqueeze_1291: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1290, 2);  unsqueeze_1290 = None
    unsqueeze_1292: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1291, 3);  unsqueeze_1291 = None
    mul_1388: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_97, primals_98);  primals_98 = None
    unsqueeze_1293: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1388, 0);  mul_1388 = None
    unsqueeze_1294: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1293, 2);  unsqueeze_1293 = None
    unsqueeze_1295: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1294, 3);  unsqueeze_1294 = None
    mul_1389: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_393, unsqueeze_1292);  sub_393 = unsqueeze_1292 = None
    sub_395: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(where_70, mul_1389);  mul_1389 = None
    sub_396: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(sub_395, unsqueeze_1289);  sub_395 = unsqueeze_1289 = None
    mul_1390: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_396, unsqueeze_1295);  sub_396 = unsqueeze_1295 = None
    mul_1391: "f32[256]" = torch.ops.aten.mul.Tensor(sum_146, squeeze_97);  sum_146 = squeeze_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:176, code: x = self.conv(torch.cat(x_children, 1))
    convolution_backward_73 = torch.ops.aten.convolution_backward.default(mul_1390, cat_3, primals_97, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1390 = cat_3 = primals_97 = None
    getitem_447: "f32[8, 512, 28, 28]" = convolution_backward_73[0]
    getitem_448: "f32[256, 512, 1, 1]" = convolution_backward_73[1];  convolution_backward_73 = None
    slice_33: "f32[8, 256, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_447, 1, 0, 256)
    slice_34: "f32[8, 256, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_447, 1, 256, 512);  getitem_447 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:176, code: x = self.conv(torch.cat(x_children, 1))
    add_617: "f32[8, 256, 28, 28]" = torch.ops.aten.add.Tensor(where_70, slice_33);  where_70 = slice_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    where_71: "f32[8, 256, 28, 28]" = torch.ops.aten.where.self(le_71, full_default, add_617);  le_71 = add_617 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    add_618: "f32[8, 256, 28, 28]" = torch.ops.aten.add.Tensor(slice_34, where_71);  slice_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    sum_147: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_71, [0, 2, 3])
    sub_397: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_31, unsqueeze_1298);  convolution_31 = unsqueeze_1298 = None
    mul_1392: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(where_71, sub_397)
    sum_148: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1392, [0, 2, 3]);  mul_1392 = None
    mul_1393: "f32[256]" = torch.ops.aten.mul.Tensor(sum_147, 0.00015943877551020407)
    unsqueeze_1299: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1393, 0);  mul_1393 = None
    unsqueeze_1300: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1299, 2);  unsqueeze_1299 = None
    unsqueeze_1301: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1300, 3);  unsqueeze_1300 = None
    mul_1394: "f32[256]" = torch.ops.aten.mul.Tensor(sum_148, 0.00015943877551020407)
    mul_1395: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_94, squeeze_94)
    mul_1396: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1394, mul_1395);  mul_1394 = mul_1395 = None
    unsqueeze_1302: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1396, 0);  mul_1396 = None
    unsqueeze_1303: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1302, 2);  unsqueeze_1302 = None
    unsqueeze_1304: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1303, 3);  unsqueeze_1303 = None
    mul_1397: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_94, primals_95);  primals_95 = None
    unsqueeze_1305: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1397, 0);  mul_1397 = None
    unsqueeze_1306: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1305, 2);  unsqueeze_1305 = None
    unsqueeze_1307: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1306, 3);  unsqueeze_1306 = None
    mul_1398: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_397, unsqueeze_1304);  sub_397 = unsqueeze_1304 = None
    sub_399: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(where_71, mul_1398);  where_71 = mul_1398 = None
    sub_400: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(sub_399, unsqueeze_1301);  sub_399 = unsqueeze_1301 = None
    mul_1399: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_400, unsqueeze_1307);  sub_400 = unsqueeze_1307 = None
    mul_1400: "f32[256]" = torch.ops.aten.mul.Tensor(sum_148, squeeze_94);  sum_148 = squeeze_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    convolution_backward_74 = torch.ops.aten.convolution_backward.default(mul_1399, relu_28, primals_94, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1399 = primals_94 = None
    getitem_450: "f32[8, 128, 28, 28]" = convolution_backward_74[0]
    getitem_451: "f32[256, 128, 1, 1]" = convolution_backward_74[1];  convolution_backward_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    le_72: "b8[8, 128, 28, 28]" = torch.ops.aten.le.Scalar(relu_28, 0);  relu_28 = None
    where_72: "f32[8, 128, 28, 28]" = torch.ops.aten.where.self(le_72, full_default, getitem_450);  le_72 = getitem_450 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    sum_149: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_72, [0, 2, 3])
    sub_401: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_1310);  convolution_30 = unsqueeze_1310 = None
    mul_1401: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_72, sub_401)
    sum_150: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1401, [0, 2, 3]);  mul_1401 = None
    mul_1402: "f32[128]" = torch.ops.aten.mul.Tensor(sum_149, 0.00015943877551020407)
    unsqueeze_1311: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1402, 0);  mul_1402 = None
    unsqueeze_1312: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1311, 2);  unsqueeze_1311 = None
    unsqueeze_1313: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1312, 3);  unsqueeze_1312 = None
    mul_1403: "f32[128]" = torch.ops.aten.mul.Tensor(sum_150, 0.00015943877551020407)
    mul_1404: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_91, squeeze_91)
    mul_1405: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1403, mul_1404);  mul_1403 = mul_1404 = None
    unsqueeze_1314: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1405, 0);  mul_1405 = None
    unsqueeze_1315: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1314, 2);  unsqueeze_1314 = None
    unsqueeze_1316: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1315, 3);  unsqueeze_1315 = None
    mul_1406: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_91, primals_92);  primals_92 = None
    unsqueeze_1317: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1406, 0);  mul_1406 = None
    unsqueeze_1318: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1317, 2);  unsqueeze_1317 = None
    unsqueeze_1319: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1318, 3);  unsqueeze_1318 = None
    mul_1407: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_401, unsqueeze_1316);  sub_401 = unsqueeze_1316 = None
    sub_403: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(where_72, mul_1407);  where_72 = mul_1407 = None
    sub_404: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(sub_403, unsqueeze_1313);  sub_403 = unsqueeze_1313 = None
    mul_1408: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_404, unsqueeze_1319);  sub_404 = unsqueeze_1319 = None
    mul_1409: "f32[128]" = torch.ops.aten.mul.Tensor(sum_150, squeeze_91);  sum_150 = squeeze_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    convolution_backward_75 = torch.ops.aten.convolution_backward.default(mul_1408, relu_27, primals_91, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1408 = primals_91 = None
    getitem_453: "f32[8, 128, 28, 28]" = convolution_backward_75[0]
    getitem_454: "f32[128, 128, 3, 3]" = convolution_backward_75[1];  convolution_backward_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    le_73: "b8[8, 128, 28, 28]" = torch.ops.aten.le.Scalar(relu_27, 0);  relu_27 = None
    where_73: "f32[8, 128, 28, 28]" = torch.ops.aten.where.self(le_73, full_default, getitem_453);  le_73 = getitem_453 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    sum_151: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_73, [0, 2, 3])
    sub_405: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_1322);  convolution_29 = unsqueeze_1322 = None
    mul_1410: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_73, sub_405)
    sum_152: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1410, [0, 2, 3]);  mul_1410 = None
    mul_1411: "f32[128]" = torch.ops.aten.mul.Tensor(sum_151, 0.00015943877551020407)
    unsqueeze_1323: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1411, 0);  mul_1411 = None
    unsqueeze_1324: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1323, 2);  unsqueeze_1323 = None
    unsqueeze_1325: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1324, 3);  unsqueeze_1324 = None
    mul_1412: "f32[128]" = torch.ops.aten.mul.Tensor(sum_152, 0.00015943877551020407)
    mul_1413: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_88, squeeze_88)
    mul_1414: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1412, mul_1413);  mul_1412 = mul_1413 = None
    unsqueeze_1326: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1414, 0);  mul_1414 = None
    unsqueeze_1327: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1326, 2);  unsqueeze_1326 = None
    unsqueeze_1328: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1327, 3);  unsqueeze_1327 = None
    mul_1415: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_88, primals_89);  primals_89 = None
    unsqueeze_1329: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1415, 0);  mul_1415 = None
    unsqueeze_1330: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1329, 2);  unsqueeze_1329 = None
    unsqueeze_1331: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1330, 3);  unsqueeze_1330 = None
    mul_1416: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_405, unsqueeze_1328);  sub_405 = unsqueeze_1328 = None
    sub_407: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(where_73, mul_1416);  where_73 = mul_1416 = None
    sub_408: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(sub_407, unsqueeze_1325);  sub_407 = unsqueeze_1325 = None
    mul_1417: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_408, unsqueeze_1331);  sub_408 = unsqueeze_1331 = None
    mul_1418: "f32[128]" = torch.ops.aten.mul.Tensor(sum_152, squeeze_88);  sum_152 = squeeze_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    convolution_backward_76 = torch.ops.aten.convolution_backward.default(mul_1417, relu_26, primals_88, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1417 = primals_88 = None
    getitem_456: "f32[8, 256, 28, 28]" = convolution_backward_76[0]
    getitem_457: "f32[128, 256, 1, 1]" = convolution_backward_76[1];  convolution_backward_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    add_619: "f32[8, 256, 28, 28]" = torch.ops.aten.add.Tensor(add_618, getitem_456);  add_618 = getitem_456 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    le_74: "b8[8, 256, 28, 28]" = torch.ops.aten.le.Scalar(relu_26, 0);  relu_26 = None
    where_74: "f32[8, 256, 28, 28]" = torch.ops.aten.where.self(le_74, full_default, add_619);  le_74 = add_619 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    add_620: "f32[8, 256, 28, 28]" = torch.ops.aten.add.Tensor(slice_31, where_74);  slice_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    sum_153: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_74, [0, 2, 3])
    sub_409: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_1334);  convolution_28 = unsqueeze_1334 = None
    mul_1419: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(where_74, sub_409)
    sum_154: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1419, [0, 2, 3]);  mul_1419 = None
    mul_1420: "f32[256]" = torch.ops.aten.mul.Tensor(sum_153, 0.00015943877551020407)
    unsqueeze_1335: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1420, 0);  mul_1420 = None
    unsqueeze_1336: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1335, 2);  unsqueeze_1335 = None
    unsqueeze_1337: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1336, 3);  unsqueeze_1336 = None
    mul_1421: "f32[256]" = torch.ops.aten.mul.Tensor(sum_154, 0.00015943877551020407)
    mul_1422: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_85, squeeze_85)
    mul_1423: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1421, mul_1422);  mul_1421 = mul_1422 = None
    unsqueeze_1338: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1423, 0);  mul_1423 = None
    unsqueeze_1339: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1338, 2);  unsqueeze_1338 = None
    unsqueeze_1340: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1339, 3);  unsqueeze_1339 = None
    mul_1424: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_85, primals_86);  primals_86 = None
    unsqueeze_1341: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1424, 0);  mul_1424 = None
    unsqueeze_1342: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1341, 2);  unsqueeze_1341 = None
    unsqueeze_1343: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1342, 3);  unsqueeze_1342 = None
    mul_1425: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_409, unsqueeze_1340);  sub_409 = unsqueeze_1340 = None
    sub_411: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(where_74, mul_1425);  where_74 = mul_1425 = None
    sub_412: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(sub_411, unsqueeze_1337);  sub_411 = unsqueeze_1337 = None
    mul_1426: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_412, unsqueeze_1343);  sub_412 = unsqueeze_1343 = None
    mul_1427: "f32[256]" = torch.ops.aten.mul.Tensor(sum_154, squeeze_85);  sum_154 = squeeze_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    convolution_backward_77 = torch.ops.aten.convolution_backward.default(mul_1426, relu_25, primals_85, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1426 = primals_85 = None
    getitem_459: "f32[8, 128, 28, 28]" = convolution_backward_77[0]
    getitem_460: "f32[256, 128, 1, 1]" = convolution_backward_77[1];  convolution_backward_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    le_75: "b8[8, 128, 28, 28]" = torch.ops.aten.le.Scalar(relu_25, 0);  relu_25 = None
    where_75: "f32[8, 128, 28, 28]" = torch.ops.aten.where.self(le_75, full_default, getitem_459);  le_75 = getitem_459 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    sum_155: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_75, [0, 2, 3])
    sub_413: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_1346);  convolution_27 = unsqueeze_1346 = None
    mul_1428: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_75, sub_413)
    sum_156: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1428, [0, 2, 3]);  mul_1428 = None
    mul_1429: "f32[128]" = torch.ops.aten.mul.Tensor(sum_155, 0.00015943877551020407)
    unsqueeze_1347: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1429, 0);  mul_1429 = None
    unsqueeze_1348: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1347, 2);  unsqueeze_1347 = None
    unsqueeze_1349: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1348, 3);  unsqueeze_1348 = None
    mul_1430: "f32[128]" = torch.ops.aten.mul.Tensor(sum_156, 0.00015943877551020407)
    mul_1431: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_82, squeeze_82)
    mul_1432: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1430, mul_1431);  mul_1430 = mul_1431 = None
    unsqueeze_1350: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1432, 0);  mul_1432 = None
    unsqueeze_1351: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1350, 2);  unsqueeze_1350 = None
    unsqueeze_1352: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1351, 3);  unsqueeze_1351 = None
    mul_1433: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_82, primals_83);  primals_83 = None
    unsqueeze_1353: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1433, 0);  mul_1433 = None
    unsqueeze_1354: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1353, 2);  unsqueeze_1353 = None
    unsqueeze_1355: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1354, 3);  unsqueeze_1354 = None
    mul_1434: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_413, unsqueeze_1352);  sub_413 = unsqueeze_1352 = None
    sub_415: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(where_75, mul_1434);  where_75 = mul_1434 = None
    sub_416: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(sub_415, unsqueeze_1349);  sub_415 = unsqueeze_1349 = None
    mul_1435: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_416, unsqueeze_1355);  sub_416 = unsqueeze_1355 = None
    mul_1436: "f32[128]" = torch.ops.aten.mul.Tensor(sum_156, squeeze_82);  sum_156 = squeeze_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    convolution_backward_78 = torch.ops.aten.convolution_backward.default(mul_1435, relu_24, primals_82, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1435 = primals_82 = None
    getitem_462: "f32[8, 128, 28, 28]" = convolution_backward_78[0]
    getitem_463: "f32[128, 128, 3, 3]" = convolution_backward_78[1];  convolution_backward_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    le_76: "b8[8, 128, 28, 28]" = torch.ops.aten.le.Scalar(relu_24, 0);  relu_24 = None
    where_76: "f32[8, 128, 28, 28]" = torch.ops.aten.where.self(le_76, full_default, getitem_462);  le_76 = getitem_462 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    sum_157: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_76, [0, 2, 3])
    sub_417: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_1358);  convolution_26 = unsqueeze_1358 = None
    mul_1437: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_76, sub_417)
    sum_158: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1437, [0, 2, 3]);  mul_1437 = None
    mul_1438: "f32[128]" = torch.ops.aten.mul.Tensor(sum_157, 0.00015943877551020407)
    unsqueeze_1359: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1438, 0);  mul_1438 = None
    unsqueeze_1360: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1359, 2);  unsqueeze_1359 = None
    unsqueeze_1361: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1360, 3);  unsqueeze_1360 = None
    mul_1439: "f32[128]" = torch.ops.aten.mul.Tensor(sum_158, 0.00015943877551020407)
    mul_1440: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_79, squeeze_79)
    mul_1441: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1439, mul_1440);  mul_1439 = mul_1440 = None
    unsqueeze_1362: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1441, 0);  mul_1441 = None
    unsqueeze_1363: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1362, 2);  unsqueeze_1362 = None
    unsqueeze_1364: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1363, 3);  unsqueeze_1363 = None
    mul_1442: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_79, primals_80);  primals_80 = None
    unsqueeze_1365: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1442, 0);  mul_1442 = None
    unsqueeze_1366: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1365, 2);  unsqueeze_1365 = None
    unsqueeze_1367: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1366, 3);  unsqueeze_1366 = None
    mul_1443: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_417, unsqueeze_1364);  sub_417 = unsqueeze_1364 = None
    sub_419: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(where_76, mul_1443);  where_76 = mul_1443 = None
    sub_420: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(sub_419, unsqueeze_1361);  sub_419 = unsqueeze_1361 = None
    mul_1444: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_420, unsqueeze_1367);  sub_420 = unsqueeze_1367 = None
    mul_1445: "f32[128]" = torch.ops.aten.mul.Tensor(sum_158, squeeze_79);  sum_158 = squeeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    convolution_backward_79 = torch.ops.aten.convolution_backward.default(mul_1444, relu_23, primals_79, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1444 = primals_79 = None
    getitem_465: "f32[8, 256, 28, 28]" = convolution_backward_79[0]
    getitem_466: "f32[128, 256, 1, 1]" = convolution_backward_79[1];  convolution_backward_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    add_621: "f32[8, 256, 28, 28]" = torch.ops.aten.add.Tensor(add_620, getitem_465);  add_620 = getitem_465 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:180, code: x = self.relu(x)
    le_77: "b8[8, 256, 28, 28]" = torch.ops.aten.le.Scalar(relu_23, 0);  relu_23 = None
    where_77: "f32[8, 256, 28, 28]" = torch.ops.aten.where.self(le_77, full_default, add_621);  le_77 = add_621 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:177, code: x = self.bn(x)
    sum_159: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_77, [0, 2, 3])
    sub_421: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_1370);  convolution_25 = unsqueeze_1370 = None
    mul_1446: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(where_77, sub_421)
    sum_160: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1446, [0, 2, 3]);  mul_1446 = None
    mul_1447: "f32[256]" = torch.ops.aten.mul.Tensor(sum_159, 0.00015943877551020407)
    unsqueeze_1371: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1447, 0);  mul_1447 = None
    unsqueeze_1372: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1371, 2);  unsqueeze_1371 = None
    unsqueeze_1373: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1372, 3);  unsqueeze_1372 = None
    mul_1448: "f32[256]" = torch.ops.aten.mul.Tensor(sum_160, 0.00015943877551020407)
    mul_1449: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_76, squeeze_76)
    mul_1450: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1448, mul_1449);  mul_1448 = mul_1449 = None
    unsqueeze_1374: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1450, 0);  mul_1450 = None
    unsqueeze_1375: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1374, 2);  unsqueeze_1374 = None
    unsqueeze_1376: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1375, 3);  unsqueeze_1375 = None
    mul_1451: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_76, primals_77);  primals_77 = None
    unsqueeze_1377: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1451, 0);  mul_1451 = None
    unsqueeze_1378: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1377, 2);  unsqueeze_1377 = None
    unsqueeze_1379: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1378, 3);  unsqueeze_1378 = None
    mul_1452: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_421, unsqueeze_1376);  sub_421 = unsqueeze_1376 = None
    sub_423: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(where_77, mul_1452);  mul_1452 = None
    sub_424: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(sub_423, unsqueeze_1373);  sub_423 = unsqueeze_1373 = None
    mul_1453: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_424, unsqueeze_1379);  sub_424 = unsqueeze_1379 = None
    mul_1454: "f32[256]" = torch.ops.aten.mul.Tensor(sum_160, squeeze_76);  sum_160 = squeeze_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:176, code: x = self.conv(torch.cat(x_children, 1))
    convolution_backward_80 = torch.ops.aten.convolution_backward.default(mul_1453, cat_2, primals_76, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1453 = cat_2 = primals_76 = None
    getitem_468: "f32[8, 768, 28, 28]" = convolution_backward_80[0]
    getitem_469: "f32[256, 768, 1, 1]" = convolution_backward_80[1];  convolution_backward_80 = None
    slice_35: "f32[8, 256, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_468, 1, 0, 256)
    slice_36: "f32[8, 256, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_468, 1, 256, 512)
    slice_37: "f32[8, 256, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_468, 1, 512, 768);  getitem_468 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:176, code: x = self.conv(torch.cat(x_children, 1))
    add_622: "f32[8, 256, 28, 28]" = torch.ops.aten.add.Tensor(where_77, slice_35);  where_77 = slice_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    where_78: "f32[8, 256, 28, 28]" = torch.ops.aten.where.self(le_78, full_default, add_622);  le_78 = add_622 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    add_623: "f32[8, 256, 28, 28]" = torch.ops.aten.add.Tensor(slice_36, where_78);  slice_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    sum_161: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_78, [0, 2, 3])
    sub_425: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_1382);  convolution_24 = unsqueeze_1382 = None
    mul_1455: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(where_78, sub_425)
    sum_162: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1455, [0, 2, 3]);  mul_1455 = None
    mul_1456: "f32[256]" = torch.ops.aten.mul.Tensor(sum_161, 0.00015943877551020407)
    unsqueeze_1383: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1456, 0);  mul_1456 = None
    unsqueeze_1384: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1383, 2);  unsqueeze_1383 = None
    unsqueeze_1385: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1384, 3);  unsqueeze_1384 = None
    mul_1457: "f32[256]" = torch.ops.aten.mul.Tensor(sum_162, 0.00015943877551020407)
    mul_1458: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_73, squeeze_73)
    mul_1459: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1457, mul_1458);  mul_1457 = mul_1458 = None
    unsqueeze_1386: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1459, 0);  mul_1459 = None
    unsqueeze_1387: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1386, 2);  unsqueeze_1386 = None
    unsqueeze_1388: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1387, 3);  unsqueeze_1387 = None
    mul_1460: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_73, primals_74);  primals_74 = None
    unsqueeze_1389: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1460, 0);  mul_1460 = None
    unsqueeze_1390: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1389, 2);  unsqueeze_1389 = None
    unsqueeze_1391: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1390, 3);  unsqueeze_1390 = None
    mul_1461: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_425, unsqueeze_1388);  sub_425 = unsqueeze_1388 = None
    sub_427: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(where_78, mul_1461);  where_78 = mul_1461 = None
    sub_428: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(sub_427, unsqueeze_1385);  sub_427 = unsqueeze_1385 = None
    mul_1462: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_428, unsqueeze_1391);  sub_428 = unsqueeze_1391 = None
    mul_1463: "f32[256]" = torch.ops.aten.mul.Tensor(sum_162, squeeze_73);  sum_162 = squeeze_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    convolution_backward_81 = torch.ops.aten.convolution_backward.default(mul_1462, relu_21, primals_73, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1462 = primals_73 = None
    getitem_471: "f32[8, 128, 28, 28]" = convolution_backward_81[0]
    getitem_472: "f32[256, 128, 1, 1]" = convolution_backward_81[1];  convolution_backward_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    le_79: "b8[8, 128, 28, 28]" = torch.ops.aten.le.Scalar(relu_21, 0);  relu_21 = None
    where_79: "f32[8, 128, 28, 28]" = torch.ops.aten.where.self(le_79, full_default, getitem_471);  le_79 = getitem_471 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    sum_163: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_79, [0, 2, 3])
    sub_429: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_1394);  convolution_23 = unsqueeze_1394 = None
    mul_1464: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_79, sub_429)
    sum_164: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1464, [0, 2, 3]);  mul_1464 = None
    mul_1465: "f32[128]" = torch.ops.aten.mul.Tensor(sum_163, 0.00015943877551020407)
    unsqueeze_1395: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1465, 0);  mul_1465 = None
    unsqueeze_1396: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1395, 2);  unsqueeze_1395 = None
    unsqueeze_1397: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1396, 3);  unsqueeze_1396 = None
    mul_1466: "f32[128]" = torch.ops.aten.mul.Tensor(sum_164, 0.00015943877551020407)
    mul_1467: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_70, squeeze_70)
    mul_1468: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1466, mul_1467);  mul_1466 = mul_1467 = None
    unsqueeze_1398: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1468, 0);  mul_1468 = None
    unsqueeze_1399: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1398, 2);  unsqueeze_1398 = None
    unsqueeze_1400: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1399, 3);  unsqueeze_1399 = None
    mul_1469: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_70, primals_71);  primals_71 = None
    unsqueeze_1401: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1469, 0);  mul_1469 = None
    unsqueeze_1402: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1401, 2);  unsqueeze_1401 = None
    unsqueeze_1403: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1402, 3);  unsqueeze_1402 = None
    mul_1470: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_429, unsqueeze_1400);  sub_429 = unsqueeze_1400 = None
    sub_431: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(where_79, mul_1470);  where_79 = mul_1470 = None
    sub_432: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(sub_431, unsqueeze_1397);  sub_431 = unsqueeze_1397 = None
    mul_1471: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_432, unsqueeze_1403);  sub_432 = unsqueeze_1403 = None
    mul_1472: "f32[128]" = torch.ops.aten.mul.Tensor(sum_164, squeeze_70);  sum_164 = squeeze_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    convolution_backward_82 = torch.ops.aten.convolution_backward.default(mul_1471, relu_20, primals_70, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1471 = primals_70 = None
    getitem_474: "f32[8, 128, 28, 28]" = convolution_backward_82[0]
    getitem_475: "f32[128, 128, 3, 3]" = convolution_backward_82[1];  convolution_backward_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    le_80: "b8[8, 128, 28, 28]" = torch.ops.aten.le.Scalar(relu_20, 0);  relu_20 = None
    where_80: "f32[8, 128, 28, 28]" = torch.ops.aten.where.self(le_80, full_default, getitem_474);  le_80 = getitem_474 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    sum_165: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_80, [0, 2, 3])
    sub_433: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_1406);  convolution_22 = unsqueeze_1406 = None
    mul_1473: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_80, sub_433)
    sum_166: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1473, [0, 2, 3]);  mul_1473 = None
    mul_1474: "f32[128]" = torch.ops.aten.mul.Tensor(sum_165, 0.00015943877551020407)
    unsqueeze_1407: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1474, 0);  mul_1474 = None
    unsqueeze_1408: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1407, 2);  unsqueeze_1407 = None
    unsqueeze_1409: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1408, 3);  unsqueeze_1408 = None
    mul_1475: "f32[128]" = torch.ops.aten.mul.Tensor(sum_166, 0.00015943877551020407)
    mul_1476: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_67, squeeze_67)
    mul_1477: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1475, mul_1476);  mul_1475 = mul_1476 = None
    unsqueeze_1410: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1477, 0);  mul_1477 = None
    unsqueeze_1411: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1410, 2);  unsqueeze_1410 = None
    unsqueeze_1412: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1411, 3);  unsqueeze_1411 = None
    mul_1478: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_67, primals_68);  primals_68 = None
    unsqueeze_1413: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1478, 0);  mul_1478 = None
    unsqueeze_1414: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1413, 2);  unsqueeze_1413 = None
    unsqueeze_1415: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1414, 3);  unsqueeze_1414 = None
    mul_1479: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_433, unsqueeze_1412);  sub_433 = unsqueeze_1412 = None
    sub_435: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(where_80, mul_1479);  where_80 = mul_1479 = None
    sub_436: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(sub_435, unsqueeze_1409);  sub_435 = unsqueeze_1409 = None
    mul_1480: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_436, unsqueeze_1415);  sub_436 = unsqueeze_1415 = None
    mul_1481: "f32[128]" = torch.ops.aten.mul.Tensor(sum_166, squeeze_67);  sum_166 = squeeze_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    convolution_backward_83 = torch.ops.aten.convolution_backward.default(mul_1480, relu_19, primals_67, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1480 = primals_67 = None
    getitem_477: "f32[8, 256, 28, 28]" = convolution_backward_83[0]
    getitem_478: "f32[128, 256, 1, 1]" = convolution_backward_83[1];  convolution_backward_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    add_624: "f32[8, 256, 28, 28]" = torch.ops.aten.add.Tensor(add_623, getitem_477);  add_623 = getitem_477 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    le_81: "b8[8, 256, 28, 28]" = torch.ops.aten.le.Scalar(relu_19, 0);  relu_19 = None
    where_81: "f32[8, 256, 28, 28]" = torch.ops.aten.where.self(le_81, full_default, add_624);  le_81 = add_624 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    add_625: "f32[8, 256, 28, 28]" = torch.ops.aten.add.Tensor(slice_37, where_81);  slice_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    sum_167: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_81, [0, 2, 3])
    sub_437: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_1418);  convolution_21 = unsqueeze_1418 = None
    mul_1482: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(where_81, sub_437)
    sum_168: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1482, [0, 2, 3]);  mul_1482 = None
    mul_1483: "f32[256]" = torch.ops.aten.mul.Tensor(sum_167, 0.00015943877551020407)
    unsqueeze_1419: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1483, 0);  mul_1483 = None
    unsqueeze_1420: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1419, 2);  unsqueeze_1419 = None
    unsqueeze_1421: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1420, 3);  unsqueeze_1420 = None
    mul_1484: "f32[256]" = torch.ops.aten.mul.Tensor(sum_168, 0.00015943877551020407)
    mul_1485: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_64, squeeze_64)
    mul_1486: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1484, mul_1485);  mul_1484 = mul_1485 = None
    unsqueeze_1422: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1486, 0);  mul_1486 = None
    unsqueeze_1423: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1422, 2);  unsqueeze_1422 = None
    unsqueeze_1424: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1423, 3);  unsqueeze_1423 = None
    mul_1487: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_64, primals_65);  primals_65 = None
    unsqueeze_1425: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1487, 0);  mul_1487 = None
    unsqueeze_1426: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1425, 2);  unsqueeze_1425 = None
    unsqueeze_1427: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1426, 3);  unsqueeze_1426 = None
    mul_1488: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_437, unsqueeze_1424);  sub_437 = unsqueeze_1424 = None
    sub_439: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(where_81, mul_1488);  where_81 = mul_1488 = None
    sub_440: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(sub_439, unsqueeze_1421);  sub_439 = unsqueeze_1421 = None
    mul_1489: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_440, unsqueeze_1427);  sub_440 = unsqueeze_1427 = None
    mul_1490: "f32[256]" = torch.ops.aten.mul.Tensor(sum_168, squeeze_64);  sum_168 = squeeze_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    convolution_backward_84 = torch.ops.aten.convolution_backward.default(mul_1489, relu_18, primals_64, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1489 = primals_64 = None
    getitem_480: "f32[8, 128, 28, 28]" = convolution_backward_84[0]
    getitem_481: "f32[256, 128, 1, 1]" = convolution_backward_84[1];  convolution_backward_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    le_82: "b8[8, 128, 28, 28]" = torch.ops.aten.le.Scalar(relu_18, 0);  relu_18 = None
    where_82: "f32[8, 128, 28, 28]" = torch.ops.aten.where.self(le_82, full_default, getitem_480);  le_82 = getitem_480 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    sum_169: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_82, [0, 2, 3])
    sub_441: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_1430);  convolution_20 = unsqueeze_1430 = None
    mul_1491: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_82, sub_441)
    sum_170: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1491, [0, 2, 3]);  mul_1491 = None
    mul_1492: "f32[128]" = torch.ops.aten.mul.Tensor(sum_169, 0.00015943877551020407)
    unsqueeze_1431: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1492, 0);  mul_1492 = None
    unsqueeze_1432: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1431, 2);  unsqueeze_1431 = None
    unsqueeze_1433: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1432, 3);  unsqueeze_1432 = None
    mul_1493: "f32[128]" = torch.ops.aten.mul.Tensor(sum_170, 0.00015943877551020407)
    mul_1494: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_61, squeeze_61)
    mul_1495: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1493, mul_1494);  mul_1493 = mul_1494 = None
    unsqueeze_1434: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1495, 0);  mul_1495 = None
    unsqueeze_1435: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1434, 2);  unsqueeze_1434 = None
    unsqueeze_1436: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1435, 3);  unsqueeze_1435 = None
    mul_1496: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_61, primals_62);  primals_62 = None
    unsqueeze_1437: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1496, 0);  mul_1496 = None
    unsqueeze_1438: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1437, 2);  unsqueeze_1437 = None
    unsqueeze_1439: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1438, 3);  unsqueeze_1438 = None
    mul_1497: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_441, unsqueeze_1436);  sub_441 = unsqueeze_1436 = None
    sub_443: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(where_82, mul_1497);  where_82 = mul_1497 = None
    sub_444: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(sub_443, unsqueeze_1433);  sub_443 = unsqueeze_1433 = None
    mul_1498: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_444, unsqueeze_1439);  sub_444 = unsqueeze_1439 = None
    mul_1499: "f32[128]" = torch.ops.aten.mul.Tensor(sum_170, squeeze_61);  sum_170 = squeeze_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    convolution_backward_85 = torch.ops.aten.convolution_backward.default(mul_1498, relu_17, primals_61, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1498 = primals_61 = None
    getitem_483: "f32[8, 128, 28, 28]" = convolution_backward_85[0]
    getitem_484: "f32[128, 128, 3, 3]" = convolution_backward_85[1];  convolution_backward_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    le_83: "b8[8, 128, 28, 28]" = torch.ops.aten.le.Scalar(relu_17, 0);  relu_17 = None
    where_83: "f32[8, 128, 28, 28]" = torch.ops.aten.where.self(le_83, full_default, getitem_483);  le_83 = getitem_483 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    sum_171: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_83, [0, 2, 3])
    sub_445: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_1442);  convolution_19 = unsqueeze_1442 = None
    mul_1500: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_83, sub_445)
    sum_172: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1500, [0, 2, 3]);  mul_1500 = None
    mul_1501: "f32[128]" = torch.ops.aten.mul.Tensor(sum_171, 0.00015943877551020407)
    unsqueeze_1443: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1501, 0);  mul_1501 = None
    unsqueeze_1444: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1443, 2);  unsqueeze_1443 = None
    unsqueeze_1445: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1444, 3);  unsqueeze_1444 = None
    mul_1502: "f32[128]" = torch.ops.aten.mul.Tensor(sum_172, 0.00015943877551020407)
    mul_1503: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_58, squeeze_58)
    mul_1504: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1502, mul_1503);  mul_1502 = mul_1503 = None
    unsqueeze_1446: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1504, 0);  mul_1504 = None
    unsqueeze_1447: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1446, 2);  unsqueeze_1446 = None
    unsqueeze_1448: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1447, 3);  unsqueeze_1447 = None
    mul_1505: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_58, primals_59);  primals_59 = None
    unsqueeze_1449: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1505, 0);  mul_1505 = None
    unsqueeze_1450: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1449, 2);  unsqueeze_1449 = None
    unsqueeze_1451: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1450, 3);  unsqueeze_1450 = None
    mul_1506: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_445, unsqueeze_1448);  sub_445 = unsqueeze_1448 = None
    sub_447: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(where_83, mul_1506);  where_83 = mul_1506 = None
    sub_448: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(sub_447, unsqueeze_1445);  sub_447 = unsqueeze_1445 = None
    mul_1507: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_448, unsqueeze_1451);  sub_448 = unsqueeze_1451 = None
    mul_1508: "f32[128]" = torch.ops.aten.mul.Tensor(sum_172, squeeze_58);  sum_172 = squeeze_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    convolution_backward_86 = torch.ops.aten.convolution_backward.default(mul_1507, relu_16, primals_58, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1507 = primals_58 = None
    getitem_486: "f32[8, 256, 28, 28]" = convolution_backward_86[0]
    getitem_487: "f32[128, 256, 1, 1]" = convolution_backward_86[1];  convolution_backward_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    add_626: "f32[8, 256, 28, 28]" = torch.ops.aten.add.Tensor(add_625, getitem_486);  add_625 = getitem_486 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:180, code: x = self.relu(x)
    le_84: "b8[8, 256, 28, 28]" = torch.ops.aten.le.Scalar(relu_16, 0);  relu_16 = None
    where_84: "f32[8, 256, 28, 28]" = torch.ops.aten.where.self(le_84, full_default, add_626);  le_84 = add_626 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:177, code: x = self.bn(x)
    sum_173: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_84, [0, 2, 3])
    sub_449: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_1454);  convolution_18 = unsqueeze_1454 = None
    mul_1509: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(where_84, sub_449)
    sum_174: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1509, [0, 2, 3]);  mul_1509 = None
    mul_1510: "f32[256]" = torch.ops.aten.mul.Tensor(sum_173, 0.00015943877551020407)
    unsqueeze_1455: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1510, 0);  mul_1510 = None
    unsqueeze_1456: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1455, 2);  unsqueeze_1455 = None
    unsqueeze_1457: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1456, 3);  unsqueeze_1456 = None
    mul_1511: "f32[256]" = torch.ops.aten.mul.Tensor(sum_174, 0.00015943877551020407)
    mul_1512: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_55, squeeze_55)
    mul_1513: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1511, mul_1512);  mul_1511 = mul_1512 = None
    unsqueeze_1458: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1513, 0);  mul_1513 = None
    unsqueeze_1459: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1458, 2);  unsqueeze_1458 = None
    unsqueeze_1460: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1459, 3);  unsqueeze_1459 = None
    mul_1514: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_55, primals_56);  primals_56 = None
    unsqueeze_1461: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1514, 0);  mul_1514 = None
    unsqueeze_1462: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1461, 2);  unsqueeze_1461 = None
    unsqueeze_1463: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1462, 3);  unsqueeze_1462 = None
    mul_1515: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_449, unsqueeze_1460);  sub_449 = unsqueeze_1460 = None
    sub_451: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(where_84, mul_1515);  mul_1515 = None
    sub_452: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(sub_451, unsqueeze_1457);  sub_451 = unsqueeze_1457 = None
    mul_1516: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_452, unsqueeze_1463);  sub_452 = unsqueeze_1463 = None
    mul_1517: "f32[256]" = torch.ops.aten.mul.Tensor(sum_174, squeeze_55);  sum_174 = squeeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:176, code: x = self.conv(torch.cat(x_children, 1))
    convolution_backward_87 = torch.ops.aten.convolution_backward.default(mul_1516, cat_1, primals_55, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1516 = cat_1 = primals_55 = None
    getitem_489: "f32[8, 512, 28, 28]" = convolution_backward_87[0]
    getitem_490: "f32[256, 512, 1, 1]" = convolution_backward_87[1];  convolution_backward_87 = None
    slice_38: "f32[8, 256, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_489, 1, 0, 256)
    slice_39: "f32[8, 256, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_489, 1, 256, 512);  getitem_489 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:176, code: x = self.conv(torch.cat(x_children, 1))
    add_627: "f32[8, 256, 28, 28]" = torch.ops.aten.add.Tensor(where_84, slice_38);  where_84 = slice_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    where_85: "f32[8, 256, 28, 28]" = torch.ops.aten.where.self(le_85, full_default, add_627);  le_85 = add_627 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    add_628: "f32[8, 256, 28, 28]" = torch.ops.aten.add.Tensor(slice_39, where_85);  slice_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    sum_175: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_85, [0, 2, 3])
    sub_453: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_1466);  convolution_17 = unsqueeze_1466 = None
    mul_1518: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(where_85, sub_453)
    sum_176: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1518, [0, 2, 3]);  mul_1518 = None
    mul_1519: "f32[256]" = torch.ops.aten.mul.Tensor(sum_175, 0.00015943877551020407)
    unsqueeze_1467: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1519, 0);  mul_1519 = None
    unsqueeze_1468: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1467, 2);  unsqueeze_1467 = None
    unsqueeze_1469: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1468, 3);  unsqueeze_1468 = None
    mul_1520: "f32[256]" = torch.ops.aten.mul.Tensor(sum_176, 0.00015943877551020407)
    mul_1521: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_52, squeeze_52)
    mul_1522: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1520, mul_1521);  mul_1520 = mul_1521 = None
    unsqueeze_1470: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1522, 0);  mul_1522 = None
    unsqueeze_1471: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1470, 2);  unsqueeze_1470 = None
    unsqueeze_1472: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1471, 3);  unsqueeze_1471 = None
    mul_1523: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_52, primals_53);  primals_53 = None
    unsqueeze_1473: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1523, 0);  mul_1523 = None
    unsqueeze_1474: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1473, 2);  unsqueeze_1473 = None
    unsqueeze_1475: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1474, 3);  unsqueeze_1474 = None
    mul_1524: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_453, unsqueeze_1472);  sub_453 = unsqueeze_1472 = None
    sub_455: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(where_85, mul_1524);  where_85 = mul_1524 = None
    sub_456: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(sub_455, unsqueeze_1469);  sub_455 = unsqueeze_1469 = None
    mul_1525: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_456, unsqueeze_1475);  sub_456 = unsqueeze_1475 = None
    mul_1526: "f32[256]" = torch.ops.aten.mul.Tensor(sum_176, squeeze_52);  sum_176 = squeeze_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    convolution_backward_88 = torch.ops.aten.convolution_backward.default(mul_1525, relu_14, primals_52, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1525 = primals_52 = None
    getitem_492: "f32[8, 128, 28, 28]" = convolution_backward_88[0]
    getitem_493: "f32[256, 128, 1, 1]" = convolution_backward_88[1];  convolution_backward_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    le_86: "b8[8, 128, 28, 28]" = torch.ops.aten.le.Scalar(relu_14, 0);  relu_14 = None
    where_86: "f32[8, 128, 28, 28]" = torch.ops.aten.where.self(le_86, full_default, getitem_492);  le_86 = getitem_492 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    sum_177: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_86, [0, 2, 3])
    sub_457: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_1478);  convolution_16 = unsqueeze_1478 = None
    mul_1527: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_86, sub_457)
    sum_178: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1527, [0, 2, 3]);  mul_1527 = None
    mul_1528: "f32[128]" = torch.ops.aten.mul.Tensor(sum_177, 0.00015943877551020407)
    unsqueeze_1479: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1528, 0);  mul_1528 = None
    unsqueeze_1480: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1479, 2);  unsqueeze_1479 = None
    unsqueeze_1481: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1480, 3);  unsqueeze_1480 = None
    mul_1529: "f32[128]" = torch.ops.aten.mul.Tensor(sum_178, 0.00015943877551020407)
    mul_1530: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_49, squeeze_49)
    mul_1531: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1529, mul_1530);  mul_1529 = mul_1530 = None
    unsqueeze_1482: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1531, 0);  mul_1531 = None
    unsqueeze_1483: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1482, 2);  unsqueeze_1482 = None
    unsqueeze_1484: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1483, 3);  unsqueeze_1483 = None
    mul_1532: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_49, primals_50);  primals_50 = None
    unsqueeze_1485: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1532, 0);  mul_1532 = None
    unsqueeze_1486: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1485, 2);  unsqueeze_1485 = None
    unsqueeze_1487: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1486, 3);  unsqueeze_1486 = None
    mul_1533: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_457, unsqueeze_1484);  sub_457 = unsqueeze_1484 = None
    sub_459: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(where_86, mul_1533);  where_86 = mul_1533 = None
    sub_460: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(sub_459, unsqueeze_1481);  sub_459 = unsqueeze_1481 = None
    mul_1534: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_460, unsqueeze_1487);  sub_460 = unsqueeze_1487 = None
    mul_1535: "f32[128]" = torch.ops.aten.mul.Tensor(sum_178, squeeze_49);  sum_178 = squeeze_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    convolution_backward_89 = torch.ops.aten.convolution_backward.default(mul_1534, relu_13, primals_49, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1534 = primals_49 = None
    getitem_495: "f32[8, 128, 28, 28]" = convolution_backward_89[0]
    getitem_496: "f32[128, 128, 3, 3]" = convolution_backward_89[1];  convolution_backward_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    le_87: "b8[8, 128, 28, 28]" = torch.ops.aten.le.Scalar(relu_13, 0);  relu_13 = None
    where_87: "f32[8, 128, 28, 28]" = torch.ops.aten.where.self(le_87, full_default, getitem_495);  le_87 = getitem_495 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    sum_179: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_87, [0, 2, 3])
    sub_461: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_1490);  convolution_15 = unsqueeze_1490 = None
    mul_1536: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_87, sub_461)
    sum_180: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1536, [0, 2, 3]);  mul_1536 = None
    mul_1537: "f32[128]" = torch.ops.aten.mul.Tensor(sum_179, 0.00015943877551020407)
    unsqueeze_1491: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1537, 0);  mul_1537 = None
    unsqueeze_1492: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1491, 2);  unsqueeze_1491 = None
    unsqueeze_1493: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1492, 3);  unsqueeze_1492 = None
    mul_1538: "f32[128]" = torch.ops.aten.mul.Tensor(sum_180, 0.00015943877551020407)
    mul_1539: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_46, squeeze_46)
    mul_1540: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1538, mul_1539);  mul_1538 = mul_1539 = None
    unsqueeze_1494: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1540, 0);  mul_1540 = None
    unsqueeze_1495: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1494, 2);  unsqueeze_1494 = None
    unsqueeze_1496: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1495, 3);  unsqueeze_1495 = None
    mul_1541: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_46, primals_47);  primals_47 = None
    unsqueeze_1497: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1541, 0);  mul_1541 = None
    unsqueeze_1498: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1497, 2);  unsqueeze_1497 = None
    unsqueeze_1499: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1498, 3);  unsqueeze_1498 = None
    mul_1542: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_461, unsqueeze_1496);  sub_461 = unsqueeze_1496 = None
    sub_463: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(where_87, mul_1542);  where_87 = mul_1542 = None
    sub_464: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(sub_463, unsqueeze_1493);  sub_463 = unsqueeze_1493 = None
    mul_1543: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_464, unsqueeze_1499);  sub_464 = unsqueeze_1499 = None
    mul_1544: "f32[128]" = torch.ops.aten.mul.Tensor(sum_180, squeeze_46);  sum_180 = squeeze_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    convolution_backward_90 = torch.ops.aten.convolution_backward.default(mul_1543, relu_12, primals_46, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1543 = primals_46 = None
    getitem_498: "f32[8, 256, 28, 28]" = convolution_backward_90[0]
    getitem_499: "f32[128, 256, 1, 1]" = convolution_backward_90[1];  convolution_backward_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    add_629: "f32[8, 256, 28, 28]" = torch.ops.aten.add.Tensor(add_628, getitem_498);  add_628 = getitem_498 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    le_88: "b8[8, 256, 28, 28]" = torch.ops.aten.le.Scalar(relu_12, 0);  relu_12 = None
    where_88: "f32[8, 256, 28, 28]" = torch.ops.aten.where.self(le_88, full_default, add_629);  le_88 = add_629 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    sum_181: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_88, [0, 2, 3])
    sub_465: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_1502);  convolution_14 = unsqueeze_1502 = None
    mul_1545: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(where_88, sub_465)
    sum_182: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1545, [0, 2, 3]);  mul_1545 = None
    mul_1546: "f32[256]" = torch.ops.aten.mul.Tensor(sum_181, 0.00015943877551020407)
    unsqueeze_1503: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1546, 0);  mul_1546 = None
    unsqueeze_1504: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1503, 2);  unsqueeze_1503 = None
    unsqueeze_1505: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1504, 3);  unsqueeze_1504 = None
    mul_1547: "f32[256]" = torch.ops.aten.mul.Tensor(sum_182, 0.00015943877551020407)
    mul_1548: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
    mul_1549: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1547, mul_1548);  mul_1547 = mul_1548 = None
    unsqueeze_1506: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1549, 0);  mul_1549 = None
    unsqueeze_1507: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1506, 2);  unsqueeze_1506 = None
    unsqueeze_1508: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1507, 3);  unsqueeze_1507 = None
    mul_1550: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_43, primals_44);  primals_44 = None
    unsqueeze_1509: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1550, 0);  mul_1550 = None
    unsqueeze_1510: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1509, 2);  unsqueeze_1509 = None
    unsqueeze_1511: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1510, 3);  unsqueeze_1510 = None
    mul_1551: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_465, unsqueeze_1508);  sub_465 = unsqueeze_1508 = None
    sub_467: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(where_88, mul_1551);  mul_1551 = None
    sub_468: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(sub_467, unsqueeze_1505);  sub_467 = None
    mul_1552: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_468, unsqueeze_1511);  sub_468 = unsqueeze_1511 = None
    mul_1553: "f32[256]" = torch.ops.aten.mul.Tensor(sum_182, squeeze_43);  sum_182 = squeeze_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    convolution_backward_91 = torch.ops.aten.convolution_backward.default(mul_1552, relu_11, primals_43, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1552 = primals_43 = None
    getitem_501: "f32[8, 128, 28, 28]" = convolution_backward_91[0]
    getitem_502: "f32[256, 128, 1, 1]" = convolution_backward_91[1];  convolution_backward_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    le_89: "b8[8, 128, 28, 28]" = torch.ops.aten.le.Scalar(relu_11, 0);  relu_11 = None
    where_89: "f32[8, 128, 28, 28]" = torch.ops.aten.where.self(le_89, full_default, getitem_501);  le_89 = getitem_501 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    sum_183: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_89, [0, 2, 3])
    sub_469: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_1514);  convolution_13 = unsqueeze_1514 = None
    mul_1554: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_89, sub_469)
    sum_184: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1554, [0, 2, 3]);  mul_1554 = None
    mul_1555: "f32[128]" = torch.ops.aten.mul.Tensor(sum_183, 0.00015943877551020407)
    unsqueeze_1515: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1555, 0);  mul_1555 = None
    unsqueeze_1516: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1515, 2);  unsqueeze_1515 = None
    unsqueeze_1517: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1516, 3);  unsqueeze_1516 = None
    mul_1556: "f32[128]" = torch.ops.aten.mul.Tensor(sum_184, 0.00015943877551020407)
    mul_1557: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_40, squeeze_40)
    mul_1558: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1556, mul_1557);  mul_1556 = mul_1557 = None
    unsqueeze_1518: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1558, 0);  mul_1558 = None
    unsqueeze_1519: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1518, 2);  unsqueeze_1518 = None
    unsqueeze_1520: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1519, 3);  unsqueeze_1519 = None
    mul_1559: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_40, primals_41);  primals_41 = None
    unsqueeze_1521: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1559, 0);  mul_1559 = None
    unsqueeze_1522: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1521, 2);  unsqueeze_1521 = None
    unsqueeze_1523: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1522, 3);  unsqueeze_1522 = None
    mul_1560: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_469, unsqueeze_1520);  sub_469 = unsqueeze_1520 = None
    sub_471: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(where_89, mul_1560);  where_89 = mul_1560 = None
    sub_472: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(sub_471, unsqueeze_1517);  sub_471 = unsqueeze_1517 = None
    mul_1561: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_472, unsqueeze_1523);  sub_472 = unsqueeze_1523 = None
    mul_1562: "f32[128]" = torch.ops.aten.mul.Tensor(sum_184, squeeze_40);  sum_184 = squeeze_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    convolution_backward_92 = torch.ops.aten.convolution_backward.default(mul_1561, relu_10, primals_40, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1561 = primals_40 = None
    getitem_504: "f32[8, 128, 56, 56]" = convolution_backward_92[0]
    getitem_505: "f32[128, 128, 3, 3]" = convolution_backward_92[1];  convolution_backward_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    le_90: "b8[8, 128, 56, 56]" = torch.ops.aten.le.Scalar(relu_10, 0);  relu_10 = None
    where_90: "f32[8, 128, 56, 56]" = torch.ops.aten.where.self(le_90, full_default, getitem_504);  le_90 = getitem_504 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    sum_185: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_90, [0, 2, 3])
    sub_473: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_1526);  convolution_12 = unsqueeze_1526 = None
    mul_1563: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(where_90, sub_473)
    sum_186: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1563, [0, 2, 3]);  mul_1563 = None
    mul_1564: "f32[128]" = torch.ops.aten.mul.Tensor(sum_185, 3.985969387755102e-05)
    unsqueeze_1527: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1564, 0);  mul_1564 = None
    unsqueeze_1528: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1527, 2);  unsqueeze_1527 = None
    unsqueeze_1529: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1528, 3);  unsqueeze_1528 = None
    mul_1565: "f32[128]" = torch.ops.aten.mul.Tensor(sum_186, 3.985969387755102e-05)
    mul_1566: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
    mul_1567: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1565, mul_1566);  mul_1565 = mul_1566 = None
    unsqueeze_1530: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1567, 0);  mul_1567 = None
    unsqueeze_1531: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1530, 2);  unsqueeze_1530 = None
    unsqueeze_1532: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1531, 3);  unsqueeze_1531 = None
    mul_1568: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_37, primals_38);  primals_38 = None
    unsqueeze_1533: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1568, 0);  mul_1568 = None
    unsqueeze_1534: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1533, 2);  unsqueeze_1533 = None
    unsqueeze_1535: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1534, 3);  unsqueeze_1534 = None
    mul_1569: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_473, unsqueeze_1532);  sub_473 = unsqueeze_1532 = None
    sub_475: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(where_90, mul_1569);  where_90 = mul_1569 = None
    sub_476: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(sub_475, unsqueeze_1529);  sub_475 = unsqueeze_1529 = None
    mul_1570: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_476, unsqueeze_1535);  sub_476 = unsqueeze_1535 = None
    mul_1571: "f32[128]" = torch.ops.aten.mul.Tensor(sum_186, squeeze_37);  sum_186 = squeeze_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    convolution_backward_93 = torch.ops.aten.convolution_backward.default(mul_1570, relu_9, primals_37, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1570 = primals_37 = None
    getitem_507: "f32[8, 128, 56, 56]" = convolution_backward_93[0]
    getitem_508: "f32[128, 128, 1, 1]" = convolution_backward_93[1];  convolution_backward_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:248, code: shortcut = self.project(bottom)
    sub_477: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_1538);  convolution_11 = unsqueeze_1538 = None
    mul_1572: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(where_88, sub_477)
    sum_188: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1572, [0, 2, 3]);  mul_1572 = None
    mul_1574: "f32[256]" = torch.ops.aten.mul.Tensor(sum_188, 0.00015943877551020407)
    mul_1575: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_34, squeeze_34)
    mul_1576: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1574, mul_1575);  mul_1574 = mul_1575 = None
    unsqueeze_1542: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1576, 0);  mul_1576 = None
    unsqueeze_1543: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1542, 2);  unsqueeze_1542 = None
    unsqueeze_1544: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1543, 3);  unsqueeze_1543 = None
    mul_1577: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_34, primals_35);  primals_35 = None
    unsqueeze_1545: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1577, 0);  mul_1577 = None
    unsqueeze_1546: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1545, 2);  unsqueeze_1545 = None
    unsqueeze_1547: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1546, 3);  unsqueeze_1546 = None
    mul_1578: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_477, unsqueeze_1544);  sub_477 = unsqueeze_1544 = None
    sub_479: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(where_88, mul_1578);  where_88 = mul_1578 = None
    sub_480: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(sub_479, unsqueeze_1505);  sub_479 = unsqueeze_1505 = None
    mul_1579: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_480, unsqueeze_1547);  sub_480 = unsqueeze_1547 = None
    mul_1580: "f32[256]" = torch.ops.aten.mul.Tensor(sum_188, squeeze_34);  sum_188 = squeeze_34 = None
    convolution_backward_94 = torch.ops.aten.convolution_backward.default(mul_1579, getitem_24, primals_34, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1579 = getitem_24 = primals_34 = None
    getitem_510: "f32[8, 128, 28, 28]" = convolution_backward_94[0]
    getitem_511: "f32[256, 128, 1, 1]" = convolution_backward_94[1];  convolution_backward_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:247, code: bottom = self.downsample(x)
    max_pool2d_with_indices_backward_3: "f32[8, 128, 56, 56]" = torch.ops.aten.max_pool2d_with_indices_backward.default(getitem_510, relu_9, [2, 2], [2, 2], [0, 0], [1, 1], False, getitem_25);  getitem_510 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:247, code: bottom = self.downsample(x)
    add_630: "f32[8, 128, 56, 56]" = torch.ops.aten.add.Tensor(getitem_507, max_pool2d_with_indices_backward_3);  getitem_507 = max_pool2d_with_indices_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:247, code: bottom = self.downsample(x)
    max_pool2d_with_indices_backward_4: "f32[8, 128, 56, 56]" = torch.ops.aten.max_pool2d_with_indices_backward.default(slice_30, relu_9, [2, 2], [2, 2], [0, 0], [1, 1], False, getitem_25);  slice_30 = getitem_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:247, code: bottom = self.downsample(x)
    add_631: "f32[8, 128, 56, 56]" = torch.ops.aten.add.Tensor(add_630, max_pool2d_with_indices_backward_4);  add_630 = max_pool2d_with_indices_backward_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:180, code: x = self.relu(x)
    le_91: "b8[8, 128, 56, 56]" = torch.ops.aten.le.Scalar(relu_9, 0);  relu_9 = None
    where_91: "f32[8, 128, 56, 56]" = torch.ops.aten.where.self(le_91, full_default, add_631);  le_91 = add_631 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:177, code: x = self.bn(x)
    sum_189: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_91, [0, 2, 3])
    sub_481: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_1550);  convolution_10 = unsqueeze_1550 = None
    mul_1581: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(where_91, sub_481)
    sum_190: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1581, [0, 2, 3]);  mul_1581 = None
    mul_1582: "f32[128]" = torch.ops.aten.mul.Tensor(sum_189, 3.985969387755102e-05)
    unsqueeze_1551: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1582, 0);  mul_1582 = None
    unsqueeze_1552: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1551, 2);  unsqueeze_1551 = None
    unsqueeze_1553: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1552, 3);  unsqueeze_1552 = None
    mul_1583: "f32[128]" = torch.ops.aten.mul.Tensor(sum_190, 3.985969387755102e-05)
    mul_1584: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
    mul_1585: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1583, mul_1584);  mul_1583 = mul_1584 = None
    unsqueeze_1554: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1585, 0);  mul_1585 = None
    unsqueeze_1555: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1554, 2);  unsqueeze_1554 = None
    unsqueeze_1556: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1555, 3);  unsqueeze_1555 = None
    mul_1586: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_31, primals_32);  primals_32 = None
    unsqueeze_1557: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1586, 0);  mul_1586 = None
    unsqueeze_1558: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1557, 2);  unsqueeze_1557 = None
    unsqueeze_1559: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1558, 3);  unsqueeze_1558 = None
    mul_1587: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_481, unsqueeze_1556);  sub_481 = unsqueeze_1556 = None
    sub_483: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(where_91, mul_1587);  mul_1587 = None
    sub_484: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(sub_483, unsqueeze_1553);  sub_483 = unsqueeze_1553 = None
    mul_1588: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_484, unsqueeze_1559);  sub_484 = unsqueeze_1559 = None
    mul_1589: "f32[128]" = torch.ops.aten.mul.Tensor(sum_190, squeeze_31);  sum_190 = squeeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:176, code: x = self.conv(torch.cat(x_children, 1))
    convolution_backward_95 = torch.ops.aten.convolution_backward.default(mul_1588, cat, primals_31, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1588 = cat = primals_31 = None
    getitem_513: "f32[8, 256, 56, 56]" = convolution_backward_95[0]
    getitem_514: "f32[128, 256, 1, 1]" = convolution_backward_95[1];  convolution_backward_95 = None
    slice_40: "f32[8, 128, 56, 56]" = torch.ops.aten.slice.Tensor(getitem_513, 1, 0, 128)
    slice_41: "f32[8, 128, 56, 56]" = torch.ops.aten.slice.Tensor(getitem_513, 1, 128, 256);  getitem_513 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:176, code: x = self.conv(torch.cat(x_children, 1))
    add_632: "f32[8, 128, 56, 56]" = torch.ops.aten.add.Tensor(where_91, slice_40);  where_91 = slice_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    where_92: "f32[8, 128, 56, 56]" = torch.ops.aten.where.self(le_92, full_default, add_632);  le_92 = add_632 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    add_633: "f32[8, 128, 56, 56]" = torch.ops.aten.add.Tensor(slice_41, where_92);  slice_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    sum_191: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_92, [0, 2, 3])
    sub_485: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_1562);  convolution_9 = unsqueeze_1562 = None
    mul_1590: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(where_92, sub_485)
    sum_192: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1590, [0, 2, 3]);  mul_1590 = None
    mul_1591: "f32[128]" = torch.ops.aten.mul.Tensor(sum_191, 3.985969387755102e-05)
    unsqueeze_1563: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1591, 0);  mul_1591 = None
    unsqueeze_1564: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1563, 2);  unsqueeze_1563 = None
    unsqueeze_1565: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1564, 3);  unsqueeze_1564 = None
    mul_1592: "f32[128]" = torch.ops.aten.mul.Tensor(sum_192, 3.985969387755102e-05)
    mul_1593: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_28, squeeze_28)
    mul_1594: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1592, mul_1593);  mul_1592 = mul_1593 = None
    unsqueeze_1566: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1594, 0);  mul_1594 = None
    unsqueeze_1567: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1566, 2);  unsqueeze_1566 = None
    unsqueeze_1568: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1567, 3);  unsqueeze_1567 = None
    mul_1595: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_28, primals_29);  primals_29 = None
    unsqueeze_1569: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1595, 0);  mul_1595 = None
    unsqueeze_1570: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1569, 2);  unsqueeze_1569 = None
    unsqueeze_1571: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1570, 3);  unsqueeze_1570 = None
    mul_1596: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_485, unsqueeze_1568);  sub_485 = unsqueeze_1568 = None
    sub_487: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(where_92, mul_1596);  where_92 = mul_1596 = None
    sub_488: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(sub_487, unsqueeze_1565);  sub_487 = unsqueeze_1565 = None
    mul_1597: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_488, unsqueeze_1571);  sub_488 = unsqueeze_1571 = None
    mul_1598: "f32[128]" = torch.ops.aten.mul.Tensor(sum_192, squeeze_28);  sum_192 = squeeze_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    convolution_backward_96 = torch.ops.aten.convolution_backward.default(mul_1597, relu_7, primals_28, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1597 = primals_28 = None
    getitem_516: "f32[8, 64, 56, 56]" = convolution_backward_96[0]
    getitem_517: "f32[128, 64, 1, 1]" = convolution_backward_96[1];  convolution_backward_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    le_93: "b8[8, 64, 56, 56]" = torch.ops.aten.le.Scalar(relu_7, 0);  relu_7 = None
    where_93: "f32[8, 64, 56, 56]" = torch.ops.aten.where.self(le_93, full_default, getitem_516);  le_93 = getitem_516 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    sum_193: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_93, [0, 2, 3])
    sub_489: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_1574);  convolution_8 = unsqueeze_1574 = None
    mul_1599: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(where_93, sub_489)
    sum_194: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1599, [0, 2, 3]);  mul_1599 = None
    mul_1600: "f32[64]" = torch.ops.aten.mul.Tensor(sum_193, 3.985969387755102e-05)
    unsqueeze_1575: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1600, 0);  mul_1600 = None
    unsqueeze_1576: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1575, 2);  unsqueeze_1575 = None
    unsqueeze_1577: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1576, 3);  unsqueeze_1576 = None
    mul_1601: "f32[64]" = torch.ops.aten.mul.Tensor(sum_194, 3.985969387755102e-05)
    mul_1602: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
    mul_1603: "f32[64]" = torch.ops.aten.mul.Tensor(mul_1601, mul_1602);  mul_1601 = mul_1602 = None
    unsqueeze_1578: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1603, 0);  mul_1603 = None
    unsqueeze_1579: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1578, 2);  unsqueeze_1578 = None
    unsqueeze_1580: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1579, 3);  unsqueeze_1579 = None
    mul_1604: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_25, primals_26);  primals_26 = None
    unsqueeze_1581: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1604, 0);  mul_1604 = None
    unsqueeze_1582: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1581, 2);  unsqueeze_1581 = None
    unsqueeze_1583: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1582, 3);  unsqueeze_1582 = None
    mul_1605: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_489, unsqueeze_1580);  sub_489 = unsqueeze_1580 = None
    sub_491: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(where_93, mul_1605);  where_93 = mul_1605 = None
    sub_492: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(sub_491, unsqueeze_1577);  sub_491 = unsqueeze_1577 = None
    mul_1606: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_492, unsqueeze_1583);  sub_492 = unsqueeze_1583 = None
    mul_1607: "f32[64]" = torch.ops.aten.mul.Tensor(sum_194, squeeze_25);  sum_194 = squeeze_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    convolution_backward_97 = torch.ops.aten.convolution_backward.default(mul_1606, relu_6, primals_25, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1606 = primals_25 = None
    getitem_519: "f32[8, 64, 56, 56]" = convolution_backward_97[0]
    getitem_520: "f32[64, 64, 3, 3]" = convolution_backward_97[1];  convolution_backward_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    le_94: "b8[8, 64, 56, 56]" = torch.ops.aten.le.Scalar(relu_6, 0);  relu_6 = None
    where_94: "f32[8, 64, 56, 56]" = torch.ops.aten.where.self(le_94, full_default, getitem_519);  le_94 = getitem_519 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    sum_195: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_94, [0, 2, 3])
    sub_493: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_1586);  convolution_7 = unsqueeze_1586 = None
    mul_1608: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(where_94, sub_493)
    sum_196: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1608, [0, 2, 3]);  mul_1608 = None
    mul_1609: "f32[64]" = torch.ops.aten.mul.Tensor(sum_195, 3.985969387755102e-05)
    unsqueeze_1587: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1609, 0);  mul_1609 = None
    unsqueeze_1588: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1587, 2);  unsqueeze_1587 = None
    unsqueeze_1589: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1588, 3);  unsqueeze_1588 = None
    mul_1610: "f32[64]" = torch.ops.aten.mul.Tensor(sum_196, 3.985969387755102e-05)
    mul_1611: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_22, squeeze_22)
    mul_1612: "f32[64]" = torch.ops.aten.mul.Tensor(mul_1610, mul_1611);  mul_1610 = mul_1611 = None
    unsqueeze_1590: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1612, 0);  mul_1612 = None
    unsqueeze_1591: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1590, 2);  unsqueeze_1590 = None
    unsqueeze_1592: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1591, 3);  unsqueeze_1591 = None
    mul_1613: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_22, primals_23);  primals_23 = None
    unsqueeze_1593: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1613, 0);  mul_1613 = None
    unsqueeze_1594: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1593, 2);  unsqueeze_1593 = None
    unsqueeze_1595: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1594, 3);  unsqueeze_1594 = None
    mul_1614: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_493, unsqueeze_1592);  sub_493 = unsqueeze_1592 = None
    sub_495: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(where_94, mul_1614);  where_94 = mul_1614 = None
    sub_496: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(sub_495, unsqueeze_1589);  sub_495 = unsqueeze_1589 = None
    mul_1615: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_496, unsqueeze_1595);  sub_496 = unsqueeze_1595 = None
    mul_1616: "f32[64]" = torch.ops.aten.mul.Tensor(sum_196, squeeze_22);  sum_196 = squeeze_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    convolution_backward_98 = torch.ops.aten.convolution_backward.default(mul_1615, relu_5, primals_22, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1615 = primals_22 = None
    getitem_522: "f32[8, 128, 56, 56]" = convolution_backward_98[0]
    getitem_523: "f32[64, 128, 1, 1]" = convolution_backward_98[1];  convolution_backward_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    add_634: "f32[8, 128, 56, 56]" = torch.ops.aten.add.Tensor(add_633, getitem_522);  add_633 = getitem_522 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    le_95: "b8[8, 128, 56, 56]" = torch.ops.aten.le.Scalar(relu_5, 0);  relu_5 = None
    where_95: "f32[8, 128, 56, 56]" = torch.ops.aten.where.self(le_95, full_default, add_634);  le_95 = add_634 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    sum_197: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_95, [0, 2, 3])
    sub_497: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_1598);  convolution_6 = unsqueeze_1598 = None
    mul_1617: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(where_95, sub_497)
    sum_198: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1617, [0, 2, 3]);  mul_1617 = None
    mul_1618: "f32[128]" = torch.ops.aten.mul.Tensor(sum_197, 3.985969387755102e-05)
    unsqueeze_1599: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1618, 0);  mul_1618 = None
    unsqueeze_1600: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1599, 2);  unsqueeze_1599 = None
    unsqueeze_1601: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1600, 3);  unsqueeze_1600 = None
    mul_1619: "f32[128]" = torch.ops.aten.mul.Tensor(sum_198, 3.985969387755102e-05)
    mul_1620: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_19, squeeze_19)
    mul_1621: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1619, mul_1620);  mul_1619 = mul_1620 = None
    unsqueeze_1602: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1621, 0);  mul_1621 = None
    unsqueeze_1603: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1602, 2);  unsqueeze_1602 = None
    unsqueeze_1604: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1603, 3);  unsqueeze_1603 = None
    mul_1622: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_19, primals_20);  primals_20 = None
    unsqueeze_1605: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1622, 0);  mul_1622 = None
    unsqueeze_1606: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1605, 2);  unsqueeze_1605 = None
    unsqueeze_1607: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1606, 3);  unsqueeze_1606 = None
    mul_1623: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_497, unsqueeze_1604);  sub_497 = unsqueeze_1604 = None
    sub_499: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(where_95, mul_1623);  mul_1623 = None
    sub_500: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(sub_499, unsqueeze_1601);  sub_499 = None
    mul_1624: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_500, unsqueeze_1607);  sub_500 = unsqueeze_1607 = None
    mul_1625: "f32[128]" = torch.ops.aten.mul.Tensor(sum_198, squeeze_19);  sum_198 = squeeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    convolution_backward_99 = torch.ops.aten.convolution_backward.default(mul_1624, relu_4, primals_19, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1624 = primals_19 = None
    getitem_525: "f32[8, 64, 56, 56]" = convolution_backward_99[0]
    getitem_526: "f32[128, 64, 1, 1]" = convolution_backward_99[1];  convolution_backward_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    le_96: "b8[8, 64, 56, 56]" = torch.ops.aten.le.Scalar(relu_4, 0);  relu_4 = None
    where_96: "f32[8, 64, 56, 56]" = torch.ops.aten.where.self(le_96, full_default, getitem_525);  le_96 = getitem_525 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    sum_199: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_96, [0, 2, 3])
    sub_501: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_1610);  convolution_5 = unsqueeze_1610 = None
    mul_1626: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(where_96, sub_501)
    sum_200: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1626, [0, 2, 3]);  mul_1626 = None
    mul_1627: "f32[64]" = torch.ops.aten.mul.Tensor(sum_199, 3.985969387755102e-05)
    unsqueeze_1611: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1627, 0);  mul_1627 = None
    unsqueeze_1612: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1611, 2);  unsqueeze_1611 = None
    unsqueeze_1613: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1612, 3);  unsqueeze_1612 = None
    mul_1628: "f32[64]" = torch.ops.aten.mul.Tensor(sum_200, 3.985969387755102e-05)
    mul_1629: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_16, squeeze_16)
    mul_1630: "f32[64]" = torch.ops.aten.mul.Tensor(mul_1628, mul_1629);  mul_1628 = mul_1629 = None
    unsqueeze_1614: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1630, 0);  mul_1630 = None
    unsqueeze_1615: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1614, 2);  unsqueeze_1614 = None
    unsqueeze_1616: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1615, 3);  unsqueeze_1615 = None
    mul_1631: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_16, primals_17);  primals_17 = None
    unsqueeze_1617: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1631, 0);  mul_1631 = None
    unsqueeze_1618: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1617, 2);  unsqueeze_1617 = None
    unsqueeze_1619: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1618, 3);  unsqueeze_1618 = None
    mul_1632: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_501, unsqueeze_1616);  sub_501 = unsqueeze_1616 = None
    sub_503: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(where_96, mul_1632);  where_96 = mul_1632 = None
    sub_504: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(sub_503, unsqueeze_1613);  sub_503 = unsqueeze_1613 = None
    mul_1633: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_504, unsqueeze_1619);  sub_504 = unsqueeze_1619 = None
    mul_1634: "f32[64]" = torch.ops.aten.mul.Tensor(sum_200, squeeze_16);  sum_200 = squeeze_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    convolution_backward_100 = torch.ops.aten.convolution_backward.default(mul_1633, relu_3, primals_16, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1633 = primals_16 = None
    getitem_528: "f32[8, 64, 112, 112]" = convolution_backward_100[0]
    getitem_529: "f32[64, 64, 3, 3]" = convolution_backward_100[1];  convolution_backward_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    le_97: "b8[8, 64, 112, 112]" = torch.ops.aten.le.Scalar(relu_3, 0);  relu_3 = None
    where_97: "f32[8, 64, 112, 112]" = torch.ops.aten.where.self(le_97, full_default, getitem_528);  le_97 = getitem_528 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    sum_201: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_97, [0, 2, 3])
    sub_505: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_1622);  convolution_4 = unsqueeze_1622 = None
    mul_1635: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(where_97, sub_505)
    sum_202: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1635, [0, 2, 3]);  mul_1635 = None
    mul_1636: "f32[64]" = torch.ops.aten.mul.Tensor(sum_201, 9.964923469387754e-06)
    unsqueeze_1623: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1636, 0);  mul_1636 = None
    unsqueeze_1624: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1623, 2);  unsqueeze_1623 = None
    unsqueeze_1625: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1624, 3);  unsqueeze_1624 = None
    mul_1637: "f32[64]" = torch.ops.aten.mul.Tensor(sum_202, 9.964923469387754e-06)
    mul_1638: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
    mul_1639: "f32[64]" = torch.ops.aten.mul.Tensor(mul_1637, mul_1638);  mul_1637 = mul_1638 = None
    unsqueeze_1626: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1639, 0);  mul_1639 = None
    unsqueeze_1627: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1626, 2);  unsqueeze_1626 = None
    unsqueeze_1628: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1627, 3);  unsqueeze_1627 = None
    mul_1640: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_13, primals_14);  primals_14 = None
    unsqueeze_1629: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1640, 0);  mul_1640 = None
    unsqueeze_1630: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1629, 2);  unsqueeze_1629 = None
    unsqueeze_1631: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1630, 3);  unsqueeze_1630 = None
    mul_1641: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_505, unsqueeze_1628);  sub_505 = unsqueeze_1628 = None
    sub_507: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(where_97, mul_1641);  where_97 = mul_1641 = None
    sub_508: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(sub_507, unsqueeze_1625);  sub_507 = unsqueeze_1625 = None
    mul_1642: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_508, unsqueeze_1631);  sub_508 = unsqueeze_1631 = None
    mul_1643: "f32[64]" = torch.ops.aten.mul.Tensor(sum_202, squeeze_13);  sum_202 = squeeze_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    convolution_backward_101 = torch.ops.aten.convolution_backward.default(mul_1642, relu_2, primals_13, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1642 = primals_13 = None
    getitem_531: "f32[8, 32, 112, 112]" = convolution_backward_101[0]
    getitem_532: "f32[64, 32, 1, 1]" = convolution_backward_101[1];  convolution_backward_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:248, code: shortcut = self.project(bottom)
    sub_509: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_1634);  convolution_3 = unsqueeze_1634 = None
    mul_1644: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(where_95, sub_509)
    sum_204: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1644, [0, 2, 3]);  mul_1644 = None
    mul_1646: "f32[128]" = torch.ops.aten.mul.Tensor(sum_204, 3.985969387755102e-05)
    mul_1647: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_10, squeeze_10)
    mul_1648: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1646, mul_1647);  mul_1646 = mul_1647 = None
    unsqueeze_1638: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1648, 0);  mul_1648 = None
    unsqueeze_1639: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1638, 2);  unsqueeze_1638 = None
    unsqueeze_1640: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1639, 3);  unsqueeze_1639 = None
    mul_1649: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_10, primals_11);  primals_11 = None
    unsqueeze_1641: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1649, 0);  mul_1649 = None
    unsqueeze_1642: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1641, 2);  unsqueeze_1641 = None
    unsqueeze_1643: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1642, 3);  unsqueeze_1642 = None
    mul_1650: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_509, unsqueeze_1640);  sub_509 = unsqueeze_1640 = None
    sub_511: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(where_95, mul_1650);  where_95 = mul_1650 = None
    sub_512: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(sub_511, unsqueeze_1601);  sub_511 = unsqueeze_1601 = None
    mul_1651: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_512, unsqueeze_1643);  sub_512 = unsqueeze_1643 = None
    mul_1652: "f32[128]" = torch.ops.aten.mul.Tensor(sum_204, squeeze_10);  sum_204 = squeeze_10 = None
    convolution_backward_102 = torch.ops.aten.convolution_backward.default(mul_1651, getitem_6, primals_10, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1651 = getitem_6 = primals_10 = None
    getitem_534: "f32[8, 32, 56, 56]" = convolution_backward_102[0]
    getitem_535: "f32[128, 32, 1, 1]" = convolution_backward_102[1];  convolution_backward_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:247, code: bottom = self.downsample(x)
    max_pool2d_with_indices_backward_5: "f32[8, 32, 112, 112]" = torch.ops.aten.max_pool2d_with_indices_backward.default(getitem_534, relu_2, [2, 2], [2, 2], [0, 0], [1, 1], False, getitem_7);  getitem_534 = getitem_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:247, code: bottom = self.downsample(x)
    add_635: "f32[8, 32, 112, 112]" = torch.ops.aten.add.Tensor(getitem_531, max_pool2d_with_indices_backward_5);  getitem_531 = max_pool2d_with_indices_backward_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:365, code: x = self.level1(x)
    le_98: "b8[8, 32, 112, 112]" = torch.ops.aten.le.Scalar(relu_2, 0);  relu_2 = None
    where_98: "f32[8, 32, 112, 112]" = torch.ops.aten.where.self(le_98, full_default, add_635);  le_98 = add_635 = None
    sum_205: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_98, [0, 2, 3])
    sub_513: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_1646);  convolution_2 = unsqueeze_1646 = None
    mul_1653: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(where_98, sub_513)
    sum_206: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_1653, [0, 2, 3]);  mul_1653 = None
    mul_1654: "f32[32]" = torch.ops.aten.mul.Tensor(sum_205, 9.964923469387754e-06)
    unsqueeze_1647: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1654, 0);  mul_1654 = None
    unsqueeze_1648: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1647, 2);  unsqueeze_1647 = None
    unsqueeze_1649: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1648, 3);  unsqueeze_1648 = None
    mul_1655: "f32[32]" = torch.ops.aten.mul.Tensor(sum_206, 9.964923469387754e-06)
    mul_1656: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
    mul_1657: "f32[32]" = torch.ops.aten.mul.Tensor(mul_1655, mul_1656);  mul_1655 = mul_1656 = None
    unsqueeze_1650: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1657, 0);  mul_1657 = None
    unsqueeze_1651: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1650, 2);  unsqueeze_1650 = None
    unsqueeze_1652: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1651, 3);  unsqueeze_1651 = None
    mul_1658: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_7, primals_8);  primals_8 = None
    unsqueeze_1653: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1658, 0);  mul_1658 = None
    unsqueeze_1654: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1653, 2);  unsqueeze_1653 = None
    unsqueeze_1655: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1654, 3);  unsqueeze_1654 = None
    mul_1659: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_513, unsqueeze_1652);  sub_513 = unsqueeze_1652 = None
    sub_515: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(where_98, mul_1659);  where_98 = mul_1659 = None
    sub_516: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(sub_515, unsqueeze_1649);  sub_515 = unsqueeze_1649 = None
    mul_1660: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_516, unsqueeze_1655);  sub_516 = unsqueeze_1655 = None
    mul_1661: "f32[32]" = torch.ops.aten.mul.Tensor(sum_206, squeeze_7);  sum_206 = squeeze_7 = None
    convolution_backward_103 = torch.ops.aten.convolution_backward.default(mul_1660, relu_1, primals_7, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1660 = primals_7 = None
    getitem_537: "f32[8, 16, 224, 224]" = convolution_backward_103[0]
    getitem_538: "f32[32, 16, 3, 3]" = convolution_backward_103[1];  convolution_backward_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:364, code: x = self.level0(x)
    le_99: "b8[8, 16, 224, 224]" = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None
    where_99: "f32[8, 16, 224, 224]" = torch.ops.aten.where.self(le_99, full_default, getitem_537);  le_99 = getitem_537 = None
    sum_207: "f32[16]" = torch.ops.aten.sum.dim_IntList(where_99, [0, 2, 3])
    sub_517: "f32[8, 16, 224, 224]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_1658);  convolution_1 = unsqueeze_1658 = None
    mul_1662: "f32[8, 16, 224, 224]" = torch.ops.aten.mul.Tensor(where_99, sub_517)
    sum_208: "f32[16]" = torch.ops.aten.sum.dim_IntList(mul_1662, [0, 2, 3]);  mul_1662 = None
    mul_1663: "f32[16]" = torch.ops.aten.mul.Tensor(sum_207, 2.4912308673469386e-06)
    unsqueeze_1659: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1663, 0);  mul_1663 = None
    unsqueeze_1660: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1659, 2);  unsqueeze_1659 = None
    unsqueeze_1661: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1660, 3);  unsqueeze_1660 = None
    mul_1664: "f32[16]" = torch.ops.aten.mul.Tensor(sum_208, 2.4912308673469386e-06)
    mul_1665: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
    mul_1666: "f32[16]" = torch.ops.aten.mul.Tensor(mul_1664, mul_1665);  mul_1664 = mul_1665 = None
    unsqueeze_1662: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1666, 0);  mul_1666 = None
    unsqueeze_1663: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1662, 2);  unsqueeze_1662 = None
    unsqueeze_1664: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1663, 3);  unsqueeze_1663 = None
    mul_1667: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_4, primals_5);  primals_5 = None
    unsqueeze_1665: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1667, 0);  mul_1667 = None
    unsqueeze_1666: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1665, 2);  unsqueeze_1665 = None
    unsqueeze_1667: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1666, 3);  unsqueeze_1666 = None
    mul_1668: "f32[8, 16, 224, 224]" = torch.ops.aten.mul.Tensor(sub_517, unsqueeze_1664);  sub_517 = unsqueeze_1664 = None
    sub_519: "f32[8, 16, 224, 224]" = torch.ops.aten.sub.Tensor(where_99, mul_1668);  where_99 = mul_1668 = None
    sub_520: "f32[8, 16, 224, 224]" = torch.ops.aten.sub.Tensor(sub_519, unsqueeze_1661);  sub_519 = unsqueeze_1661 = None
    mul_1669: "f32[8, 16, 224, 224]" = torch.ops.aten.mul.Tensor(sub_520, unsqueeze_1667);  sub_520 = unsqueeze_1667 = None
    mul_1670: "f32[16]" = torch.ops.aten.mul.Tensor(sum_208, squeeze_4);  sum_208 = squeeze_4 = None
    convolution_backward_104 = torch.ops.aten.convolution_backward.default(mul_1669, relu, primals_4, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1669 = primals_4 = None
    getitem_540: "f32[8, 16, 224, 224]" = convolution_backward_104[0]
    getitem_541: "f32[16, 16, 3, 3]" = convolution_backward_104[1];  convolution_backward_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:363, code: x = self.base_layer(x)
    le_100: "b8[8, 16, 224, 224]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
    where_100: "f32[8, 16, 224, 224]" = torch.ops.aten.where.self(le_100, full_default, getitem_540);  le_100 = full_default = getitem_540 = None
    sum_209: "f32[16]" = torch.ops.aten.sum.dim_IntList(where_100, [0, 2, 3])
    sub_521: "f32[8, 16, 224, 224]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_1670);  convolution = unsqueeze_1670 = None
    mul_1671: "f32[8, 16, 224, 224]" = torch.ops.aten.mul.Tensor(where_100, sub_521)
    sum_210: "f32[16]" = torch.ops.aten.sum.dim_IntList(mul_1671, [0, 2, 3]);  mul_1671 = None
    mul_1672: "f32[16]" = torch.ops.aten.mul.Tensor(sum_209, 2.4912308673469386e-06)
    unsqueeze_1671: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1672, 0);  mul_1672 = None
    unsqueeze_1672: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1671, 2);  unsqueeze_1671 = None
    unsqueeze_1673: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1672, 3);  unsqueeze_1672 = None
    mul_1673: "f32[16]" = torch.ops.aten.mul.Tensor(sum_210, 2.4912308673469386e-06)
    mul_1674: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
    mul_1675: "f32[16]" = torch.ops.aten.mul.Tensor(mul_1673, mul_1674);  mul_1673 = mul_1674 = None
    unsqueeze_1674: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1675, 0);  mul_1675 = None
    unsqueeze_1675: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1674, 2);  unsqueeze_1674 = None
    unsqueeze_1676: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1675, 3);  unsqueeze_1675 = None
    mul_1676: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_1, primals_2);  primals_2 = None
    unsqueeze_1677: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1676, 0);  mul_1676 = None
    unsqueeze_1678: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1677, 2);  unsqueeze_1677 = None
    unsqueeze_1679: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1678, 3);  unsqueeze_1678 = None
    mul_1677: "f32[8, 16, 224, 224]" = torch.ops.aten.mul.Tensor(sub_521, unsqueeze_1676);  sub_521 = unsqueeze_1676 = None
    sub_523: "f32[8, 16, 224, 224]" = torch.ops.aten.sub.Tensor(where_100, mul_1677);  where_100 = mul_1677 = None
    sub_524: "f32[8, 16, 224, 224]" = torch.ops.aten.sub.Tensor(sub_523, unsqueeze_1673);  sub_523 = unsqueeze_1673 = None
    mul_1678: "f32[8, 16, 224, 224]" = torch.ops.aten.mul.Tensor(sub_524, unsqueeze_1679);  sub_524 = unsqueeze_1679 = None
    mul_1679: "f32[16]" = torch.ops.aten.mul.Tensor(sum_210, squeeze_1);  sum_210 = squeeze_1 = None
    convolution_backward_105 = torch.ops.aten.convolution_backward.default(mul_1678, primals_633, primals_1, [0], [1, 1], [3, 3], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_1678 = primals_633 = primals_1 = None
    getitem_544: "f32[16, 3, 7, 7]" = convolution_backward_105[1];  convolution_backward_105 = None
    return [getitem_544, mul_1679, sum_209, getitem_541, mul_1670, sum_207, getitem_538, mul_1661, sum_205, getitem_535, mul_1652, sum_197, getitem_532, mul_1643, sum_201, getitem_529, mul_1634, sum_199, getitem_526, mul_1625, sum_197, getitem_523, mul_1616, sum_195, getitem_520, mul_1607, sum_193, getitem_517, mul_1598, sum_191, getitem_514, mul_1589, sum_189, getitem_511, mul_1580, sum_181, getitem_508, mul_1571, sum_185, getitem_505, mul_1562, sum_183, getitem_502, mul_1553, sum_181, getitem_499, mul_1544, sum_179, getitem_496, mul_1535, sum_177, getitem_493, mul_1526, sum_175, getitem_490, mul_1517, sum_173, getitem_487, mul_1508, sum_171, getitem_484, mul_1499, sum_169, getitem_481, mul_1490, sum_167, getitem_478, mul_1481, sum_165, getitem_475, mul_1472, sum_163, getitem_472, mul_1463, sum_161, getitem_469, mul_1454, sum_159, getitem_466, mul_1445, sum_157, getitem_463, mul_1436, sum_155, getitem_460, mul_1427, sum_153, getitem_457, mul_1418, sum_151, getitem_454, mul_1409, sum_149, getitem_451, mul_1400, sum_147, getitem_448, mul_1391, sum_145, getitem_445, mul_1382, sum_143, getitem_442, mul_1373, sum_141, getitem_439, mul_1364, sum_139, getitem_436, mul_1355, sum_137, getitem_433, mul_1346, sum_135, getitem_430, mul_1337, sum_133, getitem_427, mul_1328, sum_131, getitem_424, mul_1319, sum_123, getitem_421, mul_1310, sum_127, getitem_418, mul_1301, sum_125, getitem_415, mul_1292, sum_123, getitem_412, mul_1283, sum_121, getitem_409, mul_1274, sum_119, getitem_406, mul_1265, sum_117, getitem_403, mul_1256, sum_115, getitem_400, mul_1247, sum_113, getitem_397, mul_1238, sum_111, getitem_394, mul_1229, sum_109, getitem_391, mul_1220, sum_107, getitem_388, mul_1211, sum_105, getitem_385, mul_1202, sum_103, getitem_382, mul_1193, sum_101, getitem_379, mul_1184, sum_99, getitem_376, mul_1175, sum_97, getitem_373, mul_1166, sum_95, getitem_370, mul_1157, sum_93, getitem_367, mul_1148, sum_91, getitem_364, mul_1139, sum_89, getitem_361, mul_1130, sum_87, getitem_358, mul_1121, sum_85, getitem_355, mul_1112, sum_83, getitem_352, mul_1103, sum_81, getitem_349, mul_1094, sum_79, getitem_346, mul_1085, sum_77, getitem_343, mul_1076, sum_75, getitem_340, mul_1067, sum_73, getitem_337, mul_1058, sum_71, getitem_334, mul_1049, sum_69, getitem_331, mul_1040, sum_67, getitem_328, mul_1031, sum_65, getitem_325, mul_1022, sum_63, getitem_322, mul_1013, sum_61, getitem_319, mul_1004, sum_59, getitem_316, mul_995, sum_57, getitem_313, mul_986, sum_55, getitem_310, mul_977, sum_53, getitem_307, mul_968, sum_51, getitem_304, mul_959, sum_49, getitem_301, mul_950, sum_47, getitem_298, mul_941, sum_45, getitem_295, mul_932, sum_43, getitem_292, mul_923, sum_41, getitem_289, mul_914, sum_39, getitem_286, mul_905, sum_37, getitem_283, mul_896, sum_35, getitem_280, mul_887, sum_33, getitem_277, mul_878, sum_31, getitem_274, mul_869, sum_29, getitem_271, mul_860, sum_27, getitem_268, mul_851, sum_25, getitem_265, mul_842, sum_23, getitem_262, mul_833, sum_21, getitem_259, mul_824, sum_19, getitem_256, mul_815, sum_17, getitem_253, mul_806, sum_9, getitem_250, mul_797, sum_13, getitem_247, mul_788, sum_11, getitem_244, mul_779, sum_9, getitem_241, mul_770, sum_7, getitem_238, mul_761, sum_5, getitem_235, mul_752, sum_3, getitem_232, mul_743, sum_1, getitem_229, getitem_230, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
    