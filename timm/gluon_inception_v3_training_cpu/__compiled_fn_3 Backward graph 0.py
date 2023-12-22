from __future__ import annotations



def forward(self, primals_1: "f32[32]", primals_3: "f32[32]", primals_5: "f32[64]", primals_7: "f32[80]", primals_9: "f32[192]", primals_11: "f32[64]", primals_13: "f32[48]", primals_15: "f32[64]", primals_17: "f32[64]", primals_19: "f32[96]", primals_21: "f32[96]", primals_23: "f32[32]", primals_25: "f32[64]", primals_27: "f32[48]", primals_29: "f32[64]", primals_31: "f32[64]", primals_33: "f32[96]", primals_35: "f32[96]", primals_37: "f32[64]", primals_39: "f32[64]", primals_41: "f32[48]", primals_43: "f32[64]", primals_45: "f32[64]", primals_47: "f32[96]", primals_49: "f32[96]", primals_51: "f32[64]", primals_53: "f32[384]", primals_55: "f32[64]", primals_57: "f32[96]", primals_59: "f32[96]", primals_61: "f32[192]", primals_63: "f32[128]", primals_65: "f32[128]", primals_67: "f32[192]", primals_69: "f32[128]", primals_71: "f32[128]", primals_73: "f32[128]", primals_75: "f32[128]", primals_77: "f32[192]", primals_79: "f32[192]", primals_81: "f32[192]", primals_83: "f32[160]", primals_85: "f32[160]", primals_87: "f32[192]", primals_89: "f32[160]", primals_91: "f32[160]", primals_93: "f32[160]", primals_95: "f32[160]", primals_97: "f32[192]", primals_99: "f32[192]", primals_101: "f32[192]", primals_103: "f32[160]", primals_105: "f32[160]", primals_107: "f32[192]", primals_109: "f32[160]", primals_111: "f32[160]", primals_113: "f32[160]", primals_115: "f32[160]", primals_117: "f32[192]", primals_119: "f32[192]", primals_121: "f32[192]", primals_123: "f32[192]", primals_125: "f32[192]", primals_127: "f32[192]", primals_129: "f32[192]", primals_131: "f32[192]", primals_133: "f32[192]", primals_135: "f32[192]", primals_137: "f32[192]", primals_139: "f32[192]", primals_141: "f32[192]", primals_143: "f32[320]", primals_145: "f32[192]", primals_147: "f32[192]", primals_149: "f32[192]", primals_151: "f32[192]", primals_153: "f32[320]", primals_155: "f32[384]", primals_157: "f32[384]", primals_159: "f32[384]", primals_161: "f32[448]", primals_163: "f32[384]", primals_165: "f32[384]", primals_167: "f32[384]", primals_169: "f32[192]", primals_171: "f32[320]", primals_173: "f32[384]", primals_175: "f32[384]", primals_177: "f32[384]", primals_179: "f32[448]", primals_181: "f32[384]", primals_183: "f32[384]", primals_185: "f32[384]", primals_187: "f32[192]", primals_189: "f32[32, 3, 3, 3]", primals_190: "f32[32, 32, 3, 3]", primals_191: "f32[64, 32, 3, 3]", primals_192: "f32[80, 64, 1, 1]", primals_193: "f32[192, 80, 3, 3]", primals_194: "f32[64, 192, 1, 1]", primals_195: "f32[48, 192, 1, 1]", primals_196: "f32[64, 48, 5, 5]", primals_197: "f32[64, 192, 1, 1]", primals_198: "f32[96, 64, 3, 3]", primals_199: "f32[96, 96, 3, 3]", primals_200: "f32[32, 192, 1, 1]", primals_201: "f32[64, 256, 1, 1]", primals_202: "f32[48, 256, 1, 1]", primals_203: "f32[64, 48, 5, 5]", primals_204: "f32[64, 256, 1, 1]", primals_205: "f32[96, 64, 3, 3]", primals_206: "f32[96, 96, 3, 3]", primals_207: "f32[64, 256, 1, 1]", primals_208: "f32[64, 288, 1, 1]", primals_209: "f32[48, 288, 1, 1]", primals_210: "f32[64, 48, 5, 5]", primals_211: "f32[64, 288, 1, 1]", primals_212: "f32[96, 64, 3, 3]", primals_213: "f32[96, 96, 3, 3]", primals_214: "f32[64, 288, 1, 1]", primals_215: "f32[384, 288, 3, 3]", primals_216: "f32[64, 288, 1, 1]", primals_217: "f32[96, 64, 3, 3]", primals_218: "f32[96, 96, 3, 3]", primals_219: "f32[192, 768, 1, 1]", primals_220: "f32[128, 768, 1, 1]", primals_221: "f32[128, 128, 1, 7]", primals_222: "f32[192, 128, 7, 1]", primals_223: "f32[128, 768, 1, 1]", primals_224: "f32[128, 128, 7, 1]", primals_225: "f32[128, 128, 1, 7]", primals_226: "f32[128, 128, 7, 1]", primals_227: "f32[192, 128, 1, 7]", primals_228: "f32[192, 768, 1, 1]", primals_229: "f32[192, 768, 1, 1]", primals_230: "f32[160, 768, 1, 1]", primals_231: "f32[160, 160, 1, 7]", primals_232: "f32[192, 160, 7, 1]", primals_233: "f32[160, 768, 1, 1]", primals_234: "f32[160, 160, 7, 1]", primals_235: "f32[160, 160, 1, 7]", primals_236: "f32[160, 160, 7, 1]", primals_237: "f32[192, 160, 1, 7]", primals_238: "f32[192, 768, 1, 1]", primals_239: "f32[192, 768, 1, 1]", primals_240: "f32[160, 768, 1, 1]", primals_241: "f32[160, 160, 1, 7]", primals_242: "f32[192, 160, 7, 1]", primals_243: "f32[160, 768, 1, 1]", primals_244: "f32[160, 160, 7, 1]", primals_245: "f32[160, 160, 1, 7]", primals_246: "f32[160, 160, 7, 1]", primals_247: "f32[192, 160, 1, 7]", primals_248: "f32[192, 768, 1, 1]", primals_249: "f32[192, 768, 1, 1]", primals_250: "f32[192, 768, 1, 1]", primals_251: "f32[192, 192, 1, 7]", primals_252: "f32[192, 192, 7, 1]", primals_253: "f32[192, 768, 1, 1]", primals_254: "f32[192, 192, 7, 1]", primals_255: "f32[192, 192, 1, 7]", primals_256: "f32[192, 192, 7, 1]", primals_257: "f32[192, 192, 1, 7]", primals_258: "f32[192, 768, 1, 1]", primals_259: "f32[192, 768, 1, 1]", primals_260: "f32[320, 192, 3, 3]", primals_261: "f32[192, 768, 1, 1]", primals_262: "f32[192, 192, 1, 7]", primals_263: "f32[192, 192, 7, 1]", primals_264: "f32[192, 192, 3, 3]", primals_265: "f32[320, 1280, 1, 1]", primals_266: "f32[384, 1280, 1, 1]", primals_267: "f32[384, 384, 1, 3]", primals_268: "f32[384, 384, 3, 1]", primals_269: "f32[448, 1280, 1, 1]", primals_270: "f32[384, 448, 3, 3]", primals_271: "f32[384, 384, 1, 3]", primals_272: "f32[384, 384, 3, 1]", primals_273: "f32[192, 1280, 1, 1]", primals_274: "f32[320, 2048, 1, 1]", primals_275: "f32[384, 2048, 1, 1]", primals_276: "f32[384, 384, 1, 3]", primals_277: "f32[384, 384, 3, 1]", primals_278: "f32[448, 2048, 1, 1]", primals_279: "f32[384, 448, 3, 3]", primals_280: "f32[384, 384, 1, 3]", primals_281: "f32[384, 384, 3, 1]", primals_282: "f32[192, 2048, 1, 1]", primals_567: "f32[8, 3, 299, 299]", convolution: "f32[8, 32, 149, 149]", squeeze_1: "f32[32]", relu: "f32[8, 32, 149, 149]", convolution_1: "f32[8, 32, 147, 147]", squeeze_4: "f32[32]", relu_1: "f32[8, 32, 147, 147]", convolution_2: "f32[8, 64, 147, 147]", squeeze_7: "f32[64]", relu_2: "f32[8, 64, 147, 147]", getitem_6: "f32[8, 64, 73, 73]", getitem_7: "i64[8, 64, 73, 73]", convolution_3: "f32[8, 80, 73, 73]", squeeze_10: "f32[80]", relu_3: "f32[8, 80, 73, 73]", convolution_4: "f32[8, 192, 71, 71]", squeeze_13: "f32[192]", relu_4: "f32[8, 192, 71, 71]", getitem_12: "f32[8, 192, 35, 35]", getitem_13: "i64[8, 192, 35, 35]", convolution_5: "f32[8, 64, 35, 35]", squeeze_16: "f32[64]", convolution_6: "f32[8, 48, 35, 35]", squeeze_19: "f32[48]", relu_6: "f32[8, 48, 35, 35]", convolution_7: "f32[8, 64, 35, 35]", squeeze_22: "f32[64]", convolution_8: "f32[8, 64, 35, 35]", squeeze_25: "f32[64]", relu_8: "f32[8, 64, 35, 35]", convolution_9: "f32[8, 96, 35, 35]", squeeze_28: "f32[96]", relu_9: "f32[8, 96, 35, 35]", convolution_10: "f32[8, 96, 35, 35]", squeeze_31: "f32[96]", avg_pool2d: "f32[8, 192, 35, 35]", convolution_11: "f32[8, 32, 35, 35]", squeeze_34: "f32[32]", cat: "f32[8, 256, 35, 35]", convolution_12: "f32[8, 64, 35, 35]", squeeze_37: "f32[64]", convolution_13: "f32[8, 48, 35, 35]", squeeze_40: "f32[48]", relu_13: "f32[8, 48, 35, 35]", convolution_14: "f32[8, 64, 35, 35]", squeeze_43: "f32[64]", convolution_15: "f32[8, 64, 35, 35]", squeeze_46: "f32[64]", relu_15: "f32[8, 64, 35, 35]", convolution_16: "f32[8, 96, 35, 35]", squeeze_49: "f32[96]", relu_16: "f32[8, 96, 35, 35]", convolution_17: "f32[8, 96, 35, 35]", squeeze_52: "f32[96]", avg_pool2d_1: "f32[8, 256, 35, 35]", convolution_18: "f32[8, 64, 35, 35]", squeeze_55: "f32[64]", cat_1: "f32[8, 288, 35, 35]", convolution_19: "f32[8, 64, 35, 35]", squeeze_58: "f32[64]", convolution_20: "f32[8, 48, 35, 35]", squeeze_61: "f32[48]", relu_20: "f32[8, 48, 35, 35]", convolution_21: "f32[8, 64, 35, 35]", squeeze_64: "f32[64]", convolution_22: "f32[8, 64, 35, 35]", squeeze_67: "f32[64]", relu_22: "f32[8, 64, 35, 35]", convolution_23: "f32[8, 96, 35, 35]", squeeze_70: "f32[96]", relu_23: "f32[8, 96, 35, 35]", convolution_24: "f32[8, 96, 35, 35]", squeeze_73: "f32[96]", avg_pool2d_2: "f32[8, 288, 35, 35]", convolution_25: "f32[8, 64, 35, 35]", squeeze_76: "f32[64]", cat_2: "f32[8, 288, 35, 35]", convolution_26: "f32[8, 384, 17, 17]", squeeze_79: "f32[384]", convolution_27: "f32[8, 64, 35, 35]", squeeze_82: "f32[64]", relu_27: "f32[8, 64, 35, 35]", convolution_28: "f32[8, 96, 35, 35]", squeeze_85: "f32[96]", relu_28: "f32[8, 96, 35, 35]", convolution_29: "f32[8, 96, 17, 17]", squeeze_88: "f32[96]", getitem_65: "i64[8, 288, 17, 17]", cat_3: "f32[8, 768, 17, 17]", convolution_30: "f32[8, 192, 17, 17]", squeeze_91: "f32[192]", convolution_31: "f32[8, 128, 17, 17]", squeeze_94: "f32[128]", relu_31: "f32[8, 128, 17, 17]", convolution_32: "f32[8, 128, 17, 17]", squeeze_97: "f32[128]", relu_32: "f32[8, 128, 17, 17]", convolution_33: "f32[8, 192, 17, 17]", squeeze_100: "f32[192]", convolution_34: "f32[8, 128, 17, 17]", squeeze_103: "f32[128]", relu_34: "f32[8, 128, 17, 17]", convolution_35: "f32[8, 128, 17, 17]", squeeze_106: "f32[128]", relu_35: "f32[8, 128, 17, 17]", convolution_36: "f32[8, 128, 17, 17]", squeeze_109: "f32[128]", relu_36: "f32[8, 128, 17, 17]", convolution_37: "f32[8, 128, 17, 17]", squeeze_112: "f32[128]", relu_37: "f32[8, 128, 17, 17]", convolution_38: "f32[8, 192, 17, 17]", squeeze_115: "f32[192]", avg_pool2d_3: "f32[8, 768, 17, 17]", convolution_39: "f32[8, 192, 17, 17]", squeeze_118: "f32[192]", cat_4: "f32[8, 768, 17, 17]", convolution_40: "f32[8, 192, 17, 17]", squeeze_121: "f32[192]", convolution_41: "f32[8, 160, 17, 17]", squeeze_124: "f32[160]", relu_41: "f32[8, 160, 17, 17]", convolution_42: "f32[8, 160, 17, 17]", squeeze_127: "f32[160]", relu_42: "f32[8, 160, 17, 17]", convolution_43: "f32[8, 192, 17, 17]", squeeze_130: "f32[192]", convolution_44: "f32[8, 160, 17, 17]", squeeze_133: "f32[160]", relu_44: "f32[8, 160, 17, 17]", convolution_45: "f32[8, 160, 17, 17]", squeeze_136: "f32[160]", relu_45: "f32[8, 160, 17, 17]", convolution_46: "f32[8, 160, 17, 17]", squeeze_139: "f32[160]", relu_46: "f32[8, 160, 17, 17]", convolution_47: "f32[8, 160, 17, 17]", squeeze_142: "f32[160]", relu_47: "f32[8, 160, 17, 17]", convolution_48: "f32[8, 192, 17, 17]", squeeze_145: "f32[192]", avg_pool2d_4: "f32[8, 768, 17, 17]", convolution_49: "f32[8, 192, 17, 17]", squeeze_148: "f32[192]", cat_5: "f32[8, 768, 17, 17]", convolution_50: "f32[8, 192, 17, 17]", squeeze_151: "f32[192]", convolution_51: "f32[8, 160, 17, 17]", squeeze_154: "f32[160]", relu_51: "f32[8, 160, 17, 17]", convolution_52: "f32[8, 160, 17, 17]", squeeze_157: "f32[160]", relu_52: "f32[8, 160, 17, 17]", convolution_53: "f32[8, 192, 17, 17]", squeeze_160: "f32[192]", convolution_54: "f32[8, 160, 17, 17]", squeeze_163: "f32[160]", relu_54: "f32[8, 160, 17, 17]", convolution_55: "f32[8, 160, 17, 17]", squeeze_166: "f32[160]", relu_55: "f32[8, 160, 17, 17]", convolution_56: "f32[8, 160, 17, 17]", squeeze_169: "f32[160]", relu_56: "f32[8, 160, 17, 17]", convolution_57: "f32[8, 160, 17, 17]", squeeze_172: "f32[160]", relu_57: "f32[8, 160, 17, 17]", convolution_58: "f32[8, 192, 17, 17]", squeeze_175: "f32[192]", avg_pool2d_5: "f32[8, 768, 17, 17]", convolution_59: "f32[8, 192, 17, 17]", squeeze_178: "f32[192]", cat_6: "f32[8, 768, 17, 17]", convolution_60: "f32[8, 192, 17, 17]", squeeze_181: "f32[192]", convolution_61: "f32[8, 192, 17, 17]", squeeze_184: "f32[192]", relu_61: "f32[8, 192, 17, 17]", convolution_62: "f32[8, 192, 17, 17]", squeeze_187: "f32[192]", relu_62: "f32[8, 192, 17, 17]", convolution_63: "f32[8, 192, 17, 17]", squeeze_190: "f32[192]", convolution_64: "f32[8, 192, 17, 17]", squeeze_193: "f32[192]", relu_64: "f32[8, 192, 17, 17]", convolution_65: "f32[8, 192, 17, 17]", squeeze_196: "f32[192]", relu_65: "f32[8, 192, 17, 17]", convolution_66: "f32[8, 192, 17, 17]", squeeze_199: "f32[192]", relu_66: "f32[8, 192, 17, 17]", convolution_67: "f32[8, 192, 17, 17]", squeeze_202: "f32[192]", relu_67: "f32[8, 192, 17, 17]", convolution_68: "f32[8, 192, 17, 17]", squeeze_205: "f32[192]", avg_pool2d_6: "f32[8, 768, 17, 17]", convolution_69: "f32[8, 192, 17, 17]", squeeze_208: "f32[192]", cat_7: "f32[8, 768, 17, 17]", convolution_70: "f32[8, 192, 17, 17]", squeeze_211: "f32[192]", relu_70: "f32[8, 192, 17, 17]", convolution_71: "f32[8, 320, 8, 8]", squeeze_214: "f32[320]", convolution_72: "f32[8, 192, 17, 17]", squeeze_217: "f32[192]", relu_72: "f32[8, 192, 17, 17]", convolution_73: "f32[8, 192, 17, 17]", squeeze_220: "f32[192]", relu_73: "f32[8, 192, 17, 17]", convolution_74: "f32[8, 192, 17, 17]", squeeze_223: "f32[192]", relu_74: "f32[8, 192, 17, 17]", convolution_75: "f32[8, 192, 8, 8]", squeeze_226: "f32[192]", getitem_159: "i64[8, 768, 8, 8]", cat_8: "f32[8, 1280, 8, 8]", convolution_76: "f32[8, 320, 8, 8]", squeeze_229: "f32[320]", convolution_77: "f32[8, 384, 8, 8]", squeeze_232: "f32[384]", relu_77: "f32[8, 384, 8, 8]", convolution_78: "f32[8, 384, 8, 8]", squeeze_235: "f32[384]", convolution_79: "f32[8, 384, 8, 8]", squeeze_238: "f32[384]", convolution_80: "f32[8, 448, 8, 8]", squeeze_241: "f32[448]", relu_80: "f32[8, 448, 8, 8]", convolution_81: "f32[8, 384, 8, 8]", squeeze_244: "f32[384]", relu_81: "f32[8, 384, 8, 8]", convolution_82: "f32[8, 384, 8, 8]", squeeze_247: "f32[384]", convolution_83: "f32[8, 384, 8, 8]", squeeze_250: "f32[384]", avg_pool2d_7: "f32[8, 1280, 8, 8]", convolution_84: "f32[8, 192, 8, 8]", squeeze_253: "f32[192]", cat_11: "f32[8, 2048, 8, 8]", convolution_85: "f32[8, 320, 8, 8]", squeeze_256: "f32[320]", convolution_86: "f32[8, 384, 8, 8]", squeeze_259: "f32[384]", relu_86: "f32[8, 384, 8, 8]", convolution_87: "f32[8, 384, 8, 8]", squeeze_262: "f32[384]", convolution_88: "f32[8, 384, 8, 8]", squeeze_265: "f32[384]", convolution_89: "f32[8, 448, 8, 8]", squeeze_268: "f32[448]", relu_89: "f32[8, 448, 8, 8]", convolution_90: "f32[8, 384, 8, 8]", squeeze_271: "f32[384]", relu_90: "f32[8, 384, 8, 8]", convolution_91: "f32[8, 384, 8, 8]", squeeze_274: "f32[384]", convolution_92: "f32[8, 384, 8, 8]", squeeze_277: "f32[384]", avg_pool2d_8: "f32[8, 2048, 8, 8]", convolution_93: "f32[8, 192, 8, 8]", squeeze_280: "f32[192]", clone: "f32[8, 2048]", permute_1: "f32[1000, 2048]", le: "b8[8, 192, 8, 8]", unsqueeze_378: "f32[1, 192, 1, 1]", le_1: "b8[8, 384, 8, 8]", unsqueeze_390: "f32[1, 384, 1, 1]", le_2: "b8[8, 384, 8, 8]", unsqueeze_402: "f32[1, 384, 1, 1]", unsqueeze_414: "f32[1, 384, 1, 1]", unsqueeze_426: "f32[1, 448, 1, 1]", le_5: "b8[8, 384, 8, 8]", unsqueeze_438: "f32[1, 384, 1, 1]", le_6: "b8[8, 384, 8, 8]", unsqueeze_450: "f32[1, 384, 1, 1]", unsqueeze_462: "f32[1, 384, 1, 1]", le_8: "b8[8, 320, 8, 8]", unsqueeze_474: "f32[1, 320, 1, 1]", le_9: "b8[8, 192, 8, 8]", unsqueeze_486: "f32[1, 192, 1, 1]", le_10: "b8[8, 384, 8, 8]", unsqueeze_498: "f32[1, 384, 1, 1]", le_11: "b8[8, 384, 8, 8]", unsqueeze_510: "f32[1, 384, 1, 1]", unsqueeze_522: "f32[1, 384, 1, 1]", unsqueeze_534: "f32[1, 448, 1, 1]", le_14: "b8[8, 384, 8, 8]", unsqueeze_546: "f32[1, 384, 1, 1]", le_15: "b8[8, 384, 8, 8]", unsqueeze_558: "f32[1, 384, 1, 1]", unsqueeze_570: "f32[1, 384, 1, 1]", le_17: "b8[8, 320, 8, 8]", unsqueeze_582: "f32[1, 320, 1, 1]", le_18: "b8[8, 192, 8, 8]", unsqueeze_594: "f32[1, 192, 1, 1]", unsqueeze_606: "f32[1, 192, 1, 1]", unsqueeze_618: "f32[1, 192, 1, 1]", unsqueeze_630: "f32[1, 192, 1, 1]", le_22: "b8[8, 320, 8, 8]", unsqueeze_642: "f32[1, 320, 1, 1]", unsqueeze_654: "f32[1, 192, 1, 1]", le_24: "b8[8, 192, 17, 17]", unsqueeze_666: "f32[1, 192, 1, 1]", le_25: "b8[8, 192, 17, 17]", unsqueeze_678: "f32[1, 192, 1, 1]", unsqueeze_690: "f32[1, 192, 1, 1]", unsqueeze_702: "f32[1, 192, 1, 1]", unsqueeze_714: "f32[1, 192, 1, 1]", unsqueeze_726: "f32[1, 192, 1, 1]", le_30: "b8[8, 192, 17, 17]", unsqueeze_738: "f32[1, 192, 1, 1]", unsqueeze_750: "f32[1, 192, 1, 1]", unsqueeze_762: "f32[1, 192, 1, 1]", le_33: "b8[8, 192, 17, 17]", unsqueeze_774: "f32[1, 192, 1, 1]", le_34: "b8[8, 192, 17, 17]", unsqueeze_786: "f32[1, 192, 1, 1]", le_35: "b8[8, 192, 17, 17]", unsqueeze_798: "f32[1, 192, 1, 1]", unsqueeze_810: "f32[1, 160, 1, 1]", unsqueeze_822: "f32[1, 160, 1, 1]", unsqueeze_834: "f32[1, 160, 1, 1]", unsqueeze_846: "f32[1, 160, 1, 1]", le_40: "b8[8, 192, 17, 17]", unsqueeze_858: "f32[1, 192, 1, 1]", unsqueeze_870: "f32[1, 160, 1, 1]", unsqueeze_882: "f32[1, 160, 1, 1]", le_43: "b8[8, 192, 17, 17]", unsqueeze_894: "f32[1, 192, 1, 1]", le_44: "b8[8, 192, 17, 17]", unsqueeze_906: "f32[1, 192, 1, 1]", le_45: "b8[8, 192, 17, 17]", unsqueeze_918: "f32[1, 192, 1, 1]", unsqueeze_930: "f32[1, 160, 1, 1]", unsqueeze_942: "f32[1, 160, 1, 1]", unsqueeze_954: "f32[1, 160, 1, 1]", unsqueeze_966: "f32[1, 160, 1, 1]", le_50: "b8[8, 192, 17, 17]", unsqueeze_978: "f32[1, 192, 1, 1]", unsqueeze_990: "f32[1, 160, 1, 1]", unsqueeze_1002: "f32[1, 160, 1, 1]", le_53: "b8[8, 192, 17, 17]", unsqueeze_1014: "f32[1, 192, 1, 1]", le_54: "b8[8, 192, 17, 17]", unsqueeze_1026: "f32[1, 192, 1, 1]", le_55: "b8[8, 192, 17, 17]", unsqueeze_1038: "f32[1, 192, 1, 1]", unsqueeze_1050: "f32[1, 128, 1, 1]", unsqueeze_1062: "f32[1, 128, 1, 1]", unsqueeze_1074: "f32[1, 128, 1, 1]", unsqueeze_1086: "f32[1, 128, 1, 1]", le_60: "b8[8, 192, 17, 17]", unsqueeze_1098: "f32[1, 192, 1, 1]", unsqueeze_1110: "f32[1, 128, 1, 1]", unsqueeze_1122: "f32[1, 128, 1, 1]", le_63: "b8[8, 192, 17, 17]", unsqueeze_1134: "f32[1, 192, 1, 1]", le_64: "b8[8, 96, 17, 17]", unsqueeze_1146: "f32[1, 96, 1, 1]", unsqueeze_1158: "f32[1, 96, 1, 1]", unsqueeze_1170: "f32[1, 64, 1, 1]", le_67: "b8[8, 384, 17, 17]", unsqueeze_1182: "f32[1, 384, 1, 1]", le_68: "b8[8, 64, 35, 35]", unsqueeze_1194: "f32[1, 64, 1, 1]", le_69: "b8[8, 96, 35, 35]", unsqueeze_1206: "f32[1, 96, 1, 1]", unsqueeze_1218: "f32[1, 96, 1, 1]", unsqueeze_1230: "f32[1, 64, 1, 1]", le_72: "b8[8, 64, 35, 35]", unsqueeze_1242: "f32[1, 64, 1, 1]", unsqueeze_1254: "f32[1, 48, 1, 1]", le_74: "b8[8, 64, 35, 35]", unsqueeze_1266: "f32[1, 64, 1, 1]", le_75: "b8[8, 64, 35, 35]", unsqueeze_1278: "f32[1, 64, 1, 1]", le_76: "b8[8, 96, 35, 35]", unsqueeze_1290: "f32[1, 96, 1, 1]", unsqueeze_1302: "f32[1, 96, 1, 1]", unsqueeze_1314: "f32[1, 64, 1, 1]", le_79: "b8[8, 64, 35, 35]", unsqueeze_1326: "f32[1, 64, 1, 1]", unsqueeze_1338: "f32[1, 48, 1, 1]", le_81: "b8[8, 64, 35, 35]", unsqueeze_1350: "f32[1, 64, 1, 1]", le_82: "b8[8, 32, 35, 35]", unsqueeze_1362: "f32[1, 32, 1, 1]", le_83: "b8[8, 96, 35, 35]", unsqueeze_1374: "f32[1, 96, 1, 1]", unsqueeze_1386: "f32[1, 96, 1, 1]", unsqueeze_1398: "f32[1, 64, 1, 1]", le_86: "b8[8, 64, 35, 35]", unsqueeze_1410: "f32[1, 64, 1, 1]", unsqueeze_1422: "f32[1, 48, 1, 1]", le_88: "b8[8, 64, 35, 35]", unsqueeze_1434: "f32[1, 64, 1, 1]", unsqueeze_1446: "f32[1, 192, 1, 1]", unsqueeze_1458: "f32[1, 80, 1, 1]", unsqueeze_1470: "f32[1, 64, 1, 1]", unsqueeze_1482: "f32[1, 32, 1, 1]", unsqueeze_1494: "f32[1, 32, 1, 1]", tangents_1: "f32[8, 1000]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/inception_v3.py:377, code: x = self.fc(x)
    mm: "f32[8, 2048]" = torch.ops.aten.mm.default(tangents_1, permute_1);  permute_1 = None
    permute_2: "f32[1000, 8]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
    mm_1: "f32[1000, 2048]" = torch.ops.aten.mm.default(permute_2, clone);  permute_2 = clone = None
    permute_3: "f32[2048, 1000]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_1: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view_1: "f32[1000]" = torch.ops.aten.view.default(sum_1, [1000]);  sum_1 = None
    permute_4: "f32[1000, 2048]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view_2: "f32[8, 2048, 1, 1]" = torch.ops.aten.view.default(mm, [8, 2048, 1, 1]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    expand: "f32[8, 2048, 8, 8]" = torch.ops.aten.expand.default(view_2, [8, 2048, 8, 8]);  view_2 = None
    div: "f32[8, 2048, 8, 8]" = torch.ops.aten.div.Scalar(expand, 64);  expand = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/inception_v3.py:206, code: return torch.cat(outputs, 1)
    slice_1: "f32[8, 320, 8, 8]" = torch.ops.aten.slice.Tensor(div, 1, 0, 320)
    slice_2: "f32[8, 768, 8, 8]" = torch.ops.aten.slice.Tensor(div, 1, 320, 1088)
    slice_3: "f32[8, 768, 8, 8]" = torch.ops.aten.slice.Tensor(div, 1, 1088, 1856)
    slice_4: "f32[8, 192, 8, 8]" = torch.ops.aten.slice.Tensor(div, 1, 1856, 2048);  div = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    where: "f32[8, 192, 8, 8]" = torch.ops.aten.where.self(le, full_default, slice_4);  le = slice_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_2: "f32[192]" = torch.ops.aten.sum.dim_IntList(where, [0, 2, 3])
    sub_94: "f32[8, 192, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_93, unsqueeze_378);  convolution_93 = unsqueeze_378 = None
    mul_658: "f32[8, 192, 8, 8]" = torch.ops.aten.mul.Tensor(where, sub_94)
    sum_3: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_658, [0, 2, 3]);  mul_658 = None
    mul_659: "f32[192]" = torch.ops.aten.mul.Tensor(sum_2, 0.001953125)
    unsqueeze_379: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_659, 0);  mul_659 = None
    unsqueeze_380: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_379, 2);  unsqueeze_379 = None
    unsqueeze_381: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_380, 3);  unsqueeze_380 = None
    mul_660: "f32[192]" = torch.ops.aten.mul.Tensor(sum_3, 0.001953125)
    mul_661: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_280, squeeze_280)
    mul_662: "f32[192]" = torch.ops.aten.mul.Tensor(mul_660, mul_661);  mul_660 = mul_661 = None
    unsqueeze_382: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_662, 0);  mul_662 = None
    unsqueeze_383: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_382, 2);  unsqueeze_382 = None
    unsqueeze_384: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_383, 3);  unsqueeze_383 = None
    mul_663: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_280, primals_187);  primals_187 = None
    unsqueeze_385: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_663, 0);  mul_663 = None
    unsqueeze_386: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_385, 2);  unsqueeze_385 = None
    unsqueeze_387: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_386, 3);  unsqueeze_386 = None
    mul_664: "f32[8, 192, 8, 8]" = torch.ops.aten.mul.Tensor(sub_94, unsqueeze_384);  sub_94 = unsqueeze_384 = None
    sub_96: "f32[8, 192, 8, 8]" = torch.ops.aten.sub.Tensor(where, mul_664);  where = mul_664 = None
    sub_97: "f32[8, 192, 8, 8]" = torch.ops.aten.sub.Tensor(sub_96, unsqueeze_381);  sub_96 = unsqueeze_381 = None
    mul_665: "f32[8, 192, 8, 8]" = torch.ops.aten.mul.Tensor(sub_97, unsqueeze_387);  sub_97 = unsqueeze_387 = None
    mul_666: "f32[192]" = torch.ops.aten.mul.Tensor(sum_3, squeeze_280);  sum_3 = squeeze_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward = torch.ops.aten.convolution_backward.default(mul_665, avg_pool2d_8, primals_282, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_665 = avg_pool2d_8 = primals_282 = None
    getitem_196: "f32[8, 2048, 8, 8]" = convolution_backward[0]
    getitem_197: "f32[192, 2048, 1, 1]" = convolution_backward[1];  convolution_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/inception_v3.py:198, code: branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    avg_pool2d_backward: "f32[8, 2048, 8, 8]" = torch.ops.aten.avg_pool2d_backward.default(getitem_196, cat_11, [3, 3], [1, 1], [1, 1], False, True, None);  getitem_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/inception_v3.py:196, code: branch3x3dbl = torch.cat(branch3x3dbl, 1)
    slice_5: "f32[8, 384, 8, 8]" = torch.ops.aten.slice.Tensor(slice_3, 1, 0, 384)
    slice_6: "f32[8, 384, 8, 8]" = torch.ops.aten.slice.Tensor(slice_3, 1, 384, 768);  slice_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    where_1: "f32[8, 384, 8, 8]" = torch.ops.aten.where.self(le_1, full_default, slice_6);  le_1 = slice_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_4: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_1, [0, 2, 3])
    sub_98: "f32[8, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_92, unsqueeze_390);  convolution_92 = unsqueeze_390 = None
    mul_667: "f32[8, 384, 8, 8]" = torch.ops.aten.mul.Tensor(where_1, sub_98)
    sum_5: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_667, [0, 2, 3]);  mul_667 = None
    mul_668: "f32[384]" = torch.ops.aten.mul.Tensor(sum_4, 0.001953125)
    unsqueeze_391: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_668, 0);  mul_668 = None
    unsqueeze_392: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_391, 2);  unsqueeze_391 = None
    unsqueeze_393: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_392, 3);  unsqueeze_392 = None
    mul_669: "f32[384]" = torch.ops.aten.mul.Tensor(sum_5, 0.001953125)
    mul_670: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_277, squeeze_277)
    mul_671: "f32[384]" = torch.ops.aten.mul.Tensor(mul_669, mul_670);  mul_669 = mul_670 = None
    unsqueeze_394: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_671, 0);  mul_671 = None
    unsqueeze_395: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_394, 2);  unsqueeze_394 = None
    unsqueeze_396: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_395, 3);  unsqueeze_395 = None
    mul_672: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_277, primals_185);  primals_185 = None
    unsqueeze_397: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_672, 0);  mul_672 = None
    unsqueeze_398: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_397, 2);  unsqueeze_397 = None
    unsqueeze_399: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_398, 3);  unsqueeze_398 = None
    mul_673: "f32[8, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_98, unsqueeze_396);  sub_98 = unsqueeze_396 = None
    sub_100: "f32[8, 384, 8, 8]" = torch.ops.aten.sub.Tensor(where_1, mul_673);  where_1 = mul_673 = None
    sub_101: "f32[8, 384, 8, 8]" = torch.ops.aten.sub.Tensor(sub_100, unsqueeze_393);  sub_100 = unsqueeze_393 = None
    mul_674: "f32[8, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_101, unsqueeze_399);  sub_101 = unsqueeze_399 = None
    mul_675: "f32[384]" = torch.ops.aten.mul.Tensor(sum_5, squeeze_277);  sum_5 = squeeze_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(mul_674, relu_90, primals_281, [0], [1, 1], [1, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_674 = primals_281 = None
    getitem_199: "f32[8, 384, 8, 8]" = convolution_backward_1[0]
    getitem_200: "f32[384, 384, 3, 1]" = convolution_backward_1[1];  convolution_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    where_2: "f32[8, 384, 8, 8]" = torch.ops.aten.where.self(le_2, full_default, slice_5);  le_2 = slice_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_6: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_2, [0, 2, 3])
    sub_102: "f32[8, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_91, unsqueeze_402);  convolution_91 = unsqueeze_402 = None
    mul_676: "f32[8, 384, 8, 8]" = torch.ops.aten.mul.Tensor(where_2, sub_102)
    sum_7: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_676, [0, 2, 3]);  mul_676 = None
    mul_677: "f32[384]" = torch.ops.aten.mul.Tensor(sum_6, 0.001953125)
    unsqueeze_403: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_677, 0);  mul_677 = None
    unsqueeze_404: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_403, 2);  unsqueeze_403 = None
    unsqueeze_405: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_404, 3);  unsqueeze_404 = None
    mul_678: "f32[384]" = torch.ops.aten.mul.Tensor(sum_7, 0.001953125)
    mul_679: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_274, squeeze_274)
    mul_680: "f32[384]" = torch.ops.aten.mul.Tensor(mul_678, mul_679);  mul_678 = mul_679 = None
    unsqueeze_406: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_680, 0);  mul_680 = None
    unsqueeze_407: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_406, 2);  unsqueeze_406 = None
    unsqueeze_408: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_407, 3);  unsqueeze_407 = None
    mul_681: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_274, primals_183);  primals_183 = None
    unsqueeze_409: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_681, 0);  mul_681 = None
    unsqueeze_410: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_409, 2);  unsqueeze_409 = None
    unsqueeze_411: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_410, 3);  unsqueeze_410 = None
    mul_682: "f32[8, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_102, unsqueeze_408);  sub_102 = unsqueeze_408 = None
    sub_104: "f32[8, 384, 8, 8]" = torch.ops.aten.sub.Tensor(where_2, mul_682);  where_2 = mul_682 = None
    sub_105: "f32[8, 384, 8, 8]" = torch.ops.aten.sub.Tensor(sub_104, unsqueeze_405);  sub_104 = unsqueeze_405 = None
    mul_683: "f32[8, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_105, unsqueeze_411);  sub_105 = unsqueeze_411 = None
    mul_684: "f32[384]" = torch.ops.aten.mul.Tensor(sum_7, squeeze_274);  sum_7 = squeeze_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(mul_683, relu_90, primals_280, [0], [1, 1], [0, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_683 = primals_280 = None
    getitem_202: "f32[8, 384, 8, 8]" = convolution_backward_2[0]
    getitem_203: "f32[384, 384, 1, 3]" = convolution_backward_2[1];  convolution_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_470: "f32[8, 384, 8, 8]" = torch.ops.aten.add.Tensor(getitem_199, getitem_202);  getitem_199 = getitem_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_104: "f32[8, 384, 8, 8]" = torch.ops.aten.alias.default(relu_90);  relu_90 = None
    alias_105: "f32[8, 384, 8, 8]" = torch.ops.aten.alias.default(alias_104);  alias_104 = None
    le_3: "b8[8, 384, 8, 8]" = torch.ops.aten.le.Scalar(alias_105, 0);  alias_105 = None
    where_3: "f32[8, 384, 8, 8]" = torch.ops.aten.where.self(le_3, full_default, add_470);  le_3 = add_470 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_8: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_3, [0, 2, 3])
    sub_106: "f32[8, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_90, unsqueeze_414);  convolution_90 = unsqueeze_414 = None
    mul_685: "f32[8, 384, 8, 8]" = torch.ops.aten.mul.Tensor(where_3, sub_106)
    sum_9: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_685, [0, 2, 3]);  mul_685 = None
    mul_686: "f32[384]" = torch.ops.aten.mul.Tensor(sum_8, 0.001953125)
    unsqueeze_415: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_686, 0);  mul_686 = None
    unsqueeze_416: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_415, 2);  unsqueeze_415 = None
    unsqueeze_417: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_416, 3);  unsqueeze_416 = None
    mul_687: "f32[384]" = torch.ops.aten.mul.Tensor(sum_9, 0.001953125)
    mul_688: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_271, squeeze_271)
    mul_689: "f32[384]" = torch.ops.aten.mul.Tensor(mul_687, mul_688);  mul_687 = mul_688 = None
    unsqueeze_418: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_689, 0);  mul_689 = None
    unsqueeze_419: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_418, 2);  unsqueeze_418 = None
    unsqueeze_420: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_419, 3);  unsqueeze_419 = None
    mul_690: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_271, primals_181);  primals_181 = None
    unsqueeze_421: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_690, 0);  mul_690 = None
    unsqueeze_422: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_421, 2);  unsqueeze_421 = None
    unsqueeze_423: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_422, 3);  unsqueeze_422 = None
    mul_691: "f32[8, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_106, unsqueeze_420);  sub_106 = unsqueeze_420 = None
    sub_108: "f32[8, 384, 8, 8]" = torch.ops.aten.sub.Tensor(where_3, mul_691);  where_3 = mul_691 = None
    sub_109: "f32[8, 384, 8, 8]" = torch.ops.aten.sub.Tensor(sub_108, unsqueeze_417);  sub_108 = unsqueeze_417 = None
    mul_692: "f32[8, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_109, unsqueeze_423);  sub_109 = unsqueeze_423 = None
    mul_693: "f32[384]" = torch.ops.aten.mul.Tensor(sum_9, squeeze_271);  sum_9 = squeeze_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_3 = torch.ops.aten.convolution_backward.default(mul_692, relu_89, primals_279, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_692 = primals_279 = None
    getitem_205: "f32[8, 448, 8, 8]" = convolution_backward_3[0]
    getitem_206: "f32[384, 448, 3, 3]" = convolution_backward_3[1];  convolution_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_107: "f32[8, 448, 8, 8]" = torch.ops.aten.alias.default(relu_89);  relu_89 = None
    alias_108: "f32[8, 448, 8, 8]" = torch.ops.aten.alias.default(alias_107);  alias_107 = None
    le_4: "b8[8, 448, 8, 8]" = torch.ops.aten.le.Scalar(alias_108, 0);  alias_108 = None
    where_4: "f32[8, 448, 8, 8]" = torch.ops.aten.where.self(le_4, full_default, getitem_205);  le_4 = getitem_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_10: "f32[448]" = torch.ops.aten.sum.dim_IntList(where_4, [0, 2, 3])
    sub_110: "f32[8, 448, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_89, unsqueeze_426);  convolution_89 = unsqueeze_426 = None
    mul_694: "f32[8, 448, 8, 8]" = torch.ops.aten.mul.Tensor(where_4, sub_110)
    sum_11: "f32[448]" = torch.ops.aten.sum.dim_IntList(mul_694, [0, 2, 3]);  mul_694 = None
    mul_695: "f32[448]" = torch.ops.aten.mul.Tensor(sum_10, 0.001953125)
    unsqueeze_427: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_695, 0);  mul_695 = None
    unsqueeze_428: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_427, 2);  unsqueeze_427 = None
    unsqueeze_429: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_428, 3);  unsqueeze_428 = None
    mul_696: "f32[448]" = torch.ops.aten.mul.Tensor(sum_11, 0.001953125)
    mul_697: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_268, squeeze_268)
    mul_698: "f32[448]" = torch.ops.aten.mul.Tensor(mul_696, mul_697);  mul_696 = mul_697 = None
    unsqueeze_430: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_698, 0);  mul_698 = None
    unsqueeze_431: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_430, 2);  unsqueeze_430 = None
    unsqueeze_432: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_431, 3);  unsqueeze_431 = None
    mul_699: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_268, primals_179);  primals_179 = None
    unsqueeze_433: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_699, 0);  mul_699 = None
    unsqueeze_434: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_433, 2);  unsqueeze_433 = None
    unsqueeze_435: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_434, 3);  unsqueeze_434 = None
    mul_700: "f32[8, 448, 8, 8]" = torch.ops.aten.mul.Tensor(sub_110, unsqueeze_432);  sub_110 = unsqueeze_432 = None
    sub_112: "f32[8, 448, 8, 8]" = torch.ops.aten.sub.Tensor(where_4, mul_700);  where_4 = mul_700 = None
    sub_113: "f32[8, 448, 8, 8]" = torch.ops.aten.sub.Tensor(sub_112, unsqueeze_429);  sub_112 = unsqueeze_429 = None
    mul_701: "f32[8, 448, 8, 8]" = torch.ops.aten.mul.Tensor(sub_113, unsqueeze_435);  sub_113 = unsqueeze_435 = None
    mul_702: "f32[448]" = torch.ops.aten.mul.Tensor(sum_11, squeeze_268);  sum_11 = squeeze_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_4 = torch.ops.aten.convolution_backward.default(mul_701, cat_11, primals_278, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_701 = primals_278 = None
    getitem_208: "f32[8, 2048, 8, 8]" = convolution_backward_4[0]
    getitem_209: "f32[448, 2048, 1, 1]" = convolution_backward_4[1];  convolution_backward_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_471: "f32[8, 2048, 8, 8]" = torch.ops.aten.add.Tensor(avg_pool2d_backward, getitem_208);  avg_pool2d_backward = getitem_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/inception_v3.py:188, code: branch3x3 = torch.cat(branch3x3, 1)
    slice_7: "f32[8, 384, 8, 8]" = torch.ops.aten.slice.Tensor(slice_2, 1, 0, 384)
    slice_8: "f32[8, 384, 8, 8]" = torch.ops.aten.slice.Tensor(slice_2, 1, 384, 768);  slice_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    where_5: "f32[8, 384, 8, 8]" = torch.ops.aten.where.self(le_5, full_default, slice_8);  le_5 = slice_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_12: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_5, [0, 2, 3])
    sub_114: "f32[8, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_88, unsqueeze_438);  convolution_88 = unsqueeze_438 = None
    mul_703: "f32[8, 384, 8, 8]" = torch.ops.aten.mul.Tensor(where_5, sub_114)
    sum_13: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_703, [0, 2, 3]);  mul_703 = None
    mul_704: "f32[384]" = torch.ops.aten.mul.Tensor(sum_12, 0.001953125)
    unsqueeze_439: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_704, 0);  mul_704 = None
    unsqueeze_440: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_439, 2);  unsqueeze_439 = None
    unsqueeze_441: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_440, 3);  unsqueeze_440 = None
    mul_705: "f32[384]" = torch.ops.aten.mul.Tensor(sum_13, 0.001953125)
    mul_706: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_265, squeeze_265)
    mul_707: "f32[384]" = torch.ops.aten.mul.Tensor(mul_705, mul_706);  mul_705 = mul_706 = None
    unsqueeze_442: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_707, 0);  mul_707 = None
    unsqueeze_443: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_442, 2);  unsqueeze_442 = None
    unsqueeze_444: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_443, 3);  unsqueeze_443 = None
    mul_708: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_265, primals_177);  primals_177 = None
    unsqueeze_445: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_708, 0);  mul_708 = None
    unsqueeze_446: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_445, 2);  unsqueeze_445 = None
    unsqueeze_447: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_446, 3);  unsqueeze_446 = None
    mul_709: "f32[8, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_114, unsqueeze_444);  sub_114 = unsqueeze_444 = None
    sub_116: "f32[8, 384, 8, 8]" = torch.ops.aten.sub.Tensor(where_5, mul_709);  where_5 = mul_709 = None
    sub_117: "f32[8, 384, 8, 8]" = torch.ops.aten.sub.Tensor(sub_116, unsqueeze_441);  sub_116 = unsqueeze_441 = None
    mul_710: "f32[8, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_117, unsqueeze_447);  sub_117 = unsqueeze_447 = None
    mul_711: "f32[384]" = torch.ops.aten.mul.Tensor(sum_13, squeeze_265);  sum_13 = squeeze_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_5 = torch.ops.aten.convolution_backward.default(mul_710, relu_86, primals_277, [0], [1, 1], [1, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_710 = primals_277 = None
    getitem_211: "f32[8, 384, 8, 8]" = convolution_backward_5[0]
    getitem_212: "f32[384, 384, 3, 1]" = convolution_backward_5[1];  convolution_backward_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    where_6: "f32[8, 384, 8, 8]" = torch.ops.aten.where.self(le_6, full_default, slice_7);  le_6 = slice_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_14: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_6, [0, 2, 3])
    sub_118: "f32[8, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_87, unsqueeze_450);  convolution_87 = unsqueeze_450 = None
    mul_712: "f32[8, 384, 8, 8]" = torch.ops.aten.mul.Tensor(where_6, sub_118)
    sum_15: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_712, [0, 2, 3]);  mul_712 = None
    mul_713: "f32[384]" = torch.ops.aten.mul.Tensor(sum_14, 0.001953125)
    unsqueeze_451: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_713, 0);  mul_713 = None
    unsqueeze_452: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_451, 2);  unsqueeze_451 = None
    unsqueeze_453: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_452, 3);  unsqueeze_452 = None
    mul_714: "f32[384]" = torch.ops.aten.mul.Tensor(sum_15, 0.001953125)
    mul_715: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_262, squeeze_262)
    mul_716: "f32[384]" = torch.ops.aten.mul.Tensor(mul_714, mul_715);  mul_714 = mul_715 = None
    unsqueeze_454: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_716, 0);  mul_716 = None
    unsqueeze_455: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_454, 2);  unsqueeze_454 = None
    unsqueeze_456: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_455, 3);  unsqueeze_455 = None
    mul_717: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_262, primals_175);  primals_175 = None
    unsqueeze_457: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_717, 0);  mul_717 = None
    unsqueeze_458: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_457, 2);  unsqueeze_457 = None
    unsqueeze_459: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_458, 3);  unsqueeze_458 = None
    mul_718: "f32[8, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_118, unsqueeze_456);  sub_118 = unsqueeze_456 = None
    sub_120: "f32[8, 384, 8, 8]" = torch.ops.aten.sub.Tensor(where_6, mul_718);  where_6 = mul_718 = None
    sub_121: "f32[8, 384, 8, 8]" = torch.ops.aten.sub.Tensor(sub_120, unsqueeze_453);  sub_120 = unsqueeze_453 = None
    mul_719: "f32[8, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_121, unsqueeze_459);  sub_121 = unsqueeze_459 = None
    mul_720: "f32[384]" = torch.ops.aten.mul.Tensor(sum_15, squeeze_262);  sum_15 = squeeze_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_6 = torch.ops.aten.convolution_backward.default(mul_719, relu_86, primals_276, [0], [1, 1], [0, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_719 = primals_276 = None
    getitem_214: "f32[8, 384, 8, 8]" = convolution_backward_6[0]
    getitem_215: "f32[384, 384, 1, 3]" = convolution_backward_6[1];  convolution_backward_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_472: "f32[8, 384, 8, 8]" = torch.ops.aten.add.Tensor(getitem_211, getitem_214);  getitem_211 = getitem_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_116: "f32[8, 384, 8, 8]" = torch.ops.aten.alias.default(relu_86);  relu_86 = None
    alias_117: "f32[8, 384, 8, 8]" = torch.ops.aten.alias.default(alias_116);  alias_116 = None
    le_7: "b8[8, 384, 8, 8]" = torch.ops.aten.le.Scalar(alias_117, 0);  alias_117 = None
    where_7: "f32[8, 384, 8, 8]" = torch.ops.aten.where.self(le_7, full_default, add_472);  le_7 = add_472 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_16: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_7, [0, 2, 3])
    sub_122: "f32[8, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_86, unsqueeze_462);  convolution_86 = unsqueeze_462 = None
    mul_721: "f32[8, 384, 8, 8]" = torch.ops.aten.mul.Tensor(where_7, sub_122)
    sum_17: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_721, [0, 2, 3]);  mul_721 = None
    mul_722: "f32[384]" = torch.ops.aten.mul.Tensor(sum_16, 0.001953125)
    unsqueeze_463: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_722, 0);  mul_722 = None
    unsqueeze_464: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_463, 2);  unsqueeze_463 = None
    unsqueeze_465: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_464, 3);  unsqueeze_464 = None
    mul_723: "f32[384]" = torch.ops.aten.mul.Tensor(sum_17, 0.001953125)
    mul_724: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_259, squeeze_259)
    mul_725: "f32[384]" = torch.ops.aten.mul.Tensor(mul_723, mul_724);  mul_723 = mul_724 = None
    unsqueeze_466: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_725, 0);  mul_725 = None
    unsqueeze_467: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_466, 2);  unsqueeze_466 = None
    unsqueeze_468: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_467, 3);  unsqueeze_467 = None
    mul_726: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_259, primals_173);  primals_173 = None
    unsqueeze_469: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_726, 0);  mul_726 = None
    unsqueeze_470: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_469, 2);  unsqueeze_469 = None
    unsqueeze_471: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_470, 3);  unsqueeze_470 = None
    mul_727: "f32[8, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_122, unsqueeze_468);  sub_122 = unsqueeze_468 = None
    sub_124: "f32[8, 384, 8, 8]" = torch.ops.aten.sub.Tensor(where_7, mul_727);  where_7 = mul_727 = None
    sub_125: "f32[8, 384, 8, 8]" = torch.ops.aten.sub.Tensor(sub_124, unsqueeze_465);  sub_124 = unsqueeze_465 = None
    mul_728: "f32[8, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_125, unsqueeze_471);  sub_125 = unsqueeze_471 = None
    mul_729: "f32[384]" = torch.ops.aten.mul.Tensor(sum_17, squeeze_259);  sum_17 = squeeze_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_7 = torch.ops.aten.convolution_backward.default(mul_728, cat_11, primals_275, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_728 = primals_275 = None
    getitem_217: "f32[8, 2048, 8, 8]" = convolution_backward_7[0]
    getitem_218: "f32[384, 2048, 1, 1]" = convolution_backward_7[1];  convolution_backward_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_473: "f32[8, 2048, 8, 8]" = torch.ops.aten.add.Tensor(add_471, getitem_217);  add_471 = getitem_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    where_8: "f32[8, 320, 8, 8]" = torch.ops.aten.where.self(le_8, full_default, slice_1);  le_8 = slice_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_18: "f32[320]" = torch.ops.aten.sum.dim_IntList(where_8, [0, 2, 3])
    sub_126: "f32[8, 320, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_85, unsqueeze_474);  convolution_85 = unsqueeze_474 = None
    mul_730: "f32[8, 320, 8, 8]" = torch.ops.aten.mul.Tensor(where_8, sub_126)
    sum_19: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_730, [0, 2, 3]);  mul_730 = None
    mul_731: "f32[320]" = torch.ops.aten.mul.Tensor(sum_18, 0.001953125)
    unsqueeze_475: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(mul_731, 0);  mul_731 = None
    unsqueeze_476: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_475, 2);  unsqueeze_475 = None
    unsqueeze_477: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_476, 3);  unsqueeze_476 = None
    mul_732: "f32[320]" = torch.ops.aten.mul.Tensor(sum_19, 0.001953125)
    mul_733: "f32[320]" = torch.ops.aten.mul.Tensor(squeeze_256, squeeze_256)
    mul_734: "f32[320]" = torch.ops.aten.mul.Tensor(mul_732, mul_733);  mul_732 = mul_733 = None
    unsqueeze_478: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(mul_734, 0);  mul_734 = None
    unsqueeze_479: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_478, 2);  unsqueeze_478 = None
    unsqueeze_480: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_479, 3);  unsqueeze_479 = None
    mul_735: "f32[320]" = torch.ops.aten.mul.Tensor(squeeze_256, primals_171);  primals_171 = None
    unsqueeze_481: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(mul_735, 0);  mul_735 = None
    unsqueeze_482: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_481, 2);  unsqueeze_481 = None
    unsqueeze_483: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_482, 3);  unsqueeze_482 = None
    mul_736: "f32[8, 320, 8, 8]" = torch.ops.aten.mul.Tensor(sub_126, unsqueeze_480);  sub_126 = unsqueeze_480 = None
    sub_128: "f32[8, 320, 8, 8]" = torch.ops.aten.sub.Tensor(where_8, mul_736);  where_8 = mul_736 = None
    sub_129: "f32[8, 320, 8, 8]" = torch.ops.aten.sub.Tensor(sub_128, unsqueeze_477);  sub_128 = unsqueeze_477 = None
    mul_737: "f32[8, 320, 8, 8]" = torch.ops.aten.mul.Tensor(sub_129, unsqueeze_483);  sub_129 = unsqueeze_483 = None
    mul_738: "f32[320]" = torch.ops.aten.mul.Tensor(sum_19, squeeze_256);  sum_19 = squeeze_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_8 = torch.ops.aten.convolution_backward.default(mul_737, cat_11, primals_274, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_737 = cat_11 = primals_274 = None
    getitem_220: "f32[8, 2048, 8, 8]" = convolution_backward_8[0]
    getitem_221: "f32[320, 2048, 1, 1]" = convolution_backward_8[1];  convolution_backward_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_474: "f32[8, 2048, 8, 8]" = torch.ops.aten.add.Tensor(add_473, getitem_220);  add_473 = getitem_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/inception_v3.py:206, code: return torch.cat(outputs, 1)
    slice_9: "f32[8, 320, 8, 8]" = torch.ops.aten.slice.Tensor(add_474, 1, 0, 320)
    slice_10: "f32[8, 768, 8, 8]" = torch.ops.aten.slice.Tensor(add_474, 1, 320, 1088)
    slice_11: "f32[8, 768, 8, 8]" = torch.ops.aten.slice.Tensor(add_474, 1, 1088, 1856)
    slice_12: "f32[8, 192, 8, 8]" = torch.ops.aten.slice.Tensor(add_474, 1, 1856, 2048);  add_474 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    where_9: "f32[8, 192, 8, 8]" = torch.ops.aten.where.self(le_9, full_default, slice_12);  le_9 = slice_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_20: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_9, [0, 2, 3])
    sub_130: "f32[8, 192, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_84, unsqueeze_486);  convolution_84 = unsqueeze_486 = None
    mul_739: "f32[8, 192, 8, 8]" = torch.ops.aten.mul.Tensor(where_9, sub_130)
    sum_21: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_739, [0, 2, 3]);  mul_739 = None
    mul_740: "f32[192]" = torch.ops.aten.mul.Tensor(sum_20, 0.001953125)
    unsqueeze_487: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_740, 0);  mul_740 = None
    unsqueeze_488: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_487, 2);  unsqueeze_487 = None
    unsqueeze_489: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_488, 3);  unsqueeze_488 = None
    mul_741: "f32[192]" = torch.ops.aten.mul.Tensor(sum_21, 0.001953125)
    mul_742: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_253, squeeze_253)
    mul_743: "f32[192]" = torch.ops.aten.mul.Tensor(mul_741, mul_742);  mul_741 = mul_742 = None
    unsqueeze_490: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_743, 0);  mul_743 = None
    unsqueeze_491: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_490, 2);  unsqueeze_490 = None
    unsqueeze_492: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_491, 3);  unsqueeze_491 = None
    mul_744: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_253, primals_169);  primals_169 = None
    unsqueeze_493: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_744, 0);  mul_744 = None
    unsqueeze_494: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_493, 2);  unsqueeze_493 = None
    unsqueeze_495: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_494, 3);  unsqueeze_494 = None
    mul_745: "f32[8, 192, 8, 8]" = torch.ops.aten.mul.Tensor(sub_130, unsqueeze_492);  sub_130 = unsqueeze_492 = None
    sub_132: "f32[8, 192, 8, 8]" = torch.ops.aten.sub.Tensor(where_9, mul_745);  where_9 = mul_745 = None
    sub_133: "f32[8, 192, 8, 8]" = torch.ops.aten.sub.Tensor(sub_132, unsqueeze_489);  sub_132 = unsqueeze_489 = None
    mul_746: "f32[8, 192, 8, 8]" = torch.ops.aten.mul.Tensor(sub_133, unsqueeze_495);  sub_133 = unsqueeze_495 = None
    mul_747: "f32[192]" = torch.ops.aten.mul.Tensor(sum_21, squeeze_253);  sum_21 = squeeze_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_9 = torch.ops.aten.convolution_backward.default(mul_746, avg_pool2d_7, primals_273, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_746 = avg_pool2d_7 = primals_273 = None
    getitem_223: "f32[8, 1280, 8, 8]" = convolution_backward_9[0]
    getitem_224: "f32[192, 1280, 1, 1]" = convolution_backward_9[1];  convolution_backward_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/inception_v3.py:198, code: branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    avg_pool2d_backward_1: "f32[8, 1280, 8, 8]" = torch.ops.aten.avg_pool2d_backward.default(getitem_223, cat_8, [3, 3], [1, 1], [1, 1], False, True, None);  getitem_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/inception_v3.py:196, code: branch3x3dbl = torch.cat(branch3x3dbl, 1)
    slice_13: "f32[8, 384, 8, 8]" = torch.ops.aten.slice.Tensor(slice_11, 1, 0, 384)
    slice_14: "f32[8, 384, 8, 8]" = torch.ops.aten.slice.Tensor(slice_11, 1, 384, 768);  slice_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    where_10: "f32[8, 384, 8, 8]" = torch.ops.aten.where.self(le_10, full_default, slice_14);  le_10 = slice_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_22: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_10, [0, 2, 3])
    sub_134: "f32[8, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_83, unsqueeze_498);  convolution_83 = unsqueeze_498 = None
    mul_748: "f32[8, 384, 8, 8]" = torch.ops.aten.mul.Tensor(where_10, sub_134)
    sum_23: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_748, [0, 2, 3]);  mul_748 = None
    mul_749: "f32[384]" = torch.ops.aten.mul.Tensor(sum_22, 0.001953125)
    unsqueeze_499: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_749, 0);  mul_749 = None
    unsqueeze_500: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_499, 2);  unsqueeze_499 = None
    unsqueeze_501: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_500, 3);  unsqueeze_500 = None
    mul_750: "f32[384]" = torch.ops.aten.mul.Tensor(sum_23, 0.001953125)
    mul_751: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_250, squeeze_250)
    mul_752: "f32[384]" = torch.ops.aten.mul.Tensor(mul_750, mul_751);  mul_750 = mul_751 = None
    unsqueeze_502: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_752, 0);  mul_752 = None
    unsqueeze_503: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_502, 2);  unsqueeze_502 = None
    unsqueeze_504: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_503, 3);  unsqueeze_503 = None
    mul_753: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_250, primals_167);  primals_167 = None
    unsqueeze_505: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_753, 0);  mul_753 = None
    unsqueeze_506: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_505, 2);  unsqueeze_505 = None
    unsqueeze_507: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_506, 3);  unsqueeze_506 = None
    mul_754: "f32[8, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_134, unsqueeze_504);  sub_134 = unsqueeze_504 = None
    sub_136: "f32[8, 384, 8, 8]" = torch.ops.aten.sub.Tensor(where_10, mul_754);  where_10 = mul_754 = None
    sub_137: "f32[8, 384, 8, 8]" = torch.ops.aten.sub.Tensor(sub_136, unsqueeze_501);  sub_136 = unsqueeze_501 = None
    mul_755: "f32[8, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_137, unsqueeze_507);  sub_137 = unsqueeze_507 = None
    mul_756: "f32[384]" = torch.ops.aten.mul.Tensor(sum_23, squeeze_250);  sum_23 = squeeze_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_10 = torch.ops.aten.convolution_backward.default(mul_755, relu_81, primals_272, [0], [1, 1], [1, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_755 = primals_272 = None
    getitem_226: "f32[8, 384, 8, 8]" = convolution_backward_10[0]
    getitem_227: "f32[384, 384, 3, 1]" = convolution_backward_10[1];  convolution_backward_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    where_11: "f32[8, 384, 8, 8]" = torch.ops.aten.where.self(le_11, full_default, slice_13);  le_11 = slice_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_24: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_11, [0, 2, 3])
    sub_138: "f32[8, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_82, unsqueeze_510);  convolution_82 = unsqueeze_510 = None
    mul_757: "f32[8, 384, 8, 8]" = torch.ops.aten.mul.Tensor(where_11, sub_138)
    sum_25: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_757, [0, 2, 3]);  mul_757 = None
    mul_758: "f32[384]" = torch.ops.aten.mul.Tensor(sum_24, 0.001953125)
    unsqueeze_511: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_758, 0);  mul_758 = None
    unsqueeze_512: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_511, 2);  unsqueeze_511 = None
    unsqueeze_513: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_512, 3);  unsqueeze_512 = None
    mul_759: "f32[384]" = torch.ops.aten.mul.Tensor(sum_25, 0.001953125)
    mul_760: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_247, squeeze_247)
    mul_761: "f32[384]" = torch.ops.aten.mul.Tensor(mul_759, mul_760);  mul_759 = mul_760 = None
    unsqueeze_514: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_761, 0);  mul_761 = None
    unsqueeze_515: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_514, 2);  unsqueeze_514 = None
    unsqueeze_516: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_515, 3);  unsqueeze_515 = None
    mul_762: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_247, primals_165);  primals_165 = None
    unsqueeze_517: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_762, 0);  mul_762 = None
    unsqueeze_518: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_517, 2);  unsqueeze_517 = None
    unsqueeze_519: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_518, 3);  unsqueeze_518 = None
    mul_763: "f32[8, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_138, unsqueeze_516);  sub_138 = unsqueeze_516 = None
    sub_140: "f32[8, 384, 8, 8]" = torch.ops.aten.sub.Tensor(where_11, mul_763);  where_11 = mul_763 = None
    sub_141: "f32[8, 384, 8, 8]" = torch.ops.aten.sub.Tensor(sub_140, unsqueeze_513);  sub_140 = unsqueeze_513 = None
    mul_764: "f32[8, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_141, unsqueeze_519);  sub_141 = unsqueeze_519 = None
    mul_765: "f32[384]" = torch.ops.aten.mul.Tensor(sum_25, squeeze_247);  sum_25 = squeeze_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_11 = torch.ops.aten.convolution_backward.default(mul_764, relu_81, primals_271, [0], [1, 1], [0, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_764 = primals_271 = None
    getitem_229: "f32[8, 384, 8, 8]" = convolution_backward_11[0]
    getitem_230: "f32[384, 384, 1, 3]" = convolution_backward_11[1];  convolution_backward_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_475: "f32[8, 384, 8, 8]" = torch.ops.aten.add.Tensor(getitem_226, getitem_229);  getitem_226 = getitem_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_131: "f32[8, 384, 8, 8]" = torch.ops.aten.alias.default(relu_81);  relu_81 = None
    alias_132: "f32[8, 384, 8, 8]" = torch.ops.aten.alias.default(alias_131);  alias_131 = None
    le_12: "b8[8, 384, 8, 8]" = torch.ops.aten.le.Scalar(alias_132, 0);  alias_132 = None
    where_12: "f32[8, 384, 8, 8]" = torch.ops.aten.where.self(le_12, full_default, add_475);  le_12 = add_475 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_26: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_12, [0, 2, 3])
    sub_142: "f32[8, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_81, unsqueeze_522);  convolution_81 = unsqueeze_522 = None
    mul_766: "f32[8, 384, 8, 8]" = torch.ops.aten.mul.Tensor(where_12, sub_142)
    sum_27: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_766, [0, 2, 3]);  mul_766 = None
    mul_767: "f32[384]" = torch.ops.aten.mul.Tensor(sum_26, 0.001953125)
    unsqueeze_523: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_767, 0);  mul_767 = None
    unsqueeze_524: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_523, 2);  unsqueeze_523 = None
    unsqueeze_525: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_524, 3);  unsqueeze_524 = None
    mul_768: "f32[384]" = torch.ops.aten.mul.Tensor(sum_27, 0.001953125)
    mul_769: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_244, squeeze_244)
    mul_770: "f32[384]" = torch.ops.aten.mul.Tensor(mul_768, mul_769);  mul_768 = mul_769 = None
    unsqueeze_526: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_770, 0);  mul_770 = None
    unsqueeze_527: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_526, 2);  unsqueeze_526 = None
    unsqueeze_528: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_527, 3);  unsqueeze_527 = None
    mul_771: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_244, primals_163);  primals_163 = None
    unsqueeze_529: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_771, 0);  mul_771 = None
    unsqueeze_530: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_529, 2);  unsqueeze_529 = None
    unsqueeze_531: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_530, 3);  unsqueeze_530 = None
    mul_772: "f32[8, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_142, unsqueeze_528);  sub_142 = unsqueeze_528 = None
    sub_144: "f32[8, 384, 8, 8]" = torch.ops.aten.sub.Tensor(where_12, mul_772);  where_12 = mul_772 = None
    sub_145: "f32[8, 384, 8, 8]" = torch.ops.aten.sub.Tensor(sub_144, unsqueeze_525);  sub_144 = unsqueeze_525 = None
    mul_773: "f32[8, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_145, unsqueeze_531);  sub_145 = unsqueeze_531 = None
    mul_774: "f32[384]" = torch.ops.aten.mul.Tensor(sum_27, squeeze_244);  sum_27 = squeeze_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_12 = torch.ops.aten.convolution_backward.default(mul_773, relu_80, primals_270, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_773 = primals_270 = None
    getitem_232: "f32[8, 448, 8, 8]" = convolution_backward_12[0]
    getitem_233: "f32[384, 448, 3, 3]" = convolution_backward_12[1];  convolution_backward_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_134: "f32[8, 448, 8, 8]" = torch.ops.aten.alias.default(relu_80);  relu_80 = None
    alias_135: "f32[8, 448, 8, 8]" = torch.ops.aten.alias.default(alias_134);  alias_134 = None
    le_13: "b8[8, 448, 8, 8]" = torch.ops.aten.le.Scalar(alias_135, 0);  alias_135 = None
    where_13: "f32[8, 448, 8, 8]" = torch.ops.aten.where.self(le_13, full_default, getitem_232);  le_13 = getitem_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_28: "f32[448]" = torch.ops.aten.sum.dim_IntList(where_13, [0, 2, 3])
    sub_146: "f32[8, 448, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_80, unsqueeze_534);  convolution_80 = unsqueeze_534 = None
    mul_775: "f32[8, 448, 8, 8]" = torch.ops.aten.mul.Tensor(where_13, sub_146)
    sum_29: "f32[448]" = torch.ops.aten.sum.dim_IntList(mul_775, [0, 2, 3]);  mul_775 = None
    mul_776: "f32[448]" = torch.ops.aten.mul.Tensor(sum_28, 0.001953125)
    unsqueeze_535: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_776, 0);  mul_776 = None
    unsqueeze_536: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_535, 2);  unsqueeze_535 = None
    unsqueeze_537: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_536, 3);  unsqueeze_536 = None
    mul_777: "f32[448]" = torch.ops.aten.mul.Tensor(sum_29, 0.001953125)
    mul_778: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_241, squeeze_241)
    mul_779: "f32[448]" = torch.ops.aten.mul.Tensor(mul_777, mul_778);  mul_777 = mul_778 = None
    unsqueeze_538: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_779, 0);  mul_779 = None
    unsqueeze_539: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_538, 2);  unsqueeze_538 = None
    unsqueeze_540: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_539, 3);  unsqueeze_539 = None
    mul_780: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_241, primals_161);  primals_161 = None
    unsqueeze_541: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_780, 0);  mul_780 = None
    unsqueeze_542: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_541, 2);  unsqueeze_541 = None
    unsqueeze_543: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_542, 3);  unsqueeze_542 = None
    mul_781: "f32[8, 448, 8, 8]" = torch.ops.aten.mul.Tensor(sub_146, unsqueeze_540);  sub_146 = unsqueeze_540 = None
    sub_148: "f32[8, 448, 8, 8]" = torch.ops.aten.sub.Tensor(where_13, mul_781);  where_13 = mul_781 = None
    sub_149: "f32[8, 448, 8, 8]" = torch.ops.aten.sub.Tensor(sub_148, unsqueeze_537);  sub_148 = unsqueeze_537 = None
    mul_782: "f32[8, 448, 8, 8]" = torch.ops.aten.mul.Tensor(sub_149, unsqueeze_543);  sub_149 = unsqueeze_543 = None
    mul_783: "f32[448]" = torch.ops.aten.mul.Tensor(sum_29, squeeze_241);  sum_29 = squeeze_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_13 = torch.ops.aten.convolution_backward.default(mul_782, cat_8, primals_269, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_782 = primals_269 = None
    getitem_235: "f32[8, 1280, 8, 8]" = convolution_backward_13[0]
    getitem_236: "f32[448, 1280, 1, 1]" = convolution_backward_13[1];  convolution_backward_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_476: "f32[8, 1280, 8, 8]" = torch.ops.aten.add.Tensor(avg_pool2d_backward_1, getitem_235);  avg_pool2d_backward_1 = getitem_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/inception_v3.py:188, code: branch3x3 = torch.cat(branch3x3, 1)
    slice_15: "f32[8, 384, 8, 8]" = torch.ops.aten.slice.Tensor(slice_10, 1, 0, 384)
    slice_16: "f32[8, 384, 8, 8]" = torch.ops.aten.slice.Tensor(slice_10, 1, 384, 768);  slice_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    where_14: "f32[8, 384, 8, 8]" = torch.ops.aten.where.self(le_14, full_default, slice_16);  le_14 = slice_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_30: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_14, [0, 2, 3])
    sub_150: "f32[8, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_79, unsqueeze_546);  convolution_79 = unsqueeze_546 = None
    mul_784: "f32[8, 384, 8, 8]" = torch.ops.aten.mul.Tensor(where_14, sub_150)
    sum_31: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_784, [0, 2, 3]);  mul_784 = None
    mul_785: "f32[384]" = torch.ops.aten.mul.Tensor(sum_30, 0.001953125)
    unsqueeze_547: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_785, 0);  mul_785 = None
    unsqueeze_548: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_547, 2);  unsqueeze_547 = None
    unsqueeze_549: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_548, 3);  unsqueeze_548 = None
    mul_786: "f32[384]" = torch.ops.aten.mul.Tensor(sum_31, 0.001953125)
    mul_787: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_238, squeeze_238)
    mul_788: "f32[384]" = torch.ops.aten.mul.Tensor(mul_786, mul_787);  mul_786 = mul_787 = None
    unsqueeze_550: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_788, 0);  mul_788 = None
    unsqueeze_551: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_550, 2);  unsqueeze_550 = None
    unsqueeze_552: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_551, 3);  unsqueeze_551 = None
    mul_789: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_238, primals_159);  primals_159 = None
    unsqueeze_553: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_789, 0);  mul_789 = None
    unsqueeze_554: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_553, 2);  unsqueeze_553 = None
    unsqueeze_555: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_554, 3);  unsqueeze_554 = None
    mul_790: "f32[8, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_150, unsqueeze_552);  sub_150 = unsqueeze_552 = None
    sub_152: "f32[8, 384, 8, 8]" = torch.ops.aten.sub.Tensor(where_14, mul_790);  where_14 = mul_790 = None
    sub_153: "f32[8, 384, 8, 8]" = torch.ops.aten.sub.Tensor(sub_152, unsqueeze_549);  sub_152 = unsqueeze_549 = None
    mul_791: "f32[8, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_153, unsqueeze_555);  sub_153 = unsqueeze_555 = None
    mul_792: "f32[384]" = torch.ops.aten.mul.Tensor(sum_31, squeeze_238);  sum_31 = squeeze_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_14 = torch.ops.aten.convolution_backward.default(mul_791, relu_77, primals_268, [0], [1, 1], [1, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_791 = primals_268 = None
    getitem_238: "f32[8, 384, 8, 8]" = convolution_backward_14[0]
    getitem_239: "f32[384, 384, 3, 1]" = convolution_backward_14[1];  convolution_backward_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    where_15: "f32[8, 384, 8, 8]" = torch.ops.aten.where.self(le_15, full_default, slice_15);  le_15 = slice_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_32: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_15, [0, 2, 3])
    sub_154: "f32[8, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_78, unsqueeze_558);  convolution_78 = unsqueeze_558 = None
    mul_793: "f32[8, 384, 8, 8]" = torch.ops.aten.mul.Tensor(where_15, sub_154)
    sum_33: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_793, [0, 2, 3]);  mul_793 = None
    mul_794: "f32[384]" = torch.ops.aten.mul.Tensor(sum_32, 0.001953125)
    unsqueeze_559: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_794, 0);  mul_794 = None
    unsqueeze_560: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_559, 2);  unsqueeze_559 = None
    unsqueeze_561: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_560, 3);  unsqueeze_560 = None
    mul_795: "f32[384]" = torch.ops.aten.mul.Tensor(sum_33, 0.001953125)
    mul_796: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_235, squeeze_235)
    mul_797: "f32[384]" = torch.ops.aten.mul.Tensor(mul_795, mul_796);  mul_795 = mul_796 = None
    unsqueeze_562: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_797, 0);  mul_797 = None
    unsqueeze_563: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_562, 2);  unsqueeze_562 = None
    unsqueeze_564: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_563, 3);  unsqueeze_563 = None
    mul_798: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_235, primals_157);  primals_157 = None
    unsqueeze_565: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_798, 0);  mul_798 = None
    unsqueeze_566: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_565, 2);  unsqueeze_565 = None
    unsqueeze_567: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_566, 3);  unsqueeze_566 = None
    mul_799: "f32[8, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_154, unsqueeze_564);  sub_154 = unsqueeze_564 = None
    sub_156: "f32[8, 384, 8, 8]" = torch.ops.aten.sub.Tensor(where_15, mul_799);  where_15 = mul_799 = None
    sub_157: "f32[8, 384, 8, 8]" = torch.ops.aten.sub.Tensor(sub_156, unsqueeze_561);  sub_156 = unsqueeze_561 = None
    mul_800: "f32[8, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_157, unsqueeze_567);  sub_157 = unsqueeze_567 = None
    mul_801: "f32[384]" = torch.ops.aten.mul.Tensor(sum_33, squeeze_235);  sum_33 = squeeze_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_15 = torch.ops.aten.convolution_backward.default(mul_800, relu_77, primals_267, [0], [1, 1], [0, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_800 = primals_267 = None
    getitem_241: "f32[8, 384, 8, 8]" = convolution_backward_15[0]
    getitem_242: "f32[384, 384, 1, 3]" = convolution_backward_15[1];  convolution_backward_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_477: "f32[8, 384, 8, 8]" = torch.ops.aten.add.Tensor(getitem_238, getitem_241);  getitem_238 = getitem_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_143: "f32[8, 384, 8, 8]" = torch.ops.aten.alias.default(relu_77);  relu_77 = None
    alias_144: "f32[8, 384, 8, 8]" = torch.ops.aten.alias.default(alias_143);  alias_143 = None
    le_16: "b8[8, 384, 8, 8]" = torch.ops.aten.le.Scalar(alias_144, 0);  alias_144 = None
    where_16: "f32[8, 384, 8, 8]" = torch.ops.aten.where.self(le_16, full_default, add_477);  le_16 = add_477 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_34: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_16, [0, 2, 3])
    sub_158: "f32[8, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_77, unsqueeze_570);  convolution_77 = unsqueeze_570 = None
    mul_802: "f32[8, 384, 8, 8]" = torch.ops.aten.mul.Tensor(where_16, sub_158)
    sum_35: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_802, [0, 2, 3]);  mul_802 = None
    mul_803: "f32[384]" = torch.ops.aten.mul.Tensor(sum_34, 0.001953125)
    unsqueeze_571: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_803, 0);  mul_803 = None
    unsqueeze_572: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_571, 2);  unsqueeze_571 = None
    unsqueeze_573: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_572, 3);  unsqueeze_572 = None
    mul_804: "f32[384]" = torch.ops.aten.mul.Tensor(sum_35, 0.001953125)
    mul_805: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_232, squeeze_232)
    mul_806: "f32[384]" = torch.ops.aten.mul.Tensor(mul_804, mul_805);  mul_804 = mul_805 = None
    unsqueeze_574: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_806, 0);  mul_806 = None
    unsqueeze_575: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_574, 2);  unsqueeze_574 = None
    unsqueeze_576: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_575, 3);  unsqueeze_575 = None
    mul_807: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_232, primals_155);  primals_155 = None
    unsqueeze_577: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_807, 0);  mul_807 = None
    unsqueeze_578: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_577, 2);  unsqueeze_577 = None
    unsqueeze_579: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_578, 3);  unsqueeze_578 = None
    mul_808: "f32[8, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_158, unsqueeze_576);  sub_158 = unsqueeze_576 = None
    sub_160: "f32[8, 384, 8, 8]" = torch.ops.aten.sub.Tensor(where_16, mul_808);  where_16 = mul_808 = None
    sub_161: "f32[8, 384, 8, 8]" = torch.ops.aten.sub.Tensor(sub_160, unsqueeze_573);  sub_160 = unsqueeze_573 = None
    mul_809: "f32[8, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_161, unsqueeze_579);  sub_161 = unsqueeze_579 = None
    mul_810: "f32[384]" = torch.ops.aten.mul.Tensor(sum_35, squeeze_232);  sum_35 = squeeze_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_16 = torch.ops.aten.convolution_backward.default(mul_809, cat_8, primals_266, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_809 = primals_266 = None
    getitem_244: "f32[8, 1280, 8, 8]" = convolution_backward_16[0]
    getitem_245: "f32[384, 1280, 1, 1]" = convolution_backward_16[1];  convolution_backward_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_478: "f32[8, 1280, 8, 8]" = torch.ops.aten.add.Tensor(add_476, getitem_244);  add_476 = getitem_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    where_17: "f32[8, 320, 8, 8]" = torch.ops.aten.where.self(le_17, full_default, slice_9);  le_17 = slice_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_36: "f32[320]" = torch.ops.aten.sum.dim_IntList(where_17, [0, 2, 3])
    sub_162: "f32[8, 320, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_76, unsqueeze_582);  convolution_76 = unsqueeze_582 = None
    mul_811: "f32[8, 320, 8, 8]" = torch.ops.aten.mul.Tensor(where_17, sub_162)
    sum_37: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_811, [0, 2, 3]);  mul_811 = None
    mul_812: "f32[320]" = torch.ops.aten.mul.Tensor(sum_36, 0.001953125)
    unsqueeze_583: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(mul_812, 0);  mul_812 = None
    unsqueeze_584: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_583, 2);  unsqueeze_583 = None
    unsqueeze_585: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_584, 3);  unsqueeze_584 = None
    mul_813: "f32[320]" = torch.ops.aten.mul.Tensor(sum_37, 0.001953125)
    mul_814: "f32[320]" = torch.ops.aten.mul.Tensor(squeeze_229, squeeze_229)
    mul_815: "f32[320]" = torch.ops.aten.mul.Tensor(mul_813, mul_814);  mul_813 = mul_814 = None
    unsqueeze_586: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(mul_815, 0);  mul_815 = None
    unsqueeze_587: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_586, 2);  unsqueeze_586 = None
    unsqueeze_588: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_587, 3);  unsqueeze_587 = None
    mul_816: "f32[320]" = torch.ops.aten.mul.Tensor(squeeze_229, primals_153);  primals_153 = None
    unsqueeze_589: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(mul_816, 0);  mul_816 = None
    unsqueeze_590: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_589, 2);  unsqueeze_589 = None
    unsqueeze_591: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_590, 3);  unsqueeze_590 = None
    mul_817: "f32[8, 320, 8, 8]" = torch.ops.aten.mul.Tensor(sub_162, unsqueeze_588);  sub_162 = unsqueeze_588 = None
    sub_164: "f32[8, 320, 8, 8]" = torch.ops.aten.sub.Tensor(where_17, mul_817);  where_17 = mul_817 = None
    sub_165: "f32[8, 320, 8, 8]" = torch.ops.aten.sub.Tensor(sub_164, unsqueeze_585);  sub_164 = unsqueeze_585 = None
    mul_818: "f32[8, 320, 8, 8]" = torch.ops.aten.mul.Tensor(sub_165, unsqueeze_591);  sub_165 = unsqueeze_591 = None
    mul_819: "f32[320]" = torch.ops.aten.mul.Tensor(sum_37, squeeze_229);  sum_37 = squeeze_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_17 = torch.ops.aten.convolution_backward.default(mul_818, cat_8, primals_265, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_818 = cat_8 = primals_265 = None
    getitem_247: "f32[8, 1280, 8, 8]" = convolution_backward_17[0]
    getitem_248: "f32[320, 1280, 1, 1]" = convolution_backward_17[1];  convolution_backward_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_479: "f32[8, 1280, 8, 8]" = torch.ops.aten.add.Tensor(add_478, getitem_247);  add_478 = getitem_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/inception_v3.py:159, code: return torch.cat(outputs, 1)
    slice_17: "f32[8, 320, 8, 8]" = torch.ops.aten.slice.Tensor(add_479, 1, 0, 320)
    slice_18: "f32[8, 192, 8, 8]" = torch.ops.aten.slice.Tensor(add_479, 1, 320, 512)
    slice_19: "f32[8, 768, 8, 8]" = torch.ops.aten.slice.Tensor(add_479, 1, 512, 1280);  add_479 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/inception_v3.py:153, code: branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
    max_pool2d_with_indices_backward: "f32[8, 768, 17, 17]" = torch.ops.aten.max_pool2d_with_indices_backward.default(slice_19, cat_7, [3, 3], [2, 2], [0, 0], [1, 1], False, getitem_159);  slice_19 = getitem_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    where_18: "f32[8, 192, 8, 8]" = torch.ops.aten.where.self(le_18, full_default, slice_18);  le_18 = slice_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_38: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_18, [0, 2, 3])
    sub_166: "f32[8, 192, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_75, unsqueeze_594);  convolution_75 = unsqueeze_594 = None
    mul_820: "f32[8, 192, 8, 8]" = torch.ops.aten.mul.Tensor(where_18, sub_166)
    sum_39: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_820, [0, 2, 3]);  mul_820 = None
    mul_821: "f32[192]" = torch.ops.aten.mul.Tensor(sum_38, 0.001953125)
    unsqueeze_595: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_821, 0);  mul_821 = None
    unsqueeze_596: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_595, 2);  unsqueeze_595 = None
    unsqueeze_597: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_596, 3);  unsqueeze_596 = None
    mul_822: "f32[192]" = torch.ops.aten.mul.Tensor(sum_39, 0.001953125)
    mul_823: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_226, squeeze_226)
    mul_824: "f32[192]" = torch.ops.aten.mul.Tensor(mul_822, mul_823);  mul_822 = mul_823 = None
    unsqueeze_598: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_824, 0);  mul_824 = None
    unsqueeze_599: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_598, 2);  unsqueeze_598 = None
    unsqueeze_600: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_599, 3);  unsqueeze_599 = None
    mul_825: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_226, primals_151);  primals_151 = None
    unsqueeze_601: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_825, 0);  mul_825 = None
    unsqueeze_602: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_601, 2);  unsqueeze_601 = None
    unsqueeze_603: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_602, 3);  unsqueeze_602 = None
    mul_826: "f32[8, 192, 8, 8]" = torch.ops.aten.mul.Tensor(sub_166, unsqueeze_600);  sub_166 = unsqueeze_600 = None
    sub_168: "f32[8, 192, 8, 8]" = torch.ops.aten.sub.Tensor(where_18, mul_826);  where_18 = mul_826 = None
    sub_169: "f32[8, 192, 8, 8]" = torch.ops.aten.sub.Tensor(sub_168, unsqueeze_597);  sub_168 = unsqueeze_597 = None
    mul_827: "f32[8, 192, 8, 8]" = torch.ops.aten.mul.Tensor(sub_169, unsqueeze_603);  sub_169 = unsqueeze_603 = None
    mul_828: "f32[192]" = torch.ops.aten.mul.Tensor(sum_39, squeeze_226);  sum_39 = squeeze_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_18 = torch.ops.aten.convolution_backward.default(mul_827, relu_74, primals_264, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_827 = primals_264 = None
    getitem_250: "f32[8, 192, 17, 17]" = convolution_backward_18[0]
    getitem_251: "f32[192, 192, 3, 3]" = convolution_backward_18[1];  convolution_backward_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_152: "f32[8, 192, 17, 17]" = torch.ops.aten.alias.default(relu_74);  relu_74 = None
    alias_153: "f32[8, 192, 17, 17]" = torch.ops.aten.alias.default(alias_152);  alias_152 = None
    le_19: "b8[8, 192, 17, 17]" = torch.ops.aten.le.Scalar(alias_153, 0);  alias_153 = None
    where_19: "f32[8, 192, 17, 17]" = torch.ops.aten.where.self(le_19, full_default, getitem_250);  le_19 = getitem_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_40: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_19, [0, 2, 3])
    sub_170: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_74, unsqueeze_606);  convolution_74 = unsqueeze_606 = None
    mul_829: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(where_19, sub_170)
    sum_41: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_829, [0, 2, 3]);  mul_829 = None
    mul_830: "f32[192]" = torch.ops.aten.mul.Tensor(sum_40, 0.00043252595155709344)
    unsqueeze_607: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_830, 0);  mul_830 = None
    unsqueeze_608: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_607, 2);  unsqueeze_607 = None
    unsqueeze_609: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_608, 3);  unsqueeze_608 = None
    mul_831: "f32[192]" = torch.ops.aten.mul.Tensor(sum_41, 0.00043252595155709344)
    mul_832: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_223, squeeze_223)
    mul_833: "f32[192]" = torch.ops.aten.mul.Tensor(mul_831, mul_832);  mul_831 = mul_832 = None
    unsqueeze_610: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_833, 0);  mul_833 = None
    unsqueeze_611: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_610, 2);  unsqueeze_610 = None
    unsqueeze_612: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_611, 3);  unsqueeze_611 = None
    mul_834: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_223, primals_149);  primals_149 = None
    unsqueeze_613: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_834, 0);  mul_834 = None
    unsqueeze_614: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_613, 2);  unsqueeze_613 = None
    unsqueeze_615: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_614, 3);  unsqueeze_614 = None
    mul_835: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_170, unsqueeze_612);  sub_170 = unsqueeze_612 = None
    sub_172: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(where_19, mul_835);  where_19 = mul_835 = None
    sub_173: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(sub_172, unsqueeze_609);  sub_172 = unsqueeze_609 = None
    mul_836: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_173, unsqueeze_615);  sub_173 = unsqueeze_615 = None
    mul_837: "f32[192]" = torch.ops.aten.mul.Tensor(sum_41, squeeze_223);  sum_41 = squeeze_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_19 = torch.ops.aten.convolution_backward.default(mul_836, relu_73, primals_263, [0], [1, 1], [3, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_836 = primals_263 = None
    getitem_253: "f32[8, 192, 17, 17]" = convolution_backward_19[0]
    getitem_254: "f32[192, 192, 7, 1]" = convolution_backward_19[1];  convolution_backward_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_155: "f32[8, 192, 17, 17]" = torch.ops.aten.alias.default(relu_73);  relu_73 = None
    alias_156: "f32[8, 192, 17, 17]" = torch.ops.aten.alias.default(alias_155);  alias_155 = None
    le_20: "b8[8, 192, 17, 17]" = torch.ops.aten.le.Scalar(alias_156, 0);  alias_156 = None
    where_20: "f32[8, 192, 17, 17]" = torch.ops.aten.where.self(le_20, full_default, getitem_253);  le_20 = getitem_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_42: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_20, [0, 2, 3])
    sub_174: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_73, unsqueeze_618);  convolution_73 = unsqueeze_618 = None
    mul_838: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(where_20, sub_174)
    sum_43: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_838, [0, 2, 3]);  mul_838 = None
    mul_839: "f32[192]" = torch.ops.aten.mul.Tensor(sum_42, 0.00043252595155709344)
    unsqueeze_619: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_839, 0);  mul_839 = None
    unsqueeze_620: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_619, 2);  unsqueeze_619 = None
    unsqueeze_621: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_620, 3);  unsqueeze_620 = None
    mul_840: "f32[192]" = torch.ops.aten.mul.Tensor(sum_43, 0.00043252595155709344)
    mul_841: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_220, squeeze_220)
    mul_842: "f32[192]" = torch.ops.aten.mul.Tensor(mul_840, mul_841);  mul_840 = mul_841 = None
    unsqueeze_622: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_842, 0);  mul_842 = None
    unsqueeze_623: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_622, 2);  unsqueeze_622 = None
    unsqueeze_624: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_623, 3);  unsqueeze_623 = None
    mul_843: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_220, primals_147);  primals_147 = None
    unsqueeze_625: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_843, 0);  mul_843 = None
    unsqueeze_626: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_625, 2);  unsqueeze_625 = None
    unsqueeze_627: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_626, 3);  unsqueeze_626 = None
    mul_844: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_174, unsqueeze_624);  sub_174 = unsqueeze_624 = None
    sub_176: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(where_20, mul_844);  where_20 = mul_844 = None
    sub_177: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(sub_176, unsqueeze_621);  sub_176 = unsqueeze_621 = None
    mul_845: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_177, unsqueeze_627);  sub_177 = unsqueeze_627 = None
    mul_846: "f32[192]" = torch.ops.aten.mul.Tensor(sum_43, squeeze_220);  sum_43 = squeeze_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_20 = torch.ops.aten.convolution_backward.default(mul_845, relu_72, primals_262, [0], [1, 1], [0, 3], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_845 = primals_262 = None
    getitem_256: "f32[8, 192, 17, 17]" = convolution_backward_20[0]
    getitem_257: "f32[192, 192, 1, 7]" = convolution_backward_20[1];  convolution_backward_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_158: "f32[8, 192, 17, 17]" = torch.ops.aten.alias.default(relu_72);  relu_72 = None
    alias_159: "f32[8, 192, 17, 17]" = torch.ops.aten.alias.default(alias_158);  alias_158 = None
    le_21: "b8[8, 192, 17, 17]" = torch.ops.aten.le.Scalar(alias_159, 0);  alias_159 = None
    where_21: "f32[8, 192, 17, 17]" = torch.ops.aten.where.self(le_21, full_default, getitem_256);  le_21 = getitem_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_44: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_21, [0, 2, 3])
    sub_178: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_72, unsqueeze_630);  convolution_72 = unsqueeze_630 = None
    mul_847: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(where_21, sub_178)
    sum_45: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_847, [0, 2, 3]);  mul_847 = None
    mul_848: "f32[192]" = torch.ops.aten.mul.Tensor(sum_44, 0.00043252595155709344)
    unsqueeze_631: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_848, 0);  mul_848 = None
    unsqueeze_632: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_631, 2);  unsqueeze_631 = None
    unsqueeze_633: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_632, 3);  unsqueeze_632 = None
    mul_849: "f32[192]" = torch.ops.aten.mul.Tensor(sum_45, 0.00043252595155709344)
    mul_850: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_217, squeeze_217)
    mul_851: "f32[192]" = torch.ops.aten.mul.Tensor(mul_849, mul_850);  mul_849 = mul_850 = None
    unsqueeze_634: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_851, 0);  mul_851 = None
    unsqueeze_635: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_634, 2);  unsqueeze_634 = None
    unsqueeze_636: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_635, 3);  unsqueeze_635 = None
    mul_852: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_217, primals_145);  primals_145 = None
    unsqueeze_637: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_852, 0);  mul_852 = None
    unsqueeze_638: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_637, 2);  unsqueeze_637 = None
    unsqueeze_639: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_638, 3);  unsqueeze_638 = None
    mul_853: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_178, unsqueeze_636);  sub_178 = unsqueeze_636 = None
    sub_180: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(where_21, mul_853);  where_21 = mul_853 = None
    sub_181: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(sub_180, unsqueeze_633);  sub_180 = unsqueeze_633 = None
    mul_854: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_181, unsqueeze_639);  sub_181 = unsqueeze_639 = None
    mul_855: "f32[192]" = torch.ops.aten.mul.Tensor(sum_45, squeeze_217);  sum_45 = squeeze_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_21 = torch.ops.aten.convolution_backward.default(mul_854, cat_7, primals_261, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_854 = primals_261 = None
    getitem_259: "f32[8, 768, 17, 17]" = convolution_backward_21[0]
    getitem_260: "f32[192, 768, 1, 1]" = convolution_backward_21[1];  convolution_backward_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_480: "f32[8, 768, 17, 17]" = torch.ops.aten.add.Tensor(max_pool2d_with_indices_backward, getitem_259);  max_pool2d_with_indices_backward = getitem_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    where_22: "f32[8, 320, 8, 8]" = torch.ops.aten.where.self(le_22, full_default, slice_17);  le_22 = slice_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_46: "f32[320]" = torch.ops.aten.sum.dim_IntList(where_22, [0, 2, 3])
    sub_182: "f32[8, 320, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_71, unsqueeze_642);  convolution_71 = unsqueeze_642 = None
    mul_856: "f32[8, 320, 8, 8]" = torch.ops.aten.mul.Tensor(where_22, sub_182)
    sum_47: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_856, [0, 2, 3]);  mul_856 = None
    mul_857: "f32[320]" = torch.ops.aten.mul.Tensor(sum_46, 0.001953125)
    unsqueeze_643: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(mul_857, 0);  mul_857 = None
    unsqueeze_644: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_643, 2);  unsqueeze_643 = None
    unsqueeze_645: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_644, 3);  unsqueeze_644 = None
    mul_858: "f32[320]" = torch.ops.aten.mul.Tensor(sum_47, 0.001953125)
    mul_859: "f32[320]" = torch.ops.aten.mul.Tensor(squeeze_214, squeeze_214)
    mul_860: "f32[320]" = torch.ops.aten.mul.Tensor(mul_858, mul_859);  mul_858 = mul_859 = None
    unsqueeze_646: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(mul_860, 0);  mul_860 = None
    unsqueeze_647: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_646, 2);  unsqueeze_646 = None
    unsqueeze_648: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_647, 3);  unsqueeze_647 = None
    mul_861: "f32[320]" = torch.ops.aten.mul.Tensor(squeeze_214, primals_143);  primals_143 = None
    unsqueeze_649: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(mul_861, 0);  mul_861 = None
    unsqueeze_650: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_649, 2);  unsqueeze_649 = None
    unsqueeze_651: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_650, 3);  unsqueeze_650 = None
    mul_862: "f32[8, 320, 8, 8]" = torch.ops.aten.mul.Tensor(sub_182, unsqueeze_648);  sub_182 = unsqueeze_648 = None
    sub_184: "f32[8, 320, 8, 8]" = torch.ops.aten.sub.Tensor(where_22, mul_862);  where_22 = mul_862 = None
    sub_185: "f32[8, 320, 8, 8]" = torch.ops.aten.sub.Tensor(sub_184, unsqueeze_645);  sub_184 = unsqueeze_645 = None
    mul_863: "f32[8, 320, 8, 8]" = torch.ops.aten.mul.Tensor(sub_185, unsqueeze_651);  sub_185 = unsqueeze_651 = None
    mul_864: "f32[320]" = torch.ops.aten.mul.Tensor(sum_47, squeeze_214);  sum_47 = squeeze_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_22 = torch.ops.aten.convolution_backward.default(mul_863, relu_70, primals_260, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_863 = primals_260 = None
    getitem_262: "f32[8, 192, 17, 17]" = convolution_backward_22[0]
    getitem_263: "f32[320, 192, 3, 3]" = convolution_backward_22[1];  convolution_backward_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_164: "f32[8, 192, 17, 17]" = torch.ops.aten.alias.default(relu_70);  relu_70 = None
    alias_165: "f32[8, 192, 17, 17]" = torch.ops.aten.alias.default(alias_164);  alias_164 = None
    le_23: "b8[8, 192, 17, 17]" = torch.ops.aten.le.Scalar(alias_165, 0);  alias_165 = None
    where_23: "f32[8, 192, 17, 17]" = torch.ops.aten.where.self(le_23, full_default, getitem_262);  le_23 = getitem_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_48: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_23, [0, 2, 3])
    sub_186: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_70, unsqueeze_654);  convolution_70 = unsqueeze_654 = None
    mul_865: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(where_23, sub_186)
    sum_49: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_865, [0, 2, 3]);  mul_865 = None
    mul_866: "f32[192]" = torch.ops.aten.mul.Tensor(sum_48, 0.00043252595155709344)
    unsqueeze_655: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_866, 0);  mul_866 = None
    unsqueeze_656: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_655, 2);  unsqueeze_655 = None
    unsqueeze_657: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_656, 3);  unsqueeze_656 = None
    mul_867: "f32[192]" = torch.ops.aten.mul.Tensor(sum_49, 0.00043252595155709344)
    mul_868: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_211, squeeze_211)
    mul_869: "f32[192]" = torch.ops.aten.mul.Tensor(mul_867, mul_868);  mul_867 = mul_868 = None
    unsqueeze_658: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_869, 0);  mul_869 = None
    unsqueeze_659: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_658, 2);  unsqueeze_658 = None
    unsqueeze_660: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_659, 3);  unsqueeze_659 = None
    mul_870: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_211, primals_141);  primals_141 = None
    unsqueeze_661: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_870, 0);  mul_870 = None
    unsqueeze_662: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_661, 2);  unsqueeze_661 = None
    unsqueeze_663: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_662, 3);  unsqueeze_662 = None
    mul_871: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_186, unsqueeze_660);  sub_186 = unsqueeze_660 = None
    sub_188: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(where_23, mul_871);  where_23 = mul_871 = None
    sub_189: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(sub_188, unsqueeze_657);  sub_188 = unsqueeze_657 = None
    mul_872: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_189, unsqueeze_663);  sub_189 = unsqueeze_663 = None
    mul_873: "f32[192]" = torch.ops.aten.mul.Tensor(sum_49, squeeze_211);  sum_49 = squeeze_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_23 = torch.ops.aten.convolution_backward.default(mul_872, cat_7, primals_259, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_872 = cat_7 = primals_259 = None
    getitem_265: "f32[8, 768, 17, 17]" = convolution_backward_23[0]
    getitem_266: "f32[192, 768, 1, 1]" = convolution_backward_23[1];  convolution_backward_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_481: "f32[8, 768, 17, 17]" = torch.ops.aten.add.Tensor(add_480, getitem_265);  add_480 = getitem_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/inception_v3.py:128, code: return torch.cat(outputs, 1)
    slice_20: "f32[8, 192, 17, 17]" = torch.ops.aten.slice.Tensor(add_481, 1, 0, 192)
    slice_21: "f32[8, 192, 17, 17]" = torch.ops.aten.slice.Tensor(add_481, 1, 192, 384)
    slice_22: "f32[8, 192, 17, 17]" = torch.ops.aten.slice.Tensor(add_481, 1, 384, 576)
    slice_23: "f32[8, 192, 17, 17]" = torch.ops.aten.slice.Tensor(add_481, 1, 576, 768);  add_481 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    where_24: "f32[8, 192, 17, 17]" = torch.ops.aten.where.self(le_24, full_default, slice_23);  le_24 = slice_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_50: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_24, [0, 2, 3])
    sub_190: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_69, unsqueeze_666);  convolution_69 = unsqueeze_666 = None
    mul_874: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(where_24, sub_190)
    sum_51: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_874, [0, 2, 3]);  mul_874 = None
    mul_875: "f32[192]" = torch.ops.aten.mul.Tensor(sum_50, 0.00043252595155709344)
    unsqueeze_667: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_875, 0);  mul_875 = None
    unsqueeze_668: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_667, 2);  unsqueeze_667 = None
    unsqueeze_669: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_668, 3);  unsqueeze_668 = None
    mul_876: "f32[192]" = torch.ops.aten.mul.Tensor(sum_51, 0.00043252595155709344)
    mul_877: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_208, squeeze_208)
    mul_878: "f32[192]" = torch.ops.aten.mul.Tensor(mul_876, mul_877);  mul_876 = mul_877 = None
    unsqueeze_670: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_878, 0);  mul_878 = None
    unsqueeze_671: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_670, 2);  unsqueeze_670 = None
    unsqueeze_672: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_671, 3);  unsqueeze_671 = None
    mul_879: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_208, primals_139);  primals_139 = None
    unsqueeze_673: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_879, 0);  mul_879 = None
    unsqueeze_674: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_673, 2);  unsqueeze_673 = None
    unsqueeze_675: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_674, 3);  unsqueeze_674 = None
    mul_880: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_190, unsqueeze_672);  sub_190 = unsqueeze_672 = None
    sub_192: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(where_24, mul_880);  where_24 = mul_880 = None
    sub_193: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(sub_192, unsqueeze_669);  sub_192 = unsqueeze_669 = None
    mul_881: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_193, unsqueeze_675);  sub_193 = unsqueeze_675 = None
    mul_882: "f32[192]" = torch.ops.aten.mul.Tensor(sum_51, squeeze_208);  sum_51 = squeeze_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_24 = torch.ops.aten.convolution_backward.default(mul_881, avg_pool2d_6, primals_258, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_881 = avg_pool2d_6 = primals_258 = None
    getitem_268: "f32[8, 768, 17, 17]" = convolution_backward_24[0]
    getitem_269: "f32[192, 768, 1, 1]" = convolution_backward_24[1];  convolution_backward_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/inception_v3.py:120, code: branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    avg_pool2d_backward_2: "f32[8, 768, 17, 17]" = torch.ops.aten.avg_pool2d_backward.default(getitem_268, cat_6, [3, 3], [1, 1], [1, 1], False, True, None);  getitem_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    where_25: "f32[8, 192, 17, 17]" = torch.ops.aten.where.self(le_25, full_default, slice_22);  le_25 = slice_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_52: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_25, [0, 2, 3])
    sub_194: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_68, unsqueeze_678);  convolution_68 = unsqueeze_678 = None
    mul_883: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(where_25, sub_194)
    sum_53: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_883, [0, 2, 3]);  mul_883 = None
    mul_884: "f32[192]" = torch.ops.aten.mul.Tensor(sum_52, 0.00043252595155709344)
    unsqueeze_679: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_884, 0);  mul_884 = None
    unsqueeze_680: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_679, 2);  unsqueeze_679 = None
    unsqueeze_681: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_680, 3);  unsqueeze_680 = None
    mul_885: "f32[192]" = torch.ops.aten.mul.Tensor(sum_53, 0.00043252595155709344)
    mul_886: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_205, squeeze_205)
    mul_887: "f32[192]" = torch.ops.aten.mul.Tensor(mul_885, mul_886);  mul_885 = mul_886 = None
    unsqueeze_682: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_887, 0);  mul_887 = None
    unsqueeze_683: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_682, 2);  unsqueeze_682 = None
    unsqueeze_684: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_683, 3);  unsqueeze_683 = None
    mul_888: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_205, primals_137);  primals_137 = None
    unsqueeze_685: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_888, 0);  mul_888 = None
    unsqueeze_686: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_685, 2);  unsqueeze_685 = None
    unsqueeze_687: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_686, 3);  unsqueeze_686 = None
    mul_889: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_194, unsqueeze_684);  sub_194 = unsqueeze_684 = None
    sub_196: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(where_25, mul_889);  where_25 = mul_889 = None
    sub_197: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(sub_196, unsqueeze_681);  sub_196 = unsqueeze_681 = None
    mul_890: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_197, unsqueeze_687);  sub_197 = unsqueeze_687 = None
    mul_891: "f32[192]" = torch.ops.aten.mul.Tensor(sum_53, squeeze_205);  sum_53 = squeeze_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_25 = torch.ops.aten.convolution_backward.default(mul_890, relu_67, primals_257, [0], [1, 1], [0, 3], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_890 = primals_257 = None
    getitem_271: "f32[8, 192, 17, 17]" = convolution_backward_25[0]
    getitem_272: "f32[192, 192, 1, 7]" = convolution_backward_25[1];  convolution_backward_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_173: "f32[8, 192, 17, 17]" = torch.ops.aten.alias.default(relu_67);  relu_67 = None
    alias_174: "f32[8, 192, 17, 17]" = torch.ops.aten.alias.default(alias_173);  alias_173 = None
    le_26: "b8[8, 192, 17, 17]" = torch.ops.aten.le.Scalar(alias_174, 0);  alias_174 = None
    where_26: "f32[8, 192, 17, 17]" = torch.ops.aten.where.self(le_26, full_default, getitem_271);  le_26 = getitem_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_54: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_26, [0, 2, 3])
    sub_198: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_67, unsqueeze_690);  convolution_67 = unsqueeze_690 = None
    mul_892: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(where_26, sub_198)
    sum_55: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_892, [0, 2, 3]);  mul_892 = None
    mul_893: "f32[192]" = torch.ops.aten.mul.Tensor(sum_54, 0.00043252595155709344)
    unsqueeze_691: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_893, 0);  mul_893 = None
    unsqueeze_692: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_691, 2);  unsqueeze_691 = None
    unsqueeze_693: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_692, 3);  unsqueeze_692 = None
    mul_894: "f32[192]" = torch.ops.aten.mul.Tensor(sum_55, 0.00043252595155709344)
    mul_895: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_202, squeeze_202)
    mul_896: "f32[192]" = torch.ops.aten.mul.Tensor(mul_894, mul_895);  mul_894 = mul_895 = None
    unsqueeze_694: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_896, 0);  mul_896 = None
    unsqueeze_695: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_694, 2);  unsqueeze_694 = None
    unsqueeze_696: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_695, 3);  unsqueeze_695 = None
    mul_897: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_202, primals_135);  primals_135 = None
    unsqueeze_697: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_897, 0);  mul_897 = None
    unsqueeze_698: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_697, 2);  unsqueeze_697 = None
    unsqueeze_699: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_698, 3);  unsqueeze_698 = None
    mul_898: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_198, unsqueeze_696);  sub_198 = unsqueeze_696 = None
    sub_200: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(where_26, mul_898);  where_26 = mul_898 = None
    sub_201: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(sub_200, unsqueeze_693);  sub_200 = unsqueeze_693 = None
    mul_899: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_201, unsqueeze_699);  sub_201 = unsqueeze_699 = None
    mul_900: "f32[192]" = torch.ops.aten.mul.Tensor(sum_55, squeeze_202);  sum_55 = squeeze_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_26 = torch.ops.aten.convolution_backward.default(mul_899, relu_66, primals_256, [0], [1, 1], [3, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_899 = primals_256 = None
    getitem_274: "f32[8, 192, 17, 17]" = convolution_backward_26[0]
    getitem_275: "f32[192, 192, 7, 1]" = convolution_backward_26[1];  convolution_backward_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_176: "f32[8, 192, 17, 17]" = torch.ops.aten.alias.default(relu_66);  relu_66 = None
    alias_177: "f32[8, 192, 17, 17]" = torch.ops.aten.alias.default(alias_176);  alias_176 = None
    le_27: "b8[8, 192, 17, 17]" = torch.ops.aten.le.Scalar(alias_177, 0);  alias_177 = None
    where_27: "f32[8, 192, 17, 17]" = torch.ops.aten.where.self(le_27, full_default, getitem_274);  le_27 = getitem_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_56: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_27, [0, 2, 3])
    sub_202: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_66, unsqueeze_702);  convolution_66 = unsqueeze_702 = None
    mul_901: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(where_27, sub_202)
    sum_57: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_901, [0, 2, 3]);  mul_901 = None
    mul_902: "f32[192]" = torch.ops.aten.mul.Tensor(sum_56, 0.00043252595155709344)
    unsqueeze_703: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_902, 0);  mul_902 = None
    unsqueeze_704: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_703, 2);  unsqueeze_703 = None
    unsqueeze_705: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_704, 3);  unsqueeze_704 = None
    mul_903: "f32[192]" = torch.ops.aten.mul.Tensor(sum_57, 0.00043252595155709344)
    mul_904: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_199, squeeze_199)
    mul_905: "f32[192]" = torch.ops.aten.mul.Tensor(mul_903, mul_904);  mul_903 = mul_904 = None
    unsqueeze_706: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_905, 0);  mul_905 = None
    unsqueeze_707: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_706, 2);  unsqueeze_706 = None
    unsqueeze_708: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_707, 3);  unsqueeze_707 = None
    mul_906: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_199, primals_133);  primals_133 = None
    unsqueeze_709: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_906, 0);  mul_906 = None
    unsqueeze_710: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_709, 2);  unsqueeze_709 = None
    unsqueeze_711: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_710, 3);  unsqueeze_710 = None
    mul_907: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_202, unsqueeze_708);  sub_202 = unsqueeze_708 = None
    sub_204: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(where_27, mul_907);  where_27 = mul_907 = None
    sub_205: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(sub_204, unsqueeze_705);  sub_204 = unsqueeze_705 = None
    mul_908: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_205, unsqueeze_711);  sub_205 = unsqueeze_711 = None
    mul_909: "f32[192]" = torch.ops.aten.mul.Tensor(sum_57, squeeze_199);  sum_57 = squeeze_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_27 = torch.ops.aten.convolution_backward.default(mul_908, relu_65, primals_255, [0], [1, 1], [0, 3], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_908 = primals_255 = None
    getitem_277: "f32[8, 192, 17, 17]" = convolution_backward_27[0]
    getitem_278: "f32[192, 192, 1, 7]" = convolution_backward_27[1];  convolution_backward_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_179: "f32[8, 192, 17, 17]" = torch.ops.aten.alias.default(relu_65);  relu_65 = None
    alias_180: "f32[8, 192, 17, 17]" = torch.ops.aten.alias.default(alias_179);  alias_179 = None
    le_28: "b8[8, 192, 17, 17]" = torch.ops.aten.le.Scalar(alias_180, 0);  alias_180 = None
    where_28: "f32[8, 192, 17, 17]" = torch.ops.aten.where.self(le_28, full_default, getitem_277);  le_28 = getitem_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_58: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_28, [0, 2, 3])
    sub_206: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_65, unsqueeze_714);  convolution_65 = unsqueeze_714 = None
    mul_910: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(where_28, sub_206)
    sum_59: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_910, [0, 2, 3]);  mul_910 = None
    mul_911: "f32[192]" = torch.ops.aten.mul.Tensor(sum_58, 0.00043252595155709344)
    unsqueeze_715: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_911, 0);  mul_911 = None
    unsqueeze_716: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_715, 2);  unsqueeze_715 = None
    unsqueeze_717: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_716, 3);  unsqueeze_716 = None
    mul_912: "f32[192]" = torch.ops.aten.mul.Tensor(sum_59, 0.00043252595155709344)
    mul_913: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_196, squeeze_196)
    mul_914: "f32[192]" = torch.ops.aten.mul.Tensor(mul_912, mul_913);  mul_912 = mul_913 = None
    unsqueeze_718: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_914, 0);  mul_914 = None
    unsqueeze_719: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_718, 2);  unsqueeze_718 = None
    unsqueeze_720: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_719, 3);  unsqueeze_719 = None
    mul_915: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_196, primals_131);  primals_131 = None
    unsqueeze_721: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_915, 0);  mul_915 = None
    unsqueeze_722: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_721, 2);  unsqueeze_721 = None
    unsqueeze_723: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_722, 3);  unsqueeze_722 = None
    mul_916: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_206, unsqueeze_720);  sub_206 = unsqueeze_720 = None
    sub_208: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(where_28, mul_916);  where_28 = mul_916 = None
    sub_209: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(sub_208, unsqueeze_717);  sub_208 = unsqueeze_717 = None
    mul_917: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_209, unsqueeze_723);  sub_209 = unsqueeze_723 = None
    mul_918: "f32[192]" = torch.ops.aten.mul.Tensor(sum_59, squeeze_196);  sum_59 = squeeze_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_28 = torch.ops.aten.convolution_backward.default(mul_917, relu_64, primals_254, [0], [1, 1], [3, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_917 = primals_254 = None
    getitem_280: "f32[8, 192, 17, 17]" = convolution_backward_28[0]
    getitem_281: "f32[192, 192, 7, 1]" = convolution_backward_28[1];  convolution_backward_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_182: "f32[8, 192, 17, 17]" = torch.ops.aten.alias.default(relu_64);  relu_64 = None
    alias_183: "f32[8, 192, 17, 17]" = torch.ops.aten.alias.default(alias_182);  alias_182 = None
    le_29: "b8[8, 192, 17, 17]" = torch.ops.aten.le.Scalar(alias_183, 0);  alias_183 = None
    where_29: "f32[8, 192, 17, 17]" = torch.ops.aten.where.self(le_29, full_default, getitem_280);  le_29 = getitem_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_60: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_29, [0, 2, 3])
    sub_210: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_64, unsqueeze_726);  convolution_64 = unsqueeze_726 = None
    mul_919: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(where_29, sub_210)
    sum_61: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_919, [0, 2, 3]);  mul_919 = None
    mul_920: "f32[192]" = torch.ops.aten.mul.Tensor(sum_60, 0.00043252595155709344)
    unsqueeze_727: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_920, 0);  mul_920 = None
    unsqueeze_728: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_727, 2);  unsqueeze_727 = None
    unsqueeze_729: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_728, 3);  unsqueeze_728 = None
    mul_921: "f32[192]" = torch.ops.aten.mul.Tensor(sum_61, 0.00043252595155709344)
    mul_922: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_193, squeeze_193)
    mul_923: "f32[192]" = torch.ops.aten.mul.Tensor(mul_921, mul_922);  mul_921 = mul_922 = None
    unsqueeze_730: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_923, 0);  mul_923 = None
    unsqueeze_731: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_730, 2);  unsqueeze_730 = None
    unsqueeze_732: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_731, 3);  unsqueeze_731 = None
    mul_924: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_193, primals_129);  primals_129 = None
    unsqueeze_733: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_924, 0);  mul_924 = None
    unsqueeze_734: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_733, 2);  unsqueeze_733 = None
    unsqueeze_735: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_734, 3);  unsqueeze_734 = None
    mul_925: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_210, unsqueeze_732);  sub_210 = unsqueeze_732 = None
    sub_212: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(where_29, mul_925);  where_29 = mul_925 = None
    sub_213: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(sub_212, unsqueeze_729);  sub_212 = unsqueeze_729 = None
    mul_926: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_213, unsqueeze_735);  sub_213 = unsqueeze_735 = None
    mul_927: "f32[192]" = torch.ops.aten.mul.Tensor(sum_61, squeeze_193);  sum_61 = squeeze_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_29 = torch.ops.aten.convolution_backward.default(mul_926, cat_6, primals_253, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_926 = primals_253 = None
    getitem_283: "f32[8, 768, 17, 17]" = convolution_backward_29[0]
    getitem_284: "f32[192, 768, 1, 1]" = convolution_backward_29[1];  convolution_backward_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_482: "f32[8, 768, 17, 17]" = torch.ops.aten.add.Tensor(avg_pool2d_backward_2, getitem_283);  avg_pool2d_backward_2 = getitem_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    where_30: "f32[8, 192, 17, 17]" = torch.ops.aten.where.self(le_30, full_default, slice_21);  le_30 = slice_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_62: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_30, [0, 2, 3])
    sub_214: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_63, unsqueeze_738);  convolution_63 = unsqueeze_738 = None
    mul_928: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(where_30, sub_214)
    sum_63: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_928, [0, 2, 3]);  mul_928 = None
    mul_929: "f32[192]" = torch.ops.aten.mul.Tensor(sum_62, 0.00043252595155709344)
    unsqueeze_739: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_929, 0);  mul_929 = None
    unsqueeze_740: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_739, 2);  unsqueeze_739 = None
    unsqueeze_741: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_740, 3);  unsqueeze_740 = None
    mul_930: "f32[192]" = torch.ops.aten.mul.Tensor(sum_63, 0.00043252595155709344)
    mul_931: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_190, squeeze_190)
    mul_932: "f32[192]" = torch.ops.aten.mul.Tensor(mul_930, mul_931);  mul_930 = mul_931 = None
    unsqueeze_742: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_932, 0);  mul_932 = None
    unsqueeze_743: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_742, 2);  unsqueeze_742 = None
    unsqueeze_744: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_743, 3);  unsqueeze_743 = None
    mul_933: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_190, primals_127);  primals_127 = None
    unsqueeze_745: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_933, 0);  mul_933 = None
    unsqueeze_746: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_745, 2);  unsqueeze_745 = None
    unsqueeze_747: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_746, 3);  unsqueeze_746 = None
    mul_934: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_214, unsqueeze_744);  sub_214 = unsqueeze_744 = None
    sub_216: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(where_30, mul_934);  where_30 = mul_934 = None
    sub_217: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(sub_216, unsqueeze_741);  sub_216 = unsqueeze_741 = None
    mul_935: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_217, unsqueeze_747);  sub_217 = unsqueeze_747 = None
    mul_936: "f32[192]" = torch.ops.aten.mul.Tensor(sum_63, squeeze_190);  sum_63 = squeeze_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_30 = torch.ops.aten.convolution_backward.default(mul_935, relu_62, primals_252, [0], [1, 1], [3, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_935 = primals_252 = None
    getitem_286: "f32[8, 192, 17, 17]" = convolution_backward_30[0]
    getitem_287: "f32[192, 192, 7, 1]" = convolution_backward_30[1];  convolution_backward_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_188: "f32[8, 192, 17, 17]" = torch.ops.aten.alias.default(relu_62);  relu_62 = None
    alias_189: "f32[8, 192, 17, 17]" = torch.ops.aten.alias.default(alias_188);  alias_188 = None
    le_31: "b8[8, 192, 17, 17]" = torch.ops.aten.le.Scalar(alias_189, 0);  alias_189 = None
    where_31: "f32[8, 192, 17, 17]" = torch.ops.aten.where.self(le_31, full_default, getitem_286);  le_31 = getitem_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_64: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_31, [0, 2, 3])
    sub_218: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_62, unsqueeze_750);  convolution_62 = unsqueeze_750 = None
    mul_937: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(where_31, sub_218)
    sum_65: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_937, [0, 2, 3]);  mul_937 = None
    mul_938: "f32[192]" = torch.ops.aten.mul.Tensor(sum_64, 0.00043252595155709344)
    unsqueeze_751: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_938, 0);  mul_938 = None
    unsqueeze_752: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_751, 2);  unsqueeze_751 = None
    unsqueeze_753: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_752, 3);  unsqueeze_752 = None
    mul_939: "f32[192]" = torch.ops.aten.mul.Tensor(sum_65, 0.00043252595155709344)
    mul_940: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_187, squeeze_187)
    mul_941: "f32[192]" = torch.ops.aten.mul.Tensor(mul_939, mul_940);  mul_939 = mul_940 = None
    unsqueeze_754: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_941, 0);  mul_941 = None
    unsqueeze_755: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_754, 2);  unsqueeze_754 = None
    unsqueeze_756: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_755, 3);  unsqueeze_755 = None
    mul_942: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_187, primals_125);  primals_125 = None
    unsqueeze_757: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_942, 0);  mul_942 = None
    unsqueeze_758: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_757, 2);  unsqueeze_757 = None
    unsqueeze_759: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_758, 3);  unsqueeze_758 = None
    mul_943: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_218, unsqueeze_756);  sub_218 = unsqueeze_756 = None
    sub_220: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(where_31, mul_943);  where_31 = mul_943 = None
    sub_221: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(sub_220, unsqueeze_753);  sub_220 = unsqueeze_753 = None
    mul_944: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_221, unsqueeze_759);  sub_221 = unsqueeze_759 = None
    mul_945: "f32[192]" = torch.ops.aten.mul.Tensor(sum_65, squeeze_187);  sum_65 = squeeze_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_31 = torch.ops.aten.convolution_backward.default(mul_944, relu_61, primals_251, [0], [1, 1], [0, 3], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_944 = primals_251 = None
    getitem_289: "f32[8, 192, 17, 17]" = convolution_backward_31[0]
    getitem_290: "f32[192, 192, 1, 7]" = convolution_backward_31[1];  convolution_backward_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_191: "f32[8, 192, 17, 17]" = torch.ops.aten.alias.default(relu_61);  relu_61 = None
    alias_192: "f32[8, 192, 17, 17]" = torch.ops.aten.alias.default(alias_191);  alias_191 = None
    le_32: "b8[8, 192, 17, 17]" = torch.ops.aten.le.Scalar(alias_192, 0);  alias_192 = None
    where_32: "f32[8, 192, 17, 17]" = torch.ops.aten.where.self(le_32, full_default, getitem_289);  le_32 = getitem_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_66: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_32, [0, 2, 3])
    sub_222: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_61, unsqueeze_762);  convolution_61 = unsqueeze_762 = None
    mul_946: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(where_32, sub_222)
    sum_67: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_946, [0, 2, 3]);  mul_946 = None
    mul_947: "f32[192]" = torch.ops.aten.mul.Tensor(sum_66, 0.00043252595155709344)
    unsqueeze_763: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_947, 0);  mul_947 = None
    unsqueeze_764: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_763, 2);  unsqueeze_763 = None
    unsqueeze_765: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_764, 3);  unsqueeze_764 = None
    mul_948: "f32[192]" = torch.ops.aten.mul.Tensor(sum_67, 0.00043252595155709344)
    mul_949: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_184, squeeze_184)
    mul_950: "f32[192]" = torch.ops.aten.mul.Tensor(mul_948, mul_949);  mul_948 = mul_949 = None
    unsqueeze_766: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_950, 0);  mul_950 = None
    unsqueeze_767: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_766, 2);  unsqueeze_766 = None
    unsqueeze_768: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_767, 3);  unsqueeze_767 = None
    mul_951: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_184, primals_123);  primals_123 = None
    unsqueeze_769: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_951, 0);  mul_951 = None
    unsqueeze_770: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_769, 2);  unsqueeze_769 = None
    unsqueeze_771: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_770, 3);  unsqueeze_770 = None
    mul_952: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_222, unsqueeze_768);  sub_222 = unsqueeze_768 = None
    sub_224: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(where_32, mul_952);  where_32 = mul_952 = None
    sub_225: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(sub_224, unsqueeze_765);  sub_224 = unsqueeze_765 = None
    mul_953: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_225, unsqueeze_771);  sub_225 = unsqueeze_771 = None
    mul_954: "f32[192]" = torch.ops.aten.mul.Tensor(sum_67, squeeze_184);  sum_67 = squeeze_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_32 = torch.ops.aten.convolution_backward.default(mul_953, cat_6, primals_250, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_953 = primals_250 = None
    getitem_292: "f32[8, 768, 17, 17]" = convolution_backward_32[0]
    getitem_293: "f32[192, 768, 1, 1]" = convolution_backward_32[1];  convolution_backward_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_483: "f32[8, 768, 17, 17]" = torch.ops.aten.add.Tensor(add_482, getitem_292);  add_482 = getitem_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    where_33: "f32[8, 192, 17, 17]" = torch.ops.aten.where.self(le_33, full_default, slice_20);  le_33 = slice_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_68: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_33, [0, 2, 3])
    sub_226: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_60, unsqueeze_774);  convolution_60 = unsqueeze_774 = None
    mul_955: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(where_33, sub_226)
    sum_69: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_955, [0, 2, 3]);  mul_955 = None
    mul_956: "f32[192]" = torch.ops.aten.mul.Tensor(sum_68, 0.00043252595155709344)
    unsqueeze_775: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_956, 0);  mul_956 = None
    unsqueeze_776: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_775, 2);  unsqueeze_775 = None
    unsqueeze_777: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_776, 3);  unsqueeze_776 = None
    mul_957: "f32[192]" = torch.ops.aten.mul.Tensor(sum_69, 0.00043252595155709344)
    mul_958: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_181, squeeze_181)
    mul_959: "f32[192]" = torch.ops.aten.mul.Tensor(mul_957, mul_958);  mul_957 = mul_958 = None
    unsqueeze_778: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_959, 0);  mul_959 = None
    unsqueeze_779: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_778, 2);  unsqueeze_778 = None
    unsqueeze_780: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_779, 3);  unsqueeze_779 = None
    mul_960: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_181, primals_121);  primals_121 = None
    unsqueeze_781: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_960, 0);  mul_960 = None
    unsqueeze_782: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_781, 2);  unsqueeze_781 = None
    unsqueeze_783: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_782, 3);  unsqueeze_782 = None
    mul_961: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_226, unsqueeze_780);  sub_226 = unsqueeze_780 = None
    sub_228: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(where_33, mul_961);  where_33 = mul_961 = None
    sub_229: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(sub_228, unsqueeze_777);  sub_228 = unsqueeze_777 = None
    mul_962: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_229, unsqueeze_783);  sub_229 = unsqueeze_783 = None
    mul_963: "f32[192]" = torch.ops.aten.mul.Tensor(sum_69, squeeze_181);  sum_69 = squeeze_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_33 = torch.ops.aten.convolution_backward.default(mul_962, cat_6, primals_249, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_962 = cat_6 = primals_249 = None
    getitem_295: "f32[8, 768, 17, 17]" = convolution_backward_33[0]
    getitem_296: "f32[192, 768, 1, 1]" = convolution_backward_33[1];  convolution_backward_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_484: "f32[8, 768, 17, 17]" = torch.ops.aten.add.Tensor(add_483, getitem_295);  add_483 = getitem_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/inception_v3.py:128, code: return torch.cat(outputs, 1)
    slice_24: "f32[8, 192, 17, 17]" = torch.ops.aten.slice.Tensor(add_484, 1, 0, 192)
    slice_25: "f32[8, 192, 17, 17]" = torch.ops.aten.slice.Tensor(add_484, 1, 192, 384)
    slice_26: "f32[8, 192, 17, 17]" = torch.ops.aten.slice.Tensor(add_484, 1, 384, 576)
    slice_27: "f32[8, 192, 17, 17]" = torch.ops.aten.slice.Tensor(add_484, 1, 576, 768);  add_484 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    where_34: "f32[8, 192, 17, 17]" = torch.ops.aten.where.self(le_34, full_default, slice_27);  le_34 = slice_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_70: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_34, [0, 2, 3])
    sub_230: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_59, unsqueeze_786);  convolution_59 = unsqueeze_786 = None
    mul_964: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(where_34, sub_230)
    sum_71: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_964, [0, 2, 3]);  mul_964 = None
    mul_965: "f32[192]" = torch.ops.aten.mul.Tensor(sum_70, 0.00043252595155709344)
    unsqueeze_787: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_965, 0);  mul_965 = None
    unsqueeze_788: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_787, 2);  unsqueeze_787 = None
    unsqueeze_789: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_788, 3);  unsqueeze_788 = None
    mul_966: "f32[192]" = torch.ops.aten.mul.Tensor(sum_71, 0.00043252595155709344)
    mul_967: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_178, squeeze_178)
    mul_968: "f32[192]" = torch.ops.aten.mul.Tensor(mul_966, mul_967);  mul_966 = mul_967 = None
    unsqueeze_790: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_968, 0);  mul_968 = None
    unsqueeze_791: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_790, 2);  unsqueeze_790 = None
    unsqueeze_792: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_791, 3);  unsqueeze_791 = None
    mul_969: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_178, primals_119);  primals_119 = None
    unsqueeze_793: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_969, 0);  mul_969 = None
    unsqueeze_794: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_793, 2);  unsqueeze_793 = None
    unsqueeze_795: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_794, 3);  unsqueeze_794 = None
    mul_970: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_230, unsqueeze_792);  sub_230 = unsqueeze_792 = None
    sub_232: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(where_34, mul_970);  where_34 = mul_970 = None
    sub_233: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(sub_232, unsqueeze_789);  sub_232 = unsqueeze_789 = None
    mul_971: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_233, unsqueeze_795);  sub_233 = unsqueeze_795 = None
    mul_972: "f32[192]" = torch.ops.aten.mul.Tensor(sum_71, squeeze_178);  sum_71 = squeeze_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_34 = torch.ops.aten.convolution_backward.default(mul_971, avg_pool2d_5, primals_248, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_971 = avg_pool2d_5 = primals_248 = None
    getitem_298: "f32[8, 768, 17, 17]" = convolution_backward_34[0]
    getitem_299: "f32[192, 768, 1, 1]" = convolution_backward_34[1];  convolution_backward_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/inception_v3.py:120, code: branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    avg_pool2d_backward_3: "f32[8, 768, 17, 17]" = torch.ops.aten.avg_pool2d_backward.default(getitem_298, cat_5, [3, 3], [1, 1], [1, 1], False, True, None);  getitem_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    where_35: "f32[8, 192, 17, 17]" = torch.ops.aten.where.self(le_35, full_default, slice_26);  le_35 = slice_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_72: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_35, [0, 2, 3])
    sub_234: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_58, unsqueeze_798);  convolution_58 = unsqueeze_798 = None
    mul_973: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(where_35, sub_234)
    sum_73: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_973, [0, 2, 3]);  mul_973 = None
    mul_974: "f32[192]" = torch.ops.aten.mul.Tensor(sum_72, 0.00043252595155709344)
    unsqueeze_799: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_974, 0);  mul_974 = None
    unsqueeze_800: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_799, 2);  unsqueeze_799 = None
    unsqueeze_801: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_800, 3);  unsqueeze_800 = None
    mul_975: "f32[192]" = torch.ops.aten.mul.Tensor(sum_73, 0.00043252595155709344)
    mul_976: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_175, squeeze_175)
    mul_977: "f32[192]" = torch.ops.aten.mul.Tensor(mul_975, mul_976);  mul_975 = mul_976 = None
    unsqueeze_802: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_977, 0);  mul_977 = None
    unsqueeze_803: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_802, 2);  unsqueeze_802 = None
    unsqueeze_804: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_803, 3);  unsqueeze_803 = None
    mul_978: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_175, primals_117);  primals_117 = None
    unsqueeze_805: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_978, 0);  mul_978 = None
    unsqueeze_806: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_805, 2);  unsqueeze_805 = None
    unsqueeze_807: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_806, 3);  unsqueeze_806 = None
    mul_979: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_234, unsqueeze_804);  sub_234 = unsqueeze_804 = None
    sub_236: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(where_35, mul_979);  where_35 = mul_979 = None
    sub_237: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(sub_236, unsqueeze_801);  sub_236 = unsqueeze_801 = None
    mul_980: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_237, unsqueeze_807);  sub_237 = unsqueeze_807 = None
    mul_981: "f32[192]" = torch.ops.aten.mul.Tensor(sum_73, squeeze_175);  sum_73 = squeeze_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_35 = torch.ops.aten.convolution_backward.default(mul_980, relu_57, primals_247, [0], [1, 1], [0, 3], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_980 = primals_247 = None
    getitem_301: "f32[8, 160, 17, 17]" = convolution_backward_35[0]
    getitem_302: "f32[192, 160, 1, 7]" = convolution_backward_35[1];  convolution_backward_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_203: "f32[8, 160, 17, 17]" = torch.ops.aten.alias.default(relu_57);  relu_57 = None
    alias_204: "f32[8, 160, 17, 17]" = torch.ops.aten.alias.default(alias_203);  alias_203 = None
    le_36: "b8[8, 160, 17, 17]" = torch.ops.aten.le.Scalar(alias_204, 0);  alias_204 = None
    where_36: "f32[8, 160, 17, 17]" = torch.ops.aten.where.self(le_36, full_default, getitem_301);  le_36 = getitem_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_74: "f32[160]" = torch.ops.aten.sum.dim_IntList(where_36, [0, 2, 3])
    sub_238: "f32[8, 160, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_57, unsqueeze_810);  convolution_57 = unsqueeze_810 = None
    mul_982: "f32[8, 160, 17, 17]" = torch.ops.aten.mul.Tensor(where_36, sub_238)
    sum_75: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_982, [0, 2, 3]);  mul_982 = None
    mul_983: "f32[160]" = torch.ops.aten.mul.Tensor(sum_74, 0.00043252595155709344)
    unsqueeze_811: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_983, 0);  mul_983 = None
    unsqueeze_812: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_811, 2);  unsqueeze_811 = None
    unsqueeze_813: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_812, 3);  unsqueeze_812 = None
    mul_984: "f32[160]" = torch.ops.aten.mul.Tensor(sum_75, 0.00043252595155709344)
    mul_985: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_172, squeeze_172)
    mul_986: "f32[160]" = torch.ops.aten.mul.Tensor(mul_984, mul_985);  mul_984 = mul_985 = None
    unsqueeze_814: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_986, 0);  mul_986 = None
    unsqueeze_815: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_814, 2);  unsqueeze_814 = None
    unsqueeze_816: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_815, 3);  unsqueeze_815 = None
    mul_987: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_172, primals_115);  primals_115 = None
    unsqueeze_817: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_987, 0);  mul_987 = None
    unsqueeze_818: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_817, 2);  unsqueeze_817 = None
    unsqueeze_819: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_818, 3);  unsqueeze_818 = None
    mul_988: "f32[8, 160, 17, 17]" = torch.ops.aten.mul.Tensor(sub_238, unsqueeze_816);  sub_238 = unsqueeze_816 = None
    sub_240: "f32[8, 160, 17, 17]" = torch.ops.aten.sub.Tensor(where_36, mul_988);  where_36 = mul_988 = None
    sub_241: "f32[8, 160, 17, 17]" = torch.ops.aten.sub.Tensor(sub_240, unsqueeze_813);  sub_240 = unsqueeze_813 = None
    mul_989: "f32[8, 160, 17, 17]" = torch.ops.aten.mul.Tensor(sub_241, unsqueeze_819);  sub_241 = unsqueeze_819 = None
    mul_990: "f32[160]" = torch.ops.aten.mul.Tensor(sum_75, squeeze_172);  sum_75 = squeeze_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_36 = torch.ops.aten.convolution_backward.default(mul_989, relu_56, primals_246, [0], [1, 1], [3, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_989 = primals_246 = None
    getitem_304: "f32[8, 160, 17, 17]" = convolution_backward_36[0]
    getitem_305: "f32[160, 160, 7, 1]" = convolution_backward_36[1];  convolution_backward_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_206: "f32[8, 160, 17, 17]" = torch.ops.aten.alias.default(relu_56);  relu_56 = None
    alias_207: "f32[8, 160, 17, 17]" = torch.ops.aten.alias.default(alias_206);  alias_206 = None
    le_37: "b8[8, 160, 17, 17]" = torch.ops.aten.le.Scalar(alias_207, 0);  alias_207 = None
    where_37: "f32[8, 160, 17, 17]" = torch.ops.aten.where.self(le_37, full_default, getitem_304);  le_37 = getitem_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_76: "f32[160]" = torch.ops.aten.sum.dim_IntList(where_37, [0, 2, 3])
    sub_242: "f32[8, 160, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_56, unsqueeze_822);  convolution_56 = unsqueeze_822 = None
    mul_991: "f32[8, 160, 17, 17]" = torch.ops.aten.mul.Tensor(where_37, sub_242)
    sum_77: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_991, [0, 2, 3]);  mul_991 = None
    mul_992: "f32[160]" = torch.ops.aten.mul.Tensor(sum_76, 0.00043252595155709344)
    unsqueeze_823: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_992, 0);  mul_992 = None
    unsqueeze_824: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_823, 2);  unsqueeze_823 = None
    unsqueeze_825: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_824, 3);  unsqueeze_824 = None
    mul_993: "f32[160]" = torch.ops.aten.mul.Tensor(sum_77, 0.00043252595155709344)
    mul_994: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_169, squeeze_169)
    mul_995: "f32[160]" = torch.ops.aten.mul.Tensor(mul_993, mul_994);  mul_993 = mul_994 = None
    unsqueeze_826: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_995, 0);  mul_995 = None
    unsqueeze_827: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_826, 2);  unsqueeze_826 = None
    unsqueeze_828: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_827, 3);  unsqueeze_827 = None
    mul_996: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_169, primals_113);  primals_113 = None
    unsqueeze_829: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_996, 0);  mul_996 = None
    unsqueeze_830: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_829, 2);  unsqueeze_829 = None
    unsqueeze_831: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_830, 3);  unsqueeze_830 = None
    mul_997: "f32[8, 160, 17, 17]" = torch.ops.aten.mul.Tensor(sub_242, unsqueeze_828);  sub_242 = unsqueeze_828 = None
    sub_244: "f32[8, 160, 17, 17]" = torch.ops.aten.sub.Tensor(where_37, mul_997);  where_37 = mul_997 = None
    sub_245: "f32[8, 160, 17, 17]" = torch.ops.aten.sub.Tensor(sub_244, unsqueeze_825);  sub_244 = unsqueeze_825 = None
    mul_998: "f32[8, 160, 17, 17]" = torch.ops.aten.mul.Tensor(sub_245, unsqueeze_831);  sub_245 = unsqueeze_831 = None
    mul_999: "f32[160]" = torch.ops.aten.mul.Tensor(sum_77, squeeze_169);  sum_77 = squeeze_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_37 = torch.ops.aten.convolution_backward.default(mul_998, relu_55, primals_245, [0], [1, 1], [0, 3], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_998 = primals_245 = None
    getitem_307: "f32[8, 160, 17, 17]" = convolution_backward_37[0]
    getitem_308: "f32[160, 160, 1, 7]" = convolution_backward_37[1];  convolution_backward_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_209: "f32[8, 160, 17, 17]" = torch.ops.aten.alias.default(relu_55);  relu_55 = None
    alias_210: "f32[8, 160, 17, 17]" = torch.ops.aten.alias.default(alias_209);  alias_209 = None
    le_38: "b8[8, 160, 17, 17]" = torch.ops.aten.le.Scalar(alias_210, 0);  alias_210 = None
    where_38: "f32[8, 160, 17, 17]" = torch.ops.aten.where.self(le_38, full_default, getitem_307);  le_38 = getitem_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_78: "f32[160]" = torch.ops.aten.sum.dim_IntList(where_38, [0, 2, 3])
    sub_246: "f32[8, 160, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_55, unsqueeze_834);  convolution_55 = unsqueeze_834 = None
    mul_1000: "f32[8, 160, 17, 17]" = torch.ops.aten.mul.Tensor(where_38, sub_246)
    sum_79: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_1000, [0, 2, 3]);  mul_1000 = None
    mul_1001: "f32[160]" = torch.ops.aten.mul.Tensor(sum_78, 0.00043252595155709344)
    unsqueeze_835: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_1001, 0);  mul_1001 = None
    unsqueeze_836: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_835, 2);  unsqueeze_835 = None
    unsqueeze_837: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_836, 3);  unsqueeze_836 = None
    mul_1002: "f32[160]" = torch.ops.aten.mul.Tensor(sum_79, 0.00043252595155709344)
    mul_1003: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_166, squeeze_166)
    mul_1004: "f32[160]" = torch.ops.aten.mul.Tensor(mul_1002, mul_1003);  mul_1002 = mul_1003 = None
    unsqueeze_838: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_1004, 0);  mul_1004 = None
    unsqueeze_839: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_838, 2);  unsqueeze_838 = None
    unsqueeze_840: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_839, 3);  unsqueeze_839 = None
    mul_1005: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_166, primals_111);  primals_111 = None
    unsqueeze_841: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_1005, 0);  mul_1005 = None
    unsqueeze_842: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_841, 2);  unsqueeze_841 = None
    unsqueeze_843: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_842, 3);  unsqueeze_842 = None
    mul_1006: "f32[8, 160, 17, 17]" = torch.ops.aten.mul.Tensor(sub_246, unsqueeze_840);  sub_246 = unsqueeze_840 = None
    sub_248: "f32[8, 160, 17, 17]" = torch.ops.aten.sub.Tensor(where_38, mul_1006);  where_38 = mul_1006 = None
    sub_249: "f32[8, 160, 17, 17]" = torch.ops.aten.sub.Tensor(sub_248, unsqueeze_837);  sub_248 = unsqueeze_837 = None
    mul_1007: "f32[8, 160, 17, 17]" = torch.ops.aten.mul.Tensor(sub_249, unsqueeze_843);  sub_249 = unsqueeze_843 = None
    mul_1008: "f32[160]" = torch.ops.aten.mul.Tensor(sum_79, squeeze_166);  sum_79 = squeeze_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_38 = torch.ops.aten.convolution_backward.default(mul_1007, relu_54, primals_244, [0], [1, 1], [3, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1007 = primals_244 = None
    getitem_310: "f32[8, 160, 17, 17]" = convolution_backward_38[0]
    getitem_311: "f32[160, 160, 7, 1]" = convolution_backward_38[1];  convolution_backward_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_212: "f32[8, 160, 17, 17]" = torch.ops.aten.alias.default(relu_54);  relu_54 = None
    alias_213: "f32[8, 160, 17, 17]" = torch.ops.aten.alias.default(alias_212);  alias_212 = None
    le_39: "b8[8, 160, 17, 17]" = torch.ops.aten.le.Scalar(alias_213, 0);  alias_213 = None
    where_39: "f32[8, 160, 17, 17]" = torch.ops.aten.where.self(le_39, full_default, getitem_310);  le_39 = getitem_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_80: "f32[160]" = torch.ops.aten.sum.dim_IntList(where_39, [0, 2, 3])
    sub_250: "f32[8, 160, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_54, unsqueeze_846);  convolution_54 = unsqueeze_846 = None
    mul_1009: "f32[8, 160, 17, 17]" = torch.ops.aten.mul.Tensor(where_39, sub_250)
    sum_81: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_1009, [0, 2, 3]);  mul_1009 = None
    mul_1010: "f32[160]" = torch.ops.aten.mul.Tensor(sum_80, 0.00043252595155709344)
    unsqueeze_847: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_1010, 0);  mul_1010 = None
    unsqueeze_848: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_847, 2);  unsqueeze_847 = None
    unsqueeze_849: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_848, 3);  unsqueeze_848 = None
    mul_1011: "f32[160]" = torch.ops.aten.mul.Tensor(sum_81, 0.00043252595155709344)
    mul_1012: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_163, squeeze_163)
    mul_1013: "f32[160]" = torch.ops.aten.mul.Tensor(mul_1011, mul_1012);  mul_1011 = mul_1012 = None
    unsqueeze_850: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_1013, 0);  mul_1013 = None
    unsqueeze_851: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_850, 2);  unsqueeze_850 = None
    unsqueeze_852: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_851, 3);  unsqueeze_851 = None
    mul_1014: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_163, primals_109);  primals_109 = None
    unsqueeze_853: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_1014, 0);  mul_1014 = None
    unsqueeze_854: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_853, 2);  unsqueeze_853 = None
    unsqueeze_855: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_854, 3);  unsqueeze_854 = None
    mul_1015: "f32[8, 160, 17, 17]" = torch.ops.aten.mul.Tensor(sub_250, unsqueeze_852);  sub_250 = unsqueeze_852 = None
    sub_252: "f32[8, 160, 17, 17]" = torch.ops.aten.sub.Tensor(where_39, mul_1015);  where_39 = mul_1015 = None
    sub_253: "f32[8, 160, 17, 17]" = torch.ops.aten.sub.Tensor(sub_252, unsqueeze_849);  sub_252 = unsqueeze_849 = None
    mul_1016: "f32[8, 160, 17, 17]" = torch.ops.aten.mul.Tensor(sub_253, unsqueeze_855);  sub_253 = unsqueeze_855 = None
    mul_1017: "f32[160]" = torch.ops.aten.mul.Tensor(sum_81, squeeze_163);  sum_81 = squeeze_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_39 = torch.ops.aten.convolution_backward.default(mul_1016, cat_5, primals_243, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1016 = primals_243 = None
    getitem_313: "f32[8, 768, 17, 17]" = convolution_backward_39[0]
    getitem_314: "f32[160, 768, 1, 1]" = convolution_backward_39[1];  convolution_backward_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_485: "f32[8, 768, 17, 17]" = torch.ops.aten.add.Tensor(avg_pool2d_backward_3, getitem_313);  avg_pool2d_backward_3 = getitem_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    where_40: "f32[8, 192, 17, 17]" = torch.ops.aten.where.self(le_40, full_default, slice_25);  le_40 = slice_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_82: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_40, [0, 2, 3])
    sub_254: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_53, unsqueeze_858);  convolution_53 = unsqueeze_858 = None
    mul_1018: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(where_40, sub_254)
    sum_83: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_1018, [0, 2, 3]);  mul_1018 = None
    mul_1019: "f32[192]" = torch.ops.aten.mul.Tensor(sum_82, 0.00043252595155709344)
    unsqueeze_859: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_1019, 0);  mul_1019 = None
    unsqueeze_860: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_859, 2);  unsqueeze_859 = None
    unsqueeze_861: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_860, 3);  unsqueeze_860 = None
    mul_1020: "f32[192]" = torch.ops.aten.mul.Tensor(sum_83, 0.00043252595155709344)
    mul_1021: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_160, squeeze_160)
    mul_1022: "f32[192]" = torch.ops.aten.mul.Tensor(mul_1020, mul_1021);  mul_1020 = mul_1021 = None
    unsqueeze_862: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_1022, 0);  mul_1022 = None
    unsqueeze_863: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_862, 2);  unsqueeze_862 = None
    unsqueeze_864: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_863, 3);  unsqueeze_863 = None
    mul_1023: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_160, primals_107);  primals_107 = None
    unsqueeze_865: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_1023, 0);  mul_1023 = None
    unsqueeze_866: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_865, 2);  unsqueeze_865 = None
    unsqueeze_867: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_866, 3);  unsqueeze_866 = None
    mul_1024: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_254, unsqueeze_864);  sub_254 = unsqueeze_864 = None
    sub_256: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(where_40, mul_1024);  where_40 = mul_1024 = None
    sub_257: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(sub_256, unsqueeze_861);  sub_256 = unsqueeze_861 = None
    mul_1025: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_257, unsqueeze_867);  sub_257 = unsqueeze_867 = None
    mul_1026: "f32[192]" = torch.ops.aten.mul.Tensor(sum_83, squeeze_160);  sum_83 = squeeze_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_40 = torch.ops.aten.convolution_backward.default(mul_1025, relu_52, primals_242, [0], [1, 1], [3, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1025 = primals_242 = None
    getitem_316: "f32[8, 160, 17, 17]" = convolution_backward_40[0]
    getitem_317: "f32[192, 160, 7, 1]" = convolution_backward_40[1];  convolution_backward_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_218: "f32[8, 160, 17, 17]" = torch.ops.aten.alias.default(relu_52);  relu_52 = None
    alias_219: "f32[8, 160, 17, 17]" = torch.ops.aten.alias.default(alias_218);  alias_218 = None
    le_41: "b8[8, 160, 17, 17]" = torch.ops.aten.le.Scalar(alias_219, 0);  alias_219 = None
    where_41: "f32[8, 160, 17, 17]" = torch.ops.aten.where.self(le_41, full_default, getitem_316);  le_41 = getitem_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_84: "f32[160]" = torch.ops.aten.sum.dim_IntList(where_41, [0, 2, 3])
    sub_258: "f32[8, 160, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_52, unsqueeze_870);  convolution_52 = unsqueeze_870 = None
    mul_1027: "f32[8, 160, 17, 17]" = torch.ops.aten.mul.Tensor(where_41, sub_258)
    sum_85: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_1027, [0, 2, 3]);  mul_1027 = None
    mul_1028: "f32[160]" = torch.ops.aten.mul.Tensor(sum_84, 0.00043252595155709344)
    unsqueeze_871: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_1028, 0);  mul_1028 = None
    unsqueeze_872: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_871, 2);  unsqueeze_871 = None
    unsqueeze_873: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_872, 3);  unsqueeze_872 = None
    mul_1029: "f32[160]" = torch.ops.aten.mul.Tensor(sum_85, 0.00043252595155709344)
    mul_1030: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_157, squeeze_157)
    mul_1031: "f32[160]" = torch.ops.aten.mul.Tensor(mul_1029, mul_1030);  mul_1029 = mul_1030 = None
    unsqueeze_874: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_1031, 0);  mul_1031 = None
    unsqueeze_875: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_874, 2);  unsqueeze_874 = None
    unsqueeze_876: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_875, 3);  unsqueeze_875 = None
    mul_1032: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_157, primals_105);  primals_105 = None
    unsqueeze_877: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_1032, 0);  mul_1032 = None
    unsqueeze_878: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_877, 2);  unsqueeze_877 = None
    unsqueeze_879: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_878, 3);  unsqueeze_878 = None
    mul_1033: "f32[8, 160, 17, 17]" = torch.ops.aten.mul.Tensor(sub_258, unsqueeze_876);  sub_258 = unsqueeze_876 = None
    sub_260: "f32[8, 160, 17, 17]" = torch.ops.aten.sub.Tensor(where_41, mul_1033);  where_41 = mul_1033 = None
    sub_261: "f32[8, 160, 17, 17]" = torch.ops.aten.sub.Tensor(sub_260, unsqueeze_873);  sub_260 = unsqueeze_873 = None
    mul_1034: "f32[8, 160, 17, 17]" = torch.ops.aten.mul.Tensor(sub_261, unsqueeze_879);  sub_261 = unsqueeze_879 = None
    mul_1035: "f32[160]" = torch.ops.aten.mul.Tensor(sum_85, squeeze_157);  sum_85 = squeeze_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_41 = torch.ops.aten.convolution_backward.default(mul_1034, relu_51, primals_241, [0], [1, 1], [0, 3], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1034 = primals_241 = None
    getitem_319: "f32[8, 160, 17, 17]" = convolution_backward_41[0]
    getitem_320: "f32[160, 160, 1, 7]" = convolution_backward_41[1];  convolution_backward_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_221: "f32[8, 160, 17, 17]" = torch.ops.aten.alias.default(relu_51);  relu_51 = None
    alias_222: "f32[8, 160, 17, 17]" = torch.ops.aten.alias.default(alias_221);  alias_221 = None
    le_42: "b8[8, 160, 17, 17]" = torch.ops.aten.le.Scalar(alias_222, 0);  alias_222 = None
    where_42: "f32[8, 160, 17, 17]" = torch.ops.aten.where.self(le_42, full_default, getitem_319);  le_42 = getitem_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_86: "f32[160]" = torch.ops.aten.sum.dim_IntList(where_42, [0, 2, 3])
    sub_262: "f32[8, 160, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_882);  convolution_51 = unsqueeze_882 = None
    mul_1036: "f32[8, 160, 17, 17]" = torch.ops.aten.mul.Tensor(where_42, sub_262)
    sum_87: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_1036, [0, 2, 3]);  mul_1036 = None
    mul_1037: "f32[160]" = torch.ops.aten.mul.Tensor(sum_86, 0.00043252595155709344)
    unsqueeze_883: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_1037, 0);  mul_1037 = None
    unsqueeze_884: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_883, 2);  unsqueeze_883 = None
    unsqueeze_885: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_884, 3);  unsqueeze_884 = None
    mul_1038: "f32[160]" = torch.ops.aten.mul.Tensor(sum_87, 0.00043252595155709344)
    mul_1039: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_154, squeeze_154)
    mul_1040: "f32[160]" = torch.ops.aten.mul.Tensor(mul_1038, mul_1039);  mul_1038 = mul_1039 = None
    unsqueeze_886: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_1040, 0);  mul_1040 = None
    unsqueeze_887: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_886, 2);  unsqueeze_886 = None
    unsqueeze_888: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_887, 3);  unsqueeze_887 = None
    mul_1041: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_154, primals_103);  primals_103 = None
    unsqueeze_889: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_1041, 0);  mul_1041 = None
    unsqueeze_890: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_889, 2);  unsqueeze_889 = None
    unsqueeze_891: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_890, 3);  unsqueeze_890 = None
    mul_1042: "f32[8, 160, 17, 17]" = torch.ops.aten.mul.Tensor(sub_262, unsqueeze_888);  sub_262 = unsqueeze_888 = None
    sub_264: "f32[8, 160, 17, 17]" = torch.ops.aten.sub.Tensor(where_42, mul_1042);  where_42 = mul_1042 = None
    sub_265: "f32[8, 160, 17, 17]" = torch.ops.aten.sub.Tensor(sub_264, unsqueeze_885);  sub_264 = unsqueeze_885 = None
    mul_1043: "f32[8, 160, 17, 17]" = torch.ops.aten.mul.Tensor(sub_265, unsqueeze_891);  sub_265 = unsqueeze_891 = None
    mul_1044: "f32[160]" = torch.ops.aten.mul.Tensor(sum_87, squeeze_154);  sum_87 = squeeze_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_42 = torch.ops.aten.convolution_backward.default(mul_1043, cat_5, primals_240, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1043 = primals_240 = None
    getitem_322: "f32[8, 768, 17, 17]" = convolution_backward_42[0]
    getitem_323: "f32[160, 768, 1, 1]" = convolution_backward_42[1];  convolution_backward_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_486: "f32[8, 768, 17, 17]" = torch.ops.aten.add.Tensor(add_485, getitem_322);  add_485 = getitem_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    where_43: "f32[8, 192, 17, 17]" = torch.ops.aten.where.self(le_43, full_default, slice_24);  le_43 = slice_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_88: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_43, [0, 2, 3])
    sub_266: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_894);  convolution_50 = unsqueeze_894 = None
    mul_1045: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(where_43, sub_266)
    sum_89: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_1045, [0, 2, 3]);  mul_1045 = None
    mul_1046: "f32[192]" = torch.ops.aten.mul.Tensor(sum_88, 0.00043252595155709344)
    unsqueeze_895: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_1046, 0);  mul_1046 = None
    unsqueeze_896: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_895, 2);  unsqueeze_895 = None
    unsqueeze_897: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_896, 3);  unsqueeze_896 = None
    mul_1047: "f32[192]" = torch.ops.aten.mul.Tensor(sum_89, 0.00043252595155709344)
    mul_1048: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_151, squeeze_151)
    mul_1049: "f32[192]" = torch.ops.aten.mul.Tensor(mul_1047, mul_1048);  mul_1047 = mul_1048 = None
    unsqueeze_898: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_1049, 0);  mul_1049 = None
    unsqueeze_899: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_898, 2);  unsqueeze_898 = None
    unsqueeze_900: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_899, 3);  unsqueeze_899 = None
    mul_1050: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_151, primals_101);  primals_101 = None
    unsqueeze_901: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_1050, 0);  mul_1050 = None
    unsqueeze_902: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_901, 2);  unsqueeze_901 = None
    unsqueeze_903: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_902, 3);  unsqueeze_902 = None
    mul_1051: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_266, unsqueeze_900);  sub_266 = unsqueeze_900 = None
    sub_268: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(where_43, mul_1051);  where_43 = mul_1051 = None
    sub_269: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(sub_268, unsqueeze_897);  sub_268 = unsqueeze_897 = None
    mul_1052: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_269, unsqueeze_903);  sub_269 = unsqueeze_903 = None
    mul_1053: "f32[192]" = torch.ops.aten.mul.Tensor(sum_89, squeeze_151);  sum_89 = squeeze_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_43 = torch.ops.aten.convolution_backward.default(mul_1052, cat_5, primals_239, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1052 = cat_5 = primals_239 = None
    getitem_325: "f32[8, 768, 17, 17]" = convolution_backward_43[0]
    getitem_326: "f32[192, 768, 1, 1]" = convolution_backward_43[1];  convolution_backward_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_487: "f32[8, 768, 17, 17]" = torch.ops.aten.add.Tensor(add_486, getitem_325);  add_486 = getitem_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/inception_v3.py:128, code: return torch.cat(outputs, 1)
    slice_28: "f32[8, 192, 17, 17]" = torch.ops.aten.slice.Tensor(add_487, 1, 0, 192)
    slice_29: "f32[8, 192, 17, 17]" = torch.ops.aten.slice.Tensor(add_487, 1, 192, 384)
    slice_30: "f32[8, 192, 17, 17]" = torch.ops.aten.slice.Tensor(add_487, 1, 384, 576)
    slice_31: "f32[8, 192, 17, 17]" = torch.ops.aten.slice.Tensor(add_487, 1, 576, 768);  add_487 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    where_44: "f32[8, 192, 17, 17]" = torch.ops.aten.where.self(le_44, full_default, slice_31);  le_44 = slice_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_90: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_44, [0, 2, 3])
    sub_270: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_49, unsqueeze_906);  convolution_49 = unsqueeze_906 = None
    mul_1054: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(where_44, sub_270)
    sum_91: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_1054, [0, 2, 3]);  mul_1054 = None
    mul_1055: "f32[192]" = torch.ops.aten.mul.Tensor(sum_90, 0.00043252595155709344)
    unsqueeze_907: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_1055, 0);  mul_1055 = None
    unsqueeze_908: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_907, 2);  unsqueeze_907 = None
    unsqueeze_909: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_908, 3);  unsqueeze_908 = None
    mul_1056: "f32[192]" = torch.ops.aten.mul.Tensor(sum_91, 0.00043252595155709344)
    mul_1057: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_148, squeeze_148)
    mul_1058: "f32[192]" = torch.ops.aten.mul.Tensor(mul_1056, mul_1057);  mul_1056 = mul_1057 = None
    unsqueeze_910: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_1058, 0);  mul_1058 = None
    unsqueeze_911: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_910, 2);  unsqueeze_910 = None
    unsqueeze_912: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_911, 3);  unsqueeze_911 = None
    mul_1059: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_148, primals_99);  primals_99 = None
    unsqueeze_913: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_1059, 0);  mul_1059 = None
    unsqueeze_914: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_913, 2);  unsqueeze_913 = None
    unsqueeze_915: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_914, 3);  unsqueeze_914 = None
    mul_1060: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_270, unsqueeze_912);  sub_270 = unsqueeze_912 = None
    sub_272: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(where_44, mul_1060);  where_44 = mul_1060 = None
    sub_273: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(sub_272, unsqueeze_909);  sub_272 = unsqueeze_909 = None
    mul_1061: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_273, unsqueeze_915);  sub_273 = unsqueeze_915 = None
    mul_1062: "f32[192]" = torch.ops.aten.mul.Tensor(sum_91, squeeze_148);  sum_91 = squeeze_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_44 = torch.ops.aten.convolution_backward.default(mul_1061, avg_pool2d_4, primals_238, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1061 = avg_pool2d_4 = primals_238 = None
    getitem_328: "f32[8, 768, 17, 17]" = convolution_backward_44[0]
    getitem_329: "f32[192, 768, 1, 1]" = convolution_backward_44[1];  convolution_backward_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/inception_v3.py:120, code: branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    avg_pool2d_backward_4: "f32[8, 768, 17, 17]" = torch.ops.aten.avg_pool2d_backward.default(getitem_328, cat_4, [3, 3], [1, 1], [1, 1], False, True, None);  getitem_328 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    where_45: "f32[8, 192, 17, 17]" = torch.ops.aten.where.self(le_45, full_default, slice_30);  le_45 = slice_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_92: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_45, [0, 2, 3])
    sub_274: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_48, unsqueeze_918);  convolution_48 = unsqueeze_918 = None
    mul_1063: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(where_45, sub_274)
    sum_93: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_1063, [0, 2, 3]);  mul_1063 = None
    mul_1064: "f32[192]" = torch.ops.aten.mul.Tensor(sum_92, 0.00043252595155709344)
    unsqueeze_919: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_1064, 0);  mul_1064 = None
    unsqueeze_920: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_919, 2);  unsqueeze_919 = None
    unsqueeze_921: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_920, 3);  unsqueeze_920 = None
    mul_1065: "f32[192]" = torch.ops.aten.mul.Tensor(sum_93, 0.00043252595155709344)
    mul_1066: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_145, squeeze_145)
    mul_1067: "f32[192]" = torch.ops.aten.mul.Tensor(mul_1065, mul_1066);  mul_1065 = mul_1066 = None
    unsqueeze_922: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_1067, 0);  mul_1067 = None
    unsqueeze_923: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_922, 2);  unsqueeze_922 = None
    unsqueeze_924: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_923, 3);  unsqueeze_923 = None
    mul_1068: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_145, primals_97);  primals_97 = None
    unsqueeze_925: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_1068, 0);  mul_1068 = None
    unsqueeze_926: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_925, 2);  unsqueeze_925 = None
    unsqueeze_927: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_926, 3);  unsqueeze_926 = None
    mul_1069: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_274, unsqueeze_924);  sub_274 = unsqueeze_924 = None
    sub_276: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(where_45, mul_1069);  where_45 = mul_1069 = None
    sub_277: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(sub_276, unsqueeze_921);  sub_276 = unsqueeze_921 = None
    mul_1070: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_277, unsqueeze_927);  sub_277 = unsqueeze_927 = None
    mul_1071: "f32[192]" = torch.ops.aten.mul.Tensor(sum_93, squeeze_145);  sum_93 = squeeze_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_45 = torch.ops.aten.convolution_backward.default(mul_1070, relu_47, primals_237, [0], [1, 1], [0, 3], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1070 = primals_237 = None
    getitem_331: "f32[8, 160, 17, 17]" = convolution_backward_45[0]
    getitem_332: "f32[192, 160, 1, 7]" = convolution_backward_45[1];  convolution_backward_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_233: "f32[8, 160, 17, 17]" = torch.ops.aten.alias.default(relu_47);  relu_47 = None
    alias_234: "f32[8, 160, 17, 17]" = torch.ops.aten.alias.default(alias_233);  alias_233 = None
    le_46: "b8[8, 160, 17, 17]" = torch.ops.aten.le.Scalar(alias_234, 0);  alias_234 = None
    where_46: "f32[8, 160, 17, 17]" = torch.ops.aten.where.self(le_46, full_default, getitem_331);  le_46 = getitem_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_94: "f32[160]" = torch.ops.aten.sum.dim_IntList(where_46, [0, 2, 3])
    sub_278: "f32[8, 160, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_47, unsqueeze_930);  convolution_47 = unsqueeze_930 = None
    mul_1072: "f32[8, 160, 17, 17]" = torch.ops.aten.mul.Tensor(where_46, sub_278)
    sum_95: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_1072, [0, 2, 3]);  mul_1072 = None
    mul_1073: "f32[160]" = torch.ops.aten.mul.Tensor(sum_94, 0.00043252595155709344)
    unsqueeze_931: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_1073, 0);  mul_1073 = None
    unsqueeze_932: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_931, 2);  unsqueeze_931 = None
    unsqueeze_933: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_932, 3);  unsqueeze_932 = None
    mul_1074: "f32[160]" = torch.ops.aten.mul.Tensor(sum_95, 0.00043252595155709344)
    mul_1075: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_142, squeeze_142)
    mul_1076: "f32[160]" = torch.ops.aten.mul.Tensor(mul_1074, mul_1075);  mul_1074 = mul_1075 = None
    unsqueeze_934: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_1076, 0);  mul_1076 = None
    unsqueeze_935: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_934, 2);  unsqueeze_934 = None
    unsqueeze_936: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_935, 3);  unsqueeze_935 = None
    mul_1077: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_142, primals_95);  primals_95 = None
    unsqueeze_937: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_1077, 0);  mul_1077 = None
    unsqueeze_938: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_937, 2);  unsqueeze_937 = None
    unsqueeze_939: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_938, 3);  unsqueeze_938 = None
    mul_1078: "f32[8, 160, 17, 17]" = torch.ops.aten.mul.Tensor(sub_278, unsqueeze_936);  sub_278 = unsqueeze_936 = None
    sub_280: "f32[8, 160, 17, 17]" = torch.ops.aten.sub.Tensor(where_46, mul_1078);  where_46 = mul_1078 = None
    sub_281: "f32[8, 160, 17, 17]" = torch.ops.aten.sub.Tensor(sub_280, unsqueeze_933);  sub_280 = unsqueeze_933 = None
    mul_1079: "f32[8, 160, 17, 17]" = torch.ops.aten.mul.Tensor(sub_281, unsqueeze_939);  sub_281 = unsqueeze_939 = None
    mul_1080: "f32[160]" = torch.ops.aten.mul.Tensor(sum_95, squeeze_142);  sum_95 = squeeze_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_46 = torch.ops.aten.convolution_backward.default(mul_1079, relu_46, primals_236, [0], [1, 1], [3, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1079 = primals_236 = None
    getitem_334: "f32[8, 160, 17, 17]" = convolution_backward_46[0]
    getitem_335: "f32[160, 160, 7, 1]" = convolution_backward_46[1];  convolution_backward_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_236: "f32[8, 160, 17, 17]" = torch.ops.aten.alias.default(relu_46);  relu_46 = None
    alias_237: "f32[8, 160, 17, 17]" = torch.ops.aten.alias.default(alias_236);  alias_236 = None
    le_47: "b8[8, 160, 17, 17]" = torch.ops.aten.le.Scalar(alias_237, 0);  alias_237 = None
    where_47: "f32[8, 160, 17, 17]" = torch.ops.aten.where.self(le_47, full_default, getitem_334);  le_47 = getitem_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_96: "f32[160]" = torch.ops.aten.sum.dim_IntList(where_47, [0, 2, 3])
    sub_282: "f32[8, 160, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_942);  convolution_46 = unsqueeze_942 = None
    mul_1081: "f32[8, 160, 17, 17]" = torch.ops.aten.mul.Tensor(where_47, sub_282)
    sum_97: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_1081, [0, 2, 3]);  mul_1081 = None
    mul_1082: "f32[160]" = torch.ops.aten.mul.Tensor(sum_96, 0.00043252595155709344)
    unsqueeze_943: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_1082, 0);  mul_1082 = None
    unsqueeze_944: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_943, 2);  unsqueeze_943 = None
    unsqueeze_945: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_944, 3);  unsqueeze_944 = None
    mul_1083: "f32[160]" = torch.ops.aten.mul.Tensor(sum_97, 0.00043252595155709344)
    mul_1084: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_139, squeeze_139)
    mul_1085: "f32[160]" = torch.ops.aten.mul.Tensor(mul_1083, mul_1084);  mul_1083 = mul_1084 = None
    unsqueeze_946: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_1085, 0);  mul_1085 = None
    unsqueeze_947: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_946, 2);  unsqueeze_946 = None
    unsqueeze_948: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_947, 3);  unsqueeze_947 = None
    mul_1086: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_139, primals_93);  primals_93 = None
    unsqueeze_949: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_1086, 0);  mul_1086 = None
    unsqueeze_950: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_949, 2);  unsqueeze_949 = None
    unsqueeze_951: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_950, 3);  unsqueeze_950 = None
    mul_1087: "f32[8, 160, 17, 17]" = torch.ops.aten.mul.Tensor(sub_282, unsqueeze_948);  sub_282 = unsqueeze_948 = None
    sub_284: "f32[8, 160, 17, 17]" = torch.ops.aten.sub.Tensor(where_47, mul_1087);  where_47 = mul_1087 = None
    sub_285: "f32[8, 160, 17, 17]" = torch.ops.aten.sub.Tensor(sub_284, unsqueeze_945);  sub_284 = unsqueeze_945 = None
    mul_1088: "f32[8, 160, 17, 17]" = torch.ops.aten.mul.Tensor(sub_285, unsqueeze_951);  sub_285 = unsqueeze_951 = None
    mul_1089: "f32[160]" = torch.ops.aten.mul.Tensor(sum_97, squeeze_139);  sum_97 = squeeze_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_47 = torch.ops.aten.convolution_backward.default(mul_1088, relu_45, primals_235, [0], [1, 1], [0, 3], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1088 = primals_235 = None
    getitem_337: "f32[8, 160, 17, 17]" = convolution_backward_47[0]
    getitem_338: "f32[160, 160, 1, 7]" = convolution_backward_47[1];  convolution_backward_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_239: "f32[8, 160, 17, 17]" = torch.ops.aten.alias.default(relu_45);  relu_45 = None
    alias_240: "f32[8, 160, 17, 17]" = torch.ops.aten.alias.default(alias_239);  alias_239 = None
    le_48: "b8[8, 160, 17, 17]" = torch.ops.aten.le.Scalar(alias_240, 0);  alias_240 = None
    where_48: "f32[8, 160, 17, 17]" = torch.ops.aten.where.self(le_48, full_default, getitem_337);  le_48 = getitem_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_98: "f32[160]" = torch.ops.aten.sum.dim_IntList(where_48, [0, 2, 3])
    sub_286: "f32[8, 160, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_954);  convolution_45 = unsqueeze_954 = None
    mul_1090: "f32[8, 160, 17, 17]" = torch.ops.aten.mul.Tensor(where_48, sub_286)
    sum_99: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_1090, [0, 2, 3]);  mul_1090 = None
    mul_1091: "f32[160]" = torch.ops.aten.mul.Tensor(sum_98, 0.00043252595155709344)
    unsqueeze_955: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_1091, 0);  mul_1091 = None
    unsqueeze_956: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_955, 2);  unsqueeze_955 = None
    unsqueeze_957: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_956, 3);  unsqueeze_956 = None
    mul_1092: "f32[160]" = torch.ops.aten.mul.Tensor(sum_99, 0.00043252595155709344)
    mul_1093: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_136, squeeze_136)
    mul_1094: "f32[160]" = torch.ops.aten.mul.Tensor(mul_1092, mul_1093);  mul_1092 = mul_1093 = None
    unsqueeze_958: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_1094, 0);  mul_1094 = None
    unsqueeze_959: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_958, 2);  unsqueeze_958 = None
    unsqueeze_960: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_959, 3);  unsqueeze_959 = None
    mul_1095: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_136, primals_91);  primals_91 = None
    unsqueeze_961: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_1095, 0);  mul_1095 = None
    unsqueeze_962: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_961, 2);  unsqueeze_961 = None
    unsqueeze_963: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_962, 3);  unsqueeze_962 = None
    mul_1096: "f32[8, 160, 17, 17]" = torch.ops.aten.mul.Tensor(sub_286, unsqueeze_960);  sub_286 = unsqueeze_960 = None
    sub_288: "f32[8, 160, 17, 17]" = torch.ops.aten.sub.Tensor(where_48, mul_1096);  where_48 = mul_1096 = None
    sub_289: "f32[8, 160, 17, 17]" = torch.ops.aten.sub.Tensor(sub_288, unsqueeze_957);  sub_288 = unsqueeze_957 = None
    mul_1097: "f32[8, 160, 17, 17]" = torch.ops.aten.mul.Tensor(sub_289, unsqueeze_963);  sub_289 = unsqueeze_963 = None
    mul_1098: "f32[160]" = torch.ops.aten.mul.Tensor(sum_99, squeeze_136);  sum_99 = squeeze_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_48 = torch.ops.aten.convolution_backward.default(mul_1097, relu_44, primals_234, [0], [1, 1], [3, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1097 = primals_234 = None
    getitem_340: "f32[8, 160, 17, 17]" = convolution_backward_48[0]
    getitem_341: "f32[160, 160, 7, 1]" = convolution_backward_48[1];  convolution_backward_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_242: "f32[8, 160, 17, 17]" = torch.ops.aten.alias.default(relu_44);  relu_44 = None
    alias_243: "f32[8, 160, 17, 17]" = torch.ops.aten.alias.default(alias_242);  alias_242 = None
    le_49: "b8[8, 160, 17, 17]" = torch.ops.aten.le.Scalar(alias_243, 0);  alias_243 = None
    where_49: "f32[8, 160, 17, 17]" = torch.ops.aten.where.self(le_49, full_default, getitem_340);  le_49 = getitem_340 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_100: "f32[160]" = torch.ops.aten.sum.dim_IntList(where_49, [0, 2, 3])
    sub_290: "f32[8, 160, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_44, unsqueeze_966);  convolution_44 = unsqueeze_966 = None
    mul_1099: "f32[8, 160, 17, 17]" = torch.ops.aten.mul.Tensor(where_49, sub_290)
    sum_101: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_1099, [0, 2, 3]);  mul_1099 = None
    mul_1100: "f32[160]" = torch.ops.aten.mul.Tensor(sum_100, 0.00043252595155709344)
    unsqueeze_967: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_1100, 0);  mul_1100 = None
    unsqueeze_968: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_967, 2);  unsqueeze_967 = None
    unsqueeze_969: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_968, 3);  unsqueeze_968 = None
    mul_1101: "f32[160]" = torch.ops.aten.mul.Tensor(sum_101, 0.00043252595155709344)
    mul_1102: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_133, squeeze_133)
    mul_1103: "f32[160]" = torch.ops.aten.mul.Tensor(mul_1101, mul_1102);  mul_1101 = mul_1102 = None
    unsqueeze_970: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_1103, 0);  mul_1103 = None
    unsqueeze_971: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_970, 2);  unsqueeze_970 = None
    unsqueeze_972: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_971, 3);  unsqueeze_971 = None
    mul_1104: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_133, primals_89);  primals_89 = None
    unsqueeze_973: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_1104, 0);  mul_1104 = None
    unsqueeze_974: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_973, 2);  unsqueeze_973 = None
    unsqueeze_975: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_974, 3);  unsqueeze_974 = None
    mul_1105: "f32[8, 160, 17, 17]" = torch.ops.aten.mul.Tensor(sub_290, unsqueeze_972);  sub_290 = unsqueeze_972 = None
    sub_292: "f32[8, 160, 17, 17]" = torch.ops.aten.sub.Tensor(where_49, mul_1105);  where_49 = mul_1105 = None
    sub_293: "f32[8, 160, 17, 17]" = torch.ops.aten.sub.Tensor(sub_292, unsqueeze_969);  sub_292 = unsqueeze_969 = None
    mul_1106: "f32[8, 160, 17, 17]" = torch.ops.aten.mul.Tensor(sub_293, unsqueeze_975);  sub_293 = unsqueeze_975 = None
    mul_1107: "f32[160]" = torch.ops.aten.mul.Tensor(sum_101, squeeze_133);  sum_101 = squeeze_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_49 = torch.ops.aten.convolution_backward.default(mul_1106, cat_4, primals_233, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1106 = primals_233 = None
    getitem_343: "f32[8, 768, 17, 17]" = convolution_backward_49[0]
    getitem_344: "f32[160, 768, 1, 1]" = convolution_backward_49[1];  convolution_backward_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_488: "f32[8, 768, 17, 17]" = torch.ops.aten.add.Tensor(avg_pool2d_backward_4, getitem_343);  avg_pool2d_backward_4 = getitem_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    where_50: "f32[8, 192, 17, 17]" = torch.ops.aten.where.self(le_50, full_default, slice_29);  le_50 = slice_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_102: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_50, [0, 2, 3])
    sub_294: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_43, unsqueeze_978);  convolution_43 = unsqueeze_978 = None
    mul_1108: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(where_50, sub_294)
    sum_103: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_1108, [0, 2, 3]);  mul_1108 = None
    mul_1109: "f32[192]" = torch.ops.aten.mul.Tensor(sum_102, 0.00043252595155709344)
    unsqueeze_979: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_1109, 0);  mul_1109 = None
    unsqueeze_980: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_979, 2);  unsqueeze_979 = None
    unsqueeze_981: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_980, 3);  unsqueeze_980 = None
    mul_1110: "f32[192]" = torch.ops.aten.mul.Tensor(sum_103, 0.00043252595155709344)
    mul_1111: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_130, squeeze_130)
    mul_1112: "f32[192]" = torch.ops.aten.mul.Tensor(mul_1110, mul_1111);  mul_1110 = mul_1111 = None
    unsqueeze_982: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_1112, 0);  mul_1112 = None
    unsqueeze_983: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_982, 2);  unsqueeze_982 = None
    unsqueeze_984: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_983, 3);  unsqueeze_983 = None
    mul_1113: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_130, primals_87);  primals_87 = None
    unsqueeze_985: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_1113, 0);  mul_1113 = None
    unsqueeze_986: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_985, 2);  unsqueeze_985 = None
    unsqueeze_987: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_986, 3);  unsqueeze_986 = None
    mul_1114: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_294, unsqueeze_984);  sub_294 = unsqueeze_984 = None
    sub_296: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(where_50, mul_1114);  where_50 = mul_1114 = None
    sub_297: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(sub_296, unsqueeze_981);  sub_296 = unsqueeze_981 = None
    mul_1115: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_297, unsqueeze_987);  sub_297 = unsqueeze_987 = None
    mul_1116: "f32[192]" = torch.ops.aten.mul.Tensor(sum_103, squeeze_130);  sum_103 = squeeze_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_50 = torch.ops.aten.convolution_backward.default(mul_1115, relu_42, primals_232, [0], [1, 1], [3, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1115 = primals_232 = None
    getitem_346: "f32[8, 160, 17, 17]" = convolution_backward_50[0]
    getitem_347: "f32[192, 160, 7, 1]" = convolution_backward_50[1];  convolution_backward_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_248: "f32[8, 160, 17, 17]" = torch.ops.aten.alias.default(relu_42);  relu_42 = None
    alias_249: "f32[8, 160, 17, 17]" = torch.ops.aten.alias.default(alias_248);  alias_248 = None
    le_51: "b8[8, 160, 17, 17]" = torch.ops.aten.le.Scalar(alias_249, 0);  alias_249 = None
    where_51: "f32[8, 160, 17, 17]" = torch.ops.aten.where.self(le_51, full_default, getitem_346);  le_51 = getitem_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_104: "f32[160]" = torch.ops.aten.sum.dim_IntList(where_51, [0, 2, 3])
    sub_298: "f32[8, 160, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_42, unsqueeze_990);  convolution_42 = unsqueeze_990 = None
    mul_1117: "f32[8, 160, 17, 17]" = torch.ops.aten.mul.Tensor(where_51, sub_298)
    sum_105: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_1117, [0, 2, 3]);  mul_1117 = None
    mul_1118: "f32[160]" = torch.ops.aten.mul.Tensor(sum_104, 0.00043252595155709344)
    unsqueeze_991: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_1118, 0);  mul_1118 = None
    unsqueeze_992: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_991, 2);  unsqueeze_991 = None
    unsqueeze_993: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_992, 3);  unsqueeze_992 = None
    mul_1119: "f32[160]" = torch.ops.aten.mul.Tensor(sum_105, 0.00043252595155709344)
    mul_1120: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_127, squeeze_127)
    mul_1121: "f32[160]" = torch.ops.aten.mul.Tensor(mul_1119, mul_1120);  mul_1119 = mul_1120 = None
    unsqueeze_994: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_1121, 0);  mul_1121 = None
    unsqueeze_995: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_994, 2);  unsqueeze_994 = None
    unsqueeze_996: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_995, 3);  unsqueeze_995 = None
    mul_1122: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_127, primals_85);  primals_85 = None
    unsqueeze_997: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_1122, 0);  mul_1122 = None
    unsqueeze_998: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_997, 2);  unsqueeze_997 = None
    unsqueeze_999: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_998, 3);  unsqueeze_998 = None
    mul_1123: "f32[8, 160, 17, 17]" = torch.ops.aten.mul.Tensor(sub_298, unsqueeze_996);  sub_298 = unsqueeze_996 = None
    sub_300: "f32[8, 160, 17, 17]" = torch.ops.aten.sub.Tensor(where_51, mul_1123);  where_51 = mul_1123 = None
    sub_301: "f32[8, 160, 17, 17]" = torch.ops.aten.sub.Tensor(sub_300, unsqueeze_993);  sub_300 = unsqueeze_993 = None
    mul_1124: "f32[8, 160, 17, 17]" = torch.ops.aten.mul.Tensor(sub_301, unsqueeze_999);  sub_301 = unsqueeze_999 = None
    mul_1125: "f32[160]" = torch.ops.aten.mul.Tensor(sum_105, squeeze_127);  sum_105 = squeeze_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_51 = torch.ops.aten.convolution_backward.default(mul_1124, relu_41, primals_231, [0], [1, 1], [0, 3], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1124 = primals_231 = None
    getitem_349: "f32[8, 160, 17, 17]" = convolution_backward_51[0]
    getitem_350: "f32[160, 160, 1, 7]" = convolution_backward_51[1];  convolution_backward_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_251: "f32[8, 160, 17, 17]" = torch.ops.aten.alias.default(relu_41);  relu_41 = None
    alias_252: "f32[8, 160, 17, 17]" = torch.ops.aten.alias.default(alias_251);  alias_251 = None
    le_52: "b8[8, 160, 17, 17]" = torch.ops.aten.le.Scalar(alias_252, 0);  alias_252 = None
    where_52: "f32[8, 160, 17, 17]" = torch.ops.aten.where.self(le_52, full_default, getitem_349);  le_52 = getitem_349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_106: "f32[160]" = torch.ops.aten.sum.dim_IntList(where_52, [0, 2, 3])
    sub_302: "f32[8, 160, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_1002);  convolution_41 = unsqueeze_1002 = None
    mul_1126: "f32[8, 160, 17, 17]" = torch.ops.aten.mul.Tensor(where_52, sub_302)
    sum_107: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_1126, [0, 2, 3]);  mul_1126 = None
    mul_1127: "f32[160]" = torch.ops.aten.mul.Tensor(sum_106, 0.00043252595155709344)
    unsqueeze_1003: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_1127, 0);  mul_1127 = None
    unsqueeze_1004: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1003, 2);  unsqueeze_1003 = None
    unsqueeze_1005: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1004, 3);  unsqueeze_1004 = None
    mul_1128: "f32[160]" = torch.ops.aten.mul.Tensor(sum_107, 0.00043252595155709344)
    mul_1129: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_124, squeeze_124)
    mul_1130: "f32[160]" = torch.ops.aten.mul.Tensor(mul_1128, mul_1129);  mul_1128 = mul_1129 = None
    unsqueeze_1006: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_1130, 0);  mul_1130 = None
    unsqueeze_1007: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1006, 2);  unsqueeze_1006 = None
    unsqueeze_1008: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1007, 3);  unsqueeze_1007 = None
    mul_1131: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_124, primals_83);  primals_83 = None
    unsqueeze_1009: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_1131, 0);  mul_1131 = None
    unsqueeze_1010: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1009, 2);  unsqueeze_1009 = None
    unsqueeze_1011: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1010, 3);  unsqueeze_1010 = None
    mul_1132: "f32[8, 160, 17, 17]" = torch.ops.aten.mul.Tensor(sub_302, unsqueeze_1008);  sub_302 = unsqueeze_1008 = None
    sub_304: "f32[8, 160, 17, 17]" = torch.ops.aten.sub.Tensor(where_52, mul_1132);  where_52 = mul_1132 = None
    sub_305: "f32[8, 160, 17, 17]" = torch.ops.aten.sub.Tensor(sub_304, unsqueeze_1005);  sub_304 = unsqueeze_1005 = None
    mul_1133: "f32[8, 160, 17, 17]" = torch.ops.aten.mul.Tensor(sub_305, unsqueeze_1011);  sub_305 = unsqueeze_1011 = None
    mul_1134: "f32[160]" = torch.ops.aten.mul.Tensor(sum_107, squeeze_124);  sum_107 = squeeze_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_52 = torch.ops.aten.convolution_backward.default(mul_1133, cat_4, primals_230, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1133 = primals_230 = None
    getitem_352: "f32[8, 768, 17, 17]" = convolution_backward_52[0]
    getitem_353: "f32[160, 768, 1, 1]" = convolution_backward_52[1];  convolution_backward_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_489: "f32[8, 768, 17, 17]" = torch.ops.aten.add.Tensor(add_488, getitem_352);  add_488 = getitem_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    where_53: "f32[8, 192, 17, 17]" = torch.ops.aten.where.self(le_53, full_default, slice_28);  le_53 = slice_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_108: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_53, [0, 2, 3])
    sub_306: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_1014);  convolution_40 = unsqueeze_1014 = None
    mul_1135: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(where_53, sub_306)
    sum_109: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_1135, [0, 2, 3]);  mul_1135 = None
    mul_1136: "f32[192]" = torch.ops.aten.mul.Tensor(sum_108, 0.00043252595155709344)
    unsqueeze_1015: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_1136, 0);  mul_1136 = None
    unsqueeze_1016: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1015, 2);  unsqueeze_1015 = None
    unsqueeze_1017: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1016, 3);  unsqueeze_1016 = None
    mul_1137: "f32[192]" = torch.ops.aten.mul.Tensor(sum_109, 0.00043252595155709344)
    mul_1138: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_121, squeeze_121)
    mul_1139: "f32[192]" = torch.ops.aten.mul.Tensor(mul_1137, mul_1138);  mul_1137 = mul_1138 = None
    unsqueeze_1018: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_1139, 0);  mul_1139 = None
    unsqueeze_1019: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1018, 2);  unsqueeze_1018 = None
    unsqueeze_1020: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1019, 3);  unsqueeze_1019 = None
    mul_1140: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_121, primals_81);  primals_81 = None
    unsqueeze_1021: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_1140, 0);  mul_1140 = None
    unsqueeze_1022: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1021, 2);  unsqueeze_1021 = None
    unsqueeze_1023: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1022, 3);  unsqueeze_1022 = None
    mul_1141: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_306, unsqueeze_1020);  sub_306 = unsqueeze_1020 = None
    sub_308: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(where_53, mul_1141);  where_53 = mul_1141 = None
    sub_309: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(sub_308, unsqueeze_1017);  sub_308 = unsqueeze_1017 = None
    mul_1142: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_309, unsqueeze_1023);  sub_309 = unsqueeze_1023 = None
    mul_1143: "f32[192]" = torch.ops.aten.mul.Tensor(sum_109, squeeze_121);  sum_109 = squeeze_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_53 = torch.ops.aten.convolution_backward.default(mul_1142, cat_4, primals_229, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1142 = cat_4 = primals_229 = None
    getitem_355: "f32[8, 768, 17, 17]" = convolution_backward_53[0]
    getitem_356: "f32[192, 768, 1, 1]" = convolution_backward_53[1];  convolution_backward_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_490: "f32[8, 768, 17, 17]" = torch.ops.aten.add.Tensor(add_489, getitem_355);  add_489 = getitem_355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/inception_v3.py:128, code: return torch.cat(outputs, 1)
    slice_32: "f32[8, 192, 17, 17]" = torch.ops.aten.slice.Tensor(add_490, 1, 0, 192)
    slice_33: "f32[8, 192, 17, 17]" = torch.ops.aten.slice.Tensor(add_490, 1, 192, 384)
    slice_34: "f32[8, 192, 17, 17]" = torch.ops.aten.slice.Tensor(add_490, 1, 384, 576)
    slice_35: "f32[8, 192, 17, 17]" = torch.ops.aten.slice.Tensor(add_490, 1, 576, 768);  add_490 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    where_54: "f32[8, 192, 17, 17]" = torch.ops.aten.where.self(le_54, full_default, slice_35);  le_54 = slice_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_110: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_54, [0, 2, 3])
    sub_310: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_39, unsqueeze_1026);  convolution_39 = unsqueeze_1026 = None
    mul_1144: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(where_54, sub_310)
    sum_111: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_1144, [0, 2, 3]);  mul_1144 = None
    mul_1145: "f32[192]" = torch.ops.aten.mul.Tensor(sum_110, 0.00043252595155709344)
    unsqueeze_1027: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_1145, 0);  mul_1145 = None
    unsqueeze_1028: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1027, 2);  unsqueeze_1027 = None
    unsqueeze_1029: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1028, 3);  unsqueeze_1028 = None
    mul_1146: "f32[192]" = torch.ops.aten.mul.Tensor(sum_111, 0.00043252595155709344)
    mul_1147: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_118, squeeze_118)
    mul_1148: "f32[192]" = torch.ops.aten.mul.Tensor(mul_1146, mul_1147);  mul_1146 = mul_1147 = None
    unsqueeze_1030: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_1148, 0);  mul_1148 = None
    unsqueeze_1031: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1030, 2);  unsqueeze_1030 = None
    unsqueeze_1032: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1031, 3);  unsqueeze_1031 = None
    mul_1149: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_118, primals_79);  primals_79 = None
    unsqueeze_1033: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_1149, 0);  mul_1149 = None
    unsqueeze_1034: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1033, 2);  unsqueeze_1033 = None
    unsqueeze_1035: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1034, 3);  unsqueeze_1034 = None
    mul_1150: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_310, unsqueeze_1032);  sub_310 = unsqueeze_1032 = None
    sub_312: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(where_54, mul_1150);  where_54 = mul_1150 = None
    sub_313: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(sub_312, unsqueeze_1029);  sub_312 = unsqueeze_1029 = None
    mul_1151: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_313, unsqueeze_1035);  sub_313 = unsqueeze_1035 = None
    mul_1152: "f32[192]" = torch.ops.aten.mul.Tensor(sum_111, squeeze_118);  sum_111 = squeeze_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_54 = torch.ops.aten.convolution_backward.default(mul_1151, avg_pool2d_3, primals_228, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1151 = avg_pool2d_3 = primals_228 = None
    getitem_358: "f32[8, 768, 17, 17]" = convolution_backward_54[0]
    getitem_359: "f32[192, 768, 1, 1]" = convolution_backward_54[1];  convolution_backward_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/inception_v3.py:120, code: branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    avg_pool2d_backward_5: "f32[8, 768, 17, 17]" = torch.ops.aten.avg_pool2d_backward.default(getitem_358, cat_3, [3, 3], [1, 1], [1, 1], False, True, None);  getitem_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    where_55: "f32[8, 192, 17, 17]" = torch.ops.aten.where.self(le_55, full_default, slice_34);  le_55 = slice_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_112: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_55, [0, 2, 3])
    sub_314: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_38, unsqueeze_1038);  convolution_38 = unsqueeze_1038 = None
    mul_1153: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(where_55, sub_314)
    sum_113: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_1153, [0, 2, 3]);  mul_1153 = None
    mul_1154: "f32[192]" = torch.ops.aten.mul.Tensor(sum_112, 0.00043252595155709344)
    unsqueeze_1039: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_1154, 0);  mul_1154 = None
    unsqueeze_1040: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1039, 2);  unsqueeze_1039 = None
    unsqueeze_1041: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1040, 3);  unsqueeze_1040 = None
    mul_1155: "f32[192]" = torch.ops.aten.mul.Tensor(sum_113, 0.00043252595155709344)
    mul_1156: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_115, squeeze_115)
    mul_1157: "f32[192]" = torch.ops.aten.mul.Tensor(mul_1155, mul_1156);  mul_1155 = mul_1156 = None
    unsqueeze_1042: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_1157, 0);  mul_1157 = None
    unsqueeze_1043: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1042, 2);  unsqueeze_1042 = None
    unsqueeze_1044: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1043, 3);  unsqueeze_1043 = None
    mul_1158: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_115, primals_77);  primals_77 = None
    unsqueeze_1045: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_1158, 0);  mul_1158 = None
    unsqueeze_1046: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1045, 2);  unsqueeze_1045 = None
    unsqueeze_1047: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1046, 3);  unsqueeze_1046 = None
    mul_1159: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_314, unsqueeze_1044);  sub_314 = unsqueeze_1044 = None
    sub_316: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(where_55, mul_1159);  where_55 = mul_1159 = None
    sub_317: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(sub_316, unsqueeze_1041);  sub_316 = unsqueeze_1041 = None
    mul_1160: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_317, unsqueeze_1047);  sub_317 = unsqueeze_1047 = None
    mul_1161: "f32[192]" = torch.ops.aten.mul.Tensor(sum_113, squeeze_115);  sum_113 = squeeze_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_55 = torch.ops.aten.convolution_backward.default(mul_1160, relu_37, primals_227, [0], [1, 1], [0, 3], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1160 = primals_227 = None
    getitem_361: "f32[8, 128, 17, 17]" = convolution_backward_55[0]
    getitem_362: "f32[192, 128, 1, 7]" = convolution_backward_55[1];  convolution_backward_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_263: "f32[8, 128, 17, 17]" = torch.ops.aten.alias.default(relu_37);  relu_37 = None
    alias_264: "f32[8, 128, 17, 17]" = torch.ops.aten.alias.default(alias_263);  alias_263 = None
    le_56: "b8[8, 128, 17, 17]" = torch.ops.aten.le.Scalar(alias_264, 0);  alias_264 = None
    where_56: "f32[8, 128, 17, 17]" = torch.ops.aten.where.self(le_56, full_default, getitem_361);  le_56 = getitem_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_114: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_56, [0, 2, 3])
    sub_318: "f32[8, 128, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_37, unsqueeze_1050);  convolution_37 = unsqueeze_1050 = None
    mul_1162: "f32[8, 128, 17, 17]" = torch.ops.aten.mul.Tensor(where_56, sub_318)
    sum_115: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1162, [0, 2, 3]);  mul_1162 = None
    mul_1163: "f32[128]" = torch.ops.aten.mul.Tensor(sum_114, 0.00043252595155709344)
    unsqueeze_1051: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1163, 0);  mul_1163 = None
    unsqueeze_1052: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1051, 2);  unsqueeze_1051 = None
    unsqueeze_1053: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1052, 3);  unsqueeze_1052 = None
    mul_1164: "f32[128]" = torch.ops.aten.mul.Tensor(sum_115, 0.00043252595155709344)
    mul_1165: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_112, squeeze_112)
    mul_1166: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1164, mul_1165);  mul_1164 = mul_1165 = None
    unsqueeze_1054: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1166, 0);  mul_1166 = None
    unsqueeze_1055: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1054, 2);  unsqueeze_1054 = None
    unsqueeze_1056: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1055, 3);  unsqueeze_1055 = None
    mul_1167: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_112, primals_75);  primals_75 = None
    unsqueeze_1057: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1167, 0);  mul_1167 = None
    unsqueeze_1058: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1057, 2);  unsqueeze_1057 = None
    unsqueeze_1059: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1058, 3);  unsqueeze_1058 = None
    mul_1168: "f32[8, 128, 17, 17]" = torch.ops.aten.mul.Tensor(sub_318, unsqueeze_1056);  sub_318 = unsqueeze_1056 = None
    sub_320: "f32[8, 128, 17, 17]" = torch.ops.aten.sub.Tensor(where_56, mul_1168);  where_56 = mul_1168 = None
    sub_321: "f32[8, 128, 17, 17]" = torch.ops.aten.sub.Tensor(sub_320, unsqueeze_1053);  sub_320 = unsqueeze_1053 = None
    mul_1169: "f32[8, 128, 17, 17]" = torch.ops.aten.mul.Tensor(sub_321, unsqueeze_1059);  sub_321 = unsqueeze_1059 = None
    mul_1170: "f32[128]" = torch.ops.aten.mul.Tensor(sum_115, squeeze_112);  sum_115 = squeeze_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_56 = torch.ops.aten.convolution_backward.default(mul_1169, relu_36, primals_226, [0], [1, 1], [3, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1169 = primals_226 = None
    getitem_364: "f32[8, 128, 17, 17]" = convolution_backward_56[0]
    getitem_365: "f32[128, 128, 7, 1]" = convolution_backward_56[1];  convolution_backward_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_266: "f32[8, 128, 17, 17]" = torch.ops.aten.alias.default(relu_36);  relu_36 = None
    alias_267: "f32[8, 128, 17, 17]" = torch.ops.aten.alias.default(alias_266);  alias_266 = None
    le_57: "b8[8, 128, 17, 17]" = torch.ops.aten.le.Scalar(alias_267, 0);  alias_267 = None
    where_57: "f32[8, 128, 17, 17]" = torch.ops.aten.where.self(le_57, full_default, getitem_364);  le_57 = getitem_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_116: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_57, [0, 2, 3])
    sub_322: "f32[8, 128, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_1062);  convolution_36 = unsqueeze_1062 = None
    mul_1171: "f32[8, 128, 17, 17]" = torch.ops.aten.mul.Tensor(where_57, sub_322)
    sum_117: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1171, [0, 2, 3]);  mul_1171 = None
    mul_1172: "f32[128]" = torch.ops.aten.mul.Tensor(sum_116, 0.00043252595155709344)
    unsqueeze_1063: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1172, 0);  mul_1172 = None
    unsqueeze_1064: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1063, 2);  unsqueeze_1063 = None
    unsqueeze_1065: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1064, 3);  unsqueeze_1064 = None
    mul_1173: "f32[128]" = torch.ops.aten.mul.Tensor(sum_117, 0.00043252595155709344)
    mul_1174: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_109, squeeze_109)
    mul_1175: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1173, mul_1174);  mul_1173 = mul_1174 = None
    unsqueeze_1066: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1175, 0);  mul_1175 = None
    unsqueeze_1067: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1066, 2);  unsqueeze_1066 = None
    unsqueeze_1068: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1067, 3);  unsqueeze_1067 = None
    mul_1176: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_109, primals_73);  primals_73 = None
    unsqueeze_1069: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1176, 0);  mul_1176 = None
    unsqueeze_1070: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1069, 2);  unsqueeze_1069 = None
    unsqueeze_1071: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1070, 3);  unsqueeze_1070 = None
    mul_1177: "f32[8, 128, 17, 17]" = torch.ops.aten.mul.Tensor(sub_322, unsqueeze_1068);  sub_322 = unsqueeze_1068 = None
    sub_324: "f32[8, 128, 17, 17]" = torch.ops.aten.sub.Tensor(where_57, mul_1177);  where_57 = mul_1177 = None
    sub_325: "f32[8, 128, 17, 17]" = torch.ops.aten.sub.Tensor(sub_324, unsqueeze_1065);  sub_324 = unsqueeze_1065 = None
    mul_1178: "f32[8, 128, 17, 17]" = torch.ops.aten.mul.Tensor(sub_325, unsqueeze_1071);  sub_325 = unsqueeze_1071 = None
    mul_1179: "f32[128]" = torch.ops.aten.mul.Tensor(sum_117, squeeze_109);  sum_117 = squeeze_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_57 = torch.ops.aten.convolution_backward.default(mul_1178, relu_35, primals_225, [0], [1, 1], [0, 3], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1178 = primals_225 = None
    getitem_367: "f32[8, 128, 17, 17]" = convolution_backward_57[0]
    getitem_368: "f32[128, 128, 1, 7]" = convolution_backward_57[1];  convolution_backward_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_269: "f32[8, 128, 17, 17]" = torch.ops.aten.alias.default(relu_35);  relu_35 = None
    alias_270: "f32[8, 128, 17, 17]" = torch.ops.aten.alias.default(alias_269);  alias_269 = None
    le_58: "b8[8, 128, 17, 17]" = torch.ops.aten.le.Scalar(alias_270, 0);  alias_270 = None
    where_58: "f32[8, 128, 17, 17]" = torch.ops.aten.where.self(le_58, full_default, getitem_367);  le_58 = getitem_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_118: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_58, [0, 2, 3])
    sub_326: "f32[8, 128, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_1074);  convolution_35 = unsqueeze_1074 = None
    mul_1180: "f32[8, 128, 17, 17]" = torch.ops.aten.mul.Tensor(where_58, sub_326)
    sum_119: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1180, [0, 2, 3]);  mul_1180 = None
    mul_1181: "f32[128]" = torch.ops.aten.mul.Tensor(sum_118, 0.00043252595155709344)
    unsqueeze_1075: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1181, 0);  mul_1181 = None
    unsqueeze_1076: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1075, 2);  unsqueeze_1075 = None
    unsqueeze_1077: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1076, 3);  unsqueeze_1076 = None
    mul_1182: "f32[128]" = torch.ops.aten.mul.Tensor(sum_119, 0.00043252595155709344)
    mul_1183: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_106, squeeze_106)
    mul_1184: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1182, mul_1183);  mul_1182 = mul_1183 = None
    unsqueeze_1078: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1184, 0);  mul_1184 = None
    unsqueeze_1079: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1078, 2);  unsqueeze_1078 = None
    unsqueeze_1080: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1079, 3);  unsqueeze_1079 = None
    mul_1185: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_106, primals_71);  primals_71 = None
    unsqueeze_1081: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1185, 0);  mul_1185 = None
    unsqueeze_1082: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1081, 2);  unsqueeze_1081 = None
    unsqueeze_1083: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1082, 3);  unsqueeze_1082 = None
    mul_1186: "f32[8, 128, 17, 17]" = torch.ops.aten.mul.Tensor(sub_326, unsqueeze_1080);  sub_326 = unsqueeze_1080 = None
    sub_328: "f32[8, 128, 17, 17]" = torch.ops.aten.sub.Tensor(where_58, mul_1186);  where_58 = mul_1186 = None
    sub_329: "f32[8, 128, 17, 17]" = torch.ops.aten.sub.Tensor(sub_328, unsqueeze_1077);  sub_328 = unsqueeze_1077 = None
    mul_1187: "f32[8, 128, 17, 17]" = torch.ops.aten.mul.Tensor(sub_329, unsqueeze_1083);  sub_329 = unsqueeze_1083 = None
    mul_1188: "f32[128]" = torch.ops.aten.mul.Tensor(sum_119, squeeze_106);  sum_119 = squeeze_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_58 = torch.ops.aten.convolution_backward.default(mul_1187, relu_34, primals_224, [0], [1, 1], [3, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1187 = primals_224 = None
    getitem_370: "f32[8, 128, 17, 17]" = convolution_backward_58[0]
    getitem_371: "f32[128, 128, 7, 1]" = convolution_backward_58[1];  convolution_backward_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_272: "f32[8, 128, 17, 17]" = torch.ops.aten.alias.default(relu_34);  relu_34 = None
    alias_273: "f32[8, 128, 17, 17]" = torch.ops.aten.alias.default(alias_272);  alias_272 = None
    le_59: "b8[8, 128, 17, 17]" = torch.ops.aten.le.Scalar(alias_273, 0);  alias_273 = None
    where_59: "f32[8, 128, 17, 17]" = torch.ops.aten.where.self(le_59, full_default, getitem_370);  le_59 = getitem_370 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_120: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_59, [0, 2, 3])
    sub_330: "f32[8, 128, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_1086);  convolution_34 = unsqueeze_1086 = None
    mul_1189: "f32[8, 128, 17, 17]" = torch.ops.aten.mul.Tensor(where_59, sub_330)
    sum_121: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1189, [0, 2, 3]);  mul_1189 = None
    mul_1190: "f32[128]" = torch.ops.aten.mul.Tensor(sum_120, 0.00043252595155709344)
    unsqueeze_1087: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1190, 0);  mul_1190 = None
    unsqueeze_1088: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1087, 2);  unsqueeze_1087 = None
    unsqueeze_1089: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1088, 3);  unsqueeze_1088 = None
    mul_1191: "f32[128]" = torch.ops.aten.mul.Tensor(sum_121, 0.00043252595155709344)
    mul_1192: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_103, squeeze_103)
    mul_1193: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1191, mul_1192);  mul_1191 = mul_1192 = None
    unsqueeze_1090: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1193, 0);  mul_1193 = None
    unsqueeze_1091: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1090, 2);  unsqueeze_1090 = None
    unsqueeze_1092: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1091, 3);  unsqueeze_1091 = None
    mul_1194: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_103, primals_69);  primals_69 = None
    unsqueeze_1093: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1194, 0);  mul_1194 = None
    unsqueeze_1094: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1093, 2);  unsqueeze_1093 = None
    unsqueeze_1095: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1094, 3);  unsqueeze_1094 = None
    mul_1195: "f32[8, 128, 17, 17]" = torch.ops.aten.mul.Tensor(sub_330, unsqueeze_1092);  sub_330 = unsqueeze_1092 = None
    sub_332: "f32[8, 128, 17, 17]" = torch.ops.aten.sub.Tensor(where_59, mul_1195);  where_59 = mul_1195 = None
    sub_333: "f32[8, 128, 17, 17]" = torch.ops.aten.sub.Tensor(sub_332, unsqueeze_1089);  sub_332 = unsqueeze_1089 = None
    mul_1196: "f32[8, 128, 17, 17]" = torch.ops.aten.mul.Tensor(sub_333, unsqueeze_1095);  sub_333 = unsqueeze_1095 = None
    mul_1197: "f32[128]" = torch.ops.aten.mul.Tensor(sum_121, squeeze_103);  sum_121 = squeeze_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_59 = torch.ops.aten.convolution_backward.default(mul_1196, cat_3, primals_223, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1196 = primals_223 = None
    getitem_373: "f32[8, 768, 17, 17]" = convolution_backward_59[0]
    getitem_374: "f32[128, 768, 1, 1]" = convolution_backward_59[1];  convolution_backward_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_491: "f32[8, 768, 17, 17]" = torch.ops.aten.add.Tensor(avg_pool2d_backward_5, getitem_373);  avg_pool2d_backward_5 = getitem_373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    where_60: "f32[8, 192, 17, 17]" = torch.ops.aten.where.self(le_60, full_default, slice_33);  le_60 = slice_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_122: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_60, [0, 2, 3])
    sub_334: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_1098);  convolution_33 = unsqueeze_1098 = None
    mul_1198: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(where_60, sub_334)
    sum_123: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_1198, [0, 2, 3]);  mul_1198 = None
    mul_1199: "f32[192]" = torch.ops.aten.mul.Tensor(sum_122, 0.00043252595155709344)
    unsqueeze_1099: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_1199, 0);  mul_1199 = None
    unsqueeze_1100: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1099, 2);  unsqueeze_1099 = None
    unsqueeze_1101: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1100, 3);  unsqueeze_1100 = None
    mul_1200: "f32[192]" = torch.ops.aten.mul.Tensor(sum_123, 0.00043252595155709344)
    mul_1201: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_100, squeeze_100)
    mul_1202: "f32[192]" = torch.ops.aten.mul.Tensor(mul_1200, mul_1201);  mul_1200 = mul_1201 = None
    unsqueeze_1102: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_1202, 0);  mul_1202 = None
    unsqueeze_1103: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1102, 2);  unsqueeze_1102 = None
    unsqueeze_1104: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1103, 3);  unsqueeze_1103 = None
    mul_1203: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_100, primals_67);  primals_67 = None
    unsqueeze_1105: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_1203, 0);  mul_1203 = None
    unsqueeze_1106: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1105, 2);  unsqueeze_1105 = None
    unsqueeze_1107: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1106, 3);  unsqueeze_1106 = None
    mul_1204: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_334, unsqueeze_1104);  sub_334 = unsqueeze_1104 = None
    sub_336: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(where_60, mul_1204);  where_60 = mul_1204 = None
    sub_337: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(sub_336, unsqueeze_1101);  sub_336 = unsqueeze_1101 = None
    mul_1205: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_337, unsqueeze_1107);  sub_337 = unsqueeze_1107 = None
    mul_1206: "f32[192]" = torch.ops.aten.mul.Tensor(sum_123, squeeze_100);  sum_123 = squeeze_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_60 = torch.ops.aten.convolution_backward.default(mul_1205, relu_32, primals_222, [0], [1, 1], [3, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1205 = primals_222 = None
    getitem_376: "f32[8, 128, 17, 17]" = convolution_backward_60[0]
    getitem_377: "f32[192, 128, 7, 1]" = convolution_backward_60[1];  convolution_backward_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_278: "f32[8, 128, 17, 17]" = torch.ops.aten.alias.default(relu_32);  relu_32 = None
    alias_279: "f32[8, 128, 17, 17]" = torch.ops.aten.alias.default(alias_278);  alias_278 = None
    le_61: "b8[8, 128, 17, 17]" = torch.ops.aten.le.Scalar(alias_279, 0);  alias_279 = None
    where_61: "f32[8, 128, 17, 17]" = torch.ops.aten.where.self(le_61, full_default, getitem_376);  le_61 = getitem_376 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_124: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_61, [0, 2, 3])
    sub_338: "f32[8, 128, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_1110);  convolution_32 = unsqueeze_1110 = None
    mul_1207: "f32[8, 128, 17, 17]" = torch.ops.aten.mul.Tensor(where_61, sub_338)
    sum_125: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1207, [0, 2, 3]);  mul_1207 = None
    mul_1208: "f32[128]" = torch.ops.aten.mul.Tensor(sum_124, 0.00043252595155709344)
    unsqueeze_1111: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1208, 0);  mul_1208 = None
    unsqueeze_1112: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1111, 2);  unsqueeze_1111 = None
    unsqueeze_1113: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1112, 3);  unsqueeze_1112 = None
    mul_1209: "f32[128]" = torch.ops.aten.mul.Tensor(sum_125, 0.00043252595155709344)
    mul_1210: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_97, squeeze_97)
    mul_1211: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1209, mul_1210);  mul_1209 = mul_1210 = None
    unsqueeze_1114: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1211, 0);  mul_1211 = None
    unsqueeze_1115: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1114, 2);  unsqueeze_1114 = None
    unsqueeze_1116: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1115, 3);  unsqueeze_1115 = None
    mul_1212: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_97, primals_65);  primals_65 = None
    unsqueeze_1117: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1212, 0);  mul_1212 = None
    unsqueeze_1118: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1117, 2);  unsqueeze_1117 = None
    unsqueeze_1119: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1118, 3);  unsqueeze_1118 = None
    mul_1213: "f32[8, 128, 17, 17]" = torch.ops.aten.mul.Tensor(sub_338, unsqueeze_1116);  sub_338 = unsqueeze_1116 = None
    sub_340: "f32[8, 128, 17, 17]" = torch.ops.aten.sub.Tensor(where_61, mul_1213);  where_61 = mul_1213 = None
    sub_341: "f32[8, 128, 17, 17]" = torch.ops.aten.sub.Tensor(sub_340, unsqueeze_1113);  sub_340 = unsqueeze_1113 = None
    mul_1214: "f32[8, 128, 17, 17]" = torch.ops.aten.mul.Tensor(sub_341, unsqueeze_1119);  sub_341 = unsqueeze_1119 = None
    mul_1215: "f32[128]" = torch.ops.aten.mul.Tensor(sum_125, squeeze_97);  sum_125 = squeeze_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_61 = torch.ops.aten.convolution_backward.default(mul_1214, relu_31, primals_221, [0], [1, 1], [0, 3], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1214 = primals_221 = None
    getitem_379: "f32[8, 128, 17, 17]" = convolution_backward_61[0]
    getitem_380: "f32[128, 128, 1, 7]" = convolution_backward_61[1];  convolution_backward_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_281: "f32[8, 128, 17, 17]" = torch.ops.aten.alias.default(relu_31);  relu_31 = None
    alias_282: "f32[8, 128, 17, 17]" = torch.ops.aten.alias.default(alias_281);  alias_281 = None
    le_62: "b8[8, 128, 17, 17]" = torch.ops.aten.le.Scalar(alias_282, 0);  alias_282 = None
    where_62: "f32[8, 128, 17, 17]" = torch.ops.aten.where.self(le_62, full_default, getitem_379);  le_62 = getitem_379 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_126: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_62, [0, 2, 3])
    sub_342: "f32[8, 128, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_31, unsqueeze_1122);  convolution_31 = unsqueeze_1122 = None
    mul_1216: "f32[8, 128, 17, 17]" = torch.ops.aten.mul.Tensor(where_62, sub_342)
    sum_127: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1216, [0, 2, 3]);  mul_1216 = None
    mul_1217: "f32[128]" = torch.ops.aten.mul.Tensor(sum_126, 0.00043252595155709344)
    unsqueeze_1123: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1217, 0);  mul_1217 = None
    unsqueeze_1124: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1123, 2);  unsqueeze_1123 = None
    unsqueeze_1125: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1124, 3);  unsqueeze_1124 = None
    mul_1218: "f32[128]" = torch.ops.aten.mul.Tensor(sum_127, 0.00043252595155709344)
    mul_1219: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_94, squeeze_94)
    mul_1220: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1218, mul_1219);  mul_1218 = mul_1219 = None
    unsqueeze_1126: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1220, 0);  mul_1220 = None
    unsqueeze_1127: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1126, 2);  unsqueeze_1126 = None
    unsqueeze_1128: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1127, 3);  unsqueeze_1127 = None
    mul_1221: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_94, primals_63);  primals_63 = None
    unsqueeze_1129: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1221, 0);  mul_1221 = None
    unsqueeze_1130: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1129, 2);  unsqueeze_1129 = None
    unsqueeze_1131: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1130, 3);  unsqueeze_1130 = None
    mul_1222: "f32[8, 128, 17, 17]" = torch.ops.aten.mul.Tensor(sub_342, unsqueeze_1128);  sub_342 = unsqueeze_1128 = None
    sub_344: "f32[8, 128, 17, 17]" = torch.ops.aten.sub.Tensor(where_62, mul_1222);  where_62 = mul_1222 = None
    sub_345: "f32[8, 128, 17, 17]" = torch.ops.aten.sub.Tensor(sub_344, unsqueeze_1125);  sub_344 = unsqueeze_1125 = None
    mul_1223: "f32[8, 128, 17, 17]" = torch.ops.aten.mul.Tensor(sub_345, unsqueeze_1131);  sub_345 = unsqueeze_1131 = None
    mul_1224: "f32[128]" = torch.ops.aten.mul.Tensor(sum_127, squeeze_94);  sum_127 = squeeze_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_62 = torch.ops.aten.convolution_backward.default(mul_1223, cat_3, primals_220, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1223 = primals_220 = None
    getitem_382: "f32[8, 768, 17, 17]" = convolution_backward_62[0]
    getitem_383: "f32[128, 768, 1, 1]" = convolution_backward_62[1];  convolution_backward_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_492: "f32[8, 768, 17, 17]" = torch.ops.aten.add.Tensor(add_491, getitem_382);  add_491 = getitem_382 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    where_63: "f32[8, 192, 17, 17]" = torch.ops.aten.where.self(le_63, full_default, slice_32);  le_63 = slice_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_128: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_63, [0, 2, 3])
    sub_346: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_1134);  convolution_30 = unsqueeze_1134 = None
    mul_1225: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(where_63, sub_346)
    sum_129: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_1225, [0, 2, 3]);  mul_1225 = None
    mul_1226: "f32[192]" = torch.ops.aten.mul.Tensor(sum_128, 0.00043252595155709344)
    unsqueeze_1135: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_1226, 0);  mul_1226 = None
    unsqueeze_1136: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1135, 2);  unsqueeze_1135 = None
    unsqueeze_1137: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1136, 3);  unsqueeze_1136 = None
    mul_1227: "f32[192]" = torch.ops.aten.mul.Tensor(sum_129, 0.00043252595155709344)
    mul_1228: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_91, squeeze_91)
    mul_1229: "f32[192]" = torch.ops.aten.mul.Tensor(mul_1227, mul_1228);  mul_1227 = mul_1228 = None
    unsqueeze_1138: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_1229, 0);  mul_1229 = None
    unsqueeze_1139: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1138, 2);  unsqueeze_1138 = None
    unsqueeze_1140: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1139, 3);  unsqueeze_1139 = None
    mul_1230: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_91, primals_61);  primals_61 = None
    unsqueeze_1141: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_1230, 0);  mul_1230 = None
    unsqueeze_1142: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1141, 2);  unsqueeze_1141 = None
    unsqueeze_1143: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1142, 3);  unsqueeze_1142 = None
    mul_1231: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_346, unsqueeze_1140);  sub_346 = unsqueeze_1140 = None
    sub_348: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(where_63, mul_1231);  where_63 = mul_1231 = None
    sub_349: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(sub_348, unsqueeze_1137);  sub_348 = unsqueeze_1137 = None
    mul_1232: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_349, unsqueeze_1143);  sub_349 = unsqueeze_1143 = None
    mul_1233: "f32[192]" = torch.ops.aten.mul.Tensor(sum_129, squeeze_91);  sum_129 = squeeze_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_63 = torch.ops.aten.convolution_backward.default(mul_1232, cat_3, primals_219, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1232 = cat_3 = primals_219 = None
    getitem_385: "f32[8, 768, 17, 17]" = convolution_backward_63[0]
    getitem_386: "f32[192, 768, 1, 1]" = convolution_backward_63[1];  convolution_backward_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_493: "f32[8, 768, 17, 17]" = torch.ops.aten.add.Tensor(add_492, getitem_385);  add_492 = getitem_385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/inception_v3.py:84, code: return torch.cat(outputs, 1)
    slice_36: "f32[8, 384, 17, 17]" = torch.ops.aten.slice.Tensor(add_493, 1, 0, 384)
    slice_37: "f32[8, 96, 17, 17]" = torch.ops.aten.slice.Tensor(add_493, 1, 384, 480)
    slice_38: "f32[8, 288, 17, 17]" = torch.ops.aten.slice.Tensor(add_493, 1, 480, 768);  add_493 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/inception_v3.py:77, code: branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
    max_pool2d_with_indices_backward_1: "f32[8, 288, 35, 35]" = torch.ops.aten.max_pool2d_with_indices_backward.default(slice_38, cat_2, [3, 3], [2, 2], [0, 0], [1, 1], False, getitem_65);  slice_38 = getitem_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    where_64: "f32[8, 96, 17, 17]" = torch.ops.aten.where.self(le_64, full_default, slice_37);  le_64 = slice_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_130: "f32[96]" = torch.ops.aten.sum.dim_IntList(where_64, [0, 2, 3])
    sub_350: "f32[8, 96, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_1146);  convolution_29 = unsqueeze_1146 = None
    mul_1234: "f32[8, 96, 17, 17]" = torch.ops.aten.mul.Tensor(where_64, sub_350)
    sum_131: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_1234, [0, 2, 3]);  mul_1234 = None
    mul_1235: "f32[96]" = torch.ops.aten.mul.Tensor(sum_130, 0.00043252595155709344)
    unsqueeze_1147: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_1235, 0);  mul_1235 = None
    unsqueeze_1148: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1147, 2);  unsqueeze_1147 = None
    unsqueeze_1149: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1148, 3);  unsqueeze_1148 = None
    mul_1236: "f32[96]" = torch.ops.aten.mul.Tensor(sum_131, 0.00043252595155709344)
    mul_1237: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_88, squeeze_88)
    mul_1238: "f32[96]" = torch.ops.aten.mul.Tensor(mul_1236, mul_1237);  mul_1236 = mul_1237 = None
    unsqueeze_1150: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_1238, 0);  mul_1238 = None
    unsqueeze_1151: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1150, 2);  unsqueeze_1150 = None
    unsqueeze_1152: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1151, 3);  unsqueeze_1151 = None
    mul_1239: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_88, primals_59);  primals_59 = None
    unsqueeze_1153: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_1239, 0);  mul_1239 = None
    unsqueeze_1154: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1153, 2);  unsqueeze_1153 = None
    unsqueeze_1155: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1154, 3);  unsqueeze_1154 = None
    mul_1240: "f32[8, 96, 17, 17]" = torch.ops.aten.mul.Tensor(sub_350, unsqueeze_1152);  sub_350 = unsqueeze_1152 = None
    sub_352: "f32[8, 96, 17, 17]" = torch.ops.aten.sub.Tensor(where_64, mul_1240);  where_64 = mul_1240 = None
    sub_353: "f32[8, 96, 17, 17]" = torch.ops.aten.sub.Tensor(sub_352, unsqueeze_1149);  sub_352 = unsqueeze_1149 = None
    mul_1241: "f32[8, 96, 17, 17]" = torch.ops.aten.mul.Tensor(sub_353, unsqueeze_1155);  sub_353 = unsqueeze_1155 = None
    mul_1242: "f32[96]" = torch.ops.aten.mul.Tensor(sum_131, squeeze_88);  sum_131 = squeeze_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_64 = torch.ops.aten.convolution_backward.default(mul_1241, relu_28, primals_218, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1241 = primals_218 = None
    getitem_388: "f32[8, 96, 35, 35]" = convolution_backward_64[0]
    getitem_389: "f32[96, 96, 3, 3]" = convolution_backward_64[1];  convolution_backward_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_290: "f32[8, 96, 35, 35]" = torch.ops.aten.alias.default(relu_28);  relu_28 = None
    alias_291: "f32[8, 96, 35, 35]" = torch.ops.aten.alias.default(alias_290);  alias_290 = None
    le_65: "b8[8, 96, 35, 35]" = torch.ops.aten.le.Scalar(alias_291, 0);  alias_291 = None
    where_65: "f32[8, 96, 35, 35]" = torch.ops.aten.where.self(le_65, full_default, getitem_388);  le_65 = getitem_388 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_132: "f32[96]" = torch.ops.aten.sum.dim_IntList(where_65, [0, 2, 3])
    sub_354: "f32[8, 96, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_1158);  convolution_28 = unsqueeze_1158 = None
    mul_1243: "f32[8, 96, 35, 35]" = torch.ops.aten.mul.Tensor(where_65, sub_354)
    sum_133: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_1243, [0, 2, 3]);  mul_1243 = None
    mul_1244: "f32[96]" = torch.ops.aten.mul.Tensor(sum_132, 0.00010204081632653062)
    unsqueeze_1159: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_1244, 0);  mul_1244 = None
    unsqueeze_1160: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1159, 2);  unsqueeze_1159 = None
    unsqueeze_1161: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1160, 3);  unsqueeze_1160 = None
    mul_1245: "f32[96]" = torch.ops.aten.mul.Tensor(sum_133, 0.00010204081632653062)
    mul_1246: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_85, squeeze_85)
    mul_1247: "f32[96]" = torch.ops.aten.mul.Tensor(mul_1245, mul_1246);  mul_1245 = mul_1246 = None
    unsqueeze_1162: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_1247, 0);  mul_1247 = None
    unsqueeze_1163: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1162, 2);  unsqueeze_1162 = None
    unsqueeze_1164: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1163, 3);  unsqueeze_1163 = None
    mul_1248: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_85, primals_57);  primals_57 = None
    unsqueeze_1165: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_1248, 0);  mul_1248 = None
    unsqueeze_1166: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1165, 2);  unsqueeze_1165 = None
    unsqueeze_1167: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1166, 3);  unsqueeze_1166 = None
    mul_1249: "f32[8, 96, 35, 35]" = torch.ops.aten.mul.Tensor(sub_354, unsqueeze_1164);  sub_354 = unsqueeze_1164 = None
    sub_356: "f32[8, 96, 35, 35]" = torch.ops.aten.sub.Tensor(where_65, mul_1249);  where_65 = mul_1249 = None
    sub_357: "f32[8, 96, 35, 35]" = torch.ops.aten.sub.Tensor(sub_356, unsqueeze_1161);  sub_356 = unsqueeze_1161 = None
    mul_1250: "f32[8, 96, 35, 35]" = torch.ops.aten.mul.Tensor(sub_357, unsqueeze_1167);  sub_357 = unsqueeze_1167 = None
    mul_1251: "f32[96]" = torch.ops.aten.mul.Tensor(sum_133, squeeze_85);  sum_133 = squeeze_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_65 = torch.ops.aten.convolution_backward.default(mul_1250, relu_27, primals_217, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1250 = primals_217 = None
    getitem_391: "f32[8, 64, 35, 35]" = convolution_backward_65[0]
    getitem_392: "f32[96, 64, 3, 3]" = convolution_backward_65[1];  convolution_backward_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_293: "f32[8, 64, 35, 35]" = torch.ops.aten.alias.default(relu_27);  relu_27 = None
    alias_294: "f32[8, 64, 35, 35]" = torch.ops.aten.alias.default(alias_293);  alias_293 = None
    le_66: "b8[8, 64, 35, 35]" = torch.ops.aten.le.Scalar(alias_294, 0);  alias_294 = None
    where_66: "f32[8, 64, 35, 35]" = torch.ops.aten.where.self(le_66, full_default, getitem_391);  le_66 = getitem_391 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_134: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_66, [0, 2, 3])
    sub_358: "f32[8, 64, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_1170);  convolution_27 = unsqueeze_1170 = None
    mul_1252: "f32[8, 64, 35, 35]" = torch.ops.aten.mul.Tensor(where_66, sub_358)
    sum_135: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1252, [0, 2, 3]);  mul_1252 = None
    mul_1253: "f32[64]" = torch.ops.aten.mul.Tensor(sum_134, 0.00010204081632653062)
    unsqueeze_1171: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1253, 0);  mul_1253 = None
    unsqueeze_1172: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1171, 2);  unsqueeze_1171 = None
    unsqueeze_1173: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1172, 3);  unsqueeze_1172 = None
    mul_1254: "f32[64]" = torch.ops.aten.mul.Tensor(sum_135, 0.00010204081632653062)
    mul_1255: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_82, squeeze_82)
    mul_1256: "f32[64]" = torch.ops.aten.mul.Tensor(mul_1254, mul_1255);  mul_1254 = mul_1255 = None
    unsqueeze_1174: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1256, 0);  mul_1256 = None
    unsqueeze_1175: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1174, 2);  unsqueeze_1174 = None
    unsqueeze_1176: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1175, 3);  unsqueeze_1175 = None
    mul_1257: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_82, primals_55);  primals_55 = None
    unsqueeze_1177: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1257, 0);  mul_1257 = None
    unsqueeze_1178: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1177, 2);  unsqueeze_1177 = None
    unsqueeze_1179: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1178, 3);  unsqueeze_1178 = None
    mul_1258: "f32[8, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_358, unsqueeze_1176);  sub_358 = unsqueeze_1176 = None
    sub_360: "f32[8, 64, 35, 35]" = torch.ops.aten.sub.Tensor(where_66, mul_1258);  where_66 = mul_1258 = None
    sub_361: "f32[8, 64, 35, 35]" = torch.ops.aten.sub.Tensor(sub_360, unsqueeze_1173);  sub_360 = unsqueeze_1173 = None
    mul_1259: "f32[8, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_361, unsqueeze_1179);  sub_361 = unsqueeze_1179 = None
    mul_1260: "f32[64]" = torch.ops.aten.mul.Tensor(sum_135, squeeze_82);  sum_135 = squeeze_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_66 = torch.ops.aten.convolution_backward.default(mul_1259, cat_2, primals_216, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1259 = primals_216 = None
    getitem_394: "f32[8, 288, 35, 35]" = convolution_backward_66[0]
    getitem_395: "f32[64, 288, 1, 1]" = convolution_backward_66[1];  convolution_backward_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_494: "f32[8, 288, 35, 35]" = torch.ops.aten.add.Tensor(max_pool2d_with_indices_backward_1, getitem_394);  max_pool2d_with_indices_backward_1 = getitem_394 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    where_67: "f32[8, 384, 17, 17]" = torch.ops.aten.where.self(le_67, full_default, slice_36);  le_67 = slice_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_136: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_67, [0, 2, 3])
    sub_362: "f32[8, 384, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_1182);  convolution_26 = unsqueeze_1182 = None
    mul_1261: "f32[8, 384, 17, 17]" = torch.ops.aten.mul.Tensor(where_67, sub_362)
    sum_137: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_1261, [0, 2, 3]);  mul_1261 = None
    mul_1262: "f32[384]" = torch.ops.aten.mul.Tensor(sum_136, 0.00043252595155709344)
    unsqueeze_1183: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_1262, 0);  mul_1262 = None
    unsqueeze_1184: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1183, 2);  unsqueeze_1183 = None
    unsqueeze_1185: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1184, 3);  unsqueeze_1184 = None
    mul_1263: "f32[384]" = torch.ops.aten.mul.Tensor(sum_137, 0.00043252595155709344)
    mul_1264: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_79, squeeze_79)
    mul_1265: "f32[384]" = torch.ops.aten.mul.Tensor(mul_1263, mul_1264);  mul_1263 = mul_1264 = None
    unsqueeze_1186: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_1265, 0);  mul_1265 = None
    unsqueeze_1187: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1186, 2);  unsqueeze_1186 = None
    unsqueeze_1188: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1187, 3);  unsqueeze_1187 = None
    mul_1266: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_79, primals_53);  primals_53 = None
    unsqueeze_1189: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_1266, 0);  mul_1266 = None
    unsqueeze_1190: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1189, 2);  unsqueeze_1189 = None
    unsqueeze_1191: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1190, 3);  unsqueeze_1190 = None
    mul_1267: "f32[8, 384, 17, 17]" = torch.ops.aten.mul.Tensor(sub_362, unsqueeze_1188);  sub_362 = unsqueeze_1188 = None
    sub_364: "f32[8, 384, 17, 17]" = torch.ops.aten.sub.Tensor(where_67, mul_1267);  where_67 = mul_1267 = None
    sub_365: "f32[8, 384, 17, 17]" = torch.ops.aten.sub.Tensor(sub_364, unsqueeze_1185);  sub_364 = unsqueeze_1185 = None
    mul_1268: "f32[8, 384, 17, 17]" = torch.ops.aten.mul.Tensor(sub_365, unsqueeze_1191);  sub_365 = unsqueeze_1191 = None
    mul_1269: "f32[384]" = torch.ops.aten.mul.Tensor(sum_137, squeeze_79);  sum_137 = squeeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_67 = torch.ops.aten.convolution_backward.default(mul_1268, cat_2, primals_215, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1268 = cat_2 = primals_215 = None
    getitem_397: "f32[8, 288, 35, 35]" = convolution_backward_67[0]
    getitem_398: "f32[384, 288, 3, 3]" = convolution_backward_67[1];  convolution_backward_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_495: "f32[8, 288, 35, 35]" = torch.ops.aten.add.Tensor(add_494, getitem_397);  add_494 = getitem_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/inception_v3.py:56, code: return torch.cat(outputs, 1)
    slice_39: "f32[8, 64, 35, 35]" = torch.ops.aten.slice.Tensor(add_495, 1, 0, 64)
    slice_40: "f32[8, 64, 35, 35]" = torch.ops.aten.slice.Tensor(add_495, 1, 64, 128)
    slice_41: "f32[8, 96, 35, 35]" = torch.ops.aten.slice.Tensor(add_495, 1, 128, 224)
    slice_42: "f32[8, 64, 35, 35]" = torch.ops.aten.slice.Tensor(add_495, 1, 224, 288);  add_495 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    where_68: "f32[8, 64, 35, 35]" = torch.ops.aten.where.self(le_68, full_default, slice_42);  le_68 = slice_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_138: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_68, [0, 2, 3])
    sub_366: "f32[8, 64, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_1194);  convolution_25 = unsqueeze_1194 = None
    mul_1270: "f32[8, 64, 35, 35]" = torch.ops.aten.mul.Tensor(where_68, sub_366)
    sum_139: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1270, [0, 2, 3]);  mul_1270 = None
    mul_1271: "f32[64]" = torch.ops.aten.mul.Tensor(sum_138, 0.00010204081632653062)
    unsqueeze_1195: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1271, 0);  mul_1271 = None
    unsqueeze_1196: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1195, 2);  unsqueeze_1195 = None
    unsqueeze_1197: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1196, 3);  unsqueeze_1196 = None
    mul_1272: "f32[64]" = torch.ops.aten.mul.Tensor(sum_139, 0.00010204081632653062)
    mul_1273: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_76, squeeze_76)
    mul_1274: "f32[64]" = torch.ops.aten.mul.Tensor(mul_1272, mul_1273);  mul_1272 = mul_1273 = None
    unsqueeze_1198: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1274, 0);  mul_1274 = None
    unsqueeze_1199: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1198, 2);  unsqueeze_1198 = None
    unsqueeze_1200: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1199, 3);  unsqueeze_1199 = None
    mul_1275: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_76, primals_51);  primals_51 = None
    unsqueeze_1201: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1275, 0);  mul_1275 = None
    unsqueeze_1202: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1201, 2);  unsqueeze_1201 = None
    unsqueeze_1203: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1202, 3);  unsqueeze_1202 = None
    mul_1276: "f32[8, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_366, unsqueeze_1200);  sub_366 = unsqueeze_1200 = None
    sub_368: "f32[8, 64, 35, 35]" = torch.ops.aten.sub.Tensor(where_68, mul_1276);  where_68 = mul_1276 = None
    sub_369: "f32[8, 64, 35, 35]" = torch.ops.aten.sub.Tensor(sub_368, unsqueeze_1197);  sub_368 = unsqueeze_1197 = None
    mul_1277: "f32[8, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_369, unsqueeze_1203);  sub_369 = unsqueeze_1203 = None
    mul_1278: "f32[64]" = torch.ops.aten.mul.Tensor(sum_139, squeeze_76);  sum_139 = squeeze_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_68 = torch.ops.aten.convolution_backward.default(mul_1277, avg_pool2d_2, primals_214, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1277 = avg_pool2d_2 = primals_214 = None
    getitem_400: "f32[8, 288, 35, 35]" = convolution_backward_68[0]
    getitem_401: "f32[64, 288, 1, 1]" = convolution_backward_68[1];  convolution_backward_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/inception_v3.py:48, code: branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    avg_pool2d_backward_6: "f32[8, 288, 35, 35]" = torch.ops.aten.avg_pool2d_backward.default(getitem_400, cat_1, [3, 3], [1, 1], [1, 1], False, True, None);  getitem_400 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    where_69: "f32[8, 96, 35, 35]" = torch.ops.aten.where.self(le_69, full_default, slice_41);  le_69 = slice_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_140: "f32[96]" = torch.ops.aten.sum.dim_IntList(where_69, [0, 2, 3])
    sub_370: "f32[8, 96, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_1206);  convolution_24 = unsqueeze_1206 = None
    mul_1279: "f32[8, 96, 35, 35]" = torch.ops.aten.mul.Tensor(where_69, sub_370)
    sum_141: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_1279, [0, 2, 3]);  mul_1279 = None
    mul_1280: "f32[96]" = torch.ops.aten.mul.Tensor(sum_140, 0.00010204081632653062)
    unsqueeze_1207: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_1280, 0);  mul_1280 = None
    unsqueeze_1208: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1207, 2);  unsqueeze_1207 = None
    unsqueeze_1209: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1208, 3);  unsqueeze_1208 = None
    mul_1281: "f32[96]" = torch.ops.aten.mul.Tensor(sum_141, 0.00010204081632653062)
    mul_1282: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_73, squeeze_73)
    mul_1283: "f32[96]" = torch.ops.aten.mul.Tensor(mul_1281, mul_1282);  mul_1281 = mul_1282 = None
    unsqueeze_1210: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_1283, 0);  mul_1283 = None
    unsqueeze_1211: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1210, 2);  unsqueeze_1210 = None
    unsqueeze_1212: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1211, 3);  unsqueeze_1211 = None
    mul_1284: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_73, primals_49);  primals_49 = None
    unsqueeze_1213: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_1284, 0);  mul_1284 = None
    unsqueeze_1214: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1213, 2);  unsqueeze_1213 = None
    unsqueeze_1215: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1214, 3);  unsqueeze_1214 = None
    mul_1285: "f32[8, 96, 35, 35]" = torch.ops.aten.mul.Tensor(sub_370, unsqueeze_1212);  sub_370 = unsqueeze_1212 = None
    sub_372: "f32[8, 96, 35, 35]" = torch.ops.aten.sub.Tensor(where_69, mul_1285);  where_69 = mul_1285 = None
    sub_373: "f32[8, 96, 35, 35]" = torch.ops.aten.sub.Tensor(sub_372, unsqueeze_1209);  sub_372 = unsqueeze_1209 = None
    mul_1286: "f32[8, 96, 35, 35]" = torch.ops.aten.mul.Tensor(sub_373, unsqueeze_1215);  sub_373 = unsqueeze_1215 = None
    mul_1287: "f32[96]" = torch.ops.aten.mul.Tensor(sum_141, squeeze_73);  sum_141 = squeeze_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_69 = torch.ops.aten.convolution_backward.default(mul_1286, relu_23, primals_213, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1286 = primals_213 = None
    getitem_403: "f32[8, 96, 35, 35]" = convolution_backward_69[0]
    getitem_404: "f32[96, 96, 3, 3]" = convolution_backward_69[1];  convolution_backward_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_305: "f32[8, 96, 35, 35]" = torch.ops.aten.alias.default(relu_23);  relu_23 = None
    alias_306: "f32[8, 96, 35, 35]" = torch.ops.aten.alias.default(alias_305);  alias_305 = None
    le_70: "b8[8, 96, 35, 35]" = torch.ops.aten.le.Scalar(alias_306, 0);  alias_306 = None
    where_70: "f32[8, 96, 35, 35]" = torch.ops.aten.where.self(le_70, full_default, getitem_403);  le_70 = getitem_403 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_142: "f32[96]" = torch.ops.aten.sum.dim_IntList(where_70, [0, 2, 3])
    sub_374: "f32[8, 96, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_1218);  convolution_23 = unsqueeze_1218 = None
    mul_1288: "f32[8, 96, 35, 35]" = torch.ops.aten.mul.Tensor(where_70, sub_374)
    sum_143: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_1288, [0, 2, 3]);  mul_1288 = None
    mul_1289: "f32[96]" = torch.ops.aten.mul.Tensor(sum_142, 0.00010204081632653062)
    unsqueeze_1219: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_1289, 0);  mul_1289 = None
    unsqueeze_1220: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1219, 2);  unsqueeze_1219 = None
    unsqueeze_1221: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1220, 3);  unsqueeze_1220 = None
    mul_1290: "f32[96]" = torch.ops.aten.mul.Tensor(sum_143, 0.00010204081632653062)
    mul_1291: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_70, squeeze_70)
    mul_1292: "f32[96]" = torch.ops.aten.mul.Tensor(mul_1290, mul_1291);  mul_1290 = mul_1291 = None
    unsqueeze_1222: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_1292, 0);  mul_1292 = None
    unsqueeze_1223: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1222, 2);  unsqueeze_1222 = None
    unsqueeze_1224: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1223, 3);  unsqueeze_1223 = None
    mul_1293: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_70, primals_47);  primals_47 = None
    unsqueeze_1225: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_1293, 0);  mul_1293 = None
    unsqueeze_1226: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1225, 2);  unsqueeze_1225 = None
    unsqueeze_1227: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1226, 3);  unsqueeze_1226 = None
    mul_1294: "f32[8, 96, 35, 35]" = torch.ops.aten.mul.Tensor(sub_374, unsqueeze_1224);  sub_374 = unsqueeze_1224 = None
    sub_376: "f32[8, 96, 35, 35]" = torch.ops.aten.sub.Tensor(where_70, mul_1294);  where_70 = mul_1294 = None
    sub_377: "f32[8, 96, 35, 35]" = torch.ops.aten.sub.Tensor(sub_376, unsqueeze_1221);  sub_376 = unsqueeze_1221 = None
    mul_1295: "f32[8, 96, 35, 35]" = torch.ops.aten.mul.Tensor(sub_377, unsqueeze_1227);  sub_377 = unsqueeze_1227 = None
    mul_1296: "f32[96]" = torch.ops.aten.mul.Tensor(sum_143, squeeze_70);  sum_143 = squeeze_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_70 = torch.ops.aten.convolution_backward.default(mul_1295, relu_22, primals_212, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1295 = primals_212 = None
    getitem_406: "f32[8, 64, 35, 35]" = convolution_backward_70[0]
    getitem_407: "f32[96, 64, 3, 3]" = convolution_backward_70[1];  convolution_backward_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_308: "f32[8, 64, 35, 35]" = torch.ops.aten.alias.default(relu_22);  relu_22 = None
    alias_309: "f32[8, 64, 35, 35]" = torch.ops.aten.alias.default(alias_308);  alias_308 = None
    le_71: "b8[8, 64, 35, 35]" = torch.ops.aten.le.Scalar(alias_309, 0);  alias_309 = None
    where_71: "f32[8, 64, 35, 35]" = torch.ops.aten.where.self(le_71, full_default, getitem_406);  le_71 = getitem_406 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_144: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_71, [0, 2, 3])
    sub_378: "f32[8, 64, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_1230);  convolution_22 = unsqueeze_1230 = None
    mul_1297: "f32[8, 64, 35, 35]" = torch.ops.aten.mul.Tensor(where_71, sub_378)
    sum_145: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1297, [0, 2, 3]);  mul_1297 = None
    mul_1298: "f32[64]" = torch.ops.aten.mul.Tensor(sum_144, 0.00010204081632653062)
    unsqueeze_1231: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1298, 0);  mul_1298 = None
    unsqueeze_1232: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1231, 2);  unsqueeze_1231 = None
    unsqueeze_1233: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1232, 3);  unsqueeze_1232 = None
    mul_1299: "f32[64]" = torch.ops.aten.mul.Tensor(sum_145, 0.00010204081632653062)
    mul_1300: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_67, squeeze_67)
    mul_1301: "f32[64]" = torch.ops.aten.mul.Tensor(mul_1299, mul_1300);  mul_1299 = mul_1300 = None
    unsqueeze_1234: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1301, 0);  mul_1301 = None
    unsqueeze_1235: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1234, 2);  unsqueeze_1234 = None
    unsqueeze_1236: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1235, 3);  unsqueeze_1235 = None
    mul_1302: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_67, primals_45);  primals_45 = None
    unsqueeze_1237: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1302, 0);  mul_1302 = None
    unsqueeze_1238: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1237, 2);  unsqueeze_1237 = None
    unsqueeze_1239: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1238, 3);  unsqueeze_1238 = None
    mul_1303: "f32[8, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_378, unsqueeze_1236);  sub_378 = unsqueeze_1236 = None
    sub_380: "f32[8, 64, 35, 35]" = torch.ops.aten.sub.Tensor(where_71, mul_1303);  where_71 = mul_1303 = None
    sub_381: "f32[8, 64, 35, 35]" = torch.ops.aten.sub.Tensor(sub_380, unsqueeze_1233);  sub_380 = unsqueeze_1233 = None
    mul_1304: "f32[8, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_381, unsqueeze_1239);  sub_381 = unsqueeze_1239 = None
    mul_1305: "f32[64]" = torch.ops.aten.mul.Tensor(sum_145, squeeze_67);  sum_145 = squeeze_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_71 = torch.ops.aten.convolution_backward.default(mul_1304, cat_1, primals_211, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1304 = primals_211 = None
    getitem_409: "f32[8, 288, 35, 35]" = convolution_backward_71[0]
    getitem_410: "f32[64, 288, 1, 1]" = convolution_backward_71[1];  convolution_backward_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_496: "f32[8, 288, 35, 35]" = torch.ops.aten.add.Tensor(avg_pool2d_backward_6, getitem_409);  avg_pool2d_backward_6 = getitem_409 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    where_72: "f32[8, 64, 35, 35]" = torch.ops.aten.where.self(le_72, full_default, slice_40);  le_72 = slice_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_146: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_72, [0, 2, 3])
    sub_382: "f32[8, 64, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_1242);  convolution_21 = unsqueeze_1242 = None
    mul_1306: "f32[8, 64, 35, 35]" = torch.ops.aten.mul.Tensor(where_72, sub_382)
    sum_147: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1306, [0, 2, 3]);  mul_1306 = None
    mul_1307: "f32[64]" = torch.ops.aten.mul.Tensor(sum_146, 0.00010204081632653062)
    unsqueeze_1243: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1307, 0);  mul_1307 = None
    unsqueeze_1244: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1243, 2);  unsqueeze_1243 = None
    unsqueeze_1245: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1244, 3);  unsqueeze_1244 = None
    mul_1308: "f32[64]" = torch.ops.aten.mul.Tensor(sum_147, 0.00010204081632653062)
    mul_1309: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_64, squeeze_64)
    mul_1310: "f32[64]" = torch.ops.aten.mul.Tensor(mul_1308, mul_1309);  mul_1308 = mul_1309 = None
    unsqueeze_1246: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1310, 0);  mul_1310 = None
    unsqueeze_1247: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1246, 2);  unsqueeze_1246 = None
    unsqueeze_1248: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1247, 3);  unsqueeze_1247 = None
    mul_1311: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_64, primals_43);  primals_43 = None
    unsqueeze_1249: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1311, 0);  mul_1311 = None
    unsqueeze_1250: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1249, 2);  unsqueeze_1249 = None
    unsqueeze_1251: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1250, 3);  unsqueeze_1250 = None
    mul_1312: "f32[8, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_382, unsqueeze_1248);  sub_382 = unsqueeze_1248 = None
    sub_384: "f32[8, 64, 35, 35]" = torch.ops.aten.sub.Tensor(where_72, mul_1312);  where_72 = mul_1312 = None
    sub_385: "f32[8, 64, 35, 35]" = torch.ops.aten.sub.Tensor(sub_384, unsqueeze_1245);  sub_384 = unsqueeze_1245 = None
    mul_1313: "f32[8, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_385, unsqueeze_1251);  sub_385 = unsqueeze_1251 = None
    mul_1314: "f32[64]" = torch.ops.aten.mul.Tensor(sum_147, squeeze_64);  sum_147 = squeeze_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_72 = torch.ops.aten.convolution_backward.default(mul_1313, relu_20, primals_210, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1313 = primals_210 = None
    getitem_412: "f32[8, 48, 35, 35]" = convolution_backward_72[0]
    getitem_413: "f32[64, 48, 5, 5]" = convolution_backward_72[1];  convolution_backward_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_314: "f32[8, 48, 35, 35]" = torch.ops.aten.alias.default(relu_20);  relu_20 = None
    alias_315: "f32[8, 48, 35, 35]" = torch.ops.aten.alias.default(alias_314);  alias_314 = None
    le_73: "b8[8, 48, 35, 35]" = torch.ops.aten.le.Scalar(alias_315, 0);  alias_315 = None
    where_73: "f32[8, 48, 35, 35]" = torch.ops.aten.where.self(le_73, full_default, getitem_412);  le_73 = getitem_412 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_148: "f32[48]" = torch.ops.aten.sum.dim_IntList(where_73, [0, 2, 3])
    sub_386: "f32[8, 48, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_1254);  convolution_20 = unsqueeze_1254 = None
    mul_1315: "f32[8, 48, 35, 35]" = torch.ops.aten.mul.Tensor(where_73, sub_386)
    sum_149: "f32[48]" = torch.ops.aten.sum.dim_IntList(mul_1315, [0, 2, 3]);  mul_1315 = None
    mul_1316: "f32[48]" = torch.ops.aten.mul.Tensor(sum_148, 0.00010204081632653062)
    unsqueeze_1255: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_1316, 0);  mul_1316 = None
    unsqueeze_1256: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1255, 2);  unsqueeze_1255 = None
    unsqueeze_1257: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1256, 3);  unsqueeze_1256 = None
    mul_1317: "f32[48]" = torch.ops.aten.mul.Tensor(sum_149, 0.00010204081632653062)
    mul_1318: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_61, squeeze_61)
    mul_1319: "f32[48]" = torch.ops.aten.mul.Tensor(mul_1317, mul_1318);  mul_1317 = mul_1318 = None
    unsqueeze_1258: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_1319, 0);  mul_1319 = None
    unsqueeze_1259: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1258, 2);  unsqueeze_1258 = None
    unsqueeze_1260: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1259, 3);  unsqueeze_1259 = None
    mul_1320: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_61, primals_41);  primals_41 = None
    unsqueeze_1261: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_1320, 0);  mul_1320 = None
    unsqueeze_1262: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1261, 2);  unsqueeze_1261 = None
    unsqueeze_1263: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1262, 3);  unsqueeze_1262 = None
    mul_1321: "f32[8, 48, 35, 35]" = torch.ops.aten.mul.Tensor(sub_386, unsqueeze_1260);  sub_386 = unsqueeze_1260 = None
    sub_388: "f32[8, 48, 35, 35]" = torch.ops.aten.sub.Tensor(where_73, mul_1321);  where_73 = mul_1321 = None
    sub_389: "f32[8, 48, 35, 35]" = torch.ops.aten.sub.Tensor(sub_388, unsqueeze_1257);  sub_388 = unsqueeze_1257 = None
    mul_1322: "f32[8, 48, 35, 35]" = torch.ops.aten.mul.Tensor(sub_389, unsqueeze_1263);  sub_389 = unsqueeze_1263 = None
    mul_1323: "f32[48]" = torch.ops.aten.mul.Tensor(sum_149, squeeze_61);  sum_149 = squeeze_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_73 = torch.ops.aten.convolution_backward.default(mul_1322, cat_1, primals_209, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1322 = primals_209 = None
    getitem_415: "f32[8, 288, 35, 35]" = convolution_backward_73[0]
    getitem_416: "f32[48, 288, 1, 1]" = convolution_backward_73[1];  convolution_backward_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_497: "f32[8, 288, 35, 35]" = torch.ops.aten.add.Tensor(add_496, getitem_415);  add_496 = getitem_415 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    where_74: "f32[8, 64, 35, 35]" = torch.ops.aten.where.self(le_74, full_default, slice_39);  le_74 = slice_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_150: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_74, [0, 2, 3])
    sub_390: "f32[8, 64, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_1266);  convolution_19 = unsqueeze_1266 = None
    mul_1324: "f32[8, 64, 35, 35]" = torch.ops.aten.mul.Tensor(where_74, sub_390)
    sum_151: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1324, [0, 2, 3]);  mul_1324 = None
    mul_1325: "f32[64]" = torch.ops.aten.mul.Tensor(sum_150, 0.00010204081632653062)
    unsqueeze_1267: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1325, 0);  mul_1325 = None
    unsqueeze_1268: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1267, 2);  unsqueeze_1267 = None
    unsqueeze_1269: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1268, 3);  unsqueeze_1268 = None
    mul_1326: "f32[64]" = torch.ops.aten.mul.Tensor(sum_151, 0.00010204081632653062)
    mul_1327: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_58, squeeze_58)
    mul_1328: "f32[64]" = torch.ops.aten.mul.Tensor(mul_1326, mul_1327);  mul_1326 = mul_1327 = None
    unsqueeze_1270: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1328, 0);  mul_1328 = None
    unsqueeze_1271: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1270, 2);  unsqueeze_1270 = None
    unsqueeze_1272: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1271, 3);  unsqueeze_1271 = None
    mul_1329: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_58, primals_39);  primals_39 = None
    unsqueeze_1273: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1329, 0);  mul_1329 = None
    unsqueeze_1274: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1273, 2);  unsqueeze_1273 = None
    unsqueeze_1275: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1274, 3);  unsqueeze_1274 = None
    mul_1330: "f32[8, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_390, unsqueeze_1272);  sub_390 = unsqueeze_1272 = None
    sub_392: "f32[8, 64, 35, 35]" = torch.ops.aten.sub.Tensor(where_74, mul_1330);  where_74 = mul_1330 = None
    sub_393: "f32[8, 64, 35, 35]" = torch.ops.aten.sub.Tensor(sub_392, unsqueeze_1269);  sub_392 = unsqueeze_1269 = None
    mul_1331: "f32[8, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_393, unsqueeze_1275);  sub_393 = unsqueeze_1275 = None
    mul_1332: "f32[64]" = torch.ops.aten.mul.Tensor(sum_151, squeeze_58);  sum_151 = squeeze_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_74 = torch.ops.aten.convolution_backward.default(mul_1331, cat_1, primals_208, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1331 = cat_1 = primals_208 = None
    getitem_418: "f32[8, 288, 35, 35]" = convolution_backward_74[0]
    getitem_419: "f32[64, 288, 1, 1]" = convolution_backward_74[1];  convolution_backward_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_498: "f32[8, 288, 35, 35]" = torch.ops.aten.add.Tensor(add_497, getitem_418);  add_497 = getitem_418 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/inception_v3.py:56, code: return torch.cat(outputs, 1)
    slice_43: "f32[8, 64, 35, 35]" = torch.ops.aten.slice.Tensor(add_498, 1, 0, 64)
    slice_44: "f32[8, 64, 35, 35]" = torch.ops.aten.slice.Tensor(add_498, 1, 64, 128)
    slice_45: "f32[8, 96, 35, 35]" = torch.ops.aten.slice.Tensor(add_498, 1, 128, 224)
    slice_46: "f32[8, 64, 35, 35]" = torch.ops.aten.slice.Tensor(add_498, 1, 224, 288);  add_498 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    where_75: "f32[8, 64, 35, 35]" = torch.ops.aten.where.self(le_75, full_default, slice_46);  le_75 = slice_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_152: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_75, [0, 2, 3])
    sub_394: "f32[8, 64, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_1278);  convolution_18 = unsqueeze_1278 = None
    mul_1333: "f32[8, 64, 35, 35]" = torch.ops.aten.mul.Tensor(where_75, sub_394)
    sum_153: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1333, [0, 2, 3]);  mul_1333 = None
    mul_1334: "f32[64]" = torch.ops.aten.mul.Tensor(sum_152, 0.00010204081632653062)
    unsqueeze_1279: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1334, 0);  mul_1334 = None
    unsqueeze_1280: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1279, 2);  unsqueeze_1279 = None
    unsqueeze_1281: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1280, 3);  unsqueeze_1280 = None
    mul_1335: "f32[64]" = torch.ops.aten.mul.Tensor(sum_153, 0.00010204081632653062)
    mul_1336: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_55, squeeze_55)
    mul_1337: "f32[64]" = torch.ops.aten.mul.Tensor(mul_1335, mul_1336);  mul_1335 = mul_1336 = None
    unsqueeze_1282: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1337, 0);  mul_1337 = None
    unsqueeze_1283: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1282, 2);  unsqueeze_1282 = None
    unsqueeze_1284: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1283, 3);  unsqueeze_1283 = None
    mul_1338: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_55, primals_37);  primals_37 = None
    unsqueeze_1285: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1338, 0);  mul_1338 = None
    unsqueeze_1286: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1285, 2);  unsqueeze_1285 = None
    unsqueeze_1287: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1286, 3);  unsqueeze_1286 = None
    mul_1339: "f32[8, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_394, unsqueeze_1284);  sub_394 = unsqueeze_1284 = None
    sub_396: "f32[8, 64, 35, 35]" = torch.ops.aten.sub.Tensor(where_75, mul_1339);  where_75 = mul_1339 = None
    sub_397: "f32[8, 64, 35, 35]" = torch.ops.aten.sub.Tensor(sub_396, unsqueeze_1281);  sub_396 = unsqueeze_1281 = None
    mul_1340: "f32[8, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_397, unsqueeze_1287);  sub_397 = unsqueeze_1287 = None
    mul_1341: "f32[64]" = torch.ops.aten.mul.Tensor(sum_153, squeeze_55);  sum_153 = squeeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_75 = torch.ops.aten.convolution_backward.default(mul_1340, avg_pool2d_1, primals_207, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1340 = avg_pool2d_1 = primals_207 = None
    getitem_421: "f32[8, 256, 35, 35]" = convolution_backward_75[0]
    getitem_422: "f32[64, 256, 1, 1]" = convolution_backward_75[1];  convolution_backward_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/inception_v3.py:48, code: branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    avg_pool2d_backward_7: "f32[8, 256, 35, 35]" = torch.ops.aten.avg_pool2d_backward.default(getitem_421, cat, [3, 3], [1, 1], [1, 1], False, True, None);  getitem_421 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    where_76: "f32[8, 96, 35, 35]" = torch.ops.aten.where.self(le_76, full_default, slice_45);  le_76 = slice_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_154: "f32[96]" = torch.ops.aten.sum.dim_IntList(where_76, [0, 2, 3])
    sub_398: "f32[8, 96, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_1290);  convolution_17 = unsqueeze_1290 = None
    mul_1342: "f32[8, 96, 35, 35]" = torch.ops.aten.mul.Tensor(where_76, sub_398)
    sum_155: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_1342, [0, 2, 3]);  mul_1342 = None
    mul_1343: "f32[96]" = torch.ops.aten.mul.Tensor(sum_154, 0.00010204081632653062)
    unsqueeze_1291: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_1343, 0);  mul_1343 = None
    unsqueeze_1292: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1291, 2);  unsqueeze_1291 = None
    unsqueeze_1293: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1292, 3);  unsqueeze_1292 = None
    mul_1344: "f32[96]" = torch.ops.aten.mul.Tensor(sum_155, 0.00010204081632653062)
    mul_1345: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_52, squeeze_52)
    mul_1346: "f32[96]" = torch.ops.aten.mul.Tensor(mul_1344, mul_1345);  mul_1344 = mul_1345 = None
    unsqueeze_1294: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_1346, 0);  mul_1346 = None
    unsqueeze_1295: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1294, 2);  unsqueeze_1294 = None
    unsqueeze_1296: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1295, 3);  unsqueeze_1295 = None
    mul_1347: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_52, primals_35);  primals_35 = None
    unsqueeze_1297: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_1347, 0);  mul_1347 = None
    unsqueeze_1298: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1297, 2);  unsqueeze_1297 = None
    unsqueeze_1299: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1298, 3);  unsqueeze_1298 = None
    mul_1348: "f32[8, 96, 35, 35]" = torch.ops.aten.mul.Tensor(sub_398, unsqueeze_1296);  sub_398 = unsqueeze_1296 = None
    sub_400: "f32[8, 96, 35, 35]" = torch.ops.aten.sub.Tensor(where_76, mul_1348);  where_76 = mul_1348 = None
    sub_401: "f32[8, 96, 35, 35]" = torch.ops.aten.sub.Tensor(sub_400, unsqueeze_1293);  sub_400 = unsqueeze_1293 = None
    mul_1349: "f32[8, 96, 35, 35]" = torch.ops.aten.mul.Tensor(sub_401, unsqueeze_1299);  sub_401 = unsqueeze_1299 = None
    mul_1350: "f32[96]" = torch.ops.aten.mul.Tensor(sum_155, squeeze_52);  sum_155 = squeeze_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_76 = torch.ops.aten.convolution_backward.default(mul_1349, relu_16, primals_206, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1349 = primals_206 = None
    getitem_424: "f32[8, 96, 35, 35]" = convolution_backward_76[0]
    getitem_425: "f32[96, 96, 3, 3]" = convolution_backward_76[1];  convolution_backward_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_326: "f32[8, 96, 35, 35]" = torch.ops.aten.alias.default(relu_16);  relu_16 = None
    alias_327: "f32[8, 96, 35, 35]" = torch.ops.aten.alias.default(alias_326);  alias_326 = None
    le_77: "b8[8, 96, 35, 35]" = torch.ops.aten.le.Scalar(alias_327, 0);  alias_327 = None
    where_77: "f32[8, 96, 35, 35]" = torch.ops.aten.where.self(le_77, full_default, getitem_424);  le_77 = getitem_424 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_156: "f32[96]" = torch.ops.aten.sum.dim_IntList(where_77, [0, 2, 3])
    sub_402: "f32[8, 96, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_1302);  convolution_16 = unsqueeze_1302 = None
    mul_1351: "f32[8, 96, 35, 35]" = torch.ops.aten.mul.Tensor(where_77, sub_402)
    sum_157: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_1351, [0, 2, 3]);  mul_1351 = None
    mul_1352: "f32[96]" = torch.ops.aten.mul.Tensor(sum_156, 0.00010204081632653062)
    unsqueeze_1303: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_1352, 0);  mul_1352 = None
    unsqueeze_1304: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1303, 2);  unsqueeze_1303 = None
    unsqueeze_1305: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1304, 3);  unsqueeze_1304 = None
    mul_1353: "f32[96]" = torch.ops.aten.mul.Tensor(sum_157, 0.00010204081632653062)
    mul_1354: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_49, squeeze_49)
    mul_1355: "f32[96]" = torch.ops.aten.mul.Tensor(mul_1353, mul_1354);  mul_1353 = mul_1354 = None
    unsqueeze_1306: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_1355, 0);  mul_1355 = None
    unsqueeze_1307: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1306, 2);  unsqueeze_1306 = None
    unsqueeze_1308: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1307, 3);  unsqueeze_1307 = None
    mul_1356: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_49, primals_33);  primals_33 = None
    unsqueeze_1309: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_1356, 0);  mul_1356 = None
    unsqueeze_1310: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1309, 2);  unsqueeze_1309 = None
    unsqueeze_1311: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1310, 3);  unsqueeze_1310 = None
    mul_1357: "f32[8, 96, 35, 35]" = torch.ops.aten.mul.Tensor(sub_402, unsqueeze_1308);  sub_402 = unsqueeze_1308 = None
    sub_404: "f32[8, 96, 35, 35]" = torch.ops.aten.sub.Tensor(where_77, mul_1357);  where_77 = mul_1357 = None
    sub_405: "f32[8, 96, 35, 35]" = torch.ops.aten.sub.Tensor(sub_404, unsqueeze_1305);  sub_404 = unsqueeze_1305 = None
    mul_1358: "f32[8, 96, 35, 35]" = torch.ops.aten.mul.Tensor(sub_405, unsqueeze_1311);  sub_405 = unsqueeze_1311 = None
    mul_1359: "f32[96]" = torch.ops.aten.mul.Tensor(sum_157, squeeze_49);  sum_157 = squeeze_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_77 = torch.ops.aten.convolution_backward.default(mul_1358, relu_15, primals_205, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1358 = primals_205 = None
    getitem_427: "f32[8, 64, 35, 35]" = convolution_backward_77[0]
    getitem_428: "f32[96, 64, 3, 3]" = convolution_backward_77[1];  convolution_backward_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_329: "f32[8, 64, 35, 35]" = torch.ops.aten.alias.default(relu_15);  relu_15 = None
    alias_330: "f32[8, 64, 35, 35]" = torch.ops.aten.alias.default(alias_329);  alias_329 = None
    le_78: "b8[8, 64, 35, 35]" = torch.ops.aten.le.Scalar(alias_330, 0);  alias_330 = None
    where_78: "f32[8, 64, 35, 35]" = torch.ops.aten.where.self(le_78, full_default, getitem_427);  le_78 = getitem_427 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_158: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_78, [0, 2, 3])
    sub_406: "f32[8, 64, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_1314);  convolution_15 = unsqueeze_1314 = None
    mul_1360: "f32[8, 64, 35, 35]" = torch.ops.aten.mul.Tensor(where_78, sub_406)
    sum_159: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1360, [0, 2, 3]);  mul_1360 = None
    mul_1361: "f32[64]" = torch.ops.aten.mul.Tensor(sum_158, 0.00010204081632653062)
    unsqueeze_1315: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1361, 0);  mul_1361 = None
    unsqueeze_1316: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1315, 2);  unsqueeze_1315 = None
    unsqueeze_1317: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1316, 3);  unsqueeze_1316 = None
    mul_1362: "f32[64]" = torch.ops.aten.mul.Tensor(sum_159, 0.00010204081632653062)
    mul_1363: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_46, squeeze_46)
    mul_1364: "f32[64]" = torch.ops.aten.mul.Tensor(mul_1362, mul_1363);  mul_1362 = mul_1363 = None
    unsqueeze_1318: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1364, 0);  mul_1364 = None
    unsqueeze_1319: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1318, 2);  unsqueeze_1318 = None
    unsqueeze_1320: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1319, 3);  unsqueeze_1319 = None
    mul_1365: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_46, primals_31);  primals_31 = None
    unsqueeze_1321: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1365, 0);  mul_1365 = None
    unsqueeze_1322: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1321, 2);  unsqueeze_1321 = None
    unsqueeze_1323: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1322, 3);  unsqueeze_1322 = None
    mul_1366: "f32[8, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_406, unsqueeze_1320);  sub_406 = unsqueeze_1320 = None
    sub_408: "f32[8, 64, 35, 35]" = torch.ops.aten.sub.Tensor(where_78, mul_1366);  where_78 = mul_1366 = None
    sub_409: "f32[8, 64, 35, 35]" = torch.ops.aten.sub.Tensor(sub_408, unsqueeze_1317);  sub_408 = unsqueeze_1317 = None
    mul_1367: "f32[8, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_409, unsqueeze_1323);  sub_409 = unsqueeze_1323 = None
    mul_1368: "f32[64]" = torch.ops.aten.mul.Tensor(sum_159, squeeze_46);  sum_159 = squeeze_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_78 = torch.ops.aten.convolution_backward.default(mul_1367, cat, primals_204, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1367 = primals_204 = None
    getitem_430: "f32[8, 256, 35, 35]" = convolution_backward_78[0]
    getitem_431: "f32[64, 256, 1, 1]" = convolution_backward_78[1];  convolution_backward_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_499: "f32[8, 256, 35, 35]" = torch.ops.aten.add.Tensor(avg_pool2d_backward_7, getitem_430);  avg_pool2d_backward_7 = getitem_430 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    where_79: "f32[8, 64, 35, 35]" = torch.ops.aten.where.self(le_79, full_default, slice_44);  le_79 = slice_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_160: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_79, [0, 2, 3])
    sub_410: "f32[8, 64, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_1326);  convolution_14 = unsqueeze_1326 = None
    mul_1369: "f32[8, 64, 35, 35]" = torch.ops.aten.mul.Tensor(where_79, sub_410)
    sum_161: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1369, [0, 2, 3]);  mul_1369 = None
    mul_1370: "f32[64]" = torch.ops.aten.mul.Tensor(sum_160, 0.00010204081632653062)
    unsqueeze_1327: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1370, 0);  mul_1370 = None
    unsqueeze_1328: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1327, 2);  unsqueeze_1327 = None
    unsqueeze_1329: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1328, 3);  unsqueeze_1328 = None
    mul_1371: "f32[64]" = torch.ops.aten.mul.Tensor(sum_161, 0.00010204081632653062)
    mul_1372: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
    mul_1373: "f32[64]" = torch.ops.aten.mul.Tensor(mul_1371, mul_1372);  mul_1371 = mul_1372 = None
    unsqueeze_1330: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1373, 0);  mul_1373 = None
    unsqueeze_1331: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1330, 2);  unsqueeze_1330 = None
    unsqueeze_1332: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1331, 3);  unsqueeze_1331 = None
    mul_1374: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_43, primals_29);  primals_29 = None
    unsqueeze_1333: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1374, 0);  mul_1374 = None
    unsqueeze_1334: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1333, 2);  unsqueeze_1333 = None
    unsqueeze_1335: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1334, 3);  unsqueeze_1334 = None
    mul_1375: "f32[8, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_410, unsqueeze_1332);  sub_410 = unsqueeze_1332 = None
    sub_412: "f32[8, 64, 35, 35]" = torch.ops.aten.sub.Tensor(where_79, mul_1375);  where_79 = mul_1375 = None
    sub_413: "f32[8, 64, 35, 35]" = torch.ops.aten.sub.Tensor(sub_412, unsqueeze_1329);  sub_412 = unsqueeze_1329 = None
    mul_1376: "f32[8, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_413, unsqueeze_1335);  sub_413 = unsqueeze_1335 = None
    mul_1377: "f32[64]" = torch.ops.aten.mul.Tensor(sum_161, squeeze_43);  sum_161 = squeeze_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_79 = torch.ops.aten.convolution_backward.default(mul_1376, relu_13, primals_203, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1376 = primals_203 = None
    getitem_433: "f32[8, 48, 35, 35]" = convolution_backward_79[0]
    getitem_434: "f32[64, 48, 5, 5]" = convolution_backward_79[1];  convolution_backward_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_335: "f32[8, 48, 35, 35]" = torch.ops.aten.alias.default(relu_13);  relu_13 = None
    alias_336: "f32[8, 48, 35, 35]" = torch.ops.aten.alias.default(alias_335);  alias_335 = None
    le_80: "b8[8, 48, 35, 35]" = torch.ops.aten.le.Scalar(alias_336, 0);  alias_336 = None
    where_80: "f32[8, 48, 35, 35]" = torch.ops.aten.where.self(le_80, full_default, getitem_433);  le_80 = getitem_433 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_162: "f32[48]" = torch.ops.aten.sum.dim_IntList(where_80, [0, 2, 3])
    sub_414: "f32[8, 48, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_1338);  convolution_13 = unsqueeze_1338 = None
    mul_1378: "f32[8, 48, 35, 35]" = torch.ops.aten.mul.Tensor(where_80, sub_414)
    sum_163: "f32[48]" = torch.ops.aten.sum.dim_IntList(mul_1378, [0, 2, 3]);  mul_1378 = None
    mul_1379: "f32[48]" = torch.ops.aten.mul.Tensor(sum_162, 0.00010204081632653062)
    unsqueeze_1339: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_1379, 0);  mul_1379 = None
    unsqueeze_1340: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1339, 2);  unsqueeze_1339 = None
    unsqueeze_1341: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1340, 3);  unsqueeze_1340 = None
    mul_1380: "f32[48]" = torch.ops.aten.mul.Tensor(sum_163, 0.00010204081632653062)
    mul_1381: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_40, squeeze_40)
    mul_1382: "f32[48]" = torch.ops.aten.mul.Tensor(mul_1380, mul_1381);  mul_1380 = mul_1381 = None
    unsqueeze_1342: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_1382, 0);  mul_1382 = None
    unsqueeze_1343: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1342, 2);  unsqueeze_1342 = None
    unsqueeze_1344: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1343, 3);  unsqueeze_1343 = None
    mul_1383: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_40, primals_27);  primals_27 = None
    unsqueeze_1345: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_1383, 0);  mul_1383 = None
    unsqueeze_1346: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1345, 2);  unsqueeze_1345 = None
    unsqueeze_1347: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1346, 3);  unsqueeze_1346 = None
    mul_1384: "f32[8, 48, 35, 35]" = torch.ops.aten.mul.Tensor(sub_414, unsqueeze_1344);  sub_414 = unsqueeze_1344 = None
    sub_416: "f32[8, 48, 35, 35]" = torch.ops.aten.sub.Tensor(where_80, mul_1384);  where_80 = mul_1384 = None
    sub_417: "f32[8, 48, 35, 35]" = torch.ops.aten.sub.Tensor(sub_416, unsqueeze_1341);  sub_416 = unsqueeze_1341 = None
    mul_1385: "f32[8, 48, 35, 35]" = torch.ops.aten.mul.Tensor(sub_417, unsqueeze_1347);  sub_417 = unsqueeze_1347 = None
    mul_1386: "f32[48]" = torch.ops.aten.mul.Tensor(sum_163, squeeze_40);  sum_163 = squeeze_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_80 = torch.ops.aten.convolution_backward.default(mul_1385, cat, primals_202, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1385 = primals_202 = None
    getitem_436: "f32[8, 256, 35, 35]" = convolution_backward_80[0]
    getitem_437: "f32[48, 256, 1, 1]" = convolution_backward_80[1];  convolution_backward_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_500: "f32[8, 256, 35, 35]" = torch.ops.aten.add.Tensor(add_499, getitem_436);  add_499 = getitem_436 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    where_81: "f32[8, 64, 35, 35]" = torch.ops.aten.where.self(le_81, full_default, slice_43);  le_81 = slice_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_164: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_81, [0, 2, 3])
    sub_418: "f32[8, 64, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_1350);  convolution_12 = unsqueeze_1350 = None
    mul_1387: "f32[8, 64, 35, 35]" = torch.ops.aten.mul.Tensor(where_81, sub_418)
    sum_165: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1387, [0, 2, 3]);  mul_1387 = None
    mul_1388: "f32[64]" = torch.ops.aten.mul.Tensor(sum_164, 0.00010204081632653062)
    unsqueeze_1351: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1388, 0);  mul_1388 = None
    unsqueeze_1352: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1351, 2);  unsqueeze_1351 = None
    unsqueeze_1353: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1352, 3);  unsqueeze_1352 = None
    mul_1389: "f32[64]" = torch.ops.aten.mul.Tensor(sum_165, 0.00010204081632653062)
    mul_1390: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
    mul_1391: "f32[64]" = torch.ops.aten.mul.Tensor(mul_1389, mul_1390);  mul_1389 = mul_1390 = None
    unsqueeze_1354: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1391, 0);  mul_1391 = None
    unsqueeze_1355: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1354, 2);  unsqueeze_1354 = None
    unsqueeze_1356: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1355, 3);  unsqueeze_1355 = None
    mul_1392: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_37, primals_25);  primals_25 = None
    unsqueeze_1357: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1392, 0);  mul_1392 = None
    unsqueeze_1358: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1357, 2);  unsqueeze_1357 = None
    unsqueeze_1359: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1358, 3);  unsqueeze_1358 = None
    mul_1393: "f32[8, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_418, unsqueeze_1356);  sub_418 = unsqueeze_1356 = None
    sub_420: "f32[8, 64, 35, 35]" = torch.ops.aten.sub.Tensor(where_81, mul_1393);  where_81 = mul_1393 = None
    sub_421: "f32[8, 64, 35, 35]" = torch.ops.aten.sub.Tensor(sub_420, unsqueeze_1353);  sub_420 = unsqueeze_1353 = None
    mul_1394: "f32[8, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_421, unsqueeze_1359);  sub_421 = unsqueeze_1359 = None
    mul_1395: "f32[64]" = torch.ops.aten.mul.Tensor(sum_165, squeeze_37);  sum_165 = squeeze_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_81 = torch.ops.aten.convolution_backward.default(mul_1394, cat, primals_201, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1394 = cat = primals_201 = None
    getitem_439: "f32[8, 256, 35, 35]" = convolution_backward_81[0]
    getitem_440: "f32[64, 256, 1, 1]" = convolution_backward_81[1];  convolution_backward_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_501: "f32[8, 256, 35, 35]" = torch.ops.aten.add.Tensor(add_500, getitem_439);  add_500 = getitem_439 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/inception_v3.py:56, code: return torch.cat(outputs, 1)
    slice_47: "f32[8, 64, 35, 35]" = torch.ops.aten.slice.Tensor(add_501, 1, 0, 64)
    slice_48: "f32[8, 64, 35, 35]" = torch.ops.aten.slice.Tensor(add_501, 1, 64, 128)
    slice_49: "f32[8, 96, 35, 35]" = torch.ops.aten.slice.Tensor(add_501, 1, 128, 224)
    slice_50: "f32[8, 32, 35, 35]" = torch.ops.aten.slice.Tensor(add_501, 1, 224, 256);  add_501 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    where_82: "f32[8, 32, 35, 35]" = torch.ops.aten.where.self(le_82, full_default, slice_50);  le_82 = slice_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_166: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_82, [0, 2, 3])
    sub_422: "f32[8, 32, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_1362);  convolution_11 = unsqueeze_1362 = None
    mul_1396: "f32[8, 32, 35, 35]" = torch.ops.aten.mul.Tensor(where_82, sub_422)
    sum_167: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_1396, [0, 2, 3]);  mul_1396 = None
    mul_1397: "f32[32]" = torch.ops.aten.mul.Tensor(sum_166, 0.00010204081632653062)
    unsqueeze_1363: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1397, 0);  mul_1397 = None
    unsqueeze_1364: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1363, 2);  unsqueeze_1363 = None
    unsqueeze_1365: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1364, 3);  unsqueeze_1364 = None
    mul_1398: "f32[32]" = torch.ops.aten.mul.Tensor(sum_167, 0.00010204081632653062)
    mul_1399: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_34, squeeze_34)
    mul_1400: "f32[32]" = torch.ops.aten.mul.Tensor(mul_1398, mul_1399);  mul_1398 = mul_1399 = None
    unsqueeze_1366: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1400, 0);  mul_1400 = None
    unsqueeze_1367: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1366, 2);  unsqueeze_1366 = None
    unsqueeze_1368: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1367, 3);  unsqueeze_1367 = None
    mul_1401: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_34, primals_23);  primals_23 = None
    unsqueeze_1369: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1401, 0);  mul_1401 = None
    unsqueeze_1370: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1369, 2);  unsqueeze_1369 = None
    unsqueeze_1371: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1370, 3);  unsqueeze_1370 = None
    mul_1402: "f32[8, 32, 35, 35]" = torch.ops.aten.mul.Tensor(sub_422, unsqueeze_1368);  sub_422 = unsqueeze_1368 = None
    sub_424: "f32[8, 32, 35, 35]" = torch.ops.aten.sub.Tensor(where_82, mul_1402);  where_82 = mul_1402 = None
    sub_425: "f32[8, 32, 35, 35]" = torch.ops.aten.sub.Tensor(sub_424, unsqueeze_1365);  sub_424 = unsqueeze_1365 = None
    mul_1403: "f32[8, 32, 35, 35]" = torch.ops.aten.mul.Tensor(sub_425, unsqueeze_1371);  sub_425 = unsqueeze_1371 = None
    mul_1404: "f32[32]" = torch.ops.aten.mul.Tensor(sum_167, squeeze_34);  sum_167 = squeeze_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_82 = torch.ops.aten.convolution_backward.default(mul_1403, avg_pool2d, primals_200, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1403 = avg_pool2d = primals_200 = None
    getitem_442: "f32[8, 192, 35, 35]" = convolution_backward_82[0]
    getitem_443: "f32[32, 192, 1, 1]" = convolution_backward_82[1];  convolution_backward_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/inception_v3.py:48, code: branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    avg_pool2d_backward_8: "f32[8, 192, 35, 35]" = torch.ops.aten.avg_pool2d_backward.default(getitem_442, getitem_12, [3, 3], [1, 1], [1, 1], False, True, None);  getitem_442 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    where_83: "f32[8, 96, 35, 35]" = torch.ops.aten.where.self(le_83, full_default, slice_49);  le_83 = slice_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_168: "f32[96]" = torch.ops.aten.sum.dim_IntList(where_83, [0, 2, 3])
    sub_426: "f32[8, 96, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_1374);  convolution_10 = unsqueeze_1374 = None
    mul_1405: "f32[8, 96, 35, 35]" = torch.ops.aten.mul.Tensor(where_83, sub_426)
    sum_169: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_1405, [0, 2, 3]);  mul_1405 = None
    mul_1406: "f32[96]" = torch.ops.aten.mul.Tensor(sum_168, 0.00010204081632653062)
    unsqueeze_1375: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_1406, 0);  mul_1406 = None
    unsqueeze_1376: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1375, 2);  unsqueeze_1375 = None
    unsqueeze_1377: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1376, 3);  unsqueeze_1376 = None
    mul_1407: "f32[96]" = torch.ops.aten.mul.Tensor(sum_169, 0.00010204081632653062)
    mul_1408: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
    mul_1409: "f32[96]" = torch.ops.aten.mul.Tensor(mul_1407, mul_1408);  mul_1407 = mul_1408 = None
    unsqueeze_1378: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_1409, 0);  mul_1409 = None
    unsqueeze_1379: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1378, 2);  unsqueeze_1378 = None
    unsqueeze_1380: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1379, 3);  unsqueeze_1379 = None
    mul_1410: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_31, primals_21);  primals_21 = None
    unsqueeze_1381: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_1410, 0);  mul_1410 = None
    unsqueeze_1382: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1381, 2);  unsqueeze_1381 = None
    unsqueeze_1383: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1382, 3);  unsqueeze_1382 = None
    mul_1411: "f32[8, 96, 35, 35]" = torch.ops.aten.mul.Tensor(sub_426, unsqueeze_1380);  sub_426 = unsqueeze_1380 = None
    sub_428: "f32[8, 96, 35, 35]" = torch.ops.aten.sub.Tensor(where_83, mul_1411);  where_83 = mul_1411 = None
    sub_429: "f32[8, 96, 35, 35]" = torch.ops.aten.sub.Tensor(sub_428, unsqueeze_1377);  sub_428 = unsqueeze_1377 = None
    mul_1412: "f32[8, 96, 35, 35]" = torch.ops.aten.mul.Tensor(sub_429, unsqueeze_1383);  sub_429 = unsqueeze_1383 = None
    mul_1413: "f32[96]" = torch.ops.aten.mul.Tensor(sum_169, squeeze_31);  sum_169 = squeeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_83 = torch.ops.aten.convolution_backward.default(mul_1412, relu_9, primals_199, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1412 = primals_199 = None
    getitem_445: "f32[8, 96, 35, 35]" = convolution_backward_83[0]
    getitem_446: "f32[96, 96, 3, 3]" = convolution_backward_83[1];  convolution_backward_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_347: "f32[8, 96, 35, 35]" = torch.ops.aten.alias.default(relu_9);  relu_9 = None
    alias_348: "f32[8, 96, 35, 35]" = torch.ops.aten.alias.default(alias_347);  alias_347 = None
    le_84: "b8[8, 96, 35, 35]" = torch.ops.aten.le.Scalar(alias_348, 0);  alias_348 = None
    where_84: "f32[8, 96, 35, 35]" = torch.ops.aten.where.self(le_84, full_default, getitem_445);  le_84 = getitem_445 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_170: "f32[96]" = torch.ops.aten.sum.dim_IntList(where_84, [0, 2, 3])
    sub_430: "f32[8, 96, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_1386);  convolution_9 = unsqueeze_1386 = None
    mul_1414: "f32[8, 96, 35, 35]" = torch.ops.aten.mul.Tensor(where_84, sub_430)
    sum_171: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_1414, [0, 2, 3]);  mul_1414 = None
    mul_1415: "f32[96]" = torch.ops.aten.mul.Tensor(sum_170, 0.00010204081632653062)
    unsqueeze_1387: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_1415, 0);  mul_1415 = None
    unsqueeze_1388: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1387, 2);  unsqueeze_1387 = None
    unsqueeze_1389: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1388, 3);  unsqueeze_1388 = None
    mul_1416: "f32[96]" = torch.ops.aten.mul.Tensor(sum_171, 0.00010204081632653062)
    mul_1417: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_28, squeeze_28)
    mul_1418: "f32[96]" = torch.ops.aten.mul.Tensor(mul_1416, mul_1417);  mul_1416 = mul_1417 = None
    unsqueeze_1390: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_1418, 0);  mul_1418 = None
    unsqueeze_1391: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1390, 2);  unsqueeze_1390 = None
    unsqueeze_1392: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1391, 3);  unsqueeze_1391 = None
    mul_1419: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_28, primals_19);  primals_19 = None
    unsqueeze_1393: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_1419, 0);  mul_1419 = None
    unsqueeze_1394: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1393, 2);  unsqueeze_1393 = None
    unsqueeze_1395: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1394, 3);  unsqueeze_1394 = None
    mul_1420: "f32[8, 96, 35, 35]" = torch.ops.aten.mul.Tensor(sub_430, unsqueeze_1392);  sub_430 = unsqueeze_1392 = None
    sub_432: "f32[8, 96, 35, 35]" = torch.ops.aten.sub.Tensor(where_84, mul_1420);  where_84 = mul_1420 = None
    sub_433: "f32[8, 96, 35, 35]" = torch.ops.aten.sub.Tensor(sub_432, unsqueeze_1389);  sub_432 = unsqueeze_1389 = None
    mul_1421: "f32[8, 96, 35, 35]" = torch.ops.aten.mul.Tensor(sub_433, unsqueeze_1395);  sub_433 = unsqueeze_1395 = None
    mul_1422: "f32[96]" = torch.ops.aten.mul.Tensor(sum_171, squeeze_28);  sum_171 = squeeze_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_84 = torch.ops.aten.convolution_backward.default(mul_1421, relu_8, primals_198, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1421 = primals_198 = None
    getitem_448: "f32[8, 64, 35, 35]" = convolution_backward_84[0]
    getitem_449: "f32[96, 64, 3, 3]" = convolution_backward_84[1];  convolution_backward_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_350: "f32[8, 64, 35, 35]" = torch.ops.aten.alias.default(relu_8);  relu_8 = None
    alias_351: "f32[8, 64, 35, 35]" = torch.ops.aten.alias.default(alias_350);  alias_350 = None
    le_85: "b8[8, 64, 35, 35]" = torch.ops.aten.le.Scalar(alias_351, 0);  alias_351 = None
    where_85: "f32[8, 64, 35, 35]" = torch.ops.aten.where.self(le_85, full_default, getitem_448);  le_85 = getitem_448 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_172: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_85, [0, 2, 3])
    sub_434: "f32[8, 64, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_1398);  convolution_8 = unsqueeze_1398 = None
    mul_1423: "f32[8, 64, 35, 35]" = torch.ops.aten.mul.Tensor(where_85, sub_434)
    sum_173: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1423, [0, 2, 3]);  mul_1423 = None
    mul_1424: "f32[64]" = torch.ops.aten.mul.Tensor(sum_172, 0.00010204081632653062)
    unsqueeze_1399: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1424, 0);  mul_1424 = None
    unsqueeze_1400: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1399, 2);  unsqueeze_1399 = None
    unsqueeze_1401: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1400, 3);  unsqueeze_1400 = None
    mul_1425: "f32[64]" = torch.ops.aten.mul.Tensor(sum_173, 0.00010204081632653062)
    mul_1426: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
    mul_1427: "f32[64]" = torch.ops.aten.mul.Tensor(mul_1425, mul_1426);  mul_1425 = mul_1426 = None
    unsqueeze_1402: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1427, 0);  mul_1427 = None
    unsqueeze_1403: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1402, 2);  unsqueeze_1402 = None
    unsqueeze_1404: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1403, 3);  unsqueeze_1403 = None
    mul_1428: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_25, primals_17);  primals_17 = None
    unsqueeze_1405: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1428, 0);  mul_1428 = None
    unsqueeze_1406: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1405, 2);  unsqueeze_1405 = None
    unsqueeze_1407: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1406, 3);  unsqueeze_1406 = None
    mul_1429: "f32[8, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_434, unsqueeze_1404);  sub_434 = unsqueeze_1404 = None
    sub_436: "f32[8, 64, 35, 35]" = torch.ops.aten.sub.Tensor(where_85, mul_1429);  where_85 = mul_1429 = None
    sub_437: "f32[8, 64, 35, 35]" = torch.ops.aten.sub.Tensor(sub_436, unsqueeze_1401);  sub_436 = unsqueeze_1401 = None
    mul_1430: "f32[8, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_437, unsqueeze_1407);  sub_437 = unsqueeze_1407 = None
    mul_1431: "f32[64]" = torch.ops.aten.mul.Tensor(sum_173, squeeze_25);  sum_173 = squeeze_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_85 = torch.ops.aten.convolution_backward.default(mul_1430, getitem_12, primals_197, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1430 = primals_197 = None
    getitem_451: "f32[8, 192, 35, 35]" = convolution_backward_85[0]
    getitem_452: "f32[64, 192, 1, 1]" = convolution_backward_85[1];  convolution_backward_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_502: "f32[8, 192, 35, 35]" = torch.ops.aten.add.Tensor(avg_pool2d_backward_8, getitem_451);  avg_pool2d_backward_8 = getitem_451 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    where_86: "f32[8, 64, 35, 35]" = torch.ops.aten.where.self(le_86, full_default, slice_48);  le_86 = slice_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_174: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_86, [0, 2, 3])
    sub_438: "f32[8, 64, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_1410);  convolution_7 = unsqueeze_1410 = None
    mul_1432: "f32[8, 64, 35, 35]" = torch.ops.aten.mul.Tensor(where_86, sub_438)
    sum_175: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1432, [0, 2, 3]);  mul_1432 = None
    mul_1433: "f32[64]" = torch.ops.aten.mul.Tensor(sum_174, 0.00010204081632653062)
    unsqueeze_1411: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1433, 0);  mul_1433 = None
    unsqueeze_1412: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1411, 2);  unsqueeze_1411 = None
    unsqueeze_1413: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1412, 3);  unsqueeze_1412 = None
    mul_1434: "f32[64]" = torch.ops.aten.mul.Tensor(sum_175, 0.00010204081632653062)
    mul_1435: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_22, squeeze_22)
    mul_1436: "f32[64]" = torch.ops.aten.mul.Tensor(mul_1434, mul_1435);  mul_1434 = mul_1435 = None
    unsqueeze_1414: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1436, 0);  mul_1436 = None
    unsqueeze_1415: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1414, 2);  unsqueeze_1414 = None
    unsqueeze_1416: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1415, 3);  unsqueeze_1415 = None
    mul_1437: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_22, primals_15);  primals_15 = None
    unsqueeze_1417: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1437, 0);  mul_1437 = None
    unsqueeze_1418: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1417, 2);  unsqueeze_1417 = None
    unsqueeze_1419: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1418, 3);  unsqueeze_1418 = None
    mul_1438: "f32[8, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_438, unsqueeze_1416);  sub_438 = unsqueeze_1416 = None
    sub_440: "f32[8, 64, 35, 35]" = torch.ops.aten.sub.Tensor(where_86, mul_1438);  where_86 = mul_1438 = None
    sub_441: "f32[8, 64, 35, 35]" = torch.ops.aten.sub.Tensor(sub_440, unsqueeze_1413);  sub_440 = unsqueeze_1413 = None
    mul_1439: "f32[8, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_441, unsqueeze_1419);  sub_441 = unsqueeze_1419 = None
    mul_1440: "f32[64]" = torch.ops.aten.mul.Tensor(sum_175, squeeze_22);  sum_175 = squeeze_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_86 = torch.ops.aten.convolution_backward.default(mul_1439, relu_6, primals_196, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1439 = primals_196 = None
    getitem_454: "f32[8, 48, 35, 35]" = convolution_backward_86[0]
    getitem_455: "f32[64, 48, 5, 5]" = convolution_backward_86[1];  convolution_backward_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_356: "f32[8, 48, 35, 35]" = torch.ops.aten.alias.default(relu_6);  relu_6 = None
    alias_357: "f32[8, 48, 35, 35]" = torch.ops.aten.alias.default(alias_356);  alias_356 = None
    le_87: "b8[8, 48, 35, 35]" = torch.ops.aten.le.Scalar(alias_357, 0);  alias_357 = None
    where_87: "f32[8, 48, 35, 35]" = torch.ops.aten.where.self(le_87, full_default, getitem_454);  le_87 = getitem_454 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_176: "f32[48]" = torch.ops.aten.sum.dim_IntList(where_87, [0, 2, 3])
    sub_442: "f32[8, 48, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_1422);  convolution_6 = unsqueeze_1422 = None
    mul_1441: "f32[8, 48, 35, 35]" = torch.ops.aten.mul.Tensor(where_87, sub_442)
    sum_177: "f32[48]" = torch.ops.aten.sum.dim_IntList(mul_1441, [0, 2, 3]);  mul_1441 = None
    mul_1442: "f32[48]" = torch.ops.aten.mul.Tensor(sum_176, 0.00010204081632653062)
    unsqueeze_1423: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_1442, 0);  mul_1442 = None
    unsqueeze_1424: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1423, 2);  unsqueeze_1423 = None
    unsqueeze_1425: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1424, 3);  unsqueeze_1424 = None
    mul_1443: "f32[48]" = torch.ops.aten.mul.Tensor(sum_177, 0.00010204081632653062)
    mul_1444: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_19, squeeze_19)
    mul_1445: "f32[48]" = torch.ops.aten.mul.Tensor(mul_1443, mul_1444);  mul_1443 = mul_1444 = None
    unsqueeze_1426: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_1445, 0);  mul_1445 = None
    unsqueeze_1427: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1426, 2);  unsqueeze_1426 = None
    unsqueeze_1428: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1427, 3);  unsqueeze_1427 = None
    mul_1446: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_19, primals_13);  primals_13 = None
    unsqueeze_1429: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_1446, 0);  mul_1446 = None
    unsqueeze_1430: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1429, 2);  unsqueeze_1429 = None
    unsqueeze_1431: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1430, 3);  unsqueeze_1430 = None
    mul_1447: "f32[8, 48, 35, 35]" = torch.ops.aten.mul.Tensor(sub_442, unsqueeze_1428);  sub_442 = unsqueeze_1428 = None
    sub_444: "f32[8, 48, 35, 35]" = torch.ops.aten.sub.Tensor(where_87, mul_1447);  where_87 = mul_1447 = None
    sub_445: "f32[8, 48, 35, 35]" = torch.ops.aten.sub.Tensor(sub_444, unsqueeze_1425);  sub_444 = unsqueeze_1425 = None
    mul_1448: "f32[8, 48, 35, 35]" = torch.ops.aten.mul.Tensor(sub_445, unsqueeze_1431);  sub_445 = unsqueeze_1431 = None
    mul_1449: "f32[48]" = torch.ops.aten.mul.Tensor(sum_177, squeeze_19);  sum_177 = squeeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_87 = torch.ops.aten.convolution_backward.default(mul_1448, getitem_12, primals_195, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1448 = primals_195 = None
    getitem_457: "f32[8, 192, 35, 35]" = convolution_backward_87[0]
    getitem_458: "f32[48, 192, 1, 1]" = convolution_backward_87[1];  convolution_backward_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_503: "f32[8, 192, 35, 35]" = torch.ops.aten.add.Tensor(add_502, getitem_457);  add_502 = getitem_457 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    where_88: "f32[8, 64, 35, 35]" = torch.ops.aten.where.self(le_88, full_default, slice_47);  le_88 = slice_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_178: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_88, [0, 2, 3])
    sub_446: "f32[8, 64, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_1434);  convolution_5 = unsqueeze_1434 = None
    mul_1450: "f32[8, 64, 35, 35]" = torch.ops.aten.mul.Tensor(where_88, sub_446)
    sum_179: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1450, [0, 2, 3]);  mul_1450 = None
    mul_1451: "f32[64]" = torch.ops.aten.mul.Tensor(sum_178, 0.00010204081632653062)
    unsqueeze_1435: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1451, 0);  mul_1451 = None
    unsqueeze_1436: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1435, 2);  unsqueeze_1435 = None
    unsqueeze_1437: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1436, 3);  unsqueeze_1436 = None
    mul_1452: "f32[64]" = torch.ops.aten.mul.Tensor(sum_179, 0.00010204081632653062)
    mul_1453: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_16, squeeze_16)
    mul_1454: "f32[64]" = torch.ops.aten.mul.Tensor(mul_1452, mul_1453);  mul_1452 = mul_1453 = None
    unsqueeze_1438: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1454, 0);  mul_1454 = None
    unsqueeze_1439: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1438, 2);  unsqueeze_1438 = None
    unsqueeze_1440: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1439, 3);  unsqueeze_1439 = None
    mul_1455: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_16, primals_11);  primals_11 = None
    unsqueeze_1441: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1455, 0);  mul_1455 = None
    unsqueeze_1442: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1441, 2);  unsqueeze_1441 = None
    unsqueeze_1443: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1442, 3);  unsqueeze_1442 = None
    mul_1456: "f32[8, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_446, unsqueeze_1440);  sub_446 = unsqueeze_1440 = None
    sub_448: "f32[8, 64, 35, 35]" = torch.ops.aten.sub.Tensor(where_88, mul_1456);  where_88 = mul_1456 = None
    sub_449: "f32[8, 64, 35, 35]" = torch.ops.aten.sub.Tensor(sub_448, unsqueeze_1437);  sub_448 = unsqueeze_1437 = None
    mul_1457: "f32[8, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_449, unsqueeze_1443);  sub_449 = unsqueeze_1443 = None
    mul_1458: "f32[64]" = torch.ops.aten.mul.Tensor(sum_179, squeeze_16);  sum_179 = squeeze_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_88 = torch.ops.aten.convolution_backward.default(mul_1457, getitem_12, primals_194, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1457 = getitem_12 = primals_194 = None
    getitem_460: "f32[8, 192, 35, 35]" = convolution_backward_88[0]
    getitem_461: "f32[64, 192, 1, 1]" = convolution_backward_88[1];  convolution_backward_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_504: "f32[8, 192, 35, 35]" = torch.ops.aten.add.Tensor(add_503, getitem_460);  add_503 = getitem_460 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/inception_v3.py:348, code: x = self.Pool2(x)  # N x 192 x 35 x 35
    max_pool2d_with_indices_backward_2: "f32[8, 192, 71, 71]" = torch.ops.aten.max_pool2d_with_indices_backward.default(add_504, relu_4, [3, 3], [2, 2], [0, 0], [1, 1], False, getitem_13);  add_504 = getitem_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_362: "f32[8, 192, 71, 71]" = torch.ops.aten.alias.default(relu_4);  relu_4 = None
    alias_363: "f32[8, 192, 71, 71]" = torch.ops.aten.alias.default(alias_362);  alias_362 = None
    le_89: "b8[8, 192, 71, 71]" = torch.ops.aten.le.Scalar(alias_363, 0);  alias_363 = None
    where_89: "f32[8, 192, 71, 71]" = torch.ops.aten.where.self(le_89, full_default, max_pool2d_with_indices_backward_2);  le_89 = max_pool2d_with_indices_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_180: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_89, [0, 2, 3])
    sub_450: "f32[8, 192, 71, 71]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_1446);  convolution_4 = unsqueeze_1446 = None
    mul_1459: "f32[8, 192, 71, 71]" = torch.ops.aten.mul.Tensor(where_89, sub_450)
    sum_181: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_1459, [0, 2, 3]);  mul_1459 = None
    mul_1460: "f32[192]" = torch.ops.aten.mul.Tensor(sum_180, 2.479666732791113e-05)
    unsqueeze_1447: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_1460, 0);  mul_1460 = None
    unsqueeze_1448: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1447, 2);  unsqueeze_1447 = None
    unsqueeze_1449: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1448, 3);  unsqueeze_1448 = None
    mul_1461: "f32[192]" = torch.ops.aten.mul.Tensor(sum_181, 2.479666732791113e-05)
    mul_1462: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
    mul_1463: "f32[192]" = torch.ops.aten.mul.Tensor(mul_1461, mul_1462);  mul_1461 = mul_1462 = None
    unsqueeze_1450: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_1463, 0);  mul_1463 = None
    unsqueeze_1451: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1450, 2);  unsqueeze_1450 = None
    unsqueeze_1452: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1451, 3);  unsqueeze_1451 = None
    mul_1464: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_13, primals_9);  primals_9 = None
    unsqueeze_1453: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_1464, 0);  mul_1464 = None
    unsqueeze_1454: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1453, 2);  unsqueeze_1453 = None
    unsqueeze_1455: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1454, 3);  unsqueeze_1454 = None
    mul_1465: "f32[8, 192, 71, 71]" = torch.ops.aten.mul.Tensor(sub_450, unsqueeze_1452);  sub_450 = unsqueeze_1452 = None
    sub_452: "f32[8, 192, 71, 71]" = torch.ops.aten.sub.Tensor(where_89, mul_1465);  where_89 = mul_1465 = None
    sub_453: "f32[8, 192, 71, 71]" = torch.ops.aten.sub.Tensor(sub_452, unsqueeze_1449);  sub_452 = unsqueeze_1449 = None
    mul_1466: "f32[8, 192, 71, 71]" = torch.ops.aten.mul.Tensor(sub_453, unsqueeze_1455);  sub_453 = unsqueeze_1455 = None
    mul_1467: "f32[192]" = torch.ops.aten.mul.Tensor(sum_181, squeeze_13);  sum_181 = squeeze_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_89 = torch.ops.aten.convolution_backward.default(mul_1466, relu_3, primals_193, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1466 = primals_193 = None
    getitem_463: "f32[8, 80, 73, 73]" = convolution_backward_89[0]
    getitem_464: "f32[192, 80, 3, 3]" = convolution_backward_89[1];  convolution_backward_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_365: "f32[8, 80, 73, 73]" = torch.ops.aten.alias.default(relu_3);  relu_3 = None
    alias_366: "f32[8, 80, 73, 73]" = torch.ops.aten.alias.default(alias_365);  alias_365 = None
    le_90: "b8[8, 80, 73, 73]" = torch.ops.aten.le.Scalar(alias_366, 0);  alias_366 = None
    where_90: "f32[8, 80, 73, 73]" = torch.ops.aten.where.self(le_90, full_default, getitem_463);  le_90 = getitem_463 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_182: "f32[80]" = torch.ops.aten.sum.dim_IntList(where_90, [0, 2, 3])
    sub_454: "f32[8, 80, 73, 73]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_1458);  convolution_3 = unsqueeze_1458 = None
    mul_1468: "f32[8, 80, 73, 73]" = torch.ops.aten.mul.Tensor(where_90, sub_454)
    sum_183: "f32[80]" = torch.ops.aten.sum.dim_IntList(mul_1468, [0, 2, 3]);  mul_1468 = None
    mul_1469: "f32[80]" = torch.ops.aten.mul.Tensor(sum_182, 2.3456558453743668e-05)
    unsqueeze_1459: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_1469, 0);  mul_1469 = None
    unsqueeze_1460: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1459, 2);  unsqueeze_1459 = None
    unsqueeze_1461: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1460, 3);  unsqueeze_1460 = None
    mul_1470: "f32[80]" = torch.ops.aten.mul.Tensor(sum_183, 2.3456558453743668e-05)
    mul_1471: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_10, squeeze_10)
    mul_1472: "f32[80]" = torch.ops.aten.mul.Tensor(mul_1470, mul_1471);  mul_1470 = mul_1471 = None
    unsqueeze_1462: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_1472, 0);  mul_1472 = None
    unsqueeze_1463: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1462, 2);  unsqueeze_1462 = None
    unsqueeze_1464: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1463, 3);  unsqueeze_1463 = None
    mul_1473: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_10, primals_7);  primals_7 = None
    unsqueeze_1465: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_1473, 0);  mul_1473 = None
    unsqueeze_1466: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1465, 2);  unsqueeze_1465 = None
    unsqueeze_1467: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1466, 3);  unsqueeze_1466 = None
    mul_1474: "f32[8, 80, 73, 73]" = torch.ops.aten.mul.Tensor(sub_454, unsqueeze_1464);  sub_454 = unsqueeze_1464 = None
    sub_456: "f32[8, 80, 73, 73]" = torch.ops.aten.sub.Tensor(where_90, mul_1474);  where_90 = mul_1474 = None
    sub_457: "f32[8, 80, 73, 73]" = torch.ops.aten.sub.Tensor(sub_456, unsqueeze_1461);  sub_456 = unsqueeze_1461 = None
    mul_1475: "f32[8, 80, 73, 73]" = torch.ops.aten.mul.Tensor(sub_457, unsqueeze_1467);  sub_457 = unsqueeze_1467 = None
    mul_1476: "f32[80]" = torch.ops.aten.mul.Tensor(sum_183, squeeze_10);  sum_183 = squeeze_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_90 = torch.ops.aten.convolution_backward.default(mul_1475, getitem_6, primals_192, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1475 = getitem_6 = primals_192 = None
    getitem_466: "f32[8, 64, 73, 73]" = convolution_backward_90[0]
    getitem_467: "f32[80, 64, 1, 1]" = convolution_backward_90[1];  convolution_backward_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/inception_v3.py:345, code: x = self.Pool1(x)  # N x 64 x 73 x 73
    max_pool2d_with_indices_backward_3: "f32[8, 64, 147, 147]" = torch.ops.aten.max_pool2d_with_indices_backward.default(getitem_466, relu_2, [3, 3], [2, 2], [0, 0], [1, 1], False, getitem_7);  getitem_466 = getitem_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_368: "f32[8, 64, 147, 147]" = torch.ops.aten.alias.default(relu_2);  relu_2 = None
    alias_369: "f32[8, 64, 147, 147]" = torch.ops.aten.alias.default(alias_368);  alias_368 = None
    le_91: "b8[8, 64, 147, 147]" = torch.ops.aten.le.Scalar(alias_369, 0);  alias_369 = None
    where_91: "f32[8, 64, 147, 147]" = torch.ops.aten.where.self(le_91, full_default, max_pool2d_with_indices_backward_3);  le_91 = max_pool2d_with_indices_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_184: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_91, [0, 2, 3])
    sub_458: "f32[8, 64, 147, 147]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_1470);  convolution_2 = unsqueeze_1470 = None
    mul_1477: "f32[8, 64, 147, 147]" = torch.ops.aten.mul.Tensor(where_91, sub_458)
    sum_185: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1477, [0, 2, 3]);  mul_1477 = None
    mul_1478: "f32[64]" = torch.ops.aten.mul.Tensor(sum_184, 5.78462677588042e-06)
    unsqueeze_1471: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1478, 0);  mul_1478 = None
    unsqueeze_1472: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1471, 2);  unsqueeze_1471 = None
    unsqueeze_1473: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1472, 3);  unsqueeze_1472 = None
    mul_1479: "f32[64]" = torch.ops.aten.mul.Tensor(sum_185, 5.78462677588042e-06)
    mul_1480: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
    mul_1481: "f32[64]" = torch.ops.aten.mul.Tensor(mul_1479, mul_1480);  mul_1479 = mul_1480 = None
    unsqueeze_1474: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1481, 0);  mul_1481 = None
    unsqueeze_1475: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1474, 2);  unsqueeze_1474 = None
    unsqueeze_1476: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1475, 3);  unsqueeze_1475 = None
    mul_1482: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_7, primals_5);  primals_5 = None
    unsqueeze_1477: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1482, 0);  mul_1482 = None
    unsqueeze_1478: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1477, 2);  unsqueeze_1477 = None
    unsqueeze_1479: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1478, 3);  unsqueeze_1478 = None
    mul_1483: "f32[8, 64, 147, 147]" = torch.ops.aten.mul.Tensor(sub_458, unsqueeze_1476);  sub_458 = unsqueeze_1476 = None
    sub_460: "f32[8, 64, 147, 147]" = torch.ops.aten.sub.Tensor(where_91, mul_1483);  where_91 = mul_1483 = None
    sub_461: "f32[8, 64, 147, 147]" = torch.ops.aten.sub.Tensor(sub_460, unsqueeze_1473);  sub_460 = unsqueeze_1473 = None
    mul_1484: "f32[8, 64, 147, 147]" = torch.ops.aten.mul.Tensor(sub_461, unsqueeze_1479);  sub_461 = unsqueeze_1479 = None
    mul_1485: "f32[64]" = torch.ops.aten.mul.Tensor(sum_185, squeeze_7);  sum_185 = squeeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_91 = torch.ops.aten.convolution_backward.default(mul_1484, relu_1, primals_191, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1484 = primals_191 = None
    getitem_469: "f32[8, 32, 147, 147]" = convolution_backward_91[0]
    getitem_470: "f32[64, 32, 3, 3]" = convolution_backward_91[1];  convolution_backward_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_371: "f32[8, 32, 147, 147]" = torch.ops.aten.alias.default(relu_1);  relu_1 = None
    alias_372: "f32[8, 32, 147, 147]" = torch.ops.aten.alias.default(alias_371);  alias_371 = None
    le_92: "b8[8, 32, 147, 147]" = torch.ops.aten.le.Scalar(alias_372, 0);  alias_372 = None
    where_92: "f32[8, 32, 147, 147]" = torch.ops.aten.where.self(le_92, full_default, getitem_469);  le_92 = getitem_469 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_186: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_92, [0, 2, 3])
    sub_462: "f32[8, 32, 147, 147]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_1482);  convolution_1 = unsqueeze_1482 = None
    mul_1486: "f32[8, 32, 147, 147]" = torch.ops.aten.mul.Tensor(where_92, sub_462)
    sum_187: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_1486, [0, 2, 3]);  mul_1486 = None
    mul_1487: "f32[32]" = torch.ops.aten.mul.Tensor(sum_186, 5.78462677588042e-06)
    unsqueeze_1483: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1487, 0);  mul_1487 = None
    unsqueeze_1484: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1483, 2);  unsqueeze_1483 = None
    unsqueeze_1485: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1484, 3);  unsqueeze_1484 = None
    mul_1488: "f32[32]" = torch.ops.aten.mul.Tensor(sum_187, 5.78462677588042e-06)
    mul_1489: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
    mul_1490: "f32[32]" = torch.ops.aten.mul.Tensor(mul_1488, mul_1489);  mul_1488 = mul_1489 = None
    unsqueeze_1486: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1490, 0);  mul_1490 = None
    unsqueeze_1487: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1486, 2);  unsqueeze_1486 = None
    unsqueeze_1488: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1487, 3);  unsqueeze_1487 = None
    mul_1491: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_4, primals_3);  primals_3 = None
    unsqueeze_1489: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1491, 0);  mul_1491 = None
    unsqueeze_1490: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1489, 2);  unsqueeze_1489 = None
    unsqueeze_1491: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1490, 3);  unsqueeze_1490 = None
    mul_1492: "f32[8, 32, 147, 147]" = torch.ops.aten.mul.Tensor(sub_462, unsqueeze_1488);  sub_462 = unsqueeze_1488 = None
    sub_464: "f32[8, 32, 147, 147]" = torch.ops.aten.sub.Tensor(where_92, mul_1492);  where_92 = mul_1492 = None
    sub_465: "f32[8, 32, 147, 147]" = torch.ops.aten.sub.Tensor(sub_464, unsqueeze_1485);  sub_464 = unsqueeze_1485 = None
    mul_1493: "f32[8, 32, 147, 147]" = torch.ops.aten.mul.Tensor(sub_465, unsqueeze_1491);  sub_465 = unsqueeze_1491 = None
    mul_1494: "f32[32]" = torch.ops.aten.mul.Tensor(sum_187, squeeze_4);  sum_187 = squeeze_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_92 = torch.ops.aten.convolution_backward.default(mul_1493, relu, primals_190, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1493 = primals_190 = None
    getitem_472: "f32[8, 32, 149, 149]" = convolution_backward_92[0]
    getitem_473: "f32[32, 32, 3, 3]" = convolution_backward_92[1];  convolution_backward_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_374: "f32[8, 32, 149, 149]" = torch.ops.aten.alias.default(relu);  relu = None
    alias_375: "f32[8, 32, 149, 149]" = torch.ops.aten.alias.default(alias_374);  alias_374 = None
    le_93: "b8[8, 32, 149, 149]" = torch.ops.aten.le.Scalar(alias_375, 0);  alias_375 = None
    where_93: "f32[8, 32, 149, 149]" = torch.ops.aten.where.self(le_93, full_default, getitem_472);  le_93 = full_default = getitem_472 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_188: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_93, [0, 2, 3])
    sub_466: "f32[8, 32, 149, 149]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_1494);  convolution = unsqueeze_1494 = None
    mul_1495: "f32[8, 32, 149, 149]" = torch.ops.aten.mul.Tensor(where_93, sub_466)
    sum_189: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_1495, [0, 2, 3]);  mul_1495 = None
    mul_1496: "f32[32]" = torch.ops.aten.mul.Tensor(sum_188, 5.630377010044593e-06)
    unsqueeze_1495: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1496, 0);  mul_1496 = None
    unsqueeze_1496: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1495, 2);  unsqueeze_1495 = None
    unsqueeze_1497: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1496, 3);  unsqueeze_1496 = None
    mul_1497: "f32[32]" = torch.ops.aten.mul.Tensor(sum_189, 5.630377010044593e-06)
    mul_1498: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
    mul_1499: "f32[32]" = torch.ops.aten.mul.Tensor(mul_1497, mul_1498);  mul_1497 = mul_1498 = None
    unsqueeze_1498: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1499, 0);  mul_1499 = None
    unsqueeze_1499: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1498, 2);  unsqueeze_1498 = None
    unsqueeze_1500: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1499, 3);  unsqueeze_1499 = None
    mul_1500: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_1, primals_1);  primals_1 = None
    unsqueeze_1501: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1500, 0);  mul_1500 = None
    unsqueeze_1502: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1501, 2);  unsqueeze_1501 = None
    unsqueeze_1503: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1502, 3);  unsqueeze_1502 = None
    mul_1501: "f32[8, 32, 149, 149]" = torch.ops.aten.mul.Tensor(sub_466, unsqueeze_1500);  sub_466 = unsqueeze_1500 = None
    sub_468: "f32[8, 32, 149, 149]" = torch.ops.aten.sub.Tensor(where_93, mul_1501);  where_93 = mul_1501 = None
    sub_469: "f32[8, 32, 149, 149]" = torch.ops.aten.sub.Tensor(sub_468, unsqueeze_1497);  sub_468 = unsqueeze_1497 = None
    mul_1502: "f32[8, 32, 149, 149]" = torch.ops.aten.mul.Tensor(sub_469, unsqueeze_1503);  sub_469 = unsqueeze_1503 = None
    mul_1503: "f32[32]" = torch.ops.aten.mul.Tensor(sum_189, squeeze_1);  sum_189 = squeeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_93 = torch.ops.aten.convolution_backward.default(mul_1502, primals_567, primals_189, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_1502 = primals_567 = primals_189 = None
    getitem_476: "f32[32, 3, 3, 3]" = convolution_backward_93[1];  convolution_backward_93 = None
    return [mul_1503, sum_188, mul_1494, sum_186, mul_1485, sum_184, mul_1476, sum_182, mul_1467, sum_180, mul_1458, sum_178, mul_1449, sum_176, mul_1440, sum_174, mul_1431, sum_172, mul_1422, sum_170, mul_1413, sum_168, mul_1404, sum_166, mul_1395, sum_164, mul_1386, sum_162, mul_1377, sum_160, mul_1368, sum_158, mul_1359, sum_156, mul_1350, sum_154, mul_1341, sum_152, mul_1332, sum_150, mul_1323, sum_148, mul_1314, sum_146, mul_1305, sum_144, mul_1296, sum_142, mul_1287, sum_140, mul_1278, sum_138, mul_1269, sum_136, mul_1260, sum_134, mul_1251, sum_132, mul_1242, sum_130, mul_1233, sum_128, mul_1224, sum_126, mul_1215, sum_124, mul_1206, sum_122, mul_1197, sum_120, mul_1188, sum_118, mul_1179, sum_116, mul_1170, sum_114, mul_1161, sum_112, mul_1152, sum_110, mul_1143, sum_108, mul_1134, sum_106, mul_1125, sum_104, mul_1116, sum_102, mul_1107, sum_100, mul_1098, sum_98, mul_1089, sum_96, mul_1080, sum_94, mul_1071, sum_92, mul_1062, sum_90, mul_1053, sum_88, mul_1044, sum_86, mul_1035, sum_84, mul_1026, sum_82, mul_1017, sum_80, mul_1008, sum_78, mul_999, sum_76, mul_990, sum_74, mul_981, sum_72, mul_972, sum_70, mul_963, sum_68, mul_954, sum_66, mul_945, sum_64, mul_936, sum_62, mul_927, sum_60, mul_918, sum_58, mul_909, sum_56, mul_900, sum_54, mul_891, sum_52, mul_882, sum_50, mul_873, sum_48, mul_864, sum_46, mul_855, sum_44, mul_846, sum_42, mul_837, sum_40, mul_828, sum_38, mul_819, sum_36, mul_810, sum_34, mul_801, sum_32, mul_792, sum_30, mul_783, sum_28, mul_774, sum_26, mul_765, sum_24, mul_756, sum_22, mul_747, sum_20, mul_738, sum_18, mul_729, sum_16, mul_720, sum_14, mul_711, sum_12, mul_702, sum_10, mul_693, sum_8, mul_684, sum_6, mul_675, sum_4, mul_666, sum_2, getitem_476, getitem_473, getitem_470, getitem_467, getitem_464, getitem_461, getitem_458, getitem_455, getitem_452, getitem_449, getitem_446, getitem_443, getitem_440, getitem_437, getitem_434, getitem_431, getitem_428, getitem_425, getitem_422, getitem_419, getitem_416, getitem_413, getitem_410, getitem_407, getitem_404, getitem_401, getitem_398, getitem_395, getitem_392, getitem_389, getitem_386, getitem_383, getitem_380, getitem_377, getitem_374, getitem_371, getitem_368, getitem_365, getitem_362, getitem_359, getitem_356, getitem_353, getitem_350, getitem_347, getitem_344, getitem_341, getitem_338, getitem_335, getitem_332, getitem_329, getitem_326, getitem_323, getitem_320, getitem_317, getitem_314, getitem_311, getitem_308, getitem_305, getitem_302, getitem_299, getitem_296, getitem_293, getitem_290, getitem_287, getitem_284, getitem_281, getitem_278, getitem_275, getitem_272, getitem_269, getitem_266, getitem_263, getitem_260, getitem_257, getitem_254, getitem_251, getitem_248, getitem_245, getitem_242, getitem_239, getitem_236, getitem_233, getitem_230, getitem_227, getitem_224, getitem_221, getitem_218, getitem_215, getitem_212, getitem_209, getitem_206, getitem_203, getitem_200, getitem_197, permute_4, view_1, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
    