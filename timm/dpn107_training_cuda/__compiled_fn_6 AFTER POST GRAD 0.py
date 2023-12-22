from __future__ import annotations



def forward(self, primals_1: "f32[128]", primals_3: "f32[128]", primals_5: "f32[128]", primals_7: "f32[200]", primals_9: "f32[200]", primals_11: "f32[316]", primals_13: "f32[200]", primals_15: "f32[200]", primals_17: "f32[336]", primals_19: "f32[200]", primals_21: "f32[200]", primals_23: "f32[356]", primals_25: "f32[200]", primals_27: "f32[200]", primals_29: "f32[376]", primals_31: "f32[376]", primals_33: "f32[400]", primals_35: "f32[400]", primals_37: "f32[704]", primals_39: "f32[400]", primals_41: "f32[400]", primals_43: "f32[768]", primals_45: "f32[400]", primals_47: "f32[400]", primals_49: "f32[832]", primals_51: "f32[400]", primals_53: "f32[400]", primals_55: "f32[896]", primals_57: "f32[400]", primals_59: "f32[400]", primals_61: "f32[960]", primals_63: "f32[400]", primals_65: "f32[400]", primals_67: "f32[1024]", primals_69: "f32[400]", primals_71: "f32[400]", primals_73: "f32[1088]", primals_75: "f32[400]", primals_77: "f32[400]", primals_79: "f32[1152]", primals_81: "f32[1152]", primals_83: "f32[800]", primals_85: "f32[800]", primals_87: "f32[1216]", primals_89: "f32[800]", primals_91: "f32[800]", primals_93: "f32[1280]", primals_95: "f32[800]", primals_97: "f32[800]", primals_99: "f32[1344]", primals_101: "f32[800]", primals_103: "f32[800]", primals_105: "f32[1408]", primals_107: "f32[800]", primals_109: "f32[800]", primals_111: "f32[1472]", primals_113: "f32[800]", primals_115: "f32[800]", primals_117: "f32[1536]", primals_119: "f32[800]", primals_121: "f32[800]", primals_123: "f32[1600]", primals_125: "f32[800]", primals_127: "f32[800]", primals_129: "f32[1664]", primals_131: "f32[800]", primals_133: "f32[800]", primals_135: "f32[1728]", primals_137: "f32[800]", primals_139: "f32[800]", primals_141: "f32[1792]", primals_143: "f32[800]", primals_145: "f32[800]", primals_147: "f32[1856]", primals_149: "f32[800]", primals_151: "f32[800]", primals_153: "f32[1920]", primals_155: "f32[800]", primals_157: "f32[800]", primals_159: "f32[1984]", primals_161: "f32[800]", primals_163: "f32[800]", primals_165: "f32[2048]", primals_167: "f32[800]", primals_169: "f32[800]", primals_171: "f32[2112]", primals_173: "f32[800]", primals_175: "f32[800]", primals_177: "f32[2176]", primals_179: "f32[800]", primals_181: "f32[800]", primals_183: "f32[2240]", primals_185: "f32[800]", primals_187: "f32[800]", primals_189: "f32[2304]", primals_191: "f32[800]", primals_193: "f32[800]", primals_195: "f32[2368]", primals_197: "f32[800]", primals_199: "f32[800]", primals_201: "f32[2432]", primals_203: "f32[2432]", primals_205: "f32[1600]", primals_207: "f32[1600]", primals_209: "f32[2432]", primals_211: "f32[1600]", primals_213: "f32[1600]", primals_215: "f32[2560]", primals_217: "f32[1600]", primals_219: "f32[1600]", primals_221: "f32[2688]", primals_223: "f32[128, 3, 7, 7]", primals_224: "f32[296, 128, 1, 1]", primals_225: "f32[200, 128, 1, 1]", primals_226: "f32[200, 4, 3, 3]", primals_227: "f32[276, 200, 1, 1]", primals_228: "f32[200, 316, 1, 1]", primals_229: "f32[200, 4, 3, 3]", primals_230: "f32[276, 200, 1, 1]", primals_231: "f32[200, 336, 1, 1]", primals_232: "f32[200, 4, 3, 3]", primals_233: "f32[276, 200, 1, 1]", primals_234: "f32[200, 356, 1, 1]", primals_235: "f32[200, 4, 3, 3]", primals_236: "f32[276, 200, 1, 1]", primals_237: "f32[640, 376, 1, 1]", primals_238: "f32[400, 376, 1, 1]", primals_239: "f32[400, 8, 3, 3]", primals_240: "f32[576, 400, 1, 1]", primals_241: "f32[400, 704, 1, 1]", primals_242: "f32[400, 8, 3, 3]", primals_243: "f32[576, 400, 1, 1]", primals_244: "f32[400, 768, 1, 1]", primals_245: "f32[400, 8, 3, 3]", primals_246: "f32[576, 400, 1, 1]", primals_247: "f32[400, 832, 1, 1]", primals_248: "f32[400, 8, 3, 3]", primals_249: "f32[576, 400, 1, 1]", primals_250: "f32[400, 896, 1, 1]", primals_251: "f32[400, 8, 3, 3]", primals_252: "f32[576, 400, 1, 1]", primals_253: "f32[400, 960, 1, 1]", primals_254: "f32[400, 8, 3, 3]", primals_255: "f32[576, 400, 1, 1]", primals_256: "f32[400, 1024, 1, 1]", primals_257: "f32[400, 8, 3, 3]", primals_258: "f32[576, 400, 1, 1]", primals_259: "f32[400, 1088, 1, 1]", primals_260: "f32[400, 8, 3, 3]", primals_261: "f32[576, 400, 1, 1]", primals_262: "f32[1152, 1152, 1, 1]", primals_263: "f32[800, 1152, 1, 1]", primals_264: "f32[800, 16, 3, 3]", primals_265: "f32[1088, 800, 1, 1]", primals_266: "f32[800, 1216, 1, 1]", primals_267: "f32[800, 16, 3, 3]", primals_268: "f32[1088, 800, 1, 1]", primals_269: "f32[800, 1280, 1, 1]", primals_270: "f32[800, 16, 3, 3]", primals_271: "f32[1088, 800, 1, 1]", primals_272: "f32[800, 1344, 1, 1]", primals_273: "f32[800, 16, 3, 3]", primals_274: "f32[1088, 800, 1, 1]", primals_275: "f32[800, 1408, 1, 1]", primals_276: "f32[800, 16, 3, 3]", primals_277: "f32[1088, 800, 1, 1]", primals_278: "f32[800, 1472, 1, 1]", primals_279: "f32[800, 16, 3, 3]", primals_280: "f32[1088, 800, 1, 1]", primals_281: "f32[800, 1536, 1, 1]", primals_282: "f32[800, 16, 3, 3]", primals_283: "f32[1088, 800, 1, 1]", primals_284: "f32[800, 1600, 1, 1]", primals_285: "f32[800, 16, 3, 3]", primals_286: "f32[1088, 800, 1, 1]", primals_287: "f32[800, 1664, 1, 1]", primals_288: "f32[800, 16, 3, 3]", primals_289: "f32[1088, 800, 1, 1]", primals_290: "f32[800, 1728, 1, 1]", primals_291: "f32[800, 16, 3, 3]", primals_292: "f32[1088, 800, 1, 1]", primals_293: "f32[800, 1792, 1, 1]", primals_294: "f32[800, 16, 3, 3]", primals_295: "f32[1088, 800, 1, 1]", primals_296: "f32[800, 1856, 1, 1]", primals_297: "f32[800, 16, 3, 3]", primals_298: "f32[1088, 800, 1, 1]", primals_299: "f32[800, 1920, 1, 1]", primals_300: "f32[800, 16, 3, 3]", primals_301: "f32[1088, 800, 1, 1]", primals_302: "f32[800, 1984, 1, 1]", primals_303: "f32[800, 16, 3, 3]", primals_304: "f32[1088, 800, 1, 1]", primals_305: "f32[800, 2048, 1, 1]", primals_306: "f32[800, 16, 3, 3]", primals_307: "f32[1088, 800, 1, 1]", primals_308: "f32[800, 2112, 1, 1]", primals_309: "f32[800, 16, 3, 3]", primals_310: "f32[1088, 800, 1, 1]", primals_311: "f32[800, 2176, 1, 1]", primals_312: "f32[800, 16, 3, 3]", primals_313: "f32[1088, 800, 1, 1]", primals_314: "f32[800, 2240, 1, 1]", primals_315: "f32[800, 16, 3, 3]", primals_316: "f32[1088, 800, 1, 1]", primals_317: "f32[800, 2304, 1, 1]", primals_318: "f32[800, 16, 3, 3]", primals_319: "f32[1088, 800, 1, 1]", primals_320: "f32[800, 2368, 1, 1]", primals_321: "f32[800, 16, 3, 3]", primals_322: "f32[1088, 800, 1, 1]", primals_323: "f32[2304, 2432, 1, 1]", primals_324: "f32[1600, 2432, 1, 1]", primals_325: "f32[1600, 32, 3, 3]", primals_326: "f32[2176, 1600, 1, 1]", primals_327: "f32[1600, 2432, 1, 1]", primals_328: "f32[1600, 32, 3, 3]", primals_329: "f32[2176, 1600, 1, 1]", primals_330: "f32[1600, 2560, 1, 1]", primals_331: "f32[1600, 32, 3, 3]", primals_332: "f32[2176, 1600, 1, 1]", primals_333: "f32[1000, 2688, 1, 1]", primals_668: "f32[8, 3, 224, 224]", convolution: "f32[8, 128, 112, 112]", squeeze_1: "f32[128]", relu: "f32[8, 128, 112, 112]", getitem_3: "i64[8, 128, 56, 56]", squeeze_4: "f32[128]", relu_1: "f32[8, 128, 56, 56]", relu_2: "f32[8, 128, 56, 56]", convolution_2: "f32[8, 200, 56, 56]", squeeze_10: "f32[200]", relu_3: "f32[8, 200, 56, 56]", convolution_3: "f32[8, 200, 56, 56]", squeeze_13: "f32[200]", relu_4: "f32[8, 200, 56, 56]", cat_1: "f32[8, 316, 56, 56]", squeeze_16: "f32[316]", relu_5: "f32[8, 316, 56, 56]", convolution_5: "f32[8, 200, 56, 56]", squeeze_19: "f32[200]", relu_6: "f32[8, 200, 56, 56]", convolution_6: "f32[8, 200, 56, 56]", squeeze_22: "f32[200]", relu_7: "f32[8, 200, 56, 56]", cat_3: "f32[8, 336, 56, 56]", squeeze_25: "f32[336]", relu_8: "f32[8, 336, 56, 56]", convolution_8: "f32[8, 200, 56, 56]", squeeze_28: "f32[200]", relu_9: "f32[8, 200, 56, 56]", convolution_9: "f32[8, 200, 56, 56]", squeeze_31: "f32[200]", relu_10: "f32[8, 200, 56, 56]", cat_5: "f32[8, 356, 56, 56]", squeeze_34: "f32[356]", relu_11: "f32[8, 356, 56, 56]", convolution_11: "f32[8, 200, 56, 56]", squeeze_37: "f32[200]", relu_12: "f32[8, 200, 56, 56]", convolution_12: "f32[8, 200, 56, 56]", squeeze_40: "f32[200]", relu_13: "f32[8, 200, 56, 56]", cat_7: "f32[8, 376, 56, 56]", squeeze_43: "f32[376]", relu_14: "f32[8, 376, 56, 56]", relu_15: "f32[8, 376, 56, 56]", convolution_15: "f32[8, 400, 56, 56]", squeeze_49: "f32[400]", relu_16: "f32[8, 400, 56, 56]", convolution_16: "f32[8, 400, 28, 28]", squeeze_52: "f32[400]", relu_17: "f32[8, 400, 28, 28]", cat_9: "f32[8, 704, 28, 28]", squeeze_55: "f32[704]", relu_18: "f32[8, 704, 28, 28]", convolution_18: "f32[8, 400, 28, 28]", squeeze_58: "f32[400]", relu_19: "f32[8, 400, 28, 28]", convolution_19: "f32[8, 400, 28, 28]", squeeze_61: "f32[400]", relu_20: "f32[8, 400, 28, 28]", cat_11: "f32[8, 768, 28, 28]", squeeze_64: "f32[768]", relu_21: "f32[8, 768, 28, 28]", convolution_21: "f32[8, 400, 28, 28]", squeeze_67: "f32[400]", relu_22: "f32[8, 400, 28, 28]", convolution_22: "f32[8, 400, 28, 28]", squeeze_70: "f32[400]", relu_23: "f32[8, 400, 28, 28]", cat_13: "f32[8, 832, 28, 28]", squeeze_73: "f32[832]", relu_24: "f32[8, 832, 28, 28]", convolution_24: "f32[8, 400, 28, 28]", squeeze_76: "f32[400]", relu_25: "f32[8, 400, 28, 28]", convolution_25: "f32[8, 400, 28, 28]", squeeze_79: "f32[400]", relu_26: "f32[8, 400, 28, 28]", cat_15: "f32[8, 896, 28, 28]", squeeze_82: "f32[896]", relu_27: "f32[8, 896, 28, 28]", convolution_27: "f32[8, 400, 28, 28]", squeeze_85: "f32[400]", relu_28: "f32[8, 400, 28, 28]", convolution_28: "f32[8, 400, 28, 28]", squeeze_88: "f32[400]", relu_29: "f32[8, 400, 28, 28]", cat_17: "f32[8, 960, 28, 28]", squeeze_91: "f32[960]", relu_30: "f32[8, 960, 28, 28]", convolution_30: "f32[8, 400, 28, 28]", squeeze_94: "f32[400]", relu_31: "f32[8, 400, 28, 28]", convolution_31: "f32[8, 400, 28, 28]", squeeze_97: "f32[400]", relu_32: "f32[8, 400, 28, 28]", cat_19: "f32[8, 1024, 28, 28]", squeeze_100: "f32[1024]", relu_33: "f32[8, 1024, 28, 28]", convolution_33: "f32[8, 400, 28, 28]", squeeze_103: "f32[400]", relu_34: "f32[8, 400, 28, 28]", convolution_34: "f32[8, 400, 28, 28]", squeeze_106: "f32[400]", relu_35: "f32[8, 400, 28, 28]", cat_21: "f32[8, 1088, 28, 28]", squeeze_109: "f32[1088]", relu_36: "f32[8, 1088, 28, 28]", convolution_36: "f32[8, 400, 28, 28]", squeeze_112: "f32[400]", relu_37: "f32[8, 400, 28, 28]", convolution_37: "f32[8, 400, 28, 28]", squeeze_115: "f32[400]", relu_38: "f32[8, 400, 28, 28]", cat_23: "f32[8, 1152, 28, 28]", squeeze_118: "f32[1152]", relu_39: "f32[8, 1152, 28, 28]", relu_40: "f32[8, 1152, 28, 28]", convolution_40: "f32[8, 800, 28, 28]", squeeze_124: "f32[800]", relu_41: "f32[8, 800, 28, 28]", convolution_41: "f32[8, 800, 14, 14]", squeeze_127: "f32[800]", relu_42: "f32[8, 800, 14, 14]", cat_25: "f32[8, 1216, 14, 14]", squeeze_130: "f32[1216]", relu_43: "f32[8, 1216, 14, 14]", convolution_43: "f32[8, 800, 14, 14]", squeeze_133: "f32[800]", relu_44: "f32[8, 800, 14, 14]", convolution_44: "f32[8, 800, 14, 14]", squeeze_136: "f32[800]", relu_45: "f32[8, 800, 14, 14]", cat_27: "f32[8, 1280, 14, 14]", squeeze_139: "f32[1280]", relu_46: "f32[8, 1280, 14, 14]", convolution_46: "f32[8, 800, 14, 14]", squeeze_142: "f32[800]", relu_47: "f32[8, 800, 14, 14]", convolution_47: "f32[8, 800, 14, 14]", squeeze_145: "f32[800]", relu_48: "f32[8, 800, 14, 14]", cat_29: "f32[8, 1344, 14, 14]", squeeze_148: "f32[1344]", relu_49: "f32[8, 1344, 14, 14]", convolution_49: "f32[8, 800, 14, 14]", squeeze_151: "f32[800]", relu_50: "f32[8, 800, 14, 14]", convolution_50: "f32[8, 800, 14, 14]", squeeze_154: "f32[800]", relu_51: "f32[8, 800, 14, 14]", cat_31: "f32[8, 1408, 14, 14]", squeeze_157: "f32[1408]", relu_52: "f32[8, 1408, 14, 14]", convolution_52: "f32[8, 800, 14, 14]", squeeze_160: "f32[800]", relu_53: "f32[8, 800, 14, 14]", convolution_53: "f32[8, 800, 14, 14]", squeeze_163: "f32[800]", relu_54: "f32[8, 800, 14, 14]", cat_33: "f32[8, 1472, 14, 14]", squeeze_166: "f32[1472]", relu_55: "f32[8, 1472, 14, 14]", convolution_55: "f32[8, 800, 14, 14]", squeeze_169: "f32[800]", relu_56: "f32[8, 800, 14, 14]", convolution_56: "f32[8, 800, 14, 14]", squeeze_172: "f32[800]", relu_57: "f32[8, 800, 14, 14]", cat_35: "f32[8, 1536, 14, 14]", squeeze_175: "f32[1536]", relu_58: "f32[8, 1536, 14, 14]", convolution_58: "f32[8, 800, 14, 14]", squeeze_178: "f32[800]", relu_59: "f32[8, 800, 14, 14]", convolution_59: "f32[8, 800, 14, 14]", squeeze_181: "f32[800]", relu_60: "f32[8, 800, 14, 14]", cat_37: "f32[8, 1600, 14, 14]", squeeze_184: "f32[1600]", relu_61: "f32[8, 1600, 14, 14]", convolution_61: "f32[8, 800, 14, 14]", squeeze_187: "f32[800]", relu_62: "f32[8, 800, 14, 14]", convolution_62: "f32[8, 800, 14, 14]", squeeze_190: "f32[800]", relu_63: "f32[8, 800, 14, 14]", cat_39: "f32[8, 1664, 14, 14]", squeeze_193: "f32[1664]", relu_64: "f32[8, 1664, 14, 14]", convolution_64: "f32[8, 800, 14, 14]", squeeze_196: "f32[800]", relu_65: "f32[8, 800, 14, 14]", convolution_65: "f32[8, 800, 14, 14]", squeeze_199: "f32[800]", relu_66: "f32[8, 800, 14, 14]", cat_41: "f32[8, 1728, 14, 14]", squeeze_202: "f32[1728]", relu_67: "f32[8, 1728, 14, 14]", convolution_67: "f32[8, 800, 14, 14]", squeeze_205: "f32[800]", relu_68: "f32[8, 800, 14, 14]", convolution_68: "f32[8, 800, 14, 14]", squeeze_208: "f32[800]", relu_69: "f32[8, 800, 14, 14]", cat_43: "f32[8, 1792, 14, 14]", squeeze_211: "f32[1792]", relu_70: "f32[8, 1792, 14, 14]", convolution_70: "f32[8, 800, 14, 14]", squeeze_214: "f32[800]", relu_71: "f32[8, 800, 14, 14]", convolution_71: "f32[8, 800, 14, 14]", squeeze_217: "f32[800]", relu_72: "f32[8, 800, 14, 14]", cat_45: "f32[8, 1856, 14, 14]", squeeze_220: "f32[1856]", relu_73: "f32[8, 1856, 14, 14]", convolution_73: "f32[8, 800, 14, 14]", squeeze_223: "f32[800]", relu_74: "f32[8, 800, 14, 14]", convolution_74: "f32[8, 800, 14, 14]", squeeze_226: "f32[800]", relu_75: "f32[8, 800, 14, 14]", cat_47: "f32[8, 1920, 14, 14]", squeeze_229: "f32[1920]", relu_76: "f32[8, 1920, 14, 14]", convolution_76: "f32[8, 800, 14, 14]", squeeze_232: "f32[800]", relu_77: "f32[8, 800, 14, 14]", convolution_77: "f32[8, 800, 14, 14]", squeeze_235: "f32[800]", relu_78: "f32[8, 800, 14, 14]", cat_49: "f32[8, 1984, 14, 14]", squeeze_238: "f32[1984]", relu_79: "f32[8, 1984, 14, 14]", convolution_79: "f32[8, 800, 14, 14]", squeeze_241: "f32[800]", relu_80: "f32[8, 800, 14, 14]", convolution_80: "f32[8, 800, 14, 14]", squeeze_244: "f32[800]", relu_81: "f32[8, 800, 14, 14]", cat_51: "f32[8, 2048, 14, 14]", squeeze_247: "f32[2048]", relu_82: "f32[8, 2048, 14, 14]", convolution_82: "f32[8, 800, 14, 14]", squeeze_250: "f32[800]", relu_83: "f32[8, 800, 14, 14]", convolution_83: "f32[8, 800, 14, 14]", squeeze_253: "f32[800]", relu_84: "f32[8, 800, 14, 14]", cat_53: "f32[8, 2112, 14, 14]", squeeze_256: "f32[2112]", relu_85: "f32[8, 2112, 14, 14]", convolution_85: "f32[8, 800, 14, 14]", squeeze_259: "f32[800]", relu_86: "f32[8, 800, 14, 14]", convolution_86: "f32[8, 800, 14, 14]", squeeze_262: "f32[800]", relu_87: "f32[8, 800, 14, 14]", cat_55: "f32[8, 2176, 14, 14]", squeeze_265: "f32[2176]", relu_88: "f32[8, 2176, 14, 14]", convolution_88: "f32[8, 800, 14, 14]", squeeze_268: "f32[800]", relu_89: "f32[8, 800, 14, 14]", convolution_89: "f32[8, 800, 14, 14]", squeeze_271: "f32[800]", relu_90: "f32[8, 800, 14, 14]", cat_57: "f32[8, 2240, 14, 14]", squeeze_274: "f32[2240]", relu_91: "f32[8, 2240, 14, 14]", convolution_91: "f32[8, 800, 14, 14]", squeeze_277: "f32[800]", relu_92: "f32[8, 800, 14, 14]", convolution_92: "f32[8, 800, 14, 14]", squeeze_280: "f32[800]", relu_93: "f32[8, 800, 14, 14]", cat_59: "f32[8, 2304, 14, 14]", squeeze_283: "f32[2304]", relu_94: "f32[8, 2304, 14, 14]", convolution_94: "f32[8, 800, 14, 14]", squeeze_286: "f32[800]", relu_95: "f32[8, 800, 14, 14]", convolution_95: "f32[8, 800, 14, 14]", squeeze_289: "f32[800]", relu_96: "f32[8, 800, 14, 14]", cat_61: "f32[8, 2368, 14, 14]", squeeze_292: "f32[2368]", relu_97: "f32[8, 2368, 14, 14]", convolution_97: "f32[8, 800, 14, 14]", squeeze_295: "f32[800]", relu_98: "f32[8, 800, 14, 14]", convolution_98: "f32[8, 800, 14, 14]", squeeze_298: "f32[800]", relu_99: "f32[8, 800, 14, 14]", cat_63: "f32[8, 2432, 14, 14]", squeeze_301: "f32[2432]", relu_100: "f32[8, 2432, 14, 14]", relu_101: "f32[8, 2432, 14, 14]", convolution_101: "f32[8, 1600, 14, 14]", squeeze_307: "f32[1600]", relu_102: "f32[8, 1600, 14, 14]", convolution_102: "f32[8, 1600, 7, 7]", squeeze_310: "f32[1600]", relu_103: "f32[8, 1600, 7, 7]", cat_65: "f32[8, 2432, 7, 7]", squeeze_313: "f32[2432]", relu_104: "f32[8, 2432, 7, 7]", convolution_104: "f32[8, 1600, 7, 7]", squeeze_316: "f32[1600]", relu_105: "f32[8, 1600, 7, 7]", convolution_105: "f32[8, 1600, 7, 7]", squeeze_319: "f32[1600]", relu_106: "f32[8, 1600, 7, 7]", cat_67: "f32[8, 2560, 7, 7]", squeeze_322: "f32[2560]", relu_107: "f32[8, 2560, 7, 7]", convolution_107: "f32[8, 1600, 7, 7]", squeeze_325: "f32[1600]", relu_108: "f32[8, 1600, 7, 7]", convolution_108: "f32[8, 1600, 7, 7]", squeeze_328: "f32[1600]", relu_109: "f32[8, 1600, 7, 7]", cat_69: "f32[8, 2688, 7, 7]", squeeze_331: "f32[2688]", mean: "f32[8, 2688, 1, 1]", le: "b8[8, 2688, 7, 7]", unsqueeze_446: "f32[1, 2688, 1, 1]", unsqueeze_458: "f32[1, 1600, 1, 1]", unsqueeze_470: "f32[1, 1600, 1, 1]", unsqueeze_482: "f32[1, 2560, 1, 1]", unsqueeze_494: "f32[1, 1600, 1, 1]", unsqueeze_506: "f32[1, 1600, 1, 1]", unsqueeze_518: "f32[1, 2432, 1, 1]", unsqueeze_530: "f32[1, 1600, 1, 1]", unsqueeze_542: "f32[1, 1600, 1, 1]", unsqueeze_554: "f32[1, 2432, 1, 1]", unsqueeze_578: "f32[1, 800, 1, 1]", unsqueeze_590: "f32[1, 800, 1, 1]", unsqueeze_602: "f32[1, 2368, 1, 1]", unsqueeze_614: "f32[1, 800, 1, 1]", unsqueeze_626: "f32[1, 800, 1, 1]", unsqueeze_638: "f32[1, 2304, 1, 1]", unsqueeze_650: "f32[1, 800, 1, 1]", unsqueeze_662: "f32[1, 800, 1, 1]", unsqueeze_674: "f32[1, 2240, 1, 1]", unsqueeze_686: "f32[1, 800, 1, 1]", unsqueeze_698: "f32[1, 800, 1, 1]", unsqueeze_710: "f32[1, 2176, 1, 1]", unsqueeze_722: "f32[1, 800, 1, 1]", unsqueeze_734: "f32[1, 800, 1, 1]", unsqueeze_746: "f32[1, 2112, 1, 1]", unsqueeze_758: "f32[1, 800, 1, 1]", unsqueeze_770: "f32[1, 800, 1, 1]", unsqueeze_782: "f32[1, 2048, 1, 1]", unsqueeze_794: "f32[1, 800, 1, 1]", unsqueeze_806: "f32[1, 800, 1, 1]", unsqueeze_818: "f32[1, 1984, 1, 1]", unsqueeze_830: "f32[1, 800, 1, 1]", unsqueeze_842: "f32[1, 800, 1, 1]", unsqueeze_854: "f32[1, 1920, 1, 1]", unsqueeze_866: "f32[1, 800, 1, 1]", unsqueeze_878: "f32[1, 800, 1, 1]", unsqueeze_890: "f32[1, 1856, 1, 1]", unsqueeze_902: "f32[1, 800, 1, 1]", unsqueeze_914: "f32[1, 800, 1, 1]", unsqueeze_926: "f32[1, 1792, 1, 1]", unsqueeze_938: "f32[1, 800, 1, 1]", unsqueeze_950: "f32[1, 800, 1, 1]", unsqueeze_962: "f32[1, 1728, 1, 1]", unsqueeze_974: "f32[1, 800, 1, 1]", unsqueeze_986: "f32[1, 800, 1, 1]", unsqueeze_998: "f32[1, 1664, 1, 1]", unsqueeze_1010: "f32[1, 800, 1, 1]", unsqueeze_1022: "f32[1, 800, 1, 1]", unsqueeze_1034: "f32[1, 1600, 1, 1]", unsqueeze_1046: "f32[1, 800, 1, 1]", unsqueeze_1058: "f32[1, 800, 1, 1]", unsqueeze_1070: "f32[1, 1536, 1, 1]", unsqueeze_1082: "f32[1, 800, 1, 1]", unsqueeze_1094: "f32[1, 800, 1, 1]", unsqueeze_1106: "f32[1, 1472, 1, 1]", unsqueeze_1118: "f32[1, 800, 1, 1]", unsqueeze_1130: "f32[1, 800, 1, 1]", unsqueeze_1142: "f32[1, 1408, 1, 1]", unsqueeze_1154: "f32[1, 800, 1, 1]", unsqueeze_1166: "f32[1, 800, 1, 1]", unsqueeze_1178: "f32[1, 1344, 1, 1]", unsqueeze_1190: "f32[1, 800, 1, 1]", unsqueeze_1202: "f32[1, 800, 1, 1]", unsqueeze_1214: "f32[1, 1280, 1, 1]", unsqueeze_1226: "f32[1, 800, 1, 1]", unsqueeze_1238: "f32[1, 800, 1, 1]", unsqueeze_1250: "f32[1, 1216, 1, 1]", unsqueeze_1262: "f32[1, 800, 1, 1]", unsqueeze_1274: "f32[1, 800, 1, 1]", unsqueeze_1286: "f32[1, 1152, 1, 1]", unsqueeze_1310: "f32[1, 400, 1, 1]", unsqueeze_1322: "f32[1, 400, 1, 1]", unsqueeze_1334: "f32[1, 1088, 1, 1]", unsqueeze_1346: "f32[1, 400, 1, 1]", unsqueeze_1358: "f32[1, 400, 1, 1]", unsqueeze_1370: "f32[1, 1024, 1, 1]", unsqueeze_1382: "f32[1, 400, 1, 1]", unsqueeze_1394: "f32[1, 400, 1, 1]", unsqueeze_1406: "f32[1, 960, 1, 1]", unsqueeze_1418: "f32[1, 400, 1, 1]", unsqueeze_1430: "f32[1, 400, 1, 1]", unsqueeze_1442: "f32[1, 896, 1, 1]", unsqueeze_1454: "f32[1, 400, 1, 1]", unsqueeze_1466: "f32[1, 400, 1, 1]", unsqueeze_1478: "f32[1, 832, 1, 1]", unsqueeze_1490: "f32[1, 400, 1, 1]", unsqueeze_1502: "f32[1, 400, 1, 1]", unsqueeze_1514: "f32[1, 768, 1, 1]", unsqueeze_1526: "f32[1, 400, 1, 1]", unsqueeze_1538: "f32[1, 400, 1, 1]", unsqueeze_1550: "f32[1, 704, 1, 1]", unsqueeze_1562: "f32[1, 400, 1, 1]", unsqueeze_1574: "f32[1, 400, 1, 1]", unsqueeze_1586: "f32[1, 376, 1, 1]", unsqueeze_1610: "f32[1, 200, 1, 1]", unsqueeze_1622: "f32[1, 200, 1, 1]", unsqueeze_1634: "f32[1, 356, 1, 1]", unsqueeze_1646: "f32[1, 200, 1, 1]", unsqueeze_1658: "f32[1, 200, 1, 1]", unsqueeze_1670: "f32[1, 336, 1, 1]", unsqueeze_1682: "f32[1, 200, 1, 1]", unsqueeze_1694: "f32[1, 200, 1, 1]", unsqueeze_1706: "f32[1, 316, 1, 1]", unsqueeze_1718: "f32[1, 200, 1, 1]", unsqueeze_1730: "f32[1, 200, 1, 1]", sub_543: "f32[8, 128, 56, 56]", unsqueeze_1766: "f32[1, 128, 1, 1]", tangents_1: "f32[8, 1000]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:275, code: return self.flatten(x)
    view_1: "f32[8, 1000, 1, 1]" = torch.ops.aten.reshape.default(tangents_1, [8, 1000, 1, 1]);  tangents_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:274, code: x = self.classifier(x)
    sum_1: "f32[1000]" = torch.ops.aten.sum.dim_IntList(view_1, [0, 2, 3])
    convolution_backward = torch.ops.aten.convolution_backward.default(view_1, mean, primals_333, [1000], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_1 = mean = primals_333 = None
    getitem_224: "f32[8, 2688, 1, 1]" = convolution_backward[0]
    getitem_225: "f32[1000, 2688, 1, 1]" = convolution_backward[1];  convolution_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    expand: "f32[8, 2688, 7, 7]" = torch.ops.aten.expand.default(getitem_224, [8, 2688, 7, 7]);  getitem_224 = None
    div: "f32[8, 2688, 7, 7]" = torch.ops.aten.div.Scalar(expand, 49);  expand = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    where: "f32[8, 2688, 7, 7]" = torch.ops.aten.where.self(le, full_default, div);  le = div = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_2: "f32[2688]" = torch.ops.aten.sum.dim_IntList(where, [0, 2, 3])
    sub_111: "f32[8, 2688, 7, 7]" = torch.ops.aten.sub.Tensor(cat_69, unsqueeze_446);  cat_69 = unsqueeze_446 = None
    mul_777: "f32[8, 2688, 7, 7]" = torch.ops.aten.mul.Tensor(where, sub_111)
    sum_3: "f32[2688]" = torch.ops.aten.sum.dim_IntList(mul_777, [0, 2, 3]);  mul_777 = None
    mul_778: "f32[2688]" = torch.ops.aten.mul.Tensor(sum_2, 0.002551020408163265)
    unsqueeze_447: "f32[1, 2688]" = torch.ops.aten.unsqueeze.default(mul_778, 0);  mul_778 = None
    unsqueeze_448: "f32[1, 2688, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_447, 2);  unsqueeze_447 = None
    unsqueeze_449: "f32[1, 2688, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_448, 3);  unsqueeze_448 = None
    mul_779: "f32[2688]" = torch.ops.aten.mul.Tensor(sum_3, 0.002551020408163265)
    mul_780: "f32[2688]" = torch.ops.aten.mul.Tensor(squeeze_331, squeeze_331)
    mul_781: "f32[2688]" = torch.ops.aten.mul.Tensor(mul_779, mul_780);  mul_779 = mul_780 = None
    unsqueeze_450: "f32[1, 2688]" = torch.ops.aten.unsqueeze.default(mul_781, 0);  mul_781 = None
    unsqueeze_451: "f32[1, 2688, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_450, 2);  unsqueeze_450 = None
    unsqueeze_452: "f32[1, 2688, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_451, 3);  unsqueeze_451 = None
    mul_782: "f32[2688]" = torch.ops.aten.mul.Tensor(squeeze_331, primals_221);  primals_221 = None
    unsqueeze_453: "f32[1, 2688]" = torch.ops.aten.unsqueeze.default(mul_782, 0);  mul_782 = None
    unsqueeze_454: "f32[1, 2688, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_453, 2);  unsqueeze_453 = None
    unsqueeze_455: "f32[1, 2688, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_454, 3);  unsqueeze_454 = None
    mul_783: "f32[8, 2688, 7, 7]" = torch.ops.aten.mul.Tensor(sub_111, unsqueeze_452);  sub_111 = unsqueeze_452 = None
    sub_113: "f32[8, 2688, 7, 7]" = torch.ops.aten.sub.Tensor(where, mul_783);  where = mul_783 = None
    sub_114: "f32[8, 2688, 7, 7]" = torch.ops.aten.sub.Tensor(sub_113, unsqueeze_449);  sub_113 = unsqueeze_449 = None
    mul_784: "f32[8, 2688, 7, 7]" = torch.ops.aten.mul.Tensor(sub_114, unsqueeze_455);  sub_114 = unsqueeze_455 = None
    mul_785: "f32[2688]" = torch.ops.aten.mul.Tensor(sum_3, squeeze_331);  sum_3 = squeeze_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:42, code: x = torch.cat(x, dim=1)
    slice_313: "f32[8, 2048, 7, 7]" = torch.ops.aten.slice.Tensor(mul_784, 1, 0, 2048)
    slice_314: "f32[8, 640, 7, 7]" = torch.ops.aten.slice.Tensor(mul_784, 1, 2048, 2688);  mul_784 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    slice_315: "f32[8, 512, 7, 7]" = torch.ops.aten.slice.Tensor(slice_314, 1, 0, 512)
    slice_316: "f32[8, 128, 7, 7]" = torch.ops.aten.slice.Tensor(slice_314, 1, 512, 640);  slice_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    full_default_1: "f32[8, 128, 7, 7]" = torch.ops.aten.full.default([8, 128, 7, 7], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter: "f32[8, 128, 7, 7]" = torch.ops.aten.slice_scatter.default(full_default_1, slice_316, 3, 0, 9223372036854775807);  slice_316 = None
    full_default_3: "f32[8, 2176, 7, 7]" = torch.ops.aten.full.default([8, 2176, 7, 7], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_2: "f32[8, 2176, 7, 7]" = torch.ops.aten.slice_scatter.default(full_default_3, slice_scatter, 1, 2048, 9223372036854775807);  slice_scatter = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    full_default_5: "f32[8, 2048, 7, 7]" = torch.ops.aten.full.default([8, 2048, 7, 7], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_4: "f32[8, 2048, 7, 7]" = torch.ops.aten.slice_scatter.default(full_default_5, slice_313, 3, 0, 9223372036854775807);  full_default_5 = None
    slice_scatter_6: "f32[8, 2176, 7, 7]" = torch.ops.aten.slice_scatter.default(full_default_3, slice_scatter_4, 1, 0, 2048);  slice_scatter_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    add_590: "f32[8, 2176, 7, 7]" = torch.ops.aten.add.Tensor(slice_scatter_2, slice_scatter_6);  slice_scatter_2 = slice_scatter_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(add_590, relu_109, primals_332, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_590 = primals_332 = None
    getitem_227: "f32[8, 1600, 7, 7]" = convolution_backward_1[0]
    getitem_228: "f32[2176, 1600, 1, 1]" = convolution_backward_1[1];  convolution_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_1: "b8[8, 1600, 7, 7]" = torch.ops.aten.le.Scalar(relu_109, 0);  relu_109 = None
    where_1: "f32[8, 1600, 7, 7]" = torch.ops.aten.where.self(le_1, full_default, getitem_227);  le_1 = getitem_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_4: "f32[1600]" = torch.ops.aten.sum.dim_IntList(where_1, [0, 2, 3])
    sub_115: "f32[8, 1600, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_108, unsqueeze_458);  convolution_108 = unsqueeze_458 = None
    mul_786: "f32[8, 1600, 7, 7]" = torch.ops.aten.mul.Tensor(where_1, sub_115)
    sum_5: "f32[1600]" = torch.ops.aten.sum.dim_IntList(mul_786, [0, 2, 3]);  mul_786 = None
    mul_787: "f32[1600]" = torch.ops.aten.mul.Tensor(sum_4, 0.002551020408163265)
    unsqueeze_459: "f32[1, 1600]" = torch.ops.aten.unsqueeze.default(mul_787, 0);  mul_787 = None
    unsqueeze_460: "f32[1, 1600, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_459, 2);  unsqueeze_459 = None
    unsqueeze_461: "f32[1, 1600, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_460, 3);  unsqueeze_460 = None
    mul_788: "f32[1600]" = torch.ops.aten.mul.Tensor(sum_5, 0.002551020408163265)
    mul_789: "f32[1600]" = torch.ops.aten.mul.Tensor(squeeze_328, squeeze_328)
    mul_790: "f32[1600]" = torch.ops.aten.mul.Tensor(mul_788, mul_789);  mul_788 = mul_789 = None
    unsqueeze_462: "f32[1, 1600]" = torch.ops.aten.unsqueeze.default(mul_790, 0);  mul_790 = None
    unsqueeze_463: "f32[1, 1600, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_462, 2);  unsqueeze_462 = None
    unsqueeze_464: "f32[1, 1600, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_463, 3);  unsqueeze_463 = None
    mul_791: "f32[1600]" = torch.ops.aten.mul.Tensor(squeeze_328, primals_219);  primals_219 = None
    unsqueeze_465: "f32[1, 1600]" = torch.ops.aten.unsqueeze.default(mul_791, 0);  mul_791 = None
    unsqueeze_466: "f32[1, 1600, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_465, 2);  unsqueeze_465 = None
    unsqueeze_467: "f32[1, 1600, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_466, 3);  unsqueeze_466 = None
    mul_792: "f32[8, 1600, 7, 7]" = torch.ops.aten.mul.Tensor(sub_115, unsqueeze_464);  sub_115 = unsqueeze_464 = None
    sub_117: "f32[8, 1600, 7, 7]" = torch.ops.aten.sub.Tensor(where_1, mul_792);  where_1 = mul_792 = None
    sub_118: "f32[8, 1600, 7, 7]" = torch.ops.aten.sub.Tensor(sub_117, unsqueeze_461);  sub_117 = unsqueeze_461 = None
    mul_793: "f32[8, 1600, 7, 7]" = torch.ops.aten.mul.Tensor(sub_118, unsqueeze_467);  sub_118 = unsqueeze_467 = None
    mul_794: "f32[1600]" = torch.ops.aten.mul.Tensor(sum_5, squeeze_328);  sum_5 = squeeze_328 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(mul_793, relu_108, primals_331, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False]);  mul_793 = primals_331 = None
    getitem_230: "f32[8, 1600, 7, 7]" = convolution_backward_2[0]
    getitem_231: "f32[1600, 32, 3, 3]" = convolution_backward_2[1];  convolution_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_2: "b8[8, 1600, 7, 7]" = torch.ops.aten.le.Scalar(relu_108, 0);  relu_108 = None
    where_2: "f32[8, 1600, 7, 7]" = torch.ops.aten.where.self(le_2, full_default, getitem_230);  le_2 = getitem_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_6: "f32[1600]" = torch.ops.aten.sum.dim_IntList(where_2, [0, 2, 3])
    sub_119: "f32[8, 1600, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_107, unsqueeze_470);  convolution_107 = unsqueeze_470 = None
    mul_795: "f32[8, 1600, 7, 7]" = torch.ops.aten.mul.Tensor(where_2, sub_119)
    sum_7: "f32[1600]" = torch.ops.aten.sum.dim_IntList(mul_795, [0, 2, 3]);  mul_795 = None
    mul_796: "f32[1600]" = torch.ops.aten.mul.Tensor(sum_6, 0.002551020408163265)
    unsqueeze_471: "f32[1, 1600]" = torch.ops.aten.unsqueeze.default(mul_796, 0);  mul_796 = None
    unsqueeze_472: "f32[1, 1600, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_471, 2);  unsqueeze_471 = None
    unsqueeze_473: "f32[1, 1600, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_472, 3);  unsqueeze_472 = None
    mul_797: "f32[1600]" = torch.ops.aten.mul.Tensor(sum_7, 0.002551020408163265)
    mul_798: "f32[1600]" = torch.ops.aten.mul.Tensor(squeeze_325, squeeze_325)
    mul_799: "f32[1600]" = torch.ops.aten.mul.Tensor(mul_797, mul_798);  mul_797 = mul_798 = None
    unsqueeze_474: "f32[1, 1600]" = torch.ops.aten.unsqueeze.default(mul_799, 0);  mul_799 = None
    unsqueeze_475: "f32[1, 1600, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_474, 2);  unsqueeze_474 = None
    unsqueeze_476: "f32[1, 1600, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_475, 3);  unsqueeze_475 = None
    mul_800: "f32[1600]" = torch.ops.aten.mul.Tensor(squeeze_325, primals_217);  primals_217 = None
    unsqueeze_477: "f32[1, 1600]" = torch.ops.aten.unsqueeze.default(mul_800, 0);  mul_800 = None
    unsqueeze_478: "f32[1, 1600, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_477, 2);  unsqueeze_477 = None
    unsqueeze_479: "f32[1, 1600, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_478, 3);  unsqueeze_478 = None
    mul_801: "f32[8, 1600, 7, 7]" = torch.ops.aten.mul.Tensor(sub_119, unsqueeze_476);  sub_119 = unsqueeze_476 = None
    sub_121: "f32[8, 1600, 7, 7]" = torch.ops.aten.sub.Tensor(where_2, mul_801);  where_2 = mul_801 = None
    sub_122: "f32[8, 1600, 7, 7]" = torch.ops.aten.sub.Tensor(sub_121, unsqueeze_473);  sub_121 = unsqueeze_473 = None
    mul_802: "f32[8, 1600, 7, 7]" = torch.ops.aten.mul.Tensor(sub_122, unsqueeze_479);  sub_122 = unsqueeze_479 = None
    mul_803: "f32[1600]" = torch.ops.aten.mul.Tensor(sum_7, squeeze_325);  sum_7 = squeeze_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_3 = torch.ops.aten.convolution_backward.default(mul_802, relu_107, primals_330, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_802 = primals_330 = None
    getitem_233: "f32[8, 2560, 7, 7]" = convolution_backward_3[0]
    getitem_234: "f32[1600, 2560, 1, 1]" = convolution_backward_3[1];  convolution_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_3: "b8[8, 2560, 7, 7]" = torch.ops.aten.le.Scalar(relu_107, 0);  relu_107 = None
    where_3: "f32[8, 2560, 7, 7]" = torch.ops.aten.where.self(le_3, full_default, getitem_233);  le_3 = getitem_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_8: "f32[2560]" = torch.ops.aten.sum.dim_IntList(where_3, [0, 2, 3])
    sub_123: "f32[8, 2560, 7, 7]" = torch.ops.aten.sub.Tensor(cat_67, unsqueeze_482);  cat_67 = unsqueeze_482 = None
    mul_804: "f32[8, 2560, 7, 7]" = torch.ops.aten.mul.Tensor(where_3, sub_123)
    sum_9: "f32[2560]" = torch.ops.aten.sum.dim_IntList(mul_804, [0, 2, 3]);  mul_804 = None
    mul_805: "f32[2560]" = torch.ops.aten.mul.Tensor(sum_8, 0.002551020408163265)
    unsqueeze_483: "f32[1, 2560]" = torch.ops.aten.unsqueeze.default(mul_805, 0);  mul_805 = None
    unsqueeze_484: "f32[1, 2560, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_483, 2);  unsqueeze_483 = None
    unsqueeze_485: "f32[1, 2560, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_484, 3);  unsqueeze_484 = None
    mul_806: "f32[2560]" = torch.ops.aten.mul.Tensor(sum_9, 0.002551020408163265)
    mul_807: "f32[2560]" = torch.ops.aten.mul.Tensor(squeeze_322, squeeze_322)
    mul_808: "f32[2560]" = torch.ops.aten.mul.Tensor(mul_806, mul_807);  mul_806 = mul_807 = None
    unsqueeze_486: "f32[1, 2560]" = torch.ops.aten.unsqueeze.default(mul_808, 0);  mul_808 = None
    unsqueeze_487: "f32[1, 2560, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_486, 2);  unsqueeze_486 = None
    unsqueeze_488: "f32[1, 2560, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_487, 3);  unsqueeze_487 = None
    mul_809: "f32[2560]" = torch.ops.aten.mul.Tensor(squeeze_322, primals_215);  primals_215 = None
    unsqueeze_489: "f32[1, 2560]" = torch.ops.aten.unsqueeze.default(mul_809, 0);  mul_809 = None
    unsqueeze_490: "f32[1, 2560, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_489, 2);  unsqueeze_489 = None
    unsqueeze_491: "f32[1, 2560, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_490, 3);  unsqueeze_490 = None
    mul_810: "f32[8, 2560, 7, 7]" = torch.ops.aten.mul.Tensor(sub_123, unsqueeze_488);  sub_123 = unsqueeze_488 = None
    sub_125: "f32[8, 2560, 7, 7]" = torch.ops.aten.sub.Tensor(where_3, mul_810);  where_3 = mul_810 = None
    sub_126: "f32[8, 2560, 7, 7]" = torch.ops.aten.sub.Tensor(sub_125, unsqueeze_485);  sub_125 = unsqueeze_485 = None
    mul_811: "f32[8, 2560, 7, 7]" = torch.ops.aten.mul.Tensor(sub_126, unsqueeze_491);  sub_126 = unsqueeze_491 = None
    mul_812: "f32[2560]" = torch.ops.aten.mul.Tensor(sum_9, squeeze_322);  sum_9 = squeeze_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    slice_317: "f32[8, 2048, 7, 7]" = torch.ops.aten.slice.Tensor(mul_811, 1, 0, 2048)
    slice_318: "f32[8, 512, 7, 7]" = torch.ops.aten.slice.Tensor(mul_811, 1, 2048, 2560);  mul_811 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    add_591: "f32[8, 2048, 7, 7]" = torch.ops.aten.add.Tensor(slice_313, slice_317);  slice_313 = slice_317 = None
    add_592: "f32[8, 512, 7, 7]" = torch.ops.aten.add.Tensor(slice_315, slice_318);  slice_315 = slice_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    slice_319: "f32[8, 384, 7, 7]" = torch.ops.aten.slice.Tensor(add_592, 1, 0, 384)
    slice_320: "f32[8, 128, 7, 7]" = torch.ops.aten.slice.Tensor(add_592, 1, 384, 512);  add_592 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    slice_scatter_8: "f32[8, 128, 7, 7]" = torch.ops.aten.slice_scatter.default(full_default_1, slice_320, 3, 0, 9223372036854775807);  slice_320 = None
    slice_scatter_10: "f32[8, 2176, 7, 7]" = torch.ops.aten.slice_scatter.default(full_default_3, slice_scatter_8, 1, 2048, 9223372036854775807);  slice_scatter_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    slice_scatter_14: "f32[8, 2176, 7, 7]" = torch.ops.aten.slice_scatter.default(full_default_3, add_591, 1, 0, 2048)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    add_593: "f32[8, 2176, 7, 7]" = torch.ops.aten.add.Tensor(slice_scatter_10, slice_scatter_14);  slice_scatter_10 = slice_scatter_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_4 = torch.ops.aten.convolution_backward.default(add_593, relu_106, primals_329, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_593 = primals_329 = None
    getitem_236: "f32[8, 1600, 7, 7]" = convolution_backward_4[0]
    getitem_237: "f32[2176, 1600, 1, 1]" = convolution_backward_4[1];  convolution_backward_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_4: "b8[8, 1600, 7, 7]" = torch.ops.aten.le.Scalar(relu_106, 0);  relu_106 = None
    where_4: "f32[8, 1600, 7, 7]" = torch.ops.aten.where.self(le_4, full_default, getitem_236);  le_4 = getitem_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_10: "f32[1600]" = torch.ops.aten.sum.dim_IntList(where_4, [0, 2, 3])
    sub_127: "f32[8, 1600, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_105, unsqueeze_494);  convolution_105 = unsqueeze_494 = None
    mul_813: "f32[8, 1600, 7, 7]" = torch.ops.aten.mul.Tensor(where_4, sub_127)
    sum_11: "f32[1600]" = torch.ops.aten.sum.dim_IntList(mul_813, [0, 2, 3]);  mul_813 = None
    mul_814: "f32[1600]" = torch.ops.aten.mul.Tensor(sum_10, 0.002551020408163265)
    unsqueeze_495: "f32[1, 1600]" = torch.ops.aten.unsqueeze.default(mul_814, 0);  mul_814 = None
    unsqueeze_496: "f32[1, 1600, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_495, 2);  unsqueeze_495 = None
    unsqueeze_497: "f32[1, 1600, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_496, 3);  unsqueeze_496 = None
    mul_815: "f32[1600]" = torch.ops.aten.mul.Tensor(sum_11, 0.002551020408163265)
    mul_816: "f32[1600]" = torch.ops.aten.mul.Tensor(squeeze_319, squeeze_319)
    mul_817: "f32[1600]" = torch.ops.aten.mul.Tensor(mul_815, mul_816);  mul_815 = mul_816 = None
    unsqueeze_498: "f32[1, 1600]" = torch.ops.aten.unsqueeze.default(mul_817, 0);  mul_817 = None
    unsqueeze_499: "f32[1, 1600, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_498, 2);  unsqueeze_498 = None
    unsqueeze_500: "f32[1, 1600, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_499, 3);  unsqueeze_499 = None
    mul_818: "f32[1600]" = torch.ops.aten.mul.Tensor(squeeze_319, primals_213);  primals_213 = None
    unsqueeze_501: "f32[1, 1600]" = torch.ops.aten.unsqueeze.default(mul_818, 0);  mul_818 = None
    unsqueeze_502: "f32[1, 1600, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_501, 2);  unsqueeze_501 = None
    unsqueeze_503: "f32[1, 1600, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_502, 3);  unsqueeze_502 = None
    mul_819: "f32[8, 1600, 7, 7]" = torch.ops.aten.mul.Tensor(sub_127, unsqueeze_500);  sub_127 = unsqueeze_500 = None
    sub_129: "f32[8, 1600, 7, 7]" = torch.ops.aten.sub.Tensor(where_4, mul_819);  where_4 = mul_819 = None
    sub_130: "f32[8, 1600, 7, 7]" = torch.ops.aten.sub.Tensor(sub_129, unsqueeze_497);  sub_129 = unsqueeze_497 = None
    mul_820: "f32[8, 1600, 7, 7]" = torch.ops.aten.mul.Tensor(sub_130, unsqueeze_503);  sub_130 = unsqueeze_503 = None
    mul_821: "f32[1600]" = torch.ops.aten.mul.Tensor(sum_11, squeeze_319);  sum_11 = squeeze_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_5 = torch.ops.aten.convolution_backward.default(mul_820, relu_105, primals_328, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False]);  mul_820 = primals_328 = None
    getitem_239: "f32[8, 1600, 7, 7]" = convolution_backward_5[0]
    getitem_240: "f32[1600, 32, 3, 3]" = convolution_backward_5[1];  convolution_backward_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_5: "b8[8, 1600, 7, 7]" = torch.ops.aten.le.Scalar(relu_105, 0);  relu_105 = None
    where_5: "f32[8, 1600, 7, 7]" = torch.ops.aten.where.self(le_5, full_default, getitem_239);  le_5 = getitem_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_12: "f32[1600]" = torch.ops.aten.sum.dim_IntList(where_5, [0, 2, 3])
    sub_131: "f32[8, 1600, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_104, unsqueeze_506);  convolution_104 = unsqueeze_506 = None
    mul_822: "f32[8, 1600, 7, 7]" = torch.ops.aten.mul.Tensor(where_5, sub_131)
    sum_13: "f32[1600]" = torch.ops.aten.sum.dim_IntList(mul_822, [0, 2, 3]);  mul_822 = None
    mul_823: "f32[1600]" = torch.ops.aten.mul.Tensor(sum_12, 0.002551020408163265)
    unsqueeze_507: "f32[1, 1600]" = torch.ops.aten.unsqueeze.default(mul_823, 0);  mul_823 = None
    unsqueeze_508: "f32[1, 1600, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_507, 2);  unsqueeze_507 = None
    unsqueeze_509: "f32[1, 1600, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_508, 3);  unsqueeze_508 = None
    mul_824: "f32[1600]" = torch.ops.aten.mul.Tensor(sum_13, 0.002551020408163265)
    mul_825: "f32[1600]" = torch.ops.aten.mul.Tensor(squeeze_316, squeeze_316)
    mul_826: "f32[1600]" = torch.ops.aten.mul.Tensor(mul_824, mul_825);  mul_824 = mul_825 = None
    unsqueeze_510: "f32[1, 1600]" = torch.ops.aten.unsqueeze.default(mul_826, 0);  mul_826 = None
    unsqueeze_511: "f32[1, 1600, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_510, 2);  unsqueeze_510 = None
    unsqueeze_512: "f32[1, 1600, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_511, 3);  unsqueeze_511 = None
    mul_827: "f32[1600]" = torch.ops.aten.mul.Tensor(squeeze_316, primals_211);  primals_211 = None
    unsqueeze_513: "f32[1, 1600]" = torch.ops.aten.unsqueeze.default(mul_827, 0);  mul_827 = None
    unsqueeze_514: "f32[1, 1600, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_513, 2);  unsqueeze_513 = None
    unsqueeze_515: "f32[1, 1600, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_514, 3);  unsqueeze_514 = None
    mul_828: "f32[8, 1600, 7, 7]" = torch.ops.aten.mul.Tensor(sub_131, unsqueeze_512);  sub_131 = unsqueeze_512 = None
    sub_133: "f32[8, 1600, 7, 7]" = torch.ops.aten.sub.Tensor(where_5, mul_828);  where_5 = mul_828 = None
    sub_134: "f32[8, 1600, 7, 7]" = torch.ops.aten.sub.Tensor(sub_133, unsqueeze_509);  sub_133 = unsqueeze_509 = None
    mul_829: "f32[8, 1600, 7, 7]" = torch.ops.aten.mul.Tensor(sub_134, unsqueeze_515);  sub_134 = unsqueeze_515 = None
    mul_830: "f32[1600]" = torch.ops.aten.mul.Tensor(sum_13, squeeze_316);  sum_13 = squeeze_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_6 = torch.ops.aten.convolution_backward.default(mul_829, relu_104, primals_327, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_829 = primals_327 = None
    getitem_242: "f32[8, 2432, 7, 7]" = convolution_backward_6[0]
    getitem_243: "f32[1600, 2432, 1, 1]" = convolution_backward_6[1];  convolution_backward_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_6: "b8[8, 2432, 7, 7]" = torch.ops.aten.le.Scalar(relu_104, 0);  relu_104 = None
    where_6: "f32[8, 2432, 7, 7]" = torch.ops.aten.where.self(le_6, full_default, getitem_242);  le_6 = getitem_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_14: "f32[2432]" = torch.ops.aten.sum.dim_IntList(where_6, [0, 2, 3])
    sub_135: "f32[8, 2432, 7, 7]" = torch.ops.aten.sub.Tensor(cat_65, unsqueeze_518);  cat_65 = unsqueeze_518 = None
    mul_831: "f32[8, 2432, 7, 7]" = torch.ops.aten.mul.Tensor(where_6, sub_135)
    sum_15: "f32[2432]" = torch.ops.aten.sum.dim_IntList(mul_831, [0, 2, 3]);  mul_831 = None
    mul_832: "f32[2432]" = torch.ops.aten.mul.Tensor(sum_14, 0.002551020408163265)
    unsqueeze_519: "f32[1, 2432]" = torch.ops.aten.unsqueeze.default(mul_832, 0);  mul_832 = None
    unsqueeze_520: "f32[1, 2432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_519, 2);  unsqueeze_519 = None
    unsqueeze_521: "f32[1, 2432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_520, 3);  unsqueeze_520 = None
    mul_833: "f32[2432]" = torch.ops.aten.mul.Tensor(sum_15, 0.002551020408163265)
    mul_834: "f32[2432]" = torch.ops.aten.mul.Tensor(squeeze_313, squeeze_313)
    mul_835: "f32[2432]" = torch.ops.aten.mul.Tensor(mul_833, mul_834);  mul_833 = mul_834 = None
    unsqueeze_522: "f32[1, 2432]" = torch.ops.aten.unsqueeze.default(mul_835, 0);  mul_835 = None
    unsqueeze_523: "f32[1, 2432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_522, 2);  unsqueeze_522 = None
    unsqueeze_524: "f32[1, 2432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_523, 3);  unsqueeze_523 = None
    mul_836: "f32[2432]" = torch.ops.aten.mul.Tensor(squeeze_313, primals_209);  primals_209 = None
    unsqueeze_525: "f32[1, 2432]" = torch.ops.aten.unsqueeze.default(mul_836, 0);  mul_836 = None
    unsqueeze_526: "f32[1, 2432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_525, 2);  unsqueeze_525 = None
    unsqueeze_527: "f32[1, 2432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_526, 3);  unsqueeze_526 = None
    mul_837: "f32[8, 2432, 7, 7]" = torch.ops.aten.mul.Tensor(sub_135, unsqueeze_524);  sub_135 = unsqueeze_524 = None
    sub_137: "f32[8, 2432, 7, 7]" = torch.ops.aten.sub.Tensor(where_6, mul_837);  where_6 = mul_837 = None
    sub_138: "f32[8, 2432, 7, 7]" = torch.ops.aten.sub.Tensor(sub_137, unsqueeze_521);  sub_137 = unsqueeze_521 = None
    mul_838: "f32[8, 2432, 7, 7]" = torch.ops.aten.mul.Tensor(sub_138, unsqueeze_527);  sub_138 = unsqueeze_527 = None
    mul_839: "f32[2432]" = torch.ops.aten.mul.Tensor(sum_15, squeeze_313);  sum_15 = squeeze_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    slice_321: "f32[8, 2048, 7, 7]" = torch.ops.aten.slice.Tensor(mul_838, 1, 0, 2048)
    slice_322: "f32[8, 384, 7, 7]" = torch.ops.aten.slice.Tensor(mul_838, 1, 2048, 2432);  mul_838 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    add_594: "f32[8, 2048, 7, 7]" = torch.ops.aten.add.Tensor(add_591, slice_321);  add_591 = slice_321 = None
    add_595: "f32[8, 384, 7, 7]" = torch.ops.aten.add.Tensor(slice_319, slice_322);  slice_319 = slice_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    slice_323: "f32[8, 256, 7, 7]" = torch.ops.aten.slice.Tensor(add_595, 1, 0, 256)
    slice_324: "f32[8, 128, 7, 7]" = torch.ops.aten.slice.Tensor(add_595, 1, 256, 384);  add_595 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    slice_scatter_16: "f32[8, 128, 7, 7]" = torch.ops.aten.slice_scatter.default(full_default_1, slice_324, 3, 0, 9223372036854775807);  full_default_1 = slice_324 = None
    slice_scatter_18: "f32[8, 2176, 7, 7]" = torch.ops.aten.slice_scatter.default(full_default_3, slice_scatter_16, 1, 2048, 9223372036854775807);  slice_scatter_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    slice_scatter_22: "f32[8, 2176, 7, 7]" = torch.ops.aten.slice_scatter.default(full_default_3, add_594, 1, 0, 2048);  full_default_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    add_596: "f32[8, 2176, 7, 7]" = torch.ops.aten.add.Tensor(slice_scatter_18, slice_scatter_22);  slice_scatter_18 = slice_scatter_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_7 = torch.ops.aten.convolution_backward.default(add_596, relu_103, primals_326, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_596 = primals_326 = None
    getitem_245: "f32[8, 1600, 7, 7]" = convolution_backward_7[0]
    getitem_246: "f32[2176, 1600, 1, 1]" = convolution_backward_7[1];  convolution_backward_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_7: "b8[8, 1600, 7, 7]" = torch.ops.aten.le.Scalar(relu_103, 0);  relu_103 = None
    where_7: "f32[8, 1600, 7, 7]" = torch.ops.aten.where.self(le_7, full_default, getitem_245);  le_7 = getitem_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_16: "f32[1600]" = torch.ops.aten.sum.dim_IntList(where_7, [0, 2, 3])
    sub_139: "f32[8, 1600, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_102, unsqueeze_530);  convolution_102 = unsqueeze_530 = None
    mul_840: "f32[8, 1600, 7, 7]" = torch.ops.aten.mul.Tensor(where_7, sub_139)
    sum_17: "f32[1600]" = torch.ops.aten.sum.dim_IntList(mul_840, [0, 2, 3]);  mul_840 = None
    mul_841: "f32[1600]" = torch.ops.aten.mul.Tensor(sum_16, 0.002551020408163265)
    unsqueeze_531: "f32[1, 1600]" = torch.ops.aten.unsqueeze.default(mul_841, 0);  mul_841 = None
    unsqueeze_532: "f32[1, 1600, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_531, 2);  unsqueeze_531 = None
    unsqueeze_533: "f32[1, 1600, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_532, 3);  unsqueeze_532 = None
    mul_842: "f32[1600]" = torch.ops.aten.mul.Tensor(sum_17, 0.002551020408163265)
    mul_843: "f32[1600]" = torch.ops.aten.mul.Tensor(squeeze_310, squeeze_310)
    mul_844: "f32[1600]" = torch.ops.aten.mul.Tensor(mul_842, mul_843);  mul_842 = mul_843 = None
    unsqueeze_534: "f32[1, 1600]" = torch.ops.aten.unsqueeze.default(mul_844, 0);  mul_844 = None
    unsqueeze_535: "f32[1, 1600, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_534, 2);  unsqueeze_534 = None
    unsqueeze_536: "f32[1, 1600, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_535, 3);  unsqueeze_535 = None
    mul_845: "f32[1600]" = torch.ops.aten.mul.Tensor(squeeze_310, primals_207);  primals_207 = None
    unsqueeze_537: "f32[1, 1600]" = torch.ops.aten.unsqueeze.default(mul_845, 0);  mul_845 = None
    unsqueeze_538: "f32[1, 1600, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_537, 2);  unsqueeze_537 = None
    unsqueeze_539: "f32[1, 1600, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_538, 3);  unsqueeze_538 = None
    mul_846: "f32[8, 1600, 7, 7]" = torch.ops.aten.mul.Tensor(sub_139, unsqueeze_536);  sub_139 = unsqueeze_536 = None
    sub_141: "f32[8, 1600, 7, 7]" = torch.ops.aten.sub.Tensor(where_7, mul_846);  where_7 = mul_846 = None
    sub_142: "f32[8, 1600, 7, 7]" = torch.ops.aten.sub.Tensor(sub_141, unsqueeze_533);  sub_141 = unsqueeze_533 = None
    mul_847: "f32[8, 1600, 7, 7]" = torch.ops.aten.mul.Tensor(sub_142, unsqueeze_539);  sub_142 = unsqueeze_539 = None
    mul_848: "f32[1600]" = torch.ops.aten.mul.Tensor(sum_17, squeeze_310);  sum_17 = squeeze_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_8 = torch.ops.aten.convolution_backward.default(mul_847, relu_102, primals_325, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False]);  mul_847 = primals_325 = None
    getitem_248: "f32[8, 1600, 14, 14]" = convolution_backward_8[0]
    getitem_249: "f32[1600, 32, 3, 3]" = convolution_backward_8[1];  convolution_backward_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_8: "b8[8, 1600, 14, 14]" = torch.ops.aten.le.Scalar(relu_102, 0);  relu_102 = None
    where_8: "f32[8, 1600, 14, 14]" = torch.ops.aten.where.self(le_8, full_default, getitem_248);  le_8 = getitem_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_18: "f32[1600]" = torch.ops.aten.sum.dim_IntList(where_8, [0, 2, 3])
    sub_143: "f32[8, 1600, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_101, unsqueeze_542);  convolution_101 = unsqueeze_542 = None
    mul_849: "f32[8, 1600, 14, 14]" = torch.ops.aten.mul.Tensor(where_8, sub_143)
    sum_19: "f32[1600]" = torch.ops.aten.sum.dim_IntList(mul_849, [0, 2, 3]);  mul_849 = None
    mul_850: "f32[1600]" = torch.ops.aten.mul.Tensor(sum_18, 0.0006377551020408163)
    unsqueeze_543: "f32[1, 1600]" = torch.ops.aten.unsqueeze.default(mul_850, 0);  mul_850 = None
    unsqueeze_544: "f32[1, 1600, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_543, 2);  unsqueeze_543 = None
    unsqueeze_545: "f32[1, 1600, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_544, 3);  unsqueeze_544 = None
    mul_851: "f32[1600]" = torch.ops.aten.mul.Tensor(sum_19, 0.0006377551020408163)
    mul_852: "f32[1600]" = torch.ops.aten.mul.Tensor(squeeze_307, squeeze_307)
    mul_853: "f32[1600]" = torch.ops.aten.mul.Tensor(mul_851, mul_852);  mul_851 = mul_852 = None
    unsqueeze_546: "f32[1, 1600]" = torch.ops.aten.unsqueeze.default(mul_853, 0);  mul_853 = None
    unsqueeze_547: "f32[1, 1600, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_546, 2);  unsqueeze_546 = None
    unsqueeze_548: "f32[1, 1600, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_547, 3);  unsqueeze_547 = None
    mul_854: "f32[1600]" = torch.ops.aten.mul.Tensor(squeeze_307, primals_205);  primals_205 = None
    unsqueeze_549: "f32[1, 1600]" = torch.ops.aten.unsqueeze.default(mul_854, 0);  mul_854 = None
    unsqueeze_550: "f32[1, 1600, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_549, 2);  unsqueeze_549 = None
    unsqueeze_551: "f32[1, 1600, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_550, 3);  unsqueeze_550 = None
    mul_855: "f32[8, 1600, 14, 14]" = torch.ops.aten.mul.Tensor(sub_143, unsqueeze_548);  sub_143 = unsqueeze_548 = None
    sub_145: "f32[8, 1600, 14, 14]" = torch.ops.aten.sub.Tensor(where_8, mul_855);  where_8 = mul_855 = None
    sub_146: "f32[8, 1600, 14, 14]" = torch.ops.aten.sub.Tensor(sub_145, unsqueeze_545);  sub_145 = unsqueeze_545 = None
    mul_856: "f32[8, 1600, 14, 14]" = torch.ops.aten.mul.Tensor(sub_146, unsqueeze_551);  sub_146 = unsqueeze_551 = None
    mul_857: "f32[1600]" = torch.ops.aten.mul.Tensor(sum_19, squeeze_307);  sum_19 = squeeze_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_9 = torch.ops.aten.convolution_backward.default(mul_856, relu_101, primals_324, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_856 = primals_324 = None
    getitem_251: "f32[8, 2432, 14, 14]" = convolution_backward_9[0]
    getitem_252: "f32[1600, 2432, 1, 1]" = convolution_backward_9[1];  convolution_backward_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_9: "b8[8, 2432, 14, 14]" = torch.ops.aten.le.Scalar(relu_101, 0);  relu_101 = None
    where_9: "f32[8, 2432, 14, 14]" = torch.ops.aten.where.self(le_9, full_default, getitem_251);  le_9 = getitem_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_20: "f32[2432]" = torch.ops.aten.sum.dim_IntList(where_9, [0, 2, 3])
    sub_147: "f32[8, 2432, 14, 14]" = torch.ops.aten.sub.Tensor(cat_63, unsqueeze_554);  cat_63 = unsqueeze_554 = None
    mul_858: "f32[8, 2432, 14, 14]" = torch.ops.aten.mul.Tensor(where_9, sub_147)
    sum_21: "f32[2432]" = torch.ops.aten.sum.dim_IntList(mul_858, [0, 2, 3]);  mul_858 = None
    mul_859: "f32[2432]" = torch.ops.aten.mul.Tensor(sum_20, 0.0006377551020408163)
    unsqueeze_555: "f32[1, 2432]" = torch.ops.aten.unsqueeze.default(mul_859, 0);  mul_859 = None
    unsqueeze_556: "f32[1, 2432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_555, 2);  unsqueeze_555 = None
    unsqueeze_557: "f32[1, 2432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_556, 3);  unsqueeze_556 = None
    mul_860: "f32[2432]" = torch.ops.aten.mul.Tensor(sum_21, 0.0006377551020408163)
    mul_861: "f32[2432]" = torch.ops.aten.mul.Tensor(squeeze_301, squeeze_301)
    mul_862: "f32[2432]" = torch.ops.aten.mul.Tensor(mul_860, mul_861);  mul_860 = None
    unsqueeze_558: "f32[1, 2432]" = torch.ops.aten.unsqueeze.default(mul_862, 0);  mul_862 = None
    unsqueeze_559: "f32[1, 2432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_558, 2);  unsqueeze_558 = None
    unsqueeze_560: "f32[1, 2432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_559, 3);  unsqueeze_559 = None
    mul_863: "f32[2432]" = torch.ops.aten.mul.Tensor(squeeze_301, primals_203);  primals_203 = None
    unsqueeze_561: "f32[1, 2432]" = torch.ops.aten.unsqueeze.default(mul_863, 0);  mul_863 = None
    unsqueeze_562: "f32[1, 2432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_561, 2);  unsqueeze_561 = None
    unsqueeze_563: "f32[1, 2432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_562, 3);  unsqueeze_562 = None
    mul_864: "f32[8, 2432, 14, 14]" = torch.ops.aten.mul.Tensor(sub_147, unsqueeze_560);  unsqueeze_560 = None
    sub_149: "f32[8, 2432, 14, 14]" = torch.ops.aten.sub.Tensor(where_9, mul_864);  where_9 = mul_864 = None
    sub_150: "f32[8, 2432, 14, 14]" = torch.ops.aten.sub.Tensor(sub_149, unsqueeze_557);  sub_149 = unsqueeze_557 = None
    mul_865: "f32[8, 2432, 14, 14]" = torch.ops.aten.mul.Tensor(sub_150, unsqueeze_563);  sub_150 = unsqueeze_563 = None
    mul_866: "f32[2432]" = torch.ops.aten.mul.Tensor(sum_21, squeeze_301);  sum_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:134, code: x_s2 = x_s[:, self.num_1x1_c:, :, :]
    full_default_34: "f32[8, 256, 7, 7]" = torch.ops.aten.full.default([8, 256, 7, 7], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_24: "f32[8, 256, 7, 7]" = torch.ops.aten.slice_scatter.default(full_default_34, slice_323, 3, 0, 9223372036854775807);  full_default_34 = slice_323 = None
    full_default_36: "f32[8, 2304, 7, 7]" = torch.ops.aten.full.default([8, 2304, 7, 7], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_26: "f32[8, 2304, 7, 7]" = torch.ops.aten.slice_scatter.default(full_default_36, slice_scatter_24, 1, 2048, 9223372036854775807);  slice_scatter_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:133, code: x_s1 = x_s[:, :self.num_1x1_c, :, :]
    slice_scatter_30: "f32[8, 2304, 7, 7]" = torch.ops.aten.slice_scatter.default(full_default_36, add_594, 1, 0, 2048);  full_default_36 = add_594 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:133, code: x_s1 = x_s[:, :self.num_1x1_c, :, :]
    add_597: "f32[8, 2304, 7, 7]" = torch.ops.aten.add.Tensor(slice_scatter_26, slice_scatter_30);  slice_scatter_26 = slice_scatter_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_10 = torch.ops.aten.convolution_backward.default(add_597, relu_100, primals_323, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_597 = primals_323 = None
    getitem_254: "f32[8, 2432, 14, 14]" = convolution_backward_10[0]
    getitem_255: "f32[2304, 2432, 1, 1]" = convolution_backward_10[1];  convolution_backward_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_10: "b8[8, 2432, 14, 14]" = torch.ops.aten.le.Scalar(relu_100, 0);  relu_100 = None
    where_10: "f32[8, 2432, 14, 14]" = torch.ops.aten.where.self(le_10, full_default, getitem_254);  le_10 = getitem_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_22: "f32[2432]" = torch.ops.aten.sum.dim_IntList(where_10, [0, 2, 3])
    mul_867: "f32[8, 2432, 14, 14]" = torch.ops.aten.mul.Tensor(where_10, sub_147)
    sum_23: "f32[2432]" = torch.ops.aten.sum.dim_IntList(mul_867, [0, 2, 3]);  mul_867 = None
    mul_868: "f32[2432]" = torch.ops.aten.mul.Tensor(sum_22, 0.0006377551020408163)
    unsqueeze_567: "f32[1, 2432]" = torch.ops.aten.unsqueeze.default(mul_868, 0);  mul_868 = None
    unsqueeze_568: "f32[1, 2432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_567, 2);  unsqueeze_567 = None
    unsqueeze_569: "f32[1, 2432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_568, 3);  unsqueeze_568 = None
    mul_869: "f32[2432]" = torch.ops.aten.mul.Tensor(sum_23, 0.0006377551020408163)
    mul_871: "f32[2432]" = torch.ops.aten.mul.Tensor(mul_869, mul_861);  mul_869 = mul_861 = None
    unsqueeze_570: "f32[1, 2432]" = torch.ops.aten.unsqueeze.default(mul_871, 0);  mul_871 = None
    unsqueeze_571: "f32[1, 2432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_570, 2);  unsqueeze_570 = None
    unsqueeze_572: "f32[1, 2432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_571, 3);  unsqueeze_571 = None
    mul_872: "f32[2432]" = torch.ops.aten.mul.Tensor(squeeze_301, primals_201);  primals_201 = None
    unsqueeze_573: "f32[1, 2432]" = torch.ops.aten.unsqueeze.default(mul_872, 0);  mul_872 = None
    unsqueeze_574: "f32[1, 2432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_573, 2);  unsqueeze_573 = None
    unsqueeze_575: "f32[1, 2432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_574, 3);  unsqueeze_574 = None
    mul_873: "f32[8, 2432, 14, 14]" = torch.ops.aten.mul.Tensor(sub_147, unsqueeze_572);  sub_147 = unsqueeze_572 = None
    sub_153: "f32[8, 2432, 14, 14]" = torch.ops.aten.sub.Tensor(where_10, mul_873);  where_10 = mul_873 = None
    sub_154: "f32[8, 2432, 14, 14]" = torch.ops.aten.sub.Tensor(sub_153, unsqueeze_569);  sub_153 = unsqueeze_569 = None
    mul_874: "f32[8, 2432, 14, 14]" = torch.ops.aten.mul.Tensor(sub_154, unsqueeze_575);  sub_154 = unsqueeze_575 = None
    mul_875: "f32[2432]" = torch.ops.aten.mul.Tensor(sum_23, squeeze_301);  sum_23 = squeeze_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_598: "f32[8, 2432, 14, 14]" = torch.ops.aten.add.Tensor(mul_865, mul_874);  mul_865 = mul_874 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    slice_325: "f32[8, 1024, 14, 14]" = torch.ops.aten.slice.Tensor(add_598, 1, 0, 1024)
    slice_326: "f32[8, 1408, 14, 14]" = torch.ops.aten.slice.Tensor(add_598, 1, 1024, 2432);  add_598 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    slice_327: "f32[8, 1344, 14, 14]" = torch.ops.aten.slice.Tensor(slice_326, 1, 0, 1344)
    slice_328: "f32[8, 64, 14, 14]" = torch.ops.aten.slice.Tensor(slice_326, 1, 1344, 1408);  slice_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    full_default_43: "f32[8, 64, 14, 14]" = torch.ops.aten.full.default([8, 64, 14, 14], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_32: "f32[8, 64, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_43, slice_328, 3, 0, 9223372036854775807);  slice_328 = None
    full_default_45: "f32[8, 1088, 14, 14]" = torch.ops.aten.full.default([8, 1088, 14, 14], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_34: "f32[8, 1088, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_45, slice_scatter_32, 1, 1024, 9223372036854775807);  slice_scatter_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    full_default_47: "f32[8, 1024, 14, 14]" = torch.ops.aten.full.default([8, 1024, 14, 14], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_36: "f32[8, 1024, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_47, slice_325, 3, 0, 9223372036854775807);  full_default_47 = None
    slice_scatter_38: "f32[8, 1088, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_45, slice_scatter_36, 1, 0, 1024);  slice_scatter_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    add_599: "f32[8, 1088, 14, 14]" = torch.ops.aten.add.Tensor(slice_scatter_34, slice_scatter_38);  slice_scatter_34 = slice_scatter_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_11 = torch.ops.aten.convolution_backward.default(add_599, relu_99, primals_322, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_599 = primals_322 = None
    getitem_257: "f32[8, 800, 14, 14]" = convolution_backward_11[0]
    getitem_258: "f32[1088, 800, 1, 1]" = convolution_backward_11[1];  convolution_backward_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_11: "b8[8, 800, 14, 14]" = torch.ops.aten.le.Scalar(relu_99, 0);  relu_99 = None
    where_11: "f32[8, 800, 14, 14]" = torch.ops.aten.where.self(le_11, full_default, getitem_257);  le_11 = getitem_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_24: "f32[800]" = torch.ops.aten.sum.dim_IntList(where_11, [0, 2, 3])
    sub_155: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_98, unsqueeze_578);  convolution_98 = unsqueeze_578 = None
    mul_876: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(where_11, sub_155)
    sum_25: "f32[800]" = torch.ops.aten.sum.dim_IntList(mul_876, [0, 2, 3]);  mul_876 = None
    mul_877: "f32[800]" = torch.ops.aten.mul.Tensor(sum_24, 0.0006377551020408163)
    unsqueeze_579: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_877, 0);  mul_877 = None
    unsqueeze_580: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_579, 2);  unsqueeze_579 = None
    unsqueeze_581: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_580, 3);  unsqueeze_580 = None
    mul_878: "f32[800]" = torch.ops.aten.mul.Tensor(sum_25, 0.0006377551020408163)
    mul_879: "f32[800]" = torch.ops.aten.mul.Tensor(squeeze_298, squeeze_298)
    mul_880: "f32[800]" = torch.ops.aten.mul.Tensor(mul_878, mul_879);  mul_878 = mul_879 = None
    unsqueeze_582: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_880, 0);  mul_880 = None
    unsqueeze_583: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_582, 2);  unsqueeze_582 = None
    unsqueeze_584: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_583, 3);  unsqueeze_583 = None
    mul_881: "f32[800]" = torch.ops.aten.mul.Tensor(squeeze_298, primals_199);  primals_199 = None
    unsqueeze_585: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_881, 0);  mul_881 = None
    unsqueeze_586: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_585, 2);  unsqueeze_585 = None
    unsqueeze_587: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_586, 3);  unsqueeze_586 = None
    mul_882: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_155, unsqueeze_584);  sub_155 = unsqueeze_584 = None
    sub_157: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(where_11, mul_882);  where_11 = mul_882 = None
    sub_158: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(sub_157, unsqueeze_581);  sub_157 = unsqueeze_581 = None
    mul_883: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_158, unsqueeze_587);  sub_158 = unsqueeze_587 = None
    mul_884: "f32[800]" = torch.ops.aten.mul.Tensor(sum_25, squeeze_298);  sum_25 = squeeze_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_12 = torch.ops.aten.convolution_backward.default(mul_883, relu_98, primals_321, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False]);  mul_883 = primals_321 = None
    getitem_260: "f32[8, 800, 14, 14]" = convolution_backward_12[0]
    getitem_261: "f32[800, 16, 3, 3]" = convolution_backward_12[1];  convolution_backward_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_12: "b8[8, 800, 14, 14]" = torch.ops.aten.le.Scalar(relu_98, 0);  relu_98 = None
    where_12: "f32[8, 800, 14, 14]" = torch.ops.aten.where.self(le_12, full_default, getitem_260);  le_12 = getitem_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_26: "f32[800]" = torch.ops.aten.sum.dim_IntList(where_12, [0, 2, 3])
    sub_159: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_97, unsqueeze_590);  convolution_97 = unsqueeze_590 = None
    mul_885: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(where_12, sub_159)
    sum_27: "f32[800]" = torch.ops.aten.sum.dim_IntList(mul_885, [0, 2, 3]);  mul_885 = None
    mul_886: "f32[800]" = torch.ops.aten.mul.Tensor(sum_26, 0.0006377551020408163)
    unsqueeze_591: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_886, 0);  mul_886 = None
    unsqueeze_592: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_591, 2);  unsqueeze_591 = None
    unsqueeze_593: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_592, 3);  unsqueeze_592 = None
    mul_887: "f32[800]" = torch.ops.aten.mul.Tensor(sum_27, 0.0006377551020408163)
    mul_888: "f32[800]" = torch.ops.aten.mul.Tensor(squeeze_295, squeeze_295)
    mul_889: "f32[800]" = torch.ops.aten.mul.Tensor(mul_887, mul_888);  mul_887 = mul_888 = None
    unsqueeze_594: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_889, 0);  mul_889 = None
    unsqueeze_595: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_594, 2);  unsqueeze_594 = None
    unsqueeze_596: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_595, 3);  unsqueeze_595 = None
    mul_890: "f32[800]" = torch.ops.aten.mul.Tensor(squeeze_295, primals_197);  primals_197 = None
    unsqueeze_597: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_890, 0);  mul_890 = None
    unsqueeze_598: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_597, 2);  unsqueeze_597 = None
    unsqueeze_599: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_598, 3);  unsqueeze_598 = None
    mul_891: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_159, unsqueeze_596);  sub_159 = unsqueeze_596 = None
    sub_161: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(where_12, mul_891);  where_12 = mul_891 = None
    sub_162: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(sub_161, unsqueeze_593);  sub_161 = unsqueeze_593 = None
    mul_892: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_162, unsqueeze_599);  sub_162 = unsqueeze_599 = None
    mul_893: "f32[800]" = torch.ops.aten.mul.Tensor(sum_27, squeeze_295);  sum_27 = squeeze_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_13 = torch.ops.aten.convolution_backward.default(mul_892, relu_97, primals_320, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_892 = primals_320 = None
    getitem_263: "f32[8, 2368, 14, 14]" = convolution_backward_13[0]
    getitem_264: "f32[800, 2368, 1, 1]" = convolution_backward_13[1];  convolution_backward_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_13: "b8[8, 2368, 14, 14]" = torch.ops.aten.le.Scalar(relu_97, 0);  relu_97 = None
    where_13: "f32[8, 2368, 14, 14]" = torch.ops.aten.where.self(le_13, full_default, getitem_263);  le_13 = getitem_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_28: "f32[2368]" = torch.ops.aten.sum.dim_IntList(where_13, [0, 2, 3])
    sub_163: "f32[8, 2368, 14, 14]" = torch.ops.aten.sub.Tensor(cat_61, unsqueeze_602);  cat_61 = unsqueeze_602 = None
    mul_894: "f32[8, 2368, 14, 14]" = torch.ops.aten.mul.Tensor(where_13, sub_163)
    sum_29: "f32[2368]" = torch.ops.aten.sum.dim_IntList(mul_894, [0, 2, 3]);  mul_894 = None
    mul_895: "f32[2368]" = torch.ops.aten.mul.Tensor(sum_28, 0.0006377551020408163)
    unsqueeze_603: "f32[1, 2368]" = torch.ops.aten.unsqueeze.default(mul_895, 0);  mul_895 = None
    unsqueeze_604: "f32[1, 2368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_603, 2);  unsqueeze_603 = None
    unsqueeze_605: "f32[1, 2368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_604, 3);  unsqueeze_604 = None
    mul_896: "f32[2368]" = torch.ops.aten.mul.Tensor(sum_29, 0.0006377551020408163)
    mul_897: "f32[2368]" = torch.ops.aten.mul.Tensor(squeeze_292, squeeze_292)
    mul_898: "f32[2368]" = torch.ops.aten.mul.Tensor(mul_896, mul_897);  mul_896 = mul_897 = None
    unsqueeze_606: "f32[1, 2368]" = torch.ops.aten.unsqueeze.default(mul_898, 0);  mul_898 = None
    unsqueeze_607: "f32[1, 2368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_606, 2);  unsqueeze_606 = None
    unsqueeze_608: "f32[1, 2368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_607, 3);  unsqueeze_607 = None
    mul_899: "f32[2368]" = torch.ops.aten.mul.Tensor(squeeze_292, primals_195);  primals_195 = None
    unsqueeze_609: "f32[1, 2368]" = torch.ops.aten.unsqueeze.default(mul_899, 0);  mul_899 = None
    unsqueeze_610: "f32[1, 2368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_609, 2);  unsqueeze_609 = None
    unsqueeze_611: "f32[1, 2368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_610, 3);  unsqueeze_610 = None
    mul_900: "f32[8, 2368, 14, 14]" = torch.ops.aten.mul.Tensor(sub_163, unsqueeze_608);  sub_163 = unsqueeze_608 = None
    sub_165: "f32[8, 2368, 14, 14]" = torch.ops.aten.sub.Tensor(where_13, mul_900);  where_13 = mul_900 = None
    sub_166: "f32[8, 2368, 14, 14]" = torch.ops.aten.sub.Tensor(sub_165, unsqueeze_605);  sub_165 = unsqueeze_605 = None
    mul_901: "f32[8, 2368, 14, 14]" = torch.ops.aten.mul.Tensor(sub_166, unsqueeze_611);  sub_166 = unsqueeze_611 = None
    mul_902: "f32[2368]" = torch.ops.aten.mul.Tensor(sum_29, squeeze_292);  sum_29 = squeeze_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    slice_329: "f32[8, 1024, 14, 14]" = torch.ops.aten.slice.Tensor(mul_901, 1, 0, 1024)
    slice_330: "f32[8, 1344, 14, 14]" = torch.ops.aten.slice.Tensor(mul_901, 1, 1024, 2368);  mul_901 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    add_600: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(slice_325, slice_329);  slice_325 = slice_329 = None
    add_601: "f32[8, 1344, 14, 14]" = torch.ops.aten.add.Tensor(slice_327, slice_330);  slice_327 = slice_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    slice_331: "f32[8, 1280, 14, 14]" = torch.ops.aten.slice.Tensor(add_601, 1, 0, 1280)
    slice_332: "f32[8, 64, 14, 14]" = torch.ops.aten.slice.Tensor(add_601, 1, 1280, 1344);  add_601 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    slice_scatter_40: "f32[8, 64, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_43, slice_332, 3, 0, 9223372036854775807);  slice_332 = None
    slice_scatter_42: "f32[8, 1088, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_45, slice_scatter_40, 1, 1024, 9223372036854775807);  slice_scatter_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    slice_scatter_46: "f32[8, 1088, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_45, add_600, 1, 0, 1024)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    add_602: "f32[8, 1088, 14, 14]" = torch.ops.aten.add.Tensor(slice_scatter_42, slice_scatter_46);  slice_scatter_42 = slice_scatter_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_14 = torch.ops.aten.convolution_backward.default(add_602, relu_96, primals_319, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_602 = primals_319 = None
    getitem_266: "f32[8, 800, 14, 14]" = convolution_backward_14[0]
    getitem_267: "f32[1088, 800, 1, 1]" = convolution_backward_14[1];  convolution_backward_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_14: "b8[8, 800, 14, 14]" = torch.ops.aten.le.Scalar(relu_96, 0);  relu_96 = None
    where_14: "f32[8, 800, 14, 14]" = torch.ops.aten.where.self(le_14, full_default, getitem_266);  le_14 = getitem_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_30: "f32[800]" = torch.ops.aten.sum.dim_IntList(where_14, [0, 2, 3])
    sub_167: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_95, unsqueeze_614);  convolution_95 = unsqueeze_614 = None
    mul_903: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(where_14, sub_167)
    sum_31: "f32[800]" = torch.ops.aten.sum.dim_IntList(mul_903, [0, 2, 3]);  mul_903 = None
    mul_904: "f32[800]" = torch.ops.aten.mul.Tensor(sum_30, 0.0006377551020408163)
    unsqueeze_615: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_904, 0);  mul_904 = None
    unsqueeze_616: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_615, 2);  unsqueeze_615 = None
    unsqueeze_617: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_616, 3);  unsqueeze_616 = None
    mul_905: "f32[800]" = torch.ops.aten.mul.Tensor(sum_31, 0.0006377551020408163)
    mul_906: "f32[800]" = torch.ops.aten.mul.Tensor(squeeze_289, squeeze_289)
    mul_907: "f32[800]" = torch.ops.aten.mul.Tensor(mul_905, mul_906);  mul_905 = mul_906 = None
    unsqueeze_618: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_907, 0);  mul_907 = None
    unsqueeze_619: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_618, 2);  unsqueeze_618 = None
    unsqueeze_620: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_619, 3);  unsqueeze_619 = None
    mul_908: "f32[800]" = torch.ops.aten.mul.Tensor(squeeze_289, primals_193);  primals_193 = None
    unsqueeze_621: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_908, 0);  mul_908 = None
    unsqueeze_622: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_621, 2);  unsqueeze_621 = None
    unsqueeze_623: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_622, 3);  unsqueeze_622 = None
    mul_909: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_167, unsqueeze_620);  sub_167 = unsqueeze_620 = None
    sub_169: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(where_14, mul_909);  where_14 = mul_909 = None
    sub_170: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(sub_169, unsqueeze_617);  sub_169 = unsqueeze_617 = None
    mul_910: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_170, unsqueeze_623);  sub_170 = unsqueeze_623 = None
    mul_911: "f32[800]" = torch.ops.aten.mul.Tensor(sum_31, squeeze_289);  sum_31 = squeeze_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_15 = torch.ops.aten.convolution_backward.default(mul_910, relu_95, primals_318, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False]);  mul_910 = primals_318 = None
    getitem_269: "f32[8, 800, 14, 14]" = convolution_backward_15[0]
    getitem_270: "f32[800, 16, 3, 3]" = convolution_backward_15[1];  convolution_backward_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_15: "b8[8, 800, 14, 14]" = torch.ops.aten.le.Scalar(relu_95, 0);  relu_95 = None
    where_15: "f32[8, 800, 14, 14]" = torch.ops.aten.where.self(le_15, full_default, getitem_269);  le_15 = getitem_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_32: "f32[800]" = torch.ops.aten.sum.dim_IntList(where_15, [0, 2, 3])
    sub_171: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_94, unsqueeze_626);  convolution_94 = unsqueeze_626 = None
    mul_912: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(where_15, sub_171)
    sum_33: "f32[800]" = torch.ops.aten.sum.dim_IntList(mul_912, [0, 2, 3]);  mul_912 = None
    mul_913: "f32[800]" = torch.ops.aten.mul.Tensor(sum_32, 0.0006377551020408163)
    unsqueeze_627: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_913, 0);  mul_913 = None
    unsqueeze_628: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_627, 2);  unsqueeze_627 = None
    unsqueeze_629: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_628, 3);  unsqueeze_628 = None
    mul_914: "f32[800]" = torch.ops.aten.mul.Tensor(sum_33, 0.0006377551020408163)
    mul_915: "f32[800]" = torch.ops.aten.mul.Tensor(squeeze_286, squeeze_286)
    mul_916: "f32[800]" = torch.ops.aten.mul.Tensor(mul_914, mul_915);  mul_914 = mul_915 = None
    unsqueeze_630: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_916, 0);  mul_916 = None
    unsqueeze_631: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_630, 2);  unsqueeze_630 = None
    unsqueeze_632: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_631, 3);  unsqueeze_631 = None
    mul_917: "f32[800]" = torch.ops.aten.mul.Tensor(squeeze_286, primals_191);  primals_191 = None
    unsqueeze_633: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_917, 0);  mul_917 = None
    unsqueeze_634: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_633, 2);  unsqueeze_633 = None
    unsqueeze_635: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_634, 3);  unsqueeze_634 = None
    mul_918: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_171, unsqueeze_632);  sub_171 = unsqueeze_632 = None
    sub_173: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(where_15, mul_918);  where_15 = mul_918 = None
    sub_174: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(sub_173, unsqueeze_629);  sub_173 = unsqueeze_629 = None
    mul_919: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_174, unsqueeze_635);  sub_174 = unsqueeze_635 = None
    mul_920: "f32[800]" = torch.ops.aten.mul.Tensor(sum_33, squeeze_286);  sum_33 = squeeze_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_16 = torch.ops.aten.convolution_backward.default(mul_919, relu_94, primals_317, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_919 = primals_317 = None
    getitem_272: "f32[8, 2304, 14, 14]" = convolution_backward_16[0]
    getitem_273: "f32[800, 2304, 1, 1]" = convolution_backward_16[1];  convolution_backward_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_16: "b8[8, 2304, 14, 14]" = torch.ops.aten.le.Scalar(relu_94, 0);  relu_94 = None
    where_16: "f32[8, 2304, 14, 14]" = torch.ops.aten.where.self(le_16, full_default, getitem_272);  le_16 = getitem_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_34: "f32[2304]" = torch.ops.aten.sum.dim_IntList(where_16, [0, 2, 3])
    sub_175: "f32[8, 2304, 14, 14]" = torch.ops.aten.sub.Tensor(cat_59, unsqueeze_638);  cat_59 = unsqueeze_638 = None
    mul_921: "f32[8, 2304, 14, 14]" = torch.ops.aten.mul.Tensor(where_16, sub_175)
    sum_35: "f32[2304]" = torch.ops.aten.sum.dim_IntList(mul_921, [0, 2, 3]);  mul_921 = None
    mul_922: "f32[2304]" = torch.ops.aten.mul.Tensor(sum_34, 0.0006377551020408163)
    unsqueeze_639: "f32[1, 2304]" = torch.ops.aten.unsqueeze.default(mul_922, 0);  mul_922 = None
    unsqueeze_640: "f32[1, 2304, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_639, 2);  unsqueeze_639 = None
    unsqueeze_641: "f32[1, 2304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_640, 3);  unsqueeze_640 = None
    mul_923: "f32[2304]" = torch.ops.aten.mul.Tensor(sum_35, 0.0006377551020408163)
    mul_924: "f32[2304]" = torch.ops.aten.mul.Tensor(squeeze_283, squeeze_283)
    mul_925: "f32[2304]" = torch.ops.aten.mul.Tensor(mul_923, mul_924);  mul_923 = mul_924 = None
    unsqueeze_642: "f32[1, 2304]" = torch.ops.aten.unsqueeze.default(mul_925, 0);  mul_925 = None
    unsqueeze_643: "f32[1, 2304, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_642, 2);  unsqueeze_642 = None
    unsqueeze_644: "f32[1, 2304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_643, 3);  unsqueeze_643 = None
    mul_926: "f32[2304]" = torch.ops.aten.mul.Tensor(squeeze_283, primals_189);  primals_189 = None
    unsqueeze_645: "f32[1, 2304]" = torch.ops.aten.unsqueeze.default(mul_926, 0);  mul_926 = None
    unsqueeze_646: "f32[1, 2304, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_645, 2);  unsqueeze_645 = None
    unsqueeze_647: "f32[1, 2304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_646, 3);  unsqueeze_646 = None
    mul_927: "f32[8, 2304, 14, 14]" = torch.ops.aten.mul.Tensor(sub_175, unsqueeze_644);  sub_175 = unsqueeze_644 = None
    sub_177: "f32[8, 2304, 14, 14]" = torch.ops.aten.sub.Tensor(where_16, mul_927);  where_16 = mul_927 = None
    sub_178: "f32[8, 2304, 14, 14]" = torch.ops.aten.sub.Tensor(sub_177, unsqueeze_641);  sub_177 = unsqueeze_641 = None
    mul_928: "f32[8, 2304, 14, 14]" = torch.ops.aten.mul.Tensor(sub_178, unsqueeze_647);  sub_178 = unsqueeze_647 = None
    mul_929: "f32[2304]" = torch.ops.aten.mul.Tensor(sum_35, squeeze_283);  sum_35 = squeeze_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    slice_333: "f32[8, 1024, 14, 14]" = torch.ops.aten.slice.Tensor(mul_928, 1, 0, 1024)
    slice_334: "f32[8, 1280, 14, 14]" = torch.ops.aten.slice.Tensor(mul_928, 1, 1024, 2304);  mul_928 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    add_603: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_600, slice_333);  add_600 = slice_333 = None
    add_604: "f32[8, 1280, 14, 14]" = torch.ops.aten.add.Tensor(slice_331, slice_334);  slice_331 = slice_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    slice_335: "f32[8, 1216, 14, 14]" = torch.ops.aten.slice.Tensor(add_604, 1, 0, 1216)
    slice_336: "f32[8, 64, 14, 14]" = torch.ops.aten.slice.Tensor(add_604, 1, 1216, 1280);  add_604 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    slice_scatter_48: "f32[8, 64, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_43, slice_336, 3, 0, 9223372036854775807);  slice_336 = None
    slice_scatter_50: "f32[8, 1088, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_45, slice_scatter_48, 1, 1024, 9223372036854775807);  slice_scatter_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    slice_scatter_54: "f32[8, 1088, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_45, add_603, 1, 0, 1024)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    add_605: "f32[8, 1088, 14, 14]" = torch.ops.aten.add.Tensor(slice_scatter_50, slice_scatter_54);  slice_scatter_50 = slice_scatter_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_17 = torch.ops.aten.convolution_backward.default(add_605, relu_93, primals_316, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_605 = primals_316 = None
    getitem_275: "f32[8, 800, 14, 14]" = convolution_backward_17[0]
    getitem_276: "f32[1088, 800, 1, 1]" = convolution_backward_17[1];  convolution_backward_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_17: "b8[8, 800, 14, 14]" = torch.ops.aten.le.Scalar(relu_93, 0);  relu_93 = None
    where_17: "f32[8, 800, 14, 14]" = torch.ops.aten.where.self(le_17, full_default, getitem_275);  le_17 = getitem_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_36: "f32[800]" = torch.ops.aten.sum.dim_IntList(where_17, [0, 2, 3])
    sub_179: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_92, unsqueeze_650);  convolution_92 = unsqueeze_650 = None
    mul_930: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(where_17, sub_179)
    sum_37: "f32[800]" = torch.ops.aten.sum.dim_IntList(mul_930, [0, 2, 3]);  mul_930 = None
    mul_931: "f32[800]" = torch.ops.aten.mul.Tensor(sum_36, 0.0006377551020408163)
    unsqueeze_651: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_931, 0);  mul_931 = None
    unsqueeze_652: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_651, 2);  unsqueeze_651 = None
    unsqueeze_653: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_652, 3);  unsqueeze_652 = None
    mul_932: "f32[800]" = torch.ops.aten.mul.Tensor(sum_37, 0.0006377551020408163)
    mul_933: "f32[800]" = torch.ops.aten.mul.Tensor(squeeze_280, squeeze_280)
    mul_934: "f32[800]" = torch.ops.aten.mul.Tensor(mul_932, mul_933);  mul_932 = mul_933 = None
    unsqueeze_654: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_934, 0);  mul_934 = None
    unsqueeze_655: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_654, 2);  unsqueeze_654 = None
    unsqueeze_656: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_655, 3);  unsqueeze_655 = None
    mul_935: "f32[800]" = torch.ops.aten.mul.Tensor(squeeze_280, primals_187);  primals_187 = None
    unsqueeze_657: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_935, 0);  mul_935 = None
    unsqueeze_658: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_657, 2);  unsqueeze_657 = None
    unsqueeze_659: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_658, 3);  unsqueeze_658 = None
    mul_936: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_179, unsqueeze_656);  sub_179 = unsqueeze_656 = None
    sub_181: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(where_17, mul_936);  where_17 = mul_936 = None
    sub_182: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(sub_181, unsqueeze_653);  sub_181 = unsqueeze_653 = None
    mul_937: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_182, unsqueeze_659);  sub_182 = unsqueeze_659 = None
    mul_938: "f32[800]" = torch.ops.aten.mul.Tensor(sum_37, squeeze_280);  sum_37 = squeeze_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_18 = torch.ops.aten.convolution_backward.default(mul_937, relu_92, primals_315, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False]);  mul_937 = primals_315 = None
    getitem_278: "f32[8, 800, 14, 14]" = convolution_backward_18[0]
    getitem_279: "f32[800, 16, 3, 3]" = convolution_backward_18[1];  convolution_backward_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_18: "b8[8, 800, 14, 14]" = torch.ops.aten.le.Scalar(relu_92, 0);  relu_92 = None
    where_18: "f32[8, 800, 14, 14]" = torch.ops.aten.where.self(le_18, full_default, getitem_278);  le_18 = getitem_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_38: "f32[800]" = torch.ops.aten.sum.dim_IntList(where_18, [0, 2, 3])
    sub_183: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_91, unsqueeze_662);  convolution_91 = unsqueeze_662 = None
    mul_939: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(where_18, sub_183)
    sum_39: "f32[800]" = torch.ops.aten.sum.dim_IntList(mul_939, [0, 2, 3]);  mul_939 = None
    mul_940: "f32[800]" = torch.ops.aten.mul.Tensor(sum_38, 0.0006377551020408163)
    unsqueeze_663: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_940, 0);  mul_940 = None
    unsqueeze_664: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_663, 2);  unsqueeze_663 = None
    unsqueeze_665: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_664, 3);  unsqueeze_664 = None
    mul_941: "f32[800]" = torch.ops.aten.mul.Tensor(sum_39, 0.0006377551020408163)
    mul_942: "f32[800]" = torch.ops.aten.mul.Tensor(squeeze_277, squeeze_277)
    mul_943: "f32[800]" = torch.ops.aten.mul.Tensor(mul_941, mul_942);  mul_941 = mul_942 = None
    unsqueeze_666: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_943, 0);  mul_943 = None
    unsqueeze_667: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_666, 2);  unsqueeze_666 = None
    unsqueeze_668: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_667, 3);  unsqueeze_667 = None
    mul_944: "f32[800]" = torch.ops.aten.mul.Tensor(squeeze_277, primals_185);  primals_185 = None
    unsqueeze_669: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_944, 0);  mul_944 = None
    unsqueeze_670: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_669, 2);  unsqueeze_669 = None
    unsqueeze_671: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_670, 3);  unsqueeze_670 = None
    mul_945: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_183, unsqueeze_668);  sub_183 = unsqueeze_668 = None
    sub_185: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(where_18, mul_945);  where_18 = mul_945 = None
    sub_186: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(sub_185, unsqueeze_665);  sub_185 = unsqueeze_665 = None
    mul_946: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_186, unsqueeze_671);  sub_186 = unsqueeze_671 = None
    mul_947: "f32[800]" = torch.ops.aten.mul.Tensor(sum_39, squeeze_277);  sum_39 = squeeze_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_19 = torch.ops.aten.convolution_backward.default(mul_946, relu_91, primals_314, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_946 = primals_314 = None
    getitem_281: "f32[8, 2240, 14, 14]" = convolution_backward_19[0]
    getitem_282: "f32[800, 2240, 1, 1]" = convolution_backward_19[1];  convolution_backward_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_19: "b8[8, 2240, 14, 14]" = torch.ops.aten.le.Scalar(relu_91, 0);  relu_91 = None
    where_19: "f32[8, 2240, 14, 14]" = torch.ops.aten.where.self(le_19, full_default, getitem_281);  le_19 = getitem_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_40: "f32[2240]" = torch.ops.aten.sum.dim_IntList(where_19, [0, 2, 3])
    sub_187: "f32[8, 2240, 14, 14]" = torch.ops.aten.sub.Tensor(cat_57, unsqueeze_674);  cat_57 = unsqueeze_674 = None
    mul_948: "f32[8, 2240, 14, 14]" = torch.ops.aten.mul.Tensor(where_19, sub_187)
    sum_41: "f32[2240]" = torch.ops.aten.sum.dim_IntList(mul_948, [0, 2, 3]);  mul_948 = None
    mul_949: "f32[2240]" = torch.ops.aten.mul.Tensor(sum_40, 0.0006377551020408163)
    unsqueeze_675: "f32[1, 2240]" = torch.ops.aten.unsqueeze.default(mul_949, 0);  mul_949 = None
    unsqueeze_676: "f32[1, 2240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_675, 2);  unsqueeze_675 = None
    unsqueeze_677: "f32[1, 2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_676, 3);  unsqueeze_676 = None
    mul_950: "f32[2240]" = torch.ops.aten.mul.Tensor(sum_41, 0.0006377551020408163)
    mul_951: "f32[2240]" = torch.ops.aten.mul.Tensor(squeeze_274, squeeze_274)
    mul_952: "f32[2240]" = torch.ops.aten.mul.Tensor(mul_950, mul_951);  mul_950 = mul_951 = None
    unsqueeze_678: "f32[1, 2240]" = torch.ops.aten.unsqueeze.default(mul_952, 0);  mul_952 = None
    unsqueeze_679: "f32[1, 2240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_678, 2);  unsqueeze_678 = None
    unsqueeze_680: "f32[1, 2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_679, 3);  unsqueeze_679 = None
    mul_953: "f32[2240]" = torch.ops.aten.mul.Tensor(squeeze_274, primals_183);  primals_183 = None
    unsqueeze_681: "f32[1, 2240]" = torch.ops.aten.unsqueeze.default(mul_953, 0);  mul_953 = None
    unsqueeze_682: "f32[1, 2240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_681, 2);  unsqueeze_681 = None
    unsqueeze_683: "f32[1, 2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_682, 3);  unsqueeze_682 = None
    mul_954: "f32[8, 2240, 14, 14]" = torch.ops.aten.mul.Tensor(sub_187, unsqueeze_680);  sub_187 = unsqueeze_680 = None
    sub_189: "f32[8, 2240, 14, 14]" = torch.ops.aten.sub.Tensor(where_19, mul_954);  where_19 = mul_954 = None
    sub_190: "f32[8, 2240, 14, 14]" = torch.ops.aten.sub.Tensor(sub_189, unsqueeze_677);  sub_189 = unsqueeze_677 = None
    mul_955: "f32[8, 2240, 14, 14]" = torch.ops.aten.mul.Tensor(sub_190, unsqueeze_683);  sub_190 = unsqueeze_683 = None
    mul_956: "f32[2240]" = torch.ops.aten.mul.Tensor(sum_41, squeeze_274);  sum_41 = squeeze_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    slice_337: "f32[8, 1024, 14, 14]" = torch.ops.aten.slice.Tensor(mul_955, 1, 0, 1024)
    slice_338: "f32[8, 1216, 14, 14]" = torch.ops.aten.slice.Tensor(mul_955, 1, 1024, 2240);  mul_955 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    add_606: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_603, slice_337);  add_603 = slice_337 = None
    add_607: "f32[8, 1216, 14, 14]" = torch.ops.aten.add.Tensor(slice_335, slice_338);  slice_335 = slice_338 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    slice_339: "f32[8, 1152, 14, 14]" = torch.ops.aten.slice.Tensor(add_607, 1, 0, 1152)
    slice_340: "f32[8, 64, 14, 14]" = torch.ops.aten.slice.Tensor(add_607, 1, 1152, 1216);  add_607 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    slice_scatter_56: "f32[8, 64, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_43, slice_340, 3, 0, 9223372036854775807);  slice_340 = None
    slice_scatter_58: "f32[8, 1088, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_45, slice_scatter_56, 1, 1024, 9223372036854775807);  slice_scatter_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    slice_scatter_62: "f32[8, 1088, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_45, add_606, 1, 0, 1024)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    add_608: "f32[8, 1088, 14, 14]" = torch.ops.aten.add.Tensor(slice_scatter_58, slice_scatter_62);  slice_scatter_58 = slice_scatter_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_20 = torch.ops.aten.convolution_backward.default(add_608, relu_90, primals_313, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_608 = primals_313 = None
    getitem_284: "f32[8, 800, 14, 14]" = convolution_backward_20[0]
    getitem_285: "f32[1088, 800, 1, 1]" = convolution_backward_20[1];  convolution_backward_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_20: "b8[8, 800, 14, 14]" = torch.ops.aten.le.Scalar(relu_90, 0);  relu_90 = None
    where_20: "f32[8, 800, 14, 14]" = torch.ops.aten.where.self(le_20, full_default, getitem_284);  le_20 = getitem_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_42: "f32[800]" = torch.ops.aten.sum.dim_IntList(where_20, [0, 2, 3])
    sub_191: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_89, unsqueeze_686);  convolution_89 = unsqueeze_686 = None
    mul_957: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(where_20, sub_191)
    sum_43: "f32[800]" = torch.ops.aten.sum.dim_IntList(mul_957, [0, 2, 3]);  mul_957 = None
    mul_958: "f32[800]" = torch.ops.aten.mul.Tensor(sum_42, 0.0006377551020408163)
    unsqueeze_687: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_958, 0);  mul_958 = None
    unsqueeze_688: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_687, 2);  unsqueeze_687 = None
    unsqueeze_689: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_688, 3);  unsqueeze_688 = None
    mul_959: "f32[800]" = torch.ops.aten.mul.Tensor(sum_43, 0.0006377551020408163)
    mul_960: "f32[800]" = torch.ops.aten.mul.Tensor(squeeze_271, squeeze_271)
    mul_961: "f32[800]" = torch.ops.aten.mul.Tensor(mul_959, mul_960);  mul_959 = mul_960 = None
    unsqueeze_690: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_961, 0);  mul_961 = None
    unsqueeze_691: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_690, 2);  unsqueeze_690 = None
    unsqueeze_692: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_691, 3);  unsqueeze_691 = None
    mul_962: "f32[800]" = torch.ops.aten.mul.Tensor(squeeze_271, primals_181);  primals_181 = None
    unsqueeze_693: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_962, 0);  mul_962 = None
    unsqueeze_694: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_693, 2);  unsqueeze_693 = None
    unsqueeze_695: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_694, 3);  unsqueeze_694 = None
    mul_963: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_191, unsqueeze_692);  sub_191 = unsqueeze_692 = None
    sub_193: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(where_20, mul_963);  where_20 = mul_963 = None
    sub_194: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(sub_193, unsqueeze_689);  sub_193 = unsqueeze_689 = None
    mul_964: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_194, unsqueeze_695);  sub_194 = unsqueeze_695 = None
    mul_965: "f32[800]" = torch.ops.aten.mul.Tensor(sum_43, squeeze_271);  sum_43 = squeeze_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_21 = torch.ops.aten.convolution_backward.default(mul_964, relu_89, primals_312, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False]);  mul_964 = primals_312 = None
    getitem_287: "f32[8, 800, 14, 14]" = convolution_backward_21[0]
    getitem_288: "f32[800, 16, 3, 3]" = convolution_backward_21[1];  convolution_backward_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_21: "b8[8, 800, 14, 14]" = torch.ops.aten.le.Scalar(relu_89, 0);  relu_89 = None
    where_21: "f32[8, 800, 14, 14]" = torch.ops.aten.where.self(le_21, full_default, getitem_287);  le_21 = getitem_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_44: "f32[800]" = torch.ops.aten.sum.dim_IntList(where_21, [0, 2, 3])
    sub_195: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_88, unsqueeze_698);  convolution_88 = unsqueeze_698 = None
    mul_966: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(where_21, sub_195)
    sum_45: "f32[800]" = torch.ops.aten.sum.dim_IntList(mul_966, [0, 2, 3]);  mul_966 = None
    mul_967: "f32[800]" = torch.ops.aten.mul.Tensor(sum_44, 0.0006377551020408163)
    unsqueeze_699: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_967, 0);  mul_967 = None
    unsqueeze_700: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_699, 2);  unsqueeze_699 = None
    unsqueeze_701: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_700, 3);  unsqueeze_700 = None
    mul_968: "f32[800]" = torch.ops.aten.mul.Tensor(sum_45, 0.0006377551020408163)
    mul_969: "f32[800]" = torch.ops.aten.mul.Tensor(squeeze_268, squeeze_268)
    mul_970: "f32[800]" = torch.ops.aten.mul.Tensor(mul_968, mul_969);  mul_968 = mul_969 = None
    unsqueeze_702: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_970, 0);  mul_970 = None
    unsqueeze_703: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_702, 2);  unsqueeze_702 = None
    unsqueeze_704: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_703, 3);  unsqueeze_703 = None
    mul_971: "f32[800]" = torch.ops.aten.mul.Tensor(squeeze_268, primals_179);  primals_179 = None
    unsqueeze_705: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_971, 0);  mul_971 = None
    unsqueeze_706: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_705, 2);  unsqueeze_705 = None
    unsqueeze_707: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_706, 3);  unsqueeze_706 = None
    mul_972: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_195, unsqueeze_704);  sub_195 = unsqueeze_704 = None
    sub_197: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(where_21, mul_972);  where_21 = mul_972 = None
    sub_198: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(sub_197, unsqueeze_701);  sub_197 = unsqueeze_701 = None
    mul_973: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_198, unsqueeze_707);  sub_198 = unsqueeze_707 = None
    mul_974: "f32[800]" = torch.ops.aten.mul.Tensor(sum_45, squeeze_268);  sum_45 = squeeze_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_22 = torch.ops.aten.convolution_backward.default(mul_973, relu_88, primals_311, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_973 = primals_311 = None
    getitem_290: "f32[8, 2176, 14, 14]" = convolution_backward_22[0]
    getitem_291: "f32[800, 2176, 1, 1]" = convolution_backward_22[1];  convolution_backward_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_22: "b8[8, 2176, 14, 14]" = torch.ops.aten.le.Scalar(relu_88, 0);  relu_88 = None
    where_22: "f32[8, 2176, 14, 14]" = torch.ops.aten.where.self(le_22, full_default, getitem_290);  le_22 = getitem_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_46: "f32[2176]" = torch.ops.aten.sum.dim_IntList(where_22, [0, 2, 3])
    sub_199: "f32[8, 2176, 14, 14]" = torch.ops.aten.sub.Tensor(cat_55, unsqueeze_710);  cat_55 = unsqueeze_710 = None
    mul_975: "f32[8, 2176, 14, 14]" = torch.ops.aten.mul.Tensor(where_22, sub_199)
    sum_47: "f32[2176]" = torch.ops.aten.sum.dim_IntList(mul_975, [0, 2, 3]);  mul_975 = None
    mul_976: "f32[2176]" = torch.ops.aten.mul.Tensor(sum_46, 0.0006377551020408163)
    unsqueeze_711: "f32[1, 2176]" = torch.ops.aten.unsqueeze.default(mul_976, 0);  mul_976 = None
    unsqueeze_712: "f32[1, 2176, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_711, 2);  unsqueeze_711 = None
    unsqueeze_713: "f32[1, 2176, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_712, 3);  unsqueeze_712 = None
    mul_977: "f32[2176]" = torch.ops.aten.mul.Tensor(sum_47, 0.0006377551020408163)
    mul_978: "f32[2176]" = torch.ops.aten.mul.Tensor(squeeze_265, squeeze_265)
    mul_979: "f32[2176]" = torch.ops.aten.mul.Tensor(mul_977, mul_978);  mul_977 = mul_978 = None
    unsqueeze_714: "f32[1, 2176]" = torch.ops.aten.unsqueeze.default(mul_979, 0);  mul_979 = None
    unsqueeze_715: "f32[1, 2176, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_714, 2);  unsqueeze_714 = None
    unsqueeze_716: "f32[1, 2176, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_715, 3);  unsqueeze_715 = None
    mul_980: "f32[2176]" = torch.ops.aten.mul.Tensor(squeeze_265, primals_177);  primals_177 = None
    unsqueeze_717: "f32[1, 2176]" = torch.ops.aten.unsqueeze.default(mul_980, 0);  mul_980 = None
    unsqueeze_718: "f32[1, 2176, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_717, 2);  unsqueeze_717 = None
    unsqueeze_719: "f32[1, 2176, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_718, 3);  unsqueeze_718 = None
    mul_981: "f32[8, 2176, 14, 14]" = torch.ops.aten.mul.Tensor(sub_199, unsqueeze_716);  sub_199 = unsqueeze_716 = None
    sub_201: "f32[8, 2176, 14, 14]" = torch.ops.aten.sub.Tensor(where_22, mul_981);  where_22 = mul_981 = None
    sub_202: "f32[8, 2176, 14, 14]" = torch.ops.aten.sub.Tensor(sub_201, unsqueeze_713);  sub_201 = unsqueeze_713 = None
    mul_982: "f32[8, 2176, 14, 14]" = torch.ops.aten.mul.Tensor(sub_202, unsqueeze_719);  sub_202 = unsqueeze_719 = None
    mul_983: "f32[2176]" = torch.ops.aten.mul.Tensor(sum_47, squeeze_265);  sum_47 = squeeze_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    slice_341: "f32[8, 1024, 14, 14]" = torch.ops.aten.slice.Tensor(mul_982, 1, 0, 1024)
    slice_342: "f32[8, 1152, 14, 14]" = torch.ops.aten.slice.Tensor(mul_982, 1, 1024, 2176);  mul_982 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    add_609: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_606, slice_341);  add_606 = slice_341 = None
    add_610: "f32[8, 1152, 14, 14]" = torch.ops.aten.add.Tensor(slice_339, slice_342);  slice_339 = slice_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    slice_343: "f32[8, 1088, 14, 14]" = torch.ops.aten.slice.Tensor(add_610, 1, 0, 1088)
    slice_344: "f32[8, 64, 14, 14]" = torch.ops.aten.slice.Tensor(add_610, 1, 1088, 1152);  add_610 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    slice_scatter_64: "f32[8, 64, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_43, slice_344, 3, 0, 9223372036854775807);  slice_344 = None
    slice_scatter_66: "f32[8, 1088, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_45, slice_scatter_64, 1, 1024, 9223372036854775807);  slice_scatter_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    slice_scatter_70: "f32[8, 1088, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_45, add_609, 1, 0, 1024)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    add_611: "f32[8, 1088, 14, 14]" = torch.ops.aten.add.Tensor(slice_scatter_66, slice_scatter_70);  slice_scatter_66 = slice_scatter_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_23 = torch.ops.aten.convolution_backward.default(add_611, relu_87, primals_310, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_611 = primals_310 = None
    getitem_293: "f32[8, 800, 14, 14]" = convolution_backward_23[0]
    getitem_294: "f32[1088, 800, 1, 1]" = convolution_backward_23[1];  convolution_backward_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_23: "b8[8, 800, 14, 14]" = torch.ops.aten.le.Scalar(relu_87, 0);  relu_87 = None
    where_23: "f32[8, 800, 14, 14]" = torch.ops.aten.where.self(le_23, full_default, getitem_293);  le_23 = getitem_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_48: "f32[800]" = torch.ops.aten.sum.dim_IntList(where_23, [0, 2, 3])
    sub_203: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_86, unsqueeze_722);  convolution_86 = unsqueeze_722 = None
    mul_984: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(where_23, sub_203)
    sum_49: "f32[800]" = torch.ops.aten.sum.dim_IntList(mul_984, [0, 2, 3]);  mul_984 = None
    mul_985: "f32[800]" = torch.ops.aten.mul.Tensor(sum_48, 0.0006377551020408163)
    unsqueeze_723: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_985, 0);  mul_985 = None
    unsqueeze_724: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_723, 2);  unsqueeze_723 = None
    unsqueeze_725: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_724, 3);  unsqueeze_724 = None
    mul_986: "f32[800]" = torch.ops.aten.mul.Tensor(sum_49, 0.0006377551020408163)
    mul_987: "f32[800]" = torch.ops.aten.mul.Tensor(squeeze_262, squeeze_262)
    mul_988: "f32[800]" = torch.ops.aten.mul.Tensor(mul_986, mul_987);  mul_986 = mul_987 = None
    unsqueeze_726: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_988, 0);  mul_988 = None
    unsqueeze_727: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_726, 2);  unsqueeze_726 = None
    unsqueeze_728: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_727, 3);  unsqueeze_727 = None
    mul_989: "f32[800]" = torch.ops.aten.mul.Tensor(squeeze_262, primals_175);  primals_175 = None
    unsqueeze_729: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_989, 0);  mul_989 = None
    unsqueeze_730: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_729, 2);  unsqueeze_729 = None
    unsqueeze_731: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_730, 3);  unsqueeze_730 = None
    mul_990: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_203, unsqueeze_728);  sub_203 = unsqueeze_728 = None
    sub_205: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(where_23, mul_990);  where_23 = mul_990 = None
    sub_206: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(sub_205, unsqueeze_725);  sub_205 = unsqueeze_725 = None
    mul_991: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_206, unsqueeze_731);  sub_206 = unsqueeze_731 = None
    mul_992: "f32[800]" = torch.ops.aten.mul.Tensor(sum_49, squeeze_262);  sum_49 = squeeze_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_24 = torch.ops.aten.convolution_backward.default(mul_991, relu_86, primals_309, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False]);  mul_991 = primals_309 = None
    getitem_296: "f32[8, 800, 14, 14]" = convolution_backward_24[0]
    getitem_297: "f32[800, 16, 3, 3]" = convolution_backward_24[1];  convolution_backward_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_24: "b8[8, 800, 14, 14]" = torch.ops.aten.le.Scalar(relu_86, 0);  relu_86 = None
    where_24: "f32[8, 800, 14, 14]" = torch.ops.aten.where.self(le_24, full_default, getitem_296);  le_24 = getitem_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_50: "f32[800]" = torch.ops.aten.sum.dim_IntList(where_24, [0, 2, 3])
    sub_207: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_85, unsqueeze_734);  convolution_85 = unsqueeze_734 = None
    mul_993: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(where_24, sub_207)
    sum_51: "f32[800]" = torch.ops.aten.sum.dim_IntList(mul_993, [0, 2, 3]);  mul_993 = None
    mul_994: "f32[800]" = torch.ops.aten.mul.Tensor(sum_50, 0.0006377551020408163)
    unsqueeze_735: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_994, 0);  mul_994 = None
    unsqueeze_736: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_735, 2);  unsqueeze_735 = None
    unsqueeze_737: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_736, 3);  unsqueeze_736 = None
    mul_995: "f32[800]" = torch.ops.aten.mul.Tensor(sum_51, 0.0006377551020408163)
    mul_996: "f32[800]" = torch.ops.aten.mul.Tensor(squeeze_259, squeeze_259)
    mul_997: "f32[800]" = torch.ops.aten.mul.Tensor(mul_995, mul_996);  mul_995 = mul_996 = None
    unsqueeze_738: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_997, 0);  mul_997 = None
    unsqueeze_739: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_738, 2);  unsqueeze_738 = None
    unsqueeze_740: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_739, 3);  unsqueeze_739 = None
    mul_998: "f32[800]" = torch.ops.aten.mul.Tensor(squeeze_259, primals_173);  primals_173 = None
    unsqueeze_741: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_998, 0);  mul_998 = None
    unsqueeze_742: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_741, 2);  unsqueeze_741 = None
    unsqueeze_743: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_742, 3);  unsqueeze_742 = None
    mul_999: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_207, unsqueeze_740);  sub_207 = unsqueeze_740 = None
    sub_209: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(where_24, mul_999);  where_24 = mul_999 = None
    sub_210: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(sub_209, unsqueeze_737);  sub_209 = unsqueeze_737 = None
    mul_1000: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_210, unsqueeze_743);  sub_210 = unsqueeze_743 = None
    mul_1001: "f32[800]" = torch.ops.aten.mul.Tensor(sum_51, squeeze_259);  sum_51 = squeeze_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_25 = torch.ops.aten.convolution_backward.default(mul_1000, relu_85, primals_308, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1000 = primals_308 = None
    getitem_299: "f32[8, 2112, 14, 14]" = convolution_backward_25[0]
    getitem_300: "f32[800, 2112, 1, 1]" = convolution_backward_25[1];  convolution_backward_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_25: "b8[8, 2112, 14, 14]" = torch.ops.aten.le.Scalar(relu_85, 0);  relu_85 = None
    where_25: "f32[8, 2112, 14, 14]" = torch.ops.aten.where.self(le_25, full_default, getitem_299);  le_25 = getitem_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_52: "f32[2112]" = torch.ops.aten.sum.dim_IntList(where_25, [0, 2, 3])
    sub_211: "f32[8, 2112, 14, 14]" = torch.ops.aten.sub.Tensor(cat_53, unsqueeze_746);  cat_53 = unsqueeze_746 = None
    mul_1002: "f32[8, 2112, 14, 14]" = torch.ops.aten.mul.Tensor(where_25, sub_211)
    sum_53: "f32[2112]" = torch.ops.aten.sum.dim_IntList(mul_1002, [0, 2, 3]);  mul_1002 = None
    mul_1003: "f32[2112]" = torch.ops.aten.mul.Tensor(sum_52, 0.0006377551020408163)
    unsqueeze_747: "f32[1, 2112]" = torch.ops.aten.unsqueeze.default(mul_1003, 0);  mul_1003 = None
    unsqueeze_748: "f32[1, 2112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_747, 2);  unsqueeze_747 = None
    unsqueeze_749: "f32[1, 2112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_748, 3);  unsqueeze_748 = None
    mul_1004: "f32[2112]" = torch.ops.aten.mul.Tensor(sum_53, 0.0006377551020408163)
    mul_1005: "f32[2112]" = torch.ops.aten.mul.Tensor(squeeze_256, squeeze_256)
    mul_1006: "f32[2112]" = torch.ops.aten.mul.Tensor(mul_1004, mul_1005);  mul_1004 = mul_1005 = None
    unsqueeze_750: "f32[1, 2112]" = torch.ops.aten.unsqueeze.default(mul_1006, 0);  mul_1006 = None
    unsqueeze_751: "f32[1, 2112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_750, 2);  unsqueeze_750 = None
    unsqueeze_752: "f32[1, 2112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_751, 3);  unsqueeze_751 = None
    mul_1007: "f32[2112]" = torch.ops.aten.mul.Tensor(squeeze_256, primals_171);  primals_171 = None
    unsqueeze_753: "f32[1, 2112]" = torch.ops.aten.unsqueeze.default(mul_1007, 0);  mul_1007 = None
    unsqueeze_754: "f32[1, 2112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_753, 2);  unsqueeze_753 = None
    unsqueeze_755: "f32[1, 2112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_754, 3);  unsqueeze_754 = None
    mul_1008: "f32[8, 2112, 14, 14]" = torch.ops.aten.mul.Tensor(sub_211, unsqueeze_752);  sub_211 = unsqueeze_752 = None
    sub_213: "f32[8, 2112, 14, 14]" = torch.ops.aten.sub.Tensor(where_25, mul_1008);  where_25 = mul_1008 = None
    sub_214: "f32[8, 2112, 14, 14]" = torch.ops.aten.sub.Tensor(sub_213, unsqueeze_749);  sub_213 = unsqueeze_749 = None
    mul_1009: "f32[8, 2112, 14, 14]" = torch.ops.aten.mul.Tensor(sub_214, unsqueeze_755);  sub_214 = unsqueeze_755 = None
    mul_1010: "f32[2112]" = torch.ops.aten.mul.Tensor(sum_53, squeeze_256);  sum_53 = squeeze_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    slice_345: "f32[8, 1024, 14, 14]" = torch.ops.aten.slice.Tensor(mul_1009, 1, 0, 1024)
    slice_346: "f32[8, 1088, 14, 14]" = torch.ops.aten.slice.Tensor(mul_1009, 1, 1024, 2112);  mul_1009 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    add_612: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_609, slice_345);  add_609 = slice_345 = None
    add_613: "f32[8, 1088, 14, 14]" = torch.ops.aten.add.Tensor(slice_343, slice_346);  slice_343 = slice_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    slice_347: "f32[8, 1024, 14, 14]" = torch.ops.aten.slice.Tensor(add_613, 1, 0, 1024)
    slice_348: "f32[8, 64, 14, 14]" = torch.ops.aten.slice.Tensor(add_613, 1, 1024, 1088);  add_613 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    slice_scatter_72: "f32[8, 64, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_43, slice_348, 3, 0, 9223372036854775807);  slice_348 = None
    slice_scatter_74: "f32[8, 1088, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_45, slice_scatter_72, 1, 1024, 9223372036854775807);  slice_scatter_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    slice_scatter_78: "f32[8, 1088, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_45, add_612, 1, 0, 1024)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    add_614: "f32[8, 1088, 14, 14]" = torch.ops.aten.add.Tensor(slice_scatter_74, slice_scatter_78);  slice_scatter_74 = slice_scatter_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_26 = torch.ops.aten.convolution_backward.default(add_614, relu_84, primals_307, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_614 = primals_307 = None
    getitem_302: "f32[8, 800, 14, 14]" = convolution_backward_26[0]
    getitem_303: "f32[1088, 800, 1, 1]" = convolution_backward_26[1];  convolution_backward_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_26: "b8[8, 800, 14, 14]" = torch.ops.aten.le.Scalar(relu_84, 0);  relu_84 = None
    where_26: "f32[8, 800, 14, 14]" = torch.ops.aten.where.self(le_26, full_default, getitem_302);  le_26 = getitem_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_54: "f32[800]" = torch.ops.aten.sum.dim_IntList(where_26, [0, 2, 3])
    sub_215: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_83, unsqueeze_758);  convolution_83 = unsqueeze_758 = None
    mul_1011: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(where_26, sub_215)
    sum_55: "f32[800]" = torch.ops.aten.sum.dim_IntList(mul_1011, [0, 2, 3]);  mul_1011 = None
    mul_1012: "f32[800]" = torch.ops.aten.mul.Tensor(sum_54, 0.0006377551020408163)
    unsqueeze_759: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_1012, 0);  mul_1012 = None
    unsqueeze_760: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_759, 2);  unsqueeze_759 = None
    unsqueeze_761: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_760, 3);  unsqueeze_760 = None
    mul_1013: "f32[800]" = torch.ops.aten.mul.Tensor(sum_55, 0.0006377551020408163)
    mul_1014: "f32[800]" = torch.ops.aten.mul.Tensor(squeeze_253, squeeze_253)
    mul_1015: "f32[800]" = torch.ops.aten.mul.Tensor(mul_1013, mul_1014);  mul_1013 = mul_1014 = None
    unsqueeze_762: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_1015, 0);  mul_1015 = None
    unsqueeze_763: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_762, 2);  unsqueeze_762 = None
    unsqueeze_764: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_763, 3);  unsqueeze_763 = None
    mul_1016: "f32[800]" = torch.ops.aten.mul.Tensor(squeeze_253, primals_169);  primals_169 = None
    unsqueeze_765: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_1016, 0);  mul_1016 = None
    unsqueeze_766: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_765, 2);  unsqueeze_765 = None
    unsqueeze_767: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_766, 3);  unsqueeze_766 = None
    mul_1017: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_215, unsqueeze_764);  sub_215 = unsqueeze_764 = None
    sub_217: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(where_26, mul_1017);  where_26 = mul_1017 = None
    sub_218: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(sub_217, unsqueeze_761);  sub_217 = unsqueeze_761 = None
    mul_1018: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_218, unsqueeze_767);  sub_218 = unsqueeze_767 = None
    mul_1019: "f32[800]" = torch.ops.aten.mul.Tensor(sum_55, squeeze_253);  sum_55 = squeeze_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_27 = torch.ops.aten.convolution_backward.default(mul_1018, relu_83, primals_306, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False]);  mul_1018 = primals_306 = None
    getitem_305: "f32[8, 800, 14, 14]" = convolution_backward_27[0]
    getitem_306: "f32[800, 16, 3, 3]" = convolution_backward_27[1];  convolution_backward_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_27: "b8[8, 800, 14, 14]" = torch.ops.aten.le.Scalar(relu_83, 0);  relu_83 = None
    where_27: "f32[8, 800, 14, 14]" = torch.ops.aten.where.self(le_27, full_default, getitem_305);  le_27 = getitem_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_56: "f32[800]" = torch.ops.aten.sum.dim_IntList(where_27, [0, 2, 3])
    sub_219: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_82, unsqueeze_770);  convolution_82 = unsqueeze_770 = None
    mul_1020: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(where_27, sub_219)
    sum_57: "f32[800]" = torch.ops.aten.sum.dim_IntList(mul_1020, [0, 2, 3]);  mul_1020 = None
    mul_1021: "f32[800]" = torch.ops.aten.mul.Tensor(sum_56, 0.0006377551020408163)
    unsqueeze_771: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_1021, 0);  mul_1021 = None
    unsqueeze_772: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_771, 2);  unsqueeze_771 = None
    unsqueeze_773: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_772, 3);  unsqueeze_772 = None
    mul_1022: "f32[800]" = torch.ops.aten.mul.Tensor(sum_57, 0.0006377551020408163)
    mul_1023: "f32[800]" = torch.ops.aten.mul.Tensor(squeeze_250, squeeze_250)
    mul_1024: "f32[800]" = torch.ops.aten.mul.Tensor(mul_1022, mul_1023);  mul_1022 = mul_1023 = None
    unsqueeze_774: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_1024, 0);  mul_1024 = None
    unsqueeze_775: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_774, 2);  unsqueeze_774 = None
    unsqueeze_776: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_775, 3);  unsqueeze_775 = None
    mul_1025: "f32[800]" = torch.ops.aten.mul.Tensor(squeeze_250, primals_167);  primals_167 = None
    unsqueeze_777: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_1025, 0);  mul_1025 = None
    unsqueeze_778: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_777, 2);  unsqueeze_777 = None
    unsqueeze_779: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_778, 3);  unsqueeze_778 = None
    mul_1026: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_219, unsqueeze_776);  sub_219 = unsqueeze_776 = None
    sub_221: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(where_27, mul_1026);  where_27 = mul_1026 = None
    sub_222: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(sub_221, unsqueeze_773);  sub_221 = unsqueeze_773 = None
    mul_1027: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_222, unsqueeze_779);  sub_222 = unsqueeze_779 = None
    mul_1028: "f32[800]" = torch.ops.aten.mul.Tensor(sum_57, squeeze_250);  sum_57 = squeeze_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_28 = torch.ops.aten.convolution_backward.default(mul_1027, relu_82, primals_305, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1027 = primals_305 = None
    getitem_308: "f32[8, 2048, 14, 14]" = convolution_backward_28[0]
    getitem_309: "f32[800, 2048, 1, 1]" = convolution_backward_28[1];  convolution_backward_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_28: "b8[8, 2048, 14, 14]" = torch.ops.aten.le.Scalar(relu_82, 0);  relu_82 = None
    where_28: "f32[8, 2048, 14, 14]" = torch.ops.aten.where.self(le_28, full_default, getitem_308);  le_28 = getitem_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_58: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_28, [0, 2, 3])
    sub_223: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(cat_51, unsqueeze_782);  cat_51 = unsqueeze_782 = None
    mul_1029: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(where_28, sub_223)
    sum_59: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_1029, [0, 2, 3]);  mul_1029 = None
    mul_1030: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_58, 0.0006377551020408163)
    unsqueeze_783: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1030, 0);  mul_1030 = None
    unsqueeze_784: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_783, 2);  unsqueeze_783 = None
    unsqueeze_785: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_784, 3);  unsqueeze_784 = None
    mul_1031: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_59, 0.0006377551020408163)
    mul_1032: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_247, squeeze_247)
    mul_1033: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_1031, mul_1032);  mul_1031 = mul_1032 = None
    unsqueeze_786: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1033, 0);  mul_1033 = None
    unsqueeze_787: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_786, 2);  unsqueeze_786 = None
    unsqueeze_788: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_787, 3);  unsqueeze_787 = None
    mul_1034: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_247, primals_165);  primals_165 = None
    unsqueeze_789: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1034, 0);  mul_1034 = None
    unsqueeze_790: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_789, 2);  unsqueeze_789 = None
    unsqueeze_791: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_790, 3);  unsqueeze_790 = None
    mul_1035: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_223, unsqueeze_788);  sub_223 = unsqueeze_788 = None
    sub_225: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(where_28, mul_1035);  where_28 = mul_1035 = None
    sub_226: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(sub_225, unsqueeze_785);  sub_225 = unsqueeze_785 = None
    mul_1036: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_226, unsqueeze_791);  sub_226 = unsqueeze_791 = None
    mul_1037: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_59, squeeze_247);  sum_59 = squeeze_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    slice_349: "f32[8, 1024, 14, 14]" = torch.ops.aten.slice.Tensor(mul_1036, 1, 0, 1024)
    slice_350: "f32[8, 1024, 14, 14]" = torch.ops.aten.slice.Tensor(mul_1036, 1, 1024, 2048);  mul_1036 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    add_615: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_612, slice_349);  add_612 = slice_349 = None
    add_616: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(slice_347, slice_350);  slice_347 = slice_350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    slice_351: "f32[8, 960, 14, 14]" = torch.ops.aten.slice.Tensor(add_616, 1, 0, 960)
    slice_352: "f32[8, 64, 14, 14]" = torch.ops.aten.slice.Tensor(add_616, 1, 960, 1024);  add_616 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    slice_scatter_80: "f32[8, 64, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_43, slice_352, 3, 0, 9223372036854775807);  slice_352 = None
    slice_scatter_82: "f32[8, 1088, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_45, slice_scatter_80, 1, 1024, 9223372036854775807);  slice_scatter_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    slice_scatter_86: "f32[8, 1088, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_45, add_615, 1, 0, 1024)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    add_617: "f32[8, 1088, 14, 14]" = torch.ops.aten.add.Tensor(slice_scatter_82, slice_scatter_86);  slice_scatter_82 = slice_scatter_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_29 = torch.ops.aten.convolution_backward.default(add_617, relu_81, primals_304, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_617 = primals_304 = None
    getitem_311: "f32[8, 800, 14, 14]" = convolution_backward_29[0]
    getitem_312: "f32[1088, 800, 1, 1]" = convolution_backward_29[1];  convolution_backward_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_29: "b8[8, 800, 14, 14]" = torch.ops.aten.le.Scalar(relu_81, 0);  relu_81 = None
    where_29: "f32[8, 800, 14, 14]" = torch.ops.aten.where.self(le_29, full_default, getitem_311);  le_29 = getitem_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_60: "f32[800]" = torch.ops.aten.sum.dim_IntList(where_29, [0, 2, 3])
    sub_227: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_80, unsqueeze_794);  convolution_80 = unsqueeze_794 = None
    mul_1038: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(where_29, sub_227)
    sum_61: "f32[800]" = torch.ops.aten.sum.dim_IntList(mul_1038, [0, 2, 3]);  mul_1038 = None
    mul_1039: "f32[800]" = torch.ops.aten.mul.Tensor(sum_60, 0.0006377551020408163)
    unsqueeze_795: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_1039, 0);  mul_1039 = None
    unsqueeze_796: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_795, 2);  unsqueeze_795 = None
    unsqueeze_797: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_796, 3);  unsqueeze_796 = None
    mul_1040: "f32[800]" = torch.ops.aten.mul.Tensor(sum_61, 0.0006377551020408163)
    mul_1041: "f32[800]" = torch.ops.aten.mul.Tensor(squeeze_244, squeeze_244)
    mul_1042: "f32[800]" = torch.ops.aten.mul.Tensor(mul_1040, mul_1041);  mul_1040 = mul_1041 = None
    unsqueeze_798: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_1042, 0);  mul_1042 = None
    unsqueeze_799: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_798, 2);  unsqueeze_798 = None
    unsqueeze_800: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_799, 3);  unsqueeze_799 = None
    mul_1043: "f32[800]" = torch.ops.aten.mul.Tensor(squeeze_244, primals_163);  primals_163 = None
    unsqueeze_801: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_1043, 0);  mul_1043 = None
    unsqueeze_802: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_801, 2);  unsqueeze_801 = None
    unsqueeze_803: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_802, 3);  unsqueeze_802 = None
    mul_1044: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_227, unsqueeze_800);  sub_227 = unsqueeze_800 = None
    sub_229: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(where_29, mul_1044);  where_29 = mul_1044 = None
    sub_230: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(sub_229, unsqueeze_797);  sub_229 = unsqueeze_797 = None
    mul_1045: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_230, unsqueeze_803);  sub_230 = unsqueeze_803 = None
    mul_1046: "f32[800]" = torch.ops.aten.mul.Tensor(sum_61, squeeze_244);  sum_61 = squeeze_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_30 = torch.ops.aten.convolution_backward.default(mul_1045, relu_80, primals_303, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False]);  mul_1045 = primals_303 = None
    getitem_314: "f32[8, 800, 14, 14]" = convolution_backward_30[0]
    getitem_315: "f32[800, 16, 3, 3]" = convolution_backward_30[1];  convolution_backward_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_30: "b8[8, 800, 14, 14]" = torch.ops.aten.le.Scalar(relu_80, 0);  relu_80 = None
    where_30: "f32[8, 800, 14, 14]" = torch.ops.aten.where.self(le_30, full_default, getitem_314);  le_30 = getitem_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_62: "f32[800]" = torch.ops.aten.sum.dim_IntList(where_30, [0, 2, 3])
    sub_231: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_79, unsqueeze_806);  convolution_79 = unsqueeze_806 = None
    mul_1047: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(where_30, sub_231)
    sum_63: "f32[800]" = torch.ops.aten.sum.dim_IntList(mul_1047, [0, 2, 3]);  mul_1047 = None
    mul_1048: "f32[800]" = torch.ops.aten.mul.Tensor(sum_62, 0.0006377551020408163)
    unsqueeze_807: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_1048, 0);  mul_1048 = None
    unsqueeze_808: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_807, 2);  unsqueeze_807 = None
    unsqueeze_809: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_808, 3);  unsqueeze_808 = None
    mul_1049: "f32[800]" = torch.ops.aten.mul.Tensor(sum_63, 0.0006377551020408163)
    mul_1050: "f32[800]" = torch.ops.aten.mul.Tensor(squeeze_241, squeeze_241)
    mul_1051: "f32[800]" = torch.ops.aten.mul.Tensor(mul_1049, mul_1050);  mul_1049 = mul_1050 = None
    unsqueeze_810: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_1051, 0);  mul_1051 = None
    unsqueeze_811: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_810, 2);  unsqueeze_810 = None
    unsqueeze_812: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_811, 3);  unsqueeze_811 = None
    mul_1052: "f32[800]" = torch.ops.aten.mul.Tensor(squeeze_241, primals_161);  primals_161 = None
    unsqueeze_813: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_1052, 0);  mul_1052 = None
    unsqueeze_814: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_813, 2);  unsqueeze_813 = None
    unsqueeze_815: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_814, 3);  unsqueeze_814 = None
    mul_1053: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_231, unsqueeze_812);  sub_231 = unsqueeze_812 = None
    sub_233: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(where_30, mul_1053);  where_30 = mul_1053 = None
    sub_234: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(sub_233, unsqueeze_809);  sub_233 = unsqueeze_809 = None
    mul_1054: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_234, unsqueeze_815);  sub_234 = unsqueeze_815 = None
    mul_1055: "f32[800]" = torch.ops.aten.mul.Tensor(sum_63, squeeze_241);  sum_63 = squeeze_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_31 = torch.ops.aten.convolution_backward.default(mul_1054, relu_79, primals_302, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1054 = primals_302 = None
    getitem_317: "f32[8, 1984, 14, 14]" = convolution_backward_31[0]
    getitem_318: "f32[800, 1984, 1, 1]" = convolution_backward_31[1];  convolution_backward_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_31: "b8[8, 1984, 14, 14]" = torch.ops.aten.le.Scalar(relu_79, 0);  relu_79 = None
    where_31: "f32[8, 1984, 14, 14]" = torch.ops.aten.where.self(le_31, full_default, getitem_317);  le_31 = getitem_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_64: "f32[1984]" = torch.ops.aten.sum.dim_IntList(where_31, [0, 2, 3])
    sub_235: "f32[8, 1984, 14, 14]" = torch.ops.aten.sub.Tensor(cat_49, unsqueeze_818);  cat_49 = unsqueeze_818 = None
    mul_1056: "f32[8, 1984, 14, 14]" = torch.ops.aten.mul.Tensor(where_31, sub_235)
    sum_65: "f32[1984]" = torch.ops.aten.sum.dim_IntList(mul_1056, [0, 2, 3]);  mul_1056 = None
    mul_1057: "f32[1984]" = torch.ops.aten.mul.Tensor(sum_64, 0.0006377551020408163)
    unsqueeze_819: "f32[1, 1984]" = torch.ops.aten.unsqueeze.default(mul_1057, 0);  mul_1057 = None
    unsqueeze_820: "f32[1, 1984, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_819, 2);  unsqueeze_819 = None
    unsqueeze_821: "f32[1, 1984, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_820, 3);  unsqueeze_820 = None
    mul_1058: "f32[1984]" = torch.ops.aten.mul.Tensor(sum_65, 0.0006377551020408163)
    mul_1059: "f32[1984]" = torch.ops.aten.mul.Tensor(squeeze_238, squeeze_238)
    mul_1060: "f32[1984]" = torch.ops.aten.mul.Tensor(mul_1058, mul_1059);  mul_1058 = mul_1059 = None
    unsqueeze_822: "f32[1, 1984]" = torch.ops.aten.unsqueeze.default(mul_1060, 0);  mul_1060 = None
    unsqueeze_823: "f32[1, 1984, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_822, 2);  unsqueeze_822 = None
    unsqueeze_824: "f32[1, 1984, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_823, 3);  unsqueeze_823 = None
    mul_1061: "f32[1984]" = torch.ops.aten.mul.Tensor(squeeze_238, primals_159);  primals_159 = None
    unsqueeze_825: "f32[1, 1984]" = torch.ops.aten.unsqueeze.default(mul_1061, 0);  mul_1061 = None
    unsqueeze_826: "f32[1, 1984, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_825, 2);  unsqueeze_825 = None
    unsqueeze_827: "f32[1, 1984, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_826, 3);  unsqueeze_826 = None
    mul_1062: "f32[8, 1984, 14, 14]" = torch.ops.aten.mul.Tensor(sub_235, unsqueeze_824);  sub_235 = unsqueeze_824 = None
    sub_237: "f32[8, 1984, 14, 14]" = torch.ops.aten.sub.Tensor(where_31, mul_1062);  where_31 = mul_1062 = None
    sub_238: "f32[8, 1984, 14, 14]" = torch.ops.aten.sub.Tensor(sub_237, unsqueeze_821);  sub_237 = unsqueeze_821 = None
    mul_1063: "f32[8, 1984, 14, 14]" = torch.ops.aten.mul.Tensor(sub_238, unsqueeze_827);  sub_238 = unsqueeze_827 = None
    mul_1064: "f32[1984]" = torch.ops.aten.mul.Tensor(sum_65, squeeze_238);  sum_65 = squeeze_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    slice_353: "f32[8, 1024, 14, 14]" = torch.ops.aten.slice.Tensor(mul_1063, 1, 0, 1024)
    slice_354: "f32[8, 960, 14, 14]" = torch.ops.aten.slice.Tensor(mul_1063, 1, 1024, 1984);  mul_1063 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    add_618: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_615, slice_353);  add_615 = slice_353 = None
    add_619: "f32[8, 960, 14, 14]" = torch.ops.aten.add.Tensor(slice_351, slice_354);  slice_351 = slice_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    slice_355: "f32[8, 896, 14, 14]" = torch.ops.aten.slice.Tensor(add_619, 1, 0, 896)
    slice_356: "f32[8, 64, 14, 14]" = torch.ops.aten.slice.Tensor(add_619, 1, 896, 960);  add_619 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    slice_scatter_88: "f32[8, 64, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_43, slice_356, 3, 0, 9223372036854775807);  slice_356 = None
    slice_scatter_90: "f32[8, 1088, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_45, slice_scatter_88, 1, 1024, 9223372036854775807);  slice_scatter_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    slice_scatter_94: "f32[8, 1088, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_45, add_618, 1, 0, 1024)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    add_620: "f32[8, 1088, 14, 14]" = torch.ops.aten.add.Tensor(slice_scatter_90, slice_scatter_94);  slice_scatter_90 = slice_scatter_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_32 = torch.ops.aten.convolution_backward.default(add_620, relu_78, primals_301, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_620 = primals_301 = None
    getitem_320: "f32[8, 800, 14, 14]" = convolution_backward_32[0]
    getitem_321: "f32[1088, 800, 1, 1]" = convolution_backward_32[1];  convolution_backward_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_32: "b8[8, 800, 14, 14]" = torch.ops.aten.le.Scalar(relu_78, 0);  relu_78 = None
    where_32: "f32[8, 800, 14, 14]" = torch.ops.aten.where.self(le_32, full_default, getitem_320);  le_32 = getitem_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_66: "f32[800]" = torch.ops.aten.sum.dim_IntList(where_32, [0, 2, 3])
    sub_239: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_77, unsqueeze_830);  convolution_77 = unsqueeze_830 = None
    mul_1065: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(where_32, sub_239)
    sum_67: "f32[800]" = torch.ops.aten.sum.dim_IntList(mul_1065, [0, 2, 3]);  mul_1065 = None
    mul_1066: "f32[800]" = torch.ops.aten.mul.Tensor(sum_66, 0.0006377551020408163)
    unsqueeze_831: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_1066, 0);  mul_1066 = None
    unsqueeze_832: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_831, 2);  unsqueeze_831 = None
    unsqueeze_833: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_832, 3);  unsqueeze_832 = None
    mul_1067: "f32[800]" = torch.ops.aten.mul.Tensor(sum_67, 0.0006377551020408163)
    mul_1068: "f32[800]" = torch.ops.aten.mul.Tensor(squeeze_235, squeeze_235)
    mul_1069: "f32[800]" = torch.ops.aten.mul.Tensor(mul_1067, mul_1068);  mul_1067 = mul_1068 = None
    unsqueeze_834: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_1069, 0);  mul_1069 = None
    unsqueeze_835: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_834, 2);  unsqueeze_834 = None
    unsqueeze_836: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_835, 3);  unsqueeze_835 = None
    mul_1070: "f32[800]" = torch.ops.aten.mul.Tensor(squeeze_235, primals_157);  primals_157 = None
    unsqueeze_837: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_1070, 0);  mul_1070 = None
    unsqueeze_838: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_837, 2);  unsqueeze_837 = None
    unsqueeze_839: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_838, 3);  unsqueeze_838 = None
    mul_1071: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_239, unsqueeze_836);  sub_239 = unsqueeze_836 = None
    sub_241: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(where_32, mul_1071);  where_32 = mul_1071 = None
    sub_242: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(sub_241, unsqueeze_833);  sub_241 = unsqueeze_833 = None
    mul_1072: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_242, unsqueeze_839);  sub_242 = unsqueeze_839 = None
    mul_1073: "f32[800]" = torch.ops.aten.mul.Tensor(sum_67, squeeze_235);  sum_67 = squeeze_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_33 = torch.ops.aten.convolution_backward.default(mul_1072, relu_77, primals_300, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False]);  mul_1072 = primals_300 = None
    getitem_323: "f32[8, 800, 14, 14]" = convolution_backward_33[0]
    getitem_324: "f32[800, 16, 3, 3]" = convolution_backward_33[1];  convolution_backward_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_33: "b8[8, 800, 14, 14]" = torch.ops.aten.le.Scalar(relu_77, 0);  relu_77 = None
    where_33: "f32[8, 800, 14, 14]" = torch.ops.aten.where.self(le_33, full_default, getitem_323);  le_33 = getitem_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_68: "f32[800]" = torch.ops.aten.sum.dim_IntList(where_33, [0, 2, 3])
    sub_243: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_76, unsqueeze_842);  convolution_76 = unsqueeze_842 = None
    mul_1074: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(where_33, sub_243)
    sum_69: "f32[800]" = torch.ops.aten.sum.dim_IntList(mul_1074, [0, 2, 3]);  mul_1074 = None
    mul_1075: "f32[800]" = torch.ops.aten.mul.Tensor(sum_68, 0.0006377551020408163)
    unsqueeze_843: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_1075, 0);  mul_1075 = None
    unsqueeze_844: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_843, 2);  unsqueeze_843 = None
    unsqueeze_845: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_844, 3);  unsqueeze_844 = None
    mul_1076: "f32[800]" = torch.ops.aten.mul.Tensor(sum_69, 0.0006377551020408163)
    mul_1077: "f32[800]" = torch.ops.aten.mul.Tensor(squeeze_232, squeeze_232)
    mul_1078: "f32[800]" = torch.ops.aten.mul.Tensor(mul_1076, mul_1077);  mul_1076 = mul_1077 = None
    unsqueeze_846: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_1078, 0);  mul_1078 = None
    unsqueeze_847: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_846, 2);  unsqueeze_846 = None
    unsqueeze_848: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_847, 3);  unsqueeze_847 = None
    mul_1079: "f32[800]" = torch.ops.aten.mul.Tensor(squeeze_232, primals_155);  primals_155 = None
    unsqueeze_849: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_1079, 0);  mul_1079 = None
    unsqueeze_850: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_849, 2);  unsqueeze_849 = None
    unsqueeze_851: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_850, 3);  unsqueeze_850 = None
    mul_1080: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_243, unsqueeze_848);  sub_243 = unsqueeze_848 = None
    sub_245: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(where_33, mul_1080);  where_33 = mul_1080 = None
    sub_246: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(sub_245, unsqueeze_845);  sub_245 = unsqueeze_845 = None
    mul_1081: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_246, unsqueeze_851);  sub_246 = unsqueeze_851 = None
    mul_1082: "f32[800]" = torch.ops.aten.mul.Tensor(sum_69, squeeze_232);  sum_69 = squeeze_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_34 = torch.ops.aten.convolution_backward.default(mul_1081, relu_76, primals_299, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1081 = primals_299 = None
    getitem_326: "f32[8, 1920, 14, 14]" = convolution_backward_34[0]
    getitem_327: "f32[800, 1920, 1, 1]" = convolution_backward_34[1];  convolution_backward_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_34: "b8[8, 1920, 14, 14]" = torch.ops.aten.le.Scalar(relu_76, 0);  relu_76 = None
    where_34: "f32[8, 1920, 14, 14]" = torch.ops.aten.where.self(le_34, full_default, getitem_326);  le_34 = getitem_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_70: "f32[1920]" = torch.ops.aten.sum.dim_IntList(where_34, [0, 2, 3])
    sub_247: "f32[8, 1920, 14, 14]" = torch.ops.aten.sub.Tensor(cat_47, unsqueeze_854);  cat_47 = unsqueeze_854 = None
    mul_1083: "f32[8, 1920, 14, 14]" = torch.ops.aten.mul.Tensor(where_34, sub_247)
    sum_71: "f32[1920]" = torch.ops.aten.sum.dim_IntList(mul_1083, [0, 2, 3]);  mul_1083 = None
    mul_1084: "f32[1920]" = torch.ops.aten.mul.Tensor(sum_70, 0.0006377551020408163)
    unsqueeze_855: "f32[1, 1920]" = torch.ops.aten.unsqueeze.default(mul_1084, 0);  mul_1084 = None
    unsqueeze_856: "f32[1, 1920, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_855, 2);  unsqueeze_855 = None
    unsqueeze_857: "f32[1, 1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_856, 3);  unsqueeze_856 = None
    mul_1085: "f32[1920]" = torch.ops.aten.mul.Tensor(sum_71, 0.0006377551020408163)
    mul_1086: "f32[1920]" = torch.ops.aten.mul.Tensor(squeeze_229, squeeze_229)
    mul_1087: "f32[1920]" = torch.ops.aten.mul.Tensor(mul_1085, mul_1086);  mul_1085 = mul_1086 = None
    unsqueeze_858: "f32[1, 1920]" = torch.ops.aten.unsqueeze.default(mul_1087, 0);  mul_1087 = None
    unsqueeze_859: "f32[1, 1920, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_858, 2);  unsqueeze_858 = None
    unsqueeze_860: "f32[1, 1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_859, 3);  unsqueeze_859 = None
    mul_1088: "f32[1920]" = torch.ops.aten.mul.Tensor(squeeze_229, primals_153);  primals_153 = None
    unsqueeze_861: "f32[1, 1920]" = torch.ops.aten.unsqueeze.default(mul_1088, 0);  mul_1088 = None
    unsqueeze_862: "f32[1, 1920, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_861, 2);  unsqueeze_861 = None
    unsqueeze_863: "f32[1, 1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_862, 3);  unsqueeze_862 = None
    mul_1089: "f32[8, 1920, 14, 14]" = torch.ops.aten.mul.Tensor(sub_247, unsqueeze_860);  sub_247 = unsqueeze_860 = None
    sub_249: "f32[8, 1920, 14, 14]" = torch.ops.aten.sub.Tensor(where_34, mul_1089);  where_34 = mul_1089 = None
    sub_250: "f32[8, 1920, 14, 14]" = torch.ops.aten.sub.Tensor(sub_249, unsqueeze_857);  sub_249 = unsqueeze_857 = None
    mul_1090: "f32[8, 1920, 14, 14]" = torch.ops.aten.mul.Tensor(sub_250, unsqueeze_863);  sub_250 = unsqueeze_863 = None
    mul_1091: "f32[1920]" = torch.ops.aten.mul.Tensor(sum_71, squeeze_229);  sum_71 = squeeze_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    slice_357: "f32[8, 1024, 14, 14]" = torch.ops.aten.slice.Tensor(mul_1090, 1, 0, 1024)
    slice_358: "f32[8, 896, 14, 14]" = torch.ops.aten.slice.Tensor(mul_1090, 1, 1024, 1920);  mul_1090 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    add_621: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_618, slice_357);  add_618 = slice_357 = None
    add_622: "f32[8, 896, 14, 14]" = torch.ops.aten.add.Tensor(slice_355, slice_358);  slice_355 = slice_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    slice_359: "f32[8, 832, 14, 14]" = torch.ops.aten.slice.Tensor(add_622, 1, 0, 832)
    slice_360: "f32[8, 64, 14, 14]" = torch.ops.aten.slice.Tensor(add_622, 1, 832, 896);  add_622 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    slice_scatter_96: "f32[8, 64, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_43, slice_360, 3, 0, 9223372036854775807);  slice_360 = None
    slice_scatter_98: "f32[8, 1088, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_45, slice_scatter_96, 1, 1024, 9223372036854775807);  slice_scatter_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    slice_scatter_102: "f32[8, 1088, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_45, add_621, 1, 0, 1024)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    add_623: "f32[8, 1088, 14, 14]" = torch.ops.aten.add.Tensor(slice_scatter_98, slice_scatter_102);  slice_scatter_98 = slice_scatter_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_35 = torch.ops.aten.convolution_backward.default(add_623, relu_75, primals_298, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_623 = primals_298 = None
    getitem_329: "f32[8, 800, 14, 14]" = convolution_backward_35[0]
    getitem_330: "f32[1088, 800, 1, 1]" = convolution_backward_35[1];  convolution_backward_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_35: "b8[8, 800, 14, 14]" = torch.ops.aten.le.Scalar(relu_75, 0);  relu_75 = None
    where_35: "f32[8, 800, 14, 14]" = torch.ops.aten.where.self(le_35, full_default, getitem_329);  le_35 = getitem_329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_72: "f32[800]" = torch.ops.aten.sum.dim_IntList(where_35, [0, 2, 3])
    sub_251: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_74, unsqueeze_866);  convolution_74 = unsqueeze_866 = None
    mul_1092: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(where_35, sub_251)
    sum_73: "f32[800]" = torch.ops.aten.sum.dim_IntList(mul_1092, [0, 2, 3]);  mul_1092 = None
    mul_1093: "f32[800]" = torch.ops.aten.mul.Tensor(sum_72, 0.0006377551020408163)
    unsqueeze_867: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_1093, 0);  mul_1093 = None
    unsqueeze_868: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_867, 2);  unsqueeze_867 = None
    unsqueeze_869: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_868, 3);  unsqueeze_868 = None
    mul_1094: "f32[800]" = torch.ops.aten.mul.Tensor(sum_73, 0.0006377551020408163)
    mul_1095: "f32[800]" = torch.ops.aten.mul.Tensor(squeeze_226, squeeze_226)
    mul_1096: "f32[800]" = torch.ops.aten.mul.Tensor(mul_1094, mul_1095);  mul_1094 = mul_1095 = None
    unsqueeze_870: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_1096, 0);  mul_1096 = None
    unsqueeze_871: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_870, 2);  unsqueeze_870 = None
    unsqueeze_872: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_871, 3);  unsqueeze_871 = None
    mul_1097: "f32[800]" = torch.ops.aten.mul.Tensor(squeeze_226, primals_151);  primals_151 = None
    unsqueeze_873: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_1097, 0);  mul_1097 = None
    unsqueeze_874: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_873, 2);  unsqueeze_873 = None
    unsqueeze_875: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_874, 3);  unsqueeze_874 = None
    mul_1098: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_251, unsqueeze_872);  sub_251 = unsqueeze_872 = None
    sub_253: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(where_35, mul_1098);  where_35 = mul_1098 = None
    sub_254: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(sub_253, unsqueeze_869);  sub_253 = unsqueeze_869 = None
    mul_1099: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_254, unsqueeze_875);  sub_254 = unsqueeze_875 = None
    mul_1100: "f32[800]" = torch.ops.aten.mul.Tensor(sum_73, squeeze_226);  sum_73 = squeeze_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_36 = torch.ops.aten.convolution_backward.default(mul_1099, relu_74, primals_297, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False]);  mul_1099 = primals_297 = None
    getitem_332: "f32[8, 800, 14, 14]" = convolution_backward_36[0]
    getitem_333: "f32[800, 16, 3, 3]" = convolution_backward_36[1];  convolution_backward_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_36: "b8[8, 800, 14, 14]" = torch.ops.aten.le.Scalar(relu_74, 0);  relu_74 = None
    where_36: "f32[8, 800, 14, 14]" = torch.ops.aten.where.self(le_36, full_default, getitem_332);  le_36 = getitem_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_74: "f32[800]" = torch.ops.aten.sum.dim_IntList(where_36, [0, 2, 3])
    sub_255: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_73, unsqueeze_878);  convolution_73 = unsqueeze_878 = None
    mul_1101: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(where_36, sub_255)
    sum_75: "f32[800]" = torch.ops.aten.sum.dim_IntList(mul_1101, [0, 2, 3]);  mul_1101 = None
    mul_1102: "f32[800]" = torch.ops.aten.mul.Tensor(sum_74, 0.0006377551020408163)
    unsqueeze_879: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_1102, 0);  mul_1102 = None
    unsqueeze_880: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_879, 2);  unsqueeze_879 = None
    unsqueeze_881: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_880, 3);  unsqueeze_880 = None
    mul_1103: "f32[800]" = torch.ops.aten.mul.Tensor(sum_75, 0.0006377551020408163)
    mul_1104: "f32[800]" = torch.ops.aten.mul.Tensor(squeeze_223, squeeze_223)
    mul_1105: "f32[800]" = torch.ops.aten.mul.Tensor(mul_1103, mul_1104);  mul_1103 = mul_1104 = None
    unsqueeze_882: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_1105, 0);  mul_1105 = None
    unsqueeze_883: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_882, 2);  unsqueeze_882 = None
    unsqueeze_884: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_883, 3);  unsqueeze_883 = None
    mul_1106: "f32[800]" = torch.ops.aten.mul.Tensor(squeeze_223, primals_149);  primals_149 = None
    unsqueeze_885: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_1106, 0);  mul_1106 = None
    unsqueeze_886: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_885, 2);  unsqueeze_885 = None
    unsqueeze_887: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_886, 3);  unsqueeze_886 = None
    mul_1107: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_255, unsqueeze_884);  sub_255 = unsqueeze_884 = None
    sub_257: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(where_36, mul_1107);  where_36 = mul_1107 = None
    sub_258: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(sub_257, unsqueeze_881);  sub_257 = unsqueeze_881 = None
    mul_1108: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_258, unsqueeze_887);  sub_258 = unsqueeze_887 = None
    mul_1109: "f32[800]" = torch.ops.aten.mul.Tensor(sum_75, squeeze_223);  sum_75 = squeeze_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_37 = torch.ops.aten.convolution_backward.default(mul_1108, relu_73, primals_296, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1108 = primals_296 = None
    getitem_335: "f32[8, 1856, 14, 14]" = convolution_backward_37[0]
    getitem_336: "f32[800, 1856, 1, 1]" = convolution_backward_37[1];  convolution_backward_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_37: "b8[8, 1856, 14, 14]" = torch.ops.aten.le.Scalar(relu_73, 0);  relu_73 = None
    where_37: "f32[8, 1856, 14, 14]" = torch.ops.aten.where.self(le_37, full_default, getitem_335);  le_37 = getitem_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_76: "f32[1856]" = torch.ops.aten.sum.dim_IntList(where_37, [0, 2, 3])
    sub_259: "f32[8, 1856, 14, 14]" = torch.ops.aten.sub.Tensor(cat_45, unsqueeze_890);  cat_45 = unsqueeze_890 = None
    mul_1110: "f32[8, 1856, 14, 14]" = torch.ops.aten.mul.Tensor(where_37, sub_259)
    sum_77: "f32[1856]" = torch.ops.aten.sum.dim_IntList(mul_1110, [0, 2, 3]);  mul_1110 = None
    mul_1111: "f32[1856]" = torch.ops.aten.mul.Tensor(sum_76, 0.0006377551020408163)
    unsqueeze_891: "f32[1, 1856]" = torch.ops.aten.unsqueeze.default(mul_1111, 0);  mul_1111 = None
    unsqueeze_892: "f32[1, 1856, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_891, 2);  unsqueeze_891 = None
    unsqueeze_893: "f32[1, 1856, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_892, 3);  unsqueeze_892 = None
    mul_1112: "f32[1856]" = torch.ops.aten.mul.Tensor(sum_77, 0.0006377551020408163)
    mul_1113: "f32[1856]" = torch.ops.aten.mul.Tensor(squeeze_220, squeeze_220)
    mul_1114: "f32[1856]" = torch.ops.aten.mul.Tensor(mul_1112, mul_1113);  mul_1112 = mul_1113 = None
    unsqueeze_894: "f32[1, 1856]" = torch.ops.aten.unsqueeze.default(mul_1114, 0);  mul_1114 = None
    unsqueeze_895: "f32[1, 1856, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_894, 2);  unsqueeze_894 = None
    unsqueeze_896: "f32[1, 1856, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_895, 3);  unsqueeze_895 = None
    mul_1115: "f32[1856]" = torch.ops.aten.mul.Tensor(squeeze_220, primals_147);  primals_147 = None
    unsqueeze_897: "f32[1, 1856]" = torch.ops.aten.unsqueeze.default(mul_1115, 0);  mul_1115 = None
    unsqueeze_898: "f32[1, 1856, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_897, 2);  unsqueeze_897 = None
    unsqueeze_899: "f32[1, 1856, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_898, 3);  unsqueeze_898 = None
    mul_1116: "f32[8, 1856, 14, 14]" = torch.ops.aten.mul.Tensor(sub_259, unsqueeze_896);  sub_259 = unsqueeze_896 = None
    sub_261: "f32[8, 1856, 14, 14]" = torch.ops.aten.sub.Tensor(where_37, mul_1116);  where_37 = mul_1116 = None
    sub_262: "f32[8, 1856, 14, 14]" = torch.ops.aten.sub.Tensor(sub_261, unsqueeze_893);  sub_261 = unsqueeze_893 = None
    mul_1117: "f32[8, 1856, 14, 14]" = torch.ops.aten.mul.Tensor(sub_262, unsqueeze_899);  sub_262 = unsqueeze_899 = None
    mul_1118: "f32[1856]" = torch.ops.aten.mul.Tensor(sum_77, squeeze_220);  sum_77 = squeeze_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    slice_361: "f32[8, 1024, 14, 14]" = torch.ops.aten.slice.Tensor(mul_1117, 1, 0, 1024)
    slice_362: "f32[8, 832, 14, 14]" = torch.ops.aten.slice.Tensor(mul_1117, 1, 1024, 1856);  mul_1117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    add_624: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_621, slice_361);  add_621 = slice_361 = None
    add_625: "f32[8, 832, 14, 14]" = torch.ops.aten.add.Tensor(slice_359, slice_362);  slice_359 = slice_362 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    slice_363: "f32[8, 768, 14, 14]" = torch.ops.aten.slice.Tensor(add_625, 1, 0, 768)
    slice_364: "f32[8, 64, 14, 14]" = torch.ops.aten.slice.Tensor(add_625, 1, 768, 832);  add_625 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    slice_scatter_104: "f32[8, 64, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_43, slice_364, 3, 0, 9223372036854775807);  slice_364 = None
    slice_scatter_106: "f32[8, 1088, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_45, slice_scatter_104, 1, 1024, 9223372036854775807);  slice_scatter_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    slice_scatter_110: "f32[8, 1088, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_45, add_624, 1, 0, 1024)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    add_626: "f32[8, 1088, 14, 14]" = torch.ops.aten.add.Tensor(slice_scatter_106, slice_scatter_110);  slice_scatter_106 = slice_scatter_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_38 = torch.ops.aten.convolution_backward.default(add_626, relu_72, primals_295, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_626 = primals_295 = None
    getitem_338: "f32[8, 800, 14, 14]" = convolution_backward_38[0]
    getitem_339: "f32[1088, 800, 1, 1]" = convolution_backward_38[1];  convolution_backward_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_38: "b8[8, 800, 14, 14]" = torch.ops.aten.le.Scalar(relu_72, 0);  relu_72 = None
    where_38: "f32[8, 800, 14, 14]" = torch.ops.aten.where.self(le_38, full_default, getitem_338);  le_38 = getitem_338 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_78: "f32[800]" = torch.ops.aten.sum.dim_IntList(where_38, [0, 2, 3])
    sub_263: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_71, unsqueeze_902);  convolution_71 = unsqueeze_902 = None
    mul_1119: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(where_38, sub_263)
    sum_79: "f32[800]" = torch.ops.aten.sum.dim_IntList(mul_1119, [0, 2, 3]);  mul_1119 = None
    mul_1120: "f32[800]" = torch.ops.aten.mul.Tensor(sum_78, 0.0006377551020408163)
    unsqueeze_903: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_1120, 0);  mul_1120 = None
    unsqueeze_904: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_903, 2);  unsqueeze_903 = None
    unsqueeze_905: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_904, 3);  unsqueeze_904 = None
    mul_1121: "f32[800]" = torch.ops.aten.mul.Tensor(sum_79, 0.0006377551020408163)
    mul_1122: "f32[800]" = torch.ops.aten.mul.Tensor(squeeze_217, squeeze_217)
    mul_1123: "f32[800]" = torch.ops.aten.mul.Tensor(mul_1121, mul_1122);  mul_1121 = mul_1122 = None
    unsqueeze_906: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_1123, 0);  mul_1123 = None
    unsqueeze_907: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_906, 2);  unsqueeze_906 = None
    unsqueeze_908: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_907, 3);  unsqueeze_907 = None
    mul_1124: "f32[800]" = torch.ops.aten.mul.Tensor(squeeze_217, primals_145);  primals_145 = None
    unsqueeze_909: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_1124, 0);  mul_1124 = None
    unsqueeze_910: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_909, 2);  unsqueeze_909 = None
    unsqueeze_911: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_910, 3);  unsqueeze_910 = None
    mul_1125: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_263, unsqueeze_908);  sub_263 = unsqueeze_908 = None
    sub_265: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(where_38, mul_1125);  where_38 = mul_1125 = None
    sub_266: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(sub_265, unsqueeze_905);  sub_265 = unsqueeze_905 = None
    mul_1126: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_266, unsqueeze_911);  sub_266 = unsqueeze_911 = None
    mul_1127: "f32[800]" = torch.ops.aten.mul.Tensor(sum_79, squeeze_217);  sum_79 = squeeze_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_39 = torch.ops.aten.convolution_backward.default(mul_1126, relu_71, primals_294, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False]);  mul_1126 = primals_294 = None
    getitem_341: "f32[8, 800, 14, 14]" = convolution_backward_39[0]
    getitem_342: "f32[800, 16, 3, 3]" = convolution_backward_39[1];  convolution_backward_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_39: "b8[8, 800, 14, 14]" = torch.ops.aten.le.Scalar(relu_71, 0);  relu_71 = None
    where_39: "f32[8, 800, 14, 14]" = torch.ops.aten.where.self(le_39, full_default, getitem_341);  le_39 = getitem_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_80: "f32[800]" = torch.ops.aten.sum.dim_IntList(where_39, [0, 2, 3])
    sub_267: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_70, unsqueeze_914);  convolution_70 = unsqueeze_914 = None
    mul_1128: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(where_39, sub_267)
    sum_81: "f32[800]" = torch.ops.aten.sum.dim_IntList(mul_1128, [0, 2, 3]);  mul_1128 = None
    mul_1129: "f32[800]" = torch.ops.aten.mul.Tensor(sum_80, 0.0006377551020408163)
    unsqueeze_915: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_1129, 0);  mul_1129 = None
    unsqueeze_916: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_915, 2);  unsqueeze_915 = None
    unsqueeze_917: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_916, 3);  unsqueeze_916 = None
    mul_1130: "f32[800]" = torch.ops.aten.mul.Tensor(sum_81, 0.0006377551020408163)
    mul_1131: "f32[800]" = torch.ops.aten.mul.Tensor(squeeze_214, squeeze_214)
    mul_1132: "f32[800]" = torch.ops.aten.mul.Tensor(mul_1130, mul_1131);  mul_1130 = mul_1131 = None
    unsqueeze_918: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_1132, 0);  mul_1132 = None
    unsqueeze_919: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_918, 2);  unsqueeze_918 = None
    unsqueeze_920: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_919, 3);  unsqueeze_919 = None
    mul_1133: "f32[800]" = torch.ops.aten.mul.Tensor(squeeze_214, primals_143);  primals_143 = None
    unsqueeze_921: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_1133, 0);  mul_1133 = None
    unsqueeze_922: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_921, 2);  unsqueeze_921 = None
    unsqueeze_923: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_922, 3);  unsqueeze_922 = None
    mul_1134: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_267, unsqueeze_920);  sub_267 = unsqueeze_920 = None
    sub_269: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(where_39, mul_1134);  where_39 = mul_1134 = None
    sub_270: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(sub_269, unsqueeze_917);  sub_269 = unsqueeze_917 = None
    mul_1135: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_270, unsqueeze_923);  sub_270 = unsqueeze_923 = None
    mul_1136: "f32[800]" = torch.ops.aten.mul.Tensor(sum_81, squeeze_214);  sum_81 = squeeze_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_40 = torch.ops.aten.convolution_backward.default(mul_1135, relu_70, primals_293, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1135 = primals_293 = None
    getitem_344: "f32[8, 1792, 14, 14]" = convolution_backward_40[0]
    getitem_345: "f32[800, 1792, 1, 1]" = convolution_backward_40[1];  convolution_backward_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_40: "b8[8, 1792, 14, 14]" = torch.ops.aten.le.Scalar(relu_70, 0);  relu_70 = None
    where_40: "f32[8, 1792, 14, 14]" = torch.ops.aten.where.self(le_40, full_default, getitem_344);  le_40 = getitem_344 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_82: "f32[1792]" = torch.ops.aten.sum.dim_IntList(where_40, [0, 2, 3])
    sub_271: "f32[8, 1792, 14, 14]" = torch.ops.aten.sub.Tensor(cat_43, unsqueeze_926);  cat_43 = unsqueeze_926 = None
    mul_1137: "f32[8, 1792, 14, 14]" = torch.ops.aten.mul.Tensor(where_40, sub_271)
    sum_83: "f32[1792]" = torch.ops.aten.sum.dim_IntList(mul_1137, [0, 2, 3]);  mul_1137 = None
    mul_1138: "f32[1792]" = torch.ops.aten.mul.Tensor(sum_82, 0.0006377551020408163)
    unsqueeze_927: "f32[1, 1792]" = torch.ops.aten.unsqueeze.default(mul_1138, 0);  mul_1138 = None
    unsqueeze_928: "f32[1, 1792, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_927, 2);  unsqueeze_927 = None
    unsqueeze_929: "f32[1, 1792, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_928, 3);  unsqueeze_928 = None
    mul_1139: "f32[1792]" = torch.ops.aten.mul.Tensor(sum_83, 0.0006377551020408163)
    mul_1140: "f32[1792]" = torch.ops.aten.mul.Tensor(squeeze_211, squeeze_211)
    mul_1141: "f32[1792]" = torch.ops.aten.mul.Tensor(mul_1139, mul_1140);  mul_1139 = mul_1140 = None
    unsqueeze_930: "f32[1, 1792]" = torch.ops.aten.unsqueeze.default(mul_1141, 0);  mul_1141 = None
    unsqueeze_931: "f32[1, 1792, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_930, 2);  unsqueeze_930 = None
    unsqueeze_932: "f32[1, 1792, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_931, 3);  unsqueeze_931 = None
    mul_1142: "f32[1792]" = torch.ops.aten.mul.Tensor(squeeze_211, primals_141);  primals_141 = None
    unsqueeze_933: "f32[1, 1792]" = torch.ops.aten.unsqueeze.default(mul_1142, 0);  mul_1142 = None
    unsqueeze_934: "f32[1, 1792, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_933, 2);  unsqueeze_933 = None
    unsqueeze_935: "f32[1, 1792, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_934, 3);  unsqueeze_934 = None
    mul_1143: "f32[8, 1792, 14, 14]" = torch.ops.aten.mul.Tensor(sub_271, unsqueeze_932);  sub_271 = unsqueeze_932 = None
    sub_273: "f32[8, 1792, 14, 14]" = torch.ops.aten.sub.Tensor(where_40, mul_1143);  where_40 = mul_1143 = None
    sub_274: "f32[8, 1792, 14, 14]" = torch.ops.aten.sub.Tensor(sub_273, unsqueeze_929);  sub_273 = unsqueeze_929 = None
    mul_1144: "f32[8, 1792, 14, 14]" = torch.ops.aten.mul.Tensor(sub_274, unsqueeze_935);  sub_274 = unsqueeze_935 = None
    mul_1145: "f32[1792]" = torch.ops.aten.mul.Tensor(sum_83, squeeze_211);  sum_83 = squeeze_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    slice_365: "f32[8, 1024, 14, 14]" = torch.ops.aten.slice.Tensor(mul_1144, 1, 0, 1024)
    slice_366: "f32[8, 768, 14, 14]" = torch.ops.aten.slice.Tensor(mul_1144, 1, 1024, 1792);  mul_1144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    add_627: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_624, slice_365);  add_624 = slice_365 = None
    add_628: "f32[8, 768, 14, 14]" = torch.ops.aten.add.Tensor(slice_363, slice_366);  slice_363 = slice_366 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    slice_367: "f32[8, 704, 14, 14]" = torch.ops.aten.slice.Tensor(add_628, 1, 0, 704)
    slice_368: "f32[8, 64, 14, 14]" = torch.ops.aten.slice.Tensor(add_628, 1, 704, 768);  add_628 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    slice_scatter_112: "f32[8, 64, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_43, slice_368, 3, 0, 9223372036854775807);  slice_368 = None
    slice_scatter_114: "f32[8, 1088, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_45, slice_scatter_112, 1, 1024, 9223372036854775807);  slice_scatter_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    slice_scatter_118: "f32[8, 1088, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_45, add_627, 1, 0, 1024)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    add_629: "f32[8, 1088, 14, 14]" = torch.ops.aten.add.Tensor(slice_scatter_114, slice_scatter_118);  slice_scatter_114 = slice_scatter_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_41 = torch.ops.aten.convolution_backward.default(add_629, relu_69, primals_292, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_629 = primals_292 = None
    getitem_347: "f32[8, 800, 14, 14]" = convolution_backward_41[0]
    getitem_348: "f32[1088, 800, 1, 1]" = convolution_backward_41[1];  convolution_backward_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_41: "b8[8, 800, 14, 14]" = torch.ops.aten.le.Scalar(relu_69, 0);  relu_69 = None
    where_41: "f32[8, 800, 14, 14]" = torch.ops.aten.where.self(le_41, full_default, getitem_347);  le_41 = getitem_347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_84: "f32[800]" = torch.ops.aten.sum.dim_IntList(where_41, [0, 2, 3])
    sub_275: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_68, unsqueeze_938);  convolution_68 = unsqueeze_938 = None
    mul_1146: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(where_41, sub_275)
    sum_85: "f32[800]" = torch.ops.aten.sum.dim_IntList(mul_1146, [0, 2, 3]);  mul_1146 = None
    mul_1147: "f32[800]" = torch.ops.aten.mul.Tensor(sum_84, 0.0006377551020408163)
    unsqueeze_939: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_1147, 0);  mul_1147 = None
    unsqueeze_940: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_939, 2);  unsqueeze_939 = None
    unsqueeze_941: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_940, 3);  unsqueeze_940 = None
    mul_1148: "f32[800]" = torch.ops.aten.mul.Tensor(sum_85, 0.0006377551020408163)
    mul_1149: "f32[800]" = torch.ops.aten.mul.Tensor(squeeze_208, squeeze_208)
    mul_1150: "f32[800]" = torch.ops.aten.mul.Tensor(mul_1148, mul_1149);  mul_1148 = mul_1149 = None
    unsqueeze_942: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_1150, 0);  mul_1150 = None
    unsqueeze_943: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_942, 2);  unsqueeze_942 = None
    unsqueeze_944: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_943, 3);  unsqueeze_943 = None
    mul_1151: "f32[800]" = torch.ops.aten.mul.Tensor(squeeze_208, primals_139);  primals_139 = None
    unsqueeze_945: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_1151, 0);  mul_1151 = None
    unsqueeze_946: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_945, 2);  unsqueeze_945 = None
    unsqueeze_947: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_946, 3);  unsqueeze_946 = None
    mul_1152: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_275, unsqueeze_944);  sub_275 = unsqueeze_944 = None
    sub_277: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(where_41, mul_1152);  where_41 = mul_1152 = None
    sub_278: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(sub_277, unsqueeze_941);  sub_277 = unsqueeze_941 = None
    mul_1153: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_278, unsqueeze_947);  sub_278 = unsqueeze_947 = None
    mul_1154: "f32[800]" = torch.ops.aten.mul.Tensor(sum_85, squeeze_208);  sum_85 = squeeze_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_42 = torch.ops.aten.convolution_backward.default(mul_1153, relu_68, primals_291, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False]);  mul_1153 = primals_291 = None
    getitem_350: "f32[8, 800, 14, 14]" = convolution_backward_42[0]
    getitem_351: "f32[800, 16, 3, 3]" = convolution_backward_42[1];  convolution_backward_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_42: "b8[8, 800, 14, 14]" = torch.ops.aten.le.Scalar(relu_68, 0);  relu_68 = None
    where_42: "f32[8, 800, 14, 14]" = torch.ops.aten.where.self(le_42, full_default, getitem_350);  le_42 = getitem_350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_86: "f32[800]" = torch.ops.aten.sum.dim_IntList(where_42, [0, 2, 3])
    sub_279: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_67, unsqueeze_950);  convolution_67 = unsqueeze_950 = None
    mul_1155: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(where_42, sub_279)
    sum_87: "f32[800]" = torch.ops.aten.sum.dim_IntList(mul_1155, [0, 2, 3]);  mul_1155 = None
    mul_1156: "f32[800]" = torch.ops.aten.mul.Tensor(sum_86, 0.0006377551020408163)
    unsqueeze_951: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_1156, 0);  mul_1156 = None
    unsqueeze_952: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_951, 2);  unsqueeze_951 = None
    unsqueeze_953: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_952, 3);  unsqueeze_952 = None
    mul_1157: "f32[800]" = torch.ops.aten.mul.Tensor(sum_87, 0.0006377551020408163)
    mul_1158: "f32[800]" = torch.ops.aten.mul.Tensor(squeeze_205, squeeze_205)
    mul_1159: "f32[800]" = torch.ops.aten.mul.Tensor(mul_1157, mul_1158);  mul_1157 = mul_1158 = None
    unsqueeze_954: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_1159, 0);  mul_1159 = None
    unsqueeze_955: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_954, 2);  unsqueeze_954 = None
    unsqueeze_956: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_955, 3);  unsqueeze_955 = None
    mul_1160: "f32[800]" = torch.ops.aten.mul.Tensor(squeeze_205, primals_137);  primals_137 = None
    unsqueeze_957: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_1160, 0);  mul_1160 = None
    unsqueeze_958: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_957, 2);  unsqueeze_957 = None
    unsqueeze_959: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_958, 3);  unsqueeze_958 = None
    mul_1161: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_279, unsqueeze_956);  sub_279 = unsqueeze_956 = None
    sub_281: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(where_42, mul_1161);  where_42 = mul_1161 = None
    sub_282: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(sub_281, unsqueeze_953);  sub_281 = unsqueeze_953 = None
    mul_1162: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_282, unsqueeze_959);  sub_282 = unsqueeze_959 = None
    mul_1163: "f32[800]" = torch.ops.aten.mul.Tensor(sum_87, squeeze_205);  sum_87 = squeeze_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_43 = torch.ops.aten.convolution_backward.default(mul_1162, relu_67, primals_290, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1162 = primals_290 = None
    getitem_353: "f32[8, 1728, 14, 14]" = convolution_backward_43[0]
    getitem_354: "f32[800, 1728, 1, 1]" = convolution_backward_43[1];  convolution_backward_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_43: "b8[8, 1728, 14, 14]" = torch.ops.aten.le.Scalar(relu_67, 0);  relu_67 = None
    where_43: "f32[8, 1728, 14, 14]" = torch.ops.aten.where.self(le_43, full_default, getitem_353);  le_43 = getitem_353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_88: "f32[1728]" = torch.ops.aten.sum.dim_IntList(where_43, [0, 2, 3])
    sub_283: "f32[8, 1728, 14, 14]" = torch.ops.aten.sub.Tensor(cat_41, unsqueeze_962);  cat_41 = unsqueeze_962 = None
    mul_1164: "f32[8, 1728, 14, 14]" = torch.ops.aten.mul.Tensor(where_43, sub_283)
    sum_89: "f32[1728]" = torch.ops.aten.sum.dim_IntList(mul_1164, [0, 2, 3]);  mul_1164 = None
    mul_1165: "f32[1728]" = torch.ops.aten.mul.Tensor(sum_88, 0.0006377551020408163)
    unsqueeze_963: "f32[1, 1728]" = torch.ops.aten.unsqueeze.default(mul_1165, 0);  mul_1165 = None
    unsqueeze_964: "f32[1, 1728, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_963, 2);  unsqueeze_963 = None
    unsqueeze_965: "f32[1, 1728, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_964, 3);  unsqueeze_964 = None
    mul_1166: "f32[1728]" = torch.ops.aten.mul.Tensor(sum_89, 0.0006377551020408163)
    mul_1167: "f32[1728]" = torch.ops.aten.mul.Tensor(squeeze_202, squeeze_202)
    mul_1168: "f32[1728]" = torch.ops.aten.mul.Tensor(mul_1166, mul_1167);  mul_1166 = mul_1167 = None
    unsqueeze_966: "f32[1, 1728]" = torch.ops.aten.unsqueeze.default(mul_1168, 0);  mul_1168 = None
    unsqueeze_967: "f32[1, 1728, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_966, 2);  unsqueeze_966 = None
    unsqueeze_968: "f32[1, 1728, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_967, 3);  unsqueeze_967 = None
    mul_1169: "f32[1728]" = torch.ops.aten.mul.Tensor(squeeze_202, primals_135);  primals_135 = None
    unsqueeze_969: "f32[1, 1728]" = torch.ops.aten.unsqueeze.default(mul_1169, 0);  mul_1169 = None
    unsqueeze_970: "f32[1, 1728, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_969, 2);  unsqueeze_969 = None
    unsqueeze_971: "f32[1, 1728, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_970, 3);  unsqueeze_970 = None
    mul_1170: "f32[8, 1728, 14, 14]" = torch.ops.aten.mul.Tensor(sub_283, unsqueeze_968);  sub_283 = unsqueeze_968 = None
    sub_285: "f32[8, 1728, 14, 14]" = torch.ops.aten.sub.Tensor(where_43, mul_1170);  where_43 = mul_1170 = None
    sub_286: "f32[8, 1728, 14, 14]" = torch.ops.aten.sub.Tensor(sub_285, unsqueeze_965);  sub_285 = unsqueeze_965 = None
    mul_1171: "f32[8, 1728, 14, 14]" = torch.ops.aten.mul.Tensor(sub_286, unsqueeze_971);  sub_286 = unsqueeze_971 = None
    mul_1172: "f32[1728]" = torch.ops.aten.mul.Tensor(sum_89, squeeze_202);  sum_89 = squeeze_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    slice_369: "f32[8, 1024, 14, 14]" = torch.ops.aten.slice.Tensor(mul_1171, 1, 0, 1024)
    slice_370: "f32[8, 704, 14, 14]" = torch.ops.aten.slice.Tensor(mul_1171, 1, 1024, 1728);  mul_1171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    add_630: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_627, slice_369);  add_627 = slice_369 = None
    add_631: "f32[8, 704, 14, 14]" = torch.ops.aten.add.Tensor(slice_367, slice_370);  slice_367 = slice_370 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    slice_371: "f32[8, 640, 14, 14]" = torch.ops.aten.slice.Tensor(add_631, 1, 0, 640)
    slice_372: "f32[8, 64, 14, 14]" = torch.ops.aten.slice.Tensor(add_631, 1, 640, 704);  add_631 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    slice_scatter_120: "f32[8, 64, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_43, slice_372, 3, 0, 9223372036854775807);  slice_372 = None
    slice_scatter_122: "f32[8, 1088, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_45, slice_scatter_120, 1, 1024, 9223372036854775807);  slice_scatter_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    slice_scatter_126: "f32[8, 1088, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_45, add_630, 1, 0, 1024)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    add_632: "f32[8, 1088, 14, 14]" = torch.ops.aten.add.Tensor(slice_scatter_122, slice_scatter_126);  slice_scatter_122 = slice_scatter_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_44 = torch.ops.aten.convolution_backward.default(add_632, relu_66, primals_289, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_632 = primals_289 = None
    getitem_356: "f32[8, 800, 14, 14]" = convolution_backward_44[0]
    getitem_357: "f32[1088, 800, 1, 1]" = convolution_backward_44[1];  convolution_backward_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_44: "b8[8, 800, 14, 14]" = torch.ops.aten.le.Scalar(relu_66, 0);  relu_66 = None
    where_44: "f32[8, 800, 14, 14]" = torch.ops.aten.where.self(le_44, full_default, getitem_356);  le_44 = getitem_356 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_90: "f32[800]" = torch.ops.aten.sum.dim_IntList(where_44, [0, 2, 3])
    sub_287: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_65, unsqueeze_974);  convolution_65 = unsqueeze_974 = None
    mul_1173: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(where_44, sub_287)
    sum_91: "f32[800]" = torch.ops.aten.sum.dim_IntList(mul_1173, [0, 2, 3]);  mul_1173 = None
    mul_1174: "f32[800]" = torch.ops.aten.mul.Tensor(sum_90, 0.0006377551020408163)
    unsqueeze_975: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_1174, 0);  mul_1174 = None
    unsqueeze_976: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_975, 2);  unsqueeze_975 = None
    unsqueeze_977: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_976, 3);  unsqueeze_976 = None
    mul_1175: "f32[800]" = torch.ops.aten.mul.Tensor(sum_91, 0.0006377551020408163)
    mul_1176: "f32[800]" = torch.ops.aten.mul.Tensor(squeeze_199, squeeze_199)
    mul_1177: "f32[800]" = torch.ops.aten.mul.Tensor(mul_1175, mul_1176);  mul_1175 = mul_1176 = None
    unsqueeze_978: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_1177, 0);  mul_1177 = None
    unsqueeze_979: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_978, 2);  unsqueeze_978 = None
    unsqueeze_980: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_979, 3);  unsqueeze_979 = None
    mul_1178: "f32[800]" = torch.ops.aten.mul.Tensor(squeeze_199, primals_133);  primals_133 = None
    unsqueeze_981: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_1178, 0);  mul_1178 = None
    unsqueeze_982: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_981, 2);  unsqueeze_981 = None
    unsqueeze_983: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_982, 3);  unsqueeze_982 = None
    mul_1179: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_287, unsqueeze_980);  sub_287 = unsqueeze_980 = None
    sub_289: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(where_44, mul_1179);  where_44 = mul_1179 = None
    sub_290: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(sub_289, unsqueeze_977);  sub_289 = unsqueeze_977 = None
    mul_1180: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_290, unsqueeze_983);  sub_290 = unsqueeze_983 = None
    mul_1181: "f32[800]" = torch.ops.aten.mul.Tensor(sum_91, squeeze_199);  sum_91 = squeeze_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_45 = torch.ops.aten.convolution_backward.default(mul_1180, relu_65, primals_288, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False]);  mul_1180 = primals_288 = None
    getitem_359: "f32[8, 800, 14, 14]" = convolution_backward_45[0]
    getitem_360: "f32[800, 16, 3, 3]" = convolution_backward_45[1];  convolution_backward_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_45: "b8[8, 800, 14, 14]" = torch.ops.aten.le.Scalar(relu_65, 0);  relu_65 = None
    where_45: "f32[8, 800, 14, 14]" = torch.ops.aten.where.self(le_45, full_default, getitem_359);  le_45 = getitem_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_92: "f32[800]" = torch.ops.aten.sum.dim_IntList(where_45, [0, 2, 3])
    sub_291: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_64, unsqueeze_986);  convolution_64 = unsqueeze_986 = None
    mul_1182: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(where_45, sub_291)
    sum_93: "f32[800]" = torch.ops.aten.sum.dim_IntList(mul_1182, [0, 2, 3]);  mul_1182 = None
    mul_1183: "f32[800]" = torch.ops.aten.mul.Tensor(sum_92, 0.0006377551020408163)
    unsqueeze_987: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_1183, 0);  mul_1183 = None
    unsqueeze_988: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_987, 2);  unsqueeze_987 = None
    unsqueeze_989: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_988, 3);  unsqueeze_988 = None
    mul_1184: "f32[800]" = torch.ops.aten.mul.Tensor(sum_93, 0.0006377551020408163)
    mul_1185: "f32[800]" = torch.ops.aten.mul.Tensor(squeeze_196, squeeze_196)
    mul_1186: "f32[800]" = torch.ops.aten.mul.Tensor(mul_1184, mul_1185);  mul_1184 = mul_1185 = None
    unsqueeze_990: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_1186, 0);  mul_1186 = None
    unsqueeze_991: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_990, 2);  unsqueeze_990 = None
    unsqueeze_992: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_991, 3);  unsqueeze_991 = None
    mul_1187: "f32[800]" = torch.ops.aten.mul.Tensor(squeeze_196, primals_131);  primals_131 = None
    unsqueeze_993: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_1187, 0);  mul_1187 = None
    unsqueeze_994: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_993, 2);  unsqueeze_993 = None
    unsqueeze_995: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_994, 3);  unsqueeze_994 = None
    mul_1188: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_291, unsqueeze_992);  sub_291 = unsqueeze_992 = None
    sub_293: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(where_45, mul_1188);  where_45 = mul_1188 = None
    sub_294: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(sub_293, unsqueeze_989);  sub_293 = unsqueeze_989 = None
    mul_1189: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_294, unsqueeze_995);  sub_294 = unsqueeze_995 = None
    mul_1190: "f32[800]" = torch.ops.aten.mul.Tensor(sum_93, squeeze_196);  sum_93 = squeeze_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_46 = torch.ops.aten.convolution_backward.default(mul_1189, relu_64, primals_287, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1189 = primals_287 = None
    getitem_362: "f32[8, 1664, 14, 14]" = convolution_backward_46[0]
    getitem_363: "f32[800, 1664, 1, 1]" = convolution_backward_46[1];  convolution_backward_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_46: "b8[8, 1664, 14, 14]" = torch.ops.aten.le.Scalar(relu_64, 0);  relu_64 = None
    where_46: "f32[8, 1664, 14, 14]" = torch.ops.aten.where.self(le_46, full_default, getitem_362);  le_46 = getitem_362 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_94: "f32[1664]" = torch.ops.aten.sum.dim_IntList(where_46, [0, 2, 3])
    sub_295: "f32[8, 1664, 14, 14]" = torch.ops.aten.sub.Tensor(cat_39, unsqueeze_998);  cat_39 = unsqueeze_998 = None
    mul_1191: "f32[8, 1664, 14, 14]" = torch.ops.aten.mul.Tensor(where_46, sub_295)
    sum_95: "f32[1664]" = torch.ops.aten.sum.dim_IntList(mul_1191, [0, 2, 3]);  mul_1191 = None
    mul_1192: "f32[1664]" = torch.ops.aten.mul.Tensor(sum_94, 0.0006377551020408163)
    unsqueeze_999: "f32[1, 1664]" = torch.ops.aten.unsqueeze.default(mul_1192, 0);  mul_1192 = None
    unsqueeze_1000: "f32[1, 1664, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_999, 2);  unsqueeze_999 = None
    unsqueeze_1001: "f32[1, 1664, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1000, 3);  unsqueeze_1000 = None
    mul_1193: "f32[1664]" = torch.ops.aten.mul.Tensor(sum_95, 0.0006377551020408163)
    mul_1194: "f32[1664]" = torch.ops.aten.mul.Tensor(squeeze_193, squeeze_193)
    mul_1195: "f32[1664]" = torch.ops.aten.mul.Tensor(mul_1193, mul_1194);  mul_1193 = mul_1194 = None
    unsqueeze_1002: "f32[1, 1664]" = torch.ops.aten.unsqueeze.default(mul_1195, 0);  mul_1195 = None
    unsqueeze_1003: "f32[1, 1664, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1002, 2);  unsqueeze_1002 = None
    unsqueeze_1004: "f32[1, 1664, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1003, 3);  unsqueeze_1003 = None
    mul_1196: "f32[1664]" = torch.ops.aten.mul.Tensor(squeeze_193, primals_129);  primals_129 = None
    unsqueeze_1005: "f32[1, 1664]" = torch.ops.aten.unsqueeze.default(mul_1196, 0);  mul_1196 = None
    unsqueeze_1006: "f32[1, 1664, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1005, 2);  unsqueeze_1005 = None
    unsqueeze_1007: "f32[1, 1664, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1006, 3);  unsqueeze_1006 = None
    mul_1197: "f32[8, 1664, 14, 14]" = torch.ops.aten.mul.Tensor(sub_295, unsqueeze_1004);  sub_295 = unsqueeze_1004 = None
    sub_297: "f32[8, 1664, 14, 14]" = torch.ops.aten.sub.Tensor(where_46, mul_1197);  where_46 = mul_1197 = None
    sub_298: "f32[8, 1664, 14, 14]" = torch.ops.aten.sub.Tensor(sub_297, unsqueeze_1001);  sub_297 = unsqueeze_1001 = None
    mul_1198: "f32[8, 1664, 14, 14]" = torch.ops.aten.mul.Tensor(sub_298, unsqueeze_1007);  sub_298 = unsqueeze_1007 = None
    mul_1199: "f32[1664]" = torch.ops.aten.mul.Tensor(sum_95, squeeze_193);  sum_95 = squeeze_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    slice_373: "f32[8, 1024, 14, 14]" = torch.ops.aten.slice.Tensor(mul_1198, 1, 0, 1024)
    slice_374: "f32[8, 640, 14, 14]" = torch.ops.aten.slice.Tensor(mul_1198, 1, 1024, 1664);  mul_1198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    add_633: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_630, slice_373);  add_630 = slice_373 = None
    add_634: "f32[8, 640, 14, 14]" = torch.ops.aten.add.Tensor(slice_371, slice_374);  slice_371 = slice_374 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    slice_375: "f32[8, 576, 14, 14]" = torch.ops.aten.slice.Tensor(add_634, 1, 0, 576)
    slice_376: "f32[8, 64, 14, 14]" = torch.ops.aten.slice.Tensor(add_634, 1, 576, 640);  add_634 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    slice_scatter_128: "f32[8, 64, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_43, slice_376, 3, 0, 9223372036854775807);  slice_376 = None
    slice_scatter_130: "f32[8, 1088, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_45, slice_scatter_128, 1, 1024, 9223372036854775807);  slice_scatter_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    slice_scatter_134: "f32[8, 1088, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_45, add_633, 1, 0, 1024)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    add_635: "f32[8, 1088, 14, 14]" = torch.ops.aten.add.Tensor(slice_scatter_130, slice_scatter_134);  slice_scatter_130 = slice_scatter_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_47 = torch.ops.aten.convolution_backward.default(add_635, relu_63, primals_286, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_635 = primals_286 = None
    getitem_365: "f32[8, 800, 14, 14]" = convolution_backward_47[0]
    getitem_366: "f32[1088, 800, 1, 1]" = convolution_backward_47[1];  convolution_backward_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_47: "b8[8, 800, 14, 14]" = torch.ops.aten.le.Scalar(relu_63, 0);  relu_63 = None
    where_47: "f32[8, 800, 14, 14]" = torch.ops.aten.where.self(le_47, full_default, getitem_365);  le_47 = getitem_365 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_96: "f32[800]" = torch.ops.aten.sum.dim_IntList(where_47, [0, 2, 3])
    sub_299: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_62, unsqueeze_1010);  convolution_62 = unsqueeze_1010 = None
    mul_1200: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(where_47, sub_299)
    sum_97: "f32[800]" = torch.ops.aten.sum.dim_IntList(mul_1200, [0, 2, 3]);  mul_1200 = None
    mul_1201: "f32[800]" = torch.ops.aten.mul.Tensor(sum_96, 0.0006377551020408163)
    unsqueeze_1011: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_1201, 0);  mul_1201 = None
    unsqueeze_1012: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1011, 2);  unsqueeze_1011 = None
    unsqueeze_1013: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1012, 3);  unsqueeze_1012 = None
    mul_1202: "f32[800]" = torch.ops.aten.mul.Tensor(sum_97, 0.0006377551020408163)
    mul_1203: "f32[800]" = torch.ops.aten.mul.Tensor(squeeze_190, squeeze_190)
    mul_1204: "f32[800]" = torch.ops.aten.mul.Tensor(mul_1202, mul_1203);  mul_1202 = mul_1203 = None
    unsqueeze_1014: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_1204, 0);  mul_1204 = None
    unsqueeze_1015: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1014, 2);  unsqueeze_1014 = None
    unsqueeze_1016: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1015, 3);  unsqueeze_1015 = None
    mul_1205: "f32[800]" = torch.ops.aten.mul.Tensor(squeeze_190, primals_127);  primals_127 = None
    unsqueeze_1017: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_1205, 0);  mul_1205 = None
    unsqueeze_1018: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1017, 2);  unsqueeze_1017 = None
    unsqueeze_1019: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1018, 3);  unsqueeze_1018 = None
    mul_1206: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_299, unsqueeze_1016);  sub_299 = unsqueeze_1016 = None
    sub_301: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(where_47, mul_1206);  where_47 = mul_1206 = None
    sub_302: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(sub_301, unsqueeze_1013);  sub_301 = unsqueeze_1013 = None
    mul_1207: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_302, unsqueeze_1019);  sub_302 = unsqueeze_1019 = None
    mul_1208: "f32[800]" = torch.ops.aten.mul.Tensor(sum_97, squeeze_190);  sum_97 = squeeze_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_48 = torch.ops.aten.convolution_backward.default(mul_1207, relu_62, primals_285, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False]);  mul_1207 = primals_285 = None
    getitem_368: "f32[8, 800, 14, 14]" = convolution_backward_48[0]
    getitem_369: "f32[800, 16, 3, 3]" = convolution_backward_48[1];  convolution_backward_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_48: "b8[8, 800, 14, 14]" = torch.ops.aten.le.Scalar(relu_62, 0);  relu_62 = None
    where_48: "f32[8, 800, 14, 14]" = torch.ops.aten.where.self(le_48, full_default, getitem_368);  le_48 = getitem_368 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_98: "f32[800]" = torch.ops.aten.sum.dim_IntList(where_48, [0, 2, 3])
    sub_303: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_61, unsqueeze_1022);  convolution_61 = unsqueeze_1022 = None
    mul_1209: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(where_48, sub_303)
    sum_99: "f32[800]" = torch.ops.aten.sum.dim_IntList(mul_1209, [0, 2, 3]);  mul_1209 = None
    mul_1210: "f32[800]" = torch.ops.aten.mul.Tensor(sum_98, 0.0006377551020408163)
    unsqueeze_1023: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_1210, 0);  mul_1210 = None
    unsqueeze_1024: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1023, 2);  unsqueeze_1023 = None
    unsqueeze_1025: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1024, 3);  unsqueeze_1024 = None
    mul_1211: "f32[800]" = torch.ops.aten.mul.Tensor(sum_99, 0.0006377551020408163)
    mul_1212: "f32[800]" = torch.ops.aten.mul.Tensor(squeeze_187, squeeze_187)
    mul_1213: "f32[800]" = torch.ops.aten.mul.Tensor(mul_1211, mul_1212);  mul_1211 = mul_1212 = None
    unsqueeze_1026: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_1213, 0);  mul_1213 = None
    unsqueeze_1027: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1026, 2);  unsqueeze_1026 = None
    unsqueeze_1028: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1027, 3);  unsqueeze_1027 = None
    mul_1214: "f32[800]" = torch.ops.aten.mul.Tensor(squeeze_187, primals_125);  primals_125 = None
    unsqueeze_1029: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_1214, 0);  mul_1214 = None
    unsqueeze_1030: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1029, 2);  unsqueeze_1029 = None
    unsqueeze_1031: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1030, 3);  unsqueeze_1030 = None
    mul_1215: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_303, unsqueeze_1028);  sub_303 = unsqueeze_1028 = None
    sub_305: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(where_48, mul_1215);  where_48 = mul_1215 = None
    sub_306: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(sub_305, unsqueeze_1025);  sub_305 = unsqueeze_1025 = None
    mul_1216: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_306, unsqueeze_1031);  sub_306 = unsqueeze_1031 = None
    mul_1217: "f32[800]" = torch.ops.aten.mul.Tensor(sum_99, squeeze_187);  sum_99 = squeeze_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_49 = torch.ops.aten.convolution_backward.default(mul_1216, relu_61, primals_284, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1216 = primals_284 = None
    getitem_371: "f32[8, 1600, 14, 14]" = convolution_backward_49[0]
    getitem_372: "f32[800, 1600, 1, 1]" = convolution_backward_49[1];  convolution_backward_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_49: "b8[8, 1600, 14, 14]" = torch.ops.aten.le.Scalar(relu_61, 0);  relu_61 = None
    where_49: "f32[8, 1600, 14, 14]" = torch.ops.aten.where.self(le_49, full_default, getitem_371);  le_49 = getitem_371 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_100: "f32[1600]" = torch.ops.aten.sum.dim_IntList(where_49, [0, 2, 3])
    sub_307: "f32[8, 1600, 14, 14]" = torch.ops.aten.sub.Tensor(cat_37, unsqueeze_1034);  cat_37 = unsqueeze_1034 = None
    mul_1218: "f32[8, 1600, 14, 14]" = torch.ops.aten.mul.Tensor(where_49, sub_307)
    sum_101: "f32[1600]" = torch.ops.aten.sum.dim_IntList(mul_1218, [0, 2, 3]);  mul_1218 = None
    mul_1219: "f32[1600]" = torch.ops.aten.mul.Tensor(sum_100, 0.0006377551020408163)
    unsqueeze_1035: "f32[1, 1600]" = torch.ops.aten.unsqueeze.default(mul_1219, 0);  mul_1219 = None
    unsqueeze_1036: "f32[1, 1600, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1035, 2);  unsqueeze_1035 = None
    unsqueeze_1037: "f32[1, 1600, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1036, 3);  unsqueeze_1036 = None
    mul_1220: "f32[1600]" = torch.ops.aten.mul.Tensor(sum_101, 0.0006377551020408163)
    mul_1221: "f32[1600]" = torch.ops.aten.mul.Tensor(squeeze_184, squeeze_184)
    mul_1222: "f32[1600]" = torch.ops.aten.mul.Tensor(mul_1220, mul_1221);  mul_1220 = mul_1221 = None
    unsqueeze_1038: "f32[1, 1600]" = torch.ops.aten.unsqueeze.default(mul_1222, 0);  mul_1222 = None
    unsqueeze_1039: "f32[1, 1600, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1038, 2);  unsqueeze_1038 = None
    unsqueeze_1040: "f32[1, 1600, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1039, 3);  unsqueeze_1039 = None
    mul_1223: "f32[1600]" = torch.ops.aten.mul.Tensor(squeeze_184, primals_123);  primals_123 = None
    unsqueeze_1041: "f32[1, 1600]" = torch.ops.aten.unsqueeze.default(mul_1223, 0);  mul_1223 = None
    unsqueeze_1042: "f32[1, 1600, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1041, 2);  unsqueeze_1041 = None
    unsqueeze_1043: "f32[1, 1600, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1042, 3);  unsqueeze_1042 = None
    mul_1224: "f32[8, 1600, 14, 14]" = torch.ops.aten.mul.Tensor(sub_307, unsqueeze_1040);  sub_307 = unsqueeze_1040 = None
    sub_309: "f32[8, 1600, 14, 14]" = torch.ops.aten.sub.Tensor(where_49, mul_1224);  where_49 = mul_1224 = None
    sub_310: "f32[8, 1600, 14, 14]" = torch.ops.aten.sub.Tensor(sub_309, unsqueeze_1037);  sub_309 = unsqueeze_1037 = None
    mul_1225: "f32[8, 1600, 14, 14]" = torch.ops.aten.mul.Tensor(sub_310, unsqueeze_1043);  sub_310 = unsqueeze_1043 = None
    mul_1226: "f32[1600]" = torch.ops.aten.mul.Tensor(sum_101, squeeze_184);  sum_101 = squeeze_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    slice_377: "f32[8, 1024, 14, 14]" = torch.ops.aten.slice.Tensor(mul_1225, 1, 0, 1024)
    slice_378: "f32[8, 576, 14, 14]" = torch.ops.aten.slice.Tensor(mul_1225, 1, 1024, 1600);  mul_1225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    add_636: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_633, slice_377);  add_633 = slice_377 = None
    add_637: "f32[8, 576, 14, 14]" = torch.ops.aten.add.Tensor(slice_375, slice_378);  slice_375 = slice_378 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    slice_379: "f32[8, 512, 14, 14]" = torch.ops.aten.slice.Tensor(add_637, 1, 0, 512)
    slice_380: "f32[8, 64, 14, 14]" = torch.ops.aten.slice.Tensor(add_637, 1, 512, 576);  add_637 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    slice_scatter_136: "f32[8, 64, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_43, slice_380, 3, 0, 9223372036854775807);  slice_380 = None
    slice_scatter_138: "f32[8, 1088, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_45, slice_scatter_136, 1, 1024, 9223372036854775807);  slice_scatter_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    slice_scatter_142: "f32[8, 1088, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_45, add_636, 1, 0, 1024)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    add_638: "f32[8, 1088, 14, 14]" = torch.ops.aten.add.Tensor(slice_scatter_138, slice_scatter_142);  slice_scatter_138 = slice_scatter_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_50 = torch.ops.aten.convolution_backward.default(add_638, relu_60, primals_283, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_638 = primals_283 = None
    getitem_374: "f32[8, 800, 14, 14]" = convolution_backward_50[0]
    getitem_375: "f32[1088, 800, 1, 1]" = convolution_backward_50[1];  convolution_backward_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_50: "b8[8, 800, 14, 14]" = torch.ops.aten.le.Scalar(relu_60, 0);  relu_60 = None
    where_50: "f32[8, 800, 14, 14]" = torch.ops.aten.where.self(le_50, full_default, getitem_374);  le_50 = getitem_374 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_102: "f32[800]" = torch.ops.aten.sum.dim_IntList(where_50, [0, 2, 3])
    sub_311: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_59, unsqueeze_1046);  convolution_59 = unsqueeze_1046 = None
    mul_1227: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(where_50, sub_311)
    sum_103: "f32[800]" = torch.ops.aten.sum.dim_IntList(mul_1227, [0, 2, 3]);  mul_1227 = None
    mul_1228: "f32[800]" = torch.ops.aten.mul.Tensor(sum_102, 0.0006377551020408163)
    unsqueeze_1047: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_1228, 0);  mul_1228 = None
    unsqueeze_1048: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1047, 2);  unsqueeze_1047 = None
    unsqueeze_1049: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1048, 3);  unsqueeze_1048 = None
    mul_1229: "f32[800]" = torch.ops.aten.mul.Tensor(sum_103, 0.0006377551020408163)
    mul_1230: "f32[800]" = torch.ops.aten.mul.Tensor(squeeze_181, squeeze_181)
    mul_1231: "f32[800]" = torch.ops.aten.mul.Tensor(mul_1229, mul_1230);  mul_1229 = mul_1230 = None
    unsqueeze_1050: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_1231, 0);  mul_1231 = None
    unsqueeze_1051: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1050, 2);  unsqueeze_1050 = None
    unsqueeze_1052: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1051, 3);  unsqueeze_1051 = None
    mul_1232: "f32[800]" = torch.ops.aten.mul.Tensor(squeeze_181, primals_121);  primals_121 = None
    unsqueeze_1053: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_1232, 0);  mul_1232 = None
    unsqueeze_1054: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1053, 2);  unsqueeze_1053 = None
    unsqueeze_1055: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1054, 3);  unsqueeze_1054 = None
    mul_1233: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_311, unsqueeze_1052);  sub_311 = unsqueeze_1052 = None
    sub_313: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(where_50, mul_1233);  where_50 = mul_1233 = None
    sub_314: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(sub_313, unsqueeze_1049);  sub_313 = unsqueeze_1049 = None
    mul_1234: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_314, unsqueeze_1055);  sub_314 = unsqueeze_1055 = None
    mul_1235: "f32[800]" = torch.ops.aten.mul.Tensor(sum_103, squeeze_181);  sum_103 = squeeze_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_51 = torch.ops.aten.convolution_backward.default(mul_1234, relu_59, primals_282, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False]);  mul_1234 = primals_282 = None
    getitem_377: "f32[8, 800, 14, 14]" = convolution_backward_51[0]
    getitem_378: "f32[800, 16, 3, 3]" = convolution_backward_51[1];  convolution_backward_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_51: "b8[8, 800, 14, 14]" = torch.ops.aten.le.Scalar(relu_59, 0);  relu_59 = None
    where_51: "f32[8, 800, 14, 14]" = torch.ops.aten.where.self(le_51, full_default, getitem_377);  le_51 = getitem_377 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_104: "f32[800]" = torch.ops.aten.sum.dim_IntList(where_51, [0, 2, 3])
    sub_315: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_58, unsqueeze_1058);  convolution_58 = unsqueeze_1058 = None
    mul_1236: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(where_51, sub_315)
    sum_105: "f32[800]" = torch.ops.aten.sum.dim_IntList(mul_1236, [0, 2, 3]);  mul_1236 = None
    mul_1237: "f32[800]" = torch.ops.aten.mul.Tensor(sum_104, 0.0006377551020408163)
    unsqueeze_1059: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_1237, 0);  mul_1237 = None
    unsqueeze_1060: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1059, 2);  unsqueeze_1059 = None
    unsqueeze_1061: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1060, 3);  unsqueeze_1060 = None
    mul_1238: "f32[800]" = torch.ops.aten.mul.Tensor(sum_105, 0.0006377551020408163)
    mul_1239: "f32[800]" = torch.ops.aten.mul.Tensor(squeeze_178, squeeze_178)
    mul_1240: "f32[800]" = torch.ops.aten.mul.Tensor(mul_1238, mul_1239);  mul_1238 = mul_1239 = None
    unsqueeze_1062: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_1240, 0);  mul_1240 = None
    unsqueeze_1063: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1062, 2);  unsqueeze_1062 = None
    unsqueeze_1064: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1063, 3);  unsqueeze_1063 = None
    mul_1241: "f32[800]" = torch.ops.aten.mul.Tensor(squeeze_178, primals_119);  primals_119 = None
    unsqueeze_1065: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_1241, 0);  mul_1241 = None
    unsqueeze_1066: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1065, 2);  unsqueeze_1065 = None
    unsqueeze_1067: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1066, 3);  unsqueeze_1066 = None
    mul_1242: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_315, unsqueeze_1064);  sub_315 = unsqueeze_1064 = None
    sub_317: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(where_51, mul_1242);  where_51 = mul_1242 = None
    sub_318: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(sub_317, unsqueeze_1061);  sub_317 = unsqueeze_1061 = None
    mul_1243: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_318, unsqueeze_1067);  sub_318 = unsqueeze_1067 = None
    mul_1244: "f32[800]" = torch.ops.aten.mul.Tensor(sum_105, squeeze_178);  sum_105 = squeeze_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_52 = torch.ops.aten.convolution_backward.default(mul_1243, relu_58, primals_281, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1243 = primals_281 = None
    getitem_380: "f32[8, 1536, 14, 14]" = convolution_backward_52[0]
    getitem_381: "f32[800, 1536, 1, 1]" = convolution_backward_52[1];  convolution_backward_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_52: "b8[8, 1536, 14, 14]" = torch.ops.aten.le.Scalar(relu_58, 0);  relu_58 = None
    where_52: "f32[8, 1536, 14, 14]" = torch.ops.aten.where.self(le_52, full_default, getitem_380);  le_52 = getitem_380 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_106: "f32[1536]" = torch.ops.aten.sum.dim_IntList(where_52, [0, 2, 3])
    sub_319: "f32[8, 1536, 14, 14]" = torch.ops.aten.sub.Tensor(cat_35, unsqueeze_1070);  cat_35 = unsqueeze_1070 = None
    mul_1245: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(where_52, sub_319)
    sum_107: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_1245, [0, 2, 3]);  mul_1245 = None
    mul_1246: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_106, 0.0006377551020408163)
    unsqueeze_1071: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_1246, 0);  mul_1246 = None
    unsqueeze_1072: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1071, 2);  unsqueeze_1071 = None
    unsqueeze_1073: "f32[1, 1536, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1072, 3);  unsqueeze_1072 = None
    mul_1247: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_107, 0.0006377551020408163)
    mul_1248: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_175, squeeze_175)
    mul_1249: "f32[1536]" = torch.ops.aten.mul.Tensor(mul_1247, mul_1248);  mul_1247 = mul_1248 = None
    unsqueeze_1074: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_1249, 0);  mul_1249 = None
    unsqueeze_1075: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1074, 2);  unsqueeze_1074 = None
    unsqueeze_1076: "f32[1, 1536, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1075, 3);  unsqueeze_1075 = None
    mul_1250: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_175, primals_117);  primals_117 = None
    unsqueeze_1077: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_1250, 0);  mul_1250 = None
    unsqueeze_1078: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1077, 2);  unsqueeze_1077 = None
    unsqueeze_1079: "f32[1, 1536, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1078, 3);  unsqueeze_1078 = None
    mul_1251: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(sub_319, unsqueeze_1076);  sub_319 = unsqueeze_1076 = None
    sub_321: "f32[8, 1536, 14, 14]" = torch.ops.aten.sub.Tensor(where_52, mul_1251);  where_52 = mul_1251 = None
    sub_322: "f32[8, 1536, 14, 14]" = torch.ops.aten.sub.Tensor(sub_321, unsqueeze_1073);  sub_321 = unsqueeze_1073 = None
    mul_1252: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(sub_322, unsqueeze_1079);  sub_322 = unsqueeze_1079 = None
    mul_1253: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_107, squeeze_175);  sum_107 = squeeze_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    slice_381: "f32[8, 1024, 14, 14]" = torch.ops.aten.slice.Tensor(mul_1252, 1, 0, 1024)
    slice_382: "f32[8, 512, 14, 14]" = torch.ops.aten.slice.Tensor(mul_1252, 1, 1024, 1536);  mul_1252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    add_639: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_636, slice_381);  add_636 = slice_381 = None
    add_640: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(slice_379, slice_382);  slice_379 = slice_382 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    slice_383: "f32[8, 448, 14, 14]" = torch.ops.aten.slice.Tensor(add_640, 1, 0, 448)
    slice_384: "f32[8, 64, 14, 14]" = torch.ops.aten.slice.Tensor(add_640, 1, 448, 512);  add_640 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    slice_scatter_144: "f32[8, 64, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_43, slice_384, 3, 0, 9223372036854775807);  slice_384 = None
    slice_scatter_146: "f32[8, 1088, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_45, slice_scatter_144, 1, 1024, 9223372036854775807);  slice_scatter_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    slice_scatter_150: "f32[8, 1088, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_45, add_639, 1, 0, 1024)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    add_641: "f32[8, 1088, 14, 14]" = torch.ops.aten.add.Tensor(slice_scatter_146, slice_scatter_150);  slice_scatter_146 = slice_scatter_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_53 = torch.ops.aten.convolution_backward.default(add_641, relu_57, primals_280, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_641 = primals_280 = None
    getitem_383: "f32[8, 800, 14, 14]" = convolution_backward_53[0]
    getitem_384: "f32[1088, 800, 1, 1]" = convolution_backward_53[1];  convolution_backward_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_53: "b8[8, 800, 14, 14]" = torch.ops.aten.le.Scalar(relu_57, 0);  relu_57 = None
    where_53: "f32[8, 800, 14, 14]" = torch.ops.aten.where.self(le_53, full_default, getitem_383);  le_53 = getitem_383 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_108: "f32[800]" = torch.ops.aten.sum.dim_IntList(where_53, [0, 2, 3])
    sub_323: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_56, unsqueeze_1082);  convolution_56 = unsqueeze_1082 = None
    mul_1254: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(where_53, sub_323)
    sum_109: "f32[800]" = torch.ops.aten.sum.dim_IntList(mul_1254, [0, 2, 3]);  mul_1254 = None
    mul_1255: "f32[800]" = torch.ops.aten.mul.Tensor(sum_108, 0.0006377551020408163)
    unsqueeze_1083: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_1255, 0);  mul_1255 = None
    unsqueeze_1084: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1083, 2);  unsqueeze_1083 = None
    unsqueeze_1085: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1084, 3);  unsqueeze_1084 = None
    mul_1256: "f32[800]" = torch.ops.aten.mul.Tensor(sum_109, 0.0006377551020408163)
    mul_1257: "f32[800]" = torch.ops.aten.mul.Tensor(squeeze_172, squeeze_172)
    mul_1258: "f32[800]" = torch.ops.aten.mul.Tensor(mul_1256, mul_1257);  mul_1256 = mul_1257 = None
    unsqueeze_1086: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_1258, 0);  mul_1258 = None
    unsqueeze_1087: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1086, 2);  unsqueeze_1086 = None
    unsqueeze_1088: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1087, 3);  unsqueeze_1087 = None
    mul_1259: "f32[800]" = torch.ops.aten.mul.Tensor(squeeze_172, primals_115);  primals_115 = None
    unsqueeze_1089: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_1259, 0);  mul_1259 = None
    unsqueeze_1090: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1089, 2);  unsqueeze_1089 = None
    unsqueeze_1091: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1090, 3);  unsqueeze_1090 = None
    mul_1260: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_323, unsqueeze_1088);  sub_323 = unsqueeze_1088 = None
    sub_325: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(where_53, mul_1260);  where_53 = mul_1260 = None
    sub_326: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(sub_325, unsqueeze_1085);  sub_325 = unsqueeze_1085 = None
    mul_1261: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_326, unsqueeze_1091);  sub_326 = unsqueeze_1091 = None
    mul_1262: "f32[800]" = torch.ops.aten.mul.Tensor(sum_109, squeeze_172);  sum_109 = squeeze_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_54 = torch.ops.aten.convolution_backward.default(mul_1261, relu_56, primals_279, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False]);  mul_1261 = primals_279 = None
    getitem_386: "f32[8, 800, 14, 14]" = convolution_backward_54[0]
    getitem_387: "f32[800, 16, 3, 3]" = convolution_backward_54[1];  convolution_backward_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_54: "b8[8, 800, 14, 14]" = torch.ops.aten.le.Scalar(relu_56, 0);  relu_56 = None
    where_54: "f32[8, 800, 14, 14]" = torch.ops.aten.where.self(le_54, full_default, getitem_386);  le_54 = getitem_386 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_110: "f32[800]" = torch.ops.aten.sum.dim_IntList(where_54, [0, 2, 3])
    sub_327: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_55, unsqueeze_1094);  convolution_55 = unsqueeze_1094 = None
    mul_1263: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(where_54, sub_327)
    sum_111: "f32[800]" = torch.ops.aten.sum.dim_IntList(mul_1263, [0, 2, 3]);  mul_1263 = None
    mul_1264: "f32[800]" = torch.ops.aten.mul.Tensor(sum_110, 0.0006377551020408163)
    unsqueeze_1095: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_1264, 0);  mul_1264 = None
    unsqueeze_1096: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1095, 2);  unsqueeze_1095 = None
    unsqueeze_1097: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1096, 3);  unsqueeze_1096 = None
    mul_1265: "f32[800]" = torch.ops.aten.mul.Tensor(sum_111, 0.0006377551020408163)
    mul_1266: "f32[800]" = torch.ops.aten.mul.Tensor(squeeze_169, squeeze_169)
    mul_1267: "f32[800]" = torch.ops.aten.mul.Tensor(mul_1265, mul_1266);  mul_1265 = mul_1266 = None
    unsqueeze_1098: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_1267, 0);  mul_1267 = None
    unsqueeze_1099: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1098, 2);  unsqueeze_1098 = None
    unsqueeze_1100: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1099, 3);  unsqueeze_1099 = None
    mul_1268: "f32[800]" = torch.ops.aten.mul.Tensor(squeeze_169, primals_113);  primals_113 = None
    unsqueeze_1101: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_1268, 0);  mul_1268 = None
    unsqueeze_1102: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1101, 2);  unsqueeze_1101 = None
    unsqueeze_1103: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1102, 3);  unsqueeze_1102 = None
    mul_1269: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_327, unsqueeze_1100);  sub_327 = unsqueeze_1100 = None
    sub_329: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(where_54, mul_1269);  where_54 = mul_1269 = None
    sub_330: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(sub_329, unsqueeze_1097);  sub_329 = unsqueeze_1097 = None
    mul_1270: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_330, unsqueeze_1103);  sub_330 = unsqueeze_1103 = None
    mul_1271: "f32[800]" = torch.ops.aten.mul.Tensor(sum_111, squeeze_169);  sum_111 = squeeze_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_55 = torch.ops.aten.convolution_backward.default(mul_1270, relu_55, primals_278, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1270 = primals_278 = None
    getitem_389: "f32[8, 1472, 14, 14]" = convolution_backward_55[0]
    getitem_390: "f32[800, 1472, 1, 1]" = convolution_backward_55[1];  convolution_backward_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_55: "b8[8, 1472, 14, 14]" = torch.ops.aten.le.Scalar(relu_55, 0);  relu_55 = None
    where_55: "f32[8, 1472, 14, 14]" = torch.ops.aten.where.self(le_55, full_default, getitem_389);  le_55 = getitem_389 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_112: "f32[1472]" = torch.ops.aten.sum.dim_IntList(where_55, [0, 2, 3])
    sub_331: "f32[8, 1472, 14, 14]" = torch.ops.aten.sub.Tensor(cat_33, unsqueeze_1106);  cat_33 = unsqueeze_1106 = None
    mul_1272: "f32[8, 1472, 14, 14]" = torch.ops.aten.mul.Tensor(where_55, sub_331)
    sum_113: "f32[1472]" = torch.ops.aten.sum.dim_IntList(mul_1272, [0, 2, 3]);  mul_1272 = None
    mul_1273: "f32[1472]" = torch.ops.aten.mul.Tensor(sum_112, 0.0006377551020408163)
    unsqueeze_1107: "f32[1, 1472]" = torch.ops.aten.unsqueeze.default(mul_1273, 0);  mul_1273 = None
    unsqueeze_1108: "f32[1, 1472, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1107, 2);  unsqueeze_1107 = None
    unsqueeze_1109: "f32[1, 1472, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1108, 3);  unsqueeze_1108 = None
    mul_1274: "f32[1472]" = torch.ops.aten.mul.Tensor(sum_113, 0.0006377551020408163)
    mul_1275: "f32[1472]" = torch.ops.aten.mul.Tensor(squeeze_166, squeeze_166)
    mul_1276: "f32[1472]" = torch.ops.aten.mul.Tensor(mul_1274, mul_1275);  mul_1274 = mul_1275 = None
    unsqueeze_1110: "f32[1, 1472]" = torch.ops.aten.unsqueeze.default(mul_1276, 0);  mul_1276 = None
    unsqueeze_1111: "f32[1, 1472, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1110, 2);  unsqueeze_1110 = None
    unsqueeze_1112: "f32[1, 1472, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1111, 3);  unsqueeze_1111 = None
    mul_1277: "f32[1472]" = torch.ops.aten.mul.Tensor(squeeze_166, primals_111);  primals_111 = None
    unsqueeze_1113: "f32[1, 1472]" = torch.ops.aten.unsqueeze.default(mul_1277, 0);  mul_1277 = None
    unsqueeze_1114: "f32[1, 1472, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1113, 2);  unsqueeze_1113 = None
    unsqueeze_1115: "f32[1, 1472, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1114, 3);  unsqueeze_1114 = None
    mul_1278: "f32[8, 1472, 14, 14]" = torch.ops.aten.mul.Tensor(sub_331, unsqueeze_1112);  sub_331 = unsqueeze_1112 = None
    sub_333: "f32[8, 1472, 14, 14]" = torch.ops.aten.sub.Tensor(where_55, mul_1278);  where_55 = mul_1278 = None
    sub_334: "f32[8, 1472, 14, 14]" = torch.ops.aten.sub.Tensor(sub_333, unsqueeze_1109);  sub_333 = unsqueeze_1109 = None
    mul_1279: "f32[8, 1472, 14, 14]" = torch.ops.aten.mul.Tensor(sub_334, unsqueeze_1115);  sub_334 = unsqueeze_1115 = None
    mul_1280: "f32[1472]" = torch.ops.aten.mul.Tensor(sum_113, squeeze_166);  sum_113 = squeeze_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    slice_385: "f32[8, 1024, 14, 14]" = torch.ops.aten.slice.Tensor(mul_1279, 1, 0, 1024)
    slice_386: "f32[8, 448, 14, 14]" = torch.ops.aten.slice.Tensor(mul_1279, 1, 1024, 1472);  mul_1279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    add_642: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_639, slice_385);  add_639 = slice_385 = None
    add_643: "f32[8, 448, 14, 14]" = torch.ops.aten.add.Tensor(slice_383, slice_386);  slice_383 = slice_386 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    slice_387: "f32[8, 384, 14, 14]" = torch.ops.aten.slice.Tensor(add_643, 1, 0, 384)
    slice_388: "f32[8, 64, 14, 14]" = torch.ops.aten.slice.Tensor(add_643, 1, 384, 448);  add_643 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    slice_scatter_152: "f32[8, 64, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_43, slice_388, 3, 0, 9223372036854775807);  slice_388 = None
    slice_scatter_154: "f32[8, 1088, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_45, slice_scatter_152, 1, 1024, 9223372036854775807);  slice_scatter_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    slice_scatter_158: "f32[8, 1088, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_45, add_642, 1, 0, 1024)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    add_644: "f32[8, 1088, 14, 14]" = torch.ops.aten.add.Tensor(slice_scatter_154, slice_scatter_158);  slice_scatter_154 = slice_scatter_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_56 = torch.ops.aten.convolution_backward.default(add_644, relu_54, primals_277, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_644 = primals_277 = None
    getitem_392: "f32[8, 800, 14, 14]" = convolution_backward_56[0]
    getitem_393: "f32[1088, 800, 1, 1]" = convolution_backward_56[1];  convolution_backward_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_56: "b8[8, 800, 14, 14]" = torch.ops.aten.le.Scalar(relu_54, 0);  relu_54 = None
    where_56: "f32[8, 800, 14, 14]" = torch.ops.aten.where.self(le_56, full_default, getitem_392);  le_56 = getitem_392 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_114: "f32[800]" = torch.ops.aten.sum.dim_IntList(where_56, [0, 2, 3])
    sub_335: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_53, unsqueeze_1118);  convolution_53 = unsqueeze_1118 = None
    mul_1281: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(where_56, sub_335)
    sum_115: "f32[800]" = torch.ops.aten.sum.dim_IntList(mul_1281, [0, 2, 3]);  mul_1281 = None
    mul_1282: "f32[800]" = torch.ops.aten.mul.Tensor(sum_114, 0.0006377551020408163)
    unsqueeze_1119: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_1282, 0);  mul_1282 = None
    unsqueeze_1120: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1119, 2);  unsqueeze_1119 = None
    unsqueeze_1121: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1120, 3);  unsqueeze_1120 = None
    mul_1283: "f32[800]" = torch.ops.aten.mul.Tensor(sum_115, 0.0006377551020408163)
    mul_1284: "f32[800]" = torch.ops.aten.mul.Tensor(squeeze_163, squeeze_163)
    mul_1285: "f32[800]" = torch.ops.aten.mul.Tensor(mul_1283, mul_1284);  mul_1283 = mul_1284 = None
    unsqueeze_1122: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_1285, 0);  mul_1285 = None
    unsqueeze_1123: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1122, 2);  unsqueeze_1122 = None
    unsqueeze_1124: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1123, 3);  unsqueeze_1123 = None
    mul_1286: "f32[800]" = torch.ops.aten.mul.Tensor(squeeze_163, primals_109);  primals_109 = None
    unsqueeze_1125: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_1286, 0);  mul_1286 = None
    unsqueeze_1126: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1125, 2);  unsqueeze_1125 = None
    unsqueeze_1127: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1126, 3);  unsqueeze_1126 = None
    mul_1287: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_335, unsqueeze_1124);  sub_335 = unsqueeze_1124 = None
    sub_337: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(where_56, mul_1287);  where_56 = mul_1287 = None
    sub_338: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(sub_337, unsqueeze_1121);  sub_337 = unsqueeze_1121 = None
    mul_1288: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_338, unsqueeze_1127);  sub_338 = unsqueeze_1127 = None
    mul_1289: "f32[800]" = torch.ops.aten.mul.Tensor(sum_115, squeeze_163);  sum_115 = squeeze_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_57 = torch.ops.aten.convolution_backward.default(mul_1288, relu_53, primals_276, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False]);  mul_1288 = primals_276 = None
    getitem_395: "f32[8, 800, 14, 14]" = convolution_backward_57[0]
    getitem_396: "f32[800, 16, 3, 3]" = convolution_backward_57[1];  convolution_backward_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_57: "b8[8, 800, 14, 14]" = torch.ops.aten.le.Scalar(relu_53, 0);  relu_53 = None
    where_57: "f32[8, 800, 14, 14]" = torch.ops.aten.where.self(le_57, full_default, getitem_395);  le_57 = getitem_395 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_116: "f32[800]" = torch.ops.aten.sum.dim_IntList(where_57, [0, 2, 3])
    sub_339: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_52, unsqueeze_1130);  convolution_52 = unsqueeze_1130 = None
    mul_1290: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(where_57, sub_339)
    sum_117: "f32[800]" = torch.ops.aten.sum.dim_IntList(mul_1290, [0, 2, 3]);  mul_1290 = None
    mul_1291: "f32[800]" = torch.ops.aten.mul.Tensor(sum_116, 0.0006377551020408163)
    unsqueeze_1131: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_1291, 0);  mul_1291 = None
    unsqueeze_1132: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1131, 2);  unsqueeze_1131 = None
    unsqueeze_1133: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1132, 3);  unsqueeze_1132 = None
    mul_1292: "f32[800]" = torch.ops.aten.mul.Tensor(sum_117, 0.0006377551020408163)
    mul_1293: "f32[800]" = torch.ops.aten.mul.Tensor(squeeze_160, squeeze_160)
    mul_1294: "f32[800]" = torch.ops.aten.mul.Tensor(mul_1292, mul_1293);  mul_1292 = mul_1293 = None
    unsqueeze_1134: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_1294, 0);  mul_1294 = None
    unsqueeze_1135: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1134, 2);  unsqueeze_1134 = None
    unsqueeze_1136: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1135, 3);  unsqueeze_1135 = None
    mul_1295: "f32[800]" = torch.ops.aten.mul.Tensor(squeeze_160, primals_107);  primals_107 = None
    unsqueeze_1137: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_1295, 0);  mul_1295 = None
    unsqueeze_1138: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1137, 2);  unsqueeze_1137 = None
    unsqueeze_1139: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1138, 3);  unsqueeze_1138 = None
    mul_1296: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_339, unsqueeze_1136);  sub_339 = unsqueeze_1136 = None
    sub_341: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(where_57, mul_1296);  where_57 = mul_1296 = None
    sub_342: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(sub_341, unsqueeze_1133);  sub_341 = unsqueeze_1133 = None
    mul_1297: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_342, unsqueeze_1139);  sub_342 = unsqueeze_1139 = None
    mul_1298: "f32[800]" = torch.ops.aten.mul.Tensor(sum_117, squeeze_160);  sum_117 = squeeze_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_58 = torch.ops.aten.convolution_backward.default(mul_1297, relu_52, primals_275, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1297 = primals_275 = None
    getitem_398: "f32[8, 1408, 14, 14]" = convolution_backward_58[0]
    getitem_399: "f32[800, 1408, 1, 1]" = convolution_backward_58[1];  convolution_backward_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_58: "b8[8, 1408, 14, 14]" = torch.ops.aten.le.Scalar(relu_52, 0);  relu_52 = None
    where_58: "f32[8, 1408, 14, 14]" = torch.ops.aten.where.self(le_58, full_default, getitem_398);  le_58 = getitem_398 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_118: "f32[1408]" = torch.ops.aten.sum.dim_IntList(where_58, [0, 2, 3])
    sub_343: "f32[8, 1408, 14, 14]" = torch.ops.aten.sub.Tensor(cat_31, unsqueeze_1142);  cat_31 = unsqueeze_1142 = None
    mul_1299: "f32[8, 1408, 14, 14]" = torch.ops.aten.mul.Tensor(where_58, sub_343)
    sum_119: "f32[1408]" = torch.ops.aten.sum.dim_IntList(mul_1299, [0, 2, 3]);  mul_1299 = None
    mul_1300: "f32[1408]" = torch.ops.aten.mul.Tensor(sum_118, 0.0006377551020408163)
    unsqueeze_1143: "f32[1, 1408]" = torch.ops.aten.unsqueeze.default(mul_1300, 0);  mul_1300 = None
    unsqueeze_1144: "f32[1, 1408, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1143, 2);  unsqueeze_1143 = None
    unsqueeze_1145: "f32[1, 1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1144, 3);  unsqueeze_1144 = None
    mul_1301: "f32[1408]" = torch.ops.aten.mul.Tensor(sum_119, 0.0006377551020408163)
    mul_1302: "f32[1408]" = torch.ops.aten.mul.Tensor(squeeze_157, squeeze_157)
    mul_1303: "f32[1408]" = torch.ops.aten.mul.Tensor(mul_1301, mul_1302);  mul_1301 = mul_1302 = None
    unsqueeze_1146: "f32[1, 1408]" = torch.ops.aten.unsqueeze.default(mul_1303, 0);  mul_1303 = None
    unsqueeze_1147: "f32[1, 1408, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1146, 2);  unsqueeze_1146 = None
    unsqueeze_1148: "f32[1, 1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1147, 3);  unsqueeze_1147 = None
    mul_1304: "f32[1408]" = torch.ops.aten.mul.Tensor(squeeze_157, primals_105);  primals_105 = None
    unsqueeze_1149: "f32[1, 1408]" = torch.ops.aten.unsqueeze.default(mul_1304, 0);  mul_1304 = None
    unsqueeze_1150: "f32[1, 1408, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1149, 2);  unsqueeze_1149 = None
    unsqueeze_1151: "f32[1, 1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1150, 3);  unsqueeze_1150 = None
    mul_1305: "f32[8, 1408, 14, 14]" = torch.ops.aten.mul.Tensor(sub_343, unsqueeze_1148);  sub_343 = unsqueeze_1148 = None
    sub_345: "f32[8, 1408, 14, 14]" = torch.ops.aten.sub.Tensor(where_58, mul_1305);  where_58 = mul_1305 = None
    sub_346: "f32[8, 1408, 14, 14]" = torch.ops.aten.sub.Tensor(sub_345, unsqueeze_1145);  sub_345 = unsqueeze_1145 = None
    mul_1306: "f32[8, 1408, 14, 14]" = torch.ops.aten.mul.Tensor(sub_346, unsqueeze_1151);  sub_346 = unsqueeze_1151 = None
    mul_1307: "f32[1408]" = torch.ops.aten.mul.Tensor(sum_119, squeeze_157);  sum_119 = squeeze_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    slice_389: "f32[8, 1024, 14, 14]" = torch.ops.aten.slice.Tensor(mul_1306, 1, 0, 1024)
    slice_390: "f32[8, 384, 14, 14]" = torch.ops.aten.slice.Tensor(mul_1306, 1, 1024, 1408);  mul_1306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    add_645: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_642, slice_389);  add_642 = slice_389 = None
    add_646: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(slice_387, slice_390);  slice_387 = slice_390 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    slice_391: "f32[8, 320, 14, 14]" = torch.ops.aten.slice.Tensor(add_646, 1, 0, 320)
    slice_392: "f32[8, 64, 14, 14]" = torch.ops.aten.slice.Tensor(add_646, 1, 320, 384);  add_646 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    slice_scatter_160: "f32[8, 64, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_43, slice_392, 3, 0, 9223372036854775807);  slice_392 = None
    slice_scatter_162: "f32[8, 1088, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_45, slice_scatter_160, 1, 1024, 9223372036854775807);  slice_scatter_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    slice_scatter_166: "f32[8, 1088, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_45, add_645, 1, 0, 1024)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    add_647: "f32[8, 1088, 14, 14]" = torch.ops.aten.add.Tensor(slice_scatter_162, slice_scatter_166);  slice_scatter_162 = slice_scatter_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_59 = torch.ops.aten.convolution_backward.default(add_647, relu_51, primals_274, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_647 = primals_274 = None
    getitem_401: "f32[8, 800, 14, 14]" = convolution_backward_59[0]
    getitem_402: "f32[1088, 800, 1, 1]" = convolution_backward_59[1];  convolution_backward_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_59: "b8[8, 800, 14, 14]" = torch.ops.aten.le.Scalar(relu_51, 0);  relu_51 = None
    where_59: "f32[8, 800, 14, 14]" = torch.ops.aten.where.self(le_59, full_default, getitem_401);  le_59 = getitem_401 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_120: "f32[800]" = torch.ops.aten.sum.dim_IntList(where_59, [0, 2, 3])
    sub_347: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_1154);  convolution_50 = unsqueeze_1154 = None
    mul_1308: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(where_59, sub_347)
    sum_121: "f32[800]" = torch.ops.aten.sum.dim_IntList(mul_1308, [0, 2, 3]);  mul_1308 = None
    mul_1309: "f32[800]" = torch.ops.aten.mul.Tensor(sum_120, 0.0006377551020408163)
    unsqueeze_1155: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_1309, 0);  mul_1309 = None
    unsqueeze_1156: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1155, 2);  unsqueeze_1155 = None
    unsqueeze_1157: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1156, 3);  unsqueeze_1156 = None
    mul_1310: "f32[800]" = torch.ops.aten.mul.Tensor(sum_121, 0.0006377551020408163)
    mul_1311: "f32[800]" = torch.ops.aten.mul.Tensor(squeeze_154, squeeze_154)
    mul_1312: "f32[800]" = torch.ops.aten.mul.Tensor(mul_1310, mul_1311);  mul_1310 = mul_1311 = None
    unsqueeze_1158: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_1312, 0);  mul_1312 = None
    unsqueeze_1159: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1158, 2);  unsqueeze_1158 = None
    unsqueeze_1160: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1159, 3);  unsqueeze_1159 = None
    mul_1313: "f32[800]" = torch.ops.aten.mul.Tensor(squeeze_154, primals_103);  primals_103 = None
    unsqueeze_1161: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_1313, 0);  mul_1313 = None
    unsqueeze_1162: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1161, 2);  unsqueeze_1161 = None
    unsqueeze_1163: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1162, 3);  unsqueeze_1162 = None
    mul_1314: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_347, unsqueeze_1160);  sub_347 = unsqueeze_1160 = None
    sub_349: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(where_59, mul_1314);  where_59 = mul_1314 = None
    sub_350: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(sub_349, unsqueeze_1157);  sub_349 = unsqueeze_1157 = None
    mul_1315: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_350, unsqueeze_1163);  sub_350 = unsqueeze_1163 = None
    mul_1316: "f32[800]" = torch.ops.aten.mul.Tensor(sum_121, squeeze_154);  sum_121 = squeeze_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_60 = torch.ops.aten.convolution_backward.default(mul_1315, relu_50, primals_273, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False]);  mul_1315 = primals_273 = None
    getitem_404: "f32[8, 800, 14, 14]" = convolution_backward_60[0]
    getitem_405: "f32[800, 16, 3, 3]" = convolution_backward_60[1];  convolution_backward_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_60: "b8[8, 800, 14, 14]" = torch.ops.aten.le.Scalar(relu_50, 0);  relu_50 = None
    where_60: "f32[8, 800, 14, 14]" = torch.ops.aten.where.self(le_60, full_default, getitem_404);  le_60 = getitem_404 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_122: "f32[800]" = torch.ops.aten.sum.dim_IntList(where_60, [0, 2, 3])
    sub_351: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_49, unsqueeze_1166);  convolution_49 = unsqueeze_1166 = None
    mul_1317: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(where_60, sub_351)
    sum_123: "f32[800]" = torch.ops.aten.sum.dim_IntList(mul_1317, [0, 2, 3]);  mul_1317 = None
    mul_1318: "f32[800]" = torch.ops.aten.mul.Tensor(sum_122, 0.0006377551020408163)
    unsqueeze_1167: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_1318, 0);  mul_1318 = None
    unsqueeze_1168: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1167, 2);  unsqueeze_1167 = None
    unsqueeze_1169: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1168, 3);  unsqueeze_1168 = None
    mul_1319: "f32[800]" = torch.ops.aten.mul.Tensor(sum_123, 0.0006377551020408163)
    mul_1320: "f32[800]" = torch.ops.aten.mul.Tensor(squeeze_151, squeeze_151)
    mul_1321: "f32[800]" = torch.ops.aten.mul.Tensor(mul_1319, mul_1320);  mul_1319 = mul_1320 = None
    unsqueeze_1170: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_1321, 0);  mul_1321 = None
    unsqueeze_1171: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1170, 2);  unsqueeze_1170 = None
    unsqueeze_1172: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1171, 3);  unsqueeze_1171 = None
    mul_1322: "f32[800]" = torch.ops.aten.mul.Tensor(squeeze_151, primals_101);  primals_101 = None
    unsqueeze_1173: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_1322, 0);  mul_1322 = None
    unsqueeze_1174: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1173, 2);  unsqueeze_1173 = None
    unsqueeze_1175: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1174, 3);  unsqueeze_1174 = None
    mul_1323: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_351, unsqueeze_1172);  sub_351 = unsqueeze_1172 = None
    sub_353: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(where_60, mul_1323);  where_60 = mul_1323 = None
    sub_354: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(sub_353, unsqueeze_1169);  sub_353 = unsqueeze_1169 = None
    mul_1324: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_354, unsqueeze_1175);  sub_354 = unsqueeze_1175 = None
    mul_1325: "f32[800]" = torch.ops.aten.mul.Tensor(sum_123, squeeze_151);  sum_123 = squeeze_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_61 = torch.ops.aten.convolution_backward.default(mul_1324, relu_49, primals_272, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1324 = primals_272 = None
    getitem_407: "f32[8, 1344, 14, 14]" = convolution_backward_61[0]
    getitem_408: "f32[800, 1344, 1, 1]" = convolution_backward_61[1];  convolution_backward_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_61: "b8[8, 1344, 14, 14]" = torch.ops.aten.le.Scalar(relu_49, 0);  relu_49 = None
    where_61: "f32[8, 1344, 14, 14]" = torch.ops.aten.where.self(le_61, full_default, getitem_407);  le_61 = getitem_407 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_124: "f32[1344]" = torch.ops.aten.sum.dim_IntList(where_61, [0, 2, 3])
    sub_355: "f32[8, 1344, 14, 14]" = torch.ops.aten.sub.Tensor(cat_29, unsqueeze_1178);  cat_29 = unsqueeze_1178 = None
    mul_1326: "f32[8, 1344, 14, 14]" = torch.ops.aten.mul.Tensor(where_61, sub_355)
    sum_125: "f32[1344]" = torch.ops.aten.sum.dim_IntList(mul_1326, [0, 2, 3]);  mul_1326 = None
    mul_1327: "f32[1344]" = torch.ops.aten.mul.Tensor(sum_124, 0.0006377551020408163)
    unsqueeze_1179: "f32[1, 1344]" = torch.ops.aten.unsqueeze.default(mul_1327, 0);  mul_1327 = None
    unsqueeze_1180: "f32[1, 1344, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1179, 2);  unsqueeze_1179 = None
    unsqueeze_1181: "f32[1, 1344, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1180, 3);  unsqueeze_1180 = None
    mul_1328: "f32[1344]" = torch.ops.aten.mul.Tensor(sum_125, 0.0006377551020408163)
    mul_1329: "f32[1344]" = torch.ops.aten.mul.Tensor(squeeze_148, squeeze_148)
    mul_1330: "f32[1344]" = torch.ops.aten.mul.Tensor(mul_1328, mul_1329);  mul_1328 = mul_1329 = None
    unsqueeze_1182: "f32[1, 1344]" = torch.ops.aten.unsqueeze.default(mul_1330, 0);  mul_1330 = None
    unsqueeze_1183: "f32[1, 1344, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1182, 2);  unsqueeze_1182 = None
    unsqueeze_1184: "f32[1, 1344, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1183, 3);  unsqueeze_1183 = None
    mul_1331: "f32[1344]" = torch.ops.aten.mul.Tensor(squeeze_148, primals_99);  primals_99 = None
    unsqueeze_1185: "f32[1, 1344]" = torch.ops.aten.unsqueeze.default(mul_1331, 0);  mul_1331 = None
    unsqueeze_1186: "f32[1, 1344, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1185, 2);  unsqueeze_1185 = None
    unsqueeze_1187: "f32[1, 1344, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1186, 3);  unsqueeze_1186 = None
    mul_1332: "f32[8, 1344, 14, 14]" = torch.ops.aten.mul.Tensor(sub_355, unsqueeze_1184);  sub_355 = unsqueeze_1184 = None
    sub_357: "f32[8, 1344, 14, 14]" = torch.ops.aten.sub.Tensor(where_61, mul_1332);  where_61 = mul_1332 = None
    sub_358: "f32[8, 1344, 14, 14]" = torch.ops.aten.sub.Tensor(sub_357, unsqueeze_1181);  sub_357 = unsqueeze_1181 = None
    mul_1333: "f32[8, 1344, 14, 14]" = torch.ops.aten.mul.Tensor(sub_358, unsqueeze_1187);  sub_358 = unsqueeze_1187 = None
    mul_1334: "f32[1344]" = torch.ops.aten.mul.Tensor(sum_125, squeeze_148);  sum_125 = squeeze_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    slice_393: "f32[8, 1024, 14, 14]" = torch.ops.aten.slice.Tensor(mul_1333, 1, 0, 1024)
    slice_394: "f32[8, 320, 14, 14]" = torch.ops.aten.slice.Tensor(mul_1333, 1, 1024, 1344);  mul_1333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    add_648: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_645, slice_393);  add_645 = slice_393 = None
    add_649: "f32[8, 320, 14, 14]" = torch.ops.aten.add.Tensor(slice_391, slice_394);  slice_391 = slice_394 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    slice_395: "f32[8, 256, 14, 14]" = torch.ops.aten.slice.Tensor(add_649, 1, 0, 256)
    slice_396: "f32[8, 64, 14, 14]" = torch.ops.aten.slice.Tensor(add_649, 1, 256, 320);  add_649 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    slice_scatter_168: "f32[8, 64, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_43, slice_396, 3, 0, 9223372036854775807);  slice_396 = None
    slice_scatter_170: "f32[8, 1088, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_45, slice_scatter_168, 1, 1024, 9223372036854775807);  slice_scatter_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    slice_scatter_174: "f32[8, 1088, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_45, add_648, 1, 0, 1024)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    add_650: "f32[8, 1088, 14, 14]" = torch.ops.aten.add.Tensor(slice_scatter_170, slice_scatter_174);  slice_scatter_170 = slice_scatter_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_62 = torch.ops.aten.convolution_backward.default(add_650, relu_48, primals_271, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_650 = primals_271 = None
    getitem_410: "f32[8, 800, 14, 14]" = convolution_backward_62[0]
    getitem_411: "f32[1088, 800, 1, 1]" = convolution_backward_62[1];  convolution_backward_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_62: "b8[8, 800, 14, 14]" = torch.ops.aten.le.Scalar(relu_48, 0);  relu_48 = None
    where_62: "f32[8, 800, 14, 14]" = torch.ops.aten.where.self(le_62, full_default, getitem_410);  le_62 = getitem_410 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_126: "f32[800]" = torch.ops.aten.sum.dim_IntList(where_62, [0, 2, 3])
    sub_359: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_47, unsqueeze_1190);  convolution_47 = unsqueeze_1190 = None
    mul_1335: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(where_62, sub_359)
    sum_127: "f32[800]" = torch.ops.aten.sum.dim_IntList(mul_1335, [0, 2, 3]);  mul_1335 = None
    mul_1336: "f32[800]" = torch.ops.aten.mul.Tensor(sum_126, 0.0006377551020408163)
    unsqueeze_1191: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_1336, 0);  mul_1336 = None
    unsqueeze_1192: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1191, 2);  unsqueeze_1191 = None
    unsqueeze_1193: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1192, 3);  unsqueeze_1192 = None
    mul_1337: "f32[800]" = torch.ops.aten.mul.Tensor(sum_127, 0.0006377551020408163)
    mul_1338: "f32[800]" = torch.ops.aten.mul.Tensor(squeeze_145, squeeze_145)
    mul_1339: "f32[800]" = torch.ops.aten.mul.Tensor(mul_1337, mul_1338);  mul_1337 = mul_1338 = None
    unsqueeze_1194: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_1339, 0);  mul_1339 = None
    unsqueeze_1195: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1194, 2);  unsqueeze_1194 = None
    unsqueeze_1196: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1195, 3);  unsqueeze_1195 = None
    mul_1340: "f32[800]" = torch.ops.aten.mul.Tensor(squeeze_145, primals_97);  primals_97 = None
    unsqueeze_1197: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_1340, 0);  mul_1340 = None
    unsqueeze_1198: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1197, 2);  unsqueeze_1197 = None
    unsqueeze_1199: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1198, 3);  unsqueeze_1198 = None
    mul_1341: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_359, unsqueeze_1196);  sub_359 = unsqueeze_1196 = None
    sub_361: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(where_62, mul_1341);  where_62 = mul_1341 = None
    sub_362: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(sub_361, unsqueeze_1193);  sub_361 = unsqueeze_1193 = None
    mul_1342: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_362, unsqueeze_1199);  sub_362 = unsqueeze_1199 = None
    mul_1343: "f32[800]" = torch.ops.aten.mul.Tensor(sum_127, squeeze_145);  sum_127 = squeeze_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_63 = torch.ops.aten.convolution_backward.default(mul_1342, relu_47, primals_270, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False]);  mul_1342 = primals_270 = None
    getitem_413: "f32[8, 800, 14, 14]" = convolution_backward_63[0]
    getitem_414: "f32[800, 16, 3, 3]" = convolution_backward_63[1];  convolution_backward_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_63: "b8[8, 800, 14, 14]" = torch.ops.aten.le.Scalar(relu_47, 0);  relu_47 = None
    where_63: "f32[8, 800, 14, 14]" = torch.ops.aten.where.self(le_63, full_default, getitem_413);  le_63 = getitem_413 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_128: "f32[800]" = torch.ops.aten.sum.dim_IntList(where_63, [0, 2, 3])
    sub_363: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_1202);  convolution_46 = unsqueeze_1202 = None
    mul_1344: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(where_63, sub_363)
    sum_129: "f32[800]" = torch.ops.aten.sum.dim_IntList(mul_1344, [0, 2, 3]);  mul_1344 = None
    mul_1345: "f32[800]" = torch.ops.aten.mul.Tensor(sum_128, 0.0006377551020408163)
    unsqueeze_1203: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_1345, 0);  mul_1345 = None
    unsqueeze_1204: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1203, 2);  unsqueeze_1203 = None
    unsqueeze_1205: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1204, 3);  unsqueeze_1204 = None
    mul_1346: "f32[800]" = torch.ops.aten.mul.Tensor(sum_129, 0.0006377551020408163)
    mul_1347: "f32[800]" = torch.ops.aten.mul.Tensor(squeeze_142, squeeze_142)
    mul_1348: "f32[800]" = torch.ops.aten.mul.Tensor(mul_1346, mul_1347);  mul_1346 = mul_1347 = None
    unsqueeze_1206: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_1348, 0);  mul_1348 = None
    unsqueeze_1207: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1206, 2);  unsqueeze_1206 = None
    unsqueeze_1208: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1207, 3);  unsqueeze_1207 = None
    mul_1349: "f32[800]" = torch.ops.aten.mul.Tensor(squeeze_142, primals_95);  primals_95 = None
    unsqueeze_1209: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_1349, 0);  mul_1349 = None
    unsqueeze_1210: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1209, 2);  unsqueeze_1209 = None
    unsqueeze_1211: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1210, 3);  unsqueeze_1210 = None
    mul_1350: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_363, unsqueeze_1208);  sub_363 = unsqueeze_1208 = None
    sub_365: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(where_63, mul_1350);  where_63 = mul_1350 = None
    sub_366: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(sub_365, unsqueeze_1205);  sub_365 = unsqueeze_1205 = None
    mul_1351: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_366, unsqueeze_1211);  sub_366 = unsqueeze_1211 = None
    mul_1352: "f32[800]" = torch.ops.aten.mul.Tensor(sum_129, squeeze_142);  sum_129 = squeeze_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_64 = torch.ops.aten.convolution_backward.default(mul_1351, relu_46, primals_269, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1351 = primals_269 = None
    getitem_416: "f32[8, 1280, 14, 14]" = convolution_backward_64[0]
    getitem_417: "f32[800, 1280, 1, 1]" = convolution_backward_64[1];  convolution_backward_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_64: "b8[8, 1280, 14, 14]" = torch.ops.aten.le.Scalar(relu_46, 0);  relu_46 = None
    where_64: "f32[8, 1280, 14, 14]" = torch.ops.aten.where.self(le_64, full_default, getitem_416);  le_64 = getitem_416 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_130: "f32[1280]" = torch.ops.aten.sum.dim_IntList(where_64, [0, 2, 3])
    sub_367: "f32[8, 1280, 14, 14]" = torch.ops.aten.sub.Tensor(cat_27, unsqueeze_1214);  cat_27 = unsqueeze_1214 = None
    mul_1353: "f32[8, 1280, 14, 14]" = torch.ops.aten.mul.Tensor(where_64, sub_367)
    sum_131: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_1353, [0, 2, 3]);  mul_1353 = None
    mul_1354: "f32[1280]" = torch.ops.aten.mul.Tensor(sum_130, 0.0006377551020408163)
    unsqueeze_1215: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(mul_1354, 0);  mul_1354 = None
    unsqueeze_1216: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1215, 2);  unsqueeze_1215 = None
    unsqueeze_1217: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1216, 3);  unsqueeze_1216 = None
    mul_1355: "f32[1280]" = torch.ops.aten.mul.Tensor(sum_131, 0.0006377551020408163)
    mul_1356: "f32[1280]" = torch.ops.aten.mul.Tensor(squeeze_139, squeeze_139)
    mul_1357: "f32[1280]" = torch.ops.aten.mul.Tensor(mul_1355, mul_1356);  mul_1355 = mul_1356 = None
    unsqueeze_1218: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(mul_1357, 0);  mul_1357 = None
    unsqueeze_1219: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1218, 2);  unsqueeze_1218 = None
    unsqueeze_1220: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1219, 3);  unsqueeze_1219 = None
    mul_1358: "f32[1280]" = torch.ops.aten.mul.Tensor(squeeze_139, primals_93);  primals_93 = None
    unsqueeze_1221: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(mul_1358, 0);  mul_1358 = None
    unsqueeze_1222: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1221, 2);  unsqueeze_1221 = None
    unsqueeze_1223: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1222, 3);  unsqueeze_1222 = None
    mul_1359: "f32[8, 1280, 14, 14]" = torch.ops.aten.mul.Tensor(sub_367, unsqueeze_1220);  sub_367 = unsqueeze_1220 = None
    sub_369: "f32[8, 1280, 14, 14]" = torch.ops.aten.sub.Tensor(where_64, mul_1359);  where_64 = mul_1359 = None
    sub_370: "f32[8, 1280, 14, 14]" = torch.ops.aten.sub.Tensor(sub_369, unsqueeze_1217);  sub_369 = unsqueeze_1217 = None
    mul_1360: "f32[8, 1280, 14, 14]" = torch.ops.aten.mul.Tensor(sub_370, unsqueeze_1223);  sub_370 = unsqueeze_1223 = None
    mul_1361: "f32[1280]" = torch.ops.aten.mul.Tensor(sum_131, squeeze_139);  sum_131 = squeeze_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    slice_397: "f32[8, 1024, 14, 14]" = torch.ops.aten.slice.Tensor(mul_1360, 1, 0, 1024)
    slice_398: "f32[8, 256, 14, 14]" = torch.ops.aten.slice.Tensor(mul_1360, 1, 1024, 1280);  mul_1360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    add_651: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_648, slice_397);  add_648 = slice_397 = None
    add_652: "f32[8, 256, 14, 14]" = torch.ops.aten.add.Tensor(slice_395, slice_398);  slice_395 = slice_398 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    slice_399: "f32[8, 192, 14, 14]" = torch.ops.aten.slice.Tensor(add_652, 1, 0, 192)
    slice_400: "f32[8, 64, 14, 14]" = torch.ops.aten.slice.Tensor(add_652, 1, 192, 256);  add_652 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    slice_scatter_176: "f32[8, 64, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_43, slice_400, 3, 0, 9223372036854775807);  slice_400 = None
    slice_scatter_178: "f32[8, 1088, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_45, slice_scatter_176, 1, 1024, 9223372036854775807);  slice_scatter_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    slice_scatter_182: "f32[8, 1088, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_45, add_651, 1, 0, 1024)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    add_653: "f32[8, 1088, 14, 14]" = torch.ops.aten.add.Tensor(slice_scatter_178, slice_scatter_182);  slice_scatter_178 = slice_scatter_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_65 = torch.ops.aten.convolution_backward.default(add_653, relu_45, primals_268, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_653 = primals_268 = None
    getitem_419: "f32[8, 800, 14, 14]" = convolution_backward_65[0]
    getitem_420: "f32[1088, 800, 1, 1]" = convolution_backward_65[1];  convolution_backward_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_65: "b8[8, 800, 14, 14]" = torch.ops.aten.le.Scalar(relu_45, 0);  relu_45 = None
    where_65: "f32[8, 800, 14, 14]" = torch.ops.aten.where.self(le_65, full_default, getitem_419);  le_65 = getitem_419 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_132: "f32[800]" = torch.ops.aten.sum.dim_IntList(where_65, [0, 2, 3])
    sub_371: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_44, unsqueeze_1226);  convolution_44 = unsqueeze_1226 = None
    mul_1362: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(where_65, sub_371)
    sum_133: "f32[800]" = torch.ops.aten.sum.dim_IntList(mul_1362, [0, 2, 3]);  mul_1362 = None
    mul_1363: "f32[800]" = torch.ops.aten.mul.Tensor(sum_132, 0.0006377551020408163)
    unsqueeze_1227: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_1363, 0);  mul_1363 = None
    unsqueeze_1228: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1227, 2);  unsqueeze_1227 = None
    unsqueeze_1229: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1228, 3);  unsqueeze_1228 = None
    mul_1364: "f32[800]" = torch.ops.aten.mul.Tensor(sum_133, 0.0006377551020408163)
    mul_1365: "f32[800]" = torch.ops.aten.mul.Tensor(squeeze_136, squeeze_136)
    mul_1366: "f32[800]" = torch.ops.aten.mul.Tensor(mul_1364, mul_1365);  mul_1364 = mul_1365 = None
    unsqueeze_1230: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_1366, 0);  mul_1366 = None
    unsqueeze_1231: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1230, 2);  unsqueeze_1230 = None
    unsqueeze_1232: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1231, 3);  unsqueeze_1231 = None
    mul_1367: "f32[800]" = torch.ops.aten.mul.Tensor(squeeze_136, primals_91);  primals_91 = None
    unsqueeze_1233: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_1367, 0);  mul_1367 = None
    unsqueeze_1234: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1233, 2);  unsqueeze_1233 = None
    unsqueeze_1235: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1234, 3);  unsqueeze_1234 = None
    mul_1368: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_371, unsqueeze_1232);  sub_371 = unsqueeze_1232 = None
    sub_373: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(where_65, mul_1368);  where_65 = mul_1368 = None
    sub_374: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(sub_373, unsqueeze_1229);  sub_373 = unsqueeze_1229 = None
    mul_1369: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_374, unsqueeze_1235);  sub_374 = unsqueeze_1235 = None
    mul_1370: "f32[800]" = torch.ops.aten.mul.Tensor(sum_133, squeeze_136);  sum_133 = squeeze_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_66 = torch.ops.aten.convolution_backward.default(mul_1369, relu_44, primals_267, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False]);  mul_1369 = primals_267 = None
    getitem_422: "f32[8, 800, 14, 14]" = convolution_backward_66[0]
    getitem_423: "f32[800, 16, 3, 3]" = convolution_backward_66[1];  convolution_backward_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_66: "b8[8, 800, 14, 14]" = torch.ops.aten.le.Scalar(relu_44, 0);  relu_44 = None
    where_66: "f32[8, 800, 14, 14]" = torch.ops.aten.where.self(le_66, full_default, getitem_422);  le_66 = getitem_422 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_134: "f32[800]" = torch.ops.aten.sum.dim_IntList(where_66, [0, 2, 3])
    sub_375: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_43, unsqueeze_1238);  convolution_43 = unsqueeze_1238 = None
    mul_1371: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(where_66, sub_375)
    sum_135: "f32[800]" = torch.ops.aten.sum.dim_IntList(mul_1371, [0, 2, 3]);  mul_1371 = None
    mul_1372: "f32[800]" = torch.ops.aten.mul.Tensor(sum_134, 0.0006377551020408163)
    unsqueeze_1239: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_1372, 0);  mul_1372 = None
    unsqueeze_1240: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1239, 2);  unsqueeze_1239 = None
    unsqueeze_1241: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1240, 3);  unsqueeze_1240 = None
    mul_1373: "f32[800]" = torch.ops.aten.mul.Tensor(sum_135, 0.0006377551020408163)
    mul_1374: "f32[800]" = torch.ops.aten.mul.Tensor(squeeze_133, squeeze_133)
    mul_1375: "f32[800]" = torch.ops.aten.mul.Tensor(mul_1373, mul_1374);  mul_1373 = mul_1374 = None
    unsqueeze_1242: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_1375, 0);  mul_1375 = None
    unsqueeze_1243: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1242, 2);  unsqueeze_1242 = None
    unsqueeze_1244: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1243, 3);  unsqueeze_1243 = None
    mul_1376: "f32[800]" = torch.ops.aten.mul.Tensor(squeeze_133, primals_89);  primals_89 = None
    unsqueeze_1245: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_1376, 0);  mul_1376 = None
    unsqueeze_1246: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1245, 2);  unsqueeze_1245 = None
    unsqueeze_1247: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1246, 3);  unsqueeze_1246 = None
    mul_1377: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_375, unsqueeze_1244);  sub_375 = unsqueeze_1244 = None
    sub_377: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(where_66, mul_1377);  where_66 = mul_1377 = None
    sub_378: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(sub_377, unsqueeze_1241);  sub_377 = unsqueeze_1241 = None
    mul_1378: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_378, unsqueeze_1247);  sub_378 = unsqueeze_1247 = None
    mul_1379: "f32[800]" = torch.ops.aten.mul.Tensor(sum_135, squeeze_133);  sum_135 = squeeze_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_67 = torch.ops.aten.convolution_backward.default(mul_1378, relu_43, primals_266, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1378 = primals_266 = None
    getitem_425: "f32[8, 1216, 14, 14]" = convolution_backward_67[0]
    getitem_426: "f32[800, 1216, 1, 1]" = convolution_backward_67[1];  convolution_backward_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_67: "b8[8, 1216, 14, 14]" = torch.ops.aten.le.Scalar(relu_43, 0);  relu_43 = None
    where_67: "f32[8, 1216, 14, 14]" = torch.ops.aten.where.self(le_67, full_default, getitem_425);  le_67 = getitem_425 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_136: "f32[1216]" = torch.ops.aten.sum.dim_IntList(where_67, [0, 2, 3])
    sub_379: "f32[8, 1216, 14, 14]" = torch.ops.aten.sub.Tensor(cat_25, unsqueeze_1250);  cat_25 = unsqueeze_1250 = None
    mul_1380: "f32[8, 1216, 14, 14]" = torch.ops.aten.mul.Tensor(where_67, sub_379)
    sum_137: "f32[1216]" = torch.ops.aten.sum.dim_IntList(mul_1380, [0, 2, 3]);  mul_1380 = None
    mul_1381: "f32[1216]" = torch.ops.aten.mul.Tensor(sum_136, 0.0006377551020408163)
    unsqueeze_1251: "f32[1, 1216]" = torch.ops.aten.unsqueeze.default(mul_1381, 0);  mul_1381 = None
    unsqueeze_1252: "f32[1, 1216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1251, 2);  unsqueeze_1251 = None
    unsqueeze_1253: "f32[1, 1216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1252, 3);  unsqueeze_1252 = None
    mul_1382: "f32[1216]" = torch.ops.aten.mul.Tensor(sum_137, 0.0006377551020408163)
    mul_1383: "f32[1216]" = torch.ops.aten.mul.Tensor(squeeze_130, squeeze_130)
    mul_1384: "f32[1216]" = torch.ops.aten.mul.Tensor(mul_1382, mul_1383);  mul_1382 = mul_1383 = None
    unsqueeze_1254: "f32[1, 1216]" = torch.ops.aten.unsqueeze.default(mul_1384, 0);  mul_1384 = None
    unsqueeze_1255: "f32[1, 1216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1254, 2);  unsqueeze_1254 = None
    unsqueeze_1256: "f32[1, 1216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1255, 3);  unsqueeze_1255 = None
    mul_1385: "f32[1216]" = torch.ops.aten.mul.Tensor(squeeze_130, primals_87);  primals_87 = None
    unsqueeze_1257: "f32[1, 1216]" = torch.ops.aten.unsqueeze.default(mul_1385, 0);  mul_1385 = None
    unsqueeze_1258: "f32[1, 1216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1257, 2);  unsqueeze_1257 = None
    unsqueeze_1259: "f32[1, 1216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1258, 3);  unsqueeze_1258 = None
    mul_1386: "f32[8, 1216, 14, 14]" = torch.ops.aten.mul.Tensor(sub_379, unsqueeze_1256);  sub_379 = unsqueeze_1256 = None
    sub_381: "f32[8, 1216, 14, 14]" = torch.ops.aten.sub.Tensor(where_67, mul_1386);  where_67 = mul_1386 = None
    sub_382: "f32[8, 1216, 14, 14]" = torch.ops.aten.sub.Tensor(sub_381, unsqueeze_1253);  sub_381 = unsqueeze_1253 = None
    mul_1387: "f32[8, 1216, 14, 14]" = torch.ops.aten.mul.Tensor(sub_382, unsqueeze_1259);  sub_382 = unsqueeze_1259 = None
    mul_1388: "f32[1216]" = torch.ops.aten.mul.Tensor(sum_137, squeeze_130);  sum_137 = squeeze_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    slice_401: "f32[8, 1024, 14, 14]" = torch.ops.aten.slice.Tensor(mul_1387, 1, 0, 1024)
    slice_402: "f32[8, 192, 14, 14]" = torch.ops.aten.slice.Tensor(mul_1387, 1, 1024, 1216);  mul_1387 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    add_654: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_651, slice_401);  add_651 = slice_401 = None
    add_655: "f32[8, 192, 14, 14]" = torch.ops.aten.add.Tensor(slice_399, slice_402);  slice_399 = slice_402 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    slice_403: "f32[8, 128, 14, 14]" = torch.ops.aten.slice.Tensor(add_655, 1, 0, 128)
    slice_404: "f32[8, 64, 14, 14]" = torch.ops.aten.slice.Tensor(add_655, 1, 128, 192);  add_655 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    slice_scatter_184: "f32[8, 64, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_43, slice_404, 3, 0, 9223372036854775807);  full_default_43 = slice_404 = None
    slice_scatter_186: "f32[8, 1088, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_45, slice_scatter_184, 1, 1024, 9223372036854775807);  slice_scatter_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    slice_scatter_190: "f32[8, 1088, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_45, add_654, 1, 0, 1024);  full_default_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    add_656: "f32[8, 1088, 14, 14]" = torch.ops.aten.add.Tensor(slice_scatter_186, slice_scatter_190);  slice_scatter_186 = slice_scatter_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_68 = torch.ops.aten.convolution_backward.default(add_656, relu_42, primals_265, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_656 = primals_265 = None
    getitem_428: "f32[8, 800, 14, 14]" = convolution_backward_68[0]
    getitem_429: "f32[1088, 800, 1, 1]" = convolution_backward_68[1];  convolution_backward_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_68: "b8[8, 800, 14, 14]" = torch.ops.aten.le.Scalar(relu_42, 0);  relu_42 = None
    where_68: "f32[8, 800, 14, 14]" = torch.ops.aten.where.self(le_68, full_default, getitem_428);  le_68 = getitem_428 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_138: "f32[800]" = torch.ops.aten.sum.dim_IntList(where_68, [0, 2, 3])
    sub_383: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_1262);  convolution_41 = unsqueeze_1262 = None
    mul_1389: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(where_68, sub_383)
    sum_139: "f32[800]" = torch.ops.aten.sum.dim_IntList(mul_1389, [0, 2, 3]);  mul_1389 = None
    mul_1390: "f32[800]" = torch.ops.aten.mul.Tensor(sum_138, 0.0006377551020408163)
    unsqueeze_1263: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_1390, 0);  mul_1390 = None
    unsqueeze_1264: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1263, 2);  unsqueeze_1263 = None
    unsqueeze_1265: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1264, 3);  unsqueeze_1264 = None
    mul_1391: "f32[800]" = torch.ops.aten.mul.Tensor(sum_139, 0.0006377551020408163)
    mul_1392: "f32[800]" = torch.ops.aten.mul.Tensor(squeeze_127, squeeze_127)
    mul_1393: "f32[800]" = torch.ops.aten.mul.Tensor(mul_1391, mul_1392);  mul_1391 = mul_1392 = None
    unsqueeze_1266: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_1393, 0);  mul_1393 = None
    unsqueeze_1267: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1266, 2);  unsqueeze_1266 = None
    unsqueeze_1268: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1267, 3);  unsqueeze_1267 = None
    mul_1394: "f32[800]" = torch.ops.aten.mul.Tensor(squeeze_127, primals_85);  primals_85 = None
    unsqueeze_1269: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_1394, 0);  mul_1394 = None
    unsqueeze_1270: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1269, 2);  unsqueeze_1269 = None
    unsqueeze_1271: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1270, 3);  unsqueeze_1270 = None
    mul_1395: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_383, unsqueeze_1268);  sub_383 = unsqueeze_1268 = None
    sub_385: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(where_68, mul_1395);  where_68 = mul_1395 = None
    sub_386: "f32[8, 800, 14, 14]" = torch.ops.aten.sub.Tensor(sub_385, unsqueeze_1265);  sub_385 = unsqueeze_1265 = None
    mul_1396: "f32[8, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_386, unsqueeze_1271);  sub_386 = unsqueeze_1271 = None
    mul_1397: "f32[800]" = torch.ops.aten.mul.Tensor(sum_139, squeeze_127);  sum_139 = squeeze_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_69 = torch.ops.aten.convolution_backward.default(mul_1396, relu_41, primals_264, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False]);  mul_1396 = primals_264 = None
    getitem_431: "f32[8, 800, 28, 28]" = convolution_backward_69[0]
    getitem_432: "f32[800, 16, 3, 3]" = convolution_backward_69[1];  convolution_backward_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_69: "b8[8, 800, 28, 28]" = torch.ops.aten.le.Scalar(relu_41, 0);  relu_41 = None
    where_69: "f32[8, 800, 28, 28]" = torch.ops.aten.where.self(le_69, full_default, getitem_431);  le_69 = getitem_431 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_140: "f32[800]" = torch.ops.aten.sum.dim_IntList(where_69, [0, 2, 3])
    sub_387: "f32[8, 800, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_1274);  convolution_40 = unsqueeze_1274 = None
    mul_1398: "f32[8, 800, 28, 28]" = torch.ops.aten.mul.Tensor(where_69, sub_387)
    sum_141: "f32[800]" = torch.ops.aten.sum.dim_IntList(mul_1398, [0, 2, 3]);  mul_1398 = None
    mul_1399: "f32[800]" = torch.ops.aten.mul.Tensor(sum_140, 0.00015943877551020407)
    unsqueeze_1275: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_1399, 0);  mul_1399 = None
    unsqueeze_1276: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1275, 2);  unsqueeze_1275 = None
    unsqueeze_1277: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1276, 3);  unsqueeze_1276 = None
    mul_1400: "f32[800]" = torch.ops.aten.mul.Tensor(sum_141, 0.00015943877551020407)
    mul_1401: "f32[800]" = torch.ops.aten.mul.Tensor(squeeze_124, squeeze_124)
    mul_1402: "f32[800]" = torch.ops.aten.mul.Tensor(mul_1400, mul_1401);  mul_1400 = mul_1401 = None
    unsqueeze_1278: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_1402, 0);  mul_1402 = None
    unsqueeze_1279: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1278, 2);  unsqueeze_1278 = None
    unsqueeze_1280: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1279, 3);  unsqueeze_1279 = None
    mul_1403: "f32[800]" = torch.ops.aten.mul.Tensor(squeeze_124, primals_83);  primals_83 = None
    unsqueeze_1281: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(mul_1403, 0);  mul_1403 = None
    unsqueeze_1282: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1281, 2);  unsqueeze_1281 = None
    unsqueeze_1283: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1282, 3);  unsqueeze_1282 = None
    mul_1404: "f32[8, 800, 28, 28]" = torch.ops.aten.mul.Tensor(sub_387, unsqueeze_1280);  sub_387 = unsqueeze_1280 = None
    sub_389: "f32[8, 800, 28, 28]" = torch.ops.aten.sub.Tensor(where_69, mul_1404);  where_69 = mul_1404 = None
    sub_390: "f32[8, 800, 28, 28]" = torch.ops.aten.sub.Tensor(sub_389, unsqueeze_1277);  sub_389 = unsqueeze_1277 = None
    mul_1405: "f32[8, 800, 28, 28]" = torch.ops.aten.mul.Tensor(sub_390, unsqueeze_1283);  sub_390 = unsqueeze_1283 = None
    mul_1406: "f32[800]" = torch.ops.aten.mul.Tensor(sum_141, squeeze_124);  sum_141 = squeeze_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_70 = torch.ops.aten.convolution_backward.default(mul_1405, relu_40, primals_263, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1405 = primals_263 = None
    getitem_434: "f32[8, 1152, 28, 28]" = convolution_backward_70[0]
    getitem_435: "f32[800, 1152, 1, 1]" = convolution_backward_70[1];  convolution_backward_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_70: "b8[8, 1152, 28, 28]" = torch.ops.aten.le.Scalar(relu_40, 0);  relu_40 = None
    where_70: "f32[8, 1152, 28, 28]" = torch.ops.aten.where.self(le_70, full_default, getitem_434);  le_70 = getitem_434 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_142: "f32[1152]" = torch.ops.aten.sum.dim_IntList(where_70, [0, 2, 3])
    sub_391: "f32[8, 1152, 28, 28]" = torch.ops.aten.sub.Tensor(cat_23, unsqueeze_1286);  cat_23 = unsqueeze_1286 = None
    mul_1407: "f32[8, 1152, 28, 28]" = torch.ops.aten.mul.Tensor(where_70, sub_391)
    sum_143: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_1407, [0, 2, 3]);  mul_1407 = None
    mul_1408: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_142, 0.00015943877551020407)
    unsqueeze_1287: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_1408, 0);  mul_1408 = None
    unsqueeze_1288: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1287, 2);  unsqueeze_1287 = None
    unsqueeze_1289: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1288, 3);  unsqueeze_1288 = None
    mul_1409: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_143, 0.00015943877551020407)
    mul_1410: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_118, squeeze_118)
    mul_1411: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_1409, mul_1410);  mul_1409 = None
    unsqueeze_1290: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_1411, 0);  mul_1411 = None
    unsqueeze_1291: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1290, 2);  unsqueeze_1290 = None
    unsqueeze_1292: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1291, 3);  unsqueeze_1291 = None
    mul_1412: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_118, primals_81);  primals_81 = None
    unsqueeze_1293: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_1412, 0);  mul_1412 = None
    unsqueeze_1294: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1293, 2);  unsqueeze_1293 = None
    unsqueeze_1295: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1294, 3);  unsqueeze_1294 = None
    mul_1413: "f32[8, 1152, 28, 28]" = torch.ops.aten.mul.Tensor(sub_391, unsqueeze_1292);  unsqueeze_1292 = None
    sub_393: "f32[8, 1152, 28, 28]" = torch.ops.aten.sub.Tensor(where_70, mul_1413);  where_70 = mul_1413 = None
    sub_394: "f32[8, 1152, 28, 28]" = torch.ops.aten.sub.Tensor(sub_393, unsqueeze_1289);  sub_393 = unsqueeze_1289 = None
    mul_1414: "f32[8, 1152, 28, 28]" = torch.ops.aten.mul.Tensor(sub_394, unsqueeze_1295);  sub_394 = unsqueeze_1295 = None
    mul_1415: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_143, squeeze_118);  sum_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:134, code: x_s2 = x_s[:, self.num_1x1_c:, :, :]
    full_default_263: "f32[8, 128, 14, 14]" = torch.ops.aten.full.default([8, 128, 14, 14], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_192: "f32[8, 128, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_263, slice_403, 3, 0, 9223372036854775807);  full_default_263 = slice_403 = None
    full_default_265: "f32[8, 1152, 14, 14]" = torch.ops.aten.full.default([8, 1152, 14, 14], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_194: "f32[8, 1152, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_265, slice_scatter_192, 1, 1024, 9223372036854775807);  slice_scatter_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:133, code: x_s1 = x_s[:, :self.num_1x1_c, :, :]
    slice_scatter_198: "f32[8, 1152, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_265, add_654, 1, 0, 1024);  full_default_265 = add_654 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:133, code: x_s1 = x_s[:, :self.num_1x1_c, :, :]
    add_657: "f32[8, 1152, 14, 14]" = torch.ops.aten.add.Tensor(slice_scatter_194, slice_scatter_198);  slice_scatter_194 = slice_scatter_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_71 = torch.ops.aten.convolution_backward.default(add_657, relu_39, primals_262, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_657 = primals_262 = None
    getitem_437: "f32[8, 1152, 28, 28]" = convolution_backward_71[0]
    getitem_438: "f32[1152, 1152, 1, 1]" = convolution_backward_71[1];  convolution_backward_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_71: "b8[8, 1152, 28, 28]" = torch.ops.aten.le.Scalar(relu_39, 0);  relu_39 = None
    where_71: "f32[8, 1152, 28, 28]" = torch.ops.aten.where.self(le_71, full_default, getitem_437);  le_71 = getitem_437 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_144: "f32[1152]" = torch.ops.aten.sum.dim_IntList(where_71, [0, 2, 3])
    mul_1416: "f32[8, 1152, 28, 28]" = torch.ops.aten.mul.Tensor(where_71, sub_391)
    sum_145: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_1416, [0, 2, 3]);  mul_1416 = None
    mul_1417: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_144, 0.00015943877551020407)
    unsqueeze_1299: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_1417, 0);  mul_1417 = None
    unsqueeze_1300: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1299, 2);  unsqueeze_1299 = None
    unsqueeze_1301: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1300, 3);  unsqueeze_1300 = None
    mul_1418: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_145, 0.00015943877551020407)
    mul_1420: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_1418, mul_1410);  mul_1418 = mul_1410 = None
    unsqueeze_1302: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_1420, 0);  mul_1420 = None
    unsqueeze_1303: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1302, 2);  unsqueeze_1302 = None
    unsqueeze_1304: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1303, 3);  unsqueeze_1303 = None
    mul_1421: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_118, primals_79);  primals_79 = None
    unsqueeze_1305: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_1421, 0);  mul_1421 = None
    unsqueeze_1306: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1305, 2);  unsqueeze_1305 = None
    unsqueeze_1307: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1306, 3);  unsqueeze_1306 = None
    mul_1422: "f32[8, 1152, 28, 28]" = torch.ops.aten.mul.Tensor(sub_391, unsqueeze_1304);  sub_391 = unsqueeze_1304 = None
    sub_397: "f32[8, 1152, 28, 28]" = torch.ops.aten.sub.Tensor(where_71, mul_1422);  where_71 = mul_1422 = None
    sub_398: "f32[8, 1152, 28, 28]" = torch.ops.aten.sub.Tensor(sub_397, unsqueeze_1301);  sub_397 = unsqueeze_1301 = None
    mul_1423: "f32[8, 1152, 28, 28]" = torch.ops.aten.mul.Tensor(sub_398, unsqueeze_1307);  sub_398 = unsqueeze_1307 = None
    mul_1424: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_145, squeeze_118);  sum_145 = squeeze_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_658: "f32[8, 1152, 28, 28]" = torch.ops.aten.add.Tensor(mul_1414, mul_1423);  mul_1414 = mul_1423 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    slice_405: "f32[8, 512, 28, 28]" = torch.ops.aten.slice.Tensor(add_658, 1, 0, 512)
    slice_406: "f32[8, 640, 28, 28]" = torch.ops.aten.slice.Tensor(add_658, 1, 512, 1152);  add_658 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    slice_407: "f32[8, 576, 28, 28]" = torch.ops.aten.slice.Tensor(slice_406, 1, 0, 576)
    slice_408: "f32[8, 64, 28, 28]" = torch.ops.aten.slice.Tensor(slice_406, 1, 576, 640);  slice_406 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    full_default_272: "f32[8, 64, 28, 28]" = torch.ops.aten.full.default([8, 64, 28, 28], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_200: "f32[8, 64, 28, 28]" = torch.ops.aten.slice_scatter.default(full_default_272, slice_408, 3, 0, 9223372036854775807);  slice_408 = None
    full_default_274: "f32[8, 576, 28, 28]" = torch.ops.aten.full.default([8, 576, 28, 28], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_202: "f32[8, 576, 28, 28]" = torch.ops.aten.slice_scatter.default(full_default_274, slice_scatter_200, 1, 512, 9223372036854775807);  slice_scatter_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    full_default_276: "f32[8, 512, 28, 28]" = torch.ops.aten.full.default([8, 512, 28, 28], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_204: "f32[8, 512, 28, 28]" = torch.ops.aten.slice_scatter.default(full_default_276, slice_405, 3, 0, 9223372036854775807);  full_default_276 = None
    slice_scatter_206: "f32[8, 576, 28, 28]" = torch.ops.aten.slice_scatter.default(full_default_274, slice_scatter_204, 1, 0, 512);  slice_scatter_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    add_659: "f32[8, 576, 28, 28]" = torch.ops.aten.add.Tensor(slice_scatter_202, slice_scatter_206);  slice_scatter_202 = slice_scatter_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_72 = torch.ops.aten.convolution_backward.default(add_659, relu_38, primals_261, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_659 = primals_261 = None
    getitem_440: "f32[8, 400, 28, 28]" = convolution_backward_72[0]
    getitem_441: "f32[576, 400, 1, 1]" = convolution_backward_72[1];  convolution_backward_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_72: "b8[8, 400, 28, 28]" = torch.ops.aten.le.Scalar(relu_38, 0);  relu_38 = None
    where_72: "f32[8, 400, 28, 28]" = torch.ops.aten.where.self(le_72, full_default, getitem_440);  le_72 = getitem_440 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_146: "f32[400]" = torch.ops.aten.sum.dim_IntList(where_72, [0, 2, 3])
    sub_399: "f32[8, 400, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_37, unsqueeze_1310);  convolution_37 = unsqueeze_1310 = None
    mul_1425: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(where_72, sub_399)
    sum_147: "f32[400]" = torch.ops.aten.sum.dim_IntList(mul_1425, [0, 2, 3]);  mul_1425 = None
    mul_1426: "f32[400]" = torch.ops.aten.mul.Tensor(sum_146, 0.00015943877551020407)
    unsqueeze_1311: "f32[1, 400]" = torch.ops.aten.unsqueeze.default(mul_1426, 0);  mul_1426 = None
    unsqueeze_1312: "f32[1, 400, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1311, 2);  unsqueeze_1311 = None
    unsqueeze_1313: "f32[1, 400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1312, 3);  unsqueeze_1312 = None
    mul_1427: "f32[400]" = torch.ops.aten.mul.Tensor(sum_147, 0.00015943877551020407)
    mul_1428: "f32[400]" = torch.ops.aten.mul.Tensor(squeeze_115, squeeze_115)
    mul_1429: "f32[400]" = torch.ops.aten.mul.Tensor(mul_1427, mul_1428);  mul_1427 = mul_1428 = None
    unsqueeze_1314: "f32[1, 400]" = torch.ops.aten.unsqueeze.default(mul_1429, 0);  mul_1429 = None
    unsqueeze_1315: "f32[1, 400, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1314, 2);  unsqueeze_1314 = None
    unsqueeze_1316: "f32[1, 400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1315, 3);  unsqueeze_1315 = None
    mul_1430: "f32[400]" = torch.ops.aten.mul.Tensor(squeeze_115, primals_77);  primals_77 = None
    unsqueeze_1317: "f32[1, 400]" = torch.ops.aten.unsqueeze.default(mul_1430, 0);  mul_1430 = None
    unsqueeze_1318: "f32[1, 400, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1317, 2);  unsqueeze_1317 = None
    unsqueeze_1319: "f32[1, 400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1318, 3);  unsqueeze_1318 = None
    mul_1431: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(sub_399, unsqueeze_1316);  sub_399 = unsqueeze_1316 = None
    sub_401: "f32[8, 400, 28, 28]" = torch.ops.aten.sub.Tensor(where_72, mul_1431);  where_72 = mul_1431 = None
    sub_402: "f32[8, 400, 28, 28]" = torch.ops.aten.sub.Tensor(sub_401, unsqueeze_1313);  sub_401 = unsqueeze_1313 = None
    mul_1432: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(sub_402, unsqueeze_1319);  sub_402 = unsqueeze_1319 = None
    mul_1433: "f32[400]" = torch.ops.aten.mul.Tensor(sum_147, squeeze_115);  sum_147 = squeeze_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_73 = torch.ops.aten.convolution_backward.default(mul_1432, relu_37, primals_260, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False]);  mul_1432 = primals_260 = None
    getitem_443: "f32[8, 400, 28, 28]" = convolution_backward_73[0]
    getitem_444: "f32[400, 8, 3, 3]" = convolution_backward_73[1];  convolution_backward_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_73: "b8[8, 400, 28, 28]" = torch.ops.aten.le.Scalar(relu_37, 0);  relu_37 = None
    where_73: "f32[8, 400, 28, 28]" = torch.ops.aten.where.self(le_73, full_default, getitem_443);  le_73 = getitem_443 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_148: "f32[400]" = torch.ops.aten.sum.dim_IntList(where_73, [0, 2, 3])
    sub_403: "f32[8, 400, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_1322);  convolution_36 = unsqueeze_1322 = None
    mul_1434: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(where_73, sub_403)
    sum_149: "f32[400]" = torch.ops.aten.sum.dim_IntList(mul_1434, [0, 2, 3]);  mul_1434 = None
    mul_1435: "f32[400]" = torch.ops.aten.mul.Tensor(sum_148, 0.00015943877551020407)
    unsqueeze_1323: "f32[1, 400]" = torch.ops.aten.unsqueeze.default(mul_1435, 0);  mul_1435 = None
    unsqueeze_1324: "f32[1, 400, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1323, 2);  unsqueeze_1323 = None
    unsqueeze_1325: "f32[1, 400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1324, 3);  unsqueeze_1324 = None
    mul_1436: "f32[400]" = torch.ops.aten.mul.Tensor(sum_149, 0.00015943877551020407)
    mul_1437: "f32[400]" = torch.ops.aten.mul.Tensor(squeeze_112, squeeze_112)
    mul_1438: "f32[400]" = torch.ops.aten.mul.Tensor(mul_1436, mul_1437);  mul_1436 = mul_1437 = None
    unsqueeze_1326: "f32[1, 400]" = torch.ops.aten.unsqueeze.default(mul_1438, 0);  mul_1438 = None
    unsqueeze_1327: "f32[1, 400, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1326, 2);  unsqueeze_1326 = None
    unsqueeze_1328: "f32[1, 400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1327, 3);  unsqueeze_1327 = None
    mul_1439: "f32[400]" = torch.ops.aten.mul.Tensor(squeeze_112, primals_75);  primals_75 = None
    unsqueeze_1329: "f32[1, 400]" = torch.ops.aten.unsqueeze.default(mul_1439, 0);  mul_1439 = None
    unsqueeze_1330: "f32[1, 400, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1329, 2);  unsqueeze_1329 = None
    unsqueeze_1331: "f32[1, 400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1330, 3);  unsqueeze_1330 = None
    mul_1440: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(sub_403, unsqueeze_1328);  sub_403 = unsqueeze_1328 = None
    sub_405: "f32[8, 400, 28, 28]" = torch.ops.aten.sub.Tensor(where_73, mul_1440);  where_73 = mul_1440 = None
    sub_406: "f32[8, 400, 28, 28]" = torch.ops.aten.sub.Tensor(sub_405, unsqueeze_1325);  sub_405 = unsqueeze_1325 = None
    mul_1441: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(sub_406, unsqueeze_1331);  sub_406 = unsqueeze_1331 = None
    mul_1442: "f32[400]" = torch.ops.aten.mul.Tensor(sum_149, squeeze_112);  sum_149 = squeeze_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_74 = torch.ops.aten.convolution_backward.default(mul_1441, relu_36, primals_259, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1441 = primals_259 = None
    getitem_446: "f32[8, 1088, 28, 28]" = convolution_backward_74[0]
    getitem_447: "f32[400, 1088, 1, 1]" = convolution_backward_74[1];  convolution_backward_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_74: "b8[8, 1088, 28, 28]" = torch.ops.aten.le.Scalar(relu_36, 0);  relu_36 = None
    where_74: "f32[8, 1088, 28, 28]" = torch.ops.aten.where.self(le_74, full_default, getitem_446);  le_74 = getitem_446 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_150: "f32[1088]" = torch.ops.aten.sum.dim_IntList(where_74, [0, 2, 3])
    sub_407: "f32[8, 1088, 28, 28]" = torch.ops.aten.sub.Tensor(cat_21, unsqueeze_1334);  cat_21 = unsqueeze_1334 = None
    mul_1443: "f32[8, 1088, 28, 28]" = torch.ops.aten.mul.Tensor(where_74, sub_407)
    sum_151: "f32[1088]" = torch.ops.aten.sum.dim_IntList(mul_1443, [0, 2, 3]);  mul_1443 = None
    mul_1444: "f32[1088]" = torch.ops.aten.mul.Tensor(sum_150, 0.00015943877551020407)
    unsqueeze_1335: "f32[1, 1088]" = torch.ops.aten.unsqueeze.default(mul_1444, 0);  mul_1444 = None
    unsqueeze_1336: "f32[1, 1088, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1335, 2);  unsqueeze_1335 = None
    unsqueeze_1337: "f32[1, 1088, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1336, 3);  unsqueeze_1336 = None
    mul_1445: "f32[1088]" = torch.ops.aten.mul.Tensor(sum_151, 0.00015943877551020407)
    mul_1446: "f32[1088]" = torch.ops.aten.mul.Tensor(squeeze_109, squeeze_109)
    mul_1447: "f32[1088]" = torch.ops.aten.mul.Tensor(mul_1445, mul_1446);  mul_1445 = mul_1446 = None
    unsqueeze_1338: "f32[1, 1088]" = torch.ops.aten.unsqueeze.default(mul_1447, 0);  mul_1447 = None
    unsqueeze_1339: "f32[1, 1088, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1338, 2);  unsqueeze_1338 = None
    unsqueeze_1340: "f32[1, 1088, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1339, 3);  unsqueeze_1339 = None
    mul_1448: "f32[1088]" = torch.ops.aten.mul.Tensor(squeeze_109, primals_73);  primals_73 = None
    unsqueeze_1341: "f32[1, 1088]" = torch.ops.aten.unsqueeze.default(mul_1448, 0);  mul_1448 = None
    unsqueeze_1342: "f32[1, 1088, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1341, 2);  unsqueeze_1341 = None
    unsqueeze_1343: "f32[1, 1088, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1342, 3);  unsqueeze_1342 = None
    mul_1449: "f32[8, 1088, 28, 28]" = torch.ops.aten.mul.Tensor(sub_407, unsqueeze_1340);  sub_407 = unsqueeze_1340 = None
    sub_409: "f32[8, 1088, 28, 28]" = torch.ops.aten.sub.Tensor(where_74, mul_1449);  where_74 = mul_1449 = None
    sub_410: "f32[8, 1088, 28, 28]" = torch.ops.aten.sub.Tensor(sub_409, unsqueeze_1337);  sub_409 = unsqueeze_1337 = None
    mul_1450: "f32[8, 1088, 28, 28]" = torch.ops.aten.mul.Tensor(sub_410, unsqueeze_1343);  sub_410 = unsqueeze_1343 = None
    mul_1451: "f32[1088]" = torch.ops.aten.mul.Tensor(sum_151, squeeze_109);  sum_151 = squeeze_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    slice_409: "f32[8, 512, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1450, 1, 0, 512)
    slice_410: "f32[8, 576, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1450, 1, 512, 1088);  mul_1450 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    add_660: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(slice_405, slice_409);  slice_405 = slice_409 = None
    add_661: "f32[8, 576, 28, 28]" = torch.ops.aten.add.Tensor(slice_407, slice_410);  slice_407 = slice_410 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    slice_411: "f32[8, 512, 28, 28]" = torch.ops.aten.slice.Tensor(add_661, 1, 0, 512)
    slice_412: "f32[8, 64, 28, 28]" = torch.ops.aten.slice.Tensor(add_661, 1, 512, 576);  add_661 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    slice_scatter_208: "f32[8, 64, 28, 28]" = torch.ops.aten.slice_scatter.default(full_default_272, slice_412, 3, 0, 9223372036854775807);  slice_412 = None
    slice_scatter_210: "f32[8, 576, 28, 28]" = torch.ops.aten.slice_scatter.default(full_default_274, slice_scatter_208, 1, 512, 9223372036854775807);  slice_scatter_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    slice_scatter_214: "f32[8, 576, 28, 28]" = torch.ops.aten.slice_scatter.default(full_default_274, add_660, 1, 0, 512)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    add_662: "f32[8, 576, 28, 28]" = torch.ops.aten.add.Tensor(slice_scatter_210, slice_scatter_214);  slice_scatter_210 = slice_scatter_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_75 = torch.ops.aten.convolution_backward.default(add_662, relu_35, primals_258, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_662 = primals_258 = None
    getitem_449: "f32[8, 400, 28, 28]" = convolution_backward_75[0]
    getitem_450: "f32[576, 400, 1, 1]" = convolution_backward_75[1];  convolution_backward_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_75: "b8[8, 400, 28, 28]" = torch.ops.aten.le.Scalar(relu_35, 0);  relu_35 = None
    where_75: "f32[8, 400, 28, 28]" = torch.ops.aten.where.self(le_75, full_default, getitem_449);  le_75 = getitem_449 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_152: "f32[400]" = torch.ops.aten.sum.dim_IntList(where_75, [0, 2, 3])
    sub_411: "f32[8, 400, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_1346);  convolution_34 = unsqueeze_1346 = None
    mul_1452: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(where_75, sub_411)
    sum_153: "f32[400]" = torch.ops.aten.sum.dim_IntList(mul_1452, [0, 2, 3]);  mul_1452 = None
    mul_1453: "f32[400]" = torch.ops.aten.mul.Tensor(sum_152, 0.00015943877551020407)
    unsqueeze_1347: "f32[1, 400]" = torch.ops.aten.unsqueeze.default(mul_1453, 0);  mul_1453 = None
    unsqueeze_1348: "f32[1, 400, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1347, 2);  unsqueeze_1347 = None
    unsqueeze_1349: "f32[1, 400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1348, 3);  unsqueeze_1348 = None
    mul_1454: "f32[400]" = torch.ops.aten.mul.Tensor(sum_153, 0.00015943877551020407)
    mul_1455: "f32[400]" = torch.ops.aten.mul.Tensor(squeeze_106, squeeze_106)
    mul_1456: "f32[400]" = torch.ops.aten.mul.Tensor(mul_1454, mul_1455);  mul_1454 = mul_1455 = None
    unsqueeze_1350: "f32[1, 400]" = torch.ops.aten.unsqueeze.default(mul_1456, 0);  mul_1456 = None
    unsqueeze_1351: "f32[1, 400, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1350, 2);  unsqueeze_1350 = None
    unsqueeze_1352: "f32[1, 400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1351, 3);  unsqueeze_1351 = None
    mul_1457: "f32[400]" = torch.ops.aten.mul.Tensor(squeeze_106, primals_71);  primals_71 = None
    unsqueeze_1353: "f32[1, 400]" = torch.ops.aten.unsqueeze.default(mul_1457, 0);  mul_1457 = None
    unsqueeze_1354: "f32[1, 400, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1353, 2);  unsqueeze_1353 = None
    unsqueeze_1355: "f32[1, 400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1354, 3);  unsqueeze_1354 = None
    mul_1458: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(sub_411, unsqueeze_1352);  sub_411 = unsqueeze_1352 = None
    sub_413: "f32[8, 400, 28, 28]" = torch.ops.aten.sub.Tensor(where_75, mul_1458);  where_75 = mul_1458 = None
    sub_414: "f32[8, 400, 28, 28]" = torch.ops.aten.sub.Tensor(sub_413, unsqueeze_1349);  sub_413 = unsqueeze_1349 = None
    mul_1459: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(sub_414, unsqueeze_1355);  sub_414 = unsqueeze_1355 = None
    mul_1460: "f32[400]" = torch.ops.aten.mul.Tensor(sum_153, squeeze_106);  sum_153 = squeeze_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_76 = torch.ops.aten.convolution_backward.default(mul_1459, relu_34, primals_257, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False]);  mul_1459 = primals_257 = None
    getitem_452: "f32[8, 400, 28, 28]" = convolution_backward_76[0]
    getitem_453: "f32[400, 8, 3, 3]" = convolution_backward_76[1];  convolution_backward_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_76: "b8[8, 400, 28, 28]" = torch.ops.aten.le.Scalar(relu_34, 0);  relu_34 = None
    where_76: "f32[8, 400, 28, 28]" = torch.ops.aten.where.self(le_76, full_default, getitem_452);  le_76 = getitem_452 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_154: "f32[400]" = torch.ops.aten.sum.dim_IntList(where_76, [0, 2, 3])
    sub_415: "f32[8, 400, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_1358);  convolution_33 = unsqueeze_1358 = None
    mul_1461: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(where_76, sub_415)
    sum_155: "f32[400]" = torch.ops.aten.sum.dim_IntList(mul_1461, [0, 2, 3]);  mul_1461 = None
    mul_1462: "f32[400]" = torch.ops.aten.mul.Tensor(sum_154, 0.00015943877551020407)
    unsqueeze_1359: "f32[1, 400]" = torch.ops.aten.unsqueeze.default(mul_1462, 0);  mul_1462 = None
    unsqueeze_1360: "f32[1, 400, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1359, 2);  unsqueeze_1359 = None
    unsqueeze_1361: "f32[1, 400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1360, 3);  unsqueeze_1360 = None
    mul_1463: "f32[400]" = torch.ops.aten.mul.Tensor(sum_155, 0.00015943877551020407)
    mul_1464: "f32[400]" = torch.ops.aten.mul.Tensor(squeeze_103, squeeze_103)
    mul_1465: "f32[400]" = torch.ops.aten.mul.Tensor(mul_1463, mul_1464);  mul_1463 = mul_1464 = None
    unsqueeze_1362: "f32[1, 400]" = torch.ops.aten.unsqueeze.default(mul_1465, 0);  mul_1465 = None
    unsqueeze_1363: "f32[1, 400, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1362, 2);  unsqueeze_1362 = None
    unsqueeze_1364: "f32[1, 400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1363, 3);  unsqueeze_1363 = None
    mul_1466: "f32[400]" = torch.ops.aten.mul.Tensor(squeeze_103, primals_69);  primals_69 = None
    unsqueeze_1365: "f32[1, 400]" = torch.ops.aten.unsqueeze.default(mul_1466, 0);  mul_1466 = None
    unsqueeze_1366: "f32[1, 400, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1365, 2);  unsqueeze_1365 = None
    unsqueeze_1367: "f32[1, 400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1366, 3);  unsqueeze_1366 = None
    mul_1467: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(sub_415, unsqueeze_1364);  sub_415 = unsqueeze_1364 = None
    sub_417: "f32[8, 400, 28, 28]" = torch.ops.aten.sub.Tensor(where_76, mul_1467);  where_76 = mul_1467 = None
    sub_418: "f32[8, 400, 28, 28]" = torch.ops.aten.sub.Tensor(sub_417, unsqueeze_1361);  sub_417 = unsqueeze_1361 = None
    mul_1468: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(sub_418, unsqueeze_1367);  sub_418 = unsqueeze_1367 = None
    mul_1469: "f32[400]" = torch.ops.aten.mul.Tensor(sum_155, squeeze_103);  sum_155 = squeeze_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_77 = torch.ops.aten.convolution_backward.default(mul_1468, relu_33, primals_256, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1468 = primals_256 = None
    getitem_455: "f32[8, 1024, 28, 28]" = convolution_backward_77[0]
    getitem_456: "f32[400, 1024, 1, 1]" = convolution_backward_77[1];  convolution_backward_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_77: "b8[8, 1024, 28, 28]" = torch.ops.aten.le.Scalar(relu_33, 0);  relu_33 = None
    where_77: "f32[8, 1024, 28, 28]" = torch.ops.aten.where.self(le_77, full_default, getitem_455);  le_77 = getitem_455 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_156: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_77, [0, 2, 3])
    sub_419: "f32[8, 1024, 28, 28]" = torch.ops.aten.sub.Tensor(cat_19, unsqueeze_1370);  cat_19 = unsqueeze_1370 = None
    mul_1470: "f32[8, 1024, 28, 28]" = torch.ops.aten.mul.Tensor(where_77, sub_419)
    sum_157: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1470, [0, 2, 3]);  mul_1470 = None
    mul_1471: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_156, 0.00015943877551020407)
    unsqueeze_1371: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1471, 0);  mul_1471 = None
    unsqueeze_1372: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1371, 2);  unsqueeze_1371 = None
    unsqueeze_1373: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1372, 3);  unsqueeze_1372 = None
    mul_1472: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_157, 0.00015943877551020407)
    mul_1473: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_100, squeeze_100)
    mul_1474: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1472, mul_1473);  mul_1472 = mul_1473 = None
    unsqueeze_1374: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1474, 0);  mul_1474 = None
    unsqueeze_1375: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1374, 2);  unsqueeze_1374 = None
    unsqueeze_1376: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1375, 3);  unsqueeze_1375 = None
    mul_1475: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_100, primals_67);  primals_67 = None
    unsqueeze_1377: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1475, 0);  mul_1475 = None
    unsqueeze_1378: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1377, 2);  unsqueeze_1377 = None
    unsqueeze_1379: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1378, 3);  unsqueeze_1378 = None
    mul_1476: "f32[8, 1024, 28, 28]" = torch.ops.aten.mul.Tensor(sub_419, unsqueeze_1376);  sub_419 = unsqueeze_1376 = None
    sub_421: "f32[8, 1024, 28, 28]" = torch.ops.aten.sub.Tensor(where_77, mul_1476);  where_77 = mul_1476 = None
    sub_422: "f32[8, 1024, 28, 28]" = torch.ops.aten.sub.Tensor(sub_421, unsqueeze_1373);  sub_421 = unsqueeze_1373 = None
    mul_1477: "f32[8, 1024, 28, 28]" = torch.ops.aten.mul.Tensor(sub_422, unsqueeze_1379);  sub_422 = unsqueeze_1379 = None
    mul_1478: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_157, squeeze_100);  sum_157 = squeeze_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    slice_413: "f32[8, 512, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1477, 1, 0, 512)
    slice_414: "f32[8, 512, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1477, 1, 512, 1024);  mul_1477 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    add_663: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(add_660, slice_413);  add_660 = slice_413 = None
    add_664: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(slice_411, slice_414);  slice_411 = slice_414 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    slice_415: "f32[8, 448, 28, 28]" = torch.ops.aten.slice.Tensor(add_664, 1, 0, 448)
    slice_416: "f32[8, 64, 28, 28]" = torch.ops.aten.slice.Tensor(add_664, 1, 448, 512);  add_664 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    slice_scatter_216: "f32[8, 64, 28, 28]" = torch.ops.aten.slice_scatter.default(full_default_272, slice_416, 3, 0, 9223372036854775807);  slice_416 = None
    slice_scatter_218: "f32[8, 576, 28, 28]" = torch.ops.aten.slice_scatter.default(full_default_274, slice_scatter_216, 1, 512, 9223372036854775807);  slice_scatter_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    slice_scatter_222: "f32[8, 576, 28, 28]" = torch.ops.aten.slice_scatter.default(full_default_274, add_663, 1, 0, 512)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    add_665: "f32[8, 576, 28, 28]" = torch.ops.aten.add.Tensor(slice_scatter_218, slice_scatter_222);  slice_scatter_218 = slice_scatter_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_78 = torch.ops.aten.convolution_backward.default(add_665, relu_32, primals_255, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_665 = primals_255 = None
    getitem_458: "f32[8, 400, 28, 28]" = convolution_backward_78[0]
    getitem_459: "f32[576, 400, 1, 1]" = convolution_backward_78[1];  convolution_backward_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_78: "b8[8, 400, 28, 28]" = torch.ops.aten.le.Scalar(relu_32, 0);  relu_32 = None
    where_78: "f32[8, 400, 28, 28]" = torch.ops.aten.where.self(le_78, full_default, getitem_458);  le_78 = getitem_458 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_158: "f32[400]" = torch.ops.aten.sum.dim_IntList(where_78, [0, 2, 3])
    sub_423: "f32[8, 400, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_31, unsqueeze_1382);  convolution_31 = unsqueeze_1382 = None
    mul_1479: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(where_78, sub_423)
    sum_159: "f32[400]" = torch.ops.aten.sum.dim_IntList(mul_1479, [0, 2, 3]);  mul_1479 = None
    mul_1480: "f32[400]" = torch.ops.aten.mul.Tensor(sum_158, 0.00015943877551020407)
    unsqueeze_1383: "f32[1, 400]" = torch.ops.aten.unsqueeze.default(mul_1480, 0);  mul_1480 = None
    unsqueeze_1384: "f32[1, 400, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1383, 2);  unsqueeze_1383 = None
    unsqueeze_1385: "f32[1, 400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1384, 3);  unsqueeze_1384 = None
    mul_1481: "f32[400]" = torch.ops.aten.mul.Tensor(sum_159, 0.00015943877551020407)
    mul_1482: "f32[400]" = torch.ops.aten.mul.Tensor(squeeze_97, squeeze_97)
    mul_1483: "f32[400]" = torch.ops.aten.mul.Tensor(mul_1481, mul_1482);  mul_1481 = mul_1482 = None
    unsqueeze_1386: "f32[1, 400]" = torch.ops.aten.unsqueeze.default(mul_1483, 0);  mul_1483 = None
    unsqueeze_1387: "f32[1, 400, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1386, 2);  unsqueeze_1386 = None
    unsqueeze_1388: "f32[1, 400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1387, 3);  unsqueeze_1387 = None
    mul_1484: "f32[400]" = torch.ops.aten.mul.Tensor(squeeze_97, primals_65);  primals_65 = None
    unsqueeze_1389: "f32[1, 400]" = torch.ops.aten.unsqueeze.default(mul_1484, 0);  mul_1484 = None
    unsqueeze_1390: "f32[1, 400, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1389, 2);  unsqueeze_1389 = None
    unsqueeze_1391: "f32[1, 400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1390, 3);  unsqueeze_1390 = None
    mul_1485: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(sub_423, unsqueeze_1388);  sub_423 = unsqueeze_1388 = None
    sub_425: "f32[8, 400, 28, 28]" = torch.ops.aten.sub.Tensor(where_78, mul_1485);  where_78 = mul_1485 = None
    sub_426: "f32[8, 400, 28, 28]" = torch.ops.aten.sub.Tensor(sub_425, unsqueeze_1385);  sub_425 = unsqueeze_1385 = None
    mul_1486: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(sub_426, unsqueeze_1391);  sub_426 = unsqueeze_1391 = None
    mul_1487: "f32[400]" = torch.ops.aten.mul.Tensor(sum_159, squeeze_97);  sum_159 = squeeze_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_79 = torch.ops.aten.convolution_backward.default(mul_1486, relu_31, primals_254, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False]);  mul_1486 = primals_254 = None
    getitem_461: "f32[8, 400, 28, 28]" = convolution_backward_79[0]
    getitem_462: "f32[400, 8, 3, 3]" = convolution_backward_79[1];  convolution_backward_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_79: "b8[8, 400, 28, 28]" = torch.ops.aten.le.Scalar(relu_31, 0);  relu_31 = None
    where_79: "f32[8, 400, 28, 28]" = torch.ops.aten.where.self(le_79, full_default, getitem_461);  le_79 = getitem_461 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_160: "f32[400]" = torch.ops.aten.sum.dim_IntList(where_79, [0, 2, 3])
    sub_427: "f32[8, 400, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_1394);  convolution_30 = unsqueeze_1394 = None
    mul_1488: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(where_79, sub_427)
    sum_161: "f32[400]" = torch.ops.aten.sum.dim_IntList(mul_1488, [0, 2, 3]);  mul_1488 = None
    mul_1489: "f32[400]" = torch.ops.aten.mul.Tensor(sum_160, 0.00015943877551020407)
    unsqueeze_1395: "f32[1, 400]" = torch.ops.aten.unsqueeze.default(mul_1489, 0);  mul_1489 = None
    unsqueeze_1396: "f32[1, 400, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1395, 2);  unsqueeze_1395 = None
    unsqueeze_1397: "f32[1, 400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1396, 3);  unsqueeze_1396 = None
    mul_1490: "f32[400]" = torch.ops.aten.mul.Tensor(sum_161, 0.00015943877551020407)
    mul_1491: "f32[400]" = torch.ops.aten.mul.Tensor(squeeze_94, squeeze_94)
    mul_1492: "f32[400]" = torch.ops.aten.mul.Tensor(mul_1490, mul_1491);  mul_1490 = mul_1491 = None
    unsqueeze_1398: "f32[1, 400]" = torch.ops.aten.unsqueeze.default(mul_1492, 0);  mul_1492 = None
    unsqueeze_1399: "f32[1, 400, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1398, 2);  unsqueeze_1398 = None
    unsqueeze_1400: "f32[1, 400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1399, 3);  unsqueeze_1399 = None
    mul_1493: "f32[400]" = torch.ops.aten.mul.Tensor(squeeze_94, primals_63);  primals_63 = None
    unsqueeze_1401: "f32[1, 400]" = torch.ops.aten.unsqueeze.default(mul_1493, 0);  mul_1493 = None
    unsqueeze_1402: "f32[1, 400, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1401, 2);  unsqueeze_1401 = None
    unsqueeze_1403: "f32[1, 400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1402, 3);  unsqueeze_1402 = None
    mul_1494: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(sub_427, unsqueeze_1400);  sub_427 = unsqueeze_1400 = None
    sub_429: "f32[8, 400, 28, 28]" = torch.ops.aten.sub.Tensor(where_79, mul_1494);  where_79 = mul_1494 = None
    sub_430: "f32[8, 400, 28, 28]" = torch.ops.aten.sub.Tensor(sub_429, unsqueeze_1397);  sub_429 = unsqueeze_1397 = None
    mul_1495: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(sub_430, unsqueeze_1403);  sub_430 = unsqueeze_1403 = None
    mul_1496: "f32[400]" = torch.ops.aten.mul.Tensor(sum_161, squeeze_94);  sum_161 = squeeze_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_80 = torch.ops.aten.convolution_backward.default(mul_1495, relu_30, primals_253, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1495 = primals_253 = None
    getitem_464: "f32[8, 960, 28, 28]" = convolution_backward_80[0]
    getitem_465: "f32[400, 960, 1, 1]" = convolution_backward_80[1];  convolution_backward_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_80: "b8[8, 960, 28, 28]" = torch.ops.aten.le.Scalar(relu_30, 0);  relu_30 = None
    where_80: "f32[8, 960, 28, 28]" = torch.ops.aten.where.self(le_80, full_default, getitem_464);  le_80 = getitem_464 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_162: "f32[960]" = torch.ops.aten.sum.dim_IntList(where_80, [0, 2, 3])
    sub_431: "f32[8, 960, 28, 28]" = torch.ops.aten.sub.Tensor(cat_17, unsqueeze_1406);  cat_17 = unsqueeze_1406 = None
    mul_1497: "f32[8, 960, 28, 28]" = torch.ops.aten.mul.Tensor(where_80, sub_431)
    sum_163: "f32[960]" = torch.ops.aten.sum.dim_IntList(mul_1497, [0, 2, 3]);  mul_1497 = None
    mul_1498: "f32[960]" = torch.ops.aten.mul.Tensor(sum_162, 0.00015943877551020407)
    unsqueeze_1407: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(mul_1498, 0);  mul_1498 = None
    unsqueeze_1408: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1407, 2);  unsqueeze_1407 = None
    unsqueeze_1409: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1408, 3);  unsqueeze_1408 = None
    mul_1499: "f32[960]" = torch.ops.aten.mul.Tensor(sum_163, 0.00015943877551020407)
    mul_1500: "f32[960]" = torch.ops.aten.mul.Tensor(squeeze_91, squeeze_91)
    mul_1501: "f32[960]" = torch.ops.aten.mul.Tensor(mul_1499, mul_1500);  mul_1499 = mul_1500 = None
    unsqueeze_1410: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(mul_1501, 0);  mul_1501 = None
    unsqueeze_1411: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1410, 2);  unsqueeze_1410 = None
    unsqueeze_1412: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1411, 3);  unsqueeze_1411 = None
    mul_1502: "f32[960]" = torch.ops.aten.mul.Tensor(squeeze_91, primals_61);  primals_61 = None
    unsqueeze_1413: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(mul_1502, 0);  mul_1502 = None
    unsqueeze_1414: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1413, 2);  unsqueeze_1413 = None
    unsqueeze_1415: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1414, 3);  unsqueeze_1414 = None
    mul_1503: "f32[8, 960, 28, 28]" = torch.ops.aten.mul.Tensor(sub_431, unsqueeze_1412);  sub_431 = unsqueeze_1412 = None
    sub_433: "f32[8, 960, 28, 28]" = torch.ops.aten.sub.Tensor(where_80, mul_1503);  where_80 = mul_1503 = None
    sub_434: "f32[8, 960, 28, 28]" = torch.ops.aten.sub.Tensor(sub_433, unsqueeze_1409);  sub_433 = unsqueeze_1409 = None
    mul_1504: "f32[8, 960, 28, 28]" = torch.ops.aten.mul.Tensor(sub_434, unsqueeze_1415);  sub_434 = unsqueeze_1415 = None
    mul_1505: "f32[960]" = torch.ops.aten.mul.Tensor(sum_163, squeeze_91);  sum_163 = squeeze_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    slice_417: "f32[8, 512, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1504, 1, 0, 512)
    slice_418: "f32[8, 448, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1504, 1, 512, 960);  mul_1504 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    add_666: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(add_663, slice_417);  add_663 = slice_417 = None
    add_667: "f32[8, 448, 28, 28]" = torch.ops.aten.add.Tensor(slice_415, slice_418);  slice_415 = slice_418 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    slice_419: "f32[8, 384, 28, 28]" = torch.ops.aten.slice.Tensor(add_667, 1, 0, 384)
    slice_420: "f32[8, 64, 28, 28]" = torch.ops.aten.slice.Tensor(add_667, 1, 384, 448);  add_667 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    slice_scatter_224: "f32[8, 64, 28, 28]" = torch.ops.aten.slice_scatter.default(full_default_272, slice_420, 3, 0, 9223372036854775807);  slice_420 = None
    slice_scatter_226: "f32[8, 576, 28, 28]" = torch.ops.aten.slice_scatter.default(full_default_274, slice_scatter_224, 1, 512, 9223372036854775807);  slice_scatter_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    slice_scatter_230: "f32[8, 576, 28, 28]" = torch.ops.aten.slice_scatter.default(full_default_274, add_666, 1, 0, 512)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    add_668: "f32[8, 576, 28, 28]" = torch.ops.aten.add.Tensor(slice_scatter_226, slice_scatter_230);  slice_scatter_226 = slice_scatter_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_81 = torch.ops.aten.convolution_backward.default(add_668, relu_29, primals_252, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_668 = primals_252 = None
    getitem_467: "f32[8, 400, 28, 28]" = convolution_backward_81[0]
    getitem_468: "f32[576, 400, 1, 1]" = convolution_backward_81[1];  convolution_backward_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_81: "b8[8, 400, 28, 28]" = torch.ops.aten.le.Scalar(relu_29, 0);  relu_29 = None
    where_81: "f32[8, 400, 28, 28]" = torch.ops.aten.where.self(le_81, full_default, getitem_467);  le_81 = getitem_467 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_164: "f32[400]" = torch.ops.aten.sum.dim_IntList(where_81, [0, 2, 3])
    sub_435: "f32[8, 400, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_1418);  convolution_28 = unsqueeze_1418 = None
    mul_1506: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(where_81, sub_435)
    sum_165: "f32[400]" = torch.ops.aten.sum.dim_IntList(mul_1506, [0, 2, 3]);  mul_1506 = None
    mul_1507: "f32[400]" = torch.ops.aten.mul.Tensor(sum_164, 0.00015943877551020407)
    unsqueeze_1419: "f32[1, 400]" = torch.ops.aten.unsqueeze.default(mul_1507, 0);  mul_1507 = None
    unsqueeze_1420: "f32[1, 400, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1419, 2);  unsqueeze_1419 = None
    unsqueeze_1421: "f32[1, 400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1420, 3);  unsqueeze_1420 = None
    mul_1508: "f32[400]" = torch.ops.aten.mul.Tensor(sum_165, 0.00015943877551020407)
    mul_1509: "f32[400]" = torch.ops.aten.mul.Tensor(squeeze_88, squeeze_88)
    mul_1510: "f32[400]" = torch.ops.aten.mul.Tensor(mul_1508, mul_1509);  mul_1508 = mul_1509 = None
    unsqueeze_1422: "f32[1, 400]" = torch.ops.aten.unsqueeze.default(mul_1510, 0);  mul_1510 = None
    unsqueeze_1423: "f32[1, 400, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1422, 2);  unsqueeze_1422 = None
    unsqueeze_1424: "f32[1, 400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1423, 3);  unsqueeze_1423 = None
    mul_1511: "f32[400]" = torch.ops.aten.mul.Tensor(squeeze_88, primals_59);  primals_59 = None
    unsqueeze_1425: "f32[1, 400]" = torch.ops.aten.unsqueeze.default(mul_1511, 0);  mul_1511 = None
    unsqueeze_1426: "f32[1, 400, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1425, 2);  unsqueeze_1425 = None
    unsqueeze_1427: "f32[1, 400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1426, 3);  unsqueeze_1426 = None
    mul_1512: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(sub_435, unsqueeze_1424);  sub_435 = unsqueeze_1424 = None
    sub_437: "f32[8, 400, 28, 28]" = torch.ops.aten.sub.Tensor(where_81, mul_1512);  where_81 = mul_1512 = None
    sub_438: "f32[8, 400, 28, 28]" = torch.ops.aten.sub.Tensor(sub_437, unsqueeze_1421);  sub_437 = unsqueeze_1421 = None
    mul_1513: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(sub_438, unsqueeze_1427);  sub_438 = unsqueeze_1427 = None
    mul_1514: "f32[400]" = torch.ops.aten.mul.Tensor(sum_165, squeeze_88);  sum_165 = squeeze_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_82 = torch.ops.aten.convolution_backward.default(mul_1513, relu_28, primals_251, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False]);  mul_1513 = primals_251 = None
    getitem_470: "f32[8, 400, 28, 28]" = convolution_backward_82[0]
    getitem_471: "f32[400, 8, 3, 3]" = convolution_backward_82[1];  convolution_backward_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_82: "b8[8, 400, 28, 28]" = torch.ops.aten.le.Scalar(relu_28, 0);  relu_28 = None
    where_82: "f32[8, 400, 28, 28]" = torch.ops.aten.where.self(le_82, full_default, getitem_470);  le_82 = getitem_470 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_166: "f32[400]" = torch.ops.aten.sum.dim_IntList(where_82, [0, 2, 3])
    sub_439: "f32[8, 400, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_1430);  convolution_27 = unsqueeze_1430 = None
    mul_1515: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(where_82, sub_439)
    sum_167: "f32[400]" = torch.ops.aten.sum.dim_IntList(mul_1515, [0, 2, 3]);  mul_1515 = None
    mul_1516: "f32[400]" = torch.ops.aten.mul.Tensor(sum_166, 0.00015943877551020407)
    unsqueeze_1431: "f32[1, 400]" = torch.ops.aten.unsqueeze.default(mul_1516, 0);  mul_1516 = None
    unsqueeze_1432: "f32[1, 400, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1431, 2);  unsqueeze_1431 = None
    unsqueeze_1433: "f32[1, 400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1432, 3);  unsqueeze_1432 = None
    mul_1517: "f32[400]" = torch.ops.aten.mul.Tensor(sum_167, 0.00015943877551020407)
    mul_1518: "f32[400]" = torch.ops.aten.mul.Tensor(squeeze_85, squeeze_85)
    mul_1519: "f32[400]" = torch.ops.aten.mul.Tensor(mul_1517, mul_1518);  mul_1517 = mul_1518 = None
    unsqueeze_1434: "f32[1, 400]" = torch.ops.aten.unsqueeze.default(mul_1519, 0);  mul_1519 = None
    unsqueeze_1435: "f32[1, 400, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1434, 2);  unsqueeze_1434 = None
    unsqueeze_1436: "f32[1, 400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1435, 3);  unsqueeze_1435 = None
    mul_1520: "f32[400]" = torch.ops.aten.mul.Tensor(squeeze_85, primals_57);  primals_57 = None
    unsqueeze_1437: "f32[1, 400]" = torch.ops.aten.unsqueeze.default(mul_1520, 0);  mul_1520 = None
    unsqueeze_1438: "f32[1, 400, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1437, 2);  unsqueeze_1437 = None
    unsqueeze_1439: "f32[1, 400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1438, 3);  unsqueeze_1438 = None
    mul_1521: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(sub_439, unsqueeze_1436);  sub_439 = unsqueeze_1436 = None
    sub_441: "f32[8, 400, 28, 28]" = torch.ops.aten.sub.Tensor(where_82, mul_1521);  where_82 = mul_1521 = None
    sub_442: "f32[8, 400, 28, 28]" = torch.ops.aten.sub.Tensor(sub_441, unsqueeze_1433);  sub_441 = unsqueeze_1433 = None
    mul_1522: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(sub_442, unsqueeze_1439);  sub_442 = unsqueeze_1439 = None
    mul_1523: "f32[400]" = torch.ops.aten.mul.Tensor(sum_167, squeeze_85);  sum_167 = squeeze_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_83 = torch.ops.aten.convolution_backward.default(mul_1522, relu_27, primals_250, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1522 = primals_250 = None
    getitem_473: "f32[8, 896, 28, 28]" = convolution_backward_83[0]
    getitem_474: "f32[400, 896, 1, 1]" = convolution_backward_83[1];  convolution_backward_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_83: "b8[8, 896, 28, 28]" = torch.ops.aten.le.Scalar(relu_27, 0);  relu_27 = None
    where_83: "f32[8, 896, 28, 28]" = torch.ops.aten.where.self(le_83, full_default, getitem_473);  le_83 = getitem_473 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_168: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_83, [0, 2, 3])
    sub_443: "f32[8, 896, 28, 28]" = torch.ops.aten.sub.Tensor(cat_15, unsqueeze_1442);  cat_15 = unsqueeze_1442 = None
    mul_1524: "f32[8, 896, 28, 28]" = torch.ops.aten.mul.Tensor(where_83, sub_443)
    sum_169: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_1524, [0, 2, 3]);  mul_1524 = None
    mul_1525: "f32[896]" = torch.ops.aten.mul.Tensor(sum_168, 0.00015943877551020407)
    unsqueeze_1443: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_1525, 0);  mul_1525 = None
    unsqueeze_1444: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1443, 2);  unsqueeze_1443 = None
    unsqueeze_1445: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1444, 3);  unsqueeze_1444 = None
    mul_1526: "f32[896]" = torch.ops.aten.mul.Tensor(sum_169, 0.00015943877551020407)
    mul_1527: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_82, squeeze_82)
    mul_1528: "f32[896]" = torch.ops.aten.mul.Tensor(mul_1526, mul_1527);  mul_1526 = mul_1527 = None
    unsqueeze_1446: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_1528, 0);  mul_1528 = None
    unsqueeze_1447: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1446, 2);  unsqueeze_1446 = None
    unsqueeze_1448: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1447, 3);  unsqueeze_1447 = None
    mul_1529: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_82, primals_55);  primals_55 = None
    unsqueeze_1449: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_1529, 0);  mul_1529 = None
    unsqueeze_1450: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1449, 2);  unsqueeze_1449 = None
    unsqueeze_1451: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1450, 3);  unsqueeze_1450 = None
    mul_1530: "f32[8, 896, 28, 28]" = torch.ops.aten.mul.Tensor(sub_443, unsqueeze_1448);  sub_443 = unsqueeze_1448 = None
    sub_445: "f32[8, 896, 28, 28]" = torch.ops.aten.sub.Tensor(where_83, mul_1530);  where_83 = mul_1530 = None
    sub_446: "f32[8, 896, 28, 28]" = torch.ops.aten.sub.Tensor(sub_445, unsqueeze_1445);  sub_445 = unsqueeze_1445 = None
    mul_1531: "f32[8, 896, 28, 28]" = torch.ops.aten.mul.Tensor(sub_446, unsqueeze_1451);  sub_446 = unsqueeze_1451 = None
    mul_1532: "f32[896]" = torch.ops.aten.mul.Tensor(sum_169, squeeze_82);  sum_169 = squeeze_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    slice_421: "f32[8, 512, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1531, 1, 0, 512)
    slice_422: "f32[8, 384, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1531, 1, 512, 896);  mul_1531 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    add_669: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(add_666, slice_421);  add_666 = slice_421 = None
    add_670: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(slice_419, slice_422);  slice_419 = slice_422 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    slice_423: "f32[8, 320, 28, 28]" = torch.ops.aten.slice.Tensor(add_670, 1, 0, 320)
    slice_424: "f32[8, 64, 28, 28]" = torch.ops.aten.slice.Tensor(add_670, 1, 320, 384);  add_670 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    slice_scatter_232: "f32[8, 64, 28, 28]" = torch.ops.aten.slice_scatter.default(full_default_272, slice_424, 3, 0, 9223372036854775807);  slice_424 = None
    slice_scatter_234: "f32[8, 576, 28, 28]" = torch.ops.aten.slice_scatter.default(full_default_274, slice_scatter_232, 1, 512, 9223372036854775807);  slice_scatter_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    slice_scatter_238: "f32[8, 576, 28, 28]" = torch.ops.aten.slice_scatter.default(full_default_274, add_669, 1, 0, 512)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    add_671: "f32[8, 576, 28, 28]" = torch.ops.aten.add.Tensor(slice_scatter_234, slice_scatter_238);  slice_scatter_234 = slice_scatter_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_84 = torch.ops.aten.convolution_backward.default(add_671, relu_26, primals_249, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_671 = primals_249 = None
    getitem_476: "f32[8, 400, 28, 28]" = convolution_backward_84[0]
    getitem_477: "f32[576, 400, 1, 1]" = convolution_backward_84[1];  convolution_backward_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_84: "b8[8, 400, 28, 28]" = torch.ops.aten.le.Scalar(relu_26, 0);  relu_26 = None
    where_84: "f32[8, 400, 28, 28]" = torch.ops.aten.where.self(le_84, full_default, getitem_476);  le_84 = getitem_476 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_170: "f32[400]" = torch.ops.aten.sum.dim_IntList(where_84, [0, 2, 3])
    sub_447: "f32[8, 400, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_1454);  convolution_25 = unsqueeze_1454 = None
    mul_1533: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(where_84, sub_447)
    sum_171: "f32[400]" = torch.ops.aten.sum.dim_IntList(mul_1533, [0, 2, 3]);  mul_1533 = None
    mul_1534: "f32[400]" = torch.ops.aten.mul.Tensor(sum_170, 0.00015943877551020407)
    unsqueeze_1455: "f32[1, 400]" = torch.ops.aten.unsqueeze.default(mul_1534, 0);  mul_1534 = None
    unsqueeze_1456: "f32[1, 400, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1455, 2);  unsqueeze_1455 = None
    unsqueeze_1457: "f32[1, 400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1456, 3);  unsqueeze_1456 = None
    mul_1535: "f32[400]" = torch.ops.aten.mul.Tensor(sum_171, 0.00015943877551020407)
    mul_1536: "f32[400]" = torch.ops.aten.mul.Tensor(squeeze_79, squeeze_79)
    mul_1537: "f32[400]" = torch.ops.aten.mul.Tensor(mul_1535, mul_1536);  mul_1535 = mul_1536 = None
    unsqueeze_1458: "f32[1, 400]" = torch.ops.aten.unsqueeze.default(mul_1537, 0);  mul_1537 = None
    unsqueeze_1459: "f32[1, 400, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1458, 2);  unsqueeze_1458 = None
    unsqueeze_1460: "f32[1, 400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1459, 3);  unsqueeze_1459 = None
    mul_1538: "f32[400]" = torch.ops.aten.mul.Tensor(squeeze_79, primals_53);  primals_53 = None
    unsqueeze_1461: "f32[1, 400]" = torch.ops.aten.unsqueeze.default(mul_1538, 0);  mul_1538 = None
    unsqueeze_1462: "f32[1, 400, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1461, 2);  unsqueeze_1461 = None
    unsqueeze_1463: "f32[1, 400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1462, 3);  unsqueeze_1462 = None
    mul_1539: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(sub_447, unsqueeze_1460);  sub_447 = unsqueeze_1460 = None
    sub_449: "f32[8, 400, 28, 28]" = torch.ops.aten.sub.Tensor(where_84, mul_1539);  where_84 = mul_1539 = None
    sub_450: "f32[8, 400, 28, 28]" = torch.ops.aten.sub.Tensor(sub_449, unsqueeze_1457);  sub_449 = unsqueeze_1457 = None
    mul_1540: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(sub_450, unsqueeze_1463);  sub_450 = unsqueeze_1463 = None
    mul_1541: "f32[400]" = torch.ops.aten.mul.Tensor(sum_171, squeeze_79);  sum_171 = squeeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_85 = torch.ops.aten.convolution_backward.default(mul_1540, relu_25, primals_248, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False]);  mul_1540 = primals_248 = None
    getitem_479: "f32[8, 400, 28, 28]" = convolution_backward_85[0]
    getitem_480: "f32[400, 8, 3, 3]" = convolution_backward_85[1];  convolution_backward_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_85: "b8[8, 400, 28, 28]" = torch.ops.aten.le.Scalar(relu_25, 0);  relu_25 = None
    where_85: "f32[8, 400, 28, 28]" = torch.ops.aten.where.self(le_85, full_default, getitem_479);  le_85 = getitem_479 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_172: "f32[400]" = torch.ops.aten.sum.dim_IntList(where_85, [0, 2, 3])
    sub_451: "f32[8, 400, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_1466);  convolution_24 = unsqueeze_1466 = None
    mul_1542: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(where_85, sub_451)
    sum_173: "f32[400]" = torch.ops.aten.sum.dim_IntList(mul_1542, [0, 2, 3]);  mul_1542 = None
    mul_1543: "f32[400]" = torch.ops.aten.mul.Tensor(sum_172, 0.00015943877551020407)
    unsqueeze_1467: "f32[1, 400]" = torch.ops.aten.unsqueeze.default(mul_1543, 0);  mul_1543 = None
    unsqueeze_1468: "f32[1, 400, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1467, 2);  unsqueeze_1467 = None
    unsqueeze_1469: "f32[1, 400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1468, 3);  unsqueeze_1468 = None
    mul_1544: "f32[400]" = torch.ops.aten.mul.Tensor(sum_173, 0.00015943877551020407)
    mul_1545: "f32[400]" = torch.ops.aten.mul.Tensor(squeeze_76, squeeze_76)
    mul_1546: "f32[400]" = torch.ops.aten.mul.Tensor(mul_1544, mul_1545);  mul_1544 = mul_1545 = None
    unsqueeze_1470: "f32[1, 400]" = torch.ops.aten.unsqueeze.default(mul_1546, 0);  mul_1546 = None
    unsqueeze_1471: "f32[1, 400, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1470, 2);  unsqueeze_1470 = None
    unsqueeze_1472: "f32[1, 400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1471, 3);  unsqueeze_1471 = None
    mul_1547: "f32[400]" = torch.ops.aten.mul.Tensor(squeeze_76, primals_51);  primals_51 = None
    unsqueeze_1473: "f32[1, 400]" = torch.ops.aten.unsqueeze.default(mul_1547, 0);  mul_1547 = None
    unsqueeze_1474: "f32[1, 400, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1473, 2);  unsqueeze_1473 = None
    unsqueeze_1475: "f32[1, 400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1474, 3);  unsqueeze_1474 = None
    mul_1548: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(sub_451, unsqueeze_1472);  sub_451 = unsqueeze_1472 = None
    sub_453: "f32[8, 400, 28, 28]" = torch.ops.aten.sub.Tensor(where_85, mul_1548);  where_85 = mul_1548 = None
    sub_454: "f32[8, 400, 28, 28]" = torch.ops.aten.sub.Tensor(sub_453, unsqueeze_1469);  sub_453 = unsqueeze_1469 = None
    mul_1549: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(sub_454, unsqueeze_1475);  sub_454 = unsqueeze_1475 = None
    mul_1550: "f32[400]" = torch.ops.aten.mul.Tensor(sum_173, squeeze_76);  sum_173 = squeeze_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_86 = torch.ops.aten.convolution_backward.default(mul_1549, relu_24, primals_247, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1549 = primals_247 = None
    getitem_482: "f32[8, 832, 28, 28]" = convolution_backward_86[0]
    getitem_483: "f32[400, 832, 1, 1]" = convolution_backward_86[1];  convolution_backward_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_86: "b8[8, 832, 28, 28]" = torch.ops.aten.le.Scalar(relu_24, 0);  relu_24 = None
    where_86: "f32[8, 832, 28, 28]" = torch.ops.aten.where.self(le_86, full_default, getitem_482);  le_86 = getitem_482 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_174: "f32[832]" = torch.ops.aten.sum.dim_IntList(where_86, [0, 2, 3])
    sub_455: "f32[8, 832, 28, 28]" = torch.ops.aten.sub.Tensor(cat_13, unsqueeze_1478);  cat_13 = unsqueeze_1478 = None
    mul_1551: "f32[8, 832, 28, 28]" = torch.ops.aten.mul.Tensor(where_86, sub_455)
    sum_175: "f32[832]" = torch.ops.aten.sum.dim_IntList(mul_1551, [0, 2, 3]);  mul_1551 = None
    mul_1552: "f32[832]" = torch.ops.aten.mul.Tensor(sum_174, 0.00015943877551020407)
    unsqueeze_1479: "f32[1, 832]" = torch.ops.aten.unsqueeze.default(mul_1552, 0);  mul_1552 = None
    unsqueeze_1480: "f32[1, 832, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1479, 2);  unsqueeze_1479 = None
    unsqueeze_1481: "f32[1, 832, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1480, 3);  unsqueeze_1480 = None
    mul_1553: "f32[832]" = torch.ops.aten.mul.Tensor(sum_175, 0.00015943877551020407)
    mul_1554: "f32[832]" = torch.ops.aten.mul.Tensor(squeeze_73, squeeze_73)
    mul_1555: "f32[832]" = torch.ops.aten.mul.Tensor(mul_1553, mul_1554);  mul_1553 = mul_1554 = None
    unsqueeze_1482: "f32[1, 832]" = torch.ops.aten.unsqueeze.default(mul_1555, 0);  mul_1555 = None
    unsqueeze_1483: "f32[1, 832, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1482, 2);  unsqueeze_1482 = None
    unsqueeze_1484: "f32[1, 832, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1483, 3);  unsqueeze_1483 = None
    mul_1556: "f32[832]" = torch.ops.aten.mul.Tensor(squeeze_73, primals_49);  primals_49 = None
    unsqueeze_1485: "f32[1, 832]" = torch.ops.aten.unsqueeze.default(mul_1556, 0);  mul_1556 = None
    unsqueeze_1486: "f32[1, 832, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1485, 2);  unsqueeze_1485 = None
    unsqueeze_1487: "f32[1, 832, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1486, 3);  unsqueeze_1486 = None
    mul_1557: "f32[8, 832, 28, 28]" = torch.ops.aten.mul.Tensor(sub_455, unsqueeze_1484);  sub_455 = unsqueeze_1484 = None
    sub_457: "f32[8, 832, 28, 28]" = torch.ops.aten.sub.Tensor(where_86, mul_1557);  where_86 = mul_1557 = None
    sub_458: "f32[8, 832, 28, 28]" = torch.ops.aten.sub.Tensor(sub_457, unsqueeze_1481);  sub_457 = unsqueeze_1481 = None
    mul_1558: "f32[8, 832, 28, 28]" = torch.ops.aten.mul.Tensor(sub_458, unsqueeze_1487);  sub_458 = unsqueeze_1487 = None
    mul_1559: "f32[832]" = torch.ops.aten.mul.Tensor(sum_175, squeeze_73);  sum_175 = squeeze_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    slice_425: "f32[8, 512, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1558, 1, 0, 512)
    slice_426: "f32[8, 320, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1558, 1, 512, 832);  mul_1558 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    add_672: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(add_669, slice_425);  add_669 = slice_425 = None
    add_673: "f32[8, 320, 28, 28]" = torch.ops.aten.add.Tensor(slice_423, slice_426);  slice_423 = slice_426 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    slice_427: "f32[8, 256, 28, 28]" = torch.ops.aten.slice.Tensor(add_673, 1, 0, 256)
    slice_428: "f32[8, 64, 28, 28]" = torch.ops.aten.slice.Tensor(add_673, 1, 256, 320);  add_673 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    slice_scatter_240: "f32[8, 64, 28, 28]" = torch.ops.aten.slice_scatter.default(full_default_272, slice_428, 3, 0, 9223372036854775807);  slice_428 = None
    slice_scatter_242: "f32[8, 576, 28, 28]" = torch.ops.aten.slice_scatter.default(full_default_274, slice_scatter_240, 1, 512, 9223372036854775807);  slice_scatter_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    slice_scatter_246: "f32[8, 576, 28, 28]" = torch.ops.aten.slice_scatter.default(full_default_274, add_672, 1, 0, 512)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    add_674: "f32[8, 576, 28, 28]" = torch.ops.aten.add.Tensor(slice_scatter_242, slice_scatter_246);  slice_scatter_242 = slice_scatter_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_87 = torch.ops.aten.convolution_backward.default(add_674, relu_23, primals_246, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_674 = primals_246 = None
    getitem_485: "f32[8, 400, 28, 28]" = convolution_backward_87[0]
    getitem_486: "f32[576, 400, 1, 1]" = convolution_backward_87[1];  convolution_backward_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_87: "b8[8, 400, 28, 28]" = torch.ops.aten.le.Scalar(relu_23, 0);  relu_23 = None
    where_87: "f32[8, 400, 28, 28]" = torch.ops.aten.where.self(le_87, full_default, getitem_485);  le_87 = getitem_485 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_176: "f32[400]" = torch.ops.aten.sum.dim_IntList(where_87, [0, 2, 3])
    sub_459: "f32[8, 400, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_1490);  convolution_22 = unsqueeze_1490 = None
    mul_1560: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(where_87, sub_459)
    sum_177: "f32[400]" = torch.ops.aten.sum.dim_IntList(mul_1560, [0, 2, 3]);  mul_1560 = None
    mul_1561: "f32[400]" = torch.ops.aten.mul.Tensor(sum_176, 0.00015943877551020407)
    unsqueeze_1491: "f32[1, 400]" = torch.ops.aten.unsqueeze.default(mul_1561, 0);  mul_1561 = None
    unsqueeze_1492: "f32[1, 400, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1491, 2);  unsqueeze_1491 = None
    unsqueeze_1493: "f32[1, 400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1492, 3);  unsqueeze_1492 = None
    mul_1562: "f32[400]" = torch.ops.aten.mul.Tensor(sum_177, 0.00015943877551020407)
    mul_1563: "f32[400]" = torch.ops.aten.mul.Tensor(squeeze_70, squeeze_70)
    mul_1564: "f32[400]" = torch.ops.aten.mul.Tensor(mul_1562, mul_1563);  mul_1562 = mul_1563 = None
    unsqueeze_1494: "f32[1, 400]" = torch.ops.aten.unsqueeze.default(mul_1564, 0);  mul_1564 = None
    unsqueeze_1495: "f32[1, 400, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1494, 2);  unsqueeze_1494 = None
    unsqueeze_1496: "f32[1, 400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1495, 3);  unsqueeze_1495 = None
    mul_1565: "f32[400]" = torch.ops.aten.mul.Tensor(squeeze_70, primals_47);  primals_47 = None
    unsqueeze_1497: "f32[1, 400]" = torch.ops.aten.unsqueeze.default(mul_1565, 0);  mul_1565 = None
    unsqueeze_1498: "f32[1, 400, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1497, 2);  unsqueeze_1497 = None
    unsqueeze_1499: "f32[1, 400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1498, 3);  unsqueeze_1498 = None
    mul_1566: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(sub_459, unsqueeze_1496);  sub_459 = unsqueeze_1496 = None
    sub_461: "f32[8, 400, 28, 28]" = torch.ops.aten.sub.Tensor(where_87, mul_1566);  where_87 = mul_1566 = None
    sub_462: "f32[8, 400, 28, 28]" = torch.ops.aten.sub.Tensor(sub_461, unsqueeze_1493);  sub_461 = unsqueeze_1493 = None
    mul_1567: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(sub_462, unsqueeze_1499);  sub_462 = unsqueeze_1499 = None
    mul_1568: "f32[400]" = torch.ops.aten.mul.Tensor(sum_177, squeeze_70);  sum_177 = squeeze_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_88 = torch.ops.aten.convolution_backward.default(mul_1567, relu_22, primals_245, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False]);  mul_1567 = primals_245 = None
    getitem_488: "f32[8, 400, 28, 28]" = convolution_backward_88[0]
    getitem_489: "f32[400, 8, 3, 3]" = convolution_backward_88[1];  convolution_backward_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_88: "b8[8, 400, 28, 28]" = torch.ops.aten.le.Scalar(relu_22, 0);  relu_22 = None
    where_88: "f32[8, 400, 28, 28]" = torch.ops.aten.where.self(le_88, full_default, getitem_488);  le_88 = getitem_488 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_178: "f32[400]" = torch.ops.aten.sum.dim_IntList(where_88, [0, 2, 3])
    sub_463: "f32[8, 400, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_1502);  convolution_21 = unsqueeze_1502 = None
    mul_1569: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(where_88, sub_463)
    sum_179: "f32[400]" = torch.ops.aten.sum.dim_IntList(mul_1569, [0, 2, 3]);  mul_1569 = None
    mul_1570: "f32[400]" = torch.ops.aten.mul.Tensor(sum_178, 0.00015943877551020407)
    unsqueeze_1503: "f32[1, 400]" = torch.ops.aten.unsqueeze.default(mul_1570, 0);  mul_1570 = None
    unsqueeze_1504: "f32[1, 400, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1503, 2);  unsqueeze_1503 = None
    unsqueeze_1505: "f32[1, 400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1504, 3);  unsqueeze_1504 = None
    mul_1571: "f32[400]" = torch.ops.aten.mul.Tensor(sum_179, 0.00015943877551020407)
    mul_1572: "f32[400]" = torch.ops.aten.mul.Tensor(squeeze_67, squeeze_67)
    mul_1573: "f32[400]" = torch.ops.aten.mul.Tensor(mul_1571, mul_1572);  mul_1571 = mul_1572 = None
    unsqueeze_1506: "f32[1, 400]" = torch.ops.aten.unsqueeze.default(mul_1573, 0);  mul_1573 = None
    unsqueeze_1507: "f32[1, 400, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1506, 2);  unsqueeze_1506 = None
    unsqueeze_1508: "f32[1, 400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1507, 3);  unsqueeze_1507 = None
    mul_1574: "f32[400]" = torch.ops.aten.mul.Tensor(squeeze_67, primals_45);  primals_45 = None
    unsqueeze_1509: "f32[1, 400]" = torch.ops.aten.unsqueeze.default(mul_1574, 0);  mul_1574 = None
    unsqueeze_1510: "f32[1, 400, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1509, 2);  unsqueeze_1509 = None
    unsqueeze_1511: "f32[1, 400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1510, 3);  unsqueeze_1510 = None
    mul_1575: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(sub_463, unsqueeze_1508);  sub_463 = unsqueeze_1508 = None
    sub_465: "f32[8, 400, 28, 28]" = torch.ops.aten.sub.Tensor(where_88, mul_1575);  where_88 = mul_1575 = None
    sub_466: "f32[8, 400, 28, 28]" = torch.ops.aten.sub.Tensor(sub_465, unsqueeze_1505);  sub_465 = unsqueeze_1505 = None
    mul_1576: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(sub_466, unsqueeze_1511);  sub_466 = unsqueeze_1511 = None
    mul_1577: "f32[400]" = torch.ops.aten.mul.Tensor(sum_179, squeeze_67);  sum_179 = squeeze_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_89 = torch.ops.aten.convolution_backward.default(mul_1576, relu_21, primals_244, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1576 = primals_244 = None
    getitem_491: "f32[8, 768, 28, 28]" = convolution_backward_89[0]
    getitem_492: "f32[400, 768, 1, 1]" = convolution_backward_89[1];  convolution_backward_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_89: "b8[8, 768, 28, 28]" = torch.ops.aten.le.Scalar(relu_21, 0);  relu_21 = None
    where_89: "f32[8, 768, 28, 28]" = torch.ops.aten.where.self(le_89, full_default, getitem_491);  le_89 = getitem_491 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_180: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_89, [0, 2, 3])
    sub_467: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(cat_11, unsqueeze_1514);  cat_11 = unsqueeze_1514 = None
    mul_1578: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(where_89, sub_467)
    sum_181: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1578, [0, 2, 3]);  mul_1578 = None
    mul_1579: "f32[768]" = torch.ops.aten.mul.Tensor(sum_180, 0.00015943877551020407)
    unsqueeze_1515: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1579, 0);  mul_1579 = None
    unsqueeze_1516: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1515, 2);  unsqueeze_1515 = None
    unsqueeze_1517: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1516, 3);  unsqueeze_1516 = None
    mul_1580: "f32[768]" = torch.ops.aten.mul.Tensor(sum_181, 0.00015943877551020407)
    mul_1581: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_64, squeeze_64)
    mul_1582: "f32[768]" = torch.ops.aten.mul.Tensor(mul_1580, mul_1581);  mul_1580 = mul_1581 = None
    unsqueeze_1518: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1582, 0);  mul_1582 = None
    unsqueeze_1519: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1518, 2);  unsqueeze_1518 = None
    unsqueeze_1520: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1519, 3);  unsqueeze_1519 = None
    mul_1583: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_64, primals_43);  primals_43 = None
    unsqueeze_1521: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_1583, 0);  mul_1583 = None
    unsqueeze_1522: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1521, 2);  unsqueeze_1521 = None
    unsqueeze_1523: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1522, 3);  unsqueeze_1522 = None
    mul_1584: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_467, unsqueeze_1520);  sub_467 = unsqueeze_1520 = None
    sub_469: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(where_89, mul_1584);  where_89 = mul_1584 = None
    sub_470: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(sub_469, unsqueeze_1517);  sub_469 = unsqueeze_1517 = None
    mul_1585: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_470, unsqueeze_1523);  sub_470 = unsqueeze_1523 = None
    mul_1586: "f32[768]" = torch.ops.aten.mul.Tensor(sum_181, squeeze_64);  sum_181 = squeeze_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    slice_429: "f32[8, 512, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1585, 1, 0, 512)
    slice_430: "f32[8, 256, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1585, 1, 512, 768);  mul_1585 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    add_675: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(add_672, slice_429);  add_672 = slice_429 = None
    add_676: "f32[8, 256, 28, 28]" = torch.ops.aten.add.Tensor(slice_427, slice_430);  slice_427 = slice_430 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    slice_431: "f32[8, 192, 28, 28]" = torch.ops.aten.slice.Tensor(add_676, 1, 0, 192)
    slice_432: "f32[8, 64, 28, 28]" = torch.ops.aten.slice.Tensor(add_676, 1, 192, 256);  add_676 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    slice_scatter_248: "f32[8, 64, 28, 28]" = torch.ops.aten.slice_scatter.default(full_default_272, slice_432, 3, 0, 9223372036854775807);  slice_432 = None
    slice_scatter_250: "f32[8, 576, 28, 28]" = torch.ops.aten.slice_scatter.default(full_default_274, slice_scatter_248, 1, 512, 9223372036854775807);  slice_scatter_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    slice_scatter_254: "f32[8, 576, 28, 28]" = torch.ops.aten.slice_scatter.default(full_default_274, add_675, 1, 0, 512)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    add_677: "f32[8, 576, 28, 28]" = torch.ops.aten.add.Tensor(slice_scatter_250, slice_scatter_254);  slice_scatter_250 = slice_scatter_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_90 = torch.ops.aten.convolution_backward.default(add_677, relu_20, primals_243, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_677 = primals_243 = None
    getitem_494: "f32[8, 400, 28, 28]" = convolution_backward_90[0]
    getitem_495: "f32[576, 400, 1, 1]" = convolution_backward_90[1];  convolution_backward_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_90: "b8[8, 400, 28, 28]" = torch.ops.aten.le.Scalar(relu_20, 0);  relu_20 = None
    where_90: "f32[8, 400, 28, 28]" = torch.ops.aten.where.self(le_90, full_default, getitem_494);  le_90 = getitem_494 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_182: "f32[400]" = torch.ops.aten.sum.dim_IntList(where_90, [0, 2, 3])
    sub_471: "f32[8, 400, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_1526);  convolution_19 = unsqueeze_1526 = None
    mul_1587: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(where_90, sub_471)
    sum_183: "f32[400]" = torch.ops.aten.sum.dim_IntList(mul_1587, [0, 2, 3]);  mul_1587 = None
    mul_1588: "f32[400]" = torch.ops.aten.mul.Tensor(sum_182, 0.00015943877551020407)
    unsqueeze_1527: "f32[1, 400]" = torch.ops.aten.unsqueeze.default(mul_1588, 0);  mul_1588 = None
    unsqueeze_1528: "f32[1, 400, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1527, 2);  unsqueeze_1527 = None
    unsqueeze_1529: "f32[1, 400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1528, 3);  unsqueeze_1528 = None
    mul_1589: "f32[400]" = torch.ops.aten.mul.Tensor(sum_183, 0.00015943877551020407)
    mul_1590: "f32[400]" = torch.ops.aten.mul.Tensor(squeeze_61, squeeze_61)
    mul_1591: "f32[400]" = torch.ops.aten.mul.Tensor(mul_1589, mul_1590);  mul_1589 = mul_1590 = None
    unsqueeze_1530: "f32[1, 400]" = torch.ops.aten.unsqueeze.default(mul_1591, 0);  mul_1591 = None
    unsqueeze_1531: "f32[1, 400, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1530, 2);  unsqueeze_1530 = None
    unsqueeze_1532: "f32[1, 400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1531, 3);  unsqueeze_1531 = None
    mul_1592: "f32[400]" = torch.ops.aten.mul.Tensor(squeeze_61, primals_41);  primals_41 = None
    unsqueeze_1533: "f32[1, 400]" = torch.ops.aten.unsqueeze.default(mul_1592, 0);  mul_1592 = None
    unsqueeze_1534: "f32[1, 400, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1533, 2);  unsqueeze_1533 = None
    unsqueeze_1535: "f32[1, 400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1534, 3);  unsqueeze_1534 = None
    mul_1593: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(sub_471, unsqueeze_1532);  sub_471 = unsqueeze_1532 = None
    sub_473: "f32[8, 400, 28, 28]" = torch.ops.aten.sub.Tensor(where_90, mul_1593);  where_90 = mul_1593 = None
    sub_474: "f32[8, 400, 28, 28]" = torch.ops.aten.sub.Tensor(sub_473, unsqueeze_1529);  sub_473 = unsqueeze_1529 = None
    mul_1594: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(sub_474, unsqueeze_1535);  sub_474 = unsqueeze_1535 = None
    mul_1595: "f32[400]" = torch.ops.aten.mul.Tensor(sum_183, squeeze_61);  sum_183 = squeeze_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_91 = torch.ops.aten.convolution_backward.default(mul_1594, relu_19, primals_242, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False]);  mul_1594 = primals_242 = None
    getitem_497: "f32[8, 400, 28, 28]" = convolution_backward_91[0]
    getitem_498: "f32[400, 8, 3, 3]" = convolution_backward_91[1];  convolution_backward_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_91: "b8[8, 400, 28, 28]" = torch.ops.aten.le.Scalar(relu_19, 0);  relu_19 = None
    where_91: "f32[8, 400, 28, 28]" = torch.ops.aten.where.self(le_91, full_default, getitem_497);  le_91 = getitem_497 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_184: "f32[400]" = torch.ops.aten.sum.dim_IntList(where_91, [0, 2, 3])
    sub_475: "f32[8, 400, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_1538);  convolution_18 = unsqueeze_1538 = None
    mul_1596: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(where_91, sub_475)
    sum_185: "f32[400]" = torch.ops.aten.sum.dim_IntList(mul_1596, [0, 2, 3]);  mul_1596 = None
    mul_1597: "f32[400]" = torch.ops.aten.mul.Tensor(sum_184, 0.00015943877551020407)
    unsqueeze_1539: "f32[1, 400]" = torch.ops.aten.unsqueeze.default(mul_1597, 0);  mul_1597 = None
    unsqueeze_1540: "f32[1, 400, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1539, 2);  unsqueeze_1539 = None
    unsqueeze_1541: "f32[1, 400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1540, 3);  unsqueeze_1540 = None
    mul_1598: "f32[400]" = torch.ops.aten.mul.Tensor(sum_185, 0.00015943877551020407)
    mul_1599: "f32[400]" = torch.ops.aten.mul.Tensor(squeeze_58, squeeze_58)
    mul_1600: "f32[400]" = torch.ops.aten.mul.Tensor(mul_1598, mul_1599);  mul_1598 = mul_1599 = None
    unsqueeze_1542: "f32[1, 400]" = torch.ops.aten.unsqueeze.default(mul_1600, 0);  mul_1600 = None
    unsqueeze_1543: "f32[1, 400, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1542, 2);  unsqueeze_1542 = None
    unsqueeze_1544: "f32[1, 400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1543, 3);  unsqueeze_1543 = None
    mul_1601: "f32[400]" = torch.ops.aten.mul.Tensor(squeeze_58, primals_39);  primals_39 = None
    unsqueeze_1545: "f32[1, 400]" = torch.ops.aten.unsqueeze.default(mul_1601, 0);  mul_1601 = None
    unsqueeze_1546: "f32[1, 400, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1545, 2);  unsqueeze_1545 = None
    unsqueeze_1547: "f32[1, 400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1546, 3);  unsqueeze_1546 = None
    mul_1602: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(sub_475, unsqueeze_1544);  sub_475 = unsqueeze_1544 = None
    sub_477: "f32[8, 400, 28, 28]" = torch.ops.aten.sub.Tensor(where_91, mul_1602);  where_91 = mul_1602 = None
    sub_478: "f32[8, 400, 28, 28]" = torch.ops.aten.sub.Tensor(sub_477, unsqueeze_1541);  sub_477 = unsqueeze_1541 = None
    mul_1603: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(sub_478, unsqueeze_1547);  sub_478 = unsqueeze_1547 = None
    mul_1604: "f32[400]" = torch.ops.aten.mul.Tensor(sum_185, squeeze_58);  sum_185 = squeeze_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_92 = torch.ops.aten.convolution_backward.default(mul_1603, relu_18, primals_241, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1603 = primals_241 = None
    getitem_500: "f32[8, 704, 28, 28]" = convolution_backward_92[0]
    getitem_501: "f32[400, 704, 1, 1]" = convolution_backward_92[1];  convolution_backward_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_92: "b8[8, 704, 28, 28]" = torch.ops.aten.le.Scalar(relu_18, 0);  relu_18 = None
    where_92: "f32[8, 704, 28, 28]" = torch.ops.aten.where.self(le_92, full_default, getitem_500);  le_92 = getitem_500 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_186: "f32[704]" = torch.ops.aten.sum.dim_IntList(where_92, [0, 2, 3])
    sub_479: "f32[8, 704, 28, 28]" = torch.ops.aten.sub.Tensor(cat_9, unsqueeze_1550);  cat_9 = unsqueeze_1550 = None
    mul_1605: "f32[8, 704, 28, 28]" = torch.ops.aten.mul.Tensor(where_92, sub_479)
    sum_187: "f32[704]" = torch.ops.aten.sum.dim_IntList(mul_1605, [0, 2, 3]);  mul_1605 = None
    mul_1606: "f32[704]" = torch.ops.aten.mul.Tensor(sum_186, 0.00015943877551020407)
    unsqueeze_1551: "f32[1, 704]" = torch.ops.aten.unsqueeze.default(mul_1606, 0);  mul_1606 = None
    unsqueeze_1552: "f32[1, 704, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1551, 2);  unsqueeze_1551 = None
    unsqueeze_1553: "f32[1, 704, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1552, 3);  unsqueeze_1552 = None
    mul_1607: "f32[704]" = torch.ops.aten.mul.Tensor(sum_187, 0.00015943877551020407)
    mul_1608: "f32[704]" = torch.ops.aten.mul.Tensor(squeeze_55, squeeze_55)
    mul_1609: "f32[704]" = torch.ops.aten.mul.Tensor(mul_1607, mul_1608);  mul_1607 = mul_1608 = None
    unsqueeze_1554: "f32[1, 704]" = torch.ops.aten.unsqueeze.default(mul_1609, 0);  mul_1609 = None
    unsqueeze_1555: "f32[1, 704, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1554, 2);  unsqueeze_1554 = None
    unsqueeze_1556: "f32[1, 704, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1555, 3);  unsqueeze_1555 = None
    mul_1610: "f32[704]" = torch.ops.aten.mul.Tensor(squeeze_55, primals_37);  primals_37 = None
    unsqueeze_1557: "f32[1, 704]" = torch.ops.aten.unsqueeze.default(mul_1610, 0);  mul_1610 = None
    unsqueeze_1558: "f32[1, 704, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1557, 2);  unsqueeze_1557 = None
    unsqueeze_1559: "f32[1, 704, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1558, 3);  unsqueeze_1558 = None
    mul_1611: "f32[8, 704, 28, 28]" = torch.ops.aten.mul.Tensor(sub_479, unsqueeze_1556);  sub_479 = unsqueeze_1556 = None
    sub_481: "f32[8, 704, 28, 28]" = torch.ops.aten.sub.Tensor(where_92, mul_1611);  where_92 = mul_1611 = None
    sub_482: "f32[8, 704, 28, 28]" = torch.ops.aten.sub.Tensor(sub_481, unsqueeze_1553);  sub_481 = unsqueeze_1553 = None
    mul_1612: "f32[8, 704, 28, 28]" = torch.ops.aten.mul.Tensor(sub_482, unsqueeze_1559);  sub_482 = unsqueeze_1559 = None
    mul_1613: "f32[704]" = torch.ops.aten.mul.Tensor(sum_187, squeeze_55);  sum_187 = squeeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    slice_433: "f32[8, 512, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1612, 1, 0, 512)
    slice_434: "f32[8, 192, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1612, 1, 512, 704);  mul_1612 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    add_678: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(add_675, slice_433);  add_675 = slice_433 = None
    add_679: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(slice_431, slice_434);  slice_431 = slice_434 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    slice_435: "f32[8, 128, 28, 28]" = torch.ops.aten.slice.Tensor(add_679, 1, 0, 128)
    slice_436: "f32[8, 64, 28, 28]" = torch.ops.aten.slice.Tensor(add_679, 1, 128, 192);  add_679 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    slice_scatter_256: "f32[8, 64, 28, 28]" = torch.ops.aten.slice_scatter.default(full_default_272, slice_436, 3, 0, 9223372036854775807);  full_default_272 = slice_436 = None
    slice_scatter_258: "f32[8, 576, 28, 28]" = torch.ops.aten.slice_scatter.default(full_default_274, slice_scatter_256, 1, 512, 9223372036854775807);  slice_scatter_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    slice_scatter_262: "f32[8, 576, 28, 28]" = torch.ops.aten.slice_scatter.default(full_default_274, add_678, 1, 0, 512);  full_default_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    add_680: "f32[8, 576, 28, 28]" = torch.ops.aten.add.Tensor(slice_scatter_258, slice_scatter_262);  slice_scatter_258 = slice_scatter_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_93 = torch.ops.aten.convolution_backward.default(add_680, relu_17, primals_240, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_680 = primals_240 = None
    getitem_503: "f32[8, 400, 28, 28]" = convolution_backward_93[0]
    getitem_504: "f32[576, 400, 1, 1]" = convolution_backward_93[1];  convolution_backward_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_93: "b8[8, 400, 28, 28]" = torch.ops.aten.le.Scalar(relu_17, 0);  relu_17 = None
    where_93: "f32[8, 400, 28, 28]" = torch.ops.aten.where.self(le_93, full_default, getitem_503);  le_93 = getitem_503 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_188: "f32[400]" = torch.ops.aten.sum.dim_IntList(where_93, [0, 2, 3])
    sub_483: "f32[8, 400, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_1562);  convolution_16 = unsqueeze_1562 = None
    mul_1614: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(where_93, sub_483)
    sum_189: "f32[400]" = torch.ops.aten.sum.dim_IntList(mul_1614, [0, 2, 3]);  mul_1614 = None
    mul_1615: "f32[400]" = torch.ops.aten.mul.Tensor(sum_188, 0.00015943877551020407)
    unsqueeze_1563: "f32[1, 400]" = torch.ops.aten.unsqueeze.default(mul_1615, 0);  mul_1615 = None
    unsqueeze_1564: "f32[1, 400, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1563, 2);  unsqueeze_1563 = None
    unsqueeze_1565: "f32[1, 400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1564, 3);  unsqueeze_1564 = None
    mul_1616: "f32[400]" = torch.ops.aten.mul.Tensor(sum_189, 0.00015943877551020407)
    mul_1617: "f32[400]" = torch.ops.aten.mul.Tensor(squeeze_52, squeeze_52)
    mul_1618: "f32[400]" = torch.ops.aten.mul.Tensor(mul_1616, mul_1617);  mul_1616 = mul_1617 = None
    unsqueeze_1566: "f32[1, 400]" = torch.ops.aten.unsqueeze.default(mul_1618, 0);  mul_1618 = None
    unsqueeze_1567: "f32[1, 400, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1566, 2);  unsqueeze_1566 = None
    unsqueeze_1568: "f32[1, 400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1567, 3);  unsqueeze_1567 = None
    mul_1619: "f32[400]" = torch.ops.aten.mul.Tensor(squeeze_52, primals_35);  primals_35 = None
    unsqueeze_1569: "f32[1, 400]" = torch.ops.aten.unsqueeze.default(mul_1619, 0);  mul_1619 = None
    unsqueeze_1570: "f32[1, 400, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1569, 2);  unsqueeze_1569 = None
    unsqueeze_1571: "f32[1, 400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1570, 3);  unsqueeze_1570 = None
    mul_1620: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(sub_483, unsqueeze_1568);  sub_483 = unsqueeze_1568 = None
    sub_485: "f32[8, 400, 28, 28]" = torch.ops.aten.sub.Tensor(where_93, mul_1620);  where_93 = mul_1620 = None
    sub_486: "f32[8, 400, 28, 28]" = torch.ops.aten.sub.Tensor(sub_485, unsqueeze_1565);  sub_485 = unsqueeze_1565 = None
    mul_1621: "f32[8, 400, 28, 28]" = torch.ops.aten.mul.Tensor(sub_486, unsqueeze_1571);  sub_486 = unsqueeze_1571 = None
    mul_1622: "f32[400]" = torch.ops.aten.mul.Tensor(sum_189, squeeze_52);  sum_189 = squeeze_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_94 = torch.ops.aten.convolution_backward.default(mul_1621, relu_16, primals_239, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False]);  mul_1621 = primals_239 = None
    getitem_506: "f32[8, 400, 56, 56]" = convolution_backward_94[0]
    getitem_507: "f32[400, 8, 3, 3]" = convolution_backward_94[1];  convolution_backward_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_94: "b8[8, 400, 56, 56]" = torch.ops.aten.le.Scalar(relu_16, 0);  relu_16 = None
    where_94: "f32[8, 400, 56, 56]" = torch.ops.aten.where.self(le_94, full_default, getitem_506);  le_94 = getitem_506 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_190: "f32[400]" = torch.ops.aten.sum.dim_IntList(where_94, [0, 2, 3])
    sub_487: "f32[8, 400, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_1574);  convolution_15 = unsqueeze_1574 = None
    mul_1623: "f32[8, 400, 56, 56]" = torch.ops.aten.mul.Tensor(where_94, sub_487)
    sum_191: "f32[400]" = torch.ops.aten.sum.dim_IntList(mul_1623, [0, 2, 3]);  mul_1623 = None
    mul_1624: "f32[400]" = torch.ops.aten.mul.Tensor(sum_190, 3.985969387755102e-05)
    unsqueeze_1575: "f32[1, 400]" = torch.ops.aten.unsqueeze.default(mul_1624, 0);  mul_1624 = None
    unsqueeze_1576: "f32[1, 400, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1575, 2);  unsqueeze_1575 = None
    unsqueeze_1577: "f32[1, 400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1576, 3);  unsqueeze_1576 = None
    mul_1625: "f32[400]" = torch.ops.aten.mul.Tensor(sum_191, 3.985969387755102e-05)
    mul_1626: "f32[400]" = torch.ops.aten.mul.Tensor(squeeze_49, squeeze_49)
    mul_1627: "f32[400]" = torch.ops.aten.mul.Tensor(mul_1625, mul_1626);  mul_1625 = mul_1626 = None
    unsqueeze_1578: "f32[1, 400]" = torch.ops.aten.unsqueeze.default(mul_1627, 0);  mul_1627 = None
    unsqueeze_1579: "f32[1, 400, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1578, 2);  unsqueeze_1578 = None
    unsqueeze_1580: "f32[1, 400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1579, 3);  unsqueeze_1579 = None
    mul_1628: "f32[400]" = torch.ops.aten.mul.Tensor(squeeze_49, primals_33);  primals_33 = None
    unsqueeze_1581: "f32[1, 400]" = torch.ops.aten.unsqueeze.default(mul_1628, 0);  mul_1628 = None
    unsqueeze_1582: "f32[1, 400, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1581, 2);  unsqueeze_1581 = None
    unsqueeze_1583: "f32[1, 400, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1582, 3);  unsqueeze_1582 = None
    mul_1629: "f32[8, 400, 56, 56]" = torch.ops.aten.mul.Tensor(sub_487, unsqueeze_1580);  sub_487 = unsqueeze_1580 = None
    sub_489: "f32[8, 400, 56, 56]" = torch.ops.aten.sub.Tensor(where_94, mul_1629);  where_94 = mul_1629 = None
    sub_490: "f32[8, 400, 56, 56]" = torch.ops.aten.sub.Tensor(sub_489, unsqueeze_1577);  sub_489 = unsqueeze_1577 = None
    mul_1630: "f32[8, 400, 56, 56]" = torch.ops.aten.mul.Tensor(sub_490, unsqueeze_1583);  sub_490 = unsqueeze_1583 = None
    mul_1631: "f32[400]" = torch.ops.aten.mul.Tensor(sum_191, squeeze_49);  sum_191 = squeeze_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_95 = torch.ops.aten.convolution_backward.default(mul_1630, relu_15, primals_238, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1630 = primals_238 = None
    getitem_509: "f32[8, 376, 56, 56]" = convolution_backward_95[0]
    getitem_510: "f32[400, 376, 1, 1]" = convolution_backward_95[1];  convolution_backward_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_95: "b8[8, 376, 56, 56]" = torch.ops.aten.le.Scalar(relu_15, 0);  relu_15 = None
    where_95: "f32[8, 376, 56, 56]" = torch.ops.aten.where.self(le_95, full_default, getitem_509);  le_95 = getitem_509 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_192: "f32[376]" = torch.ops.aten.sum.dim_IntList(where_95, [0, 2, 3])
    sub_491: "f32[8, 376, 56, 56]" = torch.ops.aten.sub.Tensor(cat_7, unsqueeze_1586);  cat_7 = unsqueeze_1586 = None
    mul_1632: "f32[8, 376, 56, 56]" = torch.ops.aten.mul.Tensor(where_95, sub_491)
    sum_193: "f32[376]" = torch.ops.aten.sum.dim_IntList(mul_1632, [0, 2, 3]);  mul_1632 = None
    mul_1633: "f32[376]" = torch.ops.aten.mul.Tensor(sum_192, 3.985969387755102e-05)
    unsqueeze_1587: "f32[1, 376]" = torch.ops.aten.unsqueeze.default(mul_1633, 0);  mul_1633 = None
    unsqueeze_1588: "f32[1, 376, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1587, 2);  unsqueeze_1587 = None
    unsqueeze_1589: "f32[1, 376, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1588, 3);  unsqueeze_1588 = None
    mul_1634: "f32[376]" = torch.ops.aten.mul.Tensor(sum_193, 3.985969387755102e-05)
    mul_1635: "f32[376]" = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
    mul_1636: "f32[376]" = torch.ops.aten.mul.Tensor(mul_1634, mul_1635);  mul_1634 = None
    unsqueeze_1590: "f32[1, 376]" = torch.ops.aten.unsqueeze.default(mul_1636, 0);  mul_1636 = None
    unsqueeze_1591: "f32[1, 376, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1590, 2);  unsqueeze_1590 = None
    unsqueeze_1592: "f32[1, 376, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1591, 3);  unsqueeze_1591 = None
    mul_1637: "f32[376]" = torch.ops.aten.mul.Tensor(squeeze_43, primals_31);  primals_31 = None
    unsqueeze_1593: "f32[1, 376]" = torch.ops.aten.unsqueeze.default(mul_1637, 0);  mul_1637 = None
    unsqueeze_1594: "f32[1, 376, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1593, 2);  unsqueeze_1593 = None
    unsqueeze_1595: "f32[1, 376, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1594, 3);  unsqueeze_1594 = None
    mul_1638: "f32[8, 376, 56, 56]" = torch.ops.aten.mul.Tensor(sub_491, unsqueeze_1592);  unsqueeze_1592 = None
    sub_493: "f32[8, 376, 56, 56]" = torch.ops.aten.sub.Tensor(where_95, mul_1638);  where_95 = mul_1638 = None
    sub_494: "f32[8, 376, 56, 56]" = torch.ops.aten.sub.Tensor(sub_493, unsqueeze_1589);  sub_493 = unsqueeze_1589 = None
    mul_1639: "f32[8, 376, 56, 56]" = torch.ops.aten.mul.Tensor(sub_494, unsqueeze_1595);  sub_494 = unsqueeze_1595 = None
    mul_1640: "f32[376]" = torch.ops.aten.mul.Tensor(sum_193, squeeze_43);  sum_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:134, code: x_s2 = x_s[:, self.num_1x1_c:, :, :]
    full_default_360: "f32[8, 128, 28, 28]" = torch.ops.aten.full.default([8, 128, 28, 28], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_264: "f32[8, 128, 28, 28]" = torch.ops.aten.slice_scatter.default(full_default_360, slice_435, 3, 0, 9223372036854775807);  full_default_360 = slice_435 = None
    full_default_362: "f32[8, 640, 28, 28]" = torch.ops.aten.full.default([8, 640, 28, 28], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_266: "f32[8, 640, 28, 28]" = torch.ops.aten.slice_scatter.default(full_default_362, slice_scatter_264, 1, 512, 9223372036854775807);  slice_scatter_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:133, code: x_s1 = x_s[:, :self.num_1x1_c, :, :]
    slice_scatter_270: "f32[8, 640, 28, 28]" = torch.ops.aten.slice_scatter.default(full_default_362, add_678, 1, 0, 512);  full_default_362 = add_678 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:133, code: x_s1 = x_s[:, :self.num_1x1_c, :, :]
    add_681: "f32[8, 640, 28, 28]" = torch.ops.aten.add.Tensor(slice_scatter_266, slice_scatter_270);  slice_scatter_266 = slice_scatter_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_96 = torch.ops.aten.convolution_backward.default(add_681, relu_14, primals_237, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_681 = primals_237 = None
    getitem_512: "f32[8, 376, 56, 56]" = convolution_backward_96[0]
    getitem_513: "f32[640, 376, 1, 1]" = convolution_backward_96[1];  convolution_backward_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_96: "b8[8, 376, 56, 56]" = torch.ops.aten.le.Scalar(relu_14, 0);  relu_14 = None
    where_96: "f32[8, 376, 56, 56]" = torch.ops.aten.where.self(le_96, full_default, getitem_512);  le_96 = getitem_512 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_194: "f32[376]" = torch.ops.aten.sum.dim_IntList(where_96, [0, 2, 3])
    mul_1641: "f32[8, 376, 56, 56]" = torch.ops.aten.mul.Tensor(where_96, sub_491)
    sum_195: "f32[376]" = torch.ops.aten.sum.dim_IntList(mul_1641, [0, 2, 3]);  mul_1641 = None
    mul_1642: "f32[376]" = torch.ops.aten.mul.Tensor(sum_194, 3.985969387755102e-05)
    unsqueeze_1599: "f32[1, 376]" = torch.ops.aten.unsqueeze.default(mul_1642, 0);  mul_1642 = None
    unsqueeze_1600: "f32[1, 376, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1599, 2);  unsqueeze_1599 = None
    unsqueeze_1601: "f32[1, 376, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1600, 3);  unsqueeze_1600 = None
    mul_1643: "f32[376]" = torch.ops.aten.mul.Tensor(sum_195, 3.985969387755102e-05)
    mul_1645: "f32[376]" = torch.ops.aten.mul.Tensor(mul_1643, mul_1635);  mul_1643 = mul_1635 = None
    unsqueeze_1602: "f32[1, 376]" = torch.ops.aten.unsqueeze.default(mul_1645, 0);  mul_1645 = None
    unsqueeze_1603: "f32[1, 376, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1602, 2);  unsqueeze_1602 = None
    unsqueeze_1604: "f32[1, 376, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1603, 3);  unsqueeze_1603 = None
    mul_1646: "f32[376]" = torch.ops.aten.mul.Tensor(squeeze_43, primals_29);  primals_29 = None
    unsqueeze_1605: "f32[1, 376]" = torch.ops.aten.unsqueeze.default(mul_1646, 0);  mul_1646 = None
    unsqueeze_1606: "f32[1, 376, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1605, 2);  unsqueeze_1605 = None
    unsqueeze_1607: "f32[1, 376, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1606, 3);  unsqueeze_1606 = None
    mul_1647: "f32[8, 376, 56, 56]" = torch.ops.aten.mul.Tensor(sub_491, unsqueeze_1604);  sub_491 = unsqueeze_1604 = None
    sub_497: "f32[8, 376, 56, 56]" = torch.ops.aten.sub.Tensor(where_96, mul_1647);  where_96 = mul_1647 = None
    sub_498: "f32[8, 376, 56, 56]" = torch.ops.aten.sub.Tensor(sub_497, unsqueeze_1601);  sub_497 = unsqueeze_1601 = None
    mul_1648: "f32[8, 376, 56, 56]" = torch.ops.aten.mul.Tensor(sub_498, unsqueeze_1607);  sub_498 = unsqueeze_1607 = None
    mul_1649: "f32[376]" = torch.ops.aten.mul.Tensor(sum_195, squeeze_43);  sum_195 = squeeze_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_682: "f32[8, 376, 56, 56]" = torch.ops.aten.add.Tensor(mul_1639, mul_1648);  mul_1639 = mul_1648 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    slice_437: "f32[8, 256, 56, 56]" = torch.ops.aten.slice.Tensor(add_682, 1, 0, 256)
    slice_438: "f32[8, 120, 56, 56]" = torch.ops.aten.slice.Tensor(add_682, 1, 256, 376);  add_682 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    slice_439: "f32[8, 100, 56, 56]" = torch.ops.aten.slice.Tensor(slice_438, 1, 0, 100)
    slice_440: "f32[8, 20, 56, 56]" = torch.ops.aten.slice.Tensor(slice_438, 1, 100, 120);  slice_438 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    full_default_369: "f32[8, 20, 56, 56]" = torch.ops.aten.full.default([8, 20, 56, 56], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_272: "f32[8, 20, 56, 56]" = torch.ops.aten.slice_scatter.default(full_default_369, slice_440, 3, 0, 9223372036854775807);  slice_440 = None
    full_default_371: "f32[8, 276, 56, 56]" = torch.ops.aten.full.default([8, 276, 56, 56], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_274: "f32[8, 276, 56, 56]" = torch.ops.aten.slice_scatter.default(full_default_371, slice_scatter_272, 1, 256, 9223372036854775807);  slice_scatter_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    full_default_373: "f32[8, 256, 56, 56]" = torch.ops.aten.full.default([8, 256, 56, 56], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_276: "f32[8, 256, 56, 56]" = torch.ops.aten.slice_scatter.default(full_default_373, slice_437, 3, 0, 9223372036854775807);  full_default_373 = None
    slice_scatter_278: "f32[8, 276, 56, 56]" = torch.ops.aten.slice_scatter.default(full_default_371, slice_scatter_276, 1, 0, 256);  slice_scatter_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    add_683: "f32[8, 276, 56, 56]" = torch.ops.aten.add.Tensor(slice_scatter_274, slice_scatter_278);  slice_scatter_274 = slice_scatter_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_97 = torch.ops.aten.convolution_backward.default(add_683, relu_13, primals_236, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_683 = primals_236 = None
    getitem_515: "f32[8, 200, 56, 56]" = convolution_backward_97[0]
    getitem_516: "f32[276, 200, 1, 1]" = convolution_backward_97[1];  convolution_backward_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_97: "b8[8, 200, 56, 56]" = torch.ops.aten.le.Scalar(relu_13, 0);  relu_13 = None
    where_97: "f32[8, 200, 56, 56]" = torch.ops.aten.where.self(le_97, full_default, getitem_515);  le_97 = getitem_515 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_196: "f32[200]" = torch.ops.aten.sum.dim_IntList(where_97, [0, 2, 3])
    sub_499: "f32[8, 200, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_1610);  convolution_12 = unsqueeze_1610 = None
    mul_1650: "f32[8, 200, 56, 56]" = torch.ops.aten.mul.Tensor(where_97, sub_499)
    sum_197: "f32[200]" = torch.ops.aten.sum.dim_IntList(mul_1650, [0, 2, 3]);  mul_1650 = None
    mul_1651: "f32[200]" = torch.ops.aten.mul.Tensor(sum_196, 3.985969387755102e-05)
    unsqueeze_1611: "f32[1, 200]" = torch.ops.aten.unsqueeze.default(mul_1651, 0);  mul_1651 = None
    unsqueeze_1612: "f32[1, 200, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1611, 2);  unsqueeze_1611 = None
    unsqueeze_1613: "f32[1, 200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1612, 3);  unsqueeze_1612 = None
    mul_1652: "f32[200]" = torch.ops.aten.mul.Tensor(sum_197, 3.985969387755102e-05)
    mul_1653: "f32[200]" = torch.ops.aten.mul.Tensor(squeeze_40, squeeze_40)
    mul_1654: "f32[200]" = torch.ops.aten.mul.Tensor(mul_1652, mul_1653);  mul_1652 = mul_1653 = None
    unsqueeze_1614: "f32[1, 200]" = torch.ops.aten.unsqueeze.default(mul_1654, 0);  mul_1654 = None
    unsqueeze_1615: "f32[1, 200, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1614, 2);  unsqueeze_1614 = None
    unsqueeze_1616: "f32[1, 200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1615, 3);  unsqueeze_1615 = None
    mul_1655: "f32[200]" = torch.ops.aten.mul.Tensor(squeeze_40, primals_27);  primals_27 = None
    unsqueeze_1617: "f32[1, 200]" = torch.ops.aten.unsqueeze.default(mul_1655, 0);  mul_1655 = None
    unsqueeze_1618: "f32[1, 200, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1617, 2);  unsqueeze_1617 = None
    unsqueeze_1619: "f32[1, 200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1618, 3);  unsqueeze_1618 = None
    mul_1656: "f32[8, 200, 56, 56]" = torch.ops.aten.mul.Tensor(sub_499, unsqueeze_1616);  sub_499 = unsqueeze_1616 = None
    sub_501: "f32[8, 200, 56, 56]" = torch.ops.aten.sub.Tensor(where_97, mul_1656);  where_97 = mul_1656 = None
    sub_502: "f32[8, 200, 56, 56]" = torch.ops.aten.sub.Tensor(sub_501, unsqueeze_1613);  sub_501 = unsqueeze_1613 = None
    mul_1657: "f32[8, 200, 56, 56]" = torch.ops.aten.mul.Tensor(sub_502, unsqueeze_1619);  sub_502 = unsqueeze_1619 = None
    mul_1658: "f32[200]" = torch.ops.aten.mul.Tensor(sum_197, squeeze_40);  sum_197 = squeeze_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_98 = torch.ops.aten.convolution_backward.default(mul_1657, relu_12, primals_235, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False]);  mul_1657 = primals_235 = None
    getitem_518: "f32[8, 200, 56, 56]" = convolution_backward_98[0]
    getitem_519: "f32[200, 4, 3, 3]" = convolution_backward_98[1];  convolution_backward_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_98: "b8[8, 200, 56, 56]" = torch.ops.aten.le.Scalar(relu_12, 0);  relu_12 = None
    where_98: "f32[8, 200, 56, 56]" = torch.ops.aten.where.self(le_98, full_default, getitem_518);  le_98 = getitem_518 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_198: "f32[200]" = torch.ops.aten.sum.dim_IntList(where_98, [0, 2, 3])
    sub_503: "f32[8, 200, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_1622);  convolution_11 = unsqueeze_1622 = None
    mul_1659: "f32[8, 200, 56, 56]" = torch.ops.aten.mul.Tensor(where_98, sub_503)
    sum_199: "f32[200]" = torch.ops.aten.sum.dim_IntList(mul_1659, [0, 2, 3]);  mul_1659 = None
    mul_1660: "f32[200]" = torch.ops.aten.mul.Tensor(sum_198, 3.985969387755102e-05)
    unsqueeze_1623: "f32[1, 200]" = torch.ops.aten.unsqueeze.default(mul_1660, 0);  mul_1660 = None
    unsqueeze_1624: "f32[1, 200, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1623, 2);  unsqueeze_1623 = None
    unsqueeze_1625: "f32[1, 200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1624, 3);  unsqueeze_1624 = None
    mul_1661: "f32[200]" = torch.ops.aten.mul.Tensor(sum_199, 3.985969387755102e-05)
    mul_1662: "f32[200]" = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
    mul_1663: "f32[200]" = torch.ops.aten.mul.Tensor(mul_1661, mul_1662);  mul_1661 = mul_1662 = None
    unsqueeze_1626: "f32[1, 200]" = torch.ops.aten.unsqueeze.default(mul_1663, 0);  mul_1663 = None
    unsqueeze_1627: "f32[1, 200, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1626, 2);  unsqueeze_1626 = None
    unsqueeze_1628: "f32[1, 200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1627, 3);  unsqueeze_1627 = None
    mul_1664: "f32[200]" = torch.ops.aten.mul.Tensor(squeeze_37, primals_25);  primals_25 = None
    unsqueeze_1629: "f32[1, 200]" = torch.ops.aten.unsqueeze.default(mul_1664, 0);  mul_1664 = None
    unsqueeze_1630: "f32[1, 200, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1629, 2);  unsqueeze_1629 = None
    unsqueeze_1631: "f32[1, 200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1630, 3);  unsqueeze_1630 = None
    mul_1665: "f32[8, 200, 56, 56]" = torch.ops.aten.mul.Tensor(sub_503, unsqueeze_1628);  sub_503 = unsqueeze_1628 = None
    sub_505: "f32[8, 200, 56, 56]" = torch.ops.aten.sub.Tensor(where_98, mul_1665);  where_98 = mul_1665 = None
    sub_506: "f32[8, 200, 56, 56]" = torch.ops.aten.sub.Tensor(sub_505, unsqueeze_1625);  sub_505 = unsqueeze_1625 = None
    mul_1666: "f32[8, 200, 56, 56]" = torch.ops.aten.mul.Tensor(sub_506, unsqueeze_1631);  sub_506 = unsqueeze_1631 = None
    mul_1667: "f32[200]" = torch.ops.aten.mul.Tensor(sum_199, squeeze_37);  sum_199 = squeeze_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_99 = torch.ops.aten.convolution_backward.default(mul_1666, relu_11, primals_234, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1666 = primals_234 = None
    getitem_521: "f32[8, 356, 56, 56]" = convolution_backward_99[0]
    getitem_522: "f32[200, 356, 1, 1]" = convolution_backward_99[1];  convolution_backward_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_99: "b8[8, 356, 56, 56]" = torch.ops.aten.le.Scalar(relu_11, 0);  relu_11 = None
    where_99: "f32[8, 356, 56, 56]" = torch.ops.aten.where.self(le_99, full_default, getitem_521);  le_99 = getitem_521 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_200: "f32[356]" = torch.ops.aten.sum.dim_IntList(where_99, [0, 2, 3])
    sub_507: "f32[8, 356, 56, 56]" = torch.ops.aten.sub.Tensor(cat_5, unsqueeze_1634);  cat_5 = unsqueeze_1634 = None
    mul_1668: "f32[8, 356, 56, 56]" = torch.ops.aten.mul.Tensor(where_99, sub_507)
    sum_201: "f32[356]" = torch.ops.aten.sum.dim_IntList(mul_1668, [0, 2, 3]);  mul_1668 = None
    mul_1669: "f32[356]" = torch.ops.aten.mul.Tensor(sum_200, 3.985969387755102e-05)
    unsqueeze_1635: "f32[1, 356]" = torch.ops.aten.unsqueeze.default(mul_1669, 0);  mul_1669 = None
    unsqueeze_1636: "f32[1, 356, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1635, 2);  unsqueeze_1635 = None
    unsqueeze_1637: "f32[1, 356, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1636, 3);  unsqueeze_1636 = None
    mul_1670: "f32[356]" = torch.ops.aten.mul.Tensor(sum_201, 3.985969387755102e-05)
    mul_1671: "f32[356]" = torch.ops.aten.mul.Tensor(squeeze_34, squeeze_34)
    mul_1672: "f32[356]" = torch.ops.aten.mul.Tensor(mul_1670, mul_1671);  mul_1670 = mul_1671 = None
    unsqueeze_1638: "f32[1, 356]" = torch.ops.aten.unsqueeze.default(mul_1672, 0);  mul_1672 = None
    unsqueeze_1639: "f32[1, 356, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1638, 2);  unsqueeze_1638 = None
    unsqueeze_1640: "f32[1, 356, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1639, 3);  unsqueeze_1639 = None
    mul_1673: "f32[356]" = torch.ops.aten.mul.Tensor(squeeze_34, primals_23);  primals_23 = None
    unsqueeze_1641: "f32[1, 356]" = torch.ops.aten.unsqueeze.default(mul_1673, 0);  mul_1673 = None
    unsqueeze_1642: "f32[1, 356, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1641, 2);  unsqueeze_1641 = None
    unsqueeze_1643: "f32[1, 356, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1642, 3);  unsqueeze_1642 = None
    mul_1674: "f32[8, 356, 56, 56]" = torch.ops.aten.mul.Tensor(sub_507, unsqueeze_1640);  sub_507 = unsqueeze_1640 = None
    sub_509: "f32[8, 356, 56, 56]" = torch.ops.aten.sub.Tensor(where_99, mul_1674);  where_99 = mul_1674 = None
    sub_510: "f32[8, 356, 56, 56]" = torch.ops.aten.sub.Tensor(sub_509, unsqueeze_1637);  sub_509 = unsqueeze_1637 = None
    mul_1675: "f32[8, 356, 56, 56]" = torch.ops.aten.mul.Tensor(sub_510, unsqueeze_1643);  sub_510 = unsqueeze_1643 = None
    mul_1676: "f32[356]" = torch.ops.aten.mul.Tensor(sum_201, squeeze_34);  sum_201 = squeeze_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    slice_441: "f32[8, 256, 56, 56]" = torch.ops.aten.slice.Tensor(mul_1675, 1, 0, 256)
    slice_442: "f32[8, 100, 56, 56]" = torch.ops.aten.slice.Tensor(mul_1675, 1, 256, 356);  mul_1675 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    add_684: "f32[8, 256, 56, 56]" = torch.ops.aten.add.Tensor(slice_437, slice_441);  slice_437 = slice_441 = None
    add_685: "f32[8, 100, 56, 56]" = torch.ops.aten.add.Tensor(slice_439, slice_442);  slice_439 = slice_442 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    slice_443: "f32[8, 80, 56, 56]" = torch.ops.aten.slice.Tensor(add_685, 1, 0, 80)
    slice_444: "f32[8, 20, 56, 56]" = torch.ops.aten.slice.Tensor(add_685, 1, 80, 100);  add_685 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    slice_scatter_280: "f32[8, 20, 56, 56]" = torch.ops.aten.slice_scatter.default(full_default_369, slice_444, 3, 0, 9223372036854775807);  slice_444 = None
    slice_scatter_282: "f32[8, 276, 56, 56]" = torch.ops.aten.slice_scatter.default(full_default_371, slice_scatter_280, 1, 256, 9223372036854775807);  slice_scatter_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    slice_scatter_286: "f32[8, 276, 56, 56]" = torch.ops.aten.slice_scatter.default(full_default_371, add_684, 1, 0, 256)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    add_686: "f32[8, 276, 56, 56]" = torch.ops.aten.add.Tensor(slice_scatter_282, slice_scatter_286);  slice_scatter_282 = slice_scatter_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_100 = torch.ops.aten.convolution_backward.default(add_686, relu_10, primals_233, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_686 = primals_233 = None
    getitem_524: "f32[8, 200, 56, 56]" = convolution_backward_100[0]
    getitem_525: "f32[276, 200, 1, 1]" = convolution_backward_100[1];  convolution_backward_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_100: "b8[8, 200, 56, 56]" = torch.ops.aten.le.Scalar(relu_10, 0);  relu_10 = None
    where_100: "f32[8, 200, 56, 56]" = torch.ops.aten.where.self(le_100, full_default, getitem_524);  le_100 = getitem_524 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_202: "f32[200]" = torch.ops.aten.sum.dim_IntList(where_100, [0, 2, 3])
    sub_511: "f32[8, 200, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_1646);  convolution_9 = unsqueeze_1646 = None
    mul_1677: "f32[8, 200, 56, 56]" = torch.ops.aten.mul.Tensor(where_100, sub_511)
    sum_203: "f32[200]" = torch.ops.aten.sum.dim_IntList(mul_1677, [0, 2, 3]);  mul_1677 = None
    mul_1678: "f32[200]" = torch.ops.aten.mul.Tensor(sum_202, 3.985969387755102e-05)
    unsqueeze_1647: "f32[1, 200]" = torch.ops.aten.unsqueeze.default(mul_1678, 0);  mul_1678 = None
    unsqueeze_1648: "f32[1, 200, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1647, 2);  unsqueeze_1647 = None
    unsqueeze_1649: "f32[1, 200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1648, 3);  unsqueeze_1648 = None
    mul_1679: "f32[200]" = torch.ops.aten.mul.Tensor(sum_203, 3.985969387755102e-05)
    mul_1680: "f32[200]" = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
    mul_1681: "f32[200]" = torch.ops.aten.mul.Tensor(mul_1679, mul_1680);  mul_1679 = mul_1680 = None
    unsqueeze_1650: "f32[1, 200]" = torch.ops.aten.unsqueeze.default(mul_1681, 0);  mul_1681 = None
    unsqueeze_1651: "f32[1, 200, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1650, 2);  unsqueeze_1650 = None
    unsqueeze_1652: "f32[1, 200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1651, 3);  unsqueeze_1651 = None
    mul_1682: "f32[200]" = torch.ops.aten.mul.Tensor(squeeze_31, primals_21);  primals_21 = None
    unsqueeze_1653: "f32[1, 200]" = torch.ops.aten.unsqueeze.default(mul_1682, 0);  mul_1682 = None
    unsqueeze_1654: "f32[1, 200, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1653, 2);  unsqueeze_1653 = None
    unsqueeze_1655: "f32[1, 200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1654, 3);  unsqueeze_1654 = None
    mul_1683: "f32[8, 200, 56, 56]" = torch.ops.aten.mul.Tensor(sub_511, unsqueeze_1652);  sub_511 = unsqueeze_1652 = None
    sub_513: "f32[8, 200, 56, 56]" = torch.ops.aten.sub.Tensor(where_100, mul_1683);  where_100 = mul_1683 = None
    sub_514: "f32[8, 200, 56, 56]" = torch.ops.aten.sub.Tensor(sub_513, unsqueeze_1649);  sub_513 = unsqueeze_1649 = None
    mul_1684: "f32[8, 200, 56, 56]" = torch.ops.aten.mul.Tensor(sub_514, unsqueeze_1655);  sub_514 = unsqueeze_1655 = None
    mul_1685: "f32[200]" = torch.ops.aten.mul.Tensor(sum_203, squeeze_31);  sum_203 = squeeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_101 = torch.ops.aten.convolution_backward.default(mul_1684, relu_9, primals_232, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False]);  mul_1684 = primals_232 = None
    getitem_527: "f32[8, 200, 56, 56]" = convolution_backward_101[0]
    getitem_528: "f32[200, 4, 3, 3]" = convolution_backward_101[1];  convolution_backward_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_101: "b8[8, 200, 56, 56]" = torch.ops.aten.le.Scalar(relu_9, 0);  relu_9 = None
    where_101: "f32[8, 200, 56, 56]" = torch.ops.aten.where.self(le_101, full_default, getitem_527);  le_101 = getitem_527 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_204: "f32[200]" = torch.ops.aten.sum.dim_IntList(where_101, [0, 2, 3])
    sub_515: "f32[8, 200, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_1658);  convolution_8 = unsqueeze_1658 = None
    mul_1686: "f32[8, 200, 56, 56]" = torch.ops.aten.mul.Tensor(where_101, sub_515)
    sum_205: "f32[200]" = torch.ops.aten.sum.dim_IntList(mul_1686, [0, 2, 3]);  mul_1686 = None
    mul_1687: "f32[200]" = torch.ops.aten.mul.Tensor(sum_204, 3.985969387755102e-05)
    unsqueeze_1659: "f32[1, 200]" = torch.ops.aten.unsqueeze.default(mul_1687, 0);  mul_1687 = None
    unsqueeze_1660: "f32[1, 200, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1659, 2);  unsqueeze_1659 = None
    unsqueeze_1661: "f32[1, 200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1660, 3);  unsqueeze_1660 = None
    mul_1688: "f32[200]" = torch.ops.aten.mul.Tensor(sum_205, 3.985969387755102e-05)
    mul_1689: "f32[200]" = torch.ops.aten.mul.Tensor(squeeze_28, squeeze_28)
    mul_1690: "f32[200]" = torch.ops.aten.mul.Tensor(mul_1688, mul_1689);  mul_1688 = mul_1689 = None
    unsqueeze_1662: "f32[1, 200]" = torch.ops.aten.unsqueeze.default(mul_1690, 0);  mul_1690 = None
    unsqueeze_1663: "f32[1, 200, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1662, 2);  unsqueeze_1662 = None
    unsqueeze_1664: "f32[1, 200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1663, 3);  unsqueeze_1663 = None
    mul_1691: "f32[200]" = torch.ops.aten.mul.Tensor(squeeze_28, primals_19);  primals_19 = None
    unsqueeze_1665: "f32[1, 200]" = torch.ops.aten.unsqueeze.default(mul_1691, 0);  mul_1691 = None
    unsqueeze_1666: "f32[1, 200, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1665, 2);  unsqueeze_1665 = None
    unsqueeze_1667: "f32[1, 200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1666, 3);  unsqueeze_1666 = None
    mul_1692: "f32[8, 200, 56, 56]" = torch.ops.aten.mul.Tensor(sub_515, unsqueeze_1664);  sub_515 = unsqueeze_1664 = None
    sub_517: "f32[8, 200, 56, 56]" = torch.ops.aten.sub.Tensor(where_101, mul_1692);  where_101 = mul_1692 = None
    sub_518: "f32[8, 200, 56, 56]" = torch.ops.aten.sub.Tensor(sub_517, unsqueeze_1661);  sub_517 = unsqueeze_1661 = None
    mul_1693: "f32[8, 200, 56, 56]" = torch.ops.aten.mul.Tensor(sub_518, unsqueeze_1667);  sub_518 = unsqueeze_1667 = None
    mul_1694: "f32[200]" = torch.ops.aten.mul.Tensor(sum_205, squeeze_28);  sum_205 = squeeze_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_102 = torch.ops.aten.convolution_backward.default(mul_1693, relu_8, primals_231, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1693 = primals_231 = None
    getitem_530: "f32[8, 336, 56, 56]" = convolution_backward_102[0]
    getitem_531: "f32[200, 336, 1, 1]" = convolution_backward_102[1];  convolution_backward_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_102: "b8[8, 336, 56, 56]" = torch.ops.aten.le.Scalar(relu_8, 0);  relu_8 = None
    where_102: "f32[8, 336, 56, 56]" = torch.ops.aten.where.self(le_102, full_default, getitem_530);  le_102 = getitem_530 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_206: "f32[336]" = torch.ops.aten.sum.dim_IntList(where_102, [0, 2, 3])
    sub_519: "f32[8, 336, 56, 56]" = torch.ops.aten.sub.Tensor(cat_3, unsqueeze_1670);  cat_3 = unsqueeze_1670 = None
    mul_1695: "f32[8, 336, 56, 56]" = torch.ops.aten.mul.Tensor(where_102, sub_519)
    sum_207: "f32[336]" = torch.ops.aten.sum.dim_IntList(mul_1695, [0, 2, 3]);  mul_1695 = None
    mul_1696: "f32[336]" = torch.ops.aten.mul.Tensor(sum_206, 3.985969387755102e-05)
    unsqueeze_1671: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(mul_1696, 0);  mul_1696 = None
    unsqueeze_1672: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1671, 2);  unsqueeze_1671 = None
    unsqueeze_1673: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1672, 3);  unsqueeze_1672 = None
    mul_1697: "f32[336]" = torch.ops.aten.mul.Tensor(sum_207, 3.985969387755102e-05)
    mul_1698: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
    mul_1699: "f32[336]" = torch.ops.aten.mul.Tensor(mul_1697, mul_1698);  mul_1697 = mul_1698 = None
    unsqueeze_1674: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(mul_1699, 0);  mul_1699 = None
    unsqueeze_1675: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1674, 2);  unsqueeze_1674 = None
    unsqueeze_1676: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1675, 3);  unsqueeze_1675 = None
    mul_1700: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_25, primals_17);  primals_17 = None
    unsqueeze_1677: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(mul_1700, 0);  mul_1700 = None
    unsqueeze_1678: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1677, 2);  unsqueeze_1677 = None
    unsqueeze_1679: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1678, 3);  unsqueeze_1678 = None
    mul_1701: "f32[8, 336, 56, 56]" = torch.ops.aten.mul.Tensor(sub_519, unsqueeze_1676);  sub_519 = unsqueeze_1676 = None
    sub_521: "f32[8, 336, 56, 56]" = torch.ops.aten.sub.Tensor(where_102, mul_1701);  where_102 = mul_1701 = None
    sub_522: "f32[8, 336, 56, 56]" = torch.ops.aten.sub.Tensor(sub_521, unsqueeze_1673);  sub_521 = unsqueeze_1673 = None
    mul_1702: "f32[8, 336, 56, 56]" = torch.ops.aten.mul.Tensor(sub_522, unsqueeze_1679);  sub_522 = unsqueeze_1679 = None
    mul_1703: "f32[336]" = torch.ops.aten.mul.Tensor(sum_207, squeeze_25);  sum_207 = squeeze_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    slice_445: "f32[8, 256, 56, 56]" = torch.ops.aten.slice.Tensor(mul_1702, 1, 0, 256)
    slice_446: "f32[8, 80, 56, 56]" = torch.ops.aten.slice.Tensor(mul_1702, 1, 256, 336);  mul_1702 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    add_687: "f32[8, 256, 56, 56]" = torch.ops.aten.add.Tensor(add_684, slice_445);  add_684 = slice_445 = None
    add_688: "f32[8, 80, 56, 56]" = torch.ops.aten.add.Tensor(slice_443, slice_446);  slice_443 = slice_446 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    slice_447: "f32[8, 60, 56, 56]" = torch.ops.aten.slice.Tensor(add_688, 1, 0, 60)
    slice_448: "f32[8, 20, 56, 56]" = torch.ops.aten.slice.Tensor(add_688, 1, 60, 80);  add_688 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    slice_scatter_288: "f32[8, 20, 56, 56]" = torch.ops.aten.slice_scatter.default(full_default_369, slice_448, 3, 0, 9223372036854775807);  slice_448 = None
    slice_scatter_290: "f32[8, 276, 56, 56]" = torch.ops.aten.slice_scatter.default(full_default_371, slice_scatter_288, 1, 256, 9223372036854775807);  slice_scatter_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    slice_scatter_294: "f32[8, 276, 56, 56]" = torch.ops.aten.slice_scatter.default(full_default_371, add_687, 1, 0, 256)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    add_689: "f32[8, 276, 56, 56]" = torch.ops.aten.add.Tensor(slice_scatter_290, slice_scatter_294);  slice_scatter_290 = slice_scatter_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_103 = torch.ops.aten.convolution_backward.default(add_689, relu_7, primals_230, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_689 = primals_230 = None
    getitem_533: "f32[8, 200, 56, 56]" = convolution_backward_103[0]
    getitem_534: "f32[276, 200, 1, 1]" = convolution_backward_103[1];  convolution_backward_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_103: "b8[8, 200, 56, 56]" = torch.ops.aten.le.Scalar(relu_7, 0);  relu_7 = None
    where_103: "f32[8, 200, 56, 56]" = torch.ops.aten.where.self(le_103, full_default, getitem_533);  le_103 = getitem_533 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_208: "f32[200]" = torch.ops.aten.sum.dim_IntList(where_103, [0, 2, 3])
    sub_523: "f32[8, 200, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_1682);  convolution_6 = unsqueeze_1682 = None
    mul_1704: "f32[8, 200, 56, 56]" = torch.ops.aten.mul.Tensor(where_103, sub_523)
    sum_209: "f32[200]" = torch.ops.aten.sum.dim_IntList(mul_1704, [0, 2, 3]);  mul_1704 = None
    mul_1705: "f32[200]" = torch.ops.aten.mul.Tensor(sum_208, 3.985969387755102e-05)
    unsqueeze_1683: "f32[1, 200]" = torch.ops.aten.unsqueeze.default(mul_1705, 0);  mul_1705 = None
    unsqueeze_1684: "f32[1, 200, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1683, 2);  unsqueeze_1683 = None
    unsqueeze_1685: "f32[1, 200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1684, 3);  unsqueeze_1684 = None
    mul_1706: "f32[200]" = torch.ops.aten.mul.Tensor(sum_209, 3.985969387755102e-05)
    mul_1707: "f32[200]" = torch.ops.aten.mul.Tensor(squeeze_22, squeeze_22)
    mul_1708: "f32[200]" = torch.ops.aten.mul.Tensor(mul_1706, mul_1707);  mul_1706 = mul_1707 = None
    unsqueeze_1686: "f32[1, 200]" = torch.ops.aten.unsqueeze.default(mul_1708, 0);  mul_1708 = None
    unsqueeze_1687: "f32[1, 200, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1686, 2);  unsqueeze_1686 = None
    unsqueeze_1688: "f32[1, 200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1687, 3);  unsqueeze_1687 = None
    mul_1709: "f32[200]" = torch.ops.aten.mul.Tensor(squeeze_22, primals_15);  primals_15 = None
    unsqueeze_1689: "f32[1, 200]" = torch.ops.aten.unsqueeze.default(mul_1709, 0);  mul_1709 = None
    unsqueeze_1690: "f32[1, 200, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1689, 2);  unsqueeze_1689 = None
    unsqueeze_1691: "f32[1, 200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1690, 3);  unsqueeze_1690 = None
    mul_1710: "f32[8, 200, 56, 56]" = torch.ops.aten.mul.Tensor(sub_523, unsqueeze_1688);  sub_523 = unsqueeze_1688 = None
    sub_525: "f32[8, 200, 56, 56]" = torch.ops.aten.sub.Tensor(where_103, mul_1710);  where_103 = mul_1710 = None
    sub_526: "f32[8, 200, 56, 56]" = torch.ops.aten.sub.Tensor(sub_525, unsqueeze_1685);  sub_525 = unsqueeze_1685 = None
    mul_1711: "f32[8, 200, 56, 56]" = torch.ops.aten.mul.Tensor(sub_526, unsqueeze_1691);  sub_526 = unsqueeze_1691 = None
    mul_1712: "f32[200]" = torch.ops.aten.mul.Tensor(sum_209, squeeze_22);  sum_209 = squeeze_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_104 = torch.ops.aten.convolution_backward.default(mul_1711, relu_6, primals_229, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False]);  mul_1711 = primals_229 = None
    getitem_536: "f32[8, 200, 56, 56]" = convolution_backward_104[0]
    getitem_537: "f32[200, 4, 3, 3]" = convolution_backward_104[1];  convolution_backward_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_104: "b8[8, 200, 56, 56]" = torch.ops.aten.le.Scalar(relu_6, 0);  relu_6 = None
    where_104: "f32[8, 200, 56, 56]" = torch.ops.aten.where.self(le_104, full_default, getitem_536);  le_104 = getitem_536 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_210: "f32[200]" = torch.ops.aten.sum.dim_IntList(where_104, [0, 2, 3])
    sub_527: "f32[8, 200, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_1694);  convolution_5 = unsqueeze_1694 = None
    mul_1713: "f32[8, 200, 56, 56]" = torch.ops.aten.mul.Tensor(where_104, sub_527)
    sum_211: "f32[200]" = torch.ops.aten.sum.dim_IntList(mul_1713, [0, 2, 3]);  mul_1713 = None
    mul_1714: "f32[200]" = torch.ops.aten.mul.Tensor(sum_210, 3.985969387755102e-05)
    unsqueeze_1695: "f32[1, 200]" = torch.ops.aten.unsqueeze.default(mul_1714, 0);  mul_1714 = None
    unsqueeze_1696: "f32[1, 200, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1695, 2);  unsqueeze_1695 = None
    unsqueeze_1697: "f32[1, 200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1696, 3);  unsqueeze_1696 = None
    mul_1715: "f32[200]" = torch.ops.aten.mul.Tensor(sum_211, 3.985969387755102e-05)
    mul_1716: "f32[200]" = torch.ops.aten.mul.Tensor(squeeze_19, squeeze_19)
    mul_1717: "f32[200]" = torch.ops.aten.mul.Tensor(mul_1715, mul_1716);  mul_1715 = mul_1716 = None
    unsqueeze_1698: "f32[1, 200]" = torch.ops.aten.unsqueeze.default(mul_1717, 0);  mul_1717 = None
    unsqueeze_1699: "f32[1, 200, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1698, 2);  unsqueeze_1698 = None
    unsqueeze_1700: "f32[1, 200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1699, 3);  unsqueeze_1699 = None
    mul_1718: "f32[200]" = torch.ops.aten.mul.Tensor(squeeze_19, primals_13);  primals_13 = None
    unsqueeze_1701: "f32[1, 200]" = torch.ops.aten.unsqueeze.default(mul_1718, 0);  mul_1718 = None
    unsqueeze_1702: "f32[1, 200, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1701, 2);  unsqueeze_1701 = None
    unsqueeze_1703: "f32[1, 200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1702, 3);  unsqueeze_1702 = None
    mul_1719: "f32[8, 200, 56, 56]" = torch.ops.aten.mul.Tensor(sub_527, unsqueeze_1700);  sub_527 = unsqueeze_1700 = None
    sub_529: "f32[8, 200, 56, 56]" = torch.ops.aten.sub.Tensor(where_104, mul_1719);  where_104 = mul_1719 = None
    sub_530: "f32[8, 200, 56, 56]" = torch.ops.aten.sub.Tensor(sub_529, unsqueeze_1697);  sub_529 = unsqueeze_1697 = None
    mul_1720: "f32[8, 200, 56, 56]" = torch.ops.aten.mul.Tensor(sub_530, unsqueeze_1703);  sub_530 = unsqueeze_1703 = None
    mul_1721: "f32[200]" = torch.ops.aten.mul.Tensor(sum_211, squeeze_19);  sum_211 = squeeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_105 = torch.ops.aten.convolution_backward.default(mul_1720, relu_5, primals_228, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1720 = primals_228 = None
    getitem_539: "f32[8, 316, 56, 56]" = convolution_backward_105[0]
    getitem_540: "f32[200, 316, 1, 1]" = convolution_backward_105[1];  convolution_backward_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_105: "b8[8, 316, 56, 56]" = torch.ops.aten.le.Scalar(relu_5, 0);  relu_5 = None
    where_105: "f32[8, 316, 56, 56]" = torch.ops.aten.where.self(le_105, full_default, getitem_539);  le_105 = getitem_539 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_212: "f32[316]" = torch.ops.aten.sum.dim_IntList(where_105, [0, 2, 3])
    sub_531: "f32[8, 316, 56, 56]" = torch.ops.aten.sub.Tensor(cat_1, unsqueeze_1706);  cat_1 = unsqueeze_1706 = None
    mul_1722: "f32[8, 316, 56, 56]" = torch.ops.aten.mul.Tensor(where_105, sub_531)
    sum_213: "f32[316]" = torch.ops.aten.sum.dim_IntList(mul_1722, [0, 2, 3]);  mul_1722 = None
    mul_1723: "f32[316]" = torch.ops.aten.mul.Tensor(sum_212, 3.985969387755102e-05)
    unsqueeze_1707: "f32[1, 316]" = torch.ops.aten.unsqueeze.default(mul_1723, 0);  mul_1723 = None
    unsqueeze_1708: "f32[1, 316, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1707, 2);  unsqueeze_1707 = None
    unsqueeze_1709: "f32[1, 316, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1708, 3);  unsqueeze_1708 = None
    mul_1724: "f32[316]" = torch.ops.aten.mul.Tensor(sum_213, 3.985969387755102e-05)
    mul_1725: "f32[316]" = torch.ops.aten.mul.Tensor(squeeze_16, squeeze_16)
    mul_1726: "f32[316]" = torch.ops.aten.mul.Tensor(mul_1724, mul_1725);  mul_1724 = mul_1725 = None
    unsqueeze_1710: "f32[1, 316]" = torch.ops.aten.unsqueeze.default(mul_1726, 0);  mul_1726 = None
    unsqueeze_1711: "f32[1, 316, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1710, 2);  unsqueeze_1710 = None
    unsqueeze_1712: "f32[1, 316, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1711, 3);  unsqueeze_1711 = None
    mul_1727: "f32[316]" = torch.ops.aten.mul.Tensor(squeeze_16, primals_11);  primals_11 = None
    unsqueeze_1713: "f32[1, 316]" = torch.ops.aten.unsqueeze.default(mul_1727, 0);  mul_1727 = None
    unsqueeze_1714: "f32[1, 316, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1713, 2);  unsqueeze_1713 = None
    unsqueeze_1715: "f32[1, 316, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1714, 3);  unsqueeze_1714 = None
    mul_1728: "f32[8, 316, 56, 56]" = torch.ops.aten.mul.Tensor(sub_531, unsqueeze_1712);  sub_531 = unsqueeze_1712 = None
    sub_533: "f32[8, 316, 56, 56]" = torch.ops.aten.sub.Tensor(where_105, mul_1728);  where_105 = mul_1728 = None
    sub_534: "f32[8, 316, 56, 56]" = torch.ops.aten.sub.Tensor(sub_533, unsqueeze_1709);  sub_533 = unsqueeze_1709 = None
    mul_1729: "f32[8, 316, 56, 56]" = torch.ops.aten.mul.Tensor(sub_534, unsqueeze_1715);  sub_534 = unsqueeze_1715 = None
    mul_1730: "f32[316]" = torch.ops.aten.mul.Tensor(sum_213, squeeze_16);  sum_213 = squeeze_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    slice_449: "f32[8, 256, 56, 56]" = torch.ops.aten.slice.Tensor(mul_1729, 1, 0, 256)
    slice_450: "f32[8, 60, 56, 56]" = torch.ops.aten.slice.Tensor(mul_1729, 1, 256, 316);  mul_1729 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:118, code: x_in = torch.cat(x, dim=1)
    add_690: "f32[8, 256, 56, 56]" = torch.ops.aten.add.Tensor(add_687, slice_449);  add_687 = slice_449 = None
    add_691: "f32[8, 60, 56, 56]" = torch.ops.aten.add.Tensor(slice_447, slice_450);  slice_447 = slice_450 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:146, code: dense = torch.cat([x_s2, out2], dim=1)
    slice_451: "f32[8, 40, 56, 56]" = torch.ops.aten.slice.Tensor(add_691, 1, 0, 40)
    slice_452: "f32[8, 20, 56, 56]" = torch.ops.aten.slice.Tensor(add_691, 1, 40, 60);  add_691 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:144, code: out2 = x_in[:, self.num_1x1_c:, :, :]
    slice_scatter_296: "f32[8, 20, 56, 56]" = torch.ops.aten.slice_scatter.default(full_default_369, slice_452, 3, 0, 9223372036854775807);  full_default_369 = slice_452 = None
    slice_scatter_298: "f32[8, 276, 56, 56]" = torch.ops.aten.slice_scatter.default(full_default_371, slice_scatter_296, 1, 256, 9223372036854775807);  slice_scatter_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    slice_scatter_302: "f32[8, 276, 56, 56]" = torch.ops.aten.slice_scatter.default(full_default_371, add_690, 1, 0, 256);  full_default_371 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:143, code: out1 = x_in[:, :self.num_1x1_c, :, :]
    add_692: "f32[8, 276, 56, 56]" = torch.ops.aten.add.Tensor(slice_scatter_298, slice_scatter_302);  slice_scatter_298 = slice_scatter_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_106 = torch.ops.aten.convolution_backward.default(add_692, relu_4, primals_227, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_692 = primals_227 = None
    getitem_542: "f32[8, 200, 56, 56]" = convolution_backward_106[0]
    getitem_543: "f32[276, 200, 1, 1]" = convolution_backward_106[1];  convolution_backward_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_106: "b8[8, 200, 56, 56]" = torch.ops.aten.le.Scalar(relu_4, 0);  relu_4 = None
    where_106: "f32[8, 200, 56, 56]" = torch.ops.aten.where.self(le_106, full_default, getitem_542);  le_106 = getitem_542 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_214: "f32[200]" = torch.ops.aten.sum.dim_IntList(where_106, [0, 2, 3])
    sub_535: "f32[8, 200, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_1718);  convolution_3 = unsqueeze_1718 = None
    mul_1731: "f32[8, 200, 56, 56]" = torch.ops.aten.mul.Tensor(where_106, sub_535)
    sum_215: "f32[200]" = torch.ops.aten.sum.dim_IntList(mul_1731, [0, 2, 3]);  mul_1731 = None
    mul_1732: "f32[200]" = torch.ops.aten.mul.Tensor(sum_214, 3.985969387755102e-05)
    unsqueeze_1719: "f32[1, 200]" = torch.ops.aten.unsqueeze.default(mul_1732, 0);  mul_1732 = None
    unsqueeze_1720: "f32[1, 200, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1719, 2);  unsqueeze_1719 = None
    unsqueeze_1721: "f32[1, 200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1720, 3);  unsqueeze_1720 = None
    mul_1733: "f32[200]" = torch.ops.aten.mul.Tensor(sum_215, 3.985969387755102e-05)
    mul_1734: "f32[200]" = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
    mul_1735: "f32[200]" = torch.ops.aten.mul.Tensor(mul_1733, mul_1734);  mul_1733 = mul_1734 = None
    unsqueeze_1722: "f32[1, 200]" = torch.ops.aten.unsqueeze.default(mul_1735, 0);  mul_1735 = None
    unsqueeze_1723: "f32[1, 200, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1722, 2);  unsqueeze_1722 = None
    unsqueeze_1724: "f32[1, 200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1723, 3);  unsqueeze_1723 = None
    mul_1736: "f32[200]" = torch.ops.aten.mul.Tensor(squeeze_13, primals_9);  primals_9 = None
    unsqueeze_1725: "f32[1, 200]" = torch.ops.aten.unsqueeze.default(mul_1736, 0);  mul_1736 = None
    unsqueeze_1726: "f32[1, 200, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1725, 2);  unsqueeze_1725 = None
    unsqueeze_1727: "f32[1, 200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1726, 3);  unsqueeze_1726 = None
    mul_1737: "f32[8, 200, 56, 56]" = torch.ops.aten.mul.Tensor(sub_535, unsqueeze_1724);  sub_535 = unsqueeze_1724 = None
    sub_537: "f32[8, 200, 56, 56]" = torch.ops.aten.sub.Tensor(where_106, mul_1737);  where_106 = mul_1737 = None
    sub_538: "f32[8, 200, 56, 56]" = torch.ops.aten.sub.Tensor(sub_537, unsqueeze_1721);  sub_537 = unsqueeze_1721 = None
    mul_1738: "f32[8, 200, 56, 56]" = torch.ops.aten.mul.Tensor(sub_538, unsqueeze_1727);  sub_538 = unsqueeze_1727 = None
    mul_1739: "f32[200]" = torch.ops.aten.mul.Tensor(sum_215, squeeze_13);  sum_215 = squeeze_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_107 = torch.ops.aten.convolution_backward.default(mul_1738, relu_3, primals_226, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 50, [True, True, False]);  mul_1738 = primals_226 = None
    getitem_545: "f32[8, 200, 56, 56]" = convolution_backward_107[0]
    getitem_546: "f32[200, 4, 3, 3]" = convolution_backward_107[1];  convolution_backward_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_107: "b8[8, 200, 56, 56]" = torch.ops.aten.le.Scalar(relu_3, 0);  relu_3 = None
    where_107: "f32[8, 200, 56, 56]" = torch.ops.aten.where.self(le_107, full_default, getitem_545);  le_107 = getitem_545 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_216: "f32[200]" = torch.ops.aten.sum.dim_IntList(where_107, [0, 2, 3])
    sub_539: "f32[8, 200, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_1730);  convolution_2 = unsqueeze_1730 = None
    mul_1740: "f32[8, 200, 56, 56]" = torch.ops.aten.mul.Tensor(where_107, sub_539)
    sum_217: "f32[200]" = torch.ops.aten.sum.dim_IntList(mul_1740, [0, 2, 3]);  mul_1740 = None
    mul_1741: "f32[200]" = torch.ops.aten.mul.Tensor(sum_216, 3.985969387755102e-05)
    unsqueeze_1731: "f32[1, 200]" = torch.ops.aten.unsqueeze.default(mul_1741, 0);  mul_1741 = None
    unsqueeze_1732: "f32[1, 200, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1731, 2);  unsqueeze_1731 = None
    unsqueeze_1733: "f32[1, 200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1732, 3);  unsqueeze_1732 = None
    mul_1742: "f32[200]" = torch.ops.aten.mul.Tensor(sum_217, 3.985969387755102e-05)
    mul_1743: "f32[200]" = torch.ops.aten.mul.Tensor(squeeze_10, squeeze_10)
    mul_1744: "f32[200]" = torch.ops.aten.mul.Tensor(mul_1742, mul_1743);  mul_1742 = mul_1743 = None
    unsqueeze_1734: "f32[1, 200]" = torch.ops.aten.unsqueeze.default(mul_1744, 0);  mul_1744 = None
    unsqueeze_1735: "f32[1, 200, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1734, 2);  unsqueeze_1734 = None
    unsqueeze_1736: "f32[1, 200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1735, 3);  unsqueeze_1735 = None
    mul_1745: "f32[200]" = torch.ops.aten.mul.Tensor(squeeze_10, primals_7);  primals_7 = None
    unsqueeze_1737: "f32[1, 200]" = torch.ops.aten.unsqueeze.default(mul_1745, 0);  mul_1745 = None
    unsqueeze_1738: "f32[1, 200, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1737, 2);  unsqueeze_1737 = None
    unsqueeze_1739: "f32[1, 200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1738, 3);  unsqueeze_1738 = None
    mul_1746: "f32[8, 200, 56, 56]" = torch.ops.aten.mul.Tensor(sub_539, unsqueeze_1736);  sub_539 = unsqueeze_1736 = None
    sub_541: "f32[8, 200, 56, 56]" = torch.ops.aten.sub.Tensor(where_107, mul_1746);  where_107 = mul_1746 = None
    sub_542: "f32[8, 200, 56, 56]" = torch.ops.aten.sub.Tensor(sub_541, unsqueeze_1733);  sub_541 = unsqueeze_1733 = None
    mul_1747: "f32[8, 200, 56, 56]" = torch.ops.aten.mul.Tensor(sub_542, unsqueeze_1739);  sub_542 = unsqueeze_1739 = None
    mul_1748: "f32[200]" = torch.ops.aten.mul.Tensor(sum_217, squeeze_10);  sum_217 = squeeze_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_108 = torch.ops.aten.convolution_backward.default(mul_1747, relu_2, primals_225, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1747 = primals_225 = None
    getitem_548: "f32[8, 128, 56, 56]" = convolution_backward_108[0]
    getitem_549: "f32[200, 128, 1, 1]" = convolution_backward_108[1];  convolution_backward_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_108: "b8[8, 128, 56, 56]" = torch.ops.aten.le.Scalar(relu_2, 0);  relu_2 = None
    where_108: "f32[8, 128, 56, 56]" = torch.ops.aten.where.self(le_108, full_default, getitem_548);  le_108 = getitem_548 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_218: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_108, [0, 2, 3])
    mul_1749: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(where_108, sub_543)
    sum_219: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1749, [0, 2, 3]);  mul_1749 = None
    mul_1750: "f32[128]" = torch.ops.aten.mul.Tensor(sum_218, 3.985969387755102e-05)
    unsqueeze_1743: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1750, 0);  mul_1750 = None
    unsqueeze_1744: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1743, 2);  unsqueeze_1743 = None
    unsqueeze_1745: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1744, 3);  unsqueeze_1744 = None
    mul_1751: "f32[128]" = torch.ops.aten.mul.Tensor(sum_219, 3.985969387755102e-05)
    mul_1752: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
    mul_1753: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1751, mul_1752);  mul_1751 = None
    unsqueeze_1746: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1753, 0);  mul_1753 = None
    unsqueeze_1747: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1746, 2);  unsqueeze_1746 = None
    unsqueeze_1748: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1747, 3);  unsqueeze_1747 = None
    mul_1754: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_4, primals_5);  primals_5 = None
    unsqueeze_1749: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1754, 0);  mul_1754 = None
    unsqueeze_1750: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1749, 2);  unsqueeze_1749 = None
    unsqueeze_1751: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1750, 3);  unsqueeze_1750 = None
    mul_1755: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_543, unsqueeze_1748);  unsqueeze_1748 = None
    sub_545: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(where_108, mul_1755);  where_108 = mul_1755 = None
    sub_546: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(sub_545, unsqueeze_1745);  sub_545 = unsqueeze_1745 = None
    mul_1756: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_546, unsqueeze_1751);  sub_546 = unsqueeze_1751 = None
    mul_1757: "f32[128]" = torch.ops.aten.mul.Tensor(sum_219, squeeze_4);  sum_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:134, code: x_s2 = x_s[:, self.num_1x1_c:, :, :]
    full_default_413: "f32[8, 40, 56, 56]" = torch.ops.aten.full.default([8, 40, 56, 56], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_304: "f32[8, 40, 56, 56]" = torch.ops.aten.slice_scatter.default(full_default_413, slice_451, 3, 0, 9223372036854775807);  full_default_413 = slice_451 = None
    full_default_415: "f32[8, 296, 56, 56]" = torch.ops.aten.full.default([8, 296, 56, 56], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_306: "f32[8, 296, 56, 56]" = torch.ops.aten.slice_scatter.default(full_default_415, slice_scatter_304, 1, 256, 9223372036854775807);  slice_scatter_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:133, code: x_s1 = x_s[:, :self.num_1x1_c, :, :]
    slice_scatter_310: "f32[8, 296, 56, 56]" = torch.ops.aten.slice_scatter.default(full_default_415, add_690, 1, 0, 256);  full_default_415 = add_690 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:133, code: x_s1 = x_s[:, :self.num_1x1_c, :, :]
    add_693: "f32[8, 296, 56, 56]" = torch.ops.aten.add.Tensor(slice_scatter_306, slice_scatter_310);  slice_scatter_306 = slice_scatter_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:53, code: return self.conv(self.bn(x))
    convolution_backward_109 = torch.ops.aten.convolution_backward.default(add_693, relu_1, primals_224, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_693 = primals_224 = None
    getitem_551: "f32[8, 128, 56, 56]" = convolution_backward_109[0]
    getitem_552: "f32[296, 128, 1, 1]" = convolution_backward_109[1];  convolution_backward_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_109: "b8[8, 128, 56, 56]" = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None
    where_109: "f32[8, 128, 56, 56]" = torch.ops.aten.where.self(le_109, full_default, getitem_551);  le_109 = getitem_551 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_220: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_109, [0, 2, 3])
    mul_1758: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(where_109, sub_543)
    sum_221: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1758, [0, 2, 3]);  mul_1758 = None
    mul_1759: "f32[128]" = torch.ops.aten.mul.Tensor(sum_220, 3.985969387755102e-05)
    unsqueeze_1755: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1759, 0);  mul_1759 = None
    unsqueeze_1756: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1755, 2);  unsqueeze_1755 = None
    unsqueeze_1757: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1756, 3);  unsqueeze_1756 = None
    mul_1760: "f32[128]" = torch.ops.aten.mul.Tensor(sum_221, 3.985969387755102e-05)
    mul_1762: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1760, mul_1752);  mul_1760 = mul_1752 = None
    unsqueeze_1758: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1762, 0);  mul_1762 = None
    unsqueeze_1759: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1758, 2);  unsqueeze_1758 = None
    unsqueeze_1760: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1759, 3);  unsqueeze_1759 = None
    mul_1763: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_4, primals_3);  primals_3 = None
    unsqueeze_1761: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1763, 0);  mul_1763 = None
    unsqueeze_1762: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1761, 2);  unsqueeze_1761 = None
    unsqueeze_1763: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1762, 3);  unsqueeze_1762 = None
    mul_1764: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_543, unsqueeze_1760);  sub_543 = unsqueeze_1760 = None
    sub_549: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(where_109, mul_1764);  where_109 = mul_1764 = None
    sub_550: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(sub_549, unsqueeze_1757);  sub_549 = unsqueeze_1757 = None
    mul_1765: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_550, unsqueeze_1763);  sub_550 = unsqueeze_1763 = None
    mul_1766: "f32[128]" = torch.ops.aten.mul.Tensor(sum_221, squeeze_4);  sum_221 = squeeze_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_694: "f32[8, 128, 56, 56]" = torch.ops.aten.add.Tensor(mul_1756, mul_1765);  mul_1756 = mul_1765 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dpn.py:266, code: return self.features(x)
    max_pool2d_with_indices_backward: "f32[8, 128, 112, 112]" = torch.ops.aten.max_pool2d_with_indices_backward.default(add_694, relu, [3, 3], [2, 2], [1, 1], [1, 1], False, getitem_3);  add_694 = getitem_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_110: "b8[8, 128, 112, 112]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
    where_110: "f32[8, 128, 112, 112]" = torch.ops.aten.where.self(le_110, full_default, max_pool2d_with_indices_backward);  le_110 = full_default = max_pool2d_with_indices_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_222: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_110, [0, 2, 3])
    sub_551: "f32[8, 128, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_1766);  convolution = unsqueeze_1766 = None
    mul_1767: "f32[8, 128, 112, 112]" = torch.ops.aten.mul.Tensor(where_110, sub_551)
    sum_223: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1767, [0, 2, 3]);  mul_1767 = None
    mul_1768: "f32[128]" = torch.ops.aten.mul.Tensor(sum_222, 9.964923469387754e-06)
    unsqueeze_1767: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1768, 0);  mul_1768 = None
    unsqueeze_1768: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1767, 2);  unsqueeze_1767 = None
    unsqueeze_1769: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1768, 3);  unsqueeze_1768 = None
    mul_1769: "f32[128]" = torch.ops.aten.mul.Tensor(sum_223, 9.964923469387754e-06)
    mul_1770: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
    mul_1771: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1769, mul_1770);  mul_1769 = mul_1770 = None
    unsqueeze_1770: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1771, 0);  mul_1771 = None
    unsqueeze_1771: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1770, 2);  unsqueeze_1770 = None
    unsqueeze_1772: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1771, 3);  unsqueeze_1771 = None
    mul_1772: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_1, primals_1);  primals_1 = None
    unsqueeze_1773: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1772, 0);  mul_1772 = None
    unsqueeze_1774: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1773, 2);  unsqueeze_1773 = None
    unsqueeze_1775: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1774, 3);  unsqueeze_1774 = None
    mul_1773: "f32[8, 128, 112, 112]" = torch.ops.aten.mul.Tensor(sub_551, unsqueeze_1772);  sub_551 = unsqueeze_1772 = None
    sub_553: "f32[8, 128, 112, 112]" = torch.ops.aten.sub.Tensor(where_110, mul_1773);  where_110 = mul_1773 = None
    sub_554: "f32[8, 128, 112, 112]" = torch.ops.aten.sub.Tensor(sub_553, unsqueeze_1769);  sub_553 = unsqueeze_1769 = None
    mul_1774: "f32[8, 128, 112, 112]" = torch.ops.aten.mul.Tensor(sub_554, unsqueeze_1775);  sub_554 = unsqueeze_1775 = None
    mul_1775: "f32[128]" = torch.ops.aten.mul.Tensor(sum_223, squeeze_1);  sum_223 = squeeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_110 = torch.ops.aten.convolution_backward.default(mul_1774, primals_668, primals_223, [0], [2, 2], [3, 3], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_1774 = primals_668 = primals_223 = None
    getitem_555: "f32[128, 3, 7, 7]" = convolution_backward_110[1];  convolution_backward_110 = None
    return [mul_1775, sum_222, mul_1766, sum_220, mul_1757, sum_218, mul_1748, sum_216, mul_1739, sum_214, mul_1730, sum_212, mul_1721, sum_210, mul_1712, sum_208, mul_1703, sum_206, mul_1694, sum_204, mul_1685, sum_202, mul_1676, sum_200, mul_1667, sum_198, mul_1658, sum_196, mul_1649, sum_194, mul_1640, sum_192, mul_1631, sum_190, mul_1622, sum_188, mul_1613, sum_186, mul_1604, sum_184, mul_1595, sum_182, mul_1586, sum_180, mul_1577, sum_178, mul_1568, sum_176, mul_1559, sum_174, mul_1550, sum_172, mul_1541, sum_170, mul_1532, sum_168, mul_1523, sum_166, mul_1514, sum_164, mul_1505, sum_162, mul_1496, sum_160, mul_1487, sum_158, mul_1478, sum_156, mul_1469, sum_154, mul_1460, sum_152, mul_1451, sum_150, mul_1442, sum_148, mul_1433, sum_146, mul_1424, sum_144, mul_1415, sum_142, mul_1406, sum_140, mul_1397, sum_138, mul_1388, sum_136, mul_1379, sum_134, mul_1370, sum_132, mul_1361, sum_130, mul_1352, sum_128, mul_1343, sum_126, mul_1334, sum_124, mul_1325, sum_122, mul_1316, sum_120, mul_1307, sum_118, mul_1298, sum_116, mul_1289, sum_114, mul_1280, sum_112, mul_1271, sum_110, mul_1262, sum_108, mul_1253, sum_106, mul_1244, sum_104, mul_1235, sum_102, mul_1226, sum_100, mul_1217, sum_98, mul_1208, sum_96, mul_1199, sum_94, mul_1190, sum_92, mul_1181, sum_90, mul_1172, sum_88, mul_1163, sum_86, mul_1154, sum_84, mul_1145, sum_82, mul_1136, sum_80, mul_1127, sum_78, mul_1118, sum_76, mul_1109, sum_74, mul_1100, sum_72, mul_1091, sum_70, mul_1082, sum_68, mul_1073, sum_66, mul_1064, sum_64, mul_1055, sum_62, mul_1046, sum_60, mul_1037, sum_58, mul_1028, sum_56, mul_1019, sum_54, mul_1010, sum_52, mul_1001, sum_50, mul_992, sum_48, mul_983, sum_46, mul_974, sum_44, mul_965, sum_42, mul_956, sum_40, mul_947, sum_38, mul_938, sum_36, mul_929, sum_34, mul_920, sum_32, mul_911, sum_30, mul_902, sum_28, mul_893, sum_26, mul_884, sum_24, mul_875, sum_22, mul_866, sum_20, mul_857, sum_18, mul_848, sum_16, mul_839, sum_14, mul_830, sum_12, mul_821, sum_10, mul_812, sum_8, mul_803, sum_6, mul_794, sum_4, mul_785, sum_2, getitem_555, getitem_552, getitem_549, getitem_546, getitem_543, getitem_540, getitem_537, getitem_534, getitem_531, getitem_528, getitem_525, getitem_522, getitem_519, getitem_516, getitem_513, getitem_510, getitem_507, getitem_504, getitem_501, getitem_498, getitem_495, getitem_492, getitem_489, getitem_486, getitem_483, getitem_480, getitem_477, getitem_474, getitem_471, getitem_468, getitem_465, getitem_462, getitem_459, getitem_456, getitem_453, getitem_450, getitem_447, getitem_444, getitem_441, getitem_438, getitem_435, getitem_432, getitem_429, getitem_426, getitem_423, getitem_420, getitem_417, getitem_414, getitem_411, getitem_408, getitem_405, getitem_402, getitem_399, getitem_396, getitem_393, getitem_390, getitem_387, getitem_384, getitem_381, getitem_378, getitem_375, getitem_372, getitem_369, getitem_366, getitem_363, getitem_360, getitem_357, getitem_354, getitem_351, getitem_348, getitem_345, getitem_342, getitem_339, getitem_336, getitem_333, getitem_330, getitem_327, getitem_324, getitem_321, getitem_318, getitem_315, getitem_312, getitem_309, getitem_306, getitem_303, getitem_300, getitem_297, getitem_294, getitem_291, getitem_288, getitem_285, getitem_282, getitem_279, getitem_276, getitem_273, getitem_270, getitem_267, getitem_264, getitem_261, getitem_258, getitem_255, getitem_252, getitem_249, getitem_246, getitem_243, getitem_240, getitem_237, getitem_234, getitem_231, getitem_228, getitem_225, sum_1, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
    