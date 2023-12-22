from __future__ import annotations



def forward(self, primals_1: "f32[32]", primals_3: "f32[224]", primals_5: "f32[224]", primals_7: "f32[224]", primals_9: "f32[224]", primals_11: "f32[224]", primals_13: "f32[224]", primals_15: "f32[224]", primals_17: "f32[448]", primals_19: "f32[448]", primals_21: "f32[448]", primals_23: "f32[448]", primals_25: "f32[448]", primals_27: "f32[448]", primals_29: "f32[448]", primals_31: "f32[448]", primals_33: "f32[448]", primals_35: "f32[448]", primals_37: "f32[448]", primals_39: "f32[448]", primals_41: "f32[448]", primals_43: "f32[448]", primals_45: "f32[448]", primals_47: "f32[448]", primals_49: "f32[896]", primals_51: "f32[896]", primals_53: "f32[896]", primals_55: "f32[896]", primals_57: "f32[896]", primals_59: "f32[896]", primals_61: "f32[896]", primals_63: "f32[896]", primals_65: "f32[896]", primals_67: "f32[896]", primals_69: "f32[896]", primals_71: "f32[896]", primals_73: "f32[896]", primals_75: "f32[896]", primals_77: "f32[896]", primals_79: "f32[896]", primals_81: "f32[896]", primals_83: "f32[896]", primals_85: "f32[896]", primals_87: "f32[896]", primals_89: "f32[896]", primals_91: "f32[896]", primals_93: "f32[896]", primals_95: "f32[896]", primals_97: "f32[896]", primals_99: "f32[896]", primals_101: "f32[896]", primals_103: "f32[896]", primals_105: "f32[896]", primals_107: "f32[896]", primals_109: "f32[896]", primals_111: "f32[896]", primals_113: "f32[896]", primals_115: "f32[896]", primals_117: "f32[2240]", primals_119: "f32[2240]", primals_121: "f32[2240]", primals_123: "f32[2240]", primals_125: "f32[32, 3, 3, 3]", primals_126: "f32[224, 32, 1, 1]", primals_127: "f32[224, 112, 3, 3]", primals_128: "f32[8, 224, 1, 1]", primals_130: "f32[224, 8, 1, 1]", primals_132: "f32[224, 224, 1, 1]", primals_133: "f32[224, 32, 1, 1]", primals_134: "f32[224, 224, 1, 1]", primals_135: "f32[224, 112, 3, 3]", primals_136: "f32[56, 224, 1, 1]", primals_138: "f32[224, 56, 1, 1]", primals_140: "f32[224, 224, 1, 1]", primals_141: "f32[448, 224, 1, 1]", primals_142: "f32[448, 112, 3, 3]", primals_143: "f32[56, 448, 1, 1]", primals_145: "f32[448, 56, 1, 1]", primals_147: "f32[448, 448, 1, 1]", primals_148: "f32[448, 224, 1, 1]", primals_149: "f32[448, 448, 1, 1]", primals_150: "f32[448, 112, 3, 3]", primals_151: "f32[112, 448, 1, 1]", primals_153: "f32[448, 112, 1, 1]", primals_155: "f32[448, 448, 1, 1]", primals_156: "f32[448, 448, 1, 1]", primals_157: "f32[448, 112, 3, 3]", primals_158: "f32[112, 448, 1, 1]", primals_160: "f32[448, 112, 1, 1]", primals_162: "f32[448, 448, 1, 1]", primals_163: "f32[448, 448, 1, 1]", primals_164: "f32[448, 112, 3, 3]", primals_165: "f32[112, 448, 1, 1]", primals_167: "f32[448, 112, 1, 1]", primals_169: "f32[448, 448, 1, 1]", primals_170: "f32[448, 448, 1, 1]", primals_171: "f32[448, 112, 3, 3]", primals_172: "f32[112, 448, 1, 1]", primals_174: "f32[448, 112, 1, 1]", primals_176: "f32[448, 448, 1, 1]", primals_177: "f32[896, 448, 1, 1]", primals_178: "f32[896, 112, 3, 3]", primals_179: "f32[112, 896, 1, 1]", primals_181: "f32[896, 112, 1, 1]", primals_183: "f32[896, 896, 1, 1]", primals_184: "f32[896, 448, 1, 1]", primals_185: "f32[896, 896, 1, 1]", primals_186: "f32[896, 112, 3, 3]", primals_187: "f32[224, 896, 1, 1]", primals_189: "f32[896, 224, 1, 1]", primals_191: "f32[896, 896, 1, 1]", primals_192: "f32[896, 896, 1, 1]", primals_193: "f32[896, 112, 3, 3]", primals_194: "f32[224, 896, 1, 1]", primals_196: "f32[896, 224, 1, 1]", primals_198: "f32[896, 896, 1, 1]", primals_199: "f32[896, 896, 1, 1]", primals_200: "f32[896, 112, 3, 3]", primals_201: "f32[224, 896, 1, 1]", primals_203: "f32[896, 224, 1, 1]", primals_205: "f32[896, 896, 1, 1]", primals_206: "f32[896, 896, 1, 1]", primals_207: "f32[896, 112, 3, 3]", primals_208: "f32[224, 896, 1, 1]", primals_210: "f32[896, 224, 1, 1]", primals_212: "f32[896, 896, 1, 1]", primals_213: "f32[896, 896, 1, 1]", primals_214: "f32[896, 112, 3, 3]", primals_215: "f32[224, 896, 1, 1]", primals_217: "f32[896, 224, 1, 1]", primals_219: "f32[896, 896, 1, 1]", primals_220: "f32[896, 896, 1, 1]", primals_221: "f32[896, 112, 3, 3]", primals_222: "f32[224, 896, 1, 1]", primals_224: "f32[896, 224, 1, 1]", primals_226: "f32[896, 896, 1, 1]", primals_227: "f32[896, 896, 1, 1]", primals_228: "f32[896, 112, 3, 3]", primals_229: "f32[224, 896, 1, 1]", primals_231: "f32[896, 224, 1, 1]", primals_233: "f32[896, 896, 1, 1]", primals_234: "f32[896, 896, 1, 1]", primals_235: "f32[896, 112, 3, 3]", primals_236: "f32[224, 896, 1, 1]", primals_238: "f32[896, 224, 1, 1]", primals_240: "f32[896, 896, 1, 1]", primals_241: "f32[896, 896, 1, 1]", primals_242: "f32[896, 112, 3, 3]", primals_243: "f32[224, 896, 1, 1]", primals_245: "f32[896, 224, 1, 1]", primals_247: "f32[896, 896, 1, 1]", primals_248: "f32[896, 896, 1, 1]", primals_249: "f32[896, 112, 3, 3]", primals_250: "f32[224, 896, 1, 1]", primals_252: "f32[896, 224, 1, 1]", primals_254: "f32[896, 896, 1, 1]", primals_255: "f32[2240, 896, 1, 1]", primals_256: "f32[2240, 112, 3, 3]", primals_257: "f32[224, 2240, 1, 1]", primals_259: "f32[2240, 224, 1, 1]", primals_261: "f32[2240, 2240, 1, 1]", primals_262: "f32[2240, 896, 1, 1]", primals_265: "f32[32]", primals_266: "f32[32]", primals_267: "f32[224]", primals_268: "f32[224]", primals_269: "f32[224]", primals_270: "f32[224]", primals_271: "f32[224]", primals_272: "f32[224]", primals_273: "f32[224]", primals_274: "f32[224]", primals_275: "f32[224]", primals_276: "f32[224]", primals_277: "f32[224]", primals_278: "f32[224]", primals_279: "f32[224]", primals_280: "f32[224]", primals_281: "f32[448]", primals_282: "f32[448]", primals_283: "f32[448]", primals_284: "f32[448]", primals_285: "f32[448]", primals_286: "f32[448]", primals_287: "f32[448]", primals_288: "f32[448]", primals_289: "f32[448]", primals_290: "f32[448]", primals_291: "f32[448]", primals_292: "f32[448]", primals_293: "f32[448]", primals_294: "f32[448]", primals_295: "f32[448]", primals_296: "f32[448]", primals_297: "f32[448]", primals_298: "f32[448]", primals_299: "f32[448]", primals_300: "f32[448]", primals_301: "f32[448]", primals_302: "f32[448]", primals_303: "f32[448]", primals_304: "f32[448]", primals_305: "f32[448]", primals_306: "f32[448]", primals_307: "f32[448]", primals_308: "f32[448]", primals_309: "f32[448]", primals_310: "f32[448]", primals_311: "f32[448]", primals_312: "f32[448]", primals_313: "f32[896]", primals_314: "f32[896]", primals_315: "f32[896]", primals_316: "f32[896]", primals_317: "f32[896]", primals_318: "f32[896]", primals_319: "f32[896]", primals_320: "f32[896]", primals_321: "f32[896]", primals_322: "f32[896]", primals_323: "f32[896]", primals_324: "f32[896]", primals_325: "f32[896]", primals_326: "f32[896]", primals_327: "f32[896]", primals_328: "f32[896]", primals_329: "f32[896]", primals_330: "f32[896]", primals_331: "f32[896]", primals_332: "f32[896]", primals_333: "f32[896]", primals_334: "f32[896]", primals_335: "f32[896]", primals_336: "f32[896]", primals_337: "f32[896]", primals_338: "f32[896]", primals_339: "f32[896]", primals_340: "f32[896]", primals_341: "f32[896]", primals_342: "f32[896]", primals_343: "f32[896]", primals_344: "f32[896]", primals_345: "f32[896]", primals_346: "f32[896]", primals_347: "f32[896]", primals_348: "f32[896]", primals_349: "f32[896]", primals_350: "f32[896]", primals_351: "f32[896]", primals_352: "f32[896]", primals_353: "f32[896]", primals_354: "f32[896]", primals_355: "f32[896]", primals_356: "f32[896]", primals_357: "f32[896]", primals_358: "f32[896]", primals_359: "f32[896]", primals_360: "f32[896]", primals_361: "f32[896]", primals_362: "f32[896]", primals_363: "f32[896]", primals_364: "f32[896]", primals_365: "f32[896]", primals_366: "f32[896]", primals_367: "f32[896]", primals_368: "f32[896]", primals_369: "f32[896]", primals_370: "f32[896]", primals_371: "f32[896]", primals_372: "f32[896]", primals_373: "f32[896]", primals_374: "f32[896]", primals_375: "f32[896]", primals_376: "f32[896]", primals_377: "f32[896]", primals_378: "f32[896]", primals_379: "f32[896]", primals_380: "f32[896]", primals_381: "f32[2240]", primals_382: "f32[2240]", primals_383: "f32[2240]", primals_384: "f32[2240]", primals_385: "f32[2240]", primals_386: "f32[2240]", primals_387: "f32[2240]", primals_388: "f32[2240]", primals_389: "f32[4, 3, 224, 224]", convolution: "f32[4, 32, 112, 112]", relu: "f32[4, 32, 112, 112]", convolution_1: "f32[4, 224, 112, 112]", relu_1: "f32[4, 224, 112, 112]", convolution_2: "f32[4, 224, 56, 56]", relu_2: "f32[4, 224, 56, 56]", mean: "f32[4, 224, 1, 1]", relu_3: "f32[4, 8, 1, 1]", convolution_4: "f32[4, 224, 1, 1]", mul_9: "f32[4, 224, 56, 56]", convolution_5: "f32[4, 224, 56, 56]", convolution_6: "f32[4, 224, 56, 56]", relu_4: "f32[4, 224, 56, 56]", convolution_7: "f32[4, 224, 56, 56]", relu_5: "f32[4, 224, 56, 56]", convolution_8: "f32[4, 224, 56, 56]", relu_6: "f32[4, 224, 56, 56]", mean_1: "f32[4, 224, 1, 1]", relu_7: "f32[4, 56, 1, 1]", convolution_10: "f32[4, 224, 1, 1]", mul_22: "f32[4, 224, 56, 56]", convolution_11: "f32[4, 224, 56, 56]", relu_8: "f32[4, 224, 56, 56]", convolution_12: "f32[4, 448, 56, 56]", relu_9: "f32[4, 448, 56, 56]", convolution_13: "f32[4, 448, 28, 28]", relu_10: "f32[4, 448, 28, 28]", mean_2: "f32[4, 448, 1, 1]", relu_11: "f32[4, 56, 1, 1]", convolution_15: "f32[4, 448, 1, 1]", mul_32: "f32[4, 448, 28, 28]", convolution_16: "f32[4, 448, 28, 28]", convolution_17: "f32[4, 448, 28, 28]", relu_12: "f32[4, 448, 28, 28]", convolution_18: "f32[4, 448, 28, 28]", relu_13: "f32[4, 448, 28, 28]", convolution_19: "f32[4, 448, 28, 28]", relu_14: "f32[4, 448, 28, 28]", mean_3: "f32[4, 448, 1, 1]", relu_15: "f32[4, 112, 1, 1]", convolution_21: "f32[4, 448, 1, 1]", mul_45: "f32[4, 448, 28, 28]", convolution_22: "f32[4, 448, 28, 28]", relu_16: "f32[4, 448, 28, 28]", convolution_23: "f32[4, 448, 28, 28]", relu_17: "f32[4, 448, 28, 28]", convolution_24: "f32[4, 448, 28, 28]", relu_18: "f32[4, 448, 28, 28]", mean_4: "f32[4, 448, 1, 1]", relu_19: "f32[4, 112, 1, 1]", convolution_26: "f32[4, 448, 1, 1]", mul_55: "f32[4, 448, 28, 28]", convolution_27: "f32[4, 448, 28, 28]", relu_20: "f32[4, 448, 28, 28]", convolution_28: "f32[4, 448, 28, 28]", relu_21: "f32[4, 448, 28, 28]", convolution_29: "f32[4, 448, 28, 28]", relu_22: "f32[4, 448, 28, 28]", mean_5: "f32[4, 448, 1, 1]", relu_23: "f32[4, 112, 1, 1]", convolution_31: "f32[4, 448, 1, 1]", mul_65: "f32[4, 448, 28, 28]", convolution_32: "f32[4, 448, 28, 28]", relu_24: "f32[4, 448, 28, 28]", convolution_33: "f32[4, 448, 28, 28]", relu_25: "f32[4, 448, 28, 28]", convolution_34: "f32[4, 448, 28, 28]", relu_26: "f32[4, 448, 28, 28]", mean_6: "f32[4, 448, 1, 1]", relu_27: "f32[4, 112, 1, 1]", convolution_36: "f32[4, 448, 1, 1]", mul_75: "f32[4, 448, 28, 28]", convolution_37: "f32[4, 448, 28, 28]", relu_28: "f32[4, 448, 28, 28]", convolution_38: "f32[4, 896, 28, 28]", relu_29: "f32[4, 896, 28, 28]", convolution_39: "f32[4, 896, 14, 14]", relu_30: "f32[4, 896, 14, 14]", mean_7: "f32[4, 896, 1, 1]", relu_31: "f32[4, 112, 1, 1]", convolution_41: "f32[4, 896, 1, 1]", mul_85: "f32[4, 896, 14, 14]", convolution_42: "f32[4, 896, 14, 14]", convolution_43: "f32[4, 896, 14, 14]", relu_32: "f32[4, 896, 14, 14]", convolution_44: "f32[4, 896, 14, 14]", relu_33: "f32[4, 896, 14, 14]", convolution_45: "f32[4, 896, 14, 14]", relu_34: "f32[4, 896, 14, 14]", mean_8: "f32[4, 896, 1, 1]", relu_35: "f32[4, 224, 1, 1]", convolution_47: "f32[4, 896, 1, 1]", mul_98: "f32[4, 896, 14, 14]", convolution_48: "f32[4, 896, 14, 14]", relu_36: "f32[4, 896, 14, 14]", convolution_49: "f32[4, 896, 14, 14]", relu_37: "f32[4, 896, 14, 14]", convolution_50: "f32[4, 896, 14, 14]", relu_38: "f32[4, 896, 14, 14]", mean_9: "f32[4, 896, 1, 1]", relu_39: "f32[4, 224, 1, 1]", convolution_52: "f32[4, 896, 1, 1]", mul_108: "f32[4, 896, 14, 14]", convolution_53: "f32[4, 896, 14, 14]", relu_40: "f32[4, 896, 14, 14]", convolution_54: "f32[4, 896, 14, 14]", relu_41: "f32[4, 896, 14, 14]", convolution_55: "f32[4, 896, 14, 14]", relu_42: "f32[4, 896, 14, 14]", mean_10: "f32[4, 896, 1, 1]", relu_43: "f32[4, 224, 1, 1]", convolution_57: "f32[4, 896, 1, 1]", mul_118: "f32[4, 896, 14, 14]", convolution_58: "f32[4, 896, 14, 14]", relu_44: "f32[4, 896, 14, 14]", convolution_59: "f32[4, 896, 14, 14]", relu_45: "f32[4, 896, 14, 14]", convolution_60: "f32[4, 896, 14, 14]", relu_46: "f32[4, 896, 14, 14]", mean_11: "f32[4, 896, 1, 1]", relu_47: "f32[4, 224, 1, 1]", convolution_62: "f32[4, 896, 1, 1]", mul_128: "f32[4, 896, 14, 14]", convolution_63: "f32[4, 896, 14, 14]", relu_48: "f32[4, 896, 14, 14]", convolution_64: "f32[4, 896, 14, 14]", relu_49: "f32[4, 896, 14, 14]", convolution_65: "f32[4, 896, 14, 14]", relu_50: "f32[4, 896, 14, 14]", mean_12: "f32[4, 896, 1, 1]", relu_51: "f32[4, 224, 1, 1]", convolution_67: "f32[4, 896, 1, 1]", mul_138: "f32[4, 896, 14, 14]", convolution_68: "f32[4, 896, 14, 14]", relu_52: "f32[4, 896, 14, 14]", convolution_69: "f32[4, 896, 14, 14]", relu_53: "f32[4, 896, 14, 14]", convolution_70: "f32[4, 896, 14, 14]", relu_54: "f32[4, 896, 14, 14]", mean_13: "f32[4, 896, 1, 1]", relu_55: "f32[4, 224, 1, 1]", convolution_72: "f32[4, 896, 1, 1]", mul_148: "f32[4, 896, 14, 14]", convolution_73: "f32[4, 896, 14, 14]", relu_56: "f32[4, 896, 14, 14]", convolution_74: "f32[4, 896, 14, 14]", relu_57: "f32[4, 896, 14, 14]", convolution_75: "f32[4, 896, 14, 14]", relu_58: "f32[4, 896, 14, 14]", mean_14: "f32[4, 896, 1, 1]", relu_59: "f32[4, 224, 1, 1]", convolution_77: "f32[4, 896, 1, 1]", mul_158: "f32[4, 896, 14, 14]", convolution_78: "f32[4, 896, 14, 14]", relu_60: "f32[4, 896, 14, 14]", convolution_79: "f32[4, 896, 14, 14]", relu_61: "f32[4, 896, 14, 14]", convolution_80: "f32[4, 896, 14, 14]", relu_62: "f32[4, 896, 14, 14]", mean_15: "f32[4, 896, 1, 1]", relu_63: "f32[4, 224, 1, 1]", convolution_82: "f32[4, 896, 1, 1]", mul_168: "f32[4, 896, 14, 14]", convolution_83: "f32[4, 896, 14, 14]", relu_64: "f32[4, 896, 14, 14]", convolution_84: "f32[4, 896, 14, 14]", relu_65: "f32[4, 896, 14, 14]", convolution_85: "f32[4, 896, 14, 14]", relu_66: "f32[4, 896, 14, 14]", mean_16: "f32[4, 896, 1, 1]", relu_67: "f32[4, 224, 1, 1]", convolution_87: "f32[4, 896, 1, 1]", mul_178: "f32[4, 896, 14, 14]", convolution_88: "f32[4, 896, 14, 14]", relu_68: "f32[4, 896, 14, 14]", convolution_89: "f32[4, 896, 14, 14]", relu_69: "f32[4, 896, 14, 14]", convolution_90: "f32[4, 896, 14, 14]", relu_70: "f32[4, 896, 14, 14]", mean_17: "f32[4, 896, 1, 1]", relu_71: "f32[4, 224, 1, 1]", convolution_92: "f32[4, 896, 1, 1]", mul_188: "f32[4, 896, 14, 14]", convolution_93: "f32[4, 896, 14, 14]", relu_72: "f32[4, 896, 14, 14]", convolution_94: "f32[4, 2240, 14, 14]", relu_73: "f32[4, 2240, 14, 14]", convolution_95: "f32[4, 2240, 7, 7]", relu_74: "f32[4, 2240, 7, 7]", mean_18: "f32[4, 2240, 1, 1]", relu_75: "f32[4, 224, 1, 1]", convolution_97: "f32[4, 2240, 1, 1]", mul_198: "f32[4, 2240, 7, 7]", convolution_98: "f32[4, 2240, 7, 7]", convolution_99: "f32[4, 2240, 7, 7]", clone: "f32[4, 2240]", permute_1: "f32[1000, 2240]", le: "b8[4, 2240, 7, 7]", tangents_1: "f32[4, 1000]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid: "f32[4, 224, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_4);  convolution_4 = None
    sigmoid_1: "f32[4, 224, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_10);  convolution_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_2: "f32[4, 448, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_15);  convolution_15 = None
    sigmoid_3: "f32[4, 448, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_21);  convolution_21 = None
    sigmoid_4: "f32[4, 448, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_26);  convolution_26 = None
    sigmoid_5: "f32[4, 448, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_31);  convolution_31 = None
    sigmoid_6: "f32[4, 448, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_36);  convolution_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_7: "f32[4, 896, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_41);  convolution_41 = None
    sigmoid_8: "f32[4, 896, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_47);  convolution_47 = None
    sigmoid_9: "f32[4, 896, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_52);  convolution_52 = None
    sigmoid_10: "f32[4, 896, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_57);  convolution_57 = None
    sigmoid_11: "f32[4, 896, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_62);  convolution_62 = None
    sigmoid_12: "f32[4, 896, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_67);  convolution_67 = None
    sigmoid_13: "f32[4, 896, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_72);  convolution_72 = None
    sigmoid_14: "f32[4, 896, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_77);  convolution_77 = None
    sigmoid_15: "f32[4, 896, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_82);  convolution_82 = None
    sigmoid_16: "f32[4, 896, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_87);  convolution_87 = None
    sigmoid_17: "f32[4, 896, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_92);  convolution_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_18: "f32[4, 2240, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_97);  convolution_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:134, code: x = self.fc(x)
    mm: "f32[4, 2240]" = torch.ops.aten.mm.default(tangents_1, permute_1);  permute_1 = None
    permute_2: "f32[1000, 4]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
    mm_1: "f32[1000, 2240]" = torch.ops.aten.mm.default(permute_2, clone);  permute_2 = clone = None
    permute_3: "f32[2240, 1000]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_1: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view_1: "f32[1000]" = torch.ops.aten.reshape.default(sum_1, [1000]);  sum_1 = None
    permute_4: "f32[1000, 2240]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view_2: "f32[4, 2240, 1, 1]" = torch.ops.aten.reshape.default(mm, [4, 2240, 1, 1]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    expand: "f32[4, 2240, 7, 7]" = torch.ops.aten.expand.default(view_2, [4, 2240, 7, 7]);  view_2 = None
    div: "f32[4, 2240, 7, 7]" = torch.ops.aten.div.Scalar(expand, 49);  expand = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    where: "f32[4, 2240, 7, 7]" = torch.ops.aten.where.self(le, full_default, div);  le = div = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_143: "f32[2240]" = torch.ops.aten.add.Tensor(primals_388, 1e-05);  primals_388 = None
    rsqrt: "f32[2240]" = torch.ops.aten.rsqrt.default(add_143);  add_143 = None
    unsqueeze_496: "f32[1, 2240]" = torch.ops.aten.unsqueeze.default(primals_387, 0);  primals_387 = None
    unsqueeze_497: "f32[1, 2240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_496, 2);  unsqueeze_496 = None
    unsqueeze_498: "f32[1, 2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_497, 3);  unsqueeze_497 = None
    sum_2: "f32[2240]" = torch.ops.aten.sum.dim_IntList(where, [0, 2, 3])
    sub_62: "f32[4, 2240, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_99, unsqueeze_498);  convolution_99 = unsqueeze_498 = None
    mul_205: "f32[4, 2240, 7, 7]" = torch.ops.aten.mul.Tensor(where, sub_62);  sub_62 = None
    sum_3: "f32[2240]" = torch.ops.aten.sum.dim_IntList(mul_205, [0, 2, 3]);  mul_205 = None
    mul_210: "f32[2240]" = torch.ops.aten.mul.Tensor(rsqrt, primals_123);  primals_123 = None
    unsqueeze_505: "f32[1, 2240]" = torch.ops.aten.unsqueeze.default(mul_210, 0);  mul_210 = None
    unsqueeze_506: "f32[1, 2240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_505, 2);  unsqueeze_505 = None
    unsqueeze_507: "f32[1, 2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_506, 3);  unsqueeze_506 = None
    mul_211: "f32[4, 2240, 7, 7]" = torch.ops.aten.mul.Tensor(where, unsqueeze_507);  unsqueeze_507 = None
    mul_212: "f32[2240]" = torch.ops.aten.mul.Tensor(sum_3, rsqrt);  sum_3 = rsqrt = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward = torch.ops.aten.convolution_backward.default(mul_211, relu_72, primals_262, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_211 = primals_262 = None
    getitem: "f32[4, 896, 14, 14]" = convolution_backward[0]
    getitem_1: "f32[2240, 896, 1, 1]" = convolution_backward[1];  convolution_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_144: "f32[2240]" = torch.ops.aten.add.Tensor(primals_386, 1e-05);  primals_386 = None
    rsqrt_1: "f32[2240]" = torch.ops.aten.rsqrt.default(add_144);  add_144 = None
    unsqueeze_508: "f32[1, 2240]" = torch.ops.aten.unsqueeze.default(primals_385, 0);  primals_385 = None
    unsqueeze_509: "f32[1, 2240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_508, 2);  unsqueeze_508 = None
    unsqueeze_510: "f32[1, 2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_509, 3);  unsqueeze_509 = None
    sub_63: "f32[4, 2240, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_98, unsqueeze_510);  convolution_98 = unsqueeze_510 = None
    mul_213: "f32[4, 2240, 7, 7]" = torch.ops.aten.mul.Tensor(where, sub_63);  sub_63 = None
    sum_5: "f32[2240]" = torch.ops.aten.sum.dim_IntList(mul_213, [0, 2, 3]);  mul_213 = None
    mul_218: "f32[2240]" = torch.ops.aten.mul.Tensor(rsqrt_1, primals_121);  primals_121 = None
    unsqueeze_517: "f32[1, 2240]" = torch.ops.aten.unsqueeze.default(mul_218, 0);  mul_218 = None
    unsqueeze_518: "f32[1, 2240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_517, 2);  unsqueeze_517 = None
    unsqueeze_519: "f32[1, 2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_518, 3);  unsqueeze_518 = None
    mul_219: "f32[4, 2240, 7, 7]" = torch.ops.aten.mul.Tensor(where, unsqueeze_519);  where = unsqueeze_519 = None
    mul_220: "f32[2240]" = torch.ops.aten.mul.Tensor(sum_5, rsqrt_1);  sum_5 = rsqrt_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(mul_219, mul_198, primals_261, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_219 = mul_198 = primals_261 = None
    getitem_3: "f32[4, 2240, 7, 7]" = convolution_backward_1[0]
    getitem_4: "f32[2240, 2240, 1, 1]" = convolution_backward_1[1];  convolution_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_221: "f32[4, 2240, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_3, relu_74)
    mul_222: "f32[4, 2240, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_3, sigmoid_18);  getitem_3 = None
    sum_6: "f32[4, 2240, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_221, [2, 3], True);  mul_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sub_64: "f32[4, 2240, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_18)
    mul_223: "f32[4, 2240, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_18, sub_64);  sigmoid_18 = sub_64 = None
    mul_224: "f32[4, 2240, 1, 1]" = torch.ops.aten.mul.Tensor(sum_6, mul_223);  sum_6 = mul_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    sum_7: "f32[2240]" = torch.ops.aten.sum.dim_IntList(mul_224, [0, 2, 3])
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(mul_224, relu_75, primals_259, [2240], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_224 = primals_259 = None
    getitem_6: "f32[4, 224, 1, 1]" = convolution_backward_2[0]
    getitem_7: "f32[2240, 224, 1, 1]" = convolution_backward_2[1];  convolution_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    le_1: "b8[4, 224, 1, 1]" = torch.ops.aten.le.Scalar(relu_75, 0);  relu_75 = None
    where_1: "f32[4, 224, 1, 1]" = torch.ops.aten.where.self(le_1, full_default, getitem_6);  le_1 = getitem_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    sum_8: "f32[224]" = torch.ops.aten.sum.dim_IntList(where_1, [0, 2, 3])
    convolution_backward_3 = torch.ops.aten.convolution_backward.default(where_1, mean_18, primals_257, [224], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_1 = mean_18 = primals_257 = None
    getitem_9: "f32[4, 2240, 1, 1]" = convolution_backward_3[0]
    getitem_10: "f32[224, 2240, 1, 1]" = convolution_backward_3[1];  convolution_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_1: "f32[4, 2240, 7, 7]" = torch.ops.aten.expand.default(getitem_9, [4, 2240, 7, 7]);  getitem_9 = None
    div_1: "f32[4, 2240, 7, 7]" = torch.ops.aten.div.Scalar(expand_1, 49);  expand_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_145: "f32[4, 2240, 7, 7]" = torch.ops.aten.add.Tensor(mul_222, div_1);  mul_222 = div_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_2: "b8[4, 2240, 7, 7]" = torch.ops.aten.le.Scalar(relu_74, 0);  relu_74 = None
    where_2: "f32[4, 2240, 7, 7]" = torch.ops.aten.where.self(le_2, full_default, add_145);  le_2 = add_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_146: "f32[2240]" = torch.ops.aten.add.Tensor(primals_384, 1e-05);  primals_384 = None
    rsqrt_2: "f32[2240]" = torch.ops.aten.rsqrt.default(add_146);  add_146 = None
    unsqueeze_520: "f32[1, 2240]" = torch.ops.aten.unsqueeze.default(primals_383, 0);  primals_383 = None
    unsqueeze_521: "f32[1, 2240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_520, 2);  unsqueeze_520 = None
    unsqueeze_522: "f32[1, 2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_521, 3);  unsqueeze_521 = None
    sum_9: "f32[2240]" = torch.ops.aten.sum.dim_IntList(where_2, [0, 2, 3])
    sub_65: "f32[4, 2240, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_95, unsqueeze_522);  convolution_95 = unsqueeze_522 = None
    mul_225: "f32[4, 2240, 7, 7]" = torch.ops.aten.mul.Tensor(where_2, sub_65);  sub_65 = None
    sum_10: "f32[2240]" = torch.ops.aten.sum.dim_IntList(mul_225, [0, 2, 3]);  mul_225 = None
    mul_230: "f32[2240]" = torch.ops.aten.mul.Tensor(rsqrt_2, primals_119);  primals_119 = None
    unsqueeze_529: "f32[1, 2240]" = torch.ops.aten.unsqueeze.default(mul_230, 0);  mul_230 = None
    unsqueeze_530: "f32[1, 2240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_529, 2);  unsqueeze_529 = None
    unsqueeze_531: "f32[1, 2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_530, 3);  unsqueeze_530 = None
    mul_231: "f32[4, 2240, 7, 7]" = torch.ops.aten.mul.Tensor(where_2, unsqueeze_531);  where_2 = unsqueeze_531 = None
    mul_232: "f32[2240]" = torch.ops.aten.mul.Tensor(sum_10, rsqrt_2);  sum_10 = rsqrt_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_4 = torch.ops.aten.convolution_backward.default(mul_231, relu_73, primals_256, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 20, [True, True, False]);  mul_231 = primals_256 = None
    getitem_12: "f32[4, 2240, 14, 14]" = convolution_backward_4[0]
    getitem_13: "f32[2240, 112, 3, 3]" = convolution_backward_4[1];  convolution_backward_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_3: "b8[4, 2240, 14, 14]" = torch.ops.aten.le.Scalar(relu_73, 0);  relu_73 = None
    where_3: "f32[4, 2240, 14, 14]" = torch.ops.aten.where.self(le_3, full_default, getitem_12);  le_3 = getitem_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_147: "f32[2240]" = torch.ops.aten.add.Tensor(primals_382, 1e-05);  primals_382 = None
    rsqrt_3: "f32[2240]" = torch.ops.aten.rsqrt.default(add_147);  add_147 = None
    unsqueeze_532: "f32[1, 2240]" = torch.ops.aten.unsqueeze.default(primals_381, 0);  primals_381 = None
    unsqueeze_533: "f32[1, 2240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_532, 2);  unsqueeze_532 = None
    unsqueeze_534: "f32[1, 2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_533, 3);  unsqueeze_533 = None
    sum_11: "f32[2240]" = torch.ops.aten.sum.dim_IntList(where_3, [0, 2, 3])
    sub_66: "f32[4, 2240, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_94, unsqueeze_534);  convolution_94 = unsqueeze_534 = None
    mul_233: "f32[4, 2240, 14, 14]" = torch.ops.aten.mul.Tensor(where_3, sub_66);  sub_66 = None
    sum_12: "f32[2240]" = torch.ops.aten.sum.dim_IntList(mul_233, [0, 2, 3]);  mul_233 = None
    mul_238: "f32[2240]" = torch.ops.aten.mul.Tensor(rsqrt_3, primals_117);  primals_117 = None
    unsqueeze_541: "f32[1, 2240]" = torch.ops.aten.unsqueeze.default(mul_238, 0);  mul_238 = None
    unsqueeze_542: "f32[1, 2240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_541, 2);  unsqueeze_541 = None
    unsqueeze_543: "f32[1, 2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_542, 3);  unsqueeze_542 = None
    mul_239: "f32[4, 2240, 14, 14]" = torch.ops.aten.mul.Tensor(where_3, unsqueeze_543);  where_3 = unsqueeze_543 = None
    mul_240: "f32[2240]" = torch.ops.aten.mul.Tensor(sum_12, rsqrt_3);  sum_12 = rsqrt_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_5 = torch.ops.aten.convolution_backward.default(mul_239, relu_72, primals_255, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_239 = primals_255 = None
    getitem_15: "f32[4, 896, 14, 14]" = convolution_backward_5[0]
    getitem_16: "f32[2240, 896, 1, 1]" = convolution_backward_5[1];  convolution_backward_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_148: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(getitem, getitem_15);  getitem = getitem_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    le_4: "b8[4, 896, 14, 14]" = torch.ops.aten.le.Scalar(relu_72, 0);  relu_72 = None
    where_4: "f32[4, 896, 14, 14]" = torch.ops.aten.where.self(le_4, full_default, add_148);  le_4 = add_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_149: "f32[896]" = torch.ops.aten.add.Tensor(primals_380, 1e-05);  primals_380 = None
    rsqrt_4: "f32[896]" = torch.ops.aten.rsqrt.default(add_149);  add_149 = None
    unsqueeze_544: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(primals_379, 0);  primals_379 = None
    unsqueeze_545: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_544, 2);  unsqueeze_544 = None
    unsqueeze_546: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_545, 3);  unsqueeze_545 = None
    sum_13: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_4, [0, 2, 3])
    sub_67: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_93, unsqueeze_546);  convolution_93 = unsqueeze_546 = None
    mul_241: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_4, sub_67);  sub_67 = None
    sum_14: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_241, [0, 2, 3]);  mul_241 = None
    mul_246: "f32[896]" = torch.ops.aten.mul.Tensor(rsqrt_4, primals_115);  primals_115 = None
    unsqueeze_553: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_246, 0);  mul_246 = None
    unsqueeze_554: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_553, 2);  unsqueeze_553 = None
    unsqueeze_555: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_554, 3);  unsqueeze_554 = None
    mul_247: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_4, unsqueeze_555);  unsqueeze_555 = None
    mul_248: "f32[896]" = torch.ops.aten.mul.Tensor(sum_14, rsqrt_4);  sum_14 = rsqrt_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_6 = torch.ops.aten.convolution_backward.default(mul_247, mul_188, primals_254, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_247 = mul_188 = primals_254 = None
    getitem_18: "f32[4, 896, 14, 14]" = convolution_backward_6[0]
    getitem_19: "f32[896, 896, 1, 1]" = convolution_backward_6[1];  convolution_backward_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_249: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_18, relu_70)
    mul_250: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_18, sigmoid_17);  getitem_18 = None
    sum_15: "f32[4, 896, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_249, [2, 3], True);  mul_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sub_68: "f32[4, 896, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_17)
    mul_251: "f32[4, 896, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_17, sub_68);  sigmoid_17 = sub_68 = None
    mul_252: "f32[4, 896, 1, 1]" = torch.ops.aten.mul.Tensor(sum_15, mul_251);  sum_15 = mul_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    sum_16: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_252, [0, 2, 3])
    convolution_backward_7 = torch.ops.aten.convolution_backward.default(mul_252, relu_71, primals_252, [896], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_252 = primals_252 = None
    getitem_21: "f32[4, 224, 1, 1]" = convolution_backward_7[0]
    getitem_22: "f32[896, 224, 1, 1]" = convolution_backward_7[1];  convolution_backward_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    le_5: "b8[4, 224, 1, 1]" = torch.ops.aten.le.Scalar(relu_71, 0);  relu_71 = None
    where_5: "f32[4, 224, 1, 1]" = torch.ops.aten.where.self(le_5, full_default, getitem_21);  le_5 = getitem_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    sum_17: "f32[224]" = torch.ops.aten.sum.dim_IntList(where_5, [0, 2, 3])
    convolution_backward_8 = torch.ops.aten.convolution_backward.default(where_5, mean_17, primals_250, [224], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_5 = mean_17 = primals_250 = None
    getitem_24: "f32[4, 896, 1, 1]" = convolution_backward_8[0]
    getitem_25: "f32[224, 896, 1, 1]" = convolution_backward_8[1];  convolution_backward_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_2: "f32[4, 896, 14, 14]" = torch.ops.aten.expand.default(getitem_24, [4, 896, 14, 14]);  getitem_24 = None
    div_2: "f32[4, 896, 14, 14]" = torch.ops.aten.div.Scalar(expand_2, 196);  expand_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_150: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_250, div_2);  mul_250 = div_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_6: "b8[4, 896, 14, 14]" = torch.ops.aten.le.Scalar(relu_70, 0);  relu_70 = None
    where_6: "f32[4, 896, 14, 14]" = torch.ops.aten.where.self(le_6, full_default, add_150);  le_6 = add_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_151: "f32[896]" = torch.ops.aten.add.Tensor(primals_378, 1e-05);  primals_378 = None
    rsqrt_5: "f32[896]" = torch.ops.aten.rsqrt.default(add_151);  add_151 = None
    unsqueeze_556: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(primals_377, 0);  primals_377 = None
    unsqueeze_557: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_556, 2);  unsqueeze_556 = None
    unsqueeze_558: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_557, 3);  unsqueeze_557 = None
    sum_18: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_6, [0, 2, 3])
    sub_69: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_90, unsqueeze_558);  convolution_90 = unsqueeze_558 = None
    mul_253: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_6, sub_69);  sub_69 = None
    sum_19: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_253, [0, 2, 3]);  mul_253 = None
    mul_258: "f32[896]" = torch.ops.aten.mul.Tensor(rsqrt_5, primals_113);  primals_113 = None
    unsqueeze_565: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_258, 0);  mul_258 = None
    unsqueeze_566: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_565, 2);  unsqueeze_565 = None
    unsqueeze_567: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_566, 3);  unsqueeze_566 = None
    mul_259: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_6, unsqueeze_567);  where_6 = unsqueeze_567 = None
    mul_260: "f32[896]" = torch.ops.aten.mul.Tensor(sum_19, rsqrt_5);  sum_19 = rsqrt_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_9 = torch.ops.aten.convolution_backward.default(mul_259, relu_69, primals_249, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_259 = primals_249 = None
    getitem_27: "f32[4, 896, 14, 14]" = convolution_backward_9[0]
    getitem_28: "f32[896, 112, 3, 3]" = convolution_backward_9[1];  convolution_backward_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_7: "b8[4, 896, 14, 14]" = torch.ops.aten.le.Scalar(relu_69, 0);  relu_69 = None
    where_7: "f32[4, 896, 14, 14]" = torch.ops.aten.where.self(le_7, full_default, getitem_27);  le_7 = getitem_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_152: "f32[896]" = torch.ops.aten.add.Tensor(primals_376, 1e-05);  primals_376 = None
    rsqrt_6: "f32[896]" = torch.ops.aten.rsqrt.default(add_152);  add_152 = None
    unsqueeze_568: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(primals_375, 0);  primals_375 = None
    unsqueeze_569: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_568, 2);  unsqueeze_568 = None
    unsqueeze_570: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_569, 3);  unsqueeze_569 = None
    sum_20: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_7, [0, 2, 3])
    sub_70: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_89, unsqueeze_570);  convolution_89 = unsqueeze_570 = None
    mul_261: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_7, sub_70);  sub_70 = None
    sum_21: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_261, [0, 2, 3]);  mul_261 = None
    mul_266: "f32[896]" = torch.ops.aten.mul.Tensor(rsqrt_6, primals_111);  primals_111 = None
    unsqueeze_577: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_266, 0);  mul_266 = None
    unsqueeze_578: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_577, 2);  unsqueeze_577 = None
    unsqueeze_579: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_578, 3);  unsqueeze_578 = None
    mul_267: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_7, unsqueeze_579);  where_7 = unsqueeze_579 = None
    mul_268: "f32[896]" = torch.ops.aten.mul.Tensor(sum_21, rsqrt_6);  sum_21 = rsqrt_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_10 = torch.ops.aten.convolution_backward.default(mul_267, relu_68, primals_248, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_267 = primals_248 = None
    getitem_30: "f32[4, 896, 14, 14]" = convolution_backward_10[0]
    getitem_31: "f32[896, 896, 1, 1]" = convolution_backward_10[1];  convolution_backward_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_153: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(where_4, getitem_30);  where_4 = getitem_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    le_8: "b8[4, 896, 14, 14]" = torch.ops.aten.le.Scalar(relu_68, 0);  relu_68 = None
    where_8: "f32[4, 896, 14, 14]" = torch.ops.aten.where.self(le_8, full_default, add_153);  le_8 = add_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_154: "f32[896]" = torch.ops.aten.add.Tensor(primals_374, 1e-05);  primals_374 = None
    rsqrt_7: "f32[896]" = torch.ops.aten.rsqrt.default(add_154);  add_154 = None
    unsqueeze_580: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(primals_373, 0);  primals_373 = None
    unsqueeze_581: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_580, 2);  unsqueeze_580 = None
    unsqueeze_582: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_581, 3);  unsqueeze_581 = None
    sum_22: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_8, [0, 2, 3])
    sub_71: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_88, unsqueeze_582);  convolution_88 = unsqueeze_582 = None
    mul_269: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_8, sub_71);  sub_71 = None
    sum_23: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_269, [0, 2, 3]);  mul_269 = None
    mul_274: "f32[896]" = torch.ops.aten.mul.Tensor(rsqrt_7, primals_109);  primals_109 = None
    unsqueeze_589: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_274, 0);  mul_274 = None
    unsqueeze_590: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_589, 2);  unsqueeze_589 = None
    unsqueeze_591: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_590, 3);  unsqueeze_590 = None
    mul_275: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_8, unsqueeze_591);  unsqueeze_591 = None
    mul_276: "f32[896]" = torch.ops.aten.mul.Tensor(sum_23, rsqrt_7);  sum_23 = rsqrt_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_11 = torch.ops.aten.convolution_backward.default(mul_275, mul_178, primals_247, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_275 = mul_178 = primals_247 = None
    getitem_33: "f32[4, 896, 14, 14]" = convolution_backward_11[0]
    getitem_34: "f32[896, 896, 1, 1]" = convolution_backward_11[1];  convolution_backward_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_277: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_33, relu_66)
    mul_278: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_33, sigmoid_16);  getitem_33 = None
    sum_24: "f32[4, 896, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_277, [2, 3], True);  mul_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sub_72: "f32[4, 896, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_16)
    mul_279: "f32[4, 896, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_16, sub_72);  sigmoid_16 = sub_72 = None
    mul_280: "f32[4, 896, 1, 1]" = torch.ops.aten.mul.Tensor(sum_24, mul_279);  sum_24 = mul_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    sum_25: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_280, [0, 2, 3])
    convolution_backward_12 = torch.ops.aten.convolution_backward.default(mul_280, relu_67, primals_245, [896], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_280 = primals_245 = None
    getitem_36: "f32[4, 224, 1, 1]" = convolution_backward_12[0]
    getitem_37: "f32[896, 224, 1, 1]" = convolution_backward_12[1];  convolution_backward_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    le_9: "b8[4, 224, 1, 1]" = torch.ops.aten.le.Scalar(relu_67, 0);  relu_67 = None
    where_9: "f32[4, 224, 1, 1]" = torch.ops.aten.where.self(le_9, full_default, getitem_36);  le_9 = getitem_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    sum_26: "f32[224]" = torch.ops.aten.sum.dim_IntList(where_9, [0, 2, 3])
    convolution_backward_13 = torch.ops.aten.convolution_backward.default(where_9, mean_16, primals_243, [224], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_9 = mean_16 = primals_243 = None
    getitem_39: "f32[4, 896, 1, 1]" = convolution_backward_13[0]
    getitem_40: "f32[224, 896, 1, 1]" = convolution_backward_13[1];  convolution_backward_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_3: "f32[4, 896, 14, 14]" = torch.ops.aten.expand.default(getitem_39, [4, 896, 14, 14]);  getitem_39 = None
    div_3: "f32[4, 896, 14, 14]" = torch.ops.aten.div.Scalar(expand_3, 196);  expand_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_155: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_278, div_3);  mul_278 = div_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_10: "b8[4, 896, 14, 14]" = torch.ops.aten.le.Scalar(relu_66, 0);  relu_66 = None
    where_10: "f32[4, 896, 14, 14]" = torch.ops.aten.where.self(le_10, full_default, add_155);  le_10 = add_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_156: "f32[896]" = torch.ops.aten.add.Tensor(primals_372, 1e-05);  primals_372 = None
    rsqrt_8: "f32[896]" = torch.ops.aten.rsqrt.default(add_156);  add_156 = None
    unsqueeze_592: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(primals_371, 0);  primals_371 = None
    unsqueeze_593: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_592, 2);  unsqueeze_592 = None
    unsqueeze_594: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_593, 3);  unsqueeze_593 = None
    sum_27: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_10, [0, 2, 3])
    sub_73: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_85, unsqueeze_594);  convolution_85 = unsqueeze_594 = None
    mul_281: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_10, sub_73);  sub_73 = None
    sum_28: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_281, [0, 2, 3]);  mul_281 = None
    mul_286: "f32[896]" = torch.ops.aten.mul.Tensor(rsqrt_8, primals_107);  primals_107 = None
    unsqueeze_601: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_286, 0);  mul_286 = None
    unsqueeze_602: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_601, 2);  unsqueeze_601 = None
    unsqueeze_603: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_602, 3);  unsqueeze_602 = None
    mul_287: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_10, unsqueeze_603);  where_10 = unsqueeze_603 = None
    mul_288: "f32[896]" = torch.ops.aten.mul.Tensor(sum_28, rsqrt_8);  sum_28 = rsqrt_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_14 = torch.ops.aten.convolution_backward.default(mul_287, relu_65, primals_242, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_287 = primals_242 = None
    getitem_42: "f32[4, 896, 14, 14]" = convolution_backward_14[0]
    getitem_43: "f32[896, 112, 3, 3]" = convolution_backward_14[1];  convolution_backward_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_11: "b8[4, 896, 14, 14]" = torch.ops.aten.le.Scalar(relu_65, 0);  relu_65 = None
    where_11: "f32[4, 896, 14, 14]" = torch.ops.aten.where.self(le_11, full_default, getitem_42);  le_11 = getitem_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_157: "f32[896]" = torch.ops.aten.add.Tensor(primals_370, 1e-05);  primals_370 = None
    rsqrt_9: "f32[896]" = torch.ops.aten.rsqrt.default(add_157);  add_157 = None
    unsqueeze_604: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(primals_369, 0);  primals_369 = None
    unsqueeze_605: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_604, 2);  unsqueeze_604 = None
    unsqueeze_606: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_605, 3);  unsqueeze_605 = None
    sum_29: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_11, [0, 2, 3])
    sub_74: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_84, unsqueeze_606);  convolution_84 = unsqueeze_606 = None
    mul_289: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_11, sub_74);  sub_74 = None
    sum_30: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_289, [0, 2, 3]);  mul_289 = None
    mul_294: "f32[896]" = torch.ops.aten.mul.Tensor(rsqrt_9, primals_105);  primals_105 = None
    unsqueeze_613: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_294, 0);  mul_294 = None
    unsqueeze_614: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_613, 2);  unsqueeze_613 = None
    unsqueeze_615: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_614, 3);  unsqueeze_614 = None
    mul_295: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_11, unsqueeze_615);  where_11 = unsqueeze_615 = None
    mul_296: "f32[896]" = torch.ops.aten.mul.Tensor(sum_30, rsqrt_9);  sum_30 = rsqrt_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_15 = torch.ops.aten.convolution_backward.default(mul_295, relu_64, primals_241, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_295 = primals_241 = None
    getitem_45: "f32[4, 896, 14, 14]" = convolution_backward_15[0]
    getitem_46: "f32[896, 896, 1, 1]" = convolution_backward_15[1];  convolution_backward_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_158: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(where_8, getitem_45);  where_8 = getitem_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    le_12: "b8[4, 896, 14, 14]" = torch.ops.aten.le.Scalar(relu_64, 0);  relu_64 = None
    where_12: "f32[4, 896, 14, 14]" = torch.ops.aten.where.self(le_12, full_default, add_158);  le_12 = add_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_159: "f32[896]" = torch.ops.aten.add.Tensor(primals_368, 1e-05);  primals_368 = None
    rsqrt_10: "f32[896]" = torch.ops.aten.rsqrt.default(add_159);  add_159 = None
    unsqueeze_616: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(primals_367, 0);  primals_367 = None
    unsqueeze_617: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_616, 2);  unsqueeze_616 = None
    unsqueeze_618: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_617, 3);  unsqueeze_617 = None
    sum_31: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_12, [0, 2, 3])
    sub_75: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_83, unsqueeze_618);  convolution_83 = unsqueeze_618 = None
    mul_297: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_12, sub_75);  sub_75 = None
    sum_32: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_297, [0, 2, 3]);  mul_297 = None
    mul_302: "f32[896]" = torch.ops.aten.mul.Tensor(rsqrt_10, primals_103);  primals_103 = None
    unsqueeze_625: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_302, 0);  mul_302 = None
    unsqueeze_626: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_625, 2);  unsqueeze_625 = None
    unsqueeze_627: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_626, 3);  unsqueeze_626 = None
    mul_303: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_12, unsqueeze_627);  unsqueeze_627 = None
    mul_304: "f32[896]" = torch.ops.aten.mul.Tensor(sum_32, rsqrt_10);  sum_32 = rsqrt_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_16 = torch.ops.aten.convolution_backward.default(mul_303, mul_168, primals_240, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_303 = mul_168 = primals_240 = None
    getitem_48: "f32[4, 896, 14, 14]" = convolution_backward_16[0]
    getitem_49: "f32[896, 896, 1, 1]" = convolution_backward_16[1];  convolution_backward_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_305: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_48, relu_62)
    mul_306: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_48, sigmoid_15);  getitem_48 = None
    sum_33: "f32[4, 896, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_305, [2, 3], True);  mul_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sub_76: "f32[4, 896, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_15)
    mul_307: "f32[4, 896, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_15, sub_76);  sigmoid_15 = sub_76 = None
    mul_308: "f32[4, 896, 1, 1]" = torch.ops.aten.mul.Tensor(sum_33, mul_307);  sum_33 = mul_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    sum_34: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_308, [0, 2, 3])
    convolution_backward_17 = torch.ops.aten.convolution_backward.default(mul_308, relu_63, primals_238, [896], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_308 = primals_238 = None
    getitem_51: "f32[4, 224, 1, 1]" = convolution_backward_17[0]
    getitem_52: "f32[896, 224, 1, 1]" = convolution_backward_17[1];  convolution_backward_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    le_13: "b8[4, 224, 1, 1]" = torch.ops.aten.le.Scalar(relu_63, 0);  relu_63 = None
    where_13: "f32[4, 224, 1, 1]" = torch.ops.aten.where.self(le_13, full_default, getitem_51);  le_13 = getitem_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    sum_35: "f32[224]" = torch.ops.aten.sum.dim_IntList(where_13, [0, 2, 3])
    convolution_backward_18 = torch.ops.aten.convolution_backward.default(where_13, mean_15, primals_236, [224], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_13 = mean_15 = primals_236 = None
    getitem_54: "f32[4, 896, 1, 1]" = convolution_backward_18[0]
    getitem_55: "f32[224, 896, 1, 1]" = convolution_backward_18[1];  convolution_backward_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_4: "f32[4, 896, 14, 14]" = torch.ops.aten.expand.default(getitem_54, [4, 896, 14, 14]);  getitem_54 = None
    div_4: "f32[4, 896, 14, 14]" = torch.ops.aten.div.Scalar(expand_4, 196);  expand_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_160: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_306, div_4);  mul_306 = div_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_14: "b8[4, 896, 14, 14]" = torch.ops.aten.le.Scalar(relu_62, 0);  relu_62 = None
    where_14: "f32[4, 896, 14, 14]" = torch.ops.aten.where.self(le_14, full_default, add_160);  le_14 = add_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_161: "f32[896]" = torch.ops.aten.add.Tensor(primals_366, 1e-05);  primals_366 = None
    rsqrt_11: "f32[896]" = torch.ops.aten.rsqrt.default(add_161);  add_161 = None
    unsqueeze_628: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(primals_365, 0);  primals_365 = None
    unsqueeze_629: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_628, 2);  unsqueeze_628 = None
    unsqueeze_630: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_629, 3);  unsqueeze_629 = None
    sum_36: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_14, [0, 2, 3])
    sub_77: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_80, unsqueeze_630);  convolution_80 = unsqueeze_630 = None
    mul_309: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_14, sub_77);  sub_77 = None
    sum_37: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_309, [0, 2, 3]);  mul_309 = None
    mul_314: "f32[896]" = torch.ops.aten.mul.Tensor(rsqrt_11, primals_101);  primals_101 = None
    unsqueeze_637: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_314, 0);  mul_314 = None
    unsqueeze_638: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_637, 2);  unsqueeze_637 = None
    unsqueeze_639: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_638, 3);  unsqueeze_638 = None
    mul_315: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_14, unsqueeze_639);  where_14 = unsqueeze_639 = None
    mul_316: "f32[896]" = torch.ops.aten.mul.Tensor(sum_37, rsqrt_11);  sum_37 = rsqrt_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_19 = torch.ops.aten.convolution_backward.default(mul_315, relu_61, primals_235, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_315 = primals_235 = None
    getitem_57: "f32[4, 896, 14, 14]" = convolution_backward_19[0]
    getitem_58: "f32[896, 112, 3, 3]" = convolution_backward_19[1];  convolution_backward_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_15: "b8[4, 896, 14, 14]" = torch.ops.aten.le.Scalar(relu_61, 0);  relu_61 = None
    where_15: "f32[4, 896, 14, 14]" = torch.ops.aten.where.self(le_15, full_default, getitem_57);  le_15 = getitem_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_162: "f32[896]" = torch.ops.aten.add.Tensor(primals_364, 1e-05);  primals_364 = None
    rsqrt_12: "f32[896]" = torch.ops.aten.rsqrt.default(add_162);  add_162 = None
    unsqueeze_640: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(primals_363, 0);  primals_363 = None
    unsqueeze_641: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_640, 2);  unsqueeze_640 = None
    unsqueeze_642: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_641, 3);  unsqueeze_641 = None
    sum_38: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_15, [0, 2, 3])
    sub_78: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_79, unsqueeze_642);  convolution_79 = unsqueeze_642 = None
    mul_317: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_15, sub_78);  sub_78 = None
    sum_39: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_317, [0, 2, 3]);  mul_317 = None
    mul_322: "f32[896]" = torch.ops.aten.mul.Tensor(rsqrt_12, primals_99);  primals_99 = None
    unsqueeze_649: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_322, 0);  mul_322 = None
    unsqueeze_650: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_649, 2);  unsqueeze_649 = None
    unsqueeze_651: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_650, 3);  unsqueeze_650 = None
    mul_323: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_15, unsqueeze_651);  where_15 = unsqueeze_651 = None
    mul_324: "f32[896]" = torch.ops.aten.mul.Tensor(sum_39, rsqrt_12);  sum_39 = rsqrt_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_20 = torch.ops.aten.convolution_backward.default(mul_323, relu_60, primals_234, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_323 = primals_234 = None
    getitem_60: "f32[4, 896, 14, 14]" = convolution_backward_20[0]
    getitem_61: "f32[896, 896, 1, 1]" = convolution_backward_20[1];  convolution_backward_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_163: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(where_12, getitem_60);  where_12 = getitem_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    le_16: "b8[4, 896, 14, 14]" = torch.ops.aten.le.Scalar(relu_60, 0);  relu_60 = None
    where_16: "f32[4, 896, 14, 14]" = torch.ops.aten.where.self(le_16, full_default, add_163);  le_16 = add_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_164: "f32[896]" = torch.ops.aten.add.Tensor(primals_362, 1e-05);  primals_362 = None
    rsqrt_13: "f32[896]" = torch.ops.aten.rsqrt.default(add_164);  add_164 = None
    unsqueeze_652: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(primals_361, 0);  primals_361 = None
    unsqueeze_653: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_652, 2);  unsqueeze_652 = None
    unsqueeze_654: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_653, 3);  unsqueeze_653 = None
    sum_40: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_16, [0, 2, 3])
    sub_79: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_78, unsqueeze_654);  convolution_78 = unsqueeze_654 = None
    mul_325: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_16, sub_79);  sub_79 = None
    sum_41: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_325, [0, 2, 3]);  mul_325 = None
    mul_330: "f32[896]" = torch.ops.aten.mul.Tensor(rsqrt_13, primals_97);  primals_97 = None
    unsqueeze_661: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_330, 0);  mul_330 = None
    unsqueeze_662: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_661, 2);  unsqueeze_661 = None
    unsqueeze_663: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_662, 3);  unsqueeze_662 = None
    mul_331: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_16, unsqueeze_663);  unsqueeze_663 = None
    mul_332: "f32[896]" = torch.ops.aten.mul.Tensor(sum_41, rsqrt_13);  sum_41 = rsqrt_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_21 = torch.ops.aten.convolution_backward.default(mul_331, mul_158, primals_233, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_331 = mul_158 = primals_233 = None
    getitem_63: "f32[4, 896, 14, 14]" = convolution_backward_21[0]
    getitem_64: "f32[896, 896, 1, 1]" = convolution_backward_21[1];  convolution_backward_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_333: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_63, relu_58)
    mul_334: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_63, sigmoid_14);  getitem_63 = None
    sum_42: "f32[4, 896, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_333, [2, 3], True);  mul_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sub_80: "f32[4, 896, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_14)
    mul_335: "f32[4, 896, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_14, sub_80);  sigmoid_14 = sub_80 = None
    mul_336: "f32[4, 896, 1, 1]" = torch.ops.aten.mul.Tensor(sum_42, mul_335);  sum_42 = mul_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    sum_43: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_336, [0, 2, 3])
    convolution_backward_22 = torch.ops.aten.convolution_backward.default(mul_336, relu_59, primals_231, [896], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_336 = primals_231 = None
    getitem_66: "f32[4, 224, 1, 1]" = convolution_backward_22[0]
    getitem_67: "f32[896, 224, 1, 1]" = convolution_backward_22[1];  convolution_backward_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    le_17: "b8[4, 224, 1, 1]" = torch.ops.aten.le.Scalar(relu_59, 0);  relu_59 = None
    where_17: "f32[4, 224, 1, 1]" = torch.ops.aten.where.self(le_17, full_default, getitem_66);  le_17 = getitem_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    sum_44: "f32[224]" = torch.ops.aten.sum.dim_IntList(where_17, [0, 2, 3])
    convolution_backward_23 = torch.ops.aten.convolution_backward.default(where_17, mean_14, primals_229, [224], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_17 = mean_14 = primals_229 = None
    getitem_69: "f32[4, 896, 1, 1]" = convolution_backward_23[0]
    getitem_70: "f32[224, 896, 1, 1]" = convolution_backward_23[1];  convolution_backward_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_5: "f32[4, 896, 14, 14]" = torch.ops.aten.expand.default(getitem_69, [4, 896, 14, 14]);  getitem_69 = None
    div_5: "f32[4, 896, 14, 14]" = torch.ops.aten.div.Scalar(expand_5, 196);  expand_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_165: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_334, div_5);  mul_334 = div_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_18: "b8[4, 896, 14, 14]" = torch.ops.aten.le.Scalar(relu_58, 0);  relu_58 = None
    where_18: "f32[4, 896, 14, 14]" = torch.ops.aten.where.self(le_18, full_default, add_165);  le_18 = add_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_166: "f32[896]" = torch.ops.aten.add.Tensor(primals_360, 1e-05);  primals_360 = None
    rsqrt_14: "f32[896]" = torch.ops.aten.rsqrt.default(add_166);  add_166 = None
    unsqueeze_664: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(primals_359, 0);  primals_359 = None
    unsqueeze_665: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_664, 2);  unsqueeze_664 = None
    unsqueeze_666: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_665, 3);  unsqueeze_665 = None
    sum_45: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_18, [0, 2, 3])
    sub_81: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_75, unsqueeze_666);  convolution_75 = unsqueeze_666 = None
    mul_337: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_18, sub_81);  sub_81 = None
    sum_46: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_337, [0, 2, 3]);  mul_337 = None
    mul_342: "f32[896]" = torch.ops.aten.mul.Tensor(rsqrt_14, primals_95);  primals_95 = None
    unsqueeze_673: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_342, 0);  mul_342 = None
    unsqueeze_674: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_673, 2);  unsqueeze_673 = None
    unsqueeze_675: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_674, 3);  unsqueeze_674 = None
    mul_343: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_18, unsqueeze_675);  where_18 = unsqueeze_675 = None
    mul_344: "f32[896]" = torch.ops.aten.mul.Tensor(sum_46, rsqrt_14);  sum_46 = rsqrt_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_24 = torch.ops.aten.convolution_backward.default(mul_343, relu_57, primals_228, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_343 = primals_228 = None
    getitem_72: "f32[4, 896, 14, 14]" = convolution_backward_24[0]
    getitem_73: "f32[896, 112, 3, 3]" = convolution_backward_24[1];  convolution_backward_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_19: "b8[4, 896, 14, 14]" = torch.ops.aten.le.Scalar(relu_57, 0);  relu_57 = None
    where_19: "f32[4, 896, 14, 14]" = torch.ops.aten.where.self(le_19, full_default, getitem_72);  le_19 = getitem_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_167: "f32[896]" = torch.ops.aten.add.Tensor(primals_358, 1e-05);  primals_358 = None
    rsqrt_15: "f32[896]" = torch.ops.aten.rsqrt.default(add_167);  add_167 = None
    unsqueeze_676: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(primals_357, 0);  primals_357 = None
    unsqueeze_677: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_676, 2);  unsqueeze_676 = None
    unsqueeze_678: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_677, 3);  unsqueeze_677 = None
    sum_47: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_19, [0, 2, 3])
    sub_82: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_74, unsqueeze_678);  convolution_74 = unsqueeze_678 = None
    mul_345: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_19, sub_82);  sub_82 = None
    sum_48: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_345, [0, 2, 3]);  mul_345 = None
    mul_350: "f32[896]" = torch.ops.aten.mul.Tensor(rsqrt_15, primals_93);  primals_93 = None
    unsqueeze_685: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_350, 0);  mul_350 = None
    unsqueeze_686: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_685, 2);  unsqueeze_685 = None
    unsqueeze_687: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_686, 3);  unsqueeze_686 = None
    mul_351: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_19, unsqueeze_687);  where_19 = unsqueeze_687 = None
    mul_352: "f32[896]" = torch.ops.aten.mul.Tensor(sum_48, rsqrt_15);  sum_48 = rsqrt_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_25 = torch.ops.aten.convolution_backward.default(mul_351, relu_56, primals_227, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_351 = primals_227 = None
    getitem_75: "f32[4, 896, 14, 14]" = convolution_backward_25[0]
    getitem_76: "f32[896, 896, 1, 1]" = convolution_backward_25[1];  convolution_backward_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_168: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(where_16, getitem_75);  where_16 = getitem_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    le_20: "b8[4, 896, 14, 14]" = torch.ops.aten.le.Scalar(relu_56, 0);  relu_56 = None
    where_20: "f32[4, 896, 14, 14]" = torch.ops.aten.where.self(le_20, full_default, add_168);  le_20 = add_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_169: "f32[896]" = torch.ops.aten.add.Tensor(primals_356, 1e-05);  primals_356 = None
    rsqrt_16: "f32[896]" = torch.ops.aten.rsqrt.default(add_169);  add_169 = None
    unsqueeze_688: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(primals_355, 0);  primals_355 = None
    unsqueeze_689: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_688, 2);  unsqueeze_688 = None
    unsqueeze_690: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_689, 3);  unsqueeze_689 = None
    sum_49: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_20, [0, 2, 3])
    sub_83: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_73, unsqueeze_690);  convolution_73 = unsqueeze_690 = None
    mul_353: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_20, sub_83);  sub_83 = None
    sum_50: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_353, [0, 2, 3]);  mul_353 = None
    mul_358: "f32[896]" = torch.ops.aten.mul.Tensor(rsqrt_16, primals_91);  primals_91 = None
    unsqueeze_697: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_358, 0);  mul_358 = None
    unsqueeze_698: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_697, 2);  unsqueeze_697 = None
    unsqueeze_699: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_698, 3);  unsqueeze_698 = None
    mul_359: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_20, unsqueeze_699);  unsqueeze_699 = None
    mul_360: "f32[896]" = torch.ops.aten.mul.Tensor(sum_50, rsqrt_16);  sum_50 = rsqrt_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_26 = torch.ops.aten.convolution_backward.default(mul_359, mul_148, primals_226, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_359 = mul_148 = primals_226 = None
    getitem_78: "f32[4, 896, 14, 14]" = convolution_backward_26[0]
    getitem_79: "f32[896, 896, 1, 1]" = convolution_backward_26[1];  convolution_backward_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_361: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_78, relu_54)
    mul_362: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_78, sigmoid_13);  getitem_78 = None
    sum_51: "f32[4, 896, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_361, [2, 3], True);  mul_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sub_84: "f32[4, 896, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_13)
    mul_363: "f32[4, 896, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_13, sub_84);  sigmoid_13 = sub_84 = None
    mul_364: "f32[4, 896, 1, 1]" = torch.ops.aten.mul.Tensor(sum_51, mul_363);  sum_51 = mul_363 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    sum_52: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_364, [0, 2, 3])
    convolution_backward_27 = torch.ops.aten.convolution_backward.default(mul_364, relu_55, primals_224, [896], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_364 = primals_224 = None
    getitem_81: "f32[4, 224, 1, 1]" = convolution_backward_27[0]
    getitem_82: "f32[896, 224, 1, 1]" = convolution_backward_27[1];  convolution_backward_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    le_21: "b8[4, 224, 1, 1]" = torch.ops.aten.le.Scalar(relu_55, 0);  relu_55 = None
    where_21: "f32[4, 224, 1, 1]" = torch.ops.aten.where.self(le_21, full_default, getitem_81);  le_21 = getitem_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    sum_53: "f32[224]" = torch.ops.aten.sum.dim_IntList(where_21, [0, 2, 3])
    convolution_backward_28 = torch.ops.aten.convolution_backward.default(where_21, mean_13, primals_222, [224], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_21 = mean_13 = primals_222 = None
    getitem_84: "f32[4, 896, 1, 1]" = convolution_backward_28[0]
    getitem_85: "f32[224, 896, 1, 1]" = convolution_backward_28[1];  convolution_backward_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_6: "f32[4, 896, 14, 14]" = torch.ops.aten.expand.default(getitem_84, [4, 896, 14, 14]);  getitem_84 = None
    div_6: "f32[4, 896, 14, 14]" = torch.ops.aten.div.Scalar(expand_6, 196);  expand_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_170: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_362, div_6);  mul_362 = div_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_22: "b8[4, 896, 14, 14]" = torch.ops.aten.le.Scalar(relu_54, 0);  relu_54 = None
    where_22: "f32[4, 896, 14, 14]" = torch.ops.aten.where.self(le_22, full_default, add_170);  le_22 = add_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_171: "f32[896]" = torch.ops.aten.add.Tensor(primals_354, 1e-05);  primals_354 = None
    rsqrt_17: "f32[896]" = torch.ops.aten.rsqrt.default(add_171);  add_171 = None
    unsqueeze_700: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(primals_353, 0);  primals_353 = None
    unsqueeze_701: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_700, 2);  unsqueeze_700 = None
    unsqueeze_702: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_701, 3);  unsqueeze_701 = None
    sum_54: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_22, [0, 2, 3])
    sub_85: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_70, unsqueeze_702);  convolution_70 = unsqueeze_702 = None
    mul_365: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_22, sub_85);  sub_85 = None
    sum_55: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_365, [0, 2, 3]);  mul_365 = None
    mul_370: "f32[896]" = torch.ops.aten.mul.Tensor(rsqrt_17, primals_89);  primals_89 = None
    unsqueeze_709: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_370, 0);  mul_370 = None
    unsqueeze_710: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_709, 2);  unsqueeze_709 = None
    unsqueeze_711: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_710, 3);  unsqueeze_710 = None
    mul_371: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_22, unsqueeze_711);  where_22 = unsqueeze_711 = None
    mul_372: "f32[896]" = torch.ops.aten.mul.Tensor(sum_55, rsqrt_17);  sum_55 = rsqrt_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_29 = torch.ops.aten.convolution_backward.default(mul_371, relu_53, primals_221, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_371 = primals_221 = None
    getitem_87: "f32[4, 896, 14, 14]" = convolution_backward_29[0]
    getitem_88: "f32[896, 112, 3, 3]" = convolution_backward_29[1];  convolution_backward_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_23: "b8[4, 896, 14, 14]" = torch.ops.aten.le.Scalar(relu_53, 0);  relu_53 = None
    where_23: "f32[4, 896, 14, 14]" = torch.ops.aten.where.self(le_23, full_default, getitem_87);  le_23 = getitem_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_172: "f32[896]" = torch.ops.aten.add.Tensor(primals_352, 1e-05);  primals_352 = None
    rsqrt_18: "f32[896]" = torch.ops.aten.rsqrt.default(add_172);  add_172 = None
    unsqueeze_712: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(primals_351, 0);  primals_351 = None
    unsqueeze_713: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_712, 2);  unsqueeze_712 = None
    unsqueeze_714: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_713, 3);  unsqueeze_713 = None
    sum_56: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_23, [0, 2, 3])
    sub_86: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_69, unsqueeze_714);  convolution_69 = unsqueeze_714 = None
    mul_373: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_23, sub_86);  sub_86 = None
    sum_57: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_373, [0, 2, 3]);  mul_373 = None
    mul_378: "f32[896]" = torch.ops.aten.mul.Tensor(rsqrt_18, primals_87);  primals_87 = None
    unsqueeze_721: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_378, 0);  mul_378 = None
    unsqueeze_722: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_721, 2);  unsqueeze_721 = None
    unsqueeze_723: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_722, 3);  unsqueeze_722 = None
    mul_379: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_23, unsqueeze_723);  where_23 = unsqueeze_723 = None
    mul_380: "f32[896]" = torch.ops.aten.mul.Tensor(sum_57, rsqrt_18);  sum_57 = rsqrt_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_30 = torch.ops.aten.convolution_backward.default(mul_379, relu_52, primals_220, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_379 = primals_220 = None
    getitem_90: "f32[4, 896, 14, 14]" = convolution_backward_30[0]
    getitem_91: "f32[896, 896, 1, 1]" = convolution_backward_30[1];  convolution_backward_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_173: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(where_20, getitem_90);  where_20 = getitem_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    le_24: "b8[4, 896, 14, 14]" = torch.ops.aten.le.Scalar(relu_52, 0);  relu_52 = None
    where_24: "f32[4, 896, 14, 14]" = torch.ops.aten.where.self(le_24, full_default, add_173);  le_24 = add_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_174: "f32[896]" = torch.ops.aten.add.Tensor(primals_350, 1e-05);  primals_350 = None
    rsqrt_19: "f32[896]" = torch.ops.aten.rsqrt.default(add_174);  add_174 = None
    unsqueeze_724: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(primals_349, 0);  primals_349 = None
    unsqueeze_725: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_724, 2);  unsqueeze_724 = None
    unsqueeze_726: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_725, 3);  unsqueeze_725 = None
    sum_58: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_24, [0, 2, 3])
    sub_87: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_68, unsqueeze_726);  convolution_68 = unsqueeze_726 = None
    mul_381: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_24, sub_87);  sub_87 = None
    sum_59: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_381, [0, 2, 3]);  mul_381 = None
    mul_386: "f32[896]" = torch.ops.aten.mul.Tensor(rsqrt_19, primals_85);  primals_85 = None
    unsqueeze_733: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_386, 0);  mul_386 = None
    unsqueeze_734: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_733, 2);  unsqueeze_733 = None
    unsqueeze_735: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_734, 3);  unsqueeze_734 = None
    mul_387: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_24, unsqueeze_735);  unsqueeze_735 = None
    mul_388: "f32[896]" = torch.ops.aten.mul.Tensor(sum_59, rsqrt_19);  sum_59 = rsqrt_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_31 = torch.ops.aten.convolution_backward.default(mul_387, mul_138, primals_219, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_387 = mul_138 = primals_219 = None
    getitem_93: "f32[4, 896, 14, 14]" = convolution_backward_31[0]
    getitem_94: "f32[896, 896, 1, 1]" = convolution_backward_31[1];  convolution_backward_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_389: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_93, relu_50)
    mul_390: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_93, sigmoid_12);  getitem_93 = None
    sum_60: "f32[4, 896, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_389, [2, 3], True);  mul_389 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sub_88: "f32[4, 896, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_12)
    mul_391: "f32[4, 896, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_12, sub_88);  sigmoid_12 = sub_88 = None
    mul_392: "f32[4, 896, 1, 1]" = torch.ops.aten.mul.Tensor(sum_60, mul_391);  sum_60 = mul_391 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    sum_61: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_392, [0, 2, 3])
    convolution_backward_32 = torch.ops.aten.convolution_backward.default(mul_392, relu_51, primals_217, [896], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_392 = primals_217 = None
    getitem_96: "f32[4, 224, 1, 1]" = convolution_backward_32[0]
    getitem_97: "f32[896, 224, 1, 1]" = convolution_backward_32[1];  convolution_backward_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    le_25: "b8[4, 224, 1, 1]" = torch.ops.aten.le.Scalar(relu_51, 0);  relu_51 = None
    where_25: "f32[4, 224, 1, 1]" = torch.ops.aten.where.self(le_25, full_default, getitem_96);  le_25 = getitem_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    sum_62: "f32[224]" = torch.ops.aten.sum.dim_IntList(where_25, [0, 2, 3])
    convolution_backward_33 = torch.ops.aten.convolution_backward.default(where_25, mean_12, primals_215, [224], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_25 = mean_12 = primals_215 = None
    getitem_99: "f32[4, 896, 1, 1]" = convolution_backward_33[0]
    getitem_100: "f32[224, 896, 1, 1]" = convolution_backward_33[1];  convolution_backward_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_7: "f32[4, 896, 14, 14]" = torch.ops.aten.expand.default(getitem_99, [4, 896, 14, 14]);  getitem_99 = None
    div_7: "f32[4, 896, 14, 14]" = torch.ops.aten.div.Scalar(expand_7, 196);  expand_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_175: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_390, div_7);  mul_390 = div_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_26: "b8[4, 896, 14, 14]" = torch.ops.aten.le.Scalar(relu_50, 0);  relu_50 = None
    where_26: "f32[4, 896, 14, 14]" = torch.ops.aten.where.self(le_26, full_default, add_175);  le_26 = add_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_176: "f32[896]" = torch.ops.aten.add.Tensor(primals_348, 1e-05);  primals_348 = None
    rsqrt_20: "f32[896]" = torch.ops.aten.rsqrt.default(add_176);  add_176 = None
    unsqueeze_736: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(primals_347, 0);  primals_347 = None
    unsqueeze_737: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_736, 2);  unsqueeze_736 = None
    unsqueeze_738: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_737, 3);  unsqueeze_737 = None
    sum_63: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_26, [0, 2, 3])
    sub_89: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_65, unsqueeze_738);  convolution_65 = unsqueeze_738 = None
    mul_393: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_26, sub_89);  sub_89 = None
    sum_64: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_393, [0, 2, 3]);  mul_393 = None
    mul_398: "f32[896]" = torch.ops.aten.mul.Tensor(rsqrt_20, primals_83);  primals_83 = None
    unsqueeze_745: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_398, 0);  mul_398 = None
    unsqueeze_746: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_745, 2);  unsqueeze_745 = None
    unsqueeze_747: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_746, 3);  unsqueeze_746 = None
    mul_399: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_26, unsqueeze_747);  where_26 = unsqueeze_747 = None
    mul_400: "f32[896]" = torch.ops.aten.mul.Tensor(sum_64, rsqrt_20);  sum_64 = rsqrt_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_34 = torch.ops.aten.convolution_backward.default(mul_399, relu_49, primals_214, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_399 = primals_214 = None
    getitem_102: "f32[4, 896, 14, 14]" = convolution_backward_34[0]
    getitem_103: "f32[896, 112, 3, 3]" = convolution_backward_34[1];  convolution_backward_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_27: "b8[4, 896, 14, 14]" = torch.ops.aten.le.Scalar(relu_49, 0);  relu_49 = None
    where_27: "f32[4, 896, 14, 14]" = torch.ops.aten.where.self(le_27, full_default, getitem_102);  le_27 = getitem_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_177: "f32[896]" = torch.ops.aten.add.Tensor(primals_346, 1e-05);  primals_346 = None
    rsqrt_21: "f32[896]" = torch.ops.aten.rsqrt.default(add_177);  add_177 = None
    unsqueeze_748: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(primals_345, 0);  primals_345 = None
    unsqueeze_749: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_748, 2);  unsqueeze_748 = None
    unsqueeze_750: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_749, 3);  unsqueeze_749 = None
    sum_65: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_27, [0, 2, 3])
    sub_90: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_64, unsqueeze_750);  convolution_64 = unsqueeze_750 = None
    mul_401: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_27, sub_90);  sub_90 = None
    sum_66: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_401, [0, 2, 3]);  mul_401 = None
    mul_406: "f32[896]" = torch.ops.aten.mul.Tensor(rsqrt_21, primals_81);  primals_81 = None
    unsqueeze_757: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_406, 0);  mul_406 = None
    unsqueeze_758: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_757, 2);  unsqueeze_757 = None
    unsqueeze_759: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_758, 3);  unsqueeze_758 = None
    mul_407: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_27, unsqueeze_759);  where_27 = unsqueeze_759 = None
    mul_408: "f32[896]" = torch.ops.aten.mul.Tensor(sum_66, rsqrt_21);  sum_66 = rsqrt_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_35 = torch.ops.aten.convolution_backward.default(mul_407, relu_48, primals_213, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_407 = primals_213 = None
    getitem_105: "f32[4, 896, 14, 14]" = convolution_backward_35[0]
    getitem_106: "f32[896, 896, 1, 1]" = convolution_backward_35[1];  convolution_backward_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_178: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(where_24, getitem_105);  where_24 = getitem_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    le_28: "b8[4, 896, 14, 14]" = torch.ops.aten.le.Scalar(relu_48, 0);  relu_48 = None
    where_28: "f32[4, 896, 14, 14]" = torch.ops.aten.where.self(le_28, full_default, add_178);  le_28 = add_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_179: "f32[896]" = torch.ops.aten.add.Tensor(primals_344, 1e-05);  primals_344 = None
    rsqrt_22: "f32[896]" = torch.ops.aten.rsqrt.default(add_179);  add_179 = None
    unsqueeze_760: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(primals_343, 0);  primals_343 = None
    unsqueeze_761: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_760, 2);  unsqueeze_760 = None
    unsqueeze_762: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_761, 3);  unsqueeze_761 = None
    sum_67: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_28, [0, 2, 3])
    sub_91: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_63, unsqueeze_762);  convolution_63 = unsqueeze_762 = None
    mul_409: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_28, sub_91);  sub_91 = None
    sum_68: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_409, [0, 2, 3]);  mul_409 = None
    mul_414: "f32[896]" = torch.ops.aten.mul.Tensor(rsqrt_22, primals_79);  primals_79 = None
    unsqueeze_769: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_414, 0);  mul_414 = None
    unsqueeze_770: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_769, 2);  unsqueeze_769 = None
    unsqueeze_771: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_770, 3);  unsqueeze_770 = None
    mul_415: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_28, unsqueeze_771);  unsqueeze_771 = None
    mul_416: "f32[896]" = torch.ops.aten.mul.Tensor(sum_68, rsqrt_22);  sum_68 = rsqrt_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_36 = torch.ops.aten.convolution_backward.default(mul_415, mul_128, primals_212, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_415 = mul_128 = primals_212 = None
    getitem_108: "f32[4, 896, 14, 14]" = convolution_backward_36[0]
    getitem_109: "f32[896, 896, 1, 1]" = convolution_backward_36[1];  convolution_backward_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_417: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_108, relu_46)
    mul_418: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_108, sigmoid_11);  getitem_108 = None
    sum_69: "f32[4, 896, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_417, [2, 3], True);  mul_417 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sub_92: "f32[4, 896, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_11)
    mul_419: "f32[4, 896, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_11, sub_92);  sigmoid_11 = sub_92 = None
    mul_420: "f32[4, 896, 1, 1]" = torch.ops.aten.mul.Tensor(sum_69, mul_419);  sum_69 = mul_419 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    sum_70: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_420, [0, 2, 3])
    convolution_backward_37 = torch.ops.aten.convolution_backward.default(mul_420, relu_47, primals_210, [896], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_420 = primals_210 = None
    getitem_111: "f32[4, 224, 1, 1]" = convolution_backward_37[0]
    getitem_112: "f32[896, 224, 1, 1]" = convolution_backward_37[1];  convolution_backward_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    le_29: "b8[4, 224, 1, 1]" = torch.ops.aten.le.Scalar(relu_47, 0);  relu_47 = None
    where_29: "f32[4, 224, 1, 1]" = torch.ops.aten.where.self(le_29, full_default, getitem_111);  le_29 = getitem_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    sum_71: "f32[224]" = torch.ops.aten.sum.dim_IntList(where_29, [0, 2, 3])
    convolution_backward_38 = torch.ops.aten.convolution_backward.default(where_29, mean_11, primals_208, [224], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_29 = mean_11 = primals_208 = None
    getitem_114: "f32[4, 896, 1, 1]" = convolution_backward_38[0]
    getitem_115: "f32[224, 896, 1, 1]" = convolution_backward_38[1];  convolution_backward_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_8: "f32[4, 896, 14, 14]" = torch.ops.aten.expand.default(getitem_114, [4, 896, 14, 14]);  getitem_114 = None
    div_8: "f32[4, 896, 14, 14]" = torch.ops.aten.div.Scalar(expand_8, 196);  expand_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_180: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_418, div_8);  mul_418 = div_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_30: "b8[4, 896, 14, 14]" = torch.ops.aten.le.Scalar(relu_46, 0);  relu_46 = None
    where_30: "f32[4, 896, 14, 14]" = torch.ops.aten.where.self(le_30, full_default, add_180);  le_30 = add_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_181: "f32[896]" = torch.ops.aten.add.Tensor(primals_342, 1e-05);  primals_342 = None
    rsqrt_23: "f32[896]" = torch.ops.aten.rsqrt.default(add_181);  add_181 = None
    unsqueeze_772: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(primals_341, 0);  primals_341 = None
    unsqueeze_773: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_772, 2);  unsqueeze_772 = None
    unsqueeze_774: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_773, 3);  unsqueeze_773 = None
    sum_72: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_30, [0, 2, 3])
    sub_93: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_60, unsqueeze_774);  convolution_60 = unsqueeze_774 = None
    mul_421: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_30, sub_93);  sub_93 = None
    sum_73: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_421, [0, 2, 3]);  mul_421 = None
    mul_426: "f32[896]" = torch.ops.aten.mul.Tensor(rsqrt_23, primals_77);  primals_77 = None
    unsqueeze_781: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_426, 0);  mul_426 = None
    unsqueeze_782: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_781, 2);  unsqueeze_781 = None
    unsqueeze_783: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_782, 3);  unsqueeze_782 = None
    mul_427: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_30, unsqueeze_783);  where_30 = unsqueeze_783 = None
    mul_428: "f32[896]" = torch.ops.aten.mul.Tensor(sum_73, rsqrt_23);  sum_73 = rsqrt_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_39 = torch.ops.aten.convolution_backward.default(mul_427, relu_45, primals_207, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_427 = primals_207 = None
    getitem_117: "f32[4, 896, 14, 14]" = convolution_backward_39[0]
    getitem_118: "f32[896, 112, 3, 3]" = convolution_backward_39[1];  convolution_backward_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_31: "b8[4, 896, 14, 14]" = torch.ops.aten.le.Scalar(relu_45, 0);  relu_45 = None
    where_31: "f32[4, 896, 14, 14]" = torch.ops.aten.where.self(le_31, full_default, getitem_117);  le_31 = getitem_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_182: "f32[896]" = torch.ops.aten.add.Tensor(primals_340, 1e-05);  primals_340 = None
    rsqrt_24: "f32[896]" = torch.ops.aten.rsqrt.default(add_182);  add_182 = None
    unsqueeze_784: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(primals_339, 0);  primals_339 = None
    unsqueeze_785: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_784, 2);  unsqueeze_784 = None
    unsqueeze_786: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_785, 3);  unsqueeze_785 = None
    sum_74: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_31, [0, 2, 3])
    sub_94: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_59, unsqueeze_786);  convolution_59 = unsqueeze_786 = None
    mul_429: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_31, sub_94);  sub_94 = None
    sum_75: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_429, [0, 2, 3]);  mul_429 = None
    mul_434: "f32[896]" = torch.ops.aten.mul.Tensor(rsqrt_24, primals_75);  primals_75 = None
    unsqueeze_793: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_434, 0);  mul_434 = None
    unsqueeze_794: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_793, 2);  unsqueeze_793 = None
    unsqueeze_795: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_794, 3);  unsqueeze_794 = None
    mul_435: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_31, unsqueeze_795);  where_31 = unsqueeze_795 = None
    mul_436: "f32[896]" = torch.ops.aten.mul.Tensor(sum_75, rsqrt_24);  sum_75 = rsqrt_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_40 = torch.ops.aten.convolution_backward.default(mul_435, relu_44, primals_206, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_435 = primals_206 = None
    getitem_120: "f32[4, 896, 14, 14]" = convolution_backward_40[0]
    getitem_121: "f32[896, 896, 1, 1]" = convolution_backward_40[1];  convolution_backward_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_183: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(where_28, getitem_120);  where_28 = getitem_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    le_32: "b8[4, 896, 14, 14]" = torch.ops.aten.le.Scalar(relu_44, 0);  relu_44 = None
    where_32: "f32[4, 896, 14, 14]" = torch.ops.aten.where.self(le_32, full_default, add_183);  le_32 = add_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_184: "f32[896]" = torch.ops.aten.add.Tensor(primals_338, 1e-05);  primals_338 = None
    rsqrt_25: "f32[896]" = torch.ops.aten.rsqrt.default(add_184);  add_184 = None
    unsqueeze_796: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(primals_337, 0);  primals_337 = None
    unsqueeze_797: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_796, 2);  unsqueeze_796 = None
    unsqueeze_798: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_797, 3);  unsqueeze_797 = None
    sum_76: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_32, [0, 2, 3])
    sub_95: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_58, unsqueeze_798);  convolution_58 = unsqueeze_798 = None
    mul_437: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_32, sub_95);  sub_95 = None
    sum_77: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_437, [0, 2, 3]);  mul_437 = None
    mul_442: "f32[896]" = torch.ops.aten.mul.Tensor(rsqrt_25, primals_73);  primals_73 = None
    unsqueeze_805: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_442, 0);  mul_442 = None
    unsqueeze_806: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_805, 2);  unsqueeze_805 = None
    unsqueeze_807: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_806, 3);  unsqueeze_806 = None
    mul_443: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_32, unsqueeze_807);  unsqueeze_807 = None
    mul_444: "f32[896]" = torch.ops.aten.mul.Tensor(sum_77, rsqrt_25);  sum_77 = rsqrt_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_41 = torch.ops.aten.convolution_backward.default(mul_443, mul_118, primals_205, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_443 = mul_118 = primals_205 = None
    getitem_123: "f32[4, 896, 14, 14]" = convolution_backward_41[0]
    getitem_124: "f32[896, 896, 1, 1]" = convolution_backward_41[1];  convolution_backward_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_445: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_123, relu_42)
    mul_446: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_123, sigmoid_10);  getitem_123 = None
    sum_78: "f32[4, 896, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_445, [2, 3], True);  mul_445 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sub_96: "f32[4, 896, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_10)
    mul_447: "f32[4, 896, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_10, sub_96);  sigmoid_10 = sub_96 = None
    mul_448: "f32[4, 896, 1, 1]" = torch.ops.aten.mul.Tensor(sum_78, mul_447);  sum_78 = mul_447 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    sum_79: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_448, [0, 2, 3])
    convolution_backward_42 = torch.ops.aten.convolution_backward.default(mul_448, relu_43, primals_203, [896], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_448 = primals_203 = None
    getitem_126: "f32[4, 224, 1, 1]" = convolution_backward_42[0]
    getitem_127: "f32[896, 224, 1, 1]" = convolution_backward_42[1];  convolution_backward_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    le_33: "b8[4, 224, 1, 1]" = torch.ops.aten.le.Scalar(relu_43, 0);  relu_43 = None
    where_33: "f32[4, 224, 1, 1]" = torch.ops.aten.where.self(le_33, full_default, getitem_126);  le_33 = getitem_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    sum_80: "f32[224]" = torch.ops.aten.sum.dim_IntList(where_33, [0, 2, 3])
    convolution_backward_43 = torch.ops.aten.convolution_backward.default(where_33, mean_10, primals_201, [224], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_33 = mean_10 = primals_201 = None
    getitem_129: "f32[4, 896, 1, 1]" = convolution_backward_43[0]
    getitem_130: "f32[224, 896, 1, 1]" = convolution_backward_43[1];  convolution_backward_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_9: "f32[4, 896, 14, 14]" = torch.ops.aten.expand.default(getitem_129, [4, 896, 14, 14]);  getitem_129 = None
    div_9: "f32[4, 896, 14, 14]" = torch.ops.aten.div.Scalar(expand_9, 196);  expand_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_185: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_446, div_9);  mul_446 = div_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_34: "b8[4, 896, 14, 14]" = torch.ops.aten.le.Scalar(relu_42, 0);  relu_42 = None
    where_34: "f32[4, 896, 14, 14]" = torch.ops.aten.where.self(le_34, full_default, add_185);  le_34 = add_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_186: "f32[896]" = torch.ops.aten.add.Tensor(primals_336, 1e-05);  primals_336 = None
    rsqrt_26: "f32[896]" = torch.ops.aten.rsqrt.default(add_186);  add_186 = None
    unsqueeze_808: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(primals_335, 0);  primals_335 = None
    unsqueeze_809: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_808, 2);  unsqueeze_808 = None
    unsqueeze_810: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_809, 3);  unsqueeze_809 = None
    sum_81: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_34, [0, 2, 3])
    sub_97: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_55, unsqueeze_810);  convolution_55 = unsqueeze_810 = None
    mul_449: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_34, sub_97);  sub_97 = None
    sum_82: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_449, [0, 2, 3]);  mul_449 = None
    mul_454: "f32[896]" = torch.ops.aten.mul.Tensor(rsqrt_26, primals_71);  primals_71 = None
    unsqueeze_817: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_454, 0);  mul_454 = None
    unsqueeze_818: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_817, 2);  unsqueeze_817 = None
    unsqueeze_819: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_818, 3);  unsqueeze_818 = None
    mul_455: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_34, unsqueeze_819);  where_34 = unsqueeze_819 = None
    mul_456: "f32[896]" = torch.ops.aten.mul.Tensor(sum_82, rsqrt_26);  sum_82 = rsqrt_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_44 = torch.ops.aten.convolution_backward.default(mul_455, relu_41, primals_200, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_455 = primals_200 = None
    getitem_132: "f32[4, 896, 14, 14]" = convolution_backward_44[0]
    getitem_133: "f32[896, 112, 3, 3]" = convolution_backward_44[1];  convolution_backward_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_35: "b8[4, 896, 14, 14]" = torch.ops.aten.le.Scalar(relu_41, 0);  relu_41 = None
    where_35: "f32[4, 896, 14, 14]" = torch.ops.aten.where.self(le_35, full_default, getitem_132);  le_35 = getitem_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_187: "f32[896]" = torch.ops.aten.add.Tensor(primals_334, 1e-05);  primals_334 = None
    rsqrt_27: "f32[896]" = torch.ops.aten.rsqrt.default(add_187);  add_187 = None
    unsqueeze_820: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(primals_333, 0);  primals_333 = None
    unsqueeze_821: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_820, 2);  unsqueeze_820 = None
    unsqueeze_822: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_821, 3);  unsqueeze_821 = None
    sum_83: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_35, [0, 2, 3])
    sub_98: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_54, unsqueeze_822);  convolution_54 = unsqueeze_822 = None
    mul_457: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_35, sub_98);  sub_98 = None
    sum_84: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_457, [0, 2, 3]);  mul_457 = None
    mul_462: "f32[896]" = torch.ops.aten.mul.Tensor(rsqrt_27, primals_69);  primals_69 = None
    unsqueeze_829: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_462, 0);  mul_462 = None
    unsqueeze_830: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_829, 2);  unsqueeze_829 = None
    unsqueeze_831: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_830, 3);  unsqueeze_830 = None
    mul_463: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_35, unsqueeze_831);  where_35 = unsqueeze_831 = None
    mul_464: "f32[896]" = torch.ops.aten.mul.Tensor(sum_84, rsqrt_27);  sum_84 = rsqrt_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_45 = torch.ops.aten.convolution_backward.default(mul_463, relu_40, primals_199, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_463 = primals_199 = None
    getitem_135: "f32[4, 896, 14, 14]" = convolution_backward_45[0]
    getitem_136: "f32[896, 896, 1, 1]" = convolution_backward_45[1];  convolution_backward_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_188: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(where_32, getitem_135);  where_32 = getitem_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    le_36: "b8[4, 896, 14, 14]" = torch.ops.aten.le.Scalar(relu_40, 0);  relu_40 = None
    where_36: "f32[4, 896, 14, 14]" = torch.ops.aten.where.self(le_36, full_default, add_188);  le_36 = add_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_189: "f32[896]" = torch.ops.aten.add.Tensor(primals_332, 1e-05);  primals_332 = None
    rsqrt_28: "f32[896]" = torch.ops.aten.rsqrt.default(add_189);  add_189 = None
    unsqueeze_832: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(primals_331, 0);  primals_331 = None
    unsqueeze_833: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_832, 2);  unsqueeze_832 = None
    unsqueeze_834: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_833, 3);  unsqueeze_833 = None
    sum_85: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_36, [0, 2, 3])
    sub_99: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_53, unsqueeze_834);  convolution_53 = unsqueeze_834 = None
    mul_465: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_36, sub_99);  sub_99 = None
    sum_86: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_465, [0, 2, 3]);  mul_465 = None
    mul_470: "f32[896]" = torch.ops.aten.mul.Tensor(rsqrt_28, primals_67);  primals_67 = None
    unsqueeze_841: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_470, 0);  mul_470 = None
    unsqueeze_842: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_841, 2);  unsqueeze_841 = None
    unsqueeze_843: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_842, 3);  unsqueeze_842 = None
    mul_471: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_36, unsqueeze_843);  unsqueeze_843 = None
    mul_472: "f32[896]" = torch.ops.aten.mul.Tensor(sum_86, rsqrt_28);  sum_86 = rsqrt_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_46 = torch.ops.aten.convolution_backward.default(mul_471, mul_108, primals_198, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_471 = mul_108 = primals_198 = None
    getitem_138: "f32[4, 896, 14, 14]" = convolution_backward_46[0]
    getitem_139: "f32[896, 896, 1, 1]" = convolution_backward_46[1];  convolution_backward_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_473: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_138, relu_38)
    mul_474: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_138, sigmoid_9);  getitem_138 = None
    sum_87: "f32[4, 896, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_473, [2, 3], True);  mul_473 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sub_100: "f32[4, 896, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_9)
    mul_475: "f32[4, 896, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_9, sub_100);  sigmoid_9 = sub_100 = None
    mul_476: "f32[4, 896, 1, 1]" = torch.ops.aten.mul.Tensor(sum_87, mul_475);  sum_87 = mul_475 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    sum_88: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_476, [0, 2, 3])
    convolution_backward_47 = torch.ops.aten.convolution_backward.default(mul_476, relu_39, primals_196, [896], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_476 = primals_196 = None
    getitem_141: "f32[4, 224, 1, 1]" = convolution_backward_47[0]
    getitem_142: "f32[896, 224, 1, 1]" = convolution_backward_47[1];  convolution_backward_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    le_37: "b8[4, 224, 1, 1]" = torch.ops.aten.le.Scalar(relu_39, 0);  relu_39 = None
    where_37: "f32[4, 224, 1, 1]" = torch.ops.aten.where.self(le_37, full_default, getitem_141);  le_37 = getitem_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    sum_89: "f32[224]" = torch.ops.aten.sum.dim_IntList(where_37, [0, 2, 3])
    convolution_backward_48 = torch.ops.aten.convolution_backward.default(where_37, mean_9, primals_194, [224], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_37 = mean_9 = primals_194 = None
    getitem_144: "f32[4, 896, 1, 1]" = convolution_backward_48[0]
    getitem_145: "f32[224, 896, 1, 1]" = convolution_backward_48[1];  convolution_backward_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_10: "f32[4, 896, 14, 14]" = torch.ops.aten.expand.default(getitem_144, [4, 896, 14, 14]);  getitem_144 = None
    div_10: "f32[4, 896, 14, 14]" = torch.ops.aten.div.Scalar(expand_10, 196);  expand_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_190: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_474, div_10);  mul_474 = div_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_38: "b8[4, 896, 14, 14]" = torch.ops.aten.le.Scalar(relu_38, 0);  relu_38 = None
    where_38: "f32[4, 896, 14, 14]" = torch.ops.aten.where.self(le_38, full_default, add_190);  le_38 = add_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_191: "f32[896]" = torch.ops.aten.add.Tensor(primals_330, 1e-05);  primals_330 = None
    rsqrt_29: "f32[896]" = torch.ops.aten.rsqrt.default(add_191);  add_191 = None
    unsqueeze_844: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(primals_329, 0);  primals_329 = None
    unsqueeze_845: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_844, 2);  unsqueeze_844 = None
    unsqueeze_846: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_845, 3);  unsqueeze_845 = None
    sum_90: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_38, [0, 2, 3])
    sub_101: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_846);  convolution_50 = unsqueeze_846 = None
    mul_477: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_38, sub_101);  sub_101 = None
    sum_91: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_477, [0, 2, 3]);  mul_477 = None
    mul_482: "f32[896]" = torch.ops.aten.mul.Tensor(rsqrt_29, primals_65);  primals_65 = None
    unsqueeze_853: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_482, 0);  mul_482 = None
    unsqueeze_854: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_853, 2);  unsqueeze_853 = None
    unsqueeze_855: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_854, 3);  unsqueeze_854 = None
    mul_483: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_38, unsqueeze_855);  where_38 = unsqueeze_855 = None
    mul_484: "f32[896]" = torch.ops.aten.mul.Tensor(sum_91, rsqrt_29);  sum_91 = rsqrt_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_49 = torch.ops.aten.convolution_backward.default(mul_483, relu_37, primals_193, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_483 = primals_193 = None
    getitem_147: "f32[4, 896, 14, 14]" = convolution_backward_49[0]
    getitem_148: "f32[896, 112, 3, 3]" = convolution_backward_49[1];  convolution_backward_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_39: "b8[4, 896, 14, 14]" = torch.ops.aten.le.Scalar(relu_37, 0);  relu_37 = None
    where_39: "f32[4, 896, 14, 14]" = torch.ops.aten.where.self(le_39, full_default, getitem_147);  le_39 = getitem_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_192: "f32[896]" = torch.ops.aten.add.Tensor(primals_328, 1e-05);  primals_328 = None
    rsqrt_30: "f32[896]" = torch.ops.aten.rsqrt.default(add_192);  add_192 = None
    unsqueeze_856: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(primals_327, 0);  primals_327 = None
    unsqueeze_857: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_856, 2);  unsqueeze_856 = None
    unsqueeze_858: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_857, 3);  unsqueeze_857 = None
    sum_92: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_39, [0, 2, 3])
    sub_102: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_49, unsqueeze_858);  convolution_49 = unsqueeze_858 = None
    mul_485: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_39, sub_102);  sub_102 = None
    sum_93: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_485, [0, 2, 3]);  mul_485 = None
    mul_490: "f32[896]" = torch.ops.aten.mul.Tensor(rsqrt_30, primals_63);  primals_63 = None
    unsqueeze_865: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_490, 0);  mul_490 = None
    unsqueeze_866: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_865, 2);  unsqueeze_865 = None
    unsqueeze_867: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_866, 3);  unsqueeze_866 = None
    mul_491: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_39, unsqueeze_867);  where_39 = unsqueeze_867 = None
    mul_492: "f32[896]" = torch.ops.aten.mul.Tensor(sum_93, rsqrt_30);  sum_93 = rsqrt_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_50 = torch.ops.aten.convolution_backward.default(mul_491, relu_36, primals_192, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_491 = primals_192 = None
    getitem_150: "f32[4, 896, 14, 14]" = convolution_backward_50[0]
    getitem_151: "f32[896, 896, 1, 1]" = convolution_backward_50[1];  convolution_backward_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_193: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(where_36, getitem_150);  where_36 = getitem_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    le_40: "b8[4, 896, 14, 14]" = torch.ops.aten.le.Scalar(relu_36, 0);  relu_36 = None
    where_40: "f32[4, 896, 14, 14]" = torch.ops.aten.where.self(le_40, full_default, add_193);  le_40 = add_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_194: "f32[896]" = torch.ops.aten.add.Tensor(primals_326, 1e-05);  primals_326 = None
    rsqrt_31: "f32[896]" = torch.ops.aten.rsqrt.default(add_194);  add_194 = None
    unsqueeze_868: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(primals_325, 0);  primals_325 = None
    unsqueeze_869: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_868, 2);  unsqueeze_868 = None
    unsqueeze_870: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_869, 3);  unsqueeze_869 = None
    sum_94: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_40, [0, 2, 3])
    sub_103: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_48, unsqueeze_870);  convolution_48 = unsqueeze_870 = None
    mul_493: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_40, sub_103);  sub_103 = None
    sum_95: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_493, [0, 2, 3]);  mul_493 = None
    mul_498: "f32[896]" = torch.ops.aten.mul.Tensor(rsqrt_31, primals_61);  primals_61 = None
    unsqueeze_877: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_498, 0);  mul_498 = None
    unsqueeze_878: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_877, 2);  unsqueeze_877 = None
    unsqueeze_879: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_878, 3);  unsqueeze_878 = None
    mul_499: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_40, unsqueeze_879);  unsqueeze_879 = None
    mul_500: "f32[896]" = torch.ops.aten.mul.Tensor(sum_95, rsqrt_31);  sum_95 = rsqrt_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_51 = torch.ops.aten.convolution_backward.default(mul_499, mul_98, primals_191, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_499 = mul_98 = primals_191 = None
    getitem_153: "f32[4, 896, 14, 14]" = convolution_backward_51[0]
    getitem_154: "f32[896, 896, 1, 1]" = convolution_backward_51[1];  convolution_backward_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_501: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_153, relu_34)
    mul_502: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_153, sigmoid_8);  getitem_153 = None
    sum_96: "f32[4, 896, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_501, [2, 3], True);  mul_501 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sub_104: "f32[4, 896, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_8)
    mul_503: "f32[4, 896, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_8, sub_104);  sigmoid_8 = sub_104 = None
    mul_504: "f32[4, 896, 1, 1]" = torch.ops.aten.mul.Tensor(sum_96, mul_503);  sum_96 = mul_503 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    sum_97: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_504, [0, 2, 3])
    convolution_backward_52 = torch.ops.aten.convolution_backward.default(mul_504, relu_35, primals_189, [896], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_504 = primals_189 = None
    getitem_156: "f32[4, 224, 1, 1]" = convolution_backward_52[0]
    getitem_157: "f32[896, 224, 1, 1]" = convolution_backward_52[1];  convolution_backward_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    le_41: "b8[4, 224, 1, 1]" = torch.ops.aten.le.Scalar(relu_35, 0);  relu_35 = None
    where_41: "f32[4, 224, 1, 1]" = torch.ops.aten.where.self(le_41, full_default, getitem_156);  le_41 = getitem_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    sum_98: "f32[224]" = torch.ops.aten.sum.dim_IntList(where_41, [0, 2, 3])
    convolution_backward_53 = torch.ops.aten.convolution_backward.default(where_41, mean_8, primals_187, [224], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_41 = mean_8 = primals_187 = None
    getitem_159: "f32[4, 896, 1, 1]" = convolution_backward_53[0]
    getitem_160: "f32[224, 896, 1, 1]" = convolution_backward_53[1];  convolution_backward_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_11: "f32[4, 896, 14, 14]" = torch.ops.aten.expand.default(getitem_159, [4, 896, 14, 14]);  getitem_159 = None
    div_11: "f32[4, 896, 14, 14]" = torch.ops.aten.div.Scalar(expand_11, 196);  expand_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_195: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_502, div_11);  mul_502 = div_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_42: "b8[4, 896, 14, 14]" = torch.ops.aten.le.Scalar(relu_34, 0);  relu_34 = None
    where_42: "f32[4, 896, 14, 14]" = torch.ops.aten.where.self(le_42, full_default, add_195);  le_42 = add_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_196: "f32[896]" = torch.ops.aten.add.Tensor(primals_324, 1e-05);  primals_324 = None
    rsqrt_32: "f32[896]" = torch.ops.aten.rsqrt.default(add_196);  add_196 = None
    unsqueeze_880: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(primals_323, 0);  primals_323 = None
    unsqueeze_881: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_880, 2);  unsqueeze_880 = None
    unsqueeze_882: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_881, 3);  unsqueeze_881 = None
    sum_99: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_42, [0, 2, 3])
    sub_105: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_882);  convolution_45 = unsqueeze_882 = None
    mul_505: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_42, sub_105);  sub_105 = None
    sum_100: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_505, [0, 2, 3]);  mul_505 = None
    mul_510: "f32[896]" = torch.ops.aten.mul.Tensor(rsqrt_32, primals_59);  primals_59 = None
    unsqueeze_889: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_510, 0);  mul_510 = None
    unsqueeze_890: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_889, 2);  unsqueeze_889 = None
    unsqueeze_891: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_890, 3);  unsqueeze_890 = None
    mul_511: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_42, unsqueeze_891);  where_42 = unsqueeze_891 = None
    mul_512: "f32[896]" = torch.ops.aten.mul.Tensor(sum_100, rsqrt_32);  sum_100 = rsqrt_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_54 = torch.ops.aten.convolution_backward.default(mul_511, relu_33, primals_186, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_511 = primals_186 = None
    getitem_162: "f32[4, 896, 14, 14]" = convolution_backward_54[0]
    getitem_163: "f32[896, 112, 3, 3]" = convolution_backward_54[1];  convolution_backward_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_43: "b8[4, 896, 14, 14]" = torch.ops.aten.le.Scalar(relu_33, 0);  relu_33 = None
    where_43: "f32[4, 896, 14, 14]" = torch.ops.aten.where.self(le_43, full_default, getitem_162);  le_43 = getitem_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_197: "f32[896]" = torch.ops.aten.add.Tensor(primals_322, 1e-05);  primals_322 = None
    rsqrt_33: "f32[896]" = torch.ops.aten.rsqrt.default(add_197);  add_197 = None
    unsqueeze_892: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(primals_321, 0);  primals_321 = None
    unsqueeze_893: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_892, 2);  unsqueeze_892 = None
    unsqueeze_894: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_893, 3);  unsqueeze_893 = None
    sum_101: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_43, [0, 2, 3])
    sub_106: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_44, unsqueeze_894);  convolution_44 = unsqueeze_894 = None
    mul_513: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_43, sub_106);  sub_106 = None
    sum_102: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_513, [0, 2, 3]);  mul_513 = None
    mul_518: "f32[896]" = torch.ops.aten.mul.Tensor(rsqrt_33, primals_57);  primals_57 = None
    unsqueeze_901: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_518, 0);  mul_518 = None
    unsqueeze_902: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_901, 2);  unsqueeze_901 = None
    unsqueeze_903: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_902, 3);  unsqueeze_902 = None
    mul_519: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_43, unsqueeze_903);  where_43 = unsqueeze_903 = None
    mul_520: "f32[896]" = torch.ops.aten.mul.Tensor(sum_102, rsqrt_33);  sum_102 = rsqrt_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_55 = torch.ops.aten.convolution_backward.default(mul_519, relu_32, primals_185, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_519 = primals_185 = None
    getitem_165: "f32[4, 896, 14, 14]" = convolution_backward_55[0]
    getitem_166: "f32[896, 896, 1, 1]" = convolution_backward_55[1];  convolution_backward_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_198: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(where_40, getitem_165);  where_40 = getitem_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    le_44: "b8[4, 896, 14, 14]" = torch.ops.aten.le.Scalar(relu_32, 0);  relu_32 = None
    where_44: "f32[4, 896, 14, 14]" = torch.ops.aten.where.self(le_44, full_default, add_198);  le_44 = add_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_199: "f32[896]" = torch.ops.aten.add.Tensor(primals_320, 1e-05);  primals_320 = None
    rsqrt_34: "f32[896]" = torch.ops.aten.rsqrt.default(add_199);  add_199 = None
    unsqueeze_904: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(primals_319, 0);  primals_319 = None
    unsqueeze_905: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_904, 2);  unsqueeze_904 = None
    unsqueeze_906: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_905, 3);  unsqueeze_905 = None
    sum_103: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_44, [0, 2, 3])
    sub_107: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_43, unsqueeze_906);  convolution_43 = unsqueeze_906 = None
    mul_521: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_44, sub_107);  sub_107 = None
    sum_104: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_521, [0, 2, 3]);  mul_521 = None
    mul_526: "f32[896]" = torch.ops.aten.mul.Tensor(rsqrt_34, primals_55);  primals_55 = None
    unsqueeze_913: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_526, 0);  mul_526 = None
    unsqueeze_914: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_913, 2);  unsqueeze_913 = None
    unsqueeze_915: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_914, 3);  unsqueeze_914 = None
    mul_527: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_44, unsqueeze_915);  unsqueeze_915 = None
    mul_528: "f32[896]" = torch.ops.aten.mul.Tensor(sum_104, rsqrt_34);  sum_104 = rsqrt_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_56 = torch.ops.aten.convolution_backward.default(mul_527, relu_28, primals_184, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_527 = primals_184 = None
    getitem_168: "f32[4, 448, 28, 28]" = convolution_backward_56[0]
    getitem_169: "f32[896, 448, 1, 1]" = convolution_backward_56[1];  convolution_backward_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_200: "f32[896]" = torch.ops.aten.add.Tensor(primals_318, 1e-05);  primals_318 = None
    rsqrt_35: "f32[896]" = torch.ops.aten.rsqrt.default(add_200);  add_200 = None
    unsqueeze_916: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(primals_317, 0);  primals_317 = None
    unsqueeze_917: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_916, 2);  unsqueeze_916 = None
    unsqueeze_918: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_917, 3);  unsqueeze_917 = None
    sub_108: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_42, unsqueeze_918);  convolution_42 = unsqueeze_918 = None
    mul_529: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_44, sub_108);  sub_108 = None
    sum_106: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_529, [0, 2, 3]);  mul_529 = None
    mul_534: "f32[896]" = torch.ops.aten.mul.Tensor(rsqrt_35, primals_53);  primals_53 = None
    unsqueeze_925: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_534, 0);  mul_534 = None
    unsqueeze_926: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_925, 2);  unsqueeze_925 = None
    unsqueeze_927: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_926, 3);  unsqueeze_926 = None
    mul_535: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_44, unsqueeze_927);  where_44 = unsqueeze_927 = None
    mul_536: "f32[896]" = torch.ops.aten.mul.Tensor(sum_106, rsqrt_35);  sum_106 = rsqrt_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_57 = torch.ops.aten.convolution_backward.default(mul_535, mul_85, primals_183, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_535 = mul_85 = primals_183 = None
    getitem_171: "f32[4, 896, 14, 14]" = convolution_backward_57[0]
    getitem_172: "f32[896, 896, 1, 1]" = convolution_backward_57[1];  convolution_backward_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_537: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_171, relu_30)
    mul_538: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_171, sigmoid_7);  getitem_171 = None
    sum_107: "f32[4, 896, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_537, [2, 3], True);  mul_537 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sub_109: "f32[4, 896, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_7)
    mul_539: "f32[4, 896, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_7, sub_109);  sigmoid_7 = sub_109 = None
    mul_540: "f32[4, 896, 1, 1]" = torch.ops.aten.mul.Tensor(sum_107, mul_539);  sum_107 = mul_539 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    sum_108: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_540, [0, 2, 3])
    convolution_backward_58 = torch.ops.aten.convolution_backward.default(mul_540, relu_31, primals_181, [896], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_540 = primals_181 = None
    getitem_174: "f32[4, 112, 1, 1]" = convolution_backward_58[0]
    getitem_175: "f32[896, 112, 1, 1]" = convolution_backward_58[1];  convolution_backward_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    le_45: "b8[4, 112, 1, 1]" = torch.ops.aten.le.Scalar(relu_31, 0);  relu_31 = None
    where_45: "f32[4, 112, 1, 1]" = torch.ops.aten.where.self(le_45, full_default, getitem_174);  le_45 = getitem_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    sum_109: "f32[112]" = torch.ops.aten.sum.dim_IntList(where_45, [0, 2, 3])
    convolution_backward_59 = torch.ops.aten.convolution_backward.default(where_45, mean_7, primals_179, [112], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_45 = mean_7 = primals_179 = None
    getitem_177: "f32[4, 896, 1, 1]" = convolution_backward_59[0]
    getitem_178: "f32[112, 896, 1, 1]" = convolution_backward_59[1];  convolution_backward_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_12: "f32[4, 896, 14, 14]" = torch.ops.aten.expand.default(getitem_177, [4, 896, 14, 14]);  getitem_177 = None
    div_12: "f32[4, 896, 14, 14]" = torch.ops.aten.div.Scalar(expand_12, 196);  expand_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_201: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_538, div_12);  mul_538 = div_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_46: "b8[4, 896, 14, 14]" = torch.ops.aten.le.Scalar(relu_30, 0);  relu_30 = None
    where_46: "f32[4, 896, 14, 14]" = torch.ops.aten.where.self(le_46, full_default, add_201);  le_46 = add_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_202: "f32[896]" = torch.ops.aten.add.Tensor(primals_316, 1e-05);  primals_316 = None
    rsqrt_36: "f32[896]" = torch.ops.aten.rsqrt.default(add_202);  add_202 = None
    unsqueeze_928: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(primals_315, 0);  primals_315 = None
    unsqueeze_929: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_928, 2);  unsqueeze_928 = None
    unsqueeze_930: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_929, 3);  unsqueeze_929 = None
    sum_110: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_46, [0, 2, 3])
    sub_110: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_39, unsqueeze_930);  convolution_39 = unsqueeze_930 = None
    mul_541: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_46, sub_110);  sub_110 = None
    sum_111: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_541, [0, 2, 3]);  mul_541 = None
    mul_546: "f32[896]" = torch.ops.aten.mul.Tensor(rsqrt_36, primals_51);  primals_51 = None
    unsqueeze_937: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_546, 0);  mul_546 = None
    unsqueeze_938: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_937, 2);  unsqueeze_937 = None
    unsqueeze_939: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_938, 3);  unsqueeze_938 = None
    mul_547: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_46, unsqueeze_939);  where_46 = unsqueeze_939 = None
    mul_548: "f32[896]" = torch.ops.aten.mul.Tensor(sum_111, rsqrt_36);  sum_111 = rsqrt_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_60 = torch.ops.aten.convolution_backward.default(mul_547, relu_29, primals_178, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_547 = primals_178 = None
    getitem_180: "f32[4, 896, 28, 28]" = convolution_backward_60[0]
    getitem_181: "f32[896, 112, 3, 3]" = convolution_backward_60[1];  convolution_backward_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_47: "b8[4, 896, 28, 28]" = torch.ops.aten.le.Scalar(relu_29, 0);  relu_29 = None
    where_47: "f32[4, 896, 28, 28]" = torch.ops.aten.where.self(le_47, full_default, getitem_180);  le_47 = getitem_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_203: "f32[896]" = torch.ops.aten.add.Tensor(primals_314, 1e-05);  primals_314 = None
    rsqrt_37: "f32[896]" = torch.ops.aten.rsqrt.default(add_203);  add_203 = None
    unsqueeze_940: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(primals_313, 0);  primals_313 = None
    unsqueeze_941: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_940, 2);  unsqueeze_940 = None
    unsqueeze_942: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_941, 3);  unsqueeze_941 = None
    sum_112: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_47, [0, 2, 3])
    sub_111: "f32[4, 896, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_38, unsqueeze_942);  convolution_38 = unsqueeze_942 = None
    mul_549: "f32[4, 896, 28, 28]" = torch.ops.aten.mul.Tensor(where_47, sub_111);  sub_111 = None
    sum_113: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_549, [0, 2, 3]);  mul_549 = None
    mul_554: "f32[896]" = torch.ops.aten.mul.Tensor(rsqrt_37, primals_49);  primals_49 = None
    unsqueeze_949: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_554, 0);  mul_554 = None
    unsqueeze_950: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_949, 2);  unsqueeze_949 = None
    unsqueeze_951: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_950, 3);  unsqueeze_950 = None
    mul_555: "f32[4, 896, 28, 28]" = torch.ops.aten.mul.Tensor(where_47, unsqueeze_951);  where_47 = unsqueeze_951 = None
    mul_556: "f32[896]" = torch.ops.aten.mul.Tensor(sum_113, rsqrt_37);  sum_113 = rsqrt_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_61 = torch.ops.aten.convolution_backward.default(mul_555, relu_28, primals_177, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_555 = primals_177 = None
    getitem_183: "f32[4, 448, 28, 28]" = convolution_backward_61[0]
    getitem_184: "f32[896, 448, 1, 1]" = convolution_backward_61[1];  convolution_backward_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_204: "f32[4, 448, 28, 28]" = torch.ops.aten.add.Tensor(getitem_168, getitem_183);  getitem_168 = getitem_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    le_48: "b8[4, 448, 28, 28]" = torch.ops.aten.le.Scalar(relu_28, 0);  relu_28 = None
    where_48: "f32[4, 448, 28, 28]" = torch.ops.aten.where.self(le_48, full_default, add_204);  le_48 = add_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_205: "f32[448]" = torch.ops.aten.add.Tensor(primals_312, 1e-05);  primals_312 = None
    rsqrt_38: "f32[448]" = torch.ops.aten.rsqrt.default(add_205);  add_205 = None
    unsqueeze_952: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(primals_311, 0);  primals_311 = None
    unsqueeze_953: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_952, 2);  unsqueeze_952 = None
    unsqueeze_954: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_953, 3);  unsqueeze_953 = None
    sum_114: "f32[448]" = torch.ops.aten.sum.dim_IntList(where_48, [0, 2, 3])
    sub_112: "f32[4, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_37, unsqueeze_954);  convolution_37 = unsqueeze_954 = None
    mul_557: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(where_48, sub_112);  sub_112 = None
    sum_115: "f32[448]" = torch.ops.aten.sum.dim_IntList(mul_557, [0, 2, 3]);  mul_557 = None
    mul_562: "f32[448]" = torch.ops.aten.mul.Tensor(rsqrt_38, primals_47);  primals_47 = None
    unsqueeze_961: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_562, 0);  mul_562 = None
    unsqueeze_962: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_961, 2);  unsqueeze_961 = None
    unsqueeze_963: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_962, 3);  unsqueeze_962 = None
    mul_563: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(where_48, unsqueeze_963);  unsqueeze_963 = None
    mul_564: "f32[448]" = torch.ops.aten.mul.Tensor(sum_115, rsqrt_38);  sum_115 = rsqrt_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_62 = torch.ops.aten.convolution_backward.default(mul_563, mul_75, primals_176, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_563 = mul_75 = primals_176 = None
    getitem_186: "f32[4, 448, 28, 28]" = convolution_backward_62[0]
    getitem_187: "f32[448, 448, 1, 1]" = convolution_backward_62[1];  convolution_backward_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_565: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_186, relu_26)
    mul_566: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_186, sigmoid_6);  getitem_186 = None
    sum_116: "f32[4, 448, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_565, [2, 3], True);  mul_565 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sub_113: "f32[4, 448, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_6)
    mul_567: "f32[4, 448, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_6, sub_113);  sigmoid_6 = sub_113 = None
    mul_568: "f32[4, 448, 1, 1]" = torch.ops.aten.mul.Tensor(sum_116, mul_567);  sum_116 = mul_567 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    sum_117: "f32[448]" = torch.ops.aten.sum.dim_IntList(mul_568, [0, 2, 3])
    convolution_backward_63 = torch.ops.aten.convolution_backward.default(mul_568, relu_27, primals_174, [448], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_568 = primals_174 = None
    getitem_189: "f32[4, 112, 1, 1]" = convolution_backward_63[0]
    getitem_190: "f32[448, 112, 1, 1]" = convolution_backward_63[1];  convolution_backward_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    le_49: "b8[4, 112, 1, 1]" = torch.ops.aten.le.Scalar(relu_27, 0);  relu_27 = None
    where_49: "f32[4, 112, 1, 1]" = torch.ops.aten.where.self(le_49, full_default, getitem_189);  le_49 = getitem_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    sum_118: "f32[112]" = torch.ops.aten.sum.dim_IntList(where_49, [0, 2, 3])
    convolution_backward_64 = torch.ops.aten.convolution_backward.default(where_49, mean_6, primals_172, [112], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_49 = mean_6 = primals_172 = None
    getitem_192: "f32[4, 448, 1, 1]" = convolution_backward_64[0]
    getitem_193: "f32[112, 448, 1, 1]" = convolution_backward_64[1];  convolution_backward_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_13: "f32[4, 448, 28, 28]" = torch.ops.aten.expand.default(getitem_192, [4, 448, 28, 28]);  getitem_192 = None
    div_13: "f32[4, 448, 28, 28]" = torch.ops.aten.div.Scalar(expand_13, 784);  expand_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_206: "f32[4, 448, 28, 28]" = torch.ops.aten.add.Tensor(mul_566, div_13);  mul_566 = div_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_50: "b8[4, 448, 28, 28]" = torch.ops.aten.le.Scalar(relu_26, 0);  relu_26 = None
    where_50: "f32[4, 448, 28, 28]" = torch.ops.aten.where.self(le_50, full_default, add_206);  le_50 = add_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_207: "f32[448]" = torch.ops.aten.add.Tensor(primals_310, 1e-05);  primals_310 = None
    rsqrt_39: "f32[448]" = torch.ops.aten.rsqrt.default(add_207);  add_207 = None
    unsqueeze_964: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(primals_309, 0);  primals_309 = None
    unsqueeze_965: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_964, 2);  unsqueeze_964 = None
    unsqueeze_966: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_965, 3);  unsqueeze_965 = None
    sum_119: "f32[448]" = torch.ops.aten.sum.dim_IntList(where_50, [0, 2, 3])
    sub_114: "f32[4, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_966);  convolution_34 = unsqueeze_966 = None
    mul_569: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(where_50, sub_114);  sub_114 = None
    sum_120: "f32[448]" = torch.ops.aten.sum.dim_IntList(mul_569, [0, 2, 3]);  mul_569 = None
    mul_574: "f32[448]" = torch.ops.aten.mul.Tensor(rsqrt_39, primals_45);  primals_45 = None
    unsqueeze_973: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_574, 0);  mul_574 = None
    unsqueeze_974: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_973, 2);  unsqueeze_973 = None
    unsqueeze_975: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_974, 3);  unsqueeze_974 = None
    mul_575: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(where_50, unsqueeze_975);  where_50 = unsqueeze_975 = None
    mul_576: "f32[448]" = torch.ops.aten.mul.Tensor(sum_120, rsqrt_39);  sum_120 = rsqrt_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_65 = torch.ops.aten.convolution_backward.default(mul_575, relu_25, primals_171, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 4, [True, True, False]);  mul_575 = primals_171 = None
    getitem_195: "f32[4, 448, 28, 28]" = convolution_backward_65[0]
    getitem_196: "f32[448, 112, 3, 3]" = convolution_backward_65[1];  convolution_backward_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_51: "b8[4, 448, 28, 28]" = torch.ops.aten.le.Scalar(relu_25, 0);  relu_25 = None
    where_51: "f32[4, 448, 28, 28]" = torch.ops.aten.where.self(le_51, full_default, getitem_195);  le_51 = getitem_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_208: "f32[448]" = torch.ops.aten.add.Tensor(primals_308, 1e-05);  primals_308 = None
    rsqrt_40: "f32[448]" = torch.ops.aten.rsqrt.default(add_208);  add_208 = None
    unsqueeze_976: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(primals_307, 0);  primals_307 = None
    unsqueeze_977: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_976, 2);  unsqueeze_976 = None
    unsqueeze_978: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_977, 3);  unsqueeze_977 = None
    sum_121: "f32[448]" = torch.ops.aten.sum.dim_IntList(where_51, [0, 2, 3])
    sub_115: "f32[4, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_978);  convolution_33 = unsqueeze_978 = None
    mul_577: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(where_51, sub_115);  sub_115 = None
    sum_122: "f32[448]" = torch.ops.aten.sum.dim_IntList(mul_577, [0, 2, 3]);  mul_577 = None
    mul_582: "f32[448]" = torch.ops.aten.mul.Tensor(rsqrt_40, primals_43);  primals_43 = None
    unsqueeze_985: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_582, 0);  mul_582 = None
    unsqueeze_986: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_985, 2);  unsqueeze_985 = None
    unsqueeze_987: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_986, 3);  unsqueeze_986 = None
    mul_583: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(where_51, unsqueeze_987);  where_51 = unsqueeze_987 = None
    mul_584: "f32[448]" = torch.ops.aten.mul.Tensor(sum_122, rsqrt_40);  sum_122 = rsqrt_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_66 = torch.ops.aten.convolution_backward.default(mul_583, relu_24, primals_170, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_583 = primals_170 = None
    getitem_198: "f32[4, 448, 28, 28]" = convolution_backward_66[0]
    getitem_199: "f32[448, 448, 1, 1]" = convolution_backward_66[1];  convolution_backward_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_209: "f32[4, 448, 28, 28]" = torch.ops.aten.add.Tensor(where_48, getitem_198);  where_48 = getitem_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    le_52: "b8[4, 448, 28, 28]" = torch.ops.aten.le.Scalar(relu_24, 0);  relu_24 = None
    where_52: "f32[4, 448, 28, 28]" = torch.ops.aten.where.self(le_52, full_default, add_209);  le_52 = add_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_210: "f32[448]" = torch.ops.aten.add.Tensor(primals_306, 1e-05);  primals_306 = None
    rsqrt_41: "f32[448]" = torch.ops.aten.rsqrt.default(add_210);  add_210 = None
    unsqueeze_988: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(primals_305, 0);  primals_305 = None
    unsqueeze_989: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_988, 2);  unsqueeze_988 = None
    unsqueeze_990: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_989, 3);  unsqueeze_989 = None
    sum_123: "f32[448]" = torch.ops.aten.sum.dim_IntList(where_52, [0, 2, 3])
    sub_116: "f32[4, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_990);  convolution_32 = unsqueeze_990 = None
    mul_585: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(where_52, sub_116);  sub_116 = None
    sum_124: "f32[448]" = torch.ops.aten.sum.dim_IntList(mul_585, [0, 2, 3]);  mul_585 = None
    mul_590: "f32[448]" = torch.ops.aten.mul.Tensor(rsqrt_41, primals_41);  primals_41 = None
    unsqueeze_997: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_590, 0);  mul_590 = None
    unsqueeze_998: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_997, 2);  unsqueeze_997 = None
    unsqueeze_999: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_998, 3);  unsqueeze_998 = None
    mul_591: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(where_52, unsqueeze_999);  unsqueeze_999 = None
    mul_592: "f32[448]" = torch.ops.aten.mul.Tensor(sum_124, rsqrt_41);  sum_124 = rsqrt_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_67 = torch.ops.aten.convolution_backward.default(mul_591, mul_65, primals_169, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_591 = mul_65 = primals_169 = None
    getitem_201: "f32[4, 448, 28, 28]" = convolution_backward_67[0]
    getitem_202: "f32[448, 448, 1, 1]" = convolution_backward_67[1];  convolution_backward_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_593: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_201, relu_22)
    mul_594: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_201, sigmoid_5);  getitem_201 = None
    sum_125: "f32[4, 448, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_593, [2, 3], True);  mul_593 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sub_117: "f32[4, 448, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_5)
    mul_595: "f32[4, 448, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_5, sub_117);  sigmoid_5 = sub_117 = None
    mul_596: "f32[4, 448, 1, 1]" = torch.ops.aten.mul.Tensor(sum_125, mul_595);  sum_125 = mul_595 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    sum_126: "f32[448]" = torch.ops.aten.sum.dim_IntList(mul_596, [0, 2, 3])
    convolution_backward_68 = torch.ops.aten.convolution_backward.default(mul_596, relu_23, primals_167, [448], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_596 = primals_167 = None
    getitem_204: "f32[4, 112, 1, 1]" = convolution_backward_68[0]
    getitem_205: "f32[448, 112, 1, 1]" = convolution_backward_68[1];  convolution_backward_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    le_53: "b8[4, 112, 1, 1]" = torch.ops.aten.le.Scalar(relu_23, 0);  relu_23 = None
    where_53: "f32[4, 112, 1, 1]" = torch.ops.aten.where.self(le_53, full_default, getitem_204);  le_53 = getitem_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    sum_127: "f32[112]" = torch.ops.aten.sum.dim_IntList(where_53, [0, 2, 3])
    convolution_backward_69 = torch.ops.aten.convolution_backward.default(where_53, mean_5, primals_165, [112], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_53 = mean_5 = primals_165 = None
    getitem_207: "f32[4, 448, 1, 1]" = convolution_backward_69[0]
    getitem_208: "f32[112, 448, 1, 1]" = convolution_backward_69[1];  convolution_backward_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_14: "f32[4, 448, 28, 28]" = torch.ops.aten.expand.default(getitem_207, [4, 448, 28, 28]);  getitem_207 = None
    div_14: "f32[4, 448, 28, 28]" = torch.ops.aten.div.Scalar(expand_14, 784);  expand_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_211: "f32[4, 448, 28, 28]" = torch.ops.aten.add.Tensor(mul_594, div_14);  mul_594 = div_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_54: "b8[4, 448, 28, 28]" = torch.ops.aten.le.Scalar(relu_22, 0);  relu_22 = None
    where_54: "f32[4, 448, 28, 28]" = torch.ops.aten.where.self(le_54, full_default, add_211);  le_54 = add_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_212: "f32[448]" = torch.ops.aten.add.Tensor(primals_304, 1e-05);  primals_304 = None
    rsqrt_42: "f32[448]" = torch.ops.aten.rsqrt.default(add_212);  add_212 = None
    unsqueeze_1000: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(primals_303, 0);  primals_303 = None
    unsqueeze_1001: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1000, 2);  unsqueeze_1000 = None
    unsqueeze_1002: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1001, 3);  unsqueeze_1001 = None
    sum_128: "f32[448]" = torch.ops.aten.sum.dim_IntList(where_54, [0, 2, 3])
    sub_118: "f32[4, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_1002);  convolution_29 = unsqueeze_1002 = None
    mul_597: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(where_54, sub_118);  sub_118 = None
    sum_129: "f32[448]" = torch.ops.aten.sum.dim_IntList(mul_597, [0, 2, 3]);  mul_597 = None
    mul_602: "f32[448]" = torch.ops.aten.mul.Tensor(rsqrt_42, primals_39);  primals_39 = None
    unsqueeze_1009: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_602, 0);  mul_602 = None
    unsqueeze_1010: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1009, 2);  unsqueeze_1009 = None
    unsqueeze_1011: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1010, 3);  unsqueeze_1010 = None
    mul_603: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(where_54, unsqueeze_1011);  where_54 = unsqueeze_1011 = None
    mul_604: "f32[448]" = torch.ops.aten.mul.Tensor(sum_129, rsqrt_42);  sum_129 = rsqrt_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_70 = torch.ops.aten.convolution_backward.default(mul_603, relu_21, primals_164, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 4, [True, True, False]);  mul_603 = primals_164 = None
    getitem_210: "f32[4, 448, 28, 28]" = convolution_backward_70[0]
    getitem_211: "f32[448, 112, 3, 3]" = convolution_backward_70[1];  convolution_backward_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_55: "b8[4, 448, 28, 28]" = torch.ops.aten.le.Scalar(relu_21, 0);  relu_21 = None
    where_55: "f32[4, 448, 28, 28]" = torch.ops.aten.where.self(le_55, full_default, getitem_210);  le_55 = getitem_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_213: "f32[448]" = torch.ops.aten.add.Tensor(primals_302, 1e-05);  primals_302 = None
    rsqrt_43: "f32[448]" = torch.ops.aten.rsqrt.default(add_213);  add_213 = None
    unsqueeze_1012: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(primals_301, 0);  primals_301 = None
    unsqueeze_1013: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1012, 2);  unsqueeze_1012 = None
    unsqueeze_1014: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1013, 3);  unsqueeze_1013 = None
    sum_130: "f32[448]" = torch.ops.aten.sum.dim_IntList(where_55, [0, 2, 3])
    sub_119: "f32[4, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_1014);  convolution_28 = unsqueeze_1014 = None
    mul_605: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(where_55, sub_119);  sub_119 = None
    sum_131: "f32[448]" = torch.ops.aten.sum.dim_IntList(mul_605, [0, 2, 3]);  mul_605 = None
    mul_610: "f32[448]" = torch.ops.aten.mul.Tensor(rsqrt_43, primals_37);  primals_37 = None
    unsqueeze_1021: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_610, 0);  mul_610 = None
    unsqueeze_1022: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1021, 2);  unsqueeze_1021 = None
    unsqueeze_1023: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1022, 3);  unsqueeze_1022 = None
    mul_611: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(where_55, unsqueeze_1023);  where_55 = unsqueeze_1023 = None
    mul_612: "f32[448]" = torch.ops.aten.mul.Tensor(sum_131, rsqrt_43);  sum_131 = rsqrt_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_71 = torch.ops.aten.convolution_backward.default(mul_611, relu_20, primals_163, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_611 = primals_163 = None
    getitem_213: "f32[4, 448, 28, 28]" = convolution_backward_71[0]
    getitem_214: "f32[448, 448, 1, 1]" = convolution_backward_71[1];  convolution_backward_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_214: "f32[4, 448, 28, 28]" = torch.ops.aten.add.Tensor(where_52, getitem_213);  where_52 = getitem_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    le_56: "b8[4, 448, 28, 28]" = torch.ops.aten.le.Scalar(relu_20, 0);  relu_20 = None
    where_56: "f32[4, 448, 28, 28]" = torch.ops.aten.where.self(le_56, full_default, add_214);  le_56 = add_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_215: "f32[448]" = torch.ops.aten.add.Tensor(primals_300, 1e-05);  primals_300 = None
    rsqrt_44: "f32[448]" = torch.ops.aten.rsqrt.default(add_215);  add_215 = None
    unsqueeze_1024: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(primals_299, 0);  primals_299 = None
    unsqueeze_1025: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1024, 2);  unsqueeze_1024 = None
    unsqueeze_1026: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1025, 3);  unsqueeze_1025 = None
    sum_132: "f32[448]" = torch.ops.aten.sum.dim_IntList(where_56, [0, 2, 3])
    sub_120: "f32[4, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_1026);  convolution_27 = unsqueeze_1026 = None
    mul_613: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(where_56, sub_120);  sub_120 = None
    sum_133: "f32[448]" = torch.ops.aten.sum.dim_IntList(mul_613, [0, 2, 3]);  mul_613 = None
    mul_618: "f32[448]" = torch.ops.aten.mul.Tensor(rsqrt_44, primals_35);  primals_35 = None
    unsqueeze_1033: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_618, 0);  mul_618 = None
    unsqueeze_1034: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1033, 2);  unsqueeze_1033 = None
    unsqueeze_1035: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1034, 3);  unsqueeze_1034 = None
    mul_619: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(where_56, unsqueeze_1035);  unsqueeze_1035 = None
    mul_620: "f32[448]" = torch.ops.aten.mul.Tensor(sum_133, rsqrt_44);  sum_133 = rsqrt_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_72 = torch.ops.aten.convolution_backward.default(mul_619, mul_55, primals_162, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_619 = mul_55 = primals_162 = None
    getitem_216: "f32[4, 448, 28, 28]" = convolution_backward_72[0]
    getitem_217: "f32[448, 448, 1, 1]" = convolution_backward_72[1];  convolution_backward_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_621: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_216, relu_18)
    mul_622: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_216, sigmoid_4);  getitem_216 = None
    sum_134: "f32[4, 448, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_621, [2, 3], True);  mul_621 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sub_121: "f32[4, 448, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_4)
    mul_623: "f32[4, 448, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_4, sub_121);  sigmoid_4 = sub_121 = None
    mul_624: "f32[4, 448, 1, 1]" = torch.ops.aten.mul.Tensor(sum_134, mul_623);  sum_134 = mul_623 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    sum_135: "f32[448]" = torch.ops.aten.sum.dim_IntList(mul_624, [0, 2, 3])
    convolution_backward_73 = torch.ops.aten.convolution_backward.default(mul_624, relu_19, primals_160, [448], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_624 = primals_160 = None
    getitem_219: "f32[4, 112, 1, 1]" = convolution_backward_73[0]
    getitem_220: "f32[448, 112, 1, 1]" = convolution_backward_73[1];  convolution_backward_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    le_57: "b8[4, 112, 1, 1]" = torch.ops.aten.le.Scalar(relu_19, 0);  relu_19 = None
    where_57: "f32[4, 112, 1, 1]" = torch.ops.aten.where.self(le_57, full_default, getitem_219);  le_57 = getitem_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    sum_136: "f32[112]" = torch.ops.aten.sum.dim_IntList(where_57, [0, 2, 3])
    convolution_backward_74 = torch.ops.aten.convolution_backward.default(where_57, mean_4, primals_158, [112], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_57 = mean_4 = primals_158 = None
    getitem_222: "f32[4, 448, 1, 1]" = convolution_backward_74[0]
    getitem_223: "f32[112, 448, 1, 1]" = convolution_backward_74[1];  convolution_backward_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_15: "f32[4, 448, 28, 28]" = torch.ops.aten.expand.default(getitem_222, [4, 448, 28, 28]);  getitem_222 = None
    div_15: "f32[4, 448, 28, 28]" = torch.ops.aten.div.Scalar(expand_15, 784);  expand_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_216: "f32[4, 448, 28, 28]" = torch.ops.aten.add.Tensor(mul_622, div_15);  mul_622 = div_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_58: "b8[4, 448, 28, 28]" = torch.ops.aten.le.Scalar(relu_18, 0);  relu_18 = None
    where_58: "f32[4, 448, 28, 28]" = torch.ops.aten.where.self(le_58, full_default, add_216);  le_58 = add_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_217: "f32[448]" = torch.ops.aten.add.Tensor(primals_298, 1e-05);  primals_298 = None
    rsqrt_45: "f32[448]" = torch.ops.aten.rsqrt.default(add_217);  add_217 = None
    unsqueeze_1036: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(primals_297, 0);  primals_297 = None
    unsqueeze_1037: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1036, 2);  unsqueeze_1036 = None
    unsqueeze_1038: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1037, 3);  unsqueeze_1037 = None
    sum_137: "f32[448]" = torch.ops.aten.sum.dim_IntList(where_58, [0, 2, 3])
    sub_122: "f32[4, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_1038);  convolution_24 = unsqueeze_1038 = None
    mul_625: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(where_58, sub_122);  sub_122 = None
    sum_138: "f32[448]" = torch.ops.aten.sum.dim_IntList(mul_625, [0, 2, 3]);  mul_625 = None
    mul_630: "f32[448]" = torch.ops.aten.mul.Tensor(rsqrt_45, primals_33);  primals_33 = None
    unsqueeze_1045: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_630, 0);  mul_630 = None
    unsqueeze_1046: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1045, 2);  unsqueeze_1045 = None
    unsqueeze_1047: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1046, 3);  unsqueeze_1046 = None
    mul_631: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(where_58, unsqueeze_1047);  where_58 = unsqueeze_1047 = None
    mul_632: "f32[448]" = torch.ops.aten.mul.Tensor(sum_138, rsqrt_45);  sum_138 = rsqrt_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_75 = torch.ops.aten.convolution_backward.default(mul_631, relu_17, primals_157, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 4, [True, True, False]);  mul_631 = primals_157 = None
    getitem_225: "f32[4, 448, 28, 28]" = convolution_backward_75[0]
    getitem_226: "f32[448, 112, 3, 3]" = convolution_backward_75[1];  convolution_backward_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_59: "b8[4, 448, 28, 28]" = torch.ops.aten.le.Scalar(relu_17, 0);  relu_17 = None
    where_59: "f32[4, 448, 28, 28]" = torch.ops.aten.where.self(le_59, full_default, getitem_225);  le_59 = getitem_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_218: "f32[448]" = torch.ops.aten.add.Tensor(primals_296, 1e-05);  primals_296 = None
    rsqrt_46: "f32[448]" = torch.ops.aten.rsqrt.default(add_218);  add_218 = None
    unsqueeze_1048: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(primals_295, 0);  primals_295 = None
    unsqueeze_1049: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1048, 2);  unsqueeze_1048 = None
    unsqueeze_1050: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1049, 3);  unsqueeze_1049 = None
    sum_139: "f32[448]" = torch.ops.aten.sum.dim_IntList(where_59, [0, 2, 3])
    sub_123: "f32[4, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_1050);  convolution_23 = unsqueeze_1050 = None
    mul_633: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(where_59, sub_123);  sub_123 = None
    sum_140: "f32[448]" = torch.ops.aten.sum.dim_IntList(mul_633, [0, 2, 3]);  mul_633 = None
    mul_638: "f32[448]" = torch.ops.aten.mul.Tensor(rsqrt_46, primals_31);  primals_31 = None
    unsqueeze_1057: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_638, 0);  mul_638 = None
    unsqueeze_1058: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1057, 2);  unsqueeze_1057 = None
    unsqueeze_1059: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1058, 3);  unsqueeze_1058 = None
    mul_639: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(where_59, unsqueeze_1059);  where_59 = unsqueeze_1059 = None
    mul_640: "f32[448]" = torch.ops.aten.mul.Tensor(sum_140, rsqrt_46);  sum_140 = rsqrt_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_76 = torch.ops.aten.convolution_backward.default(mul_639, relu_16, primals_156, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_639 = primals_156 = None
    getitem_228: "f32[4, 448, 28, 28]" = convolution_backward_76[0]
    getitem_229: "f32[448, 448, 1, 1]" = convolution_backward_76[1];  convolution_backward_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_219: "f32[4, 448, 28, 28]" = torch.ops.aten.add.Tensor(where_56, getitem_228);  where_56 = getitem_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    le_60: "b8[4, 448, 28, 28]" = torch.ops.aten.le.Scalar(relu_16, 0);  relu_16 = None
    where_60: "f32[4, 448, 28, 28]" = torch.ops.aten.where.self(le_60, full_default, add_219);  le_60 = add_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_220: "f32[448]" = torch.ops.aten.add.Tensor(primals_294, 1e-05);  primals_294 = None
    rsqrt_47: "f32[448]" = torch.ops.aten.rsqrt.default(add_220);  add_220 = None
    unsqueeze_1060: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(primals_293, 0);  primals_293 = None
    unsqueeze_1061: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1060, 2);  unsqueeze_1060 = None
    unsqueeze_1062: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1061, 3);  unsqueeze_1061 = None
    sum_141: "f32[448]" = torch.ops.aten.sum.dim_IntList(where_60, [0, 2, 3])
    sub_124: "f32[4, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_1062);  convolution_22 = unsqueeze_1062 = None
    mul_641: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(where_60, sub_124);  sub_124 = None
    sum_142: "f32[448]" = torch.ops.aten.sum.dim_IntList(mul_641, [0, 2, 3]);  mul_641 = None
    mul_646: "f32[448]" = torch.ops.aten.mul.Tensor(rsqrt_47, primals_29);  primals_29 = None
    unsqueeze_1069: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_646, 0);  mul_646 = None
    unsqueeze_1070: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1069, 2);  unsqueeze_1069 = None
    unsqueeze_1071: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1070, 3);  unsqueeze_1070 = None
    mul_647: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(where_60, unsqueeze_1071);  unsqueeze_1071 = None
    mul_648: "f32[448]" = torch.ops.aten.mul.Tensor(sum_142, rsqrt_47);  sum_142 = rsqrt_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_77 = torch.ops.aten.convolution_backward.default(mul_647, mul_45, primals_155, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_647 = mul_45 = primals_155 = None
    getitem_231: "f32[4, 448, 28, 28]" = convolution_backward_77[0]
    getitem_232: "f32[448, 448, 1, 1]" = convolution_backward_77[1];  convolution_backward_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_649: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_231, relu_14)
    mul_650: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_231, sigmoid_3);  getitem_231 = None
    sum_143: "f32[4, 448, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_649, [2, 3], True);  mul_649 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sub_125: "f32[4, 448, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_3)
    mul_651: "f32[4, 448, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_3, sub_125);  sigmoid_3 = sub_125 = None
    mul_652: "f32[4, 448, 1, 1]" = torch.ops.aten.mul.Tensor(sum_143, mul_651);  sum_143 = mul_651 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    sum_144: "f32[448]" = torch.ops.aten.sum.dim_IntList(mul_652, [0, 2, 3])
    convolution_backward_78 = torch.ops.aten.convolution_backward.default(mul_652, relu_15, primals_153, [448], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_652 = primals_153 = None
    getitem_234: "f32[4, 112, 1, 1]" = convolution_backward_78[0]
    getitem_235: "f32[448, 112, 1, 1]" = convolution_backward_78[1];  convolution_backward_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    le_61: "b8[4, 112, 1, 1]" = torch.ops.aten.le.Scalar(relu_15, 0);  relu_15 = None
    where_61: "f32[4, 112, 1, 1]" = torch.ops.aten.where.self(le_61, full_default, getitem_234);  le_61 = getitem_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    sum_145: "f32[112]" = torch.ops.aten.sum.dim_IntList(where_61, [0, 2, 3])
    convolution_backward_79 = torch.ops.aten.convolution_backward.default(where_61, mean_3, primals_151, [112], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_61 = mean_3 = primals_151 = None
    getitem_237: "f32[4, 448, 1, 1]" = convolution_backward_79[0]
    getitem_238: "f32[112, 448, 1, 1]" = convolution_backward_79[1];  convolution_backward_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_16: "f32[4, 448, 28, 28]" = torch.ops.aten.expand.default(getitem_237, [4, 448, 28, 28]);  getitem_237 = None
    div_16: "f32[4, 448, 28, 28]" = torch.ops.aten.div.Scalar(expand_16, 784);  expand_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_221: "f32[4, 448, 28, 28]" = torch.ops.aten.add.Tensor(mul_650, div_16);  mul_650 = div_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_62: "b8[4, 448, 28, 28]" = torch.ops.aten.le.Scalar(relu_14, 0);  relu_14 = None
    where_62: "f32[4, 448, 28, 28]" = torch.ops.aten.where.self(le_62, full_default, add_221);  le_62 = add_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_222: "f32[448]" = torch.ops.aten.add.Tensor(primals_292, 1e-05);  primals_292 = None
    rsqrt_48: "f32[448]" = torch.ops.aten.rsqrt.default(add_222);  add_222 = None
    unsqueeze_1072: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(primals_291, 0);  primals_291 = None
    unsqueeze_1073: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1072, 2);  unsqueeze_1072 = None
    unsqueeze_1074: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1073, 3);  unsqueeze_1073 = None
    sum_146: "f32[448]" = torch.ops.aten.sum.dim_IntList(where_62, [0, 2, 3])
    sub_126: "f32[4, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_1074);  convolution_19 = unsqueeze_1074 = None
    mul_653: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(where_62, sub_126);  sub_126 = None
    sum_147: "f32[448]" = torch.ops.aten.sum.dim_IntList(mul_653, [0, 2, 3]);  mul_653 = None
    mul_658: "f32[448]" = torch.ops.aten.mul.Tensor(rsqrt_48, primals_27);  primals_27 = None
    unsqueeze_1081: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_658, 0);  mul_658 = None
    unsqueeze_1082: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1081, 2);  unsqueeze_1081 = None
    unsqueeze_1083: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1082, 3);  unsqueeze_1082 = None
    mul_659: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(where_62, unsqueeze_1083);  where_62 = unsqueeze_1083 = None
    mul_660: "f32[448]" = torch.ops.aten.mul.Tensor(sum_147, rsqrt_48);  sum_147 = rsqrt_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_80 = torch.ops.aten.convolution_backward.default(mul_659, relu_13, primals_150, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 4, [True, True, False]);  mul_659 = primals_150 = None
    getitem_240: "f32[4, 448, 28, 28]" = convolution_backward_80[0]
    getitem_241: "f32[448, 112, 3, 3]" = convolution_backward_80[1];  convolution_backward_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_63: "b8[4, 448, 28, 28]" = torch.ops.aten.le.Scalar(relu_13, 0);  relu_13 = None
    where_63: "f32[4, 448, 28, 28]" = torch.ops.aten.where.self(le_63, full_default, getitem_240);  le_63 = getitem_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_223: "f32[448]" = torch.ops.aten.add.Tensor(primals_290, 1e-05);  primals_290 = None
    rsqrt_49: "f32[448]" = torch.ops.aten.rsqrt.default(add_223);  add_223 = None
    unsqueeze_1084: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(primals_289, 0);  primals_289 = None
    unsqueeze_1085: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1084, 2);  unsqueeze_1084 = None
    unsqueeze_1086: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1085, 3);  unsqueeze_1085 = None
    sum_148: "f32[448]" = torch.ops.aten.sum.dim_IntList(where_63, [0, 2, 3])
    sub_127: "f32[4, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_1086);  convolution_18 = unsqueeze_1086 = None
    mul_661: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(where_63, sub_127);  sub_127 = None
    sum_149: "f32[448]" = torch.ops.aten.sum.dim_IntList(mul_661, [0, 2, 3]);  mul_661 = None
    mul_666: "f32[448]" = torch.ops.aten.mul.Tensor(rsqrt_49, primals_25);  primals_25 = None
    unsqueeze_1093: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_666, 0);  mul_666 = None
    unsqueeze_1094: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1093, 2);  unsqueeze_1093 = None
    unsqueeze_1095: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1094, 3);  unsqueeze_1094 = None
    mul_667: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(where_63, unsqueeze_1095);  where_63 = unsqueeze_1095 = None
    mul_668: "f32[448]" = torch.ops.aten.mul.Tensor(sum_149, rsqrt_49);  sum_149 = rsqrt_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_81 = torch.ops.aten.convolution_backward.default(mul_667, relu_12, primals_149, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_667 = primals_149 = None
    getitem_243: "f32[4, 448, 28, 28]" = convolution_backward_81[0]
    getitem_244: "f32[448, 448, 1, 1]" = convolution_backward_81[1];  convolution_backward_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_224: "f32[4, 448, 28, 28]" = torch.ops.aten.add.Tensor(where_60, getitem_243);  where_60 = getitem_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    le_64: "b8[4, 448, 28, 28]" = torch.ops.aten.le.Scalar(relu_12, 0);  relu_12 = None
    where_64: "f32[4, 448, 28, 28]" = torch.ops.aten.where.self(le_64, full_default, add_224);  le_64 = add_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_225: "f32[448]" = torch.ops.aten.add.Tensor(primals_288, 1e-05);  primals_288 = None
    rsqrt_50: "f32[448]" = torch.ops.aten.rsqrt.default(add_225);  add_225 = None
    unsqueeze_1096: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(primals_287, 0);  primals_287 = None
    unsqueeze_1097: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1096, 2);  unsqueeze_1096 = None
    unsqueeze_1098: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1097, 3);  unsqueeze_1097 = None
    sum_150: "f32[448]" = torch.ops.aten.sum.dim_IntList(where_64, [0, 2, 3])
    sub_128: "f32[4, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_1098);  convolution_17 = unsqueeze_1098 = None
    mul_669: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(where_64, sub_128);  sub_128 = None
    sum_151: "f32[448]" = torch.ops.aten.sum.dim_IntList(mul_669, [0, 2, 3]);  mul_669 = None
    mul_674: "f32[448]" = torch.ops.aten.mul.Tensor(rsqrt_50, primals_23);  primals_23 = None
    unsqueeze_1105: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_674, 0);  mul_674 = None
    unsqueeze_1106: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1105, 2);  unsqueeze_1105 = None
    unsqueeze_1107: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1106, 3);  unsqueeze_1106 = None
    mul_675: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(where_64, unsqueeze_1107);  unsqueeze_1107 = None
    mul_676: "f32[448]" = torch.ops.aten.mul.Tensor(sum_151, rsqrt_50);  sum_151 = rsqrt_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_82 = torch.ops.aten.convolution_backward.default(mul_675, relu_8, primals_148, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_675 = primals_148 = None
    getitem_246: "f32[4, 224, 56, 56]" = convolution_backward_82[0]
    getitem_247: "f32[448, 224, 1, 1]" = convolution_backward_82[1];  convolution_backward_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_226: "f32[448]" = torch.ops.aten.add.Tensor(primals_286, 1e-05);  primals_286 = None
    rsqrt_51: "f32[448]" = torch.ops.aten.rsqrt.default(add_226);  add_226 = None
    unsqueeze_1108: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(primals_285, 0);  primals_285 = None
    unsqueeze_1109: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1108, 2);  unsqueeze_1108 = None
    unsqueeze_1110: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1109, 3);  unsqueeze_1109 = None
    sub_129: "f32[4, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_1110);  convolution_16 = unsqueeze_1110 = None
    mul_677: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(where_64, sub_129);  sub_129 = None
    sum_153: "f32[448]" = torch.ops.aten.sum.dim_IntList(mul_677, [0, 2, 3]);  mul_677 = None
    mul_682: "f32[448]" = torch.ops.aten.mul.Tensor(rsqrt_51, primals_21);  primals_21 = None
    unsqueeze_1117: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_682, 0);  mul_682 = None
    unsqueeze_1118: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1117, 2);  unsqueeze_1117 = None
    unsqueeze_1119: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1118, 3);  unsqueeze_1118 = None
    mul_683: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(where_64, unsqueeze_1119);  where_64 = unsqueeze_1119 = None
    mul_684: "f32[448]" = torch.ops.aten.mul.Tensor(sum_153, rsqrt_51);  sum_153 = rsqrt_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_83 = torch.ops.aten.convolution_backward.default(mul_683, mul_32, primals_147, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_683 = mul_32 = primals_147 = None
    getitem_249: "f32[4, 448, 28, 28]" = convolution_backward_83[0]
    getitem_250: "f32[448, 448, 1, 1]" = convolution_backward_83[1];  convolution_backward_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_685: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_249, relu_10)
    mul_686: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_249, sigmoid_2);  getitem_249 = None
    sum_154: "f32[4, 448, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_685, [2, 3], True);  mul_685 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sub_130: "f32[4, 448, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_2)
    mul_687: "f32[4, 448, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_2, sub_130);  sigmoid_2 = sub_130 = None
    mul_688: "f32[4, 448, 1, 1]" = torch.ops.aten.mul.Tensor(sum_154, mul_687);  sum_154 = mul_687 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    sum_155: "f32[448]" = torch.ops.aten.sum.dim_IntList(mul_688, [0, 2, 3])
    convolution_backward_84 = torch.ops.aten.convolution_backward.default(mul_688, relu_11, primals_145, [448], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_688 = primals_145 = None
    getitem_252: "f32[4, 56, 1, 1]" = convolution_backward_84[0]
    getitem_253: "f32[448, 56, 1, 1]" = convolution_backward_84[1];  convolution_backward_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    le_65: "b8[4, 56, 1, 1]" = torch.ops.aten.le.Scalar(relu_11, 0);  relu_11 = None
    where_65: "f32[4, 56, 1, 1]" = torch.ops.aten.where.self(le_65, full_default, getitem_252);  le_65 = getitem_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    sum_156: "f32[56]" = torch.ops.aten.sum.dim_IntList(where_65, [0, 2, 3])
    convolution_backward_85 = torch.ops.aten.convolution_backward.default(where_65, mean_2, primals_143, [56], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_65 = mean_2 = primals_143 = None
    getitem_255: "f32[4, 448, 1, 1]" = convolution_backward_85[0]
    getitem_256: "f32[56, 448, 1, 1]" = convolution_backward_85[1];  convolution_backward_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_17: "f32[4, 448, 28, 28]" = torch.ops.aten.expand.default(getitem_255, [4, 448, 28, 28]);  getitem_255 = None
    div_17: "f32[4, 448, 28, 28]" = torch.ops.aten.div.Scalar(expand_17, 784);  expand_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_227: "f32[4, 448, 28, 28]" = torch.ops.aten.add.Tensor(mul_686, div_17);  mul_686 = div_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_66: "b8[4, 448, 28, 28]" = torch.ops.aten.le.Scalar(relu_10, 0);  relu_10 = None
    where_66: "f32[4, 448, 28, 28]" = torch.ops.aten.where.self(le_66, full_default, add_227);  le_66 = add_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_228: "f32[448]" = torch.ops.aten.add.Tensor(primals_284, 1e-05);  primals_284 = None
    rsqrt_52: "f32[448]" = torch.ops.aten.rsqrt.default(add_228);  add_228 = None
    unsqueeze_1120: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(primals_283, 0);  primals_283 = None
    unsqueeze_1121: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1120, 2);  unsqueeze_1120 = None
    unsqueeze_1122: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1121, 3);  unsqueeze_1121 = None
    sum_157: "f32[448]" = torch.ops.aten.sum.dim_IntList(where_66, [0, 2, 3])
    sub_131: "f32[4, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_1122);  convolution_13 = unsqueeze_1122 = None
    mul_689: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(where_66, sub_131);  sub_131 = None
    sum_158: "f32[448]" = torch.ops.aten.sum.dim_IntList(mul_689, [0, 2, 3]);  mul_689 = None
    mul_694: "f32[448]" = torch.ops.aten.mul.Tensor(rsqrt_52, primals_19);  primals_19 = None
    unsqueeze_1129: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_694, 0);  mul_694 = None
    unsqueeze_1130: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1129, 2);  unsqueeze_1129 = None
    unsqueeze_1131: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1130, 3);  unsqueeze_1130 = None
    mul_695: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(where_66, unsqueeze_1131);  where_66 = unsqueeze_1131 = None
    mul_696: "f32[448]" = torch.ops.aten.mul.Tensor(sum_158, rsqrt_52);  sum_158 = rsqrt_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_86 = torch.ops.aten.convolution_backward.default(mul_695, relu_9, primals_142, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 4, [True, True, False]);  mul_695 = primals_142 = None
    getitem_258: "f32[4, 448, 56, 56]" = convolution_backward_86[0]
    getitem_259: "f32[448, 112, 3, 3]" = convolution_backward_86[1];  convolution_backward_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_67: "b8[4, 448, 56, 56]" = torch.ops.aten.le.Scalar(relu_9, 0);  relu_9 = None
    where_67: "f32[4, 448, 56, 56]" = torch.ops.aten.where.self(le_67, full_default, getitem_258);  le_67 = getitem_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_229: "f32[448]" = torch.ops.aten.add.Tensor(primals_282, 1e-05);  primals_282 = None
    rsqrt_53: "f32[448]" = torch.ops.aten.rsqrt.default(add_229);  add_229 = None
    unsqueeze_1132: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(primals_281, 0);  primals_281 = None
    unsqueeze_1133: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1132, 2);  unsqueeze_1132 = None
    unsqueeze_1134: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1133, 3);  unsqueeze_1133 = None
    sum_159: "f32[448]" = torch.ops.aten.sum.dim_IntList(where_67, [0, 2, 3])
    sub_132: "f32[4, 448, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_1134);  convolution_12 = unsqueeze_1134 = None
    mul_697: "f32[4, 448, 56, 56]" = torch.ops.aten.mul.Tensor(where_67, sub_132);  sub_132 = None
    sum_160: "f32[448]" = torch.ops.aten.sum.dim_IntList(mul_697, [0, 2, 3]);  mul_697 = None
    mul_702: "f32[448]" = torch.ops.aten.mul.Tensor(rsqrt_53, primals_17);  primals_17 = None
    unsqueeze_1141: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_702, 0);  mul_702 = None
    unsqueeze_1142: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1141, 2);  unsqueeze_1141 = None
    unsqueeze_1143: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1142, 3);  unsqueeze_1142 = None
    mul_703: "f32[4, 448, 56, 56]" = torch.ops.aten.mul.Tensor(where_67, unsqueeze_1143);  where_67 = unsqueeze_1143 = None
    mul_704: "f32[448]" = torch.ops.aten.mul.Tensor(sum_160, rsqrt_53);  sum_160 = rsqrt_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_87 = torch.ops.aten.convolution_backward.default(mul_703, relu_8, primals_141, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_703 = primals_141 = None
    getitem_261: "f32[4, 224, 56, 56]" = convolution_backward_87[0]
    getitem_262: "f32[448, 224, 1, 1]" = convolution_backward_87[1];  convolution_backward_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_230: "f32[4, 224, 56, 56]" = torch.ops.aten.add.Tensor(getitem_246, getitem_261);  getitem_246 = getitem_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    le_68: "b8[4, 224, 56, 56]" = torch.ops.aten.le.Scalar(relu_8, 0);  relu_8 = None
    where_68: "f32[4, 224, 56, 56]" = torch.ops.aten.where.self(le_68, full_default, add_230);  le_68 = add_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_231: "f32[224]" = torch.ops.aten.add.Tensor(primals_280, 1e-05);  primals_280 = None
    rsqrt_54: "f32[224]" = torch.ops.aten.rsqrt.default(add_231);  add_231 = None
    unsqueeze_1144: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(primals_279, 0);  primals_279 = None
    unsqueeze_1145: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1144, 2);  unsqueeze_1144 = None
    unsqueeze_1146: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1145, 3);  unsqueeze_1145 = None
    sum_161: "f32[224]" = torch.ops.aten.sum.dim_IntList(where_68, [0, 2, 3])
    sub_133: "f32[4, 224, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_1146);  convolution_11 = unsqueeze_1146 = None
    mul_705: "f32[4, 224, 56, 56]" = torch.ops.aten.mul.Tensor(where_68, sub_133);  sub_133 = None
    sum_162: "f32[224]" = torch.ops.aten.sum.dim_IntList(mul_705, [0, 2, 3]);  mul_705 = None
    mul_710: "f32[224]" = torch.ops.aten.mul.Tensor(rsqrt_54, primals_15);  primals_15 = None
    unsqueeze_1153: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_710, 0);  mul_710 = None
    unsqueeze_1154: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1153, 2);  unsqueeze_1153 = None
    unsqueeze_1155: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1154, 3);  unsqueeze_1154 = None
    mul_711: "f32[4, 224, 56, 56]" = torch.ops.aten.mul.Tensor(where_68, unsqueeze_1155);  unsqueeze_1155 = None
    mul_712: "f32[224]" = torch.ops.aten.mul.Tensor(sum_162, rsqrt_54);  sum_162 = rsqrt_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_88 = torch.ops.aten.convolution_backward.default(mul_711, mul_22, primals_140, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_711 = mul_22 = primals_140 = None
    getitem_264: "f32[4, 224, 56, 56]" = convolution_backward_88[0]
    getitem_265: "f32[224, 224, 1, 1]" = convolution_backward_88[1];  convolution_backward_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_713: "f32[4, 224, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_264, relu_6)
    mul_714: "f32[4, 224, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_264, sigmoid_1);  getitem_264 = None
    sum_163: "f32[4, 224, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_713, [2, 3], True);  mul_713 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sub_134: "f32[4, 224, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_1)
    mul_715: "f32[4, 224, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_1, sub_134);  sigmoid_1 = sub_134 = None
    mul_716: "f32[4, 224, 1, 1]" = torch.ops.aten.mul.Tensor(sum_163, mul_715);  sum_163 = mul_715 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    sum_164: "f32[224]" = torch.ops.aten.sum.dim_IntList(mul_716, [0, 2, 3])
    convolution_backward_89 = torch.ops.aten.convolution_backward.default(mul_716, relu_7, primals_138, [224], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_716 = primals_138 = None
    getitem_267: "f32[4, 56, 1, 1]" = convolution_backward_89[0]
    getitem_268: "f32[224, 56, 1, 1]" = convolution_backward_89[1];  convolution_backward_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    le_69: "b8[4, 56, 1, 1]" = torch.ops.aten.le.Scalar(relu_7, 0);  relu_7 = None
    where_69: "f32[4, 56, 1, 1]" = torch.ops.aten.where.self(le_69, full_default, getitem_267);  le_69 = getitem_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    sum_165: "f32[56]" = torch.ops.aten.sum.dim_IntList(where_69, [0, 2, 3])
    convolution_backward_90 = torch.ops.aten.convolution_backward.default(where_69, mean_1, primals_136, [56], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_69 = mean_1 = primals_136 = None
    getitem_270: "f32[4, 224, 1, 1]" = convolution_backward_90[0]
    getitem_271: "f32[56, 224, 1, 1]" = convolution_backward_90[1];  convolution_backward_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_18: "f32[4, 224, 56, 56]" = torch.ops.aten.expand.default(getitem_270, [4, 224, 56, 56]);  getitem_270 = None
    div_18: "f32[4, 224, 56, 56]" = torch.ops.aten.div.Scalar(expand_18, 3136);  expand_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_232: "f32[4, 224, 56, 56]" = torch.ops.aten.add.Tensor(mul_714, div_18);  mul_714 = div_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_70: "b8[4, 224, 56, 56]" = torch.ops.aten.le.Scalar(relu_6, 0);  relu_6 = None
    where_70: "f32[4, 224, 56, 56]" = torch.ops.aten.where.self(le_70, full_default, add_232);  le_70 = add_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_233: "f32[224]" = torch.ops.aten.add.Tensor(primals_278, 1e-05);  primals_278 = None
    rsqrt_55: "f32[224]" = torch.ops.aten.rsqrt.default(add_233);  add_233 = None
    unsqueeze_1156: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(primals_277, 0);  primals_277 = None
    unsqueeze_1157: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1156, 2);  unsqueeze_1156 = None
    unsqueeze_1158: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1157, 3);  unsqueeze_1157 = None
    sum_166: "f32[224]" = torch.ops.aten.sum.dim_IntList(where_70, [0, 2, 3])
    sub_135: "f32[4, 224, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_1158);  convolution_8 = unsqueeze_1158 = None
    mul_717: "f32[4, 224, 56, 56]" = torch.ops.aten.mul.Tensor(where_70, sub_135);  sub_135 = None
    sum_167: "f32[224]" = torch.ops.aten.sum.dim_IntList(mul_717, [0, 2, 3]);  mul_717 = None
    mul_722: "f32[224]" = torch.ops.aten.mul.Tensor(rsqrt_55, primals_13);  primals_13 = None
    unsqueeze_1165: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_722, 0);  mul_722 = None
    unsqueeze_1166: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1165, 2);  unsqueeze_1165 = None
    unsqueeze_1167: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1166, 3);  unsqueeze_1166 = None
    mul_723: "f32[4, 224, 56, 56]" = torch.ops.aten.mul.Tensor(where_70, unsqueeze_1167);  where_70 = unsqueeze_1167 = None
    mul_724: "f32[224]" = torch.ops.aten.mul.Tensor(sum_167, rsqrt_55);  sum_167 = rsqrt_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_91 = torch.ops.aten.convolution_backward.default(mul_723, relu_5, primals_135, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False]);  mul_723 = primals_135 = None
    getitem_273: "f32[4, 224, 56, 56]" = convolution_backward_91[0]
    getitem_274: "f32[224, 112, 3, 3]" = convolution_backward_91[1];  convolution_backward_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_71: "b8[4, 224, 56, 56]" = torch.ops.aten.le.Scalar(relu_5, 0);  relu_5 = None
    where_71: "f32[4, 224, 56, 56]" = torch.ops.aten.where.self(le_71, full_default, getitem_273);  le_71 = getitem_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_234: "f32[224]" = torch.ops.aten.add.Tensor(primals_276, 1e-05);  primals_276 = None
    rsqrt_56: "f32[224]" = torch.ops.aten.rsqrt.default(add_234);  add_234 = None
    unsqueeze_1168: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(primals_275, 0);  primals_275 = None
    unsqueeze_1169: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1168, 2);  unsqueeze_1168 = None
    unsqueeze_1170: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1169, 3);  unsqueeze_1169 = None
    sum_168: "f32[224]" = torch.ops.aten.sum.dim_IntList(where_71, [0, 2, 3])
    sub_136: "f32[4, 224, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_1170);  convolution_7 = unsqueeze_1170 = None
    mul_725: "f32[4, 224, 56, 56]" = torch.ops.aten.mul.Tensor(where_71, sub_136);  sub_136 = None
    sum_169: "f32[224]" = torch.ops.aten.sum.dim_IntList(mul_725, [0, 2, 3]);  mul_725 = None
    mul_730: "f32[224]" = torch.ops.aten.mul.Tensor(rsqrt_56, primals_11);  primals_11 = None
    unsqueeze_1177: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_730, 0);  mul_730 = None
    unsqueeze_1178: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1177, 2);  unsqueeze_1177 = None
    unsqueeze_1179: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1178, 3);  unsqueeze_1178 = None
    mul_731: "f32[4, 224, 56, 56]" = torch.ops.aten.mul.Tensor(where_71, unsqueeze_1179);  where_71 = unsqueeze_1179 = None
    mul_732: "f32[224]" = torch.ops.aten.mul.Tensor(sum_169, rsqrt_56);  sum_169 = rsqrt_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_92 = torch.ops.aten.convolution_backward.default(mul_731, relu_4, primals_134, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_731 = primals_134 = None
    getitem_276: "f32[4, 224, 56, 56]" = convolution_backward_92[0]
    getitem_277: "f32[224, 224, 1, 1]" = convolution_backward_92[1];  convolution_backward_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_235: "f32[4, 224, 56, 56]" = torch.ops.aten.add.Tensor(where_68, getitem_276);  where_68 = getitem_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    le_72: "b8[4, 224, 56, 56]" = torch.ops.aten.le.Scalar(relu_4, 0);  relu_4 = None
    where_72: "f32[4, 224, 56, 56]" = torch.ops.aten.where.self(le_72, full_default, add_235);  le_72 = add_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_236: "f32[224]" = torch.ops.aten.add.Tensor(primals_274, 1e-05);  primals_274 = None
    rsqrt_57: "f32[224]" = torch.ops.aten.rsqrt.default(add_236);  add_236 = None
    unsqueeze_1180: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(primals_273, 0);  primals_273 = None
    unsqueeze_1181: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1180, 2);  unsqueeze_1180 = None
    unsqueeze_1182: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1181, 3);  unsqueeze_1181 = None
    sum_170: "f32[224]" = torch.ops.aten.sum.dim_IntList(where_72, [0, 2, 3])
    sub_137: "f32[4, 224, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_1182);  convolution_6 = unsqueeze_1182 = None
    mul_733: "f32[4, 224, 56, 56]" = torch.ops.aten.mul.Tensor(where_72, sub_137);  sub_137 = None
    sum_171: "f32[224]" = torch.ops.aten.sum.dim_IntList(mul_733, [0, 2, 3]);  mul_733 = None
    mul_738: "f32[224]" = torch.ops.aten.mul.Tensor(rsqrt_57, primals_9);  primals_9 = None
    unsqueeze_1189: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_738, 0);  mul_738 = None
    unsqueeze_1190: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1189, 2);  unsqueeze_1189 = None
    unsqueeze_1191: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1190, 3);  unsqueeze_1190 = None
    mul_739: "f32[4, 224, 56, 56]" = torch.ops.aten.mul.Tensor(where_72, unsqueeze_1191);  unsqueeze_1191 = None
    mul_740: "f32[224]" = torch.ops.aten.mul.Tensor(sum_171, rsqrt_57);  sum_171 = rsqrt_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_93 = torch.ops.aten.convolution_backward.default(mul_739, relu, primals_133, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_739 = primals_133 = None
    getitem_279: "f32[4, 32, 112, 112]" = convolution_backward_93[0]
    getitem_280: "f32[224, 32, 1, 1]" = convolution_backward_93[1];  convolution_backward_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_237: "f32[224]" = torch.ops.aten.add.Tensor(primals_272, 1e-05);  primals_272 = None
    rsqrt_58: "f32[224]" = torch.ops.aten.rsqrt.default(add_237);  add_237 = None
    unsqueeze_1192: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(primals_271, 0);  primals_271 = None
    unsqueeze_1193: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1192, 2);  unsqueeze_1192 = None
    unsqueeze_1194: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1193, 3);  unsqueeze_1193 = None
    sub_138: "f32[4, 224, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_1194);  convolution_5 = unsqueeze_1194 = None
    mul_741: "f32[4, 224, 56, 56]" = torch.ops.aten.mul.Tensor(where_72, sub_138);  sub_138 = None
    sum_173: "f32[224]" = torch.ops.aten.sum.dim_IntList(mul_741, [0, 2, 3]);  mul_741 = None
    mul_746: "f32[224]" = torch.ops.aten.mul.Tensor(rsqrt_58, primals_7);  primals_7 = None
    unsqueeze_1201: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_746, 0);  mul_746 = None
    unsqueeze_1202: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1201, 2);  unsqueeze_1201 = None
    unsqueeze_1203: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1202, 3);  unsqueeze_1202 = None
    mul_747: "f32[4, 224, 56, 56]" = torch.ops.aten.mul.Tensor(where_72, unsqueeze_1203);  where_72 = unsqueeze_1203 = None
    mul_748: "f32[224]" = torch.ops.aten.mul.Tensor(sum_173, rsqrt_58);  sum_173 = rsqrt_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_94 = torch.ops.aten.convolution_backward.default(mul_747, mul_9, primals_132, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_747 = mul_9 = primals_132 = None
    getitem_282: "f32[4, 224, 56, 56]" = convolution_backward_94[0]
    getitem_283: "f32[224, 224, 1, 1]" = convolution_backward_94[1];  convolution_backward_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_749: "f32[4, 224, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_282, relu_2)
    mul_750: "f32[4, 224, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_282, sigmoid);  getitem_282 = None
    sum_174: "f32[4, 224, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_749, [2, 3], True);  mul_749 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sub_139: "f32[4, 224, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid)
    mul_751: "f32[4, 224, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid, sub_139);  sigmoid = sub_139 = None
    mul_752: "f32[4, 224, 1, 1]" = torch.ops.aten.mul.Tensor(sum_174, mul_751);  sum_174 = mul_751 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    sum_175: "f32[224]" = torch.ops.aten.sum.dim_IntList(mul_752, [0, 2, 3])
    convolution_backward_95 = torch.ops.aten.convolution_backward.default(mul_752, relu_3, primals_130, [224], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_752 = primals_130 = None
    getitem_285: "f32[4, 8, 1, 1]" = convolution_backward_95[0]
    getitem_286: "f32[224, 8, 1, 1]" = convolution_backward_95[1];  convolution_backward_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    le_73: "b8[4, 8, 1, 1]" = torch.ops.aten.le.Scalar(relu_3, 0);  relu_3 = None
    where_73: "f32[4, 8, 1, 1]" = torch.ops.aten.where.self(le_73, full_default, getitem_285);  le_73 = getitem_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    sum_176: "f32[8]" = torch.ops.aten.sum.dim_IntList(where_73, [0, 2, 3])
    convolution_backward_96 = torch.ops.aten.convolution_backward.default(where_73, mean, primals_128, [8], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_73 = mean = primals_128 = None
    getitem_288: "f32[4, 224, 1, 1]" = convolution_backward_96[0]
    getitem_289: "f32[8, 224, 1, 1]" = convolution_backward_96[1];  convolution_backward_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_19: "f32[4, 224, 56, 56]" = torch.ops.aten.expand.default(getitem_288, [4, 224, 56, 56]);  getitem_288 = None
    div_19: "f32[4, 224, 56, 56]" = torch.ops.aten.div.Scalar(expand_19, 3136);  expand_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_238: "f32[4, 224, 56, 56]" = torch.ops.aten.add.Tensor(mul_750, div_19);  mul_750 = div_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_74: "b8[4, 224, 56, 56]" = torch.ops.aten.le.Scalar(relu_2, 0);  relu_2 = None
    where_74: "f32[4, 224, 56, 56]" = torch.ops.aten.where.self(le_74, full_default, add_238);  le_74 = add_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_239: "f32[224]" = torch.ops.aten.add.Tensor(primals_270, 1e-05);  primals_270 = None
    rsqrt_59: "f32[224]" = torch.ops.aten.rsqrt.default(add_239);  add_239 = None
    unsqueeze_1204: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(primals_269, 0);  primals_269 = None
    unsqueeze_1205: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1204, 2);  unsqueeze_1204 = None
    unsqueeze_1206: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1205, 3);  unsqueeze_1205 = None
    sum_177: "f32[224]" = torch.ops.aten.sum.dim_IntList(where_74, [0, 2, 3])
    sub_140: "f32[4, 224, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_1206);  convolution_2 = unsqueeze_1206 = None
    mul_753: "f32[4, 224, 56, 56]" = torch.ops.aten.mul.Tensor(where_74, sub_140);  sub_140 = None
    sum_178: "f32[224]" = torch.ops.aten.sum.dim_IntList(mul_753, [0, 2, 3]);  mul_753 = None
    mul_758: "f32[224]" = torch.ops.aten.mul.Tensor(rsqrt_59, primals_5);  primals_5 = None
    unsqueeze_1213: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_758, 0);  mul_758 = None
    unsqueeze_1214: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1213, 2);  unsqueeze_1213 = None
    unsqueeze_1215: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1214, 3);  unsqueeze_1214 = None
    mul_759: "f32[4, 224, 56, 56]" = torch.ops.aten.mul.Tensor(where_74, unsqueeze_1215);  where_74 = unsqueeze_1215 = None
    mul_760: "f32[224]" = torch.ops.aten.mul.Tensor(sum_178, rsqrt_59);  sum_178 = rsqrt_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_97 = torch.ops.aten.convolution_backward.default(mul_759, relu_1, primals_127, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False]);  mul_759 = primals_127 = None
    getitem_291: "f32[4, 224, 112, 112]" = convolution_backward_97[0]
    getitem_292: "f32[224, 112, 3, 3]" = convolution_backward_97[1];  convolution_backward_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_75: "b8[4, 224, 112, 112]" = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None
    where_75: "f32[4, 224, 112, 112]" = torch.ops.aten.where.self(le_75, full_default, getitem_291);  le_75 = getitem_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_240: "f32[224]" = torch.ops.aten.add.Tensor(primals_268, 1e-05);  primals_268 = None
    rsqrt_60: "f32[224]" = torch.ops.aten.rsqrt.default(add_240);  add_240 = None
    unsqueeze_1216: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(primals_267, 0);  primals_267 = None
    unsqueeze_1217: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1216, 2);  unsqueeze_1216 = None
    unsqueeze_1218: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1217, 3);  unsqueeze_1217 = None
    sum_179: "f32[224]" = torch.ops.aten.sum.dim_IntList(where_75, [0, 2, 3])
    sub_141: "f32[4, 224, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_1218);  convolution_1 = unsqueeze_1218 = None
    mul_761: "f32[4, 224, 112, 112]" = torch.ops.aten.mul.Tensor(where_75, sub_141);  sub_141 = None
    sum_180: "f32[224]" = torch.ops.aten.sum.dim_IntList(mul_761, [0, 2, 3]);  mul_761 = None
    mul_766: "f32[224]" = torch.ops.aten.mul.Tensor(rsqrt_60, primals_3);  primals_3 = None
    unsqueeze_1225: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_766, 0);  mul_766 = None
    unsqueeze_1226: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1225, 2);  unsqueeze_1225 = None
    unsqueeze_1227: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1226, 3);  unsqueeze_1226 = None
    mul_767: "f32[4, 224, 112, 112]" = torch.ops.aten.mul.Tensor(where_75, unsqueeze_1227);  where_75 = unsqueeze_1227 = None
    mul_768: "f32[224]" = torch.ops.aten.mul.Tensor(sum_180, rsqrt_60);  sum_180 = rsqrt_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_98 = torch.ops.aten.convolution_backward.default(mul_767, relu, primals_126, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_767 = primals_126 = None
    getitem_294: "f32[4, 32, 112, 112]" = convolution_backward_98[0]
    getitem_295: "f32[224, 32, 1, 1]" = convolution_backward_98[1];  convolution_backward_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_241: "f32[4, 32, 112, 112]" = torch.ops.aten.add.Tensor(getitem_279, getitem_294);  getitem_279 = getitem_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_76: "b8[4, 32, 112, 112]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
    where_76: "f32[4, 32, 112, 112]" = torch.ops.aten.where.self(le_76, full_default, add_241);  le_76 = full_default = add_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_242: "f32[32]" = torch.ops.aten.add.Tensor(primals_266, 1e-05);  primals_266 = None
    rsqrt_61: "f32[32]" = torch.ops.aten.rsqrt.default(add_242);  add_242 = None
    unsqueeze_1228: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(primals_265, 0);  primals_265 = None
    unsqueeze_1229: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1228, 2);  unsqueeze_1228 = None
    unsqueeze_1230: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1229, 3);  unsqueeze_1229 = None
    sum_181: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_76, [0, 2, 3])
    sub_142: "f32[4, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_1230);  convolution = unsqueeze_1230 = None
    mul_769: "f32[4, 32, 112, 112]" = torch.ops.aten.mul.Tensor(where_76, sub_142);  sub_142 = None
    sum_182: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_769, [0, 2, 3]);  mul_769 = None
    mul_774: "f32[32]" = torch.ops.aten.mul.Tensor(rsqrt_61, primals_1);  primals_1 = None
    unsqueeze_1237: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_774, 0);  mul_774 = None
    unsqueeze_1238: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1237, 2);  unsqueeze_1237 = None
    unsqueeze_1239: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1238, 3);  unsqueeze_1238 = None
    mul_775: "f32[4, 32, 112, 112]" = torch.ops.aten.mul.Tensor(where_76, unsqueeze_1239);  where_76 = unsqueeze_1239 = None
    mul_776: "f32[32]" = torch.ops.aten.mul.Tensor(sum_182, rsqrt_61);  sum_182 = rsqrt_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_99 = torch.ops.aten.convolution_backward.default(mul_775, primals_389, primals_125, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_775 = primals_389 = primals_125 = None
    getitem_298: "f32[32, 3, 3, 3]" = convolution_backward_99[1];  convolution_backward_99 = None
    return [mul_776, sum_181, mul_768, sum_179, mul_760, sum_177, mul_748, sum_170, mul_740, sum_170, mul_732, sum_168, mul_724, sum_166, mul_712, sum_161, mul_704, sum_159, mul_696, sum_157, mul_684, sum_150, mul_676, sum_150, mul_668, sum_148, mul_660, sum_146, mul_648, sum_141, mul_640, sum_139, mul_632, sum_137, mul_620, sum_132, mul_612, sum_130, mul_604, sum_128, mul_592, sum_123, mul_584, sum_121, mul_576, sum_119, mul_564, sum_114, mul_556, sum_112, mul_548, sum_110, mul_536, sum_103, mul_528, sum_103, mul_520, sum_101, mul_512, sum_99, mul_500, sum_94, mul_492, sum_92, mul_484, sum_90, mul_472, sum_85, mul_464, sum_83, mul_456, sum_81, mul_444, sum_76, mul_436, sum_74, mul_428, sum_72, mul_416, sum_67, mul_408, sum_65, mul_400, sum_63, mul_388, sum_58, mul_380, sum_56, mul_372, sum_54, mul_360, sum_49, mul_352, sum_47, mul_344, sum_45, mul_332, sum_40, mul_324, sum_38, mul_316, sum_36, mul_304, sum_31, mul_296, sum_29, mul_288, sum_27, mul_276, sum_22, mul_268, sum_20, mul_260, sum_18, mul_248, sum_13, mul_240, sum_11, mul_232, sum_9, mul_220, sum_2, mul_212, sum_2, getitem_298, getitem_295, getitem_292, getitem_289, sum_176, getitem_286, sum_175, getitem_283, getitem_280, getitem_277, getitem_274, getitem_271, sum_165, getitem_268, sum_164, getitem_265, getitem_262, getitem_259, getitem_256, sum_156, getitem_253, sum_155, getitem_250, getitem_247, getitem_244, getitem_241, getitem_238, sum_145, getitem_235, sum_144, getitem_232, getitem_229, getitem_226, getitem_223, sum_136, getitem_220, sum_135, getitem_217, getitem_214, getitem_211, getitem_208, sum_127, getitem_205, sum_126, getitem_202, getitem_199, getitem_196, getitem_193, sum_118, getitem_190, sum_117, getitem_187, getitem_184, getitem_181, getitem_178, sum_109, getitem_175, sum_108, getitem_172, getitem_169, getitem_166, getitem_163, getitem_160, sum_98, getitem_157, sum_97, getitem_154, getitem_151, getitem_148, getitem_145, sum_89, getitem_142, sum_88, getitem_139, getitem_136, getitem_133, getitem_130, sum_80, getitem_127, sum_79, getitem_124, getitem_121, getitem_118, getitem_115, sum_71, getitem_112, sum_70, getitem_109, getitem_106, getitem_103, getitem_100, sum_62, getitem_97, sum_61, getitem_94, getitem_91, getitem_88, getitem_85, sum_53, getitem_82, sum_52, getitem_79, getitem_76, getitem_73, getitem_70, sum_44, getitem_67, sum_43, getitem_64, getitem_61, getitem_58, getitem_55, sum_35, getitem_52, sum_34, getitem_49, getitem_46, getitem_43, getitem_40, sum_26, getitem_37, sum_25, getitem_34, getitem_31, getitem_28, getitem_25, sum_17, getitem_22, sum_16, getitem_19, getitem_16, getitem_13, getitem_10, sum_8, getitem_7, sum_7, getitem_4, getitem_1, permute_4, view_1, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
    