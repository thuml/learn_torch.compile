from __future__ import annotations



def forward(self, primals_1: "f32[16]", primals_3: "f32[16]", primals_5: "f32[16]", primals_7: "f32[16]", primals_9: "f32[16]", primals_11: "f32[64]", primals_13: "f32[64]", primals_15: "f32[24]", primals_17: "f32[48]", primals_19: "f32[48]", primals_21: "f32[24]", primals_23: "f32[48]", primals_25: "f32[48]", primals_27: "f32[24]", primals_29: "f32[48]", primals_31: "f32[48]", primals_33: "f32[24]", primals_35: "f32[120]", primals_37: "f32[120]", primals_39: "f32[40]", primals_41: "f32[120]", primals_43: "f32[120]", primals_45: "f32[40]", primals_47: "f32[120]", primals_49: "f32[120]", primals_51: "f32[40]", primals_53: "f32[120]", primals_55: "f32[120]", primals_57: "f32[40]", primals_59: "f32[120]", primals_61: "f32[120]", primals_63: "f32[40]", primals_65: "f32[200]", primals_67: "f32[200]", primals_69: "f32[72]", primals_71: "f32[216]", primals_73: "f32[216]", primals_75: "f32[72]", primals_77: "f32[216]", primals_79: "f32[216]", primals_81: "f32[72]", primals_83: "f32[216]", primals_85: "f32[216]", primals_87: "f32[72]", primals_89: "f32[216]", primals_91: "f32[216]", primals_93: "f32[72]", primals_95: "f32[360]", primals_97: "f32[360]", primals_99: "f32[120]", primals_101: "f32[360]", primals_103: "f32[360]", primals_105: "f32[120]", primals_107: "f32[360]", primals_109: "f32[360]", primals_111: "f32[120]", primals_113: "f32[360]", primals_115: "f32[360]", primals_117: "f32[120]", primals_119: "f32[360]", primals_121: "f32[360]", primals_123: "f32[120]", primals_125: "f32[360]", primals_127: "f32[360]", primals_129: "f32[120]", primals_131: "f32[720]", primals_133: "f32[720]", primals_135: "f32[184]", primals_137: "f32[736]", primals_139: "f32[736]", primals_141: "f32[184]", primals_143: "f32[736]", primals_145: "f32[736]", primals_147: "f32[184]", primals_149: "f32[736]", primals_151: "f32[736]", primals_153: "f32[184]", primals_155: "f32[736]", primals_157: "f32[736]", primals_159: "f32[184]", primals_161: "f32[736]", primals_163: "f32[736]", primals_165: "f32[184]", primals_167: "f32[1104]", primals_169: "f32[1104]", primals_171: "f32[224]", primals_173: "f32[1344]", primals_177: "f32[16, 3, 3, 3]", primals_178: "f32[16, 1, 3, 3]", primals_179: "f32[16, 16, 1, 1]", primals_180: "f32[16, 1, 3, 3]", primals_181: "f32[16, 16, 1, 1]", primals_182: "f32[64, 16, 1, 1]", primals_183: "f32[64, 1, 5, 5]", primals_184: "f32[24, 64, 1, 1]", primals_185: "f32[48, 24, 1, 1]", primals_186: "f32[48, 1, 5, 5]", primals_187: "f32[24, 48, 1, 1]", primals_188: "f32[48, 24, 1, 1]", primals_189: "f32[48, 1, 5, 5]", primals_190: "f32[24, 48, 1, 1]", primals_191: "f32[48, 24, 1, 1]", primals_192: "f32[48, 1, 5, 5]", primals_193: "f32[24, 48, 1, 1]", primals_194: "f32[120, 24, 1, 1]", primals_195: "f32[120, 1, 5, 5]", primals_196: "f32[8, 120, 1, 1]", primals_198: "f32[120, 8, 1, 1]", primals_200: "f32[40, 120, 1, 1]", primals_201: "f32[120, 40, 1, 1]", primals_202: "f32[120, 1, 5, 5]", primals_203: "f32[16, 120, 1, 1]", primals_205: "f32[120, 16, 1, 1]", primals_207: "f32[40, 120, 1, 1]", primals_208: "f32[120, 40, 1, 1]", primals_209: "f32[120, 1, 5, 5]", primals_210: "f32[16, 120, 1, 1]", primals_212: "f32[120, 16, 1, 1]", primals_214: "f32[40, 120, 1, 1]", primals_215: "f32[120, 40, 1, 1]", primals_216: "f32[120, 1, 5, 5]", primals_217: "f32[16, 120, 1, 1]", primals_219: "f32[120, 16, 1, 1]", primals_221: "f32[40, 120, 1, 1]", primals_222: "f32[120, 40, 1, 1]", primals_223: "f32[120, 1, 5, 5]", primals_224: "f32[16, 120, 1, 1]", primals_226: "f32[120, 16, 1, 1]", primals_228: "f32[40, 120, 1, 1]", primals_229: "f32[200, 40, 1, 1]", primals_230: "f32[200, 1, 5, 5]", primals_231: "f32[72, 200, 1, 1]", primals_232: "f32[216, 72, 1, 1]", primals_233: "f32[216, 1, 3, 3]", primals_234: "f32[72, 216, 1, 1]", primals_235: "f32[216, 72, 1, 1]", primals_236: "f32[216, 1, 3, 3]", primals_237: "f32[72, 216, 1, 1]", primals_238: "f32[216, 72, 1, 1]", primals_239: "f32[216, 1, 3, 3]", primals_240: "f32[72, 216, 1, 1]", primals_241: "f32[216, 72, 1, 1]", primals_242: "f32[216, 1, 3, 3]", primals_243: "f32[72, 216, 1, 1]", primals_244: "f32[360, 72, 1, 1]", primals_245: "f32[360, 1, 3, 3]", primals_246: "f32[24, 360, 1, 1]", primals_248: "f32[360, 24, 1, 1]", primals_250: "f32[120, 360, 1, 1]", primals_251: "f32[360, 120, 1, 1]", primals_252: "f32[360, 1, 5, 5]", primals_253: "f32[32, 360, 1, 1]", primals_255: "f32[360, 32, 1, 1]", primals_257: "f32[120, 360, 1, 1]", primals_258: "f32[360, 120, 1, 1]", primals_259: "f32[360, 1, 5, 5]", primals_260: "f32[32, 360, 1, 1]", primals_262: "f32[360, 32, 1, 1]", primals_264: "f32[120, 360, 1, 1]", primals_265: "f32[360, 120, 1, 1]", primals_266: "f32[360, 1, 5, 5]", primals_267: "f32[32, 360, 1, 1]", primals_269: "f32[360, 32, 1, 1]", primals_271: "f32[120, 360, 1, 1]", primals_272: "f32[360, 120, 1, 1]", primals_273: "f32[360, 1, 5, 5]", primals_274: "f32[32, 360, 1, 1]", primals_276: "f32[360, 32, 1, 1]", primals_278: "f32[120, 360, 1, 1]", primals_279: "f32[360, 120, 1, 1]", primals_280: "f32[360, 1, 5, 5]", primals_281: "f32[32, 360, 1, 1]", primals_283: "f32[360, 32, 1, 1]", primals_285: "f32[120, 360, 1, 1]", primals_286: "f32[720, 120, 1, 1]", primals_287: "f32[720, 1, 3, 3]", primals_288: "f32[32, 720, 1, 1]", primals_290: "f32[720, 32, 1, 1]", primals_292: "f32[184, 720, 1, 1]", primals_293: "f32[736, 184, 1, 1]", primals_294: "f32[736, 1, 5, 5]", primals_295: "f32[48, 736, 1, 1]", primals_297: "f32[736, 48, 1, 1]", primals_299: "f32[184, 736, 1, 1]", primals_300: "f32[736, 184, 1, 1]", primals_301: "f32[736, 1, 5, 5]", primals_302: "f32[48, 736, 1, 1]", primals_304: "f32[736, 48, 1, 1]", primals_306: "f32[184, 736, 1, 1]", primals_307: "f32[736, 184, 1, 1]", primals_308: "f32[736, 1, 5, 5]", primals_309: "f32[48, 736, 1, 1]", primals_311: "f32[736, 48, 1, 1]", primals_313: "f32[184, 736, 1, 1]", primals_314: "f32[736, 184, 1, 1]", primals_315: "f32[736, 1, 5, 5]", primals_316: "f32[48, 736, 1, 1]", primals_318: "f32[736, 48, 1, 1]", primals_320: "f32[184, 736, 1, 1]", primals_321: "f32[736, 184, 1, 1]", primals_322: "f32[736, 1, 5, 5]", primals_323: "f32[48, 736, 1, 1]", primals_325: "f32[736, 48, 1, 1]", primals_327: "f32[184, 736, 1, 1]", primals_328: "f32[1104, 184, 1, 1]", primals_329: "f32[1104, 1, 5, 5]", primals_330: "f32[48, 1104, 1, 1]", primals_332: "f32[1104, 48, 1, 1]", primals_334: "f32[224, 1104, 1, 1]", primals_335: "f32[1344, 224, 1, 1]", primals_336: "f32[1984, 1344, 1, 1]", primals_598: "f32[8, 3, 224, 224]", convolution: "f32[8, 16, 112, 112]", squeeze_1: "f32[16]", clone: "f32[8, 16, 112, 112]", div: "f32[8, 16, 112, 112]", convolution_1: "f32[8, 16, 112, 112]", squeeze_4: "f32[16]", clone_1: "f32[8, 16, 112, 112]", div_1: "f32[8, 16, 112, 112]", convolution_2: "f32[8, 16, 112, 112]", squeeze_7: "f32[16]", add_17: "f32[8, 16, 112, 112]", convolution_3: "f32[8, 16, 112, 112]", squeeze_10: "f32[16]", clone_2: "f32[8, 16, 112, 112]", div_2: "f32[8, 16, 112, 112]", convolution_4: "f32[8, 16, 112, 112]", squeeze_13: "f32[16]", add_29: "f32[8, 16, 112, 112]", convolution_5: "f32[8, 64, 112, 112]", squeeze_16: "f32[64]", clone_3: "f32[8, 64, 112, 112]", div_3: "f32[8, 64, 112, 112]", convolution_6: "f32[8, 64, 56, 56]", squeeze_19: "f32[64]", clone_4: "f32[8, 64, 56, 56]", div_4: "f32[8, 64, 56, 56]", convolution_7: "f32[8, 24, 56, 56]", squeeze_22: "f32[24]", add_46: "f32[8, 24, 56, 56]", convolution_8: "f32[8, 48, 56, 56]", squeeze_25: "f32[48]", clone_5: "f32[8, 48, 56, 56]", div_5: "f32[8, 48, 56, 56]", convolution_9: "f32[8, 48, 56, 56]", squeeze_28: "f32[48]", clone_6: "f32[8, 48, 56, 56]", div_6: "f32[8, 48, 56, 56]", convolution_10: "f32[8, 24, 56, 56]", squeeze_31: "f32[24]", add_64: "f32[8, 24, 56, 56]", convolution_11: "f32[8, 48, 56, 56]", squeeze_34: "f32[48]", clone_7: "f32[8, 48, 56, 56]", div_7: "f32[8, 48, 56, 56]", convolution_12: "f32[8, 48, 56, 56]", squeeze_37: "f32[48]", clone_8: "f32[8, 48, 56, 56]", div_8: "f32[8, 48, 56, 56]", convolution_13: "f32[8, 24, 56, 56]", squeeze_40: "f32[24]", add_82: "f32[8, 24, 56, 56]", convolution_14: "f32[8, 48, 56, 56]", squeeze_43: "f32[48]", clone_9: "f32[8, 48, 56, 56]", div_9: "f32[8, 48, 56, 56]", convolution_15: "f32[8, 48, 56, 56]", squeeze_46: "f32[48]", clone_10: "f32[8, 48, 56, 56]", div_10: "f32[8, 48, 56, 56]", convolution_16: "f32[8, 24, 56, 56]", squeeze_49: "f32[24]", add_100: "f32[8, 24, 56, 56]", convolution_17: "f32[8, 120, 56, 56]", squeeze_52: "f32[120]", clone_11: "f32[8, 120, 56, 56]", div_11: "f32[8, 120, 56, 56]", convolution_18: "f32[8, 120, 28, 28]", squeeze_55: "f32[120]", clone_12: "f32[8, 120, 28, 28]", div_12: "f32[8, 120, 28, 28]", mean: "f32[8, 120, 1, 1]", convolution_19: "f32[8, 8, 1, 1]", div_13: "f32[8, 8, 1, 1]", div_14: "f32[8, 120, 1, 1]", mul_147: "f32[8, 120, 28, 28]", convolution_21: "f32[8, 40, 28, 28]", squeeze_58: "f32[40]", add_119: "f32[8, 40, 28, 28]", convolution_22: "f32[8, 120, 28, 28]", squeeze_61: "f32[120]", clone_14: "f32[8, 120, 28, 28]", div_15: "f32[8, 120, 28, 28]", convolution_23: "f32[8, 120, 28, 28]", squeeze_64: "f32[120]", clone_15: "f32[8, 120, 28, 28]", div_16: "f32[8, 120, 28, 28]", mean_1: "f32[8, 120, 1, 1]", convolution_24: "f32[8, 16, 1, 1]", div_17: "f32[8, 16, 1, 1]", div_18: "f32[8, 120, 1, 1]", mul_172: "f32[8, 120, 28, 28]", convolution_26: "f32[8, 40, 28, 28]", squeeze_67: "f32[40]", add_139: "f32[8, 40, 28, 28]", convolution_27: "f32[8, 120, 28, 28]", squeeze_70: "f32[120]", clone_17: "f32[8, 120, 28, 28]", div_19: "f32[8, 120, 28, 28]", convolution_28: "f32[8, 120, 28, 28]", squeeze_73: "f32[120]", clone_18: "f32[8, 120, 28, 28]", div_20: "f32[8, 120, 28, 28]", mean_2: "f32[8, 120, 1, 1]", convolution_29: "f32[8, 16, 1, 1]", div_21: "f32[8, 16, 1, 1]", div_22: "f32[8, 120, 1, 1]", mul_197: "f32[8, 120, 28, 28]", convolution_31: "f32[8, 40, 28, 28]", squeeze_76: "f32[40]", add_159: "f32[8, 40, 28, 28]", convolution_32: "f32[8, 120, 28, 28]", squeeze_79: "f32[120]", clone_20: "f32[8, 120, 28, 28]", div_23: "f32[8, 120, 28, 28]", convolution_33: "f32[8, 120, 28, 28]", squeeze_82: "f32[120]", clone_21: "f32[8, 120, 28, 28]", div_24: "f32[8, 120, 28, 28]", mean_3: "f32[8, 120, 1, 1]", convolution_34: "f32[8, 16, 1, 1]", div_25: "f32[8, 16, 1, 1]", div_26: "f32[8, 120, 1, 1]", mul_222: "f32[8, 120, 28, 28]", convolution_36: "f32[8, 40, 28, 28]", squeeze_85: "f32[40]", add_179: "f32[8, 40, 28, 28]", convolution_37: "f32[8, 120, 28, 28]", squeeze_88: "f32[120]", clone_23: "f32[8, 120, 28, 28]", div_27: "f32[8, 120, 28, 28]", convolution_38: "f32[8, 120, 28, 28]", squeeze_91: "f32[120]", clone_24: "f32[8, 120, 28, 28]", div_28: "f32[8, 120, 28, 28]", mean_4: "f32[8, 120, 1, 1]", convolution_39: "f32[8, 16, 1, 1]", div_29: "f32[8, 16, 1, 1]", div_30: "f32[8, 120, 1, 1]", mul_247: "f32[8, 120, 28, 28]", convolution_41: "f32[8, 40, 28, 28]", squeeze_94: "f32[40]", add_199: "f32[8, 40, 28, 28]", convolution_42: "f32[8, 200, 28, 28]", squeeze_97: "f32[200]", clone_26: "f32[8, 200, 28, 28]", div_31: "f32[8, 200, 28, 28]", convolution_43: "f32[8, 200, 14, 14]", squeeze_100: "f32[200]", clone_27: "f32[8, 200, 14, 14]", div_32: "f32[8, 200, 14, 14]", convolution_44: "f32[8, 72, 14, 14]", squeeze_103: "f32[72]", add_216: "f32[8, 72, 14, 14]", convolution_45: "f32[8, 216, 14, 14]", squeeze_106: "f32[216]", clone_28: "f32[8, 216, 14, 14]", div_33: "f32[8, 216, 14, 14]", convolution_46: "f32[8, 216, 14, 14]", squeeze_109: "f32[216]", clone_29: "f32[8, 216, 14, 14]", div_34: "f32[8, 216, 14, 14]", convolution_47: "f32[8, 72, 14, 14]", squeeze_112: "f32[72]", add_234: "f32[8, 72, 14, 14]", convolution_48: "f32[8, 216, 14, 14]", squeeze_115: "f32[216]", clone_30: "f32[8, 216, 14, 14]", div_35: "f32[8, 216, 14, 14]", convolution_49: "f32[8, 216, 14, 14]", squeeze_118: "f32[216]", clone_31: "f32[8, 216, 14, 14]", div_36: "f32[8, 216, 14, 14]", convolution_50: "f32[8, 72, 14, 14]", squeeze_121: "f32[72]", add_252: "f32[8, 72, 14, 14]", convolution_51: "f32[8, 216, 14, 14]", squeeze_124: "f32[216]", clone_32: "f32[8, 216, 14, 14]", div_37: "f32[8, 216, 14, 14]", convolution_52: "f32[8, 216, 14, 14]", squeeze_127: "f32[216]", clone_33: "f32[8, 216, 14, 14]", div_38: "f32[8, 216, 14, 14]", convolution_53: "f32[8, 72, 14, 14]", squeeze_130: "f32[72]", add_270: "f32[8, 72, 14, 14]", convolution_54: "f32[8, 216, 14, 14]", squeeze_133: "f32[216]", clone_34: "f32[8, 216, 14, 14]", div_39: "f32[8, 216, 14, 14]", convolution_55: "f32[8, 216, 14, 14]", squeeze_136: "f32[216]", clone_35: "f32[8, 216, 14, 14]", div_40: "f32[8, 216, 14, 14]", convolution_56: "f32[8, 72, 14, 14]", squeeze_139: "f32[72]", add_288: "f32[8, 72, 14, 14]", convolution_57: "f32[8, 360, 14, 14]", squeeze_142: "f32[360]", clone_36: "f32[8, 360, 14, 14]", div_41: "f32[8, 360, 14, 14]", convolution_58: "f32[8, 360, 14, 14]", squeeze_145: "f32[360]", clone_37: "f32[8, 360, 14, 14]", div_42: "f32[8, 360, 14, 14]", mean_5: "f32[8, 360, 1, 1]", convolution_59: "f32[8, 24, 1, 1]", div_43: "f32[8, 24, 1, 1]", div_44: "f32[8, 360, 1, 1]", mul_387: "f32[8, 360, 14, 14]", convolution_61: "f32[8, 120, 14, 14]", squeeze_148: "f32[120]", add_307: "f32[8, 120, 14, 14]", convolution_62: "f32[8, 360, 14, 14]", squeeze_151: "f32[360]", clone_39: "f32[8, 360, 14, 14]", div_45: "f32[8, 360, 14, 14]", convolution_63: "f32[8, 360, 14, 14]", squeeze_154: "f32[360]", clone_40: "f32[8, 360, 14, 14]", div_46: "f32[8, 360, 14, 14]", mean_6: "f32[8, 360, 1, 1]", convolution_64: "f32[8, 32, 1, 1]", div_47: "f32[8, 32, 1, 1]", div_48: "f32[8, 360, 1, 1]", mul_412: "f32[8, 360, 14, 14]", convolution_66: "f32[8, 120, 14, 14]", squeeze_157: "f32[120]", add_327: "f32[8, 120, 14, 14]", convolution_67: "f32[8, 360, 14, 14]", squeeze_160: "f32[360]", clone_42: "f32[8, 360, 14, 14]", div_49: "f32[8, 360, 14, 14]", convolution_68: "f32[8, 360, 14, 14]", squeeze_163: "f32[360]", clone_43: "f32[8, 360, 14, 14]", div_50: "f32[8, 360, 14, 14]", mean_7: "f32[8, 360, 1, 1]", convolution_69: "f32[8, 32, 1, 1]", div_51: "f32[8, 32, 1, 1]", div_52: "f32[8, 360, 1, 1]", mul_437: "f32[8, 360, 14, 14]", convolution_71: "f32[8, 120, 14, 14]", squeeze_166: "f32[120]", add_347: "f32[8, 120, 14, 14]", convolution_72: "f32[8, 360, 14, 14]", squeeze_169: "f32[360]", clone_45: "f32[8, 360, 14, 14]", div_53: "f32[8, 360, 14, 14]", convolution_73: "f32[8, 360, 14, 14]", squeeze_172: "f32[360]", clone_46: "f32[8, 360, 14, 14]", div_54: "f32[8, 360, 14, 14]", mean_8: "f32[8, 360, 1, 1]", convolution_74: "f32[8, 32, 1, 1]", div_55: "f32[8, 32, 1, 1]", div_56: "f32[8, 360, 1, 1]", mul_462: "f32[8, 360, 14, 14]", convolution_76: "f32[8, 120, 14, 14]", squeeze_175: "f32[120]", add_367: "f32[8, 120, 14, 14]", convolution_77: "f32[8, 360, 14, 14]", squeeze_178: "f32[360]", clone_48: "f32[8, 360, 14, 14]", div_57: "f32[8, 360, 14, 14]", convolution_78: "f32[8, 360, 14, 14]", squeeze_181: "f32[360]", clone_49: "f32[8, 360, 14, 14]", div_58: "f32[8, 360, 14, 14]", mean_9: "f32[8, 360, 1, 1]", convolution_79: "f32[8, 32, 1, 1]", div_59: "f32[8, 32, 1, 1]", div_60: "f32[8, 360, 1, 1]", mul_487: "f32[8, 360, 14, 14]", convolution_81: "f32[8, 120, 14, 14]", squeeze_184: "f32[120]", add_387: "f32[8, 120, 14, 14]", convolution_82: "f32[8, 360, 14, 14]", squeeze_187: "f32[360]", clone_51: "f32[8, 360, 14, 14]", div_61: "f32[8, 360, 14, 14]", convolution_83: "f32[8, 360, 14, 14]", squeeze_190: "f32[360]", clone_52: "f32[8, 360, 14, 14]", div_62: "f32[8, 360, 14, 14]", mean_10: "f32[8, 360, 1, 1]", convolution_84: "f32[8, 32, 1, 1]", div_63: "f32[8, 32, 1, 1]", div_64: "f32[8, 360, 1, 1]", mul_512: "f32[8, 360, 14, 14]", convolution_86: "f32[8, 120, 14, 14]", squeeze_193: "f32[120]", add_407: "f32[8, 120, 14, 14]", convolution_87: "f32[8, 720, 14, 14]", squeeze_196: "f32[720]", clone_54: "f32[8, 720, 14, 14]", div_65: "f32[8, 720, 14, 14]", convolution_88: "f32[8, 720, 7, 7]", squeeze_199: "f32[720]", clone_55: "f32[8, 720, 7, 7]", div_66: "f32[8, 720, 7, 7]", mean_11: "f32[8, 720, 1, 1]", convolution_89: "f32[8, 32, 1, 1]", div_67: "f32[8, 32, 1, 1]", div_68: "f32[8, 720, 1, 1]", mul_537: "f32[8, 720, 7, 7]", convolution_91: "f32[8, 184, 7, 7]", squeeze_202: "f32[184]", add_426: "f32[8, 184, 7, 7]", convolution_92: "f32[8, 736, 7, 7]", squeeze_205: "f32[736]", clone_57: "f32[8, 736, 7, 7]", div_69: "f32[8, 736, 7, 7]", convolution_93: "f32[8, 736, 7, 7]", squeeze_208: "f32[736]", clone_58: "f32[8, 736, 7, 7]", div_70: "f32[8, 736, 7, 7]", mean_12: "f32[8, 736, 1, 1]", convolution_94: "f32[8, 48, 1, 1]", div_71: "f32[8, 48, 1, 1]", div_72: "f32[8, 736, 1, 1]", mul_562: "f32[8, 736, 7, 7]", convolution_96: "f32[8, 184, 7, 7]", squeeze_211: "f32[184]", add_446: "f32[8, 184, 7, 7]", convolution_97: "f32[8, 736, 7, 7]", squeeze_214: "f32[736]", clone_60: "f32[8, 736, 7, 7]", div_73: "f32[8, 736, 7, 7]", convolution_98: "f32[8, 736, 7, 7]", squeeze_217: "f32[736]", clone_61: "f32[8, 736, 7, 7]", div_74: "f32[8, 736, 7, 7]", mean_13: "f32[8, 736, 1, 1]", convolution_99: "f32[8, 48, 1, 1]", div_75: "f32[8, 48, 1, 1]", div_76: "f32[8, 736, 1, 1]", mul_587: "f32[8, 736, 7, 7]", convolution_101: "f32[8, 184, 7, 7]", squeeze_220: "f32[184]", add_466: "f32[8, 184, 7, 7]", convolution_102: "f32[8, 736, 7, 7]", squeeze_223: "f32[736]", clone_63: "f32[8, 736, 7, 7]", div_77: "f32[8, 736, 7, 7]", convolution_103: "f32[8, 736, 7, 7]", squeeze_226: "f32[736]", clone_64: "f32[8, 736, 7, 7]", div_78: "f32[8, 736, 7, 7]", mean_14: "f32[8, 736, 1, 1]", convolution_104: "f32[8, 48, 1, 1]", div_79: "f32[8, 48, 1, 1]", div_80: "f32[8, 736, 1, 1]", mul_612: "f32[8, 736, 7, 7]", convolution_106: "f32[8, 184, 7, 7]", squeeze_229: "f32[184]", add_486: "f32[8, 184, 7, 7]", convolution_107: "f32[8, 736, 7, 7]", squeeze_232: "f32[736]", clone_66: "f32[8, 736, 7, 7]", div_81: "f32[8, 736, 7, 7]", convolution_108: "f32[8, 736, 7, 7]", squeeze_235: "f32[736]", clone_67: "f32[8, 736, 7, 7]", div_82: "f32[8, 736, 7, 7]", mean_15: "f32[8, 736, 1, 1]", convolution_109: "f32[8, 48, 1, 1]", div_83: "f32[8, 48, 1, 1]", div_84: "f32[8, 736, 1, 1]", mul_637: "f32[8, 736, 7, 7]", convolution_111: "f32[8, 184, 7, 7]", squeeze_238: "f32[184]", add_506: "f32[8, 184, 7, 7]", convolution_112: "f32[8, 736, 7, 7]", squeeze_241: "f32[736]", clone_69: "f32[8, 736, 7, 7]", div_85: "f32[8, 736, 7, 7]", convolution_113: "f32[8, 736, 7, 7]", squeeze_244: "f32[736]", clone_70: "f32[8, 736, 7, 7]", div_86: "f32[8, 736, 7, 7]", mean_16: "f32[8, 736, 1, 1]", convolution_114: "f32[8, 48, 1, 1]", div_87: "f32[8, 48, 1, 1]", div_88: "f32[8, 736, 1, 1]", mul_662: "f32[8, 736, 7, 7]", convolution_116: "f32[8, 184, 7, 7]", squeeze_247: "f32[184]", add_526: "f32[8, 184, 7, 7]", convolution_117: "f32[8, 1104, 7, 7]", squeeze_250: "f32[1104]", clone_72: "f32[8, 1104, 7, 7]", div_89: "f32[8, 1104, 7, 7]", convolution_118: "f32[8, 1104, 7, 7]", squeeze_253: "f32[1104]", clone_73: "f32[8, 1104, 7, 7]", div_90: "f32[8, 1104, 7, 7]", mean_17: "f32[8, 1104, 1, 1]", convolution_119: "f32[8, 48, 1, 1]", div_91: "f32[8, 48, 1, 1]", div_92: "f32[8, 1104, 1, 1]", mul_687: "f32[8, 1104, 7, 7]", convolution_121: "f32[8, 224, 7, 7]", squeeze_256: "f32[224]", add_545: "f32[8, 224, 7, 7]", convolution_122: "f32[8, 1344, 7, 7]", squeeze_259: "f32[1344]", clone_75: "f32[8, 1344, 7, 7]", mean_18: "f32[8, 1344, 1, 1]", convolution_123: "f32[8, 1984, 1, 1]", view_1: "f32[8, 1984]", permute_1: "f32[1000, 1984]", unsqueeze_350: "f32[1, 1344, 1, 1]", unsqueeze_362: "f32[1, 224, 1, 1]", bitwise_and: "b8[8, 1104, 1, 1]", unsqueeze_374: "f32[1, 1104, 1, 1]", unsqueeze_386: "f32[1, 1104, 1, 1]", unsqueeze_398: "f32[1, 184, 1, 1]", bitwise_and_1: "b8[8, 736, 1, 1]", unsqueeze_410: "f32[1, 736, 1, 1]", unsqueeze_422: "f32[1, 736, 1, 1]", unsqueeze_434: "f32[1, 184, 1, 1]", bitwise_and_2: "b8[8, 736, 1, 1]", unsqueeze_446: "f32[1, 736, 1, 1]", unsqueeze_458: "f32[1, 736, 1, 1]", unsqueeze_470: "f32[1, 184, 1, 1]", bitwise_and_3: "b8[8, 736, 1, 1]", unsqueeze_482: "f32[1, 736, 1, 1]", unsqueeze_494: "f32[1, 736, 1, 1]", unsqueeze_506: "f32[1, 184, 1, 1]", bitwise_and_4: "b8[8, 736, 1, 1]", unsqueeze_518: "f32[1, 736, 1, 1]", unsqueeze_530: "f32[1, 736, 1, 1]", unsqueeze_542: "f32[1, 184, 1, 1]", bitwise_and_5: "b8[8, 736, 1, 1]", unsqueeze_554: "f32[1, 736, 1, 1]", unsqueeze_566: "f32[1, 736, 1, 1]", unsqueeze_578: "f32[1, 184, 1, 1]", bitwise_and_6: "b8[8, 720, 1, 1]", unsqueeze_590: "f32[1, 720, 1, 1]", unsqueeze_602: "f32[1, 720, 1, 1]", unsqueeze_614: "f32[1, 120, 1, 1]", bitwise_and_7: "b8[8, 360, 1, 1]", unsqueeze_626: "f32[1, 360, 1, 1]", unsqueeze_638: "f32[1, 360, 1, 1]", unsqueeze_650: "f32[1, 120, 1, 1]", bitwise_and_8: "b8[8, 360, 1, 1]", unsqueeze_662: "f32[1, 360, 1, 1]", unsqueeze_674: "f32[1, 360, 1, 1]", unsqueeze_686: "f32[1, 120, 1, 1]", bitwise_and_9: "b8[8, 360, 1, 1]", unsqueeze_698: "f32[1, 360, 1, 1]", unsqueeze_710: "f32[1, 360, 1, 1]", unsqueeze_722: "f32[1, 120, 1, 1]", bitwise_and_10: "b8[8, 360, 1, 1]", unsqueeze_734: "f32[1, 360, 1, 1]", unsqueeze_746: "f32[1, 360, 1, 1]", unsqueeze_758: "f32[1, 120, 1, 1]", bitwise_and_11: "b8[8, 360, 1, 1]", unsqueeze_770: "f32[1, 360, 1, 1]", unsqueeze_782: "f32[1, 360, 1, 1]", unsqueeze_794: "f32[1, 120, 1, 1]", bitwise_and_12: "b8[8, 360, 1, 1]", unsqueeze_806: "f32[1, 360, 1, 1]", unsqueeze_818: "f32[1, 360, 1, 1]", unsqueeze_830: "f32[1, 72, 1, 1]", unsqueeze_842: "f32[1, 216, 1, 1]", unsqueeze_854: "f32[1, 216, 1, 1]", unsqueeze_866: "f32[1, 72, 1, 1]", unsqueeze_878: "f32[1, 216, 1, 1]", unsqueeze_890: "f32[1, 216, 1, 1]", unsqueeze_902: "f32[1, 72, 1, 1]", unsqueeze_914: "f32[1, 216, 1, 1]", unsqueeze_926: "f32[1, 216, 1, 1]", unsqueeze_938: "f32[1, 72, 1, 1]", unsqueeze_950: "f32[1, 216, 1, 1]", unsqueeze_962: "f32[1, 216, 1, 1]", unsqueeze_974: "f32[1, 72, 1, 1]", unsqueeze_986: "f32[1, 200, 1, 1]", unsqueeze_998: "f32[1, 200, 1, 1]", unsqueeze_1010: "f32[1, 40, 1, 1]", bitwise_and_13: "b8[8, 120, 1, 1]", unsqueeze_1022: "f32[1, 120, 1, 1]", unsqueeze_1034: "f32[1, 120, 1, 1]", unsqueeze_1046: "f32[1, 40, 1, 1]", bitwise_and_14: "b8[8, 120, 1, 1]", unsqueeze_1058: "f32[1, 120, 1, 1]", unsqueeze_1070: "f32[1, 120, 1, 1]", unsqueeze_1082: "f32[1, 40, 1, 1]", bitwise_and_15: "b8[8, 120, 1, 1]", unsqueeze_1094: "f32[1, 120, 1, 1]", unsqueeze_1106: "f32[1, 120, 1, 1]", unsqueeze_1118: "f32[1, 40, 1, 1]", bitwise_and_16: "b8[8, 120, 1, 1]", unsqueeze_1130: "f32[1, 120, 1, 1]", unsqueeze_1142: "f32[1, 120, 1, 1]", unsqueeze_1154: "f32[1, 40, 1, 1]", bitwise_and_17: "b8[8, 120, 1, 1]", unsqueeze_1166: "f32[1, 120, 1, 1]", unsqueeze_1178: "f32[1, 120, 1, 1]", unsqueeze_1190: "f32[1, 24, 1, 1]", unsqueeze_1202: "f32[1, 48, 1, 1]", unsqueeze_1214: "f32[1, 48, 1, 1]", unsqueeze_1226: "f32[1, 24, 1, 1]", unsqueeze_1238: "f32[1, 48, 1, 1]", unsqueeze_1250: "f32[1, 48, 1, 1]", unsqueeze_1262: "f32[1, 24, 1, 1]", unsqueeze_1274: "f32[1, 48, 1, 1]", unsqueeze_1286: "f32[1, 48, 1, 1]", unsqueeze_1298: "f32[1, 24, 1, 1]", unsqueeze_1310: "f32[1, 64, 1, 1]", unsqueeze_1322: "f32[1, 64, 1, 1]", unsqueeze_1334: "f32[1, 16, 1, 1]", unsqueeze_1346: "f32[1, 16, 1, 1]", unsqueeze_1358: "f32[1, 16, 1, 1]", unsqueeze_1370: "f32[1, 16, 1, 1]", unsqueeze_1382: "f32[1, 16, 1, 1]", tangents_1: "f32[8, 1000]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/linear.py:19, code: return F.linear(input, self.weight, self.bias)
    mm: "f32[8, 1984]" = torch.ops.aten.mm.default(tangents_1, permute_1);  permute_1 = None
    permute_2: "f32[1000, 8]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
    mm_1: "f32[1000, 1984]" = torch.ops.aten.mm.default(permute_2, view_1);  permute_2 = view_1 = None
    permute_3: "f32[1984, 1000]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_1: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view_2: "f32[1000]" = torch.ops.aten.reshape.default(sum_1, [1000]);  sum_1 = None
    permute_4: "f32[1000, 1984]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilenetv3.py:147, code: x = self.flatten(x)
    view_3: "f32[8, 1984, 1, 1]" = torch.ops.aten.reshape.default(mm, [8, 1984, 1, 1]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilenetv3.py:146, code: x = self.act2(x)
    lt: "b8[8, 1984, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_123, -3)
    le: "b8[8, 1984, 1, 1]" = torch.ops.aten.le.Scalar(convolution_123, 3)
    div_95: "f32[8, 1984, 1, 1]" = torch.ops.aten.div.Tensor(convolution_123, 3);  convolution_123 = None
    add_553: "f32[8, 1984, 1, 1]" = torch.ops.aten.add.Tensor(div_95, 0.5);  div_95 = None
    mul_704: "f32[8, 1984, 1, 1]" = torch.ops.aten.mul.Tensor(view_3, add_553);  add_553 = None
    where: "f32[8, 1984, 1, 1]" = torch.ops.aten.where.self(le, mul_704, view_3);  le = mul_704 = view_3 = None
    full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    where_1: "f32[8, 1984, 1, 1]" = torch.ops.aten.where.self(lt, full_default, where);  lt = where = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilenetv3.py:145, code: x = self.conv_head(x)
    convolution_backward = torch.ops.aten.convolution_backward.default(where_1, mean_18, primals_336, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_1 = mean_18 = primals_336 = None
    getitem_174: "f32[8, 1344, 1, 1]" = convolution_backward[0]
    getitem_175: "f32[1984, 1344, 1, 1]" = convolution_backward[1];  convolution_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    expand: "f32[8, 1344, 7, 7]" = torch.ops.aten.expand.default(getitem_174, [8, 1344, 7, 7]);  getitem_174 = None
    div_96: "f32[8, 1344, 7, 7]" = torch.ops.aten.div.Scalar(expand, 49);  expand = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_1: "b8[8, 1344, 7, 7]" = torch.ops.aten.lt.Scalar(clone_75, -3)
    le_1: "b8[8, 1344, 7, 7]" = torch.ops.aten.le.Scalar(clone_75, 3)
    div_97: "f32[8, 1344, 7, 7]" = torch.ops.aten.div.Tensor(clone_75, 3);  clone_75 = None
    add_554: "f32[8, 1344, 7, 7]" = torch.ops.aten.add.Tensor(div_97, 0.5);  div_97 = None
    mul_705: "f32[8, 1344, 7, 7]" = torch.ops.aten.mul.Tensor(div_96, add_554);  add_554 = None
    where_2: "f32[8, 1344, 7, 7]" = torch.ops.aten.where.self(le_1, mul_705, div_96);  le_1 = mul_705 = div_96 = None
    where_3: "f32[8, 1344, 7, 7]" = torch.ops.aten.where.self(lt_1, full_default, where_2);  lt_1 = where_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_2: "f32[1344]" = torch.ops.aten.sum.dim_IntList(where_3, [0, 2, 3])
    sub_87: "f32[8, 1344, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_122, unsqueeze_350);  convolution_122 = unsqueeze_350 = None
    mul_706: "f32[8, 1344, 7, 7]" = torch.ops.aten.mul.Tensor(where_3, sub_87)
    sum_3: "f32[1344]" = torch.ops.aten.sum.dim_IntList(mul_706, [0, 2, 3]);  mul_706 = None
    mul_707: "f32[1344]" = torch.ops.aten.mul.Tensor(sum_2, 0.002551020408163265)
    unsqueeze_351: "f32[1, 1344]" = torch.ops.aten.unsqueeze.default(mul_707, 0);  mul_707 = None
    unsqueeze_352: "f32[1, 1344, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_351, 2);  unsqueeze_351 = None
    unsqueeze_353: "f32[1, 1344, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_352, 3);  unsqueeze_352 = None
    mul_708: "f32[1344]" = torch.ops.aten.mul.Tensor(sum_3, 0.002551020408163265)
    mul_709: "f32[1344]" = torch.ops.aten.mul.Tensor(squeeze_259, squeeze_259)
    mul_710: "f32[1344]" = torch.ops.aten.mul.Tensor(mul_708, mul_709);  mul_708 = mul_709 = None
    unsqueeze_354: "f32[1, 1344]" = torch.ops.aten.unsqueeze.default(mul_710, 0);  mul_710 = None
    unsqueeze_355: "f32[1, 1344, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_354, 2);  unsqueeze_354 = None
    unsqueeze_356: "f32[1, 1344, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_355, 3);  unsqueeze_355 = None
    mul_711: "f32[1344]" = torch.ops.aten.mul.Tensor(squeeze_259, primals_173);  primals_173 = None
    unsqueeze_357: "f32[1, 1344]" = torch.ops.aten.unsqueeze.default(mul_711, 0);  mul_711 = None
    unsqueeze_358: "f32[1, 1344, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_357, 2);  unsqueeze_357 = None
    unsqueeze_359: "f32[1, 1344, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_358, 3);  unsqueeze_358 = None
    mul_712: "f32[8, 1344, 7, 7]" = torch.ops.aten.mul.Tensor(sub_87, unsqueeze_356);  sub_87 = unsqueeze_356 = None
    sub_89: "f32[8, 1344, 7, 7]" = torch.ops.aten.sub.Tensor(where_3, mul_712);  where_3 = mul_712 = None
    sub_90: "f32[8, 1344, 7, 7]" = torch.ops.aten.sub.Tensor(sub_89, unsqueeze_353);  sub_89 = unsqueeze_353 = None
    mul_713: "f32[8, 1344, 7, 7]" = torch.ops.aten.mul.Tensor(sub_90, unsqueeze_359);  sub_90 = unsqueeze_359 = None
    mul_714: "f32[1344]" = torch.ops.aten.mul.Tensor(sum_3, squeeze_259);  sum_3 = squeeze_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:82, code: x = self.conv(x)
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(mul_713, add_545, primals_335, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_713 = add_545 = primals_335 = None
    getitem_177: "f32[8, 224, 7, 7]" = convolution_backward_1[0]
    getitem_178: "f32[1344, 224, 1, 1]" = convolution_backward_1[1];  convolution_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_4: "f32[224]" = torch.ops.aten.sum.dim_IntList(getitem_177, [0, 2, 3])
    sub_91: "f32[8, 224, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_121, unsqueeze_362);  convolution_121 = unsqueeze_362 = None
    mul_715: "f32[8, 224, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_177, sub_91)
    sum_5: "f32[224]" = torch.ops.aten.sum.dim_IntList(mul_715, [0, 2, 3]);  mul_715 = None
    mul_716: "f32[224]" = torch.ops.aten.mul.Tensor(sum_4, 0.002551020408163265)
    unsqueeze_363: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_716, 0);  mul_716 = None
    unsqueeze_364: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_363, 2);  unsqueeze_363 = None
    unsqueeze_365: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_364, 3);  unsqueeze_364 = None
    mul_717: "f32[224]" = torch.ops.aten.mul.Tensor(sum_5, 0.002551020408163265)
    mul_718: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_256, squeeze_256)
    mul_719: "f32[224]" = torch.ops.aten.mul.Tensor(mul_717, mul_718);  mul_717 = mul_718 = None
    unsqueeze_366: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_719, 0);  mul_719 = None
    unsqueeze_367: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_366, 2);  unsqueeze_366 = None
    unsqueeze_368: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_367, 3);  unsqueeze_367 = None
    mul_720: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_256, primals_171);  primals_171 = None
    unsqueeze_369: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_720, 0);  mul_720 = None
    unsqueeze_370: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_369, 2);  unsqueeze_369 = None
    unsqueeze_371: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_370, 3);  unsqueeze_370 = None
    mul_721: "f32[8, 224, 7, 7]" = torch.ops.aten.mul.Tensor(sub_91, unsqueeze_368);  sub_91 = unsqueeze_368 = None
    sub_93: "f32[8, 224, 7, 7]" = torch.ops.aten.sub.Tensor(getitem_177, mul_721);  getitem_177 = mul_721 = None
    sub_94: "f32[8, 224, 7, 7]" = torch.ops.aten.sub.Tensor(sub_93, unsqueeze_365);  sub_93 = unsqueeze_365 = None
    mul_722: "f32[8, 224, 7, 7]" = torch.ops.aten.mul.Tensor(sub_94, unsqueeze_371);  sub_94 = unsqueeze_371 = None
    mul_723: "f32[224]" = torch.ops.aten.mul.Tensor(sum_5, squeeze_256);  sum_5 = squeeze_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(mul_722, mul_687, primals_334, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_722 = mul_687 = primals_334 = None
    getitem_180: "f32[8, 1104, 7, 7]" = convolution_backward_2[0]
    getitem_181: "f32[224, 1104, 1, 1]" = convolution_backward_2[1];  convolution_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_724: "f32[8, 1104, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_180, div_90);  div_90 = None
    mul_725: "f32[8, 1104, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_180, div_92);  getitem_180 = div_92 = None
    sum_6: "f32[8, 1104, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_724, [2, 3], True);  mul_724 = None
    mul_726: "f32[8, 1104, 1, 1]" = torch.ops.aten.mul.Tensor(sum_6, 0.16666666666666666);  sum_6 = None
    where_4: "f32[8, 1104, 1, 1]" = torch.ops.aten.where.self(bitwise_and, mul_726, full_default);  bitwise_and = mul_726 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_3 = torch.ops.aten.convolution_backward.default(where_4, div_91, primals_332, [1104], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_4 = div_91 = primals_332 = None
    getitem_183: "f32[8, 48, 1, 1]" = convolution_backward_3[0]
    getitem_184: "f32[1104, 48, 1, 1]" = convolution_backward_3[1]
    getitem_185: "f32[1104]" = convolution_backward_3[2];  convolution_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    lt_3: "b8[8, 48, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_119, -3)
    le_2: "b8[8, 48, 1, 1]" = torch.ops.aten.le.Scalar(convolution_119, 3)
    div_98: "f32[8, 48, 1, 1]" = torch.ops.aten.div.Tensor(convolution_119, 3);  convolution_119 = None
    add_555: "f32[8, 48, 1, 1]" = torch.ops.aten.add.Tensor(div_98, 0.5);  div_98 = None
    mul_727: "f32[8, 48, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_183, add_555);  add_555 = None
    where_5: "f32[8, 48, 1, 1]" = torch.ops.aten.where.self(le_2, mul_727, getitem_183);  le_2 = mul_727 = getitem_183 = None
    where_6: "f32[8, 48, 1, 1]" = torch.ops.aten.where.self(lt_3, full_default, where_5);  lt_3 = where_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_4 = torch.ops.aten.convolution_backward.default(where_6, mean_17, primals_330, [48], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_6 = mean_17 = primals_330 = None
    getitem_186: "f32[8, 1104, 1, 1]" = convolution_backward_4[0]
    getitem_187: "f32[48, 1104, 1, 1]" = convolution_backward_4[1]
    getitem_188: "f32[48]" = convolution_backward_4[2];  convolution_backward_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_1: "f32[8, 1104, 7, 7]" = torch.ops.aten.expand.default(getitem_186, [8, 1104, 7, 7]);  getitem_186 = None
    div_99: "f32[8, 1104, 7, 7]" = torch.ops.aten.div.Scalar(expand_1, 49);  expand_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_556: "f32[8, 1104, 7, 7]" = torch.ops.aten.add.Tensor(mul_725, div_99);  mul_725 = div_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_4: "b8[8, 1104, 7, 7]" = torch.ops.aten.lt.Scalar(clone_73, -3)
    le_3: "b8[8, 1104, 7, 7]" = torch.ops.aten.le.Scalar(clone_73, 3)
    div_100: "f32[8, 1104, 7, 7]" = torch.ops.aten.div.Tensor(clone_73, 3);  clone_73 = None
    add_557: "f32[8, 1104, 7, 7]" = torch.ops.aten.add.Tensor(div_100, 0.5);  div_100 = None
    mul_728: "f32[8, 1104, 7, 7]" = torch.ops.aten.mul.Tensor(add_556, add_557);  add_557 = None
    where_7: "f32[8, 1104, 7, 7]" = torch.ops.aten.where.self(le_3, mul_728, add_556);  le_3 = mul_728 = add_556 = None
    where_8: "f32[8, 1104, 7, 7]" = torch.ops.aten.where.self(lt_4, full_default, where_7);  lt_4 = where_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_7: "f32[1104]" = torch.ops.aten.sum.dim_IntList(where_8, [0, 2, 3])
    sub_95: "f32[8, 1104, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_118, unsqueeze_374);  convolution_118 = unsqueeze_374 = None
    mul_729: "f32[8, 1104, 7, 7]" = torch.ops.aten.mul.Tensor(where_8, sub_95)
    sum_8: "f32[1104]" = torch.ops.aten.sum.dim_IntList(mul_729, [0, 2, 3]);  mul_729 = None
    mul_730: "f32[1104]" = torch.ops.aten.mul.Tensor(sum_7, 0.002551020408163265)
    unsqueeze_375: "f32[1, 1104]" = torch.ops.aten.unsqueeze.default(mul_730, 0);  mul_730 = None
    unsqueeze_376: "f32[1, 1104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_375, 2);  unsqueeze_375 = None
    unsqueeze_377: "f32[1, 1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_376, 3);  unsqueeze_376 = None
    mul_731: "f32[1104]" = torch.ops.aten.mul.Tensor(sum_8, 0.002551020408163265)
    mul_732: "f32[1104]" = torch.ops.aten.mul.Tensor(squeeze_253, squeeze_253)
    mul_733: "f32[1104]" = torch.ops.aten.mul.Tensor(mul_731, mul_732);  mul_731 = mul_732 = None
    unsqueeze_378: "f32[1, 1104]" = torch.ops.aten.unsqueeze.default(mul_733, 0);  mul_733 = None
    unsqueeze_379: "f32[1, 1104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_378, 2);  unsqueeze_378 = None
    unsqueeze_380: "f32[1, 1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_379, 3);  unsqueeze_379 = None
    mul_734: "f32[1104]" = torch.ops.aten.mul.Tensor(squeeze_253, primals_169);  primals_169 = None
    unsqueeze_381: "f32[1, 1104]" = torch.ops.aten.unsqueeze.default(mul_734, 0);  mul_734 = None
    unsqueeze_382: "f32[1, 1104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_381, 2);  unsqueeze_381 = None
    unsqueeze_383: "f32[1, 1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_382, 3);  unsqueeze_382 = None
    mul_735: "f32[8, 1104, 7, 7]" = torch.ops.aten.mul.Tensor(sub_95, unsqueeze_380);  sub_95 = unsqueeze_380 = None
    sub_97: "f32[8, 1104, 7, 7]" = torch.ops.aten.sub.Tensor(where_8, mul_735);  where_8 = mul_735 = None
    sub_98: "f32[8, 1104, 7, 7]" = torch.ops.aten.sub.Tensor(sub_97, unsqueeze_377);  sub_97 = unsqueeze_377 = None
    mul_736: "f32[8, 1104, 7, 7]" = torch.ops.aten.mul.Tensor(sub_98, unsqueeze_383);  sub_98 = unsqueeze_383 = None
    mul_737: "f32[1104]" = torch.ops.aten.mul.Tensor(sum_8, squeeze_253);  sum_8 = squeeze_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_5 = torch.ops.aten.convolution_backward.default(mul_736, div_89, primals_329, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 1104, [True, True, False]);  mul_736 = div_89 = primals_329 = None
    getitem_189: "f32[8, 1104, 7, 7]" = convolution_backward_5[0]
    getitem_190: "f32[1104, 1, 5, 5]" = convolution_backward_5[1];  convolution_backward_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_5: "b8[8, 1104, 7, 7]" = torch.ops.aten.lt.Scalar(clone_72, -3)
    le_4: "b8[8, 1104, 7, 7]" = torch.ops.aten.le.Scalar(clone_72, 3)
    div_101: "f32[8, 1104, 7, 7]" = torch.ops.aten.div.Tensor(clone_72, 3);  clone_72 = None
    add_558: "f32[8, 1104, 7, 7]" = torch.ops.aten.add.Tensor(div_101, 0.5);  div_101 = None
    mul_738: "f32[8, 1104, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_189, add_558);  add_558 = None
    where_9: "f32[8, 1104, 7, 7]" = torch.ops.aten.where.self(le_4, mul_738, getitem_189);  le_4 = mul_738 = getitem_189 = None
    where_10: "f32[8, 1104, 7, 7]" = torch.ops.aten.where.self(lt_5, full_default, where_9);  lt_5 = where_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_9: "f32[1104]" = torch.ops.aten.sum.dim_IntList(where_10, [0, 2, 3])
    sub_99: "f32[8, 1104, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_117, unsqueeze_386);  convolution_117 = unsqueeze_386 = None
    mul_739: "f32[8, 1104, 7, 7]" = torch.ops.aten.mul.Tensor(where_10, sub_99)
    sum_10: "f32[1104]" = torch.ops.aten.sum.dim_IntList(mul_739, [0, 2, 3]);  mul_739 = None
    mul_740: "f32[1104]" = torch.ops.aten.mul.Tensor(sum_9, 0.002551020408163265)
    unsqueeze_387: "f32[1, 1104]" = torch.ops.aten.unsqueeze.default(mul_740, 0);  mul_740 = None
    unsqueeze_388: "f32[1, 1104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_387, 2);  unsqueeze_387 = None
    unsqueeze_389: "f32[1, 1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_388, 3);  unsqueeze_388 = None
    mul_741: "f32[1104]" = torch.ops.aten.mul.Tensor(sum_10, 0.002551020408163265)
    mul_742: "f32[1104]" = torch.ops.aten.mul.Tensor(squeeze_250, squeeze_250)
    mul_743: "f32[1104]" = torch.ops.aten.mul.Tensor(mul_741, mul_742);  mul_741 = mul_742 = None
    unsqueeze_390: "f32[1, 1104]" = torch.ops.aten.unsqueeze.default(mul_743, 0);  mul_743 = None
    unsqueeze_391: "f32[1, 1104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_390, 2);  unsqueeze_390 = None
    unsqueeze_392: "f32[1, 1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_391, 3);  unsqueeze_391 = None
    mul_744: "f32[1104]" = torch.ops.aten.mul.Tensor(squeeze_250, primals_167);  primals_167 = None
    unsqueeze_393: "f32[1, 1104]" = torch.ops.aten.unsqueeze.default(mul_744, 0);  mul_744 = None
    unsqueeze_394: "f32[1, 1104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_393, 2);  unsqueeze_393 = None
    unsqueeze_395: "f32[1, 1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_394, 3);  unsqueeze_394 = None
    mul_745: "f32[8, 1104, 7, 7]" = torch.ops.aten.mul.Tensor(sub_99, unsqueeze_392);  sub_99 = unsqueeze_392 = None
    sub_101: "f32[8, 1104, 7, 7]" = torch.ops.aten.sub.Tensor(where_10, mul_745);  where_10 = mul_745 = None
    sub_102: "f32[8, 1104, 7, 7]" = torch.ops.aten.sub.Tensor(sub_101, unsqueeze_389);  sub_101 = unsqueeze_389 = None
    mul_746: "f32[8, 1104, 7, 7]" = torch.ops.aten.mul.Tensor(sub_102, unsqueeze_395);  sub_102 = unsqueeze_395 = None
    mul_747: "f32[1104]" = torch.ops.aten.mul.Tensor(sum_10, squeeze_250);  sum_10 = squeeze_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_6 = torch.ops.aten.convolution_backward.default(mul_746, add_526, primals_328, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_746 = add_526 = primals_328 = None
    getitem_192: "f32[8, 184, 7, 7]" = convolution_backward_6[0]
    getitem_193: "f32[1104, 184, 1, 1]" = convolution_backward_6[1];  convolution_backward_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_11: "f32[184]" = torch.ops.aten.sum.dim_IntList(getitem_192, [0, 2, 3])
    sub_103: "f32[8, 184, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_116, unsqueeze_398);  convolution_116 = unsqueeze_398 = None
    mul_748: "f32[8, 184, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_192, sub_103)
    sum_12: "f32[184]" = torch.ops.aten.sum.dim_IntList(mul_748, [0, 2, 3]);  mul_748 = None
    mul_749: "f32[184]" = torch.ops.aten.mul.Tensor(sum_11, 0.002551020408163265)
    unsqueeze_399: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(mul_749, 0);  mul_749 = None
    unsqueeze_400: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_399, 2);  unsqueeze_399 = None
    unsqueeze_401: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_400, 3);  unsqueeze_400 = None
    mul_750: "f32[184]" = torch.ops.aten.mul.Tensor(sum_12, 0.002551020408163265)
    mul_751: "f32[184]" = torch.ops.aten.mul.Tensor(squeeze_247, squeeze_247)
    mul_752: "f32[184]" = torch.ops.aten.mul.Tensor(mul_750, mul_751);  mul_750 = mul_751 = None
    unsqueeze_402: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(mul_752, 0);  mul_752 = None
    unsqueeze_403: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_402, 2);  unsqueeze_402 = None
    unsqueeze_404: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_403, 3);  unsqueeze_403 = None
    mul_753: "f32[184]" = torch.ops.aten.mul.Tensor(squeeze_247, primals_165);  primals_165 = None
    unsqueeze_405: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(mul_753, 0);  mul_753 = None
    unsqueeze_406: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_405, 2);  unsqueeze_405 = None
    unsqueeze_407: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_406, 3);  unsqueeze_406 = None
    mul_754: "f32[8, 184, 7, 7]" = torch.ops.aten.mul.Tensor(sub_103, unsqueeze_404);  sub_103 = unsqueeze_404 = None
    sub_105: "f32[8, 184, 7, 7]" = torch.ops.aten.sub.Tensor(getitem_192, mul_754);  mul_754 = None
    sub_106: "f32[8, 184, 7, 7]" = torch.ops.aten.sub.Tensor(sub_105, unsqueeze_401);  sub_105 = unsqueeze_401 = None
    mul_755: "f32[8, 184, 7, 7]" = torch.ops.aten.mul.Tensor(sub_106, unsqueeze_407);  sub_106 = unsqueeze_407 = None
    mul_756: "f32[184]" = torch.ops.aten.mul.Tensor(sum_12, squeeze_247);  sum_12 = squeeze_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_7 = torch.ops.aten.convolution_backward.default(mul_755, mul_662, primals_327, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_755 = mul_662 = primals_327 = None
    getitem_195: "f32[8, 736, 7, 7]" = convolution_backward_7[0]
    getitem_196: "f32[184, 736, 1, 1]" = convolution_backward_7[1];  convolution_backward_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_757: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_195, div_86);  div_86 = None
    mul_758: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_195, div_88);  getitem_195 = div_88 = None
    sum_13: "f32[8, 736, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_757, [2, 3], True);  mul_757 = None
    mul_759: "f32[8, 736, 1, 1]" = torch.ops.aten.mul.Tensor(sum_13, 0.16666666666666666);  sum_13 = None
    where_11: "f32[8, 736, 1, 1]" = torch.ops.aten.where.self(bitwise_and_1, mul_759, full_default);  bitwise_and_1 = mul_759 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_8 = torch.ops.aten.convolution_backward.default(where_11, div_87, primals_325, [736], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_11 = div_87 = primals_325 = None
    getitem_198: "f32[8, 48, 1, 1]" = convolution_backward_8[0]
    getitem_199: "f32[736, 48, 1, 1]" = convolution_backward_8[1]
    getitem_200: "f32[736]" = convolution_backward_8[2];  convolution_backward_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    lt_7: "b8[8, 48, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_114, -3)
    le_5: "b8[8, 48, 1, 1]" = torch.ops.aten.le.Scalar(convolution_114, 3)
    div_102: "f32[8, 48, 1, 1]" = torch.ops.aten.div.Tensor(convolution_114, 3);  convolution_114 = None
    add_559: "f32[8, 48, 1, 1]" = torch.ops.aten.add.Tensor(div_102, 0.5);  div_102 = None
    mul_760: "f32[8, 48, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_198, add_559);  add_559 = None
    where_12: "f32[8, 48, 1, 1]" = torch.ops.aten.where.self(le_5, mul_760, getitem_198);  le_5 = mul_760 = getitem_198 = None
    where_13: "f32[8, 48, 1, 1]" = torch.ops.aten.where.self(lt_7, full_default, where_12);  lt_7 = where_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_9 = torch.ops.aten.convolution_backward.default(where_13, mean_16, primals_323, [48], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_13 = mean_16 = primals_323 = None
    getitem_201: "f32[8, 736, 1, 1]" = convolution_backward_9[0]
    getitem_202: "f32[48, 736, 1, 1]" = convolution_backward_9[1]
    getitem_203: "f32[48]" = convolution_backward_9[2];  convolution_backward_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_2: "f32[8, 736, 7, 7]" = torch.ops.aten.expand.default(getitem_201, [8, 736, 7, 7]);  getitem_201 = None
    div_103: "f32[8, 736, 7, 7]" = torch.ops.aten.div.Scalar(expand_2, 49);  expand_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_560: "f32[8, 736, 7, 7]" = torch.ops.aten.add.Tensor(mul_758, div_103);  mul_758 = div_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_8: "b8[8, 736, 7, 7]" = torch.ops.aten.lt.Scalar(clone_70, -3)
    le_6: "b8[8, 736, 7, 7]" = torch.ops.aten.le.Scalar(clone_70, 3)
    div_104: "f32[8, 736, 7, 7]" = torch.ops.aten.div.Tensor(clone_70, 3);  clone_70 = None
    add_561: "f32[8, 736, 7, 7]" = torch.ops.aten.add.Tensor(div_104, 0.5);  div_104 = None
    mul_761: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(add_560, add_561);  add_561 = None
    where_14: "f32[8, 736, 7, 7]" = torch.ops.aten.where.self(le_6, mul_761, add_560);  le_6 = mul_761 = add_560 = None
    where_15: "f32[8, 736, 7, 7]" = torch.ops.aten.where.self(lt_8, full_default, where_14);  lt_8 = where_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_14: "f32[736]" = torch.ops.aten.sum.dim_IntList(where_15, [0, 2, 3])
    sub_107: "f32[8, 736, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_113, unsqueeze_410);  convolution_113 = unsqueeze_410 = None
    mul_762: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(where_15, sub_107)
    sum_15: "f32[736]" = torch.ops.aten.sum.dim_IntList(mul_762, [0, 2, 3]);  mul_762 = None
    mul_763: "f32[736]" = torch.ops.aten.mul.Tensor(sum_14, 0.002551020408163265)
    unsqueeze_411: "f32[1, 736]" = torch.ops.aten.unsqueeze.default(mul_763, 0);  mul_763 = None
    unsqueeze_412: "f32[1, 736, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_411, 2);  unsqueeze_411 = None
    unsqueeze_413: "f32[1, 736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_412, 3);  unsqueeze_412 = None
    mul_764: "f32[736]" = torch.ops.aten.mul.Tensor(sum_15, 0.002551020408163265)
    mul_765: "f32[736]" = torch.ops.aten.mul.Tensor(squeeze_244, squeeze_244)
    mul_766: "f32[736]" = torch.ops.aten.mul.Tensor(mul_764, mul_765);  mul_764 = mul_765 = None
    unsqueeze_414: "f32[1, 736]" = torch.ops.aten.unsqueeze.default(mul_766, 0);  mul_766 = None
    unsqueeze_415: "f32[1, 736, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_414, 2);  unsqueeze_414 = None
    unsqueeze_416: "f32[1, 736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_415, 3);  unsqueeze_415 = None
    mul_767: "f32[736]" = torch.ops.aten.mul.Tensor(squeeze_244, primals_163);  primals_163 = None
    unsqueeze_417: "f32[1, 736]" = torch.ops.aten.unsqueeze.default(mul_767, 0);  mul_767 = None
    unsqueeze_418: "f32[1, 736, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_417, 2);  unsqueeze_417 = None
    unsqueeze_419: "f32[1, 736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_418, 3);  unsqueeze_418 = None
    mul_768: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(sub_107, unsqueeze_416);  sub_107 = unsqueeze_416 = None
    sub_109: "f32[8, 736, 7, 7]" = torch.ops.aten.sub.Tensor(where_15, mul_768);  where_15 = mul_768 = None
    sub_110: "f32[8, 736, 7, 7]" = torch.ops.aten.sub.Tensor(sub_109, unsqueeze_413);  sub_109 = unsqueeze_413 = None
    mul_769: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(sub_110, unsqueeze_419);  sub_110 = unsqueeze_419 = None
    mul_770: "f32[736]" = torch.ops.aten.mul.Tensor(sum_15, squeeze_244);  sum_15 = squeeze_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_10 = torch.ops.aten.convolution_backward.default(mul_769, div_85, primals_322, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 736, [True, True, False]);  mul_769 = div_85 = primals_322 = None
    getitem_204: "f32[8, 736, 7, 7]" = convolution_backward_10[0]
    getitem_205: "f32[736, 1, 5, 5]" = convolution_backward_10[1];  convolution_backward_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_9: "b8[8, 736, 7, 7]" = torch.ops.aten.lt.Scalar(clone_69, -3)
    le_7: "b8[8, 736, 7, 7]" = torch.ops.aten.le.Scalar(clone_69, 3)
    div_105: "f32[8, 736, 7, 7]" = torch.ops.aten.div.Tensor(clone_69, 3);  clone_69 = None
    add_562: "f32[8, 736, 7, 7]" = torch.ops.aten.add.Tensor(div_105, 0.5);  div_105 = None
    mul_771: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_204, add_562);  add_562 = None
    where_16: "f32[8, 736, 7, 7]" = torch.ops.aten.where.self(le_7, mul_771, getitem_204);  le_7 = mul_771 = getitem_204 = None
    where_17: "f32[8, 736, 7, 7]" = torch.ops.aten.where.self(lt_9, full_default, where_16);  lt_9 = where_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_16: "f32[736]" = torch.ops.aten.sum.dim_IntList(where_17, [0, 2, 3])
    sub_111: "f32[8, 736, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_112, unsqueeze_422);  convolution_112 = unsqueeze_422 = None
    mul_772: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(where_17, sub_111)
    sum_17: "f32[736]" = torch.ops.aten.sum.dim_IntList(mul_772, [0, 2, 3]);  mul_772 = None
    mul_773: "f32[736]" = torch.ops.aten.mul.Tensor(sum_16, 0.002551020408163265)
    unsqueeze_423: "f32[1, 736]" = torch.ops.aten.unsqueeze.default(mul_773, 0);  mul_773 = None
    unsqueeze_424: "f32[1, 736, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_423, 2);  unsqueeze_423 = None
    unsqueeze_425: "f32[1, 736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_424, 3);  unsqueeze_424 = None
    mul_774: "f32[736]" = torch.ops.aten.mul.Tensor(sum_17, 0.002551020408163265)
    mul_775: "f32[736]" = torch.ops.aten.mul.Tensor(squeeze_241, squeeze_241)
    mul_776: "f32[736]" = torch.ops.aten.mul.Tensor(mul_774, mul_775);  mul_774 = mul_775 = None
    unsqueeze_426: "f32[1, 736]" = torch.ops.aten.unsqueeze.default(mul_776, 0);  mul_776 = None
    unsqueeze_427: "f32[1, 736, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_426, 2);  unsqueeze_426 = None
    unsqueeze_428: "f32[1, 736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_427, 3);  unsqueeze_427 = None
    mul_777: "f32[736]" = torch.ops.aten.mul.Tensor(squeeze_241, primals_161);  primals_161 = None
    unsqueeze_429: "f32[1, 736]" = torch.ops.aten.unsqueeze.default(mul_777, 0);  mul_777 = None
    unsqueeze_430: "f32[1, 736, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_429, 2);  unsqueeze_429 = None
    unsqueeze_431: "f32[1, 736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_430, 3);  unsqueeze_430 = None
    mul_778: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(sub_111, unsqueeze_428);  sub_111 = unsqueeze_428 = None
    sub_113: "f32[8, 736, 7, 7]" = torch.ops.aten.sub.Tensor(where_17, mul_778);  where_17 = mul_778 = None
    sub_114: "f32[8, 736, 7, 7]" = torch.ops.aten.sub.Tensor(sub_113, unsqueeze_425);  sub_113 = unsqueeze_425 = None
    mul_779: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(sub_114, unsqueeze_431);  sub_114 = unsqueeze_431 = None
    mul_780: "f32[736]" = torch.ops.aten.mul.Tensor(sum_17, squeeze_241);  sum_17 = squeeze_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_11 = torch.ops.aten.convolution_backward.default(mul_779, add_506, primals_321, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_779 = add_506 = primals_321 = None
    getitem_207: "f32[8, 184, 7, 7]" = convolution_backward_11[0]
    getitem_208: "f32[736, 184, 1, 1]" = convolution_backward_11[1];  convolution_backward_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_563: "f32[8, 184, 7, 7]" = torch.ops.aten.add.Tensor(getitem_192, getitem_207);  getitem_192 = getitem_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_18: "f32[184]" = torch.ops.aten.sum.dim_IntList(add_563, [0, 2, 3])
    sub_115: "f32[8, 184, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_111, unsqueeze_434);  convolution_111 = unsqueeze_434 = None
    mul_781: "f32[8, 184, 7, 7]" = torch.ops.aten.mul.Tensor(add_563, sub_115)
    sum_19: "f32[184]" = torch.ops.aten.sum.dim_IntList(mul_781, [0, 2, 3]);  mul_781 = None
    mul_782: "f32[184]" = torch.ops.aten.mul.Tensor(sum_18, 0.002551020408163265)
    unsqueeze_435: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(mul_782, 0);  mul_782 = None
    unsqueeze_436: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_435, 2);  unsqueeze_435 = None
    unsqueeze_437: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_436, 3);  unsqueeze_436 = None
    mul_783: "f32[184]" = torch.ops.aten.mul.Tensor(sum_19, 0.002551020408163265)
    mul_784: "f32[184]" = torch.ops.aten.mul.Tensor(squeeze_238, squeeze_238)
    mul_785: "f32[184]" = torch.ops.aten.mul.Tensor(mul_783, mul_784);  mul_783 = mul_784 = None
    unsqueeze_438: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(mul_785, 0);  mul_785 = None
    unsqueeze_439: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_438, 2);  unsqueeze_438 = None
    unsqueeze_440: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_439, 3);  unsqueeze_439 = None
    mul_786: "f32[184]" = torch.ops.aten.mul.Tensor(squeeze_238, primals_159);  primals_159 = None
    unsqueeze_441: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(mul_786, 0);  mul_786 = None
    unsqueeze_442: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_441, 2);  unsqueeze_441 = None
    unsqueeze_443: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_442, 3);  unsqueeze_442 = None
    mul_787: "f32[8, 184, 7, 7]" = torch.ops.aten.mul.Tensor(sub_115, unsqueeze_440);  sub_115 = unsqueeze_440 = None
    sub_117: "f32[8, 184, 7, 7]" = torch.ops.aten.sub.Tensor(add_563, mul_787);  mul_787 = None
    sub_118: "f32[8, 184, 7, 7]" = torch.ops.aten.sub.Tensor(sub_117, unsqueeze_437);  sub_117 = unsqueeze_437 = None
    mul_788: "f32[8, 184, 7, 7]" = torch.ops.aten.mul.Tensor(sub_118, unsqueeze_443);  sub_118 = unsqueeze_443 = None
    mul_789: "f32[184]" = torch.ops.aten.mul.Tensor(sum_19, squeeze_238);  sum_19 = squeeze_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_12 = torch.ops.aten.convolution_backward.default(mul_788, mul_637, primals_320, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_788 = mul_637 = primals_320 = None
    getitem_210: "f32[8, 736, 7, 7]" = convolution_backward_12[0]
    getitem_211: "f32[184, 736, 1, 1]" = convolution_backward_12[1];  convolution_backward_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_790: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_210, div_82);  div_82 = None
    mul_791: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_210, div_84);  getitem_210 = div_84 = None
    sum_20: "f32[8, 736, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_790, [2, 3], True);  mul_790 = None
    mul_792: "f32[8, 736, 1, 1]" = torch.ops.aten.mul.Tensor(sum_20, 0.16666666666666666);  sum_20 = None
    where_18: "f32[8, 736, 1, 1]" = torch.ops.aten.where.self(bitwise_and_2, mul_792, full_default);  bitwise_and_2 = mul_792 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_13 = torch.ops.aten.convolution_backward.default(where_18, div_83, primals_318, [736], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_18 = div_83 = primals_318 = None
    getitem_213: "f32[8, 48, 1, 1]" = convolution_backward_13[0]
    getitem_214: "f32[736, 48, 1, 1]" = convolution_backward_13[1]
    getitem_215: "f32[736]" = convolution_backward_13[2];  convolution_backward_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    lt_11: "b8[8, 48, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_109, -3)
    le_8: "b8[8, 48, 1, 1]" = torch.ops.aten.le.Scalar(convolution_109, 3)
    div_106: "f32[8, 48, 1, 1]" = torch.ops.aten.div.Tensor(convolution_109, 3);  convolution_109 = None
    add_564: "f32[8, 48, 1, 1]" = torch.ops.aten.add.Tensor(div_106, 0.5);  div_106 = None
    mul_793: "f32[8, 48, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_213, add_564);  add_564 = None
    where_19: "f32[8, 48, 1, 1]" = torch.ops.aten.where.self(le_8, mul_793, getitem_213);  le_8 = mul_793 = getitem_213 = None
    where_20: "f32[8, 48, 1, 1]" = torch.ops.aten.where.self(lt_11, full_default, where_19);  lt_11 = where_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_14 = torch.ops.aten.convolution_backward.default(where_20, mean_15, primals_316, [48], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_20 = mean_15 = primals_316 = None
    getitem_216: "f32[8, 736, 1, 1]" = convolution_backward_14[0]
    getitem_217: "f32[48, 736, 1, 1]" = convolution_backward_14[1]
    getitem_218: "f32[48]" = convolution_backward_14[2];  convolution_backward_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_3: "f32[8, 736, 7, 7]" = torch.ops.aten.expand.default(getitem_216, [8, 736, 7, 7]);  getitem_216 = None
    div_107: "f32[8, 736, 7, 7]" = torch.ops.aten.div.Scalar(expand_3, 49);  expand_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_565: "f32[8, 736, 7, 7]" = torch.ops.aten.add.Tensor(mul_791, div_107);  mul_791 = div_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_12: "b8[8, 736, 7, 7]" = torch.ops.aten.lt.Scalar(clone_67, -3)
    le_9: "b8[8, 736, 7, 7]" = torch.ops.aten.le.Scalar(clone_67, 3)
    div_108: "f32[8, 736, 7, 7]" = torch.ops.aten.div.Tensor(clone_67, 3);  clone_67 = None
    add_566: "f32[8, 736, 7, 7]" = torch.ops.aten.add.Tensor(div_108, 0.5);  div_108 = None
    mul_794: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(add_565, add_566);  add_566 = None
    where_21: "f32[8, 736, 7, 7]" = torch.ops.aten.where.self(le_9, mul_794, add_565);  le_9 = mul_794 = add_565 = None
    where_22: "f32[8, 736, 7, 7]" = torch.ops.aten.where.self(lt_12, full_default, where_21);  lt_12 = where_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_21: "f32[736]" = torch.ops.aten.sum.dim_IntList(where_22, [0, 2, 3])
    sub_119: "f32[8, 736, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_108, unsqueeze_446);  convolution_108 = unsqueeze_446 = None
    mul_795: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(where_22, sub_119)
    sum_22: "f32[736]" = torch.ops.aten.sum.dim_IntList(mul_795, [0, 2, 3]);  mul_795 = None
    mul_796: "f32[736]" = torch.ops.aten.mul.Tensor(sum_21, 0.002551020408163265)
    unsqueeze_447: "f32[1, 736]" = torch.ops.aten.unsqueeze.default(mul_796, 0);  mul_796 = None
    unsqueeze_448: "f32[1, 736, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_447, 2);  unsqueeze_447 = None
    unsqueeze_449: "f32[1, 736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_448, 3);  unsqueeze_448 = None
    mul_797: "f32[736]" = torch.ops.aten.mul.Tensor(sum_22, 0.002551020408163265)
    mul_798: "f32[736]" = torch.ops.aten.mul.Tensor(squeeze_235, squeeze_235)
    mul_799: "f32[736]" = torch.ops.aten.mul.Tensor(mul_797, mul_798);  mul_797 = mul_798 = None
    unsqueeze_450: "f32[1, 736]" = torch.ops.aten.unsqueeze.default(mul_799, 0);  mul_799 = None
    unsqueeze_451: "f32[1, 736, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_450, 2);  unsqueeze_450 = None
    unsqueeze_452: "f32[1, 736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_451, 3);  unsqueeze_451 = None
    mul_800: "f32[736]" = torch.ops.aten.mul.Tensor(squeeze_235, primals_157);  primals_157 = None
    unsqueeze_453: "f32[1, 736]" = torch.ops.aten.unsqueeze.default(mul_800, 0);  mul_800 = None
    unsqueeze_454: "f32[1, 736, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_453, 2);  unsqueeze_453 = None
    unsqueeze_455: "f32[1, 736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_454, 3);  unsqueeze_454 = None
    mul_801: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(sub_119, unsqueeze_452);  sub_119 = unsqueeze_452 = None
    sub_121: "f32[8, 736, 7, 7]" = torch.ops.aten.sub.Tensor(where_22, mul_801);  where_22 = mul_801 = None
    sub_122: "f32[8, 736, 7, 7]" = torch.ops.aten.sub.Tensor(sub_121, unsqueeze_449);  sub_121 = unsqueeze_449 = None
    mul_802: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(sub_122, unsqueeze_455);  sub_122 = unsqueeze_455 = None
    mul_803: "f32[736]" = torch.ops.aten.mul.Tensor(sum_22, squeeze_235);  sum_22 = squeeze_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_15 = torch.ops.aten.convolution_backward.default(mul_802, div_81, primals_315, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 736, [True, True, False]);  mul_802 = div_81 = primals_315 = None
    getitem_219: "f32[8, 736, 7, 7]" = convolution_backward_15[0]
    getitem_220: "f32[736, 1, 5, 5]" = convolution_backward_15[1];  convolution_backward_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_13: "b8[8, 736, 7, 7]" = torch.ops.aten.lt.Scalar(clone_66, -3)
    le_10: "b8[8, 736, 7, 7]" = torch.ops.aten.le.Scalar(clone_66, 3)
    div_109: "f32[8, 736, 7, 7]" = torch.ops.aten.div.Tensor(clone_66, 3);  clone_66 = None
    add_567: "f32[8, 736, 7, 7]" = torch.ops.aten.add.Tensor(div_109, 0.5);  div_109 = None
    mul_804: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_219, add_567);  add_567 = None
    where_23: "f32[8, 736, 7, 7]" = torch.ops.aten.where.self(le_10, mul_804, getitem_219);  le_10 = mul_804 = getitem_219 = None
    where_24: "f32[8, 736, 7, 7]" = torch.ops.aten.where.self(lt_13, full_default, where_23);  lt_13 = where_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_23: "f32[736]" = torch.ops.aten.sum.dim_IntList(where_24, [0, 2, 3])
    sub_123: "f32[8, 736, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_107, unsqueeze_458);  convolution_107 = unsqueeze_458 = None
    mul_805: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(where_24, sub_123)
    sum_24: "f32[736]" = torch.ops.aten.sum.dim_IntList(mul_805, [0, 2, 3]);  mul_805 = None
    mul_806: "f32[736]" = torch.ops.aten.mul.Tensor(sum_23, 0.002551020408163265)
    unsqueeze_459: "f32[1, 736]" = torch.ops.aten.unsqueeze.default(mul_806, 0);  mul_806 = None
    unsqueeze_460: "f32[1, 736, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_459, 2);  unsqueeze_459 = None
    unsqueeze_461: "f32[1, 736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_460, 3);  unsqueeze_460 = None
    mul_807: "f32[736]" = torch.ops.aten.mul.Tensor(sum_24, 0.002551020408163265)
    mul_808: "f32[736]" = torch.ops.aten.mul.Tensor(squeeze_232, squeeze_232)
    mul_809: "f32[736]" = torch.ops.aten.mul.Tensor(mul_807, mul_808);  mul_807 = mul_808 = None
    unsqueeze_462: "f32[1, 736]" = torch.ops.aten.unsqueeze.default(mul_809, 0);  mul_809 = None
    unsqueeze_463: "f32[1, 736, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_462, 2);  unsqueeze_462 = None
    unsqueeze_464: "f32[1, 736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_463, 3);  unsqueeze_463 = None
    mul_810: "f32[736]" = torch.ops.aten.mul.Tensor(squeeze_232, primals_155);  primals_155 = None
    unsqueeze_465: "f32[1, 736]" = torch.ops.aten.unsqueeze.default(mul_810, 0);  mul_810 = None
    unsqueeze_466: "f32[1, 736, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_465, 2);  unsqueeze_465 = None
    unsqueeze_467: "f32[1, 736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_466, 3);  unsqueeze_466 = None
    mul_811: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(sub_123, unsqueeze_464);  sub_123 = unsqueeze_464 = None
    sub_125: "f32[8, 736, 7, 7]" = torch.ops.aten.sub.Tensor(where_24, mul_811);  where_24 = mul_811 = None
    sub_126: "f32[8, 736, 7, 7]" = torch.ops.aten.sub.Tensor(sub_125, unsqueeze_461);  sub_125 = unsqueeze_461 = None
    mul_812: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(sub_126, unsqueeze_467);  sub_126 = unsqueeze_467 = None
    mul_813: "f32[736]" = torch.ops.aten.mul.Tensor(sum_24, squeeze_232);  sum_24 = squeeze_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_16 = torch.ops.aten.convolution_backward.default(mul_812, add_486, primals_314, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_812 = add_486 = primals_314 = None
    getitem_222: "f32[8, 184, 7, 7]" = convolution_backward_16[0]
    getitem_223: "f32[736, 184, 1, 1]" = convolution_backward_16[1];  convolution_backward_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_568: "f32[8, 184, 7, 7]" = torch.ops.aten.add.Tensor(add_563, getitem_222);  add_563 = getitem_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_25: "f32[184]" = torch.ops.aten.sum.dim_IntList(add_568, [0, 2, 3])
    sub_127: "f32[8, 184, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_106, unsqueeze_470);  convolution_106 = unsqueeze_470 = None
    mul_814: "f32[8, 184, 7, 7]" = torch.ops.aten.mul.Tensor(add_568, sub_127)
    sum_26: "f32[184]" = torch.ops.aten.sum.dim_IntList(mul_814, [0, 2, 3]);  mul_814 = None
    mul_815: "f32[184]" = torch.ops.aten.mul.Tensor(sum_25, 0.002551020408163265)
    unsqueeze_471: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(mul_815, 0);  mul_815 = None
    unsqueeze_472: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_471, 2);  unsqueeze_471 = None
    unsqueeze_473: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_472, 3);  unsqueeze_472 = None
    mul_816: "f32[184]" = torch.ops.aten.mul.Tensor(sum_26, 0.002551020408163265)
    mul_817: "f32[184]" = torch.ops.aten.mul.Tensor(squeeze_229, squeeze_229)
    mul_818: "f32[184]" = torch.ops.aten.mul.Tensor(mul_816, mul_817);  mul_816 = mul_817 = None
    unsqueeze_474: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(mul_818, 0);  mul_818 = None
    unsqueeze_475: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_474, 2);  unsqueeze_474 = None
    unsqueeze_476: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_475, 3);  unsqueeze_475 = None
    mul_819: "f32[184]" = torch.ops.aten.mul.Tensor(squeeze_229, primals_153);  primals_153 = None
    unsqueeze_477: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(mul_819, 0);  mul_819 = None
    unsqueeze_478: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_477, 2);  unsqueeze_477 = None
    unsqueeze_479: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_478, 3);  unsqueeze_478 = None
    mul_820: "f32[8, 184, 7, 7]" = torch.ops.aten.mul.Tensor(sub_127, unsqueeze_476);  sub_127 = unsqueeze_476 = None
    sub_129: "f32[8, 184, 7, 7]" = torch.ops.aten.sub.Tensor(add_568, mul_820);  mul_820 = None
    sub_130: "f32[8, 184, 7, 7]" = torch.ops.aten.sub.Tensor(sub_129, unsqueeze_473);  sub_129 = unsqueeze_473 = None
    mul_821: "f32[8, 184, 7, 7]" = torch.ops.aten.mul.Tensor(sub_130, unsqueeze_479);  sub_130 = unsqueeze_479 = None
    mul_822: "f32[184]" = torch.ops.aten.mul.Tensor(sum_26, squeeze_229);  sum_26 = squeeze_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_17 = torch.ops.aten.convolution_backward.default(mul_821, mul_612, primals_313, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_821 = mul_612 = primals_313 = None
    getitem_225: "f32[8, 736, 7, 7]" = convolution_backward_17[0]
    getitem_226: "f32[184, 736, 1, 1]" = convolution_backward_17[1];  convolution_backward_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_823: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_225, div_78);  div_78 = None
    mul_824: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_225, div_80);  getitem_225 = div_80 = None
    sum_27: "f32[8, 736, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_823, [2, 3], True);  mul_823 = None
    mul_825: "f32[8, 736, 1, 1]" = torch.ops.aten.mul.Tensor(sum_27, 0.16666666666666666);  sum_27 = None
    where_25: "f32[8, 736, 1, 1]" = torch.ops.aten.where.self(bitwise_and_3, mul_825, full_default);  bitwise_and_3 = mul_825 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_18 = torch.ops.aten.convolution_backward.default(where_25, div_79, primals_311, [736], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_25 = div_79 = primals_311 = None
    getitem_228: "f32[8, 48, 1, 1]" = convolution_backward_18[0]
    getitem_229: "f32[736, 48, 1, 1]" = convolution_backward_18[1]
    getitem_230: "f32[736]" = convolution_backward_18[2];  convolution_backward_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    lt_15: "b8[8, 48, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_104, -3)
    le_11: "b8[8, 48, 1, 1]" = torch.ops.aten.le.Scalar(convolution_104, 3)
    div_110: "f32[8, 48, 1, 1]" = torch.ops.aten.div.Tensor(convolution_104, 3);  convolution_104 = None
    add_569: "f32[8, 48, 1, 1]" = torch.ops.aten.add.Tensor(div_110, 0.5);  div_110 = None
    mul_826: "f32[8, 48, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_228, add_569);  add_569 = None
    where_26: "f32[8, 48, 1, 1]" = torch.ops.aten.where.self(le_11, mul_826, getitem_228);  le_11 = mul_826 = getitem_228 = None
    where_27: "f32[8, 48, 1, 1]" = torch.ops.aten.where.self(lt_15, full_default, where_26);  lt_15 = where_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_19 = torch.ops.aten.convolution_backward.default(where_27, mean_14, primals_309, [48], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_27 = mean_14 = primals_309 = None
    getitem_231: "f32[8, 736, 1, 1]" = convolution_backward_19[0]
    getitem_232: "f32[48, 736, 1, 1]" = convolution_backward_19[1]
    getitem_233: "f32[48]" = convolution_backward_19[2];  convolution_backward_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_4: "f32[8, 736, 7, 7]" = torch.ops.aten.expand.default(getitem_231, [8, 736, 7, 7]);  getitem_231 = None
    div_111: "f32[8, 736, 7, 7]" = torch.ops.aten.div.Scalar(expand_4, 49);  expand_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_570: "f32[8, 736, 7, 7]" = torch.ops.aten.add.Tensor(mul_824, div_111);  mul_824 = div_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_16: "b8[8, 736, 7, 7]" = torch.ops.aten.lt.Scalar(clone_64, -3)
    le_12: "b8[8, 736, 7, 7]" = torch.ops.aten.le.Scalar(clone_64, 3)
    div_112: "f32[8, 736, 7, 7]" = torch.ops.aten.div.Tensor(clone_64, 3);  clone_64 = None
    add_571: "f32[8, 736, 7, 7]" = torch.ops.aten.add.Tensor(div_112, 0.5);  div_112 = None
    mul_827: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(add_570, add_571);  add_571 = None
    where_28: "f32[8, 736, 7, 7]" = torch.ops.aten.where.self(le_12, mul_827, add_570);  le_12 = mul_827 = add_570 = None
    where_29: "f32[8, 736, 7, 7]" = torch.ops.aten.where.self(lt_16, full_default, where_28);  lt_16 = where_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_28: "f32[736]" = torch.ops.aten.sum.dim_IntList(where_29, [0, 2, 3])
    sub_131: "f32[8, 736, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_103, unsqueeze_482);  convolution_103 = unsqueeze_482 = None
    mul_828: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(where_29, sub_131)
    sum_29: "f32[736]" = torch.ops.aten.sum.dim_IntList(mul_828, [0, 2, 3]);  mul_828 = None
    mul_829: "f32[736]" = torch.ops.aten.mul.Tensor(sum_28, 0.002551020408163265)
    unsqueeze_483: "f32[1, 736]" = torch.ops.aten.unsqueeze.default(mul_829, 0);  mul_829 = None
    unsqueeze_484: "f32[1, 736, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_483, 2);  unsqueeze_483 = None
    unsqueeze_485: "f32[1, 736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_484, 3);  unsqueeze_484 = None
    mul_830: "f32[736]" = torch.ops.aten.mul.Tensor(sum_29, 0.002551020408163265)
    mul_831: "f32[736]" = torch.ops.aten.mul.Tensor(squeeze_226, squeeze_226)
    mul_832: "f32[736]" = torch.ops.aten.mul.Tensor(mul_830, mul_831);  mul_830 = mul_831 = None
    unsqueeze_486: "f32[1, 736]" = torch.ops.aten.unsqueeze.default(mul_832, 0);  mul_832 = None
    unsqueeze_487: "f32[1, 736, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_486, 2);  unsqueeze_486 = None
    unsqueeze_488: "f32[1, 736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_487, 3);  unsqueeze_487 = None
    mul_833: "f32[736]" = torch.ops.aten.mul.Tensor(squeeze_226, primals_151);  primals_151 = None
    unsqueeze_489: "f32[1, 736]" = torch.ops.aten.unsqueeze.default(mul_833, 0);  mul_833 = None
    unsqueeze_490: "f32[1, 736, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_489, 2);  unsqueeze_489 = None
    unsqueeze_491: "f32[1, 736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_490, 3);  unsqueeze_490 = None
    mul_834: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(sub_131, unsqueeze_488);  sub_131 = unsqueeze_488 = None
    sub_133: "f32[8, 736, 7, 7]" = torch.ops.aten.sub.Tensor(where_29, mul_834);  where_29 = mul_834 = None
    sub_134: "f32[8, 736, 7, 7]" = torch.ops.aten.sub.Tensor(sub_133, unsqueeze_485);  sub_133 = unsqueeze_485 = None
    mul_835: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(sub_134, unsqueeze_491);  sub_134 = unsqueeze_491 = None
    mul_836: "f32[736]" = torch.ops.aten.mul.Tensor(sum_29, squeeze_226);  sum_29 = squeeze_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_20 = torch.ops.aten.convolution_backward.default(mul_835, div_77, primals_308, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 736, [True, True, False]);  mul_835 = div_77 = primals_308 = None
    getitem_234: "f32[8, 736, 7, 7]" = convolution_backward_20[0]
    getitem_235: "f32[736, 1, 5, 5]" = convolution_backward_20[1];  convolution_backward_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_17: "b8[8, 736, 7, 7]" = torch.ops.aten.lt.Scalar(clone_63, -3)
    le_13: "b8[8, 736, 7, 7]" = torch.ops.aten.le.Scalar(clone_63, 3)
    div_113: "f32[8, 736, 7, 7]" = torch.ops.aten.div.Tensor(clone_63, 3);  clone_63 = None
    add_572: "f32[8, 736, 7, 7]" = torch.ops.aten.add.Tensor(div_113, 0.5);  div_113 = None
    mul_837: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_234, add_572);  add_572 = None
    where_30: "f32[8, 736, 7, 7]" = torch.ops.aten.where.self(le_13, mul_837, getitem_234);  le_13 = mul_837 = getitem_234 = None
    where_31: "f32[8, 736, 7, 7]" = torch.ops.aten.where.self(lt_17, full_default, where_30);  lt_17 = where_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_30: "f32[736]" = torch.ops.aten.sum.dim_IntList(where_31, [0, 2, 3])
    sub_135: "f32[8, 736, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_102, unsqueeze_494);  convolution_102 = unsqueeze_494 = None
    mul_838: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(where_31, sub_135)
    sum_31: "f32[736]" = torch.ops.aten.sum.dim_IntList(mul_838, [0, 2, 3]);  mul_838 = None
    mul_839: "f32[736]" = torch.ops.aten.mul.Tensor(sum_30, 0.002551020408163265)
    unsqueeze_495: "f32[1, 736]" = torch.ops.aten.unsqueeze.default(mul_839, 0);  mul_839 = None
    unsqueeze_496: "f32[1, 736, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_495, 2);  unsqueeze_495 = None
    unsqueeze_497: "f32[1, 736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_496, 3);  unsqueeze_496 = None
    mul_840: "f32[736]" = torch.ops.aten.mul.Tensor(sum_31, 0.002551020408163265)
    mul_841: "f32[736]" = torch.ops.aten.mul.Tensor(squeeze_223, squeeze_223)
    mul_842: "f32[736]" = torch.ops.aten.mul.Tensor(mul_840, mul_841);  mul_840 = mul_841 = None
    unsqueeze_498: "f32[1, 736]" = torch.ops.aten.unsqueeze.default(mul_842, 0);  mul_842 = None
    unsqueeze_499: "f32[1, 736, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_498, 2);  unsqueeze_498 = None
    unsqueeze_500: "f32[1, 736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_499, 3);  unsqueeze_499 = None
    mul_843: "f32[736]" = torch.ops.aten.mul.Tensor(squeeze_223, primals_149);  primals_149 = None
    unsqueeze_501: "f32[1, 736]" = torch.ops.aten.unsqueeze.default(mul_843, 0);  mul_843 = None
    unsqueeze_502: "f32[1, 736, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_501, 2);  unsqueeze_501 = None
    unsqueeze_503: "f32[1, 736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_502, 3);  unsqueeze_502 = None
    mul_844: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(sub_135, unsqueeze_500);  sub_135 = unsqueeze_500 = None
    sub_137: "f32[8, 736, 7, 7]" = torch.ops.aten.sub.Tensor(where_31, mul_844);  where_31 = mul_844 = None
    sub_138: "f32[8, 736, 7, 7]" = torch.ops.aten.sub.Tensor(sub_137, unsqueeze_497);  sub_137 = unsqueeze_497 = None
    mul_845: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(sub_138, unsqueeze_503);  sub_138 = unsqueeze_503 = None
    mul_846: "f32[736]" = torch.ops.aten.mul.Tensor(sum_31, squeeze_223);  sum_31 = squeeze_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_21 = torch.ops.aten.convolution_backward.default(mul_845, add_466, primals_307, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_845 = add_466 = primals_307 = None
    getitem_237: "f32[8, 184, 7, 7]" = convolution_backward_21[0]
    getitem_238: "f32[736, 184, 1, 1]" = convolution_backward_21[1];  convolution_backward_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_573: "f32[8, 184, 7, 7]" = torch.ops.aten.add.Tensor(add_568, getitem_237);  add_568 = getitem_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_32: "f32[184]" = torch.ops.aten.sum.dim_IntList(add_573, [0, 2, 3])
    sub_139: "f32[8, 184, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_101, unsqueeze_506);  convolution_101 = unsqueeze_506 = None
    mul_847: "f32[8, 184, 7, 7]" = torch.ops.aten.mul.Tensor(add_573, sub_139)
    sum_33: "f32[184]" = torch.ops.aten.sum.dim_IntList(mul_847, [0, 2, 3]);  mul_847 = None
    mul_848: "f32[184]" = torch.ops.aten.mul.Tensor(sum_32, 0.002551020408163265)
    unsqueeze_507: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(mul_848, 0);  mul_848 = None
    unsqueeze_508: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_507, 2);  unsqueeze_507 = None
    unsqueeze_509: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_508, 3);  unsqueeze_508 = None
    mul_849: "f32[184]" = torch.ops.aten.mul.Tensor(sum_33, 0.002551020408163265)
    mul_850: "f32[184]" = torch.ops.aten.mul.Tensor(squeeze_220, squeeze_220)
    mul_851: "f32[184]" = torch.ops.aten.mul.Tensor(mul_849, mul_850);  mul_849 = mul_850 = None
    unsqueeze_510: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(mul_851, 0);  mul_851 = None
    unsqueeze_511: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_510, 2);  unsqueeze_510 = None
    unsqueeze_512: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_511, 3);  unsqueeze_511 = None
    mul_852: "f32[184]" = torch.ops.aten.mul.Tensor(squeeze_220, primals_147);  primals_147 = None
    unsqueeze_513: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(mul_852, 0);  mul_852 = None
    unsqueeze_514: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_513, 2);  unsqueeze_513 = None
    unsqueeze_515: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_514, 3);  unsqueeze_514 = None
    mul_853: "f32[8, 184, 7, 7]" = torch.ops.aten.mul.Tensor(sub_139, unsqueeze_512);  sub_139 = unsqueeze_512 = None
    sub_141: "f32[8, 184, 7, 7]" = torch.ops.aten.sub.Tensor(add_573, mul_853);  mul_853 = None
    sub_142: "f32[8, 184, 7, 7]" = torch.ops.aten.sub.Tensor(sub_141, unsqueeze_509);  sub_141 = unsqueeze_509 = None
    mul_854: "f32[8, 184, 7, 7]" = torch.ops.aten.mul.Tensor(sub_142, unsqueeze_515);  sub_142 = unsqueeze_515 = None
    mul_855: "f32[184]" = torch.ops.aten.mul.Tensor(sum_33, squeeze_220);  sum_33 = squeeze_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_22 = torch.ops.aten.convolution_backward.default(mul_854, mul_587, primals_306, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_854 = mul_587 = primals_306 = None
    getitem_240: "f32[8, 736, 7, 7]" = convolution_backward_22[0]
    getitem_241: "f32[184, 736, 1, 1]" = convolution_backward_22[1];  convolution_backward_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_856: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_240, div_74);  div_74 = None
    mul_857: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_240, div_76);  getitem_240 = div_76 = None
    sum_34: "f32[8, 736, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_856, [2, 3], True);  mul_856 = None
    mul_858: "f32[8, 736, 1, 1]" = torch.ops.aten.mul.Tensor(sum_34, 0.16666666666666666);  sum_34 = None
    where_32: "f32[8, 736, 1, 1]" = torch.ops.aten.where.self(bitwise_and_4, mul_858, full_default);  bitwise_and_4 = mul_858 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_23 = torch.ops.aten.convolution_backward.default(where_32, div_75, primals_304, [736], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_32 = div_75 = primals_304 = None
    getitem_243: "f32[8, 48, 1, 1]" = convolution_backward_23[0]
    getitem_244: "f32[736, 48, 1, 1]" = convolution_backward_23[1]
    getitem_245: "f32[736]" = convolution_backward_23[2];  convolution_backward_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    lt_19: "b8[8, 48, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_99, -3)
    le_14: "b8[8, 48, 1, 1]" = torch.ops.aten.le.Scalar(convolution_99, 3)
    div_114: "f32[8, 48, 1, 1]" = torch.ops.aten.div.Tensor(convolution_99, 3);  convolution_99 = None
    add_574: "f32[8, 48, 1, 1]" = torch.ops.aten.add.Tensor(div_114, 0.5);  div_114 = None
    mul_859: "f32[8, 48, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_243, add_574);  add_574 = None
    where_33: "f32[8, 48, 1, 1]" = torch.ops.aten.where.self(le_14, mul_859, getitem_243);  le_14 = mul_859 = getitem_243 = None
    where_34: "f32[8, 48, 1, 1]" = torch.ops.aten.where.self(lt_19, full_default, where_33);  lt_19 = where_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_24 = torch.ops.aten.convolution_backward.default(where_34, mean_13, primals_302, [48], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_34 = mean_13 = primals_302 = None
    getitem_246: "f32[8, 736, 1, 1]" = convolution_backward_24[0]
    getitem_247: "f32[48, 736, 1, 1]" = convolution_backward_24[1]
    getitem_248: "f32[48]" = convolution_backward_24[2];  convolution_backward_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_5: "f32[8, 736, 7, 7]" = torch.ops.aten.expand.default(getitem_246, [8, 736, 7, 7]);  getitem_246 = None
    div_115: "f32[8, 736, 7, 7]" = torch.ops.aten.div.Scalar(expand_5, 49);  expand_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_575: "f32[8, 736, 7, 7]" = torch.ops.aten.add.Tensor(mul_857, div_115);  mul_857 = div_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_20: "b8[8, 736, 7, 7]" = torch.ops.aten.lt.Scalar(clone_61, -3)
    le_15: "b8[8, 736, 7, 7]" = torch.ops.aten.le.Scalar(clone_61, 3)
    div_116: "f32[8, 736, 7, 7]" = torch.ops.aten.div.Tensor(clone_61, 3);  clone_61 = None
    add_576: "f32[8, 736, 7, 7]" = torch.ops.aten.add.Tensor(div_116, 0.5);  div_116 = None
    mul_860: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(add_575, add_576);  add_576 = None
    where_35: "f32[8, 736, 7, 7]" = torch.ops.aten.where.self(le_15, mul_860, add_575);  le_15 = mul_860 = add_575 = None
    where_36: "f32[8, 736, 7, 7]" = torch.ops.aten.where.self(lt_20, full_default, where_35);  lt_20 = where_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_35: "f32[736]" = torch.ops.aten.sum.dim_IntList(where_36, [0, 2, 3])
    sub_143: "f32[8, 736, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_98, unsqueeze_518);  convolution_98 = unsqueeze_518 = None
    mul_861: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(where_36, sub_143)
    sum_36: "f32[736]" = torch.ops.aten.sum.dim_IntList(mul_861, [0, 2, 3]);  mul_861 = None
    mul_862: "f32[736]" = torch.ops.aten.mul.Tensor(sum_35, 0.002551020408163265)
    unsqueeze_519: "f32[1, 736]" = torch.ops.aten.unsqueeze.default(mul_862, 0);  mul_862 = None
    unsqueeze_520: "f32[1, 736, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_519, 2);  unsqueeze_519 = None
    unsqueeze_521: "f32[1, 736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_520, 3);  unsqueeze_520 = None
    mul_863: "f32[736]" = torch.ops.aten.mul.Tensor(sum_36, 0.002551020408163265)
    mul_864: "f32[736]" = torch.ops.aten.mul.Tensor(squeeze_217, squeeze_217)
    mul_865: "f32[736]" = torch.ops.aten.mul.Tensor(mul_863, mul_864);  mul_863 = mul_864 = None
    unsqueeze_522: "f32[1, 736]" = torch.ops.aten.unsqueeze.default(mul_865, 0);  mul_865 = None
    unsqueeze_523: "f32[1, 736, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_522, 2);  unsqueeze_522 = None
    unsqueeze_524: "f32[1, 736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_523, 3);  unsqueeze_523 = None
    mul_866: "f32[736]" = torch.ops.aten.mul.Tensor(squeeze_217, primals_145);  primals_145 = None
    unsqueeze_525: "f32[1, 736]" = torch.ops.aten.unsqueeze.default(mul_866, 0);  mul_866 = None
    unsqueeze_526: "f32[1, 736, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_525, 2);  unsqueeze_525 = None
    unsqueeze_527: "f32[1, 736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_526, 3);  unsqueeze_526 = None
    mul_867: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(sub_143, unsqueeze_524);  sub_143 = unsqueeze_524 = None
    sub_145: "f32[8, 736, 7, 7]" = torch.ops.aten.sub.Tensor(where_36, mul_867);  where_36 = mul_867 = None
    sub_146: "f32[8, 736, 7, 7]" = torch.ops.aten.sub.Tensor(sub_145, unsqueeze_521);  sub_145 = unsqueeze_521 = None
    mul_868: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(sub_146, unsqueeze_527);  sub_146 = unsqueeze_527 = None
    mul_869: "f32[736]" = torch.ops.aten.mul.Tensor(sum_36, squeeze_217);  sum_36 = squeeze_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_25 = torch.ops.aten.convolution_backward.default(mul_868, div_73, primals_301, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 736, [True, True, False]);  mul_868 = div_73 = primals_301 = None
    getitem_249: "f32[8, 736, 7, 7]" = convolution_backward_25[0]
    getitem_250: "f32[736, 1, 5, 5]" = convolution_backward_25[1];  convolution_backward_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_21: "b8[8, 736, 7, 7]" = torch.ops.aten.lt.Scalar(clone_60, -3)
    le_16: "b8[8, 736, 7, 7]" = torch.ops.aten.le.Scalar(clone_60, 3)
    div_117: "f32[8, 736, 7, 7]" = torch.ops.aten.div.Tensor(clone_60, 3);  clone_60 = None
    add_577: "f32[8, 736, 7, 7]" = torch.ops.aten.add.Tensor(div_117, 0.5);  div_117 = None
    mul_870: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_249, add_577);  add_577 = None
    where_37: "f32[8, 736, 7, 7]" = torch.ops.aten.where.self(le_16, mul_870, getitem_249);  le_16 = mul_870 = getitem_249 = None
    where_38: "f32[8, 736, 7, 7]" = torch.ops.aten.where.self(lt_21, full_default, where_37);  lt_21 = where_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_37: "f32[736]" = torch.ops.aten.sum.dim_IntList(where_38, [0, 2, 3])
    sub_147: "f32[8, 736, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_97, unsqueeze_530);  convolution_97 = unsqueeze_530 = None
    mul_871: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(where_38, sub_147)
    sum_38: "f32[736]" = torch.ops.aten.sum.dim_IntList(mul_871, [0, 2, 3]);  mul_871 = None
    mul_872: "f32[736]" = torch.ops.aten.mul.Tensor(sum_37, 0.002551020408163265)
    unsqueeze_531: "f32[1, 736]" = torch.ops.aten.unsqueeze.default(mul_872, 0);  mul_872 = None
    unsqueeze_532: "f32[1, 736, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_531, 2);  unsqueeze_531 = None
    unsqueeze_533: "f32[1, 736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_532, 3);  unsqueeze_532 = None
    mul_873: "f32[736]" = torch.ops.aten.mul.Tensor(sum_38, 0.002551020408163265)
    mul_874: "f32[736]" = torch.ops.aten.mul.Tensor(squeeze_214, squeeze_214)
    mul_875: "f32[736]" = torch.ops.aten.mul.Tensor(mul_873, mul_874);  mul_873 = mul_874 = None
    unsqueeze_534: "f32[1, 736]" = torch.ops.aten.unsqueeze.default(mul_875, 0);  mul_875 = None
    unsqueeze_535: "f32[1, 736, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_534, 2);  unsqueeze_534 = None
    unsqueeze_536: "f32[1, 736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_535, 3);  unsqueeze_535 = None
    mul_876: "f32[736]" = torch.ops.aten.mul.Tensor(squeeze_214, primals_143);  primals_143 = None
    unsqueeze_537: "f32[1, 736]" = torch.ops.aten.unsqueeze.default(mul_876, 0);  mul_876 = None
    unsqueeze_538: "f32[1, 736, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_537, 2);  unsqueeze_537 = None
    unsqueeze_539: "f32[1, 736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_538, 3);  unsqueeze_538 = None
    mul_877: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(sub_147, unsqueeze_536);  sub_147 = unsqueeze_536 = None
    sub_149: "f32[8, 736, 7, 7]" = torch.ops.aten.sub.Tensor(where_38, mul_877);  where_38 = mul_877 = None
    sub_150: "f32[8, 736, 7, 7]" = torch.ops.aten.sub.Tensor(sub_149, unsqueeze_533);  sub_149 = unsqueeze_533 = None
    mul_878: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(sub_150, unsqueeze_539);  sub_150 = unsqueeze_539 = None
    mul_879: "f32[736]" = torch.ops.aten.mul.Tensor(sum_38, squeeze_214);  sum_38 = squeeze_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_26 = torch.ops.aten.convolution_backward.default(mul_878, add_446, primals_300, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_878 = add_446 = primals_300 = None
    getitem_252: "f32[8, 184, 7, 7]" = convolution_backward_26[0]
    getitem_253: "f32[736, 184, 1, 1]" = convolution_backward_26[1];  convolution_backward_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_578: "f32[8, 184, 7, 7]" = torch.ops.aten.add.Tensor(add_573, getitem_252);  add_573 = getitem_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_39: "f32[184]" = torch.ops.aten.sum.dim_IntList(add_578, [0, 2, 3])
    sub_151: "f32[8, 184, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_96, unsqueeze_542);  convolution_96 = unsqueeze_542 = None
    mul_880: "f32[8, 184, 7, 7]" = torch.ops.aten.mul.Tensor(add_578, sub_151)
    sum_40: "f32[184]" = torch.ops.aten.sum.dim_IntList(mul_880, [0, 2, 3]);  mul_880 = None
    mul_881: "f32[184]" = torch.ops.aten.mul.Tensor(sum_39, 0.002551020408163265)
    unsqueeze_543: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(mul_881, 0);  mul_881 = None
    unsqueeze_544: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_543, 2);  unsqueeze_543 = None
    unsqueeze_545: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_544, 3);  unsqueeze_544 = None
    mul_882: "f32[184]" = torch.ops.aten.mul.Tensor(sum_40, 0.002551020408163265)
    mul_883: "f32[184]" = torch.ops.aten.mul.Tensor(squeeze_211, squeeze_211)
    mul_884: "f32[184]" = torch.ops.aten.mul.Tensor(mul_882, mul_883);  mul_882 = mul_883 = None
    unsqueeze_546: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(mul_884, 0);  mul_884 = None
    unsqueeze_547: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_546, 2);  unsqueeze_546 = None
    unsqueeze_548: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_547, 3);  unsqueeze_547 = None
    mul_885: "f32[184]" = torch.ops.aten.mul.Tensor(squeeze_211, primals_141);  primals_141 = None
    unsqueeze_549: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(mul_885, 0);  mul_885 = None
    unsqueeze_550: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_549, 2);  unsqueeze_549 = None
    unsqueeze_551: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_550, 3);  unsqueeze_550 = None
    mul_886: "f32[8, 184, 7, 7]" = torch.ops.aten.mul.Tensor(sub_151, unsqueeze_548);  sub_151 = unsqueeze_548 = None
    sub_153: "f32[8, 184, 7, 7]" = torch.ops.aten.sub.Tensor(add_578, mul_886);  mul_886 = None
    sub_154: "f32[8, 184, 7, 7]" = torch.ops.aten.sub.Tensor(sub_153, unsqueeze_545);  sub_153 = unsqueeze_545 = None
    mul_887: "f32[8, 184, 7, 7]" = torch.ops.aten.mul.Tensor(sub_154, unsqueeze_551);  sub_154 = unsqueeze_551 = None
    mul_888: "f32[184]" = torch.ops.aten.mul.Tensor(sum_40, squeeze_211);  sum_40 = squeeze_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_27 = torch.ops.aten.convolution_backward.default(mul_887, mul_562, primals_299, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_887 = mul_562 = primals_299 = None
    getitem_255: "f32[8, 736, 7, 7]" = convolution_backward_27[0]
    getitem_256: "f32[184, 736, 1, 1]" = convolution_backward_27[1];  convolution_backward_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_889: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_255, div_70);  div_70 = None
    mul_890: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_255, div_72);  getitem_255 = div_72 = None
    sum_41: "f32[8, 736, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_889, [2, 3], True);  mul_889 = None
    mul_891: "f32[8, 736, 1, 1]" = torch.ops.aten.mul.Tensor(sum_41, 0.16666666666666666);  sum_41 = None
    where_39: "f32[8, 736, 1, 1]" = torch.ops.aten.where.self(bitwise_and_5, mul_891, full_default);  bitwise_and_5 = mul_891 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_28 = torch.ops.aten.convolution_backward.default(where_39, div_71, primals_297, [736], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_39 = div_71 = primals_297 = None
    getitem_258: "f32[8, 48, 1, 1]" = convolution_backward_28[0]
    getitem_259: "f32[736, 48, 1, 1]" = convolution_backward_28[1]
    getitem_260: "f32[736]" = convolution_backward_28[2];  convolution_backward_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    lt_23: "b8[8, 48, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_94, -3)
    le_17: "b8[8, 48, 1, 1]" = torch.ops.aten.le.Scalar(convolution_94, 3)
    div_118: "f32[8, 48, 1, 1]" = torch.ops.aten.div.Tensor(convolution_94, 3);  convolution_94 = None
    add_579: "f32[8, 48, 1, 1]" = torch.ops.aten.add.Tensor(div_118, 0.5);  div_118 = None
    mul_892: "f32[8, 48, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_258, add_579);  add_579 = None
    where_40: "f32[8, 48, 1, 1]" = torch.ops.aten.where.self(le_17, mul_892, getitem_258);  le_17 = mul_892 = getitem_258 = None
    where_41: "f32[8, 48, 1, 1]" = torch.ops.aten.where.self(lt_23, full_default, where_40);  lt_23 = where_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_29 = torch.ops.aten.convolution_backward.default(where_41, mean_12, primals_295, [48], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_41 = mean_12 = primals_295 = None
    getitem_261: "f32[8, 736, 1, 1]" = convolution_backward_29[0]
    getitem_262: "f32[48, 736, 1, 1]" = convolution_backward_29[1]
    getitem_263: "f32[48]" = convolution_backward_29[2];  convolution_backward_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_6: "f32[8, 736, 7, 7]" = torch.ops.aten.expand.default(getitem_261, [8, 736, 7, 7]);  getitem_261 = None
    div_119: "f32[8, 736, 7, 7]" = torch.ops.aten.div.Scalar(expand_6, 49);  expand_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_580: "f32[8, 736, 7, 7]" = torch.ops.aten.add.Tensor(mul_890, div_119);  mul_890 = div_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_24: "b8[8, 736, 7, 7]" = torch.ops.aten.lt.Scalar(clone_58, -3)
    le_18: "b8[8, 736, 7, 7]" = torch.ops.aten.le.Scalar(clone_58, 3)
    div_120: "f32[8, 736, 7, 7]" = torch.ops.aten.div.Tensor(clone_58, 3);  clone_58 = None
    add_581: "f32[8, 736, 7, 7]" = torch.ops.aten.add.Tensor(div_120, 0.5);  div_120 = None
    mul_893: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(add_580, add_581);  add_581 = None
    where_42: "f32[8, 736, 7, 7]" = torch.ops.aten.where.self(le_18, mul_893, add_580);  le_18 = mul_893 = add_580 = None
    where_43: "f32[8, 736, 7, 7]" = torch.ops.aten.where.self(lt_24, full_default, where_42);  lt_24 = where_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_42: "f32[736]" = torch.ops.aten.sum.dim_IntList(where_43, [0, 2, 3])
    sub_155: "f32[8, 736, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_93, unsqueeze_554);  convolution_93 = unsqueeze_554 = None
    mul_894: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(where_43, sub_155)
    sum_43: "f32[736]" = torch.ops.aten.sum.dim_IntList(mul_894, [0, 2, 3]);  mul_894 = None
    mul_895: "f32[736]" = torch.ops.aten.mul.Tensor(sum_42, 0.002551020408163265)
    unsqueeze_555: "f32[1, 736]" = torch.ops.aten.unsqueeze.default(mul_895, 0);  mul_895 = None
    unsqueeze_556: "f32[1, 736, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_555, 2);  unsqueeze_555 = None
    unsqueeze_557: "f32[1, 736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_556, 3);  unsqueeze_556 = None
    mul_896: "f32[736]" = torch.ops.aten.mul.Tensor(sum_43, 0.002551020408163265)
    mul_897: "f32[736]" = torch.ops.aten.mul.Tensor(squeeze_208, squeeze_208)
    mul_898: "f32[736]" = torch.ops.aten.mul.Tensor(mul_896, mul_897);  mul_896 = mul_897 = None
    unsqueeze_558: "f32[1, 736]" = torch.ops.aten.unsqueeze.default(mul_898, 0);  mul_898 = None
    unsqueeze_559: "f32[1, 736, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_558, 2);  unsqueeze_558 = None
    unsqueeze_560: "f32[1, 736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_559, 3);  unsqueeze_559 = None
    mul_899: "f32[736]" = torch.ops.aten.mul.Tensor(squeeze_208, primals_139);  primals_139 = None
    unsqueeze_561: "f32[1, 736]" = torch.ops.aten.unsqueeze.default(mul_899, 0);  mul_899 = None
    unsqueeze_562: "f32[1, 736, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_561, 2);  unsqueeze_561 = None
    unsqueeze_563: "f32[1, 736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_562, 3);  unsqueeze_562 = None
    mul_900: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(sub_155, unsqueeze_560);  sub_155 = unsqueeze_560 = None
    sub_157: "f32[8, 736, 7, 7]" = torch.ops.aten.sub.Tensor(where_43, mul_900);  where_43 = mul_900 = None
    sub_158: "f32[8, 736, 7, 7]" = torch.ops.aten.sub.Tensor(sub_157, unsqueeze_557);  sub_157 = unsqueeze_557 = None
    mul_901: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(sub_158, unsqueeze_563);  sub_158 = unsqueeze_563 = None
    mul_902: "f32[736]" = torch.ops.aten.mul.Tensor(sum_43, squeeze_208);  sum_43 = squeeze_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_30 = torch.ops.aten.convolution_backward.default(mul_901, div_69, primals_294, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 736, [True, True, False]);  mul_901 = div_69 = primals_294 = None
    getitem_264: "f32[8, 736, 7, 7]" = convolution_backward_30[0]
    getitem_265: "f32[736, 1, 5, 5]" = convolution_backward_30[1];  convolution_backward_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_25: "b8[8, 736, 7, 7]" = torch.ops.aten.lt.Scalar(clone_57, -3)
    le_19: "b8[8, 736, 7, 7]" = torch.ops.aten.le.Scalar(clone_57, 3)
    div_121: "f32[8, 736, 7, 7]" = torch.ops.aten.div.Tensor(clone_57, 3);  clone_57 = None
    add_582: "f32[8, 736, 7, 7]" = torch.ops.aten.add.Tensor(div_121, 0.5);  div_121 = None
    mul_903: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_264, add_582);  add_582 = None
    where_44: "f32[8, 736, 7, 7]" = torch.ops.aten.where.self(le_19, mul_903, getitem_264);  le_19 = mul_903 = getitem_264 = None
    where_45: "f32[8, 736, 7, 7]" = torch.ops.aten.where.self(lt_25, full_default, where_44);  lt_25 = where_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_44: "f32[736]" = torch.ops.aten.sum.dim_IntList(where_45, [0, 2, 3])
    sub_159: "f32[8, 736, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_92, unsqueeze_566);  convolution_92 = unsqueeze_566 = None
    mul_904: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(where_45, sub_159)
    sum_45: "f32[736]" = torch.ops.aten.sum.dim_IntList(mul_904, [0, 2, 3]);  mul_904 = None
    mul_905: "f32[736]" = torch.ops.aten.mul.Tensor(sum_44, 0.002551020408163265)
    unsqueeze_567: "f32[1, 736]" = torch.ops.aten.unsqueeze.default(mul_905, 0);  mul_905 = None
    unsqueeze_568: "f32[1, 736, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_567, 2);  unsqueeze_567 = None
    unsqueeze_569: "f32[1, 736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_568, 3);  unsqueeze_568 = None
    mul_906: "f32[736]" = torch.ops.aten.mul.Tensor(sum_45, 0.002551020408163265)
    mul_907: "f32[736]" = torch.ops.aten.mul.Tensor(squeeze_205, squeeze_205)
    mul_908: "f32[736]" = torch.ops.aten.mul.Tensor(mul_906, mul_907);  mul_906 = mul_907 = None
    unsqueeze_570: "f32[1, 736]" = torch.ops.aten.unsqueeze.default(mul_908, 0);  mul_908 = None
    unsqueeze_571: "f32[1, 736, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_570, 2);  unsqueeze_570 = None
    unsqueeze_572: "f32[1, 736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_571, 3);  unsqueeze_571 = None
    mul_909: "f32[736]" = torch.ops.aten.mul.Tensor(squeeze_205, primals_137);  primals_137 = None
    unsqueeze_573: "f32[1, 736]" = torch.ops.aten.unsqueeze.default(mul_909, 0);  mul_909 = None
    unsqueeze_574: "f32[1, 736, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_573, 2);  unsqueeze_573 = None
    unsqueeze_575: "f32[1, 736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_574, 3);  unsqueeze_574 = None
    mul_910: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(sub_159, unsqueeze_572);  sub_159 = unsqueeze_572 = None
    sub_161: "f32[8, 736, 7, 7]" = torch.ops.aten.sub.Tensor(where_45, mul_910);  where_45 = mul_910 = None
    sub_162: "f32[8, 736, 7, 7]" = torch.ops.aten.sub.Tensor(sub_161, unsqueeze_569);  sub_161 = unsqueeze_569 = None
    mul_911: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(sub_162, unsqueeze_575);  sub_162 = unsqueeze_575 = None
    mul_912: "f32[736]" = torch.ops.aten.mul.Tensor(sum_45, squeeze_205);  sum_45 = squeeze_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_31 = torch.ops.aten.convolution_backward.default(mul_911, add_426, primals_293, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_911 = add_426 = primals_293 = None
    getitem_267: "f32[8, 184, 7, 7]" = convolution_backward_31[0]
    getitem_268: "f32[736, 184, 1, 1]" = convolution_backward_31[1];  convolution_backward_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_583: "f32[8, 184, 7, 7]" = torch.ops.aten.add.Tensor(add_578, getitem_267);  add_578 = getitem_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_46: "f32[184]" = torch.ops.aten.sum.dim_IntList(add_583, [0, 2, 3])
    sub_163: "f32[8, 184, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_91, unsqueeze_578);  convolution_91 = unsqueeze_578 = None
    mul_913: "f32[8, 184, 7, 7]" = torch.ops.aten.mul.Tensor(add_583, sub_163)
    sum_47: "f32[184]" = torch.ops.aten.sum.dim_IntList(mul_913, [0, 2, 3]);  mul_913 = None
    mul_914: "f32[184]" = torch.ops.aten.mul.Tensor(sum_46, 0.002551020408163265)
    unsqueeze_579: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(mul_914, 0);  mul_914 = None
    unsqueeze_580: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_579, 2);  unsqueeze_579 = None
    unsqueeze_581: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_580, 3);  unsqueeze_580 = None
    mul_915: "f32[184]" = torch.ops.aten.mul.Tensor(sum_47, 0.002551020408163265)
    mul_916: "f32[184]" = torch.ops.aten.mul.Tensor(squeeze_202, squeeze_202)
    mul_917: "f32[184]" = torch.ops.aten.mul.Tensor(mul_915, mul_916);  mul_915 = mul_916 = None
    unsqueeze_582: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(mul_917, 0);  mul_917 = None
    unsqueeze_583: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_582, 2);  unsqueeze_582 = None
    unsqueeze_584: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_583, 3);  unsqueeze_583 = None
    mul_918: "f32[184]" = torch.ops.aten.mul.Tensor(squeeze_202, primals_135);  primals_135 = None
    unsqueeze_585: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(mul_918, 0);  mul_918 = None
    unsqueeze_586: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_585, 2);  unsqueeze_585 = None
    unsqueeze_587: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_586, 3);  unsqueeze_586 = None
    mul_919: "f32[8, 184, 7, 7]" = torch.ops.aten.mul.Tensor(sub_163, unsqueeze_584);  sub_163 = unsqueeze_584 = None
    sub_165: "f32[8, 184, 7, 7]" = torch.ops.aten.sub.Tensor(add_583, mul_919);  add_583 = mul_919 = None
    sub_166: "f32[8, 184, 7, 7]" = torch.ops.aten.sub.Tensor(sub_165, unsqueeze_581);  sub_165 = unsqueeze_581 = None
    mul_920: "f32[8, 184, 7, 7]" = torch.ops.aten.mul.Tensor(sub_166, unsqueeze_587);  sub_166 = unsqueeze_587 = None
    mul_921: "f32[184]" = torch.ops.aten.mul.Tensor(sum_47, squeeze_202);  sum_47 = squeeze_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_32 = torch.ops.aten.convolution_backward.default(mul_920, mul_537, primals_292, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_920 = mul_537 = primals_292 = None
    getitem_270: "f32[8, 720, 7, 7]" = convolution_backward_32[0]
    getitem_271: "f32[184, 720, 1, 1]" = convolution_backward_32[1];  convolution_backward_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_922: "f32[8, 720, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_270, div_66);  div_66 = None
    mul_923: "f32[8, 720, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_270, div_68);  getitem_270 = div_68 = None
    sum_48: "f32[8, 720, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_922, [2, 3], True);  mul_922 = None
    mul_924: "f32[8, 720, 1, 1]" = torch.ops.aten.mul.Tensor(sum_48, 0.16666666666666666);  sum_48 = None
    where_46: "f32[8, 720, 1, 1]" = torch.ops.aten.where.self(bitwise_and_6, mul_924, full_default);  bitwise_and_6 = mul_924 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_33 = torch.ops.aten.convolution_backward.default(where_46, div_67, primals_290, [720], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_46 = div_67 = primals_290 = None
    getitem_273: "f32[8, 32, 1, 1]" = convolution_backward_33[0]
    getitem_274: "f32[720, 32, 1, 1]" = convolution_backward_33[1]
    getitem_275: "f32[720]" = convolution_backward_33[2];  convolution_backward_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    lt_27: "b8[8, 32, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_89, -3)
    le_20: "b8[8, 32, 1, 1]" = torch.ops.aten.le.Scalar(convolution_89, 3)
    div_122: "f32[8, 32, 1, 1]" = torch.ops.aten.div.Tensor(convolution_89, 3);  convolution_89 = None
    add_584: "f32[8, 32, 1, 1]" = torch.ops.aten.add.Tensor(div_122, 0.5);  div_122 = None
    mul_925: "f32[8, 32, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_273, add_584);  add_584 = None
    where_47: "f32[8, 32, 1, 1]" = torch.ops.aten.where.self(le_20, mul_925, getitem_273);  le_20 = mul_925 = getitem_273 = None
    where_48: "f32[8, 32, 1, 1]" = torch.ops.aten.where.self(lt_27, full_default, where_47);  lt_27 = where_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_34 = torch.ops.aten.convolution_backward.default(where_48, mean_11, primals_288, [32], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_48 = mean_11 = primals_288 = None
    getitem_276: "f32[8, 720, 1, 1]" = convolution_backward_34[0]
    getitem_277: "f32[32, 720, 1, 1]" = convolution_backward_34[1]
    getitem_278: "f32[32]" = convolution_backward_34[2];  convolution_backward_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_7: "f32[8, 720, 7, 7]" = torch.ops.aten.expand.default(getitem_276, [8, 720, 7, 7]);  getitem_276 = None
    div_123: "f32[8, 720, 7, 7]" = torch.ops.aten.div.Scalar(expand_7, 49);  expand_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_585: "f32[8, 720, 7, 7]" = torch.ops.aten.add.Tensor(mul_923, div_123);  mul_923 = div_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_28: "b8[8, 720, 7, 7]" = torch.ops.aten.lt.Scalar(clone_55, -3)
    le_21: "b8[8, 720, 7, 7]" = torch.ops.aten.le.Scalar(clone_55, 3)
    div_124: "f32[8, 720, 7, 7]" = torch.ops.aten.div.Tensor(clone_55, 3);  clone_55 = None
    add_586: "f32[8, 720, 7, 7]" = torch.ops.aten.add.Tensor(div_124, 0.5);  div_124 = None
    mul_926: "f32[8, 720, 7, 7]" = torch.ops.aten.mul.Tensor(add_585, add_586);  add_586 = None
    where_49: "f32[8, 720, 7, 7]" = torch.ops.aten.where.self(le_21, mul_926, add_585);  le_21 = mul_926 = add_585 = None
    where_50: "f32[8, 720, 7, 7]" = torch.ops.aten.where.self(lt_28, full_default, where_49);  lt_28 = where_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_49: "f32[720]" = torch.ops.aten.sum.dim_IntList(where_50, [0, 2, 3])
    sub_167: "f32[8, 720, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_88, unsqueeze_590);  convolution_88 = unsqueeze_590 = None
    mul_927: "f32[8, 720, 7, 7]" = torch.ops.aten.mul.Tensor(where_50, sub_167)
    sum_50: "f32[720]" = torch.ops.aten.sum.dim_IntList(mul_927, [0, 2, 3]);  mul_927 = None
    mul_928: "f32[720]" = torch.ops.aten.mul.Tensor(sum_49, 0.002551020408163265)
    unsqueeze_591: "f32[1, 720]" = torch.ops.aten.unsqueeze.default(mul_928, 0);  mul_928 = None
    unsqueeze_592: "f32[1, 720, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_591, 2);  unsqueeze_591 = None
    unsqueeze_593: "f32[1, 720, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_592, 3);  unsqueeze_592 = None
    mul_929: "f32[720]" = torch.ops.aten.mul.Tensor(sum_50, 0.002551020408163265)
    mul_930: "f32[720]" = torch.ops.aten.mul.Tensor(squeeze_199, squeeze_199)
    mul_931: "f32[720]" = torch.ops.aten.mul.Tensor(mul_929, mul_930);  mul_929 = mul_930 = None
    unsqueeze_594: "f32[1, 720]" = torch.ops.aten.unsqueeze.default(mul_931, 0);  mul_931 = None
    unsqueeze_595: "f32[1, 720, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_594, 2);  unsqueeze_594 = None
    unsqueeze_596: "f32[1, 720, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_595, 3);  unsqueeze_595 = None
    mul_932: "f32[720]" = torch.ops.aten.mul.Tensor(squeeze_199, primals_133);  primals_133 = None
    unsqueeze_597: "f32[1, 720]" = torch.ops.aten.unsqueeze.default(mul_932, 0);  mul_932 = None
    unsqueeze_598: "f32[1, 720, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_597, 2);  unsqueeze_597 = None
    unsqueeze_599: "f32[1, 720, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_598, 3);  unsqueeze_598 = None
    mul_933: "f32[8, 720, 7, 7]" = torch.ops.aten.mul.Tensor(sub_167, unsqueeze_596);  sub_167 = unsqueeze_596 = None
    sub_169: "f32[8, 720, 7, 7]" = torch.ops.aten.sub.Tensor(where_50, mul_933);  where_50 = mul_933 = None
    sub_170: "f32[8, 720, 7, 7]" = torch.ops.aten.sub.Tensor(sub_169, unsqueeze_593);  sub_169 = unsqueeze_593 = None
    mul_934: "f32[8, 720, 7, 7]" = torch.ops.aten.mul.Tensor(sub_170, unsqueeze_599);  sub_170 = unsqueeze_599 = None
    mul_935: "f32[720]" = torch.ops.aten.mul.Tensor(sum_50, squeeze_199);  sum_50 = squeeze_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_35 = torch.ops.aten.convolution_backward.default(mul_934, div_65, primals_287, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 720, [True, True, False]);  mul_934 = div_65 = primals_287 = None
    getitem_279: "f32[8, 720, 14, 14]" = convolution_backward_35[0]
    getitem_280: "f32[720, 1, 3, 3]" = convolution_backward_35[1];  convolution_backward_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_29: "b8[8, 720, 14, 14]" = torch.ops.aten.lt.Scalar(clone_54, -3)
    le_22: "b8[8, 720, 14, 14]" = torch.ops.aten.le.Scalar(clone_54, 3)
    div_125: "f32[8, 720, 14, 14]" = torch.ops.aten.div.Tensor(clone_54, 3);  clone_54 = None
    add_587: "f32[8, 720, 14, 14]" = torch.ops.aten.add.Tensor(div_125, 0.5);  div_125 = None
    mul_936: "f32[8, 720, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_279, add_587);  add_587 = None
    where_51: "f32[8, 720, 14, 14]" = torch.ops.aten.where.self(le_22, mul_936, getitem_279);  le_22 = mul_936 = getitem_279 = None
    where_52: "f32[8, 720, 14, 14]" = torch.ops.aten.where.self(lt_29, full_default, where_51);  lt_29 = where_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_51: "f32[720]" = torch.ops.aten.sum.dim_IntList(where_52, [0, 2, 3])
    sub_171: "f32[8, 720, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_87, unsqueeze_602);  convolution_87 = unsqueeze_602 = None
    mul_937: "f32[8, 720, 14, 14]" = torch.ops.aten.mul.Tensor(where_52, sub_171)
    sum_52: "f32[720]" = torch.ops.aten.sum.dim_IntList(mul_937, [0, 2, 3]);  mul_937 = None
    mul_938: "f32[720]" = torch.ops.aten.mul.Tensor(sum_51, 0.0006377551020408163)
    unsqueeze_603: "f32[1, 720]" = torch.ops.aten.unsqueeze.default(mul_938, 0);  mul_938 = None
    unsqueeze_604: "f32[1, 720, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_603, 2);  unsqueeze_603 = None
    unsqueeze_605: "f32[1, 720, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_604, 3);  unsqueeze_604 = None
    mul_939: "f32[720]" = torch.ops.aten.mul.Tensor(sum_52, 0.0006377551020408163)
    mul_940: "f32[720]" = torch.ops.aten.mul.Tensor(squeeze_196, squeeze_196)
    mul_941: "f32[720]" = torch.ops.aten.mul.Tensor(mul_939, mul_940);  mul_939 = mul_940 = None
    unsqueeze_606: "f32[1, 720]" = torch.ops.aten.unsqueeze.default(mul_941, 0);  mul_941 = None
    unsqueeze_607: "f32[1, 720, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_606, 2);  unsqueeze_606 = None
    unsqueeze_608: "f32[1, 720, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_607, 3);  unsqueeze_607 = None
    mul_942: "f32[720]" = torch.ops.aten.mul.Tensor(squeeze_196, primals_131);  primals_131 = None
    unsqueeze_609: "f32[1, 720]" = torch.ops.aten.unsqueeze.default(mul_942, 0);  mul_942 = None
    unsqueeze_610: "f32[1, 720, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_609, 2);  unsqueeze_609 = None
    unsqueeze_611: "f32[1, 720, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_610, 3);  unsqueeze_610 = None
    mul_943: "f32[8, 720, 14, 14]" = torch.ops.aten.mul.Tensor(sub_171, unsqueeze_608);  sub_171 = unsqueeze_608 = None
    sub_173: "f32[8, 720, 14, 14]" = torch.ops.aten.sub.Tensor(where_52, mul_943);  where_52 = mul_943 = None
    sub_174: "f32[8, 720, 14, 14]" = torch.ops.aten.sub.Tensor(sub_173, unsqueeze_605);  sub_173 = unsqueeze_605 = None
    mul_944: "f32[8, 720, 14, 14]" = torch.ops.aten.mul.Tensor(sub_174, unsqueeze_611);  sub_174 = unsqueeze_611 = None
    mul_945: "f32[720]" = torch.ops.aten.mul.Tensor(sum_52, squeeze_196);  sum_52 = squeeze_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_36 = torch.ops.aten.convolution_backward.default(mul_944, add_407, primals_286, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_944 = add_407 = primals_286 = None
    getitem_282: "f32[8, 120, 14, 14]" = convolution_backward_36[0]
    getitem_283: "f32[720, 120, 1, 1]" = convolution_backward_36[1];  convolution_backward_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_53: "f32[120]" = torch.ops.aten.sum.dim_IntList(getitem_282, [0, 2, 3])
    sub_175: "f32[8, 120, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_86, unsqueeze_614);  convolution_86 = unsqueeze_614 = None
    mul_946: "f32[8, 120, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_282, sub_175)
    sum_54: "f32[120]" = torch.ops.aten.sum.dim_IntList(mul_946, [0, 2, 3]);  mul_946 = None
    mul_947: "f32[120]" = torch.ops.aten.mul.Tensor(sum_53, 0.0006377551020408163)
    unsqueeze_615: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_947, 0);  mul_947 = None
    unsqueeze_616: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_615, 2);  unsqueeze_615 = None
    unsqueeze_617: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_616, 3);  unsqueeze_616 = None
    mul_948: "f32[120]" = torch.ops.aten.mul.Tensor(sum_54, 0.0006377551020408163)
    mul_949: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_193, squeeze_193)
    mul_950: "f32[120]" = torch.ops.aten.mul.Tensor(mul_948, mul_949);  mul_948 = mul_949 = None
    unsqueeze_618: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_950, 0);  mul_950 = None
    unsqueeze_619: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_618, 2);  unsqueeze_618 = None
    unsqueeze_620: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_619, 3);  unsqueeze_619 = None
    mul_951: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_193, primals_129);  primals_129 = None
    unsqueeze_621: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_951, 0);  mul_951 = None
    unsqueeze_622: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_621, 2);  unsqueeze_621 = None
    unsqueeze_623: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_622, 3);  unsqueeze_622 = None
    mul_952: "f32[8, 120, 14, 14]" = torch.ops.aten.mul.Tensor(sub_175, unsqueeze_620);  sub_175 = unsqueeze_620 = None
    sub_177: "f32[8, 120, 14, 14]" = torch.ops.aten.sub.Tensor(getitem_282, mul_952);  mul_952 = None
    sub_178: "f32[8, 120, 14, 14]" = torch.ops.aten.sub.Tensor(sub_177, unsqueeze_617);  sub_177 = unsqueeze_617 = None
    mul_953: "f32[8, 120, 14, 14]" = torch.ops.aten.mul.Tensor(sub_178, unsqueeze_623);  sub_178 = unsqueeze_623 = None
    mul_954: "f32[120]" = torch.ops.aten.mul.Tensor(sum_54, squeeze_193);  sum_54 = squeeze_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_37 = torch.ops.aten.convolution_backward.default(mul_953, mul_512, primals_285, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_953 = mul_512 = primals_285 = None
    getitem_285: "f32[8, 360, 14, 14]" = convolution_backward_37[0]
    getitem_286: "f32[120, 360, 1, 1]" = convolution_backward_37[1];  convolution_backward_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_955: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_285, div_62);  div_62 = None
    mul_956: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_285, div_64);  getitem_285 = div_64 = None
    sum_55: "f32[8, 360, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_955, [2, 3], True);  mul_955 = None
    mul_957: "f32[8, 360, 1, 1]" = torch.ops.aten.mul.Tensor(sum_55, 0.16666666666666666);  sum_55 = None
    where_53: "f32[8, 360, 1, 1]" = torch.ops.aten.where.self(bitwise_and_7, mul_957, full_default);  bitwise_and_7 = mul_957 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_38 = torch.ops.aten.convolution_backward.default(where_53, div_63, primals_283, [360], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_53 = div_63 = primals_283 = None
    getitem_288: "f32[8, 32, 1, 1]" = convolution_backward_38[0]
    getitem_289: "f32[360, 32, 1, 1]" = convolution_backward_38[1]
    getitem_290: "f32[360]" = convolution_backward_38[2];  convolution_backward_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    lt_31: "b8[8, 32, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_84, -3)
    le_23: "b8[8, 32, 1, 1]" = torch.ops.aten.le.Scalar(convolution_84, 3)
    div_126: "f32[8, 32, 1, 1]" = torch.ops.aten.div.Tensor(convolution_84, 3);  convolution_84 = None
    add_588: "f32[8, 32, 1, 1]" = torch.ops.aten.add.Tensor(div_126, 0.5);  div_126 = None
    mul_958: "f32[8, 32, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_288, add_588);  add_588 = None
    where_54: "f32[8, 32, 1, 1]" = torch.ops.aten.where.self(le_23, mul_958, getitem_288);  le_23 = mul_958 = getitem_288 = None
    where_55: "f32[8, 32, 1, 1]" = torch.ops.aten.where.self(lt_31, full_default, where_54);  lt_31 = where_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_39 = torch.ops.aten.convolution_backward.default(where_55, mean_10, primals_281, [32], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_55 = mean_10 = primals_281 = None
    getitem_291: "f32[8, 360, 1, 1]" = convolution_backward_39[0]
    getitem_292: "f32[32, 360, 1, 1]" = convolution_backward_39[1]
    getitem_293: "f32[32]" = convolution_backward_39[2];  convolution_backward_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_8: "f32[8, 360, 14, 14]" = torch.ops.aten.expand.default(getitem_291, [8, 360, 14, 14]);  getitem_291 = None
    div_127: "f32[8, 360, 14, 14]" = torch.ops.aten.div.Scalar(expand_8, 196);  expand_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_589: "f32[8, 360, 14, 14]" = torch.ops.aten.add.Tensor(mul_956, div_127);  mul_956 = div_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_32: "b8[8, 360, 14, 14]" = torch.ops.aten.lt.Scalar(clone_52, -3)
    le_24: "b8[8, 360, 14, 14]" = torch.ops.aten.le.Scalar(clone_52, 3)
    div_128: "f32[8, 360, 14, 14]" = torch.ops.aten.div.Tensor(clone_52, 3);  clone_52 = None
    add_590: "f32[8, 360, 14, 14]" = torch.ops.aten.add.Tensor(div_128, 0.5);  div_128 = None
    mul_959: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(add_589, add_590);  add_590 = None
    where_56: "f32[8, 360, 14, 14]" = torch.ops.aten.where.self(le_24, mul_959, add_589);  le_24 = mul_959 = add_589 = None
    where_57: "f32[8, 360, 14, 14]" = torch.ops.aten.where.self(lt_32, full_default, where_56);  lt_32 = where_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_56: "f32[360]" = torch.ops.aten.sum.dim_IntList(where_57, [0, 2, 3])
    sub_179: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_83, unsqueeze_626);  convolution_83 = unsqueeze_626 = None
    mul_960: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(where_57, sub_179)
    sum_57: "f32[360]" = torch.ops.aten.sum.dim_IntList(mul_960, [0, 2, 3]);  mul_960 = None
    mul_961: "f32[360]" = torch.ops.aten.mul.Tensor(sum_56, 0.0006377551020408163)
    unsqueeze_627: "f32[1, 360]" = torch.ops.aten.unsqueeze.default(mul_961, 0);  mul_961 = None
    unsqueeze_628: "f32[1, 360, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_627, 2);  unsqueeze_627 = None
    unsqueeze_629: "f32[1, 360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_628, 3);  unsqueeze_628 = None
    mul_962: "f32[360]" = torch.ops.aten.mul.Tensor(sum_57, 0.0006377551020408163)
    mul_963: "f32[360]" = torch.ops.aten.mul.Tensor(squeeze_190, squeeze_190)
    mul_964: "f32[360]" = torch.ops.aten.mul.Tensor(mul_962, mul_963);  mul_962 = mul_963 = None
    unsqueeze_630: "f32[1, 360]" = torch.ops.aten.unsqueeze.default(mul_964, 0);  mul_964 = None
    unsqueeze_631: "f32[1, 360, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_630, 2);  unsqueeze_630 = None
    unsqueeze_632: "f32[1, 360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_631, 3);  unsqueeze_631 = None
    mul_965: "f32[360]" = torch.ops.aten.mul.Tensor(squeeze_190, primals_127);  primals_127 = None
    unsqueeze_633: "f32[1, 360]" = torch.ops.aten.unsqueeze.default(mul_965, 0);  mul_965 = None
    unsqueeze_634: "f32[1, 360, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_633, 2);  unsqueeze_633 = None
    unsqueeze_635: "f32[1, 360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_634, 3);  unsqueeze_634 = None
    mul_966: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(sub_179, unsqueeze_632);  sub_179 = unsqueeze_632 = None
    sub_181: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(where_57, mul_966);  where_57 = mul_966 = None
    sub_182: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(sub_181, unsqueeze_629);  sub_181 = unsqueeze_629 = None
    mul_967: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(sub_182, unsqueeze_635);  sub_182 = unsqueeze_635 = None
    mul_968: "f32[360]" = torch.ops.aten.mul.Tensor(sum_57, squeeze_190);  sum_57 = squeeze_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_40 = torch.ops.aten.convolution_backward.default(mul_967, div_61, primals_280, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 360, [True, True, False]);  mul_967 = div_61 = primals_280 = None
    getitem_294: "f32[8, 360, 14, 14]" = convolution_backward_40[0]
    getitem_295: "f32[360, 1, 5, 5]" = convolution_backward_40[1];  convolution_backward_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_33: "b8[8, 360, 14, 14]" = torch.ops.aten.lt.Scalar(clone_51, -3)
    le_25: "b8[8, 360, 14, 14]" = torch.ops.aten.le.Scalar(clone_51, 3)
    div_129: "f32[8, 360, 14, 14]" = torch.ops.aten.div.Tensor(clone_51, 3);  clone_51 = None
    add_591: "f32[8, 360, 14, 14]" = torch.ops.aten.add.Tensor(div_129, 0.5);  div_129 = None
    mul_969: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_294, add_591);  add_591 = None
    where_58: "f32[8, 360, 14, 14]" = torch.ops.aten.where.self(le_25, mul_969, getitem_294);  le_25 = mul_969 = getitem_294 = None
    where_59: "f32[8, 360, 14, 14]" = torch.ops.aten.where.self(lt_33, full_default, where_58);  lt_33 = where_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_58: "f32[360]" = torch.ops.aten.sum.dim_IntList(where_59, [0, 2, 3])
    sub_183: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_82, unsqueeze_638);  convolution_82 = unsqueeze_638 = None
    mul_970: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(where_59, sub_183)
    sum_59: "f32[360]" = torch.ops.aten.sum.dim_IntList(mul_970, [0, 2, 3]);  mul_970 = None
    mul_971: "f32[360]" = torch.ops.aten.mul.Tensor(sum_58, 0.0006377551020408163)
    unsqueeze_639: "f32[1, 360]" = torch.ops.aten.unsqueeze.default(mul_971, 0);  mul_971 = None
    unsqueeze_640: "f32[1, 360, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_639, 2);  unsqueeze_639 = None
    unsqueeze_641: "f32[1, 360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_640, 3);  unsqueeze_640 = None
    mul_972: "f32[360]" = torch.ops.aten.mul.Tensor(sum_59, 0.0006377551020408163)
    mul_973: "f32[360]" = torch.ops.aten.mul.Tensor(squeeze_187, squeeze_187)
    mul_974: "f32[360]" = torch.ops.aten.mul.Tensor(mul_972, mul_973);  mul_972 = mul_973 = None
    unsqueeze_642: "f32[1, 360]" = torch.ops.aten.unsqueeze.default(mul_974, 0);  mul_974 = None
    unsqueeze_643: "f32[1, 360, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_642, 2);  unsqueeze_642 = None
    unsqueeze_644: "f32[1, 360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_643, 3);  unsqueeze_643 = None
    mul_975: "f32[360]" = torch.ops.aten.mul.Tensor(squeeze_187, primals_125);  primals_125 = None
    unsqueeze_645: "f32[1, 360]" = torch.ops.aten.unsqueeze.default(mul_975, 0);  mul_975 = None
    unsqueeze_646: "f32[1, 360, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_645, 2);  unsqueeze_645 = None
    unsqueeze_647: "f32[1, 360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_646, 3);  unsqueeze_646 = None
    mul_976: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(sub_183, unsqueeze_644);  sub_183 = unsqueeze_644 = None
    sub_185: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(where_59, mul_976);  where_59 = mul_976 = None
    sub_186: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(sub_185, unsqueeze_641);  sub_185 = unsqueeze_641 = None
    mul_977: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(sub_186, unsqueeze_647);  sub_186 = unsqueeze_647 = None
    mul_978: "f32[360]" = torch.ops.aten.mul.Tensor(sum_59, squeeze_187);  sum_59 = squeeze_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_41 = torch.ops.aten.convolution_backward.default(mul_977, add_387, primals_279, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_977 = add_387 = primals_279 = None
    getitem_297: "f32[8, 120, 14, 14]" = convolution_backward_41[0]
    getitem_298: "f32[360, 120, 1, 1]" = convolution_backward_41[1];  convolution_backward_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_592: "f32[8, 120, 14, 14]" = torch.ops.aten.add.Tensor(getitem_282, getitem_297);  getitem_282 = getitem_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_60: "f32[120]" = torch.ops.aten.sum.dim_IntList(add_592, [0, 2, 3])
    sub_187: "f32[8, 120, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_81, unsqueeze_650);  convolution_81 = unsqueeze_650 = None
    mul_979: "f32[8, 120, 14, 14]" = torch.ops.aten.mul.Tensor(add_592, sub_187)
    sum_61: "f32[120]" = torch.ops.aten.sum.dim_IntList(mul_979, [0, 2, 3]);  mul_979 = None
    mul_980: "f32[120]" = torch.ops.aten.mul.Tensor(sum_60, 0.0006377551020408163)
    unsqueeze_651: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_980, 0);  mul_980 = None
    unsqueeze_652: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_651, 2);  unsqueeze_651 = None
    unsqueeze_653: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_652, 3);  unsqueeze_652 = None
    mul_981: "f32[120]" = torch.ops.aten.mul.Tensor(sum_61, 0.0006377551020408163)
    mul_982: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_184, squeeze_184)
    mul_983: "f32[120]" = torch.ops.aten.mul.Tensor(mul_981, mul_982);  mul_981 = mul_982 = None
    unsqueeze_654: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_983, 0);  mul_983 = None
    unsqueeze_655: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_654, 2);  unsqueeze_654 = None
    unsqueeze_656: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_655, 3);  unsqueeze_655 = None
    mul_984: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_184, primals_123);  primals_123 = None
    unsqueeze_657: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_984, 0);  mul_984 = None
    unsqueeze_658: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_657, 2);  unsqueeze_657 = None
    unsqueeze_659: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_658, 3);  unsqueeze_658 = None
    mul_985: "f32[8, 120, 14, 14]" = torch.ops.aten.mul.Tensor(sub_187, unsqueeze_656);  sub_187 = unsqueeze_656 = None
    sub_189: "f32[8, 120, 14, 14]" = torch.ops.aten.sub.Tensor(add_592, mul_985);  mul_985 = None
    sub_190: "f32[8, 120, 14, 14]" = torch.ops.aten.sub.Tensor(sub_189, unsqueeze_653);  sub_189 = unsqueeze_653 = None
    mul_986: "f32[8, 120, 14, 14]" = torch.ops.aten.mul.Tensor(sub_190, unsqueeze_659);  sub_190 = unsqueeze_659 = None
    mul_987: "f32[120]" = torch.ops.aten.mul.Tensor(sum_61, squeeze_184);  sum_61 = squeeze_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_42 = torch.ops.aten.convolution_backward.default(mul_986, mul_487, primals_278, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_986 = mul_487 = primals_278 = None
    getitem_300: "f32[8, 360, 14, 14]" = convolution_backward_42[0]
    getitem_301: "f32[120, 360, 1, 1]" = convolution_backward_42[1];  convolution_backward_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_988: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_300, div_58);  div_58 = None
    mul_989: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_300, div_60);  getitem_300 = div_60 = None
    sum_62: "f32[8, 360, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_988, [2, 3], True);  mul_988 = None
    mul_990: "f32[8, 360, 1, 1]" = torch.ops.aten.mul.Tensor(sum_62, 0.16666666666666666);  sum_62 = None
    where_60: "f32[8, 360, 1, 1]" = torch.ops.aten.where.self(bitwise_and_8, mul_990, full_default);  bitwise_and_8 = mul_990 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_43 = torch.ops.aten.convolution_backward.default(where_60, div_59, primals_276, [360], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_60 = div_59 = primals_276 = None
    getitem_303: "f32[8, 32, 1, 1]" = convolution_backward_43[0]
    getitem_304: "f32[360, 32, 1, 1]" = convolution_backward_43[1]
    getitem_305: "f32[360]" = convolution_backward_43[2];  convolution_backward_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    lt_35: "b8[8, 32, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_79, -3)
    le_26: "b8[8, 32, 1, 1]" = torch.ops.aten.le.Scalar(convolution_79, 3)
    div_130: "f32[8, 32, 1, 1]" = torch.ops.aten.div.Tensor(convolution_79, 3);  convolution_79 = None
    add_593: "f32[8, 32, 1, 1]" = torch.ops.aten.add.Tensor(div_130, 0.5);  div_130 = None
    mul_991: "f32[8, 32, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_303, add_593);  add_593 = None
    where_61: "f32[8, 32, 1, 1]" = torch.ops.aten.where.self(le_26, mul_991, getitem_303);  le_26 = mul_991 = getitem_303 = None
    where_62: "f32[8, 32, 1, 1]" = torch.ops.aten.where.self(lt_35, full_default, where_61);  lt_35 = where_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_44 = torch.ops.aten.convolution_backward.default(where_62, mean_9, primals_274, [32], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_62 = mean_9 = primals_274 = None
    getitem_306: "f32[8, 360, 1, 1]" = convolution_backward_44[0]
    getitem_307: "f32[32, 360, 1, 1]" = convolution_backward_44[1]
    getitem_308: "f32[32]" = convolution_backward_44[2];  convolution_backward_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_9: "f32[8, 360, 14, 14]" = torch.ops.aten.expand.default(getitem_306, [8, 360, 14, 14]);  getitem_306 = None
    div_131: "f32[8, 360, 14, 14]" = torch.ops.aten.div.Scalar(expand_9, 196);  expand_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_594: "f32[8, 360, 14, 14]" = torch.ops.aten.add.Tensor(mul_989, div_131);  mul_989 = div_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_36: "b8[8, 360, 14, 14]" = torch.ops.aten.lt.Scalar(clone_49, -3)
    le_27: "b8[8, 360, 14, 14]" = torch.ops.aten.le.Scalar(clone_49, 3)
    div_132: "f32[8, 360, 14, 14]" = torch.ops.aten.div.Tensor(clone_49, 3);  clone_49 = None
    add_595: "f32[8, 360, 14, 14]" = torch.ops.aten.add.Tensor(div_132, 0.5);  div_132 = None
    mul_992: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(add_594, add_595);  add_595 = None
    where_63: "f32[8, 360, 14, 14]" = torch.ops.aten.where.self(le_27, mul_992, add_594);  le_27 = mul_992 = add_594 = None
    where_64: "f32[8, 360, 14, 14]" = torch.ops.aten.where.self(lt_36, full_default, where_63);  lt_36 = where_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_63: "f32[360]" = torch.ops.aten.sum.dim_IntList(where_64, [0, 2, 3])
    sub_191: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_78, unsqueeze_662);  convolution_78 = unsqueeze_662 = None
    mul_993: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(where_64, sub_191)
    sum_64: "f32[360]" = torch.ops.aten.sum.dim_IntList(mul_993, [0, 2, 3]);  mul_993 = None
    mul_994: "f32[360]" = torch.ops.aten.mul.Tensor(sum_63, 0.0006377551020408163)
    unsqueeze_663: "f32[1, 360]" = torch.ops.aten.unsqueeze.default(mul_994, 0);  mul_994 = None
    unsqueeze_664: "f32[1, 360, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_663, 2);  unsqueeze_663 = None
    unsqueeze_665: "f32[1, 360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_664, 3);  unsqueeze_664 = None
    mul_995: "f32[360]" = torch.ops.aten.mul.Tensor(sum_64, 0.0006377551020408163)
    mul_996: "f32[360]" = torch.ops.aten.mul.Tensor(squeeze_181, squeeze_181)
    mul_997: "f32[360]" = torch.ops.aten.mul.Tensor(mul_995, mul_996);  mul_995 = mul_996 = None
    unsqueeze_666: "f32[1, 360]" = torch.ops.aten.unsqueeze.default(mul_997, 0);  mul_997 = None
    unsqueeze_667: "f32[1, 360, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_666, 2);  unsqueeze_666 = None
    unsqueeze_668: "f32[1, 360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_667, 3);  unsqueeze_667 = None
    mul_998: "f32[360]" = torch.ops.aten.mul.Tensor(squeeze_181, primals_121);  primals_121 = None
    unsqueeze_669: "f32[1, 360]" = torch.ops.aten.unsqueeze.default(mul_998, 0);  mul_998 = None
    unsqueeze_670: "f32[1, 360, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_669, 2);  unsqueeze_669 = None
    unsqueeze_671: "f32[1, 360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_670, 3);  unsqueeze_670 = None
    mul_999: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(sub_191, unsqueeze_668);  sub_191 = unsqueeze_668 = None
    sub_193: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(where_64, mul_999);  where_64 = mul_999 = None
    sub_194: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(sub_193, unsqueeze_665);  sub_193 = unsqueeze_665 = None
    mul_1000: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(sub_194, unsqueeze_671);  sub_194 = unsqueeze_671 = None
    mul_1001: "f32[360]" = torch.ops.aten.mul.Tensor(sum_64, squeeze_181);  sum_64 = squeeze_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_45 = torch.ops.aten.convolution_backward.default(mul_1000, div_57, primals_273, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 360, [True, True, False]);  mul_1000 = div_57 = primals_273 = None
    getitem_309: "f32[8, 360, 14, 14]" = convolution_backward_45[0]
    getitem_310: "f32[360, 1, 5, 5]" = convolution_backward_45[1];  convolution_backward_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_37: "b8[8, 360, 14, 14]" = torch.ops.aten.lt.Scalar(clone_48, -3)
    le_28: "b8[8, 360, 14, 14]" = torch.ops.aten.le.Scalar(clone_48, 3)
    div_133: "f32[8, 360, 14, 14]" = torch.ops.aten.div.Tensor(clone_48, 3);  clone_48 = None
    add_596: "f32[8, 360, 14, 14]" = torch.ops.aten.add.Tensor(div_133, 0.5);  div_133 = None
    mul_1002: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_309, add_596);  add_596 = None
    where_65: "f32[8, 360, 14, 14]" = torch.ops.aten.where.self(le_28, mul_1002, getitem_309);  le_28 = mul_1002 = getitem_309 = None
    where_66: "f32[8, 360, 14, 14]" = torch.ops.aten.where.self(lt_37, full_default, where_65);  lt_37 = where_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_65: "f32[360]" = torch.ops.aten.sum.dim_IntList(where_66, [0, 2, 3])
    sub_195: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_77, unsqueeze_674);  convolution_77 = unsqueeze_674 = None
    mul_1003: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(where_66, sub_195)
    sum_66: "f32[360]" = torch.ops.aten.sum.dim_IntList(mul_1003, [0, 2, 3]);  mul_1003 = None
    mul_1004: "f32[360]" = torch.ops.aten.mul.Tensor(sum_65, 0.0006377551020408163)
    unsqueeze_675: "f32[1, 360]" = torch.ops.aten.unsqueeze.default(mul_1004, 0);  mul_1004 = None
    unsqueeze_676: "f32[1, 360, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_675, 2);  unsqueeze_675 = None
    unsqueeze_677: "f32[1, 360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_676, 3);  unsqueeze_676 = None
    mul_1005: "f32[360]" = torch.ops.aten.mul.Tensor(sum_66, 0.0006377551020408163)
    mul_1006: "f32[360]" = torch.ops.aten.mul.Tensor(squeeze_178, squeeze_178)
    mul_1007: "f32[360]" = torch.ops.aten.mul.Tensor(mul_1005, mul_1006);  mul_1005 = mul_1006 = None
    unsqueeze_678: "f32[1, 360]" = torch.ops.aten.unsqueeze.default(mul_1007, 0);  mul_1007 = None
    unsqueeze_679: "f32[1, 360, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_678, 2);  unsqueeze_678 = None
    unsqueeze_680: "f32[1, 360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_679, 3);  unsqueeze_679 = None
    mul_1008: "f32[360]" = torch.ops.aten.mul.Tensor(squeeze_178, primals_119);  primals_119 = None
    unsqueeze_681: "f32[1, 360]" = torch.ops.aten.unsqueeze.default(mul_1008, 0);  mul_1008 = None
    unsqueeze_682: "f32[1, 360, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_681, 2);  unsqueeze_681 = None
    unsqueeze_683: "f32[1, 360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_682, 3);  unsqueeze_682 = None
    mul_1009: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(sub_195, unsqueeze_680);  sub_195 = unsqueeze_680 = None
    sub_197: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(where_66, mul_1009);  where_66 = mul_1009 = None
    sub_198: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(sub_197, unsqueeze_677);  sub_197 = unsqueeze_677 = None
    mul_1010: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(sub_198, unsqueeze_683);  sub_198 = unsqueeze_683 = None
    mul_1011: "f32[360]" = torch.ops.aten.mul.Tensor(sum_66, squeeze_178);  sum_66 = squeeze_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_46 = torch.ops.aten.convolution_backward.default(mul_1010, add_367, primals_272, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1010 = add_367 = primals_272 = None
    getitem_312: "f32[8, 120, 14, 14]" = convolution_backward_46[0]
    getitem_313: "f32[360, 120, 1, 1]" = convolution_backward_46[1];  convolution_backward_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_597: "f32[8, 120, 14, 14]" = torch.ops.aten.add.Tensor(add_592, getitem_312);  add_592 = getitem_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_67: "f32[120]" = torch.ops.aten.sum.dim_IntList(add_597, [0, 2, 3])
    sub_199: "f32[8, 120, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_76, unsqueeze_686);  convolution_76 = unsqueeze_686 = None
    mul_1012: "f32[8, 120, 14, 14]" = torch.ops.aten.mul.Tensor(add_597, sub_199)
    sum_68: "f32[120]" = torch.ops.aten.sum.dim_IntList(mul_1012, [0, 2, 3]);  mul_1012 = None
    mul_1013: "f32[120]" = torch.ops.aten.mul.Tensor(sum_67, 0.0006377551020408163)
    unsqueeze_687: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1013, 0);  mul_1013 = None
    unsqueeze_688: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_687, 2);  unsqueeze_687 = None
    unsqueeze_689: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_688, 3);  unsqueeze_688 = None
    mul_1014: "f32[120]" = torch.ops.aten.mul.Tensor(sum_68, 0.0006377551020408163)
    mul_1015: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_175, squeeze_175)
    mul_1016: "f32[120]" = torch.ops.aten.mul.Tensor(mul_1014, mul_1015);  mul_1014 = mul_1015 = None
    unsqueeze_690: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1016, 0);  mul_1016 = None
    unsqueeze_691: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_690, 2);  unsqueeze_690 = None
    unsqueeze_692: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_691, 3);  unsqueeze_691 = None
    mul_1017: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_175, primals_117);  primals_117 = None
    unsqueeze_693: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1017, 0);  mul_1017 = None
    unsqueeze_694: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_693, 2);  unsqueeze_693 = None
    unsqueeze_695: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_694, 3);  unsqueeze_694 = None
    mul_1018: "f32[8, 120, 14, 14]" = torch.ops.aten.mul.Tensor(sub_199, unsqueeze_692);  sub_199 = unsqueeze_692 = None
    sub_201: "f32[8, 120, 14, 14]" = torch.ops.aten.sub.Tensor(add_597, mul_1018);  mul_1018 = None
    sub_202: "f32[8, 120, 14, 14]" = torch.ops.aten.sub.Tensor(sub_201, unsqueeze_689);  sub_201 = unsqueeze_689 = None
    mul_1019: "f32[8, 120, 14, 14]" = torch.ops.aten.mul.Tensor(sub_202, unsqueeze_695);  sub_202 = unsqueeze_695 = None
    mul_1020: "f32[120]" = torch.ops.aten.mul.Tensor(sum_68, squeeze_175);  sum_68 = squeeze_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_47 = torch.ops.aten.convolution_backward.default(mul_1019, mul_462, primals_271, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1019 = mul_462 = primals_271 = None
    getitem_315: "f32[8, 360, 14, 14]" = convolution_backward_47[0]
    getitem_316: "f32[120, 360, 1, 1]" = convolution_backward_47[1];  convolution_backward_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_1021: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_315, div_54);  div_54 = None
    mul_1022: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_315, div_56);  getitem_315 = div_56 = None
    sum_69: "f32[8, 360, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1021, [2, 3], True);  mul_1021 = None
    mul_1023: "f32[8, 360, 1, 1]" = torch.ops.aten.mul.Tensor(sum_69, 0.16666666666666666);  sum_69 = None
    where_67: "f32[8, 360, 1, 1]" = torch.ops.aten.where.self(bitwise_and_9, mul_1023, full_default);  bitwise_and_9 = mul_1023 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_48 = torch.ops.aten.convolution_backward.default(where_67, div_55, primals_269, [360], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_67 = div_55 = primals_269 = None
    getitem_318: "f32[8, 32, 1, 1]" = convolution_backward_48[0]
    getitem_319: "f32[360, 32, 1, 1]" = convolution_backward_48[1]
    getitem_320: "f32[360]" = convolution_backward_48[2];  convolution_backward_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    lt_39: "b8[8, 32, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_74, -3)
    le_29: "b8[8, 32, 1, 1]" = torch.ops.aten.le.Scalar(convolution_74, 3)
    div_134: "f32[8, 32, 1, 1]" = torch.ops.aten.div.Tensor(convolution_74, 3);  convolution_74 = None
    add_598: "f32[8, 32, 1, 1]" = torch.ops.aten.add.Tensor(div_134, 0.5);  div_134 = None
    mul_1024: "f32[8, 32, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_318, add_598);  add_598 = None
    where_68: "f32[8, 32, 1, 1]" = torch.ops.aten.where.self(le_29, mul_1024, getitem_318);  le_29 = mul_1024 = getitem_318 = None
    where_69: "f32[8, 32, 1, 1]" = torch.ops.aten.where.self(lt_39, full_default, where_68);  lt_39 = where_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_49 = torch.ops.aten.convolution_backward.default(where_69, mean_8, primals_267, [32], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_69 = mean_8 = primals_267 = None
    getitem_321: "f32[8, 360, 1, 1]" = convolution_backward_49[0]
    getitem_322: "f32[32, 360, 1, 1]" = convolution_backward_49[1]
    getitem_323: "f32[32]" = convolution_backward_49[2];  convolution_backward_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_10: "f32[8, 360, 14, 14]" = torch.ops.aten.expand.default(getitem_321, [8, 360, 14, 14]);  getitem_321 = None
    div_135: "f32[8, 360, 14, 14]" = torch.ops.aten.div.Scalar(expand_10, 196);  expand_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_599: "f32[8, 360, 14, 14]" = torch.ops.aten.add.Tensor(mul_1022, div_135);  mul_1022 = div_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_40: "b8[8, 360, 14, 14]" = torch.ops.aten.lt.Scalar(clone_46, -3)
    le_30: "b8[8, 360, 14, 14]" = torch.ops.aten.le.Scalar(clone_46, 3)
    div_136: "f32[8, 360, 14, 14]" = torch.ops.aten.div.Tensor(clone_46, 3);  clone_46 = None
    add_600: "f32[8, 360, 14, 14]" = torch.ops.aten.add.Tensor(div_136, 0.5);  div_136 = None
    mul_1025: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(add_599, add_600);  add_600 = None
    where_70: "f32[8, 360, 14, 14]" = torch.ops.aten.where.self(le_30, mul_1025, add_599);  le_30 = mul_1025 = add_599 = None
    where_71: "f32[8, 360, 14, 14]" = torch.ops.aten.where.self(lt_40, full_default, where_70);  lt_40 = where_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_70: "f32[360]" = torch.ops.aten.sum.dim_IntList(where_71, [0, 2, 3])
    sub_203: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_73, unsqueeze_698);  convolution_73 = unsqueeze_698 = None
    mul_1026: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(where_71, sub_203)
    sum_71: "f32[360]" = torch.ops.aten.sum.dim_IntList(mul_1026, [0, 2, 3]);  mul_1026 = None
    mul_1027: "f32[360]" = torch.ops.aten.mul.Tensor(sum_70, 0.0006377551020408163)
    unsqueeze_699: "f32[1, 360]" = torch.ops.aten.unsqueeze.default(mul_1027, 0);  mul_1027 = None
    unsqueeze_700: "f32[1, 360, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_699, 2);  unsqueeze_699 = None
    unsqueeze_701: "f32[1, 360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_700, 3);  unsqueeze_700 = None
    mul_1028: "f32[360]" = torch.ops.aten.mul.Tensor(sum_71, 0.0006377551020408163)
    mul_1029: "f32[360]" = torch.ops.aten.mul.Tensor(squeeze_172, squeeze_172)
    mul_1030: "f32[360]" = torch.ops.aten.mul.Tensor(mul_1028, mul_1029);  mul_1028 = mul_1029 = None
    unsqueeze_702: "f32[1, 360]" = torch.ops.aten.unsqueeze.default(mul_1030, 0);  mul_1030 = None
    unsqueeze_703: "f32[1, 360, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_702, 2);  unsqueeze_702 = None
    unsqueeze_704: "f32[1, 360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_703, 3);  unsqueeze_703 = None
    mul_1031: "f32[360]" = torch.ops.aten.mul.Tensor(squeeze_172, primals_115);  primals_115 = None
    unsqueeze_705: "f32[1, 360]" = torch.ops.aten.unsqueeze.default(mul_1031, 0);  mul_1031 = None
    unsqueeze_706: "f32[1, 360, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_705, 2);  unsqueeze_705 = None
    unsqueeze_707: "f32[1, 360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_706, 3);  unsqueeze_706 = None
    mul_1032: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(sub_203, unsqueeze_704);  sub_203 = unsqueeze_704 = None
    sub_205: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(where_71, mul_1032);  where_71 = mul_1032 = None
    sub_206: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(sub_205, unsqueeze_701);  sub_205 = unsqueeze_701 = None
    mul_1033: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(sub_206, unsqueeze_707);  sub_206 = unsqueeze_707 = None
    mul_1034: "f32[360]" = torch.ops.aten.mul.Tensor(sum_71, squeeze_172);  sum_71 = squeeze_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_50 = torch.ops.aten.convolution_backward.default(mul_1033, div_53, primals_266, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 360, [True, True, False]);  mul_1033 = div_53 = primals_266 = None
    getitem_324: "f32[8, 360, 14, 14]" = convolution_backward_50[0]
    getitem_325: "f32[360, 1, 5, 5]" = convolution_backward_50[1];  convolution_backward_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_41: "b8[8, 360, 14, 14]" = torch.ops.aten.lt.Scalar(clone_45, -3)
    le_31: "b8[8, 360, 14, 14]" = torch.ops.aten.le.Scalar(clone_45, 3)
    div_137: "f32[8, 360, 14, 14]" = torch.ops.aten.div.Tensor(clone_45, 3);  clone_45 = None
    add_601: "f32[8, 360, 14, 14]" = torch.ops.aten.add.Tensor(div_137, 0.5);  div_137 = None
    mul_1035: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_324, add_601);  add_601 = None
    where_72: "f32[8, 360, 14, 14]" = torch.ops.aten.where.self(le_31, mul_1035, getitem_324);  le_31 = mul_1035 = getitem_324 = None
    where_73: "f32[8, 360, 14, 14]" = torch.ops.aten.where.self(lt_41, full_default, where_72);  lt_41 = where_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_72: "f32[360]" = torch.ops.aten.sum.dim_IntList(where_73, [0, 2, 3])
    sub_207: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_72, unsqueeze_710);  convolution_72 = unsqueeze_710 = None
    mul_1036: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(where_73, sub_207)
    sum_73: "f32[360]" = torch.ops.aten.sum.dim_IntList(mul_1036, [0, 2, 3]);  mul_1036 = None
    mul_1037: "f32[360]" = torch.ops.aten.mul.Tensor(sum_72, 0.0006377551020408163)
    unsqueeze_711: "f32[1, 360]" = torch.ops.aten.unsqueeze.default(mul_1037, 0);  mul_1037 = None
    unsqueeze_712: "f32[1, 360, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_711, 2);  unsqueeze_711 = None
    unsqueeze_713: "f32[1, 360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_712, 3);  unsqueeze_712 = None
    mul_1038: "f32[360]" = torch.ops.aten.mul.Tensor(sum_73, 0.0006377551020408163)
    mul_1039: "f32[360]" = torch.ops.aten.mul.Tensor(squeeze_169, squeeze_169)
    mul_1040: "f32[360]" = torch.ops.aten.mul.Tensor(mul_1038, mul_1039);  mul_1038 = mul_1039 = None
    unsqueeze_714: "f32[1, 360]" = torch.ops.aten.unsqueeze.default(mul_1040, 0);  mul_1040 = None
    unsqueeze_715: "f32[1, 360, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_714, 2);  unsqueeze_714 = None
    unsqueeze_716: "f32[1, 360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_715, 3);  unsqueeze_715 = None
    mul_1041: "f32[360]" = torch.ops.aten.mul.Tensor(squeeze_169, primals_113);  primals_113 = None
    unsqueeze_717: "f32[1, 360]" = torch.ops.aten.unsqueeze.default(mul_1041, 0);  mul_1041 = None
    unsqueeze_718: "f32[1, 360, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_717, 2);  unsqueeze_717 = None
    unsqueeze_719: "f32[1, 360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_718, 3);  unsqueeze_718 = None
    mul_1042: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(sub_207, unsqueeze_716);  sub_207 = unsqueeze_716 = None
    sub_209: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(where_73, mul_1042);  where_73 = mul_1042 = None
    sub_210: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(sub_209, unsqueeze_713);  sub_209 = unsqueeze_713 = None
    mul_1043: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(sub_210, unsqueeze_719);  sub_210 = unsqueeze_719 = None
    mul_1044: "f32[360]" = torch.ops.aten.mul.Tensor(sum_73, squeeze_169);  sum_73 = squeeze_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_51 = torch.ops.aten.convolution_backward.default(mul_1043, add_347, primals_265, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1043 = add_347 = primals_265 = None
    getitem_327: "f32[8, 120, 14, 14]" = convolution_backward_51[0]
    getitem_328: "f32[360, 120, 1, 1]" = convolution_backward_51[1];  convolution_backward_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_602: "f32[8, 120, 14, 14]" = torch.ops.aten.add.Tensor(add_597, getitem_327);  add_597 = getitem_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_74: "f32[120]" = torch.ops.aten.sum.dim_IntList(add_602, [0, 2, 3])
    sub_211: "f32[8, 120, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_71, unsqueeze_722);  convolution_71 = unsqueeze_722 = None
    mul_1045: "f32[8, 120, 14, 14]" = torch.ops.aten.mul.Tensor(add_602, sub_211)
    sum_75: "f32[120]" = torch.ops.aten.sum.dim_IntList(mul_1045, [0, 2, 3]);  mul_1045 = None
    mul_1046: "f32[120]" = torch.ops.aten.mul.Tensor(sum_74, 0.0006377551020408163)
    unsqueeze_723: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1046, 0);  mul_1046 = None
    unsqueeze_724: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_723, 2);  unsqueeze_723 = None
    unsqueeze_725: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_724, 3);  unsqueeze_724 = None
    mul_1047: "f32[120]" = torch.ops.aten.mul.Tensor(sum_75, 0.0006377551020408163)
    mul_1048: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_166, squeeze_166)
    mul_1049: "f32[120]" = torch.ops.aten.mul.Tensor(mul_1047, mul_1048);  mul_1047 = mul_1048 = None
    unsqueeze_726: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1049, 0);  mul_1049 = None
    unsqueeze_727: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_726, 2);  unsqueeze_726 = None
    unsqueeze_728: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_727, 3);  unsqueeze_727 = None
    mul_1050: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_166, primals_111);  primals_111 = None
    unsqueeze_729: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1050, 0);  mul_1050 = None
    unsqueeze_730: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_729, 2);  unsqueeze_729 = None
    unsqueeze_731: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_730, 3);  unsqueeze_730 = None
    mul_1051: "f32[8, 120, 14, 14]" = torch.ops.aten.mul.Tensor(sub_211, unsqueeze_728);  sub_211 = unsqueeze_728 = None
    sub_213: "f32[8, 120, 14, 14]" = torch.ops.aten.sub.Tensor(add_602, mul_1051);  mul_1051 = None
    sub_214: "f32[8, 120, 14, 14]" = torch.ops.aten.sub.Tensor(sub_213, unsqueeze_725);  sub_213 = unsqueeze_725 = None
    mul_1052: "f32[8, 120, 14, 14]" = torch.ops.aten.mul.Tensor(sub_214, unsqueeze_731);  sub_214 = unsqueeze_731 = None
    mul_1053: "f32[120]" = torch.ops.aten.mul.Tensor(sum_75, squeeze_166);  sum_75 = squeeze_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_52 = torch.ops.aten.convolution_backward.default(mul_1052, mul_437, primals_264, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1052 = mul_437 = primals_264 = None
    getitem_330: "f32[8, 360, 14, 14]" = convolution_backward_52[0]
    getitem_331: "f32[120, 360, 1, 1]" = convolution_backward_52[1];  convolution_backward_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_1054: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_330, div_50);  div_50 = None
    mul_1055: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_330, div_52);  getitem_330 = div_52 = None
    sum_76: "f32[8, 360, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1054, [2, 3], True);  mul_1054 = None
    mul_1056: "f32[8, 360, 1, 1]" = torch.ops.aten.mul.Tensor(sum_76, 0.16666666666666666);  sum_76 = None
    where_74: "f32[8, 360, 1, 1]" = torch.ops.aten.where.self(bitwise_and_10, mul_1056, full_default);  bitwise_and_10 = mul_1056 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_53 = torch.ops.aten.convolution_backward.default(where_74, div_51, primals_262, [360], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_74 = div_51 = primals_262 = None
    getitem_333: "f32[8, 32, 1, 1]" = convolution_backward_53[0]
    getitem_334: "f32[360, 32, 1, 1]" = convolution_backward_53[1]
    getitem_335: "f32[360]" = convolution_backward_53[2];  convolution_backward_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    lt_43: "b8[8, 32, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_69, -3)
    le_32: "b8[8, 32, 1, 1]" = torch.ops.aten.le.Scalar(convolution_69, 3)
    div_138: "f32[8, 32, 1, 1]" = torch.ops.aten.div.Tensor(convolution_69, 3);  convolution_69 = None
    add_603: "f32[8, 32, 1, 1]" = torch.ops.aten.add.Tensor(div_138, 0.5);  div_138 = None
    mul_1057: "f32[8, 32, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_333, add_603);  add_603 = None
    where_75: "f32[8, 32, 1, 1]" = torch.ops.aten.where.self(le_32, mul_1057, getitem_333);  le_32 = mul_1057 = getitem_333 = None
    where_76: "f32[8, 32, 1, 1]" = torch.ops.aten.where.self(lt_43, full_default, where_75);  lt_43 = where_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_54 = torch.ops.aten.convolution_backward.default(where_76, mean_7, primals_260, [32], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_76 = mean_7 = primals_260 = None
    getitem_336: "f32[8, 360, 1, 1]" = convolution_backward_54[0]
    getitem_337: "f32[32, 360, 1, 1]" = convolution_backward_54[1]
    getitem_338: "f32[32]" = convolution_backward_54[2];  convolution_backward_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_11: "f32[8, 360, 14, 14]" = torch.ops.aten.expand.default(getitem_336, [8, 360, 14, 14]);  getitem_336 = None
    div_139: "f32[8, 360, 14, 14]" = torch.ops.aten.div.Scalar(expand_11, 196);  expand_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_604: "f32[8, 360, 14, 14]" = torch.ops.aten.add.Tensor(mul_1055, div_139);  mul_1055 = div_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_44: "b8[8, 360, 14, 14]" = torch.ops.aten.lt.Scalar(clone_43, -3)
    le_33: "b8[8, 360, 14, 14]" = torch.ops.aten.le.Scalar(clone_43, 3)
    div_140: "f32[8, 360, 14, 14]" = torch.ops.aten.div.Tensor(clone_43, 3);  clone_43 = None
    add_605: "f32[8, 360, 14, 14]" = torch.ops.aten.add.Tensor(div_140, 0.5);  div_140 = None
    mul_1058: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(add_604, add_605);  add_605 = None
    where_77: "f32[8, 360, 14, 14]" = torch.ops.aten.where.self(le_33, mul_1058, add_604);  le_33 = mul_1058 = add_604 = None
    where_78: "f32[8, 360, 14, 14]" = torch.ops.aten.where.self(lt_44, full_default, where_77);  lt_44 = where_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_77: "f32[360]" = torch.ops.aten.sum.dim_IntList(where_78, [0, 2, 3])
    sub_215: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_68, unsqueeze_734);  convolution_68 = unsqueeze_734 = None
    mul_1059: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(where_78, sub_215)
    sum_78: "f32[360]" = torch.ops.aten.sum.dim_IntList(mul_1059, [0, 2, 3]);  mul_1059 = None
    mul_1060: "f32[360]" = torch.ops.aten.mul.Tensor(sum_77, 0.0006377551020408163)
    unsqueeze_735: "f32[1, 360]" = torch.ops.aten.unsqueeze.default(mul_1060, 0);  mul_1060 = None
    unsqueeze_736: "f32[1, 360, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_735, 2);  unsqueeze_735 = None
    unsqueeze_737: "f32[1, 360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_736, 3);  unsqueeze_736 = None
    mul_1061: "f32[360]" = torch.ops.aten.mul.Tensor(sum_78, 0.0006377551020408163)
    mul_1062: "f32[360]" = torch.ops.aten.mul.Tensor(squeeze_163, squeeze_163)
    mul_1063: "f32[360]" = torch.ops.aten.mul.Tensor(mul_1061, mul_1062);  mul_1061 = mul_1062 = None
    unsqueeze_738: "f32[1, 360]" = torch.ops.aten.unsqueeze.default(mul_1063, 0);  mul_1063 = None
    unsqueeze_739: "f32[1, 360, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_738, 2);  unsqueeze_738 = None
    unsqueeze_740: "f32[1, 360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_739, 3);  unsqueeze_739 = None
    mul_1064: "f32[360]" = torch.ops.aten.mul.Tensor(squeeze_163, primals_109);  primals_109 = None
    unsqueeze_741: "f32[1, 360]" = torch.ops.aten.unsqueeze.default(mul_1064, 0);  mul_1064 = None
    unsqueeze_742: "f32[1, 360, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_741, 2);  unsqueeze_741 = None
    unsqueeze_743: "f32[1, 360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_742, 3);  unsqueeze_742 = None
    mul_1065: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(sub_215, unsqueeze_740);  sub_215 = unsqueeze_740 = None
    sub_217: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(where_78, mul_1065);  where_78 = mul_1065 = None
    sub_218: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(sub_217, unsqueeze_737);  sub_217 = unsqueeze_737 = None
    mul_1066: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(sub_218, unsqueeze_743);  sub_218 = unsqueeze_743 = None
    mul_1067: "f32[360]" = torch.ops.aten.mul.Tensor(sum_78, squeeze_163);  sum_78 = squeeze_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_55 = torch.ops.aten.convolution_backward.default(mul_1066, div_49, primals_259, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 360, [True, True, False]);  mul_1066 = div_49 = primals_259 = None
    getitem_339: "f32[8, 360, 14, 14]" = convolution_backward_55[0]
    getitem_340: "f32[360, 1, 5, 5]" = convolution_backward_55[1];  convolution_backward_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_45: "b8[8, 360, 14, 14]" = torch.ops.aten.lt.Scalar(clone_42, -3)
    le_34: "b8[8, 360, 14, 14]" = torch.ops.aten.le.Scalar(clone_42, 3)
    div_141: "f32[8, 360, 14, 14]" = torch.ops.aten.div.Tensor(clone_42, 3);  clone_42 = None
    add_606: "f32[8, 360, 14, 14]" = torch.ops.aten.add.Tensor(div_141, 0.5);  div_141 = None
    mul_1068: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_339, add_606);  add_606 = None
    where_79: "f32[8, 360, 14, 14]" = torch.ops.aten.where.self(le_34, mul_1068, getitem_339);  le_34 = mul_1068 = getitem_339 = None
    where_80: "f32[8, 360, 14, 14]" = torch.ops.aten.where.self(lt_45, full_default, where_79);  lt_45 = where_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_79: "f32[360]" = torch.ops.aten.sum.dim_IntList(where_80, [0, 2, 3])
    sub_219: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_67, unsqueeze_746);  convolution_67 = unsqueeze_746 = None
    mul_1069: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(where_80, sub_219)
    sum_80: "f32[360]" = torch.ops.aten.sum.dim_IntList(mul_1069, [0, 2, 3]);  mul_1069 = None
    mul_1070: "f32[360]" = torch.ops.aten.mul.Tensor(sum_79, 0.0006377551020408163)
    unsqueeze_747: "f32[1, 360]" = torch.ops.aten.unsqueeze.default(mul_1070, 0);  mul_1070 = None
    unsqueeze_748: "f32[1, 360, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_747, 2);  unsqueeze_747 = None
    unsqueeze_749: "f32[1, 360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_748, 3);  unsqueeze_748 = None
    mul_1071: "f32[360]" = torch.ops.aten.mul.Tensor(sum_80, 0.0006377551020408163)
    mul_1072: "f32[360]" = torch.ops.aten.mul.Tensor(squeeze_160, squeeze_160)
    mul_1073: "f32[360]" = torch.ops.aten.mul.Tensor(mul_1071, mul_1072);  mul_1071 = mul_1072 = None
    unsqueeze_750: "f32[1, 360]" = torch.ops.aten.unsqueeze.default(mul_1073, 0);  mul_1073 = None
    unsqueeze_751: "f32[1, 360, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_750, 2);  unsqueeze_750 = None
    unsqueeze_752: "f32[1, 360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_751, 3);  unsqueeze_751 = None
    mul_1074: "f32[360]" = torch.ops.aten.mul.Tensor(squeeze_160, primals_107);  primals_107 = None
    unsqueeze_753: "f32[1, 360]" = torch.ops.aten.unsqueeze.default(mul_1074, 0);  mul_1074 = None
    unsqueeze_754: "f32[1, 360, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_753, 2);  unsqueeze_753 = None
    unsqueeze_755: "f32[1, 360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_754, 3);  unsqueeze_754 = None
    mul_1075: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(sub_219, unsqueeze_752);  sub_219 = unsqueeze_752 = None
    sub_221: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(where_80, mul_1075);  where_80 = mul_1075 = None
    sub_222: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(sub_221, unsqueeze_749);  sub_221 = unsqueeze_749 = None
    mul_1076: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(sub_222, unsqueeze_755);  sub_222 = unsqueeze_755 = None
    mul_1077: "f32[360]" = torch.ops.aten.mul.Tensor(sum_80, squeeze_160);  sum_80 = squeeze_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_56 = torch.ops.aten.convolution_backward.default(mul_1076, add_327, primals_258, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1076 = add_327 = primals_258 = None
    getitem_342: "f32[8, 120, 14, 14]" = convolution_backward_56[0]
    getitem_343: "f32[360, 120, 1, 1]" = convolution_backward_56[1];  convolution_backward_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_607: "f32[8, 120, 14, 14]" = torch.ops.aten.add.Tensor(add_602, getitem_342);  add_602 = getitem_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_81: "f32[120]" = torch.ops.aten.sum.dim_IntList(add_607, [0, 2, 3])
    sub_223: "f32[8, 120, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_66, unsqueeze_758);  convolution_66 = unsqueeze_758 = None
    mul_1078: "f32[8, 120, 14, 14]" = torch.ops.aten.mul.Tensor(add_607, sub_223)
    sum_82: "f32[120]" = torch.ops.aten.sum.dim_IntList(mul_1078, [0, 2, 3]);  mul_1078 = None
    mul_1079: "f32[120]" = torch.ops.aten.mul.Tensor(sum_81, 0.0006377551020408163)
    unsqueeze_759: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1079, 0);  mul_1079 = None
    unsqueeze_760: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_759, 2);  unsqueeze_759 = None
    unsqueeze_761: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_760, 3);  unsqueeze_760 = None
    mul_1080: "f32[120]" = torch.ops.aten.mul.Tensor(sum_82, 0.0006377551020408163)
    mul_1081: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_157, squeeze_157)
    mul_1082: "f32[120]" = torch.ops.aten.mul.Tensor(mul_1080, mul_1081);  mul_1080 = mul_1081 = None
    unsqueeze_762: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1082, 0);  mul_1082 = None
    unsqueeze_763: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_762, 2);  unsqueeze_762 = None
    unsqueeze_764: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_763, 3);  unsqueeze_763 = None
    mul_1083: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_157, primals_105);  primals_105 = None
    unsqueeze_765: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1083, 0);  mul_1083 = None
    unsqueeze_766: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_765, 2);  unsqueeze_765 = None
    unsqueeze_767: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_766, 3);  unsqueeze_766 = None
    mul_1084: "f32[8, 120, 14, 14]" = torch.ops.aten.mul.Tensor(sub_223, unsqueeze_764);  sub_223 = unsqueeze_764 = None
    sub_225: "f32[8, 120, 14, 14]" = torch.ops.aten.sub.Tensor(add_607, mul_1084);  mul_1084 = None
    sub_226: "f32[8, 120, 14, 14]" = torch.ops.aten.sub.Tensor(sub_225, unsqueeze_761);  sub_225 = unsqueeze_761 = None
    mul_1085: "f32[8, 120, 14, 14]" = torch.ops.aten.mul.Tensor(sub_226, unsqueeze_767);  sub_226 = unsqueeze_767 = None
    mul_1086: "f32[120]" = torch.ops.aten.mul.Tensor(sum_82, squeeze_157);  sum_82 = squeeze_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_57 = torch.ops.aten.convolution_backward.default(mul_1085, mul_412, primals_257, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1085 = mul_412 = primals_257 = None
    getitem_345: "f32[8, 360, 14, 14]" = convolution_backward_57[0]
    getitem_346: "f32[120, 360, 1, 1]" = convolution_backward_57[1];  convolution_backward_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_1087: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_345, div_46);  div_46 = None
    mul_1088: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_345, div_48);  getitem_345 = div_48 = None
    sum_83: "f32[8, 360, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1087, [2, 3], True);  mul_1087 = None
    mul_1089: "f32[8, 360, 1, 1]" = torch.ops.aten.mul.Tensor(sum_83, 0.16666666666666666);  sum_83 = None
    where_81: "f32[8, 360, 1, 1]" = torch.ops.aten.where.self(bitwise_and_11, mul_1089, full_default);  bitwise_and_11 = mul_1089 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_58 = torch.ops.aten.convolution_backward.default(where_81, div_47, primals_255, [360], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_81 = div_47 = primals_255 = None
    getitem_348: "f32[8, 32, 1, 1]" = convolution_backward_58[0]
    getitem_349: "f32[360, 32, 1, 1]" = convolution_backward_58[1]
    getitem_350: "f32[360]" = convolution_backward_58[2];  convolution_backward_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    lt_47: "b8[8, 32, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_64, -3)
    le_35: "b8[8, 32, 1, 1]" = torch.ops.aten.le.Scalar(convolution_64, 3)
    div_142: "f32[8, 32, 1, 1]" = torch.ops.aten.div.Tensor(convolution_64, 3);  convolution_64 = None
    add_608: "f32[8, 32, 1, 1]" = torch.ops.aten.add.Tensor(div_142, 0.5);  div_142 = None
    mul_1090: "f32[8, 32, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_348, add_608);  add_608 = None
    where_82: "f32[8, 32, 1, 1]" = torch.ops.aten.where.self(le_35, mul_1090, getitem_348);  le_35 = mul_1090 = getitem_348 = None
    where_83: "f32[8, 32, 1, 1]" = torch.ops.aten.where.self(lt_47, full_default, where_82);  lt_47 = where_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_59 = torch.ops.aten.convolution_backward.default(where_83, mean_6, primals_253, [32], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_83 = mean_6 = primals_253 = None
    getitem_351: "f32[8, 360, 1, 1]" = convolution_backward_59[0]
    getitem_352: "f32[32, 360, 1, 1]" = convolution_backward_59[1]
    getitem_353: "f32[32]" = convolution_backward_59[2];  convolution_backward_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_12: "f32[8, 360, 14, 14]" = torch.ops.aten.expand.default(getitem_351, [8, 360, 14, 14]);  getitem_351 = None
    div_143: "f32[8, 360, 14, 14]" = torch.ops.aten.div.Scalar(expand_12, 196);  expand_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_609: "f32[8, 360, 14, 14]" = torch.ops.aten.add.Tensor(mul_1088, div_143);  mul_1088 = div_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_48: "b8[8, 360, 14, 14]" = torch.ops.aten.lt.Scalar(clone_40, -3)
    le_36: "b8[8, 360, 14, 14]" = torch.ops.aten.le.Scalar(clone_40, 3)
    div_144: "f32[8, 360, 14, 14]" = torch.ops.aten.div.Tensor(clone_40, 3);  clone_40 = None
    add_610: "f32[8, 360, 14, 14]" = torch.ops.aten.add.Tensor(div_144, 0.5);  div_144 = None
    mul_1091: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(add_609, add_610);  add_610 = None
    where_84: "f32[8, 360, 14, 14]" = torch.ops.aten.where.self(le_36, mul_1091, add_609);  le_36 = mul_1091 = add_609 = None
    where_85: "f32[8, 360, 14, 14]" = torch.ops.aten.where.self(lt_48, full_default, where_84);  lt_48 = where_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_84: "f32[360]" = torch.ops.aten.sum.dim_IntList(where_85, [0, 2, 3])
    sub_227: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_63, unsqueeze_770);  convolution_63 = unsqueeze_770 = None
    mul_1092: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(where_85, sub_227)
    sum_85: "f32[360]" = torch.ops.aten.sum.dim_IntList(mul_1092, [0, 2, 3]);  mul_1092 = None
    mul_1093: "f32[360]" = torch.ops.aten.mul.Tensor(sum_84, 0.0006377551020408163)
    unsqueeze_771: "f32[1, 360]" = torch.ops.aten.unsqueeze.default(mul_1093, 0);  mul_1093 = None
    unsqueeze_772: "f32[1, 360, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_771, 2);  unsqueeze_771 = None
    unsqueeze_773: "f32[1, 360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_772, 3);  unsqueeze_772 = None
    mul_1094: "f32[360]" = torch.ops.aten.mul.Tensor(sum_85, 0.0006377551020408163)
    mul_1095: "f32[360]" = torch.ops.aten.mul.Tensor(squeeze_154, squeeze_154)
    mul_1096: "f32[360]" = torch.ops.aten.mul.Tensor(mul_1094, mul_1095);  mul_1094 = mul_1095 = None
    unsqueeze_774: "f32[1, 360]" = torch.ops.aten.unsqueeze.default(mul_1096, 0);  mul_1096 = None
    unsqueeze_775: "f32[1, 360, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_774, 2);  unsqueeze_774 = None
    unsqueeze_776: "f32[1, 360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_775, 3);  unsqueeze_775 = None
    mul_1097: "f32[360]" = torch.ops.aten.mul.Tensor(squeeze_154, primals_103);  primals_103 = None
    unsqueeze_777: "f32[1, 360]" = torch.ops.aten.unsqueeze.default(mul_1097, 0);  mul_1097 = None
    unsqueeze_778: "f32[1, 360, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_777, 2);  unsqueeze_777 = None
    unsqueeze_779: "f32[1, 360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_778, 3);  unsqueeze_778 = None
    mul_1098: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(sub_227, unsqueeze_776);  sub_227 = unsqueeze_776 = None
    sub_229: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(where_85, mul_1098);  where_85 = mul_1098 = None
    sub_230: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(sub_229, unsqueeze_773);  sub_229 = unsqueeze_773 = None
    mul_1099: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(sub_230, unsqueeze_779);  sub_230 = unsqueeze_779 = None
    mul_1100: "f32[360]" = torch.ops.aten.mul.Tensor(sum_85, squeeze_154);  sum_85 = squeeze_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_60 = torch.ops.aten.convolution_backward.default(mul_1099, div_45, primals_252, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 360, [True, True, False]);  mul_1099 = div_45 = primals_252 = None
    getitem_354: "f32[8, 360, 14, 14]" = convolution_backward_60[0]
    getitem_355: "f32[360, 1, 5, 5]" = convolution_backward_60[1];  convolution_backward_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_49: "b8[8, 360, 14, 14]" = torch.ops.aten.lt.Scalar(clone_39, -3)
    le_37: "b8[8, 360, 14, 14]" = torch.ops.aten.le.Scalar(clone_39, 3)
    div_145: "f32[8, 360, 14, 14]" = torch.ops.aten.div.Tensor(clone_39, 3);  clone_39 = None
    add_611: "f32[8, 360, 14, 14]" = torch.ops.aten.add.Tensor(div_145, 0.5);  div_145 = None
    mul_1101: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_354, add_611);  add_611 = None
    where_86: "f32[8, 360, 14, 14]" = torch.ops.aten.where.self(le_37, mul_1101, getitem_354);  le_37 = mul_1101 = getitem_354 = None
    where_87: "f32[8, 360, 14, 14]" = torch.ops.aten.where.self(lt_49, full_default, where_86);  lt_49 = where_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_86: "f32[360]" = torch.ops.aten.sum.dim_IntList(where_87, [0, 2, 3])
    sub_231: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_62, unsqueeze_782);  convolution_62 = unsqueeze_782 = None
    mul_1102: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(where_87, sub_231)
    sum_87: "f32[360]" = torch.ops.aten.sum.dim_IntList(mul_1102, [0, 2, 3]);  mul_1102 = None
    mul_1103: "f32[360]" = torch.ops.aten.mul.Tensor(sum_86, 0.0006377551020408163)
    unsqueeze_783: "f32[1, 360]" = torch.ops.aten.unsqueeze.default(mul_1103, 0);  mul_1103 = None
    unsqueeze_784: "f32[1, 360, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_783, 2);  unsqueeze_783 = None
    unsqueeze_785: "f32[1, 360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_784, 3);  unsqueeze_784 = None
    mul_1104: "f32[360]" = torch.ops.aten.mul.Tensor(sum_87, 0.0006377551020408163)
    mul_1105: "f32[360]" = torch.ops.aten.mul.Tensor(squeeze_151, squeeze_151)
    mul_1106: "f32[360]" = torch.ops.aten.mul.Tensor(mul_1104, mul_1105);  mul_1104 = mul_1105 = None
    unsqueeze_786: "f32[1, 360]" = torch.ops.aten.unsqueeze.default(mul_1106, 0);  mul_1106 = None
    unsqueeze_787: "f32[1, 360, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_786, 2);  unsqueeze_786 = None
    unsqueeze_788: "f32[1, 360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_787, 3);  unsqueeze_787 = None
    mul_1107: "f32[360]" = torch.ops.aten.mul.Tensor(squeeze_151, primals_101);  primals_101 = None
    unsqueeze_789: "f32[1, 360]" = torch.ops.aten.unsqueeze.default(mul_1107, 0);  mul_1107 = None
    unsqueeze_790: "f32[1, 360, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_789, 2);  unsqueeze_789 = None
    unsqueeze_791: "f32[1, 360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_790, 3);  unsqueeze_790 = None
    mul_1108: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(sub_231, unsqueeze_788);  sub_231 = unsqueeze_788 = None
    sub_233: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(where_87, mul_1108);  where_87 = mul_1108 = None
    sub_234: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(sub_233, unsqueeze_785);  sub_233 = unsqueeze_785 = None
    mul_1109: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(sub_234, unsqueeze_791);  sub_234 = unsqueeze_791 = None
    mul_1110: "f32[360]" = torch.ops.aten.mul.Tensor(sum_87, squeeze_151);  sum_87 = squeeze_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_61 = torch.ops.aten.convolution_backward.default(mul_1109, add_307, primals_251, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1109 = add_307 = primals_251 = None
    getitem_357: "f32[8, 120, 14, 14]" = convolution_backward_61[0]
    getitem_358: "f32[360, 120, 1, 1]" = convolution_backward_61[1];  convolution_backward_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_612: "f32[8, 120, 14, 14]" = torch.ops.aten.add.Tensor(add_607, getitem_357);  add_607 = getitem_357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_88: "f32[120]" = torch.ops.aten.sum.dim_IntList(add_612, [0, 2, 3])
    sub_235: "f32[8, 120, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_61, unsqueeze_794);  convolution_61 = unsqueeze_794 = None
    mul_1111: "f32[8, 120, 14, 14]" = torch.ops.aten.mul.Tensor(add_612, sub_235)
    sum_89: "f32[120]" = torch.ops.aten.sum.dim_IntList(mul_1111, [0, 2, 3]);  mul_1111 = None
    mul_1112: "f32[120]" = torch.ops.aten.mul.Tensor(sum_88, 0.0006377551020408163)
    unsqueeze_795: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1112, 0);  mul_1112 = None
    unsqueeze_796: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_795, 2);  unsqueeze_795 = None
    unsqueeze_797: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_796, 3);  unsqueeze_796 = None
    mul_1113: "f32[120]" = torch.ops.aten.mul.Tensor(sum_89, 0.0006377551020408163)
    mul_1114: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_148, squeeze_148)
    mul_1115: "f32[120]" = torch.ops.aten.mul.Tensor(mul_1113, mul_1114);  mul_1113 = mul_1114 = None
    unsqueeze_798: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1115, 0);  mul_1115 = None
    unsqueeze_799: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_798, 2);  unsqueeze_798 = None
    unsqueeze_800: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_799, 3);  unsqueeze_799 = None
    mul_1116: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_148, primals_99);  primals_99 = None
    unsqueeze_801: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1116, 0);  mul_1116 = None
    unsqueeze_802: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_801, 2);  unsqueeze_801 = None
    unsqueeze_803: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_802, 3);  unsqueeze_802 = None
    mul_1117: "f32[8, 120, 14, 14]" = torch.ops.aten.mul.Tensor(sub_235, unsqueeze_800);  sub_235 = unsqueeze_800 = None
    sub_237: "f32[8, 120, 14, 14]" = torch.ops.aten.sub.Tensor(add_612, mul_1117);  add_612 = mul_1117 = None
    sub_238: "f32[8, 120, 14, 14]" = torch.ops.aten.sub.Tensor(sub_237, unsqueeze_797);  sub_237 = unsqueeze_797 = None
    mul_1118: "f32[8, 120, 14, 14]" = torch.ops.aten.mul.Tensor(sub_238, unsqueeze_803);  sub_238 = unsqueeze_803 = None
    mul_1119: "f32[120]" = torch.ops.aten.mul.Tensor(sum_89, squeeze_148);  sum_89 = squeeze_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_62 = torch.ops.aten.convolution_backward.default(mul_1118, mul_387, primals_250, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1118 = mul_387 = primals_250 = None
    getitem_360: "f32[8, 360, 14, 14]" = convolution_backward_62[0]
    getitem_361: "f32[120, 360, 1, 1]" = convolution_backward_62[1];  convolution_backward_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_1120: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_360, div_42);  div_42 = None
    mul_1121: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_360, div_44);  getitem_360 = div_44 = None
    sum_90: "f32[8, 360, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1120, [2, 3], True);  mul_1120 = None
    mul_1122: "f32[8, 360, 1, 1]" = torch.ops.aten.mul.Tensor(sum_90, 0.16666666666666666);  sum_90 = None
    where_88: "f32[8, 360, 1, 1]" = torch.ops.aten.where.self(bitwise_and_12, mul_1122, full_default);  bitwise_and_12 = mul_1122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_63 = torch.ops.aten.convolution_backward.default(where_88, div_43, primals_248, [360], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_88 = div_43 = primals_248 = None
    getitem_363: "f32[8, 24, 1, 1]" = convolution_backward_63[0]
    getitem_364: "f32[360, 24, 1, 1]" = convolution_backward_63[1]
    getitem_365: "f32[360]" = convolution_backward_63[2];  convolution_backward_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    lt_51: "b8[8, 24, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_59, -3)
    le_38: "b8[8, 24, 1, 1]" = torch.ops.aten.le.Scalar(convolution_59, 3)
    div_146: "f32[8, 24, 1, 1]" = torch.ops.aten.div.Tensor(convolution_59, 3);  convolution_59 = None
    add_613: "f32[8, 24, 1, 1]" = torch.ops.aten.add.Tensor(div_146, 0.5);  div_146 = None
    mul_1123: "f32[8, 24, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_363, add_613);  add_613 = None
    where_89: "f32[8, 24, 1, 1]" = torch.ops.aten.where.self(le_38, mul_1123, getitem_363);  le_38 = mul_1123 = getitem_363 = None
    where_90: "f32[8, 24, 1, 1]" = torch.ops.aten.where.self(lt_51, full_default, where_89);  lt_51 = where_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_64 = torch.ops.aten.convolution_backward.default(where_90, mean_5, primals_246, [24], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_90 = mean_5 = primals_246 = None
    getitem_366: "f32[8, 360, 1, 1]" = convolution_backward_64[0]
    getitem_367: "f32[24, 360, 1, 1]" = convolution_backward_64[1]
    getitem_368: "f32[24]" = convolution_backward_64[2];  convolution_backward_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_13: "f32[8, 360, 14, 14]" = torch.ops.aten.expand.default(getitem_366, [8, 360, 14, 14]);  getitem_366 = None
    div_147: "f32[8, 360, 14, 14]" = torch.ops.aten.div.Scalar(expand_13, 196);  expand_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_614: "f32[8, 360, 14, 14]" = torch.ops.aten.add.Tensor(mul_1121, div_147);  mul_1121 = div_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_52: "b8[8, 360, 14, 14]" = torch.ops.aten.lt.Scalar(clone_37, -3)
    le_39: "b8[8, 360, 14, 14]" = torch.ops.aten.le.Scalar(clone_37, 3)
    div_148: "f32[8, 360, 14, 14]" = torch.ops.aten.div.Tensor(clone_37, 3);  clone_37 = None
    add_615: "f32[8, 360, 14, 14]" = torch.ops.aten.add.Tensor(div_148, 0.5);  div_148 = None
    mul_1124: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(add_614, add_615);  add_615 = None
    where_91: "f32[8, 360, 14, 14]" = torch.ops.aten.where.self(le_39, mul_1124, add_614);  le_39 = mul_1124 = add_614 = None
    where_92: "f32[8, 360, 14, 14]" = torch.ops.aten.where.self(lt_52, full_default, where_91);  lt_52 = where_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_91: "f32[360]" = torch.ops.aten.sum.dim_IntList(where_92, [0, 2, 3])
    sub_239: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_58, unsqueeze_806);  convolution_58 = unsqueeze_806 = None
    mul_1125: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(where_92, sub_239)
    sum_92: "f32[360]" = torch.ops.aten.sum.dim_IntList(mul_1125, [0, 2, 3]);  mul_1125 = None
    mul_1126: "f32[360]" = torch.ops.aten.mul.Tensor(sum_91, 0.0006377551020408163)
    unsqueeze_807: "f32[1, 360]" = torch.ops.aten.unsqueeze.default(mul_1126, 0);  mul_1126 = None
    unsqueeze_808: "f32[1, 360, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_807, 2);  unsqueeze_807 = None
    unsqueeze_809: "f32[1, 360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_808, 3);  unsqueeze_808 = None
    mul_1127: "f32[360]" = torch.ops.aten.mul.Tensor(sum_92, 0.0006377551020408163)
    mul_1128: "f32[360]" = torch.ops.aten.mul.Tensor(squeeze_145, squeeze_145)
    mul_1129: "f32[360]" = torch.ops.aten.mul.Tensor(mul_1127, mul_1128);  mul_1127 = mul_1128 = None
    unsqueeze_810: "f32[1, 360]" = torch.ops.aten.unsqueeze.default(mul_1129, 0);  mul_1129 = None
    unsqueeze_811: "f32[1, 360, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_810, 2);  unsqueeze_810 = None
    unsqueeze_812: "f32[1, 360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_811, 3);  unsqueeze_811 = None
    mul_1130: "f32[360]" = torch.ops.aten.mul.Tensor(squeeze_145, primals_97);  primals_97 = None
    unsqueeze_813: "f32[1, 360]" = torch.ops.aten.unsqueeze.default(mul_1130, 0);  mul_1130 = None
    unsqueeze_814: "f32[1, 360, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_813, 2);  unsqueeze_813 = None
    unsqueeze_815: "f32[1, 360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_814, 3);  unsqueeze_814 = None
    mul_1131: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(sub_239, unsqueeze_812);  sub_239 = unsqueeze_812 = None
    sub_241: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(where_92, mul_1131);  where_92 = mul_1131 = None
    sub_242: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(sub_241, unsqueeze_809);  sub_241 = unsqueeze_809 = None
    mul_1132: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(sub_242, unsqueeze_815);  sub_242 = unsqueeze_815 = None
    mul_1133: "f32[360]" = torch.ops.aten.mul.Tensor(sum_92, squeeze_145);  sum_92 = squeeze_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_65 = torch.ops.aten.convolution_backward.default(mul_1132, div_41, primals_245, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 360, [True, True, False]);  mul_1132 = div_41 = primals_245 = None
    getitem_369: "f32[8, 360, 14, 14]" = convolution_backward_65[0]
    getitem_370: "f32[360, 1, 3, 3]" = convolution_backward_65[1];  convolution_backward_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_53: "b8[8, 360, 14, 14]" = torch.ops.aten.lt.Scalar(clone_36, -3)
    le_40: "b8[8, 360, 14, 14]" = torch.ops.aten.le.Scalar(clone_36, 3)
    div_149: "f32[8, 360, 14, 14]" = torch.ops.aten.div.Tensor(clone_36, 3);  clone_36 = None
    add_616: "f32[8, 360, 14, 14]" = torch.ops.aten.add.Tensor(div_149, 0.5);  div_149 = None
    mul_1134: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_369, add_616);  add_616 = None
    where_93: "f32[8, 360, 14, 14]" = torch.ops.aten.where.self(le_40, mul_1134, getitem_369);  le_40 = mul_1134 = getitem_369 = None
    where_94: "f32[8, 360, 14, 14]" = torch.ops.aten.where.self(lt_53, full_default, where_93);  lt_53 = where_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_93: "f32[360]" = torch.ops.aten.sum.dim_IntList(where_94, [0, 2, 3])
    sub_243: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_57, unsqueeze_818);  convolution_57 = unsqueeze_818 = None
    mul_1135: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(where_94, sub_243)
    sum_94: "f32[360]" = torch.ops.aten.sum.dim_IntList(mul_1135, [0, 2, 3]);  mul_1135 = None
    mul_1136: "f32[360]" = torch.ops.aten.mul.Tensor(sum_93, 0.0006377551020408163)
    unsqueeze_819: "f32[1, 360]" = torch.ops.aten.unsqueeze.default(mul_1136, 0);  mul_1136 = None
    unsqueeze_820: "f32[1, 360, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_819, 2);  unsqueeze_819 = None
    unsqueeze_821: "f32[1, 360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_820, 3);  unsqueeze_820 = None
    mul_1137: "f32[360]" = torch.ops.aten.mul.Tensor(sum_94, 0.0006377551020408163)
    mul_1138: "f32[360]" = torch.ops.aten.mul.Tensor(squeeze_142, squeeze_142)
    mul_1139: "f32[360]" = torch.ops.aten.mul.Tensor(mul_1137, mul_1138);  mul_1137 = mul_1138 = None
    unsqueeze_822: "f32[1, 360]" = torch.ops.aten.unsqueeze.default(mul_1139, 0);  mul_1139 = None
    unsqueeze_823: "f32[1, 360, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_822, 2);  unsqueeze_822 = None
    unsqueeze_824: "f32[1, 360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_823, 3);  unsqueeze_823 = None
    mul_1140: "f32[360]" = torch.ops.aten.mul.Tensor(squeeze_142, primals_95);  primals_95 = None
    unsqueeze_825: "f32[1, 360]" = torch.ops.aten.unsqueeze.default(mul_1140, 0);  mul_1140 = None
    unsqueeze_826: "f32[1, 360, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_825, 2);  unsqueeze_825 = None
    unsqueeze_827: "f32[1, 360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_826, 3);  unsqueeze_826 = None
    mul_1141: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(sub_243, unsqueeze_824);  sub_243 = unsqueeze_824 = None
    sub_245: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(where_94, mul_1141);  where_94 = mul_1141 = None
    sub_246: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(sub_245, unsqueeze_821);  sub_245 = unsqueeze_821 = None
    mul_1142: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(sub_246, unsqueeze_827);  sub_246 = unsqueeze_827 = None
    mul_1143: "f32[360]" = torch.ops.aten.mul.Tensor(sum_94, squeeze_142);  sum_94 = squeeze_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_66 = torch.ops.aten.convolution_backward.default(mul_1142, add_288, primals_244, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1142 = add_288 = primals_244 = None
    getitem_372: "f32[8, 72, 14, 14]" = convolution_backward_66[0]
    getitem_373: "f32[360, 72, 1, 1]" = convolution_backward_66[1];  convolution_backward_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_95: "f32[72]" = torch.ops.aten.sum.dim_IntList(getitem_372, [0, 2, 3])
    sub_247: "f32[8, 72, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_56, unsqueeze_830);  convolution_56 = unsqueeze_830 = None
    mul_1144: "f32[8, 72, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_372, sub_247)
    sum_96: "f32[72]" = torch.ops.aten.sum.dim_IntList(mul_1144, [0, 2, 3]);  mul_1144 = None
    mul_1145: "f32[72]" = torch.ops.aten.mul.Tensor(sum_95, 0.0006377551020408163)
    unsqueeze_831: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_1145, 0);  mul_1145 = None
    unsqueeze_832: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_831, 2);  unsqueeze_831 = None
    unsqueeze_833: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_832, 3);  unsqueeze_832 = None
    mul_1146: "f32[72]" = torch.ops.aten.mul.Tensor(sum_96, 0.0006377551020408163)
    mul_1147: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_139, squeeze_139)
    mul_1148: "f32[72]" = torch.ops.aten.mul.Tensor(mul_1146, mul_1147);  mul_1146 = mul_1147 = None
    unsqueeze_834: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_1148, 0);  mul_1148 = None
    unsqueeze_835: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_834, 2);  unsqueeze_834 = None
    unsqueeze_836: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_835, 3);  unsqueeze_835 = None
    mul_1149: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_139, primals_93);  primals_93 = None
    unsqueeze_837: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_1149, 0);  mul_1149 = None
    unsqueeze_838: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_837, 2);  unsqueeze_837 = None
    unsqueeze_839: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_838, 3);  unsqueeze_838 = None
    mul_1150: "f32[8, 72, 14, 14]" = torch.ops.aten.mul.Tensor(sub_247, unsqueeze_836);  sub_247 = unsqueeze_836 = None
    sub_249: "f32[8, 72, 14, 14]" = torch.ops.aten.sub.Tensor(getitem_372, mul_1150);  mul_1150 = None
    sub_250: "f32[8, 72, 14, 14]" = torch.ops.aten.sub.Tensor(sub_249, unsqueeze_833);  sub_249 = unsqueeze_833 = None
    mul_1151: "f32[8, 72, 14, 14]" = torch.ops.aten.mul.Tensor(sub_250, unsqueeze_839);  sub_250 = unsqueeze_839 = None
    mul_1152: "f32[72]" = torch.ops.aten.mul.Tensor(sum_96, squeeze_139);  sum_96 = squeeze_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_67 = torch.ops.aten.convolution_backward.default(mul_1151, div_40, primals_243, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1151 = div_40 = primals_243 = None
    getitem_375: "f32[8, 216, 14, 14]" = convolution_backward_67[0]
    getitem_376: "f32[72, 216, 1, 1]" = convolution_backward_67[1];  convolution_backward_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_54: "b8[8, 216, 14, 14]" = torch.ops.aten.lt.Scalar(clone_35, -3)
    le_41: "b8[8, 216, 14, 14]" = torch.ops.aten.le.Scalar(clone_35, 3)
    div_150: "f32[8, 216, 14, 14]" = torch.ops.aten.div.Tensor(clone_35, 3);  clone_35 = None
    add_617: "f32[8, 216, 14, 14]" = torch.ops.aten.add.Tensor(div_150, 0.5);  div_150 = None
    mul_1153: "f32[8, 216, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_375, add_617);  add_617 = None
    where_95: "f32[8, 216, 14, 14]" = torch.ops.aten.where.self(le_41, mul_1153, getitem_375);  le_41 = mul_1153 = getitem_375 = None
    where_96: "f32[8, 216, 14, 14]" = torch.ops.aten.where.self(lt_54, full_default, where_95);  lt_54 = where_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_97: "f32[216]" = torch.ops.aten.sum.dim_IntList(where_96, [0, 2, 3])
    sub_251: "f32[8, 216, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_55, unsqueeze_842);  convolution_55 = unsqueeze_842 = None
    mul_1154: "f32[8, 216, 14, 14]" = torch.ops.aten.mul.Tensor(where_96, sub_251)
    sum_98: "f32[216]" = torch.ops.aten.sum.dim_IntList(mul_1154, [0, 2, 3]);  mul_1154 = None
    mul_1155: "f32[216]" = torch.ops.aten.mul.Tensor(sum_97, 0.0006377551020408163)
    unsqueeze_843: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_1155, 0);  mul_1155 = None
    unsqueeze_844: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_843, 2);  unsqueeze_843 = None
    unsqueeze_845: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_844, 3);  unsqueeze_844 = None
    mul_1156: "f32[216]" = torch.ops.aten.mul.Tensor(sum_98, 0.0006377551020408163)
    mul_1157: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_136, squeeze_136)
    mul_1158: "f32[216]" = torch.ops.aten.mul.Tensor(mul_1156, mul_1157);  mul_1156 = mul_1157 = None
    unsqueeze_846: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_1158, 0);  mul_1158 = None
    unsqueeze_847: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_846, 2);  unsqueeze_846 = None
    unsqueeze_848: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_847, 3);  unsqueeze_847 = None
    mul_1159: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_136, primals_91);  primals_91 = None
    unsqueeze_849: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_1159, 0);  mul_1159 = None
    unsqueeze_850: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_849, 2);  unsqueeze_849 = None
    unsqueeze_851: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_850, 3);  unsqueeze_850 = None
    mul_1160: "f32[8, 216, 14, 14]" = torch.ops.aten.mul.Tensor(sub_251, unsqueeze_848);  sub_251 = unsqueeze_848 = None
    sub_253: "f32[8, 216, 14, 14]" = torch.ops.aten.sub.Tensor(where_96, mul_1160);  where_96 = mul_1160 = None
    sub_254: "f32[8, 216, 14, 14]" = torch.ops.aten.sub.Tensor(sub_253, unsqueeze_845);  sub_253 = unsqueeze_845 = None
    mul_1161: "f32[8, 216, 14, 14]" = torch.ops.aten.mul.Tensor(sub_254, unsqueeze_851);  sub_254 = unsqueeze_851 = None
    mul_1162: "f32[216]" = torch.ops.aten.mul.Tensor(sum_98, squeeze_136);  sum_98 = squeeze_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_68 = torch.ops.aten.convolution_backward.default(mul_1161, div_39, primals_242, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 216, [True, True, False]);  mul_1161 = div_39 = primals_242 = None
    getitem_378: "f32[8, 216, 14, 14]" = convolution_backward_68[0]
    getitem_379: "f32[216, 1, 3, 3]" = convolution_backward_68[1];  convolution_backward_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_55: "b8[8, 216, 14, 14]" = torch.ops.aten.lt.Scalar(clone_34, -3)
    le_42: "b8[8, 216, 14, 14]" = torch.ops.aten.le.Scalar(clone_34, 3)
    div_151: "f32[8, 216, 14, 14]" = torch.ops.aten.div.Tensor(clone_34, 3);  clone_34 = None
    add_618: "f32[8, 216, 14, 14]" = torch.ops.aten.add.Tensor(div_151, 0.5);  div_151 = None
    mul_1163: "f32[8, 216, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_378, add_618);  add_618 = None
    where_97: "f32[8, 216, 14, 14]" = torch.ops.aten.where.self(le_42, mul_1163, getitem_378);  le_42 = mul_1163 = getitem_378 = None
    where_98: "f32[8, 216, 14, 14]" = torch.ops.aten.where.self(lt_55, full_default, where_97);  lt_55 = where_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_99: "f32[216]" = torch.ops.aten.sum.dim_IntList(where_98, [0, 2, 3])
    sub_255: "f32[8, 216, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_54, unsqueeze_854);  convolution_54 = unsqueeze_854 = None
    mul_1164: "f32[8, 216, 14, 14]" = torch.ops.aten.mul.Tensor(where_98, sub_255)
    sum_100: "f32[216]" = torch.ops.aten.sum.dim_IntList(mul_1164, [0, 2, 3]);  mul_1164 = None
    mul_1165: "f32[216]" = torch.ops.aten.mul.Tensor(sum_99, 0.0006377551020408163)
    unsqueeze_855: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_1165, 0);  mul_1165 = None
    unsqueeze_856: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_855, 2);  unsqueeze_855 = None
    unsqueeze_857: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_856, 3);  unsqueeze_856 = None
    mul_1166: "f32[216]" = torch.ops.aten.mul.Tensor(sum_100, 0.0006377551020408163)
    mul_1167: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_133, squeeze_133)
    mul_1168: "f32[216]" = torch.ops.aten.mul.Tensor(mul_1166, mul_1167);  mul_1166 = mul_1167 = None
    unsqueeze_858: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_1168, 0);  mul_1168 = None
    unsqueeze_859: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_858, 2);  unsqueeze_858 = None
    unsqueeze_860: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_859, 3);  unsqueeze_859 = None
    mul_1169: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_133, primals_89);  primals_89 = None
    unsqueeze_861: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_1169, 0);  mul_1169 = None
    unsqueeze_862: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_861, 2);  unsqueeze_861 = None
    unsqueeze_863: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_862, 3);  unsqueeze_862 = None
    mul_1170: "f32[8, 216, 14, 14]" = torch.ops.aten.mul.Tensor(sub_255, unsqueeze_860);  sub_255 = unsqueeze_860 = None
    sub_257: "f32[8, 216, 14, 14]" = torch.ops.aten.sub.Tensor(where_98, mul_1170);  where_98 = mul_1170 = None
    sub_258: "f32[8, 216, 14, 14]" = torch.ops.aten.sub.Tensor(sub_257, unsqueeze_857);  sub_257 = unsqueeze_857 = None
    mul_1171: "f32[8, 216, 14, 14]" = torch.ops.aten.mul.Tensor(sub_258, unsqueeze_863);  sub_258 = unsqueeze_863 = None
    mul_1172: "f32[216]" = torch.ops.aten.mul.Tensor(sum_100, squeeze_133);  sum_100 = squeeze_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_69 = torch.ops.aten.convolution_backward.default(mul_1171, add_270, primals_241, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1171 = add_270 = primals_241 = None
    getitem_381: "f32[8, 72, 14, 14]" = convolution_backward_69[0]
    getitem_382: "f32[216, 72, 1, 1]" = convolution_backward_69[1];  convolution_backward_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_619: "f32[8, 72, 14, 14]" = torch.ops.aten.add.Tensor(getitem_372, getitem_381);  getitem_372 = getitem_381 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_101: "f32[72]" = torch.ops.aten.sum.dim_IntList(add_619, [0, 2, 3])
    sub_259: "f32[8, 72, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_53, unsqueeze_866);  convolution_53 = unsqueeze_866 = None
    mul_1173: "f32[8, 72, 14, 14]" = torch.ops.aten.mul.Tensor(add_619, sub_259)
    sum_102: "f32[72]" = torch.ops.aten.sum.dim_IntList(mul_1173, [0, 2, 3]);  mul_1173 = None
    mul_1174: "f32[72]" = torch.ops.aten.mul.Tensor(sum_101, 0.0006377551020408163)
    unsqueeze_867: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_1174, 0);  mul_1174 = None
    unsqueeze_868: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_867, 2);  unsqueeze_867 = None
    unsqueeze_869: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_868, 3);  unsqueeze_868 = None
    mul_1175: "f32[72]" = torch.ops.aten.mul.Tensor(sum_102, 0.0006377551020408163)
    mul_1176: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_130, squeeze_130)
    mul_1177: "f32[72]" = torch.ops.aten.mul.Tensor(mul_1175, mul_1176);  mul_1175 = mul_1176 = None
    unsqueeze_870: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_1177, 0);  mul_1177 = None
    unsqueeze_871: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_870, 2);  unsqueeze_870 = None
    unsqueeze_872: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_871, 3);  unsqueeze_871 = None
    mul_1178: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_130, primals_87);  primals_87 = None
    unsqueeze_873: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_1178, 0);  mul_1178 = None
    unsqueeze_874: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_873, 2);  unsqueeze_873 = None
    unsqueeze_875: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_874, 3);  unsqueeze_874 = None
    mul_1179: "f32[8, 72, 14, 14]" = torch.ops.aten.mul.Tensor(sub_259, unsqueeze_872);  sub_259 = unsqueeze_872 = None
    sub_261: "f32[8, 72, 14, 14]" = torch.ops.aten.sub.Tensor(add_619, mul_1179);  mul_1179 = None
    sub_262: "f32[8, 72, 14, 14]" = torch.ops.aten.sub.Tensor(sub_261, unsqueeze_869);  sub_261 = unsqueeze_869 = None
    mul_1180: "f32[8, 72, 14, 14]" = torch.ops.aten.mul.Tensor(sub_262, unsqueeze_875);  sub_262 = unsqueeze_875 = None
    mul_1181: "f32[72]" = torch.ops.aten.mul.Tensor(sum_102, squeeze_130);  sum_102 = squeeze_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_70 = torch.ops.aten.convolution_backward.default(mul_1180, div_38, primals_240, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1180 = div_38 = primals_240 = None
    getitem_384: "f32[8, 216, 14, 14]" = convolution_backward_70[0]
    getitem_385: "f32[72, 216, 1, 1]" = convolution_backward_70[1];  convolution_backward_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_56: "b8[8, 216, 14, 14]" = torch.ops.aten.lt.Scalar(clone_33, -3)
    le_43: "b8[8, 216, 14, 14]" = torch.ops.aten.le.Scalar(clone_33, 3)
    div_152: "f32[8, 216, 14, 14]" = torch.ops.aten.div.Tensor(clone_33, 3);  clone_33 = None
    add_620: "f32[8, 216, 14, 14]" = torch.ops.aten.add.Tensor(div_152, 0.5);  div_152 = None
    mul_1182: "f32[8, 216, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_384, add_620);  add_620 = None
    where_99: "f32[8, 216, 14, 14]" = torch.ops.aten.where.self(le_43, mul_1182, getitem_384);  le_43 = mul_1182 = getitem_384 = None
    where_100: "f32[8, 216, 14, 14]" = torch.ops.aten.where.self(lt_56, full_default, where_99);  lt_56 = where_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_103: "f32[216]" = torch.ops.aten.sum.dim_IntList(where_100, [0, 2, 3])
    sub_263: "f32[8, 216, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_52, unsqueeze_878);  convolution_52 = unsqueeze_878 = None
    mul_1183: "f32[8, 216, 14, 14]" = torch.ops.aten.mul.Tensor(where_100, sub_263)
    sum_104: "f32[216]" = torch.ops.aten.sum.dim_IntList(mul_1183, [0, 2, 3]);  mul_1183 = None
    mul_1184: "f32[216]" = torch.ops.aten.mul.Tensor(sum_103, 0.0006377551020408163)
    unsqueeze_879: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_1184, 0);  mul_1184 = None
    unsqueeze_880: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_879, 2);  unsqueeze_879 = None
    unsqueeze_881: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_880, 3);  unsqueeze_880 = None
    mul_1185: "f32[216]" = torch.ops.aten.mul.Tensor(sum_104, 0.0006377551020408163)
    mul_1186: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_127, squeeze_127)
    mul_1187: "f32[216]" = torch.ops.aten.mul.Tensor(mul_1185, mul_1186);  mul_1185 = mul_1186 = None
    unsqueeze_882: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_1187, 0);  mul_1187 = None
    unsqueeze_883: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_882, 2);  unsqueeze_882 = None
    unsqueeze_884: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_883, 3);  unsqueeze_883 = None
    mul_1188: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_127, primals_85);  primals_85 = None
    unsqueeze_885: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_1188, 0);  mul_1188 = None
    unsqueeze_886: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_885, 2);  unsqueeze_885 = None
    unsqueeze_887: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_886, 3);  unsqueeze_886 = None
    mul_1189: "f32[8, 216, 14, 14]" = torch.ops.aten.mul.Tensor(sub_263, unsqueeze_884);  sub_263 = unsqueeze_884 = None
    sub_265: "f32[8, 216, 14, 14]" = torch.ops.aten.sub.Tensor(where_100, mul_1189);  where_100 = mul_1189 = None
    sub_266: "f32[8, 216, 14, 14]" = torch.ops.aten.sub.Tensor(sub_265, unsqueeze_881);  sub_265 = unsqueeze_881 = None
    mul_1190: "f32[8, 216, 14, 14]" = torch.ops.aten.mul.Tensor(sub_266, unsqueeze_887);  sub_266 = unsqueeze_887 = None
    mul_1191: "f32[216]" = torch.ops.aten.mul.Tensor(sum_104, squeeze_127);  sum_104 = squeeze_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_71 = torch.ops.aten.convolution_backward.default(mul_1190, div_37, primals_239, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 216, [True, True, False]);  mul_1190 = div_37 = primals_239 = None
    getitem_387: "f32[8, 216, 14, 14]" = convolution_backward_71[0]
    getitem_388: "f32[216, 1, 3, 3]" = convolution_backward_71[1];  convolution_backward_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_57: "b8[8, 216, 14, 14]" = torch.ops.aten.lt.Scalar(clone_32, -3)
    le_44: "b8[8, 216, 14, 14]" = torch.ops.aten.le.Scalar(clone_32, 3)
    div_153: "f32[8, 216, 14, 14]" = torch.ops.aten.div.Tensor(clone_32, 3);  clone_32 = None
    add_621: "f32[8, 216, 14, 14]" = torch.ops.aten.add.Tensor(div_153, 0.5);  div_153 = None
    mul_1192: "f32[8, 216, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_387, add_621);  add_621 = None
    where_101: "f32[8, 216, 14, 14]" = torch.ops.aten.where.self(le_44, mul_1192, getitem_387);  le_44 = mul_1192 = getitem_387 = None
    where_102: "f32[8, 216, 14, 14]" = torch.ops.aten.where.self(lt_57, full_default, where_101);  lt_57 = where_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_105: "f32[216]" = torch.ops.aten.sum.dim_IntList(where_102, [0, 2, 3])
    sub_267: "f32[8, 216, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_890);  convolution_51 = unsqueeze_890 = None
    mul_1193: "f32[8, 216, 14, 14]" = torch.ops.aten.mul.Tensor(where_102, sub_267)
    sum_106: "f32[216]" = torch.ops.aten.sum.dim_IntList(mul_1193, [0, 2, 3]);  mul_1193 = None
    mul_1194: "f32[216]" = torch.ops.aten.mul.Tensor(sum_105, 0.0006377551020408163)
    unsqueeze_891: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_1194, 0);  mul_1194 = None
    unsqueeze_892: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_891, 2);  unsqueeze_891 = None
    unsqueeze_893: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_892, 3);  unsqueeze_892 = None
    mul_1195: "f32[216]" = torch.ops.aten.mul.Tensor(sum_106, 0.0006377551020408163)
    mul_1196: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_124, squeeze_124)
    mul_1197: "f32[216]" = torch.ops.aten.mul.Tensor(mul_1195, mul_1196);  mul_1195 = mul_1196 = None
    unsqueeze_894: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_1197, 0);  mul_1197 = None
    unsqueeze_895: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_894, 2);  unsqueeze_894 = None
    unsqueeze_896: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_895, 3);  unsqueeze_895 = None
    mul_1198: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_124, primals_83);  primals_83 = None
    unsqueeze_897: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_1198, 0);  mul_1198 = None
    unsqueeze_898: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_897, 2);  unsqueeze_897 = None
    unsqueeze_899: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_898, 3);  unsqueeze_898 = None
    mul_1199: "f32[8, 216, 14, 14]" = torch.ops.aten.mul.Tensor(sub_267, unsqueeze_896);  sub_267 = unsqueeze_896 = None
    sub_269: "f32[8, 216, 14, 14]" = torch.ops.aten.sub.Tensor(where_102, mul_1199);  where_102 = mul_1199 = None
    sub_270: "f32[8, 216, 14, 14]" = torch.ops.aten.sub.Tensor(sub_269, unsqueeze_893);  sub_269 = unsqueeze_893 = None
    mul_1200: "f32[8, 216, 14, 14]" = torch.ops.aten.mul.Tensor(sub_270, unsqueeze_899);  sub_270 = unsqueeze_899 = None
    mul_1201: "f32[216]" = torch.ops.aten.mul.Tensor(sum_106, squeeze_124);  sum_106 = squeeze_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_72 = torch.ops.aten.convolution_backward.default(mul_1200, add_252, primals_238, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1200 = add_252 = primals_238 = None
    getitem_390: "f32[8, 72, 14, 14]" = convolution_backward_72[0]
    getitem_391: "f32[216, 72, 1, 1]" = convolution_backward_72[1];  convolution_backward_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_622: "f32[8, 72, 14, 14]" = torch.ops.aten.add.Tensor(add_619, getitem_390);  add_619 = getitem_390 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_107: "f32[72]" = torch.ops.aten.sum.dim_IntList(add_622, [0, 2, 3])
    sub_271: "f32[8, 72, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_902);  convolution_50 = unsqueeze_902 = None
    mul_1202: "f32[8, 72, 14, 14]" = torch.ops.aten.mul.Tensor(add_622, sub_271)
    sum_108: "f32[72]" = torch.ops.aten.sum.dim_IntList(mul_1202, [0, 2, 3]);  mul_1202 = None
    mul_1203: "f32[72]" = torch.ops.aten.mul.Tensor(sum_107, 0.0006377551020408163)
    unsqueeze_903: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_1203, 0);  mul_1203 = None
    unsqueeze_904: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_903, 2);  unsqueeze_903 = None
    unsqueeze_905: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_904, 3);  unsqueeze_904 = None
    mul_1204: "f32[72]" = torch.ops.aten.mul.Tensor(sum_108, 0.0006377551020408163)
    mul_1205: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_121, squeeze_121)
    mul_1206: "f32[72]" = torch.ops.aten.mul.Tensor(mul_1204, mul_1205);  mul_1204 = mul_1205 = None
    unsqueeze_906: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_1206, 0);  mul_1206 = None
    unsqueeze_907: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_906, 2);  unsqueeze_906 = None
    unsqueeze_908: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_907, 3);  unsqueeze_907 = None
    mul_1207: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_121, primals_81);  primals_81 = None
    unsqueeze_909: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_1207, 0);  mul_1207 = None
    unsqueeze_910: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_909, 2);  unsqueeze_909 = None
    unsqueeze_911: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_910, 3);  unsqueeze_910 = None
    mul_1208: "f32[8, 72, 14, 14]" = torch.ops.aten.mul.Tensor(sub_271, unsqueeze_908);  sub_271 = unsqueeze_908 = None
    sub_273: "f32[8, 72, 14, 14]" = torch.ops.aten.sub.Tensor(add_622, mul_1208);  mul_1208 = None
    sub_274: "f32[8, 72, 14, 14]" = torch.ops.aten.sub.Tensor(sub_273, unsqueeze_905);  sub_273 = unsqueeze_905 = None
    mul_1209: "f32[8, 72, 14, 14]" = torch.ops.aten.mul.Tensor(sub_274, unsqueeze_911);  sub_274 = unsqueeze_911 = None
    mul_1210: "f32[72]" = torch.ops.aten.mul.Tensor(sum_108, squeeze_121);  sum_108 = squeeze_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_73 = torch.ops.aten.convolution_backward.default(mul_1209, div_36, primals_237, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1209 = div_36 = primals_237 = None
    getitem_393: "f32[8, 216, 14, 14]" = convolution_backward_73[0]
    getitem_394: "f32[72, 216, 1, 1]" = convolution_backward_73[1];  convolution_backward_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_58: "b8[8, 216, 14, 14]" = torch.ops.aten.lt.Scalar(clone_31, -3)
    le_45: "b8[8, 216, 14, 14]" = torch.ops.aten.le.Scalar(clone_31, 3)
    div_154: "f32[8, 216, 14, 14]" = torch.ops.aten.div.Tensor(clone_31, 3);  clone_31 = None
    add_623: "f32[8, 216, 14, 14]" = torch.ops.aten.add.Tensor(div_154, 0.5);  div_154 = None
    mul_1211: "f32[8, 216, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_393, add_623);  add_623 = None
    where_103: "f32[8, 216, 14, 14]" = torch.ops.aten.where.self(le_45, mul_1211, getitem_393);  le_45 = mul_1211 = getitem_393 = None
    where_104: "f32[8, 216, 14, 14]" = torch.ops.aten.where.self(lt_58, full_default, where_103);  lt_58 = where_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_109: "f32[216]" = torch.ops.aten.sum.dim_IntList(where_104, [0, 2, 3])
    sub_275: "f32[8, 216, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_49, unsqueeze_914);  convolution_49 = unsqueeze_914 = None
    mul_1212: "f32[8, 216, 14, 14]" = torch.ops.aten.mul.Tensor(where_104, sub_275)
    sum_110: "f32[216]" = torch.ops.aten.sum.dim_IntList(mul_1212, [0, 2, 3]);  mul_1212 = None
    mul_1213: "f32[216]" = torch.ops.aten.mul.Tensor(sum_109, 0.0006377551020408163)
    unsqueeze_915: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_1213, 0);  mul_1213 = None
    unsqueeze_916: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_915, 2);  unsqueeze_915 = None
    unsqueeze_917: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_916, 3);  unsqueeze_916 = None
    mul_1214: "f32[216]" = torch.ops.aten.mul.Tensor(sum_110, 0.0006377551020408163)
    mul_1215: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_118, squeeze_118)
    mul_1216: "f32[216]" = torch.ops.aten.mul.Tensor(mul_1214, mul_1215);  mul_1214 = mul_1215 = None
    unsqueeze_918: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_1216, 0);  mul_1216 = None
    unsqueeze_919: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_918, 2);  unsqueeze_918 = None
    unsqueeze_920: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_919, 3);  unsqueeze_919 = None
    mul_1217: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_118, primals_79);  primals_79 = None
    unsqueeze_921: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_1217, 0);  mul_1217 = None
    unsqueeze_922: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_921, 2);  unsqueeze_921 = None
    unsqueeze_923: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_922, 3);  unsqueeze_922 = None
    mul_1218: "f32[8, 216, 14, 14]" = torch.ops.aten.mul.Tensor(sub_275, unsqueeze_920);  sub_275 = unsqueeze_920 = None
    sub_277: "f32[8, 216, 14, 14]" = torch.ops.aten.sub.Tensor(where_104, mul_1218);  where_104 = mul_1218 = None
    sub_278: "f32[8, 216, 14, 14]" = torch.ops.aten.sub.Tensor(sub_277, unsqueeze_917);  sub_277 = unsqueeze_917 = None
    mul_1219: "f32[8, 216, 14, 14]" = torch.ops.aten.mul.Tensor(sub_278, unsqueeze_923);  sub_278 = unsqueeze_923 = None
    mul_1220: "f32[216]" = torch.ops.aten.mul.Tensor(sum_110, squeeze_118);  sum_110 = squeeze_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_74 = torch.ops.aten.convolution_backward.default(mul_1219, div_35, primals_236, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 216, [True, True, False]);  mul_1219 = div_35 = primals_236 = None
    getitem_396: "f32[8, 216, 14, 14]" = convolution_backward_74[0]
    getitem_397: "f32[216, 1, 3, 3]" = convolution_backward_74[1];  convolution_backward_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_59: "b8[8, 216, 14, 14]" = torch.ops.aten.lt.Scalar(clone_30, -3)
    le_46: "b8[8, 216, 14, 14]" = torch.ops.aten.le.Scalar(clone_30, 3)
    div_155: "f32[8, 216, 14, 14]" = torch.ops.aten.div.Tensor(clone_30, 3);  clone_30 = None
    add_624: "f32[8, 216, 14, 14]" = torch.ops.aten.add.Tensor(div_155, 0.5);  div_155 = None
    mul_1221: "f32[8, 216, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_396, add_624);  add_624 = None
    where_105: "f32[8, 216, 14, 14]" = torch.ops.aten.where.self(le_46, mul_1221, getitem_396);  le_46 = mul_1221 = getitem_396 = None
    where_106: "f32[8, 216, 14, 14]" = torch.ops.aten.where.self(lt_59, full_default, where_105);  lt_59 = where_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_111: "f32[216]" = torch.ops.aten.sum.dim_IntList(where_106, [0, 2, 3])
    sub_279: "f32[8, 216, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_48, unsqueeze_926);  convolution_48 = unsqueeze_926 = None
    mul_1222: "f32[8, 216, 14, 14]" = torch.ops.aten.mul.Tensor(where_106, sub_279)
    sum_112: "f32[216]" = torch.ops.aten.sum.dim_IntList(mul_1222, [0, 2, 3]);  mul_1222 = None
    mul_1223: "f32[216]" = torch.ops.aten.mul.Tensor(sum_111, 0.0006377551020408163)
    unsqueeze_927: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_1223, 0);  mul_1223 = None
    unsqueeze_928: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_927, 2);  unsqueeze_927 = None
    unsqueeze_929: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_928, 3);  unsqueeze_928 = None
    mul_1224: "f32[216]" = torch.ops.aten.mul.Tensor(sum_112, 0.0006377551020408163)
    mul_1225: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_115, squeeze_115)
    mul_1226: "f32[216]" = torch.ops.aten.mul.Tensor(mul_1224, mul_1225);  mul_1224 = mul_1225 = None
    unsqueeze_930: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_1226, 0);  mul_1226 = None
    unsqueeze_931: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_930, 2);  unsqueeze_930 = None
    unsqueeze_932: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_931, 3);  unsqueeze_931 = None
    mul_1227: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_115, primals_77);  primals_77 = None
    unsqueeze_933: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_1227, 0);  mul_1227 = None
    unsqueeze_934: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_933, 2);  unsqueeze_933 = None
    unsqueeze_935: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_934, 3);  unsqueeze_934 = None
    mul_1228: "f32[8, 216, 14, 14]" = torch.ops.aten.mul.Tensor(sub_279, unsqueeze_932);  sub_279 = unsqueeze_932 = None
    sub_281: "f32[8, 216, 14, 14]" = torch.ops.aten.sub.Tensor(where_106, mul_1228);  where_106 = mul_1228 = None
    sub_282: "f32[8, 216, 14, 14]" = torch.ops.aten.sub.Tensor(sub_281, unsqueeze_929);  sub_281 = unsqueeze_929 = None
    mul_1229: "f32[8, 216, 14, 14]" = torch.ops.aten.mul.Tensor(sub_282, unsqueeze_935);  sub_282 = unsqueeze_935 = None
    mul_1230: "f32[216]" = torch.ops.aten.mul.Tensor(sum_112, squeeze_115);  sum_112 = squeeze_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_75 = torch.ops.aten.convolution_backward.default(mul_1229, add_234, primals_235, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1229 = add_234 = primals_235 = None
    getitem_399: "f32[8, 72, 14, 14]" = convolution_backward_75[0]
    getitem_400: "f32[216, 72, 1, 1]" = convolution_backward_75[1];  convolution_backward_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_625: "f32[8, 72, 14, 14]" = torch.ops.aten.add.Tensor(add_622, getitem_399);  add_622 = getitem_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_113: "f32[72]" = torch.ops.aten.sum.dim_IntList(add_625, [0, 2, 3])
    sub_283: "f32[8, 72, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_47, unsqueeze_938);  convolution_47 = unsqueeze_938 = None
    mul_1231: "f32[8, 72, 14, 14]" = torch.ops.aten.mul.Tensor(add_625, sub_283)
    sum_114: "f32[72]" = torch.ops.aten.sum.dim_IntList(mul_1231, [0, 2, 3]);  mul_1231 = None
    mul_1232: "f32[72]" = torch.ops.aten.mul.Tensor(sum_113, 0.0006377551020408163)
    unsqueeze_939: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_1232, 0);  mul_1232 = None
    unsqueeze_940: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_939, 2);  unsqueeze_939 = None
    unsqueeze_941: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_940, 3);  unsqueeze_940 = None
    mul_1233: "f32[72]" = torch.ops.aten.mul.Tensor(sum_114, 0.0006377551020408163)
    mul_1234: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_112, squeeze_112)
    mul_1235: "f32[72]" = torch.ops.aten.mul.Tensor(mul_1233, mul_1234);  mul_1233 = mul_1234 = None
    unsqueeze_942: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_1235, 0);  mul_1235 = None
    unsqueeze_943: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_942, 2);  unsqueeze_942 = None
    unsqueeze_944: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_943, 3);  unsqueeze_943 = None
    mul_1236: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_112, primals_75);  primals_75 = None
    unsqueeze_945: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_1236, 0);  mul_1236 = None
    unsqueeze_946: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_945, 2);  unsqueeze_945 = None
    unsqueeze_947: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_946, 3);  unsqueeze_946 = None
    mul_1237: "f32[8, 72, 14, 14]" = torch.ops.aten.mul.Tensor(sub_283, unsqueeze_944);  sub_283 = unsqueeze_944 = None
    sub_285: "f32[8, 72, 14, 14]" = torch.ops.aten.sub.Tensor(add_625, mul_1237);  mul_1237 = None
    sub_286: "f32[8, 72, 14, 14]" = torch.ops.aten.sub.Tensor(sub_285, unsqueeze_941);  sub_285 = unsqueeze_941 = None
    mul_1238: "f32[8, 72, 14, 14]" = torch.ops.aten.mul.Tensor(sub_286, unsqueeze_947);  sub_286 = unsqueeze_947 = None
    mul_1239: "f32[72]" = torch.ops.aten.mul.Tensor(sum_114, squeeze_112);  sum_114 = squeeze_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_76 = torch.ops.aten.convolution_backward.default(mul_1238, div_34, primals_234, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1238 = div_34 = primals_234 = None
    getitem_402: "f32[8, 216, 14, 14]" = convolution_backward_76[0]
    getitem_403: "f32[72, 216, 1, 1]" = convolution_backward_76[1];  convolution_backward_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_60: "b8[8, 216, 14, 14]" = torch.ops.aten.lt.Scalar(clone_29, -3)
    le_47: "b8[8, 216, 14, 14]" = torch.ops.aten.le.Scalar(clone_29, 3)
    div_156: "f32[8, 216, 14, 14]" = torch.ops.aten.div.Tensor(clone_29, 3);  clone_29 = None
    add_626: "f32[8, 216, 14, 14]" = torch.ops.aten.add.Tensor(div_156, 0.5);  div_156 = None
    mul_1240: "f32[8, 216, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_402, add_626);  add_626 = None
    where_107: "f32[8, 216, 14, 14]" = torch.ops.aten.where.self(le_47, mul_1240, getitem_402);  le_47 = mul_1240 = getitem_402 = None
    where_108: "f32[8, 216, 14, 14]" = torch.ops.aten.where.self(lt_60, full_default, where_107);  lt_60 = where_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_115: "f32[216]" = torch.ops.aten.sum.dim_IntList(where_108, [0, 2, 3])
    sub_287: "f32[8, 216, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_950);  convolution_46 = unsqueeze_950 = None
    mul_1241: "f32[8, 216, 14, 14]" = torch.ops.aten.mul.Tensor(where_108, sub_287)
    sum_116: "f32[216]" = torch.ops.aten.sum.dim_IntList(mul_1241, [0, 2, 3]);  mul_1241 = None
    mul_1242: "f32[216]" = torch.ops.aten.mul.Tensor(sum_115, 0.0006377551020408163)
    unsqueeze_951: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_1242, 0);  mul_1242 = None
    unsqueeze_952: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_951, 2);  unsqueeze_951 = None
    unsqueeze_953: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_952, 3);  unsqueeze_952 = None
    mul_1243: "f32[216]" = torch.ops.aten.mul.Tensor(sum_116, 0.0006377551020408163)
    mul_1244: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_109, squeeze_109)
    mul_1245: "f32[216]" = torch.ops.aten.mul.Tensor(mul_1243, mul_1244);  mul_1243 = mul_1244 = None
    unsqueeze_954: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_1245, 0);  mul_1245 = None
    unsqueeze_955: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_954, 2);  unsqueeze_954 = None
    unsqueeze_956: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_955, 3);  unsqueeze_955 = None
    mul_1246: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_109, primals_73);  primals_73 = None
    unsqueeze_957: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_1246, 0);  mul_1246 = None
    unsqueeze_958: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_957, 2);  unsqueeze_957 = None
    unsqueeze_959: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_958, 3);  unsqueeze_958 = None
    mul_1247: "f32[8, 216, 14, 14]" = torch.ops.aten.mul.Tensor(sub_287, unsqueeze_956);  sub_287 = unsqueeze_956 = None
    sub_289: "f32[8, 216, 14, 14]" = torch.ops.aten.sub.Tensor(where_108, mul_1247);  where_108 = mul_1247 = None
    sub_290: "f32[8, 216, 14, 14]" = torch.ops.aten.sub.Tensor(sub_289, unsqueeze_953);  sub_289 = unsqueeze_953 = None
    mul_1248: "f32[8, 216, 14, 14]" = torch.ops.aten.mul.Tensor(sub_290, unsqueeze_959);  sub_290 = unsqueeze_959 = None
    mul_1249: "f32[216]" = torch.ops.aten.mul.Tensor(sum_116, squeeze_109);  sum_116 = squeeze_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_77 = torch.ops.aten.convolution_backward.default(mul_1248, div_33, primals_233, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 216, [True, True, False]);  mul_1248 = div_33 = primals_233 = None
    getitem_405: "f32[8, 216, 14, 14]" = convolution_backward_77[0]
    getitem_406: "f32[216, 1, 3, 3]" = convolution_backward_77[1];  convolution_backward_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_61: "b8[8, 216, 14, 14]" = torch.ops.aten.lt.Scalar(clone_28, -3)
    le_48: "b8[8, 216, 14, 14]" = torch.ops.aten.le.Scalar(clone_28, 3)
    div_157: "f32[8, 216, 14, 14]" = torch.ops.aten.div.Tensor(clone_28, 3);  clone_28 = None
    add_627: "f32[8, 216, 14, 14]" = torch.ops.aten.add.Tensor(div_157, 0.5);  div_157 = None
    mul_1250: "f32[8, 216, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_405, add_627);  add_627 = None
    where_109: "f32[8, 216, 14, 14]" = torch.ops.aten.where.self(le_48, mul_1250, getitem_405);  le_48 = mul_1250 = getitem_405 = None
    where_110: "f32[8, 216, 14, 14]" = torch.ops.aten.where.self(lt_61, full_default, where_109);  lt_61 = where_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_117: "f32[216]" = torch.ops.aten.sum.dim_IntList(where_110, [0, 2, 3])
    sub_291: "f32[8, 216, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_962);  convolution_45 = unsqueeze_962 = None
    mul_1251: "f32[8, 216, 14, 14]" = torch.ops.aten.mul.Tensor(where_110, sub_291)
    sum_118: "f32[216]" = torch.ops.aten.sum.dim_IntList(mul_1251, [0, 2, 3]);  mul_1251 = None
    mul_1252: "f32[216]" = torch.ops.aten.mul.Tensor(sum_117, 0.0006377551020408163)
    unsqueeze_963: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_1252, 0);  mul_1252 = None
    unsqueeze_964: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_963, 2);  unsqueeze_963 = None
    unsqueeze_965: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_964, 3);  unsqueeze_964 = None
    mul_1253: "f32[216]" = torch.ops.aten.mul.Tensor(sum_118, 0.0006377551020408163)
    mul_1254: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_106, squeeze_106)
    mul_1255: "f32[216]" = torch.ops.aten.mul.Tensor(mul_1253, mul_1254);  mul_1253 = mul_1254 = None
    unsqueeze_966: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_1255, 0);  mul_1255 = None
    unsqueeze_967: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_966, 2);  unsqueeze_966 = None
    unsqueeze_968: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_967, 3);  unsqueeze_967 = None
    mul_1256: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_106, primals_71);  primals_71 = None
    unsqueeze_969: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_1256, 0);  mul_1256 = None
    unsqueeze_970: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_969, 2);  unsqueeze_969 = None
    unsqueeze_971: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_970, 3);  unsqueeze_970 = None
    mul_1257: "f32[8, 216, 14, 14]" = torch.ops.aten.mul.Tensor(sub_291, unsqueeze_968);  sub_291 = unsqueeze_968 = None
    sub_293: "f32[8, 216, 14, 14]" = torch.ops.aten.sub.Tensor(where_110, mul_1257);  where_110 = mul_1257 = None
    sub_294: "f32[8, 216, 14, 14]" = torch.ops.aten.sub.Tensor(sub_293, unsqueeze_965);  sub_293 = unsqueeze_965 = None
    mul_1258: "f32[8, 216, 14, 14]" = torch.ops.aten.mul.Tensor(sub_294, unsqueeze_971);  sub_294 = unsqueeze_971 = None
    mul_1259: "f32[216]" = torch.ops.aten.mul.Tensor(sum_118, squeeze_106);  sum_118 = squeeze_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_78 = torch.ops.aten.convolution_backward.default(mul_1258, add_216, primals_232, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1258 = add_216 = primals_232 = None
    getitem_408: "f32[8, 72, 14, 14]" = convolution_backward_78[0]
    getitem_409: "f32[216, 72, 1, 1]" = convolution_backward_78[1];  convolution_backward_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_628: "f32[8, 72, 14, 14]" = torch.ops.aten.add.Tensor(add_625, getitem_408);  add_625 = getitem_408 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_119: "f32[72]" = torch.ops.aten.sum.dim_IntList(add_628, [0, 2, 3])
    sub_295: "f32[8, 72, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_44, unsqueeze_974);  convolution_44 = unsqueeze_974 = None
    mul_1260: "f32[8, 72, 14, 14]" = torch.ops.aten.mul.Tensor(add_628, sub_295)
    sum_120: "f32[72]" = torch.ops.aten.sum.dim_IntList(mul_1260, [0, 2, 3]);  mul_1260 = None
    mul_1261: "f32[72]" = torch.ops.aten.mul.Tensor(sum_119, 0.0006377551020408163)
    unsqueeze_975: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_1261, 0);  mul_1261 = None
    unsqueeze_976: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_975, 2);  unsqueeze_975 = None
    unsqueeze_977: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_976, 3);  unsqueeze_976 = None
    mul_1262: "f32[72]" = torch.ops.aten.mul.Tensor(sum_120, 0.0006377551020408163)
    mul_1263: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_103, squeeze_103)
    mul_1264: "f32[72]" = torch.ops.aten.mul.Tensor(mul_1262, mul_1263);  mul_1262 = mul_1263 = None
    unsqueeze_978: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_1264, 0);  mul_1264 = None
    unsqueeze_979: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_978, 2);  unsqueeze_978 = None
    unsqueeze_980: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_979, 3);  unsqueeze_979 = None
    mul_1265: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_103, primals_69);  primals_69 = None
    unsqueeze_981: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_1265, 0);  mul_1265 = None
    unsqueeze_982: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_981, 2);  unsqueeze_981 = None
    unsqueeze_983: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_982, 3);  unsqueeze_982 = None
    mul_1266: "f32[8, 72, 14, 14]" = torch.ops.aten.mul.Tensor(sub_295, unsqueeze_980);  sub_295 = unsqueeze_980 = None
    sub_297: "f32[8, 72, 14, 14]" = torch.ops.aten.sub.Tensor(add_628, mul_1266);  add_628 = mul_1266 = None
    sub_298: "f32[8, 72, 14, 14]" = torch.ops.aten.sub.Tensor(sub_297, unsqueeze_977);  sub_297 = unsqueeze_977 = None
    mul_1267: "f32[8, 72, 14, 14]" = torch.ops.aten.mul.Tensor(sub_298, unsqueeze_983);  sub_298 = unsqueeze_983 = None
    mul_1268: "f32[72]" = torch.ops.aten.mul.Tensor(sum_120, squeeze_103);  sum_120 = squeeze_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_79 = torch.ops.aten.convolution_backward.default(mul_1267, div_32, primals_231, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1267 = div_32 = primals_231 = None
    getitem_411: "f32[8, 200, 14, 14]" = convolution_backward_79[0]
    getitem_412: "f32[72, 200, 1, 1]" = convolution_backward_79[1];  convolution_backward_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_62: "b8[8, 200, 14, 14]" = torch.ops.aten.lt.Scalar(clone_27, -3)
    le_49: "b8[8, 200, 14, 14]" = torch.ops.aten.le.Scalar(clone_27, 3)
    div_158: "f32[8, 200, 14, 14]" = torch.ops.aten.div.Tensor(clone_27, 3);  clone_27 = None
    add_629: "f32[8, 200, 14, 14]" = torch.ops.aten.add.Tensor(div_158, 0.5);  div_158 = None
    mul_1269: "f32[8, 200, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_411, add_629);  add_629 = None
    where_111: "f32[8, 200, 14, 14]" = torch.ops.aten.where.self(le_49, mul_1269, getitem_411);  le_49 = mul_1269 = getitem_411 = None
    where_112: "f32[8, 200, 14, 14]" = torch.ops.aten.where.self(lt_62, full_default, where_111);  lt_62 = where_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_121: "f32[200]" = torch.ops.aten.sum.dim_IntList(where_112, [0, 2, 3])
    sub_299: "f32[8, 200, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_43, unsqueeze_986);  convolution_43 = unsqueeze_986 = None
    mul_1270: "f32[8, 200, 14, 14]" = torch.ops.aten.mul.Tensor(where_112, sub_299)
    sum_122: "f32[200]" = torch.ops.aten.sum.dim_IntList(mul_1270, [0, 2, 3]);  mul_1270 = None
    mul_1271: "f32[200]" = torch.ops.aten.mul.Tensor(sum_121, 0.0006377551020408163)
    unsqueeze_987: "f32[1, 200]" = torch.ops.aten.unsqueeze.default(mul_1271, 0);  mul_1271 = None
    unsqueeze_988: "f32[1, 200, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_987, 2);  unsqueeze_987 = None
    unsqueeze_989: "f32[1, 200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_988, 3);  unsqueeze_988 = None
    mul_1272: "f32[200]" = torch.ops.aten.mul.Tensor(sum_122, 0.0006377551020408163)
    mul_1273: "f32[200]" = torch.ops.aten.mul.Tensor(squeeze_100, squeeze_100)
    mul_1274: "f32[200]" = torch.ops.aten.mul.Tensor(mul_1272, mul_1273);  mul_1272 = mul_1273 = None
    unsqueeze_990: "f32[1, 200]" = torch.ops.aten.unsqueeze.default(mul_1274, 0);  mul_1274 = None
    unsqueeze_991: "f32[1, 200, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_990, 2);  unsqueeze_990 = None
    unsqueeze_992: "f32[1, 200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_991, 3);  unsqueeze_991 = None
    mul_1275: "f32[200]" = torch.ops.aten.mul.Tensor(squeeze_100, primals_67);  primals_67 = None
    unsqueeze_993: "f32[1, 200]" = torch.ops.aten.unsqueeze.default(mul_1275, 0);  mul_1275 = None
    unsqueeze_994: "f32[1, 200, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_993, 2);  unsqueeze_993 = None
    unsqueeze_995: "f32[1, 200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_994, 3);  unsqueeze_994 = None
    mul_1276: "f32[8, 200, 14, 14]" = torch.ops.aten.mul.Tensor(sub_299, unsqueeze_992);  sub_299 = unsqueeze_992 = None
    sub_301: "f32[8, 200, 14, 14]" = torch.ops.aten.sub.Tensor(where_112, mul_1276);  where_112 = mul_1276 = None
    sub_302: "f32[8, 200, 14, 14]" = torch.ops.aten.sub.Tensor(sub_301, unsqueeze_989);  sub_301 = unsqueeze_989 = None
    mul_1277: "f32[8, 200, 14, 14]" = torch.ops.aten.mul.Tensor(sub_302, unsqueeze_995);  sub_302 = unsqueeze_995 = None
    mul_1278: "f32[200]" = torch.ops.aten.mul.Tensor(sum_122, squeeze_100);  sum_122 = squeeze_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_80 = torch.ops.aten.convolution_backward.default(mul_1277, div_31, primals_230, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 200, [True, True, False]);  mul_1277 = div_31 = primals_230 = None
    getitem_414: "f32[8, 200, 28, 28]" = convolution_backward_80[0]
    getitem_415: "f32[200, 1, 5, 5]" = convolution_backward_80[1];  convolution_backward_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_63: "b8[8, 200, 28, 28]" = torch.ops.aten.lt.Scalar(clone_26, -3)
    le_50: "b8[8, 200, 28, 28]" = torch.ops.aten.le.Scalar(clone_26, 3)
    div_159: "f32[8, 200, 28, 28]" = torch.ops.aten.div.Tensor(clone_26, 3);  clone_26 = None
    add_630: "f32[8, 200, 28, 28]" = torch.ops.aten.add.Tensor(div_159, 0.5);  div_159 = None
    mul_1279: "f32[8, 200, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_414, add_630);  add_630 = None
    where_113: "f32[8, 200, 28, 28]" = torch.ops.aten.where.self(le_50, mul_1279, getitem_414);  le_50 = mul_1279 = getitem_414 = None
    where_114: "f32[8, 200, 28, 28]" = torch.ops.aten.where.self(lt_63, full_default, where_113);  lt_63 = where_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_123: "f32[200]" = torch.ops.aten.sum.dim_IntList(where_114, [0, 2, 3])
    sub_303: "f32[8, 200, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_42, unsqueeze_998);  convolution_42 = unsqueeze_998 = None
    mul_1280: "f32[8, 200, 28, 28]" = torch.ops.aten.mul.Tensor(where_114, sub_303)
    sum_124: "f32[200]" = torch.ops.aten.sum.dim_IntList(mul_1280, [0, 2, 3]);  mul_1280 = None
    mul_1281: "f32[200]" = torch.ops.aten.mul.Tensor(sum_123, 0.00015943877551020407)
    unsqueeze_999: "f32[1, 200]" = torch.ops.aten.unsqueeze.default(mul_1281, 0);  mul_1281 = None
    unsqueeze_1000: "f32[1, 200, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_999, 2);  unsqueeze_999 = None
    unsqueeze_1001: "f32[1, 200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1000, 3);  unsqueeze_1000 = None
    mul_1282: "f32[200]" = torch.ops.aten.mul.Tensor(sum_124, 0.00015943877551020407)
    mul_1283: "f32[200]" = torch.ops.aten.mul.Tensor(squeeze_97, squeeze_97)
    mul_1284: "f32[200]" = torch.ops.aten.mul.Tensor(mul_1282, mul_1283);  mul_1282 = mul_1283 = None
    unsqueeze_1002: "f32[1, 200]" = torch.ops.aten.unsqueeze.default(mul_1284, 0);  mul_1284 = None
    unsqueeze_1003: "f32[1, 200, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1002, 2);  unsqueeze_1002 = None
    unsqueeze_1004: "f32[1, 200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1003, 3);  unsqueeze_1003 = None
    mul_1285: "f32[200]" = torch.ops.aten.mul.Tensor(squeeze_97, primals_65);  primals_65 = None
    unsqueeze_1005: "f32[1, 200]" = torch.ops.aten.unsqueeze.default(mul_1285, 0);  mul_1285 = None
    unsqueeze_1006: "f32[1, 200, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1005, 2);  unsqueeze_1005 = None
    unsqueeze_1007: "f32[1, 200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1006, 3);  unsqueeze_1006 = None
    mul_1286: "f32[8, 200, 28, 28]" = torch.ops.aten.mul.Tensor(sub_303, unsqueeze_1004);  sub_303 = unsqueeze_1004 = None
    sub_305: "f32[8, 200, 28, 28]" = torch.ops.aten.sub.Tensor(where_114, mul_1286);  where_114 = mul_1286 = None
    sub_306: "f32[8, 200, 28, 28]" = torch.ops.aten.sub.Tensor(sub_305, unsqueeze_1001);  sub_305 = unsqueeze_1001 = None
    mul_1287: "f32[8, 200, 28, 28]" = torch.ops.aten.mul.Tensor(sub_306, unsqueeze_1007);  sub_306 = unsqueeze_1007 = None
    mul_1288: "f32[200]" = torch.ops.aten.mul.Tensor(sum_124, squeeze_97);  sum_124 = squeeze_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_81 = torch.ops.aten.convolution_backward.default(mul_1287, add_199, primals_229, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1287 = add_199 = primals_229 = None
    getitem_417: "f32[8, 40, 28, 28]" = convolution_backward_81[0]
    getitem_418: "f32[200, 40, 1, 1]" = convolution_backward_81[1];  convolution_backward_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_125: "f32[40]" = torch.ops.aten.sum.dim_IntList(getitem_417, [0, 2, 3])
    sub_307: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_1010);  convolution_41 = unsqueeze_1010 = None
    mul_1289: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_417, sub_307)
    sum_126: "f32[40]" = torch.ops.aten.sum.dim_IntList(mul_1289, [0, 2, 3]);  mul_1289 = None
    mul_1290: "f32[40]" = torch.ops.aten.mul.Tensor(sum_125, 0.00015943877551020407)
    unsqueeze_1011: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_1290, 0);  mul_1290 = None
    unsqueeze_1012: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1011, 2);  unsqueeze_1011 = None
    unsqueeze_1013: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1012, 3);  unsqueeze_1012 = None
    mul_1291: "f32[40]" = torch.ops.aten.mul.Tensor(sum_126, 0.00015943877551020407)
    mul_1292: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_94, squeeze_94)
    mul_1293: "f32[40]" = torch.ops.aten.mul.Tensor(mul_1291, mul_1292);  mul_1291 = mul_1292 = None
    unsqueeze_1014: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_1293, 0);  mul_1293 = None
    unsqueeze_1015: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1014, 2);  unsqueeze_1014 = None
    unsqueeze_1016: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1015, 3);  unsqueeze_1015 = None
    mul_1294: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_94, primals_63);  primals_63 = None
    unsqueeze_1017: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_1294, 0);  mul_1294 = None
    unsqueeze_1018: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1017, 2);  unsqueeze_1017 = None
    unsqueeze_1019: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1018, 3);  unsqueeze_1018 = None
    mul_1295: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_307, unsqueeze_1016);  sub_307 = unsqueeze_1016 = None
    sub_309: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(getitem_417, mul_1295);  mul_1295 = None
    sub_310: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(sub_309, unsqueeze_1013);  sub_309 = unsqueeze_1013 = None
    mul_1296: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_310, unsqueeze_1019);  sub_310 = unsqueeze_1019 = None
    mul_1297: "f32[40]" = torch.ops.aten.mul.Tensor(sum_126, squeeze_94);  sum_126 = squeeze_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_82 = torch.ops.aten.convolution_backward.default(mul_1296, mul_247, primals_228, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1296 = mul_247 = primals_228 = None
    getitem_420: "f32[8, 120, 28, 28]" = convolution_backward_82[0]
    getitem_421: "f32[40, 120, 1, 1]" = convolution_backward_82[1];  convolution_backward_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_1298: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_420, div_28);  div_28 = None
    mul_1299: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_420, div_30);  getitem_420 = div_30 = None
    sum_127: "f32[8, 120, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1298, [2, 3], True);  mul_1298 = None
    mul_1300: "f32[8, 120, 1, 1]" = torch.ops.aten.mul.Tensor(sum_127, 0.16666666666666666);  sum_127 = None
    where_115: "f32[8, 120, 1, 1]" = torch.ops.aten.where.self(bitwise_and_13, mul_1300, full_default);  bitwise_and_13 = mul_1300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_83 = torch.ops.aten.convolution_backward.default(where_115, div_29, primals_226, [120], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_115 = div_29 = primals_226 = None
    getitem_423: "f32[8, 16, 1, 1]" = convolution_backward_83[0]
    getitem_424: "f32[120, 16, 1, 1]" = convolution_backward_83[1]
    getitem_425: "f32[120]" = convolution_backward_83[2];  convolution_backward_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    lt_65: "b8[8, 16, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_39, -3)
    le_51: "b8[8, 16, 1, 1]" = torch.ops.aten.le.Scalar(convolution_39, 3)
    div_160: "f32[8, 16, 1, 1]" = torch.ops.aten.div.Tensor(convolution_39, 3);  convolution_39 = None
    add_631: "f32[8, 16, 1, 1]" = torch.ops.aten.add.Tensor(div_160, 0.5);  div_160 = None
    mul_1301: "f32[8, 16, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_423, add_631);  add_631 = None
    where_116: "f32[8, 16, 1, 1]" = torch.ops.aten.where.self(le_51, mul_1301, getitem_423);  le_51 = mul_1301 = getitem_423 = None
    where_117: "f32[8, 16, 1, 1]" = torch.ops.aten.where.self(lt_65, full_default, where_116);  lt_65 = where_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_84 = torch.ops.aten.convolution_backward.default(where_117, mean_4, primals_224, [16], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_117 = mean_4 = primals_224 = None
    getitem_426: "f32[8, 120, 1, 1]" = convolution_backward_84[0]
    getitem_427: "f32[16, 120, 1, 1]" = convolution_backward_84[1]
    getitem_428: "f32[16]" = convolution_backward_84[2];  convolution_backward_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_14: "f32[8, 120, 28, 28]" = torch.ops.aten.expand.default(getitem_426, [8, 120, 28, 28]);  getitem_426 = None
    div_161: "f32[8, 120, 28, 28]" = torch.ops.aten.div.Scalar(expand_14, 784);  expand_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_632: "f32[8, 120, 28, 28]" = torch.ops.aten.add.Tensor(mul_1299, div_161);  mul_1299 = div_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_66: "b8[8, 120, 28, 28]" = torch.ops.aten.lt.Scalar(clone_24, -3)
    le_52: "b8[8, 120, 28, 28]" = torch.ops.aten.le.Scalar(clone_24, 3)
    div_162: "f32[8, 120, 28, 28]" = torch.ops.aten.div.Tensor(clone_24, 3);  clone_24 = None
    add_633: "f32[8, 120, 28, 28]" = torch.ops.aten.add.Tensor(div_162, 0.5);  div_162 = None
    mul_1302: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(add_632, add_633);  add_633 = None
    where_118: "f32[8, 120, 28, 28]" = torch.ops.aten.where.self(le_52, mul_1302, add_632);  le_52 = mul_1302 = add_632 = None
    where_119: "f32[8, 120, 28, 28]" = torch.ops.aten.where.self(lt_66, full_default, where_118);  lt_66 = where_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_128: "f32[120]" = torch.ops.aten.sum.dim_IntList(where_119, [0, 2, 3])
    sub_311: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_38, unsqueeze_1022);  convolution_38 = unsqueeze_1022 = None
    mul_1303: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(where_119, sub_311)
    sum_129: "f32[120]" = torch.ops.aten.sum.dim_IntList(mul_1303, [0, 2, 3]);  mul_1303 = None
    mul_1304: "f32[120]" = torch.ops.aten.mul.Tensor(sum_128, 0.00015943877551020407)
    unsqueeze_1023: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1304, 0);  mul_1304 = None
    unsqueeze_1024: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1023, 2);  unsqueeze_1023 = None
    unsqueeze_1025: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1024, 3);  unsqueeze_1024 = None
    mul_1305: "f32[120]" = torch.ops.aten.mul.Tensor(sum_129, 0.00015943877551020407)
    mul_1306: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_91, squeeze_91)
    mul_1307: "f32[120]" = torch.ops.aten.mul.Tensor(mul_1305, mul_1306);  mul_1305 = mul_1306 = None
    unsqueeze_1026: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1307, 0);  mul_1307 = None
    unsqueeze_1027: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1026, 2);  unsqueeze_1026 = None
    unsqueeze_1028: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1027, 3);  unsqueeze_1027 = None
    mul_1308: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_91, primals_61);  primals_61 = None
    unsqueeze_1029: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1308, 0);  mul_1308 = None
    unsqueeze_1030: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1029, 2);  unsqueeze_1029 = None
    unsqueeze_1031: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1030, 3);  unsqueeze_1030 = None
    mul_1309: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_311, unsqueeze_1028);  sub_311 = unsqueeze_1028 = None
    sub_313: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(where_119, mul_1309);  where_119 = mul_1309 = None
    sub_314: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(sub_313, unsqueeze_1025);  sub_313 = unsqueeze_1025 = None
    mul_1310: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_314, unsqueeze_1031);  sub_314 = unsqueeze_1031 = None
    mul_1311: "f32[120]" = torch.ops.aten.mul.Tensor(sum_129, squeeze_91);  sum_129 = squeeze_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_85 = torch.ops.aten.convolution_backward.default(mul_1310, div_27, primals_223, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 120, [True, True, False]);  mul_1310 = div_27 = primals_223 = None
    getitem_429: "f32[8, 120, 28, 28]" = convolution_backward_85[0]
    getitem_430: "f32[120, 1, 5, 5]" = convolution_backward_85[1];  convolution_backward_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_67: "b8[8, 120, 28, 28]" = torch.ops.aten.lt.Scalar(clone_23, -3)
    le_53: "b8[8, 120, 28, 28]" = torch.ops.aten.le.Scalar(clone_23, 3)
    div_163: "f32[8, 120, 28, 28]" = torch.ops.aten.div.Tensor(clone_23, 3);  clone_23 = None
    add_634: "f32[8, 120, 28, 28]" = torch.ops.aten.add.Tensor(div_163, 0.5);  div_163 = None
    mul_1312: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_429, add_634);  add_634 = None
    where_120: "f32[8, 120, 28, 28]" = torch.ops.aten.where.self(le_53, mul_1312, getitem_429);  le_53 = mul_1312 = getitem_429 = None
    where_121: "f32[8, 120, 28, 28]" = torch.ops.aten.where.self(lt_67, full_default, where_120);  lt_67 = where_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_130: "f32[120]" = torch.ops.aten.sum.dim_IntList(where_121, [0, 2, 3])
    sub_315: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_37, unsqueeze_1034);  convolution_37 = unsqueeze_1034 = None
    mul_1313: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(where_121, sub_315)
    sum_131: "f32[120]" = torch.ops.aten.sum.dim_IntList(mul_1313, [0, 2, 3]);  mul_1313 = None
    mul_1314: "f32[120]" = torch.ops.aten.mul.Tensor(sum_130, 0.00015943877551020407)
    unsqueeze_1035: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1314, 0);  mul_1314 = None
    unsqueeze_1036: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1035, 2);  unsqueeze_1035 = None
    unsqueeze_1037: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1036, 3);  unsqueeze_1036 = None
    mul_1315: "f32[120]" = torch.ops.aten.mul.Tensor(sum_131, 0.00015943877551020407)
    mul_1316: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_88, squeeze_88)
    mul_1317: "f32[120]" = torch.ops.aten.mul.Tensor(mul_1315, mul_1316);  mul_1315 = mul_1316 = None
    unsqueeze_1038: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1317, 0);  mul_1317 = None
    unsqueeze_1039: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1038, 2);  unsqueeze_1038 = None
    unsqueeze_1040: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1039, 3);  unsqueeze_1039 = None
    mul_1318: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_88, primals_59);  primals_59 = None
    unsqueeze_1041: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1318, 0);  mul_1318 = None
    unsqueeze_1042: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1041, 2);  unsqueeze_1041 = None
    unsqueeze_1043: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1042, 3);  unsqueeze_1042 = None
    mul_1319: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_315, unsqueeze_1040);  sub_315 = unsqueeze_1040 = None
    sub_317: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(where_121, mul_1319);  where_121 = mul_1319 = None
    sub_318: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(sub_317, unsqueeze_1037);  sub_317 = unsqueeze_1037 = None
    mul_1320: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_318, unsqueeze_1043);  sub_318 = unsqueeze_1043 = None
    mul_1321: "f32[120]" = torch.ops.aten.mul.Tensor(sum_131, squeeze_88);  sum_131 = squeeze_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_86 = torch.ops.aten.convolution_backward.default(mul_1320, add_179, primals_222, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1320 = add_179 = primals_222 = None
    getitem_432: "f32[8, 40, 28, 28]" = convolution_backward_86[0]
    getitem_433: "f32[120, 40, 1, 1]" = convolution_backward_86[1];  convolution_backward_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_635: "f32[8, 40, 28, 28]" = torch.ops.aten.add.Tensor(getitem_417, getitem_432);  getitem_417 = getitem_432 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_132: "f32[40]" = torch.ops.aten.sum.dim_IntList(add_635, [0, 2, 3])
    sub_319: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_1046);  convolution_36 = unsqueeze_1046 = None
    mul_1322: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(add_635, sub_319)
    sum_133: "f32[40]" = torch.ops.aten.sum.dim_IntList(mul_1322, [0, 2, 3]);  mul_1322 = None
    mul_1323: "f32[40]" = torch.ops.aten.mul.Tensor(sum_132, 0.00015943877551020407)
    unsqueeze_1047: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_1323, 0);  mul_1323 = None
    unsqueeze_1048: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1047, 2);  unsqueeze_1047 = None
    unsqueeze_1049: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1048, 3);  unsqueeze_1048 = None
    mul_1324: "f32[40]" = torch.ops.aten.mul.Tensor(sum_133, 0.00015943877551020407)
    mul_1325: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_85, squeeze_85)
    mul_1326: "f32[40]" = torch.ops.aten.mul.Tensor(mul_1324, mul_1325);  mul_1324 = mul_1325 = None
    unsqueeze_1050: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_1326, 0);  mul_1326 = None
    unsqueeze_1051: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1050, 2);  unsqueeze_1050 = None
    unsqueeze_1052: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1051, 3);  unsqueeze_1051 = None
    mul_1327: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_85, primals_57);  primals_57 = None
    unsqueeze_1053: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_1327, 0);  mul_1327 = None
    unsqueeze_1054: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1053, 2);  unsqueeze_1053 = None
    unsqueeze_1055: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1054, 3);  unsqueeze_1054 = None
    mul_1328: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_319, unsqueeze_1052);  sub_319 = unsqueeze_1052 = None
    sub_321: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(add_635, mul_1328);  mul_1328 = None
    sub_322: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(sub_321, unsqueeze_1049);  sub_321 = unsqueeze_1049 = None
    mul_1329: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_322, unsqueeze_1055);  sub_322 = unsqueeze_1055 = None
    mul_1330: "f32[40]" = torch.ops.aten.mul.Tensor(sum_133, squeeze_85);  sum_133 = squeeze_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_87 = torch.ops.aten.convolution_backward.default(mul_1329, mul_222, primals_221, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1329 = mul_222 = primals_221 = None
    getitem_435: "f32[8, 120, 28, 28]" = convolution_backward_87[0]
    getitem_436: "f32[40, 120, 1, 1]" = convolution_backward_87[1];  convolution_backward_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_1331: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_435, div_24);  div_24 = None
    mul_1332: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_435, div_26);  getitem_435 = div_26 = None
    sum_134: "f32[8, 120, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1331, [2, 3], True);  mul_1331 = None
    mul_1333: "f32[8, 120, 1, 1]" = torch.ops.aten.mul.Tensor(sum_134, 0.16666666666666666);  sum_134 = None
    where_122: "f32[8, 120, 1, 1]" = torch.ops.aten.where.self(bitwise_and_14, mul_1333, full_default);  bitwise_and_14 = mul_1333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_88 = torch.ops.aten.convolution_backward.default(where_122, div_25, primals_219, [120], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_122 = div_25 = primals_219 = None
    getitem_438: "f32[8, 16, 1, 1]" = convolution_backward_88[0]
    getitem_439: "f32[120, 16, 1, 1]" = convolution_backward_88[1]
    getitem_440: "f32[120]" = convolution_backward_88[2];  convolution_backward_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    lt_69: "b8[8, 16, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_34, -3)
    le_54: "b8[8, 16, 1, 1]" = torch.ops.aten.le.Scalar(convolution_34, 3)
    div_164: "f32[8, 16, 1, 1]" = torch.ops.aten.div.Tensor(convolution_34, 3);  convolution_34 = None
    add_636: "f32[8, 16, 1, 1]" = torch.ops.aten.add.Tensor(div_164, 0.5);  div_164 = None
    mul_1334: "f32[8, 16, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_438, add_636);  add_636 = None
    where_123: "f32[8, 16, 1, 1]" = torch.ops.aten.where.self(le_54, mul_1334, getitem_438);  le_54 = mul_1334 = getitem_438 = None
    where_124: "f32[8, 16, 1, 1]" = torch.ops.aten.where.self(lt_69, full_default, where_123);  lt_69 = where_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_89 = torch.ops.aten.convolution_backward.default(where_124, mean_3, primals_217, [16], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_124 = mean_3 = primals_217 = None
    getitem_441: "f32[8, 120, 1, 1]" = convolution_backward_89[0]
    getitem_442: "f32[16, 120, 1, 1]" = convolution_backward_89[1]
    getitem_443: "f32[16]" = convolution_backward_89[2];  convolution_backward_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_15: "f32[8, 120, 28, 28]" = torch.ops.aten.expand.default(getitem_441, [8, 120, 28, 28]);  getitem_441 = None
    div_165: "f32[8, 120, 28, 28]" = torch.ops.aten.div.Scalar(expand_15, 784);  expand_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_637: "f32[8, 120, 28, 28]" = torch.ops.aten.add.Tensor(mul_1332, div_165);  mul_1332 = div_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_70: "b8[8, 120, 28, 28]" = torch.ops.aten.lt.Scalar(clone_21, -3)
    le_55: "b8[8, 120, 28, 28]" = torch.ops.aten.le.Scalar(clone_21, 3)
    div_166: "f32[8, 120, 28, 28]" = torch.ops.aten.div.Tensor(clone_21, 3);  clone_21 = None
    add_638: "f32[8, 120, 28, 28]" = torch.ops.aten.add.Tensor(div_166, 0.5);  div_166 = None
    mul_1335: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(add_637, add_638);  add_638 = None
    where_125: "f32[8, 120, 28, 28]" = torch.ops.aten.where.self(le_55, mul_1335, add_637);  le_55 = mul_1335 = add_637 = None
    where_126: "f32[8, 120, 28, 28]" = torch.ops.aten.where.self(lt_70, full_default, where_125);  lt_70 = where_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_135: "f32[120]" = torch.ops.aten.sum.dim_IntList(where_126, [0, 2, 3])
    sub_323: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_1058);  convolution_33 = unsqueeze_1058 = None
    mul_1336: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(where_126, sub_323)
    sum_136: "f32[120]" = torch.ops.aten.sum.dim_IntList(mul_1336, [0, 2, 3]);  mul_1336 = None
    mul_1337: "f32[120]" = torch.ops.aten.mul.Tensor(sum_135, 0.00015943877551020407)
    unsqueeze_1059: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1337, 0);  mul_1337 = None
    unsqueeze_1060: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1059, 2);  unsqueeze_1059 = None
    unsqueeze_1061: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1060, 3);  unsqueeze_1060 = None
    mul_1338: "f32[120]" = torch.ops.aten.mul.Tensor(sum_136, 0.00015943877551020407)
    mul_1339: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_82, squeeze_82)
    mul_1340: "f32[120]" = torch.ops.aten.mul.Tensor(mul_1338, mul_1339);  mul_1338 = mul_1339 = None
    unsqueeze_1062: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1340, 0);  mul_1340 = None
    unsqueeze_1063: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1062, 2);  unsqueeze_1062 = None
    unsqueeze_1064: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1063, 3);  unsqueeze_1063 = None
    mul_1341: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_82, primals_55);  primals_55 = None
    unsqueeze_1065: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1341, 0);  mul_1341 = None
    unsqueeze_1066: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1065, 2);  unsqueeze_1065 = None
    unsqueeze_1067: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1066, 3);  unsqueeze_1066 = None
    mul_1342: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_323, unsqueeze_1064);  sub_323 = unsqueeze_1064 = None
    sub_325: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(where_126, mul_1342);  where_126 = mul_1342 = None
    sub_326: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(sub_325, unsqueeze_1061);  sub_325 = unsqueeze_1061 = None
    mul_1343: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_326, unsqueeze_1067);  sub_326 = unsqueeze_1067 = None
    mul_1344: "f32[120]" = torch.ops.aten.mul.Tensor(sum_136, squeeze_82);  sum_136 = squeeze_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_90 = torch.ops.aten.convolution_backward.default(mul_1343, div_23, primals_216, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 120, [True, True, False]);  mul_1343 = div_23 = primals_216 = None
    getitem_444: "f32[8, 120, 28, 28]" = convolution_backward_90[0]
    getitem_445: "f32[120, 1, 5, 5]" = convolution_backward_90[1];  convolution_backward_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_71: "b8[8, 120, 28, 28]" = torch.ops.aten.lt.Scalar(clone_20, -3)
    le_56: "b8[8, 120, 28, 28]" = torch.ops.aten.le.Scalar(clone_20, 3)
    div_167: "f32[8, 120, 28, 28]" = torch.ops.aten.div.Tensor(clone_20, 3);  clone_20 = None
    add_639: "f32[8, 120, 28, 28]" = torch.ops.aten.add.Tensor(div_167, 0.5);  div_167 = None
    mul_1345: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_444, add_639);  add_639 = None
    where_127: "f32[8, 120, 28, 28]" = torch.ops.aten.where.self(le_56, mul_1345, getitem_444);  le_56 = mul_1345 = getitem_444 = None
    where_128: "f32[8, 120, 28, 28]" = torch.ops.aten.where.self(lt_71, full_default, where_127);  lt_71 = where_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_137: "f32[120]" = torch.ops.aten.sum.dim_IntList(where_128, [0, 2, 3])
    sub_327: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_1070);  convolution_32 = unsqueeze_1070 = None
    mul_1346: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(where_128, sub_327)
    sum_138: "f32[120]" = torch.ops.aten.sum.dim_IntList(mul_1346, [0, 2, 3]);  mul_1346 = None
    mul_1347: "f32[120]" = torch.ops.aten.mul.Tensor(sum_137, 0.00015943877551020407)
    unsqueeze_1071: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1347, 0);  mul_1347 = None
    unsqueeze_1072: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1071, 2);  unsqueeze_1071 = None
    unsqueeze_1073: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1072, 3);  unsqueeze_1072 = None
    mul_1348: "f32[120]" = torch.ops.aten.mul.Tensor(sum_138, 0.00015943877551020407)
    mul_1349: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_79, squeeze_79)
    mul_1350: "f32[120]" = torch.ops.aten.mul.Tensor(mul_1348, mul_1349);  mul_1348 = mul_1349 = None
    unsqueeze_1074: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1350, 0);  mul_1350 = None
    unsqueeze_1075: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1074, 2);  unsqueeze_1074 = None
    unsqueeze_1076: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1075, 3);  unsqueeze_1075 = None
    mul_1351: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_79, primals_53);  primals_53 = None
    unsqueeze_1077: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1351, 0);  mul_1351 = None
    unsqueeze_1078: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1077, 2);  unsqueeze_1077 = None
    unsqueeze_1079: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1078, 3);  unsqueeze_1078 = None
    mul_1352: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_327, unsqueeze_1076);  sub_327 = unsqueeze_1076 = None
    sub_329: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(where_128, mul_1352);  where_128 = mul_1352 = None
    sub_330: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(sub_329, unsqueeze_1073);  sub_329 = unsqueeze_1073 = None
    mul_1353: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_330, unsqueeze_1079);  sub_330 = unsqueeze_1079 = None
    mul_1354: "f32[120]" = torch.ops.aten.mul.Tensor(sum_138, squeeze_79);  sum_138 = squeeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_91 = torch.ops.aten.convolution_backward.default(mul_1353, add_159, primals_215, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1353 = add_159 = primals_215 = None
    getitem_447: "f32[8, 40, 28, 28]" = convolution_backward_91[0]
    getitem_448: "f32[120, 40, 1, 1]" = convolution_backward_91[1];  convolution_backward_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_640: "f32[8, 40, 28, 28]" = torch.ops.aten.add.Tensor(add_635, getitem_447);  add_635 = getitem_447 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_139: "f32[40]" = torch.ops.aten.sum.dim_IntList(add_640, [0, 2, 3])
    sub_331: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_31, unsqueeze_1082);  convolution_31 = unsqueeze_1082 = None
    mul_1355: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(add_640, sub_331)
    sum_140: "f32[40]" = torch.ops.aten.sum.dim_IntList(mul_1355, [0, 2, 3]);  mul_1355 = None
    mul_1356: "f32[40]" = torch.ops.aten.mul.Tensor(sum_139, 0.00015943877551020407)
    unsqueeze_1083: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_1356, 0);  mul_1356 = None
    unsqueeze_1084: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1083, 2);  unsqueeze_1083 = None
    unsqueeze_1085: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1084, 3);  unsqueeze_1084 = None
    mul_1357: "f32[40]" = torch.ops.aten.mul.Tensor(sum_140, 0.00015943877551020407)
    mul_1358: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_76, squeeze_76)
    mul_1359: "f32[40]" = torch.ops.aten.mul.Tensor(mul_1357, mul_1358);  mul_1357 = mul_1358 = None
    unsqueeze_1086: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_1359, 0);  mul_1359 = None
    unsqueeze_1087: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1086, 2);  unsqueeze_1086 = None
    unsqueeze_1088: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1087, 3);  unsqueeze_1087 = None
    mul_1360: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_76, primals_51);  primals_51 = None
    unsqueeze_1089: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_1360, 0);  mul_1360 = None
    unsqueeze_1090: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1089, 2);  unsqueeze_1089 = None
    unsqueeze_1091: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1090, 3);  unsqueeze_1090 = None
    mul_1361: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_331, unsqueeze_1088);  sub_331 = unsqueeze_1088 = None
    sub_333: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(add_640, mul_1361);  mul_1361 = None
    sub_334: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(sub_333, unsqueeze_1085);  sub_333 = unsqueeze_1085 = None
    mul_1362: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_334, unsqueeze_1091);  sub_334 = unsqueeze_1091 = None
    mul_1363: "f32[40]" = torch.ops.aten.mul.Tensor(sum_140, squeeze_76);  sum_140 = squeeze_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_92 = torch.ops.aten.convolution_backward.default(mul_1362, mul_197, primals_214, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1362 = mul_197 = primals_214 = None
    getitem_450: "f32[8, 120, 28, 28]" = convolution_backward_92[0]
    getitem_451: "f32[40, 120, 1, 1]" = convolution_backward_92[1];  convolution_backward_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_1364: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_450, div_20);  div_20 = None
    mul_1365: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_450, div_22);  getitem_450 = div_22 = None
    sum_141: "f32[8, 120, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1364, [2, 3], True);  mul_1364 = None
    mul_1366: "f32[8, 120, 1, 1]" = torch.ops.aten.mul.Tensor(sum_141, 0.16666666666666666);  sum_141 = None
    where_129: "f32[8, 120, 1, 1]" = torch.ops.aten.where.self(bitwise_and_15, mul_1366, full_default);  bitwise_and_15 = mul_1366 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_93 = torch.ops.aten.convolution_backward.default(where_129, div_21, primals_212, [120], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_129 = div_21 = primals_212 = None
    getitem_453: "f32[8, 16, 1, 1]" = convolution_backward_93[0]
    getitem_454: "f32[120, 16, 1, 1]" = convolution_backward_93[1]
    getitem_455: "f32[120]" = convolution_backward_93[2];  convolution_backward_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    lt_73: "b8[8, 16, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_29, -3)
    le_57: "b8[8, 16, 1, 1]" = torch.ops.aten.le.Scalar(convolution_29, 3)
    div_168: "f32[8, 16, 1, 1]" = torch.ops.aten.div.Tensor(convolution_29, 3);  convolution_29 = None
    add_641: "f32[8, 16, 1, 1]" = torch.ops.aten.add.Tensor(div_168, 0.5);  div_168 = None
    mul_1367: "f32[8, 16, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_453, add_641);  add_641 = None
    where_130: "f32[8, 16, 1, 1]" = torch.ops.aten.where.self(le_57, mul_1367, getitem_453);  le_57 = mul_1367 = getitem_453 = None
    where_131: "f32[8, 16, 1, 1]" = torch.ops.aten.where.self(lt_73, full_default, where_130);  lt_73 = where_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_94 = torch.ops.aten.convolution_backward.default(where_131, mean_2, primals_210, [16], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_131 = mean_2 = primals_210 = None
    getitem_456: "f32[8, 120, 1, 1]" = convolution_backward_94[0]
    getitem_457: "f32[16, 120, 1, 1]" = convolution_backward_94[1]
    getitem_458: "f32[16]" = convolution_backward_94[2];  convolution_backward_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_16: "f32[8, 120, 28, 28]" = torch.ops.aten.expand.default(getitem_456, [8, 120, 28, 28]);  getitem_456 = None
    div_169: "f32[8, 120, 28, 28]" = torch.ops.aten.div.Scalar(expand_16, 784);  expand_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_642: "f32[8, 120, 28, 28]" = torch.ops.aten.add.Tensor(mul_1365, div_169);  mul_1365 = div_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_74: "b8[8, 120, 28, 28]" = torch.ops.aten.lt.Scalar(clone_18, -3)
    le_58: "b8[8, 120, 28, 28]" = torch.ops.aten.le.Scalar(clone_18, 3)
    div_170: "f32[8, 120, 28, 28]" = torch.ops.aten.div.Tensor(clone_18, 3);  clone_18 = None
    add_643: "f32[8, 120, 28, 28]" = torch.ops.aten.add.Tensor(div_170, 0.5);  div_170 = None
    mul_1368: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(add_642, add_643);  add_643 = None
    where_132: "f32[8, 120, 28, 28]" = torch.ops.aten.where.self(le_58, mul_1368, add_642);  le_58 = mul_1368 = add_642 = None
    where_133: "f32[8, 120, 28, 28]" = torch.ops.aten.where.self(lt_74, full_default, where_132);  lt_74 = where_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_142: "f32[120]" = torch.ops.aten.sum.dim_IntList(where_133, [0, 2, 3])
    sub_335: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_1094);  convolution_28 = unsqueeze_1094 = None
    mul_1369: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(where_133, sub_335)
    sum_143: "f32[120]" = torch.ops.aten.sum.dim_IntList(mul_1369, [0, 2, 3]);  mul_1369 = None
    mul_1370: "f32[120]" = torch.ops.aten.mul.Tensor(sum_142, 0.00015943877551020407)
    unsqueeze_1095: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1370, 0);  mul_1370 = None
    unsqueeze_1096: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1095, 2);  unsqueeze_1095 = None
    unsqueeze_1097: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1096, 3);  unsqueeze_1096 = None
    mul_1371: "f32[120]" = torch.ops.aten.mul.Tensor(sum_143, 0.00015943877551020407)
    mul_1372: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_73, squeeze_73)
    mul_1373: "f32[120]" = torch.ops.aten.mul.Tensor(mul_1371, mul_1372);  mul_1371 = mul_1372 = None
    unsqueeze_1098: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1373, 0);  mul_1373 = None
    unsqueeze_1099: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1098, 2);  unsqueeze_1098 = None
    unsqueeze_1100: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1099, 3);  unsqueeze_1099 = None
    mul_1374: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_73, primals_49);  primals_49 = None
    unsqueeze_1101: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1374, 0);  mul_1374 = None
    unsqueeze_1102: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1101, 2);  unsqueeze_1101 = None
    unsqueeze_1103: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1102, 3);  unsqueeze_1102 = None
    mul_1375: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_335, unsqueeze_1100);  sub_335 = unsqueeze_1100 = None
    sub_337: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(where_133, mul_1375);  where_133 = mul_1375 = None
    sub_338: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(sub_337, unsqueeze_1097);  sub_337 = unsqueeze_1097 = None
    mul_1376: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_338, unsqueeze_1103);  sub_338 = unsqueeze_1103 = None
    mul_1377: "f32[120]" = torch.ops.aten.mul.Tensor(sum_143, squeeze_73);  sum_143 = squeeze_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_95 = torch.ops.aten.convolution_backward.default(mul_1376, div_19, primals_209, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 120, [True, True, False]);  mul_1376 = div_19 = primals_209 = None
    getitem_459: "f32[8, 120, 28, 28]" = convolution_backward_95[0]
    getitem_460: "f32[120, 1, 5, 5]" = convolution_backward_95[1];  convolution_backward_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_75: "b8[8, 120, 28, 28]" = torch.ops.aten.lt.Scalar(clone_17, -3)
    le_59: "b8[8, 120, 28, 28]" = torch.ops.aten.le.Scalar(clone_17, 3)
    div_171: "f32[8, 120, 28, 28]" = torch.ops.aten.div.Tensor(clone_17, 3);  clone_17 = None
    add_644: "f32[8, 120, 28, 28]" = torch.ops.aten.add.Tensor(div_171, 0.5);  div_171 = None
    mul_1378: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_459, add_644);  add_644 = None
    where_134: "f32[8, 120, 28, 28]" = torch.ops.aten.where.self(le_59, mul_1378, getitem_459);  le_59 = mul_1378 = getitem_459 = None
    where_135: "f32[8, 120, 28, 28]" = torch.ops.aten.where.self(lt_75, full_default, where_134);  lt_75 = where_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_144: "f32[120]" = torch.ops.aten.sum.dim_IntList(where_135, [0, 2, 3])
    sub_339: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_1106);  convolution_27 = unsqueeze_1106 = None
    mul_1379: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(where_135, sub_339)
    sum_145: "f32[120]" = torch.ops.aten.sum.dim_IntList(mul_1379, [0, 2, 3]);  mul_1379 = None
    mul_1380: "f32[120]" = torch.ops.aten.mul.Tensor(sum_144, 0.00015943877551020407)
    unsqueeze_1107: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1380, 0);  mul_1380 = None
    unsqueeze_1108: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1107, 2);  unsqueeze_1107 = None
    unsqueeze_1109: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1108, 3);  unsqueeze_1108 = None
    mul_1381: "f32[120]" = torch.ops.aten.mul.Tensor(sum_145, 0.00015943877551020407)
    mul_1382: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_70, squeeze_70)
    mul_1383: "f32[120]" = torch.ops.aten.mul.Tensor(mul_1381, mul_1382);  mul_1381 = mul_1382 = None
    unsqueeze_1110: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1383, 0);  mul_1383 = None
    unsqueeze_1111: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1110, 2);  unsqueeze_1110 = None
    unsqueeze_1112: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1111, 3);  unsqueeze_1111 = None
    mul_1384: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_70, primals_47);  primals_47 = None
    unsqueeze_1113: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1384, 0);  mul_1384 = None
    unsqueeze_1114: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1113, 2);  unsqueeze_1113 = None
    unsqueeze_1115: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1114, 3);  unsqueeze_1114 = None
    mul_1385: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_339, unsqueeze_1112);  sub_339 = unsqueeze_1112 = None
    sub_341: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(where_135, mul_1385);  where_135 = mul_1385 = None
    sub_342: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(sub_341, unsqueeze_1109);  sub_341 = unsqueeze_1109 = None
    mul_1386: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_342, unsqueeze_1115);  sub_342 = unsqueeze_1115 = None
    mul_1387: "f32[120]" = torch.ops.aten.mul.Tensor(sum_145, squeeze_70);  sum_145 = squeeze_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_96 = torch.ops.aten.convolution_backward.default(mul_1386, add_139, primals_208, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1386 = add_139 = primals_208 = None
    getitem_462: "f32[8, 40, 28, 28]" = convolution_backward_96[0]
    getitem_463: "f32[120, 40, 1, 1]" = convolution_backward_96[1];  convolution_backward_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_645: "f32[8, 40, 28, 28]" = torch.ops.aten.add.Tensor(add_640, getitem_462);  add_640 = getitem_462 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_146: "f32[40]" = torch.ops.aten.sum.dim_IntList(add_645, [0, 2, 3])
    sub_343: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_1118);  convolution_26 = unsqueeze_1118 = None
    mul_1388: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(add_645, sub_343)
    sum_147: "f32[40]" = torch.ops.aten.sum.dim_IntList(mul_1388, [0, 2, 3]);  mul_1388 = None
    mul_1389: "f32[40]" = torch.ops.aten.mul.Tensor(sum_146, 0.00015943877551020407)
    unsqueeze_1119: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_1389, 0);  mul_1389 = None
    unsqueeze_1120: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1119, 2);  unsqueeze_1119 = None
    unsqueeze_1121: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1120, 3);  unsqueeze_1120 = None
    mul_1390: "f32[40]" = torch.ops.aten.mul.Tensor(sum_147, 0.00015943877551020407)
    mul_1391: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_67, squeeze_67)
    mul_1392: "f32[40]" = torch.ops.aten.mul.Tensor(mul_1390, mul_1391);  mul_1390 = mul_1391 = None
    unsqueeze_1122: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_1392, 0);  mul_1392 = None
    unsqueeze_1123: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1122, 2);  unsqueeze_1122 = None
    unsqueeze_1124: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1123, 3);  unsqueeze_1123 = None
    mul_1393: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_67, primals_45);  primals_45 = None
    unsqueeze_1125: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_1393, 0);  mul_1393 = None
    unsqueeze_1126: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1125, 2);  unsqueeze_1125 = None
    unsqueeze_1127: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1126, 3);  unsqueeze_1126 = None
    mul_1394: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_343, unsqueeze_1124);  sub_343 = unsqueeze_1124 = None
    sub_345: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(add_645, mul_1394);  mul_1394 = None
    sub_346: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(sub_345, unsqueeze_1121);  sub_345 = unsqueeze_1121 = None
    mul_1395: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_346, unsqueeze_1127);  sub_346 = unsqueeze_1127 = None
    mul_1396: "f32[40]" = torch.ops.aten.mul.Tensor(sum_147, squeeze_67);  sum_147 = squeeze_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_97 = torch.ops.aten.convolution_backward.default(mul_1395, mul_172, primals_207, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1395 = mul_172 = primals_207 = None
    getitem_465: "f32[8, 120, 28, 28]" = convolution_backward_97[0]
    getitem_466: "f32[40, 120, 1, 1]" = convolution_backward_97[1];  convolution_backward_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_1397: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_465, div_16);  div_16 = None
    mul_1398: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_465, div_18);  getitem_465 = div_18 = None
    sum_148: "f32[8, 120, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1397, [2, 3], True);  mul_1397 = None
    mul_1399: "f32[8, 120, 1, 1]" = torch.ops.aten.mul.Tensor(sum_148, 0.16666666666666666);  sum_148 = None
    where_136: "f32[8, 120, 1, 1]" = torch.ops.aten.where.self(bitwise_and_16, mul_1399, full_default);  bitwise_and_16 = mul_1399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_98 = torch.ops.aten.convolution_backward.default(where_136, div_17, primals_205, [120], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_136 = div_17 = primals_205 = None
    getitem_468: "f32[8, 16, 1, 1]" = convolution_backward_98[0]
    getitem_469: "f32[120, 16, 1, 1]" = convolution_backward_98[1]
    getitem_470: "f32[120]" = convolution_backward_98[2];  convolution_backward_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    lt_77: "b8[8, 16, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_24, -3)
    le_60: "b8[8, 16, 1, 1]" = torch.ops.aten.le.Scalar(convolution_24, 3)
    div_172: "f32[8, 16, 1, 1]" = torch.ops.aten.div.Tensor(convolution_24, 3);  convolution_24 = None
    add_646: "f32[8, 16, 1, 1]" = torch.ops.aten.add.Tensor(div_172, 0.5);  div_172 = None
    mul_1400: "f32[8, 16, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_468, add_646);  add_646 = None
    where_137: "f32[8, 16, 1, 1]" = torch.ops.aten.where.self(le_60, mul_1400, getitem_468);  le_60 = mul_1400 = getitem_468 = None
    where_138: "f32[8, 16, 1, 1]" = torch.ops.aten.where.self(lt_77, full_default, where_137);  lt_77 = where_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_99 = torch.ops.aten.convolution_backward.default(where_138, mean_1, primals_203, [16], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_138 = mean_1 = primals_203 = None
    getitem_471: "f32[8, 120, 1, 1]" = convolution_backward_99[0]
    getitem_472: "f32[16, 120, 1, 1]" = convolution_backward_99[1]
    getitem_473: "f32[16]" = convolution_backward_99[2];  convolution_backward_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_17: "f32[8, 120, 28, 28]" = torch.ops.aten.expand.default(getitem_471, [8, 120, 28, 28]);  getitem_471 = None
    div_173: "f32[8, 120, 28, 28]" = torch.ops.aten.div.Scalar(expand_17, 784);  expand_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_647: "f32[8, 120, 28, 28]" = torch.ops.aten.add.Tensor(mul_1398, div_173);  mul_1398 = div_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_78: "b8[8, 120, 28, 28]" = torch.ops.aten.lt.Scalar(clone_15, -3)
    le_61: "b8[8, 120, 28, 28]" = torch.ops.aten.le.Scalar(clone_15, 3)
    div_174: "f32[8, 120, 28, 28]" = torch.ops.aten.div.Tensor(clone_15, 3);  clone_15 = None
    add_648: "f32[8, 120, 28, 28]" = torch.ops.aten.add.Tensor(div_174, 0.5);  div_174 = None
    mul_1401: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(add_647, add_648);  add_648 = None
    where_139: "f32[8, 120, 28, 28]" = torch.ops.aten.where.self(le_61, mul_1401, add_647);  le_61 = mul_1401 = add_647 = None
    where_140: "f32[8, 120, 28, 28]" = torch.ops.aten.where.self(lt_78, full_default, where_139);  lt_78 = where_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_149: "f32[120]" = torch.ops.aten.sum.dim_IntList(where_140, [0, 2, 3])
    sub_347: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_1130);  convolution_23 = unsqueeze_1130 = None
    mul_1402: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(where_140, sub_347)
    sum_150: "f32[120]" = torch.ops.aten.sum.dim_IntList(mul_1402, [0, 2, 3]);  mul_1402 = None
    mul_1403: "f32[120]" = torch.ops.aten.mul.Tensor(sum_149, 0.00015943877551020407)
    unsqueeze_1131: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1403, 0);  mul_1403 = None
    unsqueeze_1132: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1131, 2);  unsqueeze_1131 = None
    unsqueeze_1133: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1132, 3);  unsqueeze_1132 = None
    mul_1404: "f32[120]" = torch.ops.aten.mul.Tensor(sum_150, 0.00015943877551020407)
    mul_1405: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_64, squeeze_64)
    mul_1406: "f32[120]" = torch.ops.aten.mul.Tensor(mul_1404, mul_1405);  mul_1404 = mul_1405 = None
    unsqueeze_1134: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1406, 0);  mul_1406 = None
    unsqueeze_1135: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1134, 2);  unsqueeze_1134 = None
    unsqueeze_1136: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1135, 3);  unsqueeze_1135 = None
    mul_1407: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_64, primals_43);  primals_43 = None
    unsqueeze_1137: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1407, 0);  mul_1407 = None
    unsqueeze_1138: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1137, 2);  unsqueeze_1137 = None
    unsqueeze_1139: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1138, 3);  unsqueeze_1138 = None
    mul_1408: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_347, unsqueeze_1136);  sub_347 = unsqueeze_1136 = None
    sub_349: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(where_140, mul_1408);  where_140 = mul_1408 = None
    sub_350: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(sub_349, unsqueeze_1133);  sub_349 = unsqueeze_1133 = None
    mul_1409: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_350, unsqueeze_1139);  sub_350 = unsqueeze_1139 = None
    mul_1410: "f32[120]" = torch.ops.aten.mul.Tensor(sum_150, squeeze_64);  sum_150 = squeeze_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_100 = torch.ops.aten.convolution_backward.default(mul_1409, div_15, primals_202, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 120, [True, True, False]);  mul_1409 = div_15 = primals_202 = None
    getitem_474: "f32[8, 120, 28, 28]" = convolution_backward_100[0]
    getitem_475: "f32[120, 1, 5, 5]" = convolution_backward_100[1];  convolution_backward_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_79: "b8[8, 120, 28, 28]" = torch.ops.aten.lt.Scalar(clone_14, -3)
    le_62: "b8[8, 120, 28, 28]" = torch.ops.aten.le.Scalar(clone_14, 3)
    div_175: "f32[8, 120, 28, 28]" = torch.ops.aten.div.Tensor(clone_14, 3);  clone_14 = None
    add_649: "f32[8, 120, 28, 28]" = torch.ops.aten.add.Tensor(div_175, 0.5);  div_175 = None
    mul_1411: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_474, add_649);  add_649 = None
    where_141: "f32[8, 120, 28, 28]" = torch.ops.aten.where.self(le_62, mul_1411, getitem_474);  le_62 = mul_1411 = getitem_474 = None
    where_142: "f32[8, 120, 28, 28]" = torch.ops.aten.where.self(lt_79, full_default, where_141);  lt_79 = where_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_151: "f32[120]" = torch.ops.aten.sum.dim_IntList(where_142, [0, 2, 3])
    sub_351: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_1142);  convolution_22 = unsqueeze_1142 = None
    mul_1412: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(where_142, sub_351)
    sum_152: "f32[120]" = torch.ops.aten.sum.dim_IntList(mul_1412, [0, 2, 3]);  mul_1412 = None
    mul_1413: "f32[120]" = torch.ops.aten.mul.Tensor(sum_151, 0.00015943877551020407)
    unsqueeze_1143: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1413, 0);  mul_1413 = None
    unsqueeze_1144: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1143, 2);  unsqueeze_1143 = None
    unsqueeze_1145: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1144, 3);  unsqueeze_1144 = None
    mul_1414: "f32[120]" = torch.ops.aten.mul.Tensor(sum_152, 0.00015943877551020407)
    mul_1415: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_61, squeeze_61)
    mul_1416: "f32[120]" = torch.ops.aten.mul.Tensor(mul_1414, mul_1415);  mul_1414 = mul_1415 = None
    unsqueeze_1146: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1416, 0);  mul_1416 = None
    unsqueeze_1147: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1146, 2);  unsqueeze_1146 = None
    unsqueeze_1148: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1147, 3);  unsqueeze_1147 = None
    mul_1417: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_61, primals_41);  primals_41 = None
    unsqueeze_1149: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1417, 0);  mul_1417 = None
    unsqueeze_1150: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1149, 2);  unsqueeze_1149 = None
    unsqueeze_1151: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1150, 3);  unsqueeze_1150 = None
    mul_1418: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_351, unsqueeze_1148);  sub_351 = unsqueeze_1148 = None
    sub_353: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(where_142, mul_1418);  where_142 = mul_1418 = None
    sub_354: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(sub_353, unsqueeze_1145);  sub_353 = unsqueeze_1145 = None
    mul_1419: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_354, unsqueeze_1151);  sub_354 = unsqueeze_1151 = None
    mul_1420: "f32[120]" = torch.ops.aten.mul.Tensor(sum_152, squeeze_61);  sum_152 = squeeze_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_101 = torch.ops.aten.convolution_backward.default(mul_1419, add_119, primals_201, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1419 = add_119 = primals_201 = None
    getitem_477: "f32[8, 40, 28, 28]" = convolution_backward_101[0]
    getitem_478: "f32[120, 40, 1, 1]" = convolution_backward_101[1];  convolution_backward_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_650: "f32[8, 40, 28, 28]" = torch.ops.aten.add.Tensor(add_645, getitem_477);  add_645 = getitem_477 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_153: "f32[40]" = torch.ops.aten.sum.dim_IntList(add_650, [0, 2, 3])
    sub_355: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_1154);  convolution_21 = unsqueeze_1154 = None
    mul_1421: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(add_650, sub_355)
    sum_154: "f32[40]" = torch.ops.aten.sum.dim_IntList(mul_1421, [0, 2, 3]);  mul_1421 = None
    mul_1422: "f32[40]" = torch.ops.aten.mul.Tensor(sum_153, 0.00015943877551020407)
    unsqueeze_1155: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_1422, 0);  mul_1422 = None
    unsqueeze_1156: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1155, 2);  unsqueeze_1155 = None
    unsqueeze_1157: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1156, 3);  unsqueeze_1156 = None
    mul_1423: "f32[40]" = torch.ops.aten.mul.Tensor(sum_154, 0.00015943877551020407)
    mul_1424: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_58, squeeze_58)
    mul_1425: "f32[40]" = torch.ops.aten.mul.Tensor(mul_1423, mul_1424);  mul_1423 = mul_1424 = None
    unsqueeze_1158: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_1425, 0);  mul_1425 = None
    unsqueeze_1159: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1158, 2);  unsqueeze_1158 = None
    unsqueeze_1160: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1159, 3);  unsqueeze_1159 = None
    mul_1426: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_58, primals_39);  primals_39 = None
    unsqueeze_1161: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_1426, 0);  mul_1426 = None
    unsqueeze_1162: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1161, 2);  unsqueeze_1161 = None
    unsqueeze_1163: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1162, 3);  unsqueeze_1162 = None
    mul_1427: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_355, unsqueeze_1160);  sub_355 = unsqueeze_1160 = None
    sub_357: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(add_650, mul_1427);  add_650 = mul_1427 = None
    sub_358: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(sub_357, unsqueeze_1157);  sub_357 = unsqueeze_1157 = None
    mul_1428: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_358, unsqueeze_1163);  sub_358 = unsqueeze_1163 = None
    mul_1429: "f32[40]" = torch.ops.aten.mul.Tensor(sum_154, squeeze_58);  sum_154 = squeeze_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_102 = torch.ops.aten.convolution_backward.default(mul_1428, mul_147, primals_200, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1428 = mul_147 = primals_200 = None
    getitem_480: "f32[8, 120, 28, 28]" = convolution_backward_102[0]
    getitem_481: "f32[40, 120, 1, 1]" = convolution_backward_102[1];  convolution_backward_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_1430: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_480, div_12);  div_12 = None
    mul_1431: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_480, div_14);  getitem_480 = div_14 = None
    sum_155: "f32[8, 120, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1430, [2, 3], True);  mul_1430 = None
    mul_1432: "f32[8, 120, 1, 1]" = torch.ops.aten.mul.Tensor(sum_155, 0.16666666666666666);  sum_155 = None
    where_143: "f32[8, 120, 1, 1]" = torch.ops.aten.where.self(bitwise_and_17, mul_1432, full_default);  bitwise_and_17 = mul_1432 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_103 = torch.ops.aten.convolution_backward.default(where_143, div_13, primals_198, [120], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_143 = div_13 = primals_198 = None
    getitem_483: "f32[8, 8, 1, 1]" = convolution_backward_103[0]
    getitem_484: "f32[120, 8, 1, 1]" = convolution_backward_103[1]
    getitem_485: "f32[120]" = convolution_backward_103[2];  convolution_backward_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    lt_81: "b8[8, 8, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_19, -3)
    le_63: "b8[8, 8, 1, 1]" = torch.ops.aten.le.Scalar(convolution_19, 3)
    div_176: "f32[8, 8, 1, 1]" = torch.ops.aten.div.Tensor(convolution_19, 3);  convolution_19 = None
    add_651: "f32[8, 8, 1, 1]" = torch.ops.aten.add.Tensor(div_176, 0.5);  div_176 = None
    mul_1433: "f32[8, 8, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_483, add_651);  add_651 = None
    where_144: "f32[8, 8, 1, 1]" = torch.ops.aten.where.self(le_63, mul_1433, getitem_483);  le_63 = mul_1433 = getitem_483 = None
    where_145: "f32[8, 8, 1, 1]" = torch.ops.aten.where.self(lt_81, full_default, where_144);  lt_81 = where_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_104 = torch.ops.aten.convolution_backward.default(where_145, mean, primals_196, [8], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_145 = mean = primals_196 = None
    getitem_486: "f32[8, 120, 1, 1]" = convolution_backward_104[0]
    getitem_487: "f32[8, 120, 1, 1]" = convolution_backward_104[1]
    getitem_488: "f32[8]" = convolution_backward_104[2];  convolution_backward_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_18: "f32[8, 120, 28, 28]" = torch.ops.aten.expand.default(getitem_486, [8, 120, 28, 28]);  getitem_486 = None
    div_177: "f32[8, 120, 28, 28]" = torch.ops.aten.div.Scalar(expand_18, 784);  expand_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_652: "f32[8, 120, 28, 28]" = torch.ops.aten.add.Tensor(mul_1431, div_177);  mul_1431 = div_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_82: "b8[8, 120, 28, 28]" = torch.ops.aten.lt.Scalar(clone_12, -3)
    le_64: "b8[8, 120, 28, 28]" = torch.ops.aten.le.Scalar(clone_12, 3)
    div_178: "f32[8, 120, 28, 28]" = torch.ops.aten.div.Tensor(clone_12, 3);  clone_12 = None
    add_653: "f32[8, 120, 28, 28]" = torch.ops.aten.add.Tensor(div_178, 0.5);  div_178 = None
    mul_1434: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(add_652, add_653);  add_653 = None
    where_146: "f32[8, 120, 28, 28]" = torch.ops.aten.where.self(le_64, mul_1434, add_652);  le_64 = mul_1434 = add_652 = None
    where_147: "f32[8, 120, 28, 28]" = torch.ops.aten.where.self(lt_82, full_default, where_146);  lt_82 = where_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_156: "f32[120]" = torch.ops.aten.sum.dim_IntList(where_147, [0, 2, 3])
    sub_359: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_1166);  convolution_18 = unsqueeze_1166 = None
    mul_1435: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(where_147, sub_359)
    sum_157: "f32[120]" = torch.ops.aten.sum.dim_IntList(mul_1435, [0, 2, 3]);  mul_1435 = None
    mul_1436: "f32[120]" = torch.ops.aten.mul.Tensor(sum_156, 0.00015943877551020407)
    unsqueeze_1167: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1436, 0);  mul_1436 = None
    unsqueeze_1168: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1167, 2);  unsqueeze_1167 = None
    unsqueeze_1169: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1168, 3);  unsqueeze_1168 = None
    mul_1437: "f32[120]" = torch.ops.aten.mul.Tensor(sum_157, 0.00015943877551020407)
    mul_1438: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_55, squeeze_55)
    mul_1439: "f32[120]" = torch.ops.aten.mul.Tensor(mul_1437, mul_1438);  mul_1437 = mul_1438 = None
    unsqueeze_1170: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1439, 0);  mul_1439 = None
    unsqueeze_1171: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1170, 2);  unsqueeze_1170 = None
    unsqueeze_1172: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1171, 3);  unsqueeze_1171 = None
    mul_1440: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_55, primals_37);  primals_37 = None
    unsqueeze_1173: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1440, 0);  mul_1440 = None
    unsqueeze_1174: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1173, 2);  unsqueeze_1173 = None
    unsqueeze_1175: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1174, 3);  unsqueeze_1174 = None
    mul_1441: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_359, unsqueeze_1172);  sub_359 = unsqueeze_1172 = None
    sub_361: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(where_147, mul_1441);  where_147 = mul_1441 = None
    sub_362: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(sub_361, unsqueeze_1169);  sub_361 = unsqueeze_1169 = None
    mul_1442: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_362, unsqueeze_1175);  sub_362 = unsqueeze_1175 = None
    mul_1443: "f32[120]" = torch.ops.aten.mul.Tensor(sum_157, squeeze_55);  sum_157 = squeeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_105 = torch.ops.aten.convolution_backward.default(mul_1442, div_11, primals_195, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 120, [True, True, False]);  mul_1442 = div_11 = primals_195 = None
    getitem_489: "f32[8, 120, 56, 56]" = convolution_backward_105[0]
    getitem_490: "f32[120, 1, 5, 5]" = convolution_backward_105[1];  convolution_backward_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_83: "b8[8, 120, 56, 56]" = torch.ops.aten.lt.Scalar(clone_11, -3)
    le_65: "b8[8, 120, 56, 56]" = torch.ops.aten.le.Scalar(clone_11, 3)
    div_179: "f32[8, 120, 56, 56]" = torch.ops.aten.div.Tensor(clone_11, 3);  clone_11 = None
    add_654: "f32[8, 120, 56, 56]" = torch.ops.aten.add.Tensor(div_179, 0.5);  div_179 = None
    mul_1444: "f32[8, 120, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_489, add_654);  add_654 = None
    where_148: "f32[8, 120, 56, 56]" = torch.ops.aten.where.self(le_65, mul_1444, getitem_489);  le_65 = mul_1444 = getitem_489 = None
    where_149: "f32[8, 120, 56, 56]" = torch.ops.aten.where.self(lt_83, full_default, where_148);  lt_83 = where_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_158: "f32[120]" = torch.ops.aten.sum.dim_IntList(where_149, [0, 2, 3])
    sub_363: "f32[8, 120, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_1178);  convolution_17 = unsqueeze_1178 = None
    mul_1445: "f32[8, 120, 56, 56]" = torch.ops.aten.mul.Tensor(where_149, sub_363)
    sum_159: "f32[120]" = torch.ops.aten.sum.dim_IntList(mul_1445, [0, 2, 3]);  mul_1445 = None
    mul_1446: "f32[120]" = torch.ops.aten.mul.Tensor(sum_158, 3.985969387755102e-05)
    unsqueeze_1179: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1446, 0);  mul_1446 = None
    unsqueeze_1180: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1179, 2);  unsqueeze_1179 = None
    unsqueeze_1181: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1180, 3);  unsqueeze_1180 = None
    mul_1447: "f32[120]" = torch.ops.aten.mul.Tensor(sum_159, 3.985969387755102e-05)
    mul_1448: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_52, squeeze_52)
    mul_1449: "f32[120]" = torch.ops.aten.mul.Tensor(mul_1447, mul_1448);  mul_1447 = mul_1448 = None
    unsqueeze_1182: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1449, 0);  mul_1449 = None
    unsqueeze_1183: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1182, 2);  unsqueeze_1182 = None
    unsqueeze_1184: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1183, 3);  unsqueeze_1183 = None
    mul_1450: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_52, primals_35);  primals_35 = None
    unsqueeze_1185: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1450, 0);  mul_1450 = None
    unsqueeze_1186: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1185, 2);  unsqueeze_1185 = None
    unsqueeze_1187: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1186, 3);  unsqueeze_1186 = None
    mul_1451: "f32[8, 120, 56, 56]" = torch.ops.aten.mul.Tensor(sub_363, unsqueeze_1184);  sub_363 = unsqueeze_1184 = None
    sub_365: "f32[8, 120, 56, 56]" = torch.ops.aten.sub.Tensor(where_149, mul_1451);  where_149 = mul_1451 = None
    sub_366: "f32[8, 120, 56, 56]" = torch.ops.aten.sub.Tensor(sub_365, unsqueeze_1181);  sub_365 = unsqueeze_1181 = None
    mul_1452: "f32[8, 120, 56, 56]" = torch.ops.aten.mul.Tensor(sub_366, unsqueeze_1187);  sub_366 = unsqueeze_1187 = None
    mul_1453: "f32[120]" = torch.ops.aten.mul.Tensor(sum_159, squeeze_52);  sum_159 = squeeze_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_106 = torch.ops.aten.convolution_backward.default(mul_1452, add_100, primals_194, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1452 = add_100 = primals_194 = None
    getitem_492: "f32[8, 24, 56, 56]" = convolution_backward_106[0]
    getitem_493: "f32[120, 24, 1, 1]" = convolution_backward_106[1];  convolution_backward_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_160: "f32[24]" = torch.ops.aten.sum.dim_IntList(getitem_492, [0, 2, 3])
    sub_367: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_1190);  convolution_16 = unsqueeze_1190 = None
    mul_1454: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_492, sub_367)
    sum_161: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_1454, [0, 2, 3]);  mul_1454 = None
    mul_1455: "f32[24]" = torch.ops.aten.mul.Tensor(sum_160, 3.985969387755102e-05)
    unsqueeze_1191: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_1455, 0);  mul_1455 = None
    unsqueeze_1192: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1191, 2);  unsqueeze_1191 = None
    unsqueeze_1193: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1192, 3);  unsqueeze_1192 = None
    mul_1456: "f32[24]" = torch.ops.aten.mul.Tensor(sum_161, 3.985969387755102e-05)
    mul_1457: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_49, squeeze_49)
    mul_1458: "f32[24]" = torch.ops.aten.mul.Tensor(mul_1456, mul_1457);  mul_1456 = mul_1457 = None
    unsqueeze_1194: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_1458, 0);  mul_1458 = None
    unsqueeze_1195: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1194, 2);  unsqueeze_1194 = None
    unsqueeze_1196: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1195, 3);  unsqueeze_1195 = None
    mul_1459: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_49, primals_33);  primals_33 = None
    unsqueeze_1197: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_1459, 0);  mul_1459 = None
    unsqueeze_1198: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1197, 2);  unsqueeze_1197 = None
    unsqueeze_1199: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1198, 3);  unsqueeze_1198 = None
    mul_1460: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_367, unsqueeze_1196);  sub_367 = unsqueeze_1196 = None
    sub_369: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(getitem_492, mul_1460);  mul_1460 = None
    sub_370: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(sub_369, unsqueeze_1193);  sub_369 = unsqueeze_1193 = None
    mul_1461: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_370, unsqueeze_1199);  sub_370 = unsqueeze_1199 = None
    mul_1462: "f32[24]" = torch.ops.aten.mul.Tensor(sum_161, squeeze_49);  sum_161 = squeeze_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_107 = torch.ops.aten.convolution_backward.default(mul_1461, div_10, primals_193, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1461 = div_10 = primals_193 = None
    getitem_495: "f32[8, 48, 56, 56]" = convolution_backward_107[0]
    getitem_496: "f32[24, 48, 1, 1]" = convolution_backward_107[1];  convolution_backward_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_84: "b8[8, 48, 56, 56]" = torch.ops.aten.lt.Scalar(clone_10, -3)
    le_66: "b8[8, 48, 56, 56]" = torch.ops.aten.le.Scalar(clone_10, 3)
    div_180: "f32[8, 48, 56, 56]" = torch.ops.aten.div.Tensor(clone_10, 3);  clone_10 = None
    add_655: "f32[8, 48, 56, 56]" = torch.ops.aten.add.Tensor(div_180, 0.5);  div_180 = None
    mul_1463: "f32[8, 48, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_495, add_655);  add_655 = None
    where_150: "f32[8, 48, 56, 56]" = torch.ops.aten.where.self(le_66, mul_1463, getitem_495);  le_66 = mul_1463 = getitem_495 = None
    where_151: "f32[8, 48, 56, 56]" = torch.ops.aten.where.self(lt_84, full_default, where_150);  lt_84 = where_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_162: "f32[48]" = torch.ops.aten.sum.dim_IntList(where_151, [0, 2, 3])
    sub_371: "f32[8, 48, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_1202);  convolution_15 = unsqueeze_1202 = None
    mul_1464: "f32[8, 48, 56, 56]" = torch.ops.aten.mul.Tensor(where_151, sub_371)
    sum_163: "f32[48]" = torch.ops.aten.sum.dim_IntList(mul_1464, [0, 2, 3]);  mul_1464 = None
    mul_1465: "f32[48]" = torch.ops.aten.mul.Tensor(sum_162, 3.985969387755102e-05)
    unsqueeze_1203: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_1465, 0);  mul_1465 = None
    unsqueeze_1204: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1203, 2);  unsqueeze_1203 = None
    unsqueeze_1205: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1204, 3);  unsqueeze_1204 = None
    mul_1466: "f32[48]" = torch.ops.aten.mul.Tensor(sum_163, 3.985969387755102e-05)
    mul_1467: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_46, squeeze_46)
    mul_1468: "f32[48]" = torch.ops.aten.mul.Tensor(mul_1466, mul_1467);  mul_1466 = mul_1467 = None
    unsqueeze_1206: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_1468, 0);  mul_1468 = None
    unsqueeze_1207: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1206, 2);  unsqueeze_1206 = None
    unsqueeze_1208: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1207, 3);  unsqueeze_1207 = None
    mul_1469: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_46, primals_31);  primals_31 = None
    unsqueeze_1209: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_1469, 0);  mul_1469 = None
    unsqueeze_1210: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1209, 2);  unsqueeze_1209 = None
    unsqueeze_1211: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1210, 3);  unsqueeze_1210 = None
    mul_1470: "f32[8, 48, 56, 56]" = torch.ops.aten.mul.Tensor(sub_371, unsqueeze_1208);  sub_371 = unsqueeze_1208 = None
    sub_373: "f32[8, 48, 56, 56]" = torch.ops.aten.sub.Tensor(where_151, mul_1470);  where_151 = mul_1470 = None
    sub_374: "f32[8, 48, 56, 56]" = torch.ops.aten.sub.Tensor(sub_373, unsqueeze_1205);  sub_373 = unsqueeze_1205 = None
    mul_1471: "f32[8, 48, 56, 56]" = torch.ops.aten.mul.Tensor(sub_374, unsqueeze_1211);  sub_374 = unsqueeze_1211 = None
    mul_1472: "f32[48]" = torch.ops.aten.mul.Tensor(sum_163, squeeze_46);  sum_163 = squeeze_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_108 = torch.ops.aten.convolution_backward.default(mul_1471, div_9, primals_192, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 48, [True, True, False]);  mul_1471 = div_9 = primals_192 = None
    getitem_498: "f32[8, 48, 56, 56]" = convolution_backward_108[0]
    getitem_499: "f32[48, 1, 5, 5]" = convolution_backward_108[1];  convolution_backward_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_85: "b8[8, 48, 56, 56]" = torch.ops.aten.lt.Scalar(clone_9, -3)
    le_67: "b8[8, 48, 56, 56]" = torch.ops.aten.le.Scalar(clone_9, 3)
    div_181: "f32[8, 48, 56, 56]" = torch.ops.aten.div.Tensor(clone_9, 3);  clone_9 = None
    add_656: "f32[8, 48, 56, 56]" = torch.ops.aten.add.Tensor(div_181, 0.5);  div_181 = None
    mul_1473: "f32[8, 48, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_498, add_656);  add_656 = None
    where_152: "f32[8, 48, 56, 56]" = torch.ops.aten.where.self(le_67, mul_1473, getitem_498);  le_67 = mul_1473 = getitem_498 = None
    where_153: "f32[8, 48, 56, 56]" = torch.ops.aten.where.self(lt_85, full_default, where_152);  lt_85 = where_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_164: "f32[48]" = torch.ops.aten.sum.dim_IntList(where_153, [0, 2, 3])
    sub_375: "f32[8, 48, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_1214);  convolution_14 = unsqueeze_1214 = None
    mul_1474: "f32[8, 48, 56, 56]" = torch.ops.aten.mul.Tensor(where_153, sub_375)
    sum_165: "f32[48]" = torch.ops.aten.sum.dim_IntList(mul_1474, [0, 2, 3]);  mul_1474 = None
    mul_1475: "f32[48]" = torch.ops.aten.mul.Tensor(sum_164, 3.985969387755102e-05)
    unsqueeze_1215: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_1475, 0);  mul_1475 = None
    unsqueeze_1216: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1215, 2);  unsqueeze_1215 = None
    unsqueeze_1217: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1216, 3);  unsqueeze_1216 = None
    mul_1476: "f32[48]" = torch.ops.aten.mul.Tensor(sum_165, 3.985969387755102e-05)
    mul_1477: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
    mul_1478: "f32[48]" = torch.ops.aten.mul.Tensor(mul_1476, mul_1477);  mul_1476 = mul_1477 = None
    unsqueeze_1218: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_1478, 0);  mul_1478 = None
    unsqueeze_1219: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1218, 2);  unsqueeze_1218 = None
    unsqueeze_1220: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1219, 3);  unsqueeze_1219 = None
    mul_1479: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_43, primals_29);  primals_29 = None
    unsqueeze_1221: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_1479, 0);  mul_1479 = None
    unsqueeze_1222: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1221, 2);  unsqueeze_1221 = None
    unsqueeze_1223: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1222, 3);  unsqueeze_1222 = None
    mul_1480: "f32[8, 48, 56, 56]" = torch.ops.aten.mul.Tensor(sub_375, unsqueeze_1220);  sub_375 = unsqueeze_1220 = None
    sub_377: "f32[8, 48, 56, 56]" = torch.ops.aten.sub.Tensor(where_153, mul_1480);  where_153 = mul_1480 = None
    sub_378: "f32[8, 48, 56, 56]" = torch.ops.aten.sub.Tensor(sub_377, unsqueeze_1217);  sub_377 = unsqueeze_1217 = None
    mul_1481: "f32[8, 48, 56, 56]" = torch.ops.aten.mul.Tensor(sub_378, unsqueeze_1223);  sub_378 = unsqueeze_1223 = None
    mul_1482: "f32[48]" = torch.ops.aten.mul.Tensor(sum_165, squeeze_43);  sum_165 = squeeze_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_109 = torch.ops.aten.convolution_backward.default(mul_1481, add_82, primals_191, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1481 = add_82 = primals_191 = None
    getitem_501: "f32[8, 24, 56, 56]" = convolution_backward_109[0]
    getitem_502: "f32[48, 24, 1, 1]" = convolution_backward_109[1];  convolution_backward_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_657: "f32[8, 24, 56, 56]" = torch.ops.aten.add.Tensor(getitem_492, getitem_501);  getitem_492 = getitem_501 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_166: "f32[24]" = torch.ops.aten.sum.dim_IntList(add_657, [0, 2, 3])
    sub_379: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_1226);  convolution_13 = unsqueeze_1226 = None
    mul_1483: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(add_657, sub_379)
    sum_167: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_1483, [0, 2, 3]);  mul_1483 = None
    mul_1484: "f32[24]" = torch.ops.aten.mul.Tensor(sum_166, 3.985969387755102e-05)
    unsqueeze_1227: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_1484, 0);  mul_1484 = None
    unsqueeze_1228: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1227, 2);  unsqueeze_1227 = None
    unsqueeze_1229: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1228, 3);  unsqueeze_1228 = None
    mul_1485: "f32[24]" = torch.ops.aten.mul.Tensor(sum_167, 3.985969387755102e-05)
    mul_1486: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_40, squeeze_40)
    mul_1487: "f32[24]" = torch.ops.aten.mul.Tensor(mul_1485, mul_1486);  mul_1485 = mul_1486 = None
    unsqueeze_1230: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_1487, 0);  mul_1487 = None
    unsqueeze_1231: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1230, 2);  unsqueeze_1230 = None
    unsqueeze_1232: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1231, 3);  unsqueeze_1231 = None
    mul_1488: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_40, primals_27);  primals_27 = None
    unsqueeze_1233: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_1488, 0);  mul_1488 = None
    unsqueeze_1234: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1233, 2);  unsqueeze_1233 = None
    unsqueeze_1235: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1234, 3);  unsqueeze_1234 = None
    mul_1489: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_379, unsqueeze_1232);  sub_379 = unsqueeze_1232 = None
    sub_381: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(add_657, mul_1489);  mul_1489 = None
    sub_382: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(sub_381, unsqueeze_1229);  sub_381 = unsqueeze_1229 = None
    mul_1490: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_382, unsqueeze_1235);  sub_382 = unsqueeze_1235 = None
    mul_1491: "f32[24]" = torch.ops.aten.mul.Tensor(sum_167, squeeze_40);  sum_167 = squeeze_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_110 = torch.ops.aten.convolution_backward.default(mul_1490, div_8, primals_190, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1490 = div_8 = primals_190 = None
    getitem_504: "f32[8, 48, 56, 56]" = convolution_backward_110[0]
    getitem_505: "f32[24, 48, 1, 1]" = convolution_backward_110[1];  convolution_backward_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_86: "b8[8, 48, 56, 56]" = torch.ops.aten.lt.Scalar(clone_8, -3)
    le_68: "b8[8, 48, 56, 56]" = torch.ops.aten.le.Scalar(clone_8, 3)
    div_182: "f32[8, 48, 56, 56]" = torch.ops.aten.div.Tensor(clone_8, 3);  clone_8 = None
    add_658: "f32[8, 48, 56, 56]" = torch.ops.aten.add.Tensor(div_182, 0.5);  div_182 = None
    mul_1492: "f32[8, 48, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_504, add_658);  add_658 = None
    where_154: "f32[8, 48, 56, 56]" = torch.ops.aten.where.self(le_68, mul_1492, getitem_504);  le_68 = mul_1492 = getitem_504 = None
    where_155: "f32[8, 48, 56, 56]" = torch.ops.aten.where.self(lt_86, full_default, where_154);  lt_86 = where_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_168: "f32[48]" = torch.ops.aten.sum.dim_IntList(where_155, [0, 2, 3])
    sub_383: "f32[8, 48, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_1238);  convolution_12 = unsqueeze_1238 = None
    mul_1493: "f32[8, 48, 56, 56]" = torch.ops.aten.mul.Tensor(where_155, sub_383)
    sum_169: "f32[48]" = torch.ops.aten.sum.dim_IntList(mul_1493, [0, 2, 3]);  mul_1493 = None
    mul_1494: "f32[48]" = torch.ops.aten.mul.Tensor(sum_168, 3.985969387755102e-05)
    unsqueeze_1239: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_1494, 0);  mul_1494 = None
    unsqueeze_1240: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1239, 2);  unsqueeze_1239 = None
    unsqueeze_1241: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1240, 3);  unsqueeze_1240 = None
    mul_1495: "f32[48]" = torch.ops.aten.mul.Tensor(sum_169, 3.985969387755102e-05)
    mul_1496: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
    mul_1497: "f32[48]" = torch.ops.aten.mul.Tensor(mul_1495, mul_1496);  mul_1495 = mul_1496 = None
    unsqueeze_1242: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_1497, 0);  mul_1497 = None
    unsqueeze_1243: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1242, 2);  unsqueeze_1242 = None
    unsqueeze_1244: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1243, 3);  unsqueeze_1243 = None
    mul_1498: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_37, primals_25);  primals_25 = None
    unsqueeze_1245: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_1498, 0);  mul_1498 = None
    unsqueeze_1246: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1245, 2);  unsqueeze_1245 = None
    unsqueeze_1247: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1246, 3);  unsqueeze_1246 = None
    mul_1499: "f32[8, 48, 56, 56]" = torch.ops.aten.mul.Tensor(sub_383, unsqueeze_1244);  sub_383 = unsqueeze_1244 = None
    sub_385: "f32[8, 48, 56, 56]" = torch.ops.aten.sub.Tensor(where_155, mul_1499);  where_155 = mul_1499 = None
    sub_386: "f32[8, 48, 56, 56]" = torch.ops.aten.sub.Tensor(sub_385, unsqueeze_1241);  sub_385 = unsqueeze_1241 = None
    mul_1500: "f32[8, 48, 56, 56]" = torch.ops.aten.mul.Tensor(sub_386, unsqueeze_1247);  sub_386 = unsqueeze_1247 = None
    mul_1501: "f32[48]" = torch.ops.aten.mul.Tensor(sum_169, squeeze_37);  sum_169 = squeeze_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_111 = torch.ops.aten.convolution_backward.default(mul_1500, div_7, primals_189, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 48, [True, True, False]);  mul_1500 = div_7 = primals_189 = None
    getitem_507: "f32[8, 48, 56, 56]" = convolution_backward_111[0]
    getitem_508: "f32[48, 1, 5, 5]" = convolution_backward_111[1];  convolution_backward_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_87: "b8[8, 48, 56, 56]" = torch.ops.aten.lt.Scalar(clone_7, -3)
    le_69: "b8[8, 48, 56, 56]" = torch.ops.aten.le.Scalar(clone_7, 3)
    div_183: "f32[8, 48, 56, 56]" = torch.ops.aten.div.Tensor(clone_7, 3);  clone_7 = None
    add_659: "f32[8, 48, 56, 56]" = torch.ops.aten.add.Tensor(div_183, 0.5);  div_183 = None
    mul_1502: "f32[8, 48, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_507, add_659);  add_659 = None
    where_156: "f32[8, 48, 56, 56]" = torch.ops.aten.where.self(le_69, mul_1502, getitem_507);  le_69 = mul_1502 = getitem_507 = None
    where_157: "f32[8, 48, 56, 56]" = torch.ops.aten.where.self(lt_87, full_default, where_156);  lt_87 = where_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_170: "f32[48]" = torch.ops.aten.sum.dim_IntList(where_157, [0, 2, 3])
    sub_387: "f32[8, 48, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_1250);  convolution_11 = unsqueeze_1250 = None
    mul_1503: "f32[8, 48, 56, 56]" = torch.ops.aten.mul.Tensor(where_157, sub_387)
    sum_171: "f32[48]" = torch.ops.aten.sum.dim_IntList(mul_1503, [0, 2, 3]);  mul_1503 = None
    mul_1504: "f32[48]" = torch.ops.aten.mul.Tensor(sum_170, 3.985969387755102e-05)
    unsqueeze_1251: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_1504, 0);  mul_1504 = None
    unsqueeze_1252: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1251, 2);  unsqueeze_1251 = None
    unsqueeze_1253: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1252, 3);  unsqueeze_1252 = None
    mul_1505: "f32[48]" = torch.ops.aten.mul.Tensor(sum_171, 3.985969387755102e-05)
    mul_1506: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_34, squeeze_34)
    mul_1507: "f32[48]" = torch.ops.aten.mul.Tensor(mul_1505, mul_1506);  mul_1505 = mul_1506 = None
    unsqueeze_1254: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_1507, 0);  mul_1507 = None
    unsqueeze_1255: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1254, 2);  unsqueeze_1254 = None
    unsqueeze_1256: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1255, 3);  unsqueeze_1255 = None
    mul_1508: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_34, primals_23);  primals_23 = None
    unsqueeze_1257: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_1508, 0);  mul_1508 = None
    unsqueeze_1258: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1257, 2);  unsqueeze_1257 = None
    unsqueeze_1259: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1258, 3);  unsqueeze_1258 = None
    mul_1509: "f32[8, 48, 56, 56]" = torch.ops.aten.mul.Tensor(sub_387, unsqueeze_1256);  sub_387 = unsqueeze_1256 = None
    sub_389: "f32[8, 48, 56, 56]" = torch.ops.aten.sub.Tensor(where_157, mul_1509);  where_157 = mul_1509 = None
    sub_390: "f32[8, 48, 56, 56]" = torch.ops.aten.sub.Tensor(sub_389, unsqueeze_1253);  sub_389 = unsqueeze_1253 = None
    mul_1510: "f32[8, 48, 56, 56]" = torch.ops.aten.mul.Tensor(sub_390, unsqueeze_1259);  sub_390 = unsqueeze_1259 = None
    mul_1511: "f32[48]" = torch.ops.aten.mul.Tensor(sum_171, squeeze_34);  sum_171 = squeeze_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_112 = torch.ops.aten.convolution_backward.default(mul_1510, add_64, primals_188, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1510 = add_64 = primals_188 = None
    getitem_510: "f32[8, 24, 56, 56]" = convolution_backward_112[0]
    getitem_511: "f32[48, 24, 1, 1]" = convolution_backward_112[1];  convolution_backward_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_660: "f32[8, 24, 56, 56]" = torch.ops.aten.add.Tensor(add_657, getitem_510);  add_657 = getitem_510 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_172: "f32[24]" = torch.ops.aten.sum.dim_IntList(add_660, [0, 2, 3])
    sub_391: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_1262);  convolution_10 = unsqueeze_1262 = None
    mul_1512: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(add_660, sub_391)
    sum_173: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_1512, [0, 2, 3]);  mul_1512 = None
    mul_1513: "f32[24]" = torch.ops.aten.mul.Tensor(sum_172, 3.985969387755102e-05)
    unsqueeze_1263: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_1513, 0);  mul_1513 = None
    unsqueeze_1264: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1263, 2);  unsqueeze_1263 = None
    unsqueeze_1265: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1264, 3);  unsqueeze_1264 = None
    mul_1514: "f32[24]" = torch.ops.aten.mul.Tensor(sum_173, 3.985969387755102e-05)
    mul_1515: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
    mul_1516: "f32[24]" = torch.ops.aten.mul.Tensor(mul_1514, mul_1515);  mul_1514 = mul_1515 = None
    unsqueeze_1266: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_1516, 0);  mul_1516 = None
    unsqueeze_1267: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1266, 2);  unsqueeze_1266 = None
    unsqueeze_1268: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1267, 3);  unsqueeze_1267 = None
    mul_1517: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_31, primals_21);  primals_21 = None
    unsqueeze_1269: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_1517, 0);  mul_1517 = None
    unsqueeze_1270: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1269, 2);  unsqueeze_1269 = None
    unsqueeze_1271: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1270, 3);  unsqueeze_1270 = None
    mul_1518: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_391, unsqueeze_1268);  sub_391 = unsqueeze_1268 = None
    sub_393: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(add_660, mul_1518);  mul_1518 = None
    sub_394: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(sub_393, unsqueeze_1265);  sub_393 = unsqueeze_1265 = None
    mul_1519: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_394, unsqueeze_1271);  sub_394 = unsqueeze_1271 = None
    mul_1520: "f32[24]" = torch.ops.aten.mul.Tensor(sum_173, squeeze_31);  sum_173 = squeeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_113 = torch.ops.aten.convolution_backward.default(mul_1519, div_6, primals_187, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1519 = div_6 = primals_187 = None
    getitem_513: "f32[8, 48, 56, 56]" = convolution_backward_113[0]
    getitem_514: "f32[24, 48, 1, 1]" = convolution_backward_113[1];  convolution_backward_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_88: "b8[8, 48, 56, 56]" = torch.ops.aten.lt.Scalar(clone_6, -3)
    le_70: "b8[8, 48, 56, 56]" = torch.ops.aten.le.Scalar(clone_6, 3)
    div_184: "f32[8, 48, 56, 56]" = torch.ops.aten.div.Tensor(clone_6, 3);  clone_6 = None
    add_661: "f32[8, 48, 56, 56]" = torch.ops.aten.add.Tensor(div_184, 0.5);  div_184 = None
    mul_1521: "f32[8, 48, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_513, add_661);  add_661 = None
    where_158: "f32[8, 48, 56, 56]" = torch.ops.aten.where.self(le_70, mul_1521, getitem_513);  le_70 = mul_1521 = getitem_513 = None
    where_159: "f32[8, 48, 56, 56]" = torch.ops.aten.where.self(lt_88, full_default, where_158);  lt_88 = where_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_174: "f32[48]" = torch.ops.aten.sum.dim_IntList(where_159, [0, 2, 3])
    sub_395: "f32[8, 48, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_1274);  convolution_9 = unsqueeze_1274 = None
    mul_1522: "f32[8, 48, 56, 56]" = torch.ops.aten.mul.Tensor(where_159, sub_395)
    sum_175: "f32[48]" = torch.ops.aten.sum.dim_IntList(mul_1522, [0, 2, 3]);  mul_1522 = None
    mul_1523: "f32[48]" = torch.ops.aten.mul.Tensor(sum_174, 3.985969387755102e-05)
    unsqueeze_1275: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_1523, 0);  mul_1523 = None
    unsqueeze_1276: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1275, 2);  unsqueeze_1275 = None
    unsqueeze_1277: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1276, 3);  unsqueeze_1276 = None
    mul_1524: "f32[48]" = torch.ops.aten.mul.Tensor(sum_175, 3.985969387755102e-05)
    mul_1525: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_28, squeeze_28)
    mul_1526: "f32[48]" = torch.ops.aten.mul.Tensor(mul_1524, mul_1525);  mul_1524 = mul_1525 = None
    unsqueeze_1278: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_1526, 0);  mul_1526 = None
    unsqueeze_1279: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1278, 2);  unsqueeze_1278 = None
    unsqueeze_1280: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1279, 3);  unsqueeze_1279 = None
    mul_1527: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_28, primals_19);  primals_19 = None
    unsqueeze_1281: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_1527, 0);  mul_1527 = None
    unsqueeze_1282: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1281, 2);  unsqueeze_1281 = None
    unsqueeze_1283: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1282, 3);  unsqueeze_1282 = None
    mul_1528: "f32[8, 48, 56, 56]" = torch.ops.aten.mul.Tensor(sub_395, unsqueeze_1280);  sub_395 = unsqueeze_1280 = None
    sub_397: "f32[8, 48, 56, 56]" = torch.ops.aten.sub.Tensor(where_159, mul_1528);  where_159 = mul_1528 = None
    sub_398: "f32[8, 48, 56, 56]" = torch.ops.aten.sub.Tensor(sub_397, unsqueeze_1277);  sub_397 = unsqueeze_1277 = None
    mul_1529: "f32[8, 48, 56, 56]" = torch.ops.aten.mul.Tensor(sub_398, unsqueeze_1283);  sub_398 = unsqueeze_1283 = None
    mul_1530: "f32[48]" = torch.ops.aten.mul.Tensor(sum_175, squeeze_28);  sum_175 = squeeze_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_114 = torch.ops.aten.convolution_backward.default(mul_1529, div_5, primals_186, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 48, [True, True, False]);  mul_1529 = div_5 = primals_186 = None
    getitem_516: "f32[8, 48, 56, 56]" = convolution_backward_114[0]
    getitem_517: "f32[48, 1, 5, 5]" = convolution_backward_114[1];  convolution_backward_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_89: "b8[8, 48, 56, 56]" = torch.ops.aten.lt.Scalar(clone_5, -3)
    le_71: "b8[8, 48, 56, 56]" = torch.ops.aten.le.Scalar(clone_5, 3)
    div_185: "f32[8, 48, 56, 56]" = torch.ops.aten.div.Tensor(clone_5, 3);  clone_5 = None
    add_662: "f32[8, 48, 56, 56]" = torch.ops.aten.add.Tensor(div_185, 0.5);  div_185 = None
    mul_1531: "f32[8, 48, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_516, add_662);  add_662 = None
    where_160: "f32[8, 48, 56, 56]" = torch.ops.aten.where.self(le_71, mul_1531, getitem_516);  le_71 = mul_1531 = getitem_516 = None
    where_161: "f32[8, 48, 56, 56]" = torch.ops.aten.where.self(lt_89, full_default, where_160);  lt_89 = where_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_176: "f32[48]" = torch.ops.aten.sum.dim_IntList(where_161, [0, 2, 3])
    sub_399: "f32[8, 48, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_1286);  convolution_8 = unsqueeze_1286 = None
    mul_1532: "f32[8, 48, 56, 56]" = torch.ops.aten.mul.Tensor(where_161, sub_399)
    sum_177: "f32[48]" = torch.ops.aten.sum.dim_IntList(mul_1532, [0, 2, 3]);  mul_1532 = None
    mul_1533: "f32[48]" = torch.ops.aten.mul.Tensor(sum_176, 3.985969387755102e-05)
    unsqueeze_1287: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_1533, 0);  mul_1533 = None
    unsqueeze_1288: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1287, 2);  unsqueeze_1287 = None
    unsqueeze_1289: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1288, 3);  unsqueeze_1288 = None
    mul_1534: "f32[48]" = torch.ops.aten.mul.Tensor(sum_177, 3.985969387755102e-05)
    mul_1535: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
    mul_1536: "f32[48]" = torch.ops.aten.mul.Tensor(mul_1534, mul_1535);  mul_1534 = mul_1535 = None
    unsqueeze_1290: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_1536, 0);  mul_1536 = None
    unsqueeze_1291: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1290, 2);  unsqueeze_1290 = None
    unsqueeze_1292: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1291, 3);  unsqueeze_1291 = None
    mul_1537: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_25, primals_17);  primals_17 = None
    unsqueeze_1293: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_1537, 0);  mul_1537 = None
    unsqueeze_1294: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1293, 2);  unsqueeze_1293 = None
    unsqueeze_1295: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1294, 3);  unsqueeze_1294 = None
    mul_1538: "f32[8, 48, 56, 56]" = torch.ops.aten.mul.Tensor(sub_399, unsqueeze_1292);  sub_399 = unsqueeze_1292 = None
    sub_401: "f32[8, 48, 56, 56]" = torch.ops.aten.sub.Tensor(where_161, mul_1538);  where_161 = mul_1538 = None
    sub_402: "f32[8, 48, 56, 56]" = torch.ops.aten.sub.Tensor(sub_401, unsqueeze_1289);  sub_401 = unsqueeze_1289 = None
    mul_1539: "f32[8, 48, 56, 56]" = torch.ops.aten.mul.Tensor(sub_402, unsqueeze_1295);  sub_402 = unsqueeze_1295 = None
    mul_1540: "f32[48]" = torch.ops.aten.mul.Tensor(sum_177, squeeze_25);  sum_177 = squeeze_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_115 = torch.ops.aten.convolution_backward.default(mul_1539, add_46, primals_185, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1539 = add_46 = primals_185 = None
    getitem_519: "f32[8, 24, 56, 56]" = convolution_backward_115[0]
    getitem_520: "f32[48, 24, 1, 1]" = convolution_backward_115[1];  convolution_backward_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_663: "f32[8, 24, 56, 56]" = torch.ops.aten.add.Tensor(add_660, getitem_519);  add_660 = getitem_519 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_178: "f32[24]" = torch.ops.aten.sum.dim_IntList(add_663, [0, 2, 3])
    sub_403: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_1298);  convolution_7 = unsqueeze_1298 = None
    mul_1541: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(add_663, sub_403)
    sum_179: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_1541, [0, 2, 3]);  mul_1541 = None
    mul_1542: "f32[24]" = torch.ops.aten.mul.Tensor(sum_178, 3.985969387755102e-05)
    unsqueeze_1299: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_1542, 0);  mul_1542 = None
    unsqueeze_1300: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1299, 2);  unsqueeze_1299 = None
    unsqueeze_1301: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1300, 3);  unsqueeze_1300 = None
    mul_1543: "f32[24]" = torch.ops.aten.mul.Tensor(sum_179, 3.985969387755102e-05)
    mul_1544: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_22, squeeze_22)
    mul_1545: "f32[24]" = torch.ops.aten.mul.Tensor(mul_1543, mul_1544);  mul_1543 = mul_1544 = None
    unsqueeze_1302: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_1545, 0);  mul_1545 = None
    unsqueeze_1303: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1302, 2);  unsqueeze_1302 = None
    unsqueeze_1304: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1303, 3);  unsqueeze_1303 = None
    mul_1546: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_22, primals_15);  primals_15 = None
    unsqueeze_1305: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_1546, 0);  mul_1546 = None
    unsqueeze_1306: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1305, 2);  unsqueeze_1305 = None
    unsqueeze_1307: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1306, 3);  unsqueeze_1306 = None
    mul_1547: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_403, unsqueeze_1304);  sub_403 = unsqueeze_1304 = None
    sub_405: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(add_663, mul_1547);  add_663 = mul_1547 = None
    sub_406: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(sub_405, unsqueeze_1301);  sub_405 = unsqueeze_1301 = None
    mul_1548: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_406, unsqueeze_1307);  sub_406 = unsqueeze_1307 = None
    mul_1549: "f32[24]" = torch.ops.aten.mul.Tensor(sum_179, squeeze_22);  sum_179 = squeeze_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_116 = torch.ops.aten.convolution_backward.default(mul_1548, div_4, primals_184, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1548 = div_4 = primals_184 = None
    getitem_522: "f32[8, 64, 56, 56]" = convolution_backward_116[0]
    getitem_523: "f32[24, 64, 1, 1]" = convolution_backward_116[1];  convolution_backward_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_90: "b8[8, 64, 56, 56]" = torch.ops.aten.lt.Scalar(clone_4, -3)
    le_72: "b8[8, 64, 56, 56]" = torch.ops.aten.le.Scalar(clone_4, 3)
    div_186: "f32[8, 64, 56, 56]" = torch.ops.aten.div.Tensor(clone_4, 3);  clone_4 = None
    add_664: "f32[8, 64, 56, 56]" = torch.ops.aten.add.Tensor(div_186, 0.5);  div_186 = None
    mul_1550: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_522, add_664);  add_664 = None
    where_162: "f32[8, 64, 56, 56]" = torch.ops.aten.where.self(le_72, mul_1550, getitem_522);  le_72 = mul_1550 = getitem_522 = None
    where_163: "f32[8, 64, 56, 56]" = torch.ops.aten.where.self(lt_90, full_default, where_162);  lt_90 = where_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_180: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_163, [0, 2, 3])
    sub_407: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_1310);  convolution_6 = unsqueeze_1310 = None
    mul_1551: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(where_163, sub_407)
    sum_181: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1551, [0, 2, 3]);  mul_1551 = None
    mul_1552: "f32[64]" = torch.ops.aten.mul.Tensor(sum_180, 3.985969387755102e-05)
    unsqueeze_1311: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1552, 0);  mul_1552 = None
    unsqueeze_1312: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1311, 2);  unsqueeze_1311 = None
    unsqueeze_1313: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1312, 3);  unsqueeze_1312 = None
    mul_1553: "f32[64]" = torch.ops.aten.mul.Tensor(sum_181, 3.985969387755102e-05)
    mul_1554: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_19, squeeze_19)
    mul_1555: "f32[64]" = torch.ops.aten.mul.Tensor(mul_1553, mul_1554);  mul_1553 = mul_1554 = None
    unsqueeze_1314: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1555, 0);  mul_1555 = None
    unsqueeze_1315: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1314, 2);  unsqueeze_1314 = None
    unsqueeze_1316: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1315, 3);  unsqueeze_1315 = None
    mul_1556: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_19, primals_13);  primals_13 = None
    unsqueeze_1317: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1556, 0);  mul_1556 = None
    unsqueeze_1318: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1317, 2);  unsqueeze_1317 = None
    unsqueeze_1319: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1318, 3);  unsqueeze_1318 = None
    mul_1557: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_407, unsqueeze_1316);  sub_407 = unsqueeze_1316 = None
    sub_409: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(where_163, mul_1557);  where_163 = mul_1557 = None
    sub_410: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(sub_409, unsqueeze_1313);  sub_409 = unsqueeze_1313 = None
    mul_1558: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_410, unsqueeze_1319);  sub_410 = unsqueeze_1319 = None
    mul_1559: "f32[64]" = torch.ops.aten.mul.Tensor(sum_181, squeeze_19);  sum_181 = squeeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_117 = torch.ops.aten.convolution_backward.default(mul_1558, div_3, primals_183, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 64, [True, True, False]);  mul_1558 = div_3 = primals_183 = None
    getitem_525: "f32[8, 64, 112, 112]" = convolution_backward_117[0]
    getitem_526: "f32[64, 1, 5, 5]" = convolution_backward_117[1];  convolution_backward_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_91: "b8[8, 64, 112, 112]" = torch.ops.aten.lt.Scalar(clone_3, -3)
    le_73: "b8[8, 64, 112, 112]" = torch.ops.aten.le.Scalar(clone_3, 3)
    div_187: "f32[8, 64, 112, 112]" = torch.ops.aten.div.Tensor(clone_3, 3);  clone_3 = None
    add_665: "f32[8, 64, 112, 112]" = torch.ops.aten.add.Tensor(div_187, 0.5);  div_187 = None
    mul_1560: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(getitem_525, add_665);  add_665 = None
    where_164: "f32[8, 64, 112, 112]" = torch.ops.aten.where.self(le_73, mul_1560, getitem_525);  le_73 = mul_1560 = getitem_525 = None
    where_165: "f32[8, 64, 112, 112]" = torch.ops.aten.where.self(lt_91, full_default, where_164);  lt_91 = where_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_182: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_165, [0, 2, 3])
    sub_411: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_1322);  convolution_5 = unsqueeze_1322 = None
    mul_1561: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(where_165, sub_411)
    sum_183: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1561, [0, 2, 3]);  mul_1561 = None
    mul_1562: "f32[64]" = torch.ops.aten.mul.Tensor(sum_182, 9.964923469387754e-06)
    unsqueeze_1323: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1562, 0);  mul_1562 = None
    unsqueeze_1324: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1323, 2);  unsqueeze_1323 = None
    unsqueeze_1325: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1324, 3);  unsqueeze_1324 = None
    mul_1563: "f32[64]" = torch.ops.aten.mul.Tensor(sum_183, 9.964923469387754e-06)
    mul_1564: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_16, squeeze_16)
    mul_1565: "f32[64]" = torch.ops.aten.mul.Tensor(mul_1563, mul_1564);  mul_1563 = mul_1564 = None
    unsqueeze_1326: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1565, 0);  mul_1565 = None
    unsqueeze_1327: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1326, 2);  unsqueeze_1326 = None
    unsqueeze_1328: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1327, 3);  unsqueeze_1327 = None
    mul_1566: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_16, primals_11);  primals_11 = None
    unsqueeze_1329: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1566, 0);  mul_1566 = None
    unsqueeze_1330: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1329, 2);  unsqueeze_1329 = None
    unsqueeze_1331: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1330, 3);  unsqueeze_1330 = None
    mul_1567: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_411, unsqueeze_1328);  sub_411 = unsqueeze_1328 = None
    sub_413: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(where_165, mul_1567);  where_165 = mul_1567 = None
    sub_414: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(sub_413, unsqueeze_1325);  sub_413 = unsqueeze_1325 = None
    mul_1568: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_414, unsqueeze_1331);  sub_414 = unsqueeze_1331 = None
    mul_1569: "f32[64]" = torch.ops.aten.mul.Tensor(sum_183, squeeze_16);  sum_183 = squeeze_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_118 = torch.ops.aten.convolution_backward.default(mul_1568, add_29, primals_182, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1568 = add_29 = primals_182 = None
    getitem_528: "f32[8, 16, 112, 112]" = convolution_backward_118[0]
    getitem_529: "f32[64, 16, 1, 1]" = convolution_backward_118[1];  convolution_backward_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_184: "f32[16]" = torch.ops.aten.sum.dim_IntList(getitem_528, [0, 2, 3])
    sub_415: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_1334);  convolution_4 = unsqueeze_1334 = None
    mul_1570: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(getitem_528, sub_415)
    sum_185: "f32[16]" = torch.ops.aten.sum.dim_IntList(mul_1570, [0, 2, 3]);  mul_1570 = None
    mul_1571: "f32[16]" = torch.ops.aten.mul.Tensor(sum_184, 9.964923469387754e-06)
    unsqueeze_1335: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1571, 0);  mul_1571 = None
    unsqueeze_1336: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1335, 2);  unsqueeze_1335 = None
    unsqueeze_1337: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1336, 3);  unsqueeze_1336 = None
    mul_1572: "f32[16]" = torch.ops.aten.mul.Tensor(sum_185, 9.964923469387754e-06)
    mul_1573: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
    mul_1574: "f32[16]" = torch.ops.aten.mul.Tensor(mul_1572, mul_1573);  mul_1572 = mul_1573 = None
    unsqueeze_1338: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1574, 0);  mul_1574 = None
    unsqueeze_1339: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1338, 2);  unsqueeze_1338 = None
    unsqueeze_1340: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1339, 3);  unsqueeze_1339 = None
    mul_1575: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_13, primals_9);  primals_9 = None
    unsqueeze_1341: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1575, 0);  mul_1575 = None
    unsqueeze_1342: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1341, 2);  unsqueeze_1341 = None
    unsqueeze_1343: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1342, 3);  unsqueeze_1342 = None
    mul_1576: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_415, unsqueeze_1340);  sub_415 = unsqueeze_1340 = None
    sub_417: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(getitem_528, mul_1576);  mul_1576 = None
    sub_418: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(sub_417, unsqueeze_1337);  sub_417 = unsqueeze_1337 = None
    mul_1577: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_418, unsqueeze_1343);  sub_418 = unsqueeze_1343 = None
    mul_1578: "f32[16]" = torch.ops.aten.mul.Tensor(sum_185, squeeze_13);  sum_185 = squeeze_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:126, code: x = self.conv_pw(x)
    convolution_backward_119 = torch.ops.aten.convolution_backward.default(mul_1577, div_2, primals_181, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1577 = div_2 = primals_181 = None
    getitem_531: "f32[8, 16, 112, 112]" = convolution_backward_119[0]
    getitem_532: "f32[16, 16, 1, 1]" = convolution_backward_119[1];  convolution_backward_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_92: "b8[8, 16, 112, 112]" = torch.ops.aten.lt.Scalar(clone_2, -3)
    le_74: "b8[8, 16, 112, 112]" = torch.ops.aten.le.Scalar(clone_2, 3)
    div_188: "f32[8, 16, 112, 112]" = torch.ops.aten.div.Tensor(clone_2, 3);  clone_2 = None
    add_666: "f32[8, 16, 112, 112]" = torch.ops.aten.add.Tensor(div_188, 0.5);  div_188 = None
    mul_1579: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(getitem_531, add_666);  add_666 = None
    where_166: "f32[8, 16, 112, 112]" = torch.ops.aten.where.self(le_74, mul_1579, getitem_531);  le_74 = mul_1579 = getitem_531 = None
    where_167: "f32[8, 16, 112, 112]" = torch.ops.aten.where.self(lt_92, full_default, where_166);  lt_92 = where_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_186: "f32[16]" = torch.ops.aten.sum.dim_IntList(where_167, [0, 2, 3])
    sub_419: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_1346);  convolution_3 = unsqueeze_1346 = None
    mul_1580: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(where_167, sub_419)
    sum_187: "f32[16]" = torch.ops.aten.sum.dim_IntList(mul_1580, [0, 2, 3]);  mul_1580 = None
    mul_1581: "f32[16]" = torch.ops.aten.mul.Tensor(sum_186, 9.964923469387754e-06)
    unsqueeze_1347: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1581, 0);  mul_1581 = None
    unsqueeze_1348: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1347, 2);  unsqueeze_1347 = None
    unsqueeze_1349: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1348, 3);  unsqueeze_1348 = None
    mul_1582: "f32[16]" = torch.ops.aten.mul.Tensor(sum_187, 9.964923469387754e-06)
    mul_1583: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_10, squeeze_10)
    mul_1584: "f32[16]" = torch.ops.aten.mul.Tensor(mul_1582, mul_1583);  mul_1582 = mul_1583 = None
    unsqueeze_1350: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1584, 0);  mul_1584 = None
    unsqueeze_1351: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1350, 2);  unsqueeze_1350 = None
    unsqueeze_1352: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1351, 3);  unsqueeze_1351 = None
    mul_1585: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_10, primals_7);  primals_7 = None
    unsqueeze_1353: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1585, 0);  mul_1585 = None
    unsqueeze_1354: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1353, 2);  unsqueeze_1353 = None
    unsqueeze_1355: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1354, 3);  unsqueeze_1354 = None
    mul_1586: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_419, unsqueeze_1352);  sub_419 = unsqueeze_1352 = None
    sub_421: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(where_167, mul_1586);  where_167 = mul_1586 = None
    sub_422: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(sub_421, unsqueeze_1349);  sub_421 = unsqueeze_1349 = None
    mul_1587: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_422, unsqueeze_1355);  sub_422 = unsqueeze_1355 = None
    mul_1588: "f32[16]" = torch.ops.aten.mul.Tensor(sum_187, squeeze_10);  sum_187 = squeeze_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:123, code: x = self.conv_dw(x)
    convolution_backward_120 = torch.ops.aten.convolution_backward.default(mul_1587, add_17, primals_180, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 16, [True, True, False]);  mul_1587 = add_17 = primals_180 = None
    getitem_534: "f32[8, 16, 112, 112]" = convolution_backward_120[0]
    getitem_535: "f32[16, 1, 3, 3]" = convolution_backward_120[1];  convolution_backward_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:123, code: x = self.conv_dw(x)
    add_667: "f32[8, 16, 112, 112]" = torch.ops.aten.add.Tensor(getitem_528, getitem_534);  getitem_528 = getitem_534 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_188: "f32[16]" = torch.ops.aten.sum.dim_IntList(add_667, [0, 2, 3])
    sub_423: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_1358);  convolution_2 = unsqueeze_1358 = None
    mul_1589: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(add_667, sub_423)
    sum_189: "f32[16]" = torch.ops.aten.sum.dim_IntList(mul_1589, [0, 2, 3]);  mul_1589 = None
    mul_1590: "f32[16]" = torch.ops.aten.mul.Tensor(sum_188, 9.964923469387754e-06)
    unsqueeze_1359: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1590, 0);  mul_1590 = None
    unsqueeze_1360: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1359, 2);  unsqueeze_1359 = None
    unsqueeze_1361: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1360, 3);  unsqueeze_1360 = None
    mul_1591: "f32[16]" = torch.ops.aten.mul.Tensor(sum_189, 9.964923469387754e-06)
    mul_1592: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
    mul_1593: "f32[16]" = torch.ops.aten.mul.Tensor(mul_1591, mul_1592);  mul_1591 = mul_1592 = None
    unsqueeze_1362: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1593, 0);  mul_1593 = None
    unsqueeze_1363: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1362, 2);  unsqueeze_1362 = None
    unsqueeze_1364: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1363, 3);  unsqueeze_1363 = None
    mul_1594: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_7, primals_5);  primals_5 = None
    unsqueeze_1365: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1594, 0);  mul_1594 = None
    unsqueeze_1366: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1365, 2);  unsqueeze_1365 = None
    unsqueeze_1367: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1366, 3);  unsqueeze_1366 = None
    mul_1595: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_423, unsqueeze_1364);  sub_423 = unsqueeze_1364 = None
    sub_425: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(add_667, mul_1595);  mul_1595 = None
    sub_426: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(sub_425, unsqueeze_1361);  sub_425 = unsqueeze_1361 = None
    mul_1596: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_426, unsqueeze_1367);  sub_426 = unsqueeze_1367 = None
    mul_1597: "f32[16]" = torch.ops.aten.mul.Tensor(sum_189, squeeze_7);  sum_189 = squeeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:126, code: x = self.conv_pw(x)
    convolution_backward_121 = torch.ops.aten.convolution_backward.default(mul_1596, div_1, primals_179, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1596 = div_1 = primals_179 = None
    getitem_537: "f32[8, 16, 112, 112]" = convolution_backward_121[0]
    getitem_538: "f32[16, 16, 1, 1]" = convolution_backward_121[1];  convolution_backward_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_93: "b8[8, 16, 112, 112]" = torch.ops.aten.lt.Scalar(clone_1, -3)
    le_75: "b8[8, 16, 112, 112]" = torch.ops.aten.le.Scalar(clone_1, 3)
    div_189: "f32[8, 16, 112, 112]" = torch.ops.aten.div.Tensor(clone_1, 3);  clone_1 = None
    add_668: "f32[8, 16, 112, 112]" = torch.ops.aten.add.Tensor(div_189, 0.5);  div_189 = None
    mul_1598: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(getitem_537, add_668);  add_668 = None
    where_168: "f32[8, 16, 112, 112]" = torch.ops.aten.where.self(le_75, mul_1598, getitem_537);  le_75 = mul_1598 = getitem_537 = None
    where_169: "f32[8, 16, 112, 112]" = torch.ops.aten.where.self(lt_93, full_default, where_168);  lt_93 = where_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_190: "f32[16]" = torch.ops.aten.sum.dim_IntList(where_169, [0, 2, 3])
    sub_427: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_1370);  convolution_1 = unsqueeze_1370 = None
    mul_1599: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(where_169, sub_427)
    sum_191: "f32[16]" = torch.ops.aten.sum.dim_IntList(mul_1599, [0, 2, 3]);  mul_1599 = None
    mul_1600: "f32[16]" = torch.ops.aten.mul.Tensor(sum_190, 9.964923469387754e-06)
    unsqueeze_1371: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1600, 0);  mul_1600 = None
    unsqueeze_1372: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1371, 2);  unsqueeze_1371 = None
    unsqueeze_1373: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1372, 3);  unsqueeze_1372 = None
    mul_1601: "f32[16]" = torch.ops.aten.mul.Tensor(sum_191, 9.964923469387754e-06)
    mul_1602: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
    mul_1603: "f32[16]" = torch.ops.aten.mul.Tensor(mul_1601, mul_1602);  mul_1601 = mul_1602 = None
    unsqueeze_1374: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1603, 0);  mul_1603 = None
    unsqueeze_1375: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1374, 2);  unsqueeze_1374 = None
    unsqueeze_1376: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1375, 3);  unsqueeze_1375 = None
    mul_1604: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_4, primals_3);  primals_3 = None
    unsqueeze_1377: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1604, 0);  mul_1604 = None
    unsqueeze_1378: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1377, 2);  unsqueeze_1377 = None
    unsqueeze_1379: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1378, 3);  unsqueeze_1378 = None
    mul_1605: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_427, unsqueeze_1376);  sub_427 = unsqueeze_1376 = None
    sub_429: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(where_169, mul_1605);  where_169 = mul_1605 = None
    sub_430: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(sub_429, unsqueeze_1373);  sub_429 = unsqueeze_1373 = None
    mul_1606: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_430, unsqueeze_1379);  sub_430 = unsqueeze_1379 = None
    mul_1607: "f32[16]" = torch.ops.aten.mul.Tensor(sum_191, squeeze_4);  sum_191 = squeeze_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:123, code: x = self.conv_dw(x)
    convolution_backward_122 = torch.ops.aten.convolution_backward.default(mul_1606, div, primals_178, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 16, [True, True, False]);  mul_1606 = div = primals_178 = None
    getitem_540: "f32[8, 16, 112, 112]" = convolution_backward_122[0]
    getitem_541: "f32[16, 1, 3, 3]" = convolution_backward_122[1];  convolution_backward_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:123, code: x = self.conv_dw(x)
    add_669: "f32[8, 16, 112, 112]" = torch.ops.aten.add.Tensor(add_667, getitem_540);  add_667 = getitem_540 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_94: "b8[8, 16, 112, 112]" = torch.ops.aten.lt.Scalar(clone, -3)
    le_76: "b8[8, 16, 112, 112]" = torch.ops.aten.le.Scalar(clone, 3)
    div_190: "f32[8, 16, 112, 112]" = torch.ops.aten.div.Tensor(clone, 3);  clone = None
    add_670: "f32[8, 16, 112, 112]" = torch.ops.aten.add.Tensor(div_190, 0.5);  div_190 = None
    mul_1608: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(add_669, add_670);  add_670 = None
    where_170: "f32[8, 16, 112, 112]" = torch.ops.aten.where.self(le_76, mul_1608, add_669);  le_76 = mul_1608 = add_669 = None
    where_171: "f32[8, 16, 112, 112]" = torch.ops.aten.where.self(lt_94, full_default, where_170);  lt_94 = full_default = where_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_192: "f32[16]" = torch.ops.aten.sum.dim_IntList(where_171, [0, 2, 3])
    sub_431: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_1382);  convolution = unsqueeze_1382 = None
    mul_1609: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(where_171, sub_431)
    sum_193: "f32[16]" = torch.ops.aten.sum.dim_IntList(mul_1609, [0, 2, 3]);  mul_1609 = None
    mul_1610: "f32[16]" = torch.ops.aten.mul.Tensor(sum_192, 9.964923469387754e-06)
    unsqueeze_1383: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1610, 0);  mul_1610 = None
    unsqueeze_1384: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1383, 2);  unsqueeze_1383 = None
    unsqueeze_1385: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1384, 3);  unsqueeze_1384 = None
    mul_1611: "f32[16]" = torch.ops.aten.mul.Tensor(sum_193, 9.964923469387754e-06)
    mul_1612: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
    mul_1613: "f32[16]" = torch.ops.aten.mul.Tensor(mul_1611, mul_1612);  mul_1611 = mul_1612 = None
    unsqueeze_1386: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1613, 0);  mul_1613 = None
    unsqueeze_1387: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1386, 2);  unsqueeze_1386 = None
    unsqueeze_1388: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1387, 3);  unsqueeze_1387 = None
    mul_1614: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_1, primals_1);  primals_1 = None
    unsqueeze_1389: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1614, 0);  mul_1614 = None
    unsqueeze_1390: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1389, 2);  unsqueeze_1389 = None
    unsqueeze_1391: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1390, 3);  unsqueeze_1390 = None
    mul_1615: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_431, unsqueeze_1388);  sub_431 = unsqueeze_1388 = None
    sub_433: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(where_171, mul_1615);  where_171 = mul_1615 = None
    sub_434: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(sub_433, unsqueeze_1385);  sub_433 = unsqueeze_1385 = None
    mul_1616: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_434, unsqueeze_1391);  sub_434 = unsqueeze_1391 = None
    mul_1617: "f32[16]" = torch.ops.aten.mul.Tensor(sum_193, squeeze_1);  sum_193 = squeeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilenetv3.py:135, code: x = self.conv_stem(x)
    convolution_backward_123 = torch.ops.aten.convolution_backward.default(mul_1616, primals_598, primals_177, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_1616 = primals_598 = primals_177 = None
    getitem_544: "f32[16, 3, 3, 3]" = convolution_backward_123[1];  convolution_backward_123 = None
    return [mul_1617, sum_192, mul_1607, sum_190, mul_1597, sum_188, mul_1588, sum_186, mul_1578, sum_184, mul_1569, sum_182, mul_1559, sum_180, mul_1549, sum_178, mul_1540, sum_176, mul_1530, sum_174, mul_1520, sum_172, mul_1511, sum_170, mul_1501, sum_168, mul_1491, sum_166, mul_1482, sum_164, mul_1472, sum_162, mul_1462, sum_160, mul_1453, sum_158, mul_1443, sum_156, mul_1429, sum_153, mul_1420, sum_151, mul_1410, sum_149, mul_1396, sum_146, mul_1387, sum_144, mul_1377, sum_142, mul_1363, sum_139, mul_1354, sum_137, mul_1344, sum_135, mul_1330, sum_132, mul_1321, sum_130, mul_1311, sum_128, mul_1297, sum_125, mul_1288, sum_123, mul_1278, sum_121, mul_1268, sum_119, mul_1259, sum_117, mul_1249, sum_115, mul_1239, sum_113, mul_1230, sum_111, mul_1220, sum_109, mul_1210, sum_107, mul_1201, sum_105, mul_1191, sum_103, mul_1181, sum_101, mul_1172, sum_99, mul_1162, sum_97, mul_1152, sum_95, mul_1143, sum_93, mul_1133, sum_91, mul_1119, sum_88, mul_1110, sum_86, mul_1100, sum_84, mul_1086, sum_81, mul_1077, sum_79, mul_1067, sum_77, mul_1053, sum_74, mul_1044, sum_72, mul_1034, sum_70, mul_1020, sum_67, mul_1011, sum_65, mul_1001, sum_63, mul_987, sum_60, mul_978, sum_58, mul_968, sum_56, mul_954, sum_53, mul_945, sum_51, mul_935, sum_49, mul_921, sum_46, mul_912, sum_44, mul_902, sum_42, mul_888, sum_39, mul_879, sum_37, mul_869, sum_35, mul_855, sum_32, mul_846, sum_30, mul_836, sum_28, mul_822, sum_25, mul_813, sum_23, mul_803, sum_21, mul_789, sum_18, mul_780, sum_16, mul_770, sum_14, mul_756, sum_11, mul_747, sum_9, mul_737, sum_7, mul_723, sum_4, mul_714, sum_2, permute_4, view_2, getitem_544, getitem_541, getitem_538, getitem_535, getitem_532, getitem_529, getitem_526, getitem_523, getitem_520, getitem_517, getitem_514, getitem_511, getitem_508, getitem_505, getitem_502, getitem_499, getitem_496, getitem_493, getitem_490, getitem_487, getitem_488, getitem_484, getitem_485, getitem_481, getitem_478, getitem_475, getitem_472, getitem_473, getitem_469, getitem_470, getitem_466, getitem_463, getitem_460, getitem_457, getitem_458, getitem_454, getitem_455, getitem_451, getitem_448, getitem_445, getitem_442, getitem_443, getitem_439, getitem_440, getitem_436, getitem_433, getitem_430, getitem_427, getitem_428, getitem_424, getitem_425, getitem_421, getitem_418, getitem_415, getitem_412, getitem_409, getitem_406, getitem_403, getitem_400, getitem_397, getitem_394, getitem_391, getitem_388, getitem_385, getitem_382, getitem_379, getitem_376, getitem_373, getitem_370, getitem_367, getitem_368, getitem_364, getitem_365, getitem_361, getitem_358, getitem_355, getitem_352, getitem_353, getitem_349, getitem_350, getitem_346, getitem_343, getitem_340, getitem_337, getitem_338, getitem_334, getitem_335, getitem_331, getitem_328, getitem_325, getitem_322, getitem_323, getitem_319, getitem_320, getitem_316, getitem_313, getitem_310, getitem_307, getitem_308, getitem_304, getitem_305, getitem_301, getitem_298, getitem_295, getitem_292, getitem_293, getitem_289, getitem_290, getitem_286, getitem_283, getitem_280, getitem_277, getitem_278, getitem_274, getitem_275, getitem_271, getitem_268, getitem_265, getitem_262, getitem_263, getitem_259, getitem_260, getitem_256, getitem_253, getitem_250, getitem_247, getitem_248, getitem_244, getitem_245, getitem_241, getitem_238, getitem_235, getitem_232, getitem_233, getitem_229, getitem_230, getitem_226, getitem_223, getitem_220, getitem_217, getitem_218, getitem_214, getitem_215, getitem_211, getitem_208, getitem_205, getitem_202, getitem_203, getitem_199, getitem_200, getitem_196, getitem_193, getitem_190, getitem_187, getitem_188, getitem_184, getitem_185, getitem_181, getitem_178, getitem_175, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
    