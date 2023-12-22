from __future__ import annotations



def forward(self, primals_1: "f32[64, 3, 4, 4]", primals_3: "f32[64]", primals_5: "f32[64]", primals_9: "f32[64, 64, 8, 8]", primals_11: "f32[64]", primals_17: "f32[64]", primals_23: "f32[64, 1, 3, 3]", primals_25: "f32[64]", primals_29: "f32[64, 64, 8, 8]", primals_31: "f32[64]", primals_37: "f32[64]", primals_43: "f32[64]", primals_47: "f32[64, 64, 8, 8]", primals_49: "f32[64]", primals_55: "f32[64]", primals_61: "f32[128, 64, 2, 2]", primals_63: "f32[128]", primals_65: "f32[128]", primals_69: "f32[128, 128, 4, 4]", primals_71: "f32[128]", primals_77: "f32[128]", primals_83: "f32[128, 1, 3, 3]", primals_85: "f32[128]", primals_89: "f32[128, 128, 4, 4]", primals_91: "f32[128]", primals_97: "f32[128]", primals_103: "f32[128]", primals_107: "f32[128, 128, 4, 4]", primals_109: "f32[128]", primals_115: "f32[128]", primals_121: "f32[128]", primals_125: "f32[128, 128, 4, 4]", primals_127: "f32[128]", primals_133: "f32[128]", primals_139: "f32[320, 128, 2, 2]", primals_141: "f32[320]", primals_143: "f32[320]", primals_147: "f32[320, 320, 2, 2]", primals_149: "f32[320]", primals_155: "f32[320]", primals_161: "f32[320, 1, 3, 3]", primals_163: "f32[320]", primals_167: "f32[320, 320, 2, 2]", primals_169: "f32[320]", primals_175: "f32[320]", primals_181: "f32[320]", primals_185: "f32[320, 320, 2, 2]", primals_187: "f32[320]", primals_193: "f32[320]", primals_199: "f32[320]", primals_203: "f32[320, 320, 2, 2]", primals_205: "f32[320]", primals_211: "f32[320]", primals_217: "f32[320]", primals_221: "f32[320, 320, 2, 2]", primals_223: "f32[320]", primals_229: "f32[320]", primals_235: "f32[320]", primals_239: "f32[320, 320, 2, 2]", primals_241: "f32[320]", primals_247: "f32[320]", primals_253: "f32[320]", primals_257: "f32[320, 320, 2, 2]", primals_259: "f32[320]", primals_265: "f32[320]", primals_271: "f32[320]", primals_275: "f32[320, 320, 2, 2]", primals_277: "f32[320]", primals_283: "f32[320]", primals_289: "f32[320]", primals_293: "f32[320, 320, 2, 2]", primals_295: "f32[320]", primals_301: "f32[320]", primals_307: "f32[320]", primals_311: "f32[320, 320, 2, 2]", primals_313: "f32[320]", primals_319: "f32[320]", primals_325: "f32[320]", primals_329: "f32[320, 320, 2, 2]", primals_331: "f32[320]", primals_337: "f32[320]", primals_343: "f32[320]", primals_347: "f32[320, 320, 2, 2]", primals_349: "f32[320]", primals_355: "f32[320]", primals_361: "f32[320]", primals_365: "f32[320, 320, 2, 2]", primals_367: "f32[320]", primals_373: "f32[320]", primals_379: "f32[320]", primals_383: "f32[320, 320, 2, 2]", primals_385: "f32[320]", primals_391: "f32[320]", primals_397: "f32[320]", primals_401: "f32[320, 320, 2, 2]", primals_403: "f32[320]", primals_409: "f32[320]", primals_415: "f32[320]", primals_419: "f32[320, 320, 2, 2]", primals_421: "f32[320]", primals_427: "f32[320]", primals_433: "f32[320]", primals_437: "f32[320, 320, 2, 2]", primals_439: "f32[320]", primals_445: "f32[320]", primals_451: "f32[320]", primals_455: "f32[320, 320, 2, 2]", primals_457: "f32[320]", primals_463: "f32[320]", primals_469: "f32[512, 320, 2, 2]", primals_471: "f32[512]", primals_473: "f32[512]", primals_481: "f32[512]", primals_487: "f32[512, 1, 3, 3]", primals_489: "f32[512]", primals_497: "f32[512]", primals_503: "f32[512]", primals_511: "f32[512]", primals_517: "f32[512]", primals_521: "f32[8, 3, 224, 224]", mul: "f32[8, 3136, 64]", mul_2: "f32[8, 3136, 64]", view_1: "f32[25088, 64]", permute_2: "f32[8, 1, 3136, 64]", view_4: "f32[8, 64, 56, 56]", mul_4: "f32[8, 49, 64]", view_6: "f32[392, 64]", getitem_6: "f32[8, 1, 49, 64]", getitem_7: "f32[8, 1, 49, 64]", getitem_9: "f32[8, 1, 3136]", getitem_10: "i64[]", getitem_11: "i64[]", view_10: "f32[25088, 64]", mul_6: "f32[8, 3136, 64]", view_12: "f32[25088, 64]", addmm_3: "f32[25088, 512]", view_14: "f32[25088, 512]", view_16: "f32[8, 64, 56, 56]", mul_11: "f32[8, 3136, 64]", view_19: "f32[25088, 64]", permute_15: "f32[8, 1, 3136, 64]", view_22: "f32[8, 64, 56, 56]", mul_13: "f32[8, 49, 64]", view_24: "f32[392, 64]", getitem_18: "f32[8, 1, 49, 64]", getitem_19: "f32[8, 1, 49, 64]", getitem_21: "f32[8, 1, 3136]", getitem_22: "i64[]", getitem_23: "i64[]", view_28: "f32[25088, 64]", mul_15: "f32[8, 3136, 64]", view_30: "f32[25088, 64]", addmm_8: "f32[25088, 512]", view_32: "f32[25088, 512]", mul_20: "f32[8, 3136, 64]", view_34: "f32[25088, 64]", permute_25: "f32[8, 1, 3136, 64]", view_37: "f32[8, 64, 56, 56]", mul_22: "f32[8, 49, 64]", view_39: "f32[392, 64]", getitem_30: "f32[8, 1, 49, 64]", getitem_31: "f32[8, 1, 49, 64]", getitem_33: "f32[8, 1, 3136]", getitem_34: "i64[]", getitem_35: "i64[]", view_43: "f32[25088, 64]", mul_24: "f32[8, 3136, 64]", view_45: "f32[25088, 64]", addmm_13: "f32[25088, 512]", view_47: "f32[25088, 512]", permute_34: "f32[8, 64, 56, 56]", mul_29: "f32[8, 784, 128]", mul_31: "f32[8, 784, 128]", view_51: "f32[6272, 128]", permute_37: "f32[8, 2, 784, 64]", view_54: "f32[8, 128, 28, 28]", mul_33: "f32[8, 49, 128]", view_56: "f32[392, 128]", getitem_44: "f32[8, 2, 49, 64]", getitem_45: "f32[8, 2, 49, 64]", getitem_47: "f32[8, 2, 800]", getitem_48: "i64[]", getitem_49: "i64[]", view_60: "f32[6272, 128]", mul_35: "f32[8, 784, 128]", view_62: "f32[6272, 128]", addmm_18: "f32[6272, 1024]", view_64: "f32[6272, 1024]", view_66: "f32[8, 128, 28, 28]", mul_40: "f32[8, 784, 128]", view_69: "f32[6272, 128]", permute_50: "f32[8, 2, 784, 64]", view_72: "f32[8, 128, 28, 28]", mul_42: "f32[8, 49, 128]", view_74: "f32[392, 128]", getitem_56: "f32[8, 2, 49, 64]", getitem_57: "f32[8, 2, 49, 64]", getitem_59: "f32[8, 2, 800]", getitem_60: "i64[]", getitem_61: "i64[]", view_78: "f32[6272, 128]", mul_44: "f32[8, 784, 128]", view_80: "f32[6272, 128]", addmm_23: "f32[6272, 1024]", view_82: "f32[6272, 1024]", mul_49: "f32[8, 784, 128]", view_84: "f32[6272, 128]", permute_60: "f32[8, 2, 784, 64]", view_87: "f32[8, 128, 28, 28]", mul_51: "f32[8, 49, 128]", view_89: "f32[392, 128]", getitem_68: "f32[8, 2, 49, 64]", getitem_69: "f32[8, 2, 49, 64]", getitem_71: "f32[8, 2, 800]", getitem_72: "i64[]", getitem_73: "i64[]", view_93: "f32[6272, 128]", mul_53: "f32[8, 784, 128]", view_95: "f32[6272, 128]", addmm_28: "f32[6272, 1024]", view_97: "f32[6272, 1024]", mul_58: "f32[8, 784, 128]", view_99: "f32[6272, 128]", permute_70: "f32[8, 2, 784, 64]", view_102: "f32[8, 128, 28, 28]", mul_60: "f32[8, 49, 128]", view_104: "f32[392, 128]", getitem_80: "f32[8, 2, 49, 64]", getitem_81: "f32[8, 2, 49, 64]", getitem_83: "f32[8, 2, 800]", getitem_84: "i64[]", getitem_85: "i64[]", view_108: "f32[6272, 128]", mul_62: "f32[8, 784, 128]", view_110: "f32[6272, 128]", addmm_33: "f32[6272, 1024]", view_112: "f32[6272, 1024]", permute_79: "f32[8, 128, 28, 28]", mul_67: "f32[8, 196, 320]", mul_69: "f32[8, 196, 320]", view_116: "f32[1568, 320]", permute_82: "f32[8, 5, 196, 64]", view_119: "f32[8, 320, 14, 14]", mul_71: "f32[8, 49, 320]", view_121: "f32[392, 320]", getitem_94: "f32[8, 5, 49, 64]", getitem_95: "f32[8, 5, 49, 64]", getitem_97: "f32[8, 5, 224]", getitem_98: "i64[]", getitem_99: "i64[]", view_125: "f32[1568, 320]", mul_73: "f32[8, 196, 320]", view_127: "f32[1568, 320]", addmm_38: "f32[1568, 1280]", view_129: "f32[1568, 1280]", view_131: "f32[8, 320, 14, 14]", mul_78: "f32[8, 196, 320]", view_134: "f32[1568, 320]", permute_95: "f32[8, 5, 196, 64]", view_137: "f32[8, 320, 14, 14]", mul_80: "f32[8, 49, 320]", view_139: "f32[392, 320]", getitem_106: "f32[8, 5, 49, 64]", getitem_107: "f32[8, 5, 49, 64]", getitem_109: "f32[8, 5, 224]", getitem_110: "i64[]", getitem_111: "i64[]", view_143: "f32[1568, 320]", mul_82: "f32[8, 196, 320]", view_145: "f32[1568, 320]", addmm_43: "f32[1568, 1280]", view_147: "f32[1568, 1280]", mul_87: "f32[8, 196, 320]", view_149: "f32[1568, 320]", permute_105: "f32[8, 5, 196, 64]", view_152: "f32[8, 320, 14, 14]", mul_89: "f32[8, 49, 320]", view_154: "f32[392, 320]", getitem_118: "f32[8, 5, 49, 64]", getitem_119: "f32[8, 5, 49, 64]", getitem_121: "f32[8, 5, 224]", getitem_122: "i64[]", getitem_123: "i64[]", view_158: "f32[1568, 320]", mul_91: "f32[8, 196, 320]", view_160: "f32[1568, 320]", addmm_48: "f32[1568, 1280]", view_162: "f32[1568, 1280]", mul_96: "f32[8, 196, 320]", view_164: "f32[1568, 320]", permute_115: "f32[8, 5, 196, 64]", view_167: "f32[8, 320, 14, 14]", mul_98: "f32[8, 49, 320]", view_169: "f32[392, 320]", getitem_130: "f32[8, 5, 49, 64]", getitem_131: "f32[8, 5, 49, 64]", getitem_133: "f32[8, 5, 224]", getitem_134: "i64[]", getitem_135: "i64[]", view_173: "f32[1568, 320]", mul_100: "f32[8, 196, 320]", view_175: "f32[1568, 320]", addmm_53: "f32[1568, 1280]", view_177: "f32[1568, 1280]", mul_105: "f32[8, 196, 320]", view_179: "f32[1568, 320]", permute_125: "f32[8, 5, 196, 64]", view_182: "f32[8, 320, 14, 14]", mul_107: "f32[8, 49, 320]", view_184: "f32[392, 320]", getitem_142: "f32[8, 5, 49, 64]", getitem_143: "f32[8, 5, 49, 64]", getitem_145: "f32[8, 5, 224]", getitem_146: "i64[]", getitem_147: "i64[]", view_188: "f32[1568, 320]", mul_109: "f32[8, 196, 320]", view_190: "f32[1568, 320]", addmm_58: "f32[1568, 1280]", view_192: "f32[1568, 1280]", mul_114: "f32[8, 196, 320]", view_194: "f32[1568, 320]", permute_135: "f32[8, 5, 196, 64]", view_197: "f32[8, 320, 14, 14]", mul_116: "f32[8, 49, 320]", view_199: "f32[392, 320]", getitem_154: "f32[8, 5, 49, 64]", getitem_155: "f32[8, 5, 49, 64]", getitem_157: "f32[8, 5, 224]", getitem_158: "i64[]", getitem_159: "i64[]", view_203: "f32[1568, 320]", mul_118: "f32[8, 196, 320]", view_205: "f32[1568, 320]", addmm_63: "f32[1568, 1280]", view_207: "f32[1568, 1280]", mul_123: "f32[8, 196, 320]", view_209: "f32[1568, 320]", permute_145: "f32[8, 5, 196, 64]", view_212: "f32[8, 320, 14, 14]", mul_125: "f32[8, 49, 320]", view_214: "f32[392, 320]", getitem_166: "f32[8, 5, 49, 64]", getitem_167: "f32[8, 5, 49, 64]", getitem_169: "f32[8, 5, 224]", getitem_170: "i64[]", getitem_171: "i64[]", view_218: "f32[1568, 320]", mul_127: "f32[8, 196, 320]", view_220: "f32[1568, 320]", addmm_68: "f32[1568, 1280]", view_222: "f32[1568, 1280]", mul_132: "f32[8, 196, 320]", view_224: "f32[1568, 320]", permute_155: "f32[8, 5, 196, 64]", view_227: "f32[8, 320, 14, 14]", mul_134: "f32[8, 49, 320]", view_229: "f32[392, 320]", getitem_178: "f32[8, 5, 49, 64]", getitem_179: "f32[8, 5, 49, 64]", getitem_181: "f32[8, 5, 224]", getitem_182: "i64[]", getitem_183: "i64[]", view_233: "f32[1568, 320]", mul_136: "f32[8, 196, 320]", view_235: "f32[1568, 320]", addmm_73: "f32[1568, 1280]", view_237: "f32[1568, 1280]", mul_141: "f32[8, 196, 320]", view_239: "f32[1568, 320]", permute_165: "f32[8, 5, 196, 64]", view_242: "f32[8, 320, 14, 14]", mul_143: "f32[8, 49, 320]", view_244: "f32[392, 320]", getitem_190: "f32[8, 5, 49, 64]", getitem_191: "f32[8, 5, 49, 64]", getitem_193: "f32[8, 5, 224]", getitem_194: "i64[]", getitem_195: "i64[]", view_248: "f32[1568, 320]", mul_145: "f32[8, 196, 320]", view_250: "f32[1568, 320]", addmm_78: "f32[1568, 1280]", view_252: "f32[1568, 1280]", mul_150: "f32[8, 196, 320]", view_254: "f32[1568, 320]", permute_175: "f32[8, 5, 196, 64]", view_257: "f32[8, 320, 14, 14]", mul_152: "f32[8, 49, 320]", view_259: "f32[392, 320]", getitem_202: "f32[8, 5, 49, 64]", getitem_203: "f32[8, 5, 49, 64]", getitem_205: "f32[8, 5, 224]", getitem_206: "i64[]", getitem_207: "i64[]", view_263: "f32[1568, 320]", mul_154: "f32[8, 196, 320]", view_265: "f32[1568, 320]", addmm_83: "f32[1568, 1280]", view_267: "f32[1568, 1280]", mul_159: "f32[8, 196, 320]", view_269: "f32[1568, 320]", permute_185: "f32[8, 5, 196, 64]", view_272: "f32[8, 320, 14, 14]", mul_161: "f32[8, 49, 320]", view_274: "f32[392, 320]", getitem_214: "f32[8, 5, 49, 64]", getitem_215: "f32[8, 5, 49, 64]", getitem_217: "f32[8, 5, 224]", getitem_218: "i64[]", getitem_219: "i64[]", view_278: "f32[1568, 320]", mul_163: "f32[8, 196, 320]", view_280: "f32[1568, 320]", addmm_88: "f32[1568, 1280]", view_282: "f32[1568, 1280]", mul_168: "f32[8, 196, 320]", view_284: "f32[1568, 320]", permute_195: "f32[8, 5, 196, 64]", view_287: "f32[8, 320, 14, 14]", mul_170: "f32[8, 49, 320]", view_289: "f32[392, 320]", getitem_226: "f32[8, 5, 49, 64]", getitem_227: "f32[8, 5, 49, 64]", getitem_229: "f32[8, 5, 224]", getitem_230: "i64[]", getitem_231: "i64[]", view_293: "f32[1568, 320]", mul_172: "f32[8, 196, 320]", view_295: "f32[1568, 320]", addmm_93: "f32[1568, 1280]", view_297: "f32[1568, 1280]", mul_177: "f32[8, 196, 320]", view_299: "f32[1568, 320]", permute_205: "f32[8, 5, 196, 64]", view_302: "f32[8, 320, 14, 14]", mul_179: "f32[8, 49, 320]", view_304: "f32[392, 320]", getitem_238: "f32[8, 5, 49, 64]", getitem_239: "f32[8, 5, 49, 64]", getitem_241: "f32[8, 5, 224]", getitem_242: "i64[]", getitem_243: "i64[]", view_308: "f32[1568, 320]", mul_181: "f32[8, 196, 320]", view_310: "f32[1568, 320]", addmm_98: "f32[1568, 1280]", view_312: "f32[1568, 1280]", mul_186: "f32[8, 196, 320]", view_314: "f32[1568, 320]", permute_215: "f32[8, 5, 196, 64]", view_317: "f32[8, 320, 14, 14]", mul_188: "f32[8, 49, 320]", view_319: "f32[392, 320]", getitem_250: "f32[8, 5, 49, 64]", getitem_251: "f32[8, 5, 49, 64]", getitem_253: "f32[8, 5, 224]", getitem_254: "i64[]", getitem_255: "i64[]", view_323: "f32[1568, 320]", mul_190: "f32[8, 196, 320]", view_325: "f32[1568, 320]", addmm_103: "f32[1568, 1280]", view_327: "f32[1568, 1280]", mul_195: "f32[8, 196, 320]", view_329: "f32[1568, 320]", permute_225: "f32[8, 5, 196, 64]", view_332: "f32[8, 320, 14, 14]", mul_197: "f32[8, 49, 320]", view_334: "f32[392, 320]", getitem_262: "f32[8, 5, 49, 64]", getitem_263: "f32[8, 5, 49, 64]", getitem_265: "f32[8, 5, 224]", getitem_266: "i64[]", getitem_267: "i64[]", view_338: "f32[1568, 320]", mul_199: "f32[8, 196, 320]", view_340: "f32[1568, 320]", addmm_108: "f32[1568, 1280]", view_342: "f32[1568, 1280]", mul_204: "f32[8, 196, 320]", view_344: "f32[1568, 320]", permute_235: "f32[8, 5, 196, 64]", view_347: "f32[8, 320, 14, 14]", mul_206: "f32[8, 49, 320]", view_349: "f32[392, 320]", getitem_274: "f32[8, 5, 49, 64]", getitem_275: "f32[8, 5, 49, 64]", getitem_277: "f32[8, 5, 224]", getitem_278: "i64[]", getitem_279: "i64[]", view_353: "f32[1568, 320]", mul_208: "f32[8, 196, 320]", view_355: "f32[1568, 320]", addmm_113: "f32[1568, 1280]", view_357: "f32[1568, 1280]", mul_213: "f32[8, 196, 320]", view_359: "f32[1568, 320]", permute_245: "f32[8, 5, 196, 64]", view_362: "f32[8, 320, 14, 14]", mul_215: "f32[8, 49, 320]", view_364: "f32[392, 320]", getitem_286: "f32[8, 5, 49, 64]", getitem_287: "f32[8, 5, 49, 64]", getitem_289: "f32[8, 5, 224]", getitem_290: "i64[]", getitem_291: "i64[]", view_368: "f32[1568, 320]", mul_217: "f32[8, 196, 320]", view_370: "f32[1568, 320]", addmm_118: "f32[1568, 1280]", view_372: "f32[1568, 1280]", mul_222: "f32[8, 196, 320]", view_374: "f32[1568, 320]", permute_255: "f32[8, 5, 196, 64]", view_377: "f32[8, 320, 14, 14]", mul_224: "f32[8, 49, 320]", view_379: "f32[392, 320]", getitem_298: "f32[8, 5, 49, 64]", getitem_299: "f32[8, 5, 49, 64]", getitem_301: "f32[8, 5, 224]", getitem_302: "i64[]", getitem_303: "i64[]", view_383: "f32[1568, 320]", mul_226: "f32[8, 196, 320]", view_385: "f32[1568, 320]", addmm_123: "f32[1568, 1280]", view_387: "f32[1568, 1280]", permute_264: "f32[8, 320, 14, 14]", mul_231: "f32[8, 49, 512]", mul_233: "f32[8, 49, 512]", view_391: "f32[392, 512]", permute_267: "f32[8, 8, 49, 64]", getitem_310: "f32[8, 8, 49, 64]", getitem_311: "f32[8, 8, 49, 64]", getitem_313: "f32[8, 8, 64]", getitem_314: "i64[]", getitem_315: "i64[]", view_398: "f32[392, 512]", mul_235: "f32[8, 49, 512]", view_400: "f32[392, 512]", addmm_128: "f32[392, 2048]", view_402: "f32[392, 2048]", view_404: "f32[8, 512, 7, 7]", mul_240: "f32[8, 49, 512]", view_407: "f32[392, 512]", permute_278: "f32[8, 8, 49, 64]", getitem_320: "f32[8, 8, 49, 64]", getitem_321: "f32[8, 8, 49, 64]", getitem_323: "f32[8, 8, 64]", getitem_324: "i64[]", getitem_325: "i64[]", view_414: "f32[392, 512]", mul_242: "f32[8, 49, 512]", view_416: "f32[392, 512]", addmm_133: "f32[392, 2048]", view_418: "f32[392, 2048]", mul_247: "f32[8, 49, 512]", view_420: "f32[392, 512]", permute_286: "f32[8, 8, 49, 64]", getitem_330: "f32[8, 8, 49, 64]", getitem_331: "f32[8, 8, 49, 64]", getitem_333: "f32[8, 8, 64]", getitem_334: "i64[]", getitem_335: "i64[]", view_427: "f32[392, 512]", mul_249: "f32[8, 49, 512]", view_429: "f32[392, 512]", addmm_138: "f32[392, 2048]", view_431: "f32[392, 2048]", mul_254: "f32[8, 49, 512]", clone_166: "f32[8, 512]", permute_294: "f32[1000, 512]", div_1: "f32[8, 49, 1]", permute_298: "f32[512, 2048]", permute_302: "f32[2048, 512]", div_2: "f32[8, 49, 1]", permute_306: "f32[512, 512]", alias_28: "f32[8, 8, 49, 64]", permute_312: "f32[1024, 512]", permute_317: "f32[512, 512]", div_3: "f32[8, 49, 1]", permute_321: "f32[512, 2048]", permute_325: "f32[2048, 512]", div_4: "f32[8, 49, 1]", permute_329: "f32[512, 512]", alias_29: "f32[8, 8, 49, 64]", permute_335: "f32[1024, 512]", permute_340: "f32[512, 512]", div_5: "f32[8, 49, 1]", permute_346: "f32[512, 2048]", permute_350: "f32[2048, 512]", div_6: "f32[8, 49, 1]", permute_354: "f32[512, 512]", alias_30: "f32[8, 8, 49, 64]", permute_360: "f32[1024, 512]", permute_365: "f32[512, 512]", div_7: "f32[8, 49, 1]", div_8: "f32[8, 49, 1]", permute_371: "f32[320, 1280]", permute_375: "f32[1280, 320]", div_9: "f32[8, 196, 1]", permute_379: "f32[320, 320]", alias_31: "f32[8, 5, 196, 64]", permute_385: "f32[640, 320]", div_10: "f32[8, 49, 1]", permute_392: "f32[320, 320]", div_11: "f32[8, 196, 1]", permute_396: "f32[320, 1280]", permute_400: "f32[1280, 320]", div_12: "f32[8, 196, 1]", permute_404: "f32[320, 320]", alias_32: "f32[8, 5, 196, 64]", permute_410: "f32[640, 320]", div_13: "f32[8, 49, 1]", permute_417: "f32[320, 320]", div_14: "f32[8, 196, 1]", permute_421: "f32[320, 1280]", permute_425: "f32[1280, 320]", div_15: "f32[8, 196, 1]", permute_429: "f32[320, 320]", alias_33: "f32[8, 5, 196, 64]", permute_435: "f32[640, 320]", div_16: "f32[8, 49, 1]", permute_442: "f32[320, 320]", div_17: "f32[8, 196, 1]", permute_446: "f32[320, 1280]", permute_450: "f32[1280, 320]", div_18: "f32[8, 196, 1]", permute_454: "f32[320, 320]", alias_34: "f32[8, 5, 196, 64]", permute_460: "f32[640, 320]", div_19: "f32[8, 49, 1]", permute_467: "f32[320, 320]", div_20: "f32[8, 196, 1]", permute_471: "f32[320, 1280]", permute_475: "f32[1280, 320]", div_21: "f32[8, 196, 1]", permute_479: "f32[320, 320]", alias_35: "f32[8, 5, 196, 64]", permute_485: "f32[640, 320]", div_22: "f32[8, 49, 1]", permute_492: "f32[320, 320]", div_23: "f32[8, 196, 1]", permute_496: "f32[320, 1280]", permute_500: "f32[1280, 320]", div_24: "f32[8, 196, 1]", permute_504: "f32[320, 320]", alias_36: "f32[8, 5, 196, 64]", permute_510: "f32[640, 320]", div_25: "f32[8, 49, 1]", permute_517: "f32[320, 320]", div_26: "f32[8, 196, 1]", permute_521: "f32[320, 1280]", permute_525: "f32[1280, 320]", div_27: "f32[8, 196, 1]", permute_529: "f32[320, 320]", alias_37: "f32[8, 5, 196, 64]", permute_535: "f32[640, 320]", div_28: "f32[8, 49, 1]", permute_542: "f32[320, 320]", div_29: "f32[8, 196, 1]", permute_546: "f32[320, 1280]", permute_550: "f32[1280, 320]", div_30: "f32[8, 196, 1]", permute_554: "f32[320, 320]", alias_38: "f32[8, 5, 196, 64]", permute_560: "f32[640, 320]", div_31: "f32[8, 49, 1]", permute_567: "f32[320, 320]", div_32: "f32[8, 196, 1]", permute_571: "f32[320, 1280]", permute_575: "f32[1280, 320]", div_33: "f32[8, 196, 1]", permute_579: "f32[320, 320]", alias_39: "f32[8, 5, 196, 64]", permute_585: "f32[640, 320]", div_34: "f32[8, 49, 1]", permute_592: "f32[320, 320]", div_35: "f32[8, 196, 1]", permute_596: "f32[320, 1280]", permute_600: "f32[1280, 320]", div_36: "f32[8, 196, 1]", permute_604: "f32[320, 320]", alias_40: "f32[8, 5, 196, 64]", permute_610: "f32[640, 320]", div_37: "f32[8, 49, 1]", permute_617: "f32[320, 320]", div_38: "f32[8, 196, 1]", permute_621: "f32[320, 1280]", permute_625: "f32[1280, 320]", div_39: "f32[8, 196, 1]", permute_629: "f32[320, 320]", alias_41: "f32[8, 5, 196, 64]", permute_635: "f32[640, 320]", div_40: "f32[8, 49, 1]", permute_642: "f32[320, 320]", div_41: "f32[8, 196, 1]", permute_646: "f32[320, 1280]", permute_650: "f32[1280, 320]", div_42: "f32[8, 196, 1]", permute_654: "f32[320, 320]", alias_42: "f32[8, 5, 196, 64]", permute_660: "f32[640, 320]", div_43: "f32[8, 49, 1]", permute_667: "f32[320, 320]", div_44: "f32[8, 196, 1]", permute_671: "f32[320, 1280]", permute_675: "f32[1280, 320]", div_45: "f32[8, 196, 1]", permute_679: "f32[320, 320]", alias_43: "f32[8, 5, 196, 64]", permute_685: "f32[640, 320]", div_46: "f32[8, 49, 1]", permute_692: "f32[320, 320]", div_47: "f32[8, 196, 1]", permute_696: "f32[320, 1280]", permute_700: "f32[1280, 320]", div_48: "f32[8, 196, 1]", permute_704: "f32[320, 320]", alias_44: "f32[8, 5, 196, 64]", permute_710: "f32[640, 320]", div_49: "f32[8, 49, 1]", permute_717: "f32[320, 320]", div_50: "f32[8, 196, 1]", permute_721: "f32[320, 1280]", permute_725: "f32[1280, 320]", div_51: "f32[8, 196, 1]", permute_729: "f32[320, 320]", alias_45: "f32[8, 5, 196, 64]", permute_735: "f32[640, 320]", div_52: "f32[8, 49, 1]", permute_742: "f32[320, 320]", div_53: "f32[8, 196, 1]", permute_746: "f32[320, 1280]", permute_750: "f32[1280, 320]", div_54: "f32[8, 196, 1]", permute_754: "f32[320, 320]", alias_46: "f32[8, 5, 196, 64]", permute_760: "f32[640, 320]", div_55: "f32[8, 49, 1]", permute_767: "f32[320, 320]", div_56: "f32[8, 196, 1]", permute_771: "f32[320, 1280]", permute_775: "f32[1280, 320]", div_57: "f32[8, 196, 1]", permute_779: "f32[320, 320]", alias_47: "f32[8, 5, 196, 64]", permute_785: "f32[640, 320]", div_58: "f32[8, 49, 1]", permute_792: "f32[320, 320]", div_59: "f32[8, 196, 1]", permute_798: "f32[320, 1280]", permute_802: "f32[1280, 320]", div_60: "f32[8, 196, 1]", permute_806: "f32[320, 320]", alias_48: "f32[8, 5, 196, 64]", permute_812: "f32[640, 320]", div_61: "f32[8, 49, 1]", permute_819: "f32[320, 320]", div_62: "f32[8, 196, 1]", div_63: "f32[8, 196, 1]", permute_825: "f32[128, 1024]", permute_829: "f32[1024, 128]", div_64: "f32[8, 784, 1]", permute_833: "f32[128, 128]", alias_49: "f32[8, 2, 784, 64]", permute_839: "f32[256, 128]", div_65: "f32[8, 49, 1]", permute_846: "f32[128, 128]", div_66: "f32[8, 784, 1]", permute_850: "f32[128, 1024]", permute_854: "f32[1024, 128]", div_67: "f32[8, 784, 1]", permute_858: "f32[128, 128]", alias_50: "f32[8, 2, 784, 64]", permute_864: "f32[256, 128]", div_68: "f32[8, 49, 1]", permute_871: "f32[128, 128]", div_69: "f32[8, 784, 1]", permute_875: "f32[128, 1024]", permute_879: "f32[1024, 128]", div_70: "f32[8, 784, 1]", permute_883: "f32[128, 128]", alias_51: "f32[8, 2, 784, 64]", permute_889: "f32[256, 128]", div_71: "f32[8, 49, 1]", permute_896: "f32[128, 128]", div_72: "f32[8, 784, 1]", permute_902: "f32[128, 1024]", permute_906: "f32[1024, 128]", div_73: "f32[8, 784, 1]", permute_910: "f32[128, 128]", alias_52: "f32[8, 2, 784, 64]", permute_916: "f32[256, 128]", div_74: "f32[8, 49, 1]", permute_923: "f32[128, 128]", div_75: "f32[8, 784, 1]", div_76: "f32[8, 784, 1]", permute_929: "f32[64, 512]", permute_933: "f32[512, 64]", div_77: "f32[8, 3136, 1]", permute_937: "f32[64, 64]", alias_53: "f32[8, 1, 3136, 64]", permute_943: "f32[128, 64]", div_78: "f32[8, 49, 1]", permute_950: "f32[64, 64]", div_79: "f32[8, 3136, 1]", permute_954: "f32[64, 512]", permute_958: "f32[512, 64]", div_80: "f32[8, 3136, 1]", permute_962: "f32[64, 64]", alias_54: "f32[8, 1, 3136, 64]", permute_968: "f32[128, 64]", div_81: "f32[8, 49, 1]", permute_975: "f32[64, 64]", div_82: "f32[8, 3136, 1]", permute_981: "f32[64, 512]", permute_985: "f32[512, 64]", div_83: "f32[8, 3136, 1]", permute_989: "f32[64, 64]", alias_55: "f32[8, 1, 3136, 64]", permute_995: "f32[128, 64]", div_84: "f32[8, 49, 1]", permute_1002: "f32[64, 64]", div_85: "f32[8, 3136, 1]", div_86: "f32[8, 3136, 1]", tangents_1: "f32[8, 1000]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_13: "f32[8, 3136, 512]" = torch.ops.aten.view.default(addmm_3, [8, 3136, 512]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_9: "f32[8, 3136, 512]" = torch.ops.aten.mul.Tensor(view_13, 0.7071067811865476)
    erf: "f32[8, 3136, 512]" = torch.ops.aten.erf.default(mul_9);  mul_9 = None
    add_9: "f32[8, 3136, 512]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_31: "f32[8, 3136, 512]" = torch.ops.aten.view.default(addmm_8, [8, 3136, 512]);  addmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_18: "f32[8, 3136, 512]" = torch.ops.aten.mul.Tensor(view_31, 0.7071067811865476)
    erf_1: "f32[8, 3136, 512]" = torch.ops.aten.erf.default(mul_18);  mul_18 = None
    add_19: "f32[8, 3136, 512]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_46: "f32[8, 3136, 512]" = torch.ops.aten.view.default(addmm_13, [8, 3136, 512]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_27: "f32[8, 3136, 512]" = torch.ops.aten.mul.Tensor(view_46, 0.7071067811865476)
    erf_2: "f32[8, 3136, 512]" = torch.ops.aten.erf.default(mul_27);  mul_27 = None
    add_28: "f32[8, 3136, 512]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_63: "f32[8, 784, 1024]" = torch.ops.aten.view.default(addmm_18, [8, 784, 1024]);  addmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_38: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(view_63, 0.7071067811865476)
    erf_3: "f32[8, 784, 1024]" = torch.ops.aten.erf.default(mul_38);  mul_38 = None
    add_39: "f32[8, 784, 1024]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_81: "f32[8, 784, 1024]" = torch.ops.aten.view.default(addmm_23, [8, 784, 1024]);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_47: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(view_81, 0.7071067811865476)
    erf_4: "f32[8, 784, 1024]" = torch.ops.aten.erf.default(mul_47);  mul_47 = None
    add_49: "f32[8, 784, 1024]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_96: "f32[8, 784, 1024]" = torch.ops.aten.view.default(addmm_28, [8, 784, 1024]);  addmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_56: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(view_96, 0.7071067811865476)
    erf_5: "f32[8, 784, 1024]" = torch.ops.aten.erf.default(mul_56);  mul_56 = None
    add_58: "f32[8, 784, 1024]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_111: "f32[8, 784, 1024]" = torch.ops.aten.view.default(addmm_33, [8, 784, 1024]);  addmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_65: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(view_111, 0.7071067811865476)
    erf_6: "f32[8, 784, 1024]" = torch.ops.aten.erf.default(mul_65);  mul_65 = None
    add_67: "f32[8, 784, 1024]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_128: "f32[8, 196, 1280]" = torch.ops.aten.view.default(addmm_38, [8, 196, 1280]);  addmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_76: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_128, 0.7071067811865476)
    erf_7: "f32[8, 196, 1280]" = torch.ops.aten.erf.default(mul_76);  mul_76 = None
    add_78: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_146: "f32[8, 196, 1280]" = torch.ops.aten.view.default(addmm_43, [8, 196, 1280]);  addmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_85: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_146, 0.7071067811865476)
    erf_8: "f32[8, 196, 1280]" = torch.ops.aten.erf.default(mul_85);  mul_85 = None
    add_88: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_161: "f32[8, 196, 1280]" = torch.ops.aten.view.default(addmm_48, [8, 196, 1280]);  addmm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_94: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_161, 0.7071067811865476)
    erf_9: "f32[8, 196, 1280]" = torch.ops.aten.erf.default(mul_94);  mul_94 = None
    add_97: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_176: "f32[8, 196, 1280]" = torch.ops.aten.view.default(addmm_53, [8, 196, 1280]);  addmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_103: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_176, 0.7071067811865476)
    erf_10: "f32[8, 196, 1280]" = torch.ops.aten.erf.default(mul_103);  mul_103 = None
    add_106: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_191: "f32[8, 196, 1280]" = torch.ops.aten.view.default(addmm_58, [8, 196, 1280]);  addmm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_112: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_191, 0.7071067811865476)
    erf_11: "f32[8, 196, 1280]" = torch.ops.aten.erf.default(mul_112);  mul_112 = None
    add_115: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_206: "f32[8, 196, 1280]" = torch.ops.aten.view.default(addmm_63, [8, 196, 1280]);  addmm_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_121: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_206, 0.7071067811865476)
    erf_12: "f32[8, 196, 1280]" = torch.ops.aten.erf.default(mul_121);  mul_121 = None
    add_124: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_221: "f32[8, 196, 1280]" = torch.ops.aten.view.default(addmm_68, [8, 196, 1280]);  addmm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_130: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_221, 0.7071067811865476)
    erf_13: "f32[8, 196, 1280]" = torch.ops.aten.erf.default(mul_130);  mul_130 = None
    add_133: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_236: "f32[8, 196, 1280]" = torch.ops.aten.view.default(addmm_73, [8, 196, 1280]);  addmm_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_139: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_236, 0.7071067811865476)
    erf_14: "f32[8, 196, 1280]" = torch.ops.aten.erf.default(mul_139);  mul_139 = None
    add_142: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_251: "f32[8, 196, 1280]" = torch.ops.aten.view.default(addmm_78, [8, 196, 1280]);  addmm_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_148: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_251, 0.7071067811865476)
    erf_15: "f32[8, 196, 1280]" = torch.ops.aten.erf.default(mul_148);  mul_148 = None
    add_151: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_266: "f32[8, 196, 1280]" = torch.ops.aten.view.default(addmm_83, [8, 196, 1280]);  addmm_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_157: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_266, 0.7071067811865476)
    erf_16: "f32[8, 196, 1280]" = torch.ops.aten.erf.default(mul_157);  mul_157 = None
    add_160: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_281: "f32[8, 196, 1280]" = torch.ops.aten.view.default(addmm_88, [8, 196, 1280]);  addmm_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_166: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_281, 0.7071067811865476)
    erf_17: "f32[8, 196, 1280]" = torch.ops.aten.erf.default(mul_166);  mul_166 = None
    add_169: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_296: "f32[8, 196, 1280]" = torch.ops.aten.view.default(addmm_93, [8, 196, 1280]);  addmm_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_175: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_296, 0.7071067811865476)
    erf_18: "f32[8, 196, 1280]" = torch.ops.aten.erf.default(mul_175);  mul_175 = None
    add_178: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_311: "f32[8, 196, 1280]" = torch.ops.aten.view.default(addmm_98, [8, 196, 1280]);  addmm_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_184: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_311, 0.7071067811865476)
    erf_19: "f32[8, 196, 1280]" = torch.ops.aten.erf.default(mul_184);  mul_184 = None
    add_187: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_326: "f32[8, 196, 1280]" = torch.ops.aten.view.default(addmm_103, [8, 196, 1280]);  addmm_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_193: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_326, 0.7071067811865476)
    erf_20: "f32[8, 196, 1280]" = torch.ops.aten.erf.default(mul_193);  mul_193 = None
    add_196: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_341: "f32[8, 196, 1280]" = torch.ops.aten.view.default(addmm_108, [8, 196, 1280]);  addmm_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_202: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_341, 0.7071067811865476)
    erf_21: "f32[8, 196, 1280]" = torch.ops.aten.erf.default(mul_202);  mul_202 = None
    add_205: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_356: "f32[8, 196, 1280]" = torch.ops.aten.view.default(addmm_113, [8, 196, 1280]);  addmm_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_211: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_356, 0.7071067811865476)
    erf_22: "f32[8, 196, 1280]" = torch.ops.aten.erf.default(mul_211);  mul_211 = None
    add_214: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_371: "f32[8, 196, 1280]" = torch.ops.aten.view.default(addmm_118, [8, 196, 1280]);  addmm_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_220: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_371, 0.7071067811865476)
    erf_23: "f32[8, 196, 1280]" = torch.ops.aten.erf.default(mul_220);  mul_220 = None
    add_223: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_386: "f32[8, 196, 1280]" = torch.ops.aten.view.default(addmm_123, [8, 196, 1280]);  addmm_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_229: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_386, 0.7071067811865476)
    erf_24: "f32[8, 196, 1280]" = torch.ops.aten.erf.default(mul_229);  mul_229 = None
    add_232: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(erf_24, 1);  erf_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_401: "f32[8, 49, 2048]" = torch.ops.aten.view.default(addmm_128, [8, 49, 2048]);  addmm_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_238: "f32[8, 49, 2048]" = torch.ops.aten.mul.Tensor(view_401, 0.7071067811865476)
    erf_25: "f32[8, 49, 2048]" = torch.ops.aten.erf.default(mul_238);  mul_238 = None
    add_241: "f32[8, 49, 2048]" = torch.ops.aten.add.Tensor(erf_25, 1);  erf_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_417: "f32[8, 49, 2048]" = torch.ops.aten.view.default(addmm_133, [8, 49, 2048]);  addmm_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_245: "f32[8, 49, 2048]" = torch.ops.aten.mul.Tensor(view_417, 0.7071067811865476)
    erf_26: "f32[8, 49, 2048]" = torch.ops.aten.erf.default(mul_245);  mul_245 = None
    add_249: "f32[8, 49, 2048]" = torch.ops.aten.add.Tensor(erf_26, 1);  erf_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_430: "f32[8, 49, 2048]" = torch.ops.aten.view.default(addmm_138, [8, 49, 2048]);  addmm_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_252: "f32[8, 49, 2048]" = torch.ops.aten.mul.Tensor(view_430, 0.7071067811865476)
    erf_27: "f32[8, 49, 2048]" = torch.ops.aten.erf.default(mul_252);  mul_252 = None
    add_256: "f32[8, 49, 2048]" = torch.ops.aten.add.Tensor(erf_27, 1);  erf_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:423, code: return x if pre_logits else self.head(x)
    mm: "f32[8, 512]" = torch.ops.aten.mm.default(tangents_1, permute_294);  permute_294 = None
    permute_295: "f32[1000, 8]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
    mm_1: "f32[1000, 512]" = torch.ops.aten.mm.default(permute_295, clone_166);  permute_295 = clone_166 = None
    permute_296: "f32[512, 1000]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_1: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view_433: "f32[1000]" = torch.ops.aten.view.default(sum_1, [1000]);  sum_1 = None
    permute_297: "f32[1000, 512]" = torch.ops.aten.permute.default(permute_296, [1, 0]);  permute_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:421, code: x = x.mean(dim=1)
    unsqueeze: "f32[8, 1, 512]" = torch.ops.aten.unsqueeze.default(mm, 1);  mm = None
    expand: "f32[8, 49, 512]" = torch.ops.aten.expand.default(unsqueeze, [8, 49, 512]);  unsqueeze = None
    div: "f32[8, 49, 512]" = torch.ops.aten.div.Scalar(expand, 49);  expand = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:416, code: x = self.norm(x)
    mul_257: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(div, primals_517);  primals_517 = None
    mul_258: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(mul_257, 512)
    sum_2: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_257, [2], True)
    mul_259: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(mul_257, mul_254);  mul_257 = None
    sum_3: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_259, [2], True);  mul_259 = None
    mul_260: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(mul_254, sum_3);  sum_3 = None
    sub_87: "f32[8, 49, 512]" = torch.ops.aten.sub.Tensor(mul_258, sum_2);  mul_258 = sum_2 = None
    sub_88: "f32[8, 49, 512]" = torch.ops.aten.sub.Tensor(sub_87, mul_260);  sub_87 = mul_260 = None
    mul_261: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(div_1, sub_88);  div_1 = sub_88 = None
    mul_262: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(div, mul_254);  mul_254 = None
    sum_4: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_262, [0, 1]);  mul_262 = None
    sum_5: "f32[512]" = torch.ops.aten.sum.dim_IntList(div, [0, 1]);  div = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_434: "f32[392, 512]" = torch.ops.aten.view.default(mul_261, [392, 512])
    mm_2: "f32[392, 2048]" = torch.ops.aten.mm.default(view_434, permute_298);  permute_298 = None
    permute_299: "f32[512, 392]" = torch.ops.aten.permute.default(view_434, [1, 0])
    mm_3: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_299, view_431);  permute_299 = view_431 = None
    permute_300: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_3, [1, 0]);  mm_3 = None
    sum_6: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_434, [0], True);  view_434 = None
    view_435: "f32[512]" = torch.ops.aten.view.default(sum_6, [512]);  sum_6 = None
    permute_301: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_300, [1, 0]);  permute_300 = None
    view_436: "f32[8, 49, 2048]" = torch.ops.aten.view.default(mm_2, [8, 49, 2048]);  mm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_264: "f32[8, 49, 2048]" = torch.ops.aten.mul.Tensor(add_256, 0.5);  add_256 = None
    mul_265: "f32[8, 49, 2048]" = torch.ops.aten.mul.Tensor(view_430, view_430)
    mul_266: "f32[8, 49, 2048]" = torch.ops.aten.mul.Tensor(mul_265, -0.5);  mul_265 = None
    exp: "f32[8, 49, 2048]" = torch.ops.aten.exp.default(mul_266);  mul_266 = None
    mul_267: "f32[8, 49, 2048]" = torch.ops.aten.mul.Tensor(exp, 0.3989422804014327);  exp = None
    mul_268: "f32[8, 49, 2048]" = torch.ops.aten.mul.Tensor(view_430, mul_267);  view_430 = mul_267 = None
    add_261: "f32[8, 49, 2048]" = torch.ops.aten.add.Tensor(mul_264, mul_268);  mul_264 = mul_268 = None
    mul_269: "f32[8, 49, 2048]" = torch.ops.aten.mul.Tensor(view_436, add_261);  view_436 = add_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_437: "f32[392, 2048]" = torch.ops.aten.view.default(mul_269, [392, 2048]);  mul_269 = None
    mm_4: "f32[392, 512]" = torch.ops.aten.mm.default(view_437, permute_302);  permute_302 = None
    permute_303: "f32[2048, 392]" = torch.ops.aten.permute.default(view_437, [1, 0])
    mm_5: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_303, view_429);  permute_303 = view_429 = None
    permute_304: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_5, [1, 0]);  mm_5 = None
    sum_7: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_437, [0], True);  view_437 = None
    view_438: "f32[2048]" = torch.ops.aten.view.default(sum_7, [2048]);  sum_7 = None
    permute_305: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_304, [1, 0]);  permute_304 = None
    view_439: "f32[8, 49, 512]" = torch.ops.aten.view.default(mm_4, [8, 49, 512]);  mm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    mul_271: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(view_439, primals_511);  primals_511 = None
    mul_272: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(mul_271, 512)
    sum_8: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_271, [2], True)
    mul_273: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(mul_271, mul_249);  mul_271 = None
    sum_9: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_273, [2], True);  mul_273 = None
    mul_274: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(mul_249, sum_9);  sum_9 = None
    sub_90: "f32[8, 49, 512]" = torch.ops.aten.sub.Tensor(mul_272, sum_8);  mul_272 = sum_8 = None
    sub_91: "f32[8, 49, 512]" = torch.ops.aten.sub.Tensor(sub_90, mul_274);  sub_90 = mul_274 = None
    mul_275: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(div_2, sub_91);  div_2 = sub_91 = None
    mul_276: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(view_439, mul_249);  mul_249 = None
    sum_10: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_276, [0, 1]);  mul_276 = None
    sum_11: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_439, [0, 1]);  view_439 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_262: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(mul_261, mul_275);  mul_261 = mul_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    view_440: "f32[392, 512]" = torch.ops.aten.view.default(add_262, [392, 512])
    mm_6: "f32[392, 512]" = torch.ops.aten.mm.default(view_440, permute_306);  permute_306 = None
    permute_307: "f32[512, 392]" = torch.ops.aten.permute.default(view_440, [1, 0])
    mm_7: "f32[512, 512]" = torch.ops.aten.mm.default(permute_307, view_427);  permute_307 = view_427 = None
    permute_308: "f32[512, 512]" = torch.ops.aten.permute.default(mm_7, [1, 0]);  mm_7 = None
    sum_12: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_440, [0], True);  view_440 = None
    view_441: "f32[512]" = torch.ops.aten.view.default(sum_12, [512]);  sum_12 = None
    permute_309: "f32[512, 512]" = torch.ops.aten.permute.default(permute_308, [1, 0]);  permute_308 = None
    view_442: "f32[8, 49, 512]" = torch.ops.aten.view.default(mm_6, [8, 49, 512]);  mm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_443: "f32[8, 49, 8, 64]" = torch.ops.aten.view.default(view_442, [8, 49, 8, 64]);  view_442 = None
    permute_310: "f32[8, 8, 49, 64]" = torch.ops.aten.permute.default(view_443, [0, 2, 1, 3]);  view_443 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_backward = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_310, permute_286, getitem_330, getitem_331, None, alias_28, getitem_333, getitem_334, getitem_335, 0.0, [True, True, True, False]);  permute_310 = permute_286 = getitem_330 = getitem_331 = alias_28 = getitem_333 = getitem_334 = getitem_335 = None
    getitem_340: "f32[8, 8, 49, 64]" = _scaled_dot_product_efficient_attention_backward[0]
    getitem_341: "f32[8, 8, 49, 64]" = _scaled_dot_product_efficient_attention_backward[1]
    getitem_342: "f32[8, 8, 49, 64]" = _scaled_dot_product_efficient_attention_backward[2];  _scaled_dot_product_efficient_attention_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    cat: "f32[16, 8, 49, 64]" = torch.ops.aten.cat.default([getitem_341, getitem_342]);  getitem_341 = getitem_342 = None
    view_444: "f32[2, 8, 8, 49, 64]" = torch.ops.aten.view.default(cat, [2, 8, 8, 49, 64]);  cat = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_311: "f32[8, 49, 2, 8, 64]" = torch.ops.aten.permute.default(view_444, [1, 3, 0, 2, 4]);  view_444 = None
    clone_169: "f32[8, 49, 2, 8, 64]" = torch.ops.aten.clone.default(permute_311, memory_format = torch.contiguous_format);  permute_311 = None
    view_445: "f32[8, 49, 1024]" = torch.ops.aten.view.default(clone_169, [8, 49, 1024]);  clone_169 = None
    view_446: "f32[392, 1024]" = torch.ops.aten.view.default(view_445, [392, 1024]);  view_445 = None
    mm_8: "f32[392, 512]" = torch.ops.aten.mm.default(view_446, permute_312);  permute_312 = None
    permute_313: "f32[1024, 392]" = torch.ops.aten.permute.default(view_446, [1, 0])
    mm_9: "f32[1024, 512]" = torch.ops.aten.mm.default(permute_313, view_420);  permute_313 = None
    permute_314: "f32[512, 1024]" = torch.ops.aten.permute.default(mm_9, [1, 0]);  mm_9 = None
    sum_13: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_446, [0], True);  view_446 = None
    view_447: "f32[1024]" = torch.ops.aten.view.default(sum_13, [1024]);  sum_13 = None
    permute_315: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_314, [1, 0]);  permute_314 = None
    view_448: "f32[8, 49, 512]" = torch.ops.aten.view.default(mm_8, [8, 49, 512]);  mm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_316: "f32[8, 49, 8, 64]" = torch.ops.aten.permute.default(getitem_340, [0, 2, 1, 3]);  getitem_340 = None
    view_449: "f32[8, 49, 512]" = torch.ops.aten.view.default(permute_316, [8, 49, 512]);  permute_316 = None
    view_450: "f32[392, 512]" = torch.ops.aten.view.default(view_449, [392, 512]);  view_449 = None
    mm_10: "f32[392, 512]" = torch.ops.aten.mm.default(view_450, permute_317);  permute_317 = None
    permute_318: "f32[512, 392]" = torch.ops.aten.permute.default(view_450, [1, 0])
    mm_11: "f32[512, 512]" = torch.ops.aten.mm.default(permute_318, view_420);  permute_318 = view_420 = None
    permute_319: "f32[512, 512]" = torch.ops.aten.permute.default(mm_11, [1, 0]);  mm_11 = None
    sum_14: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_450, [0], True);  view_450 = None
    view_451: "f32[512]" = torch.ops.aten.view.default(sum_14, [512]);  sum_14 = None
    permute_320: "f32[512, 512]" = torch.ops.aten.permute.default(permute_319, [1, 0]);  permute_319 = None
    view_452: "f32[8, 49, 512]" = torch.ops.aten.view.default(mm_10, [8, 49, 512]);  mm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    add_263: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(view_448, view_452);  view_448 = view_452 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    mul_278: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(add_263, primals_503);  primals_503 = None
    mul_279: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(mul_278, 512)
    sum_15: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_278, [2], True)
    mul_280: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(mul_278, mul_247);  mul_278 = None
    sum_16: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_280, [2], True);  mul_280 = None
    mul_281: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(mul_247, sum_16);  sum_16 = None
    sub_93: "f32[8, 49, 512]" = torch.ops.aten.sub.Tensor(mul_279, sum_15);  mul_279 = sum_15 = None
    sub_94: "f32[8, 49, 512]" = torch.ops.aten.sub.Tensor(sub_93, mul_281);  sub_93 = mul_281 = None
    mul_282: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(div_3, sub_94);  div_3 = sub_94 = None
    mul_283: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(add_263, mul_247);  mul_247 = None
    sum_17: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_283, [0, 1]);  mul_283 = None
    sum_18: "f32[512]" = torch.ops.aten.sum.dim_IntList(add_263, [0, 1]);  add_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_264: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(add_262, mul_282);  add_262 = mul_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_453: "f32[392, 512]" = torch.ops.aten.view.default(add_264, [392, 512])
    mm_12: "f32[392, 2048]" = torch.ops.aten.mm.default(view_453, permute_321);  permute_321 = None
    permute_322: "f32[512, 392]" = torch.ops.aten.permute.default(view_453, [1, 0])
    mm_13: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_322, view_418);  permute_322 = view_418 = None
    permute_323: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_13, [1, 0]);  mm_13 = None
    sum_19: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_453, [0], True);  view_453 = None
    view_454: "f32[512]" = torch.ops.aten.view.default(sum_19, [512]);  sum_19 = None
    permute_324: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_323, [1, 0]);  permute_323 = None
    view_455: "f32[8, 49, 2048]" = torch.ops.aten.view.default(mm_12, [8, 49, 2048]);  mm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_285: "f32[8, 49, 2048]" = torch.ops.aten.mul.Tensor(add_249, 0.5);  add_249 = None
    mul_286: "f32[8, 49, 2048]" = torch.ops.aten.mul.Tensor(view_417, view_417)
    mul_287: "f32[8, 49, 2048]" = torch.ops.aten.mul.Tensor(mul_286, -0.5);  mul_286 = None
    exp_1: "f32[8, 49, 2048]" = torch.ops.aten.exp.default(mul_287);  mul_287 = None
    mul_288: "f32[8, 49, 2048]" = torch.ops.aten.mul.Tensor(exp_1, 0.3989422804014327);  exp_1 = None
    mul_289: "f32[8, 49, 2048]" = torch.ops.aten.mul.Tensor(view_417, mul_288);  view_417 = mul_288 = None
    add_266: "f32[8, 49, 2048]" = torch.ops.aten.add.Tensor(mul_285, mul_289);  mul_285 = mul_289 = None
    mul_290: "f32[8, 49, 2048]" = torch.ops.aten.mul.Tensor(view_455, add_266);  view_455 = add_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_456: "f32[392, 2048]" = torch.ops.aten.view.default(mul_290, [392, 2048]);  mul_290 = None
    mm_14: "f32[392, 512]" = torch.ops.aten.mm.default(view_456, permute_325);  permute_325 = None
    permute_326: "f32[2048, 392]" = torch.ops.aten.permute.default(view_456, [1, 0])
    mm_15: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_326, view_416);  permute_326 = view_416 = None
    permute_327: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_15, [1, 0]);  mm_15 = None
    sum_20: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_456, [0], True);  view_456 = None
    view_457: "f32[2048]" = torch.ops.aten.view.default(sum_20, [2048]);  sum_20 = None
    permute_328: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_327, [1, 0]);  permute_327 = None
    view_458: "f32[8, 49, 512]" = torch.ops.aten.view.default(mm_14, [8, 49, 512]);  mm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    mul_292: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(view_458, primals_497);  primals_497 = None
    mul_293: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(mul_292, 512)
    sum_21: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_292, [2], True)
    mul_294: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(mul_292, mul_242);  mul_292 = None
    sum_22: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_294, [2], True);  mul_294 = None
    mul_295: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(mul_242, sum_22);  sum_22 = None
    sub_96: "f32[8, 49, 512]" = torch.ops.aten.sub.Tensor(mul_293, sum_21);  mul_293 = sum_21 = None
    sub_97: "f32[8, 49, 512]" = torch.ops.aten.sub.Tensor(sub_96, mul_295);  sub_96 = mul_295 = None
    mul_296: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(div_4, sub_97);  div_4 = sub_97 = None
    mul_297: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(view_458, mul_242);  mul_242 = None
    sum_23: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_297, [0, 1]);  mul_297 = None
    sum_24: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_458, [0, 1]);  view_458 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_267: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(add_264, mul_296);  add_264 = mul_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    view_459: "f32[392, 512]" = torch.ops.aten.view.default(add_267, [392, 512])
    mm_16: "f32[392, 512]" = torch.ops.aten.mm.default(view_459, permute_329);  permute_329 = None
    permute_330: "f32[512, 392]" = torch.ops.aten.permute.default(view_459, [1, 0])
    mm_17: "f32[512, 512]" = torch.ops.aten.mm.default(permute_330, view_414);  permute_330 = view_414 = None
    permute_331: "f32[512, 512]" = torch.ops.aten.permute.default(mm_17, [1, 0]);  mm_17 = None
    sum_25: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_459, [0], True);  view_459 = None
    view_460: "f32[512]" = torch.ops.aten.view.default(sum_25, [512]);  sum_25 = None
    permute_332: "f32[512, 512]" = torch.ops.aten.permute.default(permute_331, [1, 0]);  permute_331 = None
    view_461: "f32[8, 49, 512]" = torch.ops.aten.view.default(mm_16, [8, 49, 512]);  mm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_462: "f32[8, 49, 8, 64]" = torch.ops.aten.view.default(view_461, [8, 49, 8, 64]);  view_461 = None
    permute_333: "f32[8, 8, 49, 64]" = torch.ops.aten.permute.default(view_462, [0, 2, 1, 3]);  view_462 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_backward_1 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_333, permute_278, getitem_320, getitem_321, None, alias_29, getitem_323, getitem_324, getitem_325, 0.0, [True, True, True, False]);  permute_333 = permute_278 = getitem_320 = getitem_321 = alias_29 = getitem_323 = getitem_324 = getitem_325 = None
    getitem_344: "f32[8, 8, 49, 64]" = _scaled_dot_product_efficient_attention_backward_1[0]
    getitem_345: "f32[8, 8, 49, 64]" = _scaled_dot_product_efficient_attention_backward_1[1]
    getitem_346: "f32[8, 8, 49, 64]" = _scaled_dot_product_efficient_attention_backward_1[2];  _scaled_dot_product_efficient_attention_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    cat_1: "f32[16, 8, 49, 64]" = torch.ops.aten.cat.default([getitem_345, getitem_346]);  getitem_345 = getitem_346 = None
    view_463: "f32[2, 8, 8, 49, 64]" = torch.ops.aten.view.default(cat_1, [2, 8, 8, 49, 64]);  cat_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_334: "f32[8, 49, 2, 8, 64]" = torch.ops.aten.permute.default(view_463, [1, 3, 0, 2, 4]);  view_463 = None
    clone_172: "f32[8, 49, 2, 8, 64]" = torch.ops.aten.clone.default(permute_334, memory_format = torch.contiguous_format);  permute_334 = None
    view_464: "f32[8, 49, 1024]" = torch.ops.aten.view.default(clone_172, [8, 49, 1024]);  clone_172 = None
    view_465: "f32[392, 1024]" = torch.ops.aten.view.default(view_464, [392, 1024]);  view_464 = None
    mm_18: "f32[392, 512]" = torch.ops.aten.mm.default(view_465, permute_335);  permute_335 = None
    permute_336: "f32[1024, 392]" = torch.ops.aten.permute.default(view_465, [1, 0])
    mm_19: "f32[1024, 512]" = torch.ops.aten.mm.default(permute_336, view_407);  permute_336 = None
    permute_337: "f32[512, 1024]" = torch.ops.aten.permute.default(mm_19, [1, 0]);  mm_19 = None
    sum_26: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_465, [0], True);  view_465 = None
    view_466: "f32[1024]" = torch.ops.aten.view.default(sum_26, [1024]);  sum_26 = None
    permute_338: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_337, [1, 0]);  permute_337 = None
    view_467: "f32[8, 49, 512]" = torch.ops.aten.view.default(mm_18, [8, 49, 512]);  mm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_339: "f32[8, 49, 8, 64]" = torch.ops.aten.permute.default(getitem_344, [0, 2, 1, 3]);  getitem_344 = None
    view_468: "f32[8, 49, 512]" = torch.ops.aten.view.default(permute_339, [8, 49, 512]);  permute_339 = None
    view_469: "f32[392, 512]" = torch.ops.aten.view.default(view_468, [392, 512]);  view_468 = None
    mm_20: "f32[392, 512]" = torch.ops.aten.mm.default(view_469, permute_340);  permute_340 = None
    permute_341: "f32[512, 392]" = torch.ops.aten.permute.default(view_469, [1, 0])
    mm_21: "f32[512, 512]" = torch.ops.aten.mm.default(permute_341, view_407);  permute_341 = view_407 = None
    permute_342: "f32[512, 512]" = torch.ops.aten.permute.default(mm_21, [1, 0]);  mm_21 = None
    sum_27: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_469, [0], True);  view_469 = None
    view_470: "f32[512]" = torch.ops.aten.view.default(sum_27, [512]);  sum_27 = None
    permute_343: "f32[512, 512]" = torch.ops.aten.permute.default(permute_342, [1, 0]);  permute_342 = None
    view_471: "f32[8, 49, 512]" = torch.ops.aten.view.default(mm_20, [8, 49, 512]);  mm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    add_268: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(view_467, view_471);  view_467 = view_471 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    mul_299: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(add_268, primals_489);  primals_489 = None
    mul_300: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(mul_299, 512)
    sum_28: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_299, [2], True)
    mul_301: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(mul_299, mul_240);  mul_299 = None
    sum_29: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_301, [2], True);  mul_301 = None
    mul_302: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(mul_240, sum_29);  sum_29 = None
    sub_99: "f32[8, 49, 512]" = torch.ops.aten.sub.Tensor(mul_300, sum_28);  mul_300 = sum_28 = None
    sub_100: "f32[8, 49, 512]" = torch.ops.aten.sub.Tensor(sub_99, mul_302);  sub_99 = mul_302 = None
    mul_303: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(div_5, sub_100);  div_5 = sub_100 = None
    mul_304: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(add_268, mul_240);  mul_240 = None
    sum_30: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_304, [0, 1]);  mul_304 = None
    sum_31: "f32[512]" = torch.ops.aten.sum.dim_IntList(add_268, [0, 1]);  add_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_269: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(add_267, mul_303);  add_267 = mul_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:246, code: x = x.flatten(2).transpose(1, 2)
    permute_344: "f32[8, 512, 49]" = torch.ops.aten.permute.default(add_269, [0, 2, 1]);  add_269 = None
    view_472: "f32[8, 512, 7, 7]" = torch.ops.aten.view.default(permute_344, [8, 512, 7, 7]);  permute_344 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:243, code: x = self.proj(cnn_feat_token)
    sum_32: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_472, [0, 2, 3])
    convolution_backward = torch.ops.aten.convolution_backward.default(view_472, view_404, primals_487, [512], [1, 1], [1, 1], [1, 1], False, [0, 0], 512, [True, True, False]);  view_404 = primals_487 = None
    getitem_348: "f32[8, 512, 7, 7]" = convolution_backward[0]
    getitem_349: "f32[512, 1, 3, 3]" = convolution_backward[1];  convolution_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:243, code: x = self.proj(cnn_feat_token)
    add_270: "f32[8, 512, 7, 7]" = torch.ops.aten.add.Tensor(view_472, getitem_348);  view_472 = getitem_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:242, code: cnn_feat_token = x.transpose(1, 2).view(B, C, *size)
    view_473: "f32[8, 512, 49]" = torch.ops.aten.view.default(add_270, [8, 512, 49]);  add_270 = None
    permute_345: "f32[8, 49, 512]" = torch.ops.aten.permute.default(view_473, [0, 2, 1]);  view_473 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_474: "f32[392, 512]" = torch.ops.aten.view.default(permute_345, [392, 512])
    mm_22: "f32[392, 2048]" = torch.ops.aten.mm.default(view_474, permute_346);  permute_346 = None
    permute_347: "f32[512, 392]" = torch.ops.aten.permute.default(view_474, [1, 0])
    mm_23: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_347, view_402);  permute_347 = view_402 = None
    permute_348: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_23, [1, 0]);  mm_23 = None
    sum_33: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_474, [0], True);  view_474 = None
    view_475: "f32[512]" = torch.ops.aten.view.default(sum_33, [512]);  sum_33 = None
    permute_349: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_348, [1, 0]);  permute_348 = None
    view_476: "f32[8, 49, 2048]" = torch.ops.aten.view.default(mm_22, [8, 49, 2048]);  mm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_306: "f32[8, 49, 2048]" = torch.ops.aten.mul.Tensor(add_241, 0.5);  add_241 = None
    mul_307: "f32[8, 49, 2048]" = torch.ops.aten.mul.Tensor(view_401, view_401)
    mul_308: "f32[8, 49, 2048]" = torch.ops.aten.mul.Tensor(mul_307, -0.5);  mul_307 = None
    exp_2: "f32[8, 49, 2048]" = torch.ops.aten.exp.default(mul_308);  mul_308 = None
    mul_309: "f32[8, 49, 2048]" = torch.ops.aten.mul.Tensor(exp_2, 0.3989422804014327);  exp_2 = None
    mul_310: "f32[8, 49, 2048]" = torch.ops.aten.mul.Tensor(view_401, mul_309);  view_401 = mul_309 = None
    add_272: "f32[8, 49, 2048]" = torch.ops.aten.add.Tensor(mul_306, mul_310);  mul_306 = mul_310 = None
    mul_311: "f32[8, 49, 2048]" = torch.ops.aten.mul.Tensor(view_476, add_272);  view_476 = add_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_477: "f32[392, 2048]" = torch.ops.aten.view.default(mul_311, [392, 2048]);  mul_311 = None
    mm_24: "f32[392, 512]" = torch.ops.aten.mm.default(view_477, permute_350);  permute_350 = None
    permute_351: "f32[2048, 392]" = torch.ops.aten.permute.default(view_477, [1, 0])
    mm_25: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_351, view_400);  permute_351 = view_400 = None
    permute_352: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_25, [1, 0]);  mm_25 = None
    sum_34: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_477, [0], True);  view_477 = None
    view_478: "f32[2048]" = torch.ops.aten.view.default(sum_34, [2048]);  sum_34 = None
    permute_353: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_352, [1, 0]);  permute_352 = None
    view_479: "f32[8, 49, 512]" = torch.ops.aten.view.default(mm_24, [8, 49, 512]);  mm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    mul_313: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(view_479, primals_481);  primals_481 = None
    mul_314: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(mul_313, 512)
    sum_35: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_313, [2], True)
    mul_315: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(mul_313, mul_235);  mul_313 = None
    sum_36: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_315, [2], True);  mul_315 = None
    mul_316: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(mul_235, sum_36);  sum_36 = None
    sub_102: "f32[8, 49, 512]" = torch.ops.aten.sub.Tensor(mul_314, sum_35);  mul_314 = sum_35 = None
    sub_103: "f32[8, 49, 512]" = torch.ops.aten.sub.Tensor(sub_102, mul_316);  sub_102 = mul_316 = None
    mul_317: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(div_6, sub_103);  div_6 = sub_103 = None
    mul_318: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(view_479, mul_235);  mul_235 = None
    sum_37: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_318, [0, 1]);  mul_318 = None
    sum_38: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_479, [0, 1]);  view_479 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_273: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(permute_345, mul_317);  permute_345 = mul_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    view_480: "f32[392, 512]" = torch.ops.aten.view.default(add_273, [392, 512])
    mm_26: "f32[392, 512]" = torch.ops.aten.mm.default(view_480, permute_354);  permute_354 = None
    permute_355: "f32[512, 392]" = torch.ops.aten.permute.default(view_480, [1, 0])
    mm_27: "f32[512, 512]" = torch.ops.aten.mm.default(permute_355, view_398);  permute_355 = view_398 = None
    permute_356: "f32[512, 512]" = torch.ops.aten.permute.default(mm_27, [1, 0]);  mm_27 = None
    sum_39: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_480, [0], True);  view_480 = None
    view_481: "f32[512]" = torch.ops.aten.view.default(sum_39, [512]);  sum_39 = None
    permute_357: "f32[512, 512]" = torch.ops.aten.permute.default(permute_356, [1, 0]);  permute_356 = None
    view_482: "f32[8, 49, 512]" = torch.ops.aten.view.default(mm_26, [8, 49, 512]);  mm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_483: "f32[8, 49, 8, 64]" = torch.ops.aten.view.default(view_482, [8, 49, 8, 64]);  view_482 = None
    permute_358: "f32[8, 8, 49, 64]" = torch.ops.aten.permute.default(view_483, [0, 2, 1, 3]);  view_483 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_backward_2 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_358, permute_267, getitem_310, getitem_311, None, alias_30, getitem_313, getitem_314, getitem_315, 0.0, [True, True, True, False]);  permute_358 = permute_267 = getitem_310 = getitem_311 = alias_30 = getitem_313 = getitem_314 = getitem_315 = None
    getitem_351: "f32[8, 8, 49, 64]" = _scaled_dot_product_efficient_attention_backward_2[0]
    getitem_352: "f32[8, 8, 49, 64]" = _scaled_dot_product_efficient_attention_backward_2[1]
    getitem_353: "f32[8, 8, 49, 64]" = _scaled_dot_product_efficient_attention_backward_2[2];  _scaled_dot_product_efficient_attention_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    cat_2: "f32[16, 8, 49, 64]" = torch.ops.aten.cat.default([getitem_352, getitem_353]);  getitem_352 = getitem_353 = None
    view_484: "f32[2, 8, 8, 49, 64]" = torch.ops.aten.view.default(cat_2, [2, 8, 8, 49, 64]);  cat_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_359: "f32[8, 49, 2, 8, 64]" = torch.ops.aten.permute.default(view_484, [1, 3, 0, 2, 4]);  view_484 = None
    clone_174: "f32[8, 49, 2, 8, 64]" = torch.ops.aten.clone.default(permute_359, memory_format = torch.contiguous_format);  permute_359 = None
    view_485: "f32[8, 49, 1024]" = torch.ops.aten.view.default(clone_174, [8, 49, 1024]);  clone_174 = None
    view_486: "f32[392, 1024]" = torch.ops.aten.view.default(view_485, [392, 1024]);  view_485 = None
    mm_28: "f32[392, 512]" = torch.ops.aten.mm.default(view_486, permute_360);  permute_360 = None
    permute_361: "f32[1024, 392]" = torch.ops.aten.permute.default(view_486, [1, 0])
    mm_29: "f32[1024, 512]" = torch.ops.aten.mm.default(permute_361, view_391);  permute_361 = None
    permute_362: "f32[512, 1024]" = torch.ops.aten.permute.default(mm_29, [1, 0]);  mm_29 = None
    sum_40: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_486, [0], True);  view_486 = None
    view_487: "f32[1024]" = torch.ops.aten.view.default(sum_40, [1024]);  sum_40 = None
    permute_363: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_362, [1, 0]);  permute_362 = None
    view_488: "f32[8, 49, 512]" = torch.ops.aten.view.default(mm_28, [8, 49, 512]);  mm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_364: "f32[8, 49, 8, 64]" = torch.ops.aten.permute.default(getitem_351, [0, 2, 1, 3]);  getitem_351 = None
    view_489: "f32[8, 49, 512]" = torch.ops.aten.view.default(permute_364, [8, 49, 512]);  permute_364 = None
    view_490: "f32[392, 512]" = torch.ops.aten.view.default(view_489, [392, 512]);  view_489 = None
    mm_30: "f32[392, 512]" = torch.ops.aten.mm.default(view_490, permute_365);  permute_365 = None
    permute_366: "f32[512, 392]" = torch.ops.aten.permute.default(view_490, [1, 0])
    mm_31: "f32[512, 512]" = torch.ops.aten.mm.default(permute_366, view_391);  permute_366 = view_391 = None
    permute_367: "f32[512, 512]" = torch.ops.aten.permute.default(mm_31, [1, 0]);  mm_31 = None
    sum_41: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_490, [0], True);  view_490 = None
    view_491: "f32[512]" = torch.ops.aten.view.default(sum_41, [512]);  sum_41 = None
    permute_368: "f32[512, 512]" = torch.ops.aten.permute.default(permute_367, [1, 0]);  permute_367 = None
    view_492: "f32[8, 49, 512]" = torch.ops.aten.view.default(mm_30, [8, 49, 512]);  mm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    add_274: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(view_488, view_492);  view_488 = view_492 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    mul_320: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(add_274, primals_473);  primals_473 = None
    mul_321: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(mul_320, 512)
    sum_42: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_320, [2], True)
    mul_322: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(mul_320, mul_233);  mul_320 = None
    sum_43: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_322, [2], True);  mul_322 = None
    mul_323: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(mul_233, sum_43);  sum_43 = None
    sub_105: "f32[8, 49, 512]" = torch.ops.aten.sub.Tensor(mul_321, sum_42);  mul_321 = sum_42 = None
    sub_106: "f32[8, 49, 512]" = torch.ops.aten.sub.Tensor(sub_105, mul_323);  sub_105 = mul_323 = None
    mul_324: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(div_7, sub_106);  div_7 = sub_106 = None
    mul_325: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(add_274, mul_233);  mul_233 = None
    sum_44: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_325, [0, 1]);  mul_325 = None
    sum_45: "f32[512]" = torch.ops.aten.sum.dim_IntList(add_274, [0, 1]);  add_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_275: "f32[8, 49, 512]" = torch.ops.aten.add.Tensor(add_273, mul_324);  add_273 = mul_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:275, code: x = self.norm(x)
    mul_327: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(add_275, primals_471);  primals_471 = None
    mul_328: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(mul_327, 512)
    sum_46: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_327, [2], True)
    mul_329: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(mul_327, mul_231);  mul_327 = None
    sum_47: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_329, [2], True);  mul_329 = None
    mul_330: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(mul_231, sum_47);  sum_47 = None
    sub_108: "f32[8, 49, 512]" = torch.ops.aten.sub.Tensor(mul_328, sum_46);  mul_328 = sum_46 = None
    sub_109: "f32[8, 49, 512]" = torch.ops.aten.sub.Tensor(sub_108, mul_330);  sub_108 = mul_330 = None
    mul_331: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(div_8, sub_109);  div_8 = sub_109 = None
    mul_332: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(add_275, mul_231);  mul_231 = None
    sum_48: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_332, [0, 1]);  mul_332 = None
    sum_49: "f32[512]" = torch.ops.aten.sum.dim_IntList(add_275, [0, 1]);  add_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:274, code: x = self.proj(x).flatten(2).transpose(1, 2)
    permute_369: "f32[8, 512, 49]" = torch.ops.aten.permute.default(mul_331, [0, 2, 1]);  mul_331 = None
    view_493: "f32[8, 512, 7, 7]" = torch.ops.aten.view.default(permute_369, [8, 512, 7, 7]);  permute_369 = None
    sum_50: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_493, [0, 2, 3])
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(view_493, permute_264, primals_469, [512], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_493 = permute_264 = primals_469 = None
    getitem_355: "f32[8, 320, 14, 14]" = convolution_backward_1[0]
    getitem_356: "f32[512, 320, 2, 2]" = convolution_backward_1[1];  convolution_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:415, code: x = x.reshape(B, *size, -1).permute(0, 3, 1, 2).contiguous()
    permute_370: "f32[8, 14, 14, 320]" = torch.ops.aten.permute.default(getitem_355, [0, 2, 3, 1]);  getitem_355 = None
    view_494: "f32[8, 196, 320]" = torch.ops.aten.view.default(permute_370, [8, 196, 320]);  permute_370 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    clone_176: "f32[8, 196, 320]" = torch.ops.aten.clone.default(view_494, memory_format = torch.contiguous_format)
    view_495: "f32[1568, 320]" = torch.ops.aten.view.default(clone_176, [1568, 320]);  clone_176 = None
    mm_32: "f32[1568, 1280]" = torch.ops.aten.mm.default(view_495, permute_371);  permute_371 = None
    permute_372: "f32[320, 1568]" = torch.ops.aten.permute.default(view_495, [1, 0])
    mm_33: "f32[320, 1280]" = torch.ops.aten.mm.default(permute_372, view_387);  permute_372 = view_387 = None
    permute_373: "f32[1280, 320]" = torch.ops.aten.permute.default(mm_33, [1, 0]);  mm_33 = None
    sum_51: "f32[1, 320]" = torch.ops.aten.sum.dim_IntList(view_495, [0], True);  view_495 = None
    view_496: "f32[320]" = torch.ops.aten.view.default(sum_51, [320]);  sum_51 = None
    permute_374: "f32[320, 1280]" = torch.ops.aten.permute.default(permute_373, [1, 0]);  permute_373 = None
    view_497: "f32[8, 196, 1280]" = torch.ops.aten.view.default(mm_32, [8, 196, 1280]);  mm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_334: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(add_232, 0.5);  add_232 = None
    mul_335: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_386, view_386)
    mul_336: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(mul_335, -0.5);  mul_335 = None
    exp_3: "f32[8, 196, 1280]" = torch.ops.aten.exp.default(mul_336);  mul_336 = None
    mul_337: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(exp_3, 0.3989422804014327);  exp_3 = None
    mul_338: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_386, mul_337);  view_386 = mul_337 = None
    add_277: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(mul_334, mul_338);  mul_334 = mul_338 = None
    mul_339: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_497, add_277);  view_497 = add_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_498: "f32[1568, 1280]" = torch.ops.aten.view.default(mul_339, [1568, 1280]);  mul_339 = None
    mm_34: "f32[1568, 320]" = torch.ops.aten.mm.default(view_498, permute_375);  permute_375 = None
    permute_376: "f32[1280, 1568]" = torch.ops.aten.permute.default(view_498, [1, 0])
    mm_35: "f32[1280, 320]" = torch.ops.aten.mm.default(permute_376, view_385);  permute_376 = view_385 = None
    permute_377: "f32[320, 1280]" = torch.ops.aten.permute.default(mm_35, [1, 0]);  mm_35 = None
    sum_52: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_498, [0], True);  view_498 = None
    view_499: "f32[1280]" = torch.ops.aten.view.default(sum_52, [1280]);  sum_52 = None
    permute_378: "f32[1280, 320]" = torch.ops.aten.permute.default(permute_377, [1, 0]);  permute_377 = None
    view_500: "f32[8, 196, 320]" = torch.ops.aten.view.default(mm_34, [8, 196, 320]);  mm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    mul_341: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(view_500, primals_463);  primals_463 = None
    mul_342: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_341, 320)
    sum_53: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_341, [2], True)
    mul_343: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_341, mul_226);  mul_341 = None
    sum_54: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_343, [2], True);  mul_343 = None
    mul_344: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_226, sum_54);  sum_54 = None
    sub_111: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(mul_342, sum_53);  mul_342 = sum_53 = None
    sub_112: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(sub_111, mul_344);  sub_111 = mul_344 = None
    mul_345: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(div_9, sub_112);  div_9 = sub_112 = None
    mul_346: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(view_500, mul_226);  mul_226 = None
    sum_55: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_346, [0, 1]);  mul_346 = None
    sum_56: "f32[320]" = torch.ops.aten.sum.dim_IntList(view_500, [0, 1]);  view_500 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_278: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(view_494, mul_345);  view_494 = mul_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    clone_178: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_278, memory_format = torch.contiguous_format)
    view_501: "f32[1568, 320]" = torch.ops.aten.view.default(clone_178, [1568, 320]);  clone_178 = None
    mm_36: "f32[1568, 320]" = torch.ops.aten.mm.default(view_501, permute_379);  permute_379 = None
    permute_380: "f32[320, 1568]" = torch.ops.aten.permute.default(view_501, [1, 0])
    mm_37: "f32[320, 320]" = torch.ops.aten.mm.default(permute_380, view_383);  permute_380 = view_383 = None
    permute_381: "f32[320, 320]" = torch.ops.aten.permute.default(mm_37, [1, 0]);  mm_37 = None
    sum_57: "f32[1, 320]" = torch.ops.aten.sum.dim_IntList(view_501, [0], True);  view_501 = None
    view_502: "f32[320]" = torch.ops.aten.view.default(sum_57, [320]);  sum_57 = None
    permute_382: "f32[320, 320]" = torch.ops.aten.permute.default(permute_381, [1, 0]);  permute_381 = None
    view_503: "f32[8, 196, 320]" = torch.ops.aten.view.default(mm_36, [8, 196, 320]);  mm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_504: "f32[8, 196, 5, 64]" = torch.ops.aten.view.default(view_503, [8, 196, 5, 64]);  view_503 = None
    permute_383: "f32[8, 5, 196, 64]" = torch.ops.aten.permute.default(view_504, [0, 2, 1, 3]);  view_504 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_backward_3 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_383, permute_255, getitem_298, getitem_299, None, alias_31, getitem_301, getitem_302, getitem_303, 0.0, [True, True, True, False]);  permute_383 = permute_255 = getitem_298 = getitem_299 = alias_31 = getitem_301 = getitem_302 = getitem_303 = None
    getitem_358: "f32[8, 5, 196, 64]" = _scaled_dot_product_efficient_attention_backward_3[0]
    getitem_359: "f32[8, 5, 49, 64]" = _scaled_dot_product_efficient_attention_backward_3[1]
    getitem_360: "f32[8, 5, 49, 64]" = _scaled_dot_product_efficient_attention_backward_3[2];  _scaled_dot_product_efficient_attention_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    cat_3: "f32[16, 5, 49, 64]" = torch.ops.aten.cat.default([getitem_359, getitem_360]);  getitem_359 = getitem_360 = None
    view_505: "f32[2, 8, 5, 49, 64]" = torch.ops.aten.view.default(cat_3, [2, 8, 5, 49, 64]);  cat_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_384: "f32[8, 49, 2, 5, 64]" = torch.ops.aten.permute.default(view_505, [1, 3, 0, 2, 4]);  view_505 = None
    clone_179: "f32[8, 49, 2, 5, 64]" = torch.ops.aten.clone.default(permute_384, memory_format = torch.contiguous_format);  permute_384 = None
    view_506: "f32[8, 49, 640]" = torch.ops.aten.view.default(clone_179, [8, 49, 640]);  clone_179 = None
    view_507: "f32[392, 640]" = torch.ops.aten.view.default(view_506, [392, 640]);  view_506 = None
    mm_38: "f32[392, 320]" = torch.ops.aten.mm.default(view_507, permute_385);  permute_385 = None
    permute_386: "f32[640, 392]" = torch.ops.aten.permute.default(view_507, [1, 0])
    mm_39: "f32[640, 320]" = torch.ops.aten.mm.default(permute_386, view_379);  permute_386 = view_379 = None
    permute_387: "f32[320, 640]" = torch.ops.aten.permute.default(mm_39, [1, 0]);  mm_39 = None
    sum_58: "f32[1, 640]" = torch.ops.aten.sum.dim_IntList(view_507, [0], True);  view_507 = None
    view_508: "f32[640]" = torch.ops.aten.view.default(sum_58, [640]);  sum_58 = None
    permute_388: "f32[640, 320]" = torch.ops.aten.permute.default(permute_387, [1, 0]);  permute_387 = None
    view_509: "f32[8, 49, 320]" = torch.ops.aten.view.default(mm_38, [8, 49, 320]);  mm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    mul_348: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(view_509, primals_457);  primals_457 = None
    mul_349: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_348, 320)
    sum_59: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_348, [2], True)
    mul_350: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_348, mul_224);  mul_348 = None
    sum_60: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_350, [2], True);  mul_350 = None
    mul_351: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_224, sum_60);  sum_60 = None
    sub_114: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(mul_349, sum_59);  mul_349 = sum_59 = None
    sub_115: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(sub_114, mul_351);  sub_114 = mul_351 = None
    mul_352: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(div_10, sub_115);  div_10 = sub_115 = None
    mul_353: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(view_509, mul_224);  mul_224 = None
    sum_61: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_353, [0, 1]);  mul_353 = None
    sum_62: "f32[320]" = torch.ops.aten.sum.dim_IntList(view_509, [0, 1]);  view_509 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    permute_389: "f32[8, 320, 49]" = torch.ops.aten.permute.default(mul_352, [0, 2, 1]);  mul_352 = None
    view_510: "f32[8, 320, 7, 7]" = torch.ops.aten.view.default(permute_389, [8, 320, 7, 7]);  permute_389 = None
    sum_63: "f32[320]" = torch.ops.aten.sum.dim_IntList(view_510, [0, 2, 3])
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(view_510, view_377, primals_455, [320], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_510 = view_377 = primals_455 = None
    getitem_362: "f32[8, 320, 14, 14]" = convolution_backward_2[0]
    getitem_363: "f32[320, 320, 2, 2]" = convolution_backward_2[1];  convolution_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    view_511: "f32[8, 320, 196]" = torch.ops.aten.view.default(getitem_362, [8, 320, 196]);  getitem_362 = None
    permute_390: "f32[8, 196, 320]" = torch.ops.aten.permute.default(view_511, [0, 2, 1]);  view_511 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_391: "f32[8, 196, 5, 64]" = torch.ops.aten.permute.default(getitem_358, [0, 2, 1, 3]);  getitem_358 = None
    view_512: "f32[8, 196, 320]" = torch.ops.aten.view.default(permute_391, [8, 196, 320]);  permute_391 = None
    view_513: "f32[1568, 320]" = torch.ops.aten.view.default(view_512, [1568, 320]);  view_512 = None
    mm_40: "f32[1568, 320]" = torch.ops.aten.mm.default(view_513, permute_392);  permute_392 = None
    permute_393: "f32[320, 1568]" = torch.ops.aten.permute.default(view_513, [1, 0])
    mm_41: "f32[320, 320]" = torch.ops.aten.mm.default(permute_393, view_374);  permute_393 = view_374 = None
    permute_394: "f32[320, 320]" = torch.ops.aten.permute.default(mm_41, [1, 0]);  mm_41 = None
    sum_64: "f32[1, 320]" = torch.ops.aten.sum.dim_IntList(view_513, [0], True);  view_513 = None
    view_514: "f32[320]" = torch.ops.aten.view.default(sum_64, [320]);  sum_64 = None
    permute_395: "f32[320, 320]" = torch.ops.aten.permute.default(permute_394, [1, 0]);  permute_394 = None
    view_515: "f32[8, 196, 320]" = torch.ops.aten.view.default(mm_40, [8, 196, 320]);  mm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    add_279: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(permute_390, view_515);  permute_390 = view_515 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    clone_181: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_279, memory_format = torch.contiguous_format);  add_279 = None
    mul_355: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(clone_181, primals_451);  primals_451 = None
    mul_356: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_355, 320)
    sum_65: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_355, [2], True)
    mul_357: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_355, mul_222);  mul_355 = None
    sum_66: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_357, [2], True);  mul_357 = None
    mul_358: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_222, sum_66);  sum_66 = None
    sub_117: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(mul_356, sum_65);  mul_356 = sum_65 = None
    sub_118: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(sub_117, mul_358);  sub_117 = mul_358 = None
    mul_359: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(div_11, sub_118);  div_11 = sub_118 = None
    mul_360: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(clone_181, mul_222);  mul_222 = None
    sum_67: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_360, [0, 1]);  mul_360 = None
    sum_68: "f32[320]" = torch.ops.aten.sum.dim_IntList(clone_181, [0, 1]);  clone_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_280: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_278, mul_359);  add_278 = mul_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    clone_183: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_280, memory_format = torch.contiguous_format)
    view_516: "f32[1568, 320]" = torch.ops.aten.view.default(clone_183, [1568, 320]);  clone_183 = None
    mm_42: "f32[1568, 1280]" = torch.ops.aten.mm.default(view_516, permute_396);  permute_396 = None
    permute_397: "f32[320, 1568]" = torch.ops.aten.permute.default(view_516, [1, 0])
    mm_43: "f32[320, 1280]" = torch.ops.aten.mm.default(permute_397, view_372);  permute_397 = view_372 = None
    permute_398: "f32[1280, 320]" = torch.ops.aten.permute.default(mm_43, [1, 0]);  mm_43 = None
    sum_69: "f32[1, 320]" = torch.ops.aten.sum.dim_IntList(view_516, [0], True);  view_516 = None
    view_517: "f32[320]" = torch.ops.aten.view.default(sum_69, [320]);  sum_69 = None
    permute_399: "f32[320, 1280]" = torch.ops.aten.permute.default(permute_398, [1, 0]);  permute_398 = None
    view_518: "f32[8, 196, 1280]" = torch.ops.aten.view.default(mm_42, [8, 196, 1280]);  mm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_362: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(add_223, 0.5);  add_223 = None
    mul_363: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_371, view_371)
    mul_364: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(mul_363, -0.5);  mul_363 = None
    exp_4: "f32[8, 196, 1280]" = torch.ops.aten.exp.default(mul_364);  mul_364 = None
    mul_365: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(exp_4, 0.3989422804014327);  exp_4 = None
    mul_366: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_371, mul_365);  view_371 = mul_365 = None
    add_282: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(mul_362, mul_366);  mul_362 = mul_366 = None
    mul_367: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_518, add_282);  view_518 = add_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_519: "f32[1568, 1280]" = torch.ops.aten.view.default(mul_367, [1568, 1280]);  mul_367 = None
    mm_44: "f32[1568, 320]" = torch.ops.aten.mm.default(view_519, permute_400);  permute_400 = None
    permute_401: "f32[1280, 1568]" = torch.ops.aten.permute.default(view_519, [1, 0])
    mm_45: "f32[1280, 320]" = torch.ops.aten.mm.default(permute_401, view_370);  permute_401 = view_370 = None
    permute_402: "f32[320, 1280]" = torch.ops.aten.permute.default(mm_45, [1, 0]);  mm_45 = None
    sum_70: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_519, [0], True);  view_519 = None
    view_520: "f32[1280]" = torch.ops.aten.view.default(sum_70, [1280]);  sum_70 = None
    permute_403: "f32[1280, 320]" = torch.ops.aten.permute.default(permute_402, [1, 0]);  permute_402 = None
    view_521: "f32[8, 196, 320]" = torch.ops.aten.view.default(mm_44, [8, 196, 320]);  mm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    mul_369: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(view_521, primals_445);  primals_445 = None
    mul_370: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_369, 320)
    sum_71: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_369, [2], True)
    mul_371: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_369, mul_217);  mul_369 = None
    sum_72: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_371, [2], True);  mul_371 = None
    mul_372: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_217, sum_72);  sum_72 = None
    sub_120: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(mul_370, sum_71);  mul_370 = sum_71 = None
    sub_121: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(sub_120, mul_372);  sub_120 = mul_372 = None
    mul_373: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(div_12, sub_121);  div_12 = sub_121 = None
    mul_374: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(view_521, mul_217);  mul_217 = None
    sum_73: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_374, [0, 1]);  mul_374 = None
    sum_74: "f32[320]" = torch.ops.aten.sum.dim_IntList(view_521, [0, 1]);  view_521 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_283: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_280, mul_373);  add_280 = mul_373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    clone_185: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_283, memory_format = torch.contiguous_format)
    view_522: "f32[1568, 320]" = torch.ops.aten.view.default(clone_185, [1568, 320]);  clone_185 = None
    mm_46: "f32[1568, 320]" = torch.ops.aten.mm.default(view_522, permute_404);  permute_404 = None
    permute_405: "f32[320, 1568]" = torch.ops.aten.permute.default(view_522, [1, 0])
    mm_47: "f32[320, 320]" = torch.ops.aten.mm.default(permute_405, view_368);  permute_405 = view_368 = None
    permute_406: "f32[320, 320]" = torch.ops.aten.permute.default(mm_47, [1, 0]);  mm_47 = None
    sum_75: "f32[1, 320]" = torch.ops.aten.sum.dim_IntList(view_522, [0], True);  view_522 = None
    view_523: "f32[320]" = torch.ops.aten.view.default(sum_75, [320]);  sum_75 = None
    permute_407: "f32[320, 320]" = torch.ops.aten.permute.default(permute_406, [1, 0]);  permute_406 = None
    view_524: "f32[8, 196, 320]" = torch.ops.aten.view.default(mm_46, [8, 196, 320]);  mm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_525: "f32[8, 196, 5, 64]" = torch.ops.aten.view.default(view_524, [8, 196, 5, 64]);  view_524 = None
    permute_408: "f32[8, 5, 196, 64]" = torch.ops.aten.permute.default(view_525, [0, 2, 1, 3]);  view_525 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_backward_4 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_408, permute_245, getitem_286, getitem_287, None, alias_32, getitem_289, getitem_290, getitem_291, 0.0, [True, True, True, False]);  permute_408 = permute_245 = getitem_286 = getitem_287 = alias_32 = getitem_289 = getitem_290 = getitem_291 = None
    getitem_365: "f32[8, 5, 196, 64]" = _scaled_dot_product_efficient_attention_backward_4[0]
    getitem_366: "f32[8, 5, 49, 64]" = _scaled_dot_product_efficient_attention_backward_4[1]
    getitem_367: "f32[8, 5, 49, 64]" = _scaled_dot_product_efficient_attention_backward_4[2];  _scaled_dot_product_efficient_attention_backward_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    cat_4: "f32[16, 5, 49, 64]" = torch.ops.aten.cat.default([getitem_366, getitem_367]);  getitem_366 = getitem_367 = None
    view_526: "f32[2, 8, 5, 49, 64]" = torch.ops.aten.view.default(cat_4, [2, 8, 5, 49, 64]);  cat_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_409: "f32[8, 49, 2, 5, 64]" = torch.ops.aten.permute.default(view_526, [1, 3, 0, 2, 4]);  view_526 = None
    clone_186: "f32[8, 49, 2, 5, 64]" = torch.ops.aten.clone.default(permute_409, memory_format = torch.contiguous_format);  permute_409 = None
    view_527: "f32[8, 49, 640]" = torch.ops.aten.view.default(clone_186, [8, 49, 640]);  clone_186 = None
    view_528: "f32[392, 640]" = torch.ops.aten.view.default(view_527, [392, 640]);  view_527 = None
    mm_48: "f32[392, 320]" = torch.ops.aten.mm.default(view_528, permute_410);  permute_410 = None
    permute_411: "f32[640, 392]" = torch.ops.aten.permute.default(view_528, [1, 0])
    mm_49: "f32[640, 320]" = torch.ops.aten.mm.default(permute_411, view_364);  permute_411 = view_364 = None
    permute_412: "f32[320, 640]" = torch.ops.aten.permute.default(mm_49, [1, 0]);  mm_49 = None
    sum_76: "f32[1, 640]" = torch.ops.aten.sum.dim_IntList(view_528, [0], True);  view_528 = None
    view_529: "f32[640]" = torch.ops.aten.view.default(sum_76, [640]);  sum_76 = None
    permute_413: "f32[640, 320]" = torch.ops.aten.permute.default(permute_412, [1, 0]);  permute_412 = None
    view_530: "f32[8, 49, 320]" = torch.ops.aten.view.default(mm_48, [8, 49, 320]);  mm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    mul_376: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(view_530, primals_439);  primals_439 = None
    mul_377: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_376, 320)
    sum_77: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_376, [2], True)
    mul_378: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_376, mul_215);  mul_376 = None
    sum_78: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_378, [2], True);  mul_378 = None
    mul_379: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_215, sum_78);  sum_78 = None
    sub_123: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(mul_377, sum_77);  mul_377 = sum_77 = None
    sub_124: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(sub_123, mul_379);  sub_123 = mul_379 = None
    mul_380: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(div_13, sub_124);  div_13 = sub_124 = None
    mul_381: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(view_530, mul_215);  mul_215 = None
    sum_79: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_381, [0, 1]);  mul_381 = None
    sum_80: "f32[320]" = torch.ops.aten.sum.dim_IntList(view_530, [0, 1]);  view_530 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    permute_414: "f32[8, 320, 49]" = torch.ops.aten.permute.default(mul_380, [0, 2, 1]);  mul_380 = None
    view_531: "f32[8, 320, 7, 7]" = torch.ops.aten.view.default(permute_414, [8, 320, 7, 7]);  permute_414 = None
    sum_81: "f32[320]" = torch.ops.aten.sum.dim_IntList(view_531, [0, 2, 3])
    convolution_backward_3 = torch.ops.aten.convolution_backward.default(view_531, view_362, primals_437, [320], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_531 = view_362 = primals_437 = None
    getitem_369: "f32[8, 320, 14, 14]" = convolution_backward_3[0]
    getitem_370: "f32[320, 320, 2, 2]" = convolution_backward_3[1];  convolution_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    view_532: "f32[8, 320, 196]" = torch.ops.aten.view.default(getitem_369, [8, 320, 196]);  getitem_369 = None
    permute_415: "f32[8, 196, 320]" = torch.ops.aten.permute.default(view_532, [0, 2, 1]);  view_532 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_416: "f32[8, 196, 5, 64]" = torch.ops.aten.permute.default(getitem_365, [0, 2, 1, 3]);  getitem_365 = None
    view_533: "f32[8, 196, 320]" = torch.ops.aten.view.default(permute_416, [8, 196, 320]);  permute_416 = None
    view_534: "f32[1568, 320]" = torch.ops.aten.view.default(view_533, [1568, 320]);  view_533 = None
    mm_50: "f32[1568, 320]" = torch.ops.aten.mm.default(view_534, permute_417);  permute_417 = None
    permute_418: "f32[320, 1568]" = torch.ops.aten.permute.default(view_534, [1, 0])
    mm_51: "f32[320, 320]" = torch.ops.aten.mm.default(permute_418, view_359);  permute_418 = view_359 = None
    permute_419: "f32[320, 320]" = torch.ops.aten.permute.default(mm_51, [1, 0]);  mm_51 = None
    sum_82: "f32[1, 320]" = torch.ops.aten.sum.dim_IntList(view_534, [0], True);  view_534 = None
    view_535: "f32[320]" = torch.ops.aten.view.default(sum_82, [320]);  sum_82 = None
    permute_420: "f32[320, 320]" = torch.ops.aten.permute.default(permute_419, [1, 0]);  permute_419 = None
    view_536: "f32[8, 196, 320]" = torch.ops.aten.view.default(mm_50, [8, 196, 320]);  mm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    add_284: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(permute_415, view_536);  permute_415 = view_536 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    clone_188: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_284, memory_format = torch.contiguous_format);  add_284 = None
    mul_383: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(clone_188, primals_433);  primals_433 = None
    mul_384: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_383, 320)
    sum_83: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_383, [2], True)
    mul_385: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_383, mul_213);  mul_383 = None
    sum_84: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_385, [2], True);  mul_385 = None
    mul_386: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_213, sum_84);  sum_84 = None
    sub_126: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(mul_384, sum_83);  mul_384 = sum_83 = None
    sub_127: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(sub_126, mul_386);  sub_126 = mul_386 = None
    mul_387: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(div_14, sub_127);  div_14 = sub_127 = None
    mul_388: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(clone_188, mul_213);  mul_213 = None
    sum_85: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_388, [0, 1]);  mul_388 = None
    sum_86: "f32[320]" = torch.ops.aten.sum.dim_IntList(clone_188, [0, 1]);  clone_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_285: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_283, mul_387);  add_283 = mul_387 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    clone_190: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_285, memory_format = torch.contiguous_format)
    view_537: "f32[1568, 320]" = torch.ops.aten.view.default(clone_190, [1568, 320]);  clone_190 = None
    mm_52: "f32[1568, 1280]" = torch.ops.aten.mm.default(view_537, permute_421);  permute_421 = None
    permute_422: "f32[320, 1568]" = torch.ops.aten.permute.default(view_537, [1, 0])
    mm_53: "f32[320, 1280]" = torch.ops.aten.mm.default(permute_422, view_357);  permute_422 = view_357 = None
    permute_423: "f32[1280, 320]" = torch.ops.aten.permute.default(mm_53, [1, 0]);  mm_53 = None
    sum_87: "f32[1, 320]" = torch.ops.aten.sum.dim_IntList(view_537, [0], True);  view_537 = None
    view_538: "f32[320]" = torch.ops.aten.view.default(sum_87, [320]);  sum_87 = None
    permute_424: "f32[320, 1280]" = torch.ops.aten.permute.default(permute_423, [1, 0]);  permute_423 = None
    view_539: "f32[8, 196, 1280]" = torch.ops.aten.view.default(mm_52, [8, 196, 1280]);  mm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_390: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(add_214, 0.5);  add_214 = None
    mul_391: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_356, view_356)
    mul_392: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(mul_391, -0.5);  mul_391 = None
    exp_5: "f32[8, 196, 1280]" = torch.ops.aten.exp.default(mul_392);  mul_392 = None
    mul_393: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(exp_5, 0.3989422804014327);  exp_5 = None
    mul_394: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_356, mul_393);  view_356 = mul_393 = None
    add_287: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(mul_390, mul_394);  mul_390 = mul_394 = None
    mul_395: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_539, add_287);  view_539 = add_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_540: "f32[1568, 1280]" = torch.ops.aten.view.default(mul_395, [1568, 1280]);  mul_395 = None
    mm_54: "f32[1568, 320]" = torch.ops.aten.mm.default(view_540, permute_425);  permute_425 = None
    permute_426: "f32[1280, 1568]" = torch.ops.aten.permute.default(view_540, [1, 0])
    mm_55: "f32[1280, 320]" = torch.ops.aten.mm.default(permute_426, view_355);  permute_426 = view_355 = None
    permute_427: "f32[320, 1280]" = torch.ops.aten.permute.default(mm_55, [1, 0]);  mm_55 = None
    sum_88: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_540, [0], True);  view_540 = None
    view_541: "f32[1280]" = torch.ops.aten.view.default(sum_88, [1280]);  sum_88 = None
    permute_428: "f32[1280, 320]" = torch.ops.aten.permute.default(permute_427, [1, 0]);  permute_427 = None
    view_542: "f32[8, 196, 320]" = torch.ops.aten.view.default(mm_54, [8, 196, 320]);  mm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    mul_397: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(view_542, primals_427);  primals_427 = None
    mul_398: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_397, 320)
    sum_89: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_397, [2], True)
    mul_399: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_397, mul_208);  mul_397 = None
    sum_90: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_399, [2], True);  mul_399 = None
    mul_400: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_208, sum_90);  sum_90 = None
    sub_129: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(mul_398, sum_89);  mul_398 = sum_89 = None
    sub_130: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(sub_129, mul_400);  sub_129 = mul_400 = None
    mul_401: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(div_15, sub_130);  div_15 = sub_130 = None
    mul_402: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(view_542, mul_208);  mul_208 = None
    sum_91: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_402, [0, 1]);  mul_402 = None
    sum_92: "f32[320]" = torch.ops.aten.sum.dim_IntList(view_542, [0, 1]);  view_542 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_288: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_285, mul_401);  add_285 = mul_401 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    clone_192: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_288, memory_format = torch.contiguous_format)
    view_543: "f32[1568, 320]" = torch.ops.aten.view.default(clone_192, [1568, 320]);  clone_192 = None
    mm_56: "f32[1568, 320]" = torch.ops.aten.mm.default(view_543, permute_429);  permute_429 = None
    permute_430: "f32[320, 1568]" = torch.ops.aten.permute.default(view_543, [1, 0])
    mm_57: "f32[320, 320]" = torch.ops.aten.mm.default(permute_430, view_353);  permute_430 = view_353 = None
    permute_431: "f32[320, 320]" = torch.ops.aten.permute.default(mm_57, [1, 0]);  mm_57 = None
    sum_93: "f32[1, 320]" = torch.ops.aten.sum.dim_IntList(view_543, [0], True);  view_543 = None
    view_544: "f32[320]" = torch.ops.aten.view.default(sum_93, [320]);  sum_93 = None
    permute_432: "f32[320, 320]" = torch.ops.aten.permute.default(permute_431, [1, 0]);  permute_431 = None
    view_545: "f32[8, 196, 320]" = torch.ops.aten.view.default(mm_56, [8, 196, 320]);  mm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_546: "f32[8, 196, 5, 64]" = torch.ops.aten.view.default(view_545, [8, 196, 5, 64]);  view_545 = None
    permute_433: "f32[8, 5, 196, 64]" = torch.ops.aten.permute.default(view_546, [0, 2, 1, 3]);  view_546 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_backward_5 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_433, permute_235, getitem_274, getitem_275, None, alias_33, getitem_277, getitem_278, getitem_279, 0.0, [True, True, True, False]);  permute_433 = permute_235 = getitem_274 = getitem_275 = alias_33 = getitem_277 = getitem_278 = getitem_279 = None
    getitem_372: "f32[8, 5, 196, 64]" = _scaled_dot_product_efficient_attention_backward_5[0]
    getitem_373: "f32[8, 5, 49, 64]" = _scaled_dot_product_efficient_attention_backward_5[1]
    getitem_374: "f32[8, 5, 49, 64]" = _scaled_dot_product_efficient_attention_backward_5[2];  _scaled_dot_product_efficient_attention_backward_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    cat_5: "f32[16, 5, 49, 64]" = torch.ops.aten.cat.default([getitem_373, getitem_374]);  getitem_373 = getitem_374 = None
    view_547: "f32[2, 8, 5, 49, 64]" = torch.ops.aten.view.default(cat_5, [2, 8, 5, 49, 64]);  cat_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_434: "f32[8, 49, 2, 5, 64]" = torch.ops.aten.permute.default(view_547, [1, 3, 0, 2, 4]);  view_547 = None
    clone_193: "f32[8, 49, 2, 5, 64]" = torch.ops.aten.clone.default(permute_434, memory_format = torch.contiguous_format);  permute_434 = None
    view_548: "f32[8, 49, 640]" = torch.ops.aten.view.default(clone_193, [8, 49, 640]);  clone_193 = None
    view_549: "f32[392, 640]" = torch.ops.aten.view.default(view_548, [392, 640]);  view_548 = None
    mm_58: "f32[392, 320]" = torch.ops.aten.mm.default(view_549, permute_435);  permute_435 = None
    permute_436: "f32[640, 392]" = torch.ops.aten.permute.default(view_549, [1, 0])
    mm_59: "f32[640, 320]" = torch.ops.aten.mm.default(permute_436, view_349);  permute_436 = view_349 = None
    permute_437: "f32[320, 640]" = torch.ops.aten.permute.default(mm_59, [1, 0]);  mm_59 = None
    sum_94: "f32[1, 640]" = torch.ops.aten.sum.dim_IntList(view_549, [0], True);  view_549 = None
    view_550: "f32[640]" = torch.ops.aten.view.default(sum_94, [640]);  sum_94 = None
    permute_438: "f32[640, 320]" = torch.ops.aten.permute.default(permute_437, [1, 0]);  permute_437 = None
    view_551: "f32[8, 49, 320]" = torch.ops.aten.view.default(mm_58, [8, 49, 320]);  mm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    mul_404: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(view_551, primals_421);  primals_421 = None
    mul_405: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_404, 320)
    sum_95: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_404, [2], True)
    mul_406: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_404, mul_206);  mul_404 = None
    sum_96: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_406, [2], True);  mul_406 = None
    mul_407: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_206, sum_96);  sum_96 = None
    sub_132: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(mul_405, sum_95);  mul_405 = sum_95 = None
    sub_133: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(sub_132, mul_407);  sub_132 = mul_407 = None
    mul_408: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(div_16, sub_133);  div_16 = sub_133 = None
    mul_409: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(view_551, mul_206);  mul_206 = None
    sum_97: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_409, [0, 1]);  mul_409 = None
    sum_98: "f32[320]" = torch.ops.aten.sum.dim_IntList(view_551, [0, 1]);  view_551 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    permute_439: "f32[8, 320, 49]" = torch.ops.aten.permute.default(mul_408, [0, 2, 1]);  mul_408 = None
    view_552: "f32[8, 320, 7, 7]" = torch.ops.aten.view.default(permute_439, [8, 320, 7, 7]);  permute_439 = None
    sum_99: "f32[320]" = torch.ops.aten.sum.dim_IntList(view_552, [0, 2, 3])
    convolution_backward_4 = torch.ops.aten.convolution_backward.default(view_552, view_347, primals_419, [320], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_552 = view_347 = primals_419 = None
    getitem_376: "f32[8, 320, 14, 14]" = convolution_backward_4[0]
    getitem_377: "f32[320, 320, 2, 2]" = convolution_backward_4[1];  convolution_backward_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    view_553: "f32[8, 320, 196]" = torch.ops.aten.view.default(getitem_376, [8, 320, 196]);  getitem_376 = None
    permute_440: "f32[8, 196, 320]" = torch.ops.aten.permute.default(view_553, [0, 2, 1]);  view_553 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_441: "f32[8, 196, 5, 64]" = torch.ops.aten.permute.default(getitem_372, [0, 2, 1, 3]);  getitem_372 = None
    view_554: "f32[8, 196, 320]" = torch.ops.aten.view.default(permute_441, [8, 196, 320]);  permute_441 = None
    view_555: "f32[1568, 320]" = torch.ops.aten.view.default(view_554, [1568, 320]);  view_554 = None
    mm_60: "f32[1568, 320]" = torch.ops.aten.mm.default(view_555, permute_442);  permute_442 = None
    permute_443: "f32[320, 1568]" = torch.ops.aten.permute.default(view_555, [1, 0])
    mm_61: "f32[320, 320]" = torch.ops.aten.mm.default(permute_443, view_344);  permute_443 = view_344 = None
    permute_444: "f32[320, 320]" = torch.ops.aten.permute.default(mm_61, [1, 0]);  mm_61 = None
    sum_100: "f32[1, 320]" = torch.ops.aten.sum.dim_IntList(view_555, [0], True);  view_555 = None
    view_556: "f32[320]" = torch.ops.aten.view.default(sum_100, [320]);  sum_100 = None
    permute_445: "f32[320, 320]" = torch.ops.aten.permute.default(permute_444, [1, 0]);  permute_444 = None
    view_557: "f32[8, 196, 320]" = torch.ops.aten.view.default(mm_60, [8, 196, 320]);  mm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    add_289: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(permute_440, view_557);  permute_440 = view_557 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    clone_195: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_289, memory_format = torch.contiguous_format);  add_289 = None
    mul_411: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(clone_195, primals_415);  primals_415 = None
    mul_412: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_411, 320)
    sum_101: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_411, [2], True)
    mul_413: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_411, mul_204);  mul_411 = None
    sum_102: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_413, [2], True);  mul_413 = None
    mul_414: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_204, sum_102);  sum_102 = None
    sub_135: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(mul_412, sum_101);  mul_412 = sum_101 = None
    sub_136: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(sub_135, mul_414);  sub_135 = mul_414 = None
    mul_415: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(div_17, sub_136);  div_17 = sub_136 = None
    mul_416: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(clone_195, mul_204);  mul_204 = None
    sum_103: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_416, [0, 1]);  mul_416 = None
    sum_104: "f32[320]" = torch.ops.aten.sum.dim_IntList(clone_195, [0, 1]);  clone_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_290: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_288, mul_415);  add_288 = mul_415 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    clone_197: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_290, memory_format = torch.contiguous_format)
    view_558: "f32[1568, 320]" = torch.ops.aten.view.default(clone_197, [1568, 320]);  clone_197 = None
    mm_62: "f32[1568, 1280]" = torch.ops.aten.mm.default(view_558, permute_446);  permute_446 = None
    permute_447: "f32[320, 1568]" = torch.ops.aten.permute.default(view_558, [1, 0])
    mm_63: "f32[320, 1280]" = torch.ops.aten.mm.default(permute_447, view_342);  permute_447 = view_342 = None
    permute_448: "f32[1280, 320]" = torch.ops.aten.permute.default(mm_63, [1, 0]);  mm_63 = None
    sum_105: "f32[1, 320]" = torch.ops.aten.sum.dim_IntList(view_558, [0], True);  view_558 = None
    view_559: "f32[320]" = torch.ops.aten.view.default(sum_105, [320]);  sum_105 = None
    permute_449: "f32[320, 1280]" = torch.ops.aten.permute.default(permute_448, [1, 0]);  permute_448 = None
    view_560: "f32[8, 196, 1280]" = torch.ops.aten.view.default(mm_62, [8, 196, 1280]);  mm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_418: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(add_205, 0.5);  add_205 = None
    mul_419: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_341, view_341)
    mul_420: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(mul_419, -0.5);  mul_419 = None
    exp_6: "f32[8, 196, 1280]" = torch.ops.aten.exp.default(mul_420);  mul_420 = None
    mul_421: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(exp_6, 0.3989422804014327);  exp_6 = None
    mul_422: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_341, mul_421);  view_341 = mul_421 = None
    add_292: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(mul_418, mul_422);  mul_418 = mul_422 = None
    mul_423: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_560, add_292);  view_560 = add_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_561: "f32[1568, 1280]" = torch.ops.aten.view.default(mul_423, [1568, 1280]);  mul_423 = None
    mm_64: "f32[1568, 320]" = torch.ops.aten.mm.default(view_561, permute_450);  permute_450 = None
    permute_451: "f32[1280, 1568]" = torch.ops.aten.permute.default(view_561, [1, 0])
    mm_65: "f32[1280, 320]" = torch.ops.aten.mm.default(permute_451, view_340);  permute_451 = view_340 = None
    permute_452: "f32[320, 1280]" = torch.ops.aten.permute.default(mm_65, [1, 0]);  mm_65 = None
    sum_106: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_561, [0], True);  view_561 = None
    view_562: "f32[1280]" = torch.ops.aten.view.default(sum_106, [1280]);  sum_106 = None
    permute_453: "f32[1280, 320]" = torch.ops.aten.permute.default(permute_452, [1, 0]);  permute_452 = None
    view_563: "f32[8, 196, 320]" = torch.ops.aten.view.default(mm_64, [8, 196, 320]);  mm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    mul_425: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(view_563, primals_409);  primals_409 = None
    mul_426: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_425, 320)
    sum_107: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_425, [2], True)
    mul_427: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_425, mul_199);  mul_425 = None
    sum_108: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_427, [2], True);  mul_427 = None
    mul_428: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_199, sum_108);  sum_108 = None
    sub_138: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(mul_426, sum_107);  mul_426 = sum_107 = None
    sub_139: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(sub_138, mul_428);  sub_138 = mul_428 = None
    mul_429: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(div_18, sub_139);  div_18 = sub_139 = None
    mul_430: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(view_563, mul_199);  mul_199 = None
    sum_109: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_430, [0, 1]);  mul_430 = None
    sum_110: "f32[320]" = torch.ops.aten.sum.dim_IntList(view_563, [0, 1]);  view_563 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_293: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_290, mul_429);  add_290 = mul_429 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    clone_199: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_293, memory_format = torch.contiguous_format)
    view_564: "f32[1568, 320]" = torch.ops.aten.view.default(clone_199, [1568, 320]);  clone_199 = None
    mm_66: "f32[1568, 320]" = torch.ops.aten.mm.default(view_564, permute_454);  permute_454 = None
    permute_455: "f32[320, 1568]" = torch.ops.aten.permute.default(view_564, [1, 0])
    mm_67: "f32[320, 320]" = torch.ops.aten.mm.default(permute_455, view_338);  permute_455 = view_338 = None
    permute_456: "f32[320, 320]" = torch.ops.aten.permute.default(mm_67, [1, 0]);  mm_67 = None
    sum_111: "f32[1, 320]" = torch.ops.aten.sum.dim_IntList(view_564, [0], True);  view_564 = None
    view_565: "f32[320]" = torch.ops.aten.view.default(sum_111, [320]);  sum_111 = None
    permute_457: "f32[320, 320]" = torch.ops.aten.permute.default(permute_456, [1, 0]);  permute_456 = None
    view_566: "f32[8, 196, 320]" = torch.ops.aten.view.default(mm_66, [8, 196, 320]);  mm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_567: "f32[8, 196, 5, 64]" = torch.ops.aten.view.default(view_566, [8, 196, 5, 64]);  view_566 = None
    permute_458: "f32[8, 5, 196, 64]" = torch.ops.aten.permute.default(view_567, [0, 2, 1, 3]);  view_567 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_backward_6 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_458, permute_225, getitem_262, getitem_263, None, alias_34, getitem_265, getitem_266, getitem_267, 0.0, [True, True, True, False]);  permute_458 = permute_225 = getitem_262 = getitem_263 = alias_34 = getitem_265 = getitem_266 = getitem_267 = None
    getitem_379: "f32[8, 5, 196, 64]" = _scaled_dot_product_efficient_attention_backward_6[0]
    getitem_380: "f32[8, 5, 49, 64]" = _scaled_dot_product_efficient_attention_backward_6[1]
    getitem_381: "f32[8, 5, 49, 64]" = _scaled_dot_product_efficient_attention_backward_6[2];  _scaled_dot_product_efficient_attention_backward_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    cat_6: "f32[16, 5, 49, 64]" = torch.ops.aten.cat.default([getitem_380, getitem_381]);  getitem_380 = getitem_381 = None
    view_568: "f32[2, 8, 5, 49, 64]" = torch.ops.aten.view.default(cat_6, [2, 8, 5, 49, 64]);  cat_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_459: "f32[8, 49, 2, 5, 64]" = torch.ops.aten.permute.default(view_568, [1, 3, 0, 2, 4]);  view_568 = None
    clone_200: "f32[8, 49, 2, 5, 64]" = torch.ops.aten.clone.default(permute_459, memory_format = torch.contiguous_format);  permute_459 = None
    view_569: "f32[8, 49, 640]" = torch.ops.aten.view.default(clone_200, [8, 49, 640]);  clone_200 = None
    view_570: "f32[392, 640]" = torch.ops.aten.view.default(view_569, [392, 640]);  view_569 = None
    mm_68: "f32[392, 320]" = torch.ops.aten.mm.default(view_570, permute_460);  permute_460 = None
    permute_461: "f32[640, 392]" = torch.ops.aten.permute.default(view_570, [1, 0])
    mm_69: "f32[640, 320]" = torch.ops.aten.mm.default(permute_461, view_334);  permute_461 = view_334 = None
    permute_462: "f32[320, 640]" = torch.ops.aten.permute.default(mm_69, [1, 0]);  mm_69 = None
    sum_112: "f32[1, 640]" = torch.ops.aten.sum.dim_IntList(view_570, [0], True);  view_570 = None
    view_571: "f32[640]" = torch.ops.aten.view.default(sum_112, [640]);  sum_112 = None
    permute_463: "f32[640, 320]" = torch.ops.aten.permute.default(permute_462, [1, 0]);  permute_462 = None
    view_572: "f32[8, 49, 320]" = torch.ops.aten.view.default(mm_68, [8, 49, 320]);  mm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    mul_432: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(view_572, primals_403);  primals_403 = None
    mul_433: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_432, 320)
    sum_113: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_432, [2], True)
    mul_434: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_432, mul_197);  mul_432 = None
    sum_114: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_434, [2], True);  mul_434 = None
    mul_435: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_197, sum_114);  sum_114 = None
    sub_141: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(mul_433, sum_113);  mul_433 = sum_113 = None
    sub_142: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(sub_141, mul_435);  sub_141 = mul_435 = None
    mul_436: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(div_19, sub_142);  div_19 = sub_142 = None
    mul_437: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(view_572, mul_197);  mul_197 = None
    sum_115: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_437, [0, 1]);  mul_437 = None
    sum_116: "f32[320]" = torch.ops.aten.sum.dim_IntList(view_572, [0, 1]);  view_572 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    permute_464: "f32[8, 320, 49]" = torch.ops.aten.permute.default(mul_436, [0, 2, 1]);  mul_436 = None
    view_573: "f32[8, 320, 7, 7]" = torch.ops.aten.view.default(permute_464, [8, 320, 7, 7]);  permute_464 = None
    sum_117: "f32[320]" = torch.ops.aten.sum.dim_IntList(view_573, [0, 2, 3])
    convolution_backward_5 = torch.ops.aten.convolution_backward.default(view_573, view_332, primals_401, [320], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_573 = view_332 = primals_401 = None
    getitem_383: "f32[8, 320, 14, 14]" = convolution_backward_5[0]
    getitem_384: "f32[320, 320, 2, 2]" = convolution_backward_5[1];  convolution_backward_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    view_574: "f32[8, 320, 196]" = torch.ops.aten.view.default(getitem_383, [8, 320, 196]);  getitem_383 = None
    permute_465: "f32[8, 196, 320]" = torch.ops.aten.permute.default(view_574, [0, 2, 1]);  view_574 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_466: "f32[8, 196, 5, 64]" = torch.ops.aten.permute.default(getitem_379, [0, 2, 1, 3]);  getitem_379 = None
    view_575: "f32[8, 196, 320]" = torch.ops.aten.view.default(permute_466, [8, 196, 320]);  permute_466 = None
    view_576: "f32[1568, 320]" = torch.ops.aten.view.default(view_575, [1568, 320]);  view_575 = None
    mm_70: "f32[1568, 320]" = torch.ops.aten.mm.default(view_576, permute_467);  permute_467 = None
    permute_468: "f32[320, 1568]" = torch.ops.aten.permute.default(view_576, [1, 0])
    mm_71: "f32[320, 320]" = torch.ops.aten.mm.default(permute_468, view_329);  permute_468 = view_329 = None
    permute_469: "f32[320, 320]" = torch.ops.aten.permute.default(mm_71, [1, 0]);  mm_71 = None
    sum_118: "f32[1, 320]" = torch.ops.aten.sum.dim_IntList(view_576, [0], True);  view_576 = None
    view_577: "f32[320]" = torch.ops.aten.view.default(sum_118, [320]);  sum_118 = None
    permute_470: "f32[320, 320]" = torch.ops.aten.permute.default(permute_469, [1, 0]);  permute_469 = None
    view_578: "f32[8, 196, 320]" = torch.ops.aten.view.default(mm_70, [8, 196, 320]);  mm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    add_294: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(permute_465, view_578);  permute_465 = view_578 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    clone_202: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_294, memory_format = torch.contiguous_format);  add_294 = None
    mul_439: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(clone_202, primals_397);  primals_397 = None
    mul_440: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_439, 320)
    sum_119: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_439, [2], True)
    mul_441: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_439, mul_195);  mul_439 = None
    sum_120: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_441, [2], True);  mul_441 = None
    mul_442: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_195, sum_120);  sum_120 = None
    sub_144: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(mul_440, sum_119);  mul_440 = sum_119 = None
    sub_145: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(sub_144, mul_442);  sub_144 = mul_442 = None
    mul_443: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(div_20, sub_145);  div_20 = sub_145 = None
    mul_444: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(clone_202, mul_195);  mul_195 = None
    sum_121: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_444, [0, 1]);  mul_444 = None
    sum_122: "f32[320]" = torch.ops.aten.sum.dim_IntList(clone_202, [0, 1]);  clone_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_295: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_293, mul_443);  add_293 = mul_443 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    clone_204: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_295, memory_format = torch.contiguous_format)
    view_579: "f32[1568, 320]" = torch.ops.aten.view.default(clone_204, [1568, 320]);  clone_204 = None
    mm_72: "f32[1568, 1280]" = torch.ops.aten.mm.default(view_579, permute_471);  permute_471 = None
    permute_472: "f32[320, 1568]" = torch.ops.aten.permute.default(view_579, [1, 0])
    mm_73: "f32[320, 1280]" = torch.ops.aten.mm.default(permute_472, view_327);  permute_472 = view_327 = None
    permute_473: "f32[1280, 320]" = torch.ops.aten.permute.default(mm_73, [1, 0]);  mm_73 = None
    sum_123: "f32[1, 320]" = torch.ops.aten.sum.dim_IntList(view_579, [0], True);  view_579 = None
    view_580: "f32[320]" = torch.ops.aten.view.default(sum_123, [320]);  sum_123 = None
    permute_474: "f32[320, 1280]" = torch.ops.aten.permute.default(permute_473, [1, 0]);  permute_473 = None
    view_581: "f32[8, 196, 1280]" = torch.ops.aten.view.default(mm_72, [8, 196, 1280]);  mm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_446: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(add_196, 0.5);  add_196 = None
    mul_447: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_326, view_326)
    mul_448: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(mul_447, -0.5);  mul_447 = None
    exp_7: "f32[8, 196, 1280]" = torch.ops.aten.exp.default(mul_448);  mul_448 = None
    mul_449: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(exp_7, 0.3989422804014327);  exp_7 = None
    mul_450: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_326, mul_449);  view_326 = mul_449 = None
    add_297: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(mul_446, mul_450);  mul_446 = mul_450 = None
    mul_451: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_581, add_297);  view_581 = add_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_582: "f32[1568, 1280]" = torch.ops.aten.view.default(mul_451, [1568, 1280]);  mul_451 = None
    mm_74: "f32[1568, 320]" = torch.ops.aten.mm.default(view_582, permute_475);  permute_475 = None
    permute_476: "f32[1280, 1568]" = torch.ops.aten.permute.default(view_582, [1, 0])
    mm_75: "f32[1280, 320]" = torch.ops.aten.mm.default(permute_476, view_325);  permute_476 = view_325 = None
    permute_477: "f32[320, 1280]" = torch.ops.aten.permute.default(mm_75, [1, 0]);  mm_75 = None
    sum_124: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_582, [0], True);  view_582 = None
    view_583: "f32[1280]" = torch.ops.aten.view.default(sum_124, [1280]);  sum_124 = None
    permute_478: "f32[1280, 320]" = torch.ops.aten.permute.default(permute_477, [1, 0]);  permute_477 = None
    view_584: "f32[8, 196, 320]" = torch.ops.aten.view.default(mm_74, [8, 196, 320]);  mm_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    mul_453: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(view_584, primals_391);  primals_391 = None
    mul_454: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_453, 320)
    sum_125: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_453, [2], True)
    mul_455: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_453, mul_190);  mul_453 = None
    sum_126: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_455, [2], True);  mul_455 = None
    mul_456: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_190, sum_126);  sum_126 = None
    sub_147: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(mul_454, sum_125);  mul_454 = sum_125 = None
    sub_148: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(sub_147, mul_456);  sub_147 = mul_456 = None
    mul_457: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(div_21, sub_148);  div_21 = sub_148 = None
    mul_458: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(view_584, mul_190);  mul_190 = None
    sum_127: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_458, [0, 1]);  mul_458 = None
    sum_128: "f32[320]" = torch.ops.aten.sum.dim_IntList(view_584, [0, 1]);  view_584 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_298: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_295, mul_457);  add_295 = mul_457 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    clone_206: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_298, memory_format = torch.contiguous_format)
    view_585: "f32[1568, 320]" = torch.ops.aten.view.default(clone_206, [1568, 320]);  clone_206 = None
    mm_76: "f32[1568, 320]" = torch.ops.aten.mm.default(view_585, permute_479);  permute_479 = None
    permute_480: "f32[320, 1568]" = torch.ops.aten.permute.default(view_585, [1, 0])
    mm_77: "f32[320, 320]" = torch.ops.aten.mm.default(permute_480, view_323);  permute_480 = view_323 = None
    permute_481: "f32[320, 320]" = torch.ops.aten.permute.default(mm_77, [1, 0]);  mm_77 = None
    sum_129: "f32[1, 320]" = torch.ops.aten.sum.dim_IntList(view_585, [0], True);  view_585 = None
    view_586: "f32[320]" = torch.ops.aten.view.default(sum_129, [320]);  sum_129 = None
    permute_482: "f32[320, 320]" = torch.ops.aten.permute.default(permute_481, [1, 0]);  permute_481 = None
    view_587: "f32[8, 196, 320]" = torch.ops.aten.view.default(mm_76, [8, 196, 320]);  mm_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_588: "f32[8, 196, 5, 64]" = torch.ops.aten.view.default(view_587, [8, 196, 5, 64]);  view_587 = None
    permute_483: "f32[8, 5, 196, 64]" = torch.ops.aten.permute.default(view_588, [0, 2, 1, 3]);  view_588 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_backward_7 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_483, permute_215, getitem_250, getitem_251, None, alias_35, getitem_253, getitem_254, getitem_255, 0.0, [True, True, True, False]);  permute_483 = permute_215 = getitem_250 = getitem_251 = alias_35 = getitem_253 = getitem_254 = getitem_255 = None
    getitem_386: "f32[8, 5, 196, 64]" = _scaled_dot_product_efficient_attention_backward_7[0]
    getitem_387: "f32[8, 5, 49, 64]" = _scaled_dot_product_efficient_attention_backward_7[1]
    getitem_388: "f32[8, 5, 49, 64]" = _scaled_dot_product_efficient_attention_backward_7[2];  _scaled_dot_product_efficient_attention_backward_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    cat_7: "f32[16, 5, 49, 64]" = torch.ops.aten.cat.default([getitem_387, getitem_388]);  getitem_387 = getitem_388 = None
    view_589: "f32[2, 8, 5, 49, 64]" = torch.ops.aten.view.default(cat_7, [2, 8, 5, 49, 64]);  cat_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_484: "f32[8, 49, 2, 5, 64]" = torch.ops.aten.permute.default(view_589, [1, 3, 0, 2, 4]);  view_589 = None
    clone_207: "f32[8, 49, 2, 5, 64]" = torch.ops.aten.clone.default(permute_484, memory_format = torch.contiguous_format);  permute_484 = None
    view_590: "f32[8, 49, 640]" = torch.ops.aten.view.default(clone_207, [8, 49, 640]);  clone_207 = None
    view_591: "f32[392, 640]" = torch.ops.aten.view.default(view_590, [392, 640]);  view_590 = None
    mm_78: "f32[392, 320]" = torch.ops.aten.mm.default(view_591, permute_485);  permute_485 = None
    permute_486: "f32[640, 392]" = torch.ops.aten.permute.default(view_591, [1, 0])
    mm_79: "f32[640, 320]" = torch.ops.aten.mm.default(permute_486, view_319);  permute_486 = view_319 = None
    permute_487: "f32[320, 640]" = torch.ops.aten.permute.default(mm_79, [1, 0]);  mm_79 = None
    sum_130: "f32[1, 640]" = torch.ops.aten.sum.dim_IntList(view_591, [0], True);  view_591 = None
    view_592: "f32[640]" = torch.ops.aten.view.default(sum_130, [640]);  sum_130 = None
    permute_488: "f32[640, 320]" = torch.ops.aten.permute.default(permute_487, [1, 0]);  permute_487 = None
    view_593: "f32[8, 49, 320]" = torch.ops.aten.view.default(mm_78, [8, 49, 320]);  mm_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    mul_460: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(view_593, primals_385);  primals_385 = None
    mul_461: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_460, 320)
    sum_131: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_460, [2], True)
    mul_462: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_460, mul_188);  mul_460 = None
    sum_132: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_462, [2], True);  mul_462 = None
    mul_463: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_188, sum_132);  sum_132 = None
    sub_150: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(mul_461, sum_131);  mul_461 = sum_131 = None
    sub_151: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(sub_150, mul_463);  sub_150 = mul_463 = None
    mul_464: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(div_22, sub_151);  div_22 = sub_151 = None
    mul_465: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(view_593, mul_188);  mul_188 = None
    sum_133: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_465, [0, 1]);  mul_465 = None
    sum_134: "f32[320]" = torch.ops.aten.sum.dim_IntList(view_593, [0, 1]);  view_593 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    permute_489: "f32[8, 320, 49]" = torch.ops.aten.permute.default(mul_464, [0, 2, 1]);  mul_464 = None
    view_594: "f32[8, 320, 7, 7]" = torch.ops.aten.view.default(permute_489, [8, 320, 7, 7]);  permute_489 = None
    sum_135: "f32[320]" = torch.ops.aten.sum.dim_IntList(view_594, [0, 2, 3])
    convolution_backward_6 = torch.ops.aten.convolution_backward.default(view_594, view_317, primals_383, [320], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_594 = view_317 = primals_383 = None
    getitem_390: "f32[8, 320, 14, 14]" = convolution_backward_6[0]
    getitem_391: "f32[320, 320, 2, 2]" = convolution_backward_6[1];  convolution_backward_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    view_595: "f32[8, 320, 196]" = torch.ops.aten.view.default(getitem_390, [8, 320, 196]);  getitem_390 = None
    permute_490: "f32[8, 196, 320]" = torch.ops.aten.permute.default(view_595, [0, 2, 1]);  view_595 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_491: "f32[8, 196, 5, 64]" = torch.ops.aten.permute.default(getitem_386, [0, 2, 1, 3]);  getitem_386 = None
    view_596: "f32[8, 196, 320]" = torch.ops.aten.view.default(permute_491, [8, 196, 320]);  permute_491 = None
    view_597: "f32[1568, 320]" = torch.ops.aten.view.default(view_596, [1568, 320]);  view_596 = None
    mm_80: "f32[1568, 320]" = torch.ops.aten.mm.default(view_597, permute_492);  permute_492 = None
    permute_493: "f32[320, 1568]" = torch.ops.aten.permute.default(view_597, [1, 0])
    mm_81: "f32[320, 320]" = torch.ops.aten.mm.default(permute_493, view_314);  permute_493 = view_314 = None
    permute_494: "f32[320, 320]" = torch.ops.aten.permute.default(mm_81, [1, 0]);  mm_81 = None
    sum_136: "f32[1, 320]" = torch.ops.aten.sum.dim_IntList(view_597, [0], True);  view_597 = None
    view_598: "f32[320]" = torch.ops.aten.view.default(sum_136, [320]);  sum_136 = None
    permute_495: "f32[320, 320]" = torch.ops.aten.permute.default(permute_494, [1, 0]);  permute_494 = None
    view_599: "f32[8, 196, 320]" = torch.ops.aten.view.default(mm_80, [8, 196, 320]);  mm_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    add_299: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(permute_490, view_599);  permute_490 = view_599 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    clone_209: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_299, memory_format = torch.contiguous_format);  add_299 = None
    mul_467: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(clone_209, primals_379);  primals_379 = None
    mul_468: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_467, 320)
    sum_137: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_467, [2], True)
    mul_469: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_467, mul_186);  mul_467 = None
    sum_138: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_469, [2], True);  mul_469 = None
    mul_470: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_186, sum_138);  sum_138 = None
    sub_153: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(mul_468, sum_137);  mul_468 = sum_137 = None
    sub_154: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(sub_153, mul_470);  sub_153 = mul_470 = None
    mul_471: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(div_23, sub_154);  div_23 = sub_154 = None
    mul_472: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(clone_209, mul_186);  mul_186 = None
    sum_139: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_472, [0, 1]);  mul_472 = None
    sum_140: "f32[320]" = torch.ops.aten.sum.dim_IntList(clone_209, [0, 1]);  clone_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_300: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_298, mul_471);  add_298 = mul_471 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    clone_211: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_300, memory_format = torch.contiguous_format)
    view_600: "f32[1568, 320]" = torch.ops.aten.view.default(clone_211, [1568, 320]);  clone_211 = None
    mm_82: "f32[1568, 1280]" = torch.ops.aten.mm.default(view_600, permute_496);  permute_496 = None
    permute_497: "f32[320, 1568]" = torch.ops.aten.permute.default(view_600, [1, 0])
    mm_83: "f32[320, 1280]" = torch.ops.aten.mm.default(permute_497, view_312);  permute_497 = view_312 = None
    permute_498: "f32[1280, 320]" = torch.ops.aten.permute.default(mm_83, [1, 0]);  mm_83 = None
    sum_141: "f32[1, 320]" = torch.ops.aten.sum.dim_IntList(view_600, [0], True);  view_600 = None
    view_601: "f32[320]" = torch.ops.aten.view.default(sum_141, [320]);  sum_141 = None
    permute_499: "f32[320, 1280]" = torch.ops.aten.permute.default(permute_498, [1, 0]);  permute_498 = None
    view_602: "f32[8, 196, 1280]" = torch.ops.aten.view.default(mm_82, [8, 196, 1280]);  mm_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_474: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(add_187, 0.5);  add_187 = None
    mul_475: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_311, view_311)
    mul_476: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(mul_475, -0.5);  mul_475 = None
    exp_8: "f32[8, 196, 1280]" = torch.ops.aten.exp.default(mul_476);  mul_476 = None
    mul_477: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(exp_8, 0.3989422804014327);  exp_8 = None
    mul_478: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_311, mul_477);  view_311 = mul_477 = None
    add_302: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(mul_474, mul_478);  mul_474 = mul_478 = None
    mul_479: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_602, add_302);  view_602 = add_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_603: "f32[1568, 1280]" = torch.ops.aten.view.default(mul_479, [1568, 1280]);  mul_479 = None
    mm_84: "f32[1568, 320]" = torch.ops.aten.mm.default(view_603, permute_500);  permute_500 = None
    permute_501: "f32[1280, 1568]" = torch.ops.aten.permute.default(view_603, [1, 0])
    mm_85: "f32[1280, 320]" = torch.ops.aten.mm.default(permute_501, view_310);  permute_501 = view_310 = None
    permute_502: "f32[320, 1280]" = torch.ops.aten.permute.default(mm_85, [1, 0]);  mm_85 = None
    sum_142: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_603, [0], True);  view_603 = None
    view_604: "f32[1280]" = torch.ops.aten.view.default(sum_142, [1280]);  sum_142 = None
    permute_503: "f32[1280, 320]" = torch.ops.aten.permute.default(permute_502, [1, 0]);  permute_502 = None
    view_605: "f32[8, 196, 320]" = torch.ops.aten.view.default(mm_84, [8, 196, 320]);  mm_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    mul_481: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(view_605, primals_373);  primals_373 = None
    mul_482: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_481, 320)
    sum_143: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_481, [2], True)
    mul_483: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_481, mul_181);  mul_481 = None
    sum_144: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_483, [2], True);  mul_483 = None
    mul_484: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_181, sum_144);  sum_144 = None
    sub_156: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(mul_482, sum_143);  mul_482 = sum_143 = None
    sub_157: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(sub_156, mul_484);  sub_156 = mul_484 = None
    mul_485: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(div_24, sub_157);  div_24 = sub_157 = None
    mul_486: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(view_605, mul_181);  mul_181 = None
    sum_145: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_486, [0, 1]);  mul_486 = None
    sum_146: "f32[320]" = torch.ops.aten.sum.dim_IntList(view_605, [0, 1]);  view_605 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_303: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_300, mul_485);  add_300 = mul_485 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    clone_213: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_303, memory_format = torch.contiguous_format)
    view_606: "f32[1568, 320]" = torch.ops.aten.view.default(clone_213, [1568, 320]);  clone_213 = None
    mm_86: "f32[1568, 320]" = torch.ops.aten.mm.default(view_606, permute_504);  permute_504 = None
    permute_505: "f32[320, 1568]" = torch.ops.aten.permute.default(view_606, [1, 0])
    mm_87: "f32[320, 320]" = torch.ops.aten.mm.default(permute_505, view_308);  permute_505 = view_308 = None
    permute_506: "f32[320, 320]" = torch.ops.aten.permute.default(mm_87, [1, 0]);  mm_87 = None
    sum_147: "f32[1, 320]" = torch.ops.aten.sum.dim_IntList(view_606, [0], True);  view_606 = None
    view_607: "f32[320]" = torch.ops.aten.view.default(sum_147, [320]);  sum_147 = None
    permute_507: "f32[320, 320]" = torch.ops.aten.permute.default(permute_506, [1, 0]);  permute_506 = None
    view_608: "f32[8, 196, 320]" = torch.ops.aten.view.default(mm_86, [8, 196, 320]);  mm_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_609: "f32[8, 196, 5, 64]" = torch.ops.aten.view.default(view_608, [8, 196, 5, 64]);  view_608 = None
    permute_508: "f32[8, 5, 196, 64]" = torch.ops.aten.permute.default(view_609, [0, 2, 1, 3]);  view_609 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_backward_8 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_508, permute_205, getitem_238, getitem_239, None, alias_36, getitem_241, getitem_242, getitem_243, 0.0, [True, True, True, False]);  permute_508 = permute_205 = getitem_238 = getitem_239 = alias_36 = getitem_241 = getitem_242 = getitem_243 = None
    getitem_393: "f32[8, 5, 196, 64]" = _scaled_dot_product_efficient_attention_backward_8[0]
    getitem_394: "f32[8, 5, 49, 64]" = _scaled_dot_product_efficient_attention_backward_8[1]
    getitem_395: "f32[8, 5, 49, 64]" = _scaled_dot_product_efficient_attention_backward_8[2];  _scaled_dot_product_efficient_attention_backward_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    cat_8: "f32[16, 5, 49, 64]" = torch.ops.aten.cat.default([getitem_394, getitem_395]);  getitem_394 = getitem_395 = None
    view_610: "f32[2, 8, 5, 49, 64]" = torch.ops.aten.view.default(cat_8, [2, 8, 5, 49, 64]);  cat_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_509: "f32[8, 49, 2, 5, 64]" = torch.ops.aten.permute.default(view_610, [1, 3, 0, 2, 4]);  view_610 = None
    clone_214: "f32[8, 49, 2, 5, 64]" = torch.ops.aten.clone.default(permute_509, memory_format = torch.contiguous_format);  permute_509 = None
    view_611: "f32[8, 49, 640]" = torch.ops.aten.view.default(clone_214, [8, 49, 640]);  clone_214 = None
    view_612: "f32[392, 640]" = torch.ops.aten.view.default(view_611, [392, 640]);  view_611 = None
    mm_88: "f32[392, 320]" = torch.ops.aten.mm.default(view_612, permute_510);  permute_510 = None
    permute_511: "f32[640, 392]" = torch.ops.aten.permute.default(view_612, [1, 0])
    mm_89: "f32[640, 320]" = torch.ops.aten.mm.default(permute_511, view_304);  permute_511 = view_304 = None
    permute_512: "f32[320, 640]" = torch.ops.aten.permute.default(mm_89, [1, 0]);  mm_89 = None
    sum_148: "f32[1, 640]" = torch.ops.aten.sum.dim_IntList(view_612, [0], True);  view_612 = None
    view_613: "f32[640]" = torch.ops.aten.view.default(sum_148, [640]);  sum_148 = None
    permute_513: "f32[640, 320]" = torch.ops.aten.permute.default(permute_512, [1, 0]);  permute_512 = None
    view_614: "f32[8, 49, 320]" = torch.ops.aten.view.default(mm_88, [8, 49, 320]);  mm_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    mul_488: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(view_614, primals_367);  primals_367 = None
    mul_489: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_488, 320)
    sum_149: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_488, [2], True)
    mul_490: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_488, mul_179);  mul_488 = None
    sum_150: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_490, [2], True);  mul_490 = None
    mul_491: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_179, sum_150);  sum_150 = None
    sub_159: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(mul_489, sum_149);  mul_489 = sum_149 = None
    sub_160: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(sub_159, mul_491);  sub_159 = mul_491 = None
    mul_492: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(div_25, sub_160);  div_25 = sub_160 = None
    mul_493: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(view_614, mul_179);  mul_179 = None
    sum_151: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_493, [0, 1]);  mul_493 = None
    sum_152: "f32[320]" = torch.ops.aten.sum.dim_IntList(view_614, [0, 1]);  view_614 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    permute_514: "f32[8, 320, 49]" = torch.ops.aten.permute.default(mul_492, [0, 2, 1]);  mul_492 = None
    view_615: "f32[8, 320, 7, 7]" = torch.ops.aten.view.default(permute_514, [8, 320, 7, 7]);  permute_514 = None
    sum_153: "f32[320]" = torch.ops.aten.sum.dim_IntList(view_615, [0, 2, 3])
    convolution_backward_7 = torch.ops.aten.convolution_backward.default(view_615, view_302, primals_365, [320], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_615 = view_302 = primals_365 = None
    getitem_397: "f32[8, 320, 14, 14]" = convolution_backward_7[0]
    getitem_398: "f32[320, 320, 2, 2]" = convolution_backward_7[1];  convolution_backward_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    view_616: "f32[8, 320, 196]" = torch.ops.aten.view.default(getitem_397, [8, 320, 196]);  getitem_397 = None
    permute_515: "f32[8, 196, 320]" = torch.ops.aten.permute.default(view_616, [0, 2, 1]);  view_616 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_516: "f32[8, 196, 5, 64]" = torch.ops.aten.permute.default(getitem_393, [0, 2, 1, 3]);  getitem_393 = None
    view_617: "f32[8, 196, 320]" = torch.ops.aten.view.default(permute_516, [8, 196, 320]);  permute_516 = None
    view_618: "f32[1568, 320]" = torch.ops.aten.view.default(view_617, [1568, 320]);  view_617 = None
    mm_90: "f32[1568, 320]" = torch.ops.aten.mm.default(view_618, permute_517);  permute_517 = None
    permute_518: "f32[320, 1568]" = torch.ops.aten.permute.default(view_618, [1, 0])
    mm_91: "f32[320, 320]" = torch.ops.aten.mm.default(permute_518, view_299);  permute_518 = view_299 = None
    permute_519: "f32[320, 320]" = torch.ops.aten.permute.default(mm_91, [1, 0]);  mm_91 = None
    sum_154: "f32[1, 320]" = torch.ops.aten.sum.dim_IntList(view_618, [0], True);  view_618 = None
    view_619: "f32[320]" = torch.ops.aten.view.default(sum_154, [320]);  sum_154 = None
    permute_520: "f32[320, 320]" = torch.ops.aten.permute.default(permute_519, [1, 0]);  permute_519 = None
    view_620: "f32[8, 196, 320]" = torch.ops.aten.view.default(mm_90, [8, 196, 320]);  mm_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    add_304: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(permute_515, view_620);  permute_515 = view_620 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    clone_216: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_304, memory_format = torch.contiguous_format);  add_304 = None
    mul_495: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(clone_216, primals_361);  primals_361 = None
    mul_496: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_495, 320)
    sum_155: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_495, [2], True)
    mul_497: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_495, mul_177);  mul_495 = None
    sum_156: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_497, [2], True);  mul_497 = None
    mul_498: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_177, sum_156);  sum_156 = None
    sub_162: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(mul_496, sum_155);  mul_496 = sum_155 = None
    sub_163: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(sub_162, mul_498);  sub_162 = mul_498 = None
    mul_499: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(div_26, sub_163);  div_26 = sub_163 = None
    mul_500: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(clone_216, mul_177);  mul_177 = None
    sum_157: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_500, [0, 1]);  mul_500 = None
    sum_158: "f32[320]" = torch.ops.aten.sum.dim_IntList(clone_216, [0, 1]);  clone_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_305: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_303, mul_499);  add_303 = mul_499 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    clone_218: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_305, memory_format = torch.contiguous_format)
    view_621: "f32[1568, 320]" = torch.ops.aten.view.default(clone_218, [1568, 320]);  clone_218 = None
    mm_92: "f32[1568, 1280]" = torch.ops.aten.mm.default(view_621, permute_521);  permute_521 = None
    permute_522: "f32[320, 1568]" = torch.ops.aten.permute.default(view_621, [1, 0])
    mm_93: "f32[320, 1280]" = torch.ops.aten.mm.default(permute_522, view_297);  permute_522 = view_297 = None
    permute_523: "f32[1280, 320]" = torch.ops.aten.permute.default(mm_93, [1, 0]);  mm_93 = None
    sum_159: "f32[1, 320]" = torch.ops.aten.sum.dim_IntList(view_621, [0], True);  view_621 = None
    view_622: "f32[320]" = torch.ops.aten.view.default(sum_159, [320]);  sum_159 = None
    permute_524: "f32[320, 1280]" = torch.ops.aten.permute.default(permute_523, [1, 0]);  permute_523 = None
    view_623: "f32[8, 196, 1280]" = torch.ops.aten.view.default(mm_92, [8, 196, 1280]);  mm_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_502: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(add_178, 0.5);  add_178 = None
    mul_503: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_296, view_296)
    mul_504: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(mul_503, -0.5);  mul_503 = None
    exp_9: "f32[8, 196, 1280]" = torch.ops.aten.exp.default(mul_504);  mul_504 = None
    mul_505: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(exp_9, 0.3989422804014327);  exp_9 = None
    mul_506: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_296, mul_505);  view_296 = mul_505 = None
    add_307: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(mul_502, mul_506);  mul_502 = mul_506 = None
    mul_507: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_623, add_307);  view_623 = add_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_624: "f32[1568, 1280]" = torch.ops.aten.view.default(mul_507, [1568, 1280]);  mul_507 = None
    mm_94: "f32[1568, 320]" = torch.ops.aten.mm.default(view_624, permute_525);  permute_525 = None
    permute_526: "f32[1280, 1568]" = torch.ops.aten.permute.default(view_624, [1, 0])
    mm_95: "f32[1280, 320]" = torch.ops.aten.mm.default(permute_526, view_295);  permute_526 = view_295 = None
    permute_527: "f32[320, 1280]" = torch.ops.aten.permute.default(mm_95, [1, 0]);  mm_95 = None
    sum_160: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_624, [0], True);  view_624 = None
    view_625: "f32[1280]" = torch.ops.aten.view.default(sum_160, [1280]);  sum_160 = None
    permute_528: "f32[1280, 320]" = torch.ops.aten.permute.default(permute_527, [1, 0]);  permute_527 = None
    view_626: "f32[8, 196, 320]" = torch.ops.aten.view.default(mm_94, [8, 196, 320]);  mm_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    mul_509: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(view_626, primals_355);  primals_355 = None
    mul_510: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_509, 320)
    sum_161: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_509, [2], True)
    mul_511: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_509, mul_172);  mul_509 = None
    sum_162: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_511, [2], True);  mul_511 = None
    mul_512: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_172, sum_162);  sum_162 = None
    sub_165: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(mul_510, sum_161);  mul_510 = sum_161 = None
    sub_166: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(sub_165, mul_512);  sub_165 = mul_512 = None
    mul_513: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(div_27, sub_166);  div_27 = sub_166 = None
    mul_514: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(view_626, mul_172);  mul_172 = None
    sum_163: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_514, [0, 1]);  mul_514 = None
    sum_164: "f32[320]" = torch.ops.aten.sum.dim_IntList(view_626, [0, 1]);  view_626 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_308: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_305, mul_513);  add_305 = mul_513 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    clone_220: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_308, memory_format = torch.contiguous_format)
    view_627: "f32[1568, 320]" = torch.ops.aten.view.default(clone_220, [1568, 320]);  clone_220 = None
    mm_96: "f32[1568, 320]" = torch.ops.aten.mm.default(view_627, permute_529);  permute_529 = None
    permute_530: "f32[320, 1568]" = torch.ops.aten.permute.default(view_627, [1, 0])
    mm_97: "f32[320, 320]" = torch.ops.aten.mm.default(permute_530, view_293);  permute_530 = view_293 = None
    permute_531: "f32[320, 320]" = torch.ops.aten.permute.default(mm_97, [1, 0]);  mm_97 = None
    sum_165: "f32[1, 320]" = torch.ops.aten.sum.dim_IntList(view_627, [0], True);  view_627 = None
    view_628: "f32[320]" = torch.ops.aten.view.default(sum_165, [320]);  sum_165 = None
    permute_532: "f32[320, 320]" = torch.ops.aten.permute.default(permute_531, [1, 0]);  permute_531 = None
    view_629: "f32[8, 196, 320]" = torch.ops.aten.view.default(mm_96, [8, 196, 320]);  mm_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_630: "f32[8, 196, 5, 64]" = torch.ops.aten.view.default(view_629, [8, 196, 5, 64]);  view_629 = None
    permute_533: "f32[8, 5, 196, 64]" = torch.ops.aten.permute.default(view_630, [0, 2, 1, 3]);  view_630 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_backward_9 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_533, permute_195, getitem_226, getitem_227, None, alias_37, getitem_229, getitem_230, getitem_231, 0.0, [True, True, True, False]);  permute_533 = permute_195 = getitem_226 = getitem_227 = alias_37 = getitem_229 = getitem_230 = getitem_231 = None
    getitem_400: "f32[8, 5, 196, 64]" = _scaled_dot_product_efficient_attention_backward_9[0]
    getitem_401: "f32[8, 5, 49, 64]" = _scaled_dot_product_efficient_attention_backward_9[1]
    getitem_402: "f32[8, 5, 49, 64]" = _scaled_dot_product_efficient_attention_backward_9[2];  _scaled_dot_product_efficient_attention_backward_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    cat_9: "f32[16, 5, 49, 64]" = torch.ops.aten.cat.default([getitem_401, getitem_402]);  getitem_401 = getitem_402 = None
    view_631: "f32[2, 8, 5, 49, 64]" = torch.ops.aten.view.default(cat_9, [2, 8, 5, 49, 64]);  cat_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_534: "f32[8, 49, 2, 5, 64]" = torch.ops.aten.permute.default(view_631, [1, 3, 0, 2, 4]);  view_631 = None
    clone_221: "f32[8, 49, 2, 5, 64]" = torch.ops.aten.clone.default(permute_534, memory_format = torch.contiguous_format);  permute_534 = None
    view_632: "f32[8, 49, 640]" = torch.ops.aten.view.default(clone_221, [8, 49, 640]);  clone_221 = None
    view_633: "f32[392, 640]" = torch.ops.aten.view.default(view_632, [392, 640]);  view_632 = None
    mm_98: "f32[392, 320]" = torch.ops.aten.mm.default(view_633, permute_535);  permute_535 = None
    permute_536: "f32[640, 392]" = torch.ops.aten.permute.default(view_633, [1, 0])
    mm_99: "f32[640, 320]" = torch.ops.aten.mm.default(permute_536, view_289);  permute_536 = view_289 = None
    permute_537: "f32[320, 640]" = torch.ops.aten.permute.default(mm_99, [1, 0]);  mm_99 = None
    sum_166: "f32[1, 640]" = torch.ops.aten.sum.dim_IntList(view_633, [0], True);  view_633 = None
    view_634: "f32[640]" = torch.ops.aten.view.default(sum_166, [640]);  sum_166 = None
    permute_538: "f32[640, 320]" = torch.ops.aten.permute.default(permute_537, [1, 0]);  permute_537 = None
    view_635: "f32[8, 49, 320]" = torch.ops.aten.view.default(mm_98, [8, 49, 320]);  mm_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    mul_516: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(view_635, primals_349);  primals_349 = None
    mul_517: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_516, 320)
    sum_167: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_516, [2], True)
    mul_518: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_516, mul_170);  mul_516 = None
    sum_168: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_518, [2], True);  mul_518 = None
    mul_519: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_170, sum_168);  sum_168 = None
    sub_168: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(mul_517, sum_167);  mul_517 = sum_167 = None
    sub_169: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(sub_168, mul_519);  sub_168 = mul_519 = None
    mul_520: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(div_28, sub_169);  div_28 = sub_169 = None
    mul_521: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(view_635, mul_170);  mul_170 = None
    sum_169: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_521, [0, 1]);  mul_521 = None
    sum_170: "f32[320]" = torch.ops.aten.sum.dim_IntList(view_635, [0, 1]);  view_635 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    permute_539: "f32[8, 320, 49]" = torch.ops.aten.permute.default(mul_520, [0, 2, 1]);  mul_520 = None
    view_636: "f32[8, 320, 7, 7]" = torch.ops.aten.view.default(permute_539, [8, 320, 7, 7]);  permute_539 = None
    sum_171: "f32[320]" = torch.ops.aten.sum.dim_IntList(view_636, [0, 2, 3])
    convolution_backward_8 = torch.ops.aten.convolution_backward.default(view_636, view_287, primals_347, [320], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_636 = view_287 = primals_347 = None
    getitem_404: "f32[8, 320, 14, 14]" = convolution_backward_8[0]
    getitem_405: "f32[320, 320, 2, 2]" = convolution_backward_8[1];  convolution_backward_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    view_637: "f32[8, 320, 196]" = torch.ops.aten.view.default(getitem_404, [8, 320, 196]);  getitem_404 = None
    permute_540: "f32[8, 196, 320]" = torch.ops.aten.permute.default(view_637, [0, 2, 1]);  view_637 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_541: "f32[8, 196, 5, 64]" = torch.ops.aten.permute.default(getitem_400, [0, 2, 1, 3]);  getitem_400 = None
    view_638: "f32[8, 196, 320]" = torch.ops.aten.view.default(permute_541, [8, 196, 320]);  permute_541 = None
    view_639: "f32[1568, 320]" = torch.ops.aten.view.default(view_638, [1568, 320]);  view_638 = None
    mm_100: "f32[1568, 320]" = torch.ops.aten.mm.default(view_639, permute_542);  permute_542 = None
    permute_543: "f32[320, 1568]" = torch.ops.aten.permute.default(view_639, [1, 0])
    mm_101: "f32[320, 320]" = torch.ops.aten.mm.default(permute_543, view_284);  permute_543 = view_284 = None
    permute_544: "f32[320, 320]" = torch.ops.aten.permute.default(mm_101, [1, 0]);  mm_101 = None
    sum_172: "f32[1, 320]" = torch.ops.aten.sum.dim_IntList(view_639, [0], True);  view_639 = None
    view_640: "f32[320]" = torch.ops.aten.view.default(sum_172, [320]);  sum_172 = None
    permute_545: "f32[320, 320]" = torch.ops.aten.permute.default(permute_544, [1, 0]);  permute_544 = None
    view_641: "f32[8, 196, 320]" = torch.ops.aten.view.default(mm_100, [8, 196, 320]);  mm_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    add_309: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(permute_540, view_641);  permute_540 = view_641 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    clone_223: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_309, memory_format = torch.contiguous_format);  add_309 = None
    mul_523: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(clone_223, primals_343);  primals_343 = None
    mul_524: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_523, 320)
    sum_173: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_523, [2], True)
    mul_525: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_523, mul_168);  mul_523 = None
    sum_174: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_525, [2], True);  mul_525 = None
    mul_526: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_168, sum_174);  sum_174 = None
    sub_171: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(mul_524, sum_173);  mul_524 = sum_173 = None
    sub_172: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(sub_171, mul_526);  sub_171 = mul_526 = None
    mul_527: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(div_29, sub_172);  div_29 = sub_172 = None
    mul_528: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(clone_223, mul_168);  mul_168 = None
    sum_175: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_528, [0, 1]);  mul_528 = None
    sum_176: "f32[320]" = torch.ops.aten.sum.dim_IntList(clone_223, [0, 1]);  clone_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_310: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_308, mul_527);  add_308 = mul_527 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    clone_225: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_310, memory_format = torch.contiguous_format)
    view_642: "f32[1568, 320]" = torch.ops.aten.view.default(clone_225, [1568, 320]);  clone_225 = None
    mm_102: "f32[1568, 1280]" = torch.ops.aten.mm.default(view_642, permute_546);  permute_546 = None
    permute_547: "f32[320, 1568]" = torch.ops.aten.permute.default(view_642, [1, 0])
    mm_103: "f32[320, 1280]" = torch.ops.aten.mm.default(permute_547, view_282);  permute_547 = view_282 = None
    permute_548: "f32[1280, 320]" = torch.ops.aten.permute.default(mm_103, [1, 0]);  mm_103 = None
    sum_177: "f32[1, 320]" = torch.ops.aten.sum.dim_IntList(view_642, [0], True);  view_642 = None
    view_643: "f32[320]" = torch.ops.aten.view.default(sum_177, [320]);  sum_177 = None
    permute_549: "f32[320, 1280]" = torch.ops.aten.permute.default(permute_548, [1, 0]);  permute_548 = None
    view_644: "f32[8, 196, 1280]" = torch.ops.aten.view.default(mm_102, [8, 196, 1280]);  mm_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_530: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(add_169, 0.5);  add_169 = None
    mul_531: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_281, view_281)
    mul_532: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(mul_531, -0.5);  mul_531 = None
    exp_10: "f32[8, 196, 1280]" = torch.ops.aten.exp.default(mul_532);  mul_532 = None
    mul_533: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(exp_10, 0.3989422804014327);  exp_10 = None
    mul_534: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_281, mul_533);  view_281 = mul_533 = None
    add_312: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(mul_530, mul_534);  mul_530 = mul_534 = None
    mul_535: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_644, add_312);  view_644 = add_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_645: "f32[1568, 1280]" = torch.ops.aten.view.default(mul_535, [1568, 1280]);  mul_535 = None
    mm_104: "f32[1568, 320]" = torch.ops.aten.mm.default(view_645, permute_550);  permute_550 = None
    permute_551: "f32[1280, 1568]" = torch.ops.aten.permute.default(view_645, [1, 0])
    mm_105: "f32[1280, 320]" = torch.ops.aten.mm.default(permute_551, view_280);  permute_551 = view_280 = None
    permute_552: "f32[320, 1280]" = torch.ops.aten.permute.default(mm_105, [1, 0]);  mm_105 = None
    sum_178: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_645, [0], True);  view_645 = None
    view_646: "f32[1280]" = torch.ops.aten.view.default(sum_178, [1280]);  sum_178 = None
    permute_553: "f32[1280, 320]" = torch.ops.aten.permute.default(permute_552, [1, 0]);  permute_552 = None
    view_647: "f32[8, 196, 320]" = torch.ops.aten.view.default(mm_104, [8, 196, 320]);  mm_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    mul_537: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(view_647, primals_337);  primals_337 = None
    mul_538: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_537, 320)
    sum_179: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_537, [2], True)
    mul_539: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_537, mul_163);  mul_537 = None
    sum_180: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_539, [2], True);  mul_539 = None
    mul_540: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_163, sum_180);  sum_180 = None
    sub_174: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(mul_538, sum_179);  mul_538 = sum_179 = None
    sub_175: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(sub_174, mul_540);  sub_174 = mul_540 = None
    mul_541: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(div_30, sub_175);  div_30 = sub_175 = None
    mul_542: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(view_647, mul_163);  mul_163 = None
    sum_181: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_542, [0, 1]);  mul_542 = None
    sum_182: "f32[320]" = torch.ops.aten.sum.dim_IntList(view_647, [0, 1]);  view_647 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_313: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_310, mul_541);  add_310 = mul_541 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    clone_227: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_313, memory_format = torch.contiguous_format)
    view_648: "f32[1568, 320]" = torch.ops.aten.view.default(clone_227, [1568, 320]);  clone_227 = None
    mm_106: "f32[1568, 320]" = torch.ops.aten.mm.default(view_648, permute_554);  permute_554 = None
    permute_555: "f32[320, 1568]" = torch.ops.aten.permute.default(view_648, [1, 0])
    mm_107: "f32[320, 320]" = torch.ops.aten.mm.default(permute_555, view_278);  permute_555 = view_278 = None
    permute_556: "f32[320, 320]" = torch.ops.aten.permute.default(mm_107, [1, 0]);  mm_107 = None
    sum_183: "f32[1, 320]" = torch.ops.aten.sum.dim_IntList(view_648, [0], True);  view_648 = None
    view_649: "f32[320]" = torch.ops.aten.view.default(sum_183, [320]);  sum_183 = None
    permute_557: "f32[320, 320]" = torch.ops.aten.permute.default(permute_556, [1, 0]);  permute_556 = None
    view_650: "f32[8, 196, 320]" = torch.ops.aten.view.default(mm_106, [8, 196, 320]);  mm_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_651: "f32[8, 196, 5, 64]" = torch.ops.aten.view.default(view_650, [8, 196, 5, 64]);  view_650 = None
    permute_558: "f32[8, 5, 196, 64]" = torch.ops.aten.permute.default(view_651, [0, 2, 1, 3]);  view_651 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_backward_10 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_558, permute_185, getitem_214, getitem_215, None, alias_38, getitem_217, getitem_218, getitem_219, 0.0, [True, True, True, False]);  permute_558 = permute_185 = getitem_214 = getitem_215 = alias_38 = getitem_217 = getitem_218 = getitem_219 = None
    getitem_407: "f32[8, 5, 196, 64]" = _scaled_dot_product_efficient_attention_backward_10[0]
    getitem_408: "f32[8, 5, 49, 64]" = _scaled_dot_product_efficient_attention_backward_10[1]
    getitem_409: "f32[8, 5, 49, 64]" = _scaled_dot_product_efficient_attention_backward_10[2];  _scaled_dot_product_efficient_attention_backward_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    cat_10: "f32[16, 5, 49, 64]" = torch.ops.aten.cat.default([getitem_408, getitem_409]);  getitem_408 = getitem_409 = None
    view_652: "f32[2, 8, 5, 49, 64]" = torch.ops.aten.view.default(cat_10, [2, 8, 5, 49, 64]);  cat_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_559: "f32[8, 49, 2, 5, 64]" = torch.ops.aten.permute.default(view_652, [1, 3, 0, 2, 4]);  view_652 = None
    clone_228: "f32[8, 49, 2, 5, 64]" = torch.ops.aten.clone.default(permute_559, memory_format = torch.contiguous_format);  permute_559 = None
    view_653: "f32[8, 49, 640]" = torch.ops.aten.view.default(clone_228, [8, 49, 640]);  clone_228 = None
    view_654: "f32[392, 640]" = torch.ops.aten.view.default(view_653, [392, 640]);  view_653 = None
    mm_108: "f32[392, 320]" = torch.ops.aten.mm.default(view_654, permute_560);  permute_560 = None
    permute_561: "f32[640, 392]" = torch.ops.aten.permute.default(view_654, [1, 0])
    mm_109: "f32[640, 320]" = torch.ops.aten.mm.default(permute_561, view_274);  permute_561 = view_274 = None
    permute_562: "f32[320, 640]" = torch.ops.aten.permute.default(mm_109, [1, 0]);  mm_109 = None
    sum_184: "f32[1, 640]" = torch.ops.aten.sum.dim_IntList(view_654, [0], True);  view_654 = None
    view_655: "f32[640]" = torch.ops.aten.view.default(sum_184, [640]);  sum_184 = None
    permute_563: "f32[640, 320]" = torch.ops.aten.permute.default(permute_562, [1, 0]);  permute_562 = None
    view_656: "f32[8, 49, 320]" = torch.ops.aten.view.default(mm_108, [8, 49, 320]);  mm_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    mul_544: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(view_656, primals_331);  primals_331 = None
    mul_545: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_544, 320)
    sum_185: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_544, [2], True)
    mul_546: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_544, mul_161);  mul_544 = None
    sum_186: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_546, [2], True);  mul_546 = None
    mul_547: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_161, sum_186);  sum_186 = None
    sub_177: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(mul_545, sum_185);  mul_545 = sum_185 = None
    sub_178: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(sub_177, mul_547);  sub_177 = mul_547 = None
    mul_548: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(div_31, sub_178);  div_31 = sub_178 = None
    mul_549: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(view_656, mul_161);  mul_161 = None
    sum_187: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_549, [0, 1]);  mul_549 = None
    sum_188: "f32[320]" = torch.ops.aten.sum.dim_IntList(view_656, [0, 1]);  view_656 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    permute_564: "f32[8, 320, 49]" = torch.ops.aten.permute.default(mul_548, [0, 2, 1]);  mul_548 = None
    view_657: "f32[8, 320, 7, 7]" = torch.ops.aten.view.default(permute_564, [8, 320, 7, 7]);  permute_564 = None
    sum_189: "f32[320]" = torch.ops.aten.sum.dim_IntList(view_657, [0, 2, 3])
    convolution_backward_9 = torch.ops.aten.convolution_backward.default(view_657, view_272, primals_329, [320], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_657 = view_272 = primals_329 = None
    getitem_411: "f32[8, 320, 14, 14]" = convolution_backward_9[0]
    getitem_412: "f32[320, 320, 2, 2]" = convolution_backward_9[1];  convolution_backward_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    view_658: "f32[8, 320, 196]" = torch.ops.aten.view.default(getitem_411, [8, 320, 196]);  getitem_411 = None
    permute_565: "f32[8, 196, 320]" = torch.ops.aten.permute.default(view_658, [0, 2, 1]);  view_658 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_566: "f32[8, 196, 5, 64]" = torch.ops.aten.permute.default(getitem_407, [0, 2, 1, 3]);  getitem_407 = None
    view_659: "f32[8, 196, 320]" = torch.ops.aten.view.default(permute_566, [8, 196, 320]);  permute_566 = None
    view_660: "f32[1568, 320]" = torch.ops.aten.view.default(view_659, [1568, 320]);  view_659 = None
    mm_110: "f32[1568, 320]" = torch.ops.aten.mm.default(view_660, permute_567);  permute_567 = None
    permute_568: "f32[320, 1568]" = torch.ops.aten.permute.default(view_660, [1, 0])
    mm_111: "f32[320, 320]" = torch.ops.aten.mm.default(permute_568, view_269);  permute_568 = view_269 = None
    permute_569: "f32[320, 320]" = torch.ops.aten.permute.default(mm_111, [1, 0]);  mm_111 = None
    sum_190: "f32[1, 320]" = torch.ops.aten.sum.dim_IntList(view_660, [0], True);  view_660 = None
    view_661: "f32[320]" = torch.ops.aten.view.default(sum_190, [320]);  sum_190 = None
    permute_570: "f32[320, 320]" = torch.ops.aten.permute.default(permute_569, [1, 0]);  permute_569 = None
    view_662: "f32[8, 196, 320]" = torch.ops.aten.view.default(mm_110, [8, 196, 320]);  mm_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    add_314: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(permute_565, view_662);  permute_565 = view_662 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    clone_230: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_314, memory_format = torch.contiguous_format);  add_314 = None
    mul_551: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(clone_230, primals_325);  primals_325 = None
    mul_552: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_551, 320)
    sum_191: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_551, [2], True)
    mul_553: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_551, mul_159);  mul_551 = None
    sum_192: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_553, [2], True);  mul_553 = None
    mul_554: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_159, sum_192);  sum_192 = None
    sub_180: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(mul_552, sum_191);  mul_552 = sum_191 = None
    sub_181: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(sub_180, mul_554);  sub_180 = mul_554 = None
    mul_555: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(div_32, sub_181);  div_32 = sub_181 = None
    mul_556: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(clone_230, mul_159);  mul_159 = None
    sum_193: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_556, [0, 1]);  mul_556 = None
    sum_194: "f32[320]" = torch.ops.aten.sum.dim_IntList(clone_230, [0, 1]);  clone_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_315: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_313, mul_555);  add_313 = mul_555 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    clone_232: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_315, memory_format = torch.contiguous_format)
    view_663: "f32[1568, 320]" = torch.ops.aten.view.default(clone_232, [1568, 320]);  clone_232 = None
    mm_112: "f32[1568, 1280]" = torch.ops.aten.mm.default(view_663, permute_571);  permute_571 = None
    permute_572: "f32[320, 1568]" = torch.ops.aten.permute.default(view_663, [1, 0])
    mm_113: "f32[320, 1280]" = torch.ops.aten.mm.default(permute_572, view_267);  permute_572 = view_267 = None
    permute_573: "f32[1280, 320]" = torch.ops.aten.permute.default(mm_113, [1, 0]);  mm_113 = None
    sum_195: "f32[1, 320]" = torch.ops.aten.sum.dim_IntList(view_663, [0], True);  view_663 = None
    view_664: "f32[320]" = torch.ops.aten.view.default(sum_195, [320]);  sum_195 = None
    permute_574: "f32[320, 1280]" = torch.ops.aten.permute.default(permute_573, [1, 0]);  permute_573 = None
    view_665: "f32[8, 196, 1280]" = torch.ops.aten.view.default(mm_112, [8, 196, 1280]);  mm_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_558: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(add_160, 0.5);  add_160 = None
    mul_559: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_266, view_266)
    mul_560: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(mul_559, -0.5);  mul_559 = None
    exp_11: "f32[8, 196, 1280]" = torch.ops.aten.exp.default(mul_560);  mul_560 = None
    mul_561: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(exp_11, 0.3989422804014327);  exp_11 = None
    mul_562: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_266, mul_561);  view_266 = mul_561 = None
    add_317: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(mul_558, mul_562);  mul_558 = mul_562 = None
    mul_563: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_665, add_317);  view_665 = add_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_666: "f32[1568, 1280]" = torch.ops.aten.view.default(mul_563, [1568, 1280]);  mul_563 = None
    mm_114: "f32[1568, 320]" = torch.ops.aten.mm.default(view_666, permute_575);  permute_575 = None
    permute_576: "f32[1280, 1568]" = torch.ops.aten.permute.default(view_666, [1, 0])
    mm_115: "f32[1280, 320]" = torch.ops.aten.mm.default(permute_576, view_265);  permute_576 = view_265 = None
    permute_577: "f32[320, 1280]" = torch.ops.aten.permute.default(mm_115, [1, 0]);  mm_115 = None
    sum_196: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_666, [0], True);  view_666 = None
    view_667: "f32[1280]" = torch.ops.aten.view.default(sum_196, [1280]);  sum_196 = None
    permute_578: "f32[1280, 320]" = torch.ops.aten.permute.default(permute_577, [1, 0]);  permute_577 = None
    view_668: "f32[8, 196, 320]" = torch.ops.aten.view.default(mm_114, [8, 196, 320]);  mm_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    mul_565: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(view_668, primals_319);  primals_319 = None
    mul_566: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_565, 320)
    sum_197: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_565, [2], True)
    mul_567: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_565, mul_154);  mul_565 = None
    sum_198: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_567, [2], True);  mul_567 = None
    mul_568: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_154, sum_198);  sum_198 = None
    sub_183: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(mul_566, sum_197);  mul_566 = sum_197 = None
    sub_184: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(sub_183, mul_568);  sub_183 = mul_568 = None
    mul_569: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(div_33, sub_184);  div_33 = sub_184 = None
    mul_570: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(view_668, mul_154);  mul_154 = None
    sum_199: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_570, [0, 1]);  mul_570 = None
    sum_200: "f32[320]" = torch.ops.aten.sum.dim_IntList(view_668, [0, 1]);  view_668 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_318: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_315, mul_569);  add_315 = mul_569 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    clone_234: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_318, memory_format = torch.contiguous_format)
    view_669: "f32[1568, 320]" = torch.ops.aten.view.default(clone_234, [1568, 320]);  clone_234 = None
    mm_116: "f32[1568, 320]" = torch.ops.aten.mm.default(view_669, permute_579);  permute_579 = None
    permute_580: "f32[320, 1568]" = torch.ops.aten.permute.default(view_669, [1, 0])
    mm_117: "f32[320, 320]" = torch.ops.aten.mm.default(permute_580, view_263);  permute_580 = view_263 = None
    permute_581: "f32[320, 320]" = torch.ops.aten.permute.default(mm_117, [1, 0]);  mm_117 = None
    sum_201: "f32[1, 320]" = torch.ops.aten.sum.dim_IntList(view_669, [0], True);  view_669 = None
    view_670: "f32[320]" = torch.ops.aten.view.default(sum_201, [320]);  sum_201 = None
    permute_582: "f32[320, 320]" = torch.ops.aten.permute.default(permute_581, [1, 0]);  permute_581 = None
    view_671: "f32[8, 196, 320]" = torch.ops.aten.view.default(mm_116, [8, 196, 320]);  mm_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_672: "f32[8, 196, 5, 64]" = torch.ops.aten.view.default(view_671, [8, 196, 5, 64]);  view_671 = None
    permute_583: "f32[8, 5, 196, 64]" = torch.ops.aten.permute.default(view_672, [0, 2, 1, 3]);  view_672 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_backward_11 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_583, permute_175, getitem_202, getitem_203, None, alias_39, getitem_205, getitem_206, getitem_207, 0.0, [True, True, True, False]);  permute_583 = permute_175 = getitem_202 = getitem_203 = alias_39 = getitem_205 = getitem_206 = getitem_207 = None
    getitem_414: "f32[8, 5, 196, 64]" = _scaled_dot_product_efficient_attention_backward_11[0]
    getitem_415: "f32[8, 5, 49, 64]" = _scaled_dot_product_efficient_attention_backward_11[1]
    getitem_416: "f32[8, 5, 49, 64]" = _scaled_dot_product_efficient_attention_backward_11[2];  _scaled_dot_product_efficient_attention_backward_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    cat_11: "f32[16, 5, 49, 64]" = torch.ops.aten.cat.default([getitem_415, getitem_416]);  getitem_415 = getitem_416 = None
    view_673: "f32[2, 8, 5, 49, 64]" = torch.ops.aten.view.default(cat_11, [2, 8, 5, 49, 64]);  cat_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_584: "f32[8, 49, 2, 5, 64]" = torch.ops.aten.permute.default(view_673, [1, 3, 0, 2, 4]);  view_673 = None
    clone_235: "f32[8, 49, 2, 5, 64]" = torch.ops.aten.clone.default(permute_584, memory_format = torch.contiguous_format);  permute_584 = None
    view_674: "f32[8, 49, 640]" = torch.ops.aten.view.default(clone_235, [8, 49, 640]);  clone_235 = None
    view_675: "f32[392, 640]" = torch.ops.aten.view.default(view_674, [392, 640]);  view_674 = None
    mm_118: "f32[392, 320]" = torch.ops.aten.mm.default(view_675, permute_585);  permute_585 = None
    permute_586: "f32[640, 392]" = torch.ops.aten.permute.default(view_675, [1, 0])
    mm_119: "f32[640, 320]" = torch.ops.aten.mm.default(permute_586, view_259);  permute_586 = view_259 = None
    permute_587: "f32[320, 640]" = torch.ops.aten.permute.default(mm_119, [1, 0]);  mm_119 = None
    sum_202: "f32[1, 640]" = torch.ops.aten.sum.dim_IntList(view_675, [0], True);  view_675 = None
    view_676: "f32[640]" = torch.ops.aten.view.default(sum_202, [640]);  sum_202 = None
    permute_588: "f32[640, 320]" = torch.ops.aten.permute.default(permute_587, [1, 0]);  permute_587 = None
    view_677: "f32[8, 49, 320]" = torch.ops.aten.view.default(mm_118, [8, 49, 320]);  mm_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    mul_572: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(view_677, primals_313);  primals_313 = None
    mul_573: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_572, 320)
    sum_203: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_572, [2], True)
    mul_574: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_572, mul_152);  mul_572 = None
    sum_204: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_574, [2], True);  mul_574 = None
    mul_575: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_152, sum_204);  sum_204 = None
    sub_186: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(mul_573, sum_203);  mul_573 = sum_203 = None
    sub_187: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(sub_186, mul_575);  sub_186 = mul_575 = None
    mul_576: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(div_34, sub_187);  div_34 = sub_187 = None
    mul_577: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(view_677, mul_152);  mul_152 = None
    sum_205: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_577, [0, 1]);  mul_577 = None
    sum_206: "f32[320]" = torch.ops.aten.sum.dim_IntList(view_677, [0, 1]);  view_677 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    permute_589: "f32[8, 320, 49]" = torch.ops.aten.permute.default(mul_576, [0, 2, 1]);  mul_576 = None
    view_678: "f32[8, 320, 7, 7]" = torch.ops.aten.view.default(permute_589, [8, 320, 7, 7]);  permute_589 = None
    sum_207: "f32[320]" = torch.ops.aten.sum.dim_IntList(view_678, [0, 2, 3])
    convolution_backward_10 = torch.ops.aten.convolution_backward.default(view_678, view_257, primals_311, [320], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_678 = view_257 = primals_311 = None
    getitem_418: "f32[8, 320, 14, 14]" = convolution_backward_10[0]
    getitem_419: "f32[320, 320, 2, 2]" = convolution_backward_10[1];  convolution_backward_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    view_679: "f32[8, 320, 196]" = torch.ops.aten.view.default(getitem_418, [8, 320, 196]);  getitem_418 = None
    permute_590: "f32[8, 196, 320]" = torch.ops.aten.permute.default(view_679, [0, 2, 1]);  view_679 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_591: "f32[8, 196, 5, 64]" = torch.ops.aten.permute.default(getitem_414, [0, 2, 1, 3]);  getitem_414 = None
    view_680: "f32[8, 196, 320]" = torch.ops.aten.view.default(permute_591, [8, 196, 320]);  permute_591 = None
    view_681: "f32[1568, 320]" = torch.ops.aten.view.default(view_680, [1568, 320]);  view_680 = None
    mm_120: "f32[1568, 320]" = torch.ops.aten.mm.default(view_681, permute_592);  permute_592 = None
    permute_593: "f32[320, 1568]" = torch.ops.aten.permute.default(view_681, [1, 0])
    mm_121: "f32[320, 320]" = torch.ops.aten.mm.default(permute_593, view_254);  permute_593 = view_254 = None
    permute_594: "f32[320, 320]" = torch.ops.aten.permute.default(mm_121, [1, 0]);  mm_121 = None
    sum_208: "f32[1, 320]" = torch.ops.aten.sum.dim_IntList(view_681, [0], True);  view_681 = None
    view_682: "f32[320]" = torch.ops.aten.view.default(sum_208, [320]);  sum_208 = None
    permute_595: "f32[320, 320]" = torch.ops.aten.permute.default(permute_594, [1, 0]);  permute_594 = None
    view_683: "f32[8, 196, 320]" = torch.ops.aten.view.default(mm_120, [8, 196, 320]);  mm_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    add_319: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(permute_590, view_683);  permute_590 = view_683 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    clone_237: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_319, memory_format = torch.contiguous_format);  add_319 = None
    mul_579: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(clone_237, primals_307);  primals_307 = None
    mul_580: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_579, 320)
    sum_209: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_579, [2], True)
    mul_581: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_579, mul_150);  mul_579 = None
    sum_210: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_581, [2], True);  mul_581 = None
    mul_582: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_150, sum_210);  sum_210 = None
    sub_189: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(mul_580, sum_209);  mul_580 = sum_209 = None
    sub_190: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(sub_189, mul_582);  sub_189 = mul_582 = None
    mul_583: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(div_35, sub_190);  div_35 = sub_190 = None
    mul_584: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(clone_237, mul_150);  mul_150 = None
    sum_211: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_584, [0, 1]);  mul_584 = None
    sum_212: "f32[320]" = torch.ops.aten.sum.dim_IntList(clone_237, [0, 1]);  clone_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_320: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_318, mul_583);  add_318 = mul_583 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    clone_239: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_320, memory_format = torch.contiguous_format)
    view_684: "f32[1568, 320]" = torch.ops.aten.view.default(clone_239, [1568, 320]);  clone_239 = None
    mm_122: "f32[1568, 1280]" = torch.ops.aten.mm.default(view_684, permute_596);  permute_596 = None
    permute_597: "f32[320, 1568]" = torch.ops.aten.permute.default(view_684, [1, 0])
    mm_123: "f32[320, 1280]" = torch.ops.aten.mm.default(permute_597, view_252);  permute_597 = view_252 = None
    permute_598: "f32[1280, 320]" = torch.ops.aten.permute.default(mm_123, [1, 0]);  mm_123 = None
    sum_213: "f32[1, 320]" = torch.ops.aten.sum.dim_IntList(view_684, [0], True);  view_684 = None
    view_685: "f32[320]" = torch.ops.aten.view.default(sum_213, [320]);  sum_213 = None
    permute_599: "f32[320, 1280]" = torch.ops.aten.permute.default(permute_598, [1, 0]);  permute_598 = None
    view_686: "f32[8, 196, 1280]" = torch.ops.aten.view.default(mm_122, [8, 196, 1280]);  mm_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_586: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(add_151, 0.5);  add_151 = None
    mul_587: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_251, view_251)
    mul_588: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(mul_587, -0.5);  mul_587 = None
    exp_12: "f32[8, 196, 1280]" = torch.ops.aten.exp.default(mul_588);  mul_588 = None
    mul_589: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(exp_12, 0.3989422804014327);  exp_12 = None
    mul_590: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_251, mul_589);  view_251 = mul_589 = None
    add_322: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(mul_586, mul_590);  mul_586 = mul_590 = None
    mul_591: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_686, add_322);  view_686 = add_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_687: "f32[1568, 1280]" = torch.ops.aten.view.default(mul_591, [1568, 1280]);  mul_591 = None
    mm_124: "f32[1568, 320]" = torch.ops.aten.mm.default(view_687, permute_600);  permute_600 = None
    permute_601: "f32[1280, 1568]" = torch.ops.aten.permute.default(view_687, [1, 0])
    mm_125: "f32[1280, 320]" = torch.ops.aten.mm.default(permute_601, view_250);  permute_601 = view_250 = None
    permute_602: "f32[320, 1280]" = torch.ops.aten.permute.default(mm_125, [1, 0]);  mm_125 = None
    sum_214: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_687, [0], True);  view_687 = None
    view_688: "f32[1280]" = torch.ops.aten.view.default(sum_214, [1280]);  sum_214 = None
    permute_603: "f32[1280, 320]" = torch.ops.aten.permute.default(permute_602, [1, 0]);  permute_602 = None
    view_689: "f32[8, 196, 320]" = torch.ops.aten.view.default(mm_124, [8, 196, 320]);  mm_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    mul_593: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(view_689, primals_301);  primals_301 = None
    mul_594: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_593, 320)
    sum_215: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_593, [2], True)
    mul_595: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_593, mul_145);  mul_593 = None
    sum_216: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_595, [2], True);  mul_595 = None
    mul_596: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_145, sum_216);  sum_216 = None
    sub_192: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(mul_594, sum_215);  mul_594 = sum_215 = None
    sub_193: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(sub_192, mul_596);  sub_192 = mul_596 = None
    mul_597: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(div_36, sub_193);  div_36 = sub_193 = None
    mul_598: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(view_689, mul_145);  mul_145 = None
    sum_217: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_598, [0, 1]);  mul_598 = None
    sum_218: "f32[320]" = torch.ops.aten.sum.dim_IntList(view_689, [0, 1]);  view_689 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_323: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_320, mul_597);  add_320 = mul_597 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    clone_241: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_323, memory_format = torch.contiguous_format)
    view_690: "f32[1568, 320]" = torch.ops.aten.view.default(clone_241, [1568, 320]);  clone_241 = None
    mm_126: "f32[1568, 320]" = torch.ops.aten.mm.default(view_690, permute_604);  permute_604 = None
    permute_605: "f32[320, 1568]" = torch.ops.aten.permute.default(view_690, [1, 0])
    mm_127: "f32[320, 320]" = torch.ops.aten.mm.default(permute_605, view_248);  permute_605 = view_248 = None
    permute_606: "f32[320, 320]" = torch.ops.aten.permute.default(mm_127, [1, 0]);  mm_127 = None
    sum_219: "f32[1, 320]" = torch.ops.aten.sum.dim_IntList(view_690, [0], True);  view_690 = None
    view_691: "f32[320]" = torch.ops.aten.view.default(sum_219, [320]);  sum_219 = None
    permute_607: "f32[320, 320]" = torch.ops.aten.permute.default(permute_606, [1, 0]);  permute_606 = None
    view_692: "f32[8, 196, 320]" = torch.ops.aten.view.default(mm_126, [8, 196, 320]);  mm_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_693: "f32[8, 196, 5, 64]" = torch.ops.aten.view.default(view_692, [8, 196, 5, 64]);  view_692 = None
    permute_608: "f32[8, 5, 196, 64]" = torch.ops.aten.permute.default(view_693, [0, 2, 1, 3]);  view_693 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_backward_12 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_608, permute_165, getitem_190, getitem_191, None, alias_40, getitem_193, getitem_194, getitem_195, 0.0, [True, True, True, False]);  permute_608 = permute_165 = getitem_190 = getitem_191 = alias_40 = getitem_193 = getitem_194 = getitem_195 = None
    getitem_421: "f32[8, 5, 196, 64]" = _scaled_dot_product_efficient_attention_backward_12[0]
    getitem_422: "f32[8, 5, 49, 64]" = _scaled_dot_product_efficient_attention_backward_12[1]
    getitem_423: "f32[8, 5, 49, 64]" = _scaled_dot_product_efficient_attention_backward_12[2];  _scaled_dot_product_efficient_attention_backward_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    cat_12: "f32[16, 5, 49, 64]" = torch.ops.aten.cat.default([getitem_422, getitem_423]);  getitem_422 = getitem_423 = None
    view_694: "f32[2, 8, 5, 49, 64]" = torch.ops.aten.view.default(cat_12, [2, 8, 5, 49, 64]);  cat_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_609: "f32[8, 49, 2, 5, 64]" = torch.ops.aten.permute.default(view_694, [1, 3, 0, 2, 4]);  view_694 = None
    clone_242: "f32[8, 49, 2, 5, 64]" = torch.ops.aten.clone.default(permute_609, memory_format = torch.contiguous_format);  permute_609 = None
    view_695: "f32[8, 49, 640]" = torch.ops.aten.view.default(clone_242, [8, 49, 640]);  clone_242 = None
    view_696: "f32[392, 640]" = torch.ops.aten.view.default(view_695, [392, 640]);  view_695 = None
    mm_128: "f32[392, 320]" = torch.ops.aten.mm.default(view_696, permute_610);  permute_610 = None
    permute_611: "f32[640, 392]" = torch.ops.aten.permute.default(view_696, [1, 0])
    mm_129: "f32[640, 320]" = torch.ops.aten.mm.default(permute_611, view_244);  permute_611 = view_244 = None
    permute_612: "f32[320, 640]" = torch.ops.aten.permute.default(mm_129, [1, 0]);  mm_129 = None
    sum_220: "f32[1, 640]" = torch.ops.aten.sum.dim_IntList(view_696, [0], True);  view_696 = None
    view_697: "f32[640]" = torch.ops.aten.view.default(sum_220, [640]);  sum_220 = None
    permute_613: "f32[640, 320]" = torch.ops.aten.permute.default(permute_612, [1, 0]);  permute_612 = None
    view_698: "f32[8, 49, 320]" = torch.ops.aten.view.default(mm_128, [8, 49, 320]);  mm_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    mul_600: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(view_698, primals_295);  primals_295 = None
    mul_601: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_600, 320)
    sum_221: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_600, [2], True)
    mul_602: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_600, mul_143);  mul_600 = None
    sum_222: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_602, [2], True);  mul_602 = None
    mul_603: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_143, sum_222);  sum_222 = None
    sub_195: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(mul_601, sum_221);  mul_601 = sum_221 = None
    sub_196: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(sub_195, mul_603);  sub_195 = mul_603 = None
    mul_604: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(div_37, sub_196);  div_37 = sub_196 = None
    mul_605: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(view_698, mul_143);  mul_143 = None
    sum_223: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_605, [0, 1]);  mul_605 = None
    sum_224: "f32[320]" = torch.ops.aten.sum.dim_IntList(view_698, [0, 1]);  view_698 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    permute_614: "f32[8, 320, 49]" = torch.ops.aten.permute.default(mul_604, [0, 2, 1]);  mul_604 = None
    view_699: "f32[8, 320, 7, 7]" = torch.ops.aten.view.default(permute_614, [8, 320, 7, 7]);  permute_614 = None
    sum_225: "f32[320]" = torch.ops.aten.sum.dim_IntList(view_699, [0, 2, 3])
    convolution_backward_11 = torch.ops.aten.convolution_backward.default(view_699, view_242, primals_293, [320], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_699 = view_242 = primals_293 = None
    getitem_425: "f32[8, 320, 14, 14]" = convolution_backward_11[0]
    getitem_426: "f32[320, 320, 2, 2]" = convolution_backward_11[1];  convolution_backward_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    view_700: "f32[8, 320, 196]" = torch.ops.aten.view.default(getitem_425, [8, 320, 196]);  getitem_425 = None
    permute_615: "f32[8, 196, 320]" = torch.ops.aten.permute.default(view_700, [0, 2, 1]);  view_700 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_616: "f32[8, 196, 5, 64]" = torch.ops.aten.permute.default(getitem_421, [0, 2, 1, 3]);  getitem_421 = None
    view_701: "f32[8, 196, 320]" = torch.ops.aten.view.default(permute_616, [8, 196, 320]);  permute_616 = None
    view_702: "f32[1568, 320]" = torch.ops.aten.view.default(view_701, [1568, 320]);  view_701 = None
    mm_130: "f32[1568, 320]" = torch.ops.aten.mm.default(view_702, permute_617);  permute_617 = None
    permute_618: "f32[320, 1568]" = torch.ops.aten.permute.default(view_702, [1, 0])
    mm_131: "f32[320, 320]" = torch.ops.aten.mm.default(permute_618, view_239);  permute_618 = view_239 = None
    permute_619: "f32[320, 320]" = torch.ops.aten.permute.default(mm_131, [1, 0]);  mm_131 = None
    sum_226: "f32[1, 320]" = torch.ops.aten.sum.dim_IntList(view_702, [0], True);  view_702 = None
    view_703: "f32[320]" = torch.ops.aten.view.default(sum_226, [320]);  sum_226 = None
    permute_620: "f32[320, 320]" = torch.ops.aten.permute.default(permute_619, [1, 0]);  permute_619 = None
    view_704: "f32[8, 196, 320]" = torch.ops.aten.view.default(mm_130, [8, 196, 320]);  mm_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    add_324: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(permute_615, view_704);  permute_615 = view_704 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    clone_244: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_324, memory_format = torch.contiguous_format);  add_324 = None
    mul_607: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(clone_244, primals_289);  primals_289 = None
    mul_608: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_607, 320)
    sum_227: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_607, [2], True)
    mul_609: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_607, mul_141);  mul_607 = None
    sum_228: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_609, [2], True);  mul_609 = None
    mul_610: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_141, sum_228);  sum_228 = None
    sub_198: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(mul_608, sum_227);  mul_608 = sum_227 = None
    sub_199: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(sub_198, mul_610);  sub_198 = mul_610 = None
    mul_611: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(div_38, sub_199);  div_38 = sub_199 = None
    mul_612: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(clone_244, mul_141);  mul_141 = None
    sum_229: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_612, [0, 1]);  mul_612 = None
    sum_230: "f32[320]" = torch.ops.aten.sum.dim_IntList(clone_244, [0, 1]);  clone_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_325: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_323, mul_611);  add_323 = mul_611 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    clone_246: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_325, memory_format = torch.contiguous_format)
    view_705: "f32[1568, 320]" = torch.ops.aten.view.default(clone_246, [1568, 320]);  clone_246 = None
    mm_132: "f32[1568, 1280]" = torch.ops.aten.mm.default(view_705, permute_621);  permute_621 = None
    permute_622: "f32[320, 1568]" = torch.ops.aten.permute.default(view_705, [1, 0])
    mm_133: "f32[320, 1280]" = torch.ops.aten.mm.default(permute_622, view_237);  permute_622 = view_237 = None
    permute_623: "f32[1280, 320]" = torch.ops.aten.permute.default(mm_133, [1, 0]);  mm_133 = None
    sum_231: "f32[1, 320]" = torch.ops.aten.sum.dim_IntList(view_705, [0], True);  view_705 = None
    view_706: "f32[320]" = torch.ops.aten.view.default(sum_231, [320]);  sum_231 = None
    permute_624: "f32[320, 1280]" = torch.ops.aten.permute.default(permute_623, [1, 0]);  permute_623 = None
    view_707: "f32[8, 196, 1280]" = torch.ops.aten.view.default(mm_132, [8, 196, 1280]);  mm_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_614: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(add_142, 0.5);  add_142 = None
    mul_615: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_236, view_236)
    mul_616: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(mul_615, -0.5);  mul_615 = None
    exp_13: "f32[8, 196, 1280]" = torch.ops.aten.exp.default(mul_616);  mul_616 = None
    mul_617: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(exp_13, 0.3989422804014327);  exp_13 = None
    mul_618: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_236, mul_617);  view_236 = mul_617 = None
    add_327: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(mul_614, mul_618);  mul_614 = mul_618 = None
    mul_619: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_707, add_327);  view_707 = add_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_708: "f32[1568, 1280]" = torch.ops.aten.view.default(mul_619, [1568, 1280]);  mul_619 = None
    mm_134: "f32[1568, 320]" = torch.ops.aten.mm.default(view_708, permute_625);  permute_625 = None
    permute_626: "f32[1280, 1568]" = torch.ops.aten.permute.default(view_708, [1, 0])
    mm_135: "f32[1280, 320]" = torch.ops.aten.mm.default(permute_626, view_235);  permute_626 = view_235 = None
    permute_627: "f32[320, 1280]" = torch.ops.aten.permute.default(mm_135, [1, 0]);  mm_135 = None
    sum_232: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_708, [0], True);  view_708 = None
    view_709: "f32[1280]" = torch.ops.aten.view.default(sum_232, [1280]);  sum_232 = None
    permute_628: "f32[1280, 320]" = torch.ops.aten.permute.default(permute_627, [1, 0]);  permute_627 = None
    view_710: "f32[8, 196, 320]" = torch.ops.aten.view.default(mm_134, [8, 196, 320]);  mm_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    mul_621: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(view_710, primals_283);  primals_283 = None
    mul_622: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_621, 320)
    sum_233: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_621, [2], True)
    mul_623: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_621, mul_136);  mul_621 = None
    sum_234: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_623, [2], True);  mul_623 = None
    mul_624: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_136, sum_234);  sum_234 = None
    sub_201: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(mul_622, sum_233);  mul_622 = sum_233 = None
    sub_202: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(sub_201, mul_624);  sub_201 = mul_624 = None
    mul_625: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(div_39, sub_202);  div_39 = sub_202 = None
    mul_626: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(view_710, mul_136);  mul_136 = None
    sum_235: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_626, [0, 1]);  mul_626 = None
    sum_236: "f32[320]" = torch.ops.aten.sum.dim_IntList(view_710, [0, 1]);  view_710 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_328: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_325, mul_625);  add_325 = mul_625 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    clone_248: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_328, memory_format = torch.contiguous_format)
    view_711: "f32[1568, 320]" = torch.ops.aten.view.default(clone_248, [1568, 320]);  clone_248 = None
    mm_136: "f32[1568, 320]" = torch.ops.aten.mm.default(view_711, permute_629);  permute_629 = None
    permute_630: "f32[320, 1568]" = torch.ops.aten.permute.default(view_711, [1, 0])
    mm_137: "f32[320, 320]" = torch.ops.aten.mm.default(permute_630, view_233);  permute_630 = view_233 = None
    permute_631: "f32[320, 320]" = torch.ops.aten.permute.default(mm_137, [1, 0]);  mm_137 = None
    sum_237: "f32[1, 320]" = torch.ops.aten.sum.dim_IntList(view_711, [0], True);  view_711 = None
    view_712: "f32[320]" = torch.ops.aten.view.default(sum_237, [320]);  sum_237 = None
    permute_632: "f32[320, 320]" = torch.ops.aten.permute.default(permute_631, [1, 0]);  permute_631 = None
    view_713: "f32[8, 196, 320]" = torch.ops.aten.view.default(mm_136, [8, 196, 320]);  mm_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_714: "f32[8, 196, 5, 64]" = torch.ops.aten.view.default(view_713, [8, 196, 5, 64]);  view_713 = None
    permute_633: "f32[8, 5, 196, 64]" = torch.ops.aten.permute.default(view_714, [0, 2, 1, 3]);  view_714 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_backward_13 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_633, permute_155, getitem_178, getitem_179, None, alias_41, getitem_181, getitem_182, getitem_183, 0.0, [True, True, True, False]);  permute_633 = permute_155 = getitem_178 = getitem_179 = alias_41 = getitem_181 = getitem_182 = getitem_183 = None
    getitem_428: "f32[8, 5, 196, 64]" = _scaled_dot_product_efficient_attention_backward_13[0]
    getitem_429: "f32[8, 5, 49, 64]" = _scaled_dot_product_efficient_attention_backward_13[1]
    getitem_430: "f32[8, 5, 49, 64]" = _scaled_dot_product_efficient_attention_backward_13[2];  _scaled_dot_product_efficient_attention_backward_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    cat_13: "f32[16, 5, 49, 64]" = torch.ops.aten.cat.default([getitem_429, getitem_430]);  getitem_429 = getitem_430 = None
    view_715: "f32[2, 8, 5, 49, 64]" = torch.ops.aten.view.default(cat_13, [2, 8, 5, 49, 64]);  cat_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_634: "f32[8, 49, 2, 5, 64]" = torch.ops.aten.permute.default(view_715, [1, 3, 0, 2, 4]);  view_715 = None
    clone_249: "f32[8, 49, 2, 5, 64]" = torch.ops.aten.clone.default(permute_634, memory_format = torch.contiguous_format);  permute_634 = None
    view_716: "f32[8, 49, 640]" = torch.ops.aten.view.default(clone_249, [8, 49, 640]);  clone_249 = None
    view_717: "f32[392, 640]" = torch.ops.aten.view.default(view_716, [392, 640]);  view_716 = None
    mm_138: "f32[392, 320]" = torch.ops.aten.mm.default(view_717, permute_635);  permute_635 = None
    permute_636: "f32[640, 392]" = torch.ops.aten.permute.default(view_717, [1, 0])
    mm_139: "f32[640, 320]" = torch.ops.aten.mm.default(permute_636, view_229);  permute_636 = view_229 = None
    permute_637: "f32[320, 640]" = torch.ops.aten.permute.default(mm_139, [1, 0]);  mm_139 = None
    sum_238: "f32[1, 640]" = torch.ops.aten.sum.dim_IntList(view_717, [0], True);  view_717 = None
    view_718: "f32[640]" = torch.ops.aten.view.default(sum_238, [640]);  sum_238 = None
    permute_638: "f32[640, 320]" = torch.ops.aten.permute.default(permute_637, [1, 0]);  permute_637 = None
    view_719: "f32[8, 49, 320]" = torch.ops.aten.view.default(mm_138, [8, 49, 320]);  mm_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    mul_628: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(view_719, primals_277);  primals_277 = None
    mul_629: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_628, 320)
    sum_239: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_628, [2], True)
    mul_630: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_628, mul_134);  mul_628 = None
    sum_240: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_630, [2], True);  mul_630 = None
    mul_631: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_134, sum_240);  sum_240 = None
    sub_204: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(mul_629, sum_239);  mul_629 = sum_239 = None
    sub_205: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(sub_204, mul_631);  sub_204 = mul_631 = None
    mul_632: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(div_40, sub_205);  div_40 = sub_205 = None
    mul_633: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(view_719, mul_134);  mul_134 = None
    sum_241: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_633, [0, 1]);  mul_633 = None
    sum_242: "f32[320]" = torch.ops.aten.sum.dim_IntList(view_719, [0, 1]);  view_719 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    permute_639: "f32[8, 320, 49]" = torch.ops.aten.permute.default(mul_632, [0, 2, 1]);  mul_632 = None
    view_720: "f32[8, 320, 7, 7]" = torch.ops.aten.view.default(permute_639, [8, 320, 7, 7]);  permute_639 = None
    sum_243: "f32[320]" = torch.ops.aten.sum.dim_IntList(view_720, [0, 2, 3])
    convolution_backward_12 = torch.ops.aten.convolution_backward.default(view_720, view_227, primals_275, [320], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_720 = view_227 = primals_275 = None
    getitem_432: "f32[8, 320, 14, 14]" = convolution_backward_12[0]
    getitem_433: "f32[320, 320, 2, 2]" = convolution_backward_12[1];  convolution_backward_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    view_721: "f32[8, 320, 196]" = torch.ops.aten.view.default(getitem_432, [8, 320, 196]);  getitem_432 = None
    permute_640: "f32[8, 196, 320]" = torch.ops.aten.permute.default(view_721, [0, 2, 1]);  view_721 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_641: "f32[8, 196, 5, 64]" = torch.ops.aten.permute.default(getitem_428, [0, 2, 1, 3]);  getitem_428 = None
    view_722: "f32[8, 196, 320]" = torch.ops.aten.view.default(permute_641, [8, 196, 320]);  permute_641 = None
    view_723: "f32[1568, 320]" = torch.ops.aten.view.default(view_722, [1568, 320]);  view_722 = None
    mm_140: "f32[1568, 320]" = torch.ops.aten.mm.default(view_723, permute_642);  permute_642 = None
    permute_643: "f32[320, 1568]" = torch.ops.aten.permute.default(view_723, [1, 0])
    mm_141: "f32[320, 320]" = torch.ops.aten.mm.default(permute_643, view_224);  permute_643 = view_224 = None
    permute_644: "f32[320, 320]" = torch.ops.aten.permute.default(mm_141, [1, 0]);  mm_141 = None
    sum_244: "f32[1, 320]" = torch.ops.aten.sum.dim_IntList(view_723, [0], True);  view_723 = None
    view_724: "f32[320]" = torch.ops.aten.view.default(sum_244, [320]);  sum_244 = None
    permute_645: "f32[320, 320]" = torch.ops.aten.permute.default(permute_644, [1, 0]);  permute_644 = None
    view_725: "f32[8, 196, 320]" = torch.ops.aten.view.default(mm_140, [8, 196, 320]);  mm_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    add_329: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(permute_640, view_725);  permute_640 = view_725 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    clone_251: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_329, memory_format = torch.contiguous_format);  add_329 = None
    mul_635: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(clone_251, primals_271);  primals_271 = None
    mul_636: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_635, 320)
    sum_245: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_635, [2], True)
    mul_637: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_635, mul_132);  mul_635 = None
    sum_246: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_637, [2], True);  mul_637 = None
    mul_638: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_132, sum_246);  sum_246 = None
    sub_207: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(mul_636, sum_245);  mul_636 = sum_245 = None
    sub_208: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(sub_207, mul_638);  sub_207 = mul_638 = None
    mul_639: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(div_41, sub_208);  div_41 = sub_208 = None
    mul_640: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(clone_251, mul_132);  mul_132 = None
    sum_247: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_640, [0, 1]);  mul_640 = None
    sum_248: "f32[320]" = torch.ops.aten.sum.dim_IntList(clone_251, [0, 1]);  clone_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_330: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_328, mul_639);  add_328 = mul_639 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    clone_253: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_330, memory_format = torch.contiguous_format)
    view_726: "f32[1568, 320]" = torch.ops.aten.view.default(clone_253, [1568, 320]);  clone_253 = None
    mm_142: "f32[1568, 1280]" = torch.ops.aten.mm.default(view_726, permute_646);  permute_646 = None
    permute_647: "f32[320, 1568]" = torch.ops.aten.permute.default(view_726, [1, 0])
    mm_143: "f32[320, 1280]" = torch.ops.aten.mm.default(permute_647, view_222);  permute_647 = view_222 = None
    permute_648: "f32[1280, 320]" = torch.ops.aten.permute.default(mm_143, [1, 0]);  mm_143 = None
    sum_249: "f32[1, 320]" = torch.ops.aten.sum.dim_IntList(view_726, [0], True);  view_726 = None
    view_727: "f32[320]" = torch.ops.aten.view.default(sum_249, [320]);  sum_249 = None
    permute_649: "f32[320, 1280]" = torch.ops.aten.permute.default(permute_648, [1, 0]);  permute_648 = None
    view_728: "f32[8, 196, 1280]" = torch.ops.aten.view.default(mm_142, [8, 196, 1280]);  mm_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_642: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(add_133, 0.5);  add_133 = None
    mul_643: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_221, view_221)
    mul_644: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(mul_643, -0.5);  mul_643 = None
    exp_14: "f32[8, 196, 1280]" = torch.ops.aten.exp.default(mul_644);  mul_644 = None
    mul_645: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(exp_14, 0.3989422804014327);  exp_14 = None
    mul_646: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_221, mul_645);  view_221 = mul_645 = None
    add_332: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(mul_642, mul_646);  mul_642 = mul_646 = None
    mul_647: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_728, add_332);  view_728 = add_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_729: "f32[1568, 1280]" = torch.ops.aten.view.default(mul_647, [1568, 1280]);  mul_647 = None
    mm_144: "f32[1568, 320]" = torch.ops.aten.mm.default(view_729, permute_650);  permute_650 = None
    permute_651: "f32[1280, 1568]" = torch.ops.aten.permute.default(view_729, [1, 0])
    mm_145: "f32[1280, 320]" = torch.ops.aten.mm.default(permute_651, view_220);  permute_651 = view_220 = None
    permute_652: "f32[320, 1280]" = torch.ops.aten.permute.default(mm_145, [1, 0]);  mm_145 = None
    sum_250: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_729, [0], True);  view_729 = None
    view_730: "f32[1280]" = torch.ops.aten.view.default(sum_250, [1280]);  sum_250 = None
    permute_653: "f32[1280, 320]" = torch.ops.aten.permute.default(permute_652, [1, 0]);  permute_652 = None
    view_731: "f32[8, 196, 320]" = torch.ops.aten.view.default(mm_144, [8, 196, 320]);  mm_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    mul_649: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(view_731, primals_265);  primals_265 = None
    mul_650: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_649, 320)
    sum_251: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_649, [2], True)
    mul_651: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_649, mul_127);  mul_649 = None
    sum_252: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_651, [2], True);  mul_651 = None
    mul_652: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_127, sum_252);  sum_252 = None
    sub_210: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(mul_650, sum_251);  mul_650 = sum_251 = None
    sub_211: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(sub_210, mul_652);  sub_210 = mul_652 = None
    mul_653: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(div_42, sub_211);  div_42 = sub_211 = None
    mul_654: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(view_731, mul_127);  mul_127 = None
    sum_253: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_654, [0, 1]);  mul_654 = None
    sum_254: "f32[320]" = torch.ops.aten.sum.dim_IntList(view_731, [0, 1]);  view_731 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_333: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_330, mul_653);  add_330 = mul_653 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    clone_255: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_333, memory_format = torch.contiguous_format)
    view_732: "f32[1568, 320]" = torch.ops.aten.view.default(clone_255, [1568, 320]);  clone_255 = None
    mm_146: "f32[1568, 320]" = torch.ops.aten.mm.default(view_732, permute_654);  permute_654 = None
    permute_655: "f32[320, 1568]" = torch.ops.aten.permute.default(view_732, [1, 0])
    mm_147: "f32[320, 320]" = torch.ops.aten.mm.default(permute_655, view_218);  permute_655 = view_218 = None
    permute_656: "f32[320, 320]" = torch.ops.aten.permute.default(mm_147, [1, 0]);  mm_147 = None
    sum_255: "f32[1, 320]" = torch.ops.aten.sum.dim_IntList(view_732, [0], True);  view_732 = None
    view_733: "f32[320]" = torch.ops.aten.view.default(sum_255, [320]);  sum_255 = None
    permute_657: "f32[320, 320]" = torch.ops.aten.permute.default(permute_656, [1, 0]);  permute_656 = None
    view_734: "f32[8, 196, 320]" = torch.ops.aten.view.default(mm_146, [8, 196, 320]);  mm_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_735: "f32[8, 196, 5, 64]" = torch.ops.aten.view.default(view_734, [8, 196, 5, 64]);  view_734 = None
    permute_658: "f32[8, 5, 196, 64]" = torch.ops.aten.permute.default(view_735, [0, 2, 1, 3]);  view_735 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_backward_14 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_658, permute_145, getitem_166, getitem_167, None, alias_42, getitem_169, getitem_170, getitem_171, 0.0, [True, True, True, False]);  permute_658 = permute_145 = getitem_166 = getitem_167 = alias_42 = getitem_169 = getitem_170 = getitem_171 = None
    getitem_435: "f32[8, 5, 196, 64]" = _scaled_dot_product_efficient_attention_backward_14[0]
    getitem_436: "f32[8, 5, 49, 64]" = _scaled_dot_product_efficient_attention_backward_14[1]
    getitem_437: "f32[8, 5, 49, 64]" = _scaled_dot_product_efficient_attention_backward_14[2];  _scaled_dot_product_efficient_attention_backward_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    cat_14: "f32[16, 5, 49, 64]" = torch.ops.aten.cat.default([getitem_436, getitem_437]);  getitem_436 = getitem_437 = None
    view_736: "f32[2, 8, 5, 49, 64]" = torch.ops.aten.view.default(cat_14, [2, 8, 5, 49, 64]);  cat_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_659: "f32[8, 49, 2, 5, 64]" = torch.ops.aten.permute.default(view_736, [1, 3, 0, 2, 4]);  view_736 = None
    clone_256: "f32[8, 49, 2, 5, 64]" = torch.ops.aten.clone.default(permute_659, memory_format = torch.contiguous_format);  permute_659 = None
    view_737: "f32[8, 49, 640]" = torch.ops.aten.view.default(clone_256, [8, 49, 640]);  clone_256 = None
    view_738: "f32[392, 640]" = torch.ops.aten.view.default(view_737, [392, 640]);  view_737 = None
    mm_148: "f32[392, 320]" = torch.ops.aten.mm.default(view_738, permute_660);  permute_660 = None
    permute_661: "f32[640, 392]" = torch.ops.aten.permute.default(view_738, [1, 0])
    mm_149: "f32[640, 320]" = torch.ops.aten.mm.default(permute_661, view_214);  permute_661 = view_214 = None
    permute_662: "f32[320, 640]" = torch.ops.aten.permute.default(mm_149, [1, 0]);  mm_149 = None
    sum_256: "f32[1, 640]" = torch.ops.aten.sum.dim_IntList(view_738, [0], True);  view_738 = None
    view_739: "f32[640]" = torch.ops.aten.view.default(sum_256, [640]);  sum_256 = None
    permute_663: "f32[640, 320]" = torch.ops.aten.permute.default(permute_662, [1, 0]);  permute_662 = None
    view_740: "f32[8, 49, 320]" = torch.ops.aten.view.default(mm_148, [8, 49, 320]);  mm_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    mul_656: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(view_740, primals_259);  primals_259 = None
    mul_657: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_656, 320)
    sum_257: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_656, [2], True)
    mul_658: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_656, mul_125);  mul_656 = None
    sum_258: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_658, [2], True);  mul_658 = None
    mul_659: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_125, sum_258);  sum_258 = None
    sub_213: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(mul_657, sum_257);  mul_657 = sum_257 = None
    sub_214: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(sub_213, mul_659);  sub_213 = mul_659 = None
    mul_660: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(div_43, sub_214);  div_43 = sub_214 = None
    mul_661: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(view_740, mul_125);  mul_125 = None
    sum_259: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_661, [0, 1]);  mul_661 = None
    sum_260: "f32[320]" = torch.ops.aten.sum.dim_IntList(view_740, [0, 1]);  view_740 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    permute_664: "f32[8, 320, 49]" = torch.ops.aten.permute.default(mul_660, [0, 2, 1]);  mul_660 = None
    view_741: "f32[8, 320, 7, 7]" = torch.ops.aten.view.default(permute_664, [8, 320, 7, 7]);  permute_664 = None
    sum_261: "f32[320]" = torch.ops.aten.sum.dim_IntList(view_741, [0, 2, 3])
    convolution_backward_13 = torch.ops.aten.convolution_backward.default(view_741, view_212, primals_257, [320], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_741 = view_212 = primals_257 = None
    getitem_439: "f32[8, 320, 14, 14]" = convolution_backward_13[0]
    getitem_440: "f32[320, 320, 2, 2]" = convolution_backward_13[1];  convolution_backward_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    view_742: "f32[8, 320, 196]" = torch.ops.aten.view.default(getitem_439, [8, 320, 196]);  getitem_439 = None
    permute_665: "f32[8, 196, 320]" = torch.ops.aten.permute.default(view_742, [0, 2, 1]);  view_742 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_666: "f32[8, 196, 5, 64]" = torch.ops.aten.permute.default(getitem_435, [0, 2, 1, 3]);  getitem_435 = None
    view_743: "f32[8, 196, 320]" = torch.ops.aten.view.default(permute_666, [8, 196, 320]);  permute_666 = None
    view_744: "f32[1568, 320]" = torch.ops.aten.view.default(view_743, [1568, 320]);  view_743 = None
    mm_150: "f32[1568, 320]" = torch.ops.aten.mm.default(view_744, permute_667);  permute_667 = None
    permute_668: "f32[320, 1568]" = torch.ops.aten.permute.default(view_744, [1, 0])
    mm_151: "f32[320, 320]" = torch.ops.aten.mm.default(permute_668, view_209);  permute_668 = view_209 = None
    permute_669: "f32[320, 320]" = torch.ops.aten.permute.default(mm_151, [1, 0]);  mm_151 = None
    sum_262: "f32[1, 320]" = torch.ops.aten.sum.dim_IntList(view_744, [0], True);  view_744 = None
    view_745: "f32[320]" = torch.ops.aten.view.default(sum_262, [320]);  sum_262 = None
    permute_670: "f32[320, 320]" = torch.ops.aten.permute.default(permute_669, [1, 0]);  permute_669 = None
    view_746: "f32[8, 196, 320]" = torch.ops.aten.view.default(mm_150, [8, 196, 320]);  mm_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    add_334: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(permute_665, view_746);  permute_665 = view_746 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    clone_258: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_334, memory_format = torch.contiguous_format);  add_334 = None
    mul_663: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(clone_258, primals_253);  primals_253 = None
    mul_664: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_663, 320)
    sum_263: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_663, [2], True)
    mul_665: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_663, mul_123);  mul_663 = None
    sum_264: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_665, [2], True);  mul_665 = None
    mul_666: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_123, sum_264);  sum_264 = None
    sub_216: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(mul_664, sum_263);  mul_664 = sum_263 = None
    sub_217: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(sub_216, mul_666);  sub_216 = mul_666 = None
    mul_667: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(div_44, sub_217);  div_44 = sub_217 = None
    mul_668: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(clone_258, mul_123);  mul_123 = None
    sum_265: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_668, [0, 1]);  mul_668 = None
    sum_266: "f32[320]" = torch.ops.aten.sum.dim_IntList(clone_258, [0, 1]);  clone_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_335: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_333, mul_667);  add_333 = mul_667 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    clone_260: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_335, memory_format = torch.contiguous_format)
    view_747: "f32[1568, 320]" = torch.ops.aten.view.default(clone_260, [1568, 320]);  clone_260 = None
    mm_152: "f32[1568, 1280]" = torch.ops.aten.mm.default(view_747, permute_671);  permute_671 = None
    permute_672: "f32[320, 1568]" = torch.ops.aten.permute.default(view_747, [1, 0])
    mm_153: "f32[320, 1280]" = torch.ops.aten.mm.default(permute_672, view_207);  permute_672 = view_207 = None
    permute_673: "f32[1280, 320]" = torch.ops.aten.permute.default(mm_153, [1, 0]);  mm_153 = None
    sum_267: "f32[1, 320]" = torch.ops.aten.sum.dim_IntList(view_747, [0], True);  view_747 = None
    view_748: "f32[320]" = torch.ops.aten.view.default(sum_267, [320]);  sum_267 = None
    permute_674: "f32[320, 1280]" = torch.ops.aten.permute.default(permute_673, [1, 0]);  permute_673 = None
    view_749: "f32[8, 196, 1280]" = torch.ops.aten.view.default(mm_152, [8, 196, 1280]);  mm_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_670: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(add_124, 0.5);  add_124 = None
    mul_671: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_206, view_206)
    mul_672: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(mul_671, -0.5);  mul_671 = None
    exp_15: "f32[8, 196, 1280]" = torch.ops.aten.exp.default(mul_672);  mul_672 = None
    mul_673: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(exp_15, 0.3989422804014327);  exp_15 = None
    mul_674: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_206, mul_673);  view_206 = mul_673 = None
    add_337: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(mul_670, mul_674);  mul_670 = mul_674 = None
    mul_675: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_749, add_337);  view_749 = add_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_750: "f32[1568, 1280]" = torch.ops.aten.view.default(mul_675, [1568, 1280]);  mul_675 = None
    mm_154: "f32[1568, 320]" = torch.ops.aten.mm.default(view_750, permute_675);  permute_675 = None
    permute_676: "f32[1280, 1568]" = torch.ops.aten.permute.default(view_750, [1, 0])
    mm_155: "f32[1280, 320]" = torch.ops.aten.mm.default(permute_676, view_205);  permute_676 = view_205 = None
    permute_677: "f32[320, 1280]" = torch.ops.aten.permute.default(mm_155, [1, 0]);  mm_155 = None
    sum_268: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_750, [0], True);  view_750 = None
    view_751: "f32[1280]" = torch.ops.aten.view.default(sum_268, [1280]);  sum_268 = None
    permute_678: "f32[1280, 320]" = torch.ops.aten.permute.default(permute_677, [1, 0]);  permute_677 = None
    view_752: "f32[8, 196, 320]" = torch.ops.aten.view.default(mm_154, [8, 196, 320]);  mm_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    mul_677: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(view_752, primals_247);  primals_247 = None
    mul_678: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_677, 320)
    sum_269: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_677, [2], True)
    mul_679: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_677, mul_118);  mul_677 = None
    sum_270: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_679, [2], True);  mul_679 = None
    mul_680: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_118, sum_270);  sum_270 = None
    sub_219: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(mul_678, sum_269);  mul_678 = sum_269 = None
    sub_220: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(sub_219, mul_680);  sub_219 = mul_680 = None
    mul_681: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(div_45, sub_220);  div_45 = sub_220 = None
    mul_682: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(view_752, mul_118);  mul_118 = None
    sum_271: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_682, [0, 1]);  mul_682 = None
    sum_272: "f32[320]" = torch.ops.aten.sum.dim_IntList(view_752, [0, 1]);  view_752 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_338: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_335, mul_681);  add_335 = mul_681 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    clone_262: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_338, memory_format = torch.contiguous_format)
    view_753: "f32[1568, 320]" = torch.ops.aten.view.default(clone_262, [1568, 320]);  clone_262 = None
    mm_156: "f32[1568, 320]" = torch.ops.aten.mm.default(view_753, permute_679);  permute_679 = None
    permute_680: "f32[320, 1568]" = torch.ops.aten.permute.default(view_753, [1, 0])
    mm_157: "f32[320, 320]" = torch.ops.aten.mm.default(permute_680, view_203);  permute_680 = view_203 = None
    permute_681: "f32[320, 320]" = torch.ops.aten.permute.default(mm_157, [1, 0]);  mm_157 = None
    sum_273: "f32[1, 320]" = torch.ops.aten.sum.dim_IntList(view_753, [0], True);  view_753 = None
    view_754: "f32[320]" = torch.ops.aten.view.default(sum_273, [320]);  sum_273 = None
    permute_682: "f32[320, 320]" = torch.ops.aten.permute.default(permute_681, [1, 0]);  permute_681 = None
    view_755: "f32[8, 196, 320]" = torch.ops.aten.view.default(mm_156, [8, 196, 320]);  mm_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_756: "f32[8, 196, 5, 64]" = torch.ops.aten.view.default(view_755, [8, 196, 5, 64]);  view_755 = None
    permute_683: "f32[8, 5, 196, 64]" = torch.ops.aten.permute.default(view_756, [0, 2, 1, 3]);  view_756 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_backward_15 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_683, permute_135, getitem_154, getitem_155, None, alias_43, getitem_157, getitem_158, getitem_159, 0.0, [True, True, True, False]);  permute_683 = permute_135 = getitem_154 = getitem_155 = alias_43 = getitem_157 = getitem_158 = getitem_159 = None
    getitem_442: "f32[8, 5, 196, 64]" = _scaled_dot_product_efficient_attention_backward_15[0]
    getitem_443: "f32[8, 5, 49, 64]" = _scaled_dot_product_efficient_attention_backward_15[1]
    getitem_444: "f32[8, 5, 49, 64]" = _scaled_dot_product_efficient_attention_backward_15[2];  _scaled_dot_product_efficient_attention_backward_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    cat_15: "f32[16, 5, 49, 64]" = torch.ops.aten.cat.default([getitem_443, getitem_444]);  getitem_443 = getitem_444 = None
    view_757: "f32[2, 8, 5, 49, 64]" = torch.ops.aten.view.default(cat_15, [2, 8, 5, 49, 64]);  cat_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_684: "f32[8, 49, 2, 5, 64]" = torch.ops.aten.permute.default(view_757, [1, 3, 0, 2, 4]);  view_757 = None
    clone_263: "f32[8, 49, 2, 5, 64]" = torch.ops.aten.clone.default(permute_684, memory_format = torch.contiguous_format);  permute_684 = None
    view_758: "f32[8, 49, 640]" = torch.ops.aten.view.default(clone_263, [8, 49, 640]);  clone_263 = None
    view_759: "f32[392, 640]" = torch.ops.aten.view.default(view_758, [392, 640]);  view_758 = None
    mm_158: "f32[392, 320]" = torch.ops.aten.mm.default(view_759, permute_685);  permute_685 = None
    permute_686: "f32[640, 392]" = torch.ops.aten.permute.default(view_759, [1, 0])
    mm_159: "f32[640, 320]" = torch.ops.aten.mm.default(permute_686, view_199);  permute_686 = view_199 = None
    permute_687: "f32[320, 640]" = torch.ops.aten.permute.default(mm_159, [1, 0]);  mm_159 = None
    sum_274: "f32[1, 640]" = torch.ops.aten.sum.dim_IntList(view_759, [0], True);  view_759 = None
    view_760: "f32[640]" = torch.ops.aten.view.default(sum_274, [640]);  sum_274 = None
    permute_688: "f32[640, 320]" = torch.ops.aten.permute.default(permute_687, [1, 0]);  permute_687 = None
    view_761: "f32[8, 49, 320]" = torch.ops.aten.view.default(mm_158, [8, 49, 320]);  mm_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    mul_684: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(view_761, primals_241);  primals_241 = None
    mul_685: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_684, 320)
    sum_275: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_684, [2], True)
    mul_686: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_684, mul_116);  mul_684 = None
    sum_276: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_686, [2], True);  mul_686 = None
    mul_687: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_116, sum_276);  sum_276 = None
    sub_222: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(mul_685, sum_275);  mul_685 = sum_275 = None
    sub_223: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(sub_222, mul_687);  sub_222 = mul_687 = None
    mul_688: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(div_46, sub_223);  div_46 = sub_223 = None
    mul_689: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(view_761, mul_116);  mul_116 = None
    sum_277: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_689, [0, 1]);  mul_689 = None
    sum_278: "f32[320]" = torch.ops.aten.sum.dim_IntList(view_761, [0, 1]);  view_761 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    permute_689: "f32[8, 320, 49]" = torch.ops.aten.permute.default(mul_688, [0, 2, 1]);  mul_688 = None
    view_762: "f32[8, 320, 7, 7]" = torch.ops.aten.view.default(permute_689, [8, 320, 7, 7]);  permute_689 = None
    sum_279: "f32[320]" = torch.ops.aten.sum.dim_IntList(view_762, [0, 2, 3])
    convolution_backward_14 = torch.ops.aten.convolution_backward.default(view_762, view_197, primals_239, [320], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_762 = view_197 = primals_239 = None
    getitem_446: "f32[8, 320, 14, 14]" = convolution_backward_14[0]
    getitem_447: "f32[320, 320, 2, 2]" = convolution_backward_14[1];  convolution_backward_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    view_763: "f32[8, 320, 196]" = torch.ops.aten.view.default(getitem_446, [8, 320, 196]);  getitem_446 = None
    permute_690: "f32[8, 196, 320]" = torch.ops.aten.permute.default(view_763, [0, 2, 1]);  view_763 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_691: "f32[8, 196, 5, 64]" = torch.ops.aten.permute.default(getitem_442, [0, 2, 1, 3]);  getitem_442 = None
    view_764: "f32[8, 196, 320]" = torch.ops.aten.view.default(permute_691, [8, 196, 320]);  permute_691 = None
    view_765: "f32[1568, 320]" = torch.ops.aten.view.default(view_764, [1568, 320]);  view_764 = None
    mm_160: "f32[1568, 320]" = torch.ops.aten.mm.default(view_765, permute_692);  permute_692 = None
    permute_693: "f32[320, 1568]" = torch.ops.aten.permute.default(view_765, [1, 0])
    mm_161: "f32[320, 320]" = torch.ops.aten.mm.default(permute_693, view_194);  permute_693 = view_194 = None
    permute_694: "f32[320, 320]" = torch.ops.aten.permute.default(mm_161, [1, 0]);  mm_161 = None
    sum_280: "f32[1, 320]" = torch.ops.aten.sum.dim_IntList(view_765, [0], True);  view_765 = None
    view_766: "f32[320]" = torch.ops.aten.view.default(sum_280, [320]);  sum_280 = None
    permute_695: "f32[320, 320]" = torch.ops.aten.permute.default(permute_694, [1, 0]);  permute_694 = None
    view_767: "f32[8, 196, 320]" = torch.ops.aten.view.default(mm_160, [8, 196, 320]);  mm_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    add_339: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(permute_690, view_767);  permute_690 = view_767 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    clone_265: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_339, memory_format = torch.contiguous_format);  add_339 = None
    mul_691: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(clone_265, primals_235);  primals_235 = None
    mul_692: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_691, 320)
    sum_281: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_691, [2], True)
    mul_693: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_691, mul_114);  mul_691 = None
    sum_282: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_693, [2], True);  mul_693 = None
    mul_694: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_114, sum_282);  sum_282 = None
    sub_225: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(mul_692, sum_281);  mul_692 = sum_281 = None
    sub_226: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(sub_225, mul_694);  sub_225 = mul_694 = None
    mul_695: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(div_47, sub_226);  div_47 = sub_226 = None
    mul_696: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(clone_265, mul_114);  mul_114 = None
    sum_283: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_696, [0, 1]);  mul_696 = None
    sum_284: "f32[320]" = torch.ops.aten.sum.dim_IntList(clone_265, [0, 1]);  clone_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_340: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_338, mul_695);  add_338 = mul_695 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    clone_267: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_340, memory_format = torch.contiguous_format)
    view_768: "f32[1568, 320]" = torch.ops.aten.view.default(clone_267, [1568, 320]);  clone_267 = None
    mm_162: "f32[1568, 1280]" = torch.ops.aten.mm.default(view_768, permute_696);  permute_696 = None
    permute_697: "f32[320, 1568]" = torch.ops.aten.permute.default(view_768, [1, 0])
    mm_163: "f32[320, 1280]" = torch.ops.aten.mm.default(permute_697, view_192);  permute_697 = view_192 = None
    permute_698: "f32[1280, 320]" = torch.ops.aten.permute.default(mm_163, [1, 0]);  mm_163 = None
    sum_285: "f32[1, 320]" = torch.ops.aten.sum.dim_IntList(view_768, [0], True);  view_768 = None
    view_769: "f32[320]" = torch.ops.aten.view.default(sum_285, [320]);  sum_285 = None
    permute_699: "f32[320, 1280]" = torch.ops.aten.permute.default(permute_698, [1, 0]);  permute_698 = None
    view_770: "f32[8, 196, 1280]" = torch.ops.aten.view.default(mm_162, [8, 196, 1280]);  mm_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_698: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(add_115, 0.5);  add_115 = None
    mul_699: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_191, view_191)
    mul_700: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(mul_699, -0.5);  mul_699 = None
    exp_16: "f32[8, 196, 1280]" = torch.ops.aten.exp.default(mul_700);  mul_700 = None
    mul_701: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(exp_16, 0.3989422804014327);  exp_16 = None
    mul_702: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_191, mul_701);  view_191 = mul_701 = None
    add_342: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(mul_698, mul_702);  mul_698 = mul_702 = None
    mul_703: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_770, add_342);  view_770 = add_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_771: "f32[1568, 1280]" = torch.ops.aten.view.default(mul_703, [1568, 1280]);  mul_703 = None
    mm_164: "f32[1568, 320]" = torch.ops.aten.mm.default(view_771, permute_700);  permute_700 = None
    permute_701: "f32[1280, 1568]" = torch.ops.aten.permute.default(view_771, [1, 0])
    mm_165: "f32[1280, 320]" = torch.ops.aten.mm.default(permute_701, view_190);  permute_701 = view_190 = None
    permute_702: "f32[320, 1280]" = torch.ops.aten.permute.default(mm_165, [1, 0]);  mm_165 = None
    sum_286: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_771, [0], True);  view_771 = None
    view_772: "f32[1280]" = torch.ops.aten.view.default(sum_286, [1280]);  sum_286 = None
    permute_703: "f32[1280, 320]" = torch.ops.aten.permute.default(permute_702, [1, 0]);  permute_702 = None
    view_773: "f32[8, 196, 320]" = torch.ops.aten.view.default(mm_164, [8, 196, 320]);  mm_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    mul_705: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(view_773, primals_229);  primals_229 = None
    mul_706: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_705, 320)
    sum_287: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_705, [2], True)
    mul_707: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_705, mul_109);  mul_705 = None
    sum_288: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_707, [2], True);  mul_707 = None
    mul_708: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_109, sum_288);  sum_288 = None
    sub_228: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(mul_706, sum_287);  mul_706 = sum_287 = None
    sub_229: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(sub_228, mul_708);  sub_228 = mul_708 = None
    mul_709: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(div_48, sub_229);  div_48 = sub_229 = None
    mul_710: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(view_773, mul_109);  mul_109 = None
    sum_289: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_710, [0, 1]);  mul_710 = None
    sum_290: "f32[320]" = torch.ops.aten.sum.dim_IntList(view_773, [0, 1]);  view_773 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_343: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_340, mul_709);  add_340 = mul_709 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    clone_269: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_343, memory_format = torch.contiguous_format)
    view_774: "f32[1568, 320]" = torch.ops.aten.view.default(clone_269, [1568, 320]);  clone_269 = None
    mm_166: "f32[1568, 320]" = torch.ops.aten.mm.default(view_774, permute_704);  permute_704 = None
    permute_705: "f32[320, 1568]" = torch.ops.aten.permute.default(view_774, [1, 0])
    mm_167: "f32[320, 320]" = torch.ops.aten.mm.default(permute_705, view_188);  permute_705 = view_188 = None
    permute_706: "f32[320, 320]" = torch.ops.aten.permute.default(mm_167, [1, 0]);  mm_167 = None
    sum_291: "f32[1, 320]" = torch.ops.aten.sum.dim_IntList(view_774, [0], True);  view_774 = None
    view_775: "f32[320]" = torch.ops.aten.view.default(sum_291, [320]);  sum_291 = None
    permute_707: "f32[320, 320]" = torch.ops.aten.permute.default(permute_706, [1, 0]);  permute_706 = None
    view_776: "f32[8, 196, 320]" = torch.ops.aten.view.default(mm_166, [8, 196, 320]);  mm_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_777: "f32[8, 196, 5, 64]" = torch.ops.aten.view.default(view_776, [8, 196, 5, 64]);  view_776 = None
    permute_708: "f32[8, 5, 196, 64]" = torch.ops.aten.permute.default(view_777, [0, 2, 1, 3]);  view_777 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_backward_16 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_708, permute_125, getitem_142, getitem_143, None, alias_44, getitem_145, getitem_146, getitem_147, 0.0, [True, True, True, False]);  permute_708 = permute_125 = getitem_142 = getitem_143 = alias_44 = getitem_145 = getitem_146 = getitem_147 = None
    getitem_449: "f32[8, 5, 196, 64]" = _scaled_dot_product_efficient_attention_backward_16[0]
    getitem_450: "f32[8, 5, 49, 64]" = _scaled_dot_product_efficient_attention_backward_16[1]
    getitem_451: "f32[8, 5, 49, 64]" = _scaled_dot_product_efficient_attention_backward_16[2];  _scaled_dot_product_efficient_attention_backward_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    cat_16: "f32[16, 5, 49, 64]" = torch.ops.aten.cat.default([getitem_450, getitem_451]);  getitem_450 = getitem_451 = None
    view_778: "f32[2, 8, 5, 49, 64]" = torch.ops.aten.view.default(cat_16, [2, 8, 5, 49, 64]);  cat_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_709: "f32[8, 49, 2, 5, 64]" = torch.ops.aten.permute.default(view_778, [1, 3, 0, 2, 4]);  view_778 = None
    clone_270: "f32[8, 49, 2, 5, 64]" = torch.ops.aten.clone.default(permute_709, memory_format = torch.contiguous_format);  permute_709 = None
    view_779: "f32[8, 49, 640]" = torch.ops.aten.view.default(clone_270, [8, 49, 640]);  clone_270 = None
    view_780: "f32[392, 640]" = torch.ops.aten.view.default(view_779, [392, 640]);  view_779 = None
    mm_168: "f32[392, 320]" = torch.ops.aten.mm.default(view_780, permute_710);  permute_710 = None
    permute_711: "f32[640, 392]" = torch.ops.aten.permute.default(view_780, [1, 0])
    mm_169: "f32[640, 320]" = torch.ops.aten.mm.default(permute_711, view_184);  permute_711 = view_184 = None
    permute_712: "f32[320, 640]" = torch.ops.aten.permute.default(mm_169, [1, 0]);  mm_169 = None
    sum_292: "f32[1, 640]" = torch.ops.aten.sum.dim_IntList(view_780, [0], True);  view_780 = None
    view_781: "f32[640]" = torch.ops.aten.view.default(sum_292, [640]);  sum_292 = None
    permute_713: "f32[640, 320]" = torch.ops.aten.permute.default(permute_712, [1, 0]);  permute_712 = None
    view_782: "f32[8, 49, 320]" = torch.ops.aten.view.default(mm_168, [8, 49, 320]);  mm_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    mul_712: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(view_782, primals_223);  primals_223 = None
    mul_713: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_712, 320)
    sum_293: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_712, [2], True)
    mul_714: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_712, mul_107);  mul_712 = None
    sum_294: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_714, [2], True);  mul_714 = None
    mul_715: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_107, sum_294);  sum_294 = None
    sub_231: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(mul_713, sum_293);  mul_713 = sum_293 = None
    sub_232: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(sub_231, mul_715);  sub_231 = mul_715 = None
    mul_716: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(div_49, sub_232);  div_49 = sub_232 = None
    mul_717: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(view_782, mul_107);  mul_107 = None
    sum_295: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_717, [0, 1]);  mul_717 = None
    sum_296: "f32[320]" = torch.ops.aten.sum.dim_IntList(view_782, [0, 1]);  view_782 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    permute_714: "f32[8, 320, 49]" = torch.ops.aten.permute.default(mul_716, [0, 2, 1]);  mul_716 = None
    view_783: "f32[8, 320, 7, 7]" = torch.ops.aten.view.default(permute_714, [8, 320, 7, 7]);  permute_714 = None
    sum_297: "f32[320]" = torch.ops.aten.sum.dim_IntList(view_783, [0, 2, 3])
    convolution_backward_15 = torch.ops.aten.convolution_backward.default(view_783, view_182, primals_221, [320], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_783 = view_182 = primals_221 = None
    getitem_453: "f32[8, 320, 14, 14]" = convolution_backward_15[0]
    getitem_454: "f32[320, 320, 2, 2]" = convolution_backward_15[1];  convolution_backward_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    view_784: "f32[8, 320, 196]" = torch.ops.aten.view.default(getitem_453, [8, 320, 196]);  getitem_453 = None
    permute_715: "f32[8, 196, 320]" = torch.ops.aten.permute.default(view_784, [0, 2, 1]);  view_784 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_716: "f32[8, 196, 5, 64]" = torch.ops.aten.permute.default(getitem_449, [0, 2, 1, 3]);  getitem_449 = None
    view_785: "f32[8, 196, 320]" = torch.ops.aten.view.default(permute_716, [8, 196, 320]);  permute_716 = None
    view_786: "f32[1568, 320]" = torch.ops.aten.view.default(view_785, [1568, 320]);  view_785 = None
    mm_170: "f32[1568, 320]" = torch.ops.aten.mm.default(view_786, permute_717);  permute_717 = None
    permute_718: "f32[320, 1568]" = torch.ops.aten.permute.default(view_786, [1, 0])
    mm_171: "f32[320, 320]" = torch.ops.aten.mm.default(permute_718, view_179);  permute_718 = view_179 = None
    permute_719: "f32[320, 320]" = torch.ops.aten.permute.default(mm_171, [1, 0]);  mm_171 = None
    sum_298: "f32[1, 320]" = torch.ops.aten.sum.dim_IntList(view_786, [0], True);  view_786 = None
    view_787: "f32[320]" = torch.ops.aten.view.default(sum_298, [320]);  sum_298 = None
    permute_720: "f32[320, 320]" = torch.ops.aten.permute.default(permute_719, [1, 0]);  permute_719 = None
    view_788: "f32[8, 196, 320]" = torch.ops.aten.view.default(mm_170, [8, 196, 320]);  mm_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    add_344: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(permute_715, view_788);  permute_715 = view_788 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    clone_272: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_344, memory_format = torch.contiguous_format);  add_344 = None
    mul_719: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(clone_272, primals_217);  primals_217 = None
    mul_720: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_719, 320)
    sum_299: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_719, [2], True)
    mul_721: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_719, mul_105);  mul_719 = None
    sum_300: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_721, [2], True);  mul_721 = None
    mul_722: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_105, sum_300);  sum_300 = None
    sub_234: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(mul_720, sum_299);  mul_720 = sum_299 = None
    sub_235: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(sub_234, mul_722);  sub_234 = mul_722 = None
    mul_723: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(div_50, sub_235);  div_50 = sub_235 = None
    mul_724: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(clone_272, mul_105);  mul_105 = None
    sum_301: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_724, [0, 1]);  mul_724 = None
    sum_302: "f32[320]" = torch.ops.aten.sum.dim_IntList(clone_272, [0, 1]);  clone_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_345: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_343, mul_723);  add_343 = mul_723 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    clone_274: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_345, memory_format = torch.contiguous_format)
    view_789: "f32[1568, 320]" = torch.ops.aten.view.default(clone_274, [1568, 320]);  clone_274 = None
    mm_172: "f32[1568, 1280]" = torch.ops.aten.mm.default(view_789, permute_721);  permute_721 = None
    permute_722: "f32[320, 1568]" = torch.ops.aten.permute.default(view_789, [1, 0])
    mm_173: "f32[320, 1280]" = torch.ops.aten.mm.default(permute_722, view_177);  permute_722 = view_177 = None
    permute_723: "f32[1280, 320]" = torch.ops.aten.permute.default(mm_173, [1, 0]);  mm_173 = None
    sum_303: "f32[1, 320]" = torch.ops.aten.sum.dim_IntList(view_789, [0], True);  view_789 = None
    view_790: "f32[320]" = torch.ops.aten.view.default(sum_303, [320]);  sum_303 = None
    permute_724: "f32[320, 1280]" = torch.ops.aten.permute.default(permute_723, [1, 0]);  permute_723 = None
    view_791: "f32[8, 196, 1280]" = torch.ops.aten.view.default(mm_172, [8, 196, 1280]);  mm_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_726: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(add_106, 0.5);  add_106 = None
    mul_727: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_176, view_176)
    mul_728: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(mul_727, -0.5);  mul_727 = None
    exp_17: "f32[8, 196, 1280]" = torch.ops.aten.exp.default(mul_728);  mul_728 = None
    mul_729: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(exp_17, 0.3989422804014327);  exp_17 = None
    mul_730: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_176, mul_729);  view_176 = mul_729 = None
    add_347: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(mul_726, mul_730);  mul_726 = mul_730 = None
    mul_731: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_791, add_347);  view_791 = add_347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_792: "f32[1568, 1280]" = torch.ops.aten.view.default(mul_731, [1568, 1280]);  mul_731 = None
    mm_174: "f32[1568, 320]" = torch.ops.aten.mm.default(view_792, permute_725);  permute_725 = None
    permute_726: "f32[1280, 1568]" = torch.ops.aten.permute.default(view_792, [1, 0])
    mm_175: "f32[1280, 320]" = torch.ops.aten.mm.default(permute_726, view_175);  permute_726 = view_175 = None
    permute_727: "f32[320, 1280]" = torch.ops.aten.permute.default(mm_175, [1, 0]);  mm_175 = None
    sum_304: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_792, [0], True);  view_792 = None
    view_793: "f32[1280]" = torch.ops.aten.view.default(sum_304, [1280]);  sum_304 = None
    permute_728: "f32[1280, 320]" = torch.ops.aten.permute.default(permute_727, [1, 0]);  permute_727 = None
    view_794: "f32[8, 196, 320]" = torch.ops.aten.view.default(mm_174, [8, 196, 320]);  mm_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    mul_733: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(view_794, primals_211);  primals_211 = None
    mul_734: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_733, 320)
    sum_305: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_733, [2], True)
    mul_735: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_733, mul_100);  mul_733 = None
    sum_306: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_735, [2], True);  mul_735 = None
    mul_736: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_100, sum_306);  sum_306 = None
    sub_237: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(mul_734, sum_305);  mul_734 = sum_305 = None
    sub_238: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(sub_237, mul_736);  sub_237 = mul_736 = None
    mul_737: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(div_51, sub_238);  div_51 = sub_238 = None
    mul_738: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(view_794, mul_100);  mul_100 = None
    sum_307: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_738, [0, 1]);  mul_738 = None
    sum_308: "f32[320]" = torch.ops.aten.sum.dim_IntList(view_794, [0, 1]);  view_794 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_348: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_345, mul_737);  add_345 = mul_737 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    clone_276: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_348, memory_format = torch.contiguous_format)
    view_795: "f32[1568, 320]" = torch.ops.aten.view.default(clone_276, [1568, 320]);  clone_276 = None
    mm_176: "f32[1568, 320]" = torch.ops.aten.mm.default(view_795, permute_729);  permute_729 = None
    permute_730: "f32[320, 1568]" = torch.ops.aten.permute.default(view_795, [1, 0])
    mm_177: "f32[320, 320]" = torch.ops.aten.mm.default(permute_730, view_173);  permute_730 = view_173 = None
    permute_731: "f32[320, 320]" = torch.ops.aten.permute.default(mm_177, [1, 0]);  mm_177 = None
    sum_309: "f32[1, 320]" = torch.ops.aten.sum.dim_IntList(view_795, [0], True);  view_795 = None
    view_796: "f32[320]" = torch.ops.aten.view.default(sum_309, [320]);  sum_309 = None
    permute_732: "f32[320, 320]" = torch.ops.aten.permute.default(permute_731, [1, 0]);  permute_731 = None
    view_797: "f32[8, 196, 320]" = torch.ops.aten.view.default(mm_176, [8, 196, 320]);  mm_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_798: "f32[8, 196, 5, 64]" = torch.ops.aten.view.default(view_797, [8, 196, 5, 64]);  view_797 = None
    permute_733: "f32[8, 5, 196, 64]" = torch.ops.aten.permute.default(view_798, [0, 2, 1, 3]);  view_798 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_backward_17 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_733, permute_115, getitem_130, getitem_131, None, alias_45, getitem_133, getitem_134, getitem_135, 0.0, [True, True, True, False]);  permute_733 = permute_115 = getitem_130 = getitem_131 = alias_45 = getitem_133 = getitem_134 = getitem_135 = None
    getitem_456: "f32[8, 5, 196, 64]" = _scaled_dot_product_efficient_attention_backward_17[0]
    getitem_457: "f32[8, 5, 49, 64]" = _scaled_dot_product_efficient_attention_backward_17[1]
    getitem_458: "f32[8, 5, 49, 64]" = _scaled_dot_product_efficient_attention_backward_17[2];  _scaled_dot_product_efficient_attention_backward_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    cat_17: "f32[16, 5, 49, 64]" = torch.ops.aten.cat.default([getitem_457, getitem_458]);  getitem_457 = getitem_458 = None
    view_799: "f32[2, 8, 5, 49, 64]" = torch.ops.aten.view.default(cat_17, [2, 8, 5, 49, 64]);  cat_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_734: "f32[8, 49, 2, 5, 64]" = torch.ops.aten.permute.default(view_799, [1, 3, 0, 2, 4]);  view_799 = None
    clone_277: "f32[8, 49, 2, 5, 64]" = torch.ops.aten.clone.default(permute_734, memory_format = torch.contiguous_format);  permute_734 = None
    view_800: "f32[8, 49, 640]" = torch.ops.aten.view.default(clone_277, [8, 49, 640]);  clone_277 = None
    view_801: "f32[392, 640]" = torch.ops.aten.view.default(view_800, [392, 640]);  view_800 = None
    mm_178: "f32[392, 320]" = torch.ops.aten.mm.default(view_801, permute_735);  permute_735 = None
    permute_736: "f32[640, 392]" = torch.ops.aten.permute.default(view_801, [1, 0])
    mm_179: "f32[640, 320]" = torch.ops.aten.mm.default(permute_736, view_169);  permute_736 = view_169 = None
    permute_737: "f32[320, 640]" = torch.ops.aten.permute.default(mm_179, [1, 0]);  mm_179 = None
    sum_310: "f32[1, 640]" = torch.ops.aten.sum.dim_IntList(view_801, [0], True);  view_801 = None
    view_802: "f32[640]" = torch.ops.aten.view.default(sum_310, [640]);  sum_310 = None
    permute_738: "f32[640, 320]" = torch.ops.aten.permute.default(permute_737, [1, 0]);  permute_737 = None
    view_803: "f32[8, 49, 320]" = torch.ops.aten.view.default(mm_178, [8, 49, 320]);  mm_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    mul_740: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(view_803, primals_205);  primals_205 = None
    mul_741: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_740, 320)
    sum_311: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_740, [2], True)
    mul_742: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_740, mul_98);  mul_740 = None
    sum_312: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_742, [2], True);  mul_742 = None
    mul_743: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_98, sum_312);  sum_312 = None
    sub_240: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(mul_741, sum_311);  mul_741 = sum_311 = None
    sub_241: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(sub_240, mul_743);  sub_240 = mul_743 = None
    mul_744: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(div_52, sub_241);  div_52 = sub_241 = None
    mul_745: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(view_803, mul_98);  mul_98 = None
    sum_313: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_745, [0, 1]);  mul_745 = None
    sum_314: "f32[320]" = torch.ops.aten.sum.dim_IntList(view_803, [0, 1]);  view_803 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    permute_739: "f32[8, 320, 49]" = torch.ops.aten.permute.default(mul_744, [0, 2, 1]);  mul_744 = None
    view_804: "f32[8, 320, 7, 7]" = torch.ops.aten.view.default(permute_739, [8, 320, 7, 7]);  permute_739 = None
    sum_315: "f32[320]" = torch.ops.aten.sum.dim_IntList(view_804, [0, 2, 3])
    convolution_backward_16 = torch.ops.aten.convolution_backward.default(view_804, view_167, primals_203, [320], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_804 = view_167 = primals_203 = None
    getitem_460: "f32[8, 320, 14, 14]" = convolution_backward_16[0]
    getitem_461: "f32[320, 320, 2, 2]" = convolution_backward_16[1];  convolution_backward_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    view_805: "f32[8, 320, 196]" = torch.ops.aten.view.default(getitem_460, [8, 320, 196]);  getitem_460 = None
    permute_740: "f32[8, 196, 320]" = torch.ops.aten.permute.default(view_805, [0, 2, 1]);  view_805 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_741: "f32[8, 196, 5, 64]" = torch.ops.aten.permute.default(getitem_456, [0, 2, 1, 3]);  getitem_456 = None
    view_806: "f32[8, 196, 320]" = torch.ops.aten.view.default(permute_741, [8, 196, 320]);  permute_741 = None
    view_807: "f32[1568, 320]" = torch.ops.aten.view.default(view_806, [1568, 320]);  view_806 = None
    mm_180: "f32[1568, 320]" = torch.ops.aten.mm.default(view_807, permute_742);  permute_742 = None
    permute_743: "f32[320, 1568]" = torch.ops.aten.permute.default(view_807, [1, 0])
    mm_181: "f32[320, 320]" = torch.ops.aten.mm.default(permute_743, view_164);  permute_743 = view_164 = None
    permute_744: "f32[320, 320]" = torch.ops.aten.permute.default(mm_181, [1, 0]);  mm_181 = None
    sum_316: "f32[1, 320]" = torch.ops.aten.sum.dim_IntList(view_807, [0], True);  view_807 = None
    view_808: "f32[320]" = torch.ops.aten.view.default(sum_316, [320]);  sum_316 = None
    permute_745: "f32[320, 320]" = torch.ops.aten.permute.default(permute_744, [1, 0]);  permute_744 = None
    view_809: "f32[8, 196, 320]" = torch.ops.aten.view.default(mm_180, [8, 196, 320]);  mm_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    add_349: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(permute_740, view_809);  permute_740 = view_809 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    clone_279: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_349, memory_format = torch.contiguous_format);  add_349 = None
    mul_747: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(clone_279, primals_199);  primals_199 = None
    mul_748: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_747, 320)
    sum_317: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_747, [2], True)
    mul_749: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_747, mul_96);  mul_747 = None
    sum_318: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_749, [2], True);  mul_749 = None
    mul_750: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_96, sum_318);  sum_318 = None
    sub_243: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(mul_748, sum_317);  mul_748 = sum_317 = None
    sub_244: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(sub_243, mul_750);  sub_243 = mul_750 = None
    mul_751: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(div_53, sub_244);  div_53 = sub_244 = None
    mul_752: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(clone_279, mul_96);  mul_96 = None
    sum_319: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_752, [0, 1]);  mul_752 = None
    sum_320: "f32[320]" = torch.ops.aten.sum.dim_IntList(clone_279, [0, 1]);  clone_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_350: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_348, mul_751);  add_348 = mul_751 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    clone_281: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_350, memory_format = torch.contiguous_format)
    view_810: "f32[1568, 320]" = torch.ops.aten.view.default(clone_281, [1568, 320]);  clone_281 = None
    mm_182: "f32[1568, 1280]" = torch.ops.aten.mm.default(view_810, permute_746);  permute_746 = None
    permute_747: "f32[320, 1568]" = torch.ops.aten.permute.default(view_810, [1, 0])
    mm_183: "f32[320, 1280]" = torch.ops.aten.mm.default(permute_747, view_162);  permute_747 = view_162 = None
    permute_748: "f32[1280, 320]" = torch.ops.aten.permute.default(mm_183, [1, 0]);  mm_183 = None
    sum_321: "f32[1, 320]" = torch.ops.aten.sum.dim_IntList(view_810, [0], True);  view_810 = None
    view_811: "f32[320]" = torch.ops.aten.view.default(sum_321, [320]);  sum_321 = None
    permute_749: "f32[320, 1280]" = torch.ops.aten.permute.default(permute_748, [1, 0]);  permute_748 = None
    view_812: "f32[8, 196, 1280]" = torch.ops.aten.view.default(mm_182, [8, 196, 1280]);  mm_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_754: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(add_97, 0.5);  add_97 = None
    mul_755: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_161, view_161)
    mul_756: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(mul_755, -0.5);  mul_755 = None
    exp_18: "f32[8, 196, 1280]" = torch.ops.aten.exp.default(mul_756);  mul_756 = None
    mul_757: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(exp_18, 0.3989422804014327);  exp_18 = None
    mul_758: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_161, mul_757);  view_161 = mul_757 = None
    add_352: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(mul_754, mul_758);  mul_754 = mul_758 = None
    mul_759: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_812, add_352);  view_812 = add_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_813: "f32[1568, 1280]" = torch.ops.aten.view.default(mul_759, [1568, 1280]);  mul_759 = None
    mm_184: "f32[1568, 320]" = torch.ops.aten.mm.default(view_813, permute_750);  permute_750 = None
    permute_751: "f32[1280, 1568]" = torch.ops.aten.permute.default(view_813, [1, 0])
    mm_185: "f32[1280, 320]" = torch.ops.aten.mm.default(permute_751, view_160);  permute_751 = view_160 = None
    permute_752: "f32[320, 1280]" = torch.ops.aten.permute.default(mm_185, [1, 0]);  mm_185 = None
    sum_322: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_813, [0], True);  view_813 = None
    view_814: "f32[1280]" = torch.ops.aten.view.default(sum_322, [1280]);  sum_322 = None
    permute_753: "f32[1280, 320]" = torch.ops.aten.permute.default(permute_752, [1, 0]);  permute_752 = None
    view_815: "f32[8, 196, 320]" = torch.ops.aten.view.default(mm_184, [8, 196, 320]);  mm_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    mul_761: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(view_815, primals_193);  primals_193 = None
    mul_762: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_761, 320)
    sum_323: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_761, [2], True)
    mul_763: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_761, mul_91);  mul_761 = None
    sum_324: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_763, [2], True);  mul_763 = None
    mul_764: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_91, sum_324);  sum_324 = None
    sub_246: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(mul_762, sum_323);  mul_762 = sum_323 = None
    sub_247: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(sub_246, mul_764);  sub_246 = mul_764 = None
    mul_765: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(div_54, sub_247);  div_54 = sub_247 = None
    mul_766: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(view_815, mul_91);  mul_91 = None
    sum_325: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_766, [0, 1]);  mul_766 = None
    sum_326: "f32[320]" = torch.ops.aten.sum.dim_IntList(view_815, [0, 1]);  view_815 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_353: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_350, mul_765);  add_350 = mul_765 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    clone_283: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_353, memory_format = torch.contiguous_format)
    view_816: "f32[1568, 320]" = torch.ops.aten.view.default(clone_283, [1568, 320]);  clone_283 = None
    mm_186: "f32[1568, 320]" = torch.ops.aten.mm.default(view_816, permute_754);  permute_754 = None
    permute_755: "f32[320, 1568]" = torch.ops.aten.permute.default(view_816, [1, 0])
    mm_187: "f32[320, 320]" = torch.ops.aten.mm.default(permute_755, view_158);  permute_755 = view_158 = None
    permute_756: "f32[320, 320]" = torch.ops.aten.permute.default(mm_187, [1, 0]);  mm_187 = None
    sum_327: "f32[1, 320]" = torch.ops.aten.sum.dim_IntList(view_816, [0], True);  view_816 = None
    view_817: "f32[320]" = torch.ops.aten.view.default(sum_327, [320]);  sum_327 = None
    permute_757: "f32[320, 320]" = torch.ops.aten.permute.default(permute_756, [1, 0]);  permute_756 = None
    view_818: "f32[8, 196, 320]" = torch.ops.aten.view.default(mm_186, [8, 196, 320]);  mm_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_819: "f32[8, 196, 5, 64]" = torch.ops.aten.view.default(view_818, [8, 196, 5, 64]);  view_818 = None
    permute_758: "f32[8, 5, 196, 64]" = torch.ops.aten.permute.default(view_819, [0, 2, 1, 3]);  view_819 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_backward_18 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_758, permute_105, getitem_118, getitem_119, None, alias_46, getitem_121, getitem_122, getitem_123, 0.0, [True, True, True, False]);  permute_758 = permute_105 = getitem_118 = getitem_119 = alias_46 = getitem_121 = getitem_122 = getitem_123 = None
    getitem_463: "f32[8, 5, 196, 64]" = _scaled_dot_product_efficient_attention_backward_18[0]
    getitem_464: "f32[8, 5, 49, 64]" = _scaled_dot_product_efficient_attention_backward_18[1]
    getitem_465: "f32[8, 5, 49, 64]" = _scaled_dot_product_efficient_attention_backward_18[2];  _scaled_dot_product_efficient_attention_backward_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    cat_18: "f32[16, 5, 49, 64]" = torch.ops.aten.cat.default([getitem_464, getitem_465]);  getitem_464 = getitem_465 = None
    view_820: "f32[2, 8, 5, 49, 64]" = torch.ops.aten.view.default(cat_18, [2, 8, 5, 49, 64]);  cat_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_759: "f32[8, 49, 2, 5, 64]" = torch.ops.aten.permute.default(view_820, [1, 3, 0, 2, 4]);  view_820 = None
    clone_284: "f32[8, 49, 2, 5, 64]" = torch.ops.aten.clone.default(permute_759, memory_format = torch.contiguous_format);  permute_759 = None
    view_821: "f32[8, 49, 640]" = torch.ops.aten.view.default(clone_284, [8, 49, 640]);  clone_284 = None
    view_822: "f32[392, 640]" = torch.ops.aten.view.default(view_821, [392, 640]);  view_821 = None
    mm_188: "f32[392, 320]" = torch.ops.aten.mm.default(view_822, permute_760);  permute_760 = None
    permute_761: "f32[640, 392]" = torch.ops.aten.permute.default(view_822, [1, 0])
    mm_189: "f32[640, 320]" = torch.ops.aten.mm.default(permute_761, view_154);  permute_761 = view_154 = None
    permute_762: "f32[320, 640]" = torch.ops.aten.permute.default(mm_189, [1, 0]);  mm_189 = None
    sum_328: "f32[1, 640]" = torch.ops.aten.sum.dim_IntList(view_822, [0], True);  view_822 = None
    view_823: "f32[640]" = torch.ops.aten.view.default(sum_328, [640]);  sum_328 = None
    permute_763: "f32[640, 320]" = torch.ops.aten.permute.default(permute_762, [1, 0]);  permute_762 = None
    view_824: "f32[8, 49, 320]" = torch.ops.aten.view.default(mm_188, [8, 49, 320]);  mm_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    mul_768: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(view_824, primals_187);  primals_187 = None
    mul_769: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_768, 320)
    sum_329: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_768, [2], True)
    mul_770: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_768, mul_89);  mul_768 = None
    sum_330: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_770, [2], True);  mul_770 = None
    mul_771: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_89, sum_330);  sum_330 = None
    sub_249: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(mul_769, sum_329);  mul_769 = sum_329 = None
    sub_250: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(sub_249, mul_771);  sub_249 = mul_771 = None
    mul_772: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(div_55, sub_250);  div_55 = sub_250 = None
    mul_773: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(view_824, mul_89);  mul_89 = None
    sum_331: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_773, [0, 1]);  mul_773 = None
    sum_332: "f32[320]" = torch.ops.aten.sum.dim_IntList(view_824, [0, 1]);  view_824 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    permute_764: "f32[8, 320, 49]" = torch.ops.aten.permute.default(mul_772, [0, 2, 1]);  mul_772 = None
    view_825: "f32[8, 320, 7, 7]" = torch.ops.aten.view.default(permute_764, [8, 320, 7, 7]);  permute_764 = None
    sum_333: "f32[320]" = torch.ops.aten.sum.dim_IntList(view_825, [0, 2, 3])
    convolution_backward_17 = torch.ops.aten.convolution_backward.default(view_825, view_152, primals_185, [320], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_825 = view_152 = primals_185 = None
    getitem_467: "f32[8, 320, 14, 14]" = convolution_backward_17[0]
    getitem_468: "f32[320, 320, 2, 2]" = convolution_backward_17[1];  convolution_backward_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    view_826: "f32[8, 320, 196]" = torch.ops.aten.view.default(getitem_467, [8, 320, 196]);  getitem_467 = None
    permute_765: "f32[8, 196, 320]" = torch.ops.aten.permute.default(view_826, [0, 2, 1]);  view_826 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_766: "f32[8, 196, 5, 64]" = torch.ops.aten.permute.default(getitem_463, [0, 2, 1, 3]);  getitem_463 = None
    view_827: "f32[8, 196, 320]" = torch.ops.aten.view.default(permute_766, [8, 196, 320]);  permute_766 = None
    view_828: "f32[1568, 320]" = torch.ops.aten.view.default(view_827, [1568, 320]);  view_827 = None
    mm_190: "f32[1568, 320]" = torch.ops.aten.mm.default(view_828, permute_767);  permute_767 = None
    permute_768: "f32[320, 1568]" = torch.ops.aten.permute.default(view_828, [1, 0])
    mm_191: "f32[320, 320]" = torch.ops.aten.mm.default(permute_768, view_149);  permute_768 = view_149 = None
    permute_769: "f32[320, 320]" = torch.ops.aten.permute.default(mm_191, [1, 0]);  mm_191 = None
    sum_334: "f32[1, 320]" = torch.ops.aten.sum.dim_IntList(view_828, [0], True);  view_828 = None
    view_829: "f32[320]" = torch.ops.aten.view.default(sum_334, [320]);  sum_334 = None
    permute_770: "f32[320, 320]" = torch.ops.aten.permute.default(permute_769, [1, 0]);  permute_769 = None
    view_830: "f32[8, 196, 320]" = torch.ops.aten.view.default(mm_190, [8, 196, 320]);  mm_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    add_354: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(permute_765, view_830);  permute_765 = view_830 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    clone_286: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_354, memory_format = torch.contiguous_format);  add_354 = None
    mul_775: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(clone_286, primals_181);  primals_181 = None
    mul_776: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_775, 320)
    sum_335: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_775, [2], True)
    mul_777: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_775, mul_87);  mul_775 = None
    sum_336: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_777, [2], True);  mul_777 = None
    mul_778: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_87, sum_336);  sum_336 = None
    sub_252: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(mul_776, sum_335);  mul_776 = sum_335 = None
    sub_253: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(sub_252, mul_778);  sub_252 = mul_778 = None
    mul_779: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(div_56, sub_253);  div_56 = sub_253 = None
    mul_780: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(clone_286, mul_87);  mul_87 = None
    sum_337: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_780, [0, 1]);  mul_780 = None
    sum_338: "f32[320]" = torch.ops.aten.sum.dim_IntList(clone_286, [0, 1]);  clone_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_355: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_353, mul_779);  add_353 = mul_779 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    clone_288: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_355, memory_format = torch.contiguous_format)
    view_831: "f32[1568, 320]" = torch.ops.aten.view.default(clone_288, [1568, 320]);  clone_288 = None
    mm_192: "f32[1568, 1280]" = torch.ops.aten.mm.default(view_831, permute_771);  permute_771 = None
    permute_772: "f32[320, 1568]" = torch.ops.aten.permute.default(view_831, [1, 0])
    mm_193: "f32[320, 1280]" = torch.ops.aten.mm.default(permute_772, view_147);  permute_772 = view_147 = None
    permute_773: "f32[1280, 320]" = torch.ops.aten.permute.default(mm_193, [1, 0]);  mm_193 = None
    sum_339: "f32[1, 320]" = torch.ops.aten.sum.dim_IntList(view_831, [0], True);  view_831 = None
    view_832: "f32[320]" = torch.ops.aten.view.default(sum_339, [320]);  sum_339 = None
    permute_774: "f32[320, 1280]" = torch.ops.aten.permute.default(permute_773, [1, 0]);  permute_773 = None
    view_833: "f32[8, 196, 1280]" = torch.ops.aten.view.default(mm_192, [8, 196, 1280]);  mm_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_782: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(add_88, 0.5);  add_88 = None
    mul_783: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_146, view_146)
    mul_784: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(mul_783, -0.5);  mul_783 = None
    exp_19: "f32[8, 196, 1280]" = torch.ops.aten.exp.default(mul_784);  mul_784 = None
    mul_785: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(exp_19, 0.3989422804014327);  exp_19 = None
    mul_786: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_146, mul_785);  view_146 = mul_785 = None
    add_357: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(mul_782, mul_786);  mul_782 = mul_786 = None
    mul_787: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_833, add_357);  view_833 = add_357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_834: "f32[1568, 1280]" = torch.ops.aten.view.default(mul_787, [1568, 1280]);  mul_787 = None
    mm_194: "f32[1568, 320]" = torch.ops.aten.mm.default(view_834, permute_775);  permute_775 = None
    permute_776: "f32[1280, 1568]" = torch.ops.aten.permute.default(view_834, [1, 0])
    mm_195: "f32[1280, 320]" = torch.ops.aten.mm.default(permute_776, view_145);  permute_776 = view_145 = None
    permute_777: "f32[320, 1280]" = torch.ops.aten.permute.default(mm_195, [1, 0]);  mm_195 = None
    sum_340: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_834, [0], True);  view_834 = None
    view_835: "f32[1280]" = torch.ops.aten.view.default(sum_340, [1280]);  sum_340 = None
    permute_778: "f32[1280, 320]" = torch.ops.aten.permute.default(permute_777, [1, 0]);  permute_777 = None
    view_836: "f32[8, 196, 320]" = torch.ops.aten.view.default(mm_194, [8, 196, 320]);  mm_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    mul_789: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(view_836, primals_175);  primals_175 = None
    mul_790: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_789, 320)
    sum_341: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_789, [2], True)
    mul_791: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_789, mul_82);  mul_789 = None
    sum_342: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_791, [2], True);  mul_791 = None
    mul_792: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_82, sum_342);  sum_342 = None
    sub_255: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(mul_790, sum_341);  mul_790 = sum_341 = None
    sub_256: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(sub_255, mul_792);  sub_255 = mul_792 = None
    mul_793: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(div_57, sub_256);  div_57 = sub_256 = None
    mul_794: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(view_836, mul_82);  mul_82 = None
    sum_343: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_794, [0, 1]);  mul_794 = None
    sum_344: "f32[320]" = torch.ops.aten.sum.dim_IntList(view_836, [0, 1]);  view_836 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_358: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_355, mul_793);  add_355 = mul_793 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    clone_290: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_358, memory_format = torch.contiguous_format)
    view_837: "f32[1568, 320]" = torch.ops.aten.view.default(clone_290, [1568, 320]);  clone_290 = None
    mm_196: "f32[1568, 320]" = torch.ops.aten.mm.default(view_837, permute_779);  permute_779 = None
    permute_780: "f32[320, 1568]" = torch.ops.aten.permute.default(view_837, [1, 0])
    mm_197: "f32[320, 320]" = torch.ops.aten.mm.default(permute_780, view_143);  permute_780 = view_143 = None
    permute_781: "f32[320, 320]" = torch.ops.aten.permute.default(mm_197, [1, 0]);  mm_197 = None
    sum_345: "f32[1, 320]" = torch.ops.aten.sum.dim_IntList(view_837, [0], True);  view_837 = None
    view_838: "f32[320]" = torch.ops.aten.view.default(sum_345, [320]);  sum_345 = None
    permute_782: "f32[320, 320]" = torch.ops.aten.permute.default(permute_781, [1, 0]);  permute_781 = None
    view_839: "f32[8, 196, 320]" = torch.ops.aten.view.default(mm_196, [8, 196, 320]);  mm_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_840: "f32[8, 196, 5, 64]" = torch.ops.aten.view.default(view_839, [8, 196, 5, 64]);  view_839 = None
    permute_783: "f32[8, 5, 196, 64]" = torch.ops.aten.permute.default(view_840, [0, 2, 1, 3]);  view_840 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_backward_19 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_783, permute_95, getitem_106, getitem_107, None, alias_47, getitem_109, getitem_110, getitem_111, 0.0, [True, True, True, False]);  permute_783 = permute_95 = getitem_106 = getitem_107 = alias_47 = getitem_109 = getitem_110 = getitem_111 = None
    getitem_470: "f32[8, 5, 196, 64]" = _scaled_dot_product_efficient_attention_backward_19[0]
    getitem_471: "f32[8, 5, 49, 64]" = _scaled_dot_product_efficient_attention_backward_19[1]
    getitem_472: "f32[8, 5, 49, 64]" = _scaled_dot_product_efficient_attention_backward_19[2];  _scaled_dot_product_efficient_attention_backward_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    cat_19: "f32[16, 5, 49, 64]" = torch.ops.aten.cat.default([getitem_471, getitem_472]);  getitem_471 = getitem_472 = None
    view_841: "f32[2, 8, 5, 49, 64]" = torch.ops.aten.view.default(cat_19, [2, 8, 5, 49, 64]);  cat_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_784: "f32[8, 49, 2, 5, 64]" = torch.ops.aten.permute.default(view_841, [1, 3, 0, 2, 4]);  view_841 = None
    clone_291: "f32[8, 49, 2, 5, 64]" = torch.ops.aten.clone.default(permute_784, memory_format = torch.contiguous_format);  permute_784 = None
    view_842: "f32[8, 49, 640]" = torch.ops.aten.view.default(clone_291, [8, 49, 640]);  clone_291 = None
    view_843: "f32[392, 640]" = torch.ops.aten.view.default(view_842, [392, 640]);  view_842 = None
    mm_198: "f32[392, 320]" = torch.ops.aten.mm.default(view_843, permute_785);  permute_785 = None
    permute_786: "f32[640, 392]" = torch.ops.aten.permute.default(view_843, [1, 0])
    mm_199: "f32[640, 320]" = torch.ops.aten.mm.default(permute_786, view_139);  permute_786 = view_139 = None
    permute_787: "f32[320, 640]" = torch.ops.aten.permute.default(mm_199, [1, 0]);  mm_199 = None
    sum_346: "f32[1, 640]" = torch.ops.aten.sum.dim_IntList(view_843, [0], True);  view_843 = None
    view_844: "f32[640]" = torch.ops.aten.view.default(sum_346, [640]);  sum_346 = None
    permute_788: "f32[640, 320]" = torch.ops.aten.permute.default(permute_787, [1, 0]);  permute_787 = None
    view_845: "f32[8, 49, 320]" = torch.ops.aten.view.default(mm_198, [8, 49, 320]);  mm_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    mul_796: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(view_845, primals_169);  primals_169 = None
    mul_797: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_796, 320)
    sum_347: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_796, [2], True)
    mul_798: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_796, mul_80);  mul_796 = None
    sum_348: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_798, [2], True);  mul_798 = None
    mul_799: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_80, sum_348);  sum_348 = None
    sub_258: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(mul_797, sum_347);  mul_797 = sum_347 = None
    sub_259: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(sub_258, mul_799);  sub_258 = mul_799 = None
    mul_800: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(div_58, sub_259);  div_58 = sub_259 = None
    mul_801: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(view_845, mul_80);  mul_80 = None
    sum_349: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_801, [0, 1]);  mul_801 = None
    sum_350: "f32[320]" = torch.ops.aten.sum.dim_IntList(view_845, [0, 1]);  view_845 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    permute_789: "f32[8, 320, 49]" = torch.ops.aten.permute.default(mul_800, [0, 2, 1]);  mul_800 = None
    view_846: "f32[8, 320, 7, 7]" = torch.ops.aten.view.default(permute_789, [8, 320, 7, 7]);  permute_789 = None
    sum_351: "f32[320]" = torch.ops.aten.sum.dim_IntList(view_846, [0, 2, 3])
    convolution_backward_18 = torch.ops.aten.convolution_backward.default(view_846, view_137, primals_167, [320], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_846 = view_137 = primals_167 = None
    getitem_474: "f32[8, 320, 14, 14]" = convolution_backward_18[0]
    getitem_475: "f32[320, 320, 2, 2]" = convolution_backward_18[1];  convolution_backward_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    view_847: "f32[8, 320, 196]" = torch.ops.aten.view.default(getitem_474, [8, 320, 196]);  getitem_474 = None
    permute_790: "f32[8, 196, 320]" = torch.ops.aten.permute.default(view_847, [0, 2, 1]);  view_847 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_791: "f32[8, 196, 5, 64]" = torch.ops.aten.permute.default(getitem_470, [0, 2, 1, 3]);  getitem_470 = None
    view_848: "f32[8, 196, 320]" = torch.ops.aten.view.default(permute_791, [8, 196, 320]);  permute_791 = None
    view_849: "f32[1568, 320]" = torch.ops.aten.view.default(view_848, [1568, 320]);  view_848 = None
    mm_200: "f32[1568, 320]" = torch.ops.aten.mm.default(view_849, permute_792);  permute_792 = None
    permute_793: "f32[320, 1568]" = torch.ops.aten.permute.default(view_849, [1, 0])
    mm_201: "f32[320, 320]" = torch.ops.aten.mm.default(permute_793, view_134);  permute_793 = view_134 = None
    permute_794: "f32[320, 320]" = torch.ops.aten.permute.default(mm_201, [1, 0]);  mm_201 = None
    sum_352: "f32[1, 320]" = torch.ops.aten.sum.dim_IntList(view_849, [0], True);  view_849 = None
    view_850: "f32[320]" = torch.ops.aten.view.default(sum_352, [320]);  sum_352 = None
    permute_795: "f32[320, 320]" = torch.ops.aten.permute.default(permute_794, [1, 0]);  permute_794 = None
    view_851: "f32[8, 196, 320]" = torch.ops.aten.view.default(mm_200, [8, 196, 320]);  mm_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    add_359: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(permute_790, view_851);  permute_790 = view_851 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    clone_293: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_359, memory_format = torch.contiguous_format);  add_359 = None
    mul_803: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(clone_293, primals_163);  primals_163 = None
    mul_804: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_803, 320)
    sum_353: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_803, [2], True)
    mul_805: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_803, mul_78);  mul_803 = None
    sum_354: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_805, [2], True);  mul_805 = None
    mul_806: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_78, sum_354);  sum_354 = None
    sub_261: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(mul_804, sum_353);  mul_804 = sum_353 = None
    sub_262: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(sub_261, mul_806);  sub_261 = mul_806 = None
    mul_807: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(div_59, sub_262);  div_59 = sub_262 = None
    mul_808: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(clone_293, mul_78);  mul_78 = None
    sum_355: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_808, [0, 1]);  mul_808 = None
    sum_356: "f32[320]" = torch.ops.aten.sum.dim_IntList(clone_293, [0, 1]);  clone_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_360: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_358, mul_807);  add_358 = mul_807 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:246, code: x = x.flatten(2).transpose(1, 2)
    permute_796: "f32[8, 320, 196]" = torch.ops.aten.permute.default(add_360, [0, 2, 1]);  add_360 = None
    view_852: "f32[8, 320, 14, 14]" = torch.ops.aten.view.default(permute_796, [8, 320, 14, 14]);  permute_796 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:243, code: x = self.proj(cnn_feat_token)
    sum_357: "f32[320]" = torch.ops.aten.sum.dim_IntList(view_852, [0, 2, 3])
    convolution_backward_19 = torch.ops.aten.convolution_backward.default(view_852, view_131, primals_161, [320], [1, 1], [1, 1], [1, 1], False, [0, 0], 320, [True, True, False]);  view_131 = primals_161 = None
    getitem_477: "f32[8, 320, 14, 14]" = convolution_backward_19[0]
    getitem_478: "f32[320, 1, 3, 3]" = convolution_backward_19[1];  convolution_backward_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:243, code: x = self.proj(cnn_feat_token)
    add_361: "f32[8, 320, 14, 14]" = torch.ops.aten.add.Tensor(view_852, getitem_477);  view_852 = getitem_477 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:242, code: cnn_feat_token = x.transpose(1, 2).view(B, C, *size)
    view_853: "f32[8, 320, 196]" = torch.ops.aten.view.default(add_361, [8, 320, 196]);  add_361 = None
    permute_797: "f32[8, 196, 320]" = torch.ops.aten.permute.default(view_853, [0, 2, 1]);  view_853 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    clone_295: "f32[8, 196, 320]" = torch.ops.aten.clone.default(permute_797, memory_format = torch.contiguous_format)
    view_854: "f32[1568, 320]" = torch.ops.aten.view.default(clone_295, [1568, 320]);  clone_295 = None
    mm_202: "f32[1568, 1280]" = torch.ops.aten.mm.default(view_854, permute_798);  permute_798 = None
    permute_799: "f32[320, 1568]" = torch.ops.aten.permute.default(view_854, [1, 0])
    mm_203: "f32[320, 1280]" = torch.ops.aten.mm.default(permute_799, view_129);  permute_799 = view_129 = None
    permute_800: "f32[1280, 320]" = torch.ops.aten.permute.default(mm_203, [1, 0]);  mm_203 = None
    sum_358: "f32[1, 320]" = torch.ops.aten.sum.dim_IntList(view_854, [0], True);  view_854 = None
    view_855: "f32[320]" = torch.ops.aten.view.default(sum_358, [320]);  sum_358 = None
    permute_801: "f32[320, 1280]" = torch.ops.aten.permute.default(permute_800, [1, 0]);  permute_800 = None
    view_856: "f32[8, 196, 1280]" = torch.ops.aten.view.default(mm_202, [8, 196, 1280]);  mm_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_810: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(add_78, 0.5);  add_78 = None
    mul_811: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_128, view_128)
    mul_812: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(mul_811, -0.5);  mul_811 = None
    exp_20: "f32[8, 196, 1280]" = torch.ops.aten.exp.default(mul_812);  mul_812 = None
    mul_813: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(exp_20, 0.3989422804014327);  exp_20 = None
    mul_814: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_128, mul_813);  view_128 = mul_813 = None
    add_363: "f32[8, 196, 1280]" = torch.ops.aten.add.Tensor(mul_810, mul_814);  mul_810 = mul_814 = None
    mul_815: "f32[8, 196, 1280]" = torch.ops.aten.mul.Tensor(view_856, add_363);  view_856 = add_363 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_857: "f32[1568, 1280]" = torch.ops.aten.view.default(mul_815, [1568, 1280]);  mul_815 = None
    mm_204: "f32[1568, 320]" = torch.ops.aten.mm.default(view_857, permute_802);  permute_802 = None
    permute_803: "f32[1280, 1568]" = torch.ops.aten.permute.default(view_857, [1, 0])
    mm_205: "f32[1280, 320]" = torch.ops.aten.mm.default(permute_803, view_127);  permute_803 = view_127 = None
    permute_804: "f32[320, 1280]" = torch.ops.aten.permute.default(mm_205, [1, 0]);  mm_205 = None
    sum_359: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_857, [0], True);  view_857 = None
    view_858: "f32[1280]" = torch.ops.aten.view.default(sum_359, [1280]);  sum_359 = None
    permute_805: "f32[1280, 320]" = torch.ops.aten.permute.default(permute_804, [1, 0]);  permute_804 = None
    view_859: "f32[8, 196, 320]" = torch.ops.aten.view.default(mm_204, [8, 196, 320]);  mm_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    mul_817: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(view_859, primals_155);  primals_155 = None
    mul_818: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_817, 320)
    sum_360: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_817, [2], True)
    mul_819: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_817, mul_73);  mul_817 = None
    sum_361: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_819, [2], True);  mul_819 = None
    mul_820: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_73, sum_361);  sum_361 = None
    sub_264: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(mul_818, sum_360);  mul_818 = sum_360 = None
    sub_265: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(sub_264, mul_820);  sub_264 = mul_820 = None
    mul_821: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(div_60, sub_265);  div_60 = sub_265 = None
    mul_822: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(view_859, mul_73);  mul_73 = None
    sum_362: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_822, [0, 1]);  mul_822 = None
    sum_363: "f32[320]" = torch.ops.aten.sum.dim_IntList(view_859, [0, 1]);  view_859 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_364: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(permute_797, mul_821);  permute_797 = mul_821 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    clone_296: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_364, memory_format = torch.contiguous_format)
    view_860: "f32[1568, 320]" = torch.ops.aten.view.default(clone_296, [1568, 320]);  clone_296 = None
    mm_206: "f32[1568, 320]" = torch.ops.aten.mm.default(view_860, permute_806);  permute_806 = None
    permute_807: "f32[320, 1568]" = torch.ops.aten.permute.default(view_860, [1, 0])
    mm_207: "f32[320, 320]" = torch.ops.aten.mm.default(permute_807, view_125);  permute_807 = view_125 = None
    permute_808: "f32[320, 320]" = torch.ops.aten.permute.default(mm_207, [1, 0]);  mm_207 = None
    sum_364: "f32[1, 320]" = torch.ops.aten.sum.dim_IntList(view_860, [0], True);  view_860 = None
    view_861: "f32[320]" = torch.ops.aten.view.default(sum_364, [320]);  sum_364 = None
    permute_809: "f32[320, 320]" = torch.ops.aten.permute.default(permute_808, [1, 0]);  permute_808 = None
    view_862: "f32[8, 196, 320]" = torch.ops.aten.view.default(mm_206, [8, 196, 320]);  mm_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_863: "f32[8, 196, 5, 64]" = torch.ops.aten.view.default(view_862, [8, 196, 5, 64]);  view_862 = None
    permute_810: "f32[8, 5, 196, 64]" = torch.ops.aten.permute.default(view_863, [0, 2, 1, 3]);  view_863 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_backward_20 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_810, permute_82, getitem_94, getitem_95, None, alias_48, getitem_97, getitem_98, getitem_99, 0.0, [True, True, True, False]);  permute_810 = permute_82 = getitem_94 = getitem_95 = alias_48 = getitem_97 = getitem_98 = getitem_99 = None
    getitem_480: "f32[8, 5, 196, 64]" = _scaled_dot_product_efficient_attention_backward_20[0]
    getitem_481: "f32[8, 5, 49, 64]" = _scaled_dot_product_efficient_attention_backward_20[1]
    getitem_482: "f32[8, 5, 49, 64]" = _scaled_dot_product_efficient_attention_backward_20[2];  _scaled_dot_product_efficient_attention_backward_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    cat_20: "f32[16, 5, 49, 64]" = torch.ops.aten.cat.default([getitem_481, getitem_482]);  getitem_481 = getitem_482 = None
    view_864: "f32[2, 8, 5, 49, 64]" = torch.ops.aten.view.default(cat_20, [2, 8, 5, 49, 64]);  cat_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_811: "f32[8, 49, 2, 5, 64]" = torch.ops.aten.permute.default(view_864, [1, 3, 0, 2, 4]);  view_864 = None
    clone_297: "f32[8, 49, 2, 5, 64]" = torch.ops.aten.clone.default(permute_811, memory_format = torch.contiguous_format);  permute_811 = None
    view_865: "f32[8, 49, 640]" = torch.ops.aten.view.default(clone_297, [8, 49, 640]);  clone_297 = None
    view_866: "f32[392, 640]" = torch.ops.aten.view.default(view_865, [392, 640]);  view_865 = None
    mm_208: "f32[392, 320]" = torch.ops.aten.mm.default(view_866, permute_812);  permute_812 = None
    permute_813: "f32[640, 392]" = torch.ops.aten.permute.default(view_866, [1, 0])
    mm_209: "f32[640, 320]" = torch.ops.aten.mm.default(permute_813, view_121);  permute_813 = view_121 = None
    permute_814: "f32[320, 640]" = torch.ops.aten.permute.default(mm_209, [1, 0]);  mm_209 = None
    sum_365: "f32[1, 640]" = torch.ops.aten.sum.dim_IntList(view_866, [0], True);  view_866 = None
    view_867: "f32[640]" = torch.ops.aten.view.default(sum_365, [640]);  sum_365 = None
    permute_815: "f32[640, 320]" = torch.ops.aten.permute.default(permute_814, [1, 0]);  permute_814 = None
    view_868: "f32[8, 49, 320]" = torch.ops.aten.view.default(mm_208, [8, 49, 320]);  mm_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    mul_824: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(view_868, primals_149);  primals_149 = None
    mul_825: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_824, 320)
    sum_366: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_824, [2], True)
    mul_826: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_824, mul_71);  mul_824 = None
    sum_367: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_826, [2], True);  mul_826 = None
    mul_827: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(mul_71, sum_367);  sum_367 = None
    sub_267: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(mul_825, sum_366);  mul_825 = sum_366 = None
    sub_268: "f32[8, 49, 320]" = torch.ops.aten.sub.Tensor(sub_267, mul_827);  sub_267 = mul_827 = None
    mul_828: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(div_61, sub_268);  div_61 = sub_268 = None
    mul_829: "f32[8, 49, 320]" = torch.ops.aten.mul.Tensor(view_868, mul_71);  mul_71 = None
    sum_368: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_829, [0, 1]);  mul_829 = None
    sum_369: "f32[320]" = torch.ops.aten.sum.dim_IntList(view_868, [0, 1]);  view_868 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    permute_816: "f32[8, 320, 49]" = torch.ops.aten.permute.default(mul_828, [0, 2, 1]);  mul_828 = None
    view_869: "f32[8, 320, 7, 7]" = torch.ops.aten.view.default(permute_816, [8, 320, 7, 7]);  permute_816 = None
    sum_370: "f32[320]" = torch.ops.aten.sum.dim_IntList(view_869, [0, 2, 3])
    convolution_backward_20 = torch.ops.aten.convolution_backward.default(view_869, view_119, primals_147, [320], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_869 = view_119 = primals_147 = None
    getitem_484: "f32[8, 320, 14, 14]" = convolution_backward_20[0]
    getitem_485: "f32[320, 320, 2, 2]" = convolution_backward_20[1];  convolution_backward_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    view_870: "f32[8, 320, 196]" = torch.ops.aten.view.default(getitem_484, [8, 320, 196]);  getitem_484 = None
    permute_817: "f32[8, 196, 320]" = torch.ops.aten.permute.default(view_870, [0, 2, 1]);  view_870 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_818: "f32[8, 196, 5, 64]" = torch.ops.aten.permute.default(getitem_480, [0, 2, 1, 3]);  getitem_480 = None
    view_871: "f32[8, 196, 320]" = torch.ops.aten.view.default(permute_818, [8, 196, 320]);  permute_818 = None
    view_872: "f32[1568, 320]" = torch.ops.aten.view.default(view_871, [1568, 320]);  view_871 = None
    mm_210: "f32[1568, 320]" = torch.ops.aten.mm.default(view_872, permute_819);  permute_819 = None
    permute_820: "f32[320, 1568]" = torch.ops.aten.permute.default(view_872, [1, 0])
    mm_211: "f32[320, 320]" = torch.ops.aten.mm.default(permute_820, view_116);  permute_820 = view_116 = None
    permute_821: "f32[320, 320]" = torch.ops.aten.permute.default(mm_211, [1, 0]);  mm_211 = None
    sum_371: "f32[1, 320]" = torch.ops.aten.sum.dim_IntList(view_872, [0], True);  view_872 = None
    view_873: "f32[320]" = torch.ops.aten.view.default(sum_371, [320]);  sum_371 = None
    permute_822: "f32[320, 320]" = torch.ops.aten.permute.default(permute_821, [1, 0]);  permute_821 = None
    view_874: "f32[8, 196, 320]" = torch.ops.aten.view.default(mm_210, [8, 196, 320]);  mm_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    add_365: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(permute_817, view_874);  permute_817 = view_874 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    clone_299: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_365, memory_format = torch.contiguous_format);  add_365 = None
    mul_831: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(clone_299, primals_143);  primals_143 = None
    mul_832: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_831, 320)
    sum_372: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_831, [2], True)
    mul_833: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_831, mul_69);  mul_831 = None
    sum_373: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_833, [2], True);  mul_833 = None
    mul_834: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_69, sum_373);  sum_373 = None
    sub_270: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(mul_832, sum_372);  mul_832 = sum_372 = None
    sub_271: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(sub_270, mul_834);  sub_270 = mul_834 = None
    mul_835: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(div_62, sub_271);  div_62 = sub_271 = None
    mul_836: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(clone_299, mul_69);  mul_69 = None
    sum_374: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_836, [0, 1]);  mul_836 = None
    sum_375: "f32[320]" = torch.ops.aten.sum.dim_IntList(clone_299, [0, 1]);  clone_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_366: "f32[8, 196, 320]" = torch.ops.aten.add.Tensor(add_364, mul_835);  add_364 = mul_835 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:275, code: x = self.norm(x)
    clone_300: "f32[8, 196, 320]" = torch.ops.aten.clone.default(add_366, memory_format = torch.contiguous_format);  add_366 = None
    mul_838: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(clone_300, primals_141);  primals_141 = None
    mul_839: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_838, 320)
    sum_376: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_838, [2], True)
    mul_840: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_838, mul_67);  mul_838 = None
    sum_377: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_840, [2], True);  mul_840 = None
    mul_841: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_67, sum_377);  sum_377 = None
    sub_273: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(mul_839, sum_376);  mul_839 = sum_376 = None
    sub_274: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(sub_273, mul_841);  sub_273 = mul_841 = None
    mul_842: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(div_63, sub_274);  div_63 = sub_274 = None
    mul_843: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(clone_300, mul_67);  mul_67 = None
    sum_378: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_843, [0, 1]);  mul_843 = None
    sum_379: "f32[320]" = torch.ops.aten.sum.dim_IntList(clone_300, [0, 1]);  clone_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:274, code: x = self.proj(x).flatten(2).transpose(1, 2)
    permute_823: "f32[8, 320, 196]" = torch.ops.aten.permute.default(mul_842, [0, 2, 1]);  mul_842 = None
    view_875: "f32[8, 320, 14, 14]" = torch.ops.aten.view.default(permute_823, [8, 320, 14, 14]);  permute_823 = None
    sum_380: "f32[320]" = torch.ops.aten.sum.dim_IntList(view_875, [0, 2, 3])
    convolution_backward_21 = torch.ops.aten.convolution_backward.default(view_875, permute_79, primals_139, [320], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_875 = permute_79 = primals_139 = None
    getitem_487: "f32[8, 128, 28, 28]" = convolution_backward_21[0]
    getitem_488: "f32[320, 128, 2, 2]" = convolution_backward_21[1];  convolution_backward_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:415, code: x = x.reshape(B, *size, -1).permute(0, 3, 1, 2).contiguous()
    permute_824: "f32[8, 28, 28, 128]" = torch.ops.aten.permute.default(getitem_487, [0, 2, 3, 1]);  getitem_487 = None
    view_876: "f32[8, 784, 128]" = torch.ops.aten.view.default(permute_824, [8, 784, 128]);  permute_824 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    clone_302: "f32[8, 784, 128]" = torch.ops.aten.clone.default(view_876, memory_format = torch.contiguous_format)
    view_877: "f32[6272, 128]" = torch.ops.aten.view.default(clone_302, [6272, 128]);  clone_302 = None
    mm_212: "f32[6272, 1024]" = torch.ops.aten.mm.default(view_877, permute_825);  permute_825 = None
    permute_826: "f32[128, 6272]" = torch.ops.aten.permute.default(view_877, [1, 0])
    mm_213: "f32[128, 1024]" = torch.ops.aten.mm.default(permute_826, view_112);  permute_826 = view_112 = None
    permute_827: "f32[1024, 128]" = torch.ops.aten.permute.default(mm_213, [1, 0]);  mm_213 = None
    sum_381: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_877, [0], True);  view_877 = None
    view_878: "f32[128]" = torch.ops.aten.view.default(sum_381, [128]);  sum_381 = None
    permute_828: "f32[128, 1024]" = torch.ops.aten.permute.default(permute_827, [1, 0]);  permute_827 = None
    view_879: "f32[8, 784, 1024]" = torch.ops.aten.view.default(mm_212, [8, 784, 1024]);  mm_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_845: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(add_67, 0.5);  add_67 = None
    mul_846: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(view_111, view_111)
    mul_847: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(mul_846, -0.5);  mul_846 = None
    exp_21: "f32[8, 784, 1024]" = torch.ops.aten.exp.default(mul_847);  mul_847 = None
    mul_848: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(exp_21, 0.3989422804014327);  exp_21 = None
    mul_849: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(view_111, mul_848);  view_111 = mul_848 = None
    add_368: "f32[8, 784, 1024]" = torch.ops.aten.add.Tensor(mul_845, mul_849);  mul_845 = mul_849 = None
    mul_850: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(view_879, add_368);  view_879 = add_368 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_880: "f32[6272, 1024]" = torch.ops.aten.view.default(mul_850, [6272, 1024]);  mul_850 = None
    mm_214: "f32[6272, 128]" = torch.ops.aten.mm.default(view_880, permute_829);  permute_829 = None
    permute_830: "f32[1024, 6272]" = torch.ops.aten.permute.default(view_880, [1, 0])
    mm_215: "f32[1024, 128]" = torch.ops.aten.mm.default(permute_830, view_110);  permute_830 = view_110 = None
    permute_831: "f32[128, 1024]" = torch.ops.aten.permute.default(mm_215, [1, 0]);  mm_215 = None
    sum_382: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_880, [0], True);  view_880 = None
    view_881: "f32[1024]" = torch.ops.aten.view.default(sum_382, [1024]);  sum_382 = None
    permute_832: "f32[1024, 128]" = torch.ops.aten.permute.default(permute_831, [1, 0]);  permute_831 = None
    view_882: "f32[8, 784, 128]" = torch.ops.aten.view.default(mm_214, [8, 784, 128]);  mm_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    mul_852: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(view_882, primals_133);  primals_133 = None
    mul_853: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(mul_852, 128)
    sum_383: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_852, [2], True)
    mul_854: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(mul_852, mul_62);  mul_852 = None
    sum_384: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_854, [2], True);  mul_854 = None
    mul_855: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(mul_62, sum_384);  sum_384 = None
    sub_276: "f32[8, 784, 128]" = torch.ops.aten.sub.Tensor(mul_853, sum_383);  mul_853 = sum_383 = None
    sub_277: "f32[8, 784, 128]" = torch.ops.aten.sub.Tensor(sub_276, mul_855);  sub_276 = mul_855 = None
    mul_856: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(div_64, sub_277);  div_64 = sub_277 = None
    mul_857: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(view_882, mul_62);  mul_62 = None
    sum_385: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_857, [0, 1]);  mul_857 = None
    sum_386: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_882, [0, 1]);  view_882 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_369: "f32[8, 784, 128]" = torch.ops.aten.add.Tensor(view_876, mul_856);  view_876 = mul_856 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    clone_304: "f32[8, 784, 128]" = torch.ops.aten.clone.default(add_369, memory_format = torch.contiguous_format)
    view_883: "f32[6272, 128]" = torch.ops.aten.view.default(clone_304, [6272, 128]);  clone_304 = None
    mm_216: "f32[6272, 128]" = torch.ops.aten.mm.default(view_883, permute_833);  permute_833 = None
    permute_834: "f32[128, 6272]" = torch.ops.aten.permute.default(view_883, [1, 0])
    mm_217: "f32[128, 128]" = torch.ops.aten.mm.default(permute_834, view_108);  permute_834 = view_108 = None
    permute_835: "f32[128, 128]" = torch.ops.aten.permute.default(mm_217, [1, 0]);  mm_217 = None
    sum_387: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_883, [0], True);  view_883 = None
    view_884: "f32[128]" = torch.ops.aten.view.default(sum_387, [128]);  sum_387 = None
    permute_836: "f32[128, 128]" = torch.ops.aten.permute.default(permute_835, [1, 0]);  permute_835 = None
    view_885: "f32[8, 784, 128]" = torch.ops.aten.view.default(mm_216, [8, 784, 128]);  mm_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_886: "f32[8, 784, 2, 64]" = torch.ops.aten.view.default(view_885, [8, 784, 2, 64]);  view_885 = None
    permute_837: "f32[8, 2, 784, 64]" = torch.ops.aten.permute.default(view_886, [0, 2, 1, 3]);  view_886 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_backward_21 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_837, permute_70, getitem_80, getitem_81, None, alias_49, getitem_83, getitem_84, getitem_85, 0.0, [True, True, True, False]);  permute_837 = permute_70 = getitem_80 = getitem_81 = alias_49 = getitem_83 = getitem_84 = getitem_85 = None
    getitem_490: "f32[8, 2, 784, 64]" = _scaled_dot_product_efficient_attention_backward_21[0]
    getitem_491: "f32[8, 2, 49, 64]" = _scaled_dot_product_efficient_attention_backward_21[1]
    getitem_492: "f32[8, 2, 49, 64]" = _scaled_dot_product_efficient_attention_backward_21[2];  _scaled_dot_product_efficient_attention_backward_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    cat_21: "f32[16, 2, 49, 64]" = torch.ops.aten.cat.default([getitem_491, getitem_492]);  getitem_491 = getitem_492 = None
    view_887: "f32[2, 8, 2, 49, 64]" = torch.ops.aten.view.default(cat_21, [2, 8, 2, 49, 64]);  cat_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_838: "f32[8, 49, 2, 2, 64]" = torch.ops.aten.permute.default(view_887, [1, 3, 0, 2, 4]);  view_887 = None
    clone_305: "f32[8, 49, 2, 2, 64]" = torch.ops.aten.clone.default(permute_838, memory_format = torch.contiguous_format);  permute_838 = None
    view_888: "f32[8, 49, 256]" = torch.ops.aten.view.default(clone_305, [8, 49, 256]);  clone_305 = None
    view_889: "f32[392, 256]" = torch.ops.aten.view.default(view_888, [392, 256]);  view_888 = None
    mm_218: "f32[392, 128]" = torch.ops.aten.mm.default(view_889, permute_839);  permute_839 = None
    permute_840: "f32[256, 392]" = torch.ops.aten.permute.default(view_889, [1, 0])
    mm_219: "f32[256, 128]" = torch.ops.aten.mm.default(permute_840, view_104);  permute_840 = view_104 = None
    permute_841: "f32[128, 256]" = torch.ops.aten.permute.default(mm_219, [1, 0]);  mm_219 = None
    sum_388: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_889, [0], True);  view_889 = None
    view_890: "f32[256]" = torch.ops.aten.view.default(sum_388, [256]);  sum_388 = None
    permute_842: "f32[256, 128]" = torch.ops.aten.permute.default(permute_841, [1, 0]);  permute_841 = None
    view_891: "f32[8, 49, 128]" = torch.ops.aten.view.default(mm_218, [8, 49, 128]);  mm_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    mul_859: "f32[8, 49, 128]" = torch.ops.aten.mul.Tensor(view_891, primals_127);  primals_127 = None
    mul_860: "f32[8, 49, 128]" = torch.ops.aten.mul.Tensor(mul_859, 128)
    sum_389: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_859, [2], True)
    mul_861: "f32[8, 49, 128]" = torch.ops.aten.mul.Tensor(mul_859, mul_60);  mul_859 = None
    sum_390: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_861, [2], True);  mul_861 = None
    mul_862: "f32[8, 49, 128]" = torch.ops.aten.mul.Tensor(mul_60, sum_390);  sum_390 = None
    sub_279: "f32[8, 49, 128]" = torch.ops.aten.sub.Tensor(mul_860, sum_389);  mul_860 = sum_389 = None
    sub_280: "f32[8, 49, 128]" = torch.ops.aten.sub.Tensor(sub_279, mul_862);  sub_279 = mul_862 = None
    mul_863: "f32[8, 49, 128]" = torch.ops.aten.mul.Tensor(div_65, sub_280);  div_65 = sub_280 = None
    mul_864: "f32[8, 49, 128]" = torch.ops.aten.mul.Tensor(view_891, mul_60);  mul_60 = None
    sum_391: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_864, [0, 1]);  mul_864 = None
    sum_392: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_891, [0, 1]);  view_891 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    permute_843: "f32[8, 128, 49]" = torch.ops.aten.permute.default(mul_863, [0, 2, 1]);  mul_863 = None
    view_892: "f32[8, 128, 7, 7]" = torch.ops.aten.view.default(permute_843, [8, 128, 7, 7]);  permute_843 = None
    sum_393: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_892, [0, 2, 3])
    convolution_backward_22 = torch.ops.aten.convolution_backward.default(view_892, view_102, primals_125, [128], [4, 4], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_892 = view_102 = primals_125 = None
    getitem_494: "f32[8, 128, 28, 28]" = convolution_backward_22[0]
    getitem_495: "f32[128, 128, 4, 4]" = convolution_backward_22[1];  convolution_backward_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    view_893: "f32[8, 128, 784]" = torch.ops.aten.view.default(getitem_494, [8, 128, 784]);  getitem_494 = None
    permute_844: "f32[8, 784, 128]" = torch.ops.aten.permute.default(view_893, [0, 2, 1]);  view_893 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_845: "f32[8, 784, 2, 64]" = torch.ops.aten.permute.default(getitem_490, [0, 2, 1, 3]);  getitem_490 = None
    view_894: "f32[8, 784, 128]" = torch.ops.aten.view.default(permute_845, [8, 784, 128]);  permute_845 = None
    view_895: "f32[6272, 128]" = torch.ops.aten.view.default(view_894, [6272, 128]);  view_894 = None
    mm_220: "f32[6272, 128]" = torch.ops.aten.mm.default(view_895, permute_846);  permute_846 = None
    permute_847: "f32[128, 6272]" = torch.ops.aten.permute.default(view_895, [1, 0])
    mm_221: "f32[128, 128]" = torch.ops.aten.mm.default(permute_847, view_99);  permute_847 = view_99 = None
    permute_848: "f32[128, 128]" = torch.ops.aten.permute.default(mm_221, [1, 0]);  mm_221 = None
    sum_394: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_895, [0], True);  view_895 = None
    view_896: "f32[128]" = torch.ops.aten.view.default(sum_394, [128]);  sum_394 = None
    permute_849: "f32[128, 128]" = torch.ops.aten.permute.default(permute_848, [1, 0]);  permute_848 = None
    view_897: "f32[8, 784, 128]" = torch.ops.aten.view.default(mm_220, [8, 784, 128]);  mm_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    add_370: "f32[8, 784, 128]" = torch.ops.aten.add.Tensor(permute_844, view_897);  permute_844 = view_897 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    clone_307: "f32[8, 784, 128]" = torch.ops.aten.clone.default(add_370, memory_format = torch.contiguous_format);  add_370 = None
    mul_866: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(clone_307, primals_121);  primals_121 = None
    mul_867: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(mul_866, 128)
    sum_395: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_866, [2], True)
    mul_868: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(mul_866, mul_58);  mul_866 = None
    sum_396: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_868, [2], True);  mul_868 = None
    mul_869: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(mul_58, sum_396);  sum_396 = None
    sub_282: "f32[8, 784, 128]" = torch.ops.aten.sub.Tensor(mul_867, sum_395);  mul_867 = sum_395 = None
    sub_283: "f32[8, 784, 128]" = torch.ops.aten.sub.Tensor(sub_282, mul_869);  sub_282 = mul_869 = None
    mul_870: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(div_66, sub_283);  div_66 = sub_283 = None
    mul_871: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(clone_307, mul_58);  mul_58 = None
    sum_397: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_871, [0, 1]);  mul_871 = None
    sum_398: "f32[128]" = torch.ops.aten.sum.dim_IntList(clone_307, [0, 1]);  clone_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_371: "f32[8, 784, 128]" = torch.ops.aten.add.Tensor(add_369, mul_870);  add_369 = mul_870 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    clone_309: "f32[8, 784, 128]" = torch.ops.aten.clone.default(add_371, memory_format = torch.contiguous_format)
    view_898: "f32[6272, 128]" = torch.ops.aten.view.default(clone_309, [6272, 128]);  clone_309 = None
    mm_222: "f32[6272, 1024]" = torch.ops.aten.mm.default(view_898, permute_850);  permute_850 = None
    permute_851: "f32[128, 6272]" = torch.ops.aten.permute.default(view_898, [1, 0])
    mm_223: "f32[128, 1024]" = torch.ops.aten.mm.default(permute_851, view_97);  permute_851 = view_97 = None
    permute_852: "f32[1024, 128]" = torch.ops.aten.permute.default(mm_223, [1, 0]);  mm_223 = None
    sum_399: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_898, [0], True);  view_898 = None
    view_899: "f32[128]" = torch.ops.aten.view.default(sum_399, [128]);  sum_399 = None
    permute_853: "f32[128, 1024]" = torch.ops.aten.permute.default(permute_852, [1, 0]);  permute_852 = None
    view_900: "f32[8, 784, 1024]" = torch.ops.aten.view.default(mm_222, [8, 784, 1024]);  mm_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_873: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(add_58, 0.5);  add_58 = None
    mul_874: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(view_96, view_96)
    mul_875: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(mul_874, -0.5);  mul_874 = None
    exp_22: "f32[8, 784, 1024]" = torch.ops.aten.exp.default(mul_875);  mul_875 = None
    mul_876: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(exp_22, 0.3989422804014327);  exp_22 = None
    mul_877: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(view_96, mul_876);  view_96 = mul_876 = None
    add_373: "f32[8, 784, 1024]" = torch.ops.aten.add.Tensor(mul_873, mul_877);  mul_873 = mul_877 = None
    mul_878: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(view_900, add_373);  view_900 = add_373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_901: "f32[6272, 1024]" = torch.ops.aten.view.default(mul_878, [6272, 1024]);  mul_878 = None
    mm_224: "f32[6272, 128]" = torch.ops.aten.mm.default(view_901, permute_854);  permute_854 = None
    permute_855: "f32[1024, 6272]" = torch.ops.aten.permute.default(view_901, [1, 0])
    mm_225: "f32[1024, 128]" = torch.ops.aten.mm.default(permute_855, view_95);  permute_855 = view_95 = None
    permute_856: "f32[128, 1024]" = torch.ops.aten.permute.default(mm_225, [1, 0]);  mm_225 = None
    sum_400: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_901, [0], True);  view_901 = None
    view_902: "f32[1024]" = torch.ops.aten.view.default(sum_400, [1024]);  sum_400 = None
    permute_857: "f32[1024, 128]" = torch.ops.aten.permute.default(permute_856, [1, 0]);  permute_856 = None
    view_903: "f32[8, 784, 128]" = torch.ops.aten.view.default(mm_224, [8, 784, 128]);  mm_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    mul_880: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(view_903, primals_115);  primals_115 = None
    mul_881: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(mul_880, 128)
    sum_401: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_880, [2], True)
    mul_882: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(mul_880, mul_53);  mul_880 = None
    sum_402: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_882, [2], True);  mul_882 = None
    mul_883: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(mul_53, sum_402);  sum_402 = None
    sub_285: "f32[8, 784, 128]" = torch.ops.aten.sub.Tensor(mul_881, sum_401);  mul_881 = sum_401 = None
    sub_286: "f32[8, 784, 128]" = torch.ops.aten.sub.Tensor(sub_285, mul_883);  sub_285 = mul_883 = None
    mul_884: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(div_67, sub_286);  div_67 = sub_286 = None
    mul_885: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(view_903, mul_53);  mul_53 = None
    sum_403: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_885, [0, 1]);  mul_885 = None
    sum_404: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_903, [0, 1]);  view_903 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_374: "f32[8, 784, 128]" = torch.ops.aten.add.Tensor(add_371, mul_884);  add_371 = mul_884 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    clone_311: "f32[8, 784, 128]" = torch.ops.aten.clone.default(add_374, memory_format = torch.contiguous_format)
    view_904: "f32[6272, 128]" = torch.ops.aten.view.default(clone_311, [6272, 128]);  clone_311 = None
    mm_226: "f32[6272, 128]" = torch.ops.aten.mm.default(view_904, permute_858);  permute_858 = None
    permute_859: "f32[128, 6272]" = torch.ops.aten.permute.default(view_904, [1, 0])
    mm_227: "f32[128, 128]" = torch.ops.aten.mm.default(permute_859, view_93);  permute_859 = view_93 = None
    permute_860: "f32[128, 128]" = torch.ops.aten.permute.default(mm_227, [1, 0]);  mm_227 = None
    sum_405: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_904, [0], True);  view_904 = None
    view_905: "f32[128]" = torch.ops.aten.view.default(sum_405, [128]);  sum_405 = None
    permute_861: "f32[128, 128]" = torch.ops.aten.permute.default(permute_860, [1, 0]);  permute_860 = None
    view_906: "f32[8, 784, 128]" = torch.ops.aten.view.default(mm_226, [8, 784, 128]);  mm_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_907: "f32[8, 784, 2, 64]" = torch.ops.aten.view.default(view_906, [8, 784, 2, 64]);  view_906 = None
    permute_862: "f32[8, 2, 784, 64]" = torch.ops.aten.permute.default(view_907, [0, 2, 1, 3]);  view_907 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_backward_22 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_862, permute_60, getitem_68, getitem_69, None, alias_50, getitem_71, getitem_72, getitem_73, 0.0, [True, True, True, False]);  permute_862 = permute_60 = getitem_68 = getitem_69 = alias_50 = getitem_71 = getitem_72 = getitem_73 = None
    getitem_497: "f32[8, 2, 784, 64]" = _scaled_dot_product_efficient_attention_backward_22[0]
    getitem_498: "f32[8, 2, 49, 64]" = _scaled_dot_product_efficient_attention_backward_22[1]
    getitem_499: "f32[8, 2, 49, 64]" = _scaled_dot_product_efficient_attention_backward_22[2];  _scaled_dot_product_efficient_attention_backward_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    cat_22: "f32[16, 2, 49, 64]" = torch.ops.aten.cat.default([getitem_498, getitem_499]);  getitem_498 = getitem_499 = None
    view_908: "f32[2, 8, 2, 49, 64]" = torch.ops.aten.view.default(cat_22, [2, 8, 2, 49, 64]);  cat_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_863: "f32[8, 49, 2, 2, 64]" = torch.ops.aten.permute.default(view_908, [1, 3, 0, 2, 4]);  view_908 = None
    clone_312: "f32[8, 49, 2, 2, 64]" = torch.ops.aten.clone.default(permute_863, memory_format = torch.contiguous_format);  permute_863 = None
    view_909: "f32[8, 49, 256]" = torch.ops.aten.view.default(clone_312, [8, 49, 256]);  clone_312 = None
    view_910: "f32[392, 256]" = torch.ops.aten.view.default(view_909, [392, 256]);  view_909 = None
    mm_228: "f32[392, 128]" = torch.ops.aten.mm.default(view_910, permute_864);  permute_864 = None
    permute_865: "f32[256, 392]" = torch.ops.aten.permute.default(view_910, [1, 0])
    mm_229: "f32[256, 128]" = torch.ops.aten.mm.default(permute_865, view_89);  permute_865 = view_89 = None
    permute_866: "f32[128, 256]" = torch.ops.aten.permute.default(mm_229, [1, 0]);  mm_229 = None
    sum_406: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_910, [0], True);  view_910 = None
    view_911: "f32[256]" = torch.ops.aten.view.default(sum_406, [256]);  sum_406 = None
    permute_867: "f32[256, 128]" = torch.ops.aten.permute.default(permute_866, [1, 0]);  permute_866 = None
    view_912: "f32[8, 49, 128]" = torch.ops.aten.view.default(mm_228, [8, 49, 128]);  mm_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    mul_887: "f32[8, 49, 128]" = torch.ops.aten.mul.Tensor(view_912, primals_109);  primals_109 = None
    mul_888: "f32[8, 49, 128]" = torch.ops.aten.mul.Tensor(mul_887, 128)
    sum_407: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_887, [2], True)
    mul_889: "f32[8, 49, 128]" = torch.ops.aten.mul.Tensor(mul_887, mul_51);  mul_887 = None
    sum_408: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_889, [2], True);  mul_889 = None
    mul_890: "f32[8, 49, 128]" = torch.ops.aten.mul.Tensor(mul_51, sum_408);  sum_408 = None
    sub_288: "f32[8, 49, 128]" = torch.ops.aten.sub.Tensor(mul_888, sum_407);  mul_888 = sum_407 = None
    sub_289: "f32[8, 49, 128]" = torch.ops.aten.sub.Tensor(sub_288, mul_890);  sub_288 = mul_890 = None
    mul_891: "f32[8, 49, 128]" = torch.ops.aten.mul.Tensor(div_68, sub_289);  div_68 = sub_289 = None
    mul_892: "f32[8, 49, 128]" = torch.ops.aten.mul.Tensor(view_912, mul_51);  mul_51 = None
    sum_409: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_892, [0, 1]);  mul_892 = None
    sum_410: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_912, [0, 1]);  view_912 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    permute_868: "f32[8, 128, 49]" = torch.ops.aten.permute.default(mul_891, [0, 2, 1]);  mul_891 = None
    view_913: "f32[8, 128, 7, 7]" = torch.ops.aten.view.default(permute_868, [8, 128, 7, 7]);  permute_868 = None
    sum_411: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_913, [0, 2, 3])
    convolution_backward_23 = torch.ops.aten.convolution_backward.default(view_913, view_87, primals_107, [128], [4, 4], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_913 = view_87 = primals_107 = None
    getitem_501: "f32[8, 128, 28, 28]" = convolution_backward_23[0]
    getitem_502: "f32[128, 128, 4, 4]" = convolution_backward_23[1];  convolution_backward_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    view_914: "f32[8, 128, 784]" = torch.ops.aten.view.default(getitem_501, [8, 128, 784]);  getitem_501 = None
    permute_869: "f32[8, 784, 128]" = torch.ops.aten.permute.default(view_914, [0, 2, 1]);  view_914 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_870: "f32[8, 784, 2, 64]" = torch.ops.aten.permute.default(getitem_497, [0, 2, 1, 3]);  getitem_497 = None
    view_915: "f32[8, 784, 128]" = torch.ops.aten.view.default(permute_870, [8, 784, 128]);  permute_870 = None
    view_916: "f32[6272, 128]" = torch.ops.aten.view.default(view_915, [6272, 128]);  view_915 = None
    mm_230: "f32[6272, 128]" = torch.ops.aten.mm.default(view_916, permute_871);  permute_871 = None
    permute_872: "f32[128, 6272]" = torch.ops.aten.permute.default(view_916, [1, 0])
    mm_231: "f32[128, 128]" = torch.ops.aten.mm.default(permute_872, view_84);  permute_872 = view_84 = None
    permute_873: "f32[128, 128]" = torch.ops.aten.permute.default(mm_231, [1, 0]);  mm_231 = None
    sum_412: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_916, [0], True);  view_916 = None
    view_917: "f32[128]" = torch.ops.aten.view.default(sum_412, [128]);  sum_412 = None
    permute_874: "f32[128, 128]" = torch.ops.aten.permute.default(permute_873, [1, 0]);  permute_873 = None
    view_918: "f32[8, 784, 128]" = torch.ops.aten.view.default(mm_230, [8, 784, 128]);  mm_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    add_375: "f32[8, 784, 128]" = torch.ops.aten.add.Tensor(permute_869, view_918);  permute_869 = view_918 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    clone_314: "f32[8, 784, 128]" = torch.ops.aten.clone.default(add_375, memory_format = torch.contiguous_format);  add_375 = None
    mul_894: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(clone_314, primals_103);  primals_103 = None
    mul_895: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(mul_894, 128)
    sum_413: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_894, [2], True)
    mul_896: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(mul_894, mul_49);  mul_894 = None
    sum_414: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_896, [2], True);  mul_896 = None
    mul_897: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(mul_49, sum_414);  sum_414 = None
    sub_291: "f32[8, 784, 128]" = torch.ops.aten.sub.Tensor(mul_895, sum_413);  mul_895 = sum_413 = None
    sub_292: "f32[8, 784, 128]" = torch.ops.aten.sub.Tensor(sub_291, mul_897);  sub_291 = mul_897 = None
    mul_898: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(div_69, sub_292);  div_69 = sub_292 = None
    mul_899: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(clone_314, mul_49);  mul_49 = None
    sum_415: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_899, [0, 1]);  mul_899 = None
    sum_416: "f32[128]" = torch.ops.aten.sum.dim_IntList(clone_314, [0, 1]);  clone_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_376: "f32[8, 784, 128]" = torch.ops.aten.add.Tensor(add_374, mul_898);  add_374 = mul_898 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    clone_316: "f32[8, 784, 128]" = torch.ops.aten.clone.default(add_376, memory_format = torch.contiguous_format)
    view_919: "f32[6272, 128]" = torch.ops.aten.view.default(clone_316, [6272, 128]);  clone_316 = None
    mm_232: "f32[6272, 1024]" = torch.ops.aten.mm.default(view_919, permute_875);  permute_875 = None
    permute_876: "f32[128, 6272]" = torch.ops.aten.permute.default(view_919, [1, 0])
    mm_233: "f32[128, 1024]" = torch.ops.aten.mm.default(permute_876, view_82);  permute_876 = view_82 = None
    permute_877: "f32[1024, 128]" = torch.ops.aten.permute.default(mm_233, [1, 0]);  mm_233 = None
    sum_417: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_919, [0], True);  view_919 = None
    view_920: "f32[128]" = torch.ops.aten.view.default(sum_417, [128]);  sum_417 = None
    permute_878: "f32[128, 1024]" = torch.ops.aten.permute.default(permute_877, [1, 0]);  permute_877 = None
    view_921: "f32[8, 784, 1024]" = torch.ops.aten.view.default(mm_232, [8, 784, 1024]);  mm_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_901: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(add_49, 0.5);  add_49 = None
    mul_902: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(view_81, view_81)
    mul_903: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(mul_902, -0.5);  mul_902 = None
    exp_23: "f32[8, 784, 1024]" = torch.ops.aten.exp.default(mul_903);  mul_903 = None
    mul_904: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(exp_23, 0.3989422804014327);  exp_23 = None
    mul_905: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(view_81, mul_904);  view_81 = mul_904 = None
    add_378: "f32[8, 784, 1024]" = torch.ops.aten.add.Tensor(mul_901, mul_905);  mul_901 = mul_905 = None
    mul_906: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(view_921, add_378);  view_921 = add_378 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_922: "f32[6272, 1024]" = torch.ops.aten.view.default(mul_906, [6272, 1024]);  mul_906 = None
    mm_234: "f32[6272, 128]" = torch.ops.aten.mm.default(view_922, permute_879);  permute_879 = None
    permute_880: "f32[1024, 6272]" = torch.ops.aten.permute.default(view_922, [1, 0])
    mm_235: "f32[1024, 128]" = torch.ops.aten.mm.default(permute_880, view_80);  permute_880 = view_80 = None
    permute_881: "f32[128, 1024]" = torch.ops.aten.permute.default(mm_235, [1, 0]);  mm_235 = None
    sum_418: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_922, [0], True);  view_922 = None
    view_923: "f32[1024]" = torch.ops.aten.view.default(sum_418, [1024]);  sum_418 = None
    permute_882: "f32[1024, 128]" = torch.ops.aten.permute.default(permute_881, [1, 0]);  permute_881 = None
    view_924: "f32[8, 784, 128]" = torch.ops.aten.view.default(mm_234, [8, 784, 128]);  mm_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    mul_908: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(view_924, primals_97);  primals_97 = None
    mul_909: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(mul_908, 128)
    sum_419: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_908, [2], True)
    mul_910: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(mul_908, mul_44);  mul_908 = None
    sum_420: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_910, [2], True);  mul_910 = None
    mul_911: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(mul_44, sum_420);  sum_420 = None
    sub_294: "f32[8, 784, 128]" = torch.ops.aten.sub.Tensor(mul_909, sum_419);  mul_909 = sum_419 = None
    sub_295: "f32[8, 784, 128]" = torch.ops.aten.sub.Tensor(sub_294, mul_911);  sub_294 = mul_911 = None
    mul_912: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(div_70, sub_295);  div_70 = sub_295 = None
    mul_913: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(view_924, mul_44);  mul_44 = None
    sum_421: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_913, [0, 1]);  mul_913 = None
    sum_422: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_924, [0, 1]);  view_924 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_379: "f32[8, 784, 128]" = torch.ops.aten.add.Tensor(add_376, mul_912);  add_376 = mul_912 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    clone_318: "f32[8, 784, 128]" = torch.ops.aten.clone.default(add_379, memory_format = torch.contiguous_format)
    view_925: "f32[6272, 128]" = torch.ops.aten.view.default(clone_318, [6272, 128]);  clone_318 = None
    mm_236: "f32[6272, 128]" = torch.ops.aten.mm.default(view_925, permute_883);  permute_883 = None
    permute_884: "f32[128, 6272]" = torch.ops.aten.permute.default(view_925, [1, 0])
    mm_237: "f32[128, 128]" = torch.ops.aten.mm.default(permute_884, view_78);  permute_884 = view_78 = None
    permute_885: "f32[128, 128]" = torch.ops.aten.permute.default(mm_237, [1, 0]);  mm_237 = None
    sum_423: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_925, [0], True);  view_925 = None
    view_926: "f32[128]" = torch.ops.aten.view.default(sum_423, [128]);  sum_423 = None
    permute_886: "f32[128, 128]" = torch.ops.aten.permute.default(permute_885, [1, 0]);  permute_885 = None
    view_927: "f32[8, 784, 128]" = torch.ops.aten.view.default(mm_236, [8, 784, 128]);  mm_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_928: "f32[8, 784, 2, 64]" = torch.ops.aten.view.default(view_927, [8, 784, 2, 64]);  view_927 = None
    permute_887: "f32[8, 2, 784, 64]" = torch.ops.aten.permute.default(view_928, [0, 2, 1, 3]);  view_928 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_backward_23 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_887, permute_50, getitem_56, getitem_57, None, alias_51, getitem_59, getitem_60, getitem_61, 0.0, [True, True, True, False]);  permute_887 = permute_50 = getitem_56 = getitem_57 = alias_51 = getitem_59 = getitem_60 = getitem_61 = None
    getitem_504: "f32[8, 2, 784, 64]" = _scaled_dot_product_efficient_attention_backward_23[0]
    getitem_505: "f32[8, 2, 49, 64]" = _scaled_dot_product_efficient_attention_backward_23[1]
    getitem_506: "f32[8, 2, 49, 64]" = _scaled_dot_product_efficient_attention_backward_23[2];  _scaled_dot_product_efficient_attention_backward_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    cat_23: "f32[16, 2, 49, 64]" = torch.ops.aten.cat.default([getitem_505, getitem_506]);  getitem_505 = getitem_506 = None
    view_929: "f32[2, 8, 2, 49, 64]" = torch.ops.aten.view.default(cat_23, [2, 8, 2, 49, 64]);  cat_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_888: "f32[8, 49, 2, 2, 64]" = torch.ops.aten.permute.default(view_929, [1, 3, 0, 2, 4]);  view_929 = None
    clone_319: "f32[8, 49, 2, 2, 64]" = torch.ops.aten.clone.default(permute_888, memory_format = torch.contiguous_format);  permute_888 = None
    view_930: "f32[8, 49, 256]" = torch.ops.aten.view.default(clone_319, [8, 49, 256]);  clone_319 = None
    view_931: "f32[392, 256]" = torch.ops.aten.view.default(view_930, [392, 256]);  view_930 = None
    mm_238: "f32[392, 128]" = torch.ops.aten.mm.default(view_931, permute_889);  permute_889 = None
    permute_890: "f32[256, 392]" = torch.ops.aten.permute.default(view_931, [1, 0])
    mm_239: "f32[256, 128]" = torch.ops.aten.mm.default(permute_890, view_74);  permute_890 = view_74 = None
    permute_891: "f32[128, 256]" = torch.ops.aten.permute.default(mm_239, [1, 0]);  mm_239 = None
    sum_424: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_931, [0], True);  view_931 = None
    view_932: "f32[256]" = torch.ops.aten.view.default(sum_424, [256]);  sum_424 = None
    permute_892: "f32[256, 128]" = torch.ops.aten.permute.default(permute_891, [1, 0]);  permute_891 = None
    view_933: "f32[8, 49, 128]" = torch.ops.aten.view.default(mm_238, [8, 49, 128]);  mm_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    mul_915: "f32[8, 49, 128]" = torch.ops.aten.mul.Tensor(view_933, primals_91);  primals_91 = None
    mul_916: "f32[8, 49, 128]" = torch.ops.aten.mul.Tensor(mul_915, 128)
    sum_425: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_915, [2], True)
    mul_917: "f32[8, 49, 128]" = torch.ops.aten.mul.Tensor(mul_915, mul_42);  mul_915 = None
    sum_426: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_917, [2], True);  mul_917 = None
    mul_918: "f32[8, 49, 128]" = torch.ops.aten.mul.Tensor(mul_42, sum_426);  sum_426 = None
    sub_297: "f32[8, 49, 128]" = torch.ops.aten.sub.Tensor(mul_916, sum_425);  mul_916 = sum_425 = None
    sub_298: "f32[8, 49, 128]" = torch.ops.aten.sub.Tensor(sub_297, mul_918);  sub_297 = mul_918 = None
    mul_919: "f32[8, 49, 128]" = torch.ops.aten.mul.Tensor(div_71, sub_298);  div_71 = sub_298 = None
    mul_920: "f32[8, 49, 128]" = torch.ops.aten.mul.Tensor(view_933, mul_42);  mul_42 = None
    sum_427: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_920, [0, 1]);  mul_920 = None
    sum_428: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_933, [0, 1]);  view_933 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    permute_893: "f32[8, 128, 49]" = torch.ops.aten.permute.default(mul_919, [0, 2, 1]);  mul_919 = None
    view_934: "f32[8, 128, 7, 7]" = torch.ops.aten.view.default(permute_893, [8, 128, 7, 7]);  permute_893 = None
    sum_429: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_934, [0, 2, 3])
    convolution_backward_24 = torch.ops.aten.convolution_backward.default(view_934, view_72, primals_89, [128], [4, 4], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_934 = view_72 = primals_89 = None
    getitem_508: "f32[8, 128, 28, 28]" = convolution_backward_24[0]
    getitem_509: "f32[128, 128, 4, 4]" = convolution_backward_24[1];  convolution_backward_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    view_935: "f32[8, 128, 784]" = torch.ops.aten.view.default(getitem_508, [8, 128, 784]);  getitem_508 = None
    permute_894: "f32[8, 784, 128]" = torch.ops.aten.permute.default(view_935, [0, 2, 1]);  view_935 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_895: "f32[8, 784, 2, 64]" = torch.ops.aten.permute.default(getitem_504, [0, 2, 1, 3]);  getitem_504 = None
    view_936: "f32[8, 784, 128]" = torch.ops.aten.view.default(permute_895, [8, 784, 128]);  permute_895 = None
    view_937: "f32[6272, 128]" = torch.ops.aten.view.default(view_936, [6272, 128]);  view_936 = None
    mm_240: "f32[6272, 128]" = torch.ops.aten.mm.default(view_937, permute_896);  permute_896 = None
    permute_897: "f32[128, 6272]" = torch.ops.aten.permute.default(view_937, [1, 0])
    mm_241: "f32[128, 128]" = torch.ops.aten.mm.default(permute_897, view_69);  permute_897 = view_69 = None
    permute_898: "f32[128, 128]" = torch.ops.aten.permute.default(mm_241, [1, 0]);  mm_241 = None
    sum_430: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_937, [0], True);  view_937 = None
    view_938: "f32[128]" = torch.ops.aten.view.default(sum_430, [128]);  sum_430 = None
    permute_899: "f32[128, 128]" = torch.ops.aten.permute.default(permute_898, [1, 0]);  permute_898 = None
    view_939: "f32[8, 784, 128]" = torch.ops.aten.view.default(mm_240, [8, 784, 128]);  mm_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    add_380: "f32[8, 784, 128]" = torch.ops.aten.add.Tensor(permute_894, view_939);  permute_894 = view_939 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    clone_321: "f32[8, 784, 128]" = torch.ops.aten.clone.default(add_380, memory_format = torch.contiguous_format);  add_380 = None
    mul_922: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(clone_321, primals_85);  primals_85 = None
    mul_923: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(mul_922, 128)
    sum_431: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_922, [2], True)
    mul_924: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(mul_922, mul_40);  mul_922 = None
    sum_432: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_924, [2], True);  mul_924 = None
    mul_925: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(mul_40, sum_432);  sum_432 = None
    sub_300: "f32[8, 784, 128]" = torch.ops.aten.sub.Tensor(mul_923, sum_431);  mul_923 = sum_431 = None
    sub_301: "f32[8, 784, 128]" = torch.ops.aten.sub.Tensor(sub_300, mul_925);  sub_300 = mul_925 = None
    mul_926: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(div_72, sub_301);  div_72 = sub_301 = None
    mul_927: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(clone_321, mul_40);  mul_40 = None
    sum_433: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_927, [0, 1]);  mul_927 = None
    sum_434: "f32[128]" = torch.ops.aten.sum.dim_IntList(clone_321, [0, 1]);  clone_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_381: "f32[8, 784, 128]" = torch.ops.aten.add.Tensor(add_379, mul_926);  add_379 = mul_926 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:246, code: x = x.flatten(2).transpose(1, 2)
    permute_900: "f32[8, 128, 784]" = torch.ops.aten.permute.default(add_381, [0, 2, 1]);  add_381 = None
    view_940: "f32[8, 128, 28, 28]" = torch.ops.aten.view.default(permute_900, [8, 128, 28, 28]);  permute_900 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:243, code: x = self.proj(cnn_feat_token)
    sum_435: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_940, [0, 2, 3])
    convolution_backward_25 = torch.ops.aten.convolution_backward.default(view_940, view_66, primals_83, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 128, [True, True, False]);  view_66 = primals_83 = None
    getitem_511: "f32[8, 128, 28, 28]" = convolution_backward_25[0]
    getitem_512: "f32[128, 1, 3, 3]" = convolution_backward_25[1];  convolution_backward_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:243, code: x = self.proj(cnn_feat_token)
    add_382: "f32[8, 128, 28, 28]" = torch.ops.aten.add.Tensor(view_940, getitem_511);  view_940 = getitem_511 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:242, code: cnn_feat_token = x.transpose(1, 2).view(B, C, *size)
    view_941: "f32[8, 128, 784]" = torch.ops.aten.view.default(add_382, [8, 128, 784]);  add_382 = None
    permute_901: "f32[8, 784, 128]" = torch.ops.aten.permute.default(view_941, [0, 2, 1]);  view_941 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    clone_323: "f32[8, 784, 128]" = torch.ops.aten.clone.default(permute_901, memory_format = torch.contiguous_format)
    view_942: "f32[6272, 128]" = torch.ops.aten.view.default(clone_323, [6272, 128]);  clone_323 = None
    mm_242: "f32[6272, 1024]" = torch.ops.aten.mm.default(view_942, permute_902);  permute_902 = None
    permute_903: "f32[128, 6272]" = torch.ops.aten.permute.default(view_942, [1, 0])
    mm_243: "f32[128, 1024]" = torch.ops.aten.mm.default(permute_903, view_64);  permute_903 = view_64 = None
    permute_904: "f32[1024, 128]" = torch.ops.aten.permute.default(mm_243, [1, 0]);  mm_243 = None
    sum_436: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_942, [0], True);  view_942 = None
    view_943: "f32[128]" = torch.ops.aten.view.default(sum_436, [128]);  sum_436 = None
    permute_905: "f32[128, 1024]" = torch.ops.aten.permute.default(permute_904, [1, 0]);  permute_904 = None
    view_944: "f32[8, 784, 1024]" = torch.ops.aten.view.default(mm_242, [8, 784, 1024]);  mm_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_929: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(add_39, 0.5);  add_39 = None
    mul_930: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(view_63, view_63)
    mul_931: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(mul_930, -0.5);  mul_930 = None
    exp_24: "f32[8, 784, 1024]" = torch.ops.aten.exp.default(mul_931);  mul_931 = None
    mul_932: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(exp_24, 0.3989422804014327);  exp_24 = None
    mul_933: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(view_63, mul_932);  view_63 = mul_932 = None
    add_384: "f32[8, 784, 1024]" = torch.ops.aten.add.Tensor(mul_929, mul_933);  mul_929 = mul_933 = None
    mul_934: "f32[8, 784, 1024]" = torch.ops.aten.mul.Tensor(view_944, add_384);  view_944 = add_384 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_945: "f32[6272, 1024]" = torch.ops.aten.view.default(mul_934, [6272, 1024]);  mul_934 = None
    mm_244: "f32[6272, 128]" = torch.ops.aten.mm.default(view_945, permute_906);  permute_906 = None
    permute_907: "f32[1024, 6272]" = torch.ops.aten.permute.default(view_945, [1, 0])
    mm_245: "f32[1024, 128]" = torch.ops.aten.mm.default(permute_907, view_62);  permute_907 = view_62 = None
    permute_908: "f32[128, 1024]" = torch.ops.aten.permute.default(mm_245, [1, 0]);  mm_245 = None
    sum_437: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_945, [0], True);  view_945 = None
    view_946: "f32[1024]" = torch.ops.aten.view.default(sum_437, [1024]);  sum_437 = None
    permute_909: "f32[1024, 128]" = torch.ops.aten.permute.default(permute_908, [1, 0]);  permute_908 = None
    view_947: "f32[8, 784, 128]" = torch.ops.aten.view.default(mm_244, [8, 784, 128]);  mm_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    mul_936: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(view_947, primals_77);  primals_77 = None
    mul_937: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(mul_936, 128)
    sum_438: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_936, [2], True)
    mul_938: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(mul_936, mul_35);  mul_936 = None
    sum_439: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_938, [2], True);  mul_938 = None
    mul_939: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(mul_35, sum_439);  sum_439 = None
    sub_303: "f32[8, 784, 128]" = torch.ops.aten.sub.Tensor(mul_937, sum_438);  mul_937 = sum_438 = None
    sub_304: "f32[8, 784, 128]" = torch.ops.aten.sub.Tensor(sub_303, mul_939);  sub_303 = mul_939 = None
    mul_940: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(div_73, sub_304);  div_73 = sub_304 = None
    mul_941: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(view_947, mul_35);  mul_35 = None
    sum_440: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_941, [0, 1]);  mul_941 = None
    sum_441: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_947, [0, 1]);  view_947 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_385: "f32[8, 784, 128]" = torch.ops.aten.add.Tensor(permute_901, mul_940);  permute_901 = mul_940 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    clone_324: "f32[8, 784, 128]" = torch.ops.aten.clone.default(add_385, memory_format = torch.contiguous_format)
    view_948: "f32[6272, 128]" = torch.ops.aten.view.default(clone_324, [6272, 128]);  clone_324 = None
    mm_246: "f32[6272, 128]" = torch.ops.aten.mm.default(view_948, permute_910);  permute_910 = None
    permute_911: "f32[128, 6272]" = torch.ops.aten.permute.default(view_948, [1, 0])
    mm_247: "f32[128, 128]" = torch.ops.aten.mm.default(permute_911, view_60);  permute_911 = view_60 = None
    permute_912: "f32[128, 128]" = torch.ops.aten.permute.default(mm_247, [1, 0]);  mm_247 = None
    sum_442: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_948, [0], True);  view_948 = None
    view_949: "f32[128]" = torch.ops.aten.view.default(sum_442, [128]);  sum_442 = None
    permute_913: "f32[128, 128]" = torch.ops.aten.permute.default(permute_912, [1, 0]);  permute_912 = None
    view_950: "f32[8, 784, 128]" = torch.ops.aten.view.default(mm_246, [8, 784, 128]);  mm_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_951: "f32[8, 784, 2, 64]" = torch.ops.aten.view.default(view_950, [8, 784, 2, 64]);  view_950 = None
    permute_914: "f32[8, 2, 784, 64]" = torch.ops.aten.permute.default(view_951, [0, 2, 1, 3]);  view_951 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_backward_24 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_914, permute_37, getitem_44, getitem_45, None, alias_52, getitem_47, getitem_48, getitem_49, 0.0, [True, True, True, False]);  permute_914 = permute_37 = getitem_44 = getitem_45 = alias_52 = getitem_47 = getitem_48 = getitem_49 = None
    getitem_514: "f32[8, 2, 784, 64]" = _scaled_dot_product_efficient_attention_backward_24[0]
    getitem_515: "f32[8, 2, 49, 64]" = _scaled_dot_product_efficient_attention_backward_24[1]
    getitem_516: "f32[8, 2, 49, 64]" = _scaled_dot_product_efficient_attention_backward_24[2];  _scaled_dot_product_efficient_attention_backward_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    cat_24: "f32[16, 2, 49, 64]" = torch.ops.aten.cat.default([getitem_515, getitem_516]);  getitem_515 = getitem_516 = None
    view_952: "f32[2, 8, 2, 49, 64]" = torch.ops.aten.view.default(cat_24, [2, 8, 2, 49, 64]);  cat_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_915: "f32[8, 49, 2, 2, 64]" = torch.ops.aten.permute.default(view_952, [1, 3, 0, 2, 4]);  view_952 = None
    clone_325: "f32[8, 49, 2, 2, 64]" = torch.ops.aten.clone.default(permute_915, memory_format = torch.contiguous_format);  permute_915 = None
    view_953: "f32[8, 49, 256]" = torch.ops.aten.view.default(clone_325, [8, 49, 256]);  clone_325 = None
    view_954: "f32[392, 256]" = torch.ops.aten.view.default(view_953, [392, 256]);  view_953 = None
    mm_248: "f32[392, 128]" = torch.ops.aten.mm.default(view_954, permute_916);  permute_916 = None
    permute_917: "f32[256, 392]" = torch.ops.aten.permute.default(view_954, [1, 0])
    mm_249: "f32[256, 128]" = torch.ops.aten.mm.default(permute_917, view_56);  permute_917 = view_56 = None
    permute_918: "f32[128, 256]" = torch.ops.aten.permute.default(mm_249, [1, 0]);  mm_249 = None
    sum_443: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_954, [0], True);  view_954 = None
    view_955: "f32[256]" = torch.ops.aten.view.default(sum_443, [256]);  sum_443 = None
    permute_919: "f32[256, 128]" = torch.ops.aten.permute.default(permute_918, [1, 0]);  permute_918 = None
    view_956: "f32[8, 49, 128]" = torch.ops.aten.view.default(mm_248, [8, 49, 128]);  mm_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    mul_943: "f32[8, 49, 128]" = torch.ops.aten.mul.Tensor(view_956, primals_71);  primals_71 = None
    mul_944: "f32[8, 49, 128]" = torch.ops.aten.mul.Tensor(mul_943, 128)
    sum_444: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_943, [2], True)
    mul_945: "f32[8, 49, 128]" = torch.ops.aten.mul.Tensor(mul_943, mul_33);  mul_943 = None
    sum_445: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_945, [2], True);  mul_945 = None
    mul_946: "f32[8, 49, 128]" = torch.ops.aten.mul.Tensor(mul_33, sum_445);  sum_445 = None
    sub_306: "f32[8, 49, 128]" = torch.ops.aten.sub.Tensor(mul_944, sum_444);  mul_944 = sum_444 = None
    sub_307: "f32[8, 49, 128]" = torch.ops.aten.sub.Tensor(sub_306, mul_946);  sub_306 = mul_946 = None
    mul_947: "f32[8, 49, 128]" = torch.ops.aten.mul.Tensor(div_74, sub_307);  div_74 = sub_307 = None
    mul_948: "f32[8, 49, 128]" = torch.ops.aten.mul.Tensor(view_956, mul_33);  mul_33 = None
    sum_446: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_948, [0, 1]);  mul_948 = None
    sum_447: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_956, [0, 1]);  view_956 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    permute_920: "f32[8, 128, 49]" = torch.ops.aten.permute.default(mul_947, [0, 2, 1]);  mul_947 = None
    view_957: "f32[8, 128, 7, 7]" = torch.ops.aten.view.default(permute_920, [8, 128, 7, 7]);  permute_920 = None
    sum_448: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_957, [0, 2, 3])
    convolution_backward_26 = torch.ops.aten.convolution_backward.default(view_957, view_54, primals_69, [128], [4, 4], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_957 = view_54 = primals_69 = None
    getitem_518: "f32[8, 128, 28, 28]" = convolution_backward_26[0]
    getitem_519: "f32[128, 128, 4, 4]" = convolution_backward_26[1];  convolution_backward_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    view_958: "f32[8, 128, 784]" = torch.ops.aten.view.default(getitem_518, [8, 128, 784]);  getitem_518 = None
    permute_921: "f32[8, 784, 128]" = torch.ops.aten.permute.default(view_958, [0, 2, 1]);  view_958 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_922: "f32[8, 784, 2, 64]" = torch.ops.aten.permute.default(getitem_514, [0, 2, 1, 3]);  getitem_514 = None
    view_959: "f32[8, 784, 128]" = torch.ops.aten.view.default(permute_922, [8, 784, 128]);  permute_922 = None
    view_960: "f32[6272, 128]" = torch.ops.aten.view.default(view_959, [6272, 128]);  view_959 = None
    mm_250: "f32[6272, 128]" = torch.ops.aten.mm.default(view_960, permute_923);  permute_923 = None
    permute_924: "f32[128, 6272]" = torch.ops.aten.permute.default(view_960, [1, 0])
    mm_251: "f32[128, 128]" = torch.ops.aten.mm.default(permute_924, view_51);  permute_924 = view_51 = None
    permute_925: "f32[128, 128]" = torch.ops.aten.permute.default(mm_251, [1, 0]);  mm_251 = None
    sum_449: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_960, [0], True);  view_960 = None
    view_961: "f32[128]" = torch.ops.aten.view.default(sum_449, [128]);  sum_449 = None
    permute_926: "f32[128, 128]" = torch.ops.aten.permute.default(permute_925, [1, 0]);  permute_925 = None
    view_962: "f32[8, 784, 128]" = torch.ops.aten.view.default(mm_250, [8, 784, 128]);  mm_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    add_386: "f32[8, 784, 128]" = torch.ops.aten.add.Tensor(permute_921, view_962);  permute_921 = view_962 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    clone_327: "f32[8, 784, 128]" = torch.ops.aten.clone.default(add_386, memory_format = torch.contiguous_format);  add_386 = None
    mul_950: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(clone_327, primals_65);  primals_65 = None
    mul_951: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(mul_950, 128)
    sum_450: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_950, [2], True)
    mul_952: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(mul_950, mul_31);  mul_950 = None
    sum_451: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_952, [2], True);  mul_952 = None
    mul_953: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(mul_31, sum_451);  sum_451 = None
    sub_309: "f32[8, 784, 128]" = torch.ops.aten.sub.Tensor(mul_951, sum_450);  mul_951 = sum_450 = None
    sub_310: "f32[8, 784, 128]" = torch.ops.aten.sub.Tensor(sub_309, mul_953);  sub_309 = mul_953 = None
    mul_954: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(div_75, sub_310);  div_75 = sub_310 = None
    mul_955: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(clone_327, mul_31);  mul_31 = None
    sum_452: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_955, [0, 1]);  mul_955 = None
    sum_453: "f32[128]" = torch.ops.aten.sum.dim_IntList(clone_327, [0, 1]);  clone_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_387: "f32[8, 784, 128]" = torch.ops.aten.add.Tensor(add_385, mul_954);  add_385 = mul_954 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:275, code: x = self.norm(x)
    clone_328: "f32[8, 784, 128]" = torch.ops.aten.clone.default(add_387, memory_format = torch.contiguous_format);  add_387 = None
    mul_957: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(clone_328, primals_63);  primals_63 = None
    mul_958: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(mul_957, 128)
    sum_454: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_957, [2], True)
    mul_959: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(mul_957, mul_29);  mul_957 = None
    sum_455: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_959, [2], True);  mul_959 = None
    mul_960: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(mul_29, sum_455);  sum_455 = None
    sub_312: "f32[8, 784, 128]" = torch.ops.aten.sub.Tensor(mul_958, sum_454);  mul_958 = sum_454 = None
    sub_313: "f32[8, 784, 128]" = torch.ops.aten.sub.Tensor(sub_312, mul_960);  sub_312 = mul_960 = None
    mul_961: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(div_76, sub_313);  div_76 = sub_313 = None
    mul_962: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(clone_328, mul_29);  mul_29 = None
    sum_456: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_962, [0, 1]);  mul_962 = None
    sum_457: "f32[128]" = torch.ops.aten.sum.dim_IntList(clone_328, [0, 1]);  clone_328 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:274, code: x = self.proj(x).flatten(2).transpose(1, 2)
    permute_927: "f32[8, 128, 784]" = torch.ops.aten.permute.default(mul_961, [0, 2, 1]);  mul_961 = None
    view_963: "f32[8, 128, 28, 28]" = torch.ops.aten.view.default(permute_927, [8, 128, 28, 28]);  permute_927 = None
    sum_458: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_963, [0, 2, 3])
    convolution_backward_27 = torch.ops.aten.convolution_backward.default(view_963, permute_34, primals_61, [128], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_963 = permute_34 = primals_61 = None
    getitem_521: "f32[8, 64, 56, 56]" = convolution_backward_27[0]
    getitem_522: "f32[128, 64, 2, 2]" = convolution_backward_27[1];  convolution_backward_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:415, code: x = x.reshape(B, *size, -1).permute(0, 3, 1, 2).contiguous()
    permute_928: "f32[8, 56, 56, 64]" = torch.ops.aten.permute.default(getitem_521, [0, 2, 3, 1]);  getitem_521 = None
    view_964: "f32[8, 3136, 64]" = torch.ops.aten.view.default(permute_928, [8, 3136, 64]);  permute_928 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    clone_330: "f32[8, 3136, 64]" = torch.ops.aten.clone.default(view_964, memory_format = torch.contiguous_format)
    view_965: "f32[25088, 64]" = torch.ops.aten.view.default(clone_330, [25088, 64]);  clone_330 = None
    mm_252: "f32[25088, 512]" = torch.ops.aten.mm.default(view_965, permute_929);  permute_929 = None
    permute_930: "f32[64, 25088]" = torch.ops.aten.permute.default(view_965, [1, 0])
    mm_253: "f32[64, 512]" = torch.ops.aten.mm.default(permute_930, view_47);  permute_930 = view_47 = None
    permute_931: "f32[512, 64]" = torch.ops.aten.permute.default(mm_253, [1, 0]);  mm_253 = None
    sum_459: "f32[1, 64]" = torch.ops.aten.sum.dim_IntList(view_965, [0], True);  view_965 = None
    view_966: "f32[64]" = torch.ops.aten.view.default(sum_459, [64]);  sum_459 = None
    permute_932: "f32[64, 512]" = torch.ops.aten.permute.default(permute_931, [1, 0]);  permute_931 = None
    view_967: "f32[8, 3136, 512]" = torch.ops.aten.view.default(mm_252, [8, 3136, 512]);  mm_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_964: "f32[8, 3136, 512]" = torch.ops.aten.mul.Tensor(add_28, 0.5);  add_28 = None
    mul_965: "f32[8, 3136, 512]" = torch.ops.aten.mul.Tensor(view_46, view_46)
    mul_966: "f32[8, 3136, 512]" = torch.ops.aten.mul.Tensor(mul_965, -0.5);  mul_965 = None
    exp_25: "f32[8, 3136, 512]" = torch.ops.aten.exp.default(mul_966);  mul_966 = None
    mul_967: "f32[8, 3136, 512]" = torch.ops.aten.mul.Tensor(exp_25, 0.3989422804014327);  exp_25 = None
    mul_968: "f32[8, 3136, 512]" = torch.ops.aten.mul.Tensor(view_46, mul_967);  view_46 = mul_967 = None
    add_389: "f32[8, 3136, 512]" = torch.ops.aten.add.Tensor(mul_964, mul_968);  mul_964 = mul_968 = None
    mul_969: "f32[8, 3136, 512]" = torch.ops.aten.mul.Tensor(view_967, add_389);  view_967 = add_389 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_968: "f32[25088, 512]" = torch.ops.aten.view.default(mul_969, [25088, 512]);  mul_969 = None
    mm_254: "f32[25088, 64]" = torch.ops.aten.mm.default(view_968, permute_933);  permute_933 = None
    permute_934: "f32[512, 25088]" = torch.ops.aten.permute.default(view_968, [1, 0])
    mm_255: "f32[512, 64]" = torch.ops.aten.mm.default(permute_934, view_45);  permute_934 = view_45 = None
    permute_935: "f32[64, 512]" = torch.ops.aten.permute.default(mm_255, [1, 0]);  mm_255 = None
    sum_460: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_968, [0], True);  view_968 = None
    view_969: "f32[512]" = torch.ops.aten.view.default(sum_460, [512]);  sum_460 = None
    permute_936: "f32[512, 64]" = torch.ops.aten.permute.default(permute_935, [1, 0]);  permute_935 = None
    view_970: "f32[8, 3136, 64]" = torch.ops.aten.view.default(mm_254, [8, 3136, 64]);  mm_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    mul_971: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(view_970, primals_55);  primals_55 = None
    mul_972: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(mul_971, 64)
    sum_461: "f32[8, 3136, 1]" = torch.ops.aten.sum.dim_IntList(mul_971, [2], True)
    mul_973: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(mul_971, mul_24);  mul_971 = None
    sum_462: "f32[8, 3136, 1]" = torch.ops.aten.sum.dim_IntList(mul_973, [2], True);  mul_973 = None
    mul_974: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(mul_24, sum_462);  sum_462 = None
    sub_315: "f32[8, 3136, 64]" = torch.ops.aten.sub.Tensor(mul_972, sum_461);  mul_972 = sum_461 = None
    sub_316: "f32[8, 3136, 64]" = torch.ops.aten.sub.Tensor(sub_315, mul_974);  sub_315 = mul_974 = None
    mul_975: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(div_77, sub_316);  div_77 = sub_316 = None
    mul_976: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(view_970, mul_24);  mul_24 = None
    sum_463: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_976, [0, 1]);  mul_976 = None
    sum_464: "f32[64]" = torch.ops.aten.sum.dim_IntList(view_970, [0, 1]);  view_970 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_390: "f32[8, 3136, 64]" = torch.ops.aten.add.Tensor(view_964, mul_975);  view_964 = mul_975 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    clone_332: "f32[8, 3136, 64]" = torch.ops.aten.clone.default(add_390, memory_format = torch.contiguous_format)
    view_971: "f32[25088, 64]" = torch.ops.aten.view.default(clone_332, [25088, 64]);  clone_332 = None
    mm_256: "f32[25088, 64]" = torch.ops.aten.mm.default(view_971, permute_937);  permute_937 = None
    permute_938: "f32[64, 25088]" = torch.ops.aten.permute.default(view_971, [1, 0])
    mm_257: "f32[64, 64]" = torch.ops.aten.mm.default(permute_938, view_43);  permute_938 = view_43 = None
    permute_939: "f32[64, 64]" = torch.ops.aten.permute.default(mm_257, [1, 0]);  mm_257 = None
    sum_465: "f32[1, 64]" = torch.ops.aten.sum.dim_IntList(view_971, [0], True);  view_971 = None
    view_972: "f32[64]" = torch.ops.aten.view.default(sum_465, [64]);  sum_465 = None
    permute_940: "f32[64, 64]" = torch.ops.aten.permute.default(permute_939, [1, 0]);  permute_939 = None
    view_973: "f32[8, 3136, 64]" = torch.ops.aten.view.default(mm_256, [8, 3136, 64]);  mm_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_974: "f32[8, 3136, 1, 64]" = torch.ops.aten.view.default(view_973, [8, 3136, 1, 64]);  view_973 = None
    permute_941: "f32[8, 1, 3136, 64]" = torch.ops.aten.permute.default(view_974, [0, 2, 1, 3]);  view_974 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_backward_25 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_941, permute_25, getitem_30, getitem_31, None, alias_53, getitem_33, getitem_34, getitem_35, 0.0, [True, True, True, False]);  permute_941 = permute_25 = getitem_30 = getitem_31 = alias_53 = getitem_33 = getitem_34 = getitem_35 = None
    getitem_524: "f32[8, 1, 3136, 64]" = _scaled_dot_product_efficient_attention_backward_25[0]
    getitem_525: "f32[8, 1, 49, 64]" = _scaled_dot_product_efficient_attention_backward_25[1]
    getitem_526: "f32[8, 1, 49, 64]" = _scaled_dot_product_efficient_attention_backward_25[2];  _scaled_dot_product_efficient_attention_backward_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    cat_25: "f32[16, 1, 49, 64]" = torch.ops.aten.cat.default([getitem_525, getitem_526]);  getitem_525 = getitem_526 = None
    view_975: "f32[2, 8, 1, 49, 64]" = torch.ops.aten.view.default(cat_25, [2, 8, 1, 49, 64]);  cat_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_942: "f32[8, 49, 2, 1, 64]" = torch.ops.aten.permute.default(view_975, [1, 3, 0, 2, 4]);  view_975 = None
    clone_333: "f32[8, 49, 2, 1, 64]" = torch.ops.aten.clone.default(permute_942, memory_format = torch.contiguous_format);  permute_942 = None
    view_976: "f32[8, 49, 128]" = torch.ops.aten.view.default(clone_333, [8, 49, 128]);  clone_333 = None
    view_977: "f32[392, 128]" = torch.ops.aten.view.default(view_976, [392, 128]);  view_976 = None
    mm_258: "f32[392, 64]" = torch.ops.aten.mm.default(view_977, permute_943);  permute_943 = None
    permute_944: "f32[128, 392]" = torch.ops.aten.permute.default(view_977, [1, 0])
    mm_259: "f32[128, 64]" = torch.ops.aten.mm.default(permute_944, view_39);  permute_944 = view_39 = None
    permute_945: "f32[64, 128]" = torch.ops.aten.permute.default(mm_259, [1, 0]);  mm_259 = None
    sum_466: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_977, [0], True);  view_977 = None
    view_978: "f32[128]" = torch.ops.aten.view.default(sum_466, [128]);  sum_466 = None
    permute_946: "f32[128, 64]" = torch.ops.aten.permute.default(permute_945, [1, 0]);  permute_945 = None
    view_979: "f32[8, 49, 64]" = torch.ops.aten.view.default(mm_258, [8, 49, 64]);  mm_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    mul_978: "f32[8, 49, 64]" = torch.ops.aten.mul.Tensor(view_979, primals_49);  primals_49 = None
    mul_979: "f32[8, 49, 64]" = torch.ops.aten.mul.Tensor(mul_978, 64)
    sum_467: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_978, [2], True)
    mul_980: "f32[8, 49, 64]" = torch.ops.aten.mul.Tensor(mul_978, mul_22);  mul_978 = None
    sum_468: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_980, [2], True);  mul_980 = None
    mul_981: "f32[8, 49, 64]" = torch.ops.aten.mul.Tensor(mul_22, sum_468);  sum_468 = None
    sub_318: "f32[8, 49, 64]" = torch.ops.aten.sub.Tensor(mul_979, sum_467);  mul_979 = sum_467 = None
    sub_319: "f32[8, 49, 64]" = torch.ops.aten.sub.Tensor(sub_318, mul_981);  sub_318 = mul_981 = None
    mul_982: "f32[8, 49, 64]" = torch.ops.aten.mul.Tensor(div_78, sub_319);  div_78 = sub_319 = None
    mul_983: "f32[8, 49, 64]" = torch.ops.aten.mul.Tensor(view_979, mul_22);  mul_22 = None
    sum_469: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_983, [0, 1]);  mul_983 = None
    sum_470: "f32[64]" = torch.ops.aten.sum.dim_IntList(view_979, [0, 1]);  view_979 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    permute_947: "f32[8, 64, 49]" = torch.ops.aten.permute.default(mul_982, [0, 2, 1]);  mul_982 = None
    view_980: "f32[8, 64, 7, 7]" = torch.ops.aten.view.default(permute_947, [8, 64, 7, 7]);  permute_947 = None
    sum_471: "f32[64]" = torch.ops.aten.sum.dim_IntList(view_980, [0, 2, 3])
    convolution_backward_28 = torch.ops.aten.convolution_backward.default(view_980, view_37, primals_47, [64], [8, 8], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_980 = view_37 = primals_47 = None
    getitem_528: "f32[8, 64, 56, 56]" = convolution_backward_28[0]
    getitem_529: "f32[64, 64, 8, 8]" = convolution_backward_28[1];  convolution_backward_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    view_981: "f32[8, 64, 3136]" = torch.ops.aten.view.default(getitem_528, [8, 64, 3136]);  getitem_528 = None
    permute_948: "f32[8, 3136, 64]" = torch.ops.aten.permute.default(view_981, [0, 2, 1]);  view_981 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_949: "f32[8, 3136, 1, 64]" = torch.ops.aten.permute.default(getitem_524, [0, 2, 1, 3]);  getitem_524 = None
    view_982: "f32[8, 3136, 64]" = torch.ops.aten.view.default(permute_949, [8, 3136, 64]);  permute_949 = None
    view_983: "f32[25088, 64]" = torch.ops.aten.view.default(view_982, [25088, 64]);  view_982 = None
    mm_260: "f32[25088, 64]" = torch.ops.aten.mm.default(view_983, permute_950);  permute_950 = None
    permute_951: "f32[64, 25088]" = torch.ops.aten.permute.default(view_983, [1, 0])
    mm_261: "f32[64, 64]" = torch.ops.aten.mm.default(permute_951, view_34);  permute_951 = view_34 = None
    permute_952: "f32[64, 64]" = torch.ops.aten.permute.default(mm_261, [1, 0]);  mm_261 = None
    sum_472: "f32[1, 64]" = torch.ops.aten.sum.dim_IntList(view_983, [0], True);  view_983 = None
    view_984: "f32[64]" = torch.ops.aten.view.default(sum_472, [64]);  sum_472 = None
    permute_953: "f32[64, 64]" = torch.ops.aten.permute.default(permute_952, [1, 0]);  permute_952 = None
    view_985: "f32[8, 3136, 64]" = torch.ops.aten.view.default(mm_260, [8, 3136, 64]);  mm_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    add_391: "f32[8, 3136, 64]" = torch.ops.aten.add.Tensor(permute_948, view_985);  permute_948 = view_985 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    clone_335: "f32[8, 3136, 64]" = torch.ops.aten.clone.default(add_391, memory_format = torch.contiguous_format);  add_391 = None
    mul_985: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(clone_335, primals_43);  primals_43 = None
    mul_986: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(mul_985, 64)
    sum_473: "f32[8, 3136, 1]" = torch.ops.aten.sum.dim_IntList(mul_985, [2], True)
    mul_987: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(mul_985, mul_20);  mul_985 = None
    sum_474: "f32[8, 3136, 1]" = torch.ops.aten.sum.dim_IntList(mul_987, [2], True);  mul_987 = None
    mul_988: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(mul_20, sum_474);  sum_474 = None
    sub_321: "f32[8, 3136, 64]" = torch.ops.aten.sub.Tensor(mul_986, sum_473);  mul_986 = sum_473 = None
    sub_322: "f32[8, 3136, 64]" = torch.ops.aten.sub.Tensor(sub_321, mul_988);  sub_321 = mul_988 = None
    mul_989: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(div_79, sub_322);  div_79 = sub_322 = None
    mul_990: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(clone_335, mul_20);  mul_20 = None
    sum_475: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_990, [0, 1]);  mul_990 = None
    sum_476: "f32[64]" = torch.ops.aten.sum.dim_IntList(clone_335, [0, 1]);  clone_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_392: "f32[8, 3136, 64]" = torch.ops.aten.add.Tensor(add_390, mul_989);  add_390 = mul_989 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    clone_337: "f32[8, 3136, 64]" = torch.ops.aten.clone.default(add_392, memory_format = torch.contiguous_format)
    view_986: "f32[25088, 64]" = torch.ops.aten.view.default(clone_337, [25088, 64]);  clone_337 = None
    mm_262: "f32[25088, 512]" = torch.ops.aten.mm.default(view_986, permute_954);  permute_954 = None
    permute_955: "f32[64, 25088]" = torch.ops.aten.permute.default(view_986, [1, 0])
    mm_263: "f32[64, 512]" = torch.ops.aten.mm.default(permute_955, view_32);  permute_955 = view_32 = None
    permute_956: "f32[512, 64]" = torch.ops.aten.permute.default(mm_263, [1, 0]);  mm_263 = None
    sum_477: "f32[1, 64]" = torch.ops.aten.sum.dim_IntList(view_986, [0], True);  view_986 = None
    view_987: "f32[64]" = torch.ops.aten.view.default(sum_477, [64]);  sum_477 = None
    permute_957: "f32[64, 512]" = torch.ops.aten.permute.default(permute_956, [1, 0]);  permute_956 = None
    view_988: "f32[8, 3136, 512]" = torch.ops.aten.view.default(mm_262, [8, 3136, 512]);  mm_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_992: "f32[8, 3136, 512]" = torch.ops.aten.mul.Tensor(add_19, 0.5);  add_19 = None
    mul_993: "f32[8, 3136, 512]" = torch.ops.aten.mul.Tensor(view_31, view_31)
    mul_994: "f32[8, 3136, 512]" = torch.ops.aten.mul.Tensor(mul_993, -0.5);  mul_993 = None
    exp_26: "f32[8, 3136, 512]" = torch.ops.aten.exp.default(mul_994);  mul_994 = None
    mul_995: "f32[8, 3136, 512]" = torch.ops.aten.mul.Tensor(exp_26, 0.3989422804014327);  exp_26 = None
    mul_996: "f32[8, 3136, 512]" = torch.ops.aten.mul.Tensor(view_31, mul_995);  view_31 = mul_995 = None
    add_394: "f32[8, 3136, 512]" = torch.ops.aten.add.Tensor(mul_992, mul_996);  mul_992 = mul_996 = None
    mul_997: "f32[8, 3136, 512]" = torch.ops.aten.mul.Tensor(view_988, add_394);  view_988 = add_394 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_989: "f32[25088, 512]" = torch.ops.aten.view.default(mul_997, [25088, 512]);  mul_997 = None
    mm_264: "f32[25088, 64]" = torch.ops.aten.mm.default(view_989, permute_958);  permute_958 = None
    permute_959: "f32[512, 25088]" = torch.ops.aten.permute.default(view_989, [1, 0])
    mm_265: "f32[512, 64]" = torch.ops.aten.mm.default(permute_959, view_30);  permute_959 = view_30 = None
    permute_960: "f32[64, 512]" = torch.ops.aten.permute.default(mm_265, [1, 0]);  mm_265 = None
    sum_478: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_989, [0], True);  view_989 = None
    view_990: "f32[512]" = torch.ops.aten.view.default(sum_478, [512]);  sum_478 = None
    permute_961: "f32[512, 64]" = torch.ops.aten.permute.default(permute_960, [1, 0]);  permute_960 = None
    view_991: "f32[8, 3136, 64]" = torch.ops.aten.view.default(mm_264, [8, 3136, 64]);  mm_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    mul_999: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(view_991, primals_37);  primals_37 = None
    mul_1000: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(mul_999, 64)
    sum_479: "f32[8, 3136, 1]" = torch.ops.aten.sum.dim_IntList(mul_999, [2], True)
    mul_1001: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(mul_999, mul_15);  mul_999 = None
    sum_480: "f32[8, 3136, 1]" = torch.ops.aten.sum.dim_IntList(mul_1001, [2], True);  mul_1001 = None
    mul_1002: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(mul_15, sum_480);  sum_480 = None
    sub_324: "f32[8, 3136, 64]" = torch.ops.aten.sub.Tensor(mul_1000, sum_479);  mul_1000 = sum_479 = None
    sub_325: "f32[8, 3136, 64]" = torch.ops.aten.sub.Tensor(sub_324, mul_1002);  sub_324 = mul_1002 = None
    mul_1003: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(div_80, sub_325);  div_80 = sub_325 = None
    mul_1004: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(view_991, mul_15);  mul_15 = None
    sum_481: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1004, [0, 1]);  mul_1004 = None
    sum_482: "f32[64]" = torch.ops.aten.sum.dim_IntList(view_991, [0, 1]);  view_991 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_395: "f32[8, 3136, 64]" = torch.ops.aten.add.Tensor(add_392, mul_1003);  add_392 = mul_1003 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    clone_339: "f32[8, 3136, 64]" = torch.ops.aten.clone.default(add_395, memory_format = torch.contiguous_format)
    view_992: "f32[25088, 64]" = torch.ops.aten.view.default(clone_339, [25088, 64]);  clone_339 = None
    mm_266: "f32[25088, 64]" = torch.ops.aten.mm.default(view_992, permute_962);  permute_962 = None
    permute_963: "f32[64, 25088]" = torch.ops.aten.permute.default(view_992, [1, 0])
    mm_267: "f32[64, 64]" = torch.ops.aten.mm.default(permute_963, view_28);  permute_963 = view_28 = None
    permute_964: "f32[64, 64]" = torch.ops.aten.permute.default(mm_267, [1, 0]);  mm_267 = None
    sum_483: "f32[1, 64]" = torch.ops.aten.sum.dim_IntList(view_992, [0], True);  view_992 = None
    view_993: "f32[64]" = torch.ops.aten.view.default(sum_483, [64]);  sum_483 = None
    permute_965: "f32[64, 64]" = torch.ops.aten.permute.default(permute_964, [1, 0]);  permute_964 = None
    view_994: "f32[8, 3136, 64]" = torch.ops.aten.view.default(mm_266, [8, 3136, 64]);  mm_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_995: "f32[8, 3136, 1, 64]" = torch.ops.aten.view.default(view_994, [8, 3136, 1, 64]);  view_994 = None
    permute_966: "f32[8, 1, 3136, 64]" = torch.ops.aten.permute.default(view_995, [0, 2, 1, 3]);  view_995 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_backward_26 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_966, permute_15, getitem_18, getitem_19, None, alias_54, getitem_21, getitem_22, getitem_23, 0.0, [True, True, True, False]);  permute_966 = permute_15 = getitem_18 = getitem_19 = alias_54 = getitem_21 = getitem_22 = getitem_23 = None
    getitem_531: "f32[8, 1, 3136, 64]" = _scaled_dot_product_efficient_attention_backward_26[0]
    getitem_532: "f32[8, 1, 49, 64]" = _scaled_dot_product_efficient_attention_backward_26[1]
    getitem_533: "f32[8, 1, 49, 64]" = _scaled_dot_product_efficient_attention_backward_26[2];  _scaled_dot_product_efficient_attention_backward_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    cat_26: "f32[16, 1, 49, 64]" = torch.ops.aten.cat.default([getitem_532, getitem_533]);  getitem_532 = getitem_533 = None
    view_996: "f32[2, 8, 1, 49, 64]" = torch.ops.aten.view.default(cat_26, [2, 8, 1, 49, 64]);  cat_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_967: "f32[8, 49, 2, 1, 64]" = torch.ops.aten.permute.default(view_996, [1, 3, 0, 2, 4]);  view_996 = None
    clone_340: "f32[8, 49, 2, 1, 64]" = torch.ops.aten.clone.default(permute_967, memory_format = torch.contiguous_format);  permute_967 = None
    view_997: "f32[8, 49, 128]" = torch.ops.aten.view.default(clone_340, [8, 49, 128]);  clone_340 = None
    view_998: "f32[392, 128]" = torch.ops.aten.view.default(view_997, [392, 128]);  view_997 = None
    mm_268: "f32[392, 64]" = torch.ops.aten.mm.default(view_998, permute_968);  permute_968 = None
    permute_969: "f32[128, 392]" = torch.ops.aten.permute.default(view_998, [1, 0])
    mm_269: "f32[128, 64]" = torch.ops.aten.mm.default(permute_969, view_24);  permute_969 = view_24 = None
    permute_970: "f32[64, 128]" = torch.ops.aten.permute.default(mm_269, [1, 0]);  mm_269 = None
    sum_484: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_998, [0], True);  view_998 = None
    view_999: "f32[128]" = torch.ops.aten.view.default(sum_484, [128]);  sum_484 = None
    permute_971: "f32[128, 64]" = torch.ops.aten.permute.default(permute_970, [1, 0]);  permute_970 = None
    view_1000: "f32[8, 49, 64]" = torch.ops.aten.view.default(mm_268, [8, 49, 64]);  mm_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    mul_1006: "f32[8, 49, 64]" = torch.ops.aten.mul.Tensor(view_1000, primals_31);  primals_31 = None
    mul_1007: "f32[8, 49, 64]" = torch.ops.aten.mul.Tensor(mul_1006, 64)
    sum_485: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_1006, [2], True)
    mul_1008: "f32[8, 49, 64]" = torch.ops.aten.mul.Tensor(mul_1006, mul_13);  mul_1006 = None
    sum_486: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_1008, [2], True);  mul_1008 = None
    mul_1009: "f32[8, 49, 64]" = torch.ops.aten.mul.Tensor(mul_13, sum_486);  sum_486 = None
    sub_327: "f32[8, 49, 64]" = torch.ops.aten.sub.Tensor(mul_1007, sum_485);  mul_1007 = sum_485 = None
    sub_328: "f32[8, 49, 64]" = torch.ops.aten.sub.Tensor(sub_327, mul_1009);  sub_327 = mul_1009 = None
    mul_1010: "f32[8, 49, 64]" = torch.ops.aten.mul.Tensor(div_81, sub_328);  div_81 = sub_328 = None
    mul_1011: "f32[8, 49, 64]" = torch.ops.aten.mul.Tensor(view_1000, mul_13);  mul_13 = None
    sum_487: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1011, [0, 1]);  mul_1011 = None
    sum_488: "f32[64]" = torch.ops.aten.sum.dim_IntList(view_1000, [0, 1]);  view_1000 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    permute_972: "f32[8, 64, 49]" = torch.ops.aten.permute.default(mul_1010, [0, 2, 1]);  mul_1010 = None
    view_1001: "f32[8, 64, 7, 7]" = torch.ops.aten.view.default(permute_972, [8, 64, 7, 7]);  permute_972 = None
    sum_489: "f32[64]" = torch.ops.aten.sum.dim_IntList(view_1001, [0, 2, 3])
    convolution_backward_29 = torch.ops.aten.convolution_backward.default(view_1001, view_22, primals_29, [64], [8, 8], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_1001 = view_22 = primals_29 = None
    getitem_535: "f32[8, 64, 56, 56]" = convolution_backward_29[0]
    getitem_536: "f32[64, 64, 8, 8]" = convolution_backward_29[1];  convolution_backward_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    view_1002: "f32[8, 64, 3136]" = torch.ops.aten.view.default(getitem_535, [8, 64, 3136]);  getitem_535 = None
    permute_973: "f32[8, 3136, 64]" = torch.ops.aten.permute.default(view_1002, [0, 2, 1]);  view_1002 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_974: "f32[8, 3136, 1, 64]" = torch.ops.aten.permute.default(getitem_531, [0, 2, 1, 3]);  getitem_531 = None
    view_1003: "f32[8, 3136, 64]" = torch.ops.aten.view.default(permute_974, [8, 3136, 64]);  permute_974 = None
    view_1004: "f32[25088, 64]" = torch.ops.aten.view.default(view_1003, [25088, 64]);  view_1003 = None
    mm_270: "f32[25088, 64]" = torch.ops.aten.mm.default(view_1004, permute_975);  permute_975 = None
    permute_976: "f32[64, 25088]" = torch.ops.aten.permute.default(view_1004, [1, 0])
    mm_271: "f32[64, 64]" = torch.ops.aten.mm.default(permute_976, view_19);  permute_976 = view_19 = None
    permute_977: "f32[64, 64]" = torch.ops.aten.permute.default(mm_271, [1, 0]);  mm_271 = None
    sum_490: "f32[1, 64]" = torch.ops.aten.sum.dim_IntList(view_1004, [0], True);  view_1004 = None
    view_1005: "f32[64]" = torch.ops.aten.view.default(sum_490, [64]);  sum_490 = None
    permute_978: "f32[64, 64]" = torch.ops.aten.permute.default(permute_977, [1, 0]);  permute_977 = None
    view_1006: "f32[8, 3136, 64]" = torch.ops.aten.view.default(mm_270, [8, 3136, 64]);  mm_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    add_396: "f32[8, 3136, 64]" = torch.ops.aten.add.Tensor(permute_973, view_1006);  permute_973 = view_1006 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    clone_342: "f32[8, 3136, 64]" = torch.ops.aten.clone.default(add_396, memory_format = torch.contiguous_format);  add_396 = None
    mul_1013: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(clone_342, primals_25);  primals_25 = None
    mul_1014: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(mul_1013, 64)
    sum_491: "f32[8, 3136, 1]" = torch.ops.aten.sum.dim_IntList(mul_1013, [2], True)
    mul_1015: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(mul_1013, mul_11);  mul_1013 = None
    sum_492: "f32[8, 3136, 1]" = torch.ops.aten.sum.dim_IntList(mul_1015, [2], True);  mul_1015 = None
    mul_1016: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(mul_11, sum_492);  sum_492 = None
    sub_330: "f32[8, 3136, 64]" = torch.ops.aten.sub.Tensor(mul_1014, sum_491);  mul_1014 = sum_491 = None
    sub_331: "f32[8, 3136, 64]" = torch.ops.aten.sub.Tensor(sub_330, mul_1016);  sub_330 = mul_1016 = None
    mul_1017: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(div_82, sub_331);  div_82 = sub_331 = None
    mul_1018: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(clone_342, mul_11);  mul_11 = None
    sum_493: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1018, [0, 1]);  mul_1018 = None
    sum_494: "f32[64]" = torch.ops.aten.sum.dim_IntList(clone_342, [0, 1]);  clone_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_397: "f32[8, 3136, 64]" = torch.ops.aten.add.Tensor(add_395, mul_1017);  add_395 = mul_1017 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:246, code: x = x.flatten(2).transpose(1, 2)
    permute_979: "f32[8, 64, 3136]" = torch.ops.aten.permute.default(add_397, [0, 2, 1]);  add_397 = None
    view_1007: "f32[8, 64, 56, 56]" = torch.ops.aten.view.default(permute_979, [8, 64, 56, 56]);  permute_979 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:243, code: x = self.proj(cnn_feat_token)
    sum_495: "f32[64]" = torch.ops.aten.sum.dim_IntList(view_1007, [0, 2, 3])
    convolution_backward_30 = torch.ops.aten.convolution_backward.default(view_1007, view_16, primals_23, [64], [1, 1], [1, 1], [1, 1], False, [0, 0], 64, [True, True, False]);  view_16 = primals_23 = None
    getitem_538: "f32[8, 64, 56, 56]" = convolution_backward_30[0]
    getitem_539: "f32[64, 1, 3, 3]" = convolution_backward_30[1];  convolution_backward_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:243, code: x = self.proj(cnn_feat_token)
    add_398: "f32[8, 64, 56, 56]" = torch.ops.aten.add.Tensor(view_1007, getitem_538);  view_1007 = getitem_538 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:242, code: cnn_feat_token = x.transpose(1, 2).view(B, C, *size)
    view_1008: "f32[8, 64, 3136]" = torch.ops.aten.view.default(add_398, [8, 64, 3136]);  add_398 = None
    permute_980: "f32[8, 3136, 64]" = torch.ops.aten.permute.default(view_1008, [0, 2, 1]);  view_1008 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    clone_344: "f32[8, 3136, 64]" = torch.ops.aten.clone.default(permute_980, memory_format = torch.contiguous_format)
    view_1009: "f32[25088, 64]" = torch.ops.aten.view.default(clone_344, [25088, 64]);  clone_344 = None
    mm_272: "f32[25088, 512]" = torch.ops.aten.mm.default(view_1009, permute_981);  permute_981 = None
    permute_982: "f32[64, 25088]" = torch.ops.aten.permute.default(view_1009, [1, 0])
    mm_273: "f32[64, 512]" = torch.ops.aten.mm.default(permute_982, view_14);  permute_982 = view_14 = None
    permute_983: "f32[512, 64]" = torch.ops.aten.permute.default(mm_273, [1, 0]);  mm_273 = None
    sum_496: "f32[1, 64]" = torch.ops.aten.sum.dim_IntList(view_1009, [0], True);  view_1009 = None
    view_1010: "f32[64]" = torch.ops.aten.view.default(sum_496, [64]);  sum_496 = None
    permute_984: "f32[64, 512]" = torch.ops.aten.permute.default(permute_983, [1, 0]);  permute_983 = None
    view_1011: "f32[8, 3136, 512]" = torch.ops.aten.view.default(mm_272, [8, 3136, 512]);  mm_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_1020: "f32[8, 3136, 512]" = torch.ops.aten.mul.Tensor(add_9, 0.5);  add_9 = None
    mul_1021: "f32[8, 3136, 512]" = torch.ops.aten.mul.Tensor(view_13, view_13)
    mul_1022: "f32[8, 3136, 512]" = torch.ops.aten.mul.Tensor(mul_1021, -0.5);  mul_1021 = None
    exp_27: "f32[8, 3136, 512]" = torch.ops.aten.exp.default(mul_1022);  mul_1022 = None
    mul_1023: "f32[8, 3136, 512]" = torch.ops.aten.mul.Tensor(exp_27, 0.3989422804014327);  exp_27 = None
    mul_1024: "f32[8, 3136, 512]" = torch.ops.aten.mul.Tensor(view_13, mul_1023);  view_13 = mul_1023 = None
    add_400: "f32[8, 3136, 512]" = torch.ops.aten.add.Tensor(mul_1020, mul_1024);  mul_1020 = mul_1024 = None
    mul_1025: "f32[8, 3136, 512]" = torch.ops.aten.mul.Tensor(view_1011, add_400);  view_1011 = add_400 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_1012: "f32[25088, 512]" = torch.ops.aten.view.default(mul_1025, [25088, 512]);  mul_1025 = None
    mm_274: "f32[25088, 64]" = torch.ops.aten.mm.default(view_1012, permute_985);  permute_985 = None
    permute_986: "f32[512, 25088]" = torch.ops.aten.permute.default(view_1012, [1, 0])
    mm_275: "f32[512, 64]" = torch.ops.aten.mm.default(permute_986, view_12);  permute_986 = view_12 = None
    permute_987: "f32[64, 512]" = torch.ops.aten.permute.default(mm_275, [1, 0]);  mm_275 = None
    sum_497: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_1012, [0], True);  view_1012 = None
    view_1013: "f32[512]" = torch.ops.aten.view.default(sum_497, [512]);  sum_497 = None
    permute_988: "f32[512, 64]" = torch.ops.aten.permute.default(permute_987, [1, 0]);  permute_987 = None
    view_1014: "f32[8, 3136, 64]" = torch.ops.aten.view.default(mm_274, [8, 3136, 64]);  mm_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    mul_1027: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(view_1014, primals_17);  primals_17 = None
    mul_1028: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(mul_1027, 64)
    sum_498: "f32[8, 3136, 1]" = torch.ops.aten.sum.dim_IntList(mul_1027, [2], True)
    mul_1029: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(mul_1027, mul_6);  mul_1027 = None
    sum_499: "f32[8, 3136, 1]" = torch.ops.aten.sum.dim_IntList(mul_1029, [2], True);  mul_1029 = None
    mul_1030: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(mul_6, sum_499);  sum_499 = None
    sub_333: "f32[8, 3136, 64]" = torch.ops.aten.sub.Tensor(mul_1028, sum_498);  mul_1028 = sum_498 = None
    sub_334: "f32[8, 3136, 64]" = torch.ops.aten.sub.Tensor(sub_333, mul_1030);  sub_333 = mul_1030 = None
    mul_1031: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(div_83, sub_334);  div_83 = sub_334 = None
    mul_1032: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(view_1014, mul_6);  mul_6 = None
    sum_500: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1032, [0, 1]);  mul_1032 = None
    sum_501: "f32[64]" = torch.ops.aten.sum.dim_IntList(view_1014, [0, 1]);  view_1014 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:227, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
    add_401: "f32[8, 3136, 64]" = torch.ops.aten.add.Tensor(permute_980, mul_1031);  permute_980 = mul_1031 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:185, code: x = self.proj(x)
    clone_345: "f32[8, 3136, 64]" = torch.ops.aten.clone.default(add_401, memory_format = torch.contiguous_format)
    view_1015: "f32[25088, 64]" = torch.ops.aten.view.default(clone_345, [25088, 64]);  clone_345 = None
    mm_276: "f32[25088, 64]" = torch.ops.aten.mm.default(view_1015, permute_989);  permute_989 = None
    permute_990: "f32[64, 25088]" = torch.ops.aten.permute.default(view_1015, [1, 0])
    mm_277: "f32[64, 64]" = torch.ops.aten.mm.default(permute_990, view_10);  permute_990 = view_10 = None
    permute_991: "f32[64, 64]" = torch.ops.aten.permute.default(mm_277, [1, 0]);  mm_277 = None
    sum_502: "f32[1, 64]" = torch.ops.aten.sum.dim_IntList(view_1015, [0], True);  view_1015 = None
    view_1016: "f32[64]" = torch.ops.aten.view.default(sum_502, [64]);  sum_502 = None
    permute_992: "f32[64, 64]" = torch.ops.aten.permute.default(permute_991, [1, 0]);  permute_991 = None
    view_1017: "f32[8, 3136, 64]" = torch.ops.aten.view.default(mm_276, [8, 3136, 64]);  mm_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:184, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_1018: "f32[8, 3136, 1, 64]" = torch.ops.aten.view.default(view_1017, [8, 3136, 1, 64]);  view_1017 = None
    permute_993: "f32[8, 1, 3136, 64]" = torch.ops.aten.permute.default(view_1018, [0, 2, 1, 3]);  view_1018 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:173, code: x = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_backward_27 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_993, permute_2, getitem_6, getitem_7, None, alias_55, getitem_9, getitem_10, getitem_11, 0.0, [True, True, True, False]);  permute_993 = permute_2 = getitem_6 = getitem_7 = alias_55 = getitem_9 = getitem_10 = getitem_11 = None
    getitem_541: "f32[8, 1, 3136, 64]" = _scaled_dot_product_efficient_attention_backward_27[0]
    getitem_542: "f32[8, 1, 49, 64]" = _scaled_dot_product_efficient_attention_backward_27[1]
    getitem_543: "f32[8, 1, 49, 64]" = _scaled_dot_product_efficient_attention_backward_27[2];  _scaled_dot_product_efficient_attention_backward_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:170, code: k, v = kv.unbind(0)
    cat_27: "f32[16, 1, 49, 64]" = torch.ops.aten.cat.default([getitem_542, getitem_543]);  getitem_542 = getitem_543 = None
    view_1019: "f32[2, 8, 1, 49, 64]" = torch.ops.aten.view.default(cat_27, [2, 8, 1, 49, 64]);  cat_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:169, code: kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_994: "f32[8, 49, 2, 1, 64]" = torch.ops.aten.permute.default(view_1019, [1, 3, 0, 2, 4]);  view_1019 = None
    clone_346: "f32[8, 49, 2, 1, 64]" = torch.ops.aten.clone.default(permute_994, memory_format = torch.contiguous_format);  permute_994 = None
    view_1020: "f32[8, 49, 128]" = torch.ops.aten.view.default(clone_346, [8, 49, 128]);  clone_346 = None
    view_1021: "f32[392, 128]" = torch.ops.aten.view.default(view_1020, [392, 128]);  view_1020 = None
    mm_278: "f32[392, 64]" = torch.ops.aten.mm.default(view_1021, permute_995);  permute_995 = None
    permute_996: "f32[128, 392]" = torch.ops.aten.permute.default(view_1021, [1, 0])
    mm_279: "f32[128, 64]" = torch.ops.aten.mm.default(permute_996, view_6);  permute_996 = view_6 = None
    permute_997: "f32[64, 128]" = torch.ops.aten.permute.default(mm_279, [1, 0]);  mm_279 = None
    sum_503: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_1021, [0], True);  view_1021 = None
    view_1022: "f32[128]" = torch.ops.aten.view.default(sum_503, [128]);  sum_503 = None
    permute_998: "f32[128, 64]" = torch.ops.aten.permute.default(permute_997, [1, 0]);  permute_997 = None
    view_1023: "f32[8, 49, 64]" = torch.ops.aten.view.default(mm_278, [8, 49, 64]);  mm_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:168, code: x = self.norm(x)
    mul_1034: "f32[8, 49, 64]" = torch.ops.aten.mul.Tensor(view_1023, primals_11);  primals_11 = None
    mul_1035: "f32[8, 49, 64]" = torch.ops.aten.mul.Tensor(mul_1034, 64)
    sum_504: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_1034, [2], True)
    mul_1036: "f32[8, 49, 64]" = torch.ops.aten.mul.Tensor(mul_1034, mul_4);  mul_1034 = None
    sum_505: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_1036, [2], True);  mul_1036 = None
    mul_1037: "f32[8, 49, 64]" = torch.ops.aten.mul.Tensor(mul_4, sum_505);  sum_505 = None
    sub_336: "f32[8, 49, 64]" = torch.ops.aten.sub.Tensor(mul_1035, sum_504);  mul_1035 = sum_504 = None
    sub_337: "f32[8, 49, 64]" = torch.ops.aten.sub.Tensor(sub_336, mul_1037);  sub_336 = mul_1037 = None
    mul_1038: "f32[8, 49, 64]" = torch.ops.aten.mul.Tensor(div_84, sub_337);  div_84 = sub_337 = None
    mul_1039: "f32[8, 49, 64]" = torch.ops.aten.mul.Tensor(view_1023, mul_4);  mul_4 = None
    sum_506: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1039, [0, 1]);  mul_1039 = None
    sum_507: "f32[64]" = torch.ops.aten.sum.dim_IntList(view_1023, [0, 1]);  view_1023 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:167, code: x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
    permute_999: "f32[8, 64, 49]" = torch.ops.aten.permute.default(mul_1038, [0, 2, 1]);  mul_1038 = None
    view_1024: "f32[8, 64, 7, 7]" = torch.ops.aten.view.default(permute_999, [8, 64, 7, 7]);  permute_999 = None
    sum_508: "f32[64]" = torch.ops.aten.sum.dim_IntList(view_1024, [0, 2, 3])
    convolution_backward_31 = torch.ops.aten.convolution_backward.default(view_1024, view_4, primals_9, [64], [8, 8], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_1024 = view_4 = primals_9 = None
    getitem_545: "f32[8, 64, 56, 56]" = convolution_backward_31[0]
    getitem_546: "f32[64, 64, 8, 8]" = convolution_backward_31[1];  convolution_backward_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:166, code: x = x.permute(0, 2, 1).reshape(B, C, *size)
    view_1025: "f32[8, 64, 3136]" = torch.ops.aten.view.default(getitem_545, [8, 64, 3136]);  getitem_545 = None
    permute_1000: "f32[8, 3136, 64]" = torch.ops.aten.permute.default(view_1025, [0, 2, 1]);  view_1025 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_1001: "f32[8, 3136, 1, 64]" = torch.ops.aten.permute.default(getitem_541, [0, 2, 1, 3]);  getitem_541 = None
    view_1026: "f32[8, 3136, 64]" = torch.ops.aten.view.default(permute_1001, [8, 3136, 64]);  permute_1001 = None
    view_1027: "f32[25088, 64]" = torch.ops.aten.view.default(view_1026, [25088, 64]);  view_1026 = None
    mm_280: "f32[25088, 64]" = torch.ops.aten.mm.default(view_1027, permute_1002);  permute_1002 = None
    permute_1003: "f32[64, 25088]" = torch.ops.aten.permute.default(view_1027, [1, 0])
    mm_281: "f32[64, 64]" = torch.ops.aten.mm.default(permute_1003, view_1);  permute_1003 = view_1 = None
    permute_1004: "f32[64, 64]" = torch.ops.aten.permute.default(mm_281, [1, 0]);  mm_281 = None
    sum_509: "f32[1, 64]" = torch.ops.aten.sum.dim_IntList(view_1027, [0], True);  view_1027 = None
    view_1028: "f32[64]" = torch.ops.aten.view.default(sum_509, [64]);  sum_509 = None
    permute_1005: "f32[64, 64]" = torch.ops.aten.permute.default(permute_1004, [1, 0]);  permute_1004 = None
    view_1029: "f32[8, 3136, 64]" = torch.ops.aten.view.default(mm_280, [8, 3136, 64]);  mm_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:163, code: q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    add_402: "f32[8, 3136, 64]" = torch.ops.aten.add.Tensor(permute_1000, view_1029);  permute_1000 = view_1029 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    clone_348: "f32[8, 3136, 64]" = torch.ops.aten.clone.default(add_402, memory_format = torch.contiguous_format);  add_402 = None
    mul_1041: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(clone_348, primals_5);  primals_5 = None
    mul_1042: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(mul_1041, 64)
    sum_510: "f32[8, 3136, 1]" = torch.ops.aten.sum.dim_IntList(mul_1041, [2], True)
    mul_1043: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(mul_1041, mul_2);  mul_1041 = None
    sum_511: "f32[8, 3136, 1]" = torch.ops.aten.sum.dim_IntList(mul_1043, [2], True);  mul_1043 = None
    mul_1044: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(mul_2, sum_511);  sum_511 = None
    sub_339: "f32[8, 3136, 64]" = torch.ops.aten.sub.Tensor(mul_1042, sum_510);  mul_1042 = sum_510 = None
    sub_340: "f32[8, 3136, 64]" = torch.ops.aten.sub.Tensor(sub_339, mul_1044);  sub_339 = mul_1044 = None
    mul_1045: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(div_85, sub_340);  div_85 = sub_340 = None
    mul_1046: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(clone_348, mul_2);  mul_2 = None
    sum_512: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1046, [0, 1]);  mul_1046 = None
    sum_513: "f32[64]" = torch.ops.aten.sum.dim_IntList(clone_348, [0, 1]);  clone_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:226, code: x = x + self.drop_path1(self.attn(self.norm1(x), size))
    add_403: "f32[8, 3136, 64]" = torch.ops.aten.add.Tensor(add_401, mul_1045);  add_401 = mul_1045 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:275, code: x = self.norm(x)
    clone_349: "f32[8, 3136, 64]" = torch.ops.aten.clone.default(add_403, memory_format = torch.contiguous_format);  add_403 = None
    mul_1048: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(clone_349, primals_3);  primals_3 = None
    mul_1049: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(mul_1048, 64)
    sum_514: "f32[8, 3136, 1]" = torch.ops.aten.sum.dim_IntList(mul_1048, [2], True)
    mul_1050: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(mul_1048, mul);  mul_1048 = None
    sum_515: "f32[8, 3136, 1]" = torch.ops.aten.sum.dim_IntList(mul_1050, [2], True);  mul_1050 = None
    mul_1051: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(mul, sum_515);  sum_515 = None
    sub_342: "f32[8, 3136, 64]" = torch.ops.aten.sub.Tensor(mul_1049, sum_514);  mul_1049 = sum_514 = None
    sub_343: "f32[8, 3136, 64]" = torch.ops.aten.sub.Tensor(sub_342, mul_1051);  sub_342 = mul_1051 = None
    mul_1052: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(div_86, sub_343);  div_86 = sub_343 = None
    mul_1053: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(clone_349, mul);  mul = None
    sum_516: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1053, [0, 1]);  mul_1053 = None
    sum_517: "f32[64]" = torch.ops.aten.sum.dim_IntList(clone_349, [0, 1]);  clone_349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/twins.py:274, code: x = self.proj(x).flatten(2).transpose(1, 2)
    permute_1006: "f32[8, 64, 3136]" = torch.ops.aten.permute.default(mul_1052, [0, 2, 1]);  mul_1052 = None
    view_1030: "f32[8, 64, 56, 56]" = torch.ops.aten.view.default(permute_1006, [8, 64, 56, 56]);  permute_1006 = None
    sum_518: "f32[64]" = torch.ops.aten.sum.dim_IntList(view_1030, [0, 2, 3])
    convolution_backward_32 = torch.ops.aten.convolution_backward.default(view_1030, primals_521, primals_1, [64], [4, 4], [0, 0], [1, 1], False, [0, 0], 1, [False, True, False]);  view_1030 = primals_521 = primals_1 = None
    getitem_549: "f32[64, 3, 4, 4]" = convolution_backward_32[1];  convolution_backward_32 = None
    return [getitem_549, sum_518, sum_516, sum_517, sum_512, sum_513, permute_1005, view_1028, getitem_546, sum_508, sum_506, sum_507, permute_998, view_1022, permute_992, view_1016, sum_500, sum_501, permute_988, view_1013, permute_984, view_1010, getitem_539, sum_495, sum_493, sum_494, permute_978, view_1005, getitem_536, sum_489, sum_487, sum_488, permute_971, view_999, permute_965, view_993, sum_481, sum_482, permute_961, view_990, permute_957, view_987, sum_475, sum_476, permute_953, view_984, getitem_529, sum_471, sum_469, sum_470, permute_946, view_978, permute_940, view_972, sum_463, sum_464, permute_936, view_969, permute_932, view_966, getitem_522, sum_458, sum_456, sum_457, sum_452, sum_453, permute_926, view_961, getitem_519, sum_448, sum_446, sum_447, permute_919, view_955, permute_913, view_949, sum_440, sum_441, permute_909, view_946, permute_905, view_943, getitem_512, sum_435, sum_433, sum_434, permute_899, view_938, getitem_509, sum_429, sum_427, sum_428, permute_892, view_932, permute_886, view_926, sum_421, sum_422, permute_882, view_923, permute_878, view_920, sum_415, sum_416, permute_874, view_917, getitem_502, sum_411, sum_409, sum_410, permute_867, view_911, permute_861, view_905, sum_403, sum_404, permute_857, view_902, permute_853, view_899, sum_397, sum_398, permute_849, view_896, getitem_495, sum_393, sum_391, sum_392, permute_842, view_890, permute_836, view_884, sum_385, sum_386, permute_832, view_881, permute_828, view_878, getitem_488, sum_380, sum_378, sum_379, sum_374, sum_375, permute_822, view_873, getitem_485, sum_370, sum_368, sum_369, permute_815, view_867, permute_809, view_861, sum_362, sum_363, permute_805, view_858, permute_801, view_855, getitem_478, sum_357, sum_355, sum_356, permute_795, view_850, getitem_475, sum_351, sum_349, sum_350, permute_788, view_844, permute_782, view_838, sum_343, sum_344, permute_778, view_835, permute_774, view_832, sum_337, sum_338, permute_770, view_829, getitem_468, sum_333, sum_331, sum_332, permute_763, view_823, permute_757, view_817, sum_325, sum_326, permute_753, view_814, permute_749, view_811, sum_319, sum_320, permute_745, view_808, getitem_461, sum_315, sum_313, sum_314, permute_738, view_802, permute_732, view_796, sum_307, sum_308, permute_728, view_793, permute_724, view_790, sum_301, sum_302, permute_720, view_787, getitem_454, sum_297, sum_295, sum_296, permute_713, view_781, permute_707, view_775, sum_289, sum_290, permute_703, view_772, permute_699, view_769, sum_283, sum_284, permute_695, view_766, getitem_447, sum_279, sum_277, sum_278, permute_688, view_760, permute_682, view_754, sum_271, sum_272, permute_678, view_751, permute_674, view_748, sum_265, sum_266, permute_670, view_745, getitem_440, sum_261, sum_259, sum_260, permute_663, view_739, permute_657, view_733, sum_253, sum_254, permute_653, view_730, permute_649, view_727, sum_247, sum_248, permute_645, view_724, getitem_433, sum_243, sum_241, sum_242, permute_638, view_718, permute_632, view_712, sum_235, sum_236, permute_628, view_709, permute_624, view_706, sum_229, sum_230, permute_620, view_703, getitem_426, sum_225, sum_223, sum_224, permute_613, view_697, permute_607, view_691, sum_217, sum_218, permute_603, view_688, permute_599, view_685, sum_211, sum_212, permute_595, view_682, getitem_419, sum_207, sum_205, sum_206, permute_588, view_676, permute_582, view_670, sum_199, sum_200, permute_578, view_667, permute_574, view_664, sum_193, sum_194, permute_570, view_661, getitem_412, sum_189, sum_187, sum_188, permute_563, view_655, permute_557, view_649, sum_181, sum_182, permute_553, view_646, permute_549, view_643, sum_175, sum_176, permute_545, view_640, getitem_405, sum_171, sum_169, sum_170, permute_538, view_634, permute_532, view_628, sum_163, sum_164, permute_528, view_625, permute_524, view_622, sum_157, sum_158, permute_520, view_619, getitem_398, sum_153, sum_151, sum_152, permute_513, view_613, permute_507, view_607, sum_145, sum_146, permute_503, view_604, permute_499, view_601, sum_139, sum_140, permute_495, view_598, getitem_391, sum_135, sum_133, sum_134, permute_488, view_592, permute_482, view_586, sum_127, sum_128, permute_478, view_583, permute_474, view_580, sum_121, sum_122, permute_470, view_577, getitem_384, sum_117, sum_115, sum_116, permute_463, view_571, permute_457, view_565, sum_109, sum_110, permute_453, view_562, permute_449, view_559, sum_103, sum_104, permute_445, view_556, getitem_377, sum_99, sum_97, sum_98, permute_438, view_550, permute_432, view_544, sum_91, sum_92, permute_428, view_541, permute_424, view_538, sum_85, sum_86, permute_420, view_535, getitem_370, sum_81, sum_79, sum_80, permute_413, view_529, permute_407, view_523, sum_73, sum_74, permute_403, view_520, permute_399, view_517, sum_67, sum_68, permute_395, view_514, getitem_363, sum_63, sum_61, sum_62, permute_388, view_508, permute_382, view_502, sum_55, sum_56, permute_378, view_499, permute_374, view_496, getitem_356, sum_50, sum_48, sum_49, sum_44, sum_45, permute_368, view_491, permute_363, view_487, permute_357, view_481, sum_37, sum_38, permute_353, view_478, permute_349, view_475, getitem_349, sum_32, sum_30, sum_31, permute_343, view_470, permute_338, view_466, permute_332, view_460, sum_23, sum_24, permute_328, view_457, permute_324, view_454, sum_17, sum_18, permute_320, view_451, permute_315, view_447, permute_309, view_441, sum_10, sum_11, permute_305, view_438, permute_301, view_435, sum_4, sum_5, permute_297, view_433, None]
    