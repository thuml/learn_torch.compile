from __future__ import annotations



def forward(self, primals, tangents):
    primals_1: "f32[50257, 2048]"; primals_2: "f32[2048, 2048]"; primals_3: "f32[2048]"; primals_4: "f32[2048]"; primals_5: "f32[2048, 2048]"; primals_6: "f32[2048, 2048]"; primals_7: "f32[2048, 2048]"; primals_8: "f32[2048, 2048]"; primals_9: "f32[2048]"; primals_10: "f32[2048]"; primals_11: "f32[2048]"; primals_12: "f32[8192, 2048]"; primals_13: "f32[8192]"; primals_14: "f32[2048, 8192]"; primals_15: "f32[2048]"; primals_16: "f32[2048]"; primals_17: "f32[2048]"; primals_18: "f32[2048, 2048]"; primals_19: "f32[2048, 2048]"; primals_20: "f32[2048, 2048]"; primals_21: "f32[2048, 2048]"; primals_22: "f32[2048]"; primals_23: "f32[2048]"; primals_24: "f32[2048]"; primals_25: "f32[8192, 2048]"; primals_26: "f32[8192]"; primals_27: "f32[2048, 8192]"; primals_28: "f32[2048]"; primals_29: "f32[2048]"; primals_30: "f32[2048]"; primals_31: "f32[2048, 2048]"; primals_32: "f32[2048, 2048]"; primals_33: "f32[2048, 2048]"; primals_34: "f32[2048, 2048]"; primals_35: "f32[2048]"; primals_36: "f32[2048]"; primals_37: "f32[2048]"; primals_38: "f32[8192, 2048]"; primals_39: "f32[8192]"; primals_40: "f32[2048, 8192]"; primals_41: "f32[2048]"; primals_42: "f32[2048]"; primals_43: "f32[2048]"; primals_44: "f32[2048, 2048]"; primals_45: "f32[2048, 2048]"; primals_46: "f32[2048, 2048]"; primals_47: "f32[2048, 2048]"; primals_48: "f32[2048]"; primals_49: "f32[2048]"; primals_50: "f32[2048]"; primals_51: "f32[8192, 2048]"; primals_52: "f32[8192]"; primals_53: "f32[2048, 8192]"; primals_54: "f32[2048]"; primals_55: "f32[2048]"; primals_56: "f32[2048]"; primals_57: "f32[2048, 2048]"; primals_58: "f32[2048, 2048]"; primals_59: "f32[2048, 2048]"; primals_60: "f32[2048, 2048]"; primals_61: "f32[2048]"; primals_62: "f32[2048]"; primals_63: "f32[2048]"; primals_64: "f32[8192, 2048]"; primals_65: "f32[8192]"; primals_66: "f32[2048, 8192]"; primals_67: "f32[2048]"; primals_68: "f32[2048]"; primals_69: "f32[2048]"; primals_70: "f32[2048, 2048]"; primals_71: "f32[2048, 2048]"; primals_72: "f32[2048, 2048]"; primals_73: "f32[2048, 2048]"; primals_74: "f32[2048]"; primals_75: "f32[2048]"; primals_76: "f32[2048]"; primals_77: "f32[8192, 2048]"; primals_78: "f32[8192]"; primals_79: "f32[2048, 8192]"; primals_80: "f32[2048]"; primals_81: "f32[2048]"; primals_82: "f32[2048]"; primals_83: "f32[2048, 2048]"; primals_84: "f32[2048, 2048]"; primals_85: "f32[2048, 2048]"; primals_86: "f32[2048, 2048]"; primals_87: "f32[2048]"; primals_88: "f32[2048]"; primals_89: "f32[2048]"; primals_90: "f32[8192, 2048]"; primals_91: "f32[8192]"; primals_92: "f32[2048, 8192]"; primals_93: "f32[2048]"; primals_94: "f32[2048]"; primals_95: "f32[2048]"; primals_96: "f32[2048, 2048]"; primals_97: "f32[2048, 2048]"; primals_98: "f32[2048, 2048]"; primals_99: "f32[2048, 2048]"; primals_100: "f32[2048]"; primals_101: "f32[2048]"; primals_102: "f32[2048]"; primals_103: "f32[8192, 2048]"; primals_104: "f32[8192]"; primals_105: "f32[2048, 8192]"; primals_106: "f32[2048]"; primals_107: "f32[2048]"; primals_108: "f32[2048]"; primals_109: "f32[2048, 2048]"; primals_110: "f32[2048, 2048]"; primals_111: "f32[2048, 2048]"; primals_112: "f32[2048, 2048]"; primals_113: "f32[2048]"; primals_114: "f32[2048]"; primals_115: "f32[2048]"; primals_116: "f32[8192, 2048]"; primals_117: "f32[8192]"; primals_118: "f32[2048, 8192]"; primals_119: "f32[2048]"; primals_120: "f32[2048]"; primals_121: "f32[2048]"; primals_122: "f32[2048, 2048]"; primals_123: "f32[2048, 2048]"; primals_124: "f32[2048, 2048]"; primals_125: "f32[2048, 2048]"; primals_126: "f32[2048]"; primals_127: "f32[2048]"; primals_128: "f32[2048]"; primals_129: "f32[8192, 2048]"; primals_130: "f32[8192]"; primals_131: "f32[2048, 8192]"; primals_132: "f32[2048]"; primals_133: "f32[2048]"; primals_134: "f32[2048]"; primals_135: "f32[2048, 2048]"; primals_136: "f32[2048, 2048]"; primals_137: "f32[2048, 2048]"; primals_138: "f32[2048, 2048]"; primals_139: "f32[2048]"; primals_140: "f32[2048]"; primals_141: "f32[2048]"; primals_142: "f32[8192, 2048]"; primals_143: "f32[8192]"; primals_144: "f32[2048, 8192]"; primals_145: "f32[2048]"; primals_146: "f32[2048]"; primals_147: "f32[2048]"; primals_148: "f32[2048, 2048]"; primals_149: "f32[2048, 2048]"; primals_150: "f32[2048, 2048]"; primals_151: "f32[2048, 2048]"; primals_152: "f32[2048]"; primals_153: "f32[2048]"; primals_154: "f32[2048]"; primals_155: "f32[8192, 2048]"; primals_156: "f32[8192]"; primals_157: "f32[2048, 8192]"; primals_158: "f32[2048]"; primals_159: "f32[2048]"; primals_160: "f32[2048]"; primals_161: "f32[2048, 2048]"; primals_162: "f32[2048, 2048]"; primals_163: "f32[2048, 2048]"; primals_164: "f32[2048, 2048]"; primals_165: "f32[2048]"; primals_166: "f32[2048]"; primals_167: "f32[2048]"; primals_168: "f32[8192, 2048]"; primals_169: "f32[8192]"; primals_170: "f32[2048, 8192]"; primals_171: "f32[2048]"; primals_172: "f32[2048]"; primals_173: "f32[2048]"; primals_174: "f32[2048, 2048]"; primals_175: "f32[2048, 2048]"; primals_176: "f32[2048, 2048]"; primals_177: "f32[2048, 2048]"; primals_178: "f32[2048]"; primals_179: "f32[2048]"; primals_180: "f32[2048]"; primals_181: "f32[8192, 2048]"; primals_182: "f32[8192]"; primals_183: "f32[2048, 8192]"; primals_184: "f32[2048]"; primals_185: "f32[2048]"; primals_186: "f32[2048]"; primals_187: "f32[2048, 2048]"; primals_188: "f32[2048, 2048]"; primals_189: "f32[2048, 2048]"; primals_190: "f32[2048, 2048]"; primals_191: "f32[2048]"; primals_192: "f32[2048]"; primals_193: "f32[2048]"; primals_194: "f32[8192, 2048]"; primals_195: "f32[8192]"; primals_196: "f32[2048, 8192]"; primals_197: "f32[2048]"; primals_198: "f32[2048]"; primals_199: "f32[2048]"; primals_200: "f32[2048, 2048]"; primals_201: "f32[2048, 2048]"; primals_202: "f32[2048, 2048]"; primals_203: "f32[2048, 2048]"; primals_204: "f32[2048]"; primals_205: "f32[2048]"; primals_206: "f32[2048]"; primals_207: "f32[8192, 2048]"; primals_208: "f32[8192]"; primals_209: "f32[2048, 8192]"; primals_210: "f32[2048]"; primals_211: "f32[2048]"; primals_212: "f32[2048]"; primals_213: "f32[2048, 2048]"; primals_214: "f32[2048, 2048]"; primals_215: "f32[2048, 2048]"; primals_216: "f32[2048, 2048]"; primals_217: "f32[2048]"; primals_218: "f32[2048]"; primals_219: "f32[2048]"; primals_220: "f32[8192, 2048]"; primals_221: "f32[8192]"; primals_222: "f32[2048, 8192]"; primals_223: "f32[2048]"; primals_224: "f32[2048]"; primals_225: "f32[2048]"; primals_226: "f32[2048, 2048]"; primals_227: "f32[2048, 2048]"; primals_228: "f32[2048, 2048]"; primals_229: "f32[2048, 2048]"; primals_230: "f32[2048]"; primals_231: "f32[2048]"; primals_232: "f32[2048]"; primals_233: "f32[8192, 2048]"; primals_234: "f32[8192]"; primals_235: "f32[2048, 8192]"; primals_236: "f32[2048]"; primals_237: "f32[2048]"; primals_238: "f32[2048]"; primals_239: "f32[2048, 2048]"; primals_240: "f32[2048, 2048]"; primals_241: "f32[2048, 2048]"; primals_242: "f32[2048, 2048]"; primals_243: "f32[2048]"; primals_244: "f32[2048]"; primals_245: "f32[2048]"; primals_246: "f32[8192, 2048]"; primals_247: "f32[8192]"; primals_248: "f32[2048, 8192]"; primals_249: "f32[2048]"; primals_250: "f32[2048]"; primals_251: "f32[2048]"; primals_252: "f32[2048, 2048]"; primals_253: "f32[2048, 2048]"; primals_254: "f32[2048, 2048]"; primals_255: "f32[2048, 2048]"; primals_256: "f32[2048]"; primals_257: "f32[2048]"; primals_258: "f32[2048]"; primals_259: "f32[8192, 2048]"; primals_260: "f32[8192]"; primals_261: "f32[2048, 8192]"; primals_262: "f32[2048]"; primals_263: "f32[2048]"; primals_264: "f32[2048]"; primals_265: "f32[2048, 2048]"; primals_266: "f32[2048, 2048]"; primals_267: "f32[2048, 2048]"; primals_268: "f32[2048, 2048]"; primals_269: "f32[2048]"; primals_270: "f32[2048]"; primals_271: "f32[2048]"; primals_272: "f32[8192, 2048]"; primals_273: "f32[8192]"; primals_274: "f32[2048, 8192]"; primals_275: "f32[2048]"; primals_276: "f32[2048]"; primals_277: "f32[2048]"; primals_278: "f32[2048, 2048]"; primals_279: "f32[2048, 2048]"; primals_280: "f32[2048, 2048]"; primals_281: "f32[2048, 2048]"; primals_282: "f32[2048]"; primals_283: "f32[2048]"; primals_284: "f32[2048]"; primals_285: "f32[8192, 2048]"; primals_286: "f32[8192]"; primals_287: "f32[2048, 8192]"; primals_288: "f32[2048]"; primals_289: "f32[2048]"; primals_290: "f32[2048]"; primals_291: "f32[2048, 2048]"; primals_292: "f32[2048, 2048]"; primals_293: "f32[2048, 2048]"; primals_294: "f32[2048, 2048]"; primals_295: "f32[2048]"; primals_296: "f32[2048]"; primals_297: "f32[2048]"; primals_298: "f32[8192, 2048]"; primals_299: "f32[8192]"; primals_300: "f32[2048, 8192]"; primals_301: "f32[2048]"; primals_302: "f32[2048]"; primals_303: "f32[2048]"; primals_304: "f32[2048, 2048]"; primals_305: "f32[2048, 2048]"; primals_306: "f32[2048, 2048]"; primals_307: "f32[2048, 2048]"; primals_308: "f32[2048]"; primals_309: "f32[2048]"; primals_310: "f32[2048]"; primals_311: "f32[8192, 2048]"; primals_312: "f32[8192]"; primals_313: "f32[2048, 8192]"; primals_314: "f32[2048]"; primals_315: "f32[2048]"; primals_316: "f32[2048]"; primals_317: "f32[2, 2048]"; primals_318: "b8[1, 1, 2048, 2048]"; primals_319: "b8[1, 1, 2048, 2048]"; primals_320: "b8[1, 1, 2048, 2048]"; primals_321: "b8[1, 1, 2048, 2048]"; primals_322: "b8[1, 1, 2048, 2048]"; primals_323: "b8[1, 1, 2048, 2048]"; primals_324: "b8[1, 1, 2048, 2048]"; primals_325: "b8[1, 1, 2048, 2048]"; primals_326: "b8[1, 1, 2048, 2048]"; primals_327: "b8[1, 1, 2048, 2048]"; primals_328: "b8[1, 1, 2048, 2048]"; primals_329: "b8[1, 1, 2048, 2048]"; primals_330: "b8[1, 1, 2048, 2048]"; primals_331: "b8[1, 1, 2048, 2048]"; primals_332: "b8[1, 1, 2048, 2048]"; primals_333: "b8[1, 1, 2048, 2048]"; primals_334: "b8[1, 1, 2048, 2048]"; primals_335: "b8[1, 1, 2048, 2048]"; primals_336: "b8[1, 1, 2048, 2048]"; primals_337: "b8[1, 1, 2048, 2048]"; primals_338: "b8[1, 1, 2048, 2048]"; primals_339: "b8[1, 1, 2048, 2048]"; primals_340: "b8[1, 1, 2048, 2048]"; primals_341: "b8[1, 1, 2048, 2048]"; primals_342: "i64[1, 128]"; tangents_1: "f32[1, 128, 2048]"; tangents_2: "f32[1, 16, 128, 128]"; tangents_3: "f32[1, 16, 128, 128]"; tangents_4: "f32[1, 16, 128, 128]"; tangents_5: "f32[1, 16, 128, 128]"; tangents_6: "f32[1, 16, 128, 128]"; tangents_7: "f32[1, 16, 128, 128]"; tangents_8: "f32[1, 16, 128, 128]"; tangents_9: "f32[1, 16, 128, 128]"; tangents_10: "f32[1, 16, 128, 128]"; tangents_11: "f32[1, 16, 128, 128]"; tangents_12: "f32[1, 16, 128, 128]"; tangents_13: "f32[1, 16, 128, 128]"; tangents_14: "f32[1, 16, 128, 128]"; tangents_15: "f32[1, 16, 128, 128]"; tangents_16: "f32[1, 16, 128, 128]"; tangents_17: "f32[1, 16, 128, 128]"; tangents_18: "f32[1, 16, 128, 128]"; tangents_19: "f32[1, 16, 128, 128]"; tangents_20: "f32[1, 16, 128, 128]"; tangents_21: "f32[1, 16, 128, 128]"; tangents_22: "f32[1, 16, 128, 128]"; tangents_23: "f32[1, 16, 128, 128]"; tangents_24: "f32[1, 16, 128, 128]"; tangents_25: "f32[1, 16, 128, 128]"; tangents_26: "f32[1, 16, 128, 128]"; tangents_27: "f32[1, 16, 128, 128]"; tangents_28: "f32[1, 16, 128, 128]"; tangents_29: "f32[1, 16, 128, 128]"; tangents_30: "f32[1, 16, 128, 128]"; tangents_31: "f32[1, 16, 128, 128]"; tangents_32: "f32[1, 16, 128, 128]"; tangents_33: "f32[1, 16, 128, 128]"; tangents_34: "f32[1, 16, 128, 128]"; tangents_35: "f32[1, 16, 128, 128]"; tangents_36: "f32[1, 16, 128, 128]"; tangents_37: "f32[1, 16, 128, 128]"; tangents_38: "f32[1, 16, 128, 128]"; tangents_39: "f32[1, 16, 128, 128]"; tangents_40: "f32[1, 16, 128, 128]"; tangents_41: "f32[1, 16, 128, 128]"; tangents_42: "f32[1, 16, 128, 128]"; tangents_43: "f32[1, 16, 128, 128]"; tangents_44: "f32[1, 16, 128, 128]"; tangents_45: "f32[1, 16, 128, 128]"; tangents_46: "f32[1, 16, 128, 128]"; tangents_47: "f32[1, 16, 128, 128]"; tangents_48: "f32[1, 16, 128, 128]"; tangents_49: "f32[1, 16, 128, 128]"; tangents_50: "f32[1, 2]"; 

    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, tangents_1, tangents_2, tangents_3, tangents_4, tangents_5, tangents_6, tangents_7, tangents_8, tangents_9, tangents_10, tangents_11, tangents_12, tangents_13, tangents_14, tangents_15, tangents_16, tangents_17, tangents_18, tangents_19, tangents_20, tangents_21, tangents_22, tangents_23, tangents_24, tangents_25, tangents_26, tangents_27, tangents_28, tangents_29, tangents_30, tangents_31, tangents_32, tangents_33, tangents_34, tangents_35, tangents_36, tangents_37, tangents_38, tangents_39, tangents_40, tangents_41, tangents_42, tangents_43, tangents_44, tangents_45, tangents_46, tangents_47, tangents_48, tangents_49, tangents_50, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:530, code: input_ids = input_ids.view(-1, input_shape[-1])
    view: "i64[1, 128]" = torch.ops.aten.view.default(primals_342, [-1, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:552, code: position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
    iota: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:553, code: position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
    unsqueeze: "i64[1, 128]" = torch.ops.aten.unsqueeze.default(iota, 0);  iota = None
    view_1: "i64[1, 128]" = torch.ops.aten.view.default(unsqueeze, [-1, 128]);  unsqueeze = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:582, code: inputs_embeds = self.wte(input_ids)
    embedding: "f32[1, 128, 2048]" = torch.ops.aten.embedding.default(primals_1, view);  primals_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:583, code: position_embeds = self.wpe(position_ids)
    embedding_1: "f32[1, 128, 2048]" = torch.ops.aten.embedding.default(primals_2, view_1);  primals_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:584, code: hidden_states = inputs_embeds + position_embeds
    add: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:590, code: hidden_states = self.drop(hidden_states)
    clone: "f32[1, 128, 2048]" = torch.ops.aten.clone.default(add);  add = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    var_mean = torch.ops.aten.var_mean.correction(clone, [2], correction = 0, keepdim = True)
    getitem: "f32[1, 128, 1]" = var_mean[0]
    getitem_1: "f32[1, 128, 1]" = var_mean[1];  var_mean = None
    add_1: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
    rsqrt: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
    sub: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(clone, getitem_1)
    mul: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
    mul_1: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul, primals_3);  mul = None
    add_2: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_1, primals_4);  mul_1 = primals_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_5, [1, 0]);  primals_5 = None
    view_2: "f32[128, 2048]" = torch.ops.aten.view.default(add_2, [128, 2048])
    mm: "f32[128, 2048]" = torch.ops.aten.mm.default(view_2, permute)
    view_3: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm, [1, 128, 2048]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_1: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_6, [1, 0]);  primals_6 = None
    view_4: "f32[128, 2048]" = torch.ops.aten.view.default(add_2, [128, 2048])
    mm_1: "f32[128, 2048]" = torch.ops.aten.mm.default(view_4, permute_1)
    view_5: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_1, [1, 128, 2048]);  mm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    permute_2: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_7, [1, 0]);  primals_7 = None
    view_6: "f32[128, 2048]" = torch.ops.aten.view.default(add_2, [128, 2048]);  add_2 = None
    mm_2: "f32[128, 2048]" = torch.ops.aten.mm.default(view_6, permute_2)
    view_7: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_2, [1, 128, 2048]);  mm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_8: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_3, [1, 128, 16, 128]);  view_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_3: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_9: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_5, [1, 128, 16, 128]);  view_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_4: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_9, [0, 2, 1, 3]);  view_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_10: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_7, [1, 128, 16, 128]);  view_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_5: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_10, [0, 2, 1, 3]);  view_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_6: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(permute_4, [0, 1, 3, 2])
    expand: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_3, [1, 16, 128, 128]);  permute_3 = None
    view_11: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand, [16, 128, 128]);  expand = None
    expand_1: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_6, [1, 16, 128, 128]);  permute_6 = None
    view_12: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_1, [16, 128, 128]);  expand_1 = None
    bmm: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_11, view_12)
    view_13: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm, [1, 16, 128, 128]);  bmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_1: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(primals_318, 0, 0, 9223372036854775807);  primals_318 = None
    slice_2: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(slice_1, 1, 0, 9223372036854775807);  slice_1 = None
    slice_3: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(slice_2, 2, 0, 128);  slice_2 = None
    slice_4: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_3, 3, 0, 128);  slice_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    _tensor_constant0 = self._tensor_constant0
    lift_fresh_copy: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant0);  _tensor_constant0 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_4, view_13, lift_fresh_copy);  view_13 = lift_fresh_copy = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(where, [-1], True)
    sub_1: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(where, amax);  where = amax = None
    exp: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
    sum_1: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    alias: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(div)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:205, code: attn_weights = self.attn_dropout(attn_weights)
    clone_1: "f32[1, 16, 128, 128]" = torch.ops.aten.clone.default(div);  div = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_2: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(clone_1, [1, 16, 128, 128]);  clone_1 = None
    view_14: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_2, [16, 128, 128]);  expand_2 = None
    expand_3: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_5, [1, 16, 128, 128])
    view_15: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_3, [16, 128, 128]);  expand_3 = None
    bmm_1: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_14, view_15)
    view_16: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_1, [1, 16, 128, 128]);  bmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_7: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_16, [0, 2, 1, 3]);  view_16 = None
    clone_2: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_17: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_2, [1, 128, 2048]);  clone_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_18: "f32[128, 2048]" = torch.ops.aten.view.default(view_17, [128, 2048]);  view_17 = None
    permute_8: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_8, [1, 0]);  primals_8 = None
    addmm: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_9, view_18, permute_8);  primals_9 = None
    view_19: "f32[1, 128, 2048]" = torch.ops.aten.view.default(addmm, [1, 128, 2048]);  addmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:247, code: attn_output = self.resid_dropout(attn_output)
    clone_3: "f32[1, 128, 2048]" = torch.ops.aten.clone.default(view_19);  view_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    add_3: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(clone_3, clone);  clone_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    var_mean_1 = torch.ops.aten.var_mean.correction(add_3, [2], correction = 0, keepdim = True)
    getitem_2: "f32[1, 128, 1]" = var_mean_1[0]
    getitem_3: "f32[1, 128, 1]" = var_mean_1[1];  var_mean_1 = None
    add_4: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
    rsqrt_1: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
    sub_2: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_3, getitem_3)
    mul_2: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = None
    mul_3: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_2, primals_10);  mul_2 = None
    add_5: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_3, primals_11);  mul_3 = primals_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_20: "f32[128, 2048]" = torch.ops.aten.view.default(add_5, [128, 2048]);  add_5 = None
    permute_9: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_12, [1, 0]);  primals_12 = None
    addmm_1: "f32[128, 8192]" = torch.ops.aten.addmm.default(primals_13, view_20, permute_9);  primals_13 = None
    view_21: "f32[1, 128, 8192]" = torch.ops.aten.view.default(addmm_1, [1, 128, 8192]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_4: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_21, 0.5)
    pow_1: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_21, 3.0)
    mul_5: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
    add_6: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(view_21, mul_5);  mul_5 = None
    mul_6: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(add_6, 0.7978845608028654);  add_6 = None
    tanh: "f32[1, 128, 8192]" = torch.ops.aten.tanh.default(mul_6);  mul_6 = None
    alias_1: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(tanh)
    add_7: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh, 1.0);  tanh = None
    mul_7: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_4, add_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_22: "f32[128, 8192]" = torch.ops.aten.view.default(mul_7, [128, 8192]);  mul_7 = None
    permute_10: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_14, [1, 0]);  primals_14 = None
    addmm_2: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_15, view_22, permute_10);  primals_15 = None
    view_23: "f32[1, 128, 2048]" = torch.ops.aten.view.default(addmm_2, [1, 128, 2048]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:303, code: hidden_states = self.dropout(hidden_states)
    clone_4: "f32[1, 128, 2048]" = torch.ops.aten.clone.default(view_23);  view_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    add_8: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_3, clone_4);  clone_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    var_mean_2 = torch.ops.aten.var_mean.correction(add_8, [2], correction = 0, keepdim = True)
    getitem_4: "f32[1, 128, 1]" = var_mean_2[0]
    getitem_5: "f32[1, 128, 1]" = var_mean_2[1];  var_mean_2 = None
    add_9: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
    rsqrt_2: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_9);  add_9 = None
    sub_3: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_8, getitem_5)
    mul_8: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = None
    mul_9: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_8, primals_16);  mul_8 = None
    add_10: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_9, primals_17);  mul_9 = primals_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_11: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_18, [1, 0]);  primals_18 = None
    view_24: "f32[128, 2048]" = torch.ops.aten.view.default(add_10, [128, 2048])
    mm_3: "f32[128, 2048]" = torch.ops.aten.mm.default(view_24, permute_11)
    view_25: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_3, [1, 128, 2048]);  mm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_12: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_19, [1, 0]);  primals_19 = None
    view_26: "f32[128, 2048]" = torch.ops.aten.view.default(add_10, [128, 2048])
    mm_4: "f32[128, 2048]" = torch.ops.aten.mm.default(view_26, permute_12)
    view_27: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_4, [1, 128, 2048]);  mm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    permute_13: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_20, [1, 0]);  primals_20 = None
    view_28: "f32[128, 2048]" = torch.ops.aten.view.default(add_10, [128, 2048]);  add_10 = None
    mm_5: "f32[128, 2048]" = torch.ops.aten.mm.default(view_28, permute_13)
    view_29: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_5, [1, 128, 2048]);  mm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_30: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_25, [1, 128, 16, 128]);  view_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_14: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_30, [0, 2, 1, 3]);  view_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_31: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_27, [1, 128, 16, 128]);  view_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_15: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_31, [0, 2, 1, 3]);  view_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_32: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_29, [1, 128, 16, 128]);  view_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_16: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_32, [0, 2, 1, 3]);  view_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_17: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(permute_15, [0, 1, 3, 2])
    expand_4: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_14, [1, 16, 128, 128]);  permute_14 = None
    view_33: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_4, [16, 128, 128]);  expand_4 = None
    expand_5: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_17, [1, 16, 128, 128]);  permute_17 = None
    view_34: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_5, [16, 128, 128]);  expand_5 = None
    bmm_2: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_33, view_34)
    view_35: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_2, [1, 16, 128, 128]);  bmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_5: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(primals_319, 0, 0, 9223372036854775807);  primals_319 = None
    slice_6: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(slice_5, 1, 0, 9223372036854775807);  slice_5 = None
    slice_7: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(slice_6, 2, 0, 128);  slice_6 = None
    slice_8: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_7, 3, 0, 128);  slice_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    _tensor_constant1 = self._tensor_constant1
    lift_fresh_copy_1: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant1);  _tensor_constant1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_1: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_8, view_35, lift_fresh_copy_1);  view_35 = lift_fresh_copy_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_1: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(where_1, [-1], True)
    sub_4: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(where_1, amax_1);  where_1 = amax_1 = None
    exp_1: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
    sum_2: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
    div_1: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
    alias_2: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(div_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:205, code: attn_weights = self.attn_dropout(attn_weights)
    clone_5: "f32[1, 16, 128, 128]" = torch.ops.aten.clone.default(div_1);  div_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_6: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(clone_5, [1, 16, 128, 128]);  clone_5 = None
    view_36: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_6, [16, 128, 128]);  expand_6 = None
    expand_7: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_16, [1, 16, 128, 128])
    view_37: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_7, [16, 128, 128]);  expand_7 = None
    bmm_3: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_36, view_37)
    view_38: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_3, [1, 16, 128, 128]);  bmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_18: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_38, [0, 2, 1, 3]);  view_38 = None
    clone_6: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_39: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_6, [1, 128, 2048]);  clone_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_40: "f32[128, 2048]" = torch.ops.aten.view.default(view_39, [128, 2048]);  view_39 = None
    permute_19: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_21, [1, 0]);  primals_21 = None
    addmm_3: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_22, view_40, permute_19);  primals_22 = None
    view_41: "f32[1, 128, 2048]" = torch.ops.aten.view.default(addmm_3, [1, 128, 2048]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:247, code: attn_output = self.resid_dropout(attn_output)
    clone_7: "f32[1, 128, 2048]" = torch.ops.aten.clone.default(view_41);  view_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    add_11: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(clone_7, add_8);  clone_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    var_mean_3 = torch.ops.aten.var_mean.correction(add_11, [2], correction = 0, keepdim = True)
    getitem_6: "f32[1, 128, 1]" = var_mean_3[0]
    getitem_7: "f32[1, 128, 1]" = var_mean_3[1];  var_mean_3 = None
    add_12: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-05);  getitem_6 = None
    rsqrt_3: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
    sub_5: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_11, getitem_7)
    mul_10: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = None
    mul_11: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_10, primals_23);  mul_10 = None
    add_13: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_11, primals_24);  mul_11 = primals_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_42: "f32[128, 2048]" = torch.ops.aten.view.default(add_13, [128, 2048]);  add_13 = None
    permute_20: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_25, [1, 0]);  primals_25 = None
    addmm_4: "f32[128, 8192]" = torch.ops.aten.addmm.default(primals_26, view_42, permute_20);  primals_26 = None
    view_43: "f32[1, 128, 8192]" = torch.ops.aten.view.default(addmm_4, [1, 128, 8192]);  addmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_12: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_43, 0.5)
    pow_2: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_43, 3.0)
    mul_13: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_2, 0.044715);  pow_2 = None
    add_14: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(view_43, mul_13);  mul_13 = None
    mul_14: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(add_14, 0.7978845608028654);  add_14 = None
    tanh_1: "f32[1, 128, 8192]" = torch.ops.aten.tanh.default(mul_14);  mul_14 = None
    alias_3: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(tanh_1)
    add_15: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_1, 1.0);  tanh_1 = None
    mul_15: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_12, add_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_44: "f32[128, 8192]" = torch.ops.aten.view.default(mul_15, [128, 8192]);  mul_15 = None
    permute_21: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_27, [1, 0]);  primals_27 = None
    addmm_5: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_28, view_44, permute_21);  primals_28 = None
    view_45: "f32[1, 128, 2048]" = torch.ops.aten.view.default(addmm_5, [1, 128, 2048]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:303, code: hidden_states = self.dropout(hidden_states)
    clone_8: "f32[1, 128, 2048]" = torch.ops.aten.clone.default(view_45);  view_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    add_16: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_11, clone_8);  clone_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    var_mean_4 = torch.ops.aten.var_mean.correction(add_16, [2], correction = 0, keepdim = True)
    getitem_8: "f32[1, 128, 1]" = var_mean_4[0]
    getitem_9: "f32[1, 128, 1]" = var_mean_4[1];  var_mean_4 = None
    add_17: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
    rsqrt_4: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_17);  add_17 = None
    sub_6: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_16, getitem_9)
    mul_16: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = None
    mul_17: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_16, primals_29);  mul_16 = None
    add_18: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_17, primals_30);  mul_17 = primals_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_22: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_31, [1, 0]);  primals_31 = None
    view_46: "f32[128, 2048]" = torch.ops.aten.view.default(add_18, [128, 2048])
    mm_6: "f32[128, 2048]" = torch.ops.aten.mm.default(view_46, permute_22)
    view_47: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_6, [1, 128, 2048]);  mm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_23: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_32, [1, 0]);  primals_32 = None
    view_48: "f32[128, 2048]" = torch.ops.aten.view.default(add_18, [128, 2048])
    mm_7: "f32[128, 2048]" = torch.ops.aten.mm.default(view_48, permute_23)
    view_49: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_7, [1, 128, 2048]);  mm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    permute_24: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_33, [1, 0]);  primals_33 = None
    view_50: "f32[128, 2048]" = torch.ops.aten.view.default(add_18, [128, 2048]);  add_18 = None
    mm_8: "f32[128, 2048]" = torch.ops.aten.mm.default(view_50, permute_24)
    view_51: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_8, [1, 128, 2048]);  mm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_52: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_47, [1, 128, 16, 128]);  view_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_25: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_53: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_49, [1, 128, 16, 128]);  view_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_26: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_53, [0, 2, 1, 3]);  view_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_54: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_51, [1, 128, 16, 128]);  view_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_27: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_54, [0, 2, 1, 3]);  view_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_28: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(permute_26, [0, 1, 3, 2])
    expand_8: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_25, [1, 16, 128, 128]);  permute_25 = None
    view_55: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_8, [16, 128, 128]);  expand_8 = None
    expand_9: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_28, [1, 16, 128, 128]);  permute_28 = None
    view_56: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_9, [16, 128, 128]);  expand_9 = None
    bmm_4: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_55, view_56)
    view_57: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_4, [1, 16, 128, 128]);  bmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_9: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(primals_320, 0, 0, 9223372036854775807);  primals_320 = None
    slice_10: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(slice_9, 1, 0, 9223372036854775807);  slice_9 = None
    slice_11: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(slice_10, 2, 0, 128);  slice_10 = None
    slice_12: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_11, 3, 0, 128);  slice_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    _tensor_constant2 = self._tensor_constant2
    lift_fresh_copy_2: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant2);  _tensor_constant2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_2: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_12, view_57, lift_fresh_copy_2);  view_57 = lift_fresh_copy_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_2: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(where_2, [-1], True)
    sub_7: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(where_2, amax_2);  where_2 = amax_2 = None
    exp_2: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
    sum_3: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
    div_2: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
    alias_4: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(div_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:205, code: attn_weights = self.attn_dropout(attn_weights)
    clone_9: "f32[1, 16, 128, 128]" = torch.ops.aten.clone.default(div_2);  div_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_10: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(clone_9, [1, 16, 128, 128]);  clone_9 = None
    view_58: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_10, [16, 128, 128]);  expand_10 = None
    expand_11: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_27, [1, 16, 128, 128])
    view_59: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_11, [16, 128, 128]);  expand_11 = None
    bmm_5: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_58, view_59)
    view_60: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_5, [1, 16, 128, 128]);  bmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_29: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_60, [0, 2, 1, 3]);  view_60 = None
    clone_10: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_61: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_10, [1, 128, 2048]);  clone_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_62: "f32[128, 2048]" = torch.ops.aten.view.default(view_61, [128, 2048]);  view_61 = None
    permute_30: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_34, [1, 0]);  primals_34 = None
    addmm_6: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_35, view_62, permute_30);  primals_35 = None
    view_63: "f32[1, 128, 2048]" = torch.ops.aten.view.default(addmm_6, [1, 128, 2048]);  addmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:247, code: attn_output = self.resid_dropout(attn_output)
    clone_11: "f32[1, 128, 2048]" = torch.ops.aten.clone.default(view_63);  view_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    add_19: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(clone_11, add_16);  clone_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    var_mean_5 = torch.ops.aten.var_mean.correction(add_19, [2], correction = 0, keepdim = True)
    getitem_10: "f32[1, 128, 1]" = var_mean_5[0]
    getitem_11: "f32[1, 128, 1]" = var_mean_5[1];  var_mean_5 = None
    add_20: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-05);  getitem_10 = None
    rsqrt_5: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
    sub_8: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_19, getitem_11)
    mul_18: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_5);  sub_8 = None
    mul_19: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_18, primals_36);  mul_18 = None
    add_21: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_19, primals_37);  mul_19 = primals_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_64: "f32[128, 2048]" = torch.ops.aten.view.default(add_21, [128, 2048]);  add_21 = None
    permute_31: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_38, [1, 0]);  primals_38 = None
    addmm_7: "f32[128, 8192]" = torch.ops.aten.addmm.default(primals_39, view_64, permute_31);  primals_39 = None
    view_65: "f32[1, 128, 8192]" = torch.ops.aten.view.default(addmm_7, [1, 128, 8192]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_20: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_65, 0.5)
    pow_3: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_65, 3.0)
    mul_21: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_3, 0.044715);  pow_3 = None
    add_22: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(view_65, mul_21);  mul_21 = None
    mul_22: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(add_22, 0.7978845608028654);  add_22 = None
    tanh_2: "f32[1, 128, 8192]" = torch.ops.aten.tanh.default(mul_22);  mul_22 = None
    alias_5: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(tanh_2)
    add_23: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_2, 1.0);  tanh_2 = None
    mul_23: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_20, add_23)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_66: "f32[128, 8192]" = torch.ops.aten.view.default(mul_23, [128, 8192]);  mul_23 = None
    permute_32: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_40, [1, 0]);  primals_40 = None
    addmm_8: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_41, view_66, permute_32);  primals_41 = None
    view_67: "f32[1, 128, 2048]" = torch.ops.aten.view.default(addmm_8, [1, 128, 2048]);  addmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:303, code: hidden_states = self.dropout(hidden_states)
    clone_12: "f32[1, 128, 2048]" = torch.ops.aten.clone.default(view_67);  view_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    add_24: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_19, clone_12);  clone_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    var_mean_6 = torch.ops.aten.var_mean.correction(add_24, [2], correction = 0, keepdim = True)
    getitem_12: "f32[1, 128, 1]" = var_mean_6[0]
    getitem_13: "f32[1, 128, 1]" = var_mean_6[1];  var_mean_6 = None
    add_25: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05);  getitem_12 = None
    rsqrt_6: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
    sub_9: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_24, getitem_13)
    mul_24: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_6);  sub_9 = None
    mul_25: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_24, primals_42);  mul_24 = None
    add_26: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_25, primals_43);  mul_25 = primals_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_33: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_44, [1, 0]);  primals_44 = None
    view_68: "f32[128, 2048]" = torch.ops.aten.view.default(add_26, [128, 2048])
    mm_9: "f32[128, 2048]" = torch.ops.aten.mm.default(view_68, permute_33)
    view_69: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_9, [1, 128, 2048]);  mm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_34: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_45, [1, 0]);  primals_45 = None
    view_70: "f32[128, 2048]" = torch.ops.aten.view.default(add_26, [128, 2048])
    mm_10: "f32[128, 2048]" = torch.ops.aten.mm.default(view_70, permute_34)
    view_71: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_10, [1, 128, 2048]);  mm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    permute_35: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_46, [1, 0]);  primals_46 = None
    view_72: "f32[128, 2048]" = torch.ops.aten.view.default(add_26, [128, 2048]);  add_26 = None
    mm_11: "f32[128, 2048]" = torch.ops.aten.mm.default(view_72, permute_35)
    view_73: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_11, [1, 128, 2048]);  mm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_74: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_69, [1, 128, 16, 128]);  view_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_36: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_74, [0, 2, 1, 3]);  view_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_75: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_71, [1, 128, 16, 128]);  view_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_37: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_75, [0, 2, 1, 3]);  view_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_76: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_73, [1, 128, 16, 128]);  view_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_38: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_76, [0, 2, 1, 3]);  view_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_39: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(permute_37, [0, 1, 3, 2])
    expand_12: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_36, [1, 16, 128, 128]);  permute_36 = None
    view_77: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_12, [16, 128, 128]);  expand_12 = None
    expand_13: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_39, [1, 16, 128, 128]);  permute_39 = None
    view_78: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_13, [16, 128, 128]);  expand_13 = None
    bmm_6: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_77, view_78)
    view_79: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_6, [1, 16, 128, 128]);  bmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_13: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(primals_321, 0, 0, 9223372036854775807);  primals_321 = None
    slice_14: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(slice_13, 1, 0, 9223372036854775807);  slice_13 = None
    slice_15: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(slice_14, 2, 0, 128);  slice_14 = None
    slice_16: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_15, 3, 0, 128);  slice_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    _tensor_constant3 = self._tensor_constant3
    lift_fresh_copy_3: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant3);  _tensor_constant3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_3: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_16, view_79, lift_fresh_copy_3);  view_79 = lift_fresh_copy_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_3: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(where_3, [-1], True)
    sub_10: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(where_3, amax_3);  where_3 = amax_3 = None
    exp_3: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
    sum_4: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
    div_3: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
    alias_6: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(div_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:205, code: attn_weights = self.attn_dropout(attn_weights)
    clone_13: "f32[1, 16, 128, 128]" = torch.ops.aten.clone.default(div_3);  div_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_14: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(clone_13, [1, 16, 128, 128]);  clone_13 = None
    view_80: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_14, [16, 128, 128]);  expand_14 = None
    expand_15: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_38, [1, 16, 128, 128])
    view_81: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_15, [16, 128, 128]);  expand_15 = None
    bmm_7: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_80, view_81)
    view_82: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_7, [1, 16, 128, 128]);  bmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_40: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_82, [0, 2, 1, 3]);  view_82 = None
    clone_14: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_83: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_14, [1, 128, 2048]);  clone_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_84: "f32[128, 2048]" = torch.ops.aten.view.default(view_83, [128, 2048]);  view_83 = None
    permute_41: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_47, [1, 0]);  primals_47 = None
    addmm_9: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_48, view_84, permute_41);  primals_48 = None
    view_85: "f32[1, 128, 2048]" = torch.ops.aten.view.default(addmm_9, [1, 128, 2048]);  addmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:247, code: attn_output = self.resid_dropout(attn_output)
    clone_15: "f32[1, 128, 2048]" = torch.ops.aten.clone.default(view_85);  view_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    add_27: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(clone_15, add_24);  clone_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    var_mean_7 = torch.ops.aten.var_mean.correction(add_27, [2], correction = 0, keepdim = True)
    getitem_14: "f32[1, 128, 1]" = var_mean_7[0]
    getitem_15: "f32[1, 128, 1]" = var_mean_7[1];  var_mean_7 = None
    add_28: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05);  getitem_14 = None
    rsqrt_7: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
    sub_11: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_27, getitem_15)
    mul_26: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_7);  sub_11 = None
    mul_27: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_26, primals_49);  mul_26 = None
    add_29: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_27, primals_50);  mul_27 = primals_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_86: "f32[128, 2048]" = torch.ops.aten.view.default(add_29, [128, 2048]);  add_29 = None
    permute_42: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_51, [1, 0]);  primals_51 = None
    addmm_10: "f32[128, 8192]" = torch.ops.aten.addmm.default(primals_52, view_86, permute_42);  primals_52 = None
    view_87: "f32[1, 128, 8192]" = torch.ops.aten.view.default(addmm_10, [1, 128, 8192]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_28: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_87, 0.5)
    pow_4: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_87, 3.0)
    mul_29: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_4, 0.044715);  pow_4 = None
    add_30: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(view_87, mul_29);  mul_29 = None
    mul_30: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(add_30, 0.7978845608028654);  add_30 = None
    tanh_3: "f32[1, 128, 8192]" = torch.ops.aten.tanh.default(mul_30);  mul_30 = None
    alias_7: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(tanh_3)
    add_31: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_3, 1.0);  tanh_3 = None
    mul_31: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_28, add_31)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_88: "f32[128, 8192]" = torch.ops.aten.view.default(mul_31, [128, 8192]);  mul_31 = None
    permute_43: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_53, [1, 0]);  primals_53 = None
    addmm_11: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_54, view_88, permute_43);  primals_54 = None
    view_89: "f32[1, 128, 2048]" = torch.ops.aten.view.default(addmm_11, [1, 128, 2048]);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:303, code: hidden_states = self.dropout(hidden_states)
    clone_16: "f32[1, 128, 2048]" = torch.ops.aten.clone.default(view_89);  view_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    add_32: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_27, clone_16);  clone_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    var_mean_8 = torch.ops.aten.var_mean.correction(add_32, [2], correction = 0, keepdim = True)
    getitem_16: "f32[1, 128, 1]" = var_mean_8[0]
    getitem_17: "f32[1, 128, 1]" = var_mean_8[1];  var_mean_8 = None
    add_33: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
    rsqrt_8: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_33);  add_33 = None
    sub_12: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_32, getitem_17)
    mul_32: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_8);  sub_12 = None
    mul_33: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_32, primals_55);  mul_32 = None
    add_34: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_33, primals_56);  mul_33 = primals_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_44: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_57, [1, 0]);  primals_57 = None
    view_90: "f32[128, 2048]" = torch.ops.aten.view.default(add_34, [128, 2048])
    mm_12: "f32[128, 2048]" = torch.ops.aten.mm.default(view_90, permute_44)
    view_91: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_12, [1, 128, 2048]);  mm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_45: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_58, [1, 0]);  primals_58 = None
    view_92: "f32[128, 2048]" = torch.ops.aten.view.default(add_34, [128, 2048])
    mm_13: "f32[128, 2048]" = torch.ops.aten.mm.default(view_92, permute_45)
    view_93: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_13, [1, 128, 2048]);  mm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    permute_46: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_59, [1, 0]);  primals_59 = None
    view_94: "f32[128, 2048]" = torch.ops.aten.view.default(add_34, [128, 2048]);  add_34 = None
    mm_14: "f32[128, 2048]" = torch.ops.aten.mm.default(view_94, permute_46)
    view_95: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_14, [1, 128, 2048]);  mm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_96: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_91, [1, 128, 16, 128]);  view_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_47: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_96, [0, 2, 1, 3]);  view_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_97: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_93, [1, 128, 16, 128]);  view_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_48: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_97, [0, 2, 1, 3]);  view_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_98: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_95, [1, 128, 16, 128]);  view_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_49: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_98, [0, 2, 1, 3]);  view_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_50: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(permute_48, [0, 1, 3, 2])
    expand_16: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_47, [1, 16, 128, 128]);  permute_47 = None
    view_99: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_16, [16, 128, 128]);  expand_16 = None
    expand_17: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_50, [1, 16, 128, 128]);  permute_50 = None
    view_100: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_17, [16, 128, 128]);  expand_17 = None
    bmm_8: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_99, view_100)
    view_101: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_8, [1, 16, 128, 128]);  bmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_17: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(primals_322, 0, 0, 9223372036854775807);  primals_322 = None
    slice_18: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(slice_17, 1, 0, 9223372036854775807);  slice_17 = None
    slice_19: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(slice_18, 2, 0, 128);  slice_18 = None
    slice_20: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_19, 3, 0, 128);  slice_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    _tensor_constant4 = self._tensor_constant4
    lift_fresh_copy_4: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant4);  _tensor_constant4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_4: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_20, view_101, lift_fresh_copy_4);  view_101 = lift_fresh_copy_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_4: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(where_4, [-1], True)
    sub_13: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(where_4, amax_4);  where_4 = amax_4 = None
    exp_4: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
    sum_5: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
    div_4: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
    alias_8: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(div_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:205, code: attn_weights = self.attn_dropout(attn_weights)
    clone_17: "f32[1, 16, 128, 128]" = torch.ops.aten.clone.default(div_4);  div_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_18: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(clone_17, [1, 16, 128, 128]);  clone_17 = None
    view_102: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_18, [16, 128, 128]);  expand_18 = None
    expand_19: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_49, [1, 16, 128, 128])
    view_103: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_19, [16, 128, 128]);  expand_19 = None
    bmm_9: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_102, view_103)
    view_104: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_9, [1, 16, 128, 128]);  bmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_51: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_104, [0, 2, 1, 3]);  view_104 = None
    clone_18: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_105: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_18, [1, 128, 2048]);  clone_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_106: "f32[128, 2048]" = torch.ops.aten.view.default(view_105, [128, 2048]);  view_105 = None
    permute_52: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_60, [1, 0]);  primals_60 = None
    addmm_12: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_61, view_106, permute_52);  primals_61 = None
    view_107: "f32[1, 128, 2048]" = torch.ops.aten.view.default(addmm_12, [1, 128, 2048]);  addmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:247, code: attn_output = self.resid_dropout(attn_output)
    clone_19: "f32[1, 128, 2048]" = torch.ops.aten.clone.default(view_107);  view_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    add_35: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(clone_19, add_32);  clone_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    var_mean_9 = torch.ops.aten.var_mean.correction(add_35, [2], correction = 0, keepdim = True)
    getitem_18: "f32[1, 128, 1]" = var_mean_9[0]
    getitem_19: "f32[1, 128, 1]" = var_mean_9[1];  var_mean_9 = None
    add_36: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-05);  getitem_18 = None
    rsqrt_9: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
    sub_14: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_35, getitem_19)
    mul_34: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_9);  sub_14 = None
    mul_35: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_34, primals_62);  mul_34 = None
    add_37: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_35, primals_63);  mul_35 = primals_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_108: "f32[128, 2048]" = torch.ops.aten.view.default(add_37, [128, 2048]);  add_37 = None
    permute_53: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_64, [1, 0]);  primals_64 = None
    addmm_13: "f32[128, 8192]" = torch.ops.aten.addmm.default(primals_65, view_108, permute_53);  primals_65 = None
    view_109: "f32[1, 128, 8192]" = torch.ops.aten.view.default(addmm_13, [1, 128, 8192]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_36: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_109, 0.5)
    pow_5: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_109, 3.0)
    mul_37: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_5, 0.044715);  pow_5 = None
    add_38: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(view_109, mul_37);  mul_37 = None
    mul_38: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(add_38, 0.7978845608028654);  add_38 = None
    tanh_4: "f32[1, 128, 8192]" = torch.ops.aten.tanh.default(mul_38);  mul_38 = None
    alias_9: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(tanh_4)
    add_39: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_4, 1.0);  tanh_4 = None
    mul_39: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_36, add_39)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_110: "f32[128, 8192]" = torch.ops.aten.view.default(mul_39, [128, 8192]);  mul_39 = None
    permute_54: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_66, [1, 0]);  primals_66 = None
    addmm_14: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_67, view_110, permute_54);  primals_67 = None
    view_111: "f32[1, 128, 2048]" = torch.ops.aten.view.default(addmm_14, [1, 128, 2048]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:303, code: hidden_states = self.dropout(hidden_states)
    clone_20: "f32[1, 128, 2048]" = torch.ops.aten.clone.default(view_111);  view_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    add_40: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_35, clone_20);  clone_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    var_mean_10 = torch.ops.aten.var_mean.correction(add_40, [2], correction = 0, keepdim = True)
    getitem_20: "f32[1, 128, 1]" = var_mean_10[0]
    getitem_21: "f32[1, 128, 1]" = var_mean_10[1];  var_mean_10 = None
    add_41: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05);  getitem_20 = None
    rsqrt_10: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_41);  add_41 = None
    sub_15: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_40, getitem_21)
    mul_40: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_10);  sub_15 = None
    mul_41: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_40, primals_68);  mul_40 = None
    add_42: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_41, primals_69);  mul_41 = primals_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_55: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_70, [1, 0]);  primals_70 = None
    view_112: "f32[128, 2048]" = torch.ops.aten.view.default(add_42, [128, 2048])
    mm_15: "f32[128, 2048]" = torch.ops.aten.mm.default(view_112, permute_55)
    view_113: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_15, [1, 128, 2048]);  mm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_56: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_71, [1, 0]);  primals_71 = None
    view_114: "f32[128, 2048]" = torch.ops.aten.view.default(add_42, [128, 2048])
    mm_16: "f32[128, 2048]" = torch.ops.aten.mm.default(view_114, permute_56)
    view_115: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_16, [1, 128, 2048]);  mm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    permute_57: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_72, [1, 0]);  primals_72 = None
    view_116: "f32[128, 2048]" = torch.ops.aten.view.default(add_42, [128, 2048]);  add_42 = None
    mm_17: "f32[128, 2048]" = torch.ops.aten.mm.default(view_116, permute_57)
    view_117: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_17, [1, 128, 2048]);  mm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_118: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_113, [1, 128, 16, 128]);  view_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_58: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_118, [0, 2, 1, 3]);  view_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_119: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_115, [1, 128, 16, 128]);  view_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_59: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_119, [0, 2, 1, 3]);  view_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_120: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_117, [1, 128, 16, 128]);  view_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_60: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_120, [0, 2, 1, 3]);  view_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_61: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(permute_59, [0, 1, 3, 2])
    expand_20: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_58, [1, 16, 128, 128]);  permute_58 = None
    view_121: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_20, [16, 128, 128]);  expand_20 = None
    expand_21: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_61, [1, 16, 128, 128]);  permute_61 = None
    view_122: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_21, [16, 128, 128]);  expand_21 = None
    bmm_10: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_121, view_122)
    view_123: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_10, [1, 16, 128, 128]);  bmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_21: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(primals_323, 0, 0, 9223372036854775807);  primals_323 = None
    slice_22: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(slice_21, 1, 0, 9223372036854775807);  slice_21 = None
    slice_23: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(slice_22, 2, 0, 128);  slice_22 = None
    slice_24: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_23, 3, 0, 128);  slice_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    _tensor_constant5 = self._tensor_constant5
    lift_fresh_copy_5: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant5);  _tensor_constant5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_5: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_24, view_123, lift_fresh_copy_5);  view_123 = lift_fresh_copy_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_5: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(where_5, [-1], True)
    sub_16: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(where_5, amax_5);  where_5 = amax_5 = None
    exp_5: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
    sum_6: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
    div_5: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
    alias_10: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(div_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:205, code: attn_weights = self.attn_dropout(attn_weights)
    clone_21: "f32[1, 16, 128, 128]" = torch.ops.aten.clone.default(div_5);  div_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_22: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(clone_21, [1, 16, 128, 128]);  clone_21 = None
    view_124: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_22, [16, 128, 128]);  expand_22 = None
    expand_23: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_60, [1, 16, 128, 128])
    view_125: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_23, [16, 128, 128]);  expand_23 = None
    bmm_11: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_124, view_125)
    view_126: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_11, [1, 16, 128, 128]);  bmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_62: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_126, [0, 2, 1, 3]);  view_126 = None
    clone_22: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_127: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_22, [1, 128, 2048]);  clone_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_128: "f32[128, 2048]" = torch.ops.aten.view.default(view_127, [128, 2048]);  view_127 = None
    permute_63: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_73, [1, 0]);  primals_73 = None
    addmm_15: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_74, view_128, permute_63);  primals_74 = None
    view_129: "f32[1, 128, 2048]" = torch.ops.aten.view.default(addmm_15, [1, 128, 2048]);  addmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:247, code: attn_output = self.resid_dropout(attn_output)
    clone_23: "f32[1, 128, 2048]" = torch.ops.aten.clone.default(view_129);  view_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    add_43: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(clone_23, add_40);  clone_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    var_mean_11 = torch.ops.aten.var_mean.correction(add_43, [2], correction = 0, keepdim = True)
    getitem_22: "f32[1, 128, 1]" = var_mean_11[0]
    getitem_23: "f32[1, 128, 1]" = var_mean_11[1];  var_mean_11 = None
    add_44: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
    rsqrt_11: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
    sub_17: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_43, getitem_23)
    mul_42: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_11);  sub_17 = None
    mul_43: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_42, primals_75);  mul_42 = None
    add_45: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_43, primals_76);  mul_43 = primals_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_130: "f32[128, 2048]" = torch.ops.aten.view.default(add_45, [128, 2048]);  add_45 = None
    permute_64: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_77, [1, 0]);  primals_77 = None
    addmm_16: "f32[128, 8192]" = torch.ops.aten.addmm.default(primals_78, view_130, permute_64);  primals_78 = None
    view_131: "f32[1, 128, 8192]" = torch.ops.aten.view.default(addmm_16, [1, 128, 8192]);  addmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_44: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_131, 0.5)
    pow_6: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_131, 3.0)
    mul_45: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_6, 0.044715);  pow_6 = None
    add_46: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(view_131, mul_45);  mul_45 = None
    mul_46: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(add_46, 0.7978845608028654);  add_46 = None
    tanh_5: "f32[1, 128, 8192]" = torch.ops.aten.tanh.default(mul_46);  mul_46 = None
    alias_11: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(tanh_5)
    add_47: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_5, 1.0);  tanh_5 = None
    mul_47: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_44, add_47)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_132: "f32[128, 8192]" = torch.ops.aten.view.default(mul_47, [128, 8192]);  mul_47 = None
    permute_65: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_79, [1, 0]);  primals_79 = None
    addmm_17: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_80, view_132, permute_65);  primals_80 = None
    view_133: "f32[1, 128, 2048]" = torch.ops.aten.view.default(addmm_17, [1, 128, 2048]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:303, code: hidden_states = self.dropout(hidden_states)
    clone_24: "f32[1, 128, 2048]" = torch.ops.aten.clone.default(view_133);  view_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    add_48: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_43, clone_24);  clone_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    var_mean_12 = torch.ops.aten.var_mean.correction(add_48, [2], correction = 0, keepdim = True)
    getitem_24: "f32[1, 128, 1]" = var_mean_12[0]
    getitem_25: "f32[1, 128, 1]" = var_mean_12[1];  var_mean_12 = None
    add_49: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05);  getitem_24 = None
    rsqrt_12: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
    sub_18: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_48, getitem_25)
    mul_48: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_12);  sub_18 = None
    mul_49: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_48, primals_81);  mul_48 = None
    add_50: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_49, primals_82);  mul_49 = primals_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_66: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_83, [1, 0]);  primals_83 = None
    view_134: "f32[128, 2048]" = torch.ops.aten.view.default(add_50, [128, 2048])
    mm_18: "f32[128, 2048]" = torch.ops.aten.mm.default(view_134, permute_66)
    view_135: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_18, [1, 128, 2048]);  mm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_67: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_84, [1, 0]);  primals_84 = None
    view_136: "f32[128, 2048]" = torch.ops.aten.view.default(add_50, [128, 2048])
    mm_19: "f32[128, 2048]" = torch.ops.aten.mm.default(view_136, permute_67)
    view_137: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_19, [1, 128, 2048]);  mm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    permute_68: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_85, [1, 0]);  primals_85 = None
    view_138: "f32[128, 2048]" = torch.ops.aten.view.default(add_50, [128, 2048]);  add_50 = None
    mm_20: "f32[128, 2048]" = torch.ops.aten.mm.default(view_138, permute_68)
    view_139: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_20, [1, 128, 2048]);  mm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_140: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_135, [1, 128, 16, 128]);  view_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_69: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_140, [0, 2, 1, 3]);  view_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_141: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_137, [1, 128, 16, 128]);  view_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_70: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_141, [0, 2, 1, 3]);  view_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_142: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_139, [1, 128, 16, 128]);  view_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_71: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_142, [0, 2, 1, 3]);  view_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_72: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(permute_70, [0, 1, 3, 2])
    expand_24: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_69, [1, 16, 128, 128]);  permute_69 = None
    view_143: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_24, [16, 128, 128]);  expand_24 = None
    expand_25: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_72, [1, 16, 128, 128]);  permute_72 = None
    view_144: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_25, [16, 128, 128]);  expand_25 = None
    bmm_12: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_143, view_144)
    view_145: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_12, [1, 16, 128, 128]);  bmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_25: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(primals_324, 0, 0, 9223372036854775807);  primals_324 = None
    slice_26: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(slice_25, 1, 0, 9223372036854775807);  slice_25 = None
    slice_27: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(slice_26, 2, 0, 128);  slice_26 = None
    slice_28: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_27, 3, 0, 128);  slice_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    _tensor_constant6 = self._tensor_constant6
    lift_fresh_copy_6: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant6);  _tensor_constant6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_6: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_28, view_145, lift_fresh_copy_6);  view_145 = lift_fresh_copy_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_6: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(where_6, [-1], True)
    sub_19: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(where_6, amax_6);  where_6 = amax_6 = None
    exp_6: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
    sum_7: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
    div_6: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
    alias_12: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(div_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:205, code: attn_weights = self.attn_dropout(attn_weights)
    clone_25: "f32[1, 16, 128, 128]" = torch.ops.aten.clone.default(div_6);  div_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_26: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(clone_25, [1, 16, 128, 128]);  clone_25 = None
    view_146: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_26, [16, 128, 128]);  expand_26 = None
    expand_27: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_71, [1, 16, 128, 128])
    view_147: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_27, [16, 128, 128]);  expand_27 = None
    bmm_13: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_146, view_147)
    view_148: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_13, [1, 16, 128, 128]);  bmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_73: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_148, [0, 2, 1, 3]);  view_148 = None
    clone_26: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_149: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_26, [1, 128, 2048]);  clone_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_150: "f32[128, 2048]" = torch.ops.aten.view.default(view_149, [128, 2048]);  view_149 = None
    permute_74: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_86, [1, 0]);  primals_86 = None
    addmm_18: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_87, view_150, permute_74);  primals_87 = None
    view_151: "f32[1, 128, 2048]" = torch.ops.aten.view.default(addmm_18, [1, 128, 2048]);  addmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:247, code: attn_output = self.resid_dropout(attn_output)
    clone_27: "f32[1, 128, 2048]" = torch.ops.aten.clone.default(view_151);  view_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    add_51: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(clone_27, add_48);  clone_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    var_mean_13 = torch.ops.aten.var_mean.correction(add_51, [2], correction = 0, keepdim = True)
    getitem_26: "f32[1, 128, 1]" = var_mean_13[0]
    getitem_27: "f32[1, 128, 1]" = var_mean_13[1];  var_mean_13 = None
    add_52: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-05);  getitem_26 = None
    rsqrt_13: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
    sub_20: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_51, getitem_27)
    mul_50: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_13);  sub_20 = None
    mul_51: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_50, primals_88);  mul_50 = None
    add_53: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_51, primals_89);  mul_51 = primals_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_152: "f32[128, 2048]" = torch.ops.aten.view.default(add_53, [128, 2048]);  add_53 = None
    permute_75: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_90, [1, 0]);  primals_90 = None
    addmm_19: "f32[128, 8192]" = torch.ops.aten.addmm.default(primals_91, view_152, permute_75);  primals_91 = None
    view_153: "f32[1, 128, 8192]" = torch.ops.aten.view.default(addmm_19, [1, 128, 8192]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_52: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_153, 0.5)
    pow_7: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_153, 3.0)
    mul_53: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_7, 0.044715);  pow_7 = None
    add_54: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(view_153, mul_53);  mul_53 = None
    mul_54: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(add_54, 0.7978845608028654);  add_54 = None
    tanh_6: "f32[1, 128, 8192]" = torch.ops.aten.tanh.default(mul_54);  mul_54 = None
    alias_13: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(tanh_6)
    add_55: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_6, 1.0);  tanh_6 = None
    mul_55: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_52, add_55)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_154: "f32[128, 8192]" = torch.ops.aten.view.default(mul_55, [128, 8192]);  mul_55 = None
    permute_76: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_92, [1, 0]);  primals_92 = None
    addmm_20: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_93, view_154, permute_76);  primals_93 = None
    view_155: "f32[1, 128, 2048]" = torch.ops.aten.view.default(addmm_20, [1, 128, 2048]);  addmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:303, code: hidden_states = self.dropout(hidden_states)
    clone_28: "f32[1, 128, 2048]" = torch.ops.aten.clone.default(view_155);  view_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    add_56: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_51, clone_28);  clone_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    var_mean_14 = torch.ops.aten.var_mean.correction(add_56, [2], correction = 0, keepdim = True)
    getitem_28: "f32[1, 128, 1]" = var_mean_14[0]
    getitem_29: "f32[1, 128, 1]" = var_mean_14[1];  var_mean_14 = None
    add_57: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05);  getitem_28 = None
    rsqrt_14: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_57);  add_57 = None
    sub_21: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_56, getitem_29)
    mul_56: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_14);  sub_21 = None
    mul_57: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_56, primals_94);  mul_56 = None
    add_58: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_57, primals_95);  mul_57 = primals_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_77: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_96, [1, 0]);  primals_96 = None
    view_156: "f32[128, 2048]" = torch.ops.aten.view.default(add_58, [128, 2048])
    mm_21: "f32[128, 2048]" = torch.ops.aten.mm.default(view_156, permute_77)
    view_157: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_21, [1, 128, 2048]);  mm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_78: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_97, [1, 0]);  primals_97 = None
    view_158: "f32[128, 2048]" = torch.ops.aten.view.default(add_58, [128, 2048])
    mm_22: "f32[128, 2048]" = torch.ops.aten.mm.default(view_158, permute_78)
    view_159: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_22, [1, 128, 2048]);  mm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    permute_79: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_98, [1, 0]);  primals_98 = None
    view_160: "f32[128, 2048]" = torch.ops.aten.view.default(add_58, [128, 2048]);  add_58 = None
    mm_23: "f32[128, 2048]" = torch.ops.aten.mm.default(view_160, permute_79)
    view_161: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_23, [1, 128, 2048]);  mm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_162: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_157, [1, 128, 16, 128]);  view_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_80: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_162, [0, 2, 1, 3]);  view_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_163: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_159, [1, 128, 16, 128]);  view_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_81: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_163, [0, 2, 1, 3]);  view_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_164: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_161, [1, 128, 16, 128]);  view_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_82: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_164, [0, 2, 1, 3]);  view_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_83: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(permute_81, [0, 1, 3, 2])
    expand_28: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_80, [1, 16, 128, 128]);  permute_80 = None
    view_165: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_28, [16, 128, 128]);  expand_28 = None
    expand_29: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_83, [1, 16, 128, 128]);  permute_83 = None
    view_166: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_29, [16, 128, 128]);  expand_29 = None
    bmm_14: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_165, view_166)
    view_167: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_14, [1, 16, 128, 128]);  bmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_29: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(primals_325, 0, 0, 9223372036854775807);  primals_325 = None
    slice_30: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(slice_29, 1, 0, 9223372036854775807);  slice_29 = None
    slice_31: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(slice_30, 2, 0, 128);  slice_30 = None
    slice_32: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_31, 3, 0, 128);  slice_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    _tensor_constant7 = self._tensor_constant7
    lift_fresh_copy_7: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant7);  _tensor_constant7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_7: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_32, view_167, lift_fresh_copy_7);  view_167 = lift_fresh_copy_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_7: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(where_7, [-1], True)
    sub_22: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(where_7, amax_7);  where_7 = amax_7 = None
    exp_7: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
    sum_8: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
    div_7: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
    alias_14: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(div_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:205, code: attn_weights = self.attn_dropout(attn_weights)
    clone_29: "f32[1, 16, 128, 128]" = torch.ops.aten.clone.default(div_7);  div_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_30: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(clone_29, [1, 16, 128, 128]);  clone_29 = None
    view_168: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_30, [16, 128, 128]);  expand_30 = None
    expand_31: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_82, [1, 16, 128, 128])
    view_169: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_31, [16, 128, 128]);  expand_31 = None
    bmm_15: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_168, view_169)
    view_170: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_15, [1, 16, 128, 128]);  bmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_84: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_170, [0, 2, 1, 3]);  view_170 = None
    clone_30: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_171: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_30, [1, 128, 2048]);  clone_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_172: "f32[128, 2048]" = torch.ops.aten.view.default(view_171, [128, 2048]);  view_171 = None
    permute_85: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_99, [1, 0]);  primals_99 = None
    addmm_21: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_100, view_172, permute_85);  primals_100 = None
    view_173: "f32[1, 128, 2048]" = torch.ops.aten.view.default(addmm_21, [1, 128, 2048]);  addmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:247, code: attn_output = self.resid_dropout(attn_output)
    clone_31: "f32[1, 128, 2048]" = torch.ops.aten.clone.default(view_173);  view_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    add_59: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(clone_31, add_56);  clone_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    var_mean_15 = torch.ops.aten.var_mean.correction(add_59, [2], correction = 0, keepdim = True)
    getitem_30: "f32[1, 128, 1]" = var_mean_15[0]
    getitem_31: "f32[1, 128, 1]" = var_mean_15[1];  var_mean_15 = None
    add_60: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-05);  getitem_30 = None
    rsqrt_15: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
    sub_23: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_59, getitem_31)
    mul_58: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_15);  sub_23 = None
    mul_59: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_58, primals_101);  mul_58 = None
    add_61: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_59, primals_102);  mul_59 = primals_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_174: "f32[128, 2048]" = torch.ops.aten.view.default(add_61, [128, 2048]);  add_61 = None
    permute_86: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_103, [1, 0]);  primals_103 = None
    addmm_22: "f32[128, 8192]" = torch.ops.aten.addmm.default(primals_104, view_174, permute_86);  primals_104 = None
    view_175: "f32[1, 128, 8192]" = torch.ops.aten.view.default(addmm_22, [1, 128, 8192]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_60: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_175, 0.5)
    pow_8: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_175, 3.0)
    mul_61: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_8, 0.044715);  pow_8 = None
    add_62: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(view_175, mul_61);  mul_61 = None
    mul_62: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(add_62, 0.7978845608028654);  add_62 = None
    tanh_7: "f32[1, 128, 8192]" = torch.ops.aten.tanh.default(mul_62);  mul_62 = None
    alias_15: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(tanh_7)
    add_63: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_7, 1.0);  tanh_7 = None
    mul_63: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_60, add_63)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_176: "f32[128, 8192]" = torch.ops.aten.view.default(mul_63, [128, 8192]);  mul_63 = None
    permute_87: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_105, [1, 0]);  primals_105 = None
    addmm_23: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_106, view_176, permute_87);  primals_106 = None
    view_177: "f32[1, 128, 2048]" = torch.ops.aten.view.default(addmm_23, [1, 128, 2048]);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:303, code: hidden_states = self.dropout(hidden_states)
    clone_32: "f32[1, 128, 2048]" = torch.ops.aten.clone.default(view_177);  view_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    add_64: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_59, clone_32);  clone_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    var_mean_16 = torch.ops.aten.var_mean.correction(add_64, [2], correction = 0, keepdim = True)
    getitem_32: "f32[1, 128, 1]" = var_mean_16[0]
    getitem_33: "f32[1, 128, 1]" = var_mean_16[1];  var_mean_16 = None
    add_65: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-05);  getitem_32 = None
    rsqrt_16: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_65);  add_65 = None
    sub_24: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_64, getitem_33)
    mul_64: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_16);  sub_24 = None
    mul_65: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_64, primals_107);  mul_64 = None
    add_66: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_65, primals_108);  mul_65 = primals_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_88: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_109, [1, 0]);  primals_109 = None
    view_178: "f32[128, 2048]" = torch.ops.aten.view.default(add_66, [128, 2048])
    mm_24: "f32[128, 2048]" = torch.ops.aten.mm.default(view_178, permute_88)
    view_179: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_24, [1, 128, 2048]);  mm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_89: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_110, [1, 0]);  primals_110 = None
    view_180: "f32[128, 2048]" = torch.ops.aten.view.default(add_66, [128, 2048])
    mm_25: "f32[128, 2048]" = torch.ops.aten.mm.default(view_180, permute_89)
    view_181: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_25, [1, 128, 2048]);  mm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    permute_90: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_111, [1, 0]);  primals_111 = None
    view_182: "f32[128, 2048]" = torch.ops.aten.view.default(add_66, [128, 2048]);  add_66 = None
    mm_26: "f32[128, 2048]" = torch.ops.aten.mm.default(view_182, permute_90)
    view_183: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_26, [1, 128, 2048]);  mm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_184: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_179, [1, 128, 16, 128]);  view_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_91: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_184, [0, 2, 1, 3]);  view_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_185: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_181, [1, 128, 16, 128]);  view_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_92: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_185, [0, 2, 1, 3]);  view_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_186: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_183, [1, 128, 16, 128]);  view_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_93: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_186, [0, 2, 1, 3]);  view_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_94: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(permute_92, [0, 1, 3, 2])
    expand_32: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_91, [1, 16, 128, 128]);  permute_91 = None
    view_187: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_32, [16, 128, 128]);  expand_32 = None
    expand_33: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_94, [1, 16, 128, 128]);  permute_94 = None
    view_188: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_33, [16, 128, 128]);  expand_33 = None
    bmm_16: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_187, view_188)
    view_189: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_16, [1, 16, 128, 128]);  bmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_33: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(primals_326, 0, 0, 9223372036854775807);  primals_326 = None
    slice_34: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(slice_33, 1, 0, 9223372036854775807);  slice_33 = None
    slice_35: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(slice_34, 2, 0, 128);  slice_34 = None
    slice_36: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_35, 3, 0, 128);  slice_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    _tensor_constant8 = self._tensor_constant8
    lift_fresh_copy_8: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant8);  _tensor_constant8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_8: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_36, view_189, lift_fresh_copy_8);  view_189 = lift_fresh_copy_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_8: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(where_8, [-1], True)
    sub_25: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(where_8, amax_8);  where_8 = amax_8 = None
    exp_8: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_25);  sub_25 = None
    sum_9: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
    div_8: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
    alias_16: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(div_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:205, code: attn_weights = self.attn_dropout(attn_weights)
    clone_33: "f32[1, 16, 128, 128]" = torch.ops.aten.clone.default(div_8);  div_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_34: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(clone_33, [1, 16, 128, 128]);  clone_33 = None
    view_190: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_34, [16, 128, 128]);  expand_34 = None
    expand_35: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_93, [1, 16, 128, 128])
    view_191: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_35, [16, 128, 128]);  expand_35 = None
    bmm_17: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_190, view_191)
    view_192: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_17, [1, 16, 128, 128]);  bmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_95: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_192, [0, 2, 1, 3]);  view_192 = None
    clone_34: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_193: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_34, [1, 128, 2048]);  clone_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_194: "f32[128, 2048]" = torch.ops.aten.view.default(view_193, [128, 2048]);  view_193 = None
    permute_96: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_112, [1, 0]);  primals_112 = None
    addmm_24: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_113, view_194, permute_96);  primals_113 = None
    view_195: "f32[1, 128, 2048]" = torch.ops.aten.view.default(addmm_24, [1, 128, 2048]);  addmm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:247, code: attn_output = self.resid_dropout(attn_output)
    clone_35: "f32[1, 128, 2048]" = torch.ops.aten.clone.default(view_195);  view_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    add_67: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(clone_35, add_64);  clone_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    var_mean_17 = torch.ops.aten.var_mean.correction(add_67, [2], correction = 0, keepdim = True)
    getitem_34: "f32[1, 128, 1]" = var_mean_17[0]
    getitem_35: "f32[1, 128, 1]" = var_mean_17[1];  var_mean_17 = None
    add_68: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-05);  getitem_34 = None
    rsqrt_17: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_68);  add_68 = None
    sub_26: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_67, getitem_35)
    mul_66: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_17);  sub_26 = None
    mul_67: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_66, primals_114);  mul_66 = None
    add_69: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_67, primals_115);  mul_67 = primals_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_196: "f32[128, 2048]" = torch.ops.aten.view.default(add_69, [128, 2048]);  add_69 = None
    permute_97: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_116, [1, 0]);  primals_116 = None
    addmm_25: "f32[128, 8192]" = torch.ops.aten.addmm.default(primals_117, view_196, permute_97);  primals_117 = None
    view_197: "f32[1, 128, 8192]" = torch.ops.aten.view.default(addmm_25, [1, 128, 8192]);  addmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_68: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_197, 0.5)
    pow_9: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_197, 3.0)
    mul_69: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_9, 0.044715);  pow_9 = None
    add_70: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(view_197, mul_69);  mul_69 = None
    mul_70: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(add_70, 0.7978845608028654);  add_70 = None
    tanh_8: "f32[1, 128, 8192]" = torch.ops.aten.tanh.default(mul_70);  mul_70 = None
    alias_17: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(tanh_8)
    add_71: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_8, 1.0);  tanh_8 = None
    mul_71: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_68, add_71)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_198: "f32[128, 8192]" = torch.ops.aten.view.default(mul_71, [128, 8192]);  mul_71 = None
    permute_98: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_118, [1, 0]);  primals_118 = None
    addmm_26: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_119, view_198, permute_98);  primals_119 = None
    view_199: "f32[1, 128, 2048]" = torch.ops.aten.view.default(addmm_26, [1, 128, 2048]);  addmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:303, code: hidden_states = self.dropout(hidden_states)
    clone_36: "f32[1, 128, 2048]" = torch.ops.aten.clone.default(view_199);  view_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    add_72: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_67, clone_36);  clone_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    var_mean_18 = torch.ops.aten.var_mean.correction(add_72, [2], correction = 0, keepdim = True)
    getitem_36: "f32[1, 128, 1]" = var_mean_18[0]
    getitem_37: "f32[1, 128, 1]" = var_mean_18[1];  var_mean_18 = None
    add_73: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-05);  getitem_36 = None
    rsqrt_18: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_73);  add_73 = None
    sub_27: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_72, getitem_37)
    mul_72: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_18);  sub_27 = None
    mul_73: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_72, primals_120);  mul_72 = None
    add_74: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_73, primals_121);  mul_73 = primals_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_99: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_122, [1, 0]);  primals_122 = None
    view_200: "f32[128, 2048]" = torch.ops.aten.view.default(add_74, [128, 2048])
    mm_27: "f32[128, 2048]" = torch.ops.aten.mm.default(view_200, permute_99)
    view_201: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_27, [1, 128, 2048]);  mm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_100: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_123, [1, 0]);  primals_123 = None
    view_202: "f32[128, 2048]" = torch.ops.aten.view.default(add_74, [128, 2048])
    mm_28: "f32[128, 2048]" = torch.ops.aten.mm.default(view_202, permute_100)
    view_203: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_28, [1, 128, 2048]);  mm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    permute_101: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_124, [1, 0]);  primals_124 = None
    view_204: "f32[128, 2048]" = torch.ops.aten.view.default(add_74, [128, 2048]);  add_74 = None
    mm_29: "f32[128, 2048]" = torch.ops.aten.mm.default(view_204, permute_101)
    view_205: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_29, [1, 128, 2048]);  mm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_206: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_201, [1, 128, 16, 128]);  view_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_102: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_206, [0, 2, 1, 3]);  view_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_207: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_203, [1, 128, 16, 128]);  view_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_103: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_207, [0, 2, 1, 3]);  view_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_208: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_205, [1, 128, 16, 128]);  view_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_104: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_208, [0, 2, 1, 3]);  view_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_105: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(permute_103, [0, 1, 3, 2])
    expand_36: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_102, [1, 16, 128, 128]);  permute_102 = None
    view_209: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_36, [16, 128, 128]);  expand_36 = None
    expand_37: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_105, [1, 16, 128, 128]);  permute_105 = None
    view_210: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_37, [16, 128, 128]);  expand_37 = None
    bmm_18: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_209, view_210)
    view_211: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_18, [1, 16, 128, 128]);  bmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_37: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(primals_327, 0, 0, 9223372036854775807);  primals_327 = None
    slice_38: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(slice_37, 1, 0, 9223372036854775807);  slice_37 = None
    slice_39: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(slice_38, 2, 0, 128);  slice_38 = None
    slice_40: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_39, 3, 0, 128);  slice_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    _tensor_constant9 = self._tensor_constant9
    lift_fresh_copy_9: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant9);  _tensor_constant9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_9: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_40, view_211, lift_fresh_copy_9);  view_211 = lift_fresh_copy_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_9: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(where_9, [-1], True)
    sub_28: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(where_9, amax_9);  where_9 = amax_9 = None
    exp_9: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_28);  sub_28 = None
    sum_10: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
    div_9: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
    alias_18: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(div_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:205, code: attn_weights = self.attn_dropout(attn_weights)
    clone_37: "f32[1, 16, 128, 128]" = torch.ops.aten.clone.default(div_9);  div_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_38: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(clone_37, [1, 16, 128, 128]);  clone_37 = None
    view_212: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_38, [16, 128, 128]);  expand_38 = None
    expand_39: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_104, [1, 16, 128, 128])
    view_213: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_39, [16, 128, 128]);  expand_39 = None
    bmm_19: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_212, view_213)
    view_214: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_19, [1, 16, 128, 128]);  bmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_106: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_214, [0, 2, 1, 3]);  view_214 = None
    clone_38: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_215: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_38, [1, 128, 2048]);  clone_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_216: "f32[128, 2048]" = torch.ops.aten.view.default(view_215, [128, 2048]);  view_215 = None
    permute_107: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_125, [1, 0]);  primals_125 = None
    addmm_27: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_126, view_216, permute_107);  primals_126 = None
    view_217: "f32[1, 128, 2048]" = torch.ops.aten.view.default(addmm_27, [1, 128, 2048]);  addmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:247, code: attn_output = self.resid_dropout(attn_output)
    clone_39: "f32[1, 128, 2048]" = torch.ops.aten.clone.default(view_217);  view_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    add_75: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(clone_39, add_72);  clone_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    var_mean_19 = torch.ops.aten.var_mean.correction(add_75, [2], correction = 0, keepdim = True)
    getitem_38: "f32[1, 128, 1]" = var_mean_19[0]
    getitem_39: "f32[1, 128, 1]" = var_mean_19[1];  var_mean_19 = None
    add_76: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-05);  getitem_38 = None
    rsqrt_19: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_76);  add_76 = None
    sub_29: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_75, getitem_39)
    mul_74: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_19);  sub_29 = None
    mul_75: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_74, primals_127);  mul_74 = None
    add_77: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_75, primals_128);  mul_75 = primals_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_218: "f32[128, 2048]" = torch.ops.aten.view.default(add_77, [128, 2048]);  add_77 = None
    permute_108: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_129, [1, 0]);  primals_129 = None
    addmm_28: "f32[128, 8192]" = torch.ops.aten.addmm.default(primals_130, view_218, permute_108);  primals_130 = None
    view_219: "f32[1, 128, 8192]" = torch.ops.aten.view.default(addmm_28, [1, 128, 8192]);  addmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_76: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_219, 0.5)
    pow_10: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_219, 3.0)
    mul_77: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_10, 0.044715);  pow_10 = None
    add_78: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(view_219, mul_77);  mul_77 = None
    mul_78: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(add_78, 0.7978845608028654);  add_78 = None
    tanh_9: "f32[1, 128, 8192]" = torch.ops.aten.tanh.default(mul_78);  mul_78 = None
    alias_19: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(tanh_9)
    add_79: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_9, 1.0);  tanh_9 = None
    mul_79: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_76, add_79)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_220: "f32[128, 8192]" = torch.ops.aten.view.default(mul_79, [128, 8192]);  mul_79 = None
    permute_109: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_131, [1, 0]);  primals_131 = None
    addmm_29: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_132, view_220, permute_109);  primals_132 = None
    view_221: "f32[1, 128, 2048]" = torch.ops.aten.view.default(addmm_29, [1, 128, 2048]);  addmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:303, code: hidden_states = self.dropout(hidden_states)
    clone_40: "f32[1, 128, 2048]" = torch.ops.aten.clone.default(view_221);  view_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    add_80: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_75, clone_40);  clone_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    var_mean_20 = torch.ops.aten.var_mean.correction(add_80, [2], correction = 0, keepdim = True)
    getitem_40: "f32[1, 128, 1]" = var_mean_20[0]
    getitem_41: "f32[1, 128, 1]" = var_mean_20[1];  var_mean_20 = None
    add_81: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-05);  getitem_40 = None
    rsqrt_20: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_81);  add_81 = None
    sub_30: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_80, getitem_41)
    mul_80: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_20);  sub_30 = None
    mul_81: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_80, primals_133);  mul_80 = None
    add_82: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_81, primals_134);  mul_81 = primals_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_110: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_135, [1, 0]);  primals_135 = None
    view_222: "f32[128, 2048]" = torch.ops.aten.view.default(add_82, [128, 2048])
    mm_30: "f32[128, 2048]" = torch.ops.aten.mm.default(view_222, permute_110)
    view_223: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_30, [1, 128, 2048]);  mm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_111: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_136, [1, 0]);  primals_136 = None
    view_224: "f32[128, 2048]" = torch.ops.aten.view.default(add_82, [128, 2048])
    mm_31: "f32[128, 2048]" = torch.ops.aten.mm.default(view_224, permute_111)
    view_225: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_31, [1, 128, 2048]);  mm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    permute_112: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_137, [1, 0]);  primals_137 = None
    view_226: "f32[128, 2048]" = torch.ops.aten.view.default(add_82, [128, 2048]);  add_82 = None
    mm_32: "f32[128, 2048]" = torch.ops.aten.mm.default(view_226, permute_112)
    view_227: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_32, [1, 128, 2048]);  mm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_228: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_223, [1, 128, 16, 128]);  view_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_113: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_228, [0, 2, 1, 3]);  view_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_229: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_225, [1, 128, 16, 128]);  view_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_114: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_229, [0, 2, 1, 3]);  view_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_230: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_227, [1, 128, 16, 128]);  view_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_115: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_230, [0, 2, 1, 3]);  view_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_116: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(permute_114, [0, 1, 3, 2])
    expand_40: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_113, [1, 16, 128, 128]);  permute_113 = None
    view_231: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_40, [16, 128, 128]);  expand_40 = None
    expand_41: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_116, [1, 16, 128, 128]);  permute_116 = None
    view_232: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_41, [16, 128, 128]);  expand_41 = None
    bmm_20: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_231, view_232)
    view_233: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_20, [1, 16, 128, 128]);  bmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_41: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(primals_328, 0, 0, 9223372036854775807);  primals_328 = None
    slice_42: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(slice_41, 1, 0, 9223372036854775807);  slice_41 = None
    slice_43: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(slice_42, 2, 0, 128);  slice_42 = None
    slice_44: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_43, 3, 0, 128);  slice_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    _tensor_constant10 = self._tensor_constant10
    lift_fresh_copy_10: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant10);  _tensor_constant10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_10: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_44, view_233, lift_fresh_copy_10);  view_233 = lift_fresh_copy_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_10: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(where_10, [-1], True)
    sub_31: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(where_10, amax_10);  where_10 = amax_10 = None
    exp_10: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_31);  sub_31 = None
    sum_11: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
    div_10: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
    alias_20: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(div_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:205, code: attn_weights = self.attn_dropout(attn_weights)
    clone_41: "f32[1, 16, 128, 128]" = torch.ops.aten.clone.default(div_10);  div_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_42: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(clone_41, [1, 16, 128, 128]);  clone_41 = None
    view_234: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_42, [16, 128, 128]);  expand_42 = None
    expand_43: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_115, [1, 16, 128, 128])
    view_235: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_43, [16, 128, 128]);  expand_43 = None
    bmm_21: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_234, view_235)
    view_236: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_21, [1, 16, 128, 128]);  bmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_117: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_236, [0, 2, 1, 3]);  view_236 = None
    clone_42: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_237: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_42, [1, 128, 2048]);  clone_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_238: "f32[128, 2048]" = torch.ops.aten.view.default(view_237, [128, 2048]);  view_237 = None
    permute_118: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_138, [1, 0]);  primals_138 = None
    addmm_30: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_139, view_238, permute_118);  primals_139 = None
    view_239: "f32[1, 128, 2048]" = torch.ops.aten.view.default(addmm_30, [1, 128, 2048]);  addmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:247, code: attn_output = self.resid_dropout(attn_output)
    clone_43: "f32[1, 128, 2048]" = torch.ops.aten.clone.default(view_239);  view_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    add_83: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(clone_43, add_80);  clone_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    var_mean_21 = torch.ops.aten.var_mean.correction(add_83, [2], correction = 0, keepdim = True)
    getitem_42: "f32[1, 128, 1]" = var_mean_21[0]
    getitem_43: "f32[1, 128, 1]" = var_mean_21[1];  var_mean_21 = None
    add_84: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-05);  getitem_42 = None
    rsqrt_21: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
    sub_32: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_83, getitem_43)
    mul_82: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_21);  sub_32 = None
    mul_83: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_82, primals_140);  mul_82 = None
    add_85: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_83, primals_141);  mul_83 = primals_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_240: "f32[128, 2048]" = torch.ops.aten.view.default(add_85, [128, 2048]);  add_85 = None
    permute_119: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_142, [1, 0]);  primals_142 = None
    addmm_31: "f32[128, 8192]" = torch.ops.aten.addmm.default(primals_143, view_240, permute_119);  primals_143 = None
    view_241: "f32[1, 128, 8192]" = torch.ops.aten.view.default(addmm_31, [1, 128, 8192]);  addmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_84: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_241, 0.5)
    pow_11: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_241, 3.0)
    mul_85: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_11, 0.044715);  pow_11 = None
    add_86: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(view_241, mul_85);  mul_85 = None
    mul_86: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(add_86, 0.7978845608028654);  add_86 = None
    tanh_10: "f32[1, 128, 8192]" = torch.ops.aten.tanh.default(mul_86);  mul_86 = None
    alias_21: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(tanh_10)
    add_87: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_10, 1.0);  tanh_10 = None
    mul_87: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_84, add_87)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_242: "f32[128, 8192]" = torch.ops.aten.view.default(mul_87, [128, 8192]);  mul_87 = None
    permute_120: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_144, [1, 0]);  primals_144 = None
    addmm_32: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_145, view_242, permute_120);  primals_145 = None
    view_243: "f32[1, 128, 2048]" = torch.ops.aten.view.default(addmm_32, [1, 128, 2048]);  addmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:303, code: hidden_states = self.dropout(hidden_states)
    clone_44: "f32[1, 128, 2048]" = torch.ops.aten.clone.default(view_243);  view_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    add_88: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_83, clone_44);  clone_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    var_mean_22 = torch.ops.aten.var_mean.correction(add_88, [2], correction = 0, keepdim = True)
    getitem_44: "f32[1, 128, 1]" = var_mean_22[0]
    getitem_45: "f32[1, 128, 1]" = var_mean_22[1];  var_mean_22 = None
    add_89: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-05);  getitem_44 = None
    rsqrt_22: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_89);  add_89 = None
    sub_33: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_88, getitem_45)
    mul_88: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_22);  sub_33 = None
    mul_89: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_88, primals_146);  mul_88 = None
    add_90: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_89, primals_147);  mul_89 = primals_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_121: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_148, [1, 0]);  primals_148 = None
    view_244: "f32[128, 2048]" = torch.ops.aten.view.default(add_90, [128, 2048])
    mm_33: "f32[128, 2048]" = torch.ops.aten.mm.default(view_244, permute_121)
    view_245: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_33, [1, 128, 2048]);  mm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_122: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_149, [1, 0]);  primals_149 = None
    view_246: "f32[128, 2048]" = torch.ops.aten.view.default(add_90, [128, 2048])
    mm_34: "f32[128, 2048]" = torch.ops.aten.mm.default(view_246, permute_122)
    view_247: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_34, [1, 128, 2048]);  mm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    permute_123: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_150, [1, 0]);  primals_150 = None
    view_248: "f32[128, 2048]" = torch.ops.aten.view.default(add_90, [128, 2048]);  add_90 = None
    mm_35: "f32[128, 2048]" = torch.ops.aten.mm.default(view_248, permute_123)
    view_249: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_35, [1, 128, 2048]);  mm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_250: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_245, [1, 128, 16, 128]);  view_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_124: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_250, [0, 2, 1, 3]);  view_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_251: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_247, [1, 128, 16, 128]);  view_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_125: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_251, [0, 2, 1, 3]);  view_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_252: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_249, [1, 128, 16, 128]);  view_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_126: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_252, [0, 2, 1, 3]);  view_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_127: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(permute_125, [0, 1, 3, 2])
    expand_44: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_124, [1, 16, 128, 128]);  permute_124 = None
    view_253: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_44, [16, 128, 128]);  expand_44 = None
    expand_45: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_127, [1, 16, 128, 128]);  permute_127 = None
    view_254: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_45, [16, 128, 128]);  expand_45 = None
    bmm_22: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_253, view_254)
    view_255: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_22, [1, 16, 128, 128]);  bmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_45: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(primals_329, 0, 0, 9223372036854775807);  primals_329 = None
    slice_46: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(slice_45, 1, 0, 9223372036854775807);  slice_45 = None
    slice_47: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(slice_46, 2, 0, 128);  slice_46 = None
    slice_48: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_47, 3, 0, 128);  slice_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    _tensor_constant11 = self._tensor_constant11
    lift_fresh_copy_11: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant11);  _tensor_constant11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_11: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_48, view_255, lift_fresh_copy_11);  view_255 = lift_fresh_copy_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_11: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(where_11, [-1], True)
    sub_34: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(where_11, amax_11);  where_11 = amax_11 = None
    exp_11: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_34);  sub_34 = None
    sum_12: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
    div_11: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
    alias_22: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(div_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:205, code: attn_weights = self.attn_dropout(attn_weights)
    clone_45: "f32[1, 16, 128, 128]" = torch.ops.aten.clone.default(div_11);  div_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_46: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(clone_45, [1, 16, 128, 128]);  clone_45 = None
    view_256: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_46, [16, 128, 128]);  expand_46 = None
    expand_47: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_126, [1, 16, 128, 128])
    view_257: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_47, [16, 128, 128]);  expand_47 = None
    bmm_23: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_256, view_257)
    view_258: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_23, [1, 16, 128, 128]);  bmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_128: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_258, [0, 2, 1, 3]);  view_258 = None
    clone_46: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_259: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_46, [1, 128, 2048]);  clone_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_260: "f32[128, 2048]" = torch.ops.aten.view.default(view_259, [128, 2048]);  view_259 = None
    permute_129: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_151, [1, 0]);  primals_151 = None
    addmm_33: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_152, view_260, permute_129);  primals_152 = None
    view_261: "f32[1, 128, 2048]" = torch.ops.aten.view.default(addmm_33, [1, 128, 2048]);  addmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:247, code: attn_output = self.resid_dropout(attn_output)
    clone_47: "f32[1, 128, 2048]" = torch.ops.aten.clone.default(view_261);  view_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    add_91: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(clone_47, add_88);  clone_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    var_mean_23 = torch.ops.aten.var_mean.correction(add_91, [2], correction = 0, keepdim = True)
    getitem_46: "f32[1, 128, 1]" = var_mean_23[0]
    getitem_47: "f32[1, 128, 1]" = var_mean_23[1];  var_mean_23 = None
    add_92: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-05);  getitem_46 = None
    rsqrt_23: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_92);  add_92 = None
    sub_35: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_91, getitem_47)
    mul_90: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_23);  sub_35 = None
    mul_91: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_90, primals_153);  mul_90 = None
    add_93: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_91, primals_154);  mul_91 = primals_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_262: "f32[128, 2048]" = torch.ops.aten.view.default(add_93, [128, 2048]);  add_93 = None
    permute_130: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_155, [1, 0]);  primals_155 = None
    addmm_34: "f32[128, 8192]" = torch.ops.aten.addmm.default(primals_156, view_262, permute_130);  primals_156 = None
    view_263: "f32[1, 128, 8192]" = torch.ops.aten.view.default(addmm_34, [1, 128, 8192]);  addmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_92: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_263, 0.5)
    pow_12: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_263, 3.0)
    mul_93: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_12, 0.044715);  pow_12 = None
    add_94: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(view_263, mul_93);  mul_93 = None
    mul_94: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(add_94, 0.7978845608028654);  add_94 = None
    tanh_11: "f32[1, 128, 8192]" = torch.ops.aten.tanh.default(mul_94);  mul_94 = None
    alias_23: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(tanh_11)
    add_95: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_11, 1.0);  tanh_11 = None
    mul_95: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_92, add_95)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_264: "f32[128, 8192]" = torch.ops.aten.view.default(mul_95, [128, 8192]);  mul_95 = None
    permute_131: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_157, [1, 0]);  primals_157 = None
    addmm_35: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_158, view_264, permute_131);  primals_158 = None
    view_265: "f32[1, 128, 2048]" = torch.ops.aten.view.default(addmm_35, [1, 128, 2048]);  addmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:303, code: hidden_states = self.dropout(hidden_states)
    clone_48: "f32[1, 128, 2048]" = torch.ops.aten.clone.default(view_265);  view_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    add_96: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_91, clone_48);  clone_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    var_mean_24 = torch.ops.aten.var_mean.correction(add_96, [2], correction = 0, keepdim = True)
    getitem_48: "f32[1, 128, 1]" = var_mean_24[0]
    getitem_49: "f32[1, 128, 1]" = var_mean_24[1];  var_mean_24 = None
    add_97: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-05);  getitem_48 = None
    rsqrt_24: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_97);  add_97 = None
    sub_36: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_96, getitem_49)
    mul_96: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_24);  sub_36 = None
    mul_97: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_96, primals_159);  mul_96 = None
    add_98: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_97, primals_160);  mul_97 = primals_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_132: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_161, [1, 0]);  primals_161 = None
    view_266: "f32[128, 2048]" = torch.ops.aten.view.default(add_98, [128, 2048])
    mm_36: "f32[128, 2048]" = torch.ops.aten.mm.default(view_266, permute_132)
    view_267: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_36, [1, 128, 2048]);  mm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_133: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_162, [1, 0]);  primals_162 = None
    view_268: "f32[128, 2048]" = torch.ops.aten.view.default(add_98, [128, 2048])
    mm_37: "f32[128, 2048]" = torch.ops.aten.mm.default(view_268, permute_133)
    view_269: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_37, [1, 128, 2048]);  mm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    permute_134: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_163, [1, 0]);  primals_163 = None
    view_270: "f32[128, 2048]" = torch.ops.aten.view.default(add_98, [128, 2048]);  add_98 = None
    mm_38: "f32[128, 2048]" = torch.ops.aten.mm.default(view_270, permute_134)
    view_271: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_38, [1, 128, 2048]);  mm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_272: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_267, [1, 128, 16, 128]);  view_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_135: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_272, [0, 2, 1, 3]);  view_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_273: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_269, [1, 128, 16, 128]);  view_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_136: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_273, [0, 2, 1, 3]);  view_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_274: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_271, [1, 128, 16, 128]);  view_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_137: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_274, [0, 2, 1, 3]);  view_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_138: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(permute_136, [0, 1, 3, 2])
    expand_48: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_135, [1, 16, 128, 128]);  permute_135 = None
    view_275: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_48, [16, 128, 128]);  expand_48 = None
    expand_49: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_138, [1, 16, 128, 128]);  permute_138 = None
    view_276: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_49, [16, 128, 128]);  expand_49 = None
    bmm_24: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_275, view_276)
    view_277: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_24, [1, 16, 128, 128]);  bmm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_49: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(primals_330, 0, 0, 9223372036854775807);  primals_330 = None
    slice_50: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(slice_49, 1, 0, 9223372036854775807);  slice_49 = None
    slice_51: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(slice_50, 2, 0, 128);  slice_50 = None
    slice_52: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_51, 3, 0, 128);  slice_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    _tensor_constant12 = self._tensor_constant12
    lift_fresh_copy_12: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant12);  _tensor_constant12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_12: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_52, view_277, lift_fresh_copy_12);  view_277 = lift_fresh_copy_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_12: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(where_12, [-1], True)
    sub_37: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(where_12, amax_12);  where_12 = amax_12 = None
    exp_12: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_37);  sub_37 = None
    sum_13: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
    div_12: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = sum_13 = None
    alias_24: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(div_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:205, code: attn_weights = self.attn_dropout(attn_weights)
    clone_49: "f32[1, 16, 128, 128]" = torch.ops.aten.clone.default(div_12);  div_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_50: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(clone_49, [1, 16, 128, 128]);  clone_49 = None
    view_278: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_50, [16, 128, 128]);  expand_50 = None
    expand_51: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_137, [1, 16, 128, 128])
    view_279: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_51, [16, 128, 128]);  expand_51 = None
    bmm_25: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_278, view_279)
    view_280: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_25, [1, 16, 128, 128]);  bmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_139: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_280, [0, 2, 1, 3]);  view_280 = None
    clone_50: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_139, memory_format = torch.contiguous_format);  permute_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_281: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_50, [1, 128, 2048]);  clone_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_282: "f32[128, 2048]" = torch.ops.aten.view.default(view_281, [128, 2048]);  view_281 = None
    permute_140: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_164, [1, 0]);  primals_164 = None
    addmm_36: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_165, view_282, permute_140);  primals_165 = None
    view_283: "f32[1, 128, 2048]" = torch.ops.aten.view.default(addmm_36, [1, 128, 2048]);  addmm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:247, code: attn_output = self.resid_dropout(attn_output)
    clone_51: "f32[1, 128, 2048]" = torch.ops.aten.clone.default(view_283);  view_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    add_99: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(clone_51, add_96);  clone_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    var_mean_25 = torch.ops.aten.var_mean.correction(add_99, [2], correction = 0, keepdim = True)
    getitem_50: "f32[1, 128, 1]" = var_mean_25[0]
    getitem_51: "f32[1, 128, 1]" = var_mean_25[1];  var_mean_25 = None
    add_100: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-05);  getitem_50 = None
    rsqrt_25: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_100);  add_100 = None
    sub_38: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_99, getitem_51)
    mul_98: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_25);  sub_38 = None
    mul_99: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_98, primals_166);  mul_98 = None
    add_101: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_99, primals_167);  mul_99 = primals_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_284: "f32[128, 2048]" = torch.ops.aten.view.default(add_101, [128, 2048]);  add_101 = None
    permute_141: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_168, [1, 0]);  primals_168 = None
    addmm_37: "f32[128, 8192]" = torch.ops.aten.addmm.default(primals_169, view_284, permute_141);  primals_169 = None
    view_285: "f32[1, 128, 8192]" = torch.ops.aten.view.default(addmm_37, [1, 128, 8192]);  addmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_100: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_285, 0.5)
    pow_13: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_285, 3.0)
    mul_101: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_13, 0.044715);  pow_13 = None
    add_102: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(view_285, mul_101);  mul_101 = None
    mul_102: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(add_102, 0.7978845608028654);  add_102 = None
    tanh_12: "f32[1, 128, 8192]" = torch.ops.aten.tanh.default(mul_102);  mul_102 = None
    alias_25: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(tanh_12)
    add_103: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_12, 1.0);  tanh_12 = None
    mul_103: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_100, add_103)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_286: "f32[128, 8192]" = torch.ops.aten.view.default(mul_103, [128, 8192]);  mul_103 = None
    permute_142: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_170, [1, 0]);  primals_170 = None
    addmm_38: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_171, view_286, permute_142);  primals_171 = None
    view_287: "f32[1, 128, 2048]" = torch.ops.aten.view.default(addmm_38, [1, 128, 2048]);  addmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:303, code: hidden_states = self.dropout(hidden_states)
    clone_52: "f32[1, 128, 2048]" = torch.ops.aten.clone.default(view_287);  view_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    add_104: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_99, clone_52);  clone_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    var_mean_26 = torch.ops.aten.var_mean.correction(add_104, [2], correction = 0, keepdim = True)
    getitem_52: "f32[1, 128, 1]" = var_mean_26[0]
    getitem_53: "f32[1, 128, 1]" = var_mean_26[1];  var_mean_26 = None
    add_105: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-05);  getitem_52 = None
    rsqrt_26: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_105);  add_105 = None
    sub_39: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_104, getitem_53)
    mul_104: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_26);  sub_39 = None
    mul_105: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_104, primals_172);  mul_104 = None
    add_106: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_105, primals_173);  mul_105 = primals_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_143: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_174, [1, 0]);  primals_174 = None
    view_288: "f32[128, 2048]" = torch.ops.aten.view.default(add_106, [128, 2048])
    mm_39: "f32[128, 2048]" = torch.ops.aten.mm.default(view_288, permute_143)
    view_289: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_39, [1, 128, 2048]);  mm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_144: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_175, [1, 0]);  primals_175 = None
    view_290: "f32[128, 2048]" = torch.ops.aten.view.default(add_106, [128, 2048])
    mm_40: "f32[128, 2048]" = torch.ops.aten.mm.default(view_290, permute_144)
    view_291: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_40, [1, 128, 2048]);  mm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    permute_145: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_176, [1, 0]);  primals_176 = None
    view_292: "f32[128, 2048]" = torch.ops.aten.view.default(add_106, [128, 2048]);  add_106 = None
    mm_41: "f32[128, 2048]" = torch.ops.aten.mm.default(view_292, permute_145)
    view_293: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_41, [1, 128, 2048]);  mm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_294: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_289, [1, 128, 16, 128]);  view_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_146: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_294, [0, 2, 1, 3]);  view_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_295: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_291, [1, 128, 16, 128]);  view_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_147: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_295, [0, 2, 1, 3]);  view_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_296: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_293, [1, 128, 16, 128]);  view_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_148: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_296, [0, 2, 1, 3]);  view_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_149: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(permute_147, [0, 1, 3, 2])
    expand_52: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_146, [1, 16, 128, 128]);  permute_146 = None
    view_297: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_52, [16, 128, 128]);  expand_52 = None
    expand_53: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_149, [1, 16, 128, 128]);  permute_149 = None
    view_298: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_53, [16, 128, 128]);  expand_53 = None
    bmm_26: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_297, view_298)
    view_299: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_26, [1, 16, 128, 128]);  bmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_53: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(primals_331, 0, 0, 9223372036854775807);  primals_331 = None
    slice_54: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(slice_53, 1, 0, 9223372036854775807);  slice_53 = None
    slice_55: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(slice_54, 2, 0, 128);  slice_54 = None
    slice_56: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_55, 3, 0, 128);  slice_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    _tensor_constant13 = self._tensor_constant13
    lift_fresh_copy_13: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant13);  _tensor_constant13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_13: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_56, view_299, lift_fresh_copy_13);  view_299 = lift_fresh_copy_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_13: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(where_13, [-1], True)
    sub_40: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(where_13, amax_13);  where_13 = amax_13 = None
    exp_13: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_40);  sub_40 = None
    sum_14: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_13, [-1], True)
    div_13: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_13, sum_14);  exp_13 = sum_14 = None
    alias_26: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(div_13)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:205, code: attn_weights = self.attn_dropout(attn_weights)
    clone_53: "f32[1, 16, 128, 128]" = torch.ops.aten.clone.default(div_13);  div_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_54: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(clone_53, [1, 16, 128, 128]);  clone_53 = None
    view_300: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_54, [16, 128, 128]);  expand_54 = None
    expand_55: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_148, [1, 16, 128, 128])
    view_301: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_55, [16, 128, 128]);  expand_55 = None
    bmm_27: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_300, view_301)
    view_302: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_27, [1, 16, 128, 128]);  bmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_150: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_302, [0, 2, 1, 3]);  view_302 = None
    clone_54: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_150, memory_format = torch.contiguous_format);  permute_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_303: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_54, [1, 128, 2048]);  clone_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_304: "f32[128, 2048]" = torch.ops.aten.view.default(view_303, [128, 2048]);  view_303 = None
    permute_151: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_177, [1, 0]);  primals_177 = None
    addmm_39: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_178, view_304, permute_151);  primals_178 = None
    view_305: "f32[1, 128, 2048]" = torch.ops.aten.view.default(addmm_39, [1, 128, 2048]);  addmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:247, code: attn_output = self.resid_dropout(attn_output)
    clone_55: "f32[1, 128, 2048]" = torch.ops.aten.clone.default(view_305);  view_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    add_107: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(clone_55, add_104);  clone_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    var_mean_27 = torch.ops.aten.var_mean.correction(add_107, [2], correction = 0, keepdim = True)
    getitem_54: "f32[1, 128, 1]" = var_mean_27[0]
    getitem_55: "f32[1, 128, 1]" = var_mean_27[1];  var_mean_27 = None
    add_108: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-05);  getitem_54 = None
    rsqrt_27: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_108);  add_108 = None
    sub_41: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_107, getitem_55)
    mul_106: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_27);  sub_41 = None
    mul_107: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_106, primals_179);  mul_106 = None
    add_109: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_107, primals_180);  mul_107 = primals_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_306: "f32[128, 2048]" = torch.ops.aten.view.default(add_109, [128, 2048]);  add_109 = None
    permute_152: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_181, [1, 0]);  primals_181 = None
    addmm_40: "f32[128, 8192]" = torch.ops.aten.addmm.default(primals_182, view_306, permute_152);  primals_182 = None
    view_307: "f32[1, 128, 8192]" = torch.ops.aten.view.default(addmm_40, [1, 128, 8192]);  addmm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_108: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_307, 0.5)
    pow_14: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_307, 3.0)
    mul_109: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_14, 0.044715);  pow_14 = None
    add_110: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(view_307, mul_109);  mul_109 = None
    mul_110: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(add_110, 0.7978845608028654);  add_110 = None
    tanh_13: "f32[1, 128, 8192]" = torch.ops.aten.tanh.default(mul_110);  mul_110 = None
    alias_27: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(tanh_13)
    add_111: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_13, 1.0);  tanh_13 = None
    mul_111: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_108, add_111)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_308: "f32[128, 8192]" = torch.ops.aten.view.default(mul_111, [128, 8192]);  mul_111 = None
    permute_153: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_183, [1, 0]);  primals_183 = None
    addmm_41: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_184, view_308, permute_153);  primals_184 = None
    view_309: "f32[1, 128, 2048]" = torch.ops.aten.view.default(addmm_41, [1, 128, 2048]);  addmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:303, code: hidden_states = self.dropout(hidden_states)
    clone_56: "f32[1, 128, 2048]" = torch.ops.aten.clone.default(view_309);  view_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    add_112: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_107, clone_56);  clone_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    var_mean_28 = torch.ops.aten.var_mean.correction(add_112, [2], correction = 0, keepdim = True)
    getitem_56: "f32[1, 128, 1]" = var_mean_28[0]
    getitem_57: "f32[1, 128, 1]" = var_mean_28[1];  var_mean_28 = None
    add_113: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-05);  getitem_56 = None
    rsqrt_28: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_113);  add_113 = None
    sub_42: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_112, getitem_57)
    mul_112: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_28);  sub_42 = None
    mul_113: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_112, primals_185);  mul_112 = None
    add_114: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_113, primals_186);  mul_113 = primals_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_154: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_187, [1, 0]);  primals_187 = None
    view_310: "f32[128, 2048]" = torch.ops.aten.view.default(add_114, [128, 2048])
    mm_42: "f32[128, 2048]" = torch.ops.aten.mm.default(view_310, permute_154)
    view_311: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_42, [1, 128, 2048]);  mm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_155: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_188, [1, 0]);  primals_188 = None
    view_312: "f32[128, 2048]" = torch.ops.aten.view.default(add_114, [128, 2048])
    mm_43: "f32[128, 2048]" = torch.ops.aten.mm.default(view_312, permute_155)
    view_313: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_43, [1, 128, 2048]);  mm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    permute_156: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_189, [1, 0]);  primals_189 = None
    view_314: "f32[128, 2048]" = torch.ops.aten.view.default(add_114, [128, 2048]);  add_114 = None
    mm_44: "f32[128, 2048]" = torch.ops.aten.mm.default(view_314, permute_156)
    view_315: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_44, [1, 128, 2048]);  mm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_316: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_311, [1, 128, 16, 128]);  view_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_157: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_316, [0, 2, 1, 3]);  view_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_317: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_313, [1, 128, 16, 128]);  view_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_158: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_317, [0, 2, 1, 3]);  view_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_318: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_315, [1, 128, 16, 128]);  view_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_159: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_318, [0, 2, 1, 3]);  view_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_160: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(permute_158, [0, 1, 3, 2])
    expand_56: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_157, [1, 16, 128, 128]);  permute_157 = None
    view_319: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_56, [16, 128, 128]);  expand_56 = None
    expand_57: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_160, [1, 16, 128, 128]);  permute_160 = None
    view_320: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_57, [16, 128, 128]);  expand_57 = None
    bmm_28: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_319, view_320)
    view_321: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_28, [1, 16, 128, 128]);  bmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_57: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(primals_332, 0, 0, 9223372036854775807);  primals_332 = None
    slice_58: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(slice_57, 1, 0, 9223372036854775807);  slice_57 = None
    slice_59: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(slice_58, 2, 0, 128);  slice_58 = None
    slice_60: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_59, 3, 0, 128);  slice_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    _tensor_constant14 = self._tensor_constant14
    lift_fresh_copy_14: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant14);  _tensor_constant14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_14: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_60, view_321, lift_fresh_copy_14);  view_321 = lift_fresh_copy_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_14: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(where_14, [-1], True)
    sub_43: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(where_14, amax_14);  where_14 = amax_14 = None
    exp_14: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_43);  sub_43 = None
    sum_15: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
    div_14: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = sum_15 = None
    alias_28: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(div_14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:205, code: attn_weights = self.attn_dropout(attn_weights)
    clone_57: "f32[1, 16, 128, 128]" = torch.ops.aten.clone.default(div_14);  div_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_58: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(clone_57, [1, 16, 128, 128]);  clone_57 = None
    view_322: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_58, [16, 128, 128]);  expand_58 = None
    expand_59: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_159, [1, 16, 128, 128])
    view_323: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_59, [16, 128, 128]);  expand_59 = None
    bmm_29: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_322, view_323)
    view_324: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_29, [1, 16, 128, 128]);  bmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_161: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_324, [0, 2, 1, 3]);  view_324 = None
    clone_58: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_161, memory_format = torch.contiguous_format);  permute_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_325: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_58, [1, 128, 2048]);  clone_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_326: "f32[128, 2048]" = torch.ops.aten.view.default(view_325, [128, 2048]);  view_325 = None
    permute_162: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_190, [1, 0]);  primals_190 = None
    addmm_42: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_191, view_326, permute_162);  primals_191 = None
    view_327: "f32[1, 128, 2048]" = torch.ops.aten.view.default(addmm_42, [1, 128, 2048]);  addmm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:247, code: attn_output = self.resid_dropout(attn_output)
    clone_59: "f32[1, 128, 2048]" = torch.ops.aten.clone.default(view_327);  view_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    add_115: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(clone_59, add_112);  clone_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    var_mean_29 = torch.ops.aten.var_mean.correction(add_115, [2], correction = 0, keepdim = True)
    getitem_58: "f32[1, 128, 1]" = var_mean_29[0]
    getitem_59: "f32[1, 128, 1]" = var_mean_29[1];  var_mean_29 = None
    add_116: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_58, 1e-05);  getitem_58 = None
    rsqrt_29: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_116);  add_116 = None
    sub_44: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_115, getitem_59)
    mul_114: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_29);  sub_44 = None
    mul_115: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_114, primals_192);  mul_114 = None
    add_117: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_115, primals_193);  mul_115 = primals_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_328: "f32[128, 2048]" = torch.ops.aten.view.default(add_117, [128, 2048]);  add_117 = None
    permute_163: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_194, [1, 0]);  primals_194 = None
    addmm_43: "f32[128, 8192]" = torch.ops.aten.addmm.default(primals_195, view_328, permute_163);  primals_195 = None
    view_329: "f32[1, 128, 8192]" = torch.ops.aten.view.default(addmm_43, [1, 128, 8192]);  addmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_116: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_329, 0.5)
    pow_15: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_329, 3.0)
    mul_117: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_15, 0.044715);  pow_15 = None
    add_118: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(view_329, mul_117);  mul_117 = None
    mul_118: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(add_118, 0.7978845608028654);  add_118 = None
    tanh_14: "f32[1, 128, 8192]" = torch.ops.aten.tanh.default(mul_118);  mul_118 = None
    alias_29: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(tanh_14)
    add_119: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_14, 1.0);  tanh_14 = None
    mul_119: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_116, add_119)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_330: "f32[128, 8192]" = torch.ops.aten.view.default(mul_119, [128, 8192]);  mul_119 = None
    permute_164: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_196, [1, 0]);  primals_196 = None
    addmm_44: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_197, view_330, permute_164);  primals_197 = None
    view_331: "f32[1, 128, 2048]" = torch.ops.aten.view.default(addmm_44, [1, 128, 2048]);  addmm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:303, code: hidden_states = self.dropout(hidden_states)
    clone_60: "f32[1, 128, 2048]" = torch.ops.aten.clone.default(view_331);  view_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    add_120: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_115, clone_60);  clone_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    var_mean_30 = torch.ops.aten.var_mean.correction(add_120, [2], correction = 0, keepdim = True)
    getitem_60: "f32[1, 128, 1]" = var_mean_30[0]
    getitem_61: "f32[1, 128, 1]" = var_mean_30[1];  var_mean_30 = None
    add_121: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_60, 1e-05);  getitem_60 = None
    rsqrt_30: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_121);  add_121 = None
    sub_45: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_120, getitem_61)
    mul_120: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_30);  sub_45 = None
    mul_121: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_120, primals_198);  mul_120 = None
    add_122: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_121, primals_199);  mul_121 = primals_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_165: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_200, [1, 0]);  primals_200 = None
    view_332: "f32[128, 2048]" = torch.ops.aten.view.default(add_122, [128, 2048])
    mm_45: "f32[128, 2048]" = torch.ops.aten.mm.default(view_332, permute_165)
    view_333: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_45, [1, 128, 2048]);  mm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_166: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_201, [1, 0]);  primals_201 = None
    view_334: "f32[128, 2048]" = torch.ops.aten.view.default(add_122, [128, 2048])
    mm_46: "f32[128, 2048]" = torch.ops.aten.mm.default(view_334, permute_166)
    view_335: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_46, [1, 128, 2048]);  mm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    permute_167: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_202, [1, 0]);  primals_202 = None
    view_336: "f32[128, 2048]" = torch.ops.aten.view.default(add_122, [128, 2048]);  add_122 = None
    mm_47: "f32[128, 2048]" = torch.ops.aten.mm.default(view_336, permute_167)
    view_337: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_47, [1, 128, 2048]);  mm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_338: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_333, [1, 128, 16, 128]);  view_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_168: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_338, [0, 2, 1, 3]);  view_338 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_339: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_335, [1, 128, 16, 128]);  view_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_169: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_339, [0, 2, 1, 3]);  view_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_340: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_337, [1, 128, 16, 128]);  view_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_170: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_340, [0, 2, 1, 3]);  view_340 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_171: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(permute_169, [0, 1, 3, 2])
    expand_60: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_168, [1, 16, 128, 128]);  permute_168 = None
    view_341: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_60, [16, 128, 128]);  expand_60 = None
    expand_61: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_171, [1, 16, 128, 128]);  permute_171 = None
    view_342: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_61, [16, 128, 128]);  expand_61 = None
    bmm_30: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_341, view_342)
    view_343: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_30, [1, 16, 128, 128]);  bmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_61: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(primals_333, 0, 0, 9223372036854775807);  primals_333 = None
    slice_62: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(slice_61, 1, 0, 9223372036854775807);  slice_61 = None
    slice_63: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(slice_62, 2, 0, 128);  slice_62 = None
    slice_64: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_63, 3, 0, 128);  slice_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    _tensor_constant15 = self._tensor_constant15
    lift_fresh_copy_15: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant15);  _tensor_constant15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_15: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_64, view_343, lift_fresh_copy_15);  view_343 = lift_fresh_copy_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_15: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(where_15, [-1], True)
    sub_46: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(where_15, amax_15);  where_15 = amax_15 = None
    exp_15: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_46);  sub_46 = None
    sum_16: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_15, [-1], True)
    div_15: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_15, sum_16);  exp_15 = sum_16 = None
    alias_30: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(div_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:205, code: attn_weights = self.attn_dropout(attn_weights)
    clone_61: "f32[1, 16, 128, 128]" = torch.ops.aten.clone.default(div_15);  div_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_62: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(clone_61, [1, 16, 128, 128]);  clone_61 = None
    view_344: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_62, [16, 128, 128]);  expand_62 = None
    expand_63: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_170, [1, 16, 128, 128])
    view_345: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_63, [16, 128, 128]);  expand_63 = None
    bmm_31: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_344, view_345)
    view_346: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_31, [1, 16, 128, 128]);  bmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_172: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_346, [0, 2, 1, 3]);  view_346 = None
    clone_62: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_172, memory_format = torch.contiguous_format);  permute_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_347: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_62, [1, 128, 2048]);  clone_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_348: "f32[128, 2048]" = torch.ops.aten.view.default(view_347, [128, 2048]);  view_347 = None
    permute_173: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_203, [1, 0]);  primals_203 = None
    addmm_45: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_204, view_348, permute_173);  primals_204 = None
    view_349: "f32[1, 128, 2048]" = torch.ops.aten.view.default(addmm_45, [1, 128, 2048]);  addmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:247, code: attn_output = self.resid_dropout(attn_output)
    clone_63: "f32[1, 128, 2048]" = torch.ops.aten.clone.default(view_349);  view_349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    add_123: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(clone_63, add_120);  clone_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    var_mean_31 = torch.ops.aten.var_mean.correction(add_123, [2], correction = 0, keepdim = True)
    getitem_62: "f32[1, 128, 1]" = var_mean_31[0]
    getitem_63: "f32[1, 128, 1]" = var_mean_31[1];  var_mean_31 = None
    add_124: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-05);  getitem_62 = None
    rsqrt_31: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_124);  add_124 = None
    sub_47: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_123, getitem_63)
    mul_122: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_31);  sub_47 = None
    mul_123: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_122, primals_205);  mul_122 = None
    add_125: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_123, primals_206);  mul_123 = primals_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_350: "f32[128, 2048]" = torch.ops.aten.view.default(add_125, [128, 2048]);  add_125 = None
    permute_174: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_207, [1, 0]);  primals_207 = None
    addmm_46: "f32[128, 8192]" = torch.ops.aten.addmm.default(primals_208, view_350, permute_174);  primals_208 = None
    view_351: "f32[1, 128, 8192]" = torch.ops.aten.view.default(addmm_46, [1, 128, 8192]);  addmm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_124: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_351, 0.5)
    pow_16: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_351, 3.0)
    mul_125: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_16, 0.044715);  pow_16 = None
    add_126: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(view_351, mul_125);  mul_125 = None
    mul_126: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(add_126, 0.7978845608028654);  add_126 = None
    tanh_15: "f32[1, 128, 8192]" = torch.ops.aten.tanh.default(mul_126);  mul_126 = None
    alias_31: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(tanh_15)
    add_127: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_15, 1.0);  tanh_15 = None
    mul_127: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_124, add_127)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_352: "f32[128, 8192]" = torch.ops.aten.view.default(mul_127, [128, 8192]);  mul_127 = None
    permute_175: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_209, [1, 0]);  primals_209 = None
    addmm_47: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_210, view_352, permute_175);  primals_210 = None
    view_353: "f32[1, 128, 2048]" = torch.ops.aten.view.default(addmm_47, [1, 128, 2048]);  addmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:303, code: hidden_states = self.dropout(hidden_states)
    clone_64: "f32[1, 128, 2048]" = torch.ops.aten.clone.default(view_353);  view_353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    add_128: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_123, clone_64);  clone_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    var_mean_32 = torch.ops.aten.var_mean.correction(add_128, [2], correction = 0, keepdim = True)
    getitem_64: "f32[1, 128, 1]" = var_mean_32[0]
    getitem_65: "f32[1, 128, 1]" = var_mean_32[1];  var_mean_32 = None
    add_129: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-05);  getitem_64 = None
    rsqrt_32: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_129);  add_129 = None
    sub_48: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_128, getitem_65)
    mul_128: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_32);  sub_48 = None
    mul_129: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_128, primals_211);  mul_128 = None
    add_130: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_129, primals_212);  mul_129 = primals_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_176: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_213, [1, 0]);  primals_213 = None
    view_354: "f32[128, 2048]" = torch.ops.aten.view.default(add_130, [128, 2048])
    mm_48: "f32[128, 2048]" = torch.ops.aten.mm.default(view_354, permute_176)
    view_355: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_48, [1, 128, 2048]);  mm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_177: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_214, [1, 0]);  primals_214 = None
    view_356: "f32[128, 2048]" = torch.ops.aten.view.default(add_130, [128, 2048])
    mm_49: "f32[128, 2048]" = torch.ops.aten.mm.default(view_356, permute_177)
    view_357: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_49, [1, 128, 2048]);  mm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    permute_178: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_215, [1, 0]);  primals_215 = None
    view_358: "f32[128, 2048]" = torch.ops.aten.view.default(add_130, [128, 2048]);  add_130 = None
    mm_50: "f32[128, 2048]" = torch.ops.aten.mm.default(view_358, permute_178)
    view_359: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_50, [1, 128, 2048]);  mm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_360: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_355, [1, 128, 16, 128]);  view_355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_179: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_360, [0, 2, 1, 3]);  view_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_361: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_357, [1, 128, 16, 128]);  view_357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_180: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_361, [0, 2, 1, 3]);  view_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_362: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_359, [1, 128, 16, 128]);  view_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_181: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_362, [0, 2, 1, 3]);  view_362 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_182: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(permute_180, [0, 1, 3, 2])
    expand_64: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_179, [1, 16, 128, 128]);  permute_179 = None
    view_363: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_64, [16, 128, 128]);  expand_64 = None
    expand_65: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_182, [1, 16, 128, 128]);  permute_182 = None
    view_364: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_65, [16, 128, 128]);  expand_65 = None
    bmm_32: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_363, view_364)
    view_365: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_32, [1, 16, 128, 128]);  bmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_65: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(primals_334, 0, 0, 9223372036854775807);  primals_334 = None
    slice_66: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(slice_65, 1, 0, 9223372036854775807);  slice_65 = None
    slice_67: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(slice_66, 2, 0, 128);  slice_66 = None
    slice_68: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_67, 3, 0, 128);  slice_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    _tensor_constant16 = self._tensor_constant16
    lift_fresh_copy_16: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant16);  _tensor_constant16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_16: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_68, view_365, lift_fresh_copy_16);  view_365 = lift_fresh_copy_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_16: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(where_16, [-1], True)
    sub_49: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(where_16, amax_16);  where_16 = amax_16 = None
    exp_16: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_49);  sub_49 = None
    sum_17: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_16, [-1], True)
    div_16: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = sum_17 = None
    alias_32: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(div_16)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:205, code: attn_weights = self.attn_dropout(attn_weights)
    clone_65: "f32[1, 16, 128, 128]" = torch.ops.aten.clone.default(div_16);  div_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_66: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(clone_65, [1, 16, 128, 128]);  clone_65 = None
    view_366: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_66, [16, 128, 128]);  expand_66 = None
    expand_67: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_181, [1, 16, 128, 128])
    view_367: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_67, [16, 128, 128]);  expand_67 = None
    bmm_33: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_366, view_367)
    view_368: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_33, [1, 16, 128, 128]);  bmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_183: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_368, [0, 2, 1, 3]);  view_368 = None
    clone_66: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_183, memory_format = torch.contiguous_format);  permute_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_369: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_66, [1, 128, 2048]);  clone_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_370: "f32[128, 2048]" = torch.ops.aten.view.default(view_369, [128, 2048]);  view_369 = None
    permute_184: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_216, [1, 0]);  primals_216 = None
    addmm_48: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_217, view_370, permute_184);  primals_217 = None
    view_371: "f32[1, 128, 2048]" = torch.ops.aten.view.default(addmm_48, [1, 128, 2048]);  addmm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:247, code: attn_output = self.resid_dropout(attn_output)
    clone_67: "f32[1, 128, 2048]" = torch.ops.aten.clone.default(view_371);  view_371 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    add_131: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(clone_67, add_128);  clone_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    var_mean_33 = torch.ops.aten.var_mean.correction(add_131, [2], correction = 0, keepdim = True)
    getitem_66: "f32[1, 128, 1]" = var_mean_33[0]
    getitem_67: "f32[1, 128, 1]" = var_mean_33[1];  var_mean_33 = None
    add_132: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-05);  getitem_66 = None
    rsqrt_33: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_132);  add_132 = None
    sub_50: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_131, getitem_67)
    mul_130: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_33);  sub_50 = None
    mul_131: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_130, primals_218);  mul_130 = None
    add_133: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_131, primals_219);  mul_131 = primals_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_372: "f32[128, 2048]" = torch.ops.aten.view.default(add_133, [128, 2048]);  add_133 = None
    permute_185: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_220, [1, 0]);  primals_220 = None
    addmm_49: "f32[128, 8192]" = torch.ops.aten.addmm.default(primals_221, view_372, permute_185);  primals_221 = None
    view_373: "f32[1, 128, 8192]" = torch.ops.aten.view.default(addmm_49, [1, 128, 8192]);  addmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_132: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_373, 0.5)
    pow_17: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_373, 3.0)
    mul_133: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_17, 0.044715);  pow_17 = None
    add_134: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(view_373, mul_133);  mul_133 = None
    mul_134: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(add_134, 0.7978845608028654);  add_134 = None
    tanh_16: "f32[1, 128, 8192]" = torch.ops.aten.tanh.default(mul_134);  mul_134 = None
    alias_33: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(tanh_16)
    add_135: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_16, 1.0);  tanh_16 = None
    mul_135: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_132, add_135)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_374: "f32[128, 8192]" = torch.ops.aten.view.default(mul_135, [128, 8192]);  mul_135 = None
    permute_186: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_222, [1, 0]);  primals_222 = None
    addmm_50: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_223, view_374, permute_186);  primals_223 = None
    view_375: "f32[1, 128, 2048]" = torch.ops.aten.view.default(addmm_50, [1, 128, 2048]);  addmm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:303, code: hidden_states = self.dropout(hidden_states)
    clone_68: "f32[1, 128, 2048]" = torch.ops.aten.clone.default(view_375);  view_375 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    add_136: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_131, clone_68);  clone_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    var_mean_34 = torch.ops.aten.var_mean.correction(add_136, [2], correction = 0, keepdim = True)
    getitem_68: "f32[1, 128, 1]" = var_mean_34[0]
    getitem_69: "f32[1, 128, 1]" = var_mean_34[1];  var_mean_34 = None
    add_137: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-05);  getitem_68 = None
    rsqrt_34: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_137);  add_137 = None
    sub_51: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_136, getitem_69)
    mul_136: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_34);  sub_51 = None
    mul_137: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_136, primals_224);  mul_136 = None
    add_138: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_137, primals_225);  mul_137 = primals_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_187: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_226, [1, 0]);  primals_226 = None
    view_376: "f32[128, 2048]" = torch.ops.aten.view.default(add_138, [128, 2048])
    mm_51: "f32[128, 2048]" = torch.ops.aten.mm.default(view_376, permute_187)
    view_377: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_51, [1, 128, 2048]);  mm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_188: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_227, [1, 0]);  primals_227 = None
    view_378: "f32[128, 2048]" = torch.ops.aten.view.default(add_138, [128, 2048])
    mm_52: "f32[128, 2048]" = torch.ops.aten.mm.default(view_378, permute_188)
    view_379: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_52, [1, 128, 2048]);  mm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    permute_189: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_228, [1, 0]);  primals_228 = None
    view_380: "f32[128, 2048]" = torch.ops.aten.view.default(add_138, [128, 2048]);  add_138 = None
    mm_53: "f32[128, 2048]" = torch.ops.aten.mm.default(view_380, permute_189)
    view_381: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_53, [1, 128, 2048]);  mm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_382: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_377, [1, 128, 16, 128]);  view_377 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_190: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_382, [0, 2, 1, 3]);  view_382 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_383: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_379, [1, 128, 16, 128]);  view_379 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_191: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_383, [0, 2, 1, 3]);  view_383 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_384: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_381, [1, 128, 16, 128]);  view_381 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_192: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_384, [0, 2, 1, 3]);  view_384 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_193: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(permute_191, [0, 1, 3, 2])
    expand_68: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_190, [1, 16, 128, 128]);  permute_190 = None
    view_385: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_68, [16, 128, 128]);  expand_68 = None
    expand_69: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_193, [1, 16, 128, 128]);  permute_193 = None
    view_386: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_69, [16, 128, 128]);  expand_69 = None
    bmm_34: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_385, view_386)
    view_387: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_34, [1, 16, 128, 128]);  bmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_69: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(primals_335, 0, 0, 9223372036854775807);  primals_335 = None
    slice_70: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(slice_69, 1, 0, 9223372036854775807);  slice_69 = None
    slice_71: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(slice_70, 2, 0, 128);  slice_70 = None
    slice_72: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_71, 3, 0, 128);  slice_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    _tensor_constant17 = self._tensor_constant17
    lift_fresh_copy_17: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant17);  _tensor_constant17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_17: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_72, view_387, lift_fresh_copy_17);  view_387 = lift_fresh_copy_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_17: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(where_17, [-1], True)
    sub_52: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(where_17, amax_17);  where_17 = amax_17 = None
    exp_17: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_52);  sub_52 = None
    sum_18: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_17, [-1], True)
    div_17: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_17, sum_18);  exp_17 = sum_18 = None
    alias_34: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(div_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:205, code: attn_weights = self.attn_dropout(attn_weights)
    clone_69: "f32[1, 16, 128, 128]" = torch.ops.aten.clone.default(div_17);  div_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_70: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(clone_69, [1, 16, 128, 128]);  clone_69 = None
    view_388: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_70, [16, 128, 128]);  expand_70 = None
    expand_71: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_192, [1, 16, 128, 128])
    view_389: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_71, [16, 128, 128]);  expand_71 = None
    bmm_35: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_388, view_389)
    view_390: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_35, [1, 16, 128, 128]);  bmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_194: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_390, [0, 2, 1, 3]);  view_390 = None
    clone_70: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_194, memory_format = torch.contiguous_format);  permute_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_391: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_70, [1, 128, 2048]);  clone_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_392: "f32[128, 2048]" = torch.ops.aten.view.default(view_391, [128, 2048]);  view_391 = None
    permute_195: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_229, [1, 0]);  primals_229 = None
    addmm_51: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_230, view_392, permute_195);  primals_230 = None
    view_393: "f32[1, 128, 2048]" = torch.ops.aten.view.default(addmm_51, [1, 128, 2048]);  addmm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:247, code: attn_output = self.resid_dropout(attn_output)
    clone_71: "f32[1, 128, 2048]" = torch.ops.aten.clone.default(view_393);  view_393 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    add_139: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(clone_71, add_136);  clone_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    var_mean_35 = torch.ops.aten.var_mean.correction(add_139, [2], correction = 0, keepdim = True)
    getitem_70: "f32[1, 128, 1]" = var_mean_35[0]
    getitem_71: "f32[1, 128, 1]" = var_mean_35[1];  var_mean_35 = None
    add_140: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_70, 1e-05);  getitem_70 = None
    rsqrt_35: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_140);  add_140 = None
    sub_53: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_139, getitem_71)
    mul_138: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_35);  sub_53 = None
    mul_139: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_138, primals_231);  mul_138 = None
    add_141: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_139, primals_232);  mul_139 = primals_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_394: "f32[128, 2048]" = torch.ops.aten.view.default(add_141, [128, 2048]);  add_141 = None
    permute_196: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_233, [1, 0]);  primals_233 = None
    addmm_52: "f32[128, 8192]" = torch.ops.aten.addmm.default(primals_234, view_394, permute_196);  primals_234 = None
    view_395: "f32[1, 128, 8192]" = torch.ops.aten.view.default(addmm_52, [1, 128, 8192]);  addmm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_140: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_395, 0.5)
    pow_18: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_395, 3.0)
    mul_141: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_18, 0.044715);  pow_18 = None
    add_142: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(view_395, mul_141);  mul_141 = None
    mul_142: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(add_142, 0.7978845608028654);  add_142 = None
    tanh_17: "f32[1, 128, 8192]" = torch.ops.aten.tanh.default(mul_142);  mul_142 = None
    alias_35: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(tanh_17)
    add_143: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_17, 1.0);  tanh_17 = None
    mul_143: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_140, add_143)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_396: "f32[128, 8192]" = torch.ops.aten.view.default(mul_143, [128, 8192]);  mul_143 = None
    permute_197: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_235, [1, 0]);  primals_235 = None
    addmm_53: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_236, view_396, permute_197);  primals_236 = None
    view_397: "f32[1, 128, 2048]" = torch.ops.aten.view.default(addmm_53, [1, 128, 2048]);  addmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:303, code: hidden_states = self.dropout(hidden_states)
    clone_72: "f32[1, 128, 2048]" = torch.ops.aten.clone.default(view_397);  view_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    add_144: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_139, clone_72);  clone_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    var_mean_36 = torch.ops.aten.var_mean.correction(add_144, [2], correction = 0, keepdim = True)
    getitem_72: "f32[1, 128, 1]" = var_mean_36[0]
    getitem_73: "f32[1, 128, 1]" = var_mean_36[1];  var_mean_36 = None
    add_145: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-05);  getitem_72 = None
    rsqrt_36: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_145);  add_145 = None
    sub_54: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_144, getitem_73)
    mul_144: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_36);  sub_54 = None
    mul_145: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_144, primals_237);  mul_144 = None
    add_146: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_145, primals_238);  mul_145 = primals_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_198: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_239, [1, 0]);  primals_239 = None
    view_398: "f32[128, 2048]" = torch.ops.aten.view.default(add_146, [128, 2048])
    mm_54: "f32[128, 2048]" = torch.ops.aten.mm.default(view_398, permute_198)
    view_399: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_54, [1, 128, 2048]);  mm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_199: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_240, [1, 0]);  primals_240 = None
    view_400: "f32[128, 2048]" = torch.ops.aten.view.default(add_146, [128, 2048])
    mm_55: "f32[128, 2048]" = torch.ops.aten.mm.default(view_400, permute_199)
    view_401: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_55, [1, 128, 2048]);  mm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    permute_200: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_241, [1, 0]);  primals_241 = None
    view_402: "f32[128, 2048]" = torch.ops.aten.view.default(add_146, [128, 2048]);  add_146 = None
    mm_56: "f32[128, 2048]" = torch.ops.aten.mm.default(view_402, permute_200)
    view_403: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_56, [1, 128, 2048]);  mm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_404: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_399, [1, 128, 16, 128]);  view_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_201: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_404, [0, 2, 1, 3]);  view_404 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_405: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_401, [1, 128, 16, 128]);  view_401 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_202: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_405, [0, 2, 1, 3]);  view_405 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_406: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_403, [1, 128, 16, 128]);  view_403 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_203: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_406, [0, 2, 1, 3]);  view_406 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_204: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(permute_202, [0, 1, 3, 2])
    expand_72: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_201, [1, 16, 128, 128]);  permute_201 = None
    view_407: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_72, [16, 128, 128]);  expand_72 = None
    expand_73: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_204, [1, 16, 128, 128]);  permute_204 = None
    view_408: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_73, [16, 128, 128]);  expand_73 = None
    bmm_36: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_407, view_408)
    view_409: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_36, [1, 16, 128, 128]);  bmm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_73: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(primals_336, 0, 0, 9223372036854775807);  primals_336 = None
    slice_74: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(slice_73, 1, 0, 9223372036854775807);  slice_73 = None
    slice_75: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(slice_74, 2, 0, 128);  slice_74 = None
    slice_76: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_75, 3, 0, 128);  slice_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    _tensor_constant18 = self._tensor_constant18
    lift_fresh_copy_18: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant18);  _tensor_constant18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_18: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_76, view_409, lift_fresh_copy_18);  view_409 = lift_fresh_copy_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_18: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(where_18, [-1], True)
    sub_55: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(where_18, amax_18);  where_18 = amax_18 = None
    exp_18: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_55);  sub_55 = None
    sum_19: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_18, [-1], True)
    div_18: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_18, sum_19);  exp_18 = sum_19 = None
    alias_36: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(div_18)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:205, code: attn_weights = self.attn_dropout(attn_weights)
    clone_73: "f32[1, 16, 128, 128]" = torch.ops.aten.clone.default(div_18);  div_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_74: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(clone_73, [1, 16, 128, 128]);  clone_73 = None
    view_410: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_74, [16, 128, 128]);  expand_74 = None
    expand_75: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_203, [1, 16, 128, 128])
    view_411: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_75, [16, 128, 128]);  expand_75 = None
    bmm_37: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_410, view_411)
    view_412: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_37, [1, 16, 128, 128]);  bmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_205: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_412, [0, 2, 1, 3]);  view_412 = None
    clone_74: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_205, memory_format = torch.contiguous_format);  permute_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_413: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_74, [1, 128, 2048]);  clone_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_414: "f32[128, 2048]" = torch.ops.aten.view.default(view_413, [128, 2048]);  view_413 = None
    permute_206: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_242, [1, 0]);  primals_242 = None
    addmm_54: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_243, view_414, permute_206);  primals_243 = None
    view_415: "f32[1, 128, 2048]" = torch.ops.aten.view.default(addmm_54, [1, 128, 2048]);  addmm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:247, code: attn_output = self.resid_dropout(attn_output)
    clone_75: "f32[1, 128, 2048]" = torch.ops.aten.clone.default(view_415);  view_415 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    add_147: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(clone_75, add_144);  clone_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    var_mean_37 = torch.ops.aten.var_mean.correction(add_147, [2], correction = 0, keepdim = True)
    getitem_74: "f32[1, 128, 1]" = var_mean_37[0]
    getitem_75: "f32[1, 128, 1]" = var_mean_37[1];  var_mean_37 = None
    add_148: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_74, 1e-05);  getitem_74 = None
    rsqrt_37: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_148);  add_148 = None
    sub_56: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_147, getitem_75)
    mul_146: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_37);  sub_56 = None
    mul_147: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_146, primals_244);  mul_146 = None
    add_149: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_147, primals_245);  mul_147 = primals_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_416: "f32[128, 2048]" = torch.ops.aten.view.default(add_149, [128, 2048]);  add_149 = None
    permute_207: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_246, [1, 0]);  primals_246 = None
    addmm_55: "f32[128, 8192]" = torch.ops.aten.addmm.default(primals_247, view_416, permute_207);  primals_247 = None
    view_417: "f32[1, 128, 8192]" = torch.ops.aten.view.default(addmm_55, [1, 128, 8192]);  addmm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_148: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_417, 0.5)
    pow_19: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_417, 3.0)
    mul_149: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_19, 0.044715);  pow_19 = None
    add_150: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(view_417, mul_149);  mul_149 = None
    mul_150: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(add_150, 0.7978845608028654);  add_150 = None
    tanh_18: "f32[1, 128, 8192]" = torch.ops.aten.tanh.default(mul_150);  mul_150 = None
    alias_37: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(tanh_18)
    add_151: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_18, 1.0);  tanh_18 = None
    mul_151: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_148, add_151)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_418: "f32[128, 8192]" = torch.ops.aten.view.default(mul_151, [128, 8192]);  mul_151 = None
    permute_208: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_248, [1, 0]);  primals_248 = None
    addmm_56: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_249, view_418, permute_208);  primals_249 = None
    view_419: "f32[1, 128, 2048]" = torch.ops.aten.view.default(addmm_56, [1, 128, 2048]);  addmm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:303, code: hidden_states = self.dropout(hidden_states)
    clone_76: "f32[1, 128, 2048]" = torch.ops.aten.clone.default(view_419);  view_419 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    add_152: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_147, clone_76);  clone_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    var_mean_38 = torch.ops.aten.var_mean.correction(add_152, [2], correction = 0, keepdim = True)
    getitem_76: "f32[1, 128, 1]" = var_mean_38[0]
    getitem_77: "f32[1, 128, 1]" = var_mean_38[1];  var_mean_38 = None
    add_153: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-05);  getitem_76 = None
    rsqrt_38: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_153);  add_153 = None
    sub_57: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_152, getitem_77)
    mul_152: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_57, rsqrt_38);  sub_57 = None
    mul_153: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_152, primals_250);  mul_152 = None
    add_154: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_153, primals_251);  mul_153 = primals_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_209: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_252, [1, 0]);  primals_252 = None
    view_420: "f32[128, 2048]" = torch.ops.aten.view.default(add_154, [128, 2048])
    mm_57: "f32[128, 2048]" = torch.ops.aten.mm.default(view_420, permute_209)
    view_421: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_57, [1, 128, 2048]);  mm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_210: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_253, [1, 0]);  primals_253 = None
    view_422: "f32[128, 2048]" = torch.ops.aten.view.default(add_154, [128, 2048])
    mm_58: "f32[128, 2048]" = torch.ops.aten.mm.default(view_422, permute_210)
    view_423: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_58, [1, 128, 2048]);  mm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    permute_211: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_254, [1, 0]);  primals_254 = None
    view_424: "f32[128, 2048]" = torch.ops.aten.view.default(add_154, [128, 2048]);  add_154 = None
    mm_59: "f32[128, 2048]" = torch.ops.aten.mm.default(view_424, permute_211)
    view_425: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_59, [1, 128, 2048]);  mm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_426: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_421, [1, 128, 16, 128]);  view_421 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_212: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_426, [0, 2, 1, 3]);  view_426 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_427: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_423, [1, 128, 16, 128]);  view_423 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_213: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_427, [0, 2, 1, 3]);  view_427 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_428: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_425, [1, 128, 16, 128]);  view_425 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_214: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_428, [0, 2, 1, 3]);  view_428 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_215: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(permute_213, [0, 1, 3, 2])
    expand_76: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_212, [1, 16, 128, 128]);  permute_212 = None
    view_429: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_76, [16, 128, 128]);  expand_76 = None
    expand_77: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_215, [1, 16, 128, 128]);  permute_215 = None
    view_430: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_77, [16, 128, 128]);  expand_77 = None
    bmm_38: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_429, view_430)
    view_431: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_38, [1, 16, 128, 128]);  bmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_77: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(primals_337, 0, 0, 9223372036854775807);  primals_337 = None
    slice_78: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(slice_77, 1, 0, 9223372036854775807);  slice_77 = None
    slice_79: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(slice_78, 2, 0, 128);  slice_78 = None
    slice_80: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_79, 3, 0, 128);  slice_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    _tensor_constant19 = self._tensor_constant19
    lift_fresh_copy_19: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant19);  _tensor_constant19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_19: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_80, view_431, lift_fresh_copy_19);  view_431 = lift_fresh_copy_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_19: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(where_19, [-1], True)
    sub_58: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(where_19, amax_19);  where_19 = amax_19 = None
    exp_19: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_58);  sub_58 = None
    sum_20: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_19, [-1], True)
    div_19: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_19, sum_20);  exp_19 = sum_20 = None
    alias_38: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(div_19)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:205, code: attn_weights = self.attn_dropout(attn_weights)
    clone_77: "f32[1, 16, 128, 128]" = torch.ops.aten.clone.default(div_19);  div_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_78: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(clone_77, [1, 16, 128, 128]);  clone_77 = None
    view_432: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_78, [16, 128, 128]);  expand_78 = None
    expand_79: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_214, [1, 16, 128, 128])
    view_433: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_79, [16, 128, 128]);  expand_79 = None
    bmm_39: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_432, view_433)
    view_434: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_39, [1, 16, 128, 128]);  bmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_216: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_434, [0, 2, 1, 3]);  view_434 = None
    clone_78: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_216, memory_format = torch.contiguous_format);  permute_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_435: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_78, [1, 128, 2048]);  clone_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_436: "f32[128, 2048]" = torch.ops.aten.view.default(view_435, [128, 2048]);  view_435 = None
    permute_217: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_255, [1, 0]);  primals_255 = None
    addmm_57: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_256, view_436, permute_217);  primals_256 = None
    view_437: "f32[1, 128, 2048]" = torch.ops.aten.view.default(addmm_57, [1, 128, 2048]);  addmm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:247, code: attn_output = self.resid_dropout(attn_output)
    clone_79: "f32[1, 128, 2048]" = torch.ops.aten.clone.default(view_437);  view_437 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    add_155: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(clone_79, add_152);  clone_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    var_mean_39 = torch.ops.aten.var_mean.correction(add_155, [2], correction = 0, keepdim = True)
    getitem_78: "f32[1, 128, 1]" = var_mean_39[0]
    getitem_79: "f32[1, 128, 1]" = var_mean_39[1];  var_mean_39 = None
    add_156: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-05);  getitem_78 = None
    rsqrt_39: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_156);  add_156 = None
    sub_59: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_155, getitem_79)
    mul_154: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_39);  sub_59 = None
    mul_155: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_154, primals_257);  mul_154 = None
    add_157: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_155, primals_258);  mul_155 = primals_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_438: "f32[128, 2048]" = torch.ops.aten.view.default(add_157, [128, 2048]);  add_157 = None
    permute_218: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_259, [1, 0]);  primals_259 = None
    addmm_58: "f32[128, 8192]" = torch.ops.aten.addmm.default(primals_260, view_438, permute_218);  primals_260 = None
    view_439: "f32[1, 128, 8192]" = torch.ops.aten.view.default(addmm_58, [1, 128, 8192]);  addmm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_156: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_439, 0.5)
    pow_20: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_439, 3.0)
    mul_157: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_20, 0.044715);  pow_20 = None
    add_158: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(view_439, mul_157);  mul_157 = None
    mul_158: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(add_158, 0.7978845608028654);  add_158 = None
    tanh_19: "f32[1, 128, 8192]" = torch.ops.aten.tanh.default(mul_158);  mul_158 = None
    alias_39: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(tanh_19)
    add_159: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_19, 1.0);  tanh_19 = None
    mul_159: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_156, add_159)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_440: "f32[128, 8192]" = torch.ops.aten.view.default(mul_159, [128, 8192]);  mul_159 = None
    permute_219: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_261, [1, 0]);  primals_261 = None
    addmm_59: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_262, view_440, permute_219);  primals_262 = None
    view_441: "f32[1, 128, 2048]" = torch.ops.aten.view.default(addmm_59, [1, 128, 2048]);  addmm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:303, code: hidden_states = self.dropout(hidden_states)
    clone_80: "f32[1, 128, 2048]" = torch.ops.aten.clone.default(view_441);  view_441 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    add_160: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_155, clone_80);  clone_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    var_mean_40 = torch.ops.aten.var_mean.correction(add_160, [2], correction = 0, keepdim = True)
    getitem_80: "f32[1, 128, 1]" = var_mean_40[0]
    getitem_81: "f32[1, 128, 1]" = var_mean_40[1];  var_mean_40 = None
    add_161: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-05);  getitem_80 = None
    rsqrt_40: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_161);  add_161 = None
    sub_60: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_160, getitem_81)
    mul_160: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_60, rsqrt_40);  sub_60 = None
    mul_161: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_160, primals_263);  mul_160 = None
    add_162: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_161, primals_264);  mul_161 = primals_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_220: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_265, [1, 0]);  primals_265 = None
    view_442: "f32[128, 2048]" = torch.ops.aten.view.default(add_162, [128, 2048])
    mm_60: "f32[128, 2048]" = torch.ops.aten.mm.default(view_442, permute_220)
    view_443: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_60, [1, 128, 2048]);  mm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_221: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_266, [1, 0]);  primals_266 = None
    view_444: "f32[128, 2048]" = torch.ops.aten.view.default(add_162, [128, 2048])
    mm_61: "f32[128, 2048]" = torch.ops.aten.mm.default(view_444, permute_221)
    view_445: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_61, [1, 128, 2048]);  mm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    permute_222: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_267, [1, 0]);  primals_267 = None
    view_446: "f32[128, 2048]" = torch.ops.aten.view.default(add_162, [128, 2048]);  add_162 = None
    mm_62: "f32[128, 2048]" = torch.ops.aten.mm.default(view_446, permute_222)
    view_447: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_62, [1, 128, 2048]);  mm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_448: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_443, [1, 128, 16, 128]);  view_443 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_223: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_448, [0, 2, 1, 3]);  view_448 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_449: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_445, [1, 128, 16, 128]);  view_445 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_224: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_449, [0, 2, 1, 3]);  view_449 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_450: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_447, [1, 128, 16, 128]);  view_447 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_225: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_450, [0, 2, 1, 3]);  view_450 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_226: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(permute_224, [0, 1, 3, 2])
    expand_80: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_223, [1, 16, 128, 128]);  permute_223 = None
    view_451: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_80, [16, 128, 128]);  expand_80 = None
    expand_81: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_226, [1, 16, 128, 128]);  permute_226 = None
    view_452: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_81, [16, 128, 128]);  expand_81 = None
    bmm_40: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_451, view_452)
    view_453: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_40, [1, 16, 128, 128]);  bmm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_81: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(primals_338, 0, 0, 9223372036854775807);  primals_338 = None
    slice_82: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(slice_81, 1, 0, 9223372036854775807);  slice_81 = None
    slice_83: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(slice_82, 2, 0, 128);  slice_82 = None
    slice_84: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_83, 3, 0, 128);  slice_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    _tensor_constant20 = self._tensor_constant20
    lift_fresh_copy_20: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant20);  _tensor_constant20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_20: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_84, view_453, lift_fresh_copy_20);  view_453 = lift_fresh_copy_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_20: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(where_20, [-1], True)
    sub_61: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(where_20, amax_20);  where_20 = amax_20 = None
    exp_20: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_61);  sub_61 = None
    sum_21: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_20, [-1], True)
    div_20: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_20, sum_21);  exp_20 = sum_21 = None
    alias_40: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(div_20)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:205, code: attn_weights = self.attn_dropout(attn_weights)
    clone_81: "f32[1, 16, 128, 128]" = torch.ops.aten.clone.default(div_20);  div_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_82: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(clone_81, [1, 16, 128, 128]);  clone_81 = None
    view_454: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_82, [16, 128, 128]);  expand_82 = None
    expand_83: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_225, [1, 16, 128, 128])
    view_455: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_83, [16, 128, 128]);  expand_83 = None
    bmm_41: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_454, view_455)
    view_456: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_41, [1, 16, 128, 128]);  bmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_227: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_456, [0, 2, 1, 3]);  view_456 = None
    clone_82: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_227, memory_format = torch.contiguous_format);  permute_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_457: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_82, [1, 128, 2048]);  clone_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_458: "f32[128, 2048]" = torch.ops.aten.view.default(view_457, [128, 2048]);  view_457 = None
    permute_228: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_268, [1, 0]);  primals_268 = None
    addmm_60: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_269, view_458, permute_228);  primals_269 = None
    view_459: "f32[1, 128, 2048]" = torch.ops.aten.view.default(addmm_60, [1, 128, 2048]);  addmm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:247, code: attn_output = self.resid_dropout(attn_output)
    clone_83: "f32[1, 128, 2048]" = torch.ops.aten.clone.default(view_459);  view_459 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    add_163: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(clone_83, add_160);  clone_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    var_mean_41 = torch.ops.aten.var_mean.correction(add_163, [2], correction = 0, keepdim = True)
    getitem_82: "f32[1, 128, 1]" = var_mean_41[0]
    getitem_83: "f32[1, 128, 1]" = var_mean_41[1];  var_mean_41 = None
    add_164: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_82, 1e-05);  getitem_82 = None
    rsqrt_41: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_164);  add_164 = None
    sub_62: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_163, getitem_83)
    mul_162: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_41);  sub_62 = None
    mul_163: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_162, primals_270);  mul_162 = None
    add_165: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_163, primals_271);  mul_163 = primals_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_460: "f32[128, 2048]" = torch.ops.aten.view.default(add_165, [128, 2048]);  add_165 = None
    permute_229: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_272, [1, 0]);  primals_272 = None
    addmm_61: "f32[128, 8192]" = torch.ops.aten.addmm.default(primals_273, view_460, permute_229);  primals_273 = None
    view_461: "f32[1, 128, 8192]" = torch.ops.aten.view.default(addmm_61, [1, 128, 8192]);  addmm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_164: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_461, 0.5)
    pow_21: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_461, 3.0)
    mul_165: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_21, 0.044715);  pow_21 = None
    add_166: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(view_461, mul_165);  mul_165 = None
    mul_166: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(add_166, 0.7978845608028654);  add_166 = None
    tanh_20: "f32[1, 128, 8192]" = torch.ops.aten.tanh.default(mul_166);  mul_166 = None
    alias_41: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(tanh_20)
    add_167: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_20, 1.0);  tanh_20 = None
    mul_167: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_164, add_167)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_462: "f32[128, 8192]" = torch.ops.aten.view.default(mul_167, [128, 8192]);  mul_167 = None
    permute_230: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_274, [1, 0]);  primals_274 = None
    addmm_62: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_275, view_462, permute_230);  primals_275 = None
    view_463: "f32[1, 128, 2048]" = torch.ops.aten.view.default(addmm_62, [1, 128, 2048]);  addmm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:303, code: hidden_states = self.dropout(hidden_states)
    clone_84: "f32[1, 128, 2048]" = torch.ops.aten.clone.default(view_463);  view_463 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    add_168: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_163, clone_84);  clone_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    var_mean_42 = torch.ops.aten.var_mean.correction(add_168, [2], correction = 0, keepdim = True)
    getitem_84: "f32[1, 128, 1]" = var_mean_42[0]
    getitem_85: "f32[1, 128, 1]" = var_mean_42[1];  var_mean_42 = None
    add_169: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_84, 1e-05);  getitem_84 = None
    rsqrt_42: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_169);  add_169 = None
    sub_63: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_168, getitem_85)
    mul_168: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_63, rsqrt_42);  sub_63 = None
    mul_169: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_168, primals_276);  mul_168 = None
    add_170: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_169, primals_277);  mul_169 = primals_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_231: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_278, [1, 0]);  primals_278 = None
    view_464: "f32[128, 2048]" = torch.ops.aten.view.default(add_170, [128, 2048])
    mm_63: "f32[128, 2048]" = torch.ops.aten.mm.default(view_464, permute_231)
    view_465: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_63, [1, 128, 2048]);  mm_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_232: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_279, [1, 0]);  primals_279 = None
    view_466: "f32[128, 2048]" = torch.ops.aten.view.default(add_170, [128, 2048])
    mm_64: "f32[128, 2048]" = torch.ops.aten.mm.default(view_466, permute_232)
    view_467: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_64, [1, 128, 2048]);  mm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    permute_233: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_280, [1, 0]);  primals_280 = None
    view_468: "f32[128, 2048]" = torch.ops.aten.view.default(add_170, [128, 2048]);  add_170 = None
    mm_65: "f32[128, 2048]" = torch.ops.aten.mm.default(view_468, permute_233)
    view_469: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_65, [1, 128, 2048]);  mm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_470: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_465, [1, 128, 16, 128]);  view_465 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_234: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_470, [0, 2, 1, 3]);  view_470 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_471: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_467, [1, 128, 16, 128]);  view_467 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_235: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_471, [0, 2, 1, 3]);  view_471 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_472: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_469, [1, 128, 16, 128]);  view_469 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_236: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_472, [0, 2, 1, 3]);  view_472 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_237: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(permute_235, [0, 1, 3, 2])
    expand_84: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_234, [1, 16, 128, 128]);  permute_234 = None
    view_473: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_84, [16, 128, 128]);  expand_84 = None
    expand_85: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_237, [1, 16, 128, 128]);  permute_237 = None
    view_474: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_85, [16, 128, 128]);  expand_85 = None
    bmm_42: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_473, view_474)
    view_475: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_42, [1, 16, 128, 128]);  bmm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_85: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(primals_339, 0, 0, 9223372036854775807);  primals_339 = None
    slice_86: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(slice_85, 1, 0, 9223372036854775807);  slice_85 = None
    slice_87: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(slice_86, 2, 0, 128);  slice_86 = None
    slice_88: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_87, 3, 0, 128);  slice_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    _tensor_constant21 = self._tensor_constant21
    lift_fresh_copy_21: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant21);  _tensor_constant21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_21: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_88, view_475, lift_fresh_copy_21);  view_475 = lift_fresh_copy_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_21: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(where_21, [-1], True)
    sub_64: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(where_21, amax_21);  where_21 = amax_21 = None
    exp_21: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_64);  sub_64 = None
    sum_22: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_21, [-1], True)
    div_21: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_21, sum_22);  exp_21 = sum_22 = None
    alias_42: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(div_21)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:205, code: attn_weights = self.attn_dropout(attn_weights)
    clone_85: "f32[1, 16, 128, 128]" = torch.ops.aten.clone.default(div_21);  div_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_86: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(clone_85, [1, 16, 128, 128]);  clone_85 = None
    view_476: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_86, [16, 128, 128]);  expand_86 = None
    expand_87: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_236, [1, 16, 128, 128])
    view_477: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_87, [16, 128, 128]);  expand_87 = None
    bmm_43: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_476, view_477)
    view_478: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_43, [1, 16, 128, 128]);  bmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_238: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_478, [0, 2, 1, 3]);  view_478 = None
    clone_86: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_238, memory_format = torch.contiguous_format);  permute_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_479: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_86, [1, 128, 2048]);  clone_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_480: "f32[128, 2048]" = torch.ops.aten.view.default(view_479, [128, 2048]);  view_479 = None
    permute_239: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_281, [1, 0]);  primals_281 = None
    addmm_63: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_282, view_480, permute_239);  primals_282 = None
    view_481: "f32[1, 128, 2048]" = torch.ops.aten.view.default(addmm_63, [1, 128, 2048]);  addmm_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:247, code: attn_output = self.resid_dropout(attn_output)
    clone_87: "f32[1, 128, 2048]" = torch.ops.aten.clone.default(view_481);  view_481 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    add_171: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(clone_87, add_168);  clone_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    var_mean_43 = torch.ops.aten.var_mean.correction(add_171, [2], correction = 0, keepdim = True)
    getitem_86: "f32[1, 128, 1]" = var_mean_43[0]
    getitem_87: "f32[1, 128, 1]" = var_mean_43[1];  var_mean_43 = None
    add_172: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_86, 1e-05);  getitem_86 = None
    rsqrt_43: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_172);  add_172 = None
    sub_65: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_171, getitem_87)
    mul_170: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_65, rsqrt_43);  sub_65 = None
    mul_171: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_170, primals_283);  mul_170 = None
    add_173: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_171, primals_284);  mul_171 = primals_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_482: "f32[128, 2048]" = torch.ops.aten.view.default(add_173, [128, 2048]);  add_173 = None
    permute_240: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_285, [1, 0]);  primals_285 = None
    addmm_64: "f32[128, 8192]" = torch.ops.aten.addmm.default(primals_286, view_482, permute_240);  primals_286 = None
    view_483: "f32[1, 128, 8192]" = torch.ops.aten.view.default(addmm_64, [1, 128, 8192]);  addmm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_172: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_483, 0.5)
    pow_22: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_483, 3.0)
    mul_173: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_22, 0.044715);  pow_22 = None
    add_174: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(view_483, mul_173);  mul_173 = None
    mul_174: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(add_174, 0.7978845608028654);  add_174 = None
    tanh_21: "f32[1, 128, 8192]" = torch.ops.aten.tanh.default(mul_174);  mul_174 = None
    alias_43: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(tanh_21)
    add_175: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_21, 1.0);  tanh_21 = None
    mul_175: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_172, add_175)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_484: "f32[128, 8192]" = torch.ops.aten.view.default(mul_175, [128, 8192]);  mul_175 = None
    permute_241: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_287, [1, 0]);  primals_287 = None
    addmm_65: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_288, view_484, permute_241);  primals_288 = None
    view_485: "f32[1, 128, 2048]" = torch.ops.aten.view.default(addmm_65, [1, 128, 2048]);  addmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:303, code: hidden_states = self.dropout(hidden_states)
    clone_88: "f32[1, 128, 2048]" = torch.ops.aten.clone.default(view_485);  view_485 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    add_176: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_171, clone_88);  clone_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    var_mean_44 = torch.ops.aten.var_mean.correction(add_176, [2], correction = 0, keepdim = True)
    getitem_88: "f32[1, 128, 1]" = var_mean_44[0]
    getitem_89: "f32[1, 128, 1]" = var_mean_44[1];  var_mean_44 = None
    add_177: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-05);  getitem_88 = None
    rsqrt_44: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_177);  add_177 = None
    sub_66: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_176, getitem_89)
    mul_176: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_66, rsqrt_44);  sub_66 = None
    mul_177: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_176, primals_289);  mul_176 = None
    add_178: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_177, primals_290);  mul_177 = primals_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_242: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_291, [1, 0]);  primals_291 = None
    view_486: "f32[128, 2048]" = torch.ops.aten.view.default(add_178, [128, 2048])
    mm_66: "f32[128, 2048]" = torch.ops.aten.mm.default(view_486, permute_242)
    view_487: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_66, [1, 128, 2048]);  mm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_243: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_292, [1, 0]);  primals_292 = None
    view_488: "f32[128, 2048]" = torch.ops.aten.view.default(add_178, [128, 2048])
    mm_67: "f32[128, 2048]" = torch.ops.aten.mm.default(view_488, permute_243)
    view_489: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_67, [1, 128, 2048]);  mm_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    permute_244: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_293, [1, 0]);  primals_293 = None
    view_490: "f32[128, 2048]" = torch.ops.aten.view.default(add_178, [128, 2048]);  add_178 = None
    mm_68: "f32[128, 2048]" = torch.ops.aten.mm.default(view_490, permute_244)
    view_491: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_68, [1, 128, 2048]);  mm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_492: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_487, [1, 128, 16, 128]);  view_487 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_245: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_492, [0, 2, 1, 3]);  view_492 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_493: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_489, [1, 128, 16, 128]);  view_489 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_246: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_493, [0, 2, 1, 3]);  view_493 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_494: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_491, [1, 128, 16, 128]);  view_491 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_247: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_494, [0, 2, 1, 3]);  view_494 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_248: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(permute_246, [0, 1, 3, 2])
    expand_88: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_245, [1, 16, 128, 128]);  permute_245 = None
    view_495: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_88, [16, 128, 128]);  expand_88 = None
    expand_89: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_248, [1, 16, 128, 128]);  permute_248 = None
    view_496: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_89, [16, 128, 128]);  expand_89 = None
    bmm_44: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_495, view_496)
    view_497: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_44, [1, 16, 128, 128]);  bmm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_89: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(primals_340, 0, 0, 9223372036854775807);  primals_340 = None
    slice_90: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(slice_89, 1, 0, 9223372036854775807);  slice_89 = None
    slice_91: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(slice_90, 2, 0, 128);  slice_90 = None
    slice_92: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_91, 3, 0, 128);  slice_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    _tensor_constant22 = self._tensor_constant22
    lift_fresh_copy_22: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant22);  _tensor_constant22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_22: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_92, view_497, lift_fresh_copy_22);  view_497 = lift_fresh_copy_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_22: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(where_22, [-1], True)
    sub_67: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(where_22, amax_22);  where_22 = amax_22 = None
    exp_22: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_67);  sub_67 = None
    sum_23: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_22, [-1], True)
    div_22: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_22, sum_23);  exp_22 = sum_23 = None
    alias_44: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(div_22)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:205, code: attn_weights = self.attn_dropout(attn_weights)
    clone_89: "f32[1, 16, 128, 128]" = torch.ops.aten.clone.default(div_22);  div_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_90: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(clone_89, [1, 16, 128, 128]);  clone_89 = None
    view_498: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_90, [16, 128, 128]);  expand_90 = None
    expand_91: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_247, [1, 16, 128, 128])
    view_499: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_91, [16, 128, 128]);  expand_91 = None
    bmm_45: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_498, view_499)
    view_500: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_45, [1, 16, 128, 128]);  bmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_249: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_500, [0, 2, 1, 3]);  view_500 = None
    clone_90: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_249, memory_format = torch.contiguous_format);  permute_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_501: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_90, [1, 128, 2048]);  clone_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_502: "f32[128, 2048]" = torch.ops.aten.view.default(view_501, [128, 2048]);  view_501 = None
    permute_250: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_294, [1, 0]);  primals_294 = None
    addmm_66: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_295, view_502, permute_250);  primals_295 = None
    view_503: "f32[1, 128, 2048]" = torch.ops.aten.view.default(addmm_66, [1, 128, 2048]);  addmm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:247, code: attn_output = self.resid_dropout(attn_output)
    clone_91: "f32[1, 128, 2048]" = torch.ops.aten.clone.default(view_503);  view_503 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    add_179: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(clone_91, add_176);  clone_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    var_mean_45 = torch.ops.aten.var_mean.correction(add_179, [2], correction = 0, keepdim = True)
    getitem_90: "f32[1, 128, 1]" = var_mean_45[0]
    getitem_91: "f32[1, 128, 1]" = var_mean_45[1];  var_mean_45 = None
    add_180: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_90, 1e-05);  getitem_90 = None
    rsqrt_45: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_180);  add_180 = None
    sub_68: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_179, getitem_91)
    mul_178: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_68, rsqrt_45);  sub_68 = None
    mul_179: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_178, primals_296);  mul_178 = None
    add_181: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_179, primals_297);  mul_179 = primals_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_504: "f32[128, 2048]" = torch.ops.aten.view.default(add_181, [128, 2048]);  add_181 = None
    permute_251: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_298, [1, 0]);  primals_298 = None
    addmm_67: "f32[128, 8192]" = torch.ops.aten.addmm.default(primals_299, view_504, permute_251);  primals_299 = None
    view_505: "f32[1, 128, 8192]" = torch.ops.aten.view.default(addmm_67, [1, 128, 8192]);  addmm_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_180: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_505, 0.5)
    pow_23: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_505, 3.0)
    mul_181: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_23, 0.044715);  pow_23 = None
    add_182: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(view_505, mul_181);  mul_181 = None
    mul_182: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(add_182, 0.7978845608028654);  add_182 = None
    tanh_22: "f32[1, 128, 8192]" = torch.ops.aten.tanh.default(mul_182);  mul_182 = None
    alias_45: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(tanh_22)
    add_183: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_22, 1.0);  tanh_22 = None
    mul_183: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_180, add_183)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_506: "f32[128, 8192]" = torch.ops.aten.view.default(mul_183, [128, 8192]);  mul_183 = None
    permute_252: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_300, [1, 0]);  primals_300 = None
    addmm_68: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_301, view_506, permute_252);  primals_301 = None
    view_507: "f32[1, 128, 2048]" = torch.ops.aten.view.default(addmm_68, [1, 128, 2048]);  addmm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:303, code: hidden_states = self.dropout(hidden_states)
    clone_92: "f32[1, 128, 2048]" = torch.ops.aten.clone.default(view_507);  view_507 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    add_184: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_179, clone_92);  clone_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    var_mean_46 = torch.ops.aten.var_mean.correction(add_184, [2], correction = 0, keepdim = True)
    getitem_92: "f32[1, 128, 1]" = var_mean_46[0]
    getitem_93: "f32[1, 128, 1]" = var_mean_46[1];  var_mean_46 = None
    add_185: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_92, 1e-05);  getitem_92 = None
    rsqrt_46: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_185);  add_185 = None
    sub_69: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_184, getitem_93)
    mul_184: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_69, rsqrt_46);  sub_69 = None
    mul_185: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_184, primals_302);  mul_184 = None
    add_186: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_185, primals_303);  mul_185 = primals_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_253: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_304, [1, 0]);  primals_304 = None
    view_508: "f32[128, 2048]" = torch.ops.aten.view.default(add_186, [128, 2048])
    mm_69: "f32[128, 2048]" = torch.ops.aten.mm.default(view_508, permute_253)
    view_509: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_69, [1, 128, 2048]);  mm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_254: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_305, [1, 0]);  primals_305 = None
    view_510: "f32[128, 2048]" = torch.ops.aten.view.default(add_186, [128, 2048])
    mm_70: "f32[128, 2048]" = torch.ops.aten.mm.default(view_510, permute_254)
    view_511: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_70, [1, 128, 2048]);  mm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    permute_255: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_306, [1, 0]);  primals_306 = None
    view_512: "f32[128, 2048]" = torch.ops.aten.view.default(add_186, [128, 2048]);  add_186 = None
    mm_71: "f32[128, 2048]" = torch.ops.aten.mm.default(view_512, permute_255)
    view_513: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_71, [1, 128, 2048]);  mm_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_514: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_509, [1, 128, 16, 128]);  view_509 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_256: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_514, [0, 2, 1, 3]);  view_514 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_515: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_511, [1, 128, 16, 128]);  view_511 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_257: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_515, [0, 2, 1, 3]);  view_515 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_516: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_513, [1, 128, 16, 128]);  view_513 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_258: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_516, [0, 2, 1, 3]);  view_516 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_259: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(permute_257, [0, 1, 3, 2])
    expand_92: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_256, [1, 16, 128, 128]);  permute_256 = None
    view_517: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_92, [16, 128, 128]);  expand_92 = None
    expand_93: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_259, [1, 16, 128, 128]);  permute_259 = None
    view_518: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_93, [16, 128, 128]);  expand_93 = None
    bmm_46: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_517, view_518)
    view_519: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_46, [1, 16, 128, 128]);  bmm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_93: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(primals_341, 0, 0, 9223372036854775807);  primals_341 = None
    slice_94: "b8[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(slice_93, 1, 0, 9223372036854775807);  slice_93 = None
    slice_95: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(slice_94, 2, 0, 128);  slice_94 = None
    slice_96: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_95, 3, 0, 128);  slice_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    _tensor_constant23 = self._tensor_constant23
    lift_fresh_copy_23: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant23);  _tensor_constant23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_23: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_96, view_519, lift_fresh_copy_23);  view_519 = lift_fresh_copy_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_23: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(where_23, [-1], True)
    sub_70: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(where_23, amax_23);  where_23 = amax_23 = None
    exp_23: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_70);  sub_70 = None
    sum_24: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_23, [-1], True)
    div_23: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_23, sum_24);  exp_23 = sum_24 = None
    alias_46: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(div_23)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:205, code: attn_weights = self.attn_dropout(attn_weights)
    clone_93: "f32[1, 16, 128, 128]" = torch.ops.aten.clone.default(div_23);  div_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_94: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(clone_93, [1, 16, 128, 128]);  clone_93 = None
    view_520: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_94, [16, 128, 128]);  expand_94 = None
    expand_95: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_258, [1, 16, 128, 128])
    view_521: "f32[16, 128, 128]" = torch.ops.aten.view.default(expand_95, [16, 128, 128]);  expand_95 = None
    bmm_47: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_520, view_521)
    view_522: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_47, [1, 16, 128, 128]);  bmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_260: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_522, [0, 2, 1, 3]);  view_522 = None
    clone_94: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_260, memory_format = torch.contiguous_format);  permute_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_523: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_94, [1, 128, 2048]);  clone_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_524: "f32[128, 2048]" = torch.ops.aten.view.default(view_523, [128, 2048]);  view_523 = None
    permute_261: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_307, [1, 0]);  primals_307 = None
    addmm_69: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_308, view_524, permute_261);  primals_308 = None
    view_525: "f32[1, 128, 2048]" = torch.ops.aten.view.default(addmm_69, [1, 128, 2048]);  addmm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:247, code: attn_output = self.resid_dropout(attn_output)
    clone_95: "f32[1, 128, 2048]" = torch.ops.aten.clone.default(view_525);  view_525 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    add_187: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(clone_95, add_184);  clone_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    var_mean_47 = torch.ops.aten.var_mean.correction(add_187, [2], correction = 0, keepdim = True)
    getitem_94: "f32[1, 128, 1]" = var_mean_47[0]
    getitem_95: "f32[1, 128, 1]" = var_mean_47[1];  var_mean_47 = None
    add_188: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_94, 1e-05);  getitem_94 = None
    rsqrt_47: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_188);  add_188 = None
    sub_71: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_187, getitem_95)
    mul_186: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_71, rsqrt_47);  sub_71 = None
    mul_187: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_186, primals_309);  mul_186 = None
    add_189: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_187, primals_310);  mul_187 = primals_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_526: "f32[128, 2048]" = torch.ops.aten.view.default(add_189, [128, 2048]);  add_189 = None
    permute_262: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_311, [1, 0]);  primals_311 = None
    addmm_70: "f32[128, 8192]" = torch.ops.aten.addmm.default(primals_312, view_526, permute_262);  primals_312 = None
    view_527: "f32[1, 128, 8192]" = torch.ops.aten.view.default(addmm_70, [1, 128, 8192]);  addmm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_188: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_527, 0.5)
    pow_24: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_527, 3.0)
    mul_189: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_24, 0.044715);  pow_24 = None
    add_190: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(view_527, mul_189);  mul_189 = None
    mul_190: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(add_190, 0.7978845608028654);  add_190 = None
    tanh_23: "f32[1, 128, 8192]" = torch.ops.aten.tanh.default(mul_190);  mul_190 = None
    alias_47: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(tanh_23)
    add_191: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_23, 1.0);  tanh_23 = None
    mul_191: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_188, add_191)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_528: "f32[128, 8192]" = torch.ops.aten.view.default(mul_191, [128, 8192]);  mul_191 = None
    permute_263: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_313, [1, 0]);  primals_313 = None
    addmm_71: "f32[128, 2048]" = torch.ops.aten.addmm.default(primals_314, view_528, permute_263);  primals_314 = None
    view_529: "f32[1, 128, 2048]" = torch.ops.aten.view.default(addmm_71, [1, 128, 2048]);  addmm_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:303, code: hidden_states = self.dropout(hidden_states)
    clone_96: "f32[1, 128, 2048]" = torch.ops.aten.clone.default(view_529);  view_529 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    add_192: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_187, clone_96);  clone_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:641, code: hidden_states = self.ln_f(hidden_states)
    var_mean_48 = torch.ops.aten.var_mean.correction(add_192, [2], correction = 0, keepdim = True)
    getitem_96: "f32[1, 128, 1]" = var_mean_48[0]
    getitem_97: "f32[1, 128, 1]" = var_mean_48[1];  var_mean_48 = None
    add_193: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_96, 1e-05);  getitem_96 = None
    rsqrt_48: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_193);  add_193 = None
    sub_72: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_192, getitem_97)
    mul_192: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_72, rsqrt_48);  sub_72 = None
    mul_193: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_192, primals_315);  mul_192 = None
    add_194: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_193, primals_316);  mul_193 = primals_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:643, code: hidden_states = hidden_states.view(output_shape)
    view_530: "f32[1, 128, 2048]" = torch.ops.aten.view.default(add_194, [-1, 128, 2048]);  add_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:878, code: logits = self.score(hidden_states)
    permute_264: "f32[2048, 2]" = torch.ops.aten.permute.default(primals_317, [1, 0]);  primals_317 = None
    view_531: "f32[128, 2048]" = torch.ops.aten.view.default(view_530, [128, 2048])
    mm_72: "f32[128, 2]" = torch.ops.aten.mm.default(view_531, permute_264)
    view_532: "f32[1, 128, 2]" = torch.ops.aten.view.default(mm_72, [1, 128, 2]);  mm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:891, code: sequence_lengths = (torch.eq(input_ids, self.config.pad_token_id).long().argmax(-1) - 1).to(
    eq: "b8[1, 128]" = torch.ops.aten.eq.Scalar(primals_342, 0);  primals_342 = None
    convert_element_type: "i64[1, 128]" = torch.ops.prims.convert_element_type.default(eq, torch.int64);  eq = None
    argmax: "i64[1]" = torch.ops.aten.argmax.default(convert_element_type, -1);  convert_element_type = None
    sub_73: "i64[1]" = torch.ops.aten.sub.Tensor(argmax, 1);  argmax = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:901, code: pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]
    iota_1: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    index: "f32[1, 2]" = torch.ops.aten.index.Tensor(view_532, [iota_1, sub_73]);  view_532 = None
    full: "f32[1, 128, 2]" = torch.ops.aten.full.default([1, 128, 2], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    index_put: "f32[1, 128, 2]" = torch.ops.aten.index_put.default(full, [iota_1, sub_73], tangents_50, True);  full = iota_1 = sub_73 = tangents_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:878, code: logits = self.score(hidden_states)
    view_533: "f32[128, 2]" = torch.ops.aten.view.default(index_put, [128, 2]);  index_put = None
    permute_265: "f32[2, 128]" = torch.ops.aten.permute.default(view_533, [1, 0])
    mm_73: "f32[2, 2048]" = torch.ops.aten.mm.default(permute_265, view_531);  permute_265 = view_531 = None
    permute_266: "f32[2048, 2]" = torch.ops.aten.permute.default(mm_73, [1, 0]);  mm_73 = None
    permute_267: "f32[2, 2048]" = torch.ops.aten.permute.default(permute_264, [1, 0]);  permute_264 = None
    mm_74: "f32[128, 2048]" = torch.ops.aten.mm.default(view_533, permute_267);  view_533 = permute_267 = None
    view_534: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_74, [1, 128, 2048]);  mm_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:878, code: logits = self.score(hidden_states)
    add_195: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(tangents_1, view_534);  tangents_1 = view_534 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:878, code: logits = self.score(hidden_states)
    permute_268: "f32[2, 2048]" = torch.ops.aten.permute.default(permute_266, [1, 0]);  permute_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:643, code: hidden_states = hidden_states.view(output_shape)
    view_535: "f32[1, 128, 2048]" = torch.ops.aten.view.default(add_195, [1, 128, 2048]);  add_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:641, code: hidden_states = self.ln_f(hidden_states)
    sub_74: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_192, getitem_97);  add_192 = getitem_97 = None
    mul_194: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_74, rsqrt_48);  sub_74 = None
    mul_195: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_535, primals_315);  primals_315 = None
    mul_196: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_195, 2048)
    sum_25: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_195, [2], True)
    mul_197: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_195, mul_194);  mul_195 = None
    sum_26: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_197, [2], True);  mul_197 = None
    mul_198: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_194, sum_26);  sum_26 = None
    sub_75: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_196, sum_25);  mul_196 = sum_25 = None
    sub_76: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_75, mul_198);  sub_75 = mul_198 = None
    div_24: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_48, 2048);  rsqrt_48 = None
    mul_199: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_24, sub_76);  div_24 = sub_76 = None
    mul_200: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_535, mul_194);  mul_194 = None
    sum_27: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_200, [0, 1]);  mul_200 = None
    sum_28: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_535, [0, 1]);  view_535 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_536: "f32[128, 2048]" = torch.ops.aten.view.default(mul_199, [128, 2048])
    permute_269: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_263, [1, 0]);  permute_263 = None
    mm_75: "f32[128, 8192]" = torch.ops.aten.mm.default(view_536, permute_269);  permute_269 = None
    permute_270: "f32[2048, 128]" = torch.ops.aten.permute.default(view_536, [1, 0])
    mm_76: "f32[2048, 8192]" = torch.ops.aten.mm.default(permute_270, view_528);  permute_270 = view_528 = None
    permute_271: "f32[8192, 2048]" = torch.ops.aten.permute.default(mm_76, [1, 0]);  mm_76 = None
    sum_29: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_536, [0], True);  view_536 = None
    view_537: "f32[2048]" = torch.ops.aten.view.default(sum_29, [2048]);  sum_29 = None
    permute_272: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_271, [1, 0]);  permute_271 = None
    view_538: "f32[1, 128, 8192]" = torch.ops.aten.view.default(mm_75, [1, 128, 8192]);  mm_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_201: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_538, mul_188);  mul_188 = None
    mul_202: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_538, add_191);  view_538 = add_191 = None
    alias_48: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(alias_47);  alias_47 = None
    mul_203: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(alias_48, alias_48);  alias_48 = None
    sub_77: "f32[1, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_203);  mul_203 = None
    mul_204: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_201, sub_77);  mul_201 = sub_77 = None
    mul_205: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_204, 0.7978845608028654);  mul_204 = None
    mul_206: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_205, 0.044715)
    pow_25: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_527, 2.0);  view_527 = None
    mul_207: "f32[1, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_25, 3.0);  pow_25 = None
    mul_208: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_206, mul_207);  mul_206 = mul_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_196: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(mul_205, mul_208);  mul_205 = mul_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_209: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_202, 0.5);  mul_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_197: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(add_196, mul_209);  add_196 = mul_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_539: "f32[128, 8192]" = torch.ops.aten.view.default(add_197, [128, 8192]);  add_197 = None
    permute_273: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_262, [1, 0]);  permute_262 = None
    mm_77: "f32[128, 2048]" = torch.ops.aten.mm.default(view_539, permute_273);  permute_273 = None
    permute_274: "f32[8192, 128]" = torch.ops.aten.permute.default(view_539, [1, 0])
    mm_78: "f32[8192, 2048]" = torch.ops.aten.mm.default(permute_274, view_526);  permute_274 = view_526 = None
    permute_275: "f32[2048, 8192]" = torch.ops.aten.permute.default(mm_78, [1, 0]);  mm_78 = None
    sum_30: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_539, [0], True);  view_539 = None
    view_540: "f32[8192]" = torch.ops.aten.view.default(sum_30, [8192]);  sum_30 = None
    permute_276: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_275, [1, 0]);  permute_275 = None
    view_541: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_77, [1, 128, 2048]);  mm_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    sub_78: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_187, getitem_95);  add_187 = getitem_95 = None
    mul_210: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_78, rsqrt_47);  sub_78 = None
    mul_211: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_541, primals_309);  primals_309 = None
    mul_212: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_211, 2048)
    sum_31: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_211, [2], True)
    mul_213: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_211, mul_210);  mul_211 = None
    sum_32: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_213, [2], True);  mul_213 = None
    mul_214: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_210, sum_32);  sum_32 = None
    sub_79: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_212, sum_31);  mul_212 = sum_31 = None
    sub_80: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_79, mul_214);  sub_79 = mul_214 = None
    div_25: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_47, 2048);  rsqrt_47 = None
    mul_215: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_25, sub_80);  div_25 = sub_80 = None
    mul_216: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_541, mul_210);  mul_210 = None
    sum_33: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_216, [0, 1]);  mul_216 = None
    sum_34: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_541, [0, 1]);  view_541 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    add_198: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_199, mul_215);  mul_199 = mul_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_542: "f32[128, 2048]" = torch.ops.aten.view.default(add_198, [128, 2048])
    permute_277: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_261, [1, 0]);  permute_261 = None
    mm_79: "f32[128, 2048]" = torch.ops.aten.mm.default(view_542, permute_277);  permute_277 = None
    permute_278: "f32[2048, 128]" = torch.ops.aten.permute.default(view_542, [1, 0])
    mm_80: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_278, view_524);  permute_278 = view_524 = None
    permute_279: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_80, [1, 0]);  mm_80 = None
    sum_35: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_542, [0], True);  view_542 = None
    view_543: "f32[2048]" = torch.ops.aten.view.default(sum_35, [2048]);  sum_35 = None
    permute_280: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_279, [1, 0]);  permute_279 = None
    view_544: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_79, [1, 128, 2048]);  mm_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_545: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_544, [1, 128, 16, 128]);  view_544 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_281: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_545, [0, 2, 1, 3]);  view_545 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_546: "f32[16, 128, 128]" = torch.ops.aten.view.default(permute_281, [16, 128, 128]);  permute_281 = None
    permute_282: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_520, [0, 2, 1]);  view_520 = None
    bmm_48: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_282, view_546);  permute_282 = None
    permute_283: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_521, [0, 2, 1]);  view_521 = None
    bmm_49: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_546, permute_283);  view_546 = permute_283 = None
    view_547: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_48, [1, 16, 128, 128]);  bmm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    add_199: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_49, view_547);  tangents_49 = view_547 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_548: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_49, [1, 16, 128, 128]);  bmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_49: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(alias_46);  alias_46 = None
    mul_217: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_548, alias_49);  view_548 = None
    sum_36: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_217, [-1], True)
    mul_218: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_49, sum_36);  alias_49 = sum_36 = None
    sub_81: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_217, mul_218);  mul_217 = mul_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    scalar_tensor: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_24: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_96, sub_81, scalar_tensor);  slice_96 = sub_81 = scalar_tensor = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_549: "f32[16, 128, 128]" = torch.ops.aten.view.default(where_24, [16, 128, 128]);  where_24 = None
    permute_284: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_517, [0, 2, 1]);  view_517 = None
    bmm_50: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_284, view_549);  permute_284 = None
    permute_285: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_518, [0, 2, 1]);  view_518 = None
    bmm_51: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_549, permute_285);  view_549 = permute_285 = None
    view_550: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_50, [1, 16, 128, 128]);  bmm_50 = None
    view_551: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_51, [1, 16, 128, 128]);  bmm_51 = None
    permute_286: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_550, [0, 1, 3, 2]);  view_550 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_200: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_48, permute_286);  tangents_48 = permute_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_287: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_199, [0, 2, 1, 3]);  add_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_97: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_287, memory_format = torch.contiguous_format);  permute_287 = None
    view_552: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_97, [1, 128, 2048]);  clone_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_288: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_200, [0, 2, 1, 3]);  add_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_98: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_288, memory_format = torch.contiguous_format);  permute_288 = None
    view_553: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_98, [1, 128, 2048]);  clone_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_289: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_551, [0, 2, 1, 3]);  view_551 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_99: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_289, memory_format = torch.contiguous_format);  permute_289 = None
    view_554: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_99, [1, 128, 2048]);  clone_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    view_555: "f32[128, 2048]" = torch.ops.aten.view.default(view_552, [128, 2048]);  view_552 = None
    permute_290: "f32[2048, 128]" = torch.ops.aten.permute.default(view_555, [1, 0])
    mm_81: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_290, view_512);  permute_290 = view_512 = None
    permute_291: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_81, [1, 0]);  mm_81 = None
    permute_292: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_255, [1, 0]);  permute_255 = None
    mm_82: "f32[128, 2048]" = torch.ops.aten.mm.default(view_555, permute_292);  view_555 = permute_292 = None
    view_556: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_82, [1, 128, 2048]);  mm_82 = None
    permute_293: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_291, [1, 0]);  permute_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    view_557: "f32[128, 2048]" = torch.ops.aten.view.default(view_553, [128, 2048]);  view_553 = None
    permute_294: "f32[2048, 128]" = torch.ops.aten.permute.default(view_557, [1, 0])
    mm_83: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_294, view_510);  permute_294 = view_510 = None
    permute_295: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_83, [1, 0]);  mm_83 = None
    permute_296: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_254, [1, 0]);  permute_254 = None
    mm_84: "f32[128, 2048]" = torch.ops.aten.mm.default(view_557, permute_296);  view_557 = permute_296 = None
    view_558: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_84, [1, 128, 2048]);  mm_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    add_201: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_556, view_558);  view_556 = view_558 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_297: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_295, [1, 0]);  permute_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    view_559: "f32[128, 2048]" = torch.ops.aten.view.default(view_554, [128, 2048]);  view_554 = None
    permute_298: "f32[2048, 128]" = torch.ops.aten.permute.default(view_559, [1, 0])
    mm_85: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_298, view_508);  permute_298 = view_508 = None
    permute_299: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_85, [1, 0]);  mm_85 = None
    permute_300: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_253, [1, 0]);  permute_253 = None
    mm_86: "f32[128, 2048]" = torch.ops.aten.mm.default(view_559, permute_300);  view_559 = permute_300 = None
    view_560: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_86, [1, 128, 2048]);  mm_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    add_202: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_201, view_560);  add_201 = view_560 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_301: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_299, [1, 0]);  permute_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    sub_82: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_184, getitem_93);  add_184 = getitem_93 = None
    mul_219: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_82, rsqrt_46);  sub_82 = None
    mul_220: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_202, primals_302);  primals_302 = None
    mul_221: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_220, 2048)
    sum_37: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_220, [2], True)
    mul_222: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_220, mul_219);  mul_220 = None
    sum_38: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_222, [2], True);  mul_222 = None
    mul_223: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_219, sum_38);  sum_38 = None
    sub_83: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_221, sum_37);  mul_221 = sum_37 = None
    sub_84: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_83, mul_223);  sub_83 = mul_223 = None
    div_26: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_46, 2048);  rsqrt_46 = None
    mul_224: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_26, sub_84);  div_26 = sub_84 = None
    mul_225: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_202, mul_219);  mul_219 = None
    sum_39: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_225, [0, 1]);  mul_225 = None
    sum_40: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_202, [0, 1]);  add_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    add_203: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_198, mul_224);  add_198 = mul_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_561: "f32[128, 2048]" = torch.ops.aten.view.default(add_203, [128, 2048])
    permute_302: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_252, [1, 0]);  permute_252 = None
    mm_87: "f32[128, 8192]" = torch.ops.aten.mm.default(view_561, permute_302);  permute_302 = None
    permute_303: "f32[2048, 128]" = torch.ops.aten.permute.default(view_561, [1, 0])
    mm_88: "f32[2048, 8192]" = torch.ops.aten.mm.default(permute_303, view_506);  permute_303 = view_506 = None
    permute_304: "f32[8192, 2048]" = torch.ops.aten.permute.default(mm_88, [1, 0]);  mm_88 = None
    sum_41: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_561, [0], True);  view_561 = None
    view_562: "f32[2048]" = torch.ops.aten.view.default(sum_41, [2048]);  sum_41 = None
    permute_305: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_304, [1, 0]);  permute_304 = None
    view_563: "f32[1, 128, 8192]" = torch.ops.aten.view.default(mm_87, [1, 128, 8192]);  mm_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_226: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_563, mul_180);  mul_180 = None
    mul_227: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_563, add_183);  view_563 = add_183 = None
    alias_50: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(alias_45);  alias_45 = None
    mul_228: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(alias_50, alias_50);  alias_50 = None
    sub_85: "f32[1, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_228);  mul_228 = None
    mul_229: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_226, sub_85);  mul_226 = sub_85 = None
    mul_230: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_229, 0.7978845608028654);  mul_229 = None
    mul_231: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_230, 0.044715)
    pow_26: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_505, 2.0);  view_505 = None
    mul_232: "f32[1, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_26, 3.0);  pow_26 = None
    mul_233: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_231, mul_232);  mul_231 = mul_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_204: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(mul_230, mul_233);  mul_230 = mul_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_234: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_227, 0.5);  mul_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_205: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(add_204, mul_234);  add_204 = mul_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_564: "f32[128, 8192]" = torch.ops.aten.view.default(add_205, [128, 8192]);  add_205 = None
    permute_306: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_251, [1, 0]);  permute_251 = None
    mm_89: "f32[128, 2048]" = torch.ops.aten.mm.default(view_564, permute_306);  permute_306 = None
    permute_307: "f32[8192, 128]" = torch.ops.aten.permute.default(view_564, [1, 0])
    mm_90: "f32[8192, 2048]" = torch.ops.aten.mm.default(permute_307, view_504);  permute_307 = view_504 = None
    permute_308: "f32[2048, 8192]" = torch.ops.aten.permute.default(mm_90, [1, 0]);  mm_90 = None
    sum_42: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_564, [0], True);  view_564 = None
    view_565: "f32[8192]" = torch.ops.aten.view.default(sum_42, [8192]);  sum_42 = None
    permute_309: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_308, [1, 0]);  permute_308 = None
    view_566: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_89, [1, 128, 2048]);  mm_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    sub_86: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_179, getitem_91);  add_179 = getitem_91 = None
    mul_235: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_86, rsqrt_45);  sub_86 = None
    mul_236: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_566, primals_296);  primals_296 = None
    mul_237: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_236, 2048)
    sum_43: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_236, [2], True)
    mul_238: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_236, mul_235);  mul_236 = None
    sum_44: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_238, [2], True);  mul_238 = None
    mul_239: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_235, sum_44);  sum_44 = None
    sub_87: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_237, sum_43);  mul_237 = sum_43 = None
    sub_88: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_87, mul_239);  sub_87 = mul_239 = None
    div_27: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_45, 2048);  rsqrt_45 = None
    mul_240: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_27, sub_88);  div_27 = sub_88 = None
    mul_241: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_566, mul_235);  mul_235 = None
    sum_45: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_241, [0, 1]);  mul_241 = None
    sum_46: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_566, [0, 1]);  view_566 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    add_206: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_203, mul_240);  add_203 = mul_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_567: "f32[128, 2048]" = torch.ops.aten.view.default(add_206, [128, 2048])
    permute_310: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_250, [1, 0]);  permute_250 = None
    mm_91: "f32[128, 2048]" = torch.ops.aten.mm.default(view_567, permute_310);  permute_310 = None
    permute_311: "f32[2048, 128]" = torch.ops.aten.permute.default(view_567, [1, 0])
    mm_92: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_311, view_502);  permute_311 = view_502 = None
    permute_312: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_92, [1, 0]);  mm_92 = None
    sum_47: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_567, [0], True);  view_567 = None
    view_568: "f32[2048]" = torch.ops.aten.view.default(sum_47, [2048]);  sum_47 = None
    permute_313: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_312, [1, 0]);  permute_312 = None
    view_569: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_91, [1, 128, 2048]);  mm_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_570: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_569, [1, 128, 16, 128]);  view_569 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_314: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_570, [0, 2, 1, 3]);  view_570 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_571: "f32[16, 128, 128]" = torch.ops.aten.view.default(permute_314, [16, 128, 128]);  permute_314 = None
    permute_315: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_498, [0, 2, 1]);  view_498 = None
    bmm_52: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_315, view_571);  permute_315 = None
    permute_316: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_499, [0, 2, 1]);  view_499 = None
    bmm_53: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_571, permute_316);  view_571 = permute_316 = None
    view_572: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_52, [1, 16, 128, 128]);  bmm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    add_207: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_47, view_572);  tangents_47 = view_572 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_573: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_53, [1, 16, 128, 128]);  bmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_51: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(alias_44);  alias_44 = None
    mul_242: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_573, alias_51);  view_573 = None
    sum_48: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_242, [-1], True)
    mul_243: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_51, sum_48);  alias_51 = sum_48 = None
    sub_89: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_242, mul_243);  mul_242 = mul_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    scalar_tensor_1: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_25: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_92, sub_89, scalar_tensor_1);  slice_92 = sub_89 = scalar_tensor_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_574: "f32[16, 128, 128]" = torch.ops.aten.view.default(where_25, [16, 128, 128]);  where_25 = None
    permute_317: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_495, [0, 2, 1]);  view_495 = None
    bmm_54: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_317, view_574);  permute_317 = None
    permute_318: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_496, [0, 2, 1]);  view_496 = None
    bmm_55: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_574, permute_318);  view_574 = permute_318 = None
    view_575: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_54, [1, 16, 128, 128]);  bmm_54 = None
    view_576: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_55, [1, 16, 128, 128]);  bmm_55 = None
    permute_319: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_575, [0, 1, 3, 2]);  view_575 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_208: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_46, permute_319);  tangents_46 = permute_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_320: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_207, [0, 2, 1, 3]);  add_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_100: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_320, memory_format = torch.contiguous_format);  permute_320 = None
    view_577: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_100, [1, 128, 2048]);  clone_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_321: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_208, [0, 2, 1, 3]);  add_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_101: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_321, memory_format = torch.contiguous_format);  permute_321 = None
    view_578: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_101, [1, 128, 2048]);  clone_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_322: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_576, [0, 2, 1, 3]);  view_576 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_102: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_322, memory_format = torch.contiguous_format);  permute_322 = None
    view_579: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_102, [1, 128, 2048]);  clone_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    view_580: "f32[128, 2048]" = torch.ops.aten.view.default(view_577, [128, 2048]);  view_577 = None
    permute_323: "f32[2048, 128]" = torch.ops.aten.permute.default(view_580, [1, 0])
    mm_93: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_323, view_490);  permute_323 = view_490 = None
    permute_324: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_93, [1, 0]);  mm_93 = None
    permute_325: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_244, [1, 0]);  permute_244 = None
    mm_94: "f32[128, 2048]" = torch.ops.aten.mm.default(view_580, permute_325);  view_580 = permute_325 = None
    view_581: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_94, [1, 128, 2048]);  mm_94 = None
    permute_326: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_324, [1, 0]);  permute_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    view_582: "f32[128, 2048]" = torch.ops.aten.view.default(view_578, [128, 2048]);  view_578 = None
    permute_327: "f32[2048, 128]" = torch.ops.aten.permute.default(view_582, [1, 0])
    mm_95: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_327, view_488);  permute_327 = view_488 = None
    permute_328: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_95, [1, 0]);  mm_95 = None
    permute_329: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_243, [1, 0]);  permute_243 = None
    mm_96: "f32[128, 2048]" = torch.ops.aten.mm.default(view_582, permute_329);  view_582 = permute_329 = None
    view_583: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_96, [1, 128, 2048]);  mm_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    add_209: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_581, view_583);  view_581 = view_583 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_330: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_328, [1, 0]);  permute_328 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    view_584: "f32[128, 2048]" = torch.ops.aten.view.default(view_579, [128, 2048]);  view_579 = None
    permute_331: "f32[2048, 128]" = torch.ops.aten.permute.default(view_584, [1, 0])
    mm_97: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_331, view_486);  permute_331 = view_486 = None
    permute_332: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_97, [1, 0]);  mm_97 = None
    permute_333: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_242, [1, 0]);  permute_242 = None
    mm_98: "f32[128, 2048]" = torch.ops.aten.mm.default(view_584, permute_333);  view_584 = permute_333 = None
    view_585: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_98, [1, 128, 2048]);  mm_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    add_210: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_209, view_585);  add_209 = view_585 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_334: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_332, [1, 0]);  permute_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    sub_90: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_176, getitem_89);  add_176 = getitem_89 = None
    mul_244: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_90, rsqrt_44);  sub_90 = None
    mul_245: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_210, primals_289);  primals_289 = None
    mul_246: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_245, 2048)
    sum_49: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_245, [2], True)
    mul_247: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_245, mul_244);  mul_245 = None
    sum_50: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_247, [2], True);  mul_247 = None
    mul_248: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_244, sum_50);  sum_50 = None
    sub_91: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_246, sum_49);  mul_246 = sum_49 = None
    sub_92: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_91, mul_248);  sub_91 = mul_248 = None
    div_28: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_44, 2048);  rsqrt_44 = None
    mul_249: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_28, sub_92);  div_28 = sub_92 = None
    mul_250: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_210, mul_244);  mul_244 = None
    sum_51: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_250, [0, 1]);  mul_250 = None
    sum_52: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_210, [0, 1]);  add_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    add_211: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_206, mul_249);  add_206 = mul_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_586: "f32[128, 2048]" = torch.ops.aten.view.default(add_211, [128, 2048])
    permute_335: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_241, [1, 0]);  permute_241 = None
    mm_99: "f32[128, 8192]" = torch.ops.aten.mm.default(view_586, permute_335);  permute_335 = None
    permute_336: "f32[2048, 128]" = torch.ops.aten.permute.default(view_586, [1, 0])
    mm_100: "f32[2048, 8192]" = torch.ops.aten.mm.default(permute_336, view_484);  permute_336 = view_484 = None
    permute_337: "f32[8192, 2048]" = torch.ops.aten.permute.default(mm_100, [1, 0]);  mm_100 = None
    sum_53: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_586, [0], True);  view_586 = None
    view_587: "f32[2048]" = torch.ops.aten.view.default(sum_53, [2048]);  sum_53 = None
    permute_338: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_337, [1, 0]);  permute_337 = None
    view_588: "f32[1, 128, 8192]" = torch.ops.aten.view.default(mm_99, [1, 128, 8192]);  mm_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_251: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_588, mul_172);  mul_172 = None
    mul_252: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_588, add_175);  view_588 = add_175 = None
    alias_52: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(alias_43);  alias_43 = None
    mul_253: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(alias_52, alias_52);  alias_52 = None
    sub_93: "f32[1, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_253);  mul_253 = None
    mul_254: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_251, sub_93);  mul_251 = sub_93 = None
    mul_255: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_254, 0.7978845608028654);  mul_254 = None
    mul_256: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_255, 0.044715)
    pow_27: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_483, 2.0);  view_483 = None
    mul_257: "f32[1, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_27, 3.0);  pow_27 = None
    mul_258: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_256, mul_257);  mul_256 = mul_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_212: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(mul_255, mul_258);  mul_255 = mul_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_259: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_252, 0.5);  mul_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_213: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(add_212, mul_259);  add_212 = mul_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_589: "f32[128, 8192]" = torch.ops.aten.view.default(add_213, [128, 8192]);  add_213 = None
    permute_339: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_240, [1, 0]);  permute_240 = None
    mm_101: "f32[128, 2048]" = torch.ops.aten.mm.default(view_589, permute_339);  permute_339 = None
    permute_340: "f32[8192, 128]" = torch.ops.aten.permute.default(view_589, [1, 0])
    mm_102: "f32[8192, 2048]" = torch.ops.aten.mm.default(permute_340, view_482);  permute_340 = view_482 = None
    permute_341: "f32[2048, 8192]" = torch.ops.aten.permute.default(mm_102, [1, 0]);  mm_102 = None
    sum_54: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_589, [0], True);  view_589 = None
    view_590: "f32[8192]" = torch.ops.aten.view.default(sum_54, [8192]);  sum_54 = None
    permute_342: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_341, [1, 0]);  permute_341 = None
    view_591: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_101, [1, 128, 2048]);  mm_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    sub_94: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_171, getitem_87);  add_171 = getitem_87 = None
    mul_260: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_94, rsqrt_43);  sub_94 = None
    mul_261: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_591, primals_283);  primals_283 = None
    mul_262: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_261, 2048)
    sum_55: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_261, [2], True)
    mul_263: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_261, mul_260);  mul_261 = None
    sum_56: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_263, [2], True);  mul_263 = None
    mul_264: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_260, sum_56);  sum_56 = None
    sub_95: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_262, sum_55);  mul_262 = sum_55 = None
    sub_96: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_95, mul_264);  sub_95 = mul_264 = None
    div_29: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_43, 2048);  rsqrt_43 = None
    mul_265: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_29, sub_96);  div_29 = sub_96 = None
    mul_266: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_591, mul_260);  mul_260 = None
    sum_57: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_266, [0, 1]);  mul_266 = None
    sum_58: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_591, [0, 1]);  view_591 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    add_214: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_211, mul_265);  add_211 = mul_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_592: "f32[128, 2048]" = torch.ops.aten.view.default(add_214, [128, 2048])
    permute_343: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_239, [1, 0]);  permute_239 = None
    mm_103: "f32[128, 2048]" = torch.ops.aten.mm.default(view_592, permute_343);  permute_343 = None
    permute_344: "f32[2048, 128]" = torch.ops.aten.permute.default(view_592, [1, 0])
    mm_104: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_344, view_480);  permute_344 = view_480 = None
    permute_345: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_104, [1, 0]);  mm_104 = None
    sum_59: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_592, [0], True);  view_592 = None
    view_593: "f32[2048]" = torch.ops.aten.view.default(sum_59, [2048]);  sum_59 = None
    permute_346: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_345, [1, 0]);  permute_345 = None
    view_594: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_103, [1, 128, 2048]);  mm_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_595: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_594, [1, 128, 16, 128]);  view_594 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_347: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_595, [0, 2, 1, 3]);  view_595 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_596: "f32[16, 128, 128]" = torch.ops.aten.view.default(permute_347, [16, 128, 128]);  permute_347 = None
    permute_348: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_476, [0, 2, 1]);  view_476 = None
    bmm_56: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_348, view_596);  permute_348 = None
    permute_349: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_477, [0, 2, 1]);  view_477 = None
    bmm_57: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_596, permute_349);  view_596 = permute_349 = None
    view_597: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_56, [1, 16, 128, 128]);  bmm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    add_215: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_45, view_597);  tangents_45 = view_597 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_598: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_57, [1, 16, 128, 128]);  bmm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_53: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(alias_42);  alias_42 = None
    mul_267: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_598, alias_53);  view_598 = None
    sum_60: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_267, [-1], True)
    mul_268: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_53, sum_60);  alias_53 = sum_60 = None
    sub_97: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_267, mul_268);  mul_267 = mul_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    scalar_tensor_2: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_26: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_88, sub_97, scalar_tensor_2);  slice_88 = sub_97 = scalar_tensor_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_599: "f32[16, 128, 128]" = torch.ops.aten.view.default(where_26, [16, 128, 128]);  where_26 = None
    permute_350: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_473, [0, 2, 1]);  view_473 = None
    bmm_58: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_350, view_599);  permute_350 = None
    permute_351: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_474, [0, 2, 1]);  view_474 = None
    bmm_59: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_599, permute_351);  view_599 = permute_351 = None
    view_600: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_58, [1, 16, 128, 128]);  bmm_58 = None
    view_601: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_59, [1, 16, 128, 128]);  bmm_59 = None
    permute_352: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_600, [0, 1, 3, 2]);  view_600 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_216: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_44, permute_352);  tangents_44 = permute_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_353: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_215, [0, 2, 1, 3]);  add_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_103: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_353, memory_format = torch.contiguous_format);  permute_353 = None
    view_602: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_103, [1, 128, 2048]);  clone_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_354: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_216, [0, 2, 1, 3]);  add_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_104: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_354, memory_format = torch.contiguous_format);  permute_354 = None
    view_603: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_104, [1, 128, 2048]);  clone_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_355: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_601, [0, 2, 1, 3]);  view_601 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_105: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_355, memory_format = torch.contiguous_format);  permute_355 = None
    view_604: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_105, [1, 128, 2048]);  clone_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    view_605: "f32[128, 2048]" = torch.ops.aten.view.default(view_602, [128, 2048]);  view_602 = None
    permute_356: "f32[2048, 128]" = torch.ops.aten.permute.default(view_605, [1, 0])
    mm_105: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_356, view_468);  permute_356 = view_468 = None
    permute_357: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_105, [1, 0]);  mm_105 = None
    permute_358: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_233, [1, 0]);  permute_233 = None
    mm_106: "f32[128, 2048]" = torch.ops.aten.mm.default(view_605, permute_358);  view_605 = permute_358 = None
    view_606: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_106, [1, 128, 2048]);  mm_106 = None
    permute_359: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_357, [1, 0]);  permute_357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    view_607: "f32[128, 2048]" = torch.ops.aten.view.default(view_603, [128, 2048]);  view_603 = None
    permute_360: "f32[2048, 128]" = torch.ops.aten.permute.default(view_607, [1, 0])
    mm_107: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_360, view_466);  permute_360 = view_466 = None
    permute_361: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_107, [1, 0]);  mm_107 = None
    permute_362: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_232, [1, 0]);  permute_232 = None
    mm_108: "f32[128, 2048]" = torch.ops.aten.mm.default(view_607, permute_362);  view_607 = permute_362 = None
    view_608: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_108, [1, 128, 2048]);  mm_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    add_217: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_606, view_608);  view_606 = view_608 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_363: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_361, [1, 0]);  permute_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    view_609: "f32[128, 2048]" = torch.ops.aten.view.default(view_604, [128, 2048]);  view_604 = None
    permute_364: "f32[2048, 128]" = torch.ops.aten.permute.default(view_609, [1, 0])
    mm_109: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_364, view_464);  permute_364 = view_464 = None
    permute_365: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_109, [1, 0]);  mm_109 = None
    permute_366: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_231, [1, 0]);  permute_231 = None
    mm_110: "f32[128, 2048]" = torch.ops.aten.mm.default(view_609, permute_366);  view_609 = permute_366 = None
    view_610: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_110, [1, 128, 2048]);  mm_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    add_218: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_217, view_610);  add_217 = view_610 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_367: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_365, [1, 0]);  permute_365 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    sub_98: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_168, getitem_85);  add_168 = getitem_85 = None
    mul_269: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_98, rsqrt_42);  sub_98 = None
    mul_270: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_218, primals_276);  primals_276 = None
    mul_271: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_270, 2048)
    sum_61: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_270, [2], True)
    mul_272: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_270, mul_269);  mul_270 = None
    sum_62: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_272, [2], True);  mul_272 = None
    mul_273: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_269, sum_62);  sum_62 = None
    sub_99: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_271, sum_61);  mul_271 = sum_61 = None
    sub_100: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_99, mul_273);  sub_99 = mul_273 = None
    div_30: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_42, 2048);  rsqrt_42 = None
    mul_274: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_30, sub_100);  div_30 = sub_100 = None
    mul_275: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_218, mul_269);  mul_269 = None
    sum_63: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_275, [0, 1]);  mul_275 = None
    sum_64: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_218, [0, 1]);  add_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    add_219: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_214, mul_274);  add_214 = mul_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_611: "f32[128, 2048]" = torch.ops.aten.view.default(add_219, [128, 2048])
    permute_368: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_230, [1, 0]);  permute_230 = None
    mm_111: "f32[128, 8192]" = torch.ops.aten.mm.default(view_611, permute_368);  permute_368 = None
    permute_369: "f32[2048, 128]" = torch.ops.aten.permute.default(view_611, [1, 0])
    mm_112: "f32[2048, 8192]" = torch.ops.aten.mm.default(permute_369, view_462);  permute_369 = view_462 = None
    permute_370: "f32[8192, 2048]" = torch.ops.aten.permute.default(mm_112, [1, 0]);  mm_112 = None
    sum_65: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_611, [0], True);  view_611 = None
    view_612: "f32[2048]" = torch.ops.aten.view.default(sum_65, [2048]);  sum_65 = None
    permute_371: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_370, [1, 0]);  permute_370 = None
    view_613: "f32[1, 128, 8192]" = torch.ops.aten.view.default(mm_111, [1, 128, 8192]);  mm_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_276: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_613, mul_164);  mul_164 = None
    mul_277: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_613, add_167);  view_613 = add_167 = None
    alias_54: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(alias_41);  alias_41 = None
    mul_278: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(alias_54, alias_54);  alias_54 = None
    sub_101: "f32[1, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_278);  mul_278 = None
    mul_279: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_276, sub_101);  mul_276 = sub_101 = None
    mul_280: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_279, 0.7978845608028654);  mul_279 = None
    mul_281: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_280, 0.044715)
    pow_28: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_461, 2.0);  view_461 = None
    mul_282: "f32[1, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_28, 3.0);  pow_28 = None
    mul_283: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_281, mul_282);  mul_281 = mul_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_220: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(mul_280, mul_283);  mul_280 = mul_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_284: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_277, 0.5);  mul_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_221: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(add_220, mul_284);  add_220 = mul_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_614: "f32[128, 8192]" = torch.ops.aten.view.default(add_221, [128, 8192]);  add_221 = None
    permute_372: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_229, [1, 0]);  permute_229 = None
    mm_113: "f32[128, 2048]" = torch.ops.aten.mm.default(view_614, permute_372);  permute_372 = None
    permute_373: "f32[8192, 128]" = torch.ops.aten.permute.default(view_614, [1, 0])
    mm_114: "f32[8192, 2048]" = torch.ops.aten.mm.default(permute_373, view_460);  permute_373 = view_460 = None
    permute_374: "f32[2048, 8192]" = torch.ops.aten.permute.default(mm_114, [1, 0]);  mm_114 = None
    sum_66: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_614, [0], True);  view_614 = None
    view_615: "f32[8192]" = torch.ops.aten.view.default(sum_66, [8192]);  sum_66 = None
    permute_375: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_374, [1, 0]);  permute_374 = None
    view_616: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_113, [1, 128, 2048]);  mm_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    sub_102: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_163, getitem_83);  add_163 = getitem_83 = None
    mul_285: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_102, rsqrt_41);  sub_102 = None
    mul_286: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_616, primals_270);  primals_270 = None
    mul_287: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_286, 2048)
    sum_67: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_286, [2], True)
    mul_288: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_286, mul_285);  mul_286 = None
    sum_68: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_288, [2], True);  mul_288 = None
    mul_289: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_285, sum_68);  sum_68 = None
    sub_103: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_287, sum_67);  mul_287 = sum_67 = None
    sub_104: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_103, mul_289);  sub_103 = mul_289 = None
    div_31: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_41, 2048);  rsqrt_41 = None
    mul_290: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_31, sub_104);  div_31 = sub_104 = None
    mul_291: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_616, mul_285);  mul_285 = None
    sum_69: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_291, [0, 1]);  mul_291 = None
    sum_70: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_616, [0, 1]);  view_616 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    add_222: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_219, mul_290);  add_219 = mul_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_617: "f32[128, 2048]" = torch.ops.aten.view.default(add_222, [128, 2048])
    permute_376: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_228, [1, 0]);  permute_228 = None
    mm_115: "f32[128, 2048]" = torch.ops.aten.mm.default(view_617, permute_376);  permute_376 = None
    permute_377: "f32[2048, 128]" = torch.ops.aten.permute.default(view_617, [1, 0])
    mm_116: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_377, view_458);  permute_377 = view_458 = None
    permute_378: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_116, [1, 0]);  mm_116 = None
    sum_71: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_617, [0], True);  view_617 = None
    view_618: "f32[2048]" = torch.ops.aten.view.default(sum_71, [2048]);  sum_71 = None
    permute_379: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_378, [1, 0]);  permute_378 = None
    view_619: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_115, [1, 128, 2048]);  mm_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_620: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_619, [1, 128, 16, 128]);  view_619 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_380: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_620, [0, 2, 1, 3]);  view_620 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_621: "f32[16, 128, 128]" = torch.ops.aten.view.default(permute_380, [16, 128, 128]);  permute_380 = None
    permute_381: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_454, [0, 2, 1]);  view_454 = None
    bmm_60: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_381, view_621);  permute_381 = None
    permute_382: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_455, [0, 2, 1]);  view_455 = None
    bmm_61: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_621, permute_382);  view_621 = permute_382 = None
    view_622: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_60, [1, 16, 128, 128]);  bmm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    add_223: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_43, view_622);  tangents_43 = view_622 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_623: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_61, [1, 16, 128, 128]);  bmm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_55: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(alias_40);  alias_40 = None
    mul_292: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_623, alias_55);  view_623 = None
    sum_72: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_292, [-1], True)
    mul_293: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_55, sum_72);  alias_55 = sum_72 = None
    sub_105: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_292, mul_293);  mul_292 = mul_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    scalar_tensor_3: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_27: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_84, sub_105, scalar_tensor_3);  slice_84 = sub_105 = scalar_tensor_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_624: "f32[16, 128, 128]" = torch.ops.aten.view.default(where_27, [16, 128, 128]);  where_27 = None
    permute_383: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_451, [0, 2, 1]);  view_451 = None
    bmm_62: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_383, view_624);  permute_383 = None
    permute_384: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_452, [0, 2, 1]);  view_452 = None
    bmm_63: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_624, permute_384);  view_624 = permute_384 = None
    view_625: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_62, [1, 16, 128, 128]);  bmm_62 = None
    view_626: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_63, [1, 16, 128, 128]);  bmm_63 = None
    permute_385: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_625, [0, 1, 3, 2]);  view_625 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_224: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_42, permute_385);  tangents_42 = permute_385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_386: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_223, [0, 2, 1, 3]);  add_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_106: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_386, memory_format = torch.contiguous_format);  permute_386 = None
    view_627: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_106, [1, 128, 2048]);  clone_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_387: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_224, [0, 2, 1, 3]);  add_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_107: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_387, memory_format = torch.contiguous_format);  permute_387 = None
    view_628: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_107, [1, 128, 2048]);  clone_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_388: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_626, [0, 2, 1, 3]);  view_626 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_108: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_388, memory_format = torch.contiguous_format);  permute_388 = None
    view_629: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_108, [1, 128, 2048]);  clone_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    view_630: "f32[128, 2048]" = torch.ops.aten.view.default(view_627, [128, 2048]);  view_627 = None
    permute_389: "f32[2048, 128]" = torch.ops.aten.permute.default(view_630, [1, 0])
    mm_117: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_389, view_446);  permute_389 = view_446 = None
    permute_390: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_117, [1, 0]);  mm_117 = None
    permute_391: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_222, [1, 0]);  permute_222 = None
    mm_118: "f32[128, 2048]" = torch.ops.aten.mm.default(view_630, permute_391);  view_630 = permute_391 = None
    view_631: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_118, [1, 128, 2048]);  mm_118 = None
    permute_392: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_390, [1, 0]);  permute_390 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    view_632: "f32[128, 2048]" = torch.ops.aten.view.default(view_628, [128, 2048]);  view_628 = None
    permute_393: "f32[2048, 128]" = torch.ops.aten.permute.default(view_632, [1, 0])
    mm_119: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_393, view_444);  permute_393 = view_444 = None
    permute_394: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_119, [1, 0]);  mm_119 = None
    permute_395: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_221, [1, 0]);  permute_221 = None
    mm_120: "f32[128, 2048]" = torch.ops.aten.mm.default(view_632, permute_395);  view_632 = permute_395 = None
    view_633: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_120, [1, 128, 2048]);  mm_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    add_225: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_631, view_633);  view_631 = view_633 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_396: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_394, [1, 0]);  permute_394 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    view_634: "f32[128, 2048]" = torch.ops.aten.view.default(view_629, [128, 2048]);  view_629 = None
    permute_397: "f32[2048, 128]" = torch.ops.aten.permute.default(view_634, [1, 0])
    mm_121: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_397, view_442);  permute_397 = view_442 = None
    permute_398: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_121, [1, 0]);  mm_121 = None
    permute_399: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_220, [1, 0]);  permute_220 = None
    mm_122: "f32[128, 2048]" = torch.ops.aten.mm.default(view_634, permute_399);  view_634 = permute_399 = None
    view_635: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_122, [1, 128, 2048]);  mm_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    add_226: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_225, view_635);  add_225 = view_635 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_400: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_398, [1, 0]);  permute_398 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    sub_106: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_160, getitem_81);  add_160 = getitem_81 = None
    mul_294: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_106, rsqrt_40);  sub_106 = None
    mul_295: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_226, primals_263);  primals_263 = None
    mul_296: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_295, 2048)
    sum_73: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_295, [2], True)
    mul_297: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_295, mul_294);  mul_295 = None
    sum_74: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_297, [2], True);  mul_297 = None
    mul_298: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_294, sum_74);  sum_74 = None
    sub_107: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_296, sum_73);  mul_296 = sum_73 = None
    sub_108: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_107, mul_298);  sub_107 = mul_298 = None
    div_32: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_40, 2048);  rsqrt_40 = None
    mul_299: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_32, sub_108);  div_32 = sub_108 = None
    mul_300: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_226, mul_294);  mul_294 = None
    sum_75: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_300, [0, 1]);  mul_300 = None
    sum_76: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_226, [0, 1]);  add_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    add_227: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_222, mul_299);  add_222 = mul_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_636: "f32[128, 2048]" = torch.ops.aten.view.default(add_227, [128, 2048])
    permute_401: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_219, [1, 0]);  permute_219 = None
    mm_123: "f32[128, 8192]" = torch.ops.aten.mm.default(view_636, permute_401);  permute_401 = None
    permute_402: "f32[2048, 128]" = torch.ops.aten.permute.default(view_636, [1, 0])
    mm_124: "f32[2048, 8192]" = torch.ops.aten.mm.default(permute_402, view_440);  permute_402 = view_440 = None
    permute_403: "f32[8192, 2048]" = torch.ops.aten.permute.default(mm_124, [1, 0]);  mm_124 = None
    sum_77: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_636, [0], True);  view_636 = None
    view_637: "f32[2048]" = torch.ops.aten.view.default(sum_77, [2048]);  sum_77 = None
    permute_404: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_403, [1, 0]);  permute_403 = None
    view_638: "f32[1, 128, 8192]" = torch.ops.aten.view.default(mm_123, [1, 128, 8192]);  mm_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_301: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_638, mul_156);  mul_156 = None
    mul_302: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_638, add_159);  view_638 = add_159 = None
    alias_56: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(alias_39);  alias_39 = None
    mul_303: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(alias_56, alias_56);  alias_56 = None
    sub_109: "f32[1, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_303);  mul_303 = None
    mul_304: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_301, sub_109);  mul_301 = sub_109 = None
    mul_305: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_304, 0.7978845608028654);  mul_304 = None
    mul_306: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_305, 0.044715)
    pow_29: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_439, 2.0);  view_439 = None
    mul_307: "f32[1, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_29, 3.0);  pow_29 = None
    mul_308: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_306, mul_307);  mul_306 = mul_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_228: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(mul_305, mul_308);  mul_305 = mul_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_309: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_302, 0.5);  mul_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_229: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(add_228, mul_309);  add_228 = mul_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_639: "f32[128, 8192]" = torch.ops.aten.view.default(add_229, [128, 8192]);  add_229 = None
    permute_405: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_218, [1, 0]);  permute_218 = None
    mm_125: "f32[128, 2048]" = torch.ops.aten.mm.default(view_639, permute_405);  permute_405 = None
    permute_406: "f32[8192, 128]" = torch.ops.aten.permute.default(view_639, [1, 0])
    mm_126: "f32[8192, 2048]" = torch.ops.aten.mm.default(permute_406, view_438);  permute_406 = view_438 = None
    permute_407: "f32[2048, 8192]" = torch.ops.aten.permute.default(mm_126, [1, 0]);  mm_126 = None
    sum_78: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_639, [0], True);  view_639 = None
    view_640: "f32[8192]" = torch.ops.aten.view.default(sum_78, [8192]);  sum_78 = None
    permute_408: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_407, [1, 0]);  permute_407 = None
    view_641: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_125, [1, 128, 2048]);  mm_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    sub_110: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_155, getitem_79);  add_155 = getitem_79 = None
    mul_310: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_110, rsqrt_39);  sub_110 = None
    mul_311: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_641, primals_257);  primals_257 = None
    mul_312: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_311, 2048)
    sum_79: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_311, [2], True)
    mul_313: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_311, mul_310);  mul_311 = None
    sum_80: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_313, [2], True);  mul_313 = None
    mul_314: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_310, sum_80);  sum_80 = None
    sub_111: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_312, sum_79);  mul_312 = sum_79 = None
    sub_112: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_111, mul_314);  sub_111 = mul_314 = None
    div_33: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_39, 2048);  rsqrt_39 = None
    mul_315: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_33, sub_112);  div_33 = sub_112 = None
    mul_316: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_641, mul_310);  mul_310 = None
    sum_81: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_316, [0, 1]);  mul_316 = None
    sum_82: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_641, [0, 1]);  view_641 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    add_230: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_227, mul_315);  add_227 = mul_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_642: "f32[128, 2048]" = torch.ops.aten.view.default(add_230, [128, 2048])
    permute_409: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_217, [1, 0]);  permute_217 = None
    mm_127: "f32[128, 2048]" = torch.ops.aten.mm.default(view_642, permute_409);  permute_409 = None
    permute_410: "f32[2048, 128]" = torch.ops.aten.permute.default(view_642, [1, 0])
    mm_128: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_410, view_436);  permute_410 = view_436 = None
    permute_411: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_128, [1, 0]);  mm_128 = None
    sum_83: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_642, [0], True);  view_642 = None
    view_643: "f32[2048]" = torch.ops.aten.view.default(sum_83, [2048]);  sum_83 = None
    permute_412: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_411, [1, 0]);  permute_411 = None
    view_644: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_127, [1, 128, 2048]);  mm_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_645: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_644, [1, 128, 16, 128]);  view_644 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_413: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_645, [0, 2, 1, 3]);  view_645 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_646: "f32[16, 128, 128]" = torch.ops.aten.view.default(permute_413, [16, 128, 128]);  permute_413 = None
    permute_414: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_432, [0, 2, 1]);  view_432 = None
    bmm_64: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_414, view_646);  permute_414 = None
    permute_415: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_433, [0, 2, 1]);  view_433 = None
    bmm_65: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_646, permute_415);  view_646 = permute_415 = None
    view_647: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_64, [1, 16, 128, 128]);  bmm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    add_231: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_41, view_647);  tangents_41 = view_647 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_648: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_65, [1, 16, 128, 128]);  bmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_57: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(alias_38);  alias_38 = None
    mul_317: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_648, alias_57);  view_648 = None
    sum_84: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_317, [-1], True)
    mul_318: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_57, sum_84);  alias_57 = sum_84 = None
    sub_113: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_317, mul_318);  mul_317 = mul_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    scalar_tensor_4: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_28: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_80, sub_113, scalar_tensor_4);  slice_80 = sub_113 = scalar_tensor_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_649: "f32[16, 128, 128]" = torch.ops.aten.view.default(where_28, [16, 128, 128]);  where_28 = None
    permute_416: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_429, [0, 2, 1]);  view_429 = None
    bmm_66: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_416, view_649);  permute_416 = None
    permute_417: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_430, [0, 2, 1]);  view_430 = None
    bmm_67: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_649, permute_417);  view_649 = permute_417 = None
    view_650: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_66, [1, 16, 128, 128]);  bmm_66 = None
    view_651: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_67, [1, 16, 128, 128]);  bmm_67 = None
    permute_418: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_650, [0, 1, 3, 2]);  view_650 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_232: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_40, permute_418);  tangents_40 = permute_418 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_419: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_231, [0, 2, 1, 3]);  add_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_109: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_419, memory_format = torch.contiguous_format);  permute_419 = None
    view_652: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_109, [1, 128, 2048]);  clone_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_420: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_232, [0, 2, 1, 3]);  add_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_110: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_420, memory_format = torch.contiguous_format);  permute_420 = None
    view_653: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_110, [1, 128, 2048]);  clone_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_421: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_651, [0, 2, 1, 3]);  view_651 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_111: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_421, memory_format = torch.contiguous_format);  permute_421 = None
    view_654: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_111, [1, 128, 2048]);  clone_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    view_655: "f32[128, 2048]" = torch.ops.aten.view.default(view_652, [128, 2048]);  view_652 = None
    permute_422: "f32[2048, 128]" = torch.ops.aten.permute.default(view_655, [1, 0])
    mm_129: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_422, view_424);  permute_422 = view_424 = None
    permute_423: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_129, [1, 0]);  mm_129 = None
    permute_424: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_211, [1, 0]);  permute_211 = None
    mm_130: "f32[128, 2048]" = torch.ops.aten.mm.default(view_655, permute_424);  view_655 = permute_424 = None
    view_656: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_130, [1, 128, 2048]);  mm_130 = None
    permute_425: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_423, [1, 0]);  permute_423 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    view_657: "f32[128, 2048]" = torch.ops.aten.view.default(view_653, [128, 2048]);  view_653 = None
    permute_426: "f32[2048, 128]" = torch.ops.aten.permute.default(view_657, [1, 0])
    mm_131: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_426, view_422);  permute_426 = view_422 = None
    permute_427: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_131, [1, 0]);  mm_131 = None
    permute_428: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_210, [1, 0]);  permute_210 = None
    mm_132: "f32[128, 2048]" = torch.ops.aten.mm.default(view_657, permute_428);  view_657 = permute_428 = None
    view_658: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_132, [1, 128, 2048]);  mm_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    add_233: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_656, view_658);  view_656 = view_658 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_429: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_427, [1, 0]);  permute_427 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    view_659: "f32[128, 2048]" = torch.ops.aten.view.default(view_654, [128, 2048]);  view_654 = None
    permute_430: "f32[2048, 128]" = torch.ops.aten.permute.default(view_659, [1, 0])
    mm_133: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_430, view_420);  permute_430 = view_420 = None
    permute_431: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_133, [1, 0]);  mm_133 = None
    permute_432: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_209, [1, 0]);  permute_209 = None
    mm_134: "f32[128, 2048]" = torch.ops.aten.mm.default(view_659, permute_432);  view_659 = permute_432 = None
    view_660: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_134, [1, 128, 2048]);  mm_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    add_234: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_233, view_660);  add_233 = view_660 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_433: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_431, [1, 0]);  permute_431 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    sub_114: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_152, getitem_77);  add_152 = getitem_77 = None
    mul_319: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_114, rsqrt_38);  sub_114 = None
    mul_320: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_234, primals_250);  primals_250 = None
    mul_321: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_320, 2048)
    sum_85: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_320, [2], True)
    mul_322: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_320, mul_319);  mul_320 = None
    sum_86: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_322, [2], True);  mul_322 = None
    mul_323: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_319, sum_86);  sum_86 = None
    sub_115: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_321, sum_85);  mul_321 = sum_85 = None
    sub_116: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_115, mul_323);  sub_115 = mul_323 = None
    div_34: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_38, 2048);  rsqrt_38 = None
    mul_324: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_34, sub_116);  div_34 = sub_116 = None
    mul_325: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_234, mul_319);  mul_319 = None
    sum_87: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_325, [0, 1]);  mul_325 = None
    sum_88: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_234, [0, 1]);  add_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    add_235: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_230, mul_324);  add_230 = mul_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_661: "f32[128, 2048]" = torch.ops.aten.view.default(add_235, [128, 2048])
    permute_434: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_208, [1, 0]);  permute_208 = None
    mm_135: "f32[128, 8192]" = torch.ops.aten.mm.default(view_661, permute_434);  permute_434 = None
    permute_435: "f32[2048, 128]" = torch.ops.aten.permute.default(view_661, [1, 0])
    mm_136: "f32[2048, 8192]" = torch.ops.aten.mm.default(permute_435, view_418);  permute_435 = view_418 = None
    permute_436: "f32[8192, 2048]" = torch.ops.aten.permute.default(mm_136, [1, 0]);  mm_136 = None
    sum_89: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_661, [0], True);  view_661 = None
    view_662: "f32[2048]" = torch.ops.aten.view.default(sum_89, [2048]);  sum_89 = None
    permute_437: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_436, [1, 0]);  permute_436 = None
    view_663: "f32[1, 128, 8192]" = torch.ops.aten.view.default(mm_135, [1, 128, 8192]);  mm_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_326: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_663, mul_148);  mul_148 = None
    mul_327: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_663, add_151);  view_663 = add_151 = None
    alias_58: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(alias_37);  alias_37 = None
    mul_328: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(alias_58, alias_58);  alias_58 = None
    sub_117: "f32[1, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_328);  mul_328 = None
    mul_329: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_326, sub_117);  mul_326 = sub_117 = None
    mul_330: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_329, 0.7978845608028654);  mul_329 = None
    mul_331: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_330, 0.044715)
    pow_30: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_417, 2.0);  view_417 = None
    mul_332: "f32[1, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_30, 3.0);  pow_30 = None
    mul_333: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_331, mul_332);  mul_331 = mul_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_236: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(mul_330, mul_333);  mul_330 = mul_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_334: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_327, 0.5);  mul_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_237: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(add_236, mul_334);  add_236 = mul_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_664: "f32[128, 8192]" = torch.ops.aten.view.default(add_237, [128, 8192]);  add_237 = None
    permute_438: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_207, [1, 0]);  permute_207 = None
    mm_137: "f32[128, 2048]" = torch.ops.aten.mm.default(view_664, permute_438);  permute_438 = None
    permute_439: "f32[8192, 128]" = torch.ops.aten.permute.default(view_664, [1, 0])
    mm_138: "f32[8192, 2048]" = torch.ops.aten.mm.default(permute_439, view_416);  permute_439 = view_416 = None
    permute_440: "f32[2048, 8192]" = torch.ops.aten.permute.default(mm_138, [1, 0]);  mm_138 = None
    sum_90: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_664, [0], True);  view_664 = None
    view_665: "f32[8192]" = torch.ops.aten.view.default(sum_90, [8192]);  sum_90 = None
    permute_441: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_440, [1, 0]);  permute_440 = None
    view_666: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_137, [1, 128, 2048]);  mm_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    sub_118: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_147, getitem_75);  add_147 = getitem_75 = None
    mul_335: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_118, rsqrt_37);  sub_118 = None
    mul_336: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_666, primals_244);  primals_244 = None
    mul_337: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_336, 2048)
    sum_91: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_336, [2], True)
    mul_338: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_336, mul_335);  mul_336 = None
    sum_92: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_338, [2], True);  mul_338 = None
    mul_339: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_335, sum_92);  sum_92 = None
    sub_119: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_337, sum_91);  mul_337 = sum_91 = None
    sub_120: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_119, mul_339);  sub_119 = mul_339 = None
    div_35: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_37, 2048);  rsqrt_37 = None
    mul_340: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_35, sub_120);  div_35 = sub_120 = None
    mul_341: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_666, mul_335);  mul_335 = None
    sum_93: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_341, [0, 1]);  mul_341 = None
    sum_94: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_666, [0, 1]);  view_666 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    add_238: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_235, mul_340);  add_235 = mul_340 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_667: "f32[128, 2048]" = torch.ops.aten.view.default(add_238, [128, 2048])
    permute_442: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_206, [1, 0]);  permute_206 = None
    mm_139: "f32[128, 2048]" = torch.ops.aten.mm.default(view_667, permute_442);  permute_442 = None
    permute_443: "f32[2048, 128]" = torch.ops.aten.permute.default(view_667, [1, 0])
    mm_140: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_443, view_414);  permute_443 = view_414 = None
    permute_444: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_140, [1, 0]);  mm_140 = None
    sum_95: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_667, [0], True);  view_667 = None
    view_668: "f32[2048]" = torch.ops.aten.view.default(sum_95, [2048]);  sum_95 = None
    permute_445: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_444, [1, 0]);  permute_444 = None
    view_669: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_139, [1, 128, 2048]);  mm_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_670: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_669, [1, 128, 16, 128]);  view_669 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_446: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_670, [0, 2, 1, 3]);  view_670 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_671: "f32[16, 128, 128]" = torch.ops.aten.view.default(permute_446, [16, 128, 128]);  permute_446 = None
    permute_447: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_410, [0, 2, 1]);  view_410 = None
    bmm_68: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_447, view_671);  permute_447 = None
    permute_448: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_411, [0, 2, 1]);  view_411 = None
    bmm_69: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_671, permute_448);  view_671 = permute_448 = None
    view_672: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_68, [1, 16, 128, 128]);  bmm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    add_239: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_39, view_672);  tangents_39 = view_672 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_673: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_69, [1, 16, 128, 128]);  bmm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_59: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(alias_36);  alias_36 = None
    mul_342: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_673, alias_59);  view_673 = None
    sum_96: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_342, [-1], True)
    mul_343: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_59, sum_96);  alias_59 = sum_96 = None
    sub_121: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_342, mul_343);  mul_342 = mul_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    scalar_tensor_5: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_29: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_76, sub_121, scalar_tensor_5);  slice_76 = sub_121 = scalar_tensor_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_674: "f32[16, 128, 128]" = torch.ops.aten.view.default(where_29, [16, 128, 128]);  where_29 = None
    permute_449: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_407, [0, 2, 1]);  view_407 = None
    bmm_70: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_449, view_674);  permute_449 = None
    permute_450: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_408, [0, 2, 1]);  view_408 = None
    bmm_71: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_674, permute_450);  view_674 = permute_450 = None
    view_675: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_70, [1, 16, 128, 128]);  bmm_70 = None
    view_676: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_71, [1, 16, 128, 128]);  bmm_71 = None
    permute_451: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_675, [0, 1, 3, 2]);  view_675 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_240: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_38, permute_451);  tangents_38 = permute_451 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_452: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_239, [0, 2, 1, 3]);  add_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_112: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_452, memory_format = torch.contiguous_format);  permute_452 = None
    view_677: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_112, [1, 128, 2048]);  clone_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_453: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_240, [0, 2, 1, 3]);  add_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_113: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_453, memory_format = torch.contiguous_format);  permute_453 = None
    view_678: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_113, [1, 128, 2048]);  clone_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_454: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_676, [0, 2, 1, 3]);  view_676 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_114: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_454, memory_format = torch.contiguous_format);  permute_454 = None
    view_679: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_114, [1, 128, 2048]);  clone_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    view_680: "f32[128, 2048]" = torch.ops.aten.view.default(view_677, [128, 2048]);  view_677 = None
    permute_455: "f32[2048, 128]" = torch.ops.aten.permute.default(view_680, [1, 0])
    mm_141: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_455, view_402);  permute_455 = view_402 = None
    permute_456: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_141, [1, 0]);  mm_141 = None
    permute_457: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_200, [1, 0]);  permute_200 = None
    mm_142: "f32[128, 2048]" = torch.ops.aten.mm.default(view_680, permute_457);  view_680 = permute_457 = None
    view_681: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_142, [1, 128, 2048]);  mm_142 = None
    permute_458: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_456, [1, 0]);  permute_456 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    view_682: "f32[128, 2048]" = torch.ops.aten.view.default(view_678, [128, 2048]);  view_678 = None
    permute_459: "f32[2048, 128]" = torch.ops.aten.permute.default(view_682, [1, 0])
    mm_143: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_459, view_400);  permute_459 = view_400 = None
    permute_460: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_143, [1, 0]);  mm_143 = None
    permute_461: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_199, [1, 0]);  permute_199 = None
    mm_144: "f32[128, 2048]" = torch.ops.aten.mm.default(view_682, permute_461);  view_682 = permute_461 = None
    view_683: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_144, [1, 128, 2048]);  mm_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    add_241: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_681, view_683);  view_681 = view_683 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_462: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_460, [1, 0]);  permute_460 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    view_684: "f32[128, 2048]" = torch.ops.aten.view.default(view_679, [128, 2048]);  view_679 = None
    permute_463: "f32[2048, 128]" = torch.ops.aten.permute.default(view_684, [1, 0])
    mm_145: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_463, view_398);  permute_463 = view_398 = None
    permute_464: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_145, [1, 0]);  mm_145 = None
    permute_465: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_198, [1, 0]);  permute_198 = None
    mm_146: "f32[128, 2048]" = torch.ops.aten.mm.default(view_684, permute_465);  view_684 = permute_465 = None
    view_685: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_146, [1, 128, 2048]);  mm_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    add_242: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_241, view_685);  add_241 = view_685 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_466: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_464, [1, 0]);  permute_464 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    sub_122: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_144, getitem_73);  add_144 = getitem_73 = None
    mul_344: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_122, rsqrt_36);  sub_122 = None
    mul_345: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_242, primals_237);  primals_237 = None
    mul_346: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_345, 2048)
    sum_97: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_345, [2], True)
    mul_347: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_345, mul_344);  mul_345 = None
    sum_98: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_347, [2], True);  mul_347 = None
    mul_348: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_344, sum_98);  sum_98 = None
    sub_123: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_346, sum_97);  mul_346 = sum_97 = None
    sub_124: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_123, mul_348);  sub_123 = mul_348 = None
    div_36: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_36, 2048);  rsqrt_36 = None
    mul_349: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_36, sub_124);  div_36 = sub_124 = None
    mul_350: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_242, mul_344);  mul_344 = None
    sum_99: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_350, [0, 1]);  mul_350 = None
    sum_100: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_242, [0, 1]);  add_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    add_243: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_238, mul_349);  add_238 = mul_349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_686: "f32[128, 2048]" = torch.ops.aten.view.default(add_243, [128, 2048])
    permute_467: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_197, [1, 0]);  permute_197 = None
    mm_147: "f32[128, 8192]" = torch.ops.aten.mm.default(view_686, permute_467);  permute_467 = None
    permute_468: "f32[2048, 128]" = torch.ops.aten.permute.default(view_686, [1, 0])
    mm_148: "f32[2048, 8192]" = torch.ops.aten.mm.default(permute_468, view_396);  permute_468 = view_396 = None
    permute_469: "f32[8192, 2048]" = torch.ops.aten.permute.default(mm_148, [1, 0]);  mm_148 = None
    sum_101: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_686, [0], True);  view_686 = None
    view_687: "f32[2048]" = torch.ops.aten.view.default(sum_101, [2048]);  sum_101 = None
    permute_470: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_469, [1, 0]);  permute_469 = None
    view_688: "f32[1, 128, 8192]" = torch.ops.aten.view.default(mm_147, [1, 128, 8192]);  mm_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_351: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_688, mul_140);  mul_140 = None
    mul_352: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_688, add_143);  view_688 = add_143 = None
    alias_60: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(alias_35);  alias_35 = None
    mul_353: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(alias_60, alias_60);  alias_60 = None
    sub_125: "f32[1, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_353);  mul_353 = None
    mul_354: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_351, sub_125);  mul_351 = sub_125 = None
    mul_355: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_354, 0.7978845608028654);  mul_354 = None
    mul_356: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_355, 0.044715)
    pow_31: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_395, 2.0);  view_395 = None
    mul_357: "f32[1, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_31, 3.0);  pow_31 = None
    mul_358: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_356, mul_357);  mul_356 = mul_357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_244: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(mul_355, mul_358);  mul_355 = mul_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_359: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_352, 0.5);  mul_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_245: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(add_244, mul_359);  add_244 = mul_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_689: "f32[128, 8192]" = torch.ops.aten.view.default(add_245, [128, 8192]);  add_245 = None
    permute_471: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_196, [1, 0]);  permute_196 = None
    mm_149: "f32[128, 2048]" = torch.ops.aten.mm.default(view_689, permute_471);  permute_471 = None
    permute_472: "f32[8192, 128]" = torch.ops.aten.permute.default(view_689, [1, 0])
    mm_150: "f32[8192, 2048]" = torch.ops.aten.mm.default(permute_472, view_394);  permute_472 = view_394 = None
    permute_473: "f32[2048, 8192]" = torch.ops.aten.permute.default(mm_150, [1, 0]);  mm_150 = None
    sum_102: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_689, [0], True);  view_689 = None
    view_690: "f32[8192]" = torch.ops.aten.view.default(sum_102, [8192]);  sum_102 = None
    permute_474: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_473, [1, 0]);  permute_473 = None
    view_691: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_149, [1, 128, 2048]);  mm_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    sub_126: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_139, getitem_71);  add_139 = getitem_71 = None
    mul_360: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_126, rsqrt_35);  sub_126 = None
    mul_361: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_691, primals_231);  primals_231 = None
    mul_362: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_361, 2048)
    sum_103: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_361, [2], True)
    mul_363: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_361, mul_360);  mul_361 = None
    sum_104: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_363, [2], True);  mul_363 = None
    mul_364: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_360, sum_104);  sum_104 = None
    sub_127: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_362, sum_103);  mul_362 = sum_103 = None
    sub_128: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_127, mul_364);  sub_127 = mul_364 = None
    div_37: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_35, 2048);  rsqrt_35 = None
    mul_365: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_37, sub_128);  div_37 = sub_128 = None
    mul_366: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_691, mul_360);  mul_360 = None
    sum_105: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_366, [0, 1]);  mul_366 = None
    sum_106: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_691, [0, 1]);  view_691 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    add_246: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_243, mul_365);  add_243 = mul_365 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_692: "f32[128, 2048]" = torch.ops.aten.view.default(add_246, [128, 2048])
    permute_475: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_195, [1, 0]);  permute_195 = None
    mm_151: "f32[128, 2048]" = torch.ops.aten.mm.default(view_692, permute_475);  permute_475 = None
    permute_476: "f32[2048, 128]" = torch.ops.aten.permute.default(view_692, [1, 0])
    mm_152: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_476, view_392);  permute_476 = view_392 = None
    permute_477: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_152, [1, 0]);  mm_152 = None
    sum_107: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_692, [0], True);  view_692 = None
    view_693: "f32[2048]" = torch.ops.aten.view.default(sum_107, [2048]);  sum_107 = None
    permute_478: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_477, [1, 0]);  permute_477 = None
    view_694: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_151, [1, 128, 2048]);  mm_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_695: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_694, [1, 128, 16, 128]);  view_694 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_479: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_695, [0, 2, 1, 3]);  view_695 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_696: "f32[16, 128, 128]" = torch.ops.aten.view.default(permute_479, [16, 128, 128]);  permute_479 = None
    permute_480: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_388, [0, 2, 1]);  view_388 = None
    bmm_72: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_480, view_696);  permute_480 = None
    permute_481: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_389, [0, 2, 1]);  view_389 = None
    bmm_73: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_696, permute_481);  view_696 = permute_481 = None
    view_697: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_72, [1, 16, 128, 128]);  bmm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    add_247: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_37, view_697);  tangents_37 = view_697 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_698: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_73, [1, 16, 128, 128]);  bmm_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_61: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(alias_34);  alias_34 = None
    mul_367: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_698, alias_61);  view_698 = None
    sum_108: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_367, [-1], True)
    mul_368: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_61, sum_108);  alias_61 = sum_108 = None
    sub_129: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_367, mul_368);  mul_367 = mul_368 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    scalar_tensor_6: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_30: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_72, sub_129, scalar_tensor_6);  slice_72 = sub_129 = scalar_tensor_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_699: "f32[16, 128, 128]" = torch.ops.aten.view.default(where_30, [16, 128, 128]);  where_30 = None
    permute_482: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_385, [0, 2, 1]);  view_385 = None
    bmm_74: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_482, view_699);  permute_482 = None
    permute_483: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_386, [0, 2, 1]);  view_386 = None
    bmm_75: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_699, permute_483);  view_699 = permute_483 = None
    view_700: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_74, [1, 16, 128, 128]);  bmm_74 = None
    view_701: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_75, [1, 16, 128, 128]);  bmm_75 = None
    permute_484: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_700, [0, 1, 3, 2]);  view_700 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_248: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_36, permute_484);  tangents_36 = permute_484 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_485: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_247, [0, 2, 1, 3]);  add_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_115: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_485, memory_format = torch.contiguous_format);  permute_485 = None
    view_702: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_115, [1, 128, 2048]);  clone_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_486: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_248, [0, 2, 1, 3]);  add_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_116: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_486, memory_format = torch.contiguous_format);  permute_486 = None
    view_703: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_116, [1, 128, 2048]);  clone_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_487: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_701, [0, 2, 1, 3]);  view_701 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_117: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_487, memory_format = torch.contiguous_format);  permute_487 = None
    view_704: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_117, [1, 128, 2048]);  clone_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    view_705: "f32[128, 2048]" = torch.ops.aten.view.default(view_702, [128, 2048]);  view_702 = None
    permute_488: "f32[2048, 128]" = torch.ops.aten.permute.default(view_705, [1, 0])
    mm_153: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_488, view_380);  permute_488 = view_380 = None
    permute_489: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_153, [1, 0]);  mm_153 = None
    permute_490: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_189, [1, 0]);  permute_189 = None
    mm_154: "f32[128, 2048]" = torch.ops.aten.mm.default(view_705, permute_490);  view_705 = permute_490 = None
    view_706: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_154, [1, 128, 2048]);  mm_154 = None
    permute_491: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_489, [1, 0]);  permute_489 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    view_707: "f32[128, 2048]" = torch.ops.aten.view.default(view_703, [128, 2048]);  view_703 = None
    permute_492: "f32[2048, 128]" = torch.ops.aten.permute.default(view_707, [1, 0])
    mm_155: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_492, view_378);  permute_492 = view_378 = None
    permute_493: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_155, [1, 0]);  mm_155 = None
    permute_494: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_188, [1, 0]);  permute_188 = None
    mm_156: "f32[128, 2048]" = torch.ops.aten.mm.default(view_707, permute_494);  view_707 = permute_494 = None
    view_708: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_156, [1, 128, 2048]);  mm_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    add_249: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_706, view_708);  view_706 = view_708 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_495: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_493, [1, 0]);  permute_493 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    view_709: "f32[128, 2048]" = torch.ops.aten.view.default(view_704, [128, 2048]);  view_704 = None
    permute_496: "f32[2048, 128]" = torch.ops.aten.permute.default(view_709, [1, 0])
    mm_157: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_496, view_376);  permute_496 = view_376 = None
    permute_497: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_157, [1, 0]);  mm_157 = None
    permute_498: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_187, [1, 0]);  permute_187 = None
    mm_158: "f32[128, 2048]" = torch.ops.aten.mm.default(view_709, permute_498);  view_709 = permute_498 = None
    view_710: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_158, [1, 128, 2048]);  mm_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    add_250: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_249, view_710);  add_249 = view_710 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_499: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_497, [1, 0]);  permute_497 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    sub_130: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_136, getitem_69);  add_136 = getitem_69 = None
    mul_369: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_130, rsqrt_34);  sub_130 = None
    mul_370: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_250, primals_224);  primals_224 = None
    mul_371: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_370, 2048)
    sum_109: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_370, [2], True)
    mul_372: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_370, mul_369);  mul_370 = None
    sum_110: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_372, [2], True);  mul_372 = None
    mul_373: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_369, sum_110);  sum_110 = None
    sub_131: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_371, sum_109);  mul_371 = sum_109 = None
    sub_132: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_131, mul_373);  sub_131 = mul_373 = None
    div_38: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_34, 2048);  rsqrt_34 = None
    mul_374: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_38, sub_132);  div_38 = sub_132 = None
    mul_375: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_250, mul_369);  mul_369 = None
    sum_111: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_375, [0, 1]);  mul_375 = None
    sum_112: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_250, [0, 1]);  add_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    add_251: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_246, mul_374);  add_246 = mul_374 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_711: "f32[128, 2048]" = torch.ops.aten.view.default(add_251, [128, 2048])
    permute_500: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_186, [1, 0]);  permute_186 = None
    mm_159: "f32[128, 8192]" = torch.ops.aten.mm.default(view_711, permute_500);  permute_500 = None
    permute_501: "f32[2048, 128]" = torch.ops.aten.permute.default(view_711, [1, 0])
    mm_160: "f32[2048, 8192]" = torch.ops.aten.mm.default(permute_501, view_374);  permute_501 = view_374 = None
    permute_502: "f32[8192, 2048]" = torch.ops.aten.permute.default(mm_160, [1, 0]);  mm_160 = None
    sum_113: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_711, [0], True);  view_711 = None
    view_712: "f32[2048]" = torch.ops.aten.view.default(sum_113, [2048]);  sum_113 = None
    permute_503: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_502, [1, 0]);  permute_502 = None
    view_713: "f32[1, 128, 8192]" = torch.ops.aten.view.default(mm_159, [1, 128, 8192]);  mm_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_376: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_713, mul_132);  mul_132 = None
    mul_377: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_713, add_135);  view_713 = add_135 = None
    alias_62: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(alias_33);  alias_33 = None
    mul_378: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(alias_62, alias_62);  alias_62 = None
    sub_133: "f32[1, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_378);  mul_378 = None
    mul_379: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_376, sub_133);  mul_376 = sub_133 = None
    mul_380: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_379, 0.7978845608028654);  mul_379 = None
    mul_381: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_380, 0.044715)
    pow_32: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_373, 2.0);  view_373 = None
    mul_382: "f32[1, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_32, 3.0);  pow_32 = None
    mul_383: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_381, mul_382);  mul_381 = mul_382 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_252: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(mul_380, mul_383);  mul_380 = mul_383 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_384: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_377, 0.5);  mul_377 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_253: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(add_252, mul_384);  add_252 = mul_384 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_714: "f32[128, 8192]" = torch.ops.aten.view.default(add_253, [128, 8192]);  add_253 = None
    permute_504: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_185, [1, 0]);  permute_185 = None
    mm_161: "f32[128, 2048]" = torch.ops.aten.mm.default(view_714, permute_504);  permute_504 = None
    permute_505: "f32[8192, 128]" = torch.ops.aten.permute.default(view_714, [1, 0])
    mm_162: "f32[8192, 2048]" = torch.ops.aten.mm.default(permute_505, view_372);  permute_505 = view_372 = None
    permute_506: "f32[2048, 8192]" = torch.ops.aten.permute.default(mm_162, [1, 0]);  mm_162 = None
    sum_114: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_714, [0], True);  view_714 = None
    view_715: "f32[8192]" = torch.ops.aten.view.default(sum_114, [8192]);  sum_114 = None
    permute_507: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_506, [1, 0]);  permute_506 = None
    view_716: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_161, [1, 128, 2048]);  mm_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    sub_134: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_131, getitem_67);  add_131 = getitem_67 = None
    mul_385: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_134, rsqrt_33);  sub_134 = None
    mul_386: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_716, primals_218);  primals_218 = None
    mul_387: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_386, 2048)
    sum_115: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_386, [2], True)
    mul_388: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_386, mul_385);  mul_386 = None
    sum_116: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_388, [2], True);  mul_388 = None
    mul_389: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_385, sum_116);  sum_116 = None
    sub_135: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_387, sum_115);  mul_387 = sum_115 = None
    sub_136: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_135, mul_389);  sub_135 = mul_389 = None
    div_39: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_33, 2048);  rsqrt_33 = None
    mul_390: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_39, sub_136);  div_39 = sub_136 = None
    mul_391: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_716, mul_385);  mul_385 = None
    sum_117: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_391, [0, 1]);  mul_391 = None
    sum_118: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_716, [0, 1]);  view_716 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    add_254: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_251, mul_390);  add_251 = mul_390 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_717: "f32[128, 2048]" = torch.ops.aten.view.default(add_254, [128, 2048])
    permute_508: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_184, [1, 0]);  permute_184 = None
    mm_163: "f32[128, 2048]" = torch.ops.aten.mm.default(view_717, permute_508);  permute_508 = None
    permute_509: "f32[2048, 128]" = torch.ops.aten.permute.default(view_717, [1, 0])
    mm_164: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_509, view_370);  permute_509 = view_370 = None
    permute_510: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_164, [1, 0]);  mm_164 = None
    sum_119: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_717, [0], True);  view_717 = None
    view_718: "f32[2048]" = torch.ops.aten.view.default(sum_119, [2048]);  sum_119 = None
    permute_511: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_510, [1, 0]);  permute_510 = None
    view_719: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_163, [1, 128, 2048]);  mm_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_720: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_719, [1, 128, 16, 128]);  view_719 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_512: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_720, [0, 2, 1, 3]);  view_720 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_721: "f32[16, 128, 128]" = torch.ops.aten.view.default(permute_512, [16, 128, 128]);  permute_512 = None
    permute_513: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_366, [0, 2, 1]);  view_366 = None
    bmm_76: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_513, view_721);  permute_513 = None
    permute_514: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_367, [0, 2, 1]);  view_367 = None
    bmm_77: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_721, permute_514);  view_721 = permute_514 = None
    view_722: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_76, [1, 16, 128, 128]);  bmm_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    add_255: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_35, view_722);  tangents_35 = view_722 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_723: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_77, [1, 16, 128, 128]);  bmm_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_63: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(alias_32);  alias_32 = None
    mul_392: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_723, alias_63);  view_723 = None
    sum_120: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_392, [-1], True)
    mul_393: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_63, sum_120);  alias_63 = sum_120 = None
    sub_137: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_392, mul_393);  mul_392 = mul_393 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    scalar_tensor_7: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_31: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_68, sub_137, scalar_tensor_7);  slice_68 = sub_137 = scalar_tensor_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_724: "f32[16, 128, 128]" = torch.ops.aten.view.default(where_31, [16, 128, 128]);  where_31 = None
    permute_515: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_363, [0, 2, 1]);  view_363 = None
    bmm_78: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_515, view_724);  permute_515 = None
    permute_516: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_364, [0, 2, 1]);  view_364 = None
    bmm_79: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_724, permute_516);  view_724 = permute_516 = None
    view_725: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_78, [1, 16, 128, 128]);  bmm_78 = None
    view_726: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_79, [1, 16, 128, 128]);  bmm_79 = None
    permute_517: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_725, [0, 1, 3, 2]);  view_725 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_256: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_34, permute_517);  tangents_34 = permute_517 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_518: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_255, [0, 2, 1, 3]);  add_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_118: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_518, memory_format = torch.contiguous_format);  permute_518 = None
    view_727: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_118, [1, 128, 2048]);  clone_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_519: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_256, [0, 2, 1, 3]);  add_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_119: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_519, memory_format = torch.contiguous_format);  permute_519 = None
    view_728: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_119, [1, 128, 2048]);  clone_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_520: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_726, [0, 2, 1, 3]);  view_726 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_120: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_520, memory_format = torch.contiguous_format);  permute_520 = None
    view_729: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_120, [1, 128, 2048]);  clone_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    view_730: "f32[128, 2048]" = torch.ops.aten.view.default(view_727, [128, 2048]);  view_727 = None
    permute_521: "f32[2048, 128]" = torch.ops.aten.permute.default(view_730, [1, 0])
    mm_165: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_521, view_358);  permute_521 = view_358 = None
    permute_522: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_165, [1, 0]);  mm_165 = None
    permute_523: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_178, [1, 0]);  permute_178 = None
    mm_166: "f32[128, 2048]" = torch.ops.aten.mm.default(view_730, permute_523);  view_730 = permute_523 = None
    view_731: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_166, [1, 128, 2048]);  mm_166 = None
    permute_524: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_522, [1, 0]);  permute_522 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    view_732: "f32[128, 2048]" = torch.ops.aten.view.default(view_728, [128, 2048]);  view_728 = None
    permute_525: "f32[2048, 128]" = torch.ops.aten.permute.default(view_732, [1, 0])
    mm_167: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_525, view_356);  permute_525 = view_356 = None
    permute_526: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_167, [1, 0]);  mm_167 = None
    permute_527: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_177, [1, 0]);  permute_177 = None
    mm_168: "f32[128, 2048]" = torch.ops.aten.mm.default(view_732, permute_527);  view_732 = permute_527 = None
    view_733: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_168, [1, 128, 2048]);  mm_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    add_257: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_731, view_733);  view_731 = view_733 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_528: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_526, [1, 0]);  permute_526 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    view_734: "f32[128, 2048]" = torch.ops.aten.view.default(view_729, [128, 2048]);  view_729 = None
    permute_529: "f32[2048, 128]" = torch.ops.aten.permute.default(view_734, [1, 0])
    mm_169: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_529, view_354);  permute_529 = view_354 = None
    permute_530: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_169, [1, 0]);  mm_169 = None
    permute_531: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_176, [1, 0]);  permute_176 = None
    mm_170: "f32[128, 2048]" = torch.ops.aten.mm.default(view_734, permute_531);  view_734 = permute_531 = None
    view_735: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_170, [1, 128, 2048]);  mm_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    add_258: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_257, view_735);  add_257 = view_735 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_532: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_530, [1, 0]);  permute_530 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    sub_138: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_128, getitem_65);  add_128 = getitem_65 = None
    mul_394: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_138, rsqrt_32);  sub_138 = None
    mul_395: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_258, primals_211);  primals_211 = None
    mul_396: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_395, 2048)
    sum_121: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_395, [2], True)
    mul_397: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_395, mul_394);  mul_395 = None
    sum_122: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_397, [2], True);  mul_397 = None
    mul_398: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_394, sum_122);  sum_122 = None
    sub_139: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_396, sum_121);  mul_396 = sum_121 = None
    sub_140: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_139, mul_398);  sub_139 = mul_398 = None
    div_40: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_32, 2048);  rsqrt_32 = None
    mul_399: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_40, sub_140);  div_40 = sub_140 = None
    mul_400: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_258, mul_394);  mul_394 = None
    sum_123: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_400, [0, 1]);  mul_400 = None
    sum_124: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_258, [0, 1]);  add_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    add_259: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_254, mul_399);  add_254 = mul_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_736: "f32[128, 2048]" = torch.ops.aten.view.default(add_259, [128, 2048])
    permute_533: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_175, [1, 0]);  permute_175 = None
    mm_171: "f32[128, 8192]" = torch.ops.aten.mm.default(view_736, permute_533);  permute_533 = None
    permute_534: "f32[2048, 128]" = torch.ops.aten.permute.default(view_736, [1, 0])
    mm_172: "f32[2048, 8192]" = torch.ops.aten.mm.default(permute_534, view_352);  permute_534 = view_352 = None
    permute_535: "f32[8192, 2048]" = torch.ops.aten.permute.default(mm_172, [1, 0]);  mm_172 = None
    sum_125: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_736, [0], True);  view_736 = None
    view_737: "f32[2048]" = torch.ops.aten.view.default(sum_125, [2048]);  sum_125 = None
    permute_536: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_535, [1, 0]);  permute_535 = None
    view_738: "f32[1, 128, 8192]" = torch.ops.aten.view.default(mm_171, [1, 128, 8192]);  mm_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_401: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_738, mul_124);  mul_124 = None
    mul_402: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_738, add_127);  view_738 = add_127 = None
    alias_64: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(alias_31);  alias_31 = None
    mul_403: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(alias_64, alias_64);  alias_64 = None
    sub_141: "f32[1, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_403);  mul_403 = None
    mul_404: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_401, sub_141);  mul_401 = sub_141 = None
    mul_405: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_404, 0.7978845608028654);  mul_404 = None
    mul_406: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_405, 0.044715)
    pow_33: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_351, 2.0);  view_351 = None
    mul_407: "f32[1, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_33, 3.0);  pow_33 = None
    mul_408: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_406, mul_407);  mul_406 = mul_407 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_260: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(mul_405, mul_408);  mul_405 = mul_408 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_409: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_402, 0.5);  mul_402 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_261: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(add_260, mul_409);  add_260 = mul_409 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_739: "f32[128, 8192]" = torch.ops.aten.view.default(add_261, [128, 8192]);  add_261 = None
    permute_537: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_174, [1, 0]);  permute_174 = None
    mm_173: "f32[128, 2048]" = torch.ops.aten.mm.default(view_739, permute_537);  permute_537 = None
    permute_538: "f32[8192, 128]" = torch.ops.aten.permute.default(view_739, [1, 0])
    mm_174: "f32[8192, 2048]" = torch.ops.aten.mm.default(permute_538, view_350);  permute_538 = view_350 = None
    permute_539: "f32[2048, 8192]" = torch.ops.aten.permute.default(mm_174, [1, 0]);  mm_174 = None
    sum_126: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_739, [0], True);  view_739 = None
    view_740: "f32[8192]" = torch.ops.aten.view.default(sum_126, [8192]);  sum_126 = None
    permute_540: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_539, [1, 0]);  permute_539 = None
    view_741: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_173, [1, 128, 2048]);  mm_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    sub_142: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_123, getitem_63);  add_123 = getitem_63 = None
    mul_410: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_142, rsqrt_31);  sub_142 = None
    mul_411: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_741, primals_205);  primals_205 = None
    mul_412: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_411, 2048)
    sum_127: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_411, [2], True)
    mul_413: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_411, mul_410);  mul_411 = None
    sum_128: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_413, [2], True);  mul_413 = None
    mul_414: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_410, sum_128);  sum_128 = None
    sub_143: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_412, sum_127);  mul_412 = sum_127 = None
    sub_144: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_143, mul_414);  sub_143 = mul_414 = None
    div_41: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_31, 2048);  rsqrt_31 = None
    mul_415: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_41, sub_144);  div_41 = sub_144 = None
    mul_416: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_741, mul_410);  mul_410 = None
    sum_129: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_416, [0, 1]);  mul_416 = None
    sum_130: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_741, [0, 1]);  view_741 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    add_262: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_259, mul_415);  add_259 = mul_415 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_742: "f32[128, 2048]" = torch.ops.aten.view.default(add_262, [128, 2048])
    permute_541: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_173, [1, 0]);  permute_173 = None
    mm_175: "f32[128, 2048]" = torch.ops.aten.mm.default(view_742, permute_541);  permute_541 = None
    permute_542: "f32[2048, 128]" = torch.ops.aten.permute.default(view_742, [1, 0])
    mm_176: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_542, view_348);  permute_542 = view_348 = None
    permute_543: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_176, [1, 0]);  mm_176 = None
    sum_131: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_742, [0], True);  view_742 = None
    view_743: "f32[2048]" = torch.ops.aten.view.default(sum_131, [2048]);  sum_131 = None
    permute_544: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_543, [1, 0]);  permute_543 = None
    view_744: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_175, [1, 128, 2048]);  mm_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_745: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_744, [1, 128, 16, 128]);  view_744 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_545: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_745, [0, 2, 1, 3]);  view_745 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_746: "f32[16, 128, 128]" = torch.ops.aten.view.default(permute_545, [16, 128, 128]);  permute_545 = None
    permute_546: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_344, [0, 2, 1]);  view_344 = None
    bmm_80: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_546, view_746);  permute_546 = None
    permute_547: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_345, [0, 2, 1]);  view_345 = None
    bmm_81: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_746, permute_547);  view_746 = permute_547 = None
    view_747: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_80, [1, 16, 128, 128]);  bmm_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    add_263: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_33, view_747);  tangents_33 = view_747 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_748: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_81, [1, 16, 128, 128]);  bmm_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_65: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(alias_30);  alias_30 = None
    mul_417: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_748, alias_65);  view_748 = None
    sum_132: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_417, [-1], True)
    mul_418: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_65, sum_132);  alias_65 = sum_132 = None
    sub_145: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_417, mul_418);  mul_417 = mul_418 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    scalar_tensor_8: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_32: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_64, sub_145, scalar_tensor_8);  slice_64 = sub_145 = scalar_tensor_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_749: "f32[16, 128, 128]" = torch.ops.aten.view.default(where_32, [16, 128, 128]);  where_32 = None
    permute_548: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_341, [0, 2, 1]);  view_341 = None
    bmm_82: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_548, view_749);  permute_548 = None
    permute_549: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_342, [0, 2, 1]);  view_342 = None
    bmm_83: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_749, permute_549);  view_749 = permute_549 = None
    view_750: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_82, [1, 16, 128, 128]);  bmm_82 = None
    view_751: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_83, [1, 16, 128, 128]);  bmm_83 = None
    permute_550: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_750, [0, 1, 3, 2]);  view_750 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_264: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_32, permute_550);  tangents_32 = permute_550 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_551: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_263, [0, 2, 1, 3]);  add_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_121: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_551, memory_format = torch.contiguous_format);  permute_551 = None
    view_752: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_121, [1, 128, 2048]);  clone_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_552: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_264, [0, 2, 1, 3]);  add_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_122: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_552, memory_format = torch.contiguous_format);  permute_552 = None
    view_753: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_122, [1, 128, 2048]);  clone_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_553: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_751, [0, 2, 1, 3]);  view_751 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_123: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_553, memory_format = torch.contiguous_format);  permute_553 = None
    view_754: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_123, [1, 128, 2048]);  clone_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    view_755: "f32[128, 2048]" = torch.ops.aten.view.default(view_752, [128, 2048]);  view_752 = None
    permute_554: "f32[2048, 128]" = torch.ops.aten.permute.default(view_755, [1, 0])
    mm_177: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_554, view_336);  permute_554 = view_336 = None
    permute_555: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_177, [1, 0]);  mm_177 = None
    permute_556: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_167, [1, 0]);  permute_167 = None
    mm_178: "f32[128, 2048]" = torch.ops.aten.mm.default(view_755, permute_556);  view_755 = permute_556 = None
    view_756: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_178, [1, 128, 2048]);  mm_178 = None
    permute_557: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_555, [1, 0]);  permute_555 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    view_757: "f32[128, 2048]" = torch.ops.aten.view.default(view_753, [128, 2048]);  view_753 = None
    permute_558: "f32[2048, 128]" = torch.ops.aten.permute.default(view_757, [1, 0])
    mm_179: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_558, view_334);  permute_558 = view_334 = None
    permute_559: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_179, [1, 0]);  mm_179 = None
    permute_560: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_166, [1, 0]);  permute_166 = None
    mm_180: "f32[128, 2048]" = torch.ops.aten.mm.default(view_757, permute_560);  view_757 = permute_560 = None
    view_758: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_180, [1, 128, 2048]);  mm_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    add_265: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_756, view_758);  view_756 = view_758 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_561: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_559, [1, 0]);  permute_559 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    view_759: "f32[128, 2048]" = torch.ops.aten.view.default(view_754, [128, 2048]);  view_754 = None
    permute_562: "f32[2048, 128]" = torch.ops.aten.permute.default(view_759, [1, 0])
    mm_181: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_562, view_332);  permute_562 = view_332 = None
    permute_563: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_181, [1, 0]);  mm_181 = None
    permute_564: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_165, [1, 0]);  permute_165 = None
    mm_182: "f32[128, 2048]" = torch.ops.aten.mm.default(view_759, permute_564);  view_759 = permute_564 = None
    view_760: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_182, [1, 128, 2048]);  mm_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    add_266: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_265, view_760);  add_265 = view_760 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_565: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_563, [1, 0]);  permute_563 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    sub_146: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_120, getitem_61);  add_120 = getitem_61 = None
    mul_419: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_146, rsqrt_30);  sub_146 = None
    mul_420: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_266, primals_198);  primals_198 = None
    mul_421: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_420, 2048)
    sum_133: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_420, [2], True)
    mul_422: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_420, mul_419);  mul_420 = None
    sum_134: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_422, [2], True);  mul_422 = None
    mul_423: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_419, sum_134);  sum_134 = None
    sub_147: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_421, sum_133);  mul_421 = sum_133 = None
    sub_148: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_147, mul_423);  sub_147 = mul_423 = None
    div_42: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_30, 2048);  rsqrt_30 = None
    mul_424: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_42, sub_148);  div_42 = sub_148 = None
    mul_425: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_266, mul_419);  mul_419 = None
    sum_135: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_425, [0, 1]);  mul_425 = None
    sum_136: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_266, [0, 1]);  add_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    add_267: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_262, mul_424);  add_262 = mul_424 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_761: "f32[128, 2048]" = torch.ops.aten.view.default(add_267, [128, 2048])
    permute_566: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_164, [1, 0]);  permute_164 = None
    mm_183: "f32[128, 8192]" = torch.ops.aten.mm.default(view_761, permute_566);  permute_566 = None
    permute_567: "f32[2048, 128]" = torch.ops.aten.permute.default(view_761, [1, 0])
    mm_184: "f32[2048, 8192]" = torch.ops.aten.mm.default(permute_567, view_330);  permute_567 = view_330 = None
    permute_568: "f32[8192, 2048]" = torch.ops.aten.permute.default(mm_184, [1, 0]);  mm_184 = None
    sum_137: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_761, [0], True);  view_761 = None
    view_762: "f32[2048]" = torch.ops.aten.view.default(sum_137, [2048]);  sum_137 = None
    permute_569: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_568, [1, 0]);  permute_568 = None
    view_763: "f32[1, 128, 8192]" = torch.ops.aten.view.default(mm_183, [1, 128, 8192]);  mm_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_426: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_763, mul_116);  mul_116 = None
    mul_427: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_763, add_119);  view_763 = add_119 = None
    alias_66: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(alias_29);  alias_29 = None
    mul_428: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(alias_66, alias_66);  alias_66 = None
    sub_149: "f32[1, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_428);  mul_428 = None
    mul_429: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_426, sub_149);  mul_426 = sub_149 = None
    mul_430: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_429, 0.7978845608028654);  mul_429 = None
    mul_431: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_430, 0.044715)
    pow_34: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_329, 2.0);  view_329 = None
    mul_432: "f32[1, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_34, 3.0);  pow_34 = None
    mul_433: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_431, mul_432);  mul_431 = mul_432 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_268: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(mul_430, mul_433);  mul_430 = mul_433 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_434: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_427, 0.5);  mul_427 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_269: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(add_268, mul_434);  add_268 = mul_434 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_764: "f32[128, 8192]" = torch.ops.aten.view.default(add_269, [128, 8192]);  add_269 = None
    permute_570: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_163, [1, 0]);  permute_163 = None
    mm_185: "f32[128, 2048]" = torch.ops.aten.mm.default(view_764, permute_570);  permute_570 = None
    permute_571: "f32[8192, 128]" = torch.ops.aten.permute.default(view_764, [1, 0])
    mm_186: "f32[8192, 2048]" = torch.ops.aten.mm.default(permute_571, view_328);  permute_571 = view_328 = None
    permute_572: "f32[2048, 8192]" = torch.ops.aten.permute.default(mm_186, [1, 0]);  mm_186 = None
    sum_138: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_764, [0], True);  view_764 = None
    view_765: "f32[8192]" = torch.ops.aten.view.default(sum_138, [8192]);  sum_138 = None
    permute_573: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_572, [1, 0]);  permute_572 = None
    view_766: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_185, [1, 128, 2048]);  mm_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    sub_150: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_115, getitem_59);  add_115 = getitem_59 = None
    mul_435: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_150, rsqrt_29);  sub_150 = None
    mul_436: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_766, primals_192);  primals_192 = None
    mul_437: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_436, 2048)
    sum_139: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_436, [2], True)
    mul_438: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_436, mul_435);  mul_436 = None
    sum_140: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_438, [2], True);  mul_438 = None
    mul_439: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_435, sum_140);  sum_140 = None
    sub_151: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_437, sum_139);  mul_437 = sum_139 = None
    sub_152: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_151, mul_439);  sub_151 = mul_439 = None
    div_43: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_29, 2048);  rsqrt_29 = None
    mul_440: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_43, sub_152);  div_43 = sub_152 = None
    mul_441: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_766, mul_435);  mul_435 = None
    sum_141: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_441, [0, 1]);  mul_441 = None
    sum_142: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_766, [0, 1]);  view_766 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    add_270: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_267, mul_440);  add_267 = mul_440 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_767: "f32[128, 2048]" = torch.ops.aten.view.default(add_270, [128, 2048])
    permute_574: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_162, [1, 0]);  permute_162 = None
    mm_187: "f32[128, 2048]" = torch.ops.aten.mm.default(view_767, permute_574);  permute_574 = None
    permute_575: "f32[2048, 128]" = torch.ops.aten.permute.default(view_767, [1, 0])
    mm_188: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_575, view_326);  permute_575 = view_326 = None
    permute_576: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_188, [1, 0]);  mm_188 = None
    sum_143: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_767, [0], True);  view_767 = None
    view_768: "f32[2048]" = torch.ops.aten.view.default(sum_143, [2048]);  sum_143 = None
    permute_577: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_576, [1, 0]);  permute_576 = None
    view_769: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_187, [1, 128, 2048]);  mm_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_770: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_769, [1, 128, 16, 128]);  view_769 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_578: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_770, [0, 2, 1, 3]);  view_770 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_771: "f32[16, 128, 128]" = torch.ops.aten.view.default(permute_578, [16, 128, 128]);  permute_578 = None
    permute_579: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_322, [0, 2, 1]);  view_322 = None
    bmm_84: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_579, view_771);  permute_579 = None
    permute_580: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_323, [0, 2, 1]);  view_323 = None
    bmm_85: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_771, permute_580);  view_771 = permute_580 = None
    view_772: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_84, [1, 16, 128, 128]);  bmm_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    add_271: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_31, view_772);  tangents_31 = view_772 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_773: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_85, [1, 16, 128, 128]);  bmm_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_67: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(alias_28);  alias_28 = None
    mul_442: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_773, alias_67);  view_773 = None
    sum_144: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_442, [-1], True)
    mul_443: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_67, sum_144);  alias_67 = sum_144 = None
    sub_153: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_442, mul_443);  mul_442 = mul_443 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    scalar_tensor_9: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_33: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_60, sub_153, scalar_tensor_9);  slice_60 = sub_153 = scalar_tensor_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_774: "f32[16, 128, 128]" = torch.ops.aten.view.default(where_33, [16, 128, 128]);  where_33 = None
    permute_581: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_319, [0, 2, 1]);  view_319 = None
    bmm_86: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_581, view_774);  permute_581 = None
    permute_582: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_320, [0, 2, 1]);  view_320 = None
    bmm_87: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_774, permute_582);  view_774 = permute_582 = None
    view_775: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_86, [1, 16, 128, 128]);  bmm_86 = None
    view_776: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_87, [1, 16, 128, 128]);  bmm_87 = None
    permute_583: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_775, [0, 1, 3, 2]);  view_775 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_272: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_30, permute_583);  tangents_30 = permute_583 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_584: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_271, [0, 2, 1, 3]);  add_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_124: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_584, memory_format = torch.contiguous_format);  permute_584 = None
    view_777: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_124, [1, 128, 2048]);  clone_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_585: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_272, [0, 2, 1, 3]);  add_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_125: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_585, memory_format = torch.contiguous_format);  permute_585 = None
    view_778: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_125, [1, 128, 2048]);  clone_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_586: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_776, [0, 2, 1, 3]);  view_776 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_126: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_586, memory_format = torch.contiguous_format);  permute_586 = None
    view_779: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_126, [1, 128, 2048]);  clone_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    view_780: "f32[128, 2048]" = torch.ops.aten.view.default(view_777, [128, 2048]);  view_777 = None
    permute_587: "f32[2048, 128]" = torch.ops.aten.permute.default(view_780, [1, 0])
    mm_189: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_587, view_314);  permute_587 = view_314 = None
    permute_588: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_189, [1, 0]);  mm_189 = None
    permute_589: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_156, [1, 0]);  permute_156 = None
    mm_190: "f32[128, 2048]" = torch.ops.aten.mm.default(view_780, permute_589);  view_780 = permute_589 = None
    view_781: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_190, [1, 128, 2048]);  mm_190 = None
    permute_590: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_588, [1, 0]);  permute_588 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    view_782: "f32[128, 2048]" = torch.ops.aten.view.default(view_778, [128, 2048]);  view_778 = None
    permute_591: "f32[2048, 128]" = torch.ops.aten.permute.default(view_782, [1, 0])
    mm_191: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_591, view_312);  permute_591 = view_312 = None
    permute_592: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_191, [1, 0]);  mm_191 = None
    permute_593: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_155, [1, 0]);  permute_155 = None
    mm_192: "f32[128, 2048]" = torch.ops.aten.mm.default(view_782, permute_593);  view_782 = permute_593 = None
    view_783: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_192, [1, 128, 2048]);  mm_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    add_273: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_781, view_783);  view_781 = view_783 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_594: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_592, [1, 0]);  permute_592 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    view_784: "f32[128, 2048]" = torch.ops.aten.view.default(view_779, [128, 2048]);  view_779 = None
    permute_595: "f32[2048, 128]" = torch.ops.aten.permute.default(view_784, [1, 0])
    mm_193: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_595, view_310);  permute_595 = view_310 = None
    permute_596: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_193, [1, 0]);  mm_193 = None
    permute_597: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_154, [1, 0]);  permute_154 = None
    mm_194: "f32[128, 2048]" = torch.ops.aten.mm.default(view_784, permute_597);  view_784 = permute_597 = None
    view_785: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_194, [1, 128, 2048]);  mm_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    add_274: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_273, view_785);  add_273 = view_785 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_598: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_596, [1, 0]);  permute_596 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    sub_154: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_112, getitem_57);  add_112 = getitem_57 = None
    mul_444: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_154, rsqrt_28);  sub_154 = None
    mul_445: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_274, primals_185);  primals_185 = None
    mul_446: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_445, 2048)
    sum_145: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_445, [2], True)
    mul_447: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_445, mul_444);  mul_445 = None
    sum_146: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_447, [2], True);  mul_447 = None
    mul_448: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_444, sum_146);  sum_146 = None
    sub_155: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_446, sum_145);  mul_446 = sum_145 = None
    sub_156: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_155, mul_448);  sub_155 = mul_448 = None
    div_44: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_28, 2048);  rsqrt_28 = None
    mul_449: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_44, sub_156);  div_44 = sub_156 = None
    mul_450: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_274, mul_444);  mul_444 = None
    sum_147: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_450, [0, 1]);  mul_450 = None
    sum_148: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_274, [0, 1]);  add_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    add_275: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_270, mul_449);  add_270 = mul_449 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_786: "f32[128, 2048]" = torch.ops.aten.view.default(add_275, [128, 2048])
    permute_599: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_153, [1, 0]);  permute_153 = None
    mm_195: "f32[128, 8192]" = torch.ops.aten.mm.default(view_786, permute_599);  permute_599 = None
    permute_600: "f32[2048, 128]" = torch.ops.aten.permute.default(view_786, [1, 0])
    mm_196: "f32[2048, 8192]" = torch.ops.aten.mm.default(permute_600, view_308);  permute_600 = view_308 = None
    permute_601: "f32[8192, 2048]" = torch.ops.aten.permute.default(mm_196, [1, 0]);  mm_196 = None
    sum_149: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_786, [0], True);  view_786 = None
    view_787: "f32[2048]" = torch.ops.aten.view.default(sum_149, [2048]);  sum_149 = None
    permute_602: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_601, [1, 0]);  permute_601 = None
    view_788: "f32[1, 128, 8192]" = torch.ops.aten.view.default(mm_195, [1, 128, 8192]);  mm_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_451: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_788, mul_108);  mul_108 = None
    mul_452: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_788, add_111);  view_788 = add_111 = None
    alias_68: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(alias_27);  alias_27 = None
    mul_453: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(alias_68, alias_68);  alias_68 = None
    sub_157: "f32[1, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_453);  mul_453 = None
    mul_454: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_451, sub_157);  mul_451 = sub_157 = None
    mul_455: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_454, 0.7978845608028654);  mul_454 = None
    mul_456: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_455, 0.044715)
    pow_35: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_307, 2.0);  view_307 = None
    mul_457: "f32[1, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_35, 3.0);  pow_35 = None
    mul_458: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_456, mul_457);  mul_456 = mul_457 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_276: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(mul_455, mul_458);  mul_455 = mul_458 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_459: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_452, 0.5);  mul_452 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_277: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(add_276, mul_459);  add_276 = mul_459 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_789: "f32[128, 8192]" = torch.ops.aten.view.default(add_277, [128, 8192]);  add_277 = None
    permute_603: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_152, [1, 0]);  permute_152 = None
    mm_197: "f32[128, 2048]" = torch.ops.aten.mm.default(view_789, permute_603);  permute_603 = None
    permute_604: "f32[8192, 128]" = torch.ops.aten.permute.default(view_789, [1, 0])
    mm_198: "f32[8192, 2048]" = torch.ops.aten.mm.default(permute_604, view_306);  permute_604 = view_306 = None
    permute_605: "f32[2048, 8192]" = torch.ops.aten.permute.default(mm_198, [1, 0]);  mm_198 = None
    sum_150: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_789, [0], True);  view_789 = None
    view_790: "f32[8192]" = torch.ops.aten.view.default(sum_150, [8192]);  sum_150 = None
    permute_606: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_605, [1, 0]);  permute_605 = None
    view_791: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_197, [1, 128, 2048]);  mm_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    sub_158: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_107, getitem_55);  add_107 = getitem_55 = None
    mul_460: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_158, rsqrt_27);  sub_158 = None
    mul_461: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_791, primals_179);  primals_179 = None
    mul_462: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_461, 2048)
    sum_151: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_461, [2], True)
    mul_463: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_461, mul_460);  mul_461 = None
    sum_152: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_463, [2], True);  mul_463 = None
    mul_464: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_460, sum_152);  sum_152 = None
    sub_159: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_462, sum_151);  mul_462 = sum_151 = None
    sub_160: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_159, mul_464);  sub_159 = mul_464 = None
    div_45: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_27, 2048);  rsqrt_27 = None
    mul_465: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_45, sub_160);  div_45 = sub_160 = None
    mul_466: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_791, mul_460);  mul_460 = None
    sum_153: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_466, [0, 1]);  mul_466 = None
    sum_154: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_791, [0, 1]);  view_791 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    add_278: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_275, mul_465);  add_275 = mul_465 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_792: "f32[128, 2048]" = torch.ops.aten.view.default(add_278, [128, 2048])
    permute_607: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_151, [1, 0]);  permute_151 = None
    mm_199: "f32[128, 2048]" = torch.ops.aten.mm.default(view_792, permute_607);  permute_607 = None
    permute_608: "f32[2048, 128]" = torch.ops.aten.permute.default(view_792, [1, 0])
    mm_200: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_608, view_304);  permute_608 = view_304 = None
    permute_609: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_200, [1, 0]);  mm_200 = None
    sum_155: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_792, [0], True);  view_792 = None
    view_793: "f32[2048]" = torch.ops.aten.view.default(sum_155, [2048]);  sum_155 = None
    permute_610: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_609, [1, 0]);  permute_609 = None
    view_794: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_199, [1, 128, 2048]);  mm_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_795: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_794, [1, 128, 16, 128]);  view_794 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_611: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_795, [0, 2, 1, 3]);  view_795 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_796: "f32[16, 128, 128]" = torch.ops.aten.view.default(permute_611, [16, 128, 128]);  permute_611 = None
    permute_612: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_300, [0, 2, 1]);  view_300 = None
    bmm_88: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_612, view_796);  permute_612 = None
    permute_613: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_301, [0, 2, 1]);  view_301 = None
    bmm_89: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_796, permute_613);  view_796 = permute_613 = None
    view_797: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_88, [1, 16, 128, 128]);  bmm_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    add_279: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_29, view_797);  tangents_29 = view_797 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_798: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_89, [1, 16, 128, 128]);  bmm_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_69: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(alias_26);  alias_26 = None
    mul_467: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_798, alias_69);  view_798 = None
    sum_156: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_467, [-1], True)
    mul_468: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_69, sum_156);  alias_69 = sum_156 = None
    sub_161: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_467, mul_468);  mul_467 = mul_468 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    scalar_tensor_10: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_34: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_56, sub_161, scalar_tensor_10);  slice_56 = sub_161 = scalar_tensor_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_799: "f32[16, 128, 128]" = torch.ops.aten.view.default(where_34, [16, 128, 128]);  where_34 = None
    permute_614: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_297, [0, 2, 1]);  view_297 = None
    bmm_90: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_614, view_799);  permute_614 = None
    permute_615: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_298, [0, 2, 1]);  view_298 = None
    bmm_91: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_799, permute_615);  view_799 = permute_615 = None
    view_800: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_90, [1, 16, 128, 128]);  bmm_90 = None
    view_801: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_91, [1, 16, 128, 128]);  bmm_91 = None
    permute_616: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_800, [0, 1, 3, 2]);  view_800 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_280: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_28, permute_616);  tangents_28 = permute_616 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_617: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_279, [0, 2, 1, 3]);  add_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_127: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_617, memory_format = torch.contiguous_format);  permute_617 = None
    view_802: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_127, [1, 128, 2048]);  clone_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_618: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_280, [0, 2, 1, 3]);  add_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_128: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_618, memory_format = torch.contiguous_format);  permute_618 = None
    view_803: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_128, [1, 128, 2048]);  clone_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_619: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_801, [0, 2, 1, 3]);  view_801 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_129: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_619, memory_format = torch.contiguous_format);  permute_619 = None
    view_804: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_129, [1, 128, 2048]);  clone_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    view_805: "f32[128, 2048]" = torch.ops.aten.view.default(view_802, [128, 2048]);  view_802 = None
    permute_620: "f32[2048, 128]" = torch.ops.aten.permute.default(view_805, [1, 0])
    mm_201: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_620, view_292);  permute_620 = view_292 = None
    permute_621: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_201, [1, 0]);  mm_201 = None
    permute_622: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_145, [1, 0]);  permute_145 = None
    mm_202: "f32[128, 2048]" = torch.ops.aten.mm.default(view_805, permute_622);  view_805 = permute_622 = None
    view_806: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_202, [1, 128, 2048]);  mm_202 = None
    permute_623: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_621, [1, 0]);  permute_621 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    view_807: "f32[128, 2048]" = torch.ops.aten.view.default(view_803, [128, 2048]);  view_803 = None
    permute_624: "f32[2048, 128]" = torch.ops.aten.permute.default(view_807, [1, 0])
    mm_203: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_624, view_290);  permute_624 = view_290 = None
    permute_625: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_203, [1, 0]);  mm_203 = None
    permute_626: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_144, [1, 0]);  permute_144 = None
    mm_204: "f32[128, 2048]" = torch.ops.aten.mm.default(view_807, permute_626);  view_807 = permute_626 = None
    view_808: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_204, [1, 128, 2048]);  mm_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    add_281: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_806, view_808);  view_806 = view_808 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_627: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_625, [1, 0]);  permute_625 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    view_809: "f32[128, 2048]" = torch.ops.aten.view.default(view_804, [128, 2048]);  view_804 = None
    permute_628: "f32[2048, 128]" = torch.ops.aten.permute.default(view_809, [1, 0])
    mm_205: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_628, view_288);  permute_628 = view_288 = None
    permute_629: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_205, [1, 0]);  mm_205 = None
    permute_630: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_143, [1, 0]);  permute_143 = None
    mm_206: "f32[128, 2048]" = torch.ops.aten.mm.default(view_809, permute_630);  view_809 = permute_630 = None
    view_810: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_206, [1, 128, 2048]);  mm_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    add_282: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_281, view_810);  add_281 = view_810 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_631: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_629, [1, 0]);  permute_629 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    sub_162: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_104, getitem_53);  add_104 = getitem_53 = None
    mul_469: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_162, rsqrt_26);  sub_162 = None
    mul_470: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_282, primals_172);  primals_172 = None
    mul_471: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_470, 2048)
    sum_157: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_470, [2], True)
    mul_472: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_470, mul_469);  mul_470 = None
    sum_158: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_472, [2], True);  mul_472 = None
    mul_473: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_469, sum_158);  sum_158 = None
    sub_163: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_471, sum_157);  mul_471 = sum_157 = None
    sub_164: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_163, mul_473);  sub_163 = mul_473 = None
    div_46: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_26, 2048);  rsqrt_26 = None
    mul_474: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_46, sub_164);  div_46 = sub_164 = None
    mul_475: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_282, mul_469);  mul_469 = None
    sum_159: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_475, [0, 1]);  mul_475 = None
    sum_160: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_282, [0, 1]);  add_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    add_283: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_278, mul_474);  add_278 = mul_474 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_811: "f32[128, 2048]" = torch.ops.aten.view.default(add_283, [128, 2048])
    permute_632: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_142, [1, 0]);  permute_142 = None
    mm_207: "f32[128, 8192]" = torch.ops.aten.mm.default(view_811, permute_632);  permute_632 = None
    permute_633: "f32[2048, 128]" = torch.ops.aten.permute.default(view_811, [1, 0])
    mm_208: "f32[2048, 8192]" = torch.ops.aten.mm.default(permute_633, view_286);  permute_633 = view_286 = None
    permute_634: "f32[8192, 2048]" = torch.ops.aten.permute.default(mm_208, [1, 0]);  mm_208 = None
    sum_161: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_811, [0], True);  view_811 = None
    view_812: "f32[2048]" = torch.ops.aten.view.default(sum_161, [2048]);  sum_161 = None
    permute_635: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_634, [1, 0]);  permute_634 = None
    view_813: "f32[1, 128, 8192]" = torch.ops.aten.view.default(mm_207, [1, 128, 8192]);  mm_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_476: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_813, mul_100);  mul_100 = None
    mul_477: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_813, add_103);  view_813 = add_103 = None
    alias_70: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(alias_25);  alias_25 = None
    mul_478: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(alias_70, alias_70);  alias_70 = None
    sub_165: "f32[1, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_478);  mul_478 = None
    mul_479: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_476, sub_165);  mul_476 = sub_165 = None
    mul_480: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_479, 0.7978845608028654);  mul_479 = None
    mul_481: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_480, 0.044715)
    pow_36: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_285, 2.0);  view_285 = None
    mul_482: "f32[1, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_36, 3.0);  pow_36 = None
    mul_483: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_481, mul_482);  mul_481 = mul_482 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_284: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(mul_480, mul_483);  mul_480 = mul_483 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_484: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_477, 0.5);  mul_477 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_285: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(add_284, mul_484);  add_284 = mul_484 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_814: "f32[128, 8192]" = torch.ops.aten.view.default(add_285, [128, 8192]);  add_285 = None
    permute_636: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_141, [1, 0]);  permute_141 = None
    mm_209: "f32[128, 2048]" = torch.ops.aten.mm.default(view_814, permute_636);  permute_636 = None
    permute_637: "f32[8192, 128]" = torch.ops.aten.permute.default(view_814, [1, 0])
    mm_210: "f32[8192, 2048]" = torch.ops.aten.mm.default(permute_637, view_284);  permute_637 = view_284 = None
    permute_638: "f32[2048, 8192]" = torch.ops.aten.permute.default(mm_210, [1, 0]);  mm_210 = None
    sum_162: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_814, [0], True);  view_814 = None
    view_815: "f32[8192]" = torch.ops.aten.view.default(sum_162, [8192]);  sum_162 = None
    permute_639: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_638, [1, 0]);  permute_638 = None
    view_816: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_209, [1, 128, 2048]);  mm_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    sub_166: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_99, getitem_51);  add_99 = getitem_51 = None
    mul_485: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_166, rsqrt_25);  sub_166 = None
    mul_486: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_816, primals_166);  primals_166 = None
    mul_487: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_486, 2048)
    sum_163: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_486, [2], True)
    mul_488: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_486, mul_485);  mul_486 = None
    sum_164: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_488, [2], True);  mul_488 = None
    mul_489: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_485, sum_164);  sum_164 = None
    sub_167: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_487, sum_163);  mul_487 = sum_163 = None
    sub_168: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_167, mul_489);  sub_167 = mul_489 = None
    div_47: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_25, 2048);  rsqrt_25 = None
    mul_490: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_47, sub_168);  div_47 = sub_168 = None
    mul_491: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_816, mul_485);  mul_485 = None
    sum_165: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_491, [0, 1]);  mul_491 = None
    sum_166: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_816, [0, 1]);  view_816 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    add_286: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_283, mul_490);  add_283 = mul_490 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_817: "f32[128, 2048]" = torch.ops.aten.view.default(add_286, [128, 2048])
    permute_640: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_140, [1, 0]);  permute_140 = None
    mm_211: "f32[128, 2048]" = torch.ops.aten.mm.default(view_817, permute_640);  permute_640 = None
    permute_641: "f32[2048, 128]" = torch.ops.aten.permute.default(view_817, [1, 0])
    mm_212: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_641, view_282);  permute_641 = view_282 = None
    permute_642: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_212, [1, 0]);  mm_212 = None
    sum_167: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_817, [0], True);  view_817 = None
    view_818: "f32[2048]" = torch.ops.aten.view.default(sum_167, [2048]);  sum_167 = None
    permute_643: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_642, [1, 0]);  permute_642 = None
    view_819: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_211, [1, 128, 2048]);  mm_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_820: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_819, [1, 128, 16, 128]);  view_819 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_644: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_820, [0, 2, 1, 3]);  view_820 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_821: "f32[16, 128, 128]" = torch.ops.aten.view.default(permute_644, [16, 128, 128]);  permute_644 = None
    permute_645: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_278, [0, 2, 1]);  view_278 = None
    bmm_92: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_645, view_821);  permute_645 = None
    permute_646: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_279, [0, 2, 1]);  view_279 = None
    bmm_93: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_821, permute_646);  view_821 = permute_646 = None
    view_822: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_92, [1, 16, 128, 128]);  bmm_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    add_287: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_27, view_822);  tangents_27 = view_822 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_823: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_93, [1, 16, 128, 128]);  bmm_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_71: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(alias_24);  alias_24 = None
    mul_492: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_823, alias_71);  view_823 = None
    sum_168: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_492, [-1], True)
    mul_493: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_71, sum_168);  alias_71 = sum_168 = None
    sub_169: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_492, mul_493);  mul_492 = mul_493 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    scalar_tensor_11: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_35: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_52, sub_169, scalar_tensor_11);  slice_52 = sub_169 = scalar_tensor_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_824: "f32[16, 128, 128]" = torch.ops.aten.view.default(where_35, [16, 128, 128]);  where_35 = None
    permute_647: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_275, [0, 2, 1]);  view_275 = None
    bmm_94: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_647, view_824);  permute_647 = None
    permute_648: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_276, [0, 2, 1]);  view_276 = None
    bmm_95: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_824, permute_648);  view_824 = permute_648 = None
    view_825: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_94, [1, 16, 128, 128]);  bmm_94 = None
    view_826: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_95, [1, 16, 128, 128]);  bmm_95 = None
    permute_649: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_825, [0, 1, 3, 2]);  view_825 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_288: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_26, permute_649);  tangents_26 = permute_649 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_650: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_287, [0, 2, 1, 3]);  add_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_130: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_650, memory_format = torch.contiguous_format);  permute_650 = None
    view_827: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_130, [1, 128, 2048]);  clone_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_651: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_288, [0, 2, 1, 3]);  add_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_131: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_651, memory_format = torch.contiguous_format);  permute_651 = None
    view_828: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_131, [1, 128, 2048]);  clone_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_652: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_826, [0, 2, 1, 3]);  view_826 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_132: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_652, memory_format = torch.contiguous_format);  permute_652 = None
    view_829: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_132, [1, 128, 2048]);  clone_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    view_830: "f32[128, 2048]" = torch.ops.aten.view.default(view_827, [128, 2048]);  view_827 = None
    permute_653: "f32[2048, 128]" = torch.ops.aten.permute.default(view_830, [1, 0])
    mm_213: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_653, view_270);  permute_653 = view_270 = None
    permute_654: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_213, [1, 0]);  mm_213 = None
    permute_655: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_134, [1, 0]);  permute_134 = None
    mm_214: "f32[128, 2048]" = torch.ops.aten.mm.default(view_830, permute_655);  view_830 = permute_655 = None
    view_831: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_214, [1, 128, 2048]);  mm_214 = None
    permute_656: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_654, [1, 0]);  permute_654 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    view_832: "f32[128, 2048]" = torch.ops.aten.view.default(view_828, [128, 2048]);  view_828 = None
    permute_657: "f32[2048, 128]" = torch.ops.aten.permute.default(view_832, [1, 0])
    mm_215: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_657, view_268);  permute_657 = view_268 = None
    permute_658: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_215, [1, 0]);  mm_215 = None
    permute_659: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_133, [1, 0]);  permute_133 = None
    mm_216: "f32[128, 2048]" = torch.ops.aten.mm.default(view_832, permute_659);  view_832 = permute_659 = None
    view_833: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_216, [1, 128, 2048]);  mm_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    add_289: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_831, view_833);  view_831 = view_833 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_660: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_658, [1, 0]);  permute_658 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    view_834: "f32[128, 2048]" = torch.ops.aten.view.default(view_829, [128, 2048]);  view_829 = None
    permute_661: "f32[2048, 128]" = torch.ops.aten.permute.default(view_834, [1, 0])
    mm_217: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_661, view_266);  permute_661 = view_266 = None
    permute_662: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_217, [1, 0]);  mm_217 = None
    permute_663: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_132, [1, 0]);  permute_132 = None
    mm_218: "f32[128, 2048]" = torch.ops.aten.mm.default(view_834, permute_663);  view_834 = permute_663 = None
    view_835: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_218, [1, 128, 2048]);  mm_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    add_290: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_289, view_835);  add_289 = view_835 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_664: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_662, [1, 0]);  permute_662 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    sub_170: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_96, getitem_49);  add_96 = getitem_49 = None
    mul_494: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_170, rsqrt_24);  sub_170 = None
    mul_495: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_290, primals_159);  primals_159 = None
    mul_496: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_495, 2048)
    sum_169: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_495, [2], True)
    mul_497: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_495, mul_494);  mul_495 = None
    sum_170: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_497, [2], True);  mul_497 = None
    mul_498: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_494, sum_170);  sum_170 = None
    sub_171: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_496, sum_169);  mul_496 = sum_169 = None
    sub_172: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_171, mul_498);  sub_171 = mul_498 = None
    div_48: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 2048);  rsqrt_24 = None
    mul_499: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_48, sub_172);  div_48 = sub_172 = None
    mul_500: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_290, mul_494);  mul_494 = None
    sum_171: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_500, [0, 1]);  mul_500 = None
    sum_172: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_290, [0, 1]);  add_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    add_291: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_286, mul_499);  add_286 = mul_499 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_836: "f32[128, 2048]" = torch.ops.aten.view.default(add_291, [128, 2048])
    permute_665: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_131, [1, 0]);  permute_131 = None
    mm_219: "f32[128, 8192]" = torch.ops.aten.mm.default(view_836, permute_665);  permute_665 = None
    permute_666: "f32[2048, 128]" = torch.ops.aten.permute.default(view_836, [1, 0])
    mm_220: "f32[2048, 8192]" = torch.ops.aten.mm.default(permute_666, view_264);  permute_666 = view_264 = None
    permute_667: "f32[8192, 2048]" = torch.ops.aten.permute.default(mm_220, [1, 0]);  mm_220 = None
    sum_173: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_836, [0], True);  view_836 = None
    view_837: "f32[2048]" = torch.ops.aten.view.default(sum_173, [2048]);  sum_173 = None
    permute_668: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_667, [1, 0]);  permute_667 = None
    view_838: "f32[1, 128, 8192]" = torch.ops.aten.view.default(mm_219, [1, 128, 8192]);  mm_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_501: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_838, mul_92);  mul_92 = None
    mul_502: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_838, add_95);  view_838 = add_95 = None
    alias_72: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(alias_23);  alias_23 = None
    mul_503: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(alias_72, alias_72);  alias_72 = None
    sub_173: "f32[1, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_503);  mul_503 = None
    mul_504: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_501, sub_173);  mul_501 = sub_173 = None
    mul_505: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_504, 0.7978845608028654);  mul_504 = None
    mul_506: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_505, 0.044715)
    pow_37: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_263, 2.0);  view_263 = None
    mul_507: "f32[1, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_37, 3.0);  pow_37 = None
    mul_508: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_506, mul_507);  mul_506 = mul_507 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_292: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(mul_505, mul_508);  mul_505 = mul_508 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_509: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_502, 0.5);  mul_502 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_293: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(add_292, mul_509);  add_292 = mul_509 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_839: "f32[128, 8192]" = torch.ops.aten.view.default(add_293, [128, 8192]);  add_293 = None
    permute_669: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_130, [1, 0]);  permute_130 = None
    mm_221: "f32[128, 2048]" = torch.ops.aten.mm.default(view_839, permute_669);  permute_669 = None
    permute_670: "f32[8192, 128]" = torch.ops.aten.permute.default(view_839, [1, 0])
    mm_222: "f32[8192, 2048]" = torch.ops.aten.mm.default(permute_670, view_262);  permute_670 = view_262 = None
    permute_671: "f32[2048, 8192]" = torch.ops.aten.permute.default(mm_222, [1, 0]);  mm_222 = None
    sum_174: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_839, [0], True);  view_839 = None
    view_840: "f32[8192]" = torch.ops.aten.view.default(sum_174, [8192]);  sum_174 = None
    permute_672: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_671, [1, 0]);  permute_671 = None
    view_841: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_221, [1, 128, 2048]);  mm_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    sub_174: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_91, getitem_47);  add_91 = getitem_47 = None
    mul_510: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_174, rsqrt_23);  sub_174 = None
    mul_511: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_841, primals_153);  primals_153 = None
    mul_512: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_511, 2048)
    sum_175: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_511, [2], True)
    mul_513: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_511, mul_510);  mul_511 = None
    sum_176: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_513, [2], True);  mul_513 = None
    mul_514: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_510, sum_176);  sum_176 = None
    sub_175: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_512, sum_175);  mul_512 = sum_175 = None
    sub_176: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_175, mul_514);  sub_175 = mul_514 = None
    div_49: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 2048);  rsqrt_23 = None
    mul_515: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_49, sub_176);  div_49 = sub_176 = None
    mul_516: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_841, mul_510);  mul_510 = None
    sum_177: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_516, [0, 1]);  mul_516 = None
    sum_178: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_841, [0, 1]);  view_841 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    add_294: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_291, mul_515);  add_291 = mul_515 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_842: "f32[128, 2048]" = torch.ops.aten.view.default(add_294, [128, 2048])
    permute_673: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_129, [1, 0]);  permute_129 = None
    mm_223: "f32[128, 2048]" = torch.ops.aten.mm.default(view_842, permute_673);  permute_673 = None
    permute_674: "f32[2048, 128]" = torch.ops.aten.permute.default(view_842, [1, 0])
    mm_224: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_674, view_260);  permute_674 = view_260 = None
    permute_675: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_224, [1, 0]);  mm_224 = None
    sum_179: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_842, [0], True);  view_842 = None
    view_843: "f32[2048]" = torch.ops.aten.view.default(sum_179, [2048]);  sum_179 = None
    permute_676: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_675, [1, 0]);  permute_675 = None
    view_844: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_223, [1, 128, 2048]);  mm_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_845: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_844, [1, 128, 16, 128]);  view_844 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_677: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_845, [0, 2, 1, 3]);  view_845 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_846: "f32[16, 128, 128]" = torch.ops.aten.view.default(permute_677, [16, 128, 128]);  permute_677 = None
    permute_678: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_256, [0, 2, 1]);  view_256 = None
    bmm_96: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_678, view_846);  permute_678 = None
    permute_679: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_257, [0, 2, 1]);  view_257 = None
    bmm_97: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_846, permute_679);  view_846 = permute_679 = None
    view_847: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_96, [1, 16, 128, 128]);  bmm_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    add_295: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_25, view_847);  tangents_25 = view_847 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_848: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_97, [1, 16, 128, 128]);  bmm_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_73: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(alias_22);  alias_22 = None
    mul_517: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_848, alias_73);  view_848 = None
    sum_180: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_517, [-1], True)
    mul_518: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_73, sum_180);  alias_73 = sum_180 = None
    sub_177: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_517, mul_518);  mul_517 = mul_518 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    scalar_tensor_12: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_36: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_48, sub_177, scalar_tensor_12);  slice_48 = sub_177 = scalar_tensor_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_849: "f32[16, 128, 128]" = torch.ops.aten.view.default(where_36, [16, 128, 128]);  where_36 = None
    permute_680: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_253, [0, 2, 1]);  view_253 = None
    bmm_98: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_680, view_849);  permute_680 = None
    permute_681: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_254, [0, 2, 1]);  view_254 = None
    bmm_99: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_849, permute_681);  view_849 = permute_681 = None
    view_850: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_98, [1, 16, 128, 128]);  bmm_98 = None
    view_851: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_99, [1, 16, 128, 128]);  bmm_99 = None
    permute_682: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_850, [0, 1, 3, 2]);  view_850 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_296: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_24, permute_682);  tangents_24 = permute_682 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_683: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_295, [0, 2, 1, 3]);  add_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_133: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_683, memory_format = torch.contiguous_format);  permute_683 = None
    view_852: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_133, [1, 128, 2048]);  clone_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_684: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_296, [0, 2, 1, 3]);  add_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_134: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_684, memory_format = torch.contiguous_format);  permute_684 = None
    view_853: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_134, [1, 128, 2048]);  clone_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_685: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_851, [0, 2, 1, 3]);  view_851 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_135: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_685, memory_format = torch.contiguous_format);  permute_685 = None
    view_854: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_135, [1, 128, 2048]);  clone_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    view_855: "f32[128, 2048]" = torch.ops.aten.view.default(view_852, [128, 2048]);  view_852 = None
    permute_686: "f32[2048, 128]" = torch.ops.aten.permute.default(view_855, [1, 0])
    mm_225: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_686, view_248);  permute_686 = view_248 = None
    permute_687: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_225, [1, 0]);  mm_225 = None
    permute_688: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_123, [1, 0]);  permute_123 = None
    mm_226: "f32[128, 2048]" = torch.ops.aten.mm.default(view_855, permute_688);  view_855 = permute_688 = None
    view_856: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_226, [1, 128, 2048]);  mm_226 = None
    permute_689: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_687, [1, 0]);  permute_687 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    view_857: "f32[128, 2048]" = torch.ops.aten.view.default(view_853, [128, 2048]);  view_853 = None
    permute_690: "f32[2048, 128]" = torch.ops.aten.permute.default(view_857, [1, 0])
    mm_227: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_690, view_246);  permute_690 = view_246 = None
    permute_691: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_227, [1, 0]);  mm_227 = None
    permute_692: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_122, [1, 0]);  permute_122 = None
    mm_228: "f32[128, 2048]" = torch.ops.aten.mm.default(view_857, permute_692);  view_857 = permute_692 = None
    view_858: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_228, [1, 128, 2048]);  mm_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    add_297: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_856, view_858);  view_856 = view_858 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_693: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_691, [1, 0]);  permute_691 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    view_859: "f32[128, 2048]" = torch.ops.aten.view.default(view_854, [128, 2048]);  view_854 = None
    permute_694: "f32[2048, 128]" = torch.ops.aten.permute.default(view_859, [1, 0])
    mm_229: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_694, view_244);  permute_694 = view_244 = None
    permute_695: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_229, [1, 0]);  mm_229 = None
    permute_696: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_121, [1, 0]);  permute_121 = None
    mm_230: "f32[128, 2048]" = torch.ops.aten.mm.default(view_859, permute_696);  view_859 = permute_696 = None
    view_860: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_230, [1, 128, 2048]);  mm_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    add_298: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_297, view_860);  add_297 = view_860 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_697: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_695, [1, 0]);  permute_695 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    sub_178: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_88, getitem_45);  add_88 = getitem_45 = None
    mul_519: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_178, rsqrt_22);  sub_178 = None
    mul_520: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_298, primals_146);  primals_146 = None
    mul_521: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_520, 2048)
    sum_181: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_520, [2], True)
    mul_522: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_520, mul_519);  mul_520 = None
    sum_182: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_522, [2], True);  mul_522 = None
    mul_523: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_519, sum_182);  sum_182 = None
    sub_179: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_521, sum_181);  mul_521 = sum_181 = None
    sub_180: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_179, mul_523);  sub_179 = mul_523 = None
    div_50: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 2048);  rsqrt_22 = None
    mul_524: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_50, sub_180);  div_50 = sub_180 = None
    mul_525: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_298, mul_519);  mul_519 = None
    sum_183: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_525, [0, 1]);  mul_525 = None
    sum_184: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_298, [0, 1]);  add_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    add_299: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_294, mul_524);  add_294 = mul_524 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_861: "f32[128, 2048]" = torch.ops.aten.view.default(add_299, [128, 2048])
    permute_698: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_120, [1, 0]);  permute_120 = None
    mm_231: "f32[128, 8192]" = torch.ops.aten.mm.default(view_861, permute_698);  permute_698 = None
    permute_699: "f32[2048, 128]" = torch.ops.aten.permute.default(view_861, [1, 0])
    mm_232: "f32[2048, 8192]" = torch.ops.aten.mm.default(permute_699, view_242);  permute_699 = view_242 = None
    permute_700: "f32[8192, 2048]" = torch.ops.aten.permute.default(mm_232, [1, 0]);  mm_232 = None
    sum_185: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_861, [0], True);  view_861 = None
    view_862: "f32[2048]" = torch.ops.aten.view.default(sum_185, [2048]);  sum_185 = None
    permute_701: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_700, [1, 0]);  permute_700 = None
    view_863: "f32[1, 128, 8192]" = torch.ops.aten.view.default(mm_231, [1, 128, 8192]);  mm_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_526: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_863, mul_84);  mul_84 = None
    mul_527: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_863, add_87);  view_863 = add_87 = None
    alias_74: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(alias_21);  alias_21 = None
    mul_528: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(alias_74, alias_74);  alias_74 = None
    sub_181: "f32[1, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_528);  mul_528 = None
    mul_529: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_526, sub_181);  mul_526 = sub_181 = None
    mul_530: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_529, 0.7978845608028654);  mul_529 = None
    mul_531: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_530, 0.044715)
    pow_38: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_241, 2.0);  view_241 = None
    mul_532: "f32[1, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_38, 3.0);  pow_38 = None
    mul_533: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_531, mul_532);  mul_531 = mul_532 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_300: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(mul_530, mul_533);  mul_530 = mul_533 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_534: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_527, 0.5);  mul_527 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_301: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(add_300, mul_534);  add_300 = mul_534 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_864: "f32[128, 8192]" = torch.ops.aten.view.default(add_301, [128, 8192]);  add_301 = None
    permute_702: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_119, [1, 0]);  permute_119 = None
    mm_233: "f32[128, 2048]" = torch.ops.aten.mm.default(view_864, permute_702);  permute_702 = None
    permute_703: "f32[8192, 128]" = torch.ops.aten.permute.default(view_864, [1, 0])
    mm_234: "f32[8192, 2048]" = torch.ops.aten.mm.default(permute_703, view_240);  permute_703 = view_240 = None
    permute_704: "f32[2048, 8192]" = torch.ops.aten.permute.default(mm_234, [1, 0]);  mm_234 = None
    sum_186: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_864, [0], True);  view_864 = None
    view_865: "f32[8192]" = torch.ops.aten.view.default(sum_186, [8192]);  sum_186 = None
    permute_705: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_704, [1, 0]);  permute_704 = None
    view_866: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_233, [1, 128, 2048]);  mm_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    sub_182: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_83, getitem_43);  add_83 = getitem_43 = None
    mul_535: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_182, rsqrt_21);  sub_182 = None
    mul_536: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_866, primals_140);  primals_140 = None
    mul_537: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_536, 2048)
    sum_187: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_536, [2], True)
    mul_538: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_536, mul_535);  mul_536 = None
    sum_188: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_538, [2], True);  mul_538 = None
    mul_539: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_535, sum_188);  sum_188 = None
    sub_183: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_537, sum_187);  mul_537 = sum_187 = None
    sub_184: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_183, mul_539);  sub_183 = mul_539 = None
    div_51: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 2048);  rsqrt_21 = None
    mul_540: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_51, sub_184);  div_51 = sub_184 = None
    mul_541: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_866, mul_535);  mul_535 = None
    sum_189: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_541, [0, 1]);  mul_541 = None
    sum_190: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_866, [0, 1]);  view_866 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    add_302: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_299, mul_540);  add_299 = mul_540 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_867: "f32[128, 2048]" = torch.ops.aten.view.default(add_302, [128, 2048])
    permute_706: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_118, [1, 0]);  permute_118 = None
    mm_235: "f32[128, 2048]" = torch.ops.aten.mm.default(view_867, permute_706);  permute_706 = None
    permute_707: "f32[2048, 128]" = torch.ops.aten.permute.default(view_867, [1, 0])
    mm_236: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_707, view_238);  permute_707 = view_238 = None
    permute_708: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_236, [1, 0]);  mm_236 = None
    sum_191: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_867, [0], True);  view_867 = None
    view_868: "f32[2048]" = torch.ops.aten.view.default(sum_191, [2048]);  sum_191 = None
    permute_709: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_708, [1, 0]);  permute_708 = None
    view_869: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_235, [1, 128, 2048]);  mm_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_870: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_869, [1, 128, 16, 128]);  view_869 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_710: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_870, [0, 2, 1, 3]);  view_870 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_871: "f32[16, 128, 128]" = torch.ops.aten.view.default(permute_710, [16, 128, 128]);  permute_710 = None
    permute_711: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_234, [0, 2, 1]);  view_234 = None
    bmm_100: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_711, view_871);  permute_711 = None
    permute_712: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_235, [0, 2, 1]);  view_235 = None
    bmm_101: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_871, permute_712);  view_871 = permute_712 = None
    view_872: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_100, [1, 16, 128, 128]);  bmm_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    add_303: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_23, view_872);  tangents_23 = view_872 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_873: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_101, [1, 16, 128, 128]);  bmm_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_75: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(alias_20);  alias_20 = None
    mul_542: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_873, alias_75);  view_873 = None
    sum_192: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_542, [-1], True)
    mul_543: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_75, sum_192);  alias_75 = sum_192 = None
    sub_185: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_542, mul_543);  mul_542 = mul_543 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    scalar_tensor_13: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_37: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_44, sub_185, scalar_tensor_13);  slice_44 = sub_185 = scalar_tensor_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_874: "f32[16, 128, 128]" = torch.ops.aten.view.default(where_37, [16, 128, 128]);  where_37 = None
    permute_713: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_231, [0, 2, 1]);  view_231 = None
    bmm_102: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_713, view_874);  permute_713 = None
    permute_714: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_232, [0, 2, 1]);  view_232 = None
    bmm_103: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_874, permute_714);  view_874 = permute_714 = None
    view_875: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_102, [1, 16, 128, 128]);  bmm_102 = None
    view_876: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_103, [1, 16, 128, 128]);  bmm_103 = None
    permute_715: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_875, [0, 1, 3, 2]);  view_875 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_304: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_22, permute_715);  tangents_22 = permute_715 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_716: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_303, [0, 2, 1, 3]);  add_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_136: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_716, memory_format = torch.contiguous_format);  permute_716 = None
    view_877: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_136, [1, 128, 2048]);  clone_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_717: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_304, [0, 2, 1, 3]);  add_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_137: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_717, memory_format = torch.contiguous_format);  permute_717 = None
    view_878: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_137, [1, 128, 2048]);  clone_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_718: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_876, [0, 2, 1, 3]);  view_876 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_138: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_718, memory_format = torch.contiguous_format);  permute_718 = None
    view_879: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_138, [1, 128, 2048]);  clone_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    view_880: "f32[128, 2048]" = torch.ops.aten.view.default(view_877, [128, 2048]);  view_877 = None
    permute_719: "f32[2048, 128]" = torch.ops.aten.permute.default(view_880, [1, 0])
    mm_237: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_719, view_226);  permute_719 = view_226 = None
    permute_720: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_237, [1, 0]);  mm_237 = None
    permute_721: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_112, [1, 0]);  permute_112 = None
    mm_238: "f32[128, 2048]" = torch.ops.aten.mm.default(view_880, permute_721);  view_880 = permute_721 = None
    view_881: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_238, [1, 128, 2048]);  mm_238 = None
    permute_722: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_720, [1, 0]);  permute_720 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    view_882: "f32[128, 2048]" = torch.ops.aten.view.default(view_878, [128, 2048]);  view_878 = None
    permute_723: "f32[2048, 128]" = torch.ops.aten.permute.default(view_882, [1, 0])
    mm_239: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_723, view_224);  permute_723 = view_224 = None
    permute_724: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_239, [1, 0]);  mm_239 = None
    permute_725: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_111, [1, 0]);  permute_111 = None
    mm_240: "f32[128, 2048]" = torch.ops.aten.mm.default(view_882, permute_725);  view_882 = permute_725 = None
    view_883: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_240, [1, 128, 2048]);  mm_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    add_305: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_881, view_883);  view_881 = view_883 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_726: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_724, [1, 0]);  permute_724 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    view_884: "f32[128, 2048]" = torch.ops.aten.view.default(view_879, [128, 2048]);  view_879 = None
    permute_727: "f32[2048, 128]" = torch.ops.aten.permute.default(view_884, [1, 0])
    mm_241: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_727, view_222);  permute_727 = view_222 = None
    permute_728: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_241, [1, 0]);  mm_241 = None
    permute_729: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_110, [1, 0]);  permute_110 = None
    mm_242: "f32[128, 2048]" = torch.ops.aten.mm.default(view_884, permute_729);  view_884 = permute_729 = None
    view_885: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_242, [1, 128, 2048]);  mm_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    add_306: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_305, view_885);  add_305 = view_885 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_730: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_728, [1, 0]);  permute_728 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    sub_186: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_80, getitem_41);  add_80 = getitem_41 = None
    mul_544: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_186, rsqrt_20);  sub_186 = None
    mul_545: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_306, primals_133);  primals_133 = None
    mul_546: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_545, 2048)
    sum_193: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_545, [2], True)
    mul_547: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_545, mul_544);  mul_545 = None
    sum_194: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_547, [2], True);  mul_547 = None
    mul_548: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_544, sum_194);  sum_194 = None
    sub_187: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_546, sum_193);  mul_546 = sum_193 = None
    sub_188: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_187, mul_548);  sub_187 = mul_548 = None
    div_52: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 2048);  rsqrt_20 = None
    mul_549: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_52, sub_188);  div_52 = sub_188 = None
    mul_550: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_306, mul_544);  mul_544 = None
    sum_195: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_550, [0, 1]);  mul_550 = None
    sum_196: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_306, [0, 1]);  add_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    add_307: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_302, mul_549);  add_302 = mul_549 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_886: "f32[128, 2048]" = torch.ops.aten.view.default(add_307, [128, 2048])
    permute_731: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_109, [1, 0]);  permute_109 = None
    mm_243: "f32[128, 8192]" = torch.ops.aten.mm.default(view_886, permute_731);  permute_731 = None
    permute_732: "f32[2048, 128]" = torch.ops.aten.permute.default(view_886, [1, 0])
    mm_244: "f32[2048, 8192]" = torch.ops.aten.mm.default(permute_732, view_220);  permute_732 = view_220 = None
    permute_733: "f32[8192, 2048]" = torch.ops.aten.permute.default(mm_244, [1, 0]);  mm_244 = None
    sum_197: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_886, [0], True);  view_886 = None
    view_887: "f32[2048]" = torch.ops.aten.view.default(sum_197, [2048]);  sum_197 = None
    permute_734: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_733, [1, 0]);  permute_733 = None
    view_888: "f32[1, 128, 8192]" = torch.ops.aten.view.default(mm_243, [1, 128, 8192]);  mm_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_551: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_888, mul_76);  mul_76 = None
    mul_552: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_888, add_79);  view_888 = add_79 = None
    alias_76: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(alias_19);  alias_19 = None
    mul_553: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(alias_76, alias_76);  alias_76 = None
    sub_189: "f32[1, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_553);  mul_553 = None
    mul_554: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_551, sub_189);  mul_551 = sub_189 = None
    mul_555: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_554, 0.7978845608028654);  mul_554 = None
    mul_556: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_555, 0.044715)
    pow_39: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_219, 2.0);  view_219 = None
    mul_557: "f32[1, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_39, 3.0);  pow_39 = None
    mul_558: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_556, mul_557);  mul_556 = mul_557 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_308: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(mul_555, mul_558);  mul_555 = mul_558 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_559: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_552, 0.5);  mul_552 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_309: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(add_308, mul_559);  add_308 = mul_559 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_889: "f32[128, 8192]" = torch.ops.aten.view.default(add_309, [128, 8192]);  add_309 = None
    permute_735: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_108, [1, 0]);  permute_108 = None
    mm_245: "f32[128, 2048]" = torch.ops.aten.mm.default(view_889, permute_735);  permute_735 = None
    permute_736: "f32[8192, 128]" = torch.ops.aten.permute.default(view_889, [1, 0])
    mm_246: "f32[8192, 2048]" = torch.ops.aten.mm.default(permute_736, view_218);  permute_736 = view_218 = None
    permute_737: "f32[2048, 8192]" = torch.ops.aten.permute.default(mm_246, [1, 0]);  mm_246 = None
    sum_198: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_889, [0], True);  view_889 = None
    view_890: "f32[8192]" = torch.ops.aten.view.default(sum_198, [8192]);  sum_198 = None
    permute_738: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_737, [1, 0]);  permute_737 = None
    view_891: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_245, [1, 128, 2048]);  mm_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    sub_190: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_75, getitem_39);  add_75 = getitem_39 = None
    mul_560: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_190, rsqrt_19);  sub_190 = None
    mul_561: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_891, primals_127);  primals_127 = None
    mul_562: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_561, 2048)
    sum_199: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_561, [2], True)
    mul_563: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_561, mul_560);  mul_561 = None
    sum_200: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_563, [2], True);  mul_563 = None
    mul_564: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_560, sum_200);  sum_200 = None
    sub_191: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_562, sum_199);  mul_562 = sum_199 = None
    sub_192: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_191, mul_564);  sub_191 = mul_564 = None
    div_53: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 2048);  rsqrt_19 = None
    mul_565: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_53, sub_192);  div_53 = sub_192 = None
    mul_566: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_891, mul_560);  mul_560 = None
    sum_201: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_566, [0, 1]);  mul_566 = None
    sum_202: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_891, [0, 1]);  view_891 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    add_310: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_307, mul_565);  add_307 = mul_565 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_892: "f32[128, 2048]" = torch.ops.aten.view.default(add_310, [128, 2048])
    permute_739: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_107, [1, 0]);  permute_107 = None
    mm_247: "f32[128, 2048]" = torch.ops.aten.mm.default(view_892, permute_739);  permute_739 = None
    permute_740: "f32[2048, 128]" = torch.ops.aten.permute.default(view_892, [1, 0])
    mm_248: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_740, view_216);  permute_740 = view_216 = None
    permute_741: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_248, [1, 0]);  mm_248 = None
    sum_203: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_892, [0], True);  view_892 = None
    view_893: "f32[2048]" = torch.ops.aten.view.default(sum_203, [2048]);  sum_203 = None
    permute_742: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_741, [1, 0]);  permute_741 = None
    view_894: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_247, [1, 128, 2048]);  mm_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_895: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_894, [1, 128, 16, 128]);  view_894 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_743: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_895, [0, 2, 1, 3]);  view_895 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_896: "f32[16, 128, 128]" = torch.ops.aten.view.default(permute_743, [16, 128, 128]);  permute_743 = None
    permute_744: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_212, [0, 2, 1]);  view_212 = None
    bmm_104: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_744, view_896);  permute_744 = None
    permute_745: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_213, [0, 2, 1]);  view_213 = None
    bmm_105: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_896, permute_745);  view_896 = permute_745 = None
    view_897: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_104, [1, 16, 128, 128]);  bmm_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    add_311: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_21, view_897);  tangents_21 = view_897 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_898: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_105, [1, 16, 128, 128]);  bmm_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_77: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(alias_18);  alias_18 = None
    mul_567: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_898, alias_77);  view_898 = None
    sum_204: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_567, [-1], True)
    mul_568: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_77, sum_204);  alias_77 = sum_204 = None
    sub_193: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_567, mul_568);  mul_567 = mul_568 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    scalar_tensor_14: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_38: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_40, sub_193, scalar_tensor_14);  slice_40 = sub_193 = scalar_tensor_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_899: "f32[16, 128, 128]" = torch.ops.aten.view.default(where_38, [16, 128, 128]);  where_38 = None
    permute_746: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_209, [0, 2, 1]);  view_209 = None
    bmm_106: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_746, view_899);  permute_746 = None
    permute_747: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_210, [0, 2, 1]);  view_210 = None
    bmm_107: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_899, permute_747);  view_899 = permute_747 = None
    view_900: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_106, [1, 16, 128, 128]);  bmm_106 = None
    view_901: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_107, [1, 16, 128, 128]);  bmm_107 = None
    permute_748: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_900, [0, 1, 3, 2]);  view_900 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_312: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_20, permute_748);  tangents_20 = permute_748 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_749: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_311, [0, 2, 1, 3]);  add_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_139: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_749, memory_format = torch.contiguous_format);  permute_749 = None
    view_902: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_139, [1, 128, 2048]);  clone_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_750: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_312, [0, 2, 1, 3]);  add_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_140: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_750, memory_format = torch.contiguous_format);  permute_750 = None
    view_903: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_140, [1, 128, 2048]);  clone_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_751: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_901, [0, 2, 1, 3]);  view_901 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_141: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_751, memory_format = torch.contiguous_format);  permute_751 = None
    view_904: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_141, [1, 128, 2048]);  clone_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    view_905: "f32[128, 2048]" = torch.ops.aten.view.default(view_902, [128, 2048]);  view_902 = None
    permute_752: "f32[2048, 128]" = torch.ops.aten.permute.default(view_905, [1, 0])
    mm_249: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_752, view_204);  permute_752 = view_204 = None
    permute_753: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_249, [1, 0]);  mm_249 = None
    permute_754: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_101, [1, 0]);  permute_101 = None
    mm_250: "f32[128, 2048]" = torch.ops.aten.mm.default(view_905, permute_754);  view_905 = permute_754 = None
    view_906: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_250, [1, 128, 2048]);  mm_250 = None
    permute_755: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_753, [1, 0]);  permute_753 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    view_907: "f32[128, 2048]" = torch.ops.aten.view.default(view_903, [128, 2048]);  view_903 = None
    permute_756: "f32[2048, 128]" = torch.ops.aten.permute.default(view_907, [1, 0])
    mm_251: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_756, view_202);  permute_756 = view_202 = None
    permute_757: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_251, [1, 0]);  mm_251 = None
    permute_758: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_100, [1, 0]);  permute_100 = None
    mm_252: "f32[128, 2048]" = torch.ops.aten.mm.default(view_907, permute_758);  view_907 = permute_758 = None
    view_908: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_252, [1, 128, 2048]);  mm_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    add_313: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_906, view_908);  view_906 = view_908 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_759: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_757, [1, 0]);  permute_757 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    view_909: "f32[128, 2048]" = torch.ops.aten.view.default(view_904, [128, 2048]);  view_904 = None
    permute_760: "f32[2048, 128]" = torch.ops.aten.permute.default(view_909, [1, 0])
    mm_253: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_760, view_200);  permute_760 = view_200 = None
    permute_761: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_253, [1, 0]);  mm_253 = None
    permute_762: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_99, [1, 0]);  permute_99 = None
    mm_254: "f32[128, 2048]" = torch.ops.aten.mm.default(view_909, permute_762);  view_909 = permute_762 = None
    view_910: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_254, [1, 128, 2048]);  mm_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    add_314: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_313, view_910);  add_313 = view_910 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_763: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_761, [1, 0]);  permute_761 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    sub_194: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_72, getitem_37);  add_72 = getitem_37 = None
    mul_569: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_194, rsqrt_18);  sub_194 = None
    mul_570: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_314, primals_120);  primals_120 = None
    mul_571: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_570, 2048)
    sum_205: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_570, [2], True)
    mul_572: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_570, mul_569);  mul_570 = None
    sum_206: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_572, [2], True);  mul_572 = None
    mul_573: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_569, sum_206);  sum_206 = None
    sub_195: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_571, sum_205);  mul_571 = sum_205 = None
    sub_196: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_195, mul_573);  sub_195 = mul_573 = None
    div_54: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 2048);  rsqrt_18 = None
    mul_574: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_54, sub_196);  div_54 = sub_196 = None
    mul_575: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_314, mul_569);  mul_569 = None
    sum_207: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_575, [0, 1]);  mul_575 = None
    sum_208: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_314, [0, 1]);  add_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    add_315: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_310, mul_574);  add_310 = mul_574 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_911: "f32[128, 2048]" = torch.ops.aten.view.default(add_315, [128, 2048])
    permute_764: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_98, [1, 0]);  permute_98 = None
    mm_255: "f32[128, 8192]" = torch.ops.aten.mm.default(view_911, permute_764);  permute_764 = None
    permute_765: "f32[2048, 128]" = torch.ops.aten.permute.default(view_911, [1, 0])
    mm_256: "f32[2048, 8192]" = torch.ops.aten.mm.default(permute_765, view_198);  permute_765 = view_198 = None
    permute_766: "f32[8192, 2048]" = torch.ops.aten.permute.default(mm_256, [1, 0]);  mm_256 = None
    sum_209: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_911, [0], True);  view_911 = None
    view_912: "f32[2048]" = torch.ops.aten.view.default(sum_209, [2048]);  sum_209 = None
    permute_767: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_766, [1, 0]);  permute_766 = None
    view_913: "f32[1, 128, 8192]" = torch.ops.aten.view.default(mm_255, [1, 128, 8192]);  mm_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_576: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_913, mul_68);  mul_68 = None
    mul_577: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_913, add_71);  view_913 = add_71 = None
    alias_78: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(alias_17);  alias_17 = None
    mul_578: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(alias_78, alias_78);  alias_78 = None
    sub_197: "f32[1, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_578);  mul_578 = None
    mul_579: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_576, sub_197);  mul_576 = sub_197 = None
    mul_580: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_579, 0.7978845608028654);  mul_579 = None
    mul_581: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_580, 0.044715)
    pow_40: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_197, 2.0);  view_197 = None
    mul_582: "f32[1, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_40, 3.0);  pow_40 = None
    mul_583: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_581, mul_582);  mul_581 = mul_582 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_316: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(mul_580, mul_583);  mul_580 = mul_583 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_584: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_577, 0.5);  mul_577 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_317: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(add_316, mul_584);  add_316 = mul_584 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_914: "f32[128, 8192]" = torch.ops.aten.view.default(add_317, [128, 8192]);  add_317 = None
    permute_768: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_97, [1, 0]);  permute_97 = None
    mm_257: "f32[128, 2048]" = torch.ops.aten.mm.default(view_914, permute_768);  permute_768 = None
    permute_769: "f32[8192, 128]" = torch.ops.aten.permute.default(view_914, [1, 0])
    mm_258: "f32[8192, 2048]" = torch.ops.aten.mm.default(permute_769, view_196);  permute_769 = view_196 = None
    permute_770: "f32[2048, 8192]" = torch.ops.aten.permute.default(mm_258, [1, 0]);  mm_258 = None
    sum_210: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_914, [0], True);  view_914 = None
    view_915: "f32[8192]" = torch.ops.aten.view.default(sum_210, [8192]);  sum_210 = None
    permute_771: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_770, [1, 0]);  permute_770 = None
    view_916: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_257, [1, 128, 2048]);  mm_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    sub_198: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_67, getitem_35);  add_67 = getitem_35 = None
    mul_585: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_198, rsqrt_17);  sub_198 = None
    mul_586: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_916, primals_114);  primals_114 = None
    mul_587: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_586, 2048)
    sum_211: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_586, [2], True)
    mul_588: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_586, mul_585);  mul_586 = None
    sum_212: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_588, [2], True);  mul_588 = None
    mul_589: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_585, sum_212);  sum_212 = None
    sub_199: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_587, sum_211);  mul_587 = sum_211 = None
    sub_200: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_199, mul_589);  sub_199 = mul_589 = None
    div_55: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 2048);  rsqrt_17 = None
    mul_590: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_55, sub_200);  div_55 = sub_200 = None
    mul_591: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_916, mul_585);  mul_585 = None
    sum_213: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_591, [0, 1]);  mul_591 = None
    sum_214: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_916, [0, 1]);  view_916 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    add_318: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_315, mul_590);  add_315 = mul_590 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_917: "f32[128, 2048]" = torch.ops.aten.view.default(add_318, [128, 2048])
    permute_772: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_96, [1, 0]);  permute_96 = None
    mm_259: "f32[128, 2048]" = torch.ops.aten.mm.default(view_917, permute_772);  permute_772 = None
    permute_773: "f32[2048, 128]" = torch.ops.aten.permute.default(view_917, [1, 0])
    mm_260: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_773, view_194);  permute_773 = view_194 = None
    permute_774: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_260, [1, 0]);  mm_260 = None
    sum_215: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_917, [0], True);  view_917 = None
    view_918: "f32[2048]" = torch.ops.aten.view.default(sum_215, [2048]);  sum_215 = None
    permute_775: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_774, [1, 0]);  permute_774 = None
    view_919: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_259, [1, 128, 2048]);  mm_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_920: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_919, [1, 128, 16, 128]);  view_919 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_776: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_920, [0, 2, 1, 3]);  view_920 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_921: "f32[16, 128, 128]" = torch.ops.aten.view.default(permute_776, [16, 128, 128]);  permute_776 = None
    permute_777: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_190, [0, 2, 1]);  view_190 = None
    bmm_108: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_777, view_921);  permute_777 = None
    permute_778: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_191, [0, 2, 1]);  view_191 = None
    bmm_109: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_921, permute_778);  view_921 = permute_778 = None
    view_922: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_108, [1, 16, 128, 128]);  bmm_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    add_319: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_19, view_922);  tangents_19 = view_922 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_923: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_109, [1, 16, 128, 128]);  bmm_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_79: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(alias_16);  alias_16 = None
    mul_592: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_923, alias_79);  view_923 = None
    sum_216: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_592, [-1], True)
    mul_593: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_79, sum_216);  alias_79 = sum_216 = None
    sub_201: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_592, mul_593);  mul_592 = mul_593 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    scalar_tensor_15: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_39: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_36, sub_201, scalar_tensor_15);  slice_36 = sub_201 = scalar_tensor_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_924: "f32[16, 128, 128]" = torch.ops.aten.view.default(where_39, [16, 128, 128]);  where_39 = None
    permute_779: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_187, [0, 2, 1]);  view_187 = None
    bmm_110: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_779, view_924);  permute_779 = None
    permute_780: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_188, [0, 2, 1]);  view_188 = None
    bmm_111: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_924, permute_780);  view_924 = permute_780 = None
    view_925: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_110, [1, 16, 128, 128]);  bmm_110 = None
    view_926: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_111, [1, 16, 128, 128]);  bmm_111 = None
    permute_781: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_925, [0, 1, 3, 2]);  view_925 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_320: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_18, permute_781);  tangents_18 = permute_781 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_782: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_319, [0, 2, 1, 3]);  add_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_142: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_782, memory_format = torch.contiguous_format);  permute_782 = None
    view_927: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_142, [1, 128, 2048]);  clone_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_783: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_320, [0, 2, 1, 3]);  add_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_143: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_783, memory_format = torch.contiguous_format);  permute_783 = None
    view_928: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_143, [1, 128, 2048]);  clone_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_784: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_926, [0, 2, 1, 3]);  view_926 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_144: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_784, memory_format = torch.contiguous_format);  permute_784 = None
    view_929: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_144, [1, 128, 2048]);  clone_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    view_930: "f32[128, 2048]" = torch.ops.aten.view.default(view_927, [128, 2048]);  view_927 = None
    permute_785: "f32[2048, 128]" = torch.ops.aten.permute.default(view_930, [1, 0])
    mm_261: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_785, view_182);  permute_785 = view_182 = None
    permute_786: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_261, [1, 0]);  mm_261 = None
    permute_787: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_90, [1, 0]);  permute_90 = None
    mm_262: "f32[128, 2048]" = torch.ops.aten.mm.default(view_930, permute_787);  view_930 = permute_787 = None
    view_931: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_262, [1, 128, 2048]);  mm_262 = None
    permute_788: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_786, [1, 0]);  permute_786 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    view_932: "f32[128, 2048]" = torch.ops.aten.view.default(view_928, [128, 2048]);  view_928 = None
    permute_789: "f32[2048, 128]" = torch.ops.aten.permute.default(view_932, [1, 0])
    mm_263: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_789, view_180);  permute_789 = view_180 = None
    permute_790: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_263, [1, 0]);  mm_263 = None
    permute_791: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_89, [1, 0]);  permute_89 = None
    mm_264: "f32[128, 2048]" = torch.ops.aten.mm.default(view_932, permute_791);  view_932 = permute_791 = None
    view_933: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_264, [1, 128, 2048]);  mm_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    add_321: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_931, view_933);  view_931 = view_933 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_792: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_790, [1, 0]);  permute_790 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    view_934: "f32[128, 2048]" = torch.ops.aten.view.default(view_929, [128, 2048]);  view_929 = None
    permute_793: "f32[2048, 128]" = torch.ops.aten.permute.default(view_934, [1, 0])
    mm_265: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_793, view_178);  permute_793 = view_178 = None
    permute_794: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_265, [1, 0]);  mm_265 = None
    permute_795: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_88, [1, 0]);  permute_88 = None
    mm_266: "f32[128, 2048]" = torch.ops.aten.mm.default(view_934, permute_795);  view_934 = permute_795 = None
    view_935: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_266, [1, 128, 2048]);  mm_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    add_322: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_321, view_935);  add_321 = view_935 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_796: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_794, [1, 0]);  permute_794 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    sub_202: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_64, getitem_33);  add_64 = getitem_33 = None
    mul_594: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_202, rsqrt_16);  sub_202 = None
    mul_595: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_322, primals_107);  primals_107 = None
    mul_596: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_595, 2048)
    sum_217: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_595, [2], True)
    mul_597: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_595, mul_594);  mul_595 = None
    sum_218: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_597, [2], True);  mul_597 = None
    mul_598: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_594, sum_218);  sum_218 = None
    sub_203: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_596, sum_217);  mul_596 = sum_217 = None
    sub_204: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_203, mul_598);  sub_203 = mul_598 = None
    div_56: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 2048);  rsqrt_16 = None
    mul_599: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_56, sub_204);  div_56 = sub_204 = None
    mul_600: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_322, mul_594);  mul_594 = None
    sum_219: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_600, [0, 1]);  mul_600 = None
    sum_220: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_322, [0, 1]);  add_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    add_323: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_318, mul_599);  add_318 = mul_599 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_936: "f32[128, 2048]" = torch.ops.aten.view.default(add_323, [128, 2048])
    permute_797: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_87, [1, 0]);  permute_87 = None
    mm_267: "f32[128, 8192]" = torch.ops.aten.mm.default(view_936, permute_797);  permute_797 = None
    permute_798: "f32[2048, 128]" = torch.ops.aten.permute.default(view_936, [1, 0])
    mm_268: "f32[2048, 8192]" = torch.ops.aten.mm.default(permute_798, view_176);  permute_798 = view_176 = None
    permute_799: "f32[8192, 2048]" = torch.ops.aten.permute.default(mm_268, [1, 0]);  mm_268 = None
    sum_221: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_936, [0], True);  view_936 = None
    view_937: "f32[2048]" = torch.ops.aten.view.default(sum_221, [2048]);  sum_221 = None
    permute_800: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_799, [1, 0]);  permute_799 = None
    view_938: "f32[1, 128, 8192]" = torch.ops.aten.view.default(mm_267, [1, 128, 8192]);  mm_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_601: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_938, mul_60);  mul_60 = None
    mul_602: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_938, add_63);  view_938 = add_63 = None
    alias_80: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(alias_15);  alias_15 = None
    mul_603: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(alias_80, alias_80);  alias_80 = None
    sub_205: "f32[1, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_603);  mul_603 = None
    mul_604: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_601, sub_205);  mul_601 = sub_205 = None
    mul_605: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_604, 0.7978845608028654);  mul_604 = None
    mul_606: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_605, 0.044715)
    pow_41: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_175, 2.0);  view_175 = None
    mul_607: "f32[1, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_41, 3.0);  pow_41 = None
    mul_608: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_606, mul_607);  mul_606 = mul_607 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_324: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(mul_605, mul_608);  mul_605 = mul_608 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_609: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_602, 0.5);  mul_602 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_325: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(add_324, mul_609);  add_324 = mul_609 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_939: "f32[128, 8192]" = torch.ops.aten.view.default(add_325, [128, 8192]);  add_325 = None
    permute_801: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_86, [1, 0]);  permute_86 = None
    mm_269: "f32[128, 2048]" = torch.ops.aten.mm.default(view_939, permute_801);  permute_801 = None
    permute_802: "f32[8192, 128]" = torch.ops.aten.permute.default(view_939, [1, 0])
    mm_270: "f32[8192, 2048]" = torch.ops.aten.mm.default(permute_802, view_174);  permute_802 = view_174 = None
    permute_803: "f32[2048, 8192]" = torch.ops.aten.permute.default(mm_270, [1, 0]);  mm_270 = None
    sum_222: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_939, [0], True);  view_939 = None
    view_940: "f32[8192]" = torch.ops.aten.view.default(sum_222, [8192]);  sum_222 = None
    permute_804: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_803, [1, 0]);  permute_803 = None
    view_941: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_269, [1, 128, 2048]);  mm_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    sub_206: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_59, getitem_31);  add_59 = getitem_31 = None
    mul_610: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_206, rsqrt_15);  sub_206 = None
    mul_611: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_941, primals_101);  primals_101 = None
    mul_612: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_611, 2048)
    sum_223: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_611, [2], True)
    mul_613: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_611, mul_610);  mul_611 = None
    sum_224: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_613, [2], True);  mul_613 = None
    mul_614: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_610, sum_224);  sum_224 = None
    sub_207: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_612, sum_223);  mul_612 = sum_223 = None
    sub_208: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_207, mul_614);  sub_207 = mul_614 = None
    div_57: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 2048);  rsqrt_15 = None
    mul_615: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_57, sub_208);  div_57 = sub_208 = None
    mul_616: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_941, mul_610);  mul_610 = None
    sum_225: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_616, [0, 1]);  mul_616 = None
    sum_226: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_941, [0, 1]);  view_941 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    add_326: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_323, mul_615);  add_323 = mul_615 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_942: "f32[128, 2048]" = torch.ops.aten.view.default(add_326, [128, 2048])
    permute_805: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_85, [1, 0]);  permute_85 = None
    mm_271: "f32[128, 2048]" = torch.ops.aten.mm.default(view_942, permute_805);  permute_805 = None
    permute_806: "f32[2048, 128]" = torch.ops.aten.permute.default(view_942, [1, 0])
    mm_272: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_806, view_172);  permute_806 = view_172 = None
    permute_807: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_272, [1, 0]);  mm_272 = None
    sum_227: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_942, [0], True);  view_942 = None
    view_943: "f32[2048]" = torch.ops.aten.view.default(sum_227, [2048]);  sum_227 = None
    permute_808: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_807, [1, 0]);  permute_807 = None
    view_944: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_271, [1, 128, 2048]);  mm_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_945: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_944, [1, 128, 16, 128]);  view_944 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_809: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_945, [0, 2, 1, 3]);  view_945 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_946: "f32[16, 128, 128]" = torch.ops.aten.view.default(permute_809, [16, 128, 128]);  permute_809 = None
    permute_810: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_168, [0, 2, 1]);  view_168 = None
    bmm_112: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_810, view_946);  permute_810 = None
    permute_811: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_169, [0, 2, 1]);  view_169 = None
    bmm_113: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_946, permute_811);  view_946 = permute_811 = None
    view_947: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_112, [1, 16, 128, 128]);  bmm_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    add_327: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_17, view_947);  tangents_17 = view_947 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_948: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_113, [1, 16, 128, 128]);  bmm_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_81: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(alias_14);  alias_14 = None
    mul_617: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_948, alias_81);  view_948 = None
    sum_228: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_617, [-1], True)
    mul_618: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_81, sum_228);  alias_81 = sum_228 = None
    sub_209: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_617, mul_618);  mul_617 = mul_618 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    scalar_tensor_16: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_40: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_32, sub_209, scalar_tensor_16);  slice_32 = sub_209 = scalar_tensor_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_949: "f32[16, 128, 128]" = torch.ops.aten.view.default(where_40, [16, 128, 128]);  where_40 = None
    permute_812: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_165, [0, 2, 1]);  view_165 = None
    bmm_114: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_812, view_949);  permute_812 = None
    permute_813: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_166, [0, 2, 1]);  view_166 = None
    bmm_115: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_949, permute_813);  view_949 = permute_813 = None
    view_950: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_114, [1, 16, 128, 128]);  bmm_114 = None
    view_951: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_115, [1, 16, 128, 128]);  bmm_115 = None
    permute_814: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_950, [0, 1, 3, 2]);  view_950 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_328: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_16, permute_814);  tangents_16 = permute_814 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_815: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_327, [0, 2, 1, 3]);  add_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_145: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_815, memory_format = torch.contiguous_format);  permute_815 = None
    view_952: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_145, [1, 128, 2048]);  clone_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_816: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_328, [0, 2, 1, 3]);  add_328 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_146: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_816, memory_format = torch.contiguous_format);  permute_816 = None
    view_953: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_146, [1, 128, 2048]);  clone_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_817: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_951, [0, 2, 1, 3]);  view_951 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_147: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_817, memory_format = torch.contiguous_format);  permute_817 = None
    view_954: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_147, [1, 128, 2048]);  clone_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    view_955: "f32[128, 2048]" = torch.ops.aten.view.default(view_952, [128, 2048]);  view_952 = None
    permute_818: "f32[2048, 128]" = torch.ops.aten.permute.default(view_955, [1, 0])
    mm_273: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_818, view_160);  permute_818 = view_160 = None
    permute_819: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_273, [1, 0]);  mm_273 = None
    permute_820: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_79, [1, 0]);  permute_79 = None
    mm_274: "f32[128, 2048]" = torch.ops.aten.mm.default(view_955, permute_820);  view_955 = permute_820 = None
    view_956: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_274, [1, 128, 2048]);  mm_274 = None
    permute_821: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_819, [1, 0]);  permute_819 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    view_957: "f32[128, 2048]" = torch.ops.aten.view.default(view_953, [128, 2048]);  view_953 = None
    permute_822: "f32[2048, 128]" = torch.ops.aten.permute.default(view_957, [1, 0])
    mm_275: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_822, view_158);  permute_822 = view_158 = None
    permute_823: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_275, [1, 0]);  mm_275 = None
    permute_824: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_78, [1, 0]);  permute_78 = None
    mm_276: "f32[128, 2048]" = torch.ops.aten.mm.default(view_957, permute_824);  view_957 = permute_824 = None
    view_958: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_276, [1, 128, 2048]);  mm_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    add_329: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_956, view_958);  view_956 = view_958 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_825: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_823, [1, 0]);  permute_823 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    view_959: "f32[128, 2048]" = torch.ops.aten.view.default(view_954, [128, 2048]);  view_954 = None
    permute_826: "f32[2048, 128]" = torch.ops.aten.permute.default(view_959, [1, 0])
    mm_277: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_826, view_156);  permute_826 = view_156 = None
    permute_827: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_277, [1, 0]);  mm_277 = None
    permute_828: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_77, [1, 0]);  permute_77 = None
    mm_278: "f32[128, 2048]" = torch.ops.aten.mm.default(view_959, permute_828);  view_959 = permute_828 = None
    view_960: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_278, [1, 128, 2048]);  mm_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    add_330: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_329, view_960);  add_329 = view_960 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_829: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_827, [1, 0]);  permute_827 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    sub_210: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_56, getitem_29);  add_56 = getitem_29 = None
    mul_619: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_210, rsqrt_14);  sub_210 = None
    mul_620: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_330, primals_94);  primals_94 = None
    mul_621: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_620, 2048)
    sum_229: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_620, [2], True)
    mul_622: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_620, mul_619);  mul_620 = None
    sum_230: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_622, [2], True);  mul_622 = None
    mul_623: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_619, sum_230);  sum_230 = None
    sub_211: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_621, sum_229);  mul_621 = sum_229 = None
    sub_212: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_211, mul_623);  sub_211 = mul_623 = None
    div_58: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 2048);  rsqrt_14 = None
    mul_624: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_58, sub_212);  div_58 = sub_212 = None
    mul_625: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_330, mul_619);  mul_619 = None
    sum_231: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_625, [0, 1]);  mul_625 = None
    sum_232: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_330, [0, 1]);  add_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    add_331: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_326, mul_624);  add_326 = mul_624 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_961: "f32[128, 2048]" = torch.ops.aten.view.default(add_331, [128, 2048])
    permute_830: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None
    mm_279: "f32[128, 8192]" = torch.ops.aten.mm.default(view_961, permute_830);  permute_830 = None
    permute_831: "f32[2048, 128]" = torch.ops.aten.permute.default(view_961, [1, 0])
    mm_280: "f32[2048, 8192]" = torch.ops.aten.mm.default(permute_831, view_154);  permute_831 = view_154 = None
    permute_832: "f32[8192, 2048]" = torch.ops.aten.permute.default(mm_280, [1, 0]);  mm_280 = None
    sum_233: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_961, [0], True);  view_961 = None
    view_962: "f32[2048]" = torch.ops.aten.view.default(sum_233, [2048]);  sum_233 = None
    permute_833: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_832, [1, 0]);  permute_832 = None
    view_963: "f32[1, 128, 8192]" = torch.ops.aten.view.default(mm_279, [1, 128, 8192]);  mm_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_626: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_963, mul_52);  mul_52 = None
    mul_627: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_963, add_55);  view_963 = add_55 = None
    alias_82: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(alias_13);  alias_13 = None
    mul_628: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(alias_82, alias_82);  alias_82 = None
    sub_213: "f32[1, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_628);  mul_628 = None
    mul_629: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_626, sub_213);  mul_626 = sub_213 = None
    mul_630: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_629, 0.7978845608028654);  mul_629 = None
    mul_631: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_630, 0.044715)
    pow_42: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_153, 2.0);  view_153 = None
    mul_632: "f32[1, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_42, 3.0);  pow_42 = None
    mul_633: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_631, mul_632);  mul_631 = mul_632 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_332: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(mul_630, mul_633);  mul_630 = mul_633 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_634: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_627, 0.5);  mul_627 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_333: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(add_332, mul_634);  add_332 = mul_634 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_964: "f32[128, 8192]" = torch.ops.aten.view.default(add_333, [128, 8192]);  add_333 = None
    permute_834: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_75, [1, 0]);  permute_75 = None
    mm_281: "f32[128, 2048]" = torch.ops.aten.mm.default(view_964, permute_834);  permute_834 = None
    permute_835: "f32[8192, 128]" = torch.ops.aten.permute.default(view_964, [1, 0])
    mm_282: "f32[8192, 2048]" = torch.ops.aten.mm.default(permute_835, view_152);  permute_835 = view_152 = None
    permute_836: "f32[2048, 8192]" = torch.ops.aten.permute.default(mm_282, [1, 0]);  mm_282 = None
    sum_234: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_964, [0], True);  view_964 = None
    view_965: "f32[8192]" = torch.ops.aten.view.default(sum_234, [8192]);  sum_234 = None
    permute_837: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_836, [1, 0]);  permute_836 = None
    view_966: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_281, [1, 128, 2048]);  mm_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    sub_214: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_51, getitem_27);  add_51 = getitem_27 = None
    mul_635: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_214, rsqrt_13);  sub_214 = None
    mul_636: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_966, primals_88);  primals_88 = None
    mul_637: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_636, 2048)
    sum_235: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_636, [2], True)
    mul_638: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_636, mul_635);  mul_636 = None
    sum_236: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_638, [2], True);  mul_638 = None
    mul_639: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_635, sum_236);  sum_236 = None
    sub_215: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_637, sum_235);  mul_637 = sum_235 = None
    sub_216: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_215, mul_639);  sub_215 = mul_639 = None
    div_59: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 2048);  rsqrt_13 = None
    mul_640: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_59, sub_216);  div_59 = sub_216 = None
    mul_641: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_966, mul_635);  mul_635 = None
    sum_237: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_641, [0, 1]);  mul_641 = None
    sum_238: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_966, [0, 1]);  view_966 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    add_334: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_331, mul_640);  add_331 = mul_640 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_967: "f32[128, 2048]" = torch.ops.aten.view.default(add_334, [128, 2048])
    permute_838: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_74, [1, 0]);  permute_74 = None
    mm_283: "f32[128, 2048]" = torch.ops.aten.mm.default(view_967, permute_838);  permute_838 = None
    permute_839: "f32[2048, 128]" = torch.ops.aten.permute.default(view_967, [1, 0])
    mm_284: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_839, view_150);  permute_839 = view_150 = None
    permute_840: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_284, [1, 0]);  mm_284 = None
    sum_239: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_967, [0], True);  view_967 = None
    view_968: "f32[2048]" = torch.ops.aten.view.default(sum_239, [2048]);  sum_239 = None
    permute_841: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_840, [1, 0]);  permute_840 = None
    view_969: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_283, [1, 128, 2048]);  mm_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_970: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_969, [1, 128, 16, 128]);  view_969 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_842: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_970, [0, 2, 1, 3]);  view_970 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_971: "f32[16, 128, 128]" = torch.ops.aten.view.default(permute_842, [16, 128, 128]);  permute_842 = None
    permute_843: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_146, [0, 2, 1]);  view_146 = None
    bmm_116: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_843, view_971);  permute_843 = None
    permute_844: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_147, [0, 2, 1]);  view_147 = None
    bmm_117: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_971, permute_844);  view_971 = permute_844 = None
    view_972: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_116, [1, 16, 128, 128]);  bmm_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    add_335: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_15, view_972);  tangents_15 = view_972 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_973: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_117, [1, 16, 128, 128]);  bmm_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_83: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(alias_12);  alias_12 = None
    mul_642: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_973, alias_83);  view_973 = None
    sum_240: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_642, [-1], True)
    mul_643: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_83, sum_240);  alias_83 = sum_240 = None
    sub_217: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_642, mul_643);  mul_642 = mul_643 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    scalar_tensor_17: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_41: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_28, sub_217, scalar_tensor_17);  slice_28 = sub_217 = scalar_tensor_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_974: "f32[16, 128, 128]" = torch.ops.aten.view.default(where_41, [16, 128, 128]);  where_41 = None
    permute_845: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_143, [0, 2, 1]);  view_143 = None
    bmm_118: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_845, view_974);  permute_845 = None
    permute_846: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_144, [0, 2, 1]);  view_144 = None
    bmm_119: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_974, permute_846);  view_974 = permute_846 = None
    view_975: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_118, [1, 16, 128, 128]);  bmm_118 = None
    view_976: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_119, [1, 16, 128, 128]);  bmm_119 = None
    permute_847: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_975, [0, 1, 3, 2]);  view_975 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_336: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_14, permute_847);  tangents_14 = permute_847 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_848: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_335, [0, 2, 1, 3]);  add_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_148: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_848, memory_format = torch.contiguous_format);  permute_848 = None
    view_977: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_148, [1, 128, 2048]);  clone_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_849: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_336, [0, 2, 1, 3]);  add_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_149: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_849, memory_format = torch.contiguous_format);  permute_849 = None
    view_978: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_149, [1, 128, 2048]);  clone_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_850: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_976, [0, 2, 1, 3]);  view_976 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_150: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_850, memory_format = torch.contiguous_format);  permute_850 = None
    view_979: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_150, [1, 128, 2048]);  clone_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    view_980: "f32[128, 2048]" = torch.ops.aten.view.default(view_977, [128, 2048]);  view_977 = None
    permute_851: "f32[2048, 128]" = torch.ops.aten.permute.default(view_980, [1, 0])
    mm_285: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_851, view_138);  permute_851 = view_138 = None
    permute_852: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_285, [1, 0]);  mm_285 = None
    permute_853: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_68, [1, 0]);  permute_68 = None
    mm_286: "f32[128, 2048]" = torch.ops.aten.mm.default(view_980, permute_853);  view_980 = permute_853 = None
    view_981: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_286, [1, 128, 2048]);  mm_286 = None
    permute_854: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_852, [1, 0]);  permute_852 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    view_982: "f32[128, 2048]" = torch.ops.aten.view.default(view_978, [128, 2048]);  view_978 = None
    permute_855: "f32[2048, 128]" = torch.ops.aten.permute.default(view_982, [1, 0])
    mm_287: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_855, view_136);  permute_855 = view_136 = None
    permute_856: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_287, [1, 0]);  mm_287 = None
    permute_857: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_67, [1, 0]);  permute_67 = None
    mm_288: "f32[128, 2048]" = torch.ops.aten.mm.default(view_982, permute_857);  view_982 = permute_857 = None
    view_983: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_288, [1, 128, 2048]);  mm_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    add_337: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_981, view_983);  view_981 = view_983 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_858: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_856, [1, 0]);  permute_856 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    view_984: "f32[128, 2048]" = torch.ops.aten.view.default(view_979, [128, 2048]);  view_979 = None
    permute_859: "f32[2048, 128]" = torch.ops.aten.permute.default(view_984, [1, 0])
    mm_289: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_859, view_134);  permute_859 = view_134 = None
    permute_860: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_289, [1, 0]);  mm_289 = None
    permute_861: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
    mm_290: "f32[128, 2048]" = torch.ops.aten.mm.default(view_984, permute_861);  view_984 = permute_861 = None
    view_985: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_290, [1, 128, 2048]);  mm_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    add_338: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_337, view_985);  add_337 = view_985 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_862: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_860, [1, 0]);  permute_860 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    sub_218: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_48, getitem_25);  add_48 = getitem_25 = None
    mul_644: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_218, rsqrt_12);  sub_218 = None
    mul_645: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_338, primals_81);  primals_81 = None
    mul_646: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_645, 2048)
    sum_241: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_645, [2], True)
    mul_647: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_645, mul_644);  mul_645 = None
    sum_242: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_647, [2], True);  mul_647 = None
    mul_648: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_644, sum_242);  sum_242 = None
    sub_219: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_646, sum_241);  mul_646 = sum_241 = None
    sub_220: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_219, mul_648);  sub_219 = mul_648 = None
    div_60: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 2048);  rsqrt_12 = None
    mul_649: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_60, sub_220);  div_60 = sub_220 = None
    mul_650: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_338, mul_644);  mul_644 = None
    sum_243: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_650, [0, 1]);  mul_650 = None
    sum_244: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_338, [0, 1]);  add_338 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    add_339: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_334, mul_649);  add_334 = mul_649 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_986: "f32[128, 2048]" = torch.ops.aten.view.default(add_339, [128, 2048])
    permute_863: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
    mm_291: "f32[128, 8192]" = torch.ops.aten.mm.default(view_986, permute_863);  permute_863 = None
    permute_864: "f32[2048, 128]" = torch.ops.aten.permute.default(view_986, [1, 0])
    mm_292: "f32[2048, 8192]" = torch.ops.aten.mm.default(permute_864, view_132);  permute_864 = view_132 = None
    permute_865: "f32[8192, 2048]" = torch.ops.aten.permute.default(mm_292, [1, 0]);  mm_292 = None
    sum_245: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_986, [0], True);  view_986 = None
    view_987: "f32[2048]" = torch.ops.aten.view.default(sum_245, [2048]);  sum_245 = None
    permute_866: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_865, [1, 0]);  permute_865 = None
    view_988: "f32[1, 128, 8192]" = torch.ops.aten.view.default(mm_291, [1, 128, 8192]);  mm_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_651: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_988, mul_44);  mul_44 = None
    mul_652: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_988, add_47);  view_988 = add_47 = None
    alias_84: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(alias_11);  alias_11 = None
    mul_653: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(alias_84, alias_84);  alias_84 = None
    sub_221: "f32[1, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_653);  mul_653 = None
    mul_654: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_651, sub_221);  mul_651 = sub_221 = None
    mul_655: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_654, 0.7978845608028654);  mul_654 = None
    mul_656: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_655, 0.044715)
    pow_43: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_131, 2.0);  view_131 = None
    mul_657: "f32[1, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_43, 3.0);  pow_43 = None
    mul_658: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_656, mul_657);  mul_656 = mul_657 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_340: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(mul_655, mul_658);  mul_655 = mul_658 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_659: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_652, 0.5);  mul_652 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_341: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(add_340, mul_659);  add_340 = mul_659 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_989: "f32[128, 8192]" = torch.ops.aten.view.default(add_341, [128, 8192]);  add_341 = None
    permute_867: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_64, [1, 0]);  permute_64 = None
    mm_293: "f32[128, 2048]" = torch.ops.aten.mm.default(view_989, permute_867);  permute_867 = None
    permute_868: "f32[8192, 128]" = torch.ops.aten.permute.default(view_989, [1, 0])
    mm_294: "f32[8192, 2048]" = torch.ops.aten.mm.default(permute_868, view_130);  permute_868 = view_130 = None
    permute_869: "f32[2048, 8192]" = torch.ops.aten.permute.default(mm_294, [1, 0]);  mm_294 = None
    sum_246: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_989, [0], True);  view_989 = None
    view_990: "f32[8192]" = torch.ops.aten.view.default(sum_246, [8192]);  sum_246 = None
    permute_870: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_869, [1, 0]);  permute_869 = None
    view_991: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_293, [1, 128, 2048]);  mm_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    sub_222: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_43, getitem_23);  add_43 = getitem_23 = None
    mul_660: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_222, rsqrt_11);  sub_222 = None
    mul_661: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_991, primals_75);  primals_75 = None
    mul_662: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_661, 2048)
    sum_247: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_661, [2], True)
    mul_663: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_661, mul_660);  mul_661 = None
    sum_248: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_663, [2], True);  mul_663 = None
    mul_664: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_660, sum_248);  sum_248 = None
    sub_223: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_662, sum_247);  mul_662 = sum_247 = None
    sub_224: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_223, mul_664);  sub_223 = mul_664 = None
    div_61: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 2048);  rsqrt_11 = None
    mul_665: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_61, sub_224);  div_61 = sub_224 = None
    mul_666: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_991, mul_660);  mul_660 = None
    sum_249: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_666, [0, 1]);  mul_666 = None
    sum_250: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_991, [0, 1]);  view_991 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    add_342: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_339, mul_665);  add_339 = mul_665 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_992: "f32[128, 2048]" = torch.ops.aten.view.default(add_342, [128, 2048])
    permute_871: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_63, [1, 0]);  permute_63 = None
    mm_295: "f32[128, 2048]" = torch.ops.aten.mm.default(view_992, permute_871);  permute_871 = None
    permute_872: "f32[2048, 128]" = torch.ops.aten.permute.default(view_992, [1, 0])
    mm_296: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_872, view_128);  permute_872 = view_128 = None
    permute_873: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_296, [1, 0]);  mm_296 = None
    sum_251: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_992, [0], True);  view_992 = None
    view_993: "f32[2048]" = torch.ops.aten.view.default(sum_251, [2048]);  sum_251 = None
    permute_874: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_873, [1, 0]);  permute_873 = None
    view_994: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_295, [1, 128, 2048]);  mm_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_995: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_994, [1, 128, 16, 128]);  view_994 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_875: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_995, [0, 2, 1, 3]);  view_995 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_996: "f32[16, 128, 128]" = torch.ops.aten.view.default(permute_875, [16, 128, 128]);  permute_875 = None
    permute_876: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_124, [0, 2, 1]);  view_124 = None
    bmm_120: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_876, view_996);  permute_876 = None
    permute_877: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_125, [0, 2, 1]);  view_125 = None
    bmm_121: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_996, permute_877);  view_996 = permute_877 = None
    view_997: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_120, [1, 16, 128, 128]);  bmm_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    add_343: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_13, view_997);  tangents_13 = view_997 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_998: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_121, [1, 16, 128, 128]);  bmm_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_85: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(alias_10);  alias_10 = None
    mul_667: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_998, alias_85);  view_998 = None
    sum_252: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_667, [-1], True)
    mul_668: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_85, sum_252);  alias_85 = sum_252 = None
    sub_225: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_667, mul_668);  mul_667 = mul_668 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    scalar_tensor_18: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_42: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_24, sub_225, scalar_tensor_18);  slice_24 = sub_225 = scalar_tensor_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_999: "f32[16, 128, 128]" = torch.ops.aten.view.default(where_42, [16, 128, 128]);  where_42 = None
    permute_878: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_121, [0, 2, 1]);  view_121 = None
    bmm_122: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_878, view_999);  permute_878 = None
    permute_879: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_122, [0, 2, 1]);  view_122 = None
    bmm_123: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_999, permute_879);  view_999 = permute_879 = None
    view_1000: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_122, [1, 16, 128, 128]);  bmm_122 = None
    view_1001: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_123, [1, 16, 128, 128]);  bmm_123 = None
    permute_880: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_1000, [0, 1, 3, 2]);  view_1000 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_344: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_12, permute_880);  tangents_12 = permute_880 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_881: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_343, [0, 2, 1, 3]);  add_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_151: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_881, memory_format = torch.contiguous_format);  permute_881 = None
    view_1002: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_151, [1, 128, 2048]);  clone_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_882: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_344, [0, 2, 1, 3]);  add_344 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_152: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_882, memory_format = torch.contiguous_format);  permute_882 = None
    view_1003: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_152, [1, 128, 2048]);  clone_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_883: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_1001, [0, 2, 1, 3]);  view_1001 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_153: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_883, memory_format = torch.contiguous_format);  permute_883 = None
    view_1004: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_153, [1, 128, 2048]);  clone_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    view_1005: "f32[128, 2048]" = torch.ops.aten.view.default(view_1002, [128, 2048]);  view_1002 = None
    permute_884: "f32[2048, 128]" = torch.ops.aten.permute.default(view_1005, [1, 0])
    mm_297: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_884, view_116);  permute_884 = view_116 = None
    permute_885: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_297, [1, 0]);  mm_297 = None
    permute_886: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_57, [1, 0]);  permute_57 = None
    mm_298: "f32[128, 2048]" = torch.ops.aten.mm.default(view_1005, permute_886);  view_1005 = permute_886 = None
    view_1006: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_298, [1, 128, 2048]);  mm_298 = None
    permute_887: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_885, [1, 0]);  permute_885 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    view_1007: "f32[128, 2048]" = torch.ops.aten.view.default(view_1003, [128, 2048]);  view_1003 = None
    permute_888: "f32[2048, 128]" = torch.ops.aten.permute.default(view_1007, [1, 0])
    mm_299: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_888, view_114);  permute_888 = view_114 = None
    permute_889: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_299, [1, 0]);  mm_299 = None
    permute_890: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_56, [1, 0]);  permute_56 = None
    mm_300: "f32[128, 2048]" = torch.ops.aten.mm.default(view_1007, permute_890);  view_1007 = permute_890 = None
    view_1008: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_300, [1, 128, 2048]);  mm_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    add_345: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_1006, view_1008);  view_1006 = view_1008 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_891: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_889, [1, 0]);  permute_889 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    view_1009: "f32[128, 2048]" = torch.ops.aten.view.default(view_1004, [128, 2048]);  view_1004 = None
    permute_892: "f32[2048, 128]" = torch.ops.aten.permute.default(view_1009, [1, 0])
    mm_301: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_892, view_112);  permute_892 = view_112 = None
    permute_893: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_301, [1, 0]);  mm_301 = None
    permute_894: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
    mm_302: "f32[128, 2048]" = torch.ops.aten.mm.default(view_1009, permute_894);  view_1009 = permute_894 = None
    view_1010: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_302, [1, 128, 2048]);  mm_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    add_346: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_345, view_1010);  add_345 = view_1010 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_895: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_893, [1, 0]);  permute_893 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    sub_226: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_40, getitem_21);  add_40 = getitem_21 = None
    mul_669: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_226, rsqrt_10);  sub_226 = None
    mul_670: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_346, primals_68);  primals_68 = None
    mul_671: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_670, 2048)
    sum_253: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_670, [2], True)
    mul_672: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_670, mul_669);  mul_670 = None
    sum_254: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_672, [2], True);  mul_672 = None
    mul_673: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_669, sum_254);  sum_254 = None
    sub_227: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_671, sum_253);  mul_671 = sum_253 = None
    sub_228: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_227, mul_673);  sub_227 = mul_673 = None
    div_62: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 2048);  rsqrt_10 = None
    mul_674: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_62, sub_228);  div_62 = sub_228 = None
    mul_675: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_346, mul_669);  mul_669 = None
    sum_255: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_675, [0, 1]);  mul_675 = None
    sum_256: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_346, [0, 1]);  add_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    add_347: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_342, mul_674);  add_342 = mul_674 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_1011: "f32[128, 2048]" = torch.ops.aten.view.default(add_347, [128, 2048])
    permute_896: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
    mm_303: "f32[128, 8192]" = torch.ops.aten.mm.default(view_1011, permute_896);  permute_896 = None
    permute_897: "f32[2048, 128]" = torch.ops.aten.permute.default(view_1011, [1, 0])
    mm_304: "f32[2048, 8192]" = torch.ops.aten.mm.default(permute_897, view_110);  permute_897 = view_110 = None
    permute_898: "f32[8192, 2048]" = torch.ops.aten.permute.default(mm_304, [1, 0]);  mm_304 = None
    sum_257: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_1011, [0], True);  view_1011 = None
    view_1012: "f32[2048]" = torch.ops.aten.view.default(sum_257, [2048]);  sum_257 = None
    permute_899: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_898, [1, 0]);  permute_898 = None
    view_1013: "f32[1, 128, 8192]" = torch.ops.aten.view.default(mm_303, [1, 128, 8192]);  mm_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_676: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_1013, mul_36);  mul_36 = None
    mul_677: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_1013, add_39);  view_1013 = add_39 = None
    alias_86: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(alias_9);  alias_9 = None
    mul_678: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(alias_86, alias_86);  alias_86 = None
    sub_229: "f32[1, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_678);  mul_678 = None
    mul_679: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_676, sub_229);  mul_676 = sub_229 = None
    mul_680: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_679, 0.7978845608028654);  mul_679 = None
    mul_681: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_680, 0.044715)
    pow_44: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_109, 2.0);  view_109 = None
    mul_682: "f32[1, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_44, 3.0);  pow_44 = None
    mul_683: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_681, mul_682);  mul_681 = mul_682 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_348: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(mul_680, mul_683);  mul_680 = mul_683 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_684: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_677, 0.5);  mul_677 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_349: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(add_348, mul_684);  add_348 = mul_684 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_1014: "f32[128, 8192]" = torch.ops.aten.view.default(add_349, [128, 8192]);  add_349 = None
    permute_900: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None
    mm_305: "f32[128, 2048]" = torch.ops.aten.mm.default(view_1014, permute_900);  permute_900 = None
    permute_901: "f32[8192, 128]" = torch.ops.aten.permute.default(view_1014, [1, 0])
    mm_306: "f32[8192, 2048]" = torch.ops.aten.mm.default(permute_901, view_108);  permute_901 = view_108 = None
    permute_902: "f32[2048, 8192]" = torch.ops.aten.permute.default(mm_306, [1, 0]);  mm_306 = None
    sum_258: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_1014, [0], True);  view_1014 = None
    view_1015: "f32[8192]" = torch.ops.aten.view.default(sum_258, [8192]);  sum_258 = None
    permute_903: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_902, [1, 0]);  permute_902 = None
    view_1016: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_305, [1, 128, 2048]);  mm_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    sub_230: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_35, getitem_19);  add_35 = getitem_19 = None
    mul_685: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_230, rsqrt_9);  sub_230 = None
    mul_686: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_1016, primals_62);  primals_62 = None
    mul_687: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_686, 2048)
    sum_259: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_686, [2], True)
    mul_688: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_686, mul_685);  mul_686 = None
    sum_260: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_688, [2], True);  mul_688 = None
    mul_689: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_685, sum_260);  sum_260 = None
    sub_231: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_687, sum_259);  mul_687 = sum_259 = None
    sub_232: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_231, mul_689);  sub_231 = mul_689 = None
    div_63: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 2048);  rsqrt_9 = None
    mul_690: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_63, sub_232);  div_63 = sub_232 = None
    mul_691: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_1016, mul_685);  mul_685 = None
    sum_261: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_691, [0, 1]);  mul_691 = None
    sum_262: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_1016, [0, 1]);  view_1016 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    add_350: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_347, mul_690);  add_347 = mul_690 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_1017: "f32[128, 2048]" = torch.ops.aten.view.default(add_350, [128, 2048])
    permute_904: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_52, [1, 0]);  permute_52 = None
    mm_307: "f32[128, 2048]" = torch.ops.aten.mm.default(view_1017, permute_904);  permute_904 = None
    permute_905: "f32[2048, 128]" = torch.ops.aten.permute.default(view_1017, [1, 0])
    mm_308: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_905, view_106);  permute_905 = view_106 = None
    permute_906: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_308, [1, 0]);  mm_308 = None
    sum_263: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_1017, [0], True);  view_1017 = None
    view_1018: "f32[2048]" = torch.ops.aten.view.default(sum_263, [2048]);  sum_263 = None
    permute_907: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_906, [1, 0]);  permute_906 = None
    view_1019: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_307, [1, 128, 2048]);  mm_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_1020: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_1019, [1, 128, 16, 128]);  view_1019 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_908: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_1020, [0, 2, 1, 3]);  view_1020 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_1021: "f32[16, 128, 128]" = torch.ops.aten.view.default(permute_908, [16, 128, 128]);  permute_908 = None
    permute_909: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_102, [0, 2, 1]);  view_102 = None
    bmm_124: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_909, view_1021);  permute_909 = None
    permute_910: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_103, [0, 2, 1]);  view_103 = None
    bmm_125: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_1021, permute_910);  view_1021 = permute_910 = None
    view_1022: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_124, [1, 16, 128, 128]);  bmm_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    add_351: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_11, view_1022);  tangents_11 = view_1022 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_1023: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_125, [1, 16, 128, 128]);  bmm_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_87: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(alias_8);  alias_8 = None
    mul_692: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_1023, alias_87);  view_1023 = None
    sum_264: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_692, [-1], True)
    mul_693: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_87, sum_264);  alias_87 = sum_264 = None
    sub_233: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_692, mul_693);  mul_692 = mul_693 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    scalar_tensor_19: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_43: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_20, sub_233, scalar_tensor_19);  slice_20 = sub_233 = scalar_tensor_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_1024: "f32[16, 128, 128]" = torch.ops.aten.view.default(where_43, [16, 128, 128]);  where_43 = None
    permute_911: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_99, [0, 2, 1]);  view_99 = None
    bmm_126: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_911, view_1024);  permute_911 = None
    permute_912: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_100, [0, 2, 1]);  view_100 = None
    bmm_127: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_1024, permute_912);  view_1024 = permute_912 = None
    view_1025: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_126, [1, 16, 128, 128]);  bmm_126 = None
    view_1026: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_127, [1, 16, 128, 128]);  bmm_127 = None
    permute_913: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_1025, [0, 1, 3, 2]);  view_1025 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_352: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_10, permute_913);  tangents_10 = permute_913 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_914: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_351, [0, 2, 1, 3]);  add_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_154: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_914, memory_format = torch.contiguous_format);  permute_914 = None
    view_1027: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_154, [1, 128, 2048]);  clone_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_915: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_352, [0, 2, 1, 3]);  add_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_155: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_915, memory_format = torch.contiguous_format);  permute_915 = None
    view_1028: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_155, [1, 128, 2048]);  clone_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_916: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_1026, [0, 2, 1, 3]);  view_1026 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_156: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_916, memory_format = torch.contiguous_format);  permute_916 = None
    view_1029: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_156, [1, 128, 2048]);  clone_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    view_1030: "f32[128, 2048]" = torch.ops.aten.view.default(view_1027, [128, 2048]);  view_1027 = None
    permute_917: "f32[2048, 128]" = torch.ops.aten.permute.default(view_1030, [1, 0])
    mm_309: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_917, view_94);  permute_917 = view_94 = None
    permute_918: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_309, [1, 0]);  mm_309 = None
    permute_919: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_46, [1, 0]);  permute_46 = None
    mm_310: "f32[128, 2048]" = torch.ops.aten.mm.default(view_1030, permute_919);  view_1030 = permute_919 = None
    view_1031: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_310, [1, 128, 2048]);  mm_310 = None
    permute_920: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_918, [1, 0]);  permute_918 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    view_1032: "f32[128, 2048]" = torch.ops.aten.view.default(view_1028, [128, 2048]);  view_1028 = None
    permute_921: "f32[2048, 128]" = torch.ops.aten.permute.default(view_1032, [1, 0])
    mm_311: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_921, view_92);  permute_921 = view_92 = None
    permute_922: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_311, [1, 0]);  mm_311 = None
    permute_923: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_45, [1, 0]);  permute_45 = None
    mm_312: "f32[128, 2048]" = torch.ops.aten.mm.default(view_1032, permute_923);  view_1032 = permute_923 = None
    view_1033: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_312, [1, 128, 2048]);  mm_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    add_353: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_1031, view_1033);  view_1031 = view_1033 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_924: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_922, [1, 0]);  permute_922 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    view_1034: "f32[128, 2048]" = torch.ops.aten.view.default(view_1029, [128, 2048]);  view_1029 = None
    permute_925: "f32[2048, 128]" = torch.ops.aten.permute.default(view_1034, [1, 0])
    mm_313: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_925, view_90);  permute_925 = view_90 = None
    permute_926: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_313, [1, 0]);  mm_313 = None
    permute_927: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_44, [1, 0]);  permute_44 = None
    mm_314: "f32[128, 2048]" = torch.ops.aten.mm.default(view_1034, permute_927);  view_1034 = permute_927 = None
    view_1035: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_314, [1, 128, 2048]);  mm_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    add_354: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_353, view_1035);  add_353 = view_1035 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_928: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_926, [1, 0]);  permute_926 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    sub_234: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_32, getitem_17);  add_32 = getitem_17 = None
    mul_694: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_234, rsqrt_8);  sub_234 = None
    mul_695: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_354, primals_55);  primals_55 = None
    mul_696: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_695, 2048)
    sum_265: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_695, [2], True)
    mul_697: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_695, mul_694);  mul_695 = None
    sum_266: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_697, [2], True);  mul_697 = None
    mul_698: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_694, sum_266);  sum_266 = None
    sub_235: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_696, sum_265);  mul_696 = sum_265 = None
    sub_236: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_235, mul_698);  sub_235 = mul_698 = None
    div_64: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 2048);  rsqrt_8 = None
    mul_699: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_64, sub_236);  div_64 = sub_236 = None
    mul_700: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_354, mul_694);  mul_694 = None
    sum_267: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_700, [0, 1]);  mul_700 = None
    sum_268: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_354, [0, 1]);  add_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    add_355: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_350, mul_699);  add_350 = mul_699 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_1036: "f32[128, 2048]" = torch.ops.aten.view.default(add_355, [128, 2048])
    permute_929: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_43, [1, 0]);  permute_43 = None
    mm_315: "f32[128, 8192]" = torch.ops.aten.mm.default(view_1036, permute_929);  permute_929 = None
    permute_930: "f32[2048, 128]" = torch.ops.aten.permute.default(view_1036, [1, 0])
    mm_316: "f32[2048, 8192]" = torch.ops.aten.mm.default(permute_930, view_88);  permute_930 = view_88 = None
    permute_931: "f32[8192, 2048]" = torch.ops.aten.permute.default(mm_316, [1, 0]);  mm_316 = None
    sum_269: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_1036, [0], True);  view_1036 = None
    view_1037: "f32[2048]" = torch.ops.aten.view.default(sum_269, [2048]);  sum_269 = None
    permute_932: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_931, [1, 0]);  permute_931 = None
    view_1038: "f32[1, 128, 8192]" = torch.ops.aten.view.default(mm_315, [1, 128, 8192]);  mm_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_701: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_1038, mul_28);  mul_28 = None
    mul_702: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_1038, add_31);  view_1038 = add_31 = None
    alias_88: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(alias_7);  alias_7 = None
    mul_703: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(alias_88, alias_88);  alias_88 = None
    sub_237: "f32[1, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_703);  mul_703 = None
    mul_704: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_701, sub_237);  mul_701 = sub_237 = None
    mul_705: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_704, 0.7978845608028654);  mul_704 = None
    mul_706: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_705, 0.044715)
    pow_45: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_87, 2.0);  view_87 = None
    mul_707: "f32[1, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_45, 3.0);  pow_45 = None
    mul_708: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_706, mul_707);  mul_706 = mul_707 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_356: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(mul_705, mul_708);  mul_705 = mul_708 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_709: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_702, 0.5);  mul_702 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_357: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(add_356, mul_709);  add_356 = mul_709 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_1039: "f32[128, 8192]" = torch.ops.aten.view.default(add_357, [128, 8192]);  add_357 = None
    permute_933: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
    mm_317: "f32[128, 2048]" = torch.ops.aten.mm.default(view_1039, permute_933);  permute_933 = None
    permute_934: "f32[8192, 128]" = torch.ops.aten.permute.default(view_1039, [1, 0])
    mm_318: "f32[8192, 2048]" = torch.ops.aten.mm.default(permute_934, view_86);  permute_934 = view_86 = None
    permute_935: "f32[2048, 8192]" = torch.ops.aten.permute.default(mm_318, [1, 0]);  mm_318 = None
    sum_270: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_1039, [0], True);  view_1039 = None
    view_1040: "f32[8192]" = torch.ops.aten.view.default(sum_270, [8192]);  sum_270 = None
    permute_936: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_935, [1, 0]);  permute_935 = None
    view_1041: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_317, [1, 128, 2048]);  mm_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    sub_238: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_27, getitem_15);  add_27 = getitem_15 = None
    mul_710: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_238, rsqrt_7);  sub_238 = None
    mul_711: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_1041, primals_49);  primals_49 = None
    mul_712: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_711, 2048)
    sum_271: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_711, [2], True)
    mul_713: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_711, mul_710);  mul_711 = None
    sum_272: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_713, [2], True);  mul_713 = None
    mul_714: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_710, sum_272);  sum_272 = None
    sub_239: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_712, sum_271);  mul_712 = sum_271 = None
    sub_240: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_239, mul_714);  sub_239 = mul_714 = None
    div_65: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 2048);  rsqrt_7 = None
    mul_715: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_65, sub_240);  div_65 = sub_240 = None
    mul_716: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_1041, mul_710);  mul_710 = None
    sum_273: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_716, [0, 1]);  mul_716 = None
    sum_274: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_1041, [0, 1]);  view_1041 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    add_358: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_355, mul_715);  add_355 = mul_715 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_1042: "f32[128, 2048]" = torch.ops.aten.view.default(add_358, [128, 2048])
    permute_937: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_41, [1, 0]);  permute_41 = None
    mm_319: "f32[128, 2048]" = torch.ops.aten.mm.default(view_1042, permute_937);  permute_937 = None
    permute_938: "f32[2048, 128]" = torch.ops.aten.permute.default(view_1042, [1, 0])
    mm_320: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_938, view_84);  permute_938 = view_84 = None
    permute_939: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_320, [1, 0]);  mm_320 = None
    sum_275: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_1042, [0], True);  view_1042 = None
    view_1043: "f32[2048]" = torch.ops.aten.view.default(sum_275, [2048]);  sum_275 = None
    permute_940: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_939, [1, 0]);  permute_939 = None
    view_1044: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_319, [1, 128, 2048]);  mm_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_1045: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_1044, [1, 128, 16, 128]);  view_1044 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_941: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_1045, [0, 2, 1, 3]);  view_1045 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_1046: "f32[16, 128, 128]" = torch.ops.aten.view.default(permute_941, [16, 128, 128]);  permute_941 = None
    permute_942: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_80, [0, 2, 1]);  view_80 = None
    bmm_128: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_942, view_1046);  permute_942 = None
    permute_943: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_81, [0, 2, 1]);  view_81 = None
    bmm_129: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_1046, permute_943);  view_1046 = permute_943 = None
    view_1047: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_128, [1, 16, 128, 128]);  bmm_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    add_359: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_9, view_1047);  tangents_9 = view_1047 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_1048: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_129, [1, 16, 128, 128]);  bmm_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_89: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(alias_6);  alias_6 = None
    mul_717: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_1048, alias_89);  view_1048 = None
    sum_276: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_717, [-1], True)
    mul_718: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_89, sum_276);  alias_89 = sum_276 = None
    sub_241: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_717, mul_718);  mul_717 = mul_718 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    scalar_tensor_20: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_44: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_16, sub_241, scalar_tensor_20);  slice_16 = sub_241 = scalar_tensor_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_1049: "f32[16, 128, 128]" = torch.ops.aten.view.default(where_44, [16, 128, 128]);  where_44 = None
    permute_944: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_77, [0, 2, 1]);  view_77 = None
    bmm_130: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_944, view_1049);  permute_944 = None
    permute_945: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_78, [0, 2, 1]);  view_78 = None
    bmm_131: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_1049, permute_945);  view_1049 = permute_945 = None
    view_1050: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_130, [1, 16, 128, 128]);  bmm_130 = None
    view_1051: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_131, [1, 16, 128, 128]);  bmm_131 = None
    permute_946: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_1050, [0, 1, 3, 2]);  view_1050 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_360: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_8, permute_946);  tangents_8 = permute_946 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_947: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_359, [0, 2, 1, 3]);  add_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_157: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_947, memory_format = torch.contiguous_format);  permute_947 = None
    view_1052: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_157, [1, 128, 2048]);  clone_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_948: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_360, [0, 2, 1, 3]);  add_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_158: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_948, memory_format = torch.contiguous_format);  permute_948 = None
    view_1053: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_158, [1, 128, 2048]);  clone_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_949: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_1051, [0, 2, 1, 3]);  view_1051 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_159: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_949, memory_format = torch.contiguous_format);  permute_949 = None
    view_1054: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_159, [1, 128, 2048]);  clone_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    view_1055: "f32[128, 2048]" = torch.ops.aten.view.default(view_1052, [128, 2048]);  view_1052 = None
    permute_950: "f32[2048, 128]" = torch.ops.aten.permute.default(view_1055, [1, 0])
    mm_321: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_950, view_72);  permute_950 = view_72 = None
    permute_951: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_321, [1, 0]);  mm_321 = None
    permute_952: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_35, [1, 0]);  permute_35 = None
    mm_322: "f32[128, 2048]" = torch.ops.aten.mm.default(view_1055, permute_952);  view_1055 = permute_952 = None
    view_1056: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_322, [1, 128, 2048]);  mm_322 = None
    permute_953: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_951, [1, 0]);  permute_951 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    view_1057: "f32[128, 2048]" = torch.ops.aten.view.default(view_1053, [128, 2048]);  view_1053 = None
    permute_954: "f32[2048, 128]" = torch.ops.aten.permute.default(view_1057, [1, 0])
    mm_323: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_954, view_70);  permute_954 = view_70 = None
    permute_955: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_323, [1, 0]);  mm_323 = None
    permute_956: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None
    mm_324: "f32[128, 2048]" = torch.ops.aten.mm.default(view_1057, permute_956);  view_1057 = permute_956 = None
    view_1058: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_324, [1, 128, 2048]);  mm_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    add_361: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_1056, view_1058);  view_1056 = view_1058 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_957: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_955, [1, 0]);  permute_955 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    view_1059: "f32[128, 2048]" = torch.ops.aten.view.default(view_1054, [128, 2048]);  view_1054 = None
    permute_958: "f32[2048, 128]" = torch.ops.aten.permute.default(view_1059, [1, 0])
    mm_325: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_958, view_68);  permute_958 = view_68 = None
    permute_959: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_325, [1, 0]);  mm_325 = None
    permute_960: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
    mm_326: "f32[128, 2048]" = torch.ops.aten.mm.default(view_1059, permute_960);  view_1059 = permute_960 = None
    view_1060: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_326, [1, 128, 2048]);  mm_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    add_362: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_361, view_1060);  add_361 = view_1060 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_961: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_959, [1, 0]);  permute_959 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    sub_242: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_24, getitem_13);  add_24 = getitem_13 = None
    mul_719: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_242, rsqrt_6);  sub_242 = None
    mul_720: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_362, primals_42);  primals_42 = None
    mul_721: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_720, 2048)
    sum_277: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_720, [2], True)
    mul_722: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_720, mul_719);  mul_720 = None
    sum_278: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_722, [2], True);  mul_722 = None
    mul_723: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_719, sum_278);  sum_278 = None
    sub_243: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_721, sum_277);  mul_721 = sum_277 = None
    sub_244: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_243, mul_723);  sub_243 = mul_723 = None
    div_66: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 2048);  rsqrt_6 = None
    mul_724: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_66, sub_244);  div_66 = sub_244 = None
    mul_725: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_362, mul_719);  mul_719 = None
    sum_279: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_725, [0, 1]);  mul_725 = None
    sum_280: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_362, [0, 1]);  add_362 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    add_363: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_358, mul_724);  add_358 = mul_724 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_1061: "f32[128, 2048]" = torch.ops.aten.view.default(add_363, [128, 2048])
    permute_962: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
    mm_327: "f32[128, 8192]" = torch.ops.aten.mm.default(view_1061, permute_962);  permute_962 = None
    permute_963: "f32[2048, 128]" = torch.ops.aten.permute.default(view_1061, [1, 0])
    mm_328: "f32[2048, 8192]" = torch.ops.aten.mm.default(permute_963, view_66);  permute_963 = view_66 = None
    permute_964: "f32[8192, 2048]" = torch.ops.aten.permute.default(mm_328, [1, 0]);  mm_328 = None
    sum_281: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_1061, [0], True);  view_1061 = None
    view_1062: "f32[2048]" = torch.ops.aten.view.default(sum_281, [2048]);  sum_281 = None
    permute_965: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_964, [1, 0]);  permute_964 = None
    view_1063: "f32[1, 128, 8192]" = torch.ops.aten.view.default(mm_327, [1, 128, 8192]);  mm_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_726: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_1063, mul_20);  mul_20 = None
    mul_727: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_1063, add_23);  view_1063 = add_23 = None
    alias_90: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(alias_5);  alias_5 = None
    mul_728: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(alias_90, alias_90);  alias_90 = None
    sub_245: "f32[1, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_728);  mul_728 = None
    mul_729: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_726, sub_245);  mul_726 = sub_245 = None
    mul_730: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_729, 0.7978845608028654);  mul_729 = None
    mul_731: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_730, 0.044715)
    pow_46: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_65, 2.0);  view_65 = None
    mul_732: "f32[1, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_46, 3.0);  pow_46 = None
    mul_733: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_731, mul_732);  mul_731 = mul_732 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_364: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(mul_730, mul_733);  mul_730 = mul_733 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_734: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_727, 0.5);  mul_727 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_365: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(add_364, mul_734);  add_364 = mul_734 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_1064: "f32[128, 8192]" = torch.ops.aten.view.default(add_365, [128, 8192]);  add_365 = None
    permute_966: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_31, [1, 0]);  permute_31 = None
    mm_329: "f32[128, 2048]" = torch.ops.aten.mm.default(view_1064, permute_966);  permute_966 = None
    permute_967: "f32[8192, 128]" = torch.ops.aten.permute.default(view_1064, [1, 0])
    mm_330: "f32[8192, 2048]" = torch.ops.aten.mm.default(permute_967, view_64);  permute_967 = view_64 = None
    permute_968: "f32[2048, 8192]" = torch.ops.aten.permute.default(mm_330, [1, 0]);  mm_330 = None
    sum_282: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_1064, [0], True);  view_1064 = None
    view_1065: "f32[8192]" = torch.ops.aten.view.default(sum_282, [8192]);  sum_282 = None
    permute_969: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_968, [1, 0]);  permute_968 = None
    view_1066: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_329, [1, 128, 2048]);  mm_329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    sub_246: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_19, getitem_11);  add_19 = getitem_11 = None
    mul_735: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_246, rsqrt_5);  sub_246 = None
    mul_736: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_1066, primals_36);  primals_36 = None
    mul_737: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_736, 2048)
    sum_283: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_736, [2], True)
    mul_738: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_736, mul_735);  mul_736 = None
    sum_284: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_738, [2], True);  mul_738 = None
    mul_739: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_735, sum_284);  sum_284 = None
    sub_247: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_737, sum_283);  mul_737 = sum_283 = None
    sub_248: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_247, mul_739);  sub_247 = mul_739 = None
    div_67: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 2048);  rsqrt_5 = None
    mul_740: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_67, sub_248);  div_67 = sub_248 = None
    mul_741: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_1066, mul_735);  mul_735 = None
    sum_285: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_741, [0, 1]);  mul_741 = None
    sum_286: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_1066, [0, 1]);  view_1066 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    add_366: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_363, mul_740);  add_363 = mul_740 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_1067: "f32[128, 2048]" = torch.ops.aten.view.default(add_366, [128, 2048])
    permute_970: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_30, [1, 0]);  permute_30 = None
    mm_331: "f32[128, 2048]" = torch.ops.aten.mm.default(view_1067, permute_970);  permute_970 = None
    permute_971: "f32[2048, 128]" = torch.ops.aten.permute.default(view_1067, [1, 0])
    mm_332: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_971, view_62);  permute_971 = view_62 = None
    permute_972: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_332, [1, 0]);  mm_332 = None
    sum_287: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_1067, [0], True);  view_1067 = None
    view_1068: "f32[2048]" = torch.ops.aten.view.default(sum_287, [2048]);  sum_287 = None
    permute_973: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_972, [1, 0]);  permute_972 = None
    view_1069: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_331, [1, 128, 2048]);  mm_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_1070: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_1069, [1, 128, 16, 128]);  view_1069 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_974: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_1070, [0, 2, 1, 3]);  view_1070 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_1071: "f32[16, 128, 128]" = torch.ops.aten.view.default(permute_974, [16, 128, 128]);  permute_974 = None
    permute_975: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_58, [0, 2, 1]);  view_58 = None
    bmm_132: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_975, view_1071);  permute_975 = None
    permute_976: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_59, [0, 2, 1]);  view_59 = None
    bmm_133: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_1071, permute_976);  view_1071 = permute_976 = None
    view_1072: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_132, [1, 16, 128, 128]);  bmm_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    add_367: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_7, view_1072);  tangents_7 = view_1072 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_1073: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_133, [1, 16, 128, 128]);  bmm_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_91: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(alias_4);  alias_4 = None
    mul_742: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_1073, alias_91);  view_1073 = None
    sum_288: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_742, [-1], True)
    mul_743: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_91, sum_288);  alias_91 = sum_288 = None
    sub_249: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_742, mul_743);  mul_742 = mul_743 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    scalar_tensor_21: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_45: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_12, sub_249, scalar_tensor_21);  slice_12 = sub_249 = scalar_tensor_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_1074: "f32[16, 128, 128]" = torch.ops.aten.view.default(where_45, [16, 128, 128]);  where_45 = None
    permute_977: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_55, [0, 2, 1]);  view_55 = None
    bmm_134: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_977, view_1074);  permute_977 = None
    permute_978: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_56, [0, 2, 1]);  view_56 = None
    bmm_135: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_1074, permute_978);  view_1074 = permute_978 = None
    view_1075: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_134, [1, 16, 128, 128]);  bmm_134 = None
    view_1076: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_135, [1, 16, 128, 128]);  bmm_135 = None
    permute_979: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_1075, [0, 1, 3, 2]);  view_1075 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_368: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_6, permute_979);  tangents_6 = permute_979 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_980: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_367, [0, 2, 1, 3]);  add_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_160: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_980, memory_format = torch.contiguous_format);  permute_980 = None
    view_1077: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_160, [1, 128, 2048]);  clone_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_981: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_368, [0, 2, 1, 3]);  add_368 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_161: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_981, memory_format = torch.contiguous_format);  permute_981 = None
    view_1078: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_161, [1, 128, 2048]);  clone_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_982: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_1076, [0, 2, 1, 3]);  view_1076 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_162: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_982, memory_format = torch.contiguous_format);  permute_982 = None
    view_1079: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_162, [1, 128, 2048]);  clone_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    view_1080: "f32[128, 2048]" = torch.ops.aten.view.default(view_1077, [128, 2048]);  view_1077 = None
    permute_983: "f32[2048, 128]" = torch.ops.aten.permute.default(view_1080, [1, 0])
    mm_333: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_983, view_50);  permute_983 = view_50 = None
    permute_984: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_333, [1, 0]);  mm_333 = None
    permute_985: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_24, [1, 0]);  permute_24 = None
    mm_334: "f32[128, 2048]" = torch.ops.aten.mm.default(view_1080, permute_985);  view_1080 = permute_985 = None
    view_1081: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_334, [1, 128, 2048]);  mm_334 = None
    permute_986: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_984, [1, 0]);  permute_984 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    view_1082: "f32[128, 2048]" = torch.ops.aten.view.default(view_1078, [128, 2048]);  view_1078 = None
    permute_987: "f32[2048, 128]" = torch.ops.aten.permute.default(view_1082, [1, 0])
    mm_335: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_987, view_48);  permute_987 = view_48 = None
    permute_988: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_335, [1, 0]);  mm_335 = None
    permute_989: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None
    mm_336: "f32[128, 2048]" = torch.ops.aten.mm.default(view_1082, permute_989);  view_1082 = permute_989 = None
    view_1083: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_336, [1, 128, 2048]);  mm_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    add_369: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_1081, view_1083);  view_1081 = view_1083 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_990: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_988, [1, 0]);  permute_988 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    view_1084: "f32[128, 2048]" = torch.ops.aten.view.default(view_1079, [128, 2048]);  view_1079 = None
    permute_991: "f32[2048, 128]" = torch.ops.aten.permute.default(view_1084, [1, 0])
    mm_337: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_991, view_46);  permute_991 = view_46 = None
    permute_992: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_337, [1, 0]);  mm_337 = None
    permute_993: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
    mm_338: "f32[128, 2048]" = torch.ops.aten.mm.default(view_1084, permute_993);  view_1084 = permute_993 = None
    view_1085: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_338, [1, 128, 2048]);  mm_338 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    add_370: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_369, view_1085);  add_369 = view_1085 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_994: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_992, [1, 0]);  permute_992 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    sub_250: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_16, getitem_9);  add_16 = getitem_9 = None
    mul_744: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_250, rsqrt_4);  sub_250 = None
    mul_745: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_370, primals_29);  primals_29 = None
    mul_746: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_745, 2048)
    sum_289: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_745, [2], True)
    mul_747: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_745, mul_744);  mul_745 = None
    sum_290: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_747, [2], True);  mul_747 = None
    mul_748: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_744, sum_290);  sum_290 = None
    sub_251: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_746, sum_289);  mul_746 = sum_289 = None
    sub_252: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_251, mul_748);  sub_251 = mul_748 = None
    div_68: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 2048);  rsqrt_4 = None
    mul_749: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_68, sub_252);  div_68 = sub_252 = None
    mul_750: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_370, mul_744);  mul_744 = None
    sum_291: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_750, [0, 1]);  mul_750 = None
    sum_292: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_370, [0, 1]);  add_370 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    add_371: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_366, mul_749);  add_366 = mul_749 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_1086: "f32[128, 2048]" = torch.ops.aten.view.default(add_371, [128, 2048])
    permute_995: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
    mm_339: "f32[128, 8192]" = torch.ops.aten.mm.default(view_1086, permute_995);  permute_995 = None
    permute_996: "f32[2048, 128]" = torch.ops.aten.permute.default(view_1086, [1, 0])
    mm_340: "f32[2048, 8192]" = torch.ops.aten.mm.default(permute_996, view_44);  permute_996 = view_44 = None
    permute_997: "f32[8192, 2048]" = torch.ops.aten.permute.default(mm_340, [1, 0]);  mm_340 = None
    sum_293: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_1086, [0], True);  view_1086 = None
    view_1087: "f32[2048]" = torch.ops.aten.view.default(sum_293, [2048]);  sum_293 = None
    permute_998: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_997, [1, 0]);  permute_997 = None
    view_1088: "f32[1, 128, 8192]" = torch.ops.aten.view.default(mm_339, [1, 128, 8192]);  mm_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_751: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_1088, mul_12);  mul_12 = None
    mul_752: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_1088, add_15);  view_1088 = add_15 = None
    alias_92: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(alias_3);  alias_3 = None
    mul_753: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(alias_92, alias_92);  alias_92 = None
    sub_253: "f32[1, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_753);  mul_753 = None
    mul_754: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_751, sub_253);  mul_751 = sub_253 = None
    mul_755: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_754, 0.7978845608028654);  mul_754 = None
    mul_756: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_755, 0.044715)
    pow_47: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_43, 2.0);  view_43 = None
    mul_757: "f32[1, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_47, 3.0);  pow_47 = None
    mul_758: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_756, mul_757);  mul_756 = mul_757 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_372: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(mul_755, mul_758);  mul_755 = mul_758 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_759: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_752, 0.5);  mul_752 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_373: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(add_372, mul_759);  add_372 = mul_759 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_1089: "f32[128, 8192]" = torch.ops.aten.view.default(add_373, [128, 8192]);  add_373 = None
    permute_999: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
    mm_341: "f32[128, 2048]" = torch.ops.aten.mm.default(view_1089, permute_999);  permute_999 = None
    permute_1000: "f32[8192, 128]" = torch.ops.aten.permute.default(view_1089, [1, 0])
    mm_342: "f32[8192, 2048]" = torch.ops.aten.mm.default(permute_1000, view_42);  permute_1000 = view_42 = None
    permute_1001: "f32[2048, 8192]" = torch.ops.aten.permute.default(mm_342, [1, 0]);  mm_342 = None
    sum_294: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_1089, [0], True);  view_1089 = None
    view_1090: "f32[8192]" = torch.ops.aten.view.default(sum_294, [8192]);  sum_294 = None
    permute_1002: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_1001, [1, 0]);  permute_1001 = None
    view_1091: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_341, [1, 128, 2048]);  mm_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    sub_254: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_11, getitem_7);  add_11 = getitem_7 = None
    mul_760: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_254, rsqrt_3);  sub_254 = None
    mul_761: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_1091, primals_23);  primals_23 = None
    mul_762: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_761, 2048)
    sum_295: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_761, [2], True)
    mul_763: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_761, mul_760);  mul_761 = None
    sum_296: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_763, [2], True);  mul_763 = None
    mul_764: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_760, sum_296);  sum_296 = None
    sub_255: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_762, sum_295);  mul_762 = sum_295 = None
    sub_256: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_255, mul_764);  sub_255 = mul_764 = None
    div_69: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 2048);  rsqrt_3 = None
    mul_765: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_69, sub_256);  div_69 = sub_256 = None
    mul_766: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_1091, mul_760);  mul_760 = None
    sum_297: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_766, [0, 1]);  mul_766 = None
    sum_298: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_1091, [0, 1]);  view_1091 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    add_374: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_371, mul_765);  add_371 = mul_765 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_1092: "f32[128, 2048]" = torch.ops.aten.view.default(add_374, [128, 2048])
    permute_1003: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_19, [1, 0]);  permute_19 = None
    mm_343: "f32[128, 2048]" = torch.ops.aten.mm.default(view_1092, permute_1003);  permute_1003 = None
    permute_1004: "f32[2048, 128]" = torch.ops.aten.permute.default(view_1092, [1, 0])
    mm_344: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_1004, view_40);  permute_1004 = view_40 = None
    permute_1005: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_344, [1, 0]);  mm_344 = None
    sum_299: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_1092, [0], True);  view_1092 = None
    view_1093: "f32[2048]" = torch.ops.aten.view.default(sum_299, [2048]);  sum_299 = None
    permute_1006: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_1005, [1, 0]);  permute_1005 = None
    view_1094: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_343, [1, 128, 2048]);  mm_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_1095: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_1094, [1, 128, 16, 128]);  view_1094 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_1007: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_1095, [0, 2, 1, 3]);  view_1095 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_1096: "f32[16, 128, 128]" = torch.ops.aten.view.default(permute_1007, [16, 128, 128]);  permute_1007 = None
    permute_1008: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_36, [0, 2, 1]);  view_36 = None
    bmm_136: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_1008, view_1096);  permute_1008 = None
    permute_1009: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_37, [0, 2, 1]);  view_37 = None
    bmm_137: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_1096, permute_1009);  view_1096 = permute_1009 = None
    view_1097: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_136, [1, 16, 128, 128]);  bmm_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    add_375: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_5, view_1097);  tangents_5 = view_1097 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_1098: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_137, [1, 16, 128, 128]);  bmm_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_93: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(alias_2);  alias_2 = None
    mul_767: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_1098, alias_93);  view_1098 = None
    sum_300: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_767, [-1], True)
    mul_768: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_93, sum_300);  alias_93 = sum_300 = None
    sub_257: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_767, mul_768);  mul_767 = mul_768 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    scalar_tensor_22: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_46: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_8, sub_257, scalar_tensor_22);  slice_8 = sub_257 = scalar_tensor_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_1099: "f32[16, 128, 128]" = torch.ops.aten.view.default(where_46, [16, 128, 128]);  where_46 = None
    permute_1010: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_33, [0, 2, 1]);  view_33 = None
    bmm_138: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_1010, view_1099);  permute_1010 = None
    permute_1011: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_34, [0, 2, 1]);  view_34 = None
    bmm_139: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_1099, permute_1011);  view_1099 = permute_1011 = None
    view_1100: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_138, [1, 16, 128, 128]);  bmm_138 = None
    view_1101: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_139, [1, 16, 128, 128]);  bmm_139 = None
    permute_1012: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_1100, [0, 1, 3, 2]);  view_1100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_376: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_4, permute_1012);  tangents_4 = permute_1012 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_1013: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_375, [0, 2, 1, 3]);  add_375 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_163: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_1013, memory_format = torch.contiguous_format);  permute_1013 = None
    view_1102: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_163, [1, 128, 2048]);  clone_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_1014: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_376, [0, 2, 1, 3]);  add_376 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_164: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_1014, memory_format = torch.contiguous_format);  permute_1014 = None
    view_1103: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_164, [1, 128, 2048]);  clone_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_1015: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_1101, [0, 2, 1, 3]);  view_1101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_165: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_1015, memory_format = torch.contiguous_format);  permute_1015 = None
    view_1104: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_165, [1, 128, 2048]);  clone_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    view_1105: "f32[128, 2048]" = torch.ops.aten.view.default(view_1102, [128, 2048]);  view_1102 = None
    permute_1016: "f32[2048, 128]" = torch.ops.aten.permute.default(view_1105, [1, 0])
    mm_345: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_1016, view_28);  permute_1016 = view_28 = None
    permute_1017: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_345, [1, 0]);  mm_345 = None
    permute_1018: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
    mm_346: "f32[128, 2048]" = torch.ops.aten.mm.default(view_1105, permute_1018);  view_1105 = permute_1018 = None
    view_1106: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_346, [1, 128, 2048]);  mm_346 = None
    permute_1019: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_1017, [1, 0]);  permute_1017 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    view_1107: "f32[128, 2048]" = torch.ops.aten.view.default(view_1103, [128, 2048]);  view_1103 = None
    permute_1020: "f32[2048, 128]" = torch.ops.aten.permute.default(view_1107, [1, 0])
    mm_347: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_1020, view_26);  permute_1020 = view_26 = None
    permute_1021: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_347, [1, 0]);  mm_347 = None
    permute_1022: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
    mm_348: "f32[128, 2048]" = torch.ops.aten.mm.default(view_1107, permute_1022);  view_1107 = permute_1022 = None
    view_1108: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_348, [1, 128, 2048]);  mm_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    add_377: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_1106, view_1108);  view_1106 = view_1108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_1023: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_1021, [1, 0]);  permute_1021 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    view_1109: "f32[128, 2048]" = torch.ops.aten.view.default(view_1104, [128, 2048]);  view_1104 = None
    permute_1024: "f32[2048, 128]" = torch.ops.aten.permute.default(view_1109, [1, 0])
    mm_349: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_1024, view_24);  permute_1024 = view_24 = None
    permute_1025: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_349, [1, 0]);  mm_349 = None
    permute_1026: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
    mm_350: "f32[128, 2048]" = torch.ops.aten.mm.default(view_1109, permute_1026);  view_1109 = permute_1026 = None
    view_1110: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_350, [1, 128, 2048]);  mm_350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    add_378: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_377, view_1110);  add_377 = view_1110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_1027: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_1025, [1, 0]);  permute_1025 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    sub_258: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_8, getitem_5);  add_8 = getitem_5 = None
    mul_769: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_258, rsqrt_2);  sub_258 = None
    mul_770: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_378, primals_16);  primals_16 = None
    mul_771: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_770, 2048)
    sum_301: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_770, [2], True)
    mul_772: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_770, mul_769);  mul_770 = None
    sum_302: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_772, [2], True);  mul_772 = None
    mul_773: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_769, sum_302);  sum_302 = None
    sub_259: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_771, sum_301);  mul_771 = sum_301 = None
    sub_260: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_259, mul_773);  sub_259 = mul_773 = None
    div_70: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 2048);  rsqrt_2 = None
    mul_774: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_70, sub_260);  div_70 = sub_260 = None
    mul_775: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_378, mul_769);  mul_769 = None
    sum_303: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_775, [0, 1]);  mul_775 = None
    sum_304: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_378, [0, 1]);  add_378 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    add_379: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_374, mul_774);  add_374 = mul_774 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_1111: "f32[128, 2048]" = torch.ops.aten.view.default(add_379, [128, 2048])
    permute_1028: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
    mm_351: "f32[128, 8192]" = torch.ops.aten.mm.default(view_1111, permute_1028);  permute_1028 = None
    permute_1029: "f32[2048, 128]" = torch.ops.aten.permute.default(view_1111, [1, 0])
    mm_352: "f32[2048, 8192]" = torch.ops.aten.mm.default(permute_1029, view_22);  permute_1029 = view_22 = None
    permute_1030: "f32[8192, 2048]" = torch.ops.aten.permute.default(mm_352, [1, 0]);  mm_352 = None
    sum_305: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_1111, [0], True);  view_1111 = None
    view_1112: "f32[2048]" = torch.ops.aten.view.default(sum_305, [2048]);  sum_305 = None
    permute_1031: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_1030, [1, 0]);  permute_1030 = None
    view_1113: "f32[1, 128, 8192]" = torch.ops.aten.view.default(mm_351, [1, 128, 8192]);  mm_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_776: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_1113, mul_4);  mul_4 = None
    mul_777: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_1113, add_7);  view_1113 = add_7 = None
    alias_94: "f32[1, 128, 8192]" = torch.ops.aten.alias.default(alias_1);  alias_1 = None
    mul_778: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(alias_94, alias_94);  alias_94 = None
    sub_261: "f32[1, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_778);  mul_778 = None
    mul_779: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_776, sub_261);  mul_776 = sub_261 = None
    mul_780: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_779, 0.7978845608028654);  mul_779 = None
    mul_781: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_780, 0.044715)
    pow_48: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_21, 2.0);  view_21 = None
    mul_782: "f32[1, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_48, 3.0);  pow_48 = None
    mul_783: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_781, mul_782);  mul_781 = mul_782 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_380: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(mul_780, mul_783);  mul_780 = mul_783 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_784: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_777, 0.5);  mul_777 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_381: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(add_380, mul_784);  add_380 = mul_784 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_1114: "f32[128, 8192]" = torch.ops.aten.view.default(add_381, [128, 8192]);  add_381 = None
    permute_1032: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
    mm_353: "f32[128, 2048]" = torch.ops.aten.mm.default(view_1114, permute_1032);  permute_1032 = None
    permute_1033: "f32[8192, 128]" = torch.ops.aten.permute.default(view_1114, [1, 0])
    mm_354: "f32[8192, 2048]" = torch.ops.aten.mm.default(permute_1033, view_20);  permute_1033 = view_20 = None
    permute_1034: "f32[2048, 8192]" = torch.ops.aten.permute.default(mm_354, [1, 0]);  mm_354 = None
    sum_306: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_1114, [0], True);  view_1114 = None
    view_1115: "f32[8192]" = torch.ops.aten.view.default(sum_306, [8192]);  sum_306 = None
    permute_1035: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_1034, [1, 0]);  permute_1034 = None
    view_1116: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_353, [1, 128, 2048]);  mm_353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    sub_262: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_3, getitem_3);  add_3 = getitem_3 = None
    mul_785: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_262, rsqrt_1);  sub_262 = None
    mul_786: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_1116, primals_10);  primals_10 = None
    mul_787: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_786, 2048)
    sum_307: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_786, [2], True)
    mul_788: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_786, mul_785);  mul_786 = None
    sum_308: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_788, [2], True);  mul_788 = None
    mul_789: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_785, sum_308);  sum_308 = None
    sub_263: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_787, sum_307);  mul_787 = sum_307 = None
    sub_264: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_263, mul_789);  sub_263 = mul_789 = None
    div_71: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 2048);  rsqrt_1 = None
    mul_790: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_71, sub_264);  div_71 = sub_264 = None
    mul_791: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_1116, mul_785);  mul_785 = None
    sum_309: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_791, [0, 1]);  mul_791 = None
    sum_310: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_1116, [0, 1]);  view_1116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    add_382: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_379, mul_790);  add_379 = mul_790 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_1117: "f32[128, 2048]" = torch.ops.aten.view.default(add_382, [128, 2048])
    permute_1036: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
    mm_355: "f32[128, 2048]" = torch.ops.aten.mm.default(view_1117, permute_1036);  permute_1036 = None
    permute_1037: "f32[2048, 128]" = torch.ops.aten.permute.default(view_1117, [1, 0])
    mm_356: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_1037, view_18);  permute_1037 = view_18 = None
    permute_1038: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_356, [1, 0]);  mm_356 = None
    sum_311: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_1117, [0], True);  view_1117 = None
    view_1118: "f32[2048]" = torch.ops.aten.view.default(sum_311, [2048]);  sum_311 = None
    permute_1039: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_1038, [1, 0]);  permute_1038 = None
    view_1119: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_355, [1, 128, 2048]);  mm_355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_1120: "f32[1, 128, 16, 128]" = torch.ops.aten.view.default(view_1119, [1, 128, 16, 128]);  view_1119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_1040: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_1120, [0, 2, 1, 3]);  view_1120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_1121: "f32[16, 128, 128]" = torch.ops.aten.view.default(permute_1040, [16, 128, 128]);  permute_1040 = None
    permute_1041: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_14, [0, 2, 1]);  view_14 = None
    bmm_140: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_1041, view_1121);  permute_1041 = None
    permute_1042: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_15, [0, 2, 1]);  view_15 = None
    bmm_141: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_1121, permute_1042);  view_1121 = permute_1042 = None
    view_1122: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_140, [1, 16, 128, 128]);  bmm_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    add_383: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_3, view_1122);  tangents_3 = view_1122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    view_1123: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_141, [1, 16, 128, 128]);  bmm_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_95: "f32[1, 16, 128, 128]" = torch.ops.aten.alias.default(alias);  alias = None
    mul_792: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_1123, alias_95);  view_1123 = None
    sum_312: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_792, [-1], True)
    mul_793: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(alias_95, sum_312);  alias_95 = sum_312 = None
    sub_265: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_792, mul_793);  mul_792 = mul_793 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    scalar_tensor_23: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_47: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_4, sub_265, scalar_tensor_23);  slice_4 = sub_265 = scalar_tensor_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_1124: "f32[16, 128, 128]" = torch.ops.aten.view.default(where_47, [16, 128, 128]);  where_47 = None
    permute_1043: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_11, [0, 2, 1]);  view_11 = None
    bmm_142: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(permute_1043, view_1124);  permute_1043 = None
    permute_1044: "f32[16, 128, 128]" = torch.ops.aten.permute.default(view_12, [0, 2, 1]);  view_12 = None
    bmm_143: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_1124, permute_1044);  view_1124 = permute_1044 = None
    view_1125: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_142, [1, 16, 128, 128]);  bmm_142 = None
    view_1126: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_143, [1, 16, 128, 128]);  bmm_143 = None
    permute_1045: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_1125, [0, 1, 3, 2]);  view_1125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_384: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(tangents_2, permute_1045);  tangents_2 = permute_1045 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_1046: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_383, [0, 2, 1, 3]);  add_383 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_166: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_1046, memory_format = torch.contiguous_format);  permute_1046 = None
    view_1127: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_166, [1, 128, 2048]);  clone_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_1047: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(add_384, [0, 2, 1, 3]);  add_384 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_167: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_1047, memory_format = torch.contiguous_format);  permute_1047 = None
    view_1128: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_167, [1, 128, 2048]);  clone_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_1048: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_1126, [0, 2, 1, 3]);  view_1126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    clone_168: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_1048, memory_format = torch.contiguous_format);  permute_1048 = None
    view_1129: "f32[1, 128, 2048]" = torch.ops.aten.view.default(clone_168, [1, 128, 2048]);  clone_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    view_1130: "f32[128, 2048]" = torch.ops.aten.view.default(view_1127, [128, 2048]);  view_1127 = None
    permute_1049: "f32[2048, 128]" = torch.ops.aten.permute.default(view_1130, [1, 0])
    mm_357: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_1049, view_6);  permute_1049 = view_6 = None
    permute_1050: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_357, [1, 0]);  mm_357 = None
    permute_1051: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
    mm_358: "f32[128, 2048]" = torch.ops.aten.mm.default(view_1130, permute_1051);  view_1130 = permute_1051 = None
    view_1131: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_358, [1, 128, 2048]);  mm_358 = None
    permute_1052: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_1050, [1, 0]);  permute_1050 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    view_1132: "f32[128, 2048]" = torch.ops.aten.view.default(view_1128, [128, 2048]);  view_1128 = None
    permute_1053: "f32[2048, 128]" = torch.ops.aten.permute.default(view_1132, [1, 0])
    mm_359: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_1053, view_4);  permute_1053 = view_4 = None
    permute_1054: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_359, [1, 0]);  mm_359 = None
    permute_1055: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
    mm_360: "f32[128, 2048]" = torch.ops.aten.mm.default(view_1132, permute_1055);  view_1132 = permute_1055 = None
    view_1133: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_360, [1, 128, 2048]);  mm_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    add_385: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_1131, view_1133);  view_1131 = view_1133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    permute_1056: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_1054, [1, 0]);  permute_1054 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    view_1134: "f32[128, 2048]" = torch.ops.aten.view.default(view_1129, [128, 2048]);  view_1129 = None
    permute_1057: "f32[2048, 128]" = torch.ops.aten.permute.default(view_1134, [1, 0])
    mm_361: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_1057, view_2);  permute_1057 = view_2 = None
    permute_1058: "f32[2048, 2048]" = torch.ops.aten.permute.default(mm_361, [1, 0]);  mm_361 = None
    permute_1059: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
    mm_362: "f32[128, 2048]" = torch.ops.aten.mm.default(view_1134, permute_1059);  view_1134 = permute_1059 = None
    view_1135: "f32[1, 128, 2048]" = torch.ops.aten.view.default(mm_362, [1, 128, 2048]);  mm_362 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    add_386: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_385, view_1135);  add_385 = view_1135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    permute_1060: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_1058, [1, 0]);  permute_1058 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    sub_266: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(clone, getitem_1);  clone = getitem_1 = None
    mul_794: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_266, rsqrt);  sub_266 = None
    mul_795: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_386, primals_3);  primals_3 = None
    mul_796: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_795, 2048)
    sum_313: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_795, [2], True)
    mul_797: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_795, mul_794);  mul_795 = None
    sum_314: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_797, [2], True);  mul_797 = None
    mul_798: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_794, sum_314);  sum_314 = None
    sub_267: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_796, sum_313);  mul_796 = sum_313 = None
    sub_268: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_267, mul_798);  sub_267 = mul_798 = None
    div_72: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt, 2048);  rsqrt = None
    mul_799: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_72, sub_268);  div_72 = sub_268 = None
    mul_800: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_386, mul_794);  mul_794 = None
    sum_315: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_800, [0, 1]);  mul_800 = None
    sum_316: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_386, [0, 1]);  add_386 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    add_387: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_382, mul_799);  add_382 = mul_799 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:583, code: position_embeds = self.wpe(position_ids)
    eq_1: "b8[1, 128]" = torch.ops.aten.eq.Scalar(view_1, -1)
    unsqueeze_1: "b8[1, 128, 1]" = torch.ops.aten.unsqueeze.default(eq_1, -1);  eq_1 = None
    scalar_tensor_24: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_48: "f32[1, 128, 2048]" = torch.ops.aten.where.self(unsqueeze_1, scalar_tensor_24, add_387);  unsqueeze_1 = scalar_tensor_24 = None
    full_1: "f32[2048, 2048]" = torch.ops.aten.full.default([2048, 2048], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    _unsafe_index_put: "f32[2048, 2048]" = torch.ops.aten._unsafe_index_put.default(full_1, [view_1], where_48, True);  full_1 = view_1 = where_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:582, code: inputs_embeds = self.wte(input_ids)
    eq_2: "b8[1, 128]" = torch.ops.aten.eq.Scalar(view, -1)
    unsqueeze_2: "b8[1, 128, 1]" = torch.ops.aten.unsqueeze.default(eq_2, -1);  eq_2 = None
    scalar_tensor_25: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_49: "f32[1, 128, 2048]" = torch.ops.aten.where.self(unsqueeze_2, scalar_tensor_25, add_387);  unsqueeze_2 = scalar_tensor_25 = add_387 = None
    full_2: "f32[50257, 2048]" = torch.ops.aten.full.default([50257, 2048], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    _unsafe_index_put_1: "f32[50257, 2048]" = torch.ops.aten._unsafe_index_put.default(full_2, [view], where_49, True);  full_2 = view = where_49 = None
    return pytree.tree_unflatten([view_530, permute_4, permute_5, permute_15, permute_16, permute_26, permute_27, permute_37, permute_38, permute_48, permute_49, permute_59, permute_60, permute_70, permute_71, permute_81, permute_82, permute_92, permute_93, permute_103, permute_104, permute_114, permute_115, permute_125, permute_126, permute_136, permute_137, permute_147, permute_148, permute_158, permute_159, permute_169, permute_170, permute_180, permute_181, permute_191, permute_192, permute_202, permute_203, permute_213, permute_214, permute_224, permute_225, permute_235, permute_236, permute_246, permute_247, permute_257, permute_258, index, _unsafe_index_put_1, _unsafe_index_put, sum_315, sum_316, permute_1060, permute_1056, permute_1052, permute_1039, view_1118, sum_309, sum_310, permute_1035, view_1115, permute_1031, view_1112, sum_303, sum_304, permute_1027, permute_1023, permute_1019, permute_1006, view_1093, sum_297, sum_298, permute_1002, view_1090, permute_998, view_1087, sum_291, sum_292, permute_994, permute_990, permute_986, permute_973, view_1068, sum_285, sum_286, permute_969, view_1065, permute_965, view_1062, sum_279, sum_280, permute_961, permute_957, permute_953, permute_940, view_1043, sum_273, sum_274, permute_936, view_1040, permute_932, view_1037, sum_267, sum_268, permute_928, permute_924, permute_920, permute_907, view_1018, sum_261, sum_262, permute_903, view_1015, permute_899, view_1012, sum_255, sum_256, permute_895, permute_891, permute_887, permute_874, view_993, sum_249, sum_250, permute_870, view_990, permute_866, view_987, sum_243, sum_244, permute_862, permute_858, permute_854, permute_841, view_968, sum_237, sum_238, permute_837, view_965, permute_833, view_962, sum_231, sum_232, permute_829, permute_825, permute_821, permute_808, view_943, sum_225, sum_226, permute_804, view_940, permute_800, view_937, sum_219, sum_220, permute_796, permute_792, permute_788, permute_775, view_918, sum_213, sum_214, permute_771, view_915, permute_767, view_912, sum_207, sum_208, permute_763, permute_759, permute_755, permute_742, view_893, sum_201, sum_202, permute_738, view_890, permute_734, view_887, sum_195, sum_196, permute_730, permute_726, permute_722, permute_709, view_868, sum_189, sum_190, permute_705, view_865, permute_701, view_862, sum_183, sum_184, permute_697, permute_693, permute_689, permute_676, view_843, sum_177, sum_178, permute_672, view_840, permute_668, view_837, sum_171, sum_172, permute_664, permute_660, permute_656, permute_643, view_818, sum_165, sum_166, permute_639, view_815, permute_635, view_812, sum_159, sum_160, permute_631, permute_627, permute_623, permute_610, view_793, sum_153, sum_154, permute_606, view_790, permute_602, view_787, sum_147, sum_148, permute_598, permute_594, permute_590, permute_577, view_768, sum_141, sum_142, permute_573, view_765, permute_569, view_762, sum_135, sum_136, permute_565, permute_561, permute_557, permute_544, view_743, sum_129, sum_130, permute_540, view_740, permute_536, view_737, sum_123, sum_124, permute_532, permute_528, permute_524, permute_511, view_718, sum_117, sum_118, permute_507, view_715, permute_503, view_712, sum_111, sum_112, permute_499, permute_495, permute_491, permute_478, view_693, sum_105, sum_106, permute_474, view_690, permute_470, view_687, sum_99, sum_100, permute_466, permute_462, permute_458, permute_445, view_668, sum_93, sum_94, permute_441, view_665, permute_437, view_662, sum_87, sum_88, permute_433, permute_429, permute_425, permute_412, view_643, sum_81, sum_82, permute_408, view_640, permute_404, view_637, sum_75, sum_76, permute_400, permute_396, permute_392, permute_379, view_618, sum_69, sum_70, permute_375, view_615, permute_371, view_612, sum_63, sum_64, permute_367, permute_363, permute_359, permute_346, view_593, sum_57, sum_58, permute_342, view_590, permute_338, view_587, sum_51, sum_52, permute_334, permute_330, permute_326, permute_313, view_568, sum_45, sum_46, permute_309, view_565, permute_305, view_562, sum_39, sum_40, permute_301, permute_297, permute_293, permute_280, view_543, sum_33, sum_34, permute_276, view_540, permute_272, view_537, sum_27, sum_28, permute_268, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None], self._out_spec)
    