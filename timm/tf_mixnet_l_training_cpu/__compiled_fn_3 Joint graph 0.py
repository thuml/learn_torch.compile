from __future__ import annotations



def forward(self, primals, tangents):
    primals_1: "f32[32, 3, 3, 3]"; primals_2: "f32[32]"; primals_3: "f32[32]"; primals_4: "f32[32]"; primals_5: "f32[32]"; primals_6: "f32[32]"; primals_7: "f32[32]"; primals_8: "f32[192]"; primals_9: "f32[192]"; primals_10: "f32[64, 1, 3, 3]"; primals_11: "f32[64, 1, 5, 5]"; primals_12: "f32[64, 1, 7, 7]"; primals_13: "f32[192]"; primals_14: "f32[192]"; primals_15: "f32[40]"; primals_16: "f32[40]"; primals_17: "f32[120]"; primals_18: "f32[120]"; primals_19: "f32[120]"; primals_20: "f32[120]"; primals_21: "f32[40]"; primals_22: "f32[40]"; primals_23: "f32[240]"; primals_24: "f32[240]"; primals_25: "f32[60, 1, 3, 3]"; primals_26: "f32[60, 1, 5, 5]"; primals_27: "f32[60, 1, 7, 7]"; primals_28: "f32[60, 1, 9, 9]"; primals_29: "f32[240]"; primals_30: "f32[240]"; primals_31: "f32[56]"; primals_32: "f32[56]"; primals_33: "f32[336]"; primals_34: "f32[336]"; primals_35: "f32[336]"; primals_36: "f32[336]"; primals_37: "f32[56]"; primals_38: "f32[56]"; primals_39: "f32[336]"; primals_40: "f32[336]"; primals_41: "f32[336]"; primals_42: "f32[336]"; primals_43: "f32[56]"; primals_44: "f32[56]"; primals_45: "f32[336]"; primals_46: "f32[336]"; primals_47: "f32[336]"; primals_48: "f32[336]"; primals_49: "f32[56]"; primals_50: "f32[56]"; primals_51: "f32[336]"; primals_52: "f32[336]"; primals_53: "f32[112, 1, 3, 3]"; primals_54: "f32[112, 1, 5, 5]"; primals_55: "f32[112, 1, 7, 7]"; primals_56: "f32[336]"; primals_57: "f32[336]"; primals_58: "f32[104]"; primals_59: "f32[104]"; primals_60: "f32[624]"; primals_61: "f32[624]"; primals_62: "f32[624]"; primals_63: "f32[624]"; primals_64: "f32[104]"; primals_65: "f32[104]"; primals_66: "f32[624]"; primals_67: "f32[624]"; primals_68: "f32[624]"; primals_69: "f32[624]"; primals_70: "f32[104]"; primals_71: "f32[104]"; primals_72: "f32[624]"; primals_73: "f32[624]"; primals_74: "f32[624]"; primals_75: "f32[624]"; primals_76: "f32[104]"; primals_77: "f32[104]"; primals_78: "f32[624]"; primals_79: "f32[624]"; primals_80: "f32[624]"; primals_81: "f32[624]"; primals_82: "f32[160]"; primals_83: "f32[160]"; primals_84: "f32[480]"; primals_85: "f32[480]"; primals_86: "f32[480]"; primals_87: "f32[480]"; primals_88: "f32[160]"; primals_89: "f32[160]"; primals_90: "f32[480]"; primals_91: "f32[480]"; primals_92: "f32[480]"; primals_93: "f32[480]"; primals_94: "f32[160]"; primals_95: "f32[160]"; primals_96: "f32[480]"; primals_97: "f32[480]"; primals_98: "f32[480]"; primals_99: "f32[480]"; primals_100: "f32[160]"; primals_101: "f32[160]"; primals_102: "f32[960]"; primals_103: "f32[960]"; primals_104: "f32[240, 1, 3, 3]"; primals_105: "f32[240, 1, 5, 5]"; primals_106: "f32[240, 1, 7, 7]"; primals_107: "f32[240, 1, 9, 9]"; primals_108: "f32[960]"; primals_109: "f32[960]"; primals_110: "f32[264]"; primals_111: "f32[264]"; primals_112: "f32[1584]"; primals_113: "f32[1584]"; primals_114: "f32[1584]"; primals_115: "f32[1584]"; primals_116: "f32[264]"; primals_117: "f32[264]"; primals_118: "f32[1584]"; primals_119: "f32[1584]"; primals_120: "f32[1584]"; primals_121: "f32[1584]"; primals_122: "f32[264]"; primals_123: "f32[264]"; primals_124: "f32[1584]"; primals_125: "f32[1584]"; primals_126: "f32[1584]"; primals_127: "f32[1584]"; primals_128: "f32[264]"; primals_129: "f32[264]"; primals_130: "f32[1536]"; primals_131: "f32[1536]"; primals_132: "f32[32, 1, 3, 3]"; primals_133: "f32[32, 32, 1, 1]"; primals_134: "f32[96, 16, 1, 1]"; primals_135: "f32[96, 16, 1, 1]"; primals_136: "f32[20, 96, 1, 1]"; primals_137: "f32[20, 96, 1, 1]"; primals_138: "f32[60, 20, 1, 1]"; primals_139: "f32[60, 20, 1, 1]"; primals_140: "f32[120, 1, 3, 3]"; primals_141: "f32[20, 60, 1, 1]"; primals_142: "f32[20, 60, 1, 1]"; primals_143: "f32[240, 40, 1, 1]"; primals_144: "f32[20, 240, 1, 1]"; primals_145: "f32[20]"; primals_146: "f32[240, 20, 1, 1]"; primals_147: "f32[240]"; primals_148: "f32[56, 240, 1, 1]"; primals_149: "f32[168, 28, 1, 1]"; primals_150: "f32[168, 28, 1, 1]"; primals_151: "f32[168, 1, 3, 3]"; primals_152: "f32[168, 1, 5, 5]"; primals_153: "f32[28, 336, 1, 1]"; primals_154: "f32[28]"; primals_155: "f32[336, 28, 1, 1]"; primals_156: "f32[336]"; primals_157: "f32[28, 168, 1, 1]"; primals_158: "f32[28, 168, 1, 1]"; primals_159: "f32[168, 28, 1, 1]"; primals_160: "f32[168, 28, 1, 1]"; primals_161: "f32[168, 1, 3, 3]"; primals_162: "f32[168, 1, 5, 5]"; primals_163: "f32[28, 336, 1, 1]"; primals_164: "f32[28]"; primals_165: "f32[336, 28, 1, 1]"; primals_166: "f32[336]"; primals_167: "f32[28, 168, 1, 1]"; primals_168: "f32[28, 168, 1, 1]"; primals_169: "f32[168, 28, 1, 1]"; primals_170: "f32[168, 28, 1, 1]"; primals_171: "f32[168, 1, 3, 3]"; primals_172: "f32[168, 1, 5, 5]"; primals_173: "f32[28, 336, 1, 1]"; primals_174: "f32[28]"; primals_175: "f32[336, 28, 1, 1]"; primals_176: "f32[336]"; primals_177: "f32[28, 168, 1, 1]"; primals_178: "f32[28, 168, 1, 1]"; primals_179: "f32[336, 56, 1, 1]"; primals_180: "f32[14, 336, 1, 1]"; primals_181: "f32[14]"; primals_182: "f32[336, 14, 1, 1]"; primals_183: "f32[336]"; primals_184: "f32[104, 336, 1, 1]"; primals_185: "f32[312, 52, 1, 1]"; primals_186: "f32[312, 52, 1, 1]"; primals_187: "f32[156, 1, 3, 3]"; primals_188: "f32[156, 1, 5, 5]"; primals_189: "f32[156, 1, 7, 7]"; primals_190: "f32[156, 1, 9, 9]"; primals_191: "f32[26, 624, 1, 1]"; primals_192: "f32[26]"; primals_193: "f32[624, 26, 1, 1]"; primals_194: "f32[624]"; primals_195: "f32[52, 312, 1, 1]"; primals_196: "f32[52, 312, 1, 1]"; primals_197: "f32[312, 52, 1, 1]"; primals_198: "f32[312, 52, 1, 1]"; primals_199: "f32[156, 1, 3, 3]"; primals_200: "f32[156, 1, 5, 5]"; primals_201: "f32[156, 1, 7, 7]"; primals_202: "f32[156, 1, 9, 9]"; primals_203: "f32[26, 624, 1, 1]"; primals_204: "f32[26]"; primals_205: "f32[624, 26, 1, 1]"; primals_206: "f32[624]"; primals_207: "f32[52, 312, 1, 1]"; primals_208: "f32[52, 312, 1, 1]"; primals_209: "f32[312, 52, 1, 1]"; primals_210: "f32[312, 52, 1, 1]"; primals_211: "f32[156, 1, 3, 3]"; primals_212: "f32[156, 1, 5, 5]"; primals_213: "f32[156, 1, 7, 7]"; primals_214: "f32[156, 1, 9, 9]"; primals_215: "f32[26, 624, 1, 1]"; primals_216: "f32[26]"; primals_217: "f32[624, 26, 1, 1]"; primals_218: "f32[624]"; primals_219: "f32[52, 312, 1, 1]"; primals_220: "f32[52, 312, 1, 1]"; primals_221: "f32[624, 104, 1, 1]"; primals_222: "f32[624, 1, 3, 3]"; primals_223: "f32[52, 624, 1, 1]"; primals_224: "f32[52]"; primals_225: "f32[624, 52, 1, 1]"; primals_226: "f32[624]"; primals_227: "f32[160, 624, 1, 1]"; primals_228: "f32[240, 80, 1, 1]"; primals_229: "f32[240, 80, 1, 1]"; primals_230: "f32[120, 1, 3, 3]"; primals_231: "f32[120, 1, 5, 5]"; primals_232: "f32[120, 1, 7, 7]"; primals_233: "f32[120, 1, 9, 9]"; primals_234: "f32[80, 480, 1, 1]"; primals_235: "f32[80]"; primals_236: "f32[480, 80, 1, 1]"; primals_237: "f32[480]"; primals_238: "f32[80, 240, 1, 1]"; primals_239: "f32[80, 240, 1, 1]"; primals_240: "f32[240, 80, 1, 1]"; primals_241: "f32[240, 80, 1, 1]"; primals_242: "f32[120, 1, 3, 3]"; primals_243: "f32[120, 1, 5, 5]"; primals_244: "f32[120, 1, 7, 7]"; primals_245: "f32[120, 1, 9, 9]"; primals_246: "f32[80, 480, 1, 1]"; primals_247: "f32[80]"; primals_248: "f32[480, 80, 1, 1]"; primals_249: "f32[480]"; primals_250: "f32[80, 240, 1, 1]"; primals_251: "f32[80, 240, 1, 1]"; primals_252: "f32[240, 80, 1, 1]"; primals_253: "f32[240, 80, 1, 1]"; primals_254: "f32[120, 1, 3, 3]"; primals_255: "f32[120, 1, 5, 5]"; primals_256: "f32[120, 1, 7, 7]"; primals_257: "f32[120, 1, 9, 9]"; primals_258: "f32[80, 480, 1, 1]"; primals_259: "f32[80]"; primals_260: "f32[480, 80, 1, 1]"; primals_261: "f32[480]"; primals_262: "f32[80, 240, 1, 1]"; primals_263: "f32[80, 240, 1, 1]"; primals_264: "f32[960, 160, 1, 1]"; primals_265: "f32[80, 960, 1, 1]"; primals_266: "f32[80]"; primals_267: "f32[960, 80, 1, 1]"; primals_268: "f32[960]"; primals_269: "f32[264, 960, 1, 1]"; primals_270: "f32[1584, 264, 1, 1]"; primals_271: "f32[396, 1, 3, 3]"; primals_272: "f32[396, 1, 5, 5]"; primals_273: "f32[396, 1, 7, 7]"; primals_274: "f32[396, 1, 9, 9]"; primals_275: "f32[132, 1584, 1, 1]"; primals_276: "f32[132]"; primals_277: "f32[1584, 132, 1, 1]"; primals_278: "f32[1584]"; primals_279: "f32[132, 792, 1, 1]"; primals_280: "f32[132, 792, 1, 1]"; primals_281: "f32[1584, 264, 1, 1]"; primals_282: "f32[396, 1, 3, 3]"; primals_283: "f32[396, 1, 5, 5]"; primals_284: "f32[396, 1, 7, 7]"; primals_285: "f32[396, 1, 9, 9]"; primals_286: "f32[132, 1584, 1, 1]"; primals_287: "f32[132]"; primals_288: "f32[1584, 132, 1, 1]"; primals_289: "f32[1584]"; primals_290: "f32[132, 792, 1, 1]"; primals_291: "f32[132, 792, 1, 1]"; primals_292: "f32[1584, 264, 1, 1]"; primals_293: "f32[396, 1, 3, 3]"; primals_294: "f32[396, 1, 5, 5]"; primals_295: "f32[396, 1, 7, 7]"; primals_296: "f32[396, 1, 9, 9]"; primals_297: "f32[132, 1584, 1, 1]"; primals_298: "f32[132]"; primals_299: "f32[1584, 132, 1, 1]"; primals_300: "f32[1584]"; primals_301: "f32[132, 792, 1, 1]"; primals_302: "f32[132, 792, 1, 1]"; primals_303: "f32[1536, 264, 1, 1]"; primals_304: "f32[1000, 1536]"; primals_305: "f32[1000]"; primals_306: "i64[]"; primals_307: "f32[32]"; primals_308: "f32[32]"; primals_309: "i64[]"; primals_310: "f32[32]"; primals_311: "f32[32]"; primals_312: "i64[]"; primals_313: "f32[32]"; primals_314: "f32[32]"; primals_315: "i64[]"; primals_316: "f32[192]"; primals_317: "f32[192]"; primals_318: "i64[]"; primals_319: "f32[192]"; primals_320: "f32[192]"; primals_321: "i64[]"; primals_322: "f32[40]"; primals_323: "f32[40]"; primals_324: "i64[]"; primals_325: "f32[120]"; primals_326: "f32[120]"; primals_327: "i64[]"; primals_328: "f32[120]"; primals_329: "f32[120]"; primals_330: "i64[]"; primals_331: "f32[40]"; primals_332: "f32[40]"; primals_333: "i64[]"; primals_334: "f32[240]"; primals_335: "f32[240]"; primals_336: "i64[]"; primals_337: "f32[240]"; primals_338: "f32[240]"; primals_339: "i64[]"; primals_340: "f32[56]"; primals_341: "f32[56]"; primals_342: "i64[]"; primals_343: "f32[336]"; primals_344: "f32[336]"; primals_345: "i64[]"; primals_346: "f32[336]"; primals_347: "f32[336]"; primals_348: "i64[]"; primals_349: "f32[56]"; primals_350: "f32[56]"; primals_351: "i64[]"; primals_352: "f32[336]"; primals_353: "f32[336]"; primals_354: "i64[]"; primals_355: "f32[336]"; primals_356: "f32[336]"; primals_357: "i64[]"; primals_358: "f32[56]"; primals_359: "f32[56]"; primals_360: "i64[]"; primals_361: "f32[336]"; primals_362: "f32[336]"; primals_363: "i64[]"; primals_364: "f32[336]"; primals_365: "f32[336]"; primals_366: "i64[]"; primals_367: "f32[56]"; primals_368: "f32[56]"; primals_369: "i64[]"; primals_370: "f32[336]"; primals_371: "f32[336]"; primals_372: "i64[]"; primals_373: "f32[336]"; primals_374: "f32[336]"; primals_375: "i64[]"; primals_376: "f32[104]"; primals_377: "f32[104]"; primals_378: "i64[]"; primals_379: "f32[624]"; primals_380: "f32[624]"; primals_381: "i64[]"; primals_382: "f32[624]"; primals_383: "f32[624]"; primals_384: "i64[]"; primals_385: "f32[104]"; primals_386: "f32[104]"; primals_387: "i64[]"; primals_388: "f32[624]"; primals_389: "f32[624]"; primals_390: "i64[]"; primals_391: "f32[624]"; primals_392: "f32[624]"; primals_393: "i64[]"; primals_394: "f32[104]"; primals_395: "f32[104]"; primals_396: "i64[]"; primals_397: "f32[624]"; primals_398: "f32[624]"; primals_399: "i64[]"; primals_400: "f32[624]"; primals_401: "f32[624]"; primals_402: "i64[]"; primals_403: "f32[104]"; primals_404: "f32[104]"; primals_405: "i64[]"; primals_406: "f32[624]"; primals_407: "f32[624]"; primals_408: "i64[]"; primals_409: "f32[624]"; primals_410: "f32[624]"; primals_411: "i64[]"; primals_412: "f32[160]"; primals_413: "f32[160]"; primals_414: "i64[]"; primals_415: "f32[480]"; primals_416: "f32[480]"; primals_417: "i64[]"; primals_418: "f32[480]"; primals_419: "f32[480]"; primals_420: "i64[]"; primals_421: "f32[160]"; primals_422: "f32[160]"; primals_423: "i64[]"; primals_424: "f32[480]"; primals_425: "f32[480]"; primals_426: "i64[]"; primals_427: "f32[480]"; primals_428: "f32[480]"; primals_429: "i64[]"; primals_430: "f32[160]"; primals_431: "f32[160]"; primals_432: "i64[]"; primals_433: "f32[480]"; primals_434: "f32[480]"; primals_435: "i64[]"; primals_436: "f32[480]"; primals_437: "f32[480]"; primals_438: "i64[]"; primals_439: "f32[160]"; primals_440: "f32[160]"; primals_441: "i64[]"; primals_442: "f32[960]"; primals_443: "f32[960]"; primals_444: "i64[]"; primals_445: "f32[960]"; primals_446: "f32[960]"; primals_447: "i64[]"; primals_448: "f32[264]"; primals_449: "f32[264]"; primals_450: "i64[]"; primals_451: "f32[1584]"; primals_452: "f32[1584]"; primals_453: "i64[]"; primals_454: "f32[1584]"; primals_455: "f32[1584]"; primals_456: "i64[]"; primals_457: "f32[264]"; primals_458: "f32[264]"; primals_459: "i64[]"; primals_460: "f32[1584]"; primals_461: "f32[1584]"; primals_462: "i64[]"; primals_463: "f32[1584]"; primals_464: "f32[1584]"; primals_465: "i64[]"; primals_466: "f32[264]"; primals_467: "f32[264]"; primals_468: "i64[]"; primals_469: "f32[1584]"; primals_470: "f32[1584]"; primals_471: "i64[]"; primals_472: "f32[1584]"; primals_473: "f32[1584]"; primals_474: "i64[]"; primals_475: "f32[264]"; primals_476: "f32[264]"; primals_477: "i64[]"; primals_478: "f32[1536]"; primals_479: "f32[1536]"; primals_480: "f32[8, 3, 224, 224]"; tangents_1: "f32[8, 1000]"; 

    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, tangents_1, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd: "f32[8, 3, 225, 225]" = torch.ops.aten.constant_pad_nd.default(primals_480, [0, 1, 0, 1], 0.0);  primals_480 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution: "f32[8, 32, 112, 112]" = torch.ops.aten.convolution.default(constant_pad_nd, primals_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add: "i64[]" = torch.ops.aten.add.Tensor(primals_306, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean = torch.ops.aten.var_mean.correction(convolution, [0, 2, 3], correction = 0, keepdim = True)
    getitem: "f32[1, 32, 1, 1]" = var_mean[0]
    getitem_1: "f32[1, 32, 1, 1]" = var_mean[1];  var_mean = None
    add_1: "f32[1, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 0.001)
    rsqrt: "f32[1, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
    sub: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, getitem_1)
    mul: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
    squeeze: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
    squeeze_1: "f32[32]" = torch.ops.aten.squeeze.dims(rsqrt, [0, 2, 3]);  rsqrt = None
    mul_1: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze, 0.1)
    mul_2: "f32[32]" = torch.ops.aten.mul.Tensor(primals_307, 0.9)
    add_2: "f32[32]" = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
    squeeze_2: "f32[32]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
    mul_3: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_2, 1.00000996502277);  squeeze_2 = None
    mul_4: "f32[32]" = torch.ops.aten.mul.Tensor(mul_3, 0.1);  mul_3 = None
    mul_5: "f32[32]" = torch.ops.aten.mul.Tensor(primals_308, 0.9)
    add_3: "f32[32]" = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
    unsqueeze: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_2, -1)
    unsqueeze_1: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
    mul_6: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
    unsqueeze_2: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_3, -1);  primals_3 = None
    unsqueeze_3: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
    add_4: "f32[8, 32, 112, 112]" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_3);  mul_6 = unsqueeze_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu: "f32[8, 32, 112, 112]" = torch.ops.aten.relu.default(add_4);  add_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:123, code: x = self.conv_dw(x)
    convolution_1: "f32[8, 32, 112, 112]" = torch.ops.aten.convolution.default(relu, primals_132, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_5: "i64[]" = torch.ops.aten.add.Tensor(primals_309, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_1 = torch.ops.aten.var_mean.correction(convolution_1, [0, 2, 3], correction = 0, keepdim = True)
    getitem_2: "f32[1, 32, 1, 1]" = var_mean_1[0]
    getitem_3: "f32[1, 32, 1, 1]" = var_mean_1[1];  var_mean_1 = None
    add_6: "f32[1, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_2, 0.001)
    rsqrt_1: "f32[1, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
    sub_1: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_1, getitem_3)
    mul_7: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
    squeeze_3: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_3, [0, 2, 3]);  getitem_3 = None
    squeeze_4: "f32[32]" = torch.ops.aten.squeeze.dims(rsqrt_1, [0, 2, 3]);  rsqrt_1 = None
    mul_8: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_3, 0.1)
    mul_9: "f32[32]" = torch.ops.aten.mul.Tensor(primals_310, 0.9)
    add_7: "f32[32]" = torch.ops.aten.add.Tensor(mul_8, mul_9);  mul_8 = mul_9 = None
    squeeze_5: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_2, [0, 2, 3]);  getitem_2 = None
    mul_10: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_5, 1.00000996502277);  squeeze_5 = None
    mul_11: "f32[32]" = torch.ops.aten.mul.Tensor(mul_10, 0.1);  mul_10 = None
    mul_12: "f32[32]" = torch.ops.aten.mul.Tensor(primals_311, 0.9)
    add_8: "f32[32]" = torch.ops.aten.add.Tensor(mul_11, mul_12);  mul_11 = mul_12 = None
    unsqueeze_4: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_4, -1)
    unsqueeze_5: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
    mul_13: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_5);  mul_7 = unsqueeze_5 = None
    unsqueeze_6: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_5, -1);  primals_5 = None
    unsqueeze_7: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
    add_9: "f32[8, 32, 112, 112]" = torch.ops.aten.add.Tensor(mul_13, unsqueeze_7);  mul_13 = unsqueeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_1: "f32[8, 32, 112, 112]" = torch.ops.aten.relu.default(add_9);  add_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:126, code: x = self.conv_pw(x)
    convolution_2: "f32[8, 32, 112, 112]" = torch.ops.aten.convolution.default(relu_1, primals_133, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_10: "i64[]" = torch.ops.aten.add.Tensor(primals_312, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_2 = torch.ops.aten.var_mean.correction(convolution_2, [0, 2, 3], correction = 0, keepdim = True)
    getitem_4: "f32[1, 32, 1, 1]" = var_mean_2[0]
    getitem_5: "f32[1, 32, 1, 1]" = var_mean_2[1];  var_mean_2 = None
    add_11: "f32[1, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_4, 0.001)
    rsqrt_2: "f32[1, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
    sub_2: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_2, getitem_5)
    mul_14: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
    squeeze_6: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2, 3]);  getitem_5 = None
    squeeze_7: "f32[32]" = torch.ops.aten.squeeze.dims(rsqrt_2, [0, 2, 3]);  rsqrt_2 = None
    mul_15: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_6, 0.1)
    mul_16: "f32[32]" = torch.ops.aten.mul.Tensor(primals_313, 0.9)
    add_12: "f32[32]" = torch.ops.aten.add.Tensor(mul_15, mul_16);  mul_15 = mul_16 = None
    squeeze_8: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_4, [0, 2, 3]);  getitem_4 = None
    mul_17: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_8, 1.00000996502277);  squeeze_8 = None
    mul_18: "f32[32]" = torch.ops.aten.mul.Tensor(mul_17, 0.1);  mul_17 = None
    mul_19: "f32[32]" = torch.ops.aten.mul.Tensor(primals_314, 0.9)
    add_13: "f32[32]" = torch.ops.aten.add.Tensor(mul_18, mul_19);  mul_18 = mul_19 = None
    unsqueeze_8: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_6, -1)
    unsqueeze_9: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
    mul_20: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(mul_14, unsqueeze_9);  mul_14 = unsqueeze_9 = None
    unsqueeze_10: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_7, -1);  primals_7 = None
    unsqueeze_11: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
    add_14: "f32[8, 32, 112, 112]" = torch.ops.aten.add.Tensor(mul_20, unsqueeze_11);  mul_20 = unsqueeze_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:129, code: x = self.drop_path(x) + shortcut
    add_15: "f32[8, 32, 112, 112]" = torch.ops.aten.add.Tensor(add_14, relu);  add_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_with_sizes = torch.ops.aten.split_with_sizes.default(add_15, [16, 16], 1);  add_15 = None
    getitem_6: "f32[8, 16, 112, 112]" = split_with_sizes[0]
    getitem_7: "f32[8, 16, 112, 112]" = split_with_sizes[1];  split_with_sizes = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_3: "f32[8, 96, 112, 112]" = torch.ops.aten.convolution.default(getitem_6, primals_134, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convolution_4: "f32[8, 96, 112, 112]" = torch.ops.aten.convolution.default(getitem_7, primals_135, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat: "f32[8, 192, 112, 112]" = torch.ops.aten.cat.default([convolution_3, convolution_4], 1);  convolution_3 = convolution_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_16: "i64[]" = torch.ops.aten.add.Tensor(primals_315, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_3 = torch.ops.aten.var_mean.correction(cat, [0, 2, 3], correction = 0, keepdim = True)
    getitem_8: "f32[1, 192, 1, 1]" = var_mean_3[0]
    getitem_9: "f32[1, 192, 1, 1]" = var_mean_3[1];  var_mean_3 = None
    add_17: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_8, 0.001)
    rsqrt_3: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_17);  add_17 = None
    sub_3: "f32[8, 192, 112, 112]" = torch.ops.aten.sub.Tensor(cat, getitem_9)
    mul_21: "f32[8, 192, 112, 112]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
    squeeze_9: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_9, [0, 2, 3]);  getitem_9 = None
    squeeze_10: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_3, [0, 2, 3]);  rsqrt_3 = None
    mul_22: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_9, 0.1)
    mul_23: "f32[192]" = torch.ops.aten.mul.Tensor(primals_316, 0.9)
    add_18: "f32[192]" = torch.ops.aten.add.Tensor(mul_22, mul_23);  mul_22 = mul_23 = None
    squeeze_11: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_8, [0, 2, 3]);  getitem_8 = None
    mul_24: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_11, 1.00000996502277);  squeeze_11 = None
    mul_25: "f32[192]" = torch.ops.aten.mul.Tensor(mul_24, 0.1);  mul_24 = None
    mul_26: "f32[192]" = torch.ops.aten.mul.Tensor(primals_317, 0.9)
    add_19: "f32[192]" = torch.ops.aten.add.Tensor(mul_25, mul_26);  mul_25 = mul_26 = None
    unsqueeze_12: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_8, -1)
    unsqueeze_13: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
    mul_27: "f32[8, 192, 112, 112]" = torch.ops.aten.mul.Tensor(mul_21, unsqueeze_13);  mul_21 = unsqueeze_13 = None
    unsqueeze_14: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_9, -1);  primals_9 = None
    unsqueeze_15: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
    add_20: "f32[8, 192, 112, 112]" = torch.ops.aten.add.Tensor(mul_27, unsqueeze_15);  mul_27 = unsqueeze_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_2: "f32[8, 192, 112, 112]" = torch.ops.aten.relu.default(add_20);  add_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    split_with_sizes_2 = torch.ops.aten.split_with_sizes.default(relu_2, [64, 64, 64], 1)
    getitem_13: "f32[8, 64, 112, 112]" = split_with_sizes_2[0];  split_with_sizes_2 = None
    constant_pad_nd_1: "f32[8, 64, 113, 113]" = torch.ops.aten.constant_pad_nd.default(getitem_13, [0, 1, 0, 1], 0.0);  getitem_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_5: "f32[8, 64, 56, 56]" = torch.ops.aten.convolution.default(constant_pad_nd_1, primals_10, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    split_with_sizes_3 = torch.ops.aten.split_with_sizes.default(relu_2, [64, 64, 64], 1)
    getitem_17: "f32[8, 64, 112, 112]" = split_with_sizes_3[1];  split_with_sizes_3 = None
    constant_pad_nd_2: "f32[8, 64, 115, 115]" = torch.ops.aten.constant_pad_nd.default(getitem_17, [1, 2, 1, 2], 0.0);  getitem_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_6: "f32[8, 64, 56, 56]" = torch.ops.aten.convolution.default(constant_pad_nd_2, primals_11, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    split_with_sizes_4 = torch.ops.aten.split_with_sizes.default(relu_2, [64, 64, 64], 1)
    getitem_21: "f32[8, 64, 112, 112]" = split_with_sizes_4[2];  split_with_sizes_4 = None
    constant_pad_nd_3: "f32[8, 64, 117, 117]" = torch.ops.aten.constant_pad_nd.default(getitem_21, [2, 3, 2, 3], 0.0);  getitem_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_7: "f32[8, 64, 56, 56]" = torch.ops.aten.convolution.default(constant_pad_nd_3, primals_12, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_1: "f32[8, 192, 56, 56]" = torch.ops.aten.cat.default([convolution_5, convolution_6, convolution_7], 1);  convolution_5 = convolution_6 = convolution_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_21: "i64[]" = torch.ops.aten.add.Tensor(primals_318, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_4 = torch.ops.aten.var_mean.correction(cat_1, [0, 2, 3], correction = 0, keepdim = True)
    getitem_22: "f32[1, 192, 1, 1]" = var_mean_4[0]
    getitem_23: "f32[1, 192, 1, 1]" = var_mean_4[1];  var_mean_4 = None
    add_22: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_22, 0.001)
    rsqrt_4: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
    sub_4: "f32[8, 192, 56, 56]" = torch.ops.aten.sub.Tensor(cat_1, getitem_23)
    mul_28: "f32[8, 192, 56, 56]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
    squeeze_12: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_23, [0, 2, 3]);  getitem_23 = None
    squeeze_13: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_4, [0, 2, 3]);  rsqrt_4 = None
    mul_29: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_12, 0.1)
    mul_30: "f32[192]" = torch.ops.aten.mul.Tensor(primals_319, 0.9)
    add_23: "f32[192]" = torch.ops.aten.add.Tensor(mul_29, mul_30);  mul_29 = mul_30 = None
    squeeze_14: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_22, [0, 2, 3]);  getitem_22 = None
    mul_31: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_14, 1.0000398612827361);  squeeze_14 = None
    mul_32: "f32[192]" = torch.ops.aten.mul.Tensor(mul_31, 0.1);  mul_31 = None
    mul_33: "f32[192]" = torch.ops.aten.mul.Tensor(primals_320, 0.9)
    add_24: "f32[192]" = torch.ops.aten.add.Tensor(mul_32, mul_33);  mul_32 = mul_33 = None
    unsqueeze_16: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_13, -1)
    unsqueeze_17: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
    mul_34: "f32[8, 192, 56, 56]" = torch.ops.aten.mul.Tensor(mul_28, unsqueeze_17);  mul_28 = unsqueeze_17 = None
    unsqueeze_18: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_14, -1);  primals_14 = None
    unsqueeze_19: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
    add_25: "f32[8, 192, 56, 56]" = torch.ops.aten.add.Tensor(mul_34, unsqueeze_19);  mul_34 = unsqueeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_3: "f32[8, 192, 56, 56]" = torch.ops.aten.relu.default(add_25);  add_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    split_with_sizes_6 = torch.ops.aten.split_with_sizes.default(relu_3, [96, 96], 1)
    getitem_26: "f32[8, 96, 56, 56]" = split_with_sizes_6[0];  split_with_sizes_6 = None
    convolution_8: "f32[8, 20, 56, 56]" = torch.ops.aten.convolution.default(getitem_26, primals_136, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    split_with_sizes_7 = torch.ops.aten.split_with_sizes.default(relu_3, [96, 96], 1)
    getitem_29: "f32[8, 96, 56, 56]" = split_with_sizes_7[1];  split_with_sizes_7 = None
    convolution_9: "f32[8, 20, 56, 56]" = torch.ops.aten.convolution.default(getitem_29, primals_137, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_2: "f32[8, 40, 56, 56]" = torch.ops.aten.cat.default([convolution_8, convolution_9], 1);  convolution_8 = convolution_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_26: "i64[]" = torch.ops.aten.add.Tensor(primals_321, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_5 = torch.ops.aten.var_mean.correction(cat_2, [0, 2, 3], correction = 0, keepdim = True)
    getitem_30: "f32[1, 40, 1, 1]" = var_mean_5[0]
    getitem_31: "f32[1, 40, 1, 1]" = var_mean_5[1];  var_mean_5 = None
    add_27: "f32[1, 40, 1, 1]" = torch.ops.aten.add.Tensor(getitem_30, 0.001)
    rsqrt_5: "f32[1, 40, 1, 1]" = torch.ops.aten.rsqrt.default(add_27);  add_27 = None
    sub_5: "f32[8, 40, 56, 56]" = torch.ops.aten.sub.Tensor(cat_2, getitem_31)
    mul_35: "f32[8, 40, 56, 56]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
    squeeze_15: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_31, [0, 2, 3]);  getitem_31 = None
    squeeze_16: "f32[40]" = torch.ops.aten.squeeze.dims(rsqrt_5, [0, 2, 3]);  rsqrt_5 = None
    mul_36: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_15, 0.1)
    mul_37: "f32[40]" = torch.ops.aten.mul.Tensor(primals_322, 0.9)
    add_28: "f32[40]" = torch.ops.aten.add.Tensor(mul_36, mul_37);  mul_36 = mul_37 = None
    squeeze_17: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_30, [0, 2, 3]);  getitem_30 = None
    mul_38: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_17, 1.0000398612827361);  squeeze_17 = None
    mul_39: "f32[40]" = torch.ops.aten.mul.Tensor(mul_38, 0.1);  mul_38 = None
    mul_40: "f32[40]" = torch.ops.aten.mul.Tensor(primals_323, 0.9)
    add_29: "f32[40]" = torch.ops.aten.add.Tensor(mul_39, mul_40);  mul_39 = mul_40 = None
    unsqueeze_20: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_15, -1)
    unsqueeze_21: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
    mul_41: "f32[8, 40, 56, 56]" = torch.ops.aten.mul.Tensor(mul_35, unsqueeze_21);  mul_35 = unsqueeze_21 = None
    unsqueeze_22: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_16, -1);  primals_16 = None
    unsqueeze_23: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
    add_30: "f32[8, 40, 56, 56]" = torch.ops.aten.add.Tensor(mul_41, unsqueeze_23);  mul_41 = unsqueeze_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_with_sizes_8 = torch.ops.aten.split_with_sizes.default(add_30, [20, 20], 1)
    getitem_32: "f32[8, 20, 56, 56]" = split_with_sizes_8[0]
    getitem_33: "f32[8, 20, 56, 56]" = split_with_sizes_8[1];  split_with_sizes_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_10: "f32[8, 60, 56, 56]" = torch.ops.aten.convolution.default(getitem_32, primals_138, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convolution_11: "f32[8, 60, 56, 56]" = torch.ops.aten.convolution.default(getitem_33, primals_139, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_3: "f32[8, 120, 56, 56]" = torch.ops.aten.cat.default([convolution_10, convolution_11], 1);  convolution_10 = convolution_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_31: "i64[]" = torch.ops.aten.add.Tensor(primals_324, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_6 = torch.ops.aten.var_mean.correction(cat_3, [0, 2, 3], correction = 0, keepdim = True)
    getitem_34: "f32[1, 120, 1, 1]" = var_mean_6[0]
    getitem_35: "f32[1, 120, 1, 1]" = var_mean_6[1];  var_mean_6 = None
    add_32: "f32[1, 120, 1, 1]" = torch.ops.aten.add.Tensor(getitem_34, 0.001)
    rsqrt_6: "f32[1, 120, 1, 1]" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
    sub_6: "f32[8, 120, 56, 56]" = torch.ops.aten.sub.Tensor(cat_3, getitem_35)
    mul_42: "f32[8, 120, 56, 56]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
    squeeze_18: "f32[120]" = torch.ops.aten.squeeze.dims(getitem_35, [0, 2, 3]);  getitem_35 = None
    squeeze_19: "f32[120]" = torch.ops.aten.squeeze.dims(rsqrt_6, [0, 2, 3]);  rsqrt_6 = None
    mul_43: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_18, 0.1)
    mul_44: "f32[120]" = torch.ops.aten.mul.Tensor(primals_325, 0.9)
    add_33: "f32[120]" = torch.ops.aten.add.Tensor(mul_43, mul_44);  mul_43 = mul_44 = None
    squeeze_20: "f32[120]" = torch.ops.aten.squeeze.dims(getitem_34, [0, 2, 3]);  getitem_34 = None
    mul_45: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_20, 1.0000398612827361);  squeeze_20 = None
    mul_46: "f32[120]" = torch.ops.aten.mul.Tensor(mul_45, 0.1);  mul_45 = None
    mul_47: "f32[120]" = torch.ops.aten.mul.Tensor(primals_326, 0.9)
    add_34: "f32[120]" = torch.ops.aten.add.Tensor(mul_46, mul_47);  mul_46 = mul_47 = None
    unsqueeze_24: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_17, -1)
    unsqueeze_25: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
    mul_48: "f32[8, 120, 56, 56]" = torch.ops.aten.mul.Tensor(mul_42, unsqueeze_25);  mul_42 = unsqueeze_25 = None
    unsqueeze_26: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_18, -1);  primals_18 = None
    unsqueeze_27: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
    add_35: "f32[8, 120, 56, 56]" = torch.ops.aten.add.Tensor(mul_48, unsqueeze_27);  mul_48 = unsqueeze_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_4: "f32[8, 120, 56, 56]" = torch.ops.aten.relu.default(add_35);  add_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_12: "f32[8, 120, 56, 56]" = torch.ops.aten.convolution.default(relu_4, primals_140, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 120)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_36: "i64[]" = torch.ops.aten.add.Tensor(primals_327, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_7 = torch.ops.aten.var_mean.correction(convolution_12, [0, 2, 3], correction = 0, keepdim = True)
    getitem_36: "f32[1, 120, 1, 1]" = var_mean_7[0]
    getitem_37: "f32[1, 120, 1, 1]" = var_mean_7[1];  var_mean_7 = None
    add_37: "f32[1, 120, 1, 1]" = torch.ops.aten.add.Tensor(getitem_36, 0.001)
    rsqrt_7: "f32[1, 120, 1, 1]" = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
    sub_7: "f32[8, 120, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_12, getitem_37)
    mul_49: "f32[8, 120, 56, 56]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
    squeeze_21: "f32[120]" = torch.ops.aten.squeeze.dims(getitem_37, [0, 2, 3]);  getitem_37 = None
    squeeze_22: "f32[120]" = torch.ops.aten.squeeze.dims(rsqrt_7, [0, 2, 3]);  rsqrt_7 = None
    mul_50: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_21, 0.1)
    mul_51: "f32[120]" = torch.ops.aten.mul.Tensor(primals_328, 0.9)
    add_38: "f32[120]" = torch.ops.aten.add.Tensor(mul_50, mul_51);  mul_50 = mul_51 = None
    squeeze_23: "f32[120]" = torch.ops.aten.squeeze.dims(getitem_36, [0, 2, 3]);  getitem_36 = None
    mul_52: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_23, 1.0000398612827361);  squeeze_23 = None
    mul_53: "f32[120]" = torch.ops.aten.mul.Tensor(mul_52, 0.1);  mul_52 = None
    mul_54: "f32[120]" = torch.ops.aten.mul.Tensor(primals_329, 0.9)
    add_39: "f32[120]" = torch.ops.aten.add.Tensor(mul_53, mul_54);  mul_53 = mul_54 = None
    unsqueeze_28: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_19, -1)
    unsqueeze_29: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
    mul_55: "f32[8, 120, 56, 56]" = torch.ops.aten.mul.Tensor(mul_49, unsqueeze_29);  mul_49 = unsqueeze_29 = None
    unsqueeze_30: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_20, -1);  primals_20 = None
    unsqueeze_31: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
    add_40: "f32[8, 120, 56, 56]" = torch.ops.aten.add.Tensor(mul_55, unsqueeze_31);  mul_55 = unsqueeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_5: "f32[8, 120, 56, 56]" = torch.ops.aten.relu.default(add_40);  add_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    split_with_sizes_10 = torch.ops.aten.split_with_sizes.default(relu_5, [60, 60], 1)
    getitem_40: "f32[8, 60, 56, 56]" = split_with_sizes_10[0];  split_with_sizes_10 = None
    convolution_13: "f32[8, 20, 56, 56]" = torch.ops.aten.convolution.default(getitem_40, primals_141, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    split_with_sizes_11 = torch.ops.aten.split_with_sizes.default(relu_5, [60, 60], 1)
    getitem_43: "f32[8, 60, 56, 56]" = split_with_sizes_11[1];  split_with_sizes_11 = None
    convolution_14: "f32[8, 20, 56, 56]" = torch.ops.aten.convolution.default(getitem_43, primals_142, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_4: "f32[8, 40, 56, 56]" = torch.ops.aten.cat.default([convolution_13, convolution_14], 1);  convolution_13 = convolution_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_41: "i64[]" = torch.ops.aten.add.Tensor(primals_330, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_8 = torch.ops.aten.var_mean.correction(cat_4, [0, 2, 3], correction = 0, keepdim = True)
    getitem_44: "f32[1, 40, 1, 1]" = var_mean_8[0]
    getitem_45: "f32[1, 40, 1, 1]" = var_mean_8[1];  var_mean_8 = None
    add_42: "f32[1, 40, 1, 1]" = torch.ops.aten.add.Tensor(getitem_44, 0.001)
    rsqrt_8: "f32[1, 40, 1, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
    sub_8: "f32[8, 40, 56, 56]" = torch.ops.aten.sub.Tensor(cat_4, getitem_45)
    mul_56: "f32[8, 40, 56, 56]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
    squeeze_24: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_45, [0, 2, 3]);  getitem_45 = None
    squeeze_25: "f32[40]" = torch.ops.aten.squeeze.dims(rsqrt_8, [0, 2, 3]);  rsqrt_8 = None
    mul_57: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_24, 0.1)
    mul_58: "f32[40]" = torch.ops.aten.mul.Tensor(primals_331, 0.9)
    add_43: "f32[40]" = torch.ops.aten.add.Tensor(mul_57, mul_58);  mul_57 = mul_58 = None
    squeeze_26: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_44, [0, 2, 3]);  getitem_44 = None
    mul_59: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_26, 1.0000398612827361);  squeeze_26 = None
    mul_60: "f32[40]" = torch.ops.aten.mul.Tensor(mul_59, 0.1);  mul_59 = None
    mul_61: "f32[40]" = torch.ops.aten.mul.Tensor(primals_332, 0.9)
    add_44: "f32[40]" = torch.ops.aten.add.Tensor(mul_60, mul_61);  mul_60 = mul_61 = None
    unsqueeze_32: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_21, -1)
    unsqueeze_33: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
    mul_62: "f32[8, 40, 56, 56]" = torch.ops.aten.mul.Tensor(mul_56, unsqueeze_33);  mul_56 = unsqueeze_33 = None
    unsqueeze_34: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_22, -1);  primals_22 = None
    unsqueeze_35: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
    add_45: "f32[8, 40, 56, 56]" = torch.ops.aten.add.Tensor(mul_62, unsqueeze_35);  mul_62 = unsqueeze_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_46: "f32[8, 40, 56, 56]" = torch.ops.aten.add.Tensor(add_45, add_30);  add_45 = add_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_15: "f32[8, 240, 56, 56]" = torch.ops.aten.convolution.default(add_46, primals_143, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_47: "i64[]" = torch.ops.aten.add.Tensor(primals_333, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_9 = torch.ops.aten.var_mean.correction(convolution_15, [0, 2, 3], correction = 0, keepdim = True)
    getitem_46: "f32[1, 240, 1, 1]" = var_mean_9[0]
    getitem_47: "f32[1, 240, 1, 1]" = var_mean_9[1];  var_mean_9 = None
    add_48: "f32[1, 240, 1, 1]" = torch.ops.aten.add.Tensor(getitem_46, 0.001)
    rsqrt_9: "f32[1, 240, 1, 1]" = torch.ops.aten.rsqrt.default(add_48);  add_48 = None
    sub_9: "f32[8, 240, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_15, getitem_47)
    mul_63: "f32[8, 240, 56, 56]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
    squeeze_27: "f32[240]" = torch.ops.aten.squeeze.dims(getitem_47, [0, 2, 3]);  getitem_47 = None
    squeeze_28: "f32[240]" = torch.ops.aten.squeeze.dims(rsqrt_9, [0, 2, 3]);  rsqrt_9 = None
    mul_64: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_27, 0.1)
    mul_65: "f32[240]" = torch.ops.aten.mul.Tensor(primals_334, 0.9)
    add_49: "f32[240]" = torch.ops.aten.add.Tensor(mul_64, mul_65);  mul_64 = mul_65 = None
    squeeze_29: "f32[240]" = torch.ops.aten.squeeze.dims(getitem_46, [0, 2, 3]);  getitem_46 = None
    mul_66: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_29, 1.0000398612827361);  squeeze_29 = None
    mul_67: "f32[240]" = torch.ops.aten.mul.Tensor(mul_66, 0.1);  mul_66 = None
    mul_68: "f32[240]" = torch.ops.aten.mul.Tensor(primals_335, 0.9)
    add_50: "f32[240]" = torch.ops.aten.add.Tensor(mul_67, mul_68);  mul_67 = mul_68 = None
    unsqueeze_36: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_23, -1)
    unsqueeze_37: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
    mul_69: "f32[8, 240, 56, 56]" = torch.ops.aten.mul.Tensor(mul_63, unsqueeze_37);  mul_63 = unsqueeze_37 = None
    unsqueeze_38: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_24, -1);  primals_24 = None
    unsqueeze_39: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
    add_51: "f32[8, 240, 56, 56]" = torch.ops.aten.add.Tensor(mul_69, unsqueeze_39);  mul_69 = unsqueeze_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone: "f32[8, 240, 56, 56]" = torch.ops.aten.clone.default(add_51)
    sigmoid: "f32[8, 240, 56, 56]" = torch.ops.aten.sigmoid.default(add_51)
    mul_70: "f32[8, 240, 56, 56]" = torch.ops.aten.mul.Tensor(add_51, sigmoid);  add_51 = sigmoid = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    split_with_sizes_13 = torch.ops.aten.split_with_sizes.default(mul_70, [60, 60, 60, 60], 1)
    getitem_52: "f32[8, 60, 56, 56]" = split_with_sizes_13[0];  split_with_sizes_13 = None
    constant_pad_nd_4: "f32[8, 60, 57, 57]" = torch.ops.aten.constant_pad_nd.default(getitem_52, [0, 1, 0, 1], 0.0);  getitem_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_16: "f32[8, 60, 28, 28]" = torch.ops.aten.convolution.default(constant_pad_nd_4, primals_25, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 60)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    split_with_sizes_14 = torch.ops.aten.split_with_sizes.default(mul_70, [60, 60, 60, 60], 1)
    getitem_57: "f32[8, 60, 56, 56]" = split_with_sizes_14[1];  split_with_sizes_14 = None
    constant_pad_nd_5: "f32[8, 60, 59, 59]" = torch.ops.aten.constant_pad_nd.default(getitem_57, [1, 2, 1, 2], 0.0);  getitem_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_17: "f32[8, 60, 28, 28]" = torch.ops.aten.convolution.default(constant_pad_nd_5, primals_26, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 60)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    split_with_sizes_15 = torch.ops.aten.split_with_sizes.default(mul_70, [60, 60, 60, 60], 1)
    getitem_62: "f32[8, 60, 56, 56]" = split_with_sizes_15[2];  split_with_sizes_15 = None
    constant_pad_nd_6: "f32[8, 60, 61, 61]" = torch.ops.aten.constant_pad_nd.default(getitem_62, [2, 3, 2, 3], 0.0);  getitem_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_18: "f32[8, 60, 28, 28]" = torch.ops.aten.convolution.default(constant_pad_nd_6, primals_27, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 60)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    split_with_sizes_16 = torch.ops.aten.split_with_sizes.default(mul_70, [60, 60, 60, 60], 1);  mul_70 = None
    getitem_67: "f32[8, 60, 56, 56]" = split_with_sizes_16[3];  split_with_sizes_16 = None
    constant_pad_nd_7: "f32[8, 60, 63, 63]" = torch.ops.aten.constant_pad_nd.default(getitem_67, [3, 4, 3, 4], 0.0);  getitem_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_19: "f32[8, 60, 28, 28]" = torch.ops.aten.convolution.default(constant_pad_nd_7, primals_28, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 60)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_5: "f32[8, 240, 28, 28]" = torch.ops.aten.cat.default([convolution_16, convolution_17, convolution_18, convolution_19], 1);  convolution_16 = convolution_17 = convolution_18 = convolution_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_52: "i64[]" = torch.ops.aten.add.Tensor(primals_336, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_10 = torch.ops.aten.var_mean.correction(cat_5, [0, 2, 3], correction = 0, keepdim = True)
    getitem_68: "f32[1, 240, 1, 1]" = var_mean_10[0]
    getitem_69: "f32[1, 240, 1, 1]" = var_mean_10[1];  var_mean_10 = None
    add_53: "f32[1, 240, 1, 1]" = torch.ops.aten.add.Tensor(getitem_68, 0.001)
    rsqrt_10: "f32[1, 240, 1, 1]" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
    sub_10: "f32[8, 240, 28, 28]" = torch.ops.aten.sub.Tensor(cat_5, getitem_69)
    mul_71: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
    squeeze_30: "f32[240]" = torch.ops.aten.squeeze.dims(getitem_69, [0, 2, 3]);  getitem_69 = None
    squeeze_31: "f32[240]" = torch.ops.aten.squeeze.dims(rsqrt_10, [0, 2, 3]);  rsqrt_10 = None
    mul_72: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_30, 0.1)
    mul_73: "f32[240]" = torch.ops.aten.mul.Tensor(primals_337, 0.9)
    add_54: "f32[240]" = torch.ops.aten.add.Tensor(mul_72, mul_73);  mul_72 = mul_73 = None
    squeeze_32: "f32[240]" = torch.ops.aten.squeeze.dims(getitem_68, [0, 2, 3]);  getitem_68 = None
    mul_74: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_32, 1.0001594642002871);  squeeze_32 = None
    mul_75: "f32[240]" = torch.ops.aten.mul.Tensor(mul_74, 0.1);  mul_74 = None
    mul_76: "f32[240]" = torch.ops.aten.mul.Tensor(primals_338, 0.9)
    add_55: "f32[240]" = torch.ops.aten.add.Tensor(mul_75, mul_76);  mul_75 = mul_76 = None
    unsqueeze_40: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_29, -1)
    unsqueeze_41: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
    mul_77: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(mul_71, unsqueeze_41);  mul_71 = unsqueeze_41 = None
    unsqueeze_42: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_30, -1);  primals_30 = None
    unsqueeze_43: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
    add_56: "f32[8, 240, 28, 28]" = torch.ops.aten.add.Tensor(mul_77, unsqueeze_43);  mul_77 = unsqueeze_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_1: "f32[8, 240, 28, 28]" = torch.ops.aten.clone.default(add_56)
    sigmoid_1: "f32[8, 240, 28, 28]" = torch.ops.aten.sigmoid.default(add_56)
    mul_78: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(add_56, sigmoid_1);  add_56 = sigmoid_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean: "f32[8, 240, 1, 1]" = torch.ops.aten.mean.dim(mul_78, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_20: "f32[8, 20, 1, 1]" = torch.ops.aten.convolution.default(mean, primals_144, primals_145, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_2: "f32[8, 20, 1, 1]" = torch.ops.aten.clone.default(convolution_20)
    sigmoid_2: "f32[8, 20, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_20)
    mul_79: "f32[8, 20, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_20, sigmoid_2);  convolution_20 = sigmoid_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_21: "f32[8, 240, 1, 1]" = torch.ops.aten.convolution.default(mul_79, primals_146, primals_147, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_3: "f32[8, 240, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_21);  convolution_21 = None
    alias_6: "f32[8, 240, 1, 1]" = torch.ops.aten.alias.default(sigmoid_3)
    mul_80: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(mul_78, sigmoid_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_22: "f32[8, 56, 28, 28]" = torch.ops.aten.convolution.default(mul_80, primals_148, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_57: "i64[]" = torch.ops.aten.add.Tensor(primals_339, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_11 = torch.ops.aten.var_mean.correction(convolution_22, [0, 2, 3], correction = 0, keepdim = True)
    getitem_70: "f32[1, 56, 1, 1]" = var_mean_11[0]
    getitem_71: "f32[1, 56, 1, 1]" = var_mean_11[1];  var_mean_11 = None
    add_58: "f32[1, 56, 1, 1]" = torch.ops.aten.add.Tensor(getitem_70, 0.001)
    rsqrt_11: "f32[1, 56, 1, 1]" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
    sub_11: "f32[8, 56, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_22, getitem_71)
    mul_81: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
    squeeze_33: "f32[56]" = torch.ops.aten.squeeze.dims(getitem_71, [0, 2, 3]);  getitem_71 = None
    squeeze_34: "f32[56]" = torch.ops.aten.squeeze.dims(rsqrt_11, [0, 2, 3]);  rsqrt_11 = None
    mul_82: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_33, 0.1)
    mul_83: "f32[56]" = torch.ops.aten.mul.Tensor(primals_340, 0.9)
    add_59: "f32[56]" = torch.ops.aten.add.Tensor(mul_82, mul_83);  mul_82 = mul_83 = None
    squeeze_35: "f32[56]" = torch.ops.aten.squeeze.dims(getitem_70, [0, 2, 3]);  getitem_70 = None
    mul_84: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_35, 1.0001594642002871);  squeeze_35 = None
    mul_85: "f32[56]" = torch.ops.aten.mul.Tensor(mul_84, 0.1);  mul_84 = None
    mul_86: "f32[56]" = torch.ops.aten.mul.Tensor(primals_341, 0.9)
    add_60: "f32[56]" = torch.ops.aten.add.Tensor(mul_85, mul_86);  mul_85 = mul_86 = None
    unsqueeze_44: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(primals_31, -1)
    unsqueeze_45: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
    mul_87: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(mul_81, unsqueeze_45);  mul_81 = unsqueeze_45 = None
    unsqueeze_46: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(primals_32, -1);  primals_32 = None
    unsqueeze_47: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
    add_61: "f32[8, 56, 28, 28]" = torch.ops.aten.add.Tensor(mul_87, unsqueeze_47);  mul_87 = unsqueeze_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_with_sizes_17 = torch.ops.aten.split_with_sizes.default(add_61, [28, 28], 1)
    getitem_72: "f32[8, 28, 28, 28]" = split_with_sizes_17[0]
    getitem_73: "f32[8, 28, 28, 28]" = split_with_sizes_17[1];  split_with_sizes_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_23: "f32[8, 168, 28, 28]" = torch.ops.aten.convolution.default(getitem_72, primals_149, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convolution_24: "f32[8, 168, 28, 28]" = torch.ops.aten.convolution.default(getitem_73, primals_150, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_6: "f32[8, 336, 28, 28]" = torch.ops.aten.cat.default([convolution_23, convolution_24], 1);  convolution_23 = convolution_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_62: "i64[]" = torch.ops.aten.add.Tensor(primals_342, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_12 = torch.ops.aten.var_mean.correction(cat_6, [0, 2, 3], correction = 0, keepdim = True)
    getitem_74: "f32[1, 336, 1, 1]" = var_mean_12[0]
    getitem_75: "f32[1, 336, 1, 1]" = var_mean_12[1];  var_mean_12 = None
    add_63: "f32[1, 336, 1, 1]" = torch.ops.aten.add.Tensor(getitem_74, 0.001)
    rsqrt_12: "f32[1, 336, 1, 1]" = torch.ops.aten.rsqrt.default(add_63);  add_63 = None
    sub_12: "f32[8, 336, 28, 28]" = torch.ops.aten.sub.Tensor(cat_6, getitem_75)
    mul_88: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
    squeeze_36: "f32[336]" = torch.ops.aten.squeeze.dims(getitem_75, [0, 2, 3]);  getitem_75 = None
    squeeze_37: "f32[336]" = torch.ops.aten.squeeze.dims(rsqrt_12, [0, 2, 3]);  rsqrt_12 = None
    mul_89: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_36, 0.1)
    mul_90: "f32[336]" = torch.ops.aten.mul.Tensor(primals_343, 0.9)
    add_64: "f32[336]" = torch.ops.aten.add.Tensor(mul_89, mul_90);  mul_89 = mul_90 = None
    squeeze_38: "f32[336]" = torch.ops.aten.squeeze.dims(getitem_74, [0, 2, 3]);  getitem_74 = None
    mul_91: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_38, 1.0001594642002871);  squeeze_38 = None
    mul_92: "f32[336]" = torch.ops.aten.mul.Tensor(mul_91, 0.1);  mul_91 = None
    mul_93: "f32[336]" = torch.ops.aten.mul.Tensor(primals_344, 0.9)
    add_65: "f32[336]" = torch.ops.aten.add.Tensor(mul_92, mul_93);  mul_92 = mul_93 = None
    unsqueeze_48: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(primals_33, -1)
    unsqueeze_49: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
    mul_94: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(mul_88, unsqueeze_49);  mul_88 = unsqueeze_49 = None
    unsqueeze_50: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(primals_34, -1);  primals_34 = None
    unsqueeze_51: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
    add_66: "f32[8, 336, 28, 28]" = torch.ops.aten.add.Tensor(mul_94, unsqueeze_51);  mul_94 = unsqueeze_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_3: "f32[8, 336, 28, 28]" = torch.ops.aten.clone.default(add_66)
    sigmoid_4: "f32[8, 336, 28, 28]" = torch.ops.aten.sigmoid.default(add_66)
    mul_95: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(add_66, sigmoid_4);  add_66 = sigmoid_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    split_with_sizes_19 = torch.ops.aten.split_with_sizes.default(mul_95, [168, 168], 1)
    getitem_78: "f32[8, 168, 28, 28]" = split_with_sizes_19[0];  split_with_sizes_19 = None
    convolution_25: "f32[8, 168, 28, 28]" = torch.ops.aten.convolution.default(getitem_78, primals_151, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 168)
    split_with_sizes_20 = torch.ops.aten.split_with_sizes.default(mul_95, [168, 168], 1);  mul_95 = None
    getitem_81: "f32[8, 168, 28, 28]" = split_with_sizes_20[1];  split_with_sizes_20 = None
    convolution_26: "f32[8, 168, 28, 28]" = torch.ops.aten.convolution.default(getitem_81, primals_152, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 168)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_7: "f32[8, 336, 28, 28]" = torch.ops.aten.cat.default([convolution_25, convolution_26], 1);  convolution_25 = convolution_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_67: "i64[]" = torch.ops.aten.add.Tensor(primals_345, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_13 = torch.ops.aten.var_mean.correction(cat_7, [0, 2, 3], correction = 0, keepdim = True)
    getitem_82: "f32[1, 336, 1, 1]" = var_mean_13[0]
    getitem_83: "f32[1, 336, 1, 1]" = var_mean_13[1];  var_mean_13 = None
    add_68: "f32[1, 336, 1, 1]" = torch.ops.aten.add.Tensor(getitem_82, 0.001)
    rsqrt_13: "f32[1, 336, 1, 1]" = torch.ops.aten.rsqrt.default(add_68);  add_68 = None
    sub_13: "f32[8, 336, 28, 28]" = torch.ops.aten.sub.Tensor(cat_7, getitem_83)
    mul_96: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
    squeeze_39: "f32[336]" = torch.ops.aten.squeeze.dims(getitem_83, [0, 2, 3]);  getitem_83 = None
    squeeze_40: "f32[336]" = torch.ops.aten.squeeze.dims(rsqrt_13, [0, 2, 3]);  rsqrt_13 = None
    mul_97: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_39, 0.1)
    mul_98: "f32[336]" = torch.ops.aten.mul.Tensor(primals_346, 0.9)
    add_69: "f32[336]" = torch.ops.aten.add.Tensor(mul_97, mul_98);  mul_97 = mul_98 = None
    squeeze_41: "f32[336]" = torch.ops.aten.squeeze.dims(getitem_82, [0, 2, 3]);  getitem_82 = None
    mul_99: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_41, 1.0001594642002871);  squeeze_41 = None
    mul_100: "f32[336]" = torch.ops.aten.mul.Tensor(mul_99, 0.1);  mul_99 = None
    mul_101: "f32[336]" = torch.ops.aten.mul.Tensor(primals_347, 0.9)
    add_70: "f32[336]" = torch.ops.aten.add.Tensor(mul_100, mul_101);  mul_100 = mul_101 = None
    unsqueeze_52: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(primals_35, -1)
    unsqueeze_53: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
    mul_102: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(mul_96, unsqueeze_53);  mul_96 = unsqueeze_53 = None
    unsqueeze_54: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(primals_36, -1);  primals_36 = None
    unsqueeze_55: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
    add_71: "f32[8, 336, 28, 28]" = torch.ops.aten.add.Tensor(mul_102, unsqueeze_55);  mul_102 = unsqueeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_4: "f32[8, 336, 28, 28]" = torch.ops.aten.clone.default(add_71)
    sigmoid_5: "f32[8, 336, 28, 28]" = torch.ops.aten.sigmoid.default(add_71)
    mul_103: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(add_71, sigmoid_5);  add_71 = sigmoid_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_1: "f32[8, 336, 1, 1]" = torch.ops.aten.mean.dim(mul_103, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_27: "f32[8, 28, 1, 1]" = torch.ops.aten.convolution.default(mean_1, primals_153, primals_154, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_5: "f32[8, 28, 1, 1]" = torch.ops.aten.clone.default(convolution_27)
    sigmoid_6: "f32[8, 28, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_27)
    mul_104: "f32[8, 28, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_27, sigmoid_6);  convolution_27 = sigmoid_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_28: "f32[8, 336, 1, 1]" = torch.ops.aten.convolution.default(mul_104, primals_155, primals_156, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_7: "f32[8, 336, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_28);  convolution_28 = None
    alias_7: "f32[8, 336, 1, 1]" = torch.ops.aten.alias.default(sigmoid_7)
    mul_105: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(mul_103, sigmoid_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_with_sizes_21 = torch.ops.aten.split_with_sizes.default(mul_105, [168, 168], 1);  mul_105 = None
    getitem_84: "f32[8, 168, 28, 28]" = split_with_sizes_21[0]
    getitem_85: "f32[8, 168, 28, 28]" = split_with_sizes_21[1];  split_with_sizes_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_29: "f32[8, 28, 28, 28]" = torch.ops.aten.convolution.default(getitem_84, primals_157, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convolution_30: "f32[8, 28, 28, 28]" = torch.ops.aten.convolution.default(getitem_85, primals_158, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_8: "f32[8, 56, 28, 28]" = torch.ops.aten.cat.default([convolution_29, convolution_30], 1);  convolution_29 = convolution_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_72: "i64[]" = torch.ops.aten.add.Tensor(primals_348, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_14 = torch.ops.aten.var_mean.correction(cat_8, [0, 2, 3], correction = 0, keepdim = True)
    getitem_86: "f32[1, 56, 1, 1]" = var_mean_14[0]
    getitem_87: "f32[1, 56, 1, 1]" = var_mean_14[1];  var_mean_14 = None
    add_73: "f32[1, 56, 1, 1]" = torch.ops.aten.add.Tensor(getitem_86, 0.001)
    rsqrt_14: "f32[1, 56, 1, 1]" = torch.ops.aten.rsqrt.default(add_73);  add_73 = None
    sub_14: "f32[8, 56, 28, 28]" = torch.ops.aten.sub.Tensor(cat_8, getitem_87)
    mul_106: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
    squeeze_42: "f32[56]" = torch.ops.aten.squeeze.dims(getitem_87, [0, 2, 3]);  getitem_87 = None
    squeeze_43: "f32[56]" = torch.ops.aten.squeeze.dims(rsqrt_14, [0, 2, 3]);  rsqrt_14 = None
    mul_107: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_42, 0.1)
    mul_108: "f32[56]" = torch.ops.aten.mul.Tensor(primals_349, 0.9)
    add_74: "f32[56]" = torch.ops.aten.add.Tensor(mul_107, mul_108);  mul_107 = mul_108 = None
    squeeze_44: "f32[56]" = torch.ops.aten.squeeze.dims(getitem_86, [0, 2, 3]);  getitem_86 = None
    mul_109: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_44, 1.0001594642002871);  squeeze_44 = None
    mul_110: "f32[56]" = torch.ops.aten.mul.Tensor(mul_109, 0.1);  mul_109 = None
    mul_111: "f32[56]" = torch.ops.aten.mul.Tensor(primals_350, 0.9)
    add_75: "f32[56]" = torch.ops.aten.add.Tensor(mul_110, mul_111);  mul_110 = mul_111 = None
    unsqueeze_56: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(primals_37, -1)
    unsqueeze_57: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
    mul_112: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(mul_106, unsqueeze_57);  mul_106 = unsqueeze_57 = None
    unsqueeze_58: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(primals_38, -1);  primals_38 = None
    unsqueeze_59: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
    add_76: "f32[8, 56, 28, 28]" = torch.ops.aten.add.Tensor(mul_112, unsqueeze_59);  mul_112 = unsqueeze_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_77: "f32[8, 56, 28, 28]" = torch.ops.aten.add.Tensor(add_76, add_61);  add_76 = add_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_with_sizes_22 = torch.ops.aten.split_with_sizes.default(add_77, [28, 28], 1)
    getitem_88: "f32[8, 28, 28, 28]" = split_with_sizes_22[0]
    getitem_89: "f32[8, 28, 28, 28]" = split_with_sizes_22[1];  split_with_sizes_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_31: "f32[8, 168, 28, 28]" = torch.ops.aten.convolution.default(getitem_88, primals_159, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convolution_32: "f32[8, 168, 28, 28]" = torch.ops.aten.convolution.default(getitem_89, primals_160, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_9: "f32[8, 336, 28, 28]" = torch.ops.aten.cat.default([convolution_31, convolution_32], 1);  convolution_31 = convolution_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_78: "i64[]" = torch.ops.aten.add.Tensor(primals_351, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_15 = torch.ops.aten.var_mean.correction(cat_9, [0, 2, 3], correction = 0, keepdim = True)
    getitem_90: "f32[1, 336, 1, 1]" = var_mean_15[0]
    getitem_91: "f32[1, 336, 1, 1]" = var_mean_15[1];  var_mean_15 = None
    add_79: "f32[1, 336, 1, 1]" = torch.ops.aten.add.Tensor(getitem_90, 0.001)
    rsqrt_15: "f32[1, 336, 1, 1]" = torch.ops.aten.rsqrt.default(add_79);  add_79 = None
    sub_15: "f32[8, 336, 28, 28]" = torch.ops.aten.sub.Tensor(cat_9, getitem_91)
    mul_113: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = None
    squeeze_45: "f32[336]" = torch.ops.aten.squeeze.dims(getitem_91, [0, 2, 3]);  getitem_91 = None
    squeeze_46: "f32[336]" = torch.ops.aten.squeeze.dims(rsqrt_15, [0, 2, 3]);  rsqrt_15 = None
    mul_114: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_45, 0.1)
    mul_115: "f32[336]" = torch.ops.aten.mul.Tensor(primals_352, 0.9)
    add_80: "f32[336]" = torch.ops.aten.add.Tensor(mul_114, mul_115);  mul_114 = mul_115 = None
    squeeze_47: "f32[336]" = torch.ops.aten.squeeze.dims(getitem_90, [0, 2, 3]);  getitem_90 = None
    mul_116: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_47, 1.0001594642002871);  squeeze_47 = None
    mul_117: "f32[336]" = torch.ops.aten.mul.Tensor(mul_116, 0.1);  mul_116 = None
    mul_118: "f32[336]" = torch.ops.aten.mul.Tensor(primals_353, 0.9)
    add_81: "f32[336]" = torch.ops.aten.add.Tensor(mul_117, mul_118);  mul_117 = mul_118 = None
    unsqueeze_60: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(primals_39, -1)
    unsqueeze_61: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
    mul_119: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(mul_113, unsqueeze_61);  mul_113 = unsqueeze_61 = None
    unsqueeze_62: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(primals_40, -1);  primals_40 = None
    unsqueeze_63: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
    add_82: "f32[8, 336, 28, 28]" = torch.ops.aten.add.Tensor(mul_119, unsqueeze_63);  mul_119 = unsqueeze_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_6: "f32[8, 336, 28, 28]" = torch.ops.aten.clone.default(add_82)
    sigmoid_8: "f32[8, 336, 28, 28]" = torch.ops.aten.sigmoid.default(add_82)
    mul_120: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(add_82, sigmoid_8);  add_82 = sigmoid_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    split_with_sizes_24 = torch.ops.aten.split_with_sizes.default(mul_120, [168, 168], 1)
    getitem_94: "f32[8, 168, 28, 28]" = split_with_sizes_24[0];  split_with_sizes_24 = None
    convolution_33: "f32[8, 168, 28, 28]" = torch.ops.aten.convolution.default(getitem_94, primals_161, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 168)
    split_with_sizes_25 = torch.ops.aten.split_with_sizes.default(mul_120, [168, 168], 1);  mul_120 = None
    getitem_97: "f32[8, 168, 28, 28]" = split_with_sizes_25[1];  split_with_sizes_25 = None
    convolution_34: "f32[8, 168, 28, 28]" = torch.ops.aten.convolution.default(getitem_97, primals_162, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 168)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_10: "f32[8, 336, 28, 28]" = torch.ops.aten.cat.default([convolution_33, convolution_34], 1);  convolution_33 = convolution_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_83: "i64[]" = torch.ops.aten.add.Tensor(primals_354, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_16 = torch.ops.aten.var_mean.correction(cat_10, [0, 2, 3], correction = 0, keepdim = True)
    getitem_98: "f32[1, 336, 1, 1]" = var_mean_16[0]
    getitem_99: "f32[1, 336, 1, 1]" = var_mean_16[1];  var_mean_16 = None
    add_84: "f32[1, 336, 1, 1]" = torch.ops.aten.add.Tensor(getitem_98, 0.001)
    rsqrt_16: "f32[1, 336, 1, 1]" = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
    sub_16: "f32[8, 336, 28, 28]" = torch.ops.aten.sub.Tensor(cat_10, getitem_99)
    mul_121: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
    squeeze_48: "f32[336]" = torch.ops.aten.squeeze.dims(getitem_99, [0, 2, 3]);  getitem_99 = None
    squeeze_49: "f32[336]" = torch.ops.aten.squeeze.dims(rsqrt_16, [0, 2, 3]);  rsqrt_16 = None
    mul_122: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_48, 0.1)
    mul_123: "f32[336]" = torch.ops.aten.mul.Tensor(primals_355, 0.9)
    add_85: "f32[336]" = torch.ops.aten.add.Tensor(mul_122, mul_123);  mul_122 = mul_123 = None
    squeeze_50: "f32[336]" = torch.ops.aten.squeeze.dims(getitem_98, [0, 2, 3]);  getitem_98 = None
    mul_124: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_50, 1.0001594642002871);  squeeze_50 = None
    mul_125: "f32[336]" = torch.ops.aten.mul.Tensor(mul_124, 0.1);  mul_124 = None
    mul_126: "f32[336]" = torch.ops.aten.mul.Tensor(primals_356, 0.9)
    add_86: "f32[336]" = torch.ops.aten.add.Tensor(mul_125, mul_126);  mul_125 = mul_126 = None
    unsqueeze_64: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(primals_41, -1)
    unsqueeze_65: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
    mul_127: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(mul_121, unsqueeze_65);  mul_121 = unsqueeze_65 = None
    unsqueeze_66: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(primals_42, -1);  primals_42 = None
    unsqueeze_67: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
    add_87: "f32[8, 336, 28, 28]" = torch.ops.aten.add.Tensor(mul_127, unsqueeze_67);  mul_127 = unsqueeze_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_7: "f32[8, 336, 28, 28]" = torch.ops.aten.clone.default(add_87)
    sigmoid_9: "f32[8, 336, 28, 28]" = torch.ops.aten.sigmoid.default(add_87)
    mul_128: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(add_87, sigmoid_9);  add_87 = sigmoid_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_2: "f32[8, 336, 1, 1]" = torch.ops.aten.mean.dim(mul_128, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_35: "f32[8, 28, 1, 1]" = torch.ops.aten.convolution.default(mean_2, primals_163, primals_164, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_8: "f32[8, 28, 1, 1]" = torch.ops.aten.clone.default(convolution_35)
    sigmoid_10: "f32[8, 28, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_35)
    mul_129: "f32[8, 28, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_35, sigmoid_10);  convolution_35 = sigmoid_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_36: "f32[8, 336, 1, 1]" = torch.ops.aten.convolution.default(mul_129, primals_165, primals_166, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_11: "f32[8, 336, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_36);  convolution_36 = None
    alias_8: "f32[8, 336, 1, 1]" = torch.ops.aten.alias.default(sigmoid_11)
    mul_130: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(mul_128, sigmoid_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_with_sizes_26 = torch.ops.aten.split_with_sizes.default(mul_130, [168, 168], 1);  mul_130 = None
    getitem_100: "f32[8, 168, 28, 28]" = split_with_sizes_26[0]
    getitem_101: "f32[8, 168, 28, 28]" = split_with_sizes_26[1];  split_with_sizes_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_37: "f32[8, 28, 28, 28]" = torch.ops.aten.convolution.default(getitem_100, primals_167, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convolution_38: "f32[8, 28, 28, 28]" = torch.ops.aten.convolution.default(getitem_101, primals_168, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_11: "f32[8, 56, 28, 28]" = torch.ops.aten.cat.default([convolution_37, convolution_38], 1);  convolution_37 = convolution_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_88: "i64[]" = torch.ops.aten.add.Tensor(primals_357, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_17 = torch.ops.aten.var_mean.correction(cat_11, [0, 2, 3], correction = 0, keepdim = True)
    getitem_102: "f32[1, 56, 1, 1]" = var_mean_17[0]
    getitem_103: "f32[1, 56, 1, 1]" = var_mean_17[1];  var_mean_17 = None
    add_89: "f32[1, 56, 1, 1]" = torch.ops.aten.add.Tensor(getitem_102, 0.001)
    rsqrt_17: "f32[1, 56, 1, 1]" = torch.ops.aten.rsqrt.default(add_89);  add_89 = None
    sub_17: "f32[8, 56, 28, 28]" = torch.ops.aten.sub.Tensor(cat_11, getitem_103)
    mul_131: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = None
    squeeze_51: "f32[56]" = torch.ops.aten.squeeze.dims(getitem_103, [0, 2, 3]);  getitem_103 = None
    squeeze_52: "f32[56]" = torch.ops.aten.squeeze.dims(rsqrt_17, [0, 2, 3]);  rsqrt_17 = None
    mul_132: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_51, 0.1)
    mul_133: "f32[56]" = torch.ops.aten.mul.Tensor(primals_358, 0.9)
    add_90: "f32[56]" = torch.ops.aten.add.Tensor(mul_132, mul_133);  mul_132 = mul_133 = None
    squeeze_53: "f32[56]" = torch.ops.aten.squeeze.dims(getitem_102, [0, 2, 3]);  getitem_102 = None
    mul_134: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_53, 1.0001594642002871);  squeeze_53 = None
    mul_135: "f32[56]" = torch.ops.aten.mul.Tensor(mul_134, 0.1);  mul_134 = None
    mul_136: "f32[56]" = torch.ops.aten.mul.Tensor(primals_359, 0.9)
    add_91: "f32[56]" = torch.ops.aten.add.Tensor(mul_135, mul_136);  mul_135 = mul_136 = None
    unsqueeze_68: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(primals_43, -1)
    unsqueeze_69: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
    mul_137: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(mul_131, unsqueeze_69);  mul_131 = unsqueeze_69 = None
    unsqueeze_70: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(primals_44, -1);  primals_44 = None
    unsqueeze_71: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
    add_92: "f32[8, 56, 28, 28]" = torch.ops.aten.add.Tensor(mul_137, unsqueeze_71);  mul_137 = unsqueeze_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_93: "f32[8, 56, 28, 28]" = torch.ops.aten.add.Tensor(add_92, add_77);  add_92 = add_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_with_sizes_27 = torch.ops.aten.split_with_sizes.default(add_93, [28, 28], 1)
    getitem_104: "f32[8, 28, 28, 28]" = split_with_sizes_27[0]
    getitem_105: "f32[8, 28, 28, 28]" = split_with_sizes_27[1];  split_with_sizes_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_39: "f32[8, 168, 28, 28]" = torch.ops.aten.convolution.default(getitem_104, primals_169, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convolution_40: "f32[8, 168, 28, 28]" = torch.ops.aten.convolution.default(getitem_105, primals_170, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_12: "f32[8, 336, 28, 28]" = torch.ops.aten.cat.default([convolution_39, convolution_40], 1);  convolution_39 = convolution_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_94: "i64[]" = torch.ops.aten.add.Tensor(primals_360, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_18 = torch.ops.aten.var_mean.correction(cat_12, [0, 2, 3], correction = 0, keepdim = True)
    getitem_106: "f32[1, 336, 1, 1]" = var_mean_18[0]
    getitem_107: "f32[1, 336, 1, 1]" = var_mean_18[1];  var_mean_18 = None
    add_95: "f32[1, 336, 1, 1]" = torch.ops.aten.add.Tensor(getitem_106, 0.001)
    rsqrt_18: "f32[1, 336, 1, 1]" = torch.ops.aten.rsqrt.default(add_95);  add_95 = None
    sub_18: "f32[8, 336, 28, 28]" = torch.ops.aten.sub.Tensor(cat_12, getitem_107)
    mul_138: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = None
    squeeze_54: "f32[336]" = torch.ops.aten.squeeze.dims(getitem_107, [0, 2, 3]);  getitem_107 = None
    squeeze_55: "f32[336]" = torch.ops.aten.squeeze.dims(rsqrt_18, [0, 2, 3]);  rsqrt_18 = None
    mul_139: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_54, 0.1)
    mul_140: "f32[336]" = torch.ops.aten.mul.Tensor(primals_361, 0.9)
    add_96: "f32[336]" = torch.ops.aten.add.Tensor(mul_139, mul_140);  mul_139 = mul_140 = None
    squeeze_56: "f32[336]" = torch.ops.aten.squeeze.dims(getitem_106, [0, 2, 3]);  getitem_106 = None
    mul_141: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_56, 1.0001594642002871);  squeeze_56 = None
    mul_142: "f32[336]" = torch.ops.aten.mul.Tensor(mul_141, 0.1);  mul_141 = None
    mul_143: "f32[336]" = torch.ops.aten.mul.Tensor(primals_362, 0.9)
    add_97: "f32[336]" = torch.ops.aten.add.Tensor(mul_142, mul_143);  mul_142 = mul_143 = None
    unsqueeze_72: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(primals_45, -1)
    unsqueeze_73: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
    mul_144: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(mul_138, unsqueeze_73);  mul_138 = unsqueeze_73 = None
    unsqueeze_74: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(primals_46, -1);  primals_46 = None
    unsqueeze_75: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
    add_98: "f32[8, 336, 28, 28]" = torch.ops.aten.add.Tensor(mul_144, unsqueeze_75);  mul_144 = unsqueeze_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_9: "f32[8, 336, 28, 28]" = torch.ops.aten.clone.default(add_98)
    sigmoid_12: "f32[8, 336, 28, 28]" = torch.ops.aten.sigmoid.default(add_98)
    mul_145: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(add_98, sigmoid_12);  add_98 = sigmoid_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    split_with_sizes_29 = torch.ops.aten.split_with_sizes.default(mul_145, [168, 168], 1)
    getitem_110: "f32[8, 168, 28, 28]" = split_with_sizes_29[0];  split_with_sizes_29 = None
    convolution_41: "f32[8, 168, 28, 28]" = torch.ops.aten.convolution.default(getitem_110, primals_171, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 168)
    split_with_sizes_30 = torch.ops.aten.split_with_sizes.default(mul_145, [168, 168], 1);  mul_145 = None
    getitem_113: "f32[8, 168, 28, 28]" = split_with_sizes_30[1];  split_with_sizes_30 = None
    convolution_42: "f32[8, 168, 28, 28]" = torch.ops.aten.convolution.default(getitem_113, primals_172, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 168)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_13: "f32[8, 336, 28, 28]" = torch.ops.aten.cat.default([convolution_41, convolution_42], 1);  convolution_41 = convolution_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_99: "i64[]" = torch.ops.aten.add.Tensor(primals_363, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_19 = torch.ops.aten.var_mean.correction(cat_13, [0, 2, 3], correction = 0, keepdim = True)
    getitem_114: "f32[1, 336, 1, 1]" = var_mean_19[0]
    getitem_115: "f32[1, 336, 1, 1]" = var_mean_19[1];  var_mean_19 = None
    add_100: "f32[1, 336, 1, 1]" = torch.ops.aten.add.Tensor(getitem_114, 0.001)
    rsqrt_19: "f32[1, 336, 1, 1]" = torch.ops.aten.rsqrt.default(add_100);  add_100 = None
    sub_19: "f32[8, 336, 28, 28]" = torch.ops.aten.sub.Tensor(cat_13, getitem_115)
    mul_146: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = None
    squeeze_57: "f32[336]" = torch.ops.aten.squeeze.dims(getitem_115, [0, 2, 3]);  getitem_115 = None
    squeeze_58: "f32[336]" = torch.ops.aten.squeeze.dims(rsqrt_19, [0, 2, 3]);  rsqrt_19 = None
    mul_147: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_57, 0.1)
    mul_148: "f32[336]" = torch.ops.aten.mul.Tensor(primals_364, 0.9)
    add_101: "f32[336]" = torch.ops.aten.add.Tensor(mul_147, mul_148);  mul_147 = mul_148 = None
    squeeze_59: "f32[336]" = torch.ops.aten.squeeze.dims(getitem_114, [0, 2, 3]);  getitem_114 = None
    mul_149: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_59, 1.0001594642002871);  squeeze_59 = None
    mul_150: "f32[336]" = torch.ops.aten.mul.Tensor(mul_149, 0.1);  mul_149 = None
    mul_151: "f32[336]" = torch.ops.aten.mul.Tensor(primals_365, 0.9)
    add_102: "f32[336]" = torch.ops.aten.add.Tensor(mul_150, mul_151);  mul_150 = mul_151 = None
    unsqueeze_76: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(primals_47, -1)
    unsqueeze_77: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
    mul_152: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(mul_146, unsqueeze_77);  mul_146 = unsqueeze_77 = None
    unsqueeze_78: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(primals_48, -1);  primals_48 = None
    unsqueeze_79: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
    add_103: "f32[8, 336, 28, 28]" = torch.ops.aten.add.Tensor(mul_152, unsqueeze_79);  mul_152 = unsqueeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_10: "f32[8, 336, 28, 28]" = torch.ops.aten.clone.default(add_103)
    sigmoid_13: "f32[8, 336, 28, 28]" = torch.ops.aten.sigmoid.default(add_103)
    mul_153: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(add_103, sigmoid_13);  add_103 = sigmoid_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_3: "f32[8, 336, 1, 1]" = torch.ops.aten.mean.dim(mul_153, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_43: "f32[8, 28, 1, 1]" = torch.ops.aten.convolution.default(mean_3, primals_173, primals_174, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_11: "f32[8, 28, 1, 1]" = torch.ops.aten.clone.default(convolution_43)
    sigmoid_14: "f32[8, 28, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_43)
    mul_154: "f32[8, 28, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_43, sigmoid_14);  convolution_43 = sigmoid_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_44: "f32[8, 336, 1, 1]" = torch.ops.aten.convolution.default(mul_154, primals_175, primals_176, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_15: "f32[8, 336, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_44);  convolution_44 = None
    alias_9: "f32[8, 336, 1, 1]" = torch.ops.aten.alias.default(sigmoid_15)
    mul_155: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(mul_153, sigmoid_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_with_sizes_31 = torch.ops.aten.split_with_sizes.default(mul_155, [168, 168], 1);  mul_155 = None
    getitem_116: "f32[8, 168, 28, 28]" = split_with_sizes_31[0]
    getitem_117: "f32[8, 168, 28, 28]" = split_with_sizes_31[1];  split_with_sizes_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_45: "f32[8, 28, 28, 28]" = torch.ops.aten.convolution.default(getitem_116, primals_177, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convolution_46: "f32[8, 28, 28, 28]" = torch.ops.aten.convolution.default(getitem_117, primals_178, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_14: "f32[8, 56, 28, 28]" = torch.ops.aten.cat.default([convolution_45, convolution_46], 1);  convolution_45 = convolution_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_104: "i64[]" = torch.ops.aten.add.Tensor(primals_366, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_20 = torch.ops.aten.var_mean.correction(cat_14, [0, 2, 3], correction = 0, keepdim = True)
    getitem_118: "f32[1, 56, 1, 1]" = var_mean_20[0]
    getitem_119: "f32[1, 56, 1, 1]" = var_mean_20[1];  var_mean_20 = None
    add_105: "f32[1, 56, 1, 1]" = torch.ops.aten.add.Tensor(getitem_118, 0.001)
    rsqrt_20: "f32[1, 56, 1, 1]" = torch.ops.aten.rsqrt.default(add_105);  add_105 = None
    sub_20: "f32[8, 56, 28, 28]" = torch.ops.aten.sub.Tensor(cat_14, getitem_119)
    mul_156: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = None
    squeeze_60: "f32[56]" = torch.ops.aten.squeeze.dims(getitem_119, [0, 2, 3]);  getitem_119 = None
    squeeze_61: "f32[56]" = torch.ops.aten.squeeze.dims(rsqrt_20, [0, 2, 3]);  rsqrt_20 = None
    mul_157: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_60, 0.1)
    mul_158: "f32[56]" = torch.ops.aten.mul.Tensor(primals_367, 0.9)
    add_106: "f32[56]" = torch.ops.aten.add.Tensor(mul_157, mul_158);  mul_157 = mul_158 = None
    squeeze_62: "f32[56]" = torch.ops.aten.squeeze.dims(getitem_118, [0, 2, 3]);  getitem_118 = None
    mul_159: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_62, 1.0001594642002871);  squeeze_62 = None
    mul_160: "f32[56]" = torch.ops.aten.mul.Tensor(mul_159, 0.1);  mul_159 = None
    mul_161: "f32[56]" = torch.ops.aten.mul.Tensor(primals_368, 0.9)
    add_107: "f32[56]" = torch.ops.aten.add.Tensor(mul_160, mul_161);  mul_160 = mul_161 = None
    unsqueeze_80: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(primals_49, -1)
    unsqueeze_81: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
    mul_162: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(mul_156, unsqueeze_81);  mul_156 = unsqueeze_81 = None
    unsqueeze_82: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(primals_50, -1);  primals_50 = None
    unsqueeze_83: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
    add_108: "f32[8, 56, 28, 28]" = torch.ops.aten.add.Tensor(mul_162, unsqueeze_83);  mul_162 = unsqueeze_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_109: "f32[8, 56, 28, 28]" = torch.ops.aten.add.Tensor(add_108, add_93);  add_108 = add_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_47: "f32[8, 336, 28, 28]" = torch.ops.aten.convolution.default(add_109, primals_179, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_110: "i64[]" = torch.ops.aten.add.Tensor(primals_369, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_21 = torch.ops.aten.var_mean.correction(convolution_47, [0, 2, 3], correction = 0, keepdim = True)
    getitem_120: "f32[1, 336, 1, 1]" = var_mean_21[0]
    getitem_121: "f32[1, 336, 1, 1]" = var_mean_21[1];  var_mean_21 = None
    add_111: "f32[1, 336, 1, 1]" = torch.ops.aten.add.Tensor(getitem_120, 0.001)
    rsqrt_21: "f32[1, 336, 1, 1]" = torch.ops.aten.rsqrt.default(add_111);  add_111 = None
    sub_21: "f32[8, 336, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_47, getitem_121)
    mul_163: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = None
    squeeze_63: "f32[336]" = torch.ops.aten.squeeze.dims(getitem_121, [0, 2, 3]);  getitem_121 = None
    squeeze_64: "f32[336]" = torch.ops.aten.squeeze.dims(rsqrt_21, [0, 2, 3]);  rsqrt_21 = None
    mul_164: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_63, 0.1)
    mul_165: "f32[336]" = torch.ops.aten.mul.Tensor(primals_370, 0.9)
    add_112: "f32[336]" = torch.ops.aten.add.Tensor(mul_164, mul_165);  mul_164 = mul_165 = None
    squeeze_65: "f32[336]" = torch.ops.aten.squeeze.dims(getitem_120, [0, 2, 3]);  getitem_120 = None
    mul_166: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_65, 1.0001594642002871);  squeeze_65 = None
    mul_167: "f32[336]" = torch.ops.aten.mul.Tensor(mul_166, 0.1);  mul_166 = None
    mul_168: "f32[336]" = torch.ops.aten.mul.Tensor(primals_371, 0.9)
    add_113: "f32[336]" = torch.ops.aten.add.Tensor(mul_167, mul_168);  mul_167 = mul_168 = None
    unsqueeze_84: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(primals_51, -1)
    unsqueeze_85: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
    mul_169: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(mul_163, unsqueeze_85);  mul_163 = unsqueeze_85 = None
    unsqueeze_86: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(primals_52, -1);  primals_52 = None
    unsqueeze_87: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
    add_114: "f32[8, 336, 28, 28]" = torch.ops.aten.add.Tensor(mul_169, unsqueeze_87);  mul_169 = unsqueeze_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_12: "f32[8, 336, 28, 28]" = torch.ops.aten.clone.default(add_114)
    sigmoid_16: "f32[8, 336, 28, 28]" = torch.ops.aten.sigmoid.default(add_114)
    mul_170: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(add_114, sigmoid_16);  add_114 = sigmoid_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    split_with_sizes_33 = torch.ops.aten.split_with_sizes.default(mul_170, [112, 112, 112], 1)
    getitem_125: "f32[8, 112, 28, 28]" = split_with_sizes_33[0];  split_with_sizes_33 = None
    constant_pad_nd_8: "f32[8, 112, 29, 29]" = torch.ops.aten.constant_pad_nd.default(getitem_125, [0, 1, 0, 1], 0.0);  getitem_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_48: "f32[8, 112, 14, 14]" = torch.ops.aten.convolution.default(constant_pad_nd_8, primals_53, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 112)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    split_with_sizes_34 = torch.ops.aten.split_with_sizes.default(mul_170, [112, 112, 112], 1)
    getitem_129: "f32[8, 112, 28, 28]" = split_with_sizes_34[1];  split_with_sizes_34 = None
    constant_pad_nd_9: "f32[8, 112, 31, 31]" = torch.ops.aten.constant_pad_nd.default(getitem_129, [1, 2, 1, 2], 0.0);  getitem_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_49: "f32[8, 112, 14, 14]" = torch.ops.aten.convolution.default(constant_pad_nd_9, primals_54, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 112)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    split_with_sizes_35 = torch.ops.aten.split_with_sizes.default(mul_170, [112, 112, 112], 1);  mul_170 = None
    getitem_133: "f32[8, 112, 28, 28]" = split_with_sizes_35[2];  split_with_sizes_35 = None
    constant_pad_nd_10: "f32[8, 112, 33, 33]" = torch.ops.aten.constant_pad_nd.default(getitem_133, [2, 3, 2, 3], 0.0);  getitem_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_50: "f32[8, 112, 14, 14]" = torch.ops.aten.convolution.default(constant_pad_nd_10, primals_55, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 112)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_15: "f32[8, 336, 14, 14]" = torch.ops.aten.cat.default([convolution_48, convolution_49, convolution_50], 1);  convolution_48 = convolution_49 = convolution_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_115: "i64[]" = torch.ops.aten.add.Tensor(primals_372, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_22 = torch.ops.aten.var_mean.correction(cat_15, [0, 2, 3], correction = 0, keepdim = True)
    getitem_134: "f32[1, 336, 1, 1]" = var_mean_22[0]
    getitem_135: "f32[1, 336, 1, 1]" = var_mean_22[1];  var_mean_22 = None
    add_116: "f32[1, 336, 1, 1]" = torch.ops.aten.add.Tensor(getitem_134, 0.001)
    rsqrt_22: "f32[1, 336, 1, 1]" = torch.ops.aten.rsqrt.default(add_116);  add_116 = None
    sub_22: "f32[8, 336, 14, 14]" = torch.ops.aten.sub.Tensor(cat_15, getitem_135)
    mul_171: "f32[8, 336, 14, 14]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = None
    squeeze_66: "f32[336]" = torch.ops.aten.squeeze.dims(getitem_135, [0, 2, 3]);  getitem_135 = None
    squeeze_67: "f32[336]" = torch.ops.aten.squeeze.dims(rsqrt_22, [0, 2, 3]);  rsqrt_22 = None
    mul_172: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_66, 0.1)
    mul_173: "f32[336]" = torch.ops.aten.mul.Tensor(primals_373, 0.9)
    add_117: "f32[336]" = torch.ops.aten.add.Tensor(mul_172, mul_173);  mul_172 = mul_173 = None
    squeeze_68: "f32[336]" = torch.ops.aten.squeeze.dims(getitem_134, [0, 2, 3]);  getitem_134 = None
    mul_174: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_68, 1.0006381620931717);  squeeze_68 = None
    mul_175: "f32[336]" = torch.ops.aten.mul.Tensor(mul_174, 0.1);  mul_174 = None
    mul_176: "f32[336]" = torch.ops.aten.mul.Tensor(primals_374, 0.9)
    add_118: "f32[336]" = torch.ops.aten.add.Tensor(mul_175, mul_176);  mul_175 = mul_176 = None
    unsqueeze_88: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(primals_56, -1)
    unsqueeze_89: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
    mul_177: "f32[8, 336, 14, 14]" = torch.ops.aten.mul.Tensor(mul_171, unsqueeze_89);  mul_171 = unsqueeze_89 = None
    unsqueeze_90: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(primals_57, -1);  primals_57 = None
    unsqueeze_91: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
    add_119: "f32[8, 336, 14, 14]" = torch.ops.aten.add.Tensor(mul_177, unsqueeze_91);  mul_177 = unsqueeze_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_13: "f32[8, 336, 14, 14]" = torch.ops.aten.clone.default(add_119)
    sigmoid_17: "f32[8, 336, 14, 14]" = torch.ops.aten.sigmoid.default(add_119)
    mul_178: "f32[8, 336, 14, 14]" = torch.ops.aten.mul.Tensor(add_119, sigmoid_17);  add_119 = sigmoid_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_4: "f32[8, 336, 1, 1]" = torch.ops.aten.mean.dim(mul_178, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_51: "f32[8, 14, 1, 1]" = torch.ops.aten.convolution.default(mean_4, primals_180, primals_181, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_14: "f32[8, 14, 1, 1]" = torch.ops.aten.clone.default(convolution_51)
    sigmoid_18: "f32[8, 14, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_51)
    mul_179: "f32[8, 14, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_51, sigmoid_18);  convolution_51 = sigmoid_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_52: "f32[8, 336, 1, 1]" = torch.ops.aten.convolution.default(mul_179, primals_182, primals_183, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_19: "f32[8, 336, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_52);  convolution_52 = None
    alias_10: "f32[8, 336, 1, 1]" = torch.ops.aten.alias.default(sigmoid_19)
    mul_180: "f32[8, 336, 14, 14]" = torch.ops.aten.mul.Tensor(mul_178, sigmoid_19)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_53: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(mul_180, primals_184, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_120: "i64[]" = torch.ops.aten.add.Tensor(primals_375, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_23 = torch.ops.aten.var_mean.correction(convolution_53, [0, 2, 3], correction = 0, keepdim = True)
    getitem_136: "f32[1, 104, 1, 1]" = var_mean_23[0]
    getitem_137: "f32[1, 104, 1, 1]" = var_mean_23[1];  var_mean_23 = None
    add_121: "f32[1, 104, 1, 1]" = torch.ops.aten.add.Tensor(getitem_136, 0.001)
    rsqrt_23: "f32[1, 104, 1, 1]" = torch.ops.aten.rsqrt.default(add_121);  add_121 = None
    sub_23: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_53, getitem_137)
    mul_181: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = None
    squeeze_69: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_137, [0, 2, 3]);  getitem_137 = None
    squeeze_70: "f32[104]" = torch.ops.aten.squeeze.dims(rsqrt_23, [0, 2, 3]);  rsqrt_23 = None
    mul_182: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_69, 0.1)
    mul_183: "f32[104]" = torch.ops.aten.mul.Tensor(primals_376, 0.9)
    add_122: "f32[104]" = torch.ops.aten.add.Tensor(mul_182, mul_183);  mul_182 = mul_183 = None
    squeeze_71: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_136, [0, 2, 3]);  getitem_136 = None
    mul_184: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_71, 1.0006381620931717);  squeeze_71 = None
    mul_185: "f32[104]" = torch.ops.aten.mul.Tensor(mul_184, 0.1);  mul_184 = None
    mul_186: "f32[104]" = torch.ops.aten.mul.Tensor(primals_377, 0.9)
    add_123: "f32[104]" = torch.ops.aten.add.Tensor(mul_185, mul_186);  mul_185 = mul_186 = None
    unsqueeze_92: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_58, -1)
    unsqueeze_93: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
    mul_187: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_181, unsqueeze_93);  mul_181 = unsqueeze_93 = None
    unsqueeze_94: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_59, -1);  primals_59 = None
    unsqueeze_95: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
    add_124: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_187, unsqueeze_95);  mul_187 = unsqueeze_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_with_sizes_36 = torch.ops.aten.split_with_sizes.default(add_124, [52, 52], 1)
    getitem_138: "f32[8, 52, 14, 14]" = split_with_sizes_36[0]
    getitem_139: "f32[8, 52, 14, 14]" = split_with_sizes_36[1];  split_with_sizes_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_54: "f32[8, 312, 14, 14]" = torch.ops.aten.convolution.default(getitem_138, primals_185, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convolution_55: "f32[8, 312, 14, 14]" = torch.ops.aten.convolution.default(getitem_139, primals_186, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_16: "f32[8, 624, 14, 14]" = torch.ops.aten.cat.default([convolution_54, convolution_55], 1);  convolution_54 = convolution_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_125: "i64[]" = torch.ops.aten.add.Tensor(primals_378, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_24 = torch.ops.aten.var_mean.correction(cat_16, [0, 2, 3], correction = 0, keepdim = True)
    getitem_140: "f32[1, 624, 1, 1]" = var_mean_24[0]
    getitem_141: "f32[1, 624, 1, 1]" = var_mean_24[1];  var_mean_24 = None
    add_126: "f32[1, 624, 1, 1]" = torch.ops.aten.add.Tensor(getitem_140, 0.001)
    rsqrt_24: "f32[1, 624, 1, 1]" = torch.ops.aten.rsqrt.default(add_126);  add_126 = None
    sub_24: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(cat_16, getitem_141)
    mul_188: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = None
    squeeze_72: "f32[624]" = torch.ops.aten.squeeze.dims(getitem_141, [0, 2, 3]);  getitem_141 = None
    squeeze_73: "f32[624]" = torch.ops.aten.squeeze.dims(rsqrt_24, [0, 2, 3]);  rsqrt_24 = None
    mul_189: "f32[624]" = torch.ops.aten.mul.Tensor(squeeze_72, 0.1)
    mul_190: "f32[624]" = torch.ops.aten.mul.Tensor(primals_379, 0.9)
    add_127: "f32[624]" = torch.ops.aten.add.Tensor(mul_189, mul_190);  mul_189 = mul_190 = None
    squeeze_74: "f32[624]" = torch.ops.aten.squeeze.dims(getitem_140, [0, 2, 3]);  getitem_140 = None
    mul_191: "f32[624]" = torch.ops.aten.mul.Tensor(squeeze_74, 1.0006381620931717);  squeeze_74 = None
    mul_192: "f32[624]" = torch.ops.aten.mul.Tensor(mul_191, 0.1);  mul_191 = None
    mul_193: "f32[624]" = torch.ops.aten.mul.Tensor(primals_380, 0.9)
    add_128: "f32[624]" = torch.ops.aten.add.Tensor(mul_192, mul_193);  mul_192 = mul_193 = None
    unsqueeze_96: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(primals_60, -1)
    unsqueeze_97: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
    mul_194: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(mul_188, unsqueeze_97);  mul_188 = unsqueeze_97 = None
    unsqueeze_98: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(primals_61, -1);  primals_61 = None
    unsqueeze_99: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
    add_129: "f32[8, 624, 14, 14]" = torch.ops.aten.add.Tensor(mul_194, unsqueeze_99);  mul_194 = unsqueeze_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_15: "f32[8, 624, 14, 14]" = torch.ops.aten.clone.default(add_129)
    sigmoid_20: "f32[8, 624, 14, 14]" = torch.ops.aten.sigmoid.default(add_129)
    mul_195: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(add_129, sigmoid_20);  add_129 = sigmoid_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    split_with_sizes_38 = torch.ops.aten.split_with_sizes.default(mul_195, [156, 156, 156, 156], 1)
    getitem_146: "f32[8, 156, 14, 14]" = split_with_sizes_38[0];  split_with_sizes_38 = None
    convolution_56: "f32[8, 156, 14, 14]" = torch.ops.aten.convolution.default(getitem_146, primals_187, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 156)
    split_with_sizes_39 = torch.ops.aten.split_with_sizes.default(mul_195, [156, 156, 156, 156], 1)
    getitem_151: "f32[8, 156, 14, 14]" = split_with_sizes_39[1];  split_with_sizes_39 = None
    convolution_57: "f32[8, 156, 14, 14]" = torch.ops.aten.convolution.default(getitem_151, primals_188, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 156)
    split_with_sizes_40 = torch.ops.aten.split_with_sizes.default(mul_195, [156, 156, 156, 156], 1)
    getitem_156: "f32[8, 156, 14, 14]" = split_with_sizes_40[2];  split_with_sizes_40 = None
    convolution_58: "f32[8, 156, 14, 14]" = torch.ops.aten.convolution.default(getitem_156, primals_189, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 156)
    split_with_sizes_41 = torch.ops.aten.split_with_sizes.default(mul_195, [156, 156, 156, 156], 1);  mul_195 = None
    getitem_161: "f32[8, 156, 14, 14]" = split_with_sizes_41[3];  split_with_sizes_41 = None
    convolution_59: "f32[8, 156, 14, 14]" = torch.ops.aten.convolution.default(getitem_161, primals_190, None, [1, 1], [4, 4], [1, 1], False, [0, 0], 156)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_17: "f32[8, 624, 14, 14]" = torch.ops.aten.cat.default([convolution_56, convolution_57, convolution_58, convolution_59], 1);  convolution_56 = convolution_57 = convolution_58 = convolution_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_130: "i64[]" = torch.ops.aten.add.Tensor(primals_381, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_25 = torch.ops.aten.var_mean.correction(cat_17, [0, 2, 3], correction = 0, keepdim = True)
    getitem_162: "f32[1, 624, 1, 1]" = var_mean_25[0]
    getitem_163: "f32[1, 624, 1, 1]" = var_mean_25[1];  var_mean_25 = None
    add_131: "f32[1, 624, 1, 1]" = torch.ops.aten.add.Tensor(getitem_162, 0.001)
    rsqrt_25: "f32[1, 624, 1, 1]" = torch.ops.aten.rsqrt.default(add_131);  add_131 = None
    sub_25: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(cat_17, getitem_163)
    mul_196: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = None
    squeeze_75: "f32[624]" = torch.ops.aten.squeeze.dims(getitem_163, [0, 2, 3]);  getitem_163 = None
    squeeze_76: "f32[624]" = torch.ops.aten.squeeze.dims(rsqrt_25, [0, 2, 3]);  rsqrt_25 = None
    mul_197: "f32[624]" = torch.ops.aten.mul.Tensor(squeeze_75, 0.1)
    mul_198: "f32[624]" = torch.ops.aten.mul.Tensor(primals_382, 0.9)
    add_132: "f32[624]" = torch.ops.aten.add.Tensor(mul_197, mul_198);  mul_197 = mul_198 = None
    squeeze_77: "f32[624]" = torch.ops.aten.squeeze.dims(getitem_162, [0, 2, 3]);  getitem_162 = None
    mul_199: "f32[624]" = torch.ops.aten.mul.Tensor(squeeze_77, 1.0006381620931717);  squeeze_77 = None
    mul_200: "f32[624]" = torch.ops.aten.mul.Tensor(mul_199, 0.1);  mul_199 = None
    mul_201: "f32[624]" = torch.ops.aten.mul.Tensor(primals_383, 0.9)
    add_133: "f32[624]" = torch.ops.aten.add.Tensor(mul_200, mul_201);  mul_200 = mul_201 = None
    unsqueeze_100: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(primals_62, -1)
    unsqueeze_101: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
    mul_202: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(mul_196, unsqueeze_101);  mul_196 = unsqueeze_101 = None
    unsqueeze_102: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(primals_63, -1);  primals_63 = None
    unsqueeze_103: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
    add_134: "f32[8, 624, 14, 14]" = torch.ops.aten.add.Tensor(mul_202, unsqueeze_103);  mul_202 = unsqueeze_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_16: "f32[8, 624, 14, 14]" = torch.ops.aten.clone.default(add_134)
    sigmoid_21: "f32[8, 624, 14, 14]" = torch.ops.aten.sigmoid.default(add_134)
    mul_203: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(add_134, sigmoid_21);  add_134 = sigmoid_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_5: "f32[8, 624, 1, 1]" = torch.ops.aten.mean.dim(mul_203, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_60: "f32[8, 26, 1, 1]" = torch.ops.aten.convolution.default(mean_5, primals_191, primals_192, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_17: "f32[8, 26, 1, 1]" = torch.ops.aten.clone.default(convolution_60)
    sigmoid_22: "f32[8, 26, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_60)
    mul_204: "f32[8, 26, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_60, sigmoid_22);  convolution_60 = sigmoid_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_61: "f32[8, 624, 1, 1]" = torch.ops.aten.convolution.default(mul_204, primals_193, primals_194, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_23: "f32[8, 624, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_61);  convolution_61 = None
    alias_11: "f32[8, 624, 1, 1]" = torch.ops.aten.alias.default(sigmoid_23)
    mul_205: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(mul_203, sigmoid_23)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_with_sizes_42 = torch.ops.aten.split_with_sizes.default(mul_205, [312, 312], 1);  mul_205 = None
    getitem_164: "f32[8, 312, 14, 14]" = split_with_sizes_42[0]
    getitem_165: "f32[8, 312, 14, 14]" = split_with_sizes_42[1];  split_with_sizes_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_62: "f32[8, 52, 14, 14]" = torch.ops.aten.convolution.default(getitem_164, primals_195, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convolution_63: "f32[8, 52, 14, 14]" = torch.ops.aten.convolution.default(getitem_165, primals_196, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_18: "f32[8, 104, 14, 14]" = torch.ops.aten.cat.default([convolution_62, convolution_63], 1);  convolution_62 = convolution_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_135: "i64[]" = torch.ops.aten.add.Tensor(primals_384, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_26 = torch.ops.aten.var_mean.correction(cat_18, [0, 2, 3], correction = 0, keepdim = True)
    getitem_166: "f32[1, 104, 1, 1]" = var_mean_26[0]
    getitem_167: "f32[1, 104, 1, 1]" = var_mean_26[1];  var_mean_26 = None
    add_136: "f32[1, 104, 1, 1]" = torch.ops.aten.add.Tensor(getitem_166, 0.001)
    rsqrt_26: "f32[1, 104, 1, 1]" = torch.ops.aten.rsqrt.default(add_136);  add_136 = None
    sub_26: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(cat_18, getitem_167)
    mul_206: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_26);  sub_26 = None
    squeeze_78: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_167, [0, 2, 3]);  getitem_167 = None
    squeeze_79: "f32[104]" = torch.ops.aten.squeeze.dims(rsqrt_26, [0, 2, 3]);  rsqrt_26 = None
    mul_207: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_78, 0.1)
    mul_208: "f32[104]" = torch.ops.aten.mul.Tensor(primals_385, 0.9)
    add_137: "f32[104]" = torch.ops.aten.add.Tensor(mul_207, mul_208);  mul_207 = mul_208 = None
    squeeze_80: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_166, [0, 2, 3]);  getitem_166 = None
    mul_209: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_80, 1.0006381620931717);  squeeze_80 = None
    mul_210: "f32[104]" = torch.ops.aten.mul.Tensor(mul_209, 0.1);  mul_209 = None
    mul_211: "f32[104]" = torch.ops.aten.mul.Tensor(primals_386, 0.9)
    add_138: "f32[104]" = torch.ops.aten.add.Tensor(mul_210, mul_211);  mul_210 = mul_211 = None
    unsqueeze_104: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_64, -1)
    unsqueeze_105: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
    mul_212: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_206, unsqueeze_105);  mul_206 = unsqueeze_105 = None
    unsqueeze_106: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_65, -1);  primals_65 = None
    unsqueeze_107: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
    add_139: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_212, unsqueeze_107);  mul_212 = unsqueeze_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_140: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(add_139, add_124);  add_139 = add_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_with_sizes_43 = torch.ops.aten.split_with_sizes.default(add_140, [52, 52], 1)
    getitem_168: "f32[8, 52, 14, 14]" = split_with_sizes_43[0]
    getitem_169: "f32[8, 52, 14, 14]" = split_with_sizes_43[1];  split_with_sizes_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_64: "f32[8, 312, 14, 14]" = torch.ops.aten.convolution.default(getitem_168, primals_197, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convolution_65: "f32[8, 312, 14, 14]" = torch.ops.aten.convolution.default(getitem_169, primals_198, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_19: "f32[8, 624, 14, 14]" = torch.ops.aten.cat.default([convolution_64, convolution_65], 1);  convolution_64 = convolution_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_141: "i64[]" = torch.ops.aten.add.Tensor(primals_387, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_27 = torch.ops.aten.var_mean.correction(cat_19, [0, 2, 3], correction = 0, keepdim = True)
    getitem_170: "f32[1, 624, 1, 1]" = var_mean_27[0]
    getitem_171: "f32[1, 624, 1, 1]" = var_mean_27[1];  var_mean_27 = None
    add_142: "f32[1, 624, 1, 1]" = torch.ops.aten.add.Tensor(getitem_170, 0.001)
    rsqrt_27: "f32[1, 624, 1, 1]" = torch.ops.aten.rsqrt.default(add_142);  add_142 = None
    sub_27: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(cat_19, getitem_171)
    mul_213: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_27);  sub_27 = None
    squeeze_81: "f32[624]" = torch.ops.aten.squeeze.dims(getitem_171, [0, 2, 3]);  getitem_171 = None
    squeeze_82: "f32[624]" = torch.ops.aten.squeeze.dims(rsqrt_27, [0, 2, 3]);  rsqrt_27 = None
    mul_214: "f32[624]" = torch.ops.aten.mul.Tensor(squeeze_81, 0.1)
    mul_215: "f32[624]" = torch.ops.aten.mul.Tensor(primals_388, 0.9)
    add_143: "f32[624]" = torch.ops.aten.add.Tensor(mul_214, mul_215);  mul_214 = mul_215 = None
    squeeze_83: "f32[624]" = torch.ops.aten.squeeze.dims(getitem_170, [0, 2, 3]);  getitem_170 = None
    mul_216: "f32[624]" = torch.ops.aten.mul.Tensor(squeeze_83, 1.0006381620931717);  squeeze_83 = None
    mul_217: "f32[624]" = torch.ops.aten.mul.Tensor(mul_216, 0.1);  mul_216 = None
    mul_218: "f32[624]" = torch.ops.aten.mul.Tensor(primals_389, 0.9)
    add_144: "f32[624]" = torch.ops.aten.add.Tensor(mul_217, mul_218);  mul_217 = mul_218 = None
    unsqueeze_108: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(primals_66, -1)
    unsqueeze_109: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
    mul_219: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(mul_213, unsqueeze_109);  mul_213 = unsqueeze_109 = None
    unsqueeze_110: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(primals_67, -1);  primals_67 = None
    unsqueeze_111: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
    add_145: "f32[8, 624, 14, 14]" = torch.ops.aten.add.Tensor(mul_219, unsqueeze_111);  mul_219 = unsqueeze_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_18: "f32[8, 624, 14, 14]" = torch.ops.aten.clone.default(add_145)
    sigmoid_24: "f32[8, 624, 14, 14]" = torch.ops.aten.sigmoid.default(add_145)
    mul_220: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(add_145, sigmoid_24);  add_145 = sigmoid_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    split_with_sizes_45 = torch.ops.aten.split_with_sizes.default(mul_220, [156, 156, 156, 156], 1)
    getitem_176: "f32[8, 156, 14, 14]" = split_with_sizes_45[0];  split_with_sizes_45 = None
    convolution_66: "f32[8, 156, 14, 14]" = torch.ops.aten.convolution.default(getitem_176, primals_199, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 156)
    split_with_sizes_46 = torch.ops.aten.split_with_sizes.default(mul_220, [156, 156, 156, 156], 1)
    getitem_181: "f32[8, 156, 14, 14]" = split_with_sizes_46[1];  split_with_sizes_46 = None
    convolution_67: "f32[8, 156, 14, 14]" = torch.ops.aten.convolution.default(getitem_181, primals_200, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 156)
    split_with_sizes_47 = torch.ops.aten.split_with_sizes.default(mul_220, [156, 156, 156, 156], 1)
    getitem_186: "f32[8, 156, 14, 14]" = split_with_sizes_47[2];  split_with_sizes_47 = None
    convolution_68: "f32[8, 156, 14, 14]" = torch.ops.aten.convolution.default(getitem_186, primals_201, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 156)
    split_with_sizes_48 = torch.ops.aten.split_with_sizes.default(mul_220, [156, 156, 156, 156], 1);  mul_220 = None
    getitem_191: "f32[8, 156, 14, 14]" = split_with_sizes_48[3];  split_with_sizes_48 = None
    convolution_69: "f32[8, 156, 14, 14]" = torch.ops.aten.convolution.default(getitem_191, primals_202, None, [1, 1], [4, 4], [1, 1], False, [0, 0], 156)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_20: "f32[8, 624, 14, 14]" = torch.ops.aten.cat.default([convolution_66, convolution_67, convolution_68, convolution_69], 1);  convolution_66 = convolution_67 = convolution_68 = convolution_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_146: "i64[]" = torch.ops.aten.add.Tensor(primals_390, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_28 = torch.ops.aten.var_mean.correction(cat_20, [0, 2, 3], correction = 0, keepdim = True)
    getitem_192: "f32[1, 624, 1, 1]" = var_mean_28[0]
    getitem_193: "f32[1, 624, 1, 1]" = var_mean_28[1];  var_mean_28 = None
    add_147: "f32[1, 624, 1, 1]" = torch.ops.aten.add.Tensor(getitem_192, 0.001)
    rsqrt_28: "f32[1, 624, 1, 1]" = torch.ops.aten.rsqrt.default(add_147);  add_147 = None
    sub_28: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(cat_20, getitem_193)
    mul_221: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_28);  sub_28 = None
    squeeze_84: "f32[624]" = torch.ops.aten.squeeze.dims(getitem_193, [0, 2, 3]);  getitem_193 = None
    squeeze_85: "f32[624]" = torch.ops.aten.squeeze.dims(rsqrt_28, [0, 2, 3]);  rsqrt_28 = None
    mul_222: "f32[624]" = torch.ops.aten.mul.Tensor(squeeze_84, 0.1)
    mul_223: "f32[624]" = torch.ops.aten.mul.Tensor(primals_391, 0.9)
    add_148: "f32[624]" = torch.ops.aten.add.Tensor(mul_222, mul_223);  mul_222 = mul_223 = None
    squeeze_86: "f32[624]" = torch.ops.aten.squeeze.dims(getitem_192, [0, 2, 3]);  getitem_192 = None
    mul_224: "f32[624]" = torch.ops.aten.mul.Tensor(squeeze_86, 1.0006381620931717);  squeeze_86 = None
    mul_225: "f32[624]" = torch.ops.aten.mul.Tensor(mul_224, 0.1);  mul_224 = None
    mul_226: "f32[624]" = torch.ops.aten.mul.Tensor(primals_392, 0.9)
    add_149: "f32[624]" = torch.ops.aten.add.Tensor(mul_225, mul_226);  mul_225 = mul_226 = None
    unsqueeze_112: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(primals_68, -1)
    unsqueeze_113: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
    mul_227: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(mul_221, unsqueeze_113);  mul_221 = unsqueeze_113 = None
    unsqueeze_114: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(primals_69, -1);  primals_69 = None
    unsqueeze_115: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
    add_150: "f32[8, 624, 14, 14]" = torch.ops.aten.add.Tensor(mul_227, unsqueeze_115);  mul_227 = unsqueeze_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_19: "f32[8, 624, 14, 14]" = torch.ops.aten.clone.default(add_150)
    sigmoid_25: "f32[8, 624, 14, 14]" = torch.ops.aten.sigmoid.default(add_150)
    mul_228: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(add_150, sigmoid_25);  add_150 = sigmoid_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_6: "f32[8, 624, 1, 1]" = torch.ops.aten.mean.dim(mul_228, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_70: "f32[8, 26, 1, 1]" = torch.ops.aten.convolution.default(mean_6, primals_203, primals_204, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_20: "f32[8, 26, 1, 1]" = torch.ops.aten.clone.default(convolution_70)
    sigmoid_26: "f32[8, 26, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_70)
    mul_229: "f32[8, 26, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_70, sigmoid_26);  convolution_70 = sigmoid_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_71: "f32[8, 624, 1, 1]" = torch.ops.aten.convolution.default(mul_229, primals_205, primals_206, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_27: "f32[8, 624, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_71);  convolution_71 = None
    alias_12: "f32[8, 624, 1, 1]" = torch.ops.aten.alias.default(sigmoid_27)
    mul_230: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(mul_228, sigmoid_27)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_with_sizes_49 = torch.ops.aten.split_with_sizes.default(mul_230, [312, 312], 1);  mul_230 = None
    getitem_194: "f32[8, 312, 14, 14]" = split_with_sizes_49[0]
    getitem_195: "f32[8, 312, 14, 14]" = split_with_sizes_49[1];  split_with_sizes_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_72: "f32[8, 52, 14, 14]" = torch.ops.aten.convolution.default(getitem_194, primals_207, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convolution_73: "f32[8, 52, 14, 14]" = torch.ops.aten.convolution.default(getitem_195, primals_208, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_21: "f32[8, 104, 14, 14]" = torch.ops.aten.cat.default([convolution_72, convolution_73], 1);  convolution_72 = convolution_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_151: "i64[]" = torch.ops.aten.add.Tensor(primals_393, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_29 = torch.ops.aten.var_mean.correction(cat_21, [0, 2, 3], correction = 0, keepdim = True)
    getitem_196: "f32[1, 104, 1, 1]" = var_mean_29[0]
    getitem_197: "f32[1, 104, 1, 1]" = var_mean_29[1];  var_mean_29 = None
    add_152: "f32[1, 104, 1, 1]" = torch.ops.aten.add.Tensor(getitem_196, 0.001)
    rsqrt_29: "f32[1, 104, 1, 1]" = torch.ops.aten.rsqrt.default(add_152);  add_152 = None
    sub_29: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(cat_21, getitem_197)
    mul_231: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_29);  sub_29 = None
    squeeze_87: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_197, [0, 2, 3]);  getitem_197 = None
    squeeze_88: "f32[104]" = torch.ops.aten.squeeze.dims(rsqrt_29, [0, 2, 3]);  rsqrt_29 = None
    mul_232: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_87, 0.1)
    mul_233: "f32[104]" = torch.ops.aten.mul.Tensor(primals_394, 0.9)
    add_153: "f32[104]" = torch.ops.aten.add.Tensor(mul_232, mul_233);  mul_232 = mul_233 = None
    squeeze_89: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_196, [0, 2, 3]);  getitem_196 = None
    mul_234: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_89, 1.0006381620931717);  squeeze_89 = None
    mul_235: "f32[104]" = torch.ops.aten.mul.Tensor(mul_234, 0.1);  mul_234 = None
    mul_236: "f32[104]" = torch.ops.aten.mul.Tensor(primals_395, 0.9)
    add_154: "f32[104]" = torch.ops.aten.add.Tensor(mul_235, mul_236);  mul_235 = mul_236 = None
    unsqueeze_116: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_70, -1)
    unsqueeze_117: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
    mul_237: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_231, unsqueeze_117);  mul_231 = unsqueeze_117 = None
    unsqueeze_118: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_71, -1);  primals_71 = None
    unsqueeze_119: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
    add_155: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_237, unsqueeze_119);  mul_237 = unsqueeze_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_156: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(add_155, add_140);  add_155 = add_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_with_sizes_50 = torch.ops.aten.split_with_sizes.default(add_156, [52, 52], 1)
    getitem_198: "f32[8, 52, 14, 14]" = split_with_sizes_50[0]
    getitem_199: "f32[8, 52, 14, 14]" = split_with_sizes_50[1];  split_with_sizes_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_74: "f32[8, 312, 14, 14]" = torch.ops.aten.convolution.default(getitem_198, primals_209, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convolution_75: "f32[8, 312, 14, 14]" = torch.ops.aten.convolution.default(getitem_199, primals_210, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_22: "f32[8, 624, 14, 14]" = torch.ops.aten.cat.default([convolution_74, convolution_75], 1);  convolution_74 = convolution_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_157: "i64[]" = torch.ops.aten.add.Tensor(primals_396, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_30 = torch.ops.aten.var_mean.correction(cat_22, [0, 2, 3], correction = 0, keepdim = True)
    getitem_200: "f32[1, 624, 1, 1]" = var_mean_30[0]
    getitem_201: "f32[1, 624, 1, 1]" = var_mean_30[1];  var_mean_30 = None
    add_158: "f32[1, 624, 1, 1]" = torch.ops.aten.add.Tensor(getitem_200, 0.001)
    rsqrt_30: "f32[1, 624, 1, 1]" = torch.ops.aten.rsqrt.default(add_158);  add_158 = None
    sub_30: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(cat_22, getitem_201)
    mul_238: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_30);  sub_30 = None
    squeeze_90: "f32[624]" = torch.ops.aten.squeeze.dims(getitem_201, [0, 2, 3]);  getitem_201 = None
    squeeze_91: "f32[624]" = torch.ops.aten.squeeze.dims(rsqrt_30, [0, 2, 3]);  rsqrt_30 = None
    mul_239: "f32[624]" = torch.ops.aten.mul.Tensor(squeeze_90, 0.1)
    mul_240: "f32[624]" = torch.ops.aten.mul.Tensor(primals_397, 0.9)
    add_159: "f32[624]" = torch.ops.aten.add.Tensor(mul_239, mul_240);  mul_239 = mul_240 = None
    squeeze_92: "f32[624]" = torch.ops.aten.squeeze.dims(getitem_200, [0, 2, 3]);  getitem_200 = None
    mul_241: "f32[624]" = torch.ops.aten.mul.Tensor(squeeze_92, 1.0006381620931717);  squeeze_92 = None
    mul_242: "f32[624]" = torch.ops.aten.mul.Tensor(mul_241, 0.1);  mul_241 = None
    mul_243: "f32[624]" = torch.ops.aten.mul.Tensor(primals_398, 0.9)
    add_160: "f32[624]" = torch.ops.aten.add.Tensor(mul_242, mul_243);  mul_242 = mul_243 = None
    unsqueeze_120: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(primals_72, -1)
    unsqueeze_121: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
    mul_244: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(mul_238, unsqueeze_121);  mul_238 = unsqueeze_121 = None
    unsqueeze_122: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(primals_73, -1);  primals_73 = None
    unsqueeze_123: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
    add_161: "f32[8, 624, 14, 14]" = torch.ops.aten.add.Tensor(mul_244, unsqueeze_123);  mul_244 = unsqueeze_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_21: "f32[8, 624, 14, 14]" = torch.ops.aten.clone.default(add_161)
    sigmoid_28: "f32[8, 624, 14, 14]" = torch.ops.aten.sigmoid.default(add_161)
    mul_245: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(add_161, sigmoid_28);  add_161 = sigmoid_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    split_with_sizes_52 = torch.ops.aten.split_with_sizes.default(mul_245, [156, 156, 156, 156], 1)
    getitem_206: "f32[8, 156, 14, 14]" = split_with_sizes_52[0];  split_with_sizes_52 = None
    convolution_76: "f32[8, 156, 14, 14]" = torch.ops.aten.convolution.default(getitem_206, primals_211, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 156)
    split_with_sizes_53 = torch.ops.aten.split_with_sizes.default(mul_245, [156, 156, 156, 156], 1)
    getitem_211: "f32[8, 156, 14, 14]" = split_with_sizes_53[1];  split_with_sizes_53 = None
    convolution_77: "f32[8, 156, 14, 14]" = torch.ops.aten.convolution.default(getitem_211, primals_212, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 156)
    split_with_sizes_54 = torch.ops.aten.split_with_sizes.default(mul_245, [156, 156, 156, 156], 1)
    getitem_216: "f32[8, 156, 14, 14]" = split_with_sizes_54[2];  split_with_sizes_54 = None
    convolution_78: "f32[8, 156, 14, 14]" = torch.ops.aten.convolution.default(getitem_216, primals_213, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 156)
    split_with_sizes_55 = torch.ops.aten.split_with_sizes.default(mul_245, [156, 156, 156, 156], 1);  mul_245 = None
    getitem_221: "f32[8, 156, 14, 14]" = split_with_sizes_55[3];  split_with_sizes_55 = None
    convolution_79: "f32[8, 156, 14, 14]" = torch.ops.aten.convolution.default(getitem_221, primals_214, None, [1, 1], [4, 4], [1, 1], False, [0, 0], 156)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_23: "f32[8, 624, 14, 14]" = torch.ops.aten.cat.default([convolution_76, convolution_77, convolution_78, convolution_79], 1);  convolution_76 = convolution_77 = convolution_78 = convolution_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_162: "i64[]" = torch.ops.aten.add.Tensor(primals_399, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_31 = torch.ops.aten.var_mean.correction(cat_23, [0, 2, 3], correction = 0, keepdim = True)
    getitem_222: "f32[1, 624, 1, 1]" = var_mean_31[0]
    getitem_223: "f32[1, 624, 1, 1]" = var_mean_31[1];  var_mean_31 = None
    add_163: "f32[1, 624, 1, 1]" = torch.ops.aten.add.Tensor(getitem_222, 0.001)
    rsqrt_31: "f32[1, 624, 1, 1]" = torch.ops.aten.rsqrt.default(add_163);  add_163 = None
    sub_31: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(cat_23, getitem_223)
    mul_246: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_31);  sub_31 = None
    squeeze_93: "f32[624]" = torch.ops.aten.squeeze.dims(getitem_223, [0, 2, 3]);  getitem_223 = None
    squeeze_94: "f32[624]" = torch.ops.aten.squeeze.dims(rsqrt_31, [0, 2, 3]);  rsqrt_31 = None
    mul_247: "f32[624]" = torch.ops.aten.mul.Tensor(squeeze_93, 0.1)
    mul_248: "f32[624]" = torch.ops.aten.mul.Tensor(primals_400, 0.9)
    add_164: "f32[624]" = torch.ops.aten.add.Tensor(mul_247, mul_248);  mul_247 = mul_248 = None
    squeeze_95: "f32[624]" = torch.ops.aten.squeeze.dims(getitem_222, [0, 2, 3]);  getitem_222 = None
    mul_249: "f32[624]" = torch.ops.aten.mul.Tensor(squeeze_95, 1.0006381620931717);  squeeze_95 = None
    mul_250: "f32[624]" = torch.ops.aten.mul.Tensor(mul_249, 0.1);  mul_249 = None
    mul_251: "f32[624]" = torch.ops.aten.mul.Tensor(primals_401, 0.9)
    add_165: "f32[624]" = torch.ops.aten.add.Tensor(mul_250, mul_251);  mul_250 = mul_251 = None
    unsqueeze_124: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(primals_74, -1)
    unsqueeze_125: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
    mul_252: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(mul_246, unsqueeze_125);  mul_246 = unsqueeze_125 = None
    unsqueeze_126: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(primals_75, -1);  primals_75 = None
    unsqueeze_127: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
    add_166: "f32[8, 624, 14, 14]" = torch.ops.aten.add.Tensor(mul_252, unsqueeze_127);  mul_252 = unsqueeze_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_22: "f32[8, 624, 14, 14]" = torch.ops.aten.clone.default(add_166)
    sigmoid_29: "f32[8, 624, 14, 14]" = torch.ops.aten.sigmoid.default(add_166)
    mul_253: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(add_166, sigmoid_29);  add_166 = sigmoid_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_7: "f32[8, 624, 1, 1]" = torch.ops.aten.mean.dim(mul_253, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_80: "f32[8, 26, 1, 1]" = torch.ops.aten.convolution.default(mean_7, primals_215, primals_216, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_23: "f32[8, 26, 1, 1]" = torch.ops.aten.clone.default(convolution_80)
    sigmoid_30: "f32[8, 26, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_80)
    mul_254: "f32[8, 26, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_80, sigmoid_30);  convolution_80 = sigmoid_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_81: "f32[8, 624, 1, 1]" = torch.ops.aten.convolution.default(mul_254, primals_217, primals_218, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_31: "f32[8, 624, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_81);  convolution_81 = None
    alias_13: "f32[8, 624, 1, 1]" = torch.ops.aten.alias.default(sigmoid_31)
    mul_255: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(mul_253, sigmoid_31)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_with_sizes_56 = torch.ops.aten.split_with_sizes.default(mul_255, [312, 312], 1);  mul_255 = None
    getitem_224: "f32[8, 312, 14, 14]" = split_with_sizes_56[0]
    getitem_225: "f32[8, 312, 14, 14]" = split_with_sizes_56[1];  split_with_sizes_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_82: "f32[8, 52, 14, 14]" = torch.ops.aten.convolution.default(getitem_224, primals_219, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convolution_83: "f32[8, 52, 14, 14]" = torch.ops.aten.convolution.default(getitem_225, primals_220, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_24: "f32[8, 104, 14, 14]" = torch.ops.aten.cat.default([convolution_82, convolution_83], 1);  convolution_82 = convolution_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_167: "i64[]" = torch.ops.aten.add.Tensor(primals_402, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_32 = torch.ops.aten.var_mean.correction(cat_24, [0, 2, 3], correction = 0, keepdim = True)
    getitem_226: "f32[1, 104, 1, 1]" = var_mean_32[0]
    getitem_227: "f32[1, 104, 1, 1]" = var_mean_32[1];  var_mean_32 = None
    add_168: "f32[1, 104, 1, 1]" = torch.ops.aten.add.Tensor(getitem_226, 0.001)
    rsqrt_32: "f32[1, 104, 1, 1]" = torch.ops.aten.rsqrt.default(add_168);  add_168 = None
    sub_32: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(cat_24, getitem_227)
    mul_256: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_32);  sub_32 = None
    squeeze_96: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_227, [0, 2, 3]);  getitem_227 = None
    squeeze_97: "f32[104]" = torch.ops.aten.squeeze.dims(rsqrt_32, [0, 2, 3]);  rsqrt_32 = None
    mul_257: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_96, 0.1)
    mul_258: "f32[104]" = torch.ops.aten.mul.Tensor(primals_403, 0.9)
    add_169: "f32[104]" = torch.ops.aten.add.Tensor(mul_257, mul_258);  mul_257 = mul_258 = None
    squeeze_98: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_226, [0, 2, 3]);  getitem_226 = None
    mul_259: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_98, 1.0006381620931717);  squeeze_98 = None
    mul_260: "f32[104]" = torch.ops.aten.mul.Tensor(mul_259, 0.1);  mul_259 = None
    mul_261: "f32[104]" = torch.ops.aten.mul.Tensor(primals_404, 0.9)
    add_170: "f32[104]" = torch.ops.aten.add.Tensor(mul_260, mul_261);  mul_260 = mul_261 = None
    unsqueeze_128: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_76, -1)
    unsqueeze_129: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_128, -1);  unsqueeze_128 = None
    mul_262: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_256, unsqueeze_129);  mul_256 = unsqueeze_129 = None
    unsqueeze_130: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_77, -1);  primals_77 = None
    unsqueeze_131: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_130, -1);  unsqueeze_130 = None
    add_171: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_262, unsqueeze_131);  mul_262 = unsqueeze_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_172: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(add_171, add_156);  add_171 = add_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_84: "f32[8, 624, 14, 14]" = torch.ops.aten.convolution.default(add_172, primals_221, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_173: "i64[]" = torch.ops.aten.add.Tensor(primals_405, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_33 = torch.ops.aten.var_mean.correction(convolution_84, [0, 2, 3], correction = 0, keepdim = True)
    getitem_228: "f32[1, 624, 1, 1]" = var_mean_33[0]
    getitem_229: "f32[1, 624, 1, 1]" = var_mean_33[1];  var_mean_33 = None
    add_174: "f32[1, 624, 1, 1]" = torch.ops.aten.add.Tensor(getitem_228, 0.001)
    rsqrt_33: "f32[1, 624, 1, 1]" = torch.ops.aten.rsqrt.default(add_174);  add_174 = None
    sub_33: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_84, getitem_229)
    mul_263: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_33);  sub_33 = None
    squeeze_99: "f32[624]" = torch.ops.aten.squeeze.dims(getitem_229, [0, 2, 3]);  getitem_229 = None
    squeeze_100: "f32[624]" = torch.ops.aten.squeeze.dims(rsqrt_33, [0, 2, 3]);  rsqrt_33 = None
    mul_264: "f32[624]" = torch.ops.aten.mul.Tensor(squeeze_99, 0.1)
    mul_265: "f32[624]" = torch.ops.aten.mul.Tensor(primals_406, 0.9)
    add_175: "f32[624]" = torch.ops.aten.add.Tensor(mul_264, mul_265);  mul_264 = mul_265 = None
    squeeze_101: "f32[624]" = torch.ops.aten.squeeze.dims(getitem_228, [0, 2, 3]);  getitem_228 = None
    mul_266: "f32[624]" = torch.ops.aten.mul.Tensor(squeeze_101, 1.0006381620931717);  squeeze_101 = None
    mul_267: "f32[624]" = torch.ops.aten.mul.Tensor(mul_266, 0.1);  mul_266 = None
    mul_268: "f32[624]" = torch.ops.aten.mul.Tensor(primals_407, 0.9)
    add_176: "f32[624]" = torch.ops.aten.add.Tensor(mul_267, mul_268);  mul_267 = mul_268 = None
    unsqueeze_132: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(primals_78, -1)
    unsqueeze_133: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_132, -1);  unsqueeze_132 = None
    mul_269: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(mul_263, unsqueeze_133);  mul_263 = unsqueeze_133 = None
    unsqueeze_134: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(primals_79, -1);  primals_79 = None
    unsqueeze_135: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_134, -1);  unsqueeze_134 = None
    add_177: "f32[8, 624, 14, 14]" = torch.ops.aten.add.Tensor(mul_269, unsqueeze_135);  mul_269 = unsqueeze_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_24: "f32[8, 624, 14, 14]" = torch.ops.aten.clone.default(add_177)
    sigmoid_32: "f32[8, 624, 14, 14]" = torch.ops.aten.sigmoid.default(add_177)
    mul_270: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(add_177, sigmoid_32);  add_177 = sigmoid_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_85: "f32[8, 624, 14, 14]" = torch.ops.aten.convolution.default(mul_270, primals_222, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 624)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_178: "i64[]" = torch.ops.aten.add.Tensor(primals_408, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_34 = torch.ops.aten.var_mean.correction(convolution_85, [0, 2, 3], correction = 0, keepdim = True)
    getitem_230: "f32[1, 624, 1, 1]" = var_mean_34[0]
    getitem_231: "f32[1, 624, 1, 1]" = var_mean_34[1];  var_mean_34 = None
    add_179: "f32[1, 624, 1, 1]" = torch.ops.aten.add.Tensor(getitem_230, 0.001)
    rsqrt_34: "f32[1, 624, 1, 1]" = torch.ops.aten.rsqrt.default(add_179);  add_179 = None
    sub_34: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_85, getitem_231)
    mul_271: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_34);  sub_34 = None
    squeeze_102: "f32[624]" = torch.ops.aten.squeeze.dims(getitem_231, [0, 2, 3]);  getitem_231 = None
    squeeze_103: "f32[624]" = torch.ops.aten.squeeze.dims(rsqrt_34, [0, 2, 3]);  rsqrt_34 = None
    mul_272: "f32[624]" = torch.ops.aten.mul.Tensor(squeeze_102, 0.1)
    mul_273: "f32[624]" = torch.ops.aten.mul.Tensor(primals_409, 0.9)
    add_180: "f32[624]" = torch.ops.aten.add.Tensor(mul_272, mul_273);  mul_272 = mul_273 = None
    squeeze_104: "f32[624]" = torch.ops.aten.squeeze.dims(getitem_230, [0, 2, 3]);  getitem_230 = None
    mul_274: "f32[624]" = torch.ops.aten.mul.Tensor(squeeze_104, 1.0006381620931717);  squeeze_104 = None
    mul_275: "f32[624]" = torch.ops.aten.mul.Tensor(mul_274, 0.1);  mul_274 = None
    mul_276: "f32[624]" = torch.ops.aten.mul.Tensor(primals_410, 0.9)
    add_181: "f32[624]" = torch.ops.aten.add.Tensor(mul_275, mul_276);  mul_275 = mul_276 = None
    unsqueeze_136: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(primals_80, -1)
    unsqueeze_137: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_136, -1);  unsqueeze_136 = None
    mul_277: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(mul_271, unsqueeze_137);  mul_271 = unsqueeze_137 = None
    unsqueeze_138: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(primals_81, -1);  primals_81 = None
    unsqueeze_139: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_138, -1);  unsqueeze_138 = None
    add_182: "f32[8, 624, 14, 14]" = torch.ops.aten.add.Tensor(mul_277, unsqueeze_139);  mul_277 = unsqueeze_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_25: "f32[8, 624, 14, 14]" = torch.ops.aten.clone.default(add_182)
    sigmoid_33: "f32[8, 624, 14, 14]" = torch.ops.aten.sigmoid.default(add_182)
    mul_278: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(add_182, sigmoid_33);  add_182 = sigmoid_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_8: "f32[8, 624, 1, 1]" = torch.ops.aten.mean.dim(mul_278, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_86: "f32[8, 52, 1, 1]" = torch.ops.aten.convolution.default(mean_8, primals_223, primals_224, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_26: "f32[8, 52, 1, 1]" = torch.ops.aten.clone.default(convolution_86)
    sigmoid_34: "f32[8, 52, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_86)
    mul_279: "f32[8, 52, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_86, sigmoid_34);  convolution_86 = sigmoid_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_87: "f32[8, 624, 1, 1]" = torch.ops.aten.convolution.default(mul_279, primals_225, primals_226, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_35: "f32[8, 624, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_87);  convolution_87 = None
    alias_14: "f32[8, 624, 1, 1]" = torch.ops.aten.alias.default(sigmoid_35)
    mul_280: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(mul_278, sigmoid_35)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_88: "f32[8, 160, 14, 14]" = torch.ops.aten.convolution.default(mul_280, primals_227, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_183: "i64[]" = torch.ops.aten.add.Tensor(primals_411, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_35 = torch.ops.aten.var_mean.correction(convolution_88, [0, 2, 3], correction = 0, keepdim = True)
    getitem_232: "f32[1, 160, 1, 1]" = var_mean_35[0]
    getitem_233: "f32[1, 160, 1, 1]" = var_mean_35[1];  var_mean_35 = None
    add_184: "f32[1, 160, 1, 1]" = torch.ops.aten.add.Tensor(getitem_232, 0.001)
    rsqrt_35: "f32[1, 160, 1, 1]" = torch.ops.aten.rsqrt.default(add_184);  add_184 = None
    sub_35: "f32[8, 160, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_88, getitem_233)
    mul_281: "f32[8, 160, 14, 14]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_35);  sub_35 = None
    squeeze_105: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_233, [0, 2, 3]);  getitem_233 = None
    squeeze_106: "f32[160]" = torch.ops.aten.squeeze.dims(rsqrt_35, [0, 2, 3]);  rsqrt_35 = None
    mul_282: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_105, 0.1)
    mul_283: "f32[160]" = torch.ops.aten.mul.Tensor(primals_412, 0.9)
    add_185: "f32[160]" = torch.ops.aten.add.Tensor(mul_282, mul_283);  mul_282 = mul_283 = None
    squeeze_107: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_232, [0, 2, 3]);  getitem_232 = None
    mul_284: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_107, 1.0006381620931717);  squeeze_107 = None
    mul_285: "f32[160]" = torch.ops.aten.mul.Tensor(mul_284, 0.1);  mul_284 = None
    mul_286: "f32[160]" = torch.ops.aten.mul.Tensor(primals_413, 0.9)
    add_186: "f32[160]" = torch.ops.aten.add.Tensor(mul_285, mul_286);  mul_285 = mul_286 = None
    unsqueeze_140: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_82, -1)
    unsqueeze_141: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_140, -1);  unsqueeze_140 = None
    mul_287: "f32[8, 160, 14, 14]" = torch.ops.aten.mul.Tensor(mul_281, unsqueeze_141);  mul_281 = unsqueeze_141 = None
    unsqueeze_142: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_83, -1);  primals_83 = None
    unsqueeze_143: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_142, -1);  unsqueeze_142 = None
    add_187: "f32[8, 160, 14, 14]" = torch.ops.aten.add.Tensor(mul_287, unsqueeze_143);  mul_287 = unsqueeze_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_with_sizes_57 = torch.ops.aten.split_with_sizes.default(add_187, [80, 80], 1)
    getitem_234: "f32[8, 80, 14, 14]" = split_with_sizes_57[0]
    getitem_235: "f32[8, 80, 14, 14]" = split_with_sizes_57[1];  split_with_sizes_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_89: "f32[8, 240, 14, 14]" = torch.ops.aten.convolution.default(getitem_234, primals_228, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convolution_90: "f32[8, 240, 14, 14]" = torch.ops.aten.convolution.default(getitem_235, primals_229, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_25: "f32[8, 480, 14, 14]" = torch.ops.aten.cat.default([convolution_89, convolution_90], 1);  convolution_89 = convolution_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_188: "i64[]" = torch.ops.aten.add.Tensor(primals_414, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_36 = torch.ops.aten.var_mean.correction(cat_25, [0, 2, 3], correction = 0, keepdim = True)
    getitem_236: "f32[1, 480, 1, 1]" = var_mean_36[0]
    getitem_237: "f32[1, 480, 1, 1]" = var_mean_36[1];  var_mean_36 = None
    add_189: "f32[1, 480, 1, 1]" = torch.ops.aten.add.Tensor(getitem_236, 0.001)
    rsqrt_36: "f32[1, 480, 1, 1]" = torch.ops.aten.rsqrt.default(add_189);  add_189 = None
    sub_36: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(cat_25, getitem_237)
    mul_288: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_36);  sub_36 = None
    squeeze_108: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_237, [0, 2, 3]);  getitem_237 = None
    squeeze_109: "f32[480]" = torch.ops.aten.squeeze.dims(rsqrt_36, [0, 2, 3]);  rsqrt_36 = None
    mul_289: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_108, 0.1)
    mul_290: "f32[480]" = torch.ops.aten.mul.Tensor(primals_415, 0.9)
    add_190: "f32[480]" = torch.ops.aten.add.Tensor(mul_289, mul_290);  mul_289 = mul_290 = None
    squeeze_110: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_236, [0, 2, 3]);  getitem_236 = None
    mul_291: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_110, 1.0006381620931717);  squeeze_110 = None
    mul_292: "f32[480]" = torch.ops.aten.mul.Tensor(mul_291, 0.1);  mul_291 = None
    mul_293: "f32[480]" = torch.ops.aten.mul.Tensor(primals_416, 0.9)
    add_191: "f32[480]" = torch.ops.aten.add.Tensor(mul_292, mul_293);  mul_292 = mul_293 = None
    unsqueeze_144: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_84, -1)
    unsqueeze_145: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_144, -1);  unsqueeze_144 = None
    mul_294: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_288, unsqueeze_145);  mul_288 = unsqueeze_145 = None
    unsqueeze_146: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_85, -1);  primals_85 = None
    unsqueeze_147: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_146, -1);  unsqueeze_146 = None
    add_192: "f32[8, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_294, unsqueeze_147);  mul_294 = unsqueeze_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_27: "f32[8, 480, 14, 14]" = torch.ops.aten.clone.default(add_192)
    sigmoid_36: "f32[8, 480, 14, 14]" = torch.ops.aten.sigmoid.default(add_192)
    mul_295: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_192, sigmoid_36);  add_192 = sigmoid_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    split_with_sizes_59 = torch.ops.aten.split_with_sizes.default(mul_295, [120, 120, 120, 120], 1)
    getitem_242: "f32[8, 120, 14, 14]" = split_with_sizes_59[0];  split_with_sizes_59 = None
    convolution_91: "f32[8, 120, 14, 14]" = torch.ops.aten.convolution.default(getitem_242, primals_230, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 120)
    split_with_sizes_60 = torch.ops.aten.split_with_sizes.default(mul_295, [120, 120, 120, 120], 1)
    getitem_247: "f32[8, 120, 14, 14]" = split_with_sizes_60[1];  split_with_sizes_60 = None
    convolution_92: "f32[8, 120, 14, 14]" = torch.ops.aten.convolution.default(getitem_247, primals_231, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 120)
    split_with_sizes_61 = torch.ops.aten.split_with_sizes.default(mul_295, [120, 120, 120, 120], 1)
    getitem_252: "f32[8, 120, 14, 14]" = split_with_sizes_61[2];  split_with_sizes_61 = None
    convolution_93: "f32[8, 120, 14, 14]" = torch.ops.aten.convolution.default(getitem_252, primals_232, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 120)
    split_with_sizes_62 = torch.ops.aten.split_with_sizes.default(mul_295, [120, 120, 120, 120], 1);  mul_295 = None
    getitem_257: "f32[8, 120, 14, 14]" = split_with_sizes_62[3];  split_with_sizes_62 = None
    convolution_94: "f32[8, 120, 14, 14]" = torch.ops.aten.convolution.default(getitem_257, primals_233, None, [1, 1], [4, 4], [1, 1], False, [0, 0], 120)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_26: "f32[8, 480, 14, 14]" = torch.ops.aten.cat.default([convolution_91, convolution_92, convolution_93, convolution_94], 1);  convolution_91 = convolution_92 = convolution_93 = convolution_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_193: "i64[]" = torch.ops.aten.add.Tensor(primals_417, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_37 = torch.ops.aten.var_mean.correction(cat_26, [0, 2, 3], correction = 0, keepdim = True)
    getitem_258: "f32[1, 480, 1, 1]" = var_mean_37[0]
    getitem_259: "f32[1, 480, 1, 1]" = var_mean_37[1];  var_mean_37 = None
    add_194: "f32[1, 480, 1, 1]" = torch.ops.aten.add.Tensor(getitem_258, 0.001)
    rsqrt_37: "f32[1, 480, 1, 1]" = torch.ops.aten.rsqrt.default(add_194);  add_194 = None
    sub_37: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(cat_26, getitem_259)
    mul_296: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_37);  sub_37 = None
    squeeze_111: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_259, [0, 2, 3]);  getitem_259 = None
    squeeze_112: "f32[480]" = torch.ops.aten.squeeze.dims(rsqrt_37, [0, 2, 3]);  rsqrt_37 = None
    mul_297: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_111, 0.1)
    mul_298: "f32[480]" = torch.ops.aten.mul.Tensor(primals_418, 0.9)
    add_195: "f32[480]" = torch.ops.aten.add.Tensor(mul_297, mul_298);  mul_297 = mul_298 = None
    squeeze_113: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_258, [0, 2, 3]);  getitem_258 = None
    mul_299: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_113, 1.0006381620931717);  squeeze_113 = None
    mul_300: "f32[480]" = torch.ops.aten.mul.Tensor(mul_299, 0.1);  mul_299 = None
    mul_301: "f32[480]" = torch.ops.aten.mul.Tensor(primals_419, 0.9)
    add_196: "f32[480]" = torch.ops.aten.add.Tensor(mul_300, mul_301);  mul_300 = mul_301 = None
    unsqueeze_148: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_86, -1)
    unsqueeze_149: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_148, -1);  unsqueeze_148 = None
    mul_302: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_296, unsqueeze_149);  mul_296 = unsqueeze_149 = None
    unsqueeze_150: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_87, -1);  primals_87 = None
    unsqueeze_151: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_150, -1);  unsqueeze_150 = None
    add_197: "f32[8, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_302, unsqueeze_151);  mul_302 = unsqueeze_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_28: "f32[8, 480, 14, 14]" = torch.ops.aten.clone.default(add_197)
    sigmoid_37: "f32[8, 480, 14, 14]" = torch.ops.aten.sigmoid.default(add_197)
    mul_303: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_197, sigmoid_37);  add_197 = sigmoid_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_9: "f32[8, 480, 1, 1]" = torch.ops.aten.mean.dim(mul_303, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_95: "f32[8, 80, 1, 1]" = torch.ops.aten.convolution.default(mean_9, primals_234, primals_235, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_29: "f32[8, 80, 1, 1]" = torch.ops.aten.clone.default(convolution_95)
    sigmoid_38: "f32[8, 80, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_95)
    mul_304: "f32[8, 80, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_95, sigmoid_38);  convolution_95 = sigmoid_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_96: "f32[8, 480, 1, 1]" = torch.ops.aten.convolution.default(mul_304, primals_236, primals_237, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_39: "f32[8, 480, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_96);  convolution_96 = None
    alias_15: "f32[8, 480, 1, 1]" = torch.ops.aten.alias.default(sigmoid_39)
    mul_305: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_303, sigmoid_39)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_with_sizes_63 = torch.ops.aten.split_with_sizes.default(mul_305, [240, 240], 1);  mul_305 = None
    getitem_260: "f32[8, 240, 14, 14]" = split_with_sizes_63[0]
    getitem_261: "f32[8, 240, 14, 14]" = split_with_sizes_63[1];  split_with_sizes_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_97: "f32[8, 80, 14, 14]" = torch.ops.aten.convolution.default(getitem_260, primals_238, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convolution_98: "f32[8, 80, 14, 14]" = torch.ops.aten.convolution.default(getitem_261, primals_239, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_27: "f32[8, 160, 14, 14]" = torch.ops.aten.cat.default([convolution_97, convolution_98], 1);  convolution_97 = convolution_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_198: "i64[]" = torch.ops.aten.add.Tensor(primals_420, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_38 = torch.ops.aten.var_mean.correction(cat_27, [0, 2, 3], correction = 0, keepdim = True)
    getitem_262: "f32[1, 160, 1, 1]" = var_mean_38[0]
    getitem_263: "f32[1, 160, 1, 1]" = var_mean_38[1];  var_mean_38 = None
    add_199: "f32[1, 160, 1, 1]" = torch.ops.aten.add.Tensor(getitem_262, 0.001)
    rsqrt_38: "f32[1, 160, 1, 1]" = torch.ops.aten.rsqrt.default(add_199);  add_199 = None
    sub_38: "f32[8, 160, 14, 14]" = torch.ops.aten.sub.Tensor(cat_27, getitem_263)
    mul_306: "f32[8, 160, 14, 14]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_38);  sub_38 = None
    squeeze_114: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_263, [0, 2, 3]);  getitem_263 = None
    squeeze_115: "f32[160]" = torch.ops.aten.squeeze.dims(rsqrt_38, [0, 2, 3]);  rsqrt_38 = None
    mul_307: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_114, 0.1)
    mul_308: "f32[160]" = torch.ops.aten.mul.Tensor(primals_421, 0.9)
    add_200: "f32[160]" = torch.ops.aten.add.Tensor(mul_307, mul_308);  mul_307 = mul_308 = None
    squeeze_116: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_262, [0, 2, 3]);  getitem_262 = None
    mul_309: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_116, 1.0006381620931717);  squeeze_116 = None
    mul_310: "f32[160]" = torch.ops.aten.mul.Tensor(mul_309, 0.1);  mul_309 = None
    mul_311: "f32[160]" = torch.ops.aten.mul.Tensor(primals_422, 0.9)
    add_201: "f32[160]" = torch.ops.aten.add.Tensor(mul_310, mul_311);  mul_310 = mul_311 = None
    unsqueeze_152: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_88, -1)
    unsqueeze_153: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_152, -1);  unsqueeze_152 = None
    mul_312: "f32[8, 160, 14, 14]" = torch.ops.aten.mul.Tensor(mul_306, unsqueeze_153);  mul_306 = unsqueeze_153 = None
    unsqueeze_154: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_89, -1);  primals_89 = None
    unsqueeze_155: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_154, -1);  unsqueeze_154 = None
    add_202: "f32[8, 160, 14, 14]" = torch.ops.aten.add.Tensor(mul_312, unsqueeze_155);  mul_312 = unsqueeze_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_203: "f32[8, 160, 14, 14]" = torch.ops.aten.add.Tensor(add_202, add_187);  add_202 = add_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_with_sizes_64 = torch.ops.aten.split_with_sizes.default(add_203, [80, 80], 1)
    getitem_264: "f32[8, 80, 14, 14]" = split_with_sizes_64[0]
    getitem_265: "f32[8, 80, 14, 14]" = split_with_sizes_64[1];  split_with_sizes_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_99: "f32[8, 240, 14, 14]" = torch.ops.aten.convolution.default(getitem_264, primals_240, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convolution_100: "f32[8, 240, 14, 14]" = torch.ops.aten.convolution.default(getitem_265, primals_241, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_28: "f32[8, 480, 14, 14]" = torch.ops.aten.cat.default([convolution_99, convolution_100], 1);  convolution_99 = convolution_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_204: "i64[]" = torch.ops.aten.add.Tensor(primals_423, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_39 = torch.ops.aten.var_mean.correction(cat_28, [0, 2, 3], correction = 0, keepdim = True)
    getitem_266: "f32[1, 480, 1, 1]" = var_mean_39[0]
    getitem_267: "f32[1, 480, 1, 1]" = var_mean_39[1];  var_mean_39 = None
    add_205: "f32[1, 480, 1, 1]" = torch.ops.aten.add.Tensor(getitem_266, 0.001)
    rsqrt_39: "f32[1, 480, 1, 1]" = torch.ops.aten.rsqrt.default(add_205);  add_205 = None
    sub_39: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(cat_28, getitem_267)
    mul_313: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_39);  sub_39 = None
    squeeze_117: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_267, [0, 2, 3]);  getitem_267 = None
    squeeze_118: "f32[480]" = torch.ops.aten.squeeze.dims(rsqrt_39, [0, 2, 3]);  rsqrt_39 = None
    mul_314: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_117, 0.1)
    mul_315: "f32[480]" = torch.ops.aten.mul.Tensor(primals_424, 0.9)
    add_206: "f32[480]" = torch.ops.aten.add.Tensor(mul_314, mul_315);  mul_314 = mul_315 = None
    squeeze_119: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_266, [0, 2, 3]);  getitem_266 = None
    mul_316: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_119, 1.0006381620931717);  squeeze_119 = None
    mul_317: "f32[480]" = torch.ops.aten.mul.Tensor(mul_316, 0.1);  mul_316 = None
    mul_318: "f32[480]" = torch.ops.aten.mul.Tensor(primals_425, 0.9)
    add_207: "f32[480]" = torch.ops.aten.add.Tensor(mul_317, mul_318);  mul_317 = mul_318 = None
    unsqueeze_156: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_90, -1)
    unsqueeze_157: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_156, -1);  unsqueeze_156 = None
    mul_319: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_313, unsqueeze_157);  mul_313 = unsqueeze_157 = None
    unsqueeze_158: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_91, -1);  primals_91 = None
    unsqueeze_159: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_158, -1);  unsqueeze_158 = None
    add_208: "f32[8, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_319, unsqueeze_159);  mul_319 = unsqueeze_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_30: "f32[8, 480, 14, 14]" = torch.ops.aten.clone.default(add_208)
    sigmoid_40: "f32[8, 480, 14, 14]" = torch.ops.aten.sigmoid.default(add_208)
    mul_320: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_208, sigmoid_40);  add_208 = sigmoid_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    split_with_sizes_66 = torch.ops.aten.split_with_sizes.default(mul_320, [120, 120, 120, 120], 1)
    getitem_272: "f32[8, 120, 14, 14]" = split_with_sizes_66[0];  split_with_sizes_66 = None
    convolution_101: "f32[8, 120, 14, 14]" = torch.ops.aten.convolution.default(getitem_272, primals_242, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 120)
    split_with_sizes_67 = torch.ops.aten.split_with_sizes.default(mul_320, [120, 120, 120, 120], 1)
    getitem_277: "f32[8, 120, 14, 14]" = split_with_sizes_67[1];  split_with_sizes_67 = None
    convolution_102: "f32[8, 120, 14, 14]" = torch.ops.aten.convolution.default(getitem_277, primals_243, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 120)
    split_with_sizes_68 = torch.ops.aten.split_with_sizes.default(mul_320, [120, 120, 120, 120], 1)
    getitem_282: "f32[8, 120, 14, 14]" = split_with_sizes_68[2];  split_with_sizes_68 = None
    convolution_103: "f32[8, 120, 14, 14]" = torch.ops.aten.convolution.default(getitem_282, primals_244, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 120)
    split_with_sizes_69 = torch.ops.aten.split_with_sizes.default(mul_320, [120, 120, 120, 120], 1);  mul_320 = None
    getitem_287: "f32[8, 120, 14, 14]" = split_with_sizes_69[3];  split_with_sizes_69 = None
    convolution_104: "f32[8, 120, 14, 14]" = torch.ops.aten.convolution.default(getitem_287, primals_245, None, [1, 1], [4, 4], [1, 1], False, [0, 0], 120)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_29: "f32[8, 480, 14, 14]" = torch.ops.aten.cat.default([convolution_101, convolution_102, convolution_103, convolution_104], 1);  convolution_101 = convolution_102 = convolution_103 = convolution_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_209: "i64[]" = torch.ops.aten.add.Tensor(primals_426, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_40 = torch.ops.aten.var_mean.correction(cat_29, [0, 2, 3], correction = 0, keepdim = True)
    getitem_288: "f32[1, 480, 1, 1]" = var_mean_40[0]
    getitem_289: "f32[1, 480, 1, 1]" = var_mean_40[1];  var_mean_40 = None
    add_210: "f32[1, 480, 1, 1]" = torch.ops.aten.add.Tensor(getitem_288, 0.001)
    rsqrt_40: "f32[1, 480, 1, 1]" = torch.ops.aten.rsqrt.default(add_210);  add_210 = None
    sub_40: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(cat_29, getitem_289)
    mul_321: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_40);  sub_40 = None
    squeeze_120: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_289, [0, 2, 3]);  getitem_289 = None
    squeeze_121: "f32[480]" = torch.ops.aten.squeeze.dims(rsqrt_40, [0, 2, 3]);  rsqrt_40 = None
    mul_322: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_120, 0.1)
    mul_323: "f32[480]" = torch.ops.aten.mul.Tensor(primals_427, 0.9)
    add_211: "f32[480]" = torch.ops.aten.add.Tensor(mul_322, mul_323);  mul_322 = mul_323 = None
    squeeze_122: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_288, [0, 2, 3]);  getitem_288 = None
    mul_324: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_122, 1.0006381620931717);  squeeze_122 = None
    mul_325: "f32[480]" = torch.ops.aten.mul.Tensor(mul_324, 0.1);  mul_324 = None
    mul_326: "f32[480]" = torch.ops.aten.mul.Tensor(primals_428, 0.9)
    add_212: "f32[480]" = torch.ops.aten.add.Tensor(mul_325, mul_326);  mul_325 = mul_326 = None
    unsqueeze_160: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_92, -1)
    unsqueeze_161: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_160, -1);  unsqueeze_160 = None
    mul_327: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_321, unsqueeze_161);  mul_321 = unsqueeze_161 = None
    unsqueeze_162: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_93, -1);  primals_93 = None
    unsqueeze_163: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_162, -1);  unsqueeze_162 = None
    add_213: "f32[8, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_327, unsqueeze_163);  mul_327 = unsqueeze_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_31: "f32[8, 480, 14, 14]" = torch.ops.aten.clone.default(add_213)
    sigmoid_41: "f32[8, 480, 14, 14]" = torch.ops.aten.sigmoid.default(add_213)
    mul_328: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_213, sigmoid_41);  add_213 = sigmoid_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_10: "f32[8, 480, 1, 1]" = torch.ops.aten.mean.dim(mul_328, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_105: "f32[8, 80, 1, 1]" = torch.ops.aten.convolution.default(mean_10, primals_246, primals_247, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_32: "f32[8, 80, 1, 1]" = torch.ops.aten.clone.default(convolution_105)
    sigmoid_42: "f32[8, 80, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_105)
    mul_329: "f32[8, 80, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_105, sigmoid_42);  convolution_105 = sigmoid_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_106: "f32[8, 480, 1, 1]" = torch.ops.aten.convolution.default(mul_329, primals_248, primals_249, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_43: "f32[8, 480, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_106);  convolution_106 = None
    alias_16: "f32[8, 480, 1, 1]" = torch.ops.aten.alias.default(sigmoid_43)
    mul_330: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_328, sigmoid_43)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_with_sizes_70 = torch.ops.aten.split_with_sizes.default(mul_330, [240, 240], 1);  mul_330 = None
    getitem_290: "f32[8, 240, 14, 14]" = split_with_sizes_70[0]
    getitem_291: "f32[8, 240, 14, 14]" = split_with_sizes_70[1];  split_with_sizes_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_107: "f32[8, 80, 14, 14]" = torch.ops.aten.convolution.default(getitem_290, primals_250, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convolution_108: "f32[8, 80, 14, 14]" = torch.ops.aten.convolution.default(getitem_291, primals_251, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_30: "f32[8, 160, 14, 14]" = torch.ops.aten.cat.default([convolution_107, convolution_108], 1);  convolution_107 = convolution_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_214: "i64[]" = torch.ops.aten.add.Tensor(primals_429, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_41 = torch.ops.aten.var_mean.correction(cat_30, [0, 2, 3], correction = 0, keepdim = True)
    getitem_292: "f32[1, 160, 1, 1]" = var_mean_41[0]
    getitem_293: "f32[1, 160, 1, 1]" = var_mean_41[1];  var_mean_41 = None
    add_215: "f32[1, 160, 1, 1]" = torch.ops.aten.add.Tensor(getitem_292, 0.001)
    rsqrt_41: "f32[1, 160, 1, 1]" = torch.ops.aten.rsqrt.default(add_215);  add_215 = None
    sub_41: "f32[8, 160, 14, 14]" = torch.ops.aten.sub.Tensor(cat_30, getitem_293)
    mul_331: "f32[8, 160, 14, 14]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_41);  sub_41 = None
    squeeze_123: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_293, [0, 2, 3]);  getitem_293 = None
    squeeze_124: "f32[160]" = torch.ops.aten.squeeze.dims(rsqrt_41, [0, 2, 3]);  rsqrt_41 = None
    mul_332: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_123, 0.1)
    mul_333: "f32[160]" = torch.ops.aten.mul.Tensor(primals_430, 0.9)
    add_216: "f32[160]" = torch.ops.aten.add.Tensor(mul_332, mul_333);  mul_332 = mul_333 = None
    squeeze_125: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_292, [0, 2, 3]);  getitem_292 = None
    mul_334: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_125, 1.0006381620931717);  squeeze_125 = None
    mul_335: "f32[160]" = torch.ops.aten.mul.Tensor(mul_334, 0.1);  mul_334 = None
    mul_336: "f32[160]" = torch.ops.aten.mul.Tensor(primals_431, 0.9)
    add_217: "f32[160]" = torch.ops.aten.add.Tensor(mul_335, mul_336);  mul_335 = mul_336 = None
    unsqueeze_164: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_94, -1)
    unsqueeze_165: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_164, -1);  unsqueeze_164 = None
    mul_337: "f32[8, 160, 14, 14]" = torch.ops.aten.mul.Tensor(mul_331, unsqueeze_165);  mul_331 = unsqueeze_165 = None
    unsqueeze_166: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_95, -1);  primals_95 = None
    unsqueeze_167: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_166, -1);  unsqueeze_166 = None
    add_218: "f32[8, 160, 14, 14]" = torch.ops.aten.add.Tensor(mul_337, unsqueeze_167);  mul_337 = unsqueeze_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_219: "f32[8, 160, 14, 14]" = torch.ops.aten.add.Tensor(add_218, add_203);  add_218 = add_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_with_sizes_71 = torch.ops.aten.split_with_sizes.default(add_219, [80, 80], 1)
    getitem_294: "f32[8, 80, 14, 14]" = split_with_sizes_71[0]
    getitem_295: "f32[8, 80, 14, 14]" = split_with_sizes_71[1];  split_with_sizes_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_109: "f32[8, 240, 14, 14]" = torch.ops.aten.convolution.default(getitem_294, primals_252, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convolution_110: "f32[8, 240, 14, 14]" = torch.ops.aten.convolution.default(getitem_295, primals_253, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_31: "f32[8, 480, 14, 14]" = torch.ops.aten.cat.default([convolution_109, convolution_110], 1);  convolution_109 = convolution_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_220: "i64[]" = torch.ops.aten.add.Tensor(primals_432, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_42 = torch.ops.aten.var_mean.correction(cat_31, [0, 2, 3], correction = 0, keepdim = True)
    getitem_296: "f32[1, 480, 1, 1]" = var_mean_42[0]
    getitem_297: "f32[1, 480, 1, 1]" = var_mean_42[1];  var_mean_42 = None
    add_221: "f32[1, 480, 1, 1]" = torch.ops.aten.add.Tensor(getitem_296, 0.001)
    rsqrt_42: "f32[1, 480, 1, 1]" = torch.ops.aten.rsqrt.default(add_221);  add_221 = None
    sub_42: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(cat_31, getitem_297)
    mul_338: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_42);  sub_42 = None
    squeeze_126: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_297, [0, 2, 3]);  getitem_297 = None
    squeeze_127: "f32[480]" = torch.ops.aten.squeeze.dims(rsqrt_42, [0, 2, 3]);  rsqrt_42 = None
    mul_339: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_126, 0.1)
    mul_340: "f32[480]" = torch.ops.aten.mul.Tensor(primals_433, 0.9)
    add_222: "f32[480]" = torch.ops.aten.add.Tensor(mul_339, mul_340);  mul_339 = mul_340 = None
    squeeze_128: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_296, [0, 2, 3]);  getitem_296 = None
    mul_341: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_128, 1.0006381620931717);  squeeze_128 = None
    mul_342: "f32[480]" = torch.ops.aten.mul.Tensor(mul_341, 0.1);  mul_341 = None
    mul_343: "f32[480]" = torch.ops.aten.mul.Tensor(primals_434, 0.9)
    add_223: "f32[480]" = torch.ops.aten.add.Tensor(mul_342, mul_343);  mul_342 = mul_343 = None
    unsqueeze_168: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_96, -1)
    unsqueeze_169: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_168, -1);  unsqueeze_168 = None
    mul_344: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_338, unsqueeze_169);  mul_338 = unsqueeze_169 = None
    unsqueeze_170: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_97, -1);  primals_97 = None
    unsqueeze_171: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_170, -1);  unsqueeze_170 = None
    add_224: "f32[8, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_344, unsqueeze_171);  mul_344 = unsqueeze_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_33: "f32[8, 480, 14, 14]" = torch.ops.aten.clone.default(add_224)
    sigmoid_44: "f32[8, 480, 14, 14]" = torch.ops.aten.sigmoid.default(add_224)
    mul_345: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_224, sigmoid_44);  add_224 = sigmoid_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    split_with_sizes_73 = torch.ops.aten.split_with_sizes.default(mul_345, [120, 120, 120, 120], 1)
    getitem_302: "f32[8, 120, 14, 14]" = split_with_sizes_73[0];  split_with_sizes_73 = None
    convolution_111: "f32[8, 120, 14, 14]" = torch.ops.aten.convolution.default(getitem_302, primals_254, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 120)
    split_with_sizes_74 = torch.ops.aten.split_with_sizes.default(mul_345, [120, 120, 120, 120], 1)
    getitem_307: "f32[8, 120, 14, 14]" = split_with_sizes_74[1];  split_with_sizes_74 = None
    convolution_112: "f32[8, 120, 14, 14]" = torch.ops.aten.convolution.default(getitem_307, primals_255, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 120)
    split_with_sizes_75 = torch.ops.aten.split_with_sizes.default(mul_345, [120, 120, 120, 120], 1)
    getitem_312: "f32[8, 120, 14, 14]" = split_with_sizes_75[2];  split_with_sizes_75 = None
    convolution_113: "f32[8, 120, 14, 14]" = torch.ops.aten.convolution.default(getitem_312, primals_256, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 120)
    split_with_sizes_76 = torch.ops.aten.split_with_sizes.default(mul_345, [120, 120, 120, 120], 1);  mul_345 = None
    getitem_317: "f32[8, 120, 14, 14]" = split_with_sizes_76[3];  split_with_sizes_76 = None
    convolution_114: "f32[8, 120, 14, 14]" = torch.ops.aten.convolution.default(getitem_317, primals_257, None, [1, 1], [4, 4], [1, 1], False, [0, 0], 120)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_32: "f32[8, 480, 14, 14]" = torch.ops.aten.cat.default([convolution_111, convolution_112, convolution_113, convolution_114], 1);  convolution_111 = convolution_112 = convolution_113 = convolution_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_225: "i64[]" = torch.ops.aten.add.Tensor(primals_435, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_43 = torch.ops.aten.var_mean.correction(cat_32, [0, 2, 3], correction = 0, keepdim = True)
    getitem_318: "f32[1, 480, 1, 1]" = var_mean_43[0]
    getitem_319: "f32[1, 480, 1, 1]" = var_mean_43[1];  var_mean_43 = None
    add_226: "f32[1, 480, 1, 1]" = torch.ops.aten.add.Tensor(getitem_318, 0.001)
    rsqrt_43: "f32[1, 480, 1, 1]" = torch.ops.aten.rsqrt.default(add_226);  add_226 = None
    sub_43: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(cat_32, getitem_319)
    mul_346: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_43);  sub_43 = None
    squeeze_129: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_319, [0, 2, 3]);  getitem_319 = None
    squeeze_130: "f32[480]" = torch.ops.aten.squeeze.dims(rsqrt_43, [0, 2, 3]);  rsqrt_43 = None
    mul_347: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_129, 0.1)
    mul_348: "f32[480]" = torch.ops.aten.mul.Tensor(primals_436, 0.9)
    add_227: "f32[480]" = torch.ops.aten.add.Tensor(mul_347, mul_348);  mul_347 = mul_348 = None
    squeeze_131: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_318, [0, 2, 3]);  getitem_318 = None
    mul_349: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_131, 1.0006381620931717);  squeeze_131 = None
    mul_350: "f32[480]" = torch.ops.aten.mul.Tensor(mul_349, 0.1);  mul_349 = None
    mul_351: "f32[480]" = torch.ops.aten.mul.Tensor(primals_437, 0.9)
    add_228: "f32[480]" = torch.ops.aten.add.Tensor(mul_350, mul_351);  mul_350 = mul_351 = None
    unsqueeze_172: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_98, -1)
    unsqueeze_173: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_172, -1);  unsqueeze_172 = None
    mul_352: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_346, unsqueeze_173);  mul_346 = unsqueeze_173 = None
    unsqueeze_174: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_99, -1);  primals_99 = None
    unsqueeze_175: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_174, -1);  unsqueeze_174 = None
    add_229: "f32[8, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_352, unsqueeze_175);  mul_352 = unsqueeze_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_34: "f32[8, 480, 14, 14]" = torch.ops.aten.clone.default(add_229)
    sigmoid_45: "f32[8, 480, 14, 14]" = torch.ops.aten.sigmoid.default(add_229)
    mul_353: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_229, sigmoid_45);  add_229 = sigmoid_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_11: "f32[8, 480, 1, 1]" = torch.ops.aten.mean.dim(mul_353, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_115: "f32[8, 80, 1, 1]" = torch.ops.aten.convolution.default(mean_11, primals_258, primals_259, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_35: "f32[8, 80, 1, 1]" = torch.ops.aten.clone.default(convolution_115)
    sigmoid_46: "f32[8, 80, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_115)
    mul_354: "f32[8, 80, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_115, sigmoid_46);  convolution_115 = sigmoid_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_116: "f32[8, 480, 1, 1]" = torch.ops.aten.convolution.default(mul_354, primals_260, primals_261, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_47: "f32[8, 480, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_116);  convolution_116 = None
    alias_17: "f32[8, 480, 1, 1]" = torch.ops.aten.alias.default(sigmoid_47)
    mul_355: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_353, sigmoid_47)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_with_sizes_77 = torch.ops.aten.split_with_sizes.default(mul_355, [240, 240], 1);  mul_355 = None
    getitem_320: "f32[8, 240, 14, 14]" = split_with_sizes_77[0]
    getitem_321: "f32[8, 240, 14, 14]" = split_with_sizes_77[1];  split_with_sizes_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_117: "f32[8, 80, 14, 14]" = torch.ops.aten.convolution.default(getitem_320, primals_262, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convolution_118: "f32[8, 80, 14, 14]" = torch.ops.aten.convolution.default(getitem_321, primals_263, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_33: "f32[8, 160, 14, 14]" = torch.ops.aten.cat.default([convolution_117, convolution_118], 1);  convolution_117 = convolution_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_230: "i64[]" = torch.ops.aten.add.Tensor(primals_438, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_44 = torch.ops.aten.var_mean.correction(cat_33, [0, 2, 3], correction = 0, keepdim = True)
    getitem_322: "f32[1, 160, 1, 1]" = var_mean_44[0]
    getitem_323: "f32[1, 160, 1, 1]" = var_mean_44[1];  var_mean_44 = None
    add_231: "f32[1, 160, 1, 1]" = torch.ops.aten.add.Tensor(getitem_322, 0.001)
    rsqrt_44: "f32[1, 160, 1, 1]" = torch.ops.aten.rsqrt.default(add_231);  add_231 = None
    sub_44: "f32[8, 160, 14, 14]" = torch.ops.aten.sub.Tensor(cat_33, getitem_323)
    mul_356: "f32[8, 160, 14, 14]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_44);  sub_44 = None
    squeeze_132: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_323, [0, 2, 3]);  getitem_323 = None
    squeeze_133: "f32[160]" = torch.ops.aten.squeeze.dims(rsqrt_44, [0, 2, 3]);  rsqrt_44 = None
    mul_357: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_132, 0.1)
    mul_358: "f32[160]" = torch.ops.aten.mul.Tensor(primals_439, 0.9)
    add_232: "f32[160]" = torch.ops.aten.add.Tensor(mul_357, mul_358);  mul_357 = mul_358 = None
    squeeze_134: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_322, [0, 2, 3]);  getitem_322 = None
    mul_359: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_134, 1.0006381620931717);  squeeze_134 = None
    mul_360: "f32[160]" = torch.ops.aten.mul.Tensor(mul_359, 0.1);  mul_359 = None
    mul_361: "f32[160]" = torch.ops.aten.mul.Tensor(primals_440, 0.9)
    add_233: "f32[160]" = torch.ops.aten.add.Tensor(mul_360, mul_361);  mul_360 = mul_361 = None
    unsqueeze_176: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_100, -1)
    unsqueeze_177: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_176, -1);  unsqueeze_176 = None
    mul_362: "f32[8, 160, 14, 14]" = torch.ops.aten.mul.Tensor(mul_356, unsqueeze_177);  mul_356 = unsqueeze_177 = None
    unsqueeze_178: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_101, -1);  primals_101 = None
    unsqueeze_179: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_178, -1);  unsqueeze_178 = None
    add_234: "f32[8, 160, 14, 14]" = torch.ops.aten.add.Tensor(mul_362, unsqueeze_179);  mul_362 = unsqueeze_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_235: "f32[8, 160, 14, 14]" = torch.ops.aten.add.Tensor(add_234, add_219);  add_234 = add_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_119: "f32[8, 960, 14, 14]" = torch.ops.aten.convolution.default(add_235, primals_264, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_236: "i64[]" = torch.ops.aten.add.Tensor(primals_441, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_45 = torch.ops.aten.var_mean.correction(convolution_119, [0, 2, 3], correction = 0, keepdim = True)
    getitem_324: "f32[1, 960, 1, 1]" = var_mean_45[0]
    getitem_325: "f32[1, 960, 1, 1]" = var_mean_45[1];  var_mean_45 = None
    add_237: "f32[1, 960, 1, 1]" = torch.ops.aten.add.Tensor(getitem_324, 0.001)
    rsqrt_45: "f32[1, 960, 1, 1]" = torch.ops.aten.rsqrt.default(add_237);  add_237 = None
    sub_45: "f32[8, 960, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_119, getitem_325)
    mul_363: "f32[8, 960, 14, 14]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_45);  sub_45 = None
    squeeze_135: "f32[960]" = torch.ops.aten.squeeze.dims(getitem_325, [0, 2, 3]);  getitem_325 = None
    squeeze_136: "f32[960]" = torch.ops.aten.squeeze.dims(rsqrt_45, [0, 2, 3]);  rsqrt_45 = None
    mul_364: "f32[960]" = torch.ops.aten.mul.Tensor(squeeze_135, 0.1)
    mul_365: "f32[960]" = torch.ops.aten.mul.Tensor(primals_442, 0.9)
    add_238: "f32[960]" = torch.ops.aten.add.Tensor(mul_364, mul_365);  mul_364 = mul_365 = None
    squeeze_137: "f32[960]" = torch.ops.aten.squeeze.dims(getitem_324, [0, 2, 3]);  getitem_324 = None
    mul_366: "f32[960]" = torch.ops.aten.mul.Tensor(squeeze_137, 1.0006381620931717);  squeeze_137 = None
    mul_367: "f32[960]" = torch.ops.aten.mul.Tensor(mul_366, 0.1);  mul_366 = None
    mul_368: "f32[960]" = torch.ops.aten.mul.Tensor(primals_443, 0.9)
    add_239: "f32[960]" = torch.ops.aten.add.Tensor(mul_367, mul_368);  mul_367 = mul_368 = None
    unsqueeze_180: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(primals_102, -1)
    unsqueeze_181: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_180, -1);  unsqueeze_180 = None
    mul_369: "f32[8, 960, 14, 14]" = torch.ops.aten.mul.Tensor(mul_363, unsqueeze_181);  mul_363 = unsqueeze_181 = None
    unsqueeze_182: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(primals_103, -1);  primals_103 = None
    unsqueeze_183: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_182, -1);  unsqueeze_182 = None
    add_240: "f32[8, 960, 14, 14]" = torch.ops.aten.add.Tensor(mul_369, unsqueeze_183);  mul_369 = unsqueeze_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_36: "f32[8, 960, 14, 14]" = torch.ops.aten.clone.default(add_240)
    sigmoid_48: "f32[8, 960, 14, 14]" = torch.ops.aten.sigmoid.default(add_240)
    mul_370: "f32[8, 960, 14, 14]" = torch.ops.aten.mul.Tensor(add_240, sigmoid_48);  add_240 = sigmoid_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    split_with_sizes_79 = torch.ops.aten.split_with_sizes.default(mul_370, [240, 240, 240, 240], 1)
    getitem_330: "f32[8, 240, 14, 14]" = split_with_sizes_79[0];  split_with_sizes_79 = None
    constant_pad_nd_11: "f32[8, 240, 15, 15]" = torch.ops.aten.constant_pad_nd.default(getitem_330, [0, 1, 0, 1], 0.0);  getitem_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_120: "f32[8, 240, 7, 7]" = torch.ops.aten.convolution.default(constant_pad_nd_11, primals_104, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 240)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    split_with_sizes_80 = torch.ops.aten.split_with_sizes.default(mul_370, [240, 240, 240, 240], 1)
    getitem_335: "f32[8, 240, 14, 14]" = split_with_sizes_80[1];  split_with_sizes_80 = None
    constant_pad_nd_12: "f32[8, 240, 17, 17]" = torch.ops.aten.constant_pad_nd.default(getitem_335, [1, 2, 1, 2], 0.0);  getitem_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_121: "f32[8, 240, 7, 7]" = torch.ops.aten.convolution.default(constant_pad_nd_12, primals_105, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 240)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    split_with_sizes_81 = torch.ops.aten.split_with_sizes.default(mul_370, [240, 240, 240, 240], 1)
    getitem_340: "f32[8, 240, 14, 14]" = split_with_sizes_81[2];  split_with_sizes_81 = None
    constant_pad_nd_13: "f32[8, 240, 19, 19]" = torch.ops.aten.constant_pad_nd.default(getitem_340, [2, 3, 2, 3], 0.0);  getitem_340 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_122: "f32[8, 240, 7, 7]" = torch.ops.aten.convolution.default(constant_pad_nd_13, primals_106, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 240)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    split_with_sizes_82 = torch.ops.aten.split_with_sizes.default(mul_370, [240, 240, 240, 240], 1);  mul_370 = None
    getitem_345: "f32[8, 240, 14, 14]" = split_with_sizes_82[3];  split_with_sizes_82 = None
    constant_pad_nd_14: "f32[8, 240, 21, 21]" = torch.ops.aten.constant_pad_nd.default(getitem_345, [3, 4, 3, 4], 0.0);  getitem_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_123: "f32[8, 240, 7, 7]" = torch.ops.aten.convolution.default(constant_pad_nd_14, primals_107, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 240)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_34: "f32[8, 960, 7, 7]" = torch.ops.aten.cat.default([convolution_120, convolution_121, convolution_122, convolution_123], 1);  convolution_120 = convolution_121 = convolution_122 = convolution_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_241: "i64[]" = torch.ops.aten.add.Tensor(primals_444, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_46 = torch.ops.aten.var_mean.correction(cat_34, [0, 2, 3], correction = 0, keepdim = True)
    getitem_346: "f32[1, 960, 1, 1]" = var_mean_46[0]
    getitem_347: "f32[1, 960, 1, 1]" = var_mean_46[1];  var_mean_46 = None
    add_242: "f32[1, 960, 1, 1]" = torch.ops.aten.add.Tensor(getitem_346, 0.001)
    rsqrt_46: "f32[1, 960, 1, 1]" = torch.ops.aten.rsqrt.default(add_242);  add_242 = None
    sub_46: "f32[8, 960, 7, 7]" = torch.ops.aten.sub.Tensor(cat_34, getitem_347)
    mul_371: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_46);  sub_46 = None
    squeeze_138: "f32[960]" = torch.ops.aten.squeeze.dims(getitem_347, [0, 2, 3]);  getitem_347 = None
    squeeze_139: "f32[960]" = torch.ops.aten.squeeze.dims(rsqrt_46, [0, 2, 3]);  rsqrt_46 = None
    mul_372: "f32[960]" = torch.ops.aten.mul.Tensor(squeeze_138, 0.1)
    mul_373: "f32[960]" = torch.ops.aten.mul.Tensor(primals_445, 0.9)
    add_243: "f32[960]" = torch.ops.aten.add.Tensor(mul_372, mul_373);  mul_372 = mul_373 = None
    squeeze_140: "f32[960]" = torch.ops.aten.squeeze.dims(getitem_346, [0, 2, 3]);  getitem_346 = None
    mul_374: "f32[960]" = torch.ops.aten.mul.Tensor(squeeze_140, 1.0025575447570332);  squeeze_140 = None
    mul_375: "f32[960]" = torch.ops.aten.mul.Tensor(mul_374, 0.1);  mul_374 = None
    mul_376: "f32[960]" = torch.ops.aten.mul.Tensor(primals_446, 0.9)
    add_244: "f32[960]" = torch.ops.aten.add.Tensor(mul_375, mul_376);  mul_375 = mul_376 = None
    unsqueeze_184: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(primals_108, -1)
    unsqueeze_185: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_184, -1);  unsqueeze_184 = None
    mul_377: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(mul_371, unsqueeze_185);  mul_371 = unsqueeze_185 = None
    unsqueeze_186: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(primals_109, -1);  primals_109 = None
    unsqueeze_187: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_186, -1);  unsqueeze_186 = None
    add_245: "f32[8, 960, 7, 7]" = torch.ops.aten.add.Tensor(mul_377, unsqueeze_187);  mul_377 = unsqueeze_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_37: "f32[8, 960, 7, 7]" = torch.ops.aten.clone.default(add_245)
    sigmoid_49: "f32[8, 960, 7, 7]" = torch.ops.aten.sigmoid.default(add_245)
    mul_378: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(add_245, sigmoid_49);  add_245 = sigmoid_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_12: "f32[8, 960, 1, 1]" = torch.ops.aten.mean.dim(mul_378, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_124: "f32[8, 80, 1, 1]" = torch.ops.aten.convolution.default(mean_12, primals_265, primals_266, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_38: "f32[8, 80, 1, 1]" = torch.ops.aten.clone.default(convolution_124)
    sigmoid_50: "f32[8, 80, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_124)
    mul_379: "f32[8, 80, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_124, sigmoid_50);  convolution_124 = sigmoid_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_125: "f32[8, 960, 1, 1]" = torch.ops.aten.convolution.default(mul_379, primals_267, primals_268, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_51: "f32[8, 960, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_125);  convolution_125 = None
    alias_18: "f32[8, 960, 1, 1]" = torch.ops.aten.alias.default(sigmoid_51)
    mul_380: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(mul_378, sigmoid_51)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_126: "f32[8, 264, 7, 7]" = torch.ops.aten.convolution.default(mul_380, primals_269, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_246: "i64[]" = torch.ops.aten.add.Tensor(primals_447, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_47 = torch.ops.aten.var_mean.correction(convolution_126, [0, 2, 3], correction = 0, keepdim = True)
    getitem_348: "f32[1, 264, 1, 1]" = var_mean_47[0]
    getitem_349: "f32[1, 264, 1, 1]" = var_mean_47[1];  var_mean_47 = None
    add_247: "f32[1, 264, 1, 1]" = torch.ops.aten.add.Tensor(getitem_348, 0.001)
    rsqrt_47: "f32[1, 264, 1, 1]" = torch.ops.aten.rsqrt.default(add_247);  add_247 = None
    sub_47: "f32[8, 264, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_126, getitem_349)
    mul_381: "f32[8, 264, 7, 7]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_47);  sub_47 = None
    squeeze_141: "f32[264]" = torch.ops.aten.squeeze.dims(getitem_349, [0, 2, 3]);  getitem_349 = None
    squeeze_142: "f32[264]" = torch.ops.aten.squeeze.dims(rsqrt_47, [0, 2, 3]);  rsqrt_47 = None
    mul_382: "f32[264]" = torch.ops.aten.mul.Tensor(squeeze_141, 0.1)
    mul_383: "f32[264]" = torch.ops.aten.mul.Tensor(primals_448, 0.9)
    add_248: "f32[264]" = torch.ops.aten.add.Tensor(mul_382, mul_383);  mul_382 = mul_383 = None
    squeeze_143: "f32[264]" = torch.ops.aten.squeeze.dims(getitem_348, [0, 2, 3]);  getitem_348 = None
    mul_384: "f32[264]" = torch.ops.aten.mul.Tensor(squeeze_143, 1.0025575447570332);  squeeze_143 = None
    mul_385: "f32[264]" = torch.ops.aten.mul.Tensor(mul_384, 0.1);  mul_384 = None
    mul_386: "f32[264]" = torch.ops.aten.mul.Tensor(primals_449, 0.9)
    add_249: "f32[264]" = torch.ops.aten.add.Tensor(mul_385, mul_386);  mul_385 = mul_386 = None
    unsqueeze_188: "f32[264, 1]" = torch.ops.aten.unsqueeze.default(primals_110, -1)
    unsqueeze_189: "f32[264, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_188, -1);  unsqueeze_188 = None
    mul_387: "f32[8, 264, 7, 7]" = torch.ops.aten.mul.Tensor(mul_381, unsqueeze_189);  mul_381 = unsqueeze_189 = None
    unsqueeze_190: "f32[264, 1]" = torch.ops.aten.unsqueeze.default(primals_111, -1);  primals_111 = None
    unsqueeze_191: "f32[264, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_190, -1);  unsqueeze_190 = None
    add_250: "f32[8, 264, 7, 7]" = torch.ops.aten.add.Tensor(mul_387, unsqueeze_191);  mul_387 = unsqueeze_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_127: "f32[8, 1584, 7, 7]" = torch.ops.aten.convolution.default(add_250, primals_270, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_251: "i64[]" = torch.ops.aten.add.Tensor(primals_450, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_48 = torch.ops.aten.var_mean.correction(convolution_127, [0, 2, 3], correction = 0, keepdim = True)
    getitem_350: "f32[1, 1584, 1, 1]" = var_mean_48[0]
    getitem_351: "f32[1, 1584, 1, 1]" = var_mean_48[1];  var_mean_48 = None
    add_252: "f32[1, 1584, 1, 1]" = torch.ops.aten.add.Tensor(getitem_350, 0.001)
    rsqrt_48: "f32[1, 1584, 1, 1]" = torch.ops.aten.rsqrt.default(add_252);  add_252 = None
    sub_48: "f32[8, 1584, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_127, getitem_351)
    mul_388: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_48);  sub_48 = None
    squeeze_144: "f32[1584]" = torch.ops.aten.squeeze.dims(getitem_351, [0, 2, 3]);  getitem_351 = None
    squeeze_145: "f32[1584]" = torch.ops.aten.squeeze.dims(rsqrt_48, [0, 2, 3]);  rsqrt_48 = None
    mul_389: "f32[1584]" = torch.ops.aten.mul.Tensor(squeeze_144, 0.1)
    mul_390: "f32[1584]" = torch.ops.aten.mul.Tensor(primals_451, 0.9)
    add_253: "f32[1584]" = torch.ops.aten.add.Tensor(mul_389, mul_390);  mul_389 = mul_390 = None
    squeeze_146: "f32[1584]" = torch.ops.aten.squeeze.dims(getitem_350, [0, 2, 3]);  getitem_350 = None
    mul_391: "f32[1584]" = torch.ops.aten.mul.Tensor(squeeze_146, 1.0025575447570332);  squeeze_146 = None
    mul_392: "f32[1584]" = torch.ops.aten.mul.Tensor(mul_391, 0.1);  mul_391 = None
    mul_393: "f32[1584]" = torch.ops.aten.mul.Tensor(primals_452, 0.9)
    add_254: "f32[1584]" = torch.ops.aten.add.Tensor(mul_392, mul_393);  mul_392 = mul_393 = None
    unsqueeze_192: "f32[1584, 1]" = torch.ops.aten.unsqueeze.default(primals_112, -1)
    unsqueeze_193: "f32[1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_192, -1);  unsqueeze_192 = None
    mul_394: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(mul_388, unsqueeze_193);  mul_388 = unsqueeze_193 = None
    unsqueeze_194: "f32[1584, 1]" = torch.ops.aten.unsqueeze.default(primals_113, -1);  primals_113 = None
    unsqueeze_195: "f32[1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_194, -1);  unsqueeze_194 = None
    add_255: "f32[8, 1584, 7, 7]" = torch.ops.aten.add.Tensor(mul_394, unsqueeze_195);  mul_394 = unsqueeze_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_39: "f32[8, 1584, 7, 7]" = torch.ops.aten.clone.default(add_255)
    sigmoid_52: "f32[8, 1584, 7, 7]" = torch.ops.aten.sigmoid.default(add_255)
    mul_395: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(add_255, sigmoid_52);  add_255 = sigmoid_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    split_with_sizes_84 = torch.ops.aten.split_with_sizes.default(mul_395, [396, 396, 396, 396], 1)
    getitem_356: "f32[8, 396, 7, 7]" = split_with_sizes_84[0];  split_with_sizes_84 = None
    convolution_128: "f32[8, 396, 7, 7]" = torch.ops.aten.convolution.default(getitem_356, primals_271, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 396)
    split_with_sizes_85 = torch.ops.aten.split_with_sizes.default(mul_395, [396, 396, 396, 396], 1)
    getitem_361: "f32[8, 396, 7, 7]" = split_with_sizes_85[1];  split_with_sizes_85 = None
    convolution_129: "f32[8, 396, 7, 7]" = torch.ops.aten.convolution.default(getitem_361, primals_272, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 396)
    split_with_sizes_86 = torch.ops.aten.split_with_sizes.default(mul_395, [396, 396, 396, 396], 1)
    getitem_366: "f32[8, 396, 7, 7]" = split_with_sizes_86[2];  split_with_sizes_86 = None
    convolution_130: "f32[8, 396, 7, 7]" = torch.ops.aten.convolution.default(getitem_366, primals_273, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 396)
    split_with_sizes_87 = torch.ops.aten.split_with_sizes.default(mul_395, [396, 396, 396, 396], 1);  mul_395 = None
    getitem_371: "f32[8, 396, 7, 7]" = split_with_sizes_87[3];  split_with_sizes_87 = None
    convolution_131: "f32[8, 396, 7, 7]" = torch.ops.aten.convolution.default(getitem_371, primals_274, None, [1, 1], [4, 4], [1, 1], False, [0, 0], 396)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_35: "f32[8, 1584, 7, 7]" = torch.ops.aten.cat.default([convolution_128, convolution_129, convolution_130, convolution_131], 1);  convolution_128 = convolution_129 = convolution_130 = convolution_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_256: "i64[]" = torch.ops.aten.add.Tensor(primals_453, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_49 = torch.ops.aten.var_mean.correction(cat_35, [0, 2, 3], correction = 0, keepdim = True)
    getitem_372: "f32[1, 1584, 1, 1]" = var_mean_49[0]
    getitem_373: "f32[1, 1584, 1, 1]" = var_mean_49[1];  var_mean_49 = None
    add_257: "f32[1, 1584, 1, 1]" = torch.ops.aten.add.Tensor(getitem_372, 0.001)
    rsqrt_49: "f32[1, 1584, 1, 1]" = torch.ops.aten.rsqrt.default(add_257);  add_257 = None
    sub_49: "f32[8, 1584, 7, 7]" = torch.ops.aten.sub.Tensor(cat_35, getitem_373)
    mul_396: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_49);  sub_49 = None
    squeeze_147: "f32[1584]" = torch.ops.aten.squeeze.dims(getitem_373, [0, 2, 3]);  getitem_373 = None
    squeeze_148: "f32[1584]" = torch.ops.aten.squeeze.dims(rsqrt_49, [0, 2, 3]);  rsqrt_49 = None
    mul_397: "f32[1584]" = torch.ops.aten.mul.Tensor(squeeze_147, 0.1)
    mul_398: "f32[1584]" = torch.ops.aten.mul.Tensor(primals_454, 0.9)
    add_258: "f32[1584]" = torch.ops.aten.add.Tensor(mul_397, mul_398);  mul_397 = mul_398 = None
    squeeze_149: "f32[1584]" = torch.ops.aten.squeeze.dims(getitem_372, [0, 2, 3]);  getitem_372 = None
    mul_399: "f32[1584]" = torch.ops.aten.mul.Tensor(squeeze_149, 1.0025575447570332);  squeeze_149 = None
    mul_400: "f32[1584]" = torch.ops.aten.mul.Tensor(mul_399, 0.1);  mul_399 = None
    mul_401: "f32[1584]" = torch.ops.aten.mul.Tensor(primals_455, 0.9)
    add_259: "f32[1584]" = torch.ops.aten.add.Tensor(mul_400, mul_401);  mul_400 = mul_401 = None
    unsqueeze_196: "f32[1584, 1]" = torch.ops.aten.unsqueeze.default(primals_114, -1)
    unsqueeze_197: "f32[1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_196, -1);  unsqueeze_196 = None
    mul_402: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(mul_396, unsqueeze_197);  mul_396 = unsqueeze_197 = None
    unsqueeze_198: "f32[1584, 1]" = torch.ops.aten.unsqueeze.default(primals_115, -1);  primals_115 = None
    unsqueeze_199: "f32[1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_198, -1);  unsqueeze_198 = None
    add_260: "f32[8, 1584, 7, 7]" = torch.ops.aten.add.Tensor(mul_402, unsqueeze_199);  mul_402 = unsqueeze_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_40: "f32[8, 1584, 7, 7]" = torch.ops.aten.clone.default(add_260)
    sigmoid_53: "f32[8, 1584, 7, 7]" = torch.ops.aten.sigmoid.default(add_260)
    mul_403: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(add_260, sigmoid_53);  add_260 = sigmoid_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_13: "f32[8, 1584, 1, 1]" = torch.ops.aten.mean.dim(mul_403, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_132: "f32[8, 132, 1, 1]" = torch.ops.aten.convolution.default(mean_13, primals_275, primals_276, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_41: "f32[8, 132, 1, 1]" = torch.ops.aten.clone.default(convolution_132)
    sigmoid_54: "f32[8, 132, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_132)
    mul_404: "f32[8, 132, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_132, sigmoid_54);  convolution_132 = sigmoid_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_133: "f32[8, 1584, 1, 1]" = torch.ops.aten.convolution.default(mul_404, primals_277, primals_278, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_55: "f32[8, 1584, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_133);  convolution_133 = None
    alias_19: "f32[8, 1584, 1, 1]" = torch.ops.aten.alias.default(sigmoid_55)
    mul_405: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(mul_403, sigmoid_55)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_with_sizes_88 = torch.ops.aten.split_with_sizes.default(mul_405, [792, 792], 1);  mul_405 = None
    getitem_374: "f32[8, 792, 7, 7]" = split_with_sizes_88[0]
    getitem_375: "f32[8, 792, 7, 7]" = split_with_sizes_88[1];  split_with_sizes_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_134: "f32[8, 132, 7, 7]" = torch.ops.aten.convolution.default(getitem_374, primals_279, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convolution_135: "f32[8, 132, 7, 7]" = torch.ops.aten.convolution.default(getitem_375, primals_280, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_36: "f32[8, 264, 7, 7]" = torch.ops.aten.cat.default([convolution_134, convolution_135], 1);  convolution_134 = convolution_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_261: "i64[]" = torch.ops.aten.add.Tensor(primals_456, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_50 = torch.ops.aten.var_mean.correction(cat_36, [0, 2, 3], correction = 0, keepdim = True)
    getitem_376: "f32[1, 264, 1, 1]" = var_mean_50[0]
    getitem_377: "f32[1, 264, 1, 1]" = var_mean_50[1];  var_mean_50 = None
    add_262: "f32[1, 264, 1, 1]" = torch.ops.aten.add.Tensor(getitem_376, 0.001)
    rsqrt_50: "f32[1, 264, 1, 1]" = torch.ops.aten.rsqrt.default(add_262);  add_262 = None
    sub_50: "f32[8, 264, 7, 7]" = torch.ops.aten.sub.Tensor(cat_36, getitem_377)
    mul_406: "f32[8, 264, 7, 7]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_50);  sub_50 = None
    squeeze_150: "f32[264]" = torch.ops.aten.squeeze.dims(getitem_377, [0, 2, 3]);  getitem_377 = None
    squeeze_151: "f32[264]" = torch.ops.aten.squeeze.dims(rsqrt_50, [0, 2, 3]);  rsqrt_50 = None
    mul_407: "f32[264]" = torch.ops.aten.mul.Tensor(squeeze_150, 0.1)
    mul_408: "f32[264]" = torch.ops.aten.mul.Tensor(primals_457, 0.9)
    add_263: "f32[264]" = torch.ops.aten.add.Tensor(mul_407, mul_408);  mul_407 = mul_408 = None
    squeeze_152: "f32[264]" = torch.ops.aten.squeeze.dims(getitem_376, [0, 2, 3]);  getitem_376 = None
    mul_409: "f32[264]" = torch.ops.aten.mul.Tensor(squeeze_152, 1.0025575447570332);  squeeze_152 = None
    mul_410: "f32[264]" = torch.ops.aten.mul.Tensor(mul_409, 0.1);  mul_409 = None
    mul_411: "f32[264]" = torch.ops.aten.mul.Tensor(primals_458, 0.9)
    add_264: "f32[264]" = torch.ops.aten.add.Tensor(mul_410, mul_411);  mul_410 = mul_411 = None
    unsqueeze_200: "f32[264, 1]" = torch.ops.aten.unsqueeze.default(primals_116, -1)
    unsqueeze_201: "f32[264, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_200, -1);  unsqueeze_200 = None
    mul_412: "f32[8, 264, 7, 7]" = torch.ops.aten.mul.Tensor(mul_406, unsqueeze_201);  mul_406 = unsqueeze_201 = None
    unsqueeze_202: "f32[264, 1]" = torch.ops.aten.unsqueeze.default(primals_117, -1);  primals_117 = None
    unsqueeze_203: "f32[264, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_202, -1);  unsqueeze_202 = None
    add_265: "f32[8, 264, 7, 7]" = torch.ops.aten.add.Tensor(mul_412, unsqueeze_203);  mul_412 = unsqueeze_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_266: "f32[8, 264, 7, 7]" = torch.ops.aten.add.Tensor(add_265, add_250);  add_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_136: "f32[8, 1584, 7, 7]" = torch.ops.aten.convolution.default(add_266, primals_281, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_267: "i64[]" = torch.ops.aten.add.Tensor(primals_459, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_51 = torch.ops.aten.var_mean.correction(convolution_136, [0, 2, 3], correction = 0, keepdim = True)
    getitem_378: "f32[1, 1584, 1, 1]" = var_mean_51[0]
    getitem_379: "f32[1, 1584, 1, 1]" = var_mean_51[1];  var_mean_51 = None
    add_268: "f32[1, 1584, 1, 1]" = torch.ops.aten.add.Tensor(getitem_378, 0.001)
    rsqrt_51: "f32[1, 1584, 1, 1]" = torch.ops.aten.rsqrt.default(add_268);  add_268 = None
    sub_51: "f32[8, 1584, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_136, getitem_379)
    mul_413: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_51);  sub_51 = None
    squeeze_153: "f32[1584]" = torch.ops.aten.squeeze.dims(getitem_379, [0, 2, 3]);  getitem_379 = None
    squeeze_154: "f32[1584]" = torch.ops.aten.squeeze.dims(rsqrt_51, [0, 2, 3]);  rsqrt_51 = None
    mul_414: "f32[1584]" = torch.ops.aten.mul.Tensor(squeeze_153, 0.1)
    mul_415: "f32[1584]" = torch.ops.aten.mul.Tensor(primals_460, 0.9)
    add_269: "f32[1584]" = torch.ops.aten.add.Tensor(mul_414, mul_415);  mul_414 = mul_415 = None
    squeeze_155: "f32[1584]" = torch.ops.aten.squeeze.dims(getitem_378, [0, 2, 3]);  getitem_378 = None
    mul_416: "f32[1584]" = torch.ops.aten.mul.Tensor(squeeze_155, 1.0025575447570332);  squeeze_155 = None
    mul_417: "f32[1584]" = torch.ops.aten.mul.Tensor(mul_416, 0.1);  mul_416 = None
    mul_418: "f32[1584]" = torch.ops.aten.mul.Tensor(primals_461, 0.9)
    add_270: "f32[1584]" = torch.ops.aten.add.Tensor(mul_417, mul_418);  mul_417 = mul_418 = None
    unsqueeze_204: "f32[1584, 1]" = torch.ops.aten.unsqueeze.default(primals_118, -1)
    unsqueeze_205: "f32[1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_204, -1);  unsqueeze_204 = None
    mul_419: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(mul_413, unsqueeze_205);  mul_413 = unsqueeze_205 = None
    unsqueeze_206: "f32[1584, 1]" = torch.ops.aten.unsqueeze.default(primals_119, -1);  primals_119 = None
    unsqueeze_207: "f32[1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_206, -1);  unsqueeze_206 = None
    add_271: "f32[8, 1584, 7, 7]" = torch.ops.aten.add.Tensor(mul_419, unsqueeze_207);  mul_419 = unsqueeze_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_42: "f32[8, 1584, 7, 7]" = torch.ops.aten.clone.default(add_271)
    sigmoid_56: "f32[8, 1584, 7, 7]" = torch.ops.aten.sigmoid.default(add_271)
    mul_420: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(add_271, sigmoid_56);  add_271 = sigmoid_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    split_with_sizes_90 = torch.ops.aten.split_with_sizes.default(mul_420, [396, 396, 396, 396], 1)
    getitem_384: "f32[8, 396, 7, 7]" = split_with_sizes_90[0];  split_with_sizes_90 = None
    convolution_137: "f32[8, 396, 7, 7]" = torch.ops.aten.convolution.default(getitem_384, primals_282, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 396)
    split_with_sizes_91 = torch.ops.aten.split_with_sizes.default(mul_420, [396, 396, 396, 396], 1)
    getitem_389: "f32[8, 396, 7, 7]" = split_with_sizes_91[1];  split_with_sizes_91 = None
    convolution_138: "f32[8, 396, 7, 7]" = torch.ops.aten.convolution.default(getitem_389, primals_283, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 396)
    split_with_sizes_92 = torch.ops.aten.split_with_sizes.default(mul_420, [396, 396, 396, 396], 1)
    getitem_394: "f32[8, 396, 7, 7]" = split_with_sizes_92[2];  split_with_sizes_92 = None
    convolution_139: "f32[8, 396, 7, 7]" = torch.ops.aten.convolution.default(getitem_394, primals_284, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 396)
    split_with_sizes_93 = torch.ops.aten.split_with_sizes.default(mul_420, [396, 396, 396, 396], 1);  mul_420 = None
    getitem_399: "f32[8, 396, 7, 7]" = split_with_sizes_93[3];  split_with_sizes_93 = None
    convolution_140: "f32[8, 396, 7, 7]" = torch.ops.aten.convolution.default(getitem_399, primals_285, None, [1, 1], [4, 4], [1, 1], False, [0, 0], 396)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_37: "f32[8, 1584, 7, 7]" = torch.ops.aten.cat.default([convolution_137, convolution_138, convolution_139, convolution_140], 1);  convolution_137 = convolution_138 = convolution_139 = convolution_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_272: "i64[]" = torch.ops.aten.add.Tensor(primals_462, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_52 = torch.ops.aten.var_mean.correction(cat_37, [0, 2, 3], correction = 0, keepdim = True)
    getitem_400: "f32[1, 1584, 1, 1]" = var_mean_52[0]
    getitem_401: "f32[1, 1584, 1, 1]" = var_mean_52[1];  var_mean_52 = None
    add_273: "f32[1, 1584, 1, 1]" = torch.ops.aten.add.Tensor(getitem_400, 0.001)
    rsqrt_52: "f32[1, 1584, 1, 1]" = torch.ops.aten.rsqrt.default(add_273);  add_273 = None
    sub_52: "f32[8, 1584, 7, 7]" = torch.ops.aten.sub.Tensor(cat_37, getitem_401)
    mul_421: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_52);  sub_52 = None
    squeeze_156: "f32[1584]" = torch.ops.aten.squeeze.dims(getitem_401, [0, 2, 3]);  getitem_401 = None
    squeeze_157: "f32[1584]" = torch.ops.aten.squeeze.dims(rsqrt_52, [0, 2, 3]);  rsqrt_52 = None
    mul_422: "f32[1584]" = torch.ops.aten.mul.Tensor(squeeze_156, 0.1)
    mul_423: "f32[1584]" = torch.ops.aten.mul.Tensor(primals_463, 0.9)
    add_274: "f32[1584]" = torch.ops.aten.add.Tensor(mul_422, mul_423);  mul_422 = mul_423 = None
    squeeze_158: "f32[1584]" = torch.ops.aten.squeeze.dims(getitem_400, [0, 2, 3]);  getitem_400 = None
    mul_424: "f32[1584]" = torch.ops.aten.mul.Tensor(squeeze_158, 1.0025575447570332);  squeeze_158 = None
    mul_425: "f32[1584]" = torch.ops.aten.mul.Tensor(mul_424, 0.1);  mul_424 = None
    mul_426: "f32[1584]" = torch.ops.aten.mul.Tensor(primals_464, 0.9)
    add_275: "f32[1584]" = torch.ops.aten.add.Tensor(mul_425, mul_426);  mul_425 = mul_426 = None
    unsqueeze_208: "f32[1584, 1]" = torch.ops.aten.unsqueeze.default(primals_120, -1)
    unsqueeze_209: "f32[1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_208, -1);  unsqueeze_208 = None
    mul_427: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(mul_421, unsqueeze_209);  mul_421 = unsqueeze_209 = None
    unsqueeze_210: "f32[1584, 1]" = torch.ops.aten.unsqueeze.default(primals_121, -1);  primals_121 = None
    unsqueeze_211: "f32[1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_210, -1);  unsqueeze_210 = None
    add_276: "f32[8, 1584, 7, 7]" = torch.ops.aten.add.Tensor(mul_427, unsqueeze_211);  mul_427 = unsqueeze_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_43: "f32[8, 1584, 7, 7]" = torch.ops.aten.clone.default(add_276)
    sigmoid_57: "f32[8, 1584, 7, 7]" = torch.ops.aten.sigmoid.default(add_276)
    mul_428: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(add_276, sigmoid_57);  add_276 = sigmoid_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_14: "f32[8, 1584, 1, 1]" = torch.ops.aten.mean.dim(mul_428, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_141: "f32[8, 132, 1, 1]" = torch.ops.aten.convolution.default(mean_14, primals_286, primals_287, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_44: "f32[8, 132, 1, 1]" = torch.ops.aten.clone.default(convolution_141)
    sigmoid_58: "f32[8, 132, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_141)
    mul_429: "f32[8, 132, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_141, sigmoid_58);  convolution_141 = sigmoid_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_142: "f32[8, 1584, 1, 1]" = torch.ops.aten.convolution.default(mul_429, primals_288, primals_289, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_59: "f32[8, 1584, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_142);  convolution_142 = None
    alias_20: "f32[8, 1584, 1, 1]" = torch.ops.aten.alias.default(sigmoid_59)
    mul_430: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(mul_428, sigmoid_59)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_with_sizes_94 = torch.ops.aten.split_with_sizes.default(mul_430, [792, 792], 1);  mul_430 = None
    getitem_402: "f32[8, 792, 7, 7]" = split_with_sizes_94[0]
    getitem_403: "f32[8, 792, 7, 7]" = split_with_sizes_94[1];  split_with_sizes_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_143: "f32[8, 132, 7, 7]" = torch.ops.aten.convolution.default(getitem_402, primals_290, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convolution_144: "f32[8, 132, 7, 7]" = torch.ops.aten.convolution.default(getitem_403, primals_291, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_38: "f32[8, 264, 7, 7]" = torch.ops.aten.cat.default([convolution_143, convolution_144], 1);  convolution_143 = convolution_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_277: "i64[]" = torch.ops.aten.add.Tensor(primals_465, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_53 = torch.ops.aten.var_mean.correction(cat_38, [0, 2, 3], correction = 0, keepdim = True)
    getitem_404: "f32[1, 264, 1, 1]" = var_mean_53[0]
    getitem_405: "f32[1, 264, 1, 1]" = var_mean_53[1];  var_mean_53 = None
    add_278: "f32[1, 264, 1, 1]" = torch.ops.aten.add.Tensor(getitem_404, 0.001)
    rsqrt_53: "f32[1, 264, 1, 1]" = torch.ops.aten.rsqrt.default(add_278);  add_278 = None
    sub_53: "f32[8, 264, 7, 7]" = torch.ops.aten.sub.Tensor(cat_38, getitem_405)
    mul_431: "f32[8, 264, 7, 7]" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_53);  sub_53 = None
    squeeze_159: "f32[264]" = torch.ops.aten.squeeze.dims(getitem_405, [0, 2, 3]);  getitem_405 = None
    squeeze_160: "f32[264]" = torch.ops.aten.squeeze.dims(rsqrt_53, [0, 2, 3]);  rsqrt_53 = None
    mul_432: "f32[264]" = torch.ops.aten.mul.Tensor(squeeze_159, 0.1)
    mul_433: "f32[264]" = torch.ops.aten.mul.Tensor(primals_466, 0.9)
    add_279: "f32[264]" = torch.ops.aten.add.Tensor(mul_432, mul_433);  mul_432 = mul_433 = None
    squeeze_161: "f32[264]" = torch.ops.aten.squeeze.dims(getitem_404, [0, 2, 3]);  getitem_404 = None
    mul_434: "f32[264]" = torch.ops.aten.mul.Tensor(squeeze_161, 1.0025575447570332);  squeeze_161 = None
    mul_435: "f32[264]" = torch.ops.aten.mul.Tensor(mul_434, 0.1);  mul_434 = None
    mul_436: "f32[264]" = torch.ops.aten.mul.Tensor(primals_467, 0.9)
    add_280: "f32[264]" = torch.ops.aten.add.Tensor(mul_435, mul_436);  mul_435 = mul_436 = None
    unsqueeze_212: "f32[264, 1]" = torch.ops.aten.unsqueeze.default(primals_122, -1)
    unsqueeze_213: "f32[264, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_212, -1);  unsqueeze_212 = None
    mul_437: "f32[8, 264, 7, 7]" = torch.ops.aten.mul.Tensor(mul_431, unsqueeze_213);  mul_431 = unsqueeze_213 = None
    unsqueeze_214: "f32[264, 1]" = torch.ops.aten.unsqueeze.default(primals_123, -1);  primals_123 = None
    unsqueeze_215: "f32[264, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_214, -1);  unsqueeze_214 = None
    add_281: "f32[8, 264, 7, 7]" = torch.ops.aten.add.Tensor(mul_437, unsqueeze_215);  mul_437 = unsqueeze_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_282: "f32[8, 264, 7, 7]" = torch.ops.aten.add.Tensor(add_281, add_266);  add_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_145: "f32[8, 1584, 7, 7]" = torch.ops.aten.convolution.default(add_282, primals_292, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_283: "i64[]" = torch.ops.aten.add.Tensor(primals_468, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_54 = torch.ops.aten.var_mean.correction(convolution_145, [0, 2, 3], correction = 0, keepdim = True)
    getitem_406: "f32[1, 1584, 1, 1]" = var_mean_54[0]
    getitem_407: "f32[1, 1584, 1, 1]" = var_mean_54[1];  var_mean_54 = None
    add_284: "f32[1, 1584, 1, 1]" = torch.ops.aten.add.Tensor(getitem_406, 0.001)
    rsqrt_54: "f32[1, 1584, 1, 1]" = torch.ops.aten.rsqrt.default(add_284);  add_284 = None
    sub_54: "f32[8, 1584, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_145, getitem_407)
    mul_438: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_54);  sub_54 = None
    squeeze_162: "f32[1584]" = torch.ops.aten.squeeze.dims(getitem_407, [0, 2, 3]);  getitem_407 = None
    squeeze_163: "f32[1584]" = torch.ops.aten.squeeze.dims(rsqrt_54, [0, 2, 3]);  rsqrt_54 = None
    mul_439: "f32[1584]" = torch.ops.aten.mul.Tensor(squeeze_162, 0.1)
    mul_440: "f32[1584]" = torch.ops.aten.mul.Tensor(primals_469, 0.9)
    add_285: "f32[1584]" = torch.ops.aten.add.Tensor(mul_439, mul_440);  mul_439 = mul_440 = None
    squeeze_164: "f32[1584]" = torch.ops.aten.squeeze.dims(getitem_406, [0, 2, 3]);  getitem_406 = None
    mul_441: "f32[1584]" = torch.ops.aten.mul.Tensor(squeeze_164, 1.0025575447570332);  squeeze_164 = None
    mul_442: "f32[1584]" = torch.ops.aten.mul.Tensor(mul_441, 0.1);  mul_441 = None
    mul_443: "f32[1584]" = torch.ops.aten.mul.Tensor(primals_470, 0.9)
    add_286: "f32[1584]" = torch.ops.aten.add.Tensor(mul_442, mul_443);  mul_442 = mul_443 = None
    unsqueeze_216: "f32[1584, 1]" = torch.ops.aten.unsqueeze.default(primals_124, -1)
    unsqueeze_217: "f32[1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_216, -1);  unsqueeze_216 = None
    mul_444: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(mul_438, unsqueeze_217);  mul_438 = unsqueeze_217 = None
    unsqueeze_218: "f32[1584, 1]" = torch.ops.aten.unsqueeze.default(primals_125, -1);  primals_125 = None
    unsqueeze_219: "f32[1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_218, -1);  unsqueeze_218 = None
    add_287: "f32[8, 1584, 7, 7]" = torch.ops.aten.add.Tensor(mul_444, unsqueeze_219);  mul_444 = unsqueeze_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_45: "f32[8, 1584, 7, 7]" = torch.ops.aten.clone.default(add_287)
    sigmoid_60: "f32[8, 1584, 7, 7]" = torch.ops.aten.sigmoid.default(add_287)
    mul_445: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(add_287, sigmoid_60);  add_287 = sigmoid_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    split_with_sizes_96 = torch.ops.aten.split_with_sizes.default(mul_445, [396, 396, 396, 396], 1)
    getitem_412: "f32[8, 396, 7, 7]" = split_with_sizes_96[0];  split_with_sizes_96 = None
    convolution_146: "f32[8, 396, 7, 7]" = torch.ops.aten.convolution.default(getitem_412, primals_293, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 396)
    split_with_sizes_97 = torch.ops.aten.split_with_sizes.default(mul_445, [396, 396, 396, 396], 1)
    getitem_417: "f32[8, 396, 7, 7]" = split_with_sizes_97[1];  split_with_sizes_97 = None
    convolution_147: "f32[8, 396, 7, 7]" = torch.ops.aten.convolution.default(getitem_417, primals_294, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 396)
    split_with_sizes_98 = torch.ops.aten.split_with_sizes.default(mul_445, [396, 396, 396, 396], 1)
    getitem_422: "f32[8, 396, 7, 7]" = split_with_sizes_98[2];  split_with_sizes_98 = None
    convolution_148: "f32[8, 396, 7, 7]" = torch.ops.aten.convolution.default(getitem_422, primals_295, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 396)
    split_with_sizes_99 = torch.ops.aten.split_with_sizes.default(mul_445, [396, 396, 396, 396], 1);  mul_445 = None
    getitem_427: "f32[8, 396, 7, 7]" = split_with_sizes_99[3];  split_with_sizes_99 = None
    convolution_149: "f32[8, 396, 7, 7]" = torch.ops.aten.convolution.default(getitem_427, primals_296, None, [1, 1], [4, 4], [1, 1], False, [0, 0], 396)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_39: "f32[8, 1584, 7, 7]" = torch.ops.aten.cat.default([convolution_146, convolution_147, convolution_148, convolution_149], 1);  convolution_146 = convolution_147 = convolution_148 = convolution_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_288: "i64[]" = torch.ops.aten.add.Tensor(primals_471, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_55 = torch.ops.aten.var_mean.correction(cat_39, [0, 2, 3], correction = 0, keepdim = True)
    getitem_428: "f32[1, 1584, 1, 1]" = var_mean_55[0]
    getitem_429: "f32[1, 1584, 1, 1]" = var_mean_55[1];  var_mean_55 = None
    add_289: "f32[1, 1584, 1, 1]" = torch.ops.aten.add.Tensor(getitem_428, 0.001)
    rsqrt_55: "f32[1, 1584, 1, 1]" = torch.ops.aten.rsqrt.default(add_289);  add_289 = None
    sub_55: "f32[8, 1584, 7, 7]" = torch.ops.aten.sub.Tensor(cat_39, getitem_429)
    mul_446: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(sub_55, rsqrt_55);  sub_55 = None
    squeeze_165: "f32[1584]" = torch.ops.aten.squeeze.dims(getitem_429, [0, 2, 3]);  getitem_429 = None
    squeeze_166: "f32[1584]" = torch.ops.aten.squeeze.dims(rsqrt_55, [0, 2, 3]);  rsqrt_55 = None
    mul_447: "f32[1584]" = torch.ops.aten.mul.Tensor(squeeze_165, 0.1)
    mul_448: "f32[1584]" = torch.ops.aten.mul.Tensor(primals_472, 0.9)
    add_290: "f32[1584]" = torch.ops.aten.add.Tensor(mul_447, mul_448);  mul_447 = mul_448 = None
    squeeze_167: "f32[1584]" = torch.ops.aten.squeeze.dims(getitem_428, [0, 2, 3]);  getitem_428 = None
    mul_449: "f32[1584]" = torch.ops.aten.mul.Tensor(squeeze_167, 1.0025575447570332);  squeeze_167 = None
    mul_450: "f32[1584]" = torch.ops.aten.mul.Tensor(mul_449, 0.1);  mul_449 = None
    mul_451: "f32[1584]" = torch.ops.aten.mul.Tensor(primals_473, 0.9)
    add_291: "f32[1584]" = torch.ops.aten.add.Tensor(mul_450, mul_451);  mul_450 = mul_451 = None
    unsqueeze_220: "f32[1584, 1]" = torch.ops.aten.unsqueeze.default(primals_126, -1)
    unsqueeze_221: "f32[1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_220, -1);  unsqueeze_220 = None
    mul_452: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(mul_446, unsqueeze_221);  mul_446 = unsqueeze_221 = None
    unsqueeze_222: "f32[1584, 1]" = torch.ops.aten.unsqueeze.default(primals_127, -1);  primals_127 = None
    unsqueeze_223: "f32[1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_222, -1);  unsqueeze_222 = None
    add_292: "f32[8, 1584, 7, 7]" = torch.ops.aten.add.Tensor(mul_452, unsqueeze_223);  mul_452 = unsqueeze_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_46: "f32[8, 1584, 7, 7]" = torch.ops.aten.clone.default(add_292)
    sigmoid_61: "f32[8, 1584, 7, 7]" = torch.ops.aten.sigmoid.default(add_292)
    mul_453: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(add_292, sigmoid_61);  add_292 = sigmoid_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_15: "f32[8, 1584, 1, 1]" = torch.ops.aten.mean.dim(mul_453, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_150: "f32[8, 132, 1, 1]" = torch.ops.aten.convolution.default(mean_15, primals_297, primals_298, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_47: "f32[8, 132, 1, 1]" = torch.ops.aten.clone.default(convolution_150)
    sigmoid_62: "f32[8, 132, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_150)
    mul_454: "f32[8, 132, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_150, sigmoid_62);  convolution_150 = sigmoid_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_151: "f32[8, 1584, 1, 1]" = torch.ops.aten.convolution.default(mul_454, primals_299, primals_300, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_63: "f32[8, 1584, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_151);  convolution_151 = None
    alias_21: "f32[8, 1584, 1, 1]" = torch.ops.aten.alias.default(sigmoid_63)
    mul_455: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(mul_453, sigmoid_63)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_with_sizes_100 = torch.ops.aten.split_with_sizes.default(mul_455, [792, 792], 1);  mul_455 = None
    getitem_430: "f32[8, 792, 7, 7]" = split_with_sizes_100[0]
    getitem_431: "f32[8, 792, 7, 7]" = split_with_sizes_100[1];  split_with_sizes_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_152: "f32[8, 132, 7, 7]" = torch.ops.aten.convolution.default(getitem_430, primals_301, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convolution_153: "f32[8, 132, 7, 7]" = torch.ops.aten.convolution.default(getitem_431, primals_302, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_40: "f32[8, 264, 7, 7]" = torch.ops.aten.cat.default([convolution_152, convolution_153], 1);  convolution_152 = convolution_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_293: "i64[]" = torch.ops.aten.add.Tensor(primals_474, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_56 = torch.ops.aten.var_mean.correction(cat_40, [0, 2, 3], correction = 0, keepdim = True)
    getitem_432: "f32[1, 264, 1, 1]" = var_mean_56[0]
    getitem_433: "f32[1, 264, 1, 1]" = var_mean_56[1];  var_mean_56 = None
    add_294: "f32[1, 264, 1, 1]" = torch.ops.aten.add.Tensor(getitem_432, 0.001)
    rsqrt_56: "f32[1, 264, 1, 1]" = torch.ops.aten.rsqrt.default(add_294);  add_294 = None
    sub_56: "f32[8, 264, 7, 7]" = torch.ops.aten.sub.Tensor(cat_40, getitem_433)
    mul_456: "f32[8, 264, 7, 7]" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_56);  sub_56 = None
    squeeze_168: "f32[264]" = torch.ops.aten.squeeze.dims(getitem_433, [0, 2, 3]);  getitem_433 = None
    squeeze_169: "f32[264]" = torch.ops.aten.squeeze.dims(rsqrt_56, [0, 2, 3]);  rsqrt_56 = None
    mul_457: "f32[264]" = torch.ops.aten.mul.Tensor(squeeze_168, 0.1)
    mul_458: "f32[264]" = torch.ops.aten.mul.Tensor(primals_475, 0.9)
    add_295: "f32[264]" = torch.ops.aten.add.Tensor(mul_457, mul_458);  mul_457 = mul_458 = None
    squeeze_170: "f32[264]" = torch.ops.aten.squeeze.dims(getitem_432, [0, 2, 3]);  getitem_432 = None
    mul_459: "f32[264]" = torch.ops.aten.mul.Tensor(squeeze_170, 1.0025575447570332);  squeeze_170 = None
    mul_460: "f32[264]" = torch.ops.aten.mul.Tensor(mul_459, 0.1);  mul_459 = None
    mul_461: "f32[264]" = torch.ops.aten.mul.Tensor(primals_476, 0.9)
    add_296: "f32[264]" = torch.ops.aten.add.Tensor(mul_460, mul_461);  mul_460 = mul_461 = None
    unsqueeze_224: "f32[264, 1]" = torch.ops.aten.unsqueeze.default(primals_128, -1)
    unsqueeze_225: "f32[264, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_224, -1);  unsqueeze_224 = None
    mul_462: "f32[8, 264, 7, 7]" = torch.ops.aten.mul.Tensor(mul_456, unsqueeze_225);  mul_456 = unsqueeze_225 = None
    unsqueeze_226: "f32[264, 1]" = torch.ops.aten.unsqueeze.default(primals_129, -1);  primals_129 = None
    unsqueeze_227: "f32[264, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_226, -1);  unsqueeze_226 = None
    add_297: "f32[8, 264, 7, 7]" = torch.ops.aten.add.Tensor(mul_462, unsqueeze_227);  mul_462 = unsqueeze_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_298: "f32[8, 264, 7, 7]" = torch.ops.aten.add.Tensor(add_297, add_282);  add_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/efficientnet.py:168, code: x = self.conv_head(x)
    convolution_154: "f32[8, 1536, 7, 7]" = torch.ops.aten.convolution.default(add_298, primals_303, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_299: "i64[]" = torch.ops.aten.add.Tensor(primals_477, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_57 = torch.ops.aten.var_mean.correction(convolution_154, [0, 2, 3], correction = 0, keepdim = True)
    getitem_434: "f32[1, 1536, 1, 1]" = var_mean_57[0]
    getitem_435: "f32[1, 1536, 1, 1]" = var_mean_57[1];  var_mean_57 = None
    add_300: "f32[1, 1536, 1, 1]" = torch.ops.aten.add.Tensor(getitem_434, 0.001)
    rsqrt_57: "f32[1, 1536, 1, 1]" = torch.ops.aten.rsqrt.default(add_300);  add_300 = None
    sub_57: "f32[8, 1536, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_154, getitem_435)
    mul_463: "f32[8, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(sub_57, rsqrt_57);  sub_57 = None
    squeeze_171: "f32[1536]" = torch.ops.aten.squeeze.dims(getitem_435, [0, 2, 3]);  getitem_435 = None
    squeeze_172: "f32[1536]" = torch.ops.aten.squeeze.dims(rsqrt_57, [0, 2, 3]);  rsqrt_57 = None
    mul_464: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_171, 0.1)
    mul_465: "f32[1536]" = torch.ops.aten.mul.Tensor(primals_478, 0.9)
    add_301: "f32[1536]" = torch.ops.aten.add.Tensor(mul_464, mul_465);  mul_464 = mul_465 = None
    squeeze_173: "f32[1536]" = torch.ops.aten.squeeze.dims(getitem_434, [0, 2, 3]);  getitem_434 = None
    mul_466: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_173, 1.0025575447570332);  squeeze_173 = None
    mul_467: "f32[1536]" = torch.ops.aten.mul.Tensor(mul_466, 0.1);  mul_466 = None
    mul_468: "f32[1536]" = torch.ops.aten.mul.Tensor(primals_479, 0.9)
    add_302: "f32[1536]" = torch.ops.aten.add.Tensor(mul_467, mul_468);  mul_467 = mul_468 = None
    unsqueeze_228: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(primals_130, -1)
    unsqueeze_229: "f32[1536, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_228, -1);  unsqueeze_228 = None
    mul_469: "f32[8, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(mul_463, unsqueeze_229);  mul_463 = unsqueeze_229 = None
    unsqueeze_230: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(primals_131, -1);  primals_131 = None
    unsqueeze_231: "f32[1536, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_230, -1);  unsqueeze_230 = None
    add_303: "f32[8, 1536, 7, 7]" = torch.ops.aten.add.Tensor(mul_469, unsqueeze_231);  mul_469 = unsqueeze_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_6: "f32[8, 1536, 7, 7]" = torch.ops.aten.relu.default(add_303);  add_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    mean_16: "f32[8, 1536, 1, 1]" = torch.ops.aten.mean.dim(relu_6, [-1, -2], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view: "f32[8, 1536]" = torch.ops.aten.view.default(mean_16, [8, 1536]);  mean_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/efficientnet.py:176, code: return x if pre_logits else self.classifier(x)
    permute: "f32[1536, 1000]" = torch.ops.aten.permute.default(primals_304, [1, 0]);  primals_304 = None
    addmm: "f32[8, 1000]" = torch.ops.aten.addmm.default(primals_305, view, permute);  primals_305 = None
    permute_1: "f32[1000, 1536]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
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
    alias_24: "f32[8, 1536, 7, 7]" = torch.ops.aten.alias.default(relu_6);  relu_6 = None
    alias_25: "f32[8, 1536, 7, 7]" = torch.ops.aten.alias.default(alias_24);  alias_24 = None
    le: "b8[8, 1536, 7, 7]" = torch.ops.aten.le.Scalar(alias_25, 0);  alias_25 = None
    scalar_tensor: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where: "f32[8, 1536, 7, 7]" = torch.ops.aten.where.self(le, scalar_tensor, div);  le = scalar_tensor = div = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_232: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(squeeze_171, 0);  squeeze_171 = None
    unsqueeze_233: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_232, 2);  unsqueeze_232 = None
    unsqueeze_234: "f32[1, 1536, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_233, 3);  unsqueeze_233 = None
    sum_2: "f32[1536]" = torch.ops.aten.sum.dim_IntList(where, [0, 2, 3])
    sub_58: "f32[8, 1536, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_154, unsqueeze_234)
    mul_470: "f32[8, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(where, sub_58);  sub_58 = None
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
    mul_475: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_172, primals_130);  primals_130 = None
    unsqueeze_241: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_475, 0);  mul_475 = None
    unsqueeze_242: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_241, 2);  unsqueeze_241 = None
    unsqueeze_243: "f32[1, 1536, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_242, 3);  unsqueeze_242 = None
    sub_59: "f32[8, 1536, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_154, unsqueeze_234);  convolution_154 = unsqueeze_234 = None
    mul_476: "f32[8, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(sub_59, unsqueeze_240);  sub_59 = unsqueeze_240 = None
    sub_60: "f32[8, 1536, 7, 7]" = torch.ops.aten.sub.Tensor(where, mul_476);  where = mul_476 = None
    sub_61: "f32[8, 1536, 7, 7]" = torch.ops.aten.sub.Tensor(sub_60, unsqueeze_237);  sub_60 = unsqueeze_237 = None
    mul_477: "f32[8, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(sub_61, unsqueeze_243);  sub_61 = unsqueeze_243 = None
    mul_478: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_3, squeeze_172);  sum_3 = squeeze_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/efficientnet.py:168, code: x = self.conv_head(x)
    convolution_backward = torch.ops.aten.convolution_backward.default(mul_477, add_298, primals_303, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_477 = add_298 = primals_303 = None
    getitem_436: "f32[8, 264, 7, 7]" = convolution_backward[0]
    getitem_437: "f32[1536, 264, 1, 1]" = convolution_backward[1];  convolution_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_244: "f32[1, 264]" = torch.ops.aten.unsqueeze.default(squeeze_168, 0);  squeeze_168 = None
    unsqueeze_245: "f32[1, 264, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_244, 2);  unsqueeze_244 = None
    unsqueeze_246: "f32[1, 264, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_245, 3);  unsqueeze_245 = None
    sum_4: "f32[264]" = torch.ops.aten.sum.dim_IntList(getitem_436, [0, 2, 3])
    sub_62: "f32[8, 264, 7, 7]" = torch.ops.aten.sub.Tensor(cat_40, unsqueeze_246)
    mul_479: "f32[8, 264, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_436, sub_62);  sub_62 = None
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
    mul_484: "f32[264]" = torch.ops.aten.mul.Tensor(squeeze_169, primals_128);  primals_128 = None
    unsqueeze_253: "f32[1, 264]" = torch.ops.aten.unsqueeze.default(mul_484, 0);  mul_484 = None
    unsqueeze_254: "f32[1, 264, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_253, 2);  unsqueeze_253 = None
    unsqueeze_255: "f32[1, 264, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_254, 3);  unsqueeze_254 = None
    sub_63: "f32[8, 264, 7, 7]" = torch.ops.aten.sub.Tensor(cat_40, unsqueeze_246);  cat_40 = unsqueeze_246 = None
    mul_485: "f32[8, 264, 7, 7]" = torch.ops.aten.mul.Tensor(sub_63, unsqueeze_252);  sub_63 = unsqueeze_252 = None
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
    full: "f32[8, 132, 1, 1]" = torch.ops.aten.full.default([8, 132, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_67: "f32[8, 132, 1, 1]" = torch.ops.aten.sub.Tensor(full, sigmoid_64);  full = None
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
    full_1: "f32[8, 1584, 7, 7]" = torch.ops.aten.full.default([8, 1584, 7, 7], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_68: "f32[8, 1584, 7, 7]" = torch.ops.aten.sub.Tensor(full_1, sigmoid_65);  full_1 = None
    mul_495: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(clone_46, sub_68);  clone_46 = sub_68 = None
    add_306: "f32[8, 1584, 7, 7]" = torch.ops.aten.add.Scalar(mul_495, 1);  mul_495 = None
    mul_496: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(sigmoid_65, add_306);  sigmoid_65 = add_306 = None
    mul_497: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(add_305, mul_496);  add_305 = mul_496 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_256: "f32[1, 1584]" = torch.ops.aten.unsqueeze.default(squeeze_165, 0);  squeeze_165 = None
    unsqueeze_257: "f32[1, 1584, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_256, 2);  unsqueeze_256 = None
    unsqueeze_258: "f32[1, 1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_257, 3);  unsqueeze_257 = None
    sum_7: "f32[1584]" = torch.ops.aten.sum.dim_IntList(mul_497, [0, 2, 3])
    sub_69: "f32[8, 1584, 7, 7]" = torch.ops.aten.sub.Tensor(cat_39, unsqueeze_258)
    mul_498: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(mul_497, sub_69);  sub_69 = None
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
    mul_503: "f32[1584]" = torch.ops.aten.mul.Tensor(squeeze_166, primals_126);  primals_126 = None
    unsqueeze_265: "f32[1, 1584]" = torch.ops.aten.unsqueeze.default(mul_503, 0);  mul_503 = None
    unsqueeze_266: "f32[1, 1584, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_265, 2);  unsqueeze_265 = None
    unsqueeze_267: "f32[1, 1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_266, 3);  unsqueeze_266 = None
    sub_70: "f32[8, 1584, 7, 7]" = torch.ops.aten.sub.Tensor(cat_39, unsqueeze_258);  cat_39 = unsqueeze_258 = None
    mul_504: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(sub_70, unsqueeze_264);  sub_70 = unsqueeze_264 = None
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
    sigmoid_66: "f32[8, 1584, 7, 7]" = torch.ops.aten.sigmoid.default(clone_45)
    full_2: "f32[8, 1584, 7, 7]" = torch.ops.aten.full.default([8, 1584, 7, 7], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_73: "f32[8, 1584, 7, 7]" = torch.ops.aten.sub.Tensor(full_2, sigmoid_66);  full_2 = None
    mul_507: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(clone_45, sub_73);  clone_45 = sub_73 = None
    add_307: "f32[8, 1584, 7, 7]" = torch.ops.aten.add.Scalar(mul_507, 1);  mul_507 = None
    mul_508: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(sigmoid_66, add_307);  sigmoid_66 = add_307 = None
    mul_509: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(cat_42, mul_508);  cat_42 = mul_508 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_268: "f32[1, 1584]" = torch.ops.aten.unsqueeze.default(squeeze_162, 0);  squeeze_162 = None
    unsqueeze_269: "f32[1, 1584, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_268, 2);  unsqueeze_268 = None
    unsqueeze_270: "f32[1, 1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_269, 3);  unsqueeze_269 = None
    sum_9: "f32[1584]" = torch.ops.aten.sum.dim_IntList(mul_509, [0, 2, 3])
    sub_74: "f32[8, 1584, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_145, unsqueeze_270)
    mul_510: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(mul_509, sub_74);  sub_74 = None
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
    mul_515: "f32[1584]" = torch.ops.aten.mul.Tensor(squeeze_163, primals_124);  primals_124 = None
    unsqueeze_277: "f32[1, 1584]" = torch.ops.aten.unsqueeze.default(mul_515, 0);  mul_515 = None
    unsqueeze_278: "f32[1, 1584, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_277, 2);  unsqueeze_277 = None
    unsqueeze_279: "f32[1, 1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_278, 3);  unsqueeze_278 = None
    sub_75: "f32[8, 1584, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_145, unsqueeze_270);  convolution_145 = unsqueeze_270 = None
    mul_516: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(sub_75, unsqueeze_276);  sub_75 = unsqueeze_276 = None
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
    unsqueeze_280: "f32[1, 264]" = torch.ops.aten.unsqueeze.default(squeeze_159, 0);  squeeze_159 = None
    unsqueeze_281: "f32[1, 264, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_280, 2);  unsqueeze_280 = None
    unsqueeze_282: "f32[1, 264, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_281, 3);  unsqueeze_281 = None
    sum_11: "f32[264]" = torch.ops.aten.sum.dim_IntList(add_308, [0, 2, 3])
    sub_78: "f32[8, 264, 7, 7]" = torch.ops.aten.sub.Tensor(cat_38, unsqueeze_282)
    mul_519: "f32[8, 264, 7, 7]" = torch.ops.aten.mul.Tensor(add_308, sub_78);  sub_78 = None
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
    mul_524: "f32[264]" = torch.ops.aten.mul.Tensor(squeeze_160, primals_122);  primals_122 = None
    unsqueeze_289: "f32[1, 264]" = torch.ops.aten.unsqueeze.default(mul_524, 0);  mul_524 = None
    unsqueeze_290: "f32[1, 264, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_289, 2);  unsqueeze_289 = None
    unsqueeze_291: "f32[1, 264, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_290, 3);  unsqueeze_290 = None
    sub_79: "f32[8, 264, 7, 7]" = torch.ops.aten.sub.Tensor(cat_38, unsqueeze_282);  cat_38 = unsqueeze_282 = None
    mul_525: "f32[8, 264, 7, 7]" = torch.ops.aten.mul.Tensor(sub_79, unsqueeze_288);  sub_79 = unsqueeze_288 = None
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
    full_3: "f32[8, 132, 1, 1]" = torch.ops.aten.full.default([8, 132, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_83: "f32[8, 132, 1, 1]" = torch.ops.aten.sub.Tensor(full_3, sigmoid_67);  full_3 = None
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
    full_4: "f32[8, 1584, 7, 7]" = torch.ops.aten.full.default([8, 1584, 7, 7], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_84: "f32[8, 1584, 7, 7]" = torch.ops.aten.sub.Tensor(full_4, sigmoid_68);  full_4 = None
    mul_535: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(clone_43, sub_84);  clone_43 = sub_84 = None
    add_311: "f32[8, 1584, 7, 7]" = torch.ops.aten.add.Scalar(mul_535, 1);  mul_535 = None
    mul_536: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(sigmoid_68, add_311);  sigmoid_68 = add_311 = None
    mul_537: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(add_310, mul_536);  add_310 = mul_536 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_292: "f32[1, 1584]" = torch.ops.aten.unsqueeze.default(squeeze_156, 0);  squeeze_156 = None
    unsqueeze_293: "f32[1, 1584, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_292, 2);  unsqueeze_292 = None
    unsqueeze_294: "f32[1, 1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_293, 3);  unsqueeze_293 = None
    sum_14: "f32[1584]" = torch.ops.aten.sum.dim_IntList(mul_537, [0, 2, 3])
    sub_85: "f32[8, 1584, 7, 7]" = torch.ops.aten.sub.Tensor(cat_37, unsqueeze_294)
    mul_538: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(mul_537, sub_85);  sub_85 = None
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
    mul_543: "f32[1584]" = torch.ops.aten.mul.Tensor(squeeze_157, primals_120);  primals_120 = None
    unsqueeze_301: "f32[1, 1584]" = torch.ops.aten.unsqueeze.default(mul_543, 0);  mul_543 = None
    unsqueeze_302: "f32[1, 1584, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_301, 2);  unsqueeze_301 = None
    unsqueeze_303: "f32[1, 1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_302, 3);  unsqueeze_302 = None
    sub_86: "f32[8, 1584, 7, 7]" = torch.ops.aten.sub.Tensor(cat_37, unsqueeze_294);  cat_37 = unsqueeze_294 = None
    mul_544: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(sub_86, unsqueeze_300);  sub_86 = unsqueeze_300 = None
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
    sigmoid_69: "f32[8, 1584, 7, 7]" = torch.ops.aten.sigmoid.default(clone_42)
    full_5: "f32[8, 1584, 7, 7]" = torch.ops.aten.full.default([8, 1584, 7, 7], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_89: "f32[8, 1584, 7, 7]" = torch.ops.aten.sub.Tensor(full_5, sigmoid_69);  full_5 = None
    mul_547: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(clone_42, sub_89);  clone_42 = sub_89 = None
    add_312: "f32[8, 1584, 7, 7]" = torch.ops.aten.add.Scalar(mul_547, 1);  mul_547 = None
    mul_548: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(sigmoid_69, add_312);  sigmoid_69 = add_312 = None
    mul_549: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(cat_44, mul_548);  cat_44 = mul_548 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_304: "f32[1, 1584]" = torch.ops.aten.unsqueeze.default(squeeze_153, 0);  squeeze_153 = None
    unsqueeze_305: "f32[1, 1584, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_304, 2);  unsqueeze_304 = None
    unsqueeze_306: "f32[1, 1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_305, 3);  unsqueeze_305 = None
    sum_16: "f32[1584]" = torch.ops.aten.sum.dim_IntList(mul_549, [0, 2, 3])
    sub_90: "f32[8, 1584, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_136, unsqueeze_306)
    mul_550: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(mul_549, sub_90);  sub_90 = None
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
    mul_555: "f32[1584]" = torch.ops.aten.mul.Tensor(squeeze_154, primals_118);  primals_118 = None
    unsqueeze_313: "f32[1, 1584]" = torch.ops.aten.unsqueeze.default(mul_555, 0);  mul_555 = None
    unsqueeze_314: "f32[1, 1584, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_313, 2);  unsqueeze_313 = None
    unsqueeze_315: "f32[1, 1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_314, 3);  unsqueeze_314 = None
    sub_91: "f32[8, 1584, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_136, unsqueeze_306);  convolution_136 = unsqueeze_306 = None
    mul_556: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(sub_91, unsqueeze_312);  sub_91 = unsqueeze_312 = None
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
    unsqueeze_316: "f32[1, 264]" = torch.ops.aten.unsqueeze.default(squeeze_150, 0);  squeeze_150 = None
    unsqueeze_317: "f32[1, 264, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_316, 2);  unsqueeze_316 = None
    unsqueeze_318: "f32[1, 264, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_317, 3);  unsqueeze_317 = None
    sum_18: "f32[264]" = torch.ops.aten.sum.dim_IntList(add_313, [0, 2, 3])
    sub_94: "f32[8, 264, 7, 7]" = torch.ops.aten.sub.Tensor(cat_36, unsqueeze_318)
    mul_559: "f32[8, 264, 7, 7]" = torch.ops.aten.mul.Tensor(add_313, sub_94);  sub_94 = None
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
    mul_564: "f32[264]" = torch.ops.aten.mul.Tensor(squeeze_151, primals_116);  primals_116 = None
    unsqueeze_325: "f32[1, 264]" = torch.ops.aten.unsqueeze.default(mul_564, 0);  mul_564 = None
    unsqueeze_326: "f32[1, 264, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_325, 2);  unsqueeze_325 = None
    unsqueeze_327: "f32[1, 264, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_326, 3);  unsqueeze_326 = None
    sub_95: "f32[8, 264, 7, 7]" = torch.ops.aten.sub.Tensor(cat_36, unsqueeze_318);  cat_36 = unsqueeze_318 = None
    mul_565: "f32[8, 264, 7, 7]" = torch.ops.aten.mul.Tensor(sub_95, unsqueeze_324);  sub_95 = unsqueeze_324 = None
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
    full_6: "f32[8, 132, 1, 1]" = torch.ops.aten.full.default([8, 132, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_99: "f32[8, 132, 1, 1]" = torch.ops.aten.sub.Tensor(full_6, sigmoid_70);  full_6 = None
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
    full_7: "f32[8, 1584, 7, 7]" = torch.ops.aten.full.default([8, 1584, 7, 7], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_100: "f32[8, 1584, 7, 7]" = torch.ops.aten.sub.Tensor(full_7, sigmoid_71);  full_7 = None
    mul_575: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(clone_40, sub_100);  clone_40 = sub_100 = None
    add_316: "f32[8, 1584, 7, 7]" = torch.ops.aten.add.Scalar(mul_575, 1);  mul_575 = None
    mul_576: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(sigmoid_71, add_316);  sigmoid_71 = add_316 = None
    mul_577: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(add_315, mul_576);  add_315 = mul_576 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_328: "f32[1, 1584]" = torch.ops.aten.unsqueeze.default(squeeze_147, 0);  squeeze_147 = None
    unsqueeze_329: "f32[1, 1584, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_328, 2);  unsqueeze_328 = None
    unsqueeze_330: "f32[1, 1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_329, 3);  unsqueeze_329 = None
    sum_21: "f32[1584]" = torch.ops.aten.sum.dim_IntList(mul_577, [0, 2, 3])
    sub_101: "f32[8, 1584, 7, 7]" = torch.ops.aten.sub.Tensor(cat_35, unsqueeze_330)
    mul_578: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(mul_577, sub_101);  sub_101 = None
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
    mul_583: "f32[1584]" = torch.ops.aten.mul.Tensor(squeeze_148, primals_114);  primals_114 = None
    unsqueeze_337: "f32[1, 1584]" = torch.ops.aten.unsqueeze.default(mul_583, 0);  mul_583 = None
    unsqueeze_338: "f32[1, 1584, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_337, 2);  unsqueeze_337 = None
    unsqueeze_339: "f32[1, 1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_338, 3);  unsqueeze_338 = None
    sub_102: "f32[8, 1584, 7, 7]" = torch.ops.aten.sub.Tensor(cat_35, unsqueeze_330);  cat_35 = unsqueeze_330 = None
    mul_584: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(sub_102, unsqueeze_336);  sub_102 = unsqueeze_336 = None
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
    sigmoid_72: "f32[8, 1584, 7, 7]" = torch.ops.aten.sigmoid.default(clone_39)
    full_8: "f32[8, 1584, 7, 7]" = torch.ops.aten.full.default([8, 1584, 7, 7], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_105: "f32[8, 1584, 7, 7]" = torch.ops.aten.sub.Tensor(full_8, sigmoid_72);  full_8 = None
    mul_587: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(clone_39, sub_105);  clone_39 = sub_105 = None
    add_317: "f32[8, 1584, 7, 7]" = torch.ops.aten.add.Scalar(mul_587, 1);  mul_587 = None
    mul_588: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(sigmoid_72, add_317);  sigmoid_72 = add_317 = None
    mul_589: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(cat_46, mul_588);  cat_46 = mul_588 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_340: "f32[1, 1584]" = torch.ops.aten.unsqueeze.default(squeeze_144, 0);  squeeze_144 = None
    unsqueeze_341: "f32[1, 1584, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_340, 2);  unsqueeze_340 = None
    unsqueeze_342: "f32[1, 1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_341, 3);  unsqueeze_341 = None
    sum_23: "f32[1584]" = torch.ops.aten.sum.dim_IntList(mul_589, [0, 2, 3])
    sub_106: "f32[8, 1584, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_127, unsqueeze_342)
    mul_590: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(mul_589, sub_106);  sub_106 = None
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
    mul_595: "f32[1584]" = torch.ops.aten.mul.Tensor(squeeze_145, primals_112);  primals_112 = None
    unsqueeze_349: "f32[1, 1584]" = torch.ops.aten.unsqueeze.default(mul_595, 0);  mul_595 = None
    unsqueeze_350: "f32[1, 1584, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_349, 2);  unsqueeze_349 = None
    unsqueeze_351: "f32[1, 1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_350, 3);  unsqueeze_350 = None
    sub_107: "f32[8, 1584, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_127, unsqueeze_342);  convolution_127 = unsqueeze_342 = None
    mul_596: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(sub_107, unsqueeze_348);  sub_107 = unsqueeze_348 = None
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
    unsqueeze_352: "f32[1, 264]" = torch.ops.aten.unsqueeze.default(squeeze_141, 0);  squeeze_141 = None
    unsqueeze_353: "f32[1, 264, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_352, 2);  unsqueeze_352 = None
    unsqueeze_354: "f32[1, 264, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_353, 3);  unsqueeze_353 = None
    sum_25: "f32[264]" = torch.ops.aten.sum.dim_IntList(add_318, [0, 2, 3])
    sub_110: "f32[8, 264, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_126, unsqueeze_354)
    mul_599: "f32[8, 264, 7, 7]" = torch.ops.aten.mul.Tensor(add_318, sub_110);  sub_110 = None
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
    mul_604: "f32[264]" = torch.ops.aten.mul.Tensor(squeeze_142, primals_110);  primals_110 = None
    unsqueeze_361: "f32[1, 264]" = torch.ops.aten.unsqueeze.default(mul_604, 0);  mul_604 = None
    unsqueeze_362: "f32[1, 264, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_361, 2);  unsqueeze_361 = None
    unsqueeze_363: "f32[1, 264, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_362, 3);  unsqueeze_362 = None
    sub_111: "f32[8, 264, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_126, unsqueeze_354);  convolution_126 = unsqueeze_354 = None
    mul_605: "f32[8, 264, 7, 7]" = torch.ops.aten.mul.Tensor(sub_111, unsqueeze_360);  sub_111 = unsqueeze_360 = None
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
    full_9: "f32[8, 80, 1, 1]" = torch.ops.aten.full.default([8, 80, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_115: "f32[8, 80, 1, 1]" = torch.ops.aten.sub.Tensor(full_9, sigmoid_73);  full_9 = None
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
    full_10: "f32[8, 960, 7, 7]" = torch.ops.aten.full.default([8, 960, 7, 7], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_116: "f32[8, 960, 7, 7]" = torch.ops.aten.sub.Tensor(full_10, sigmoid_74);  full_10 = None
    mul_615: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(clone_37, sub_116);  clone_37 = sub_116 = None
    add_321: "f32[8, 960, 7, 7]" = torch.ops.aten.add.Scalar(mul_615, 1);  mul_615 = None
    mul_616: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(sigmoid_74, add_321);  sigmoid_74 = add_321 = None
    mul_617: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(add_320, mul_616);  add_320 = mul_616 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_364: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(squeeze_138, 0);  squeeze_138 = None
    unsqueeze_365: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_364, 2);  unsqueeze_364 = None
    unsqueeze_366: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_365, 3);  unsqueeze_365 = None
    sum_28: "f32[960]" = torch.ops.aten.sum.dim_IntList(mul_617, [0, 2, 3])
    sub_117: "f32[8, 960, 7, 7]" = torch.ops.aten.sub.Tensor(cat_34, unsqueeze_366)
    mul_618: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(mul_617, sub_117);  sub_117 = None
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
    mul_623: "f32[960]" = torch.ops.aten.mul.Tensor(squeeze_139, primals_108);  primals_108 = None
    unsqueeze_373: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(mul_623, 0);  mul_623 = None
    unsqueeze_374: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_373, 2);  unsqueeze_373 = None
    unsqueeze_375: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_374, 3);  unsqueeze_374 = None
    sub_118: "f32[8, 960, 7, 7]" = torch.ops.aten.sub.Tensor(cat_34, unsqueeze_366);  cat_34 = unsqueeze_366 = None
    mul_624: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(sub_118, unsqueeze_372);  sub_118 = unsqueeze_372 = None
    sub_119: "f32[8, 960, 7, 7]" = torch.ops.aten.sub.Tensor(mul_617, mul_624);  mul_617 = mul_624 = None
    sub_120: "f32[8, 960, 7, 7]" = torch.ops.aten.sub.Tensor(sub_119, unsqueeze_369);  sub_119 = unsqueeze_369 = None
    mul_625: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(sub_120, unsqueeze_375);  sub_120 = unsqueeze_375 = None
    mul_626: "f32[960]" = torch.ops.aten.mul.Tensor(sum_29, squeeze_139);  sum_29 = squeeze_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    slice_19: "f32[8, 240, 7, 7]" = torch.ops.aten.slice.Tensor(mul_625, 1, 0, 240)
    slice_20: "f32[8, 240, 7, 7]" = torch.ops.aten.slice.Tensor(mul_625, 1, 240, 480)
    slice_21: "f32[8, 240, 7, 7]" = torch.ops.aten.slice.Tensor(mul_625, 1, 480, 720)
    slice_22: "f32[8, 240, 7, 7]" = torch.ops.aten.slice.Tensor(mul_625, 1, 720, 960);  mul_625 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_backward_31 = torch.ops.aten.convolution_backward.default(slice_22, constant_pad_nd_14, primals_107, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 240, [True, True, False]);  slice_22 = constant_pad_nd_14 = primals_107 = None
    getitem_529: "f32[8, 240, 21, 21]" = convolution_backward_31[0]
    getitem_530: "f32[240, 1, 9, 9]" = convolution_backward_31[1];  convolution_backward_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_15: "f32[8, 240, 14, 14]" = torch.ops.aten.constant_pad_nd.default(getitem_529, [-3, -4, -3, -4]);  getitem_529 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_backward_32 = torch.ops.aten.convolution_backward.default(slice_21, constant_pad_nd_13, primals_106, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 240, [True, True, False]);  slice_21 = constant_pad_nd_13 = primals_106 = None
    getitem_532: "f32[8, 240, 19, 19]" = convolution_backward_32[0]
    getitem_533: "f32[240, 1, 7, 7]" = convolution_backward_32[1];  convolution_backward_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_16: "f32[8, 240, 14, 14]" = torch.ops.aten.constant_pad_nd.default(getitem_532, [-2, -3, -2, -3]);  getitem_532 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_backward_33 = torch.ops.aten.convolution_backward.default(slice_20, constant_pad_nd_12, primals_105, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 240, [True, True, False]);  slice_20 = constant_pad_nd_12 = primals_105 = None
    getitem_535: "f32[8, 240, 17, 17]" = convolution_backward_33[0]
    getitem_536: "f32[240, 1, 5, 5]" = convolution_backward_33[1];  convolution_backward_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_17: "f32[8, 240, 14, 14]" = torch.ops.aten.constant_pad_nd.default(getitem_535, [-1, -2, -1, -2]);  getitem_535 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_backward_34 = torch.ops.aten.convolution_backward.default(slice_19, constant_pad_nd_11, primals_104, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 240, [True, True, False]);  slice_19 = constant_pad_nd_11 = primals_104 = None
    getitem_538: "f32[8, 240, 15, 15]" = convolution_backward_34[0]
    getitem_539: "f32[240, 1, 3, 3]" = convolution_backward_34[1];  convolution_backward_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_18: "f32[8, 240, 14, 14]" = torch.ops.aten.constant_pad_nd.default(getitem_538, [0, -1, 0, -1]);  getitem_538 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    cat_47: "f32[8, 960, 14, 14]" = torch.ops.aten.cat.default([constant_pad_nd_18, constant_pad_nd_17, constant_pad_nd_16, constant_pad_nd_15], 1);  constant_pad_nd_18 = constant_pad_nd_17 = constant_pad_nd_16 = constant_pad_nd_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_75: "f32[8, 960, 14, 14]" = torch.ops.aten.sigmoid.default(clone_36)
    full_11: "f32[8, 960, 14, 14]" = torch.ops.aten.full.default([8, 960, 14, 14], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_121: "f32[8, 960, 14, 14]" = torch.ops.aten.sub.Tensor(full_11, sigmoid_75);  full_11 = None
    mul_627: "f32[8, 960, 14, 14]" = torch.ops.aten.mul.Tensor(clone_36, sub_121);  clone_36 = sub_121 = None
    add_322: "f32[8, 960, 14, 14]" = torch.ops.aten.add.Scalar(mul_627, 1);  mul_627 = None
    mul_628: "f32[8, 960, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_75, add_322);  sigmoid_75 = add_322 = None
    mul_629: "f32[8, 960, 14, 14]" = torch.ops.aten.mul.Tensor(cat_47, mul_628);  cat_47 = mul_628 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_376: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(squeeze_135, 0);  squeeze_135 = None
    unsqueeze_377: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_376, 2);  unsqueeze_376 = None
    unsqueeze_378: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_377, 3);  unsqueeze_377 = None
    sum_30: "f32[960]" = torch.ops.aten.sum.dim_IntList(mul_629, [0, 2, 3])
    sub_122: "f32[8, 960, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_119, unsqueeze_378)
    mul_630: "f32[8, 960, 14, 14]" = torch.ops.aten.mul.Tensor(mul_629, sub_122);  sub_122 = None
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
    mul_635: "f32[960]" = torch.ops.aten.mul.Tensor(squeeze_136, primals_102);  primals_102 = None
    unsqueeze_385: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(mul_635, 0);  mul_635 = None
    unsqueeze_386: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_385, 2);  unsqueeze_385 = None
    unsqueeze_387: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_386, 3);  unsqueeze_386 = None
    sub_123: "f32[8, 960, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_119, unsqueeze_378);  convolution_119 = unsqueeze_378 = None
    mul_636: "f32[8, 960, 14, 14]" = torch.ops.aten.mul.Tensor(sub_123, unsqueeze_384);  sub_123 = unsqueeze_384 = None
    sub_124: "f32[8, 960, 14, 14]" = torch.ops.aten.sub.Tensor(mul_629, mul_636);  mul_629 = mul_636 = None
    sub_125: "f32[8, 960, 14, 14]" = torch.ops.aten.sub.Tensor(sub_124, unsqueeze_381);  sub_124 = unsqueeze_381 = None
    mul_637: "f32[8, 960, 14, 14]" = torch.ops.aten.mul.Tensor(sub_125, unsqueeze_387);  sub_125 = unsqueeze_387 = None
    mul_638: "f32[960]" = torch.ops.aten.mul.Tensor(sum_31, squeeze_136);  sum_31 = squeeze_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_35 = torch.ops.aten.convolution_backward.default(mul_637, add_235, primals_264, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_637 = add_235 = primals_264 = None
    getitem_541: "f32[8, 160, 14, 14]" = convolution_backward_35[0]
    getitem_542: "f32[960, 160, 1, 1]" = convolution_backward_35[1];  convolution_backward_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_388: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(squeeze_132, 0);  squeeze_132 = None
    unsqueeze_389: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_388, 2);  unsqueeze_388 = None
    unsqueeze_390: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_389, 3);  unsqueeze_389 = None
    sum_32: "f32[160]" = torch.ops.aten.sum.dim_IntList(getitem_541, [0, 2, 3])
    sub_126: "f32[8, 160, 14, 14]" = torch.ops.aten.sub.Tensor(cat_33, unsqueeze_390)
    mul_639: "f32[8, 160, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_541, sub_126);  sub_126 = None
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
    mul_644: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_133, primals_100);  primals_100 = None
    unsqueeze_397: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_644, 0);  mul_644 = None
    unsqueeze_398: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_397, 2);  unsqueeze_397 = None
    unsqueeze_399: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_398, 3);  unsqueeze_398 = None
    sub_127: "f32[8, 160, 14, 14]" = torch.ops.aten.sub.Tensor(cat_33, unsqueeze_390);  cat_33 = unsqueeze_390 = None
    mul_645: "f32[8, 160, 14, 14]" = torch.ops.aten.mul.Tensor(sub_127, unsqueeze_396);  sub_127 = unsqueeze_396 = None
    sub_128: "f32[8, 160, 14, 14]" = torch.ops.aten.sub.Tensor(getitem_541, mul_645);  mul_645 = None
    sub_129: "f32[8, 160, 14, 14]" = torch.ops.aten.sub.Tensor(sub_128, unsqueeze_393);  sub_128 = unsqueeze_393 = None
    mul_646: "f32[8, 160, 14, 14]" = torch.ops.aten.mul.Tensor(sub_129, unsqueeze_399);  sub_129 = unsqueeze_399 = None
    mul_647: "f32[160]" = torch.ops.aten.mul.Tensor(sum_33, squeeze_133);  sum_33 = squeeze_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    slice_23: "f32[8, 80, 14, 14]" = torch.ops.aten.slice.Tensor(mul_646, 1, 0, 80)
    slice_24: "f32[8, 80, 14, 14]" = torch.ops.aten.slice.Tensor(mul_646, 1, 80, 160);  mul_646 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_backward_36 = torch.ops.aten.convolution_backward.default(slice_24, getitem_321, primals_263, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_24 = getitem_321 = primals_263 = None
    getitem_544: "f32[8, 240, 14, 14]" = convolution_backward_36[0]
    getitem_545: "f32[80, 240, 1, 1]" = convolution_backward_36[1];  convolution_backward_36 = None
    convolution_backward_37 = torch.ops.aten.convolution_backward.default(slice_23, getitem_320, primals_262, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_23 = getitem_320 = primals_262 = None
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
    convolution_backward_38 = torch.ops.aten.convolution_backward.default(mul_651, mul_354, primals_260, [480], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_651 = mul_354 = primals_260 = None
    getitem_550: "f32[8, 80, 1, 1]" = convolution_backward_38[0]
    getitem_551: "f32[480, 80, 1, 1]" = convolution_backward_38[1]
    getitem_552: "f32[480]" = convolution_backward_38[2];  convolution_backward_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_76: "f32[8, 80, 1, 1]" = torch.ops.aten.sigmoid.default(clone_35)
    full_12: "f32[8, 80, 1, 1]" = torch.ops.aten.full.default([8, 80, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_131: "f32[8, 80, 1, 1]" = torch.ops.aten.sub.Tensor(full_12, sigmoid_76);  full_12 = None
    mul_652: "f32[8, 80, 1, 1]" = torch.ops.aten.mul.Tensor(clone_35, sub_131);  clone_35 = sub_131 = None
    add_323: "f32[8, 80, 1, 1]" = torch.ops.aten.add.Scalar(mul_652, 1);  mul_652 = None
    mul_653: "f32[8, 80, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_76, add_323);  sigmoid_76 = add_323 = None
    mul_654: "f32[8, 80, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_550, mul_653);  getitem_550 = mul_653 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_39 = torch.ops.aten.convolution_backward.default(mul_654, mean_11, primals_258, [80], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_654 = mean_11 = primals_258 = None
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
    full_13: "f32[8, 480, 14, 14]" = torch.ops.aten.full.default([8, 480, 14, 14], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_132: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(full_13, sigmoid_77);  full_13 = None
    mul_655: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(clone_34, sub_132);  clone_34 = sub_132 = None
    add_325: "f32[8, 480, 14, 14]" = torch.ops.aten.add.Scalar(mul_655, 1);  mul_655 = None
    mul_656: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_77, add_325);  sigmoid_77 = add_325 = None
    mul_657: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_324, mul_656);  add_324 = mul_656 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_400: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(squeeze_129, 0);  squeeze_129 = None
    unsqueeze_401: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_400, 2);  unsqueeze_400 = None
    unsqueeze_402: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_401, 3);  unsqueeze_401 = None
    sum_35: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_657, [0, 2, 3])
    sub_133: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(cat_32, unsqueeze_402)
    mul_658: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_657, sub_133);  sub_133 = None
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
    mul_663: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_130, primals_98);  primals_98 = None
    unsqueeze_409: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_663, 0);  mul_663 = None
    unsqueeze_410: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_409, 2);  unsqueeze_409 = None
    unsqueeze_411: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_410, 3);  unsqueeze_410 = None
    sub_134: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(cat_32, unsqueeze_402);  cat_32 = unsqueeze_402 = None
    mul_664: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_134, unsqueeze_408);  sub_134 = unsqueeze_408 = None
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
    convolution_backward_40 = torch.ops.aten.convolution_backward.default(slice_28, getitem_317, primals_257, [0], [1, 1], [4, 4], [1, 1], False, [0, 0], 120, [True, True, False]);  slice_28 = getitem_317 = primals_257 = None
    getitem_556: "f32[8, 120, 14, 14]" = convolution_backward_40[0]
    getitem_557: "f32[120, 1, 9, 9]" = convolution_backward_40[1];  convolution_backward_40 = None
    convolution_backward_41 = torch.ops.aten.convolution_backward.default(slice_27, getitem_312, primals_256, [0], [1, 1], [3, 3], [1, 1], False, [0, 0], 120, [True, True, False]);  slice_27 = getitem_312 = primals_256 = None
    getitem_559: "f32[8, 120, 14, 14]" = convolution_backward_41[0]
    getitem_560: "f32[120, 1, 7, 7]" = convolution_backward_41[1];  convolution_backward_41 = None
    convolution_backward_42 = torch.ops.aten.convolution_backward.default(slice_26, getitem_307, primals_255, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 120, [True, True, False]);  slice_26 = getitem_307 = primals_255 = None
    getitem_562: "f32[8, 120, 14, 14]" = convolution_backward_42[0]
    getitem_563: "f32[120, 1, 5, 5]" = convolution_backward_42[1];  convolution_backward_42 = None
    convolution_backward_43 = torch.ops.aten.convolution_backward.default(slice_25, getitem_302, primals_254, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 120, [True, True, False]);  slice_25 = getitem_302 = primals_254 = None
    getitem_565: "f32[8, 120, 14, 14]" = convolution_backward_43[0]
    getitem_566: "f32[120, 1, 3, 3]" = convolution_backward_43[1];  convolution_backward_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    cat_49: "f32[8, 480, 14, 14]" = torch.ops.aten.cat.default([getitem_565, getitem_562, getitem_559, getitem_556], 1);  getitem_565 = getitem_562 = getitem_559 = getitem_556 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_78: "f32[8, 480, 14, 14]" = torch.ops.aten.sigmoid.default(clone_33)
    full_14: "f32[8, 480, 14, 14]" = torch.ops.aten.full.default([8, 480, 14, 14], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_137: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(full_14, sigmoid_78);  full_14 = None
    mul_667: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(clone_33, sub_137);  clone_33 = sub_137 = None
    add_326: "f32[8, 480, 14, 14]" = torch.ops.aten.add.Scalar(mul_667, 1);  mul_667 = None
    mul_668: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_78, add_326);  sigmoid_78 = add_326 = None
    mul_669: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(cat_49, mul_668);  cat_49 = mul_668 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_412: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(squeeze_126, 0);  squeeze_126 = None
    unsqueeze_413: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_412, 2);  unsqueeze_412 = None
    unsqueeze_414: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_413, 3);  unsqueeze_413 = None
    sum_37: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_669, [0, 2, 3])
    sub_138: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(cat_31, unsqueeze_414)
    mul_670: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_669, sub_138);  sub_138 = None
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
    mul_675: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_127, primals_96);  primals_96 = None
    unsqueeze_421: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_675, 0);  mul_675 = None
    unsqueeze_422: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_421, 2);  unsqueeze_421 = None
    unsqueeze_423: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_422, 3);  unsqueeze_422 = None
    sub_139: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(cat_31, unsqueeze_414);  cat_31 = unsqueeze_414 = None
    mul_676: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_139, unsqueeze_420);  sub_139 = unsqueeze_420 = None
    sub_140: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(mul_669, mul_676);  mul_669 = mul_676 = None
    sub_141: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(sub_140, unsqueeze_417);  sub_140 = unsqueeze_417 = None
    mul_677: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_141, unsqueeze_423);  sub_141 = unsqueeze_423 = None
    mul_678: "f32[480]" = torch.ops.aten.mul.Tensor(sum_38, squeeze_127);  sum_38 = squeeze_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    slice_29: "f32[8, 240, 14, 14]" = torch.ops.aten.slice.Tensor(mul_677, 1, 0, 240)
    slice_30: "f32[8, 240, 14, 14]" = torch.ops.aten.slice.Tensor(mul_677, 1, 240, 480);  mul_677 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_backward_44 = torch.ops.aten.convolution_backward.default(slice_30, getitem_295, primals_253, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_30 = getitem_295 = primals_253 = None
    getitem_568: "f32[8, 80, 14, 14]" = convolution_backward_44[0]
    getitem_569: "f32[240, 80, 1, 1]" = convolution_backward_44[1];  convolution_backward_44 = None
    convolution_backward_45 = torch.ops.aten.convolution_backward.default(slice_29, getitem_294, primals_252, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_29 = getitem_294 = primals_252 = None
    getitem_571: "f32[8, 80, 14, 14]" = convolution_backward_45[0]
    getitem_572: "f32[240, 80, 1, 1]" = convolution_backward_45[1];  convolution_backward_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    cat_50: "f32[8, 160, 14, 14]" = torch.ops.aten.cat.default([getitem_571, getitem_568], 1);  getitem_571 = getitem_568 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    add_327: "f32[8, 160, 14, 14]" = torch.ops.aten.add.Tensor(getitem_541, cat_50);  getitem_541 = cat_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_424: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(squeeze_123, 0);  squeeze_123 = None
    unsqueeze_425: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_424, 2);  unsqueeze_424 = None
    unsqueeze_426: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_425, 3);  unsqueeze_425 = None
    sum_39: "f32[160]" = torch.ops.aten.sum.dim_IntList(add_327, [0, 2, 3])
    sub_142: "f32[8, 160, 14, 14]" = torch.ops.aten.sub.Tensor(cat_30, unsqueeze_426)
    mul_679: "f32[8, 160, 14, 14]" = torch.ops.aten.mul.Tensor(add_327, sub_142);  sub_142 = None
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
    mul_684: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_124, primals_94);  primals_94 = None
    unsqueeze_433: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_684, 0);  mul_684 = None
    unsqueeze_434: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_433, 2);  unsqueeze_433 = None
    unsqueeze_435: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_434, 3);  unsqueeze_434 = None
    sub_143: "f32[8, 160, 14, 14]" = torch.ops.aten.sub.Tensor(cat_30, unsqueeze_426);  cat_30 = unsqueeze_426 = None
    mul_685: "f32[8, 160, 14, 14]" = torch.ops.aten.mul.Tensor(sub_143, unsqueeze_432);  sub_143 = unsqueeze_432 = None
    sub_144: "f32[8, 160, 14, 14]" = torch.ops.aten.sub.Tensor(add_327, mul_685);  mul_685 = None
    sub_145: "f32[8, 160, 14, 14]" = torch.ops.aten.sub.Tensor(sub_144, unsqueeze_429);  sub_144 = unsqueeze_429 = None
    mul_686: "f32[8, 160, 14, 14]" = torch.ops.aten.mul.Tensor(sub_145, unsqueeze_435);  sub_145 = unsqueeze_435 = None
    mul_687: "f32[160]" = torch.ops.aten.mul.Tensor(sum_40, squeeze_124);  sum_40 = squeeze_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    slice_31: "f32[8, 80, 14, 14]" = torch.ops.aten.slice.Tensor(mul_686, 1, 0, 80)
    slice_32: "f32[8, 80, 14, 14]" = torch.ops.aten.slice.Tensor(mul_686, 1, 80, 160);  mul_686 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_backward_46 = torch.ops.aten.convolution_backward.default(slice_32, getitem_291, primals_251, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_32 = getitem_291 = primals_251 = None
    getitem_574: "f32[8, 240, 14, 14]" = convolution_backward_46[0]
    getitem_575: "f32[80, 240, 1, 1]" = convolution_backward_46[1];  convolution_backward_46 = None
    convolution_backward_47 = torch.ops.aten.convolution_backward.default(slice_31, getitem_290, primals_250, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_31 = getitem_290 = primals_250 = None
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
    convolution_backward_48 = torch.ops.aten.convolution_backward.default(mul_691, mul_329, primals_248, [480], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_691 = mul_329 = primals_248 = None
    getitem_580: "f32[8, 80, 1, 1]" = convolution_backward_48[0]
    getitem_581: "f32[480, 80, 1, 1]" = convolution_backward_48[1]
    getitem_582: "f32[480]" = convolution_backward_48[2];  convolution_backward_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_79: "f32[8, 80, 1, 1]" = torch.ops.aten.sigmoid.default(clone_32)
    full_15: "f32[8, 80, 1, 1]" = torch.ops.aten.full.default([8, 80, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_147: "f32[8, 80, 1, 1]" = torch.ops.aten.sub.Tensor(full_15, sigmoid_79);  full_15 = None
    mul_692: "f32[8, 80, 1, 1]" = torch.ops.aten.mul.Tensor(clone_32, sub_147);  clone_32 = sub_147 = None
    add_328: "f32[8, 80, 1, 1]" = torch.ops.aten.add.Scalar(mul_692, 1);  mul_692 = None
    mul_693: "f32[8, 80, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_79, add_328);  sigmoid_79 = add_328 = None
    mul_694: "f32[8, 80, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_580, mul_693);  getitem_580 = mul_693 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_49 = torch.ops.aten.convolution_backward.default(mul_694, mean_10, primals_246, [80], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_694 = mean_10 = primals_246 = None
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
    full_16: "f32[8, 480, 14, 14]" = torch.ops.aten.full.default([8, 480, 14, 14], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_148: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(full_16, sigmoid_80);  full_16 = None
    mul_695: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(clone_31, sub_148);  clone_31 = sub_148 = None
    add_330: "f32[8, 480, 14, 14]" = torch.ops.aten.add.Scalar(mul_695, 1);  mul_695 = None
    mul_696: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_80, add_330);  sigmoid_80 = add_330 = None
    mul_697: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_329, mul_696);  add_329 = mul_696 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_436: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(squeeze_120, 0);  squeeze_120 = None
    unsqueeze_437: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_436, 2);  unsqueeze_436 = None
    unsqueeze_438: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_437, 3);  unsqueeze_437 = None
    sum_42: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_697, [0, 2, 3])
    sub_149: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(cat_29, unsqueeze_438)
    mul_698: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_697, sub_149);  sub_149 = None
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
    mul_703: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_121, primals_92);  primals_92 = None
    unsqueeze_445: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_703, 0);  mul_703 = None
    unsqueeze_446: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_445, 2);  unsqueeze_445 = None
    unsqueeze_447: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_446, 3);  unsqueeze_446 = None
    sub_150: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(cat_29, unsqueeze_438);  cat_29 = unsqueeze_438 = None
    mul_704: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_150, unsqueeze_444);  sub_150 = unsqueeze_444 = None
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
    convolution_backward_50 = torch.ops.aten.convolution_backward.default(slice_36, getitem_287, primals_245, [0], [1, 1], [4, 4], [1, 1], False, [0, 0], 120, [True, True, False]);  slice_36 = getitem_287 = primals_245 = None
    getitem_586: "f32[8, 120, 14, 14]" = convolution_backward_50[0]
    getitem_587: "f32[120, 1, 9, 9]" = convolution_backward_50[1];  convolution_backward_50 = None
    convolution_backward_51 = torch.ops.aten.convolution_backward.default(slice_35, getitem_282, primals_244, [0], [1, 1], [3, 3], [1, 1], False, [0, 0], 120, [True, True, False]);  slice_35 = getitem_282 = primals_244 = None
    getitem_589: "f32[8, 120, 14, 14]" = convolution_backward_51[0]
    getitem_590: "f32[120, 1, 7, 7]" = convolution_backward_51[1];  convolution_backward_51 = None
    convolution_backward_52 = torch.ops.aten.convolution_backward.default(slice_34, getitem_277, primals_243, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 120, [True, True, False]);  slice_34 = getitem_277 = primals_243 = None
    getitem_592: "f32[8, 120, 14, 14]" = convolution_backward_52[0]
    getitem_593: "f32[120, 1, 5, 5]" = convolution_backward_52[1];  convolution_backward_52 = None
    convolution_backward_53 = torch.ops.aten.convolution_backward.default(slice_33, getitem_272, primals_242, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 120, [True, True, False]);  slice_33 = getitem_272 = primals_242 = None
    getitem_595: "f32[8, 120, 14, 14]" = convolution_backward_53[0]
    getitem_596: "f32[120, 1, 3, 3]" = convolution_backward_53[1];  convolution_backward_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    cat_52: "f32[8, 480, 14, 14]" = torch.ops.aten.cat.default([getitem_595, getitem_592, getitem_589, getitem_586], 1);  getitem_595 = getitem_592 = getitem_589 = getitem_586 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_81: "f32[8, 480, 14, 14]" = torch.ops.aten.sigmoid.default(clone_30)
    full_17: "f32[8, 480, 14, 14]" = torch.ops.aten.full.default([8, 480, 14, 14], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_153: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(full_17, sigmoid_81);  full_17 = None
    mul_707: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(clone_30, sub_153);  clone_30 = sub_153 = None
    add_331: "f32[8, 480, 14, 14]" = torch.ops.aten.add.Scalar(mul_707, 1);  mul_707 = None
    mul_708: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_81, add_331);  sigmoid_81 = add_331 = None
    mul_709: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(cat_52, mul_708);  cat_52 = mul_708 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_448: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(squeeze_117, 0);  squeeze_117 = None
    unsqueeze_449: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_448, 2);  unsqueeze_448 = None
    unsqueeze_450: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_449, 3);  unsqueeze_449 = None
    sum_44: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_709, [0, 2, 3])
    sub_154: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(cat_28, unsqueeze_450)
    mul_710: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_709, sub_154);  sub_154 = None
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
    mul_715: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_118, primals_90);  primals_90 = None
    unsqueeze_457: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_715, 0);  mul_715 = None
    unsqueeze_458: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_457, 2);  unsqueeze_457 = None
    unsqueeze_459: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_458, 3);  unsqueeze_458 = None
    sub_155: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(cat_28, unsqueeze_450);  cat_28 = unsqueeze_450 = None
    mul_716: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_155, unsqueeze_456);  sub_155 = unsqueeze_456 = None
    sub_156: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(mul_709, mul_716);  mul_709 = mul_716 = None
    sub_157: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(sub_156, unsqueeze_453);  sub_156 = unsqueeze_453 = None
    mul_717: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_157, unsqueeze_459);  sub_157 = unsqueeze_459 = None
    mul_718: "f32[480]" = torch.ops.aten.mul.Tensor(sum_45, squeeze_118);  sum_45 = squeeze_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    slice_37: "f32[8, 240, 14, 14]" = torch.ops.aten.slice.Tensor(mul_717, 1, 0, 240)
    slice_38: "f32[8, 240, 14, 14]" = torch.ops.aten.slice.Tensor(mul_717, 1, 240, 480);  mul_717 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_backward_54 = torch.ops.aten.convolution_backward.default(slice_38, getitem_265, primals_241, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_38 = getitem_265 = primals_241 = None
    getitem_598: "f32[8, 80, 14, 14]" = convolution_backward_54[0]
    getitem_599: "f32[240, 80, 1, 1]" = convolution_backward_54[1];  convolution_backward_54 = None
    convolution_backward_55 = torch.ops.aten.convolution_backward.default(slice_37, getitem_264, primals_240, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_37 = getitem_264 = primals_240 = None
    getitem_601: "f32[8, 80, 14, 14]" = convolution_backward_55[0]
    getitem_602: "f32[240, 80, 1, 1]" = convolution_backward_55[1];  convolution_backward_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    cat_53: "f32[8, 160, 14, 14]" = torch.ops.aten.cat.default([getitem_601, getitem_598], 1);  getitem_601 = getitem_598 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    add_332: "f32[8, 160, 14, 14]" = torch.ops.aten.add.Tensor(add_327, cat_53);  add_327 = cat_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_460: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(squeeze_114, 0);  squeeze_114 = None
    unsqueeze_461: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_460, 2);  unsqueeze_460 = None
    unsqueeze_462: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_461, 3);  unsqueeze_461 = None
    sum_46: "f32[160]" = torch.ops.aten.sum.dim_IntList(add_332, [0, 2, 3])
    sub_158: "f32[8, 160, 14, 14]" = torch.ops.aten.sub.Tensor(cat_27, unsqueeze_462)
    mul_719: "f32[8, 160, 14, 14]" = torch.ops.aten.mul.Tensor(add_332, sub_158);  sub_158 = None
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
    mul_724: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_115, primals_88);  primals_88 = None
    unsqueeze_469: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_724, 0);  mul_724 = None
    unsqueeze_470: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_469, 2);  unsqueeze_469 = None
    unsqueeze_471: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_470, 3);  unsqueeze_470 = None
    sub_159: "f32[8, 160, 14, 14]" = torch.ops.aten.sub.Tensor(cat_27, unsqueeze_462);  cat_27 = unsqueeze_462 = None
    mul_725: "f32[8, 160, 14, 14]" = torch.ops.aten.mul.Tensor(sub_159, unsqueeze_468);  sub_159 = unsqueeze_468 = None
    sub_160: "f32[8, 160, 14, 14]" = torch.ops.aten.sub.Tensor(add_332, mul_725);  mul_725 = None
    sub_161: "f32[8, 160, 14, 14]" = torch.ops.aten.sub.Tensor(sub_160, unsqueeze_465);  sub_160 = unsqueeze_465 = None
    mul_726: "f32[8, 160, 14, 14]" = torch.ops.aten.mul.Tensor(sub_161, unsqueeze_471);  sub_161 = unsqueeze_471 = None
    mul_727: "f32[160]" = torch.ops.aten.mul.Tensor(sum_47, squeeze_115);  sum_47 = squeeze_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    slice_39: "f32[8, 80, 14, 14]" = torch.ops.aten.slice.Tensor(mul_726, 1, 0, 80)
    slice_40: "f32[8, 80, 14, 14]" = torch.ops.aten.slice.Tensor(mul_726, 1, 80, 160);  mul_726 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_backward_56 = torch.ops.aten.convolution_backward.default(slice_40, getitem_261, primals_239, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_40 = getitem_261 = primals_239 = None
    getitem_604: "f32[8, 240, 14, 14]" = convolution_backward_56[0]
    getitem_605: "f32[80, 240, 1, 1]" = convolution_backward_56[1];  convolution_backward_56 = None
    convolution_backward_57 = torch.ops.aten.convolution_backward.default(slice_39, getitem_260, primals_238, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_39 = getitem_260 = primals_238 = None
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
    convolution_backward_58 = torch.ops.aten.convolution_backward.default(mul_731, mul_304, primals_236, [480], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_731 = mul_304 = primals_236 = None
    getitem_610: "f32[8, 80, 1, 1]" = convolution_backward_58[0]
    getitem_611: "f32[480, 80, 1, 1]" = convolution_backward_58[1]
    getitem_612: "f32[480]" = convolution_backward_58[2];  convolution_backward_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_82: "f32[8, 80, 1, 1]" = torch.ops.aten.sigmoid.default(clone_29)
    full_18: "f32[8, 80, 1, 1]" = torch.ops.aten.full.default([8, 80, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_163: "f32[8, 80, 1, 1]" = torch.ops.aten.sub.Tensor(full_18, sigmoid_82);  full_18 = None
    mul_732: "f32[8, 80, 1, 1]" = torch.ops.aten.mul.Tensor(clone_29, sub_163);  clone_29 = sub_163 = None
    add_333: "f32[8, 80, 1, 1]" = torch.ops.aten.add.Scalar(mul_732, 1);  mul_732 = None
    mul_733: "f32[8, 80, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_82, add_333);  sigmoid_82 = add_333 = None
    mul_734: "f32[8, 80, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_610, mul_733);  getitem_610 = mul_733 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_59 = torch.ops.aten.convolution_backward.default(mul_734, mean_9, primals_234, [80], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_734 = mean_9 = primals_234 = None
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
    full_19: "f32[8, 480, 14, 14]" = torch.ops.aten.full.default([8, 480, 14, 14], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_164: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(full_19, sigmoid_83);  full_19 = None
    mul_735: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(clone_28, sub_164);  clone_28 = sub_164 = None
    add_335: "f32[8, 480, 14, 14]" = torch.ops.aten.add.Scalar(mul_735, 1);  mul_735 = None
    mul_736: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_83, add_335);  sigmoid_83 = add_335 = None
    mul_737: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_334, mul_736);  add_334 = mul_736 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_472: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(squeeze_111, 0);  squeeze_111 = None
    unsqueeze_473: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_472, 2);  unsqueeze_472 = None
    unsqueeze_474: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_473, 3);  unsqueeze_473 = None
    sum_49: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_737, [0, 2, 3])
    sub_165: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(cat_26, unsqueeze_474)
    mul_738: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_737, sub_165);  sub_165 = None
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
    mul_743: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_112, primals_86);  primals_86 = None
    unsqueeze_481: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_743, 0);  mul_743 = None
    unsqueeze_482: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_481, 2);  unsqueeze_481 = None
    unsqueeze_483: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_482, 3);  unsqueeze_482 = None
    sub_166: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(cat_26, unsqueeze_474);  cat_26 = unsqueeze_474 = None
    mul_744: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_166, unsqueeze_480);  sub_166 = unsqueeze_480 = None
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
    convolution_backward_60 = torch.ops.aten.convolution_backward.default(slice_44, getitem_257, primals_233, [0], [1, 1], [4, 4], [1, 1], False, [0, 0], 120, [True, True, False]);  slice_44 = getitem_257 = primals_233 = None
    getitem_616: "f32[8, 120, 14, 14]" = convolution_backward_60[0]
    getitem_617: "f32[120, 1, 9, 9]" = convolution_backward_60[1];  convolution_backward_60 = None
    convolution_backward_61 = torch.ops.aten.convolution_backward.default(slice_43, getitem_252, primals_232, [0], [1, 1], [3, 3], [1, 1], False, [0, 0], 120, [True, True, False]);  slice_43 = getitem_252 = primals_232 = None
    getitem_619: "f32[8, 120, 14, 14]" = convolution_backward_61[0]
    getitem_620: "f32[120, 1, 7, 7]" = convolution_backward_61[1];  convolution_backward_61 = None
    convolution_backward_62 = torch.ops.aten.convolution_backward.default(slice_42, getitem_247, primals_231, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 120, [True, True, False]);  slice_42 = getitem_247 = primals_231 = None
    getitem_622: "f32[8, 120, 14, 14]" = convolution_backward_62[0]
    getitem_623: "f32[120, 1, 5, 5]" = convolution_backward_62[1];  convolution_backward_62 = None
    convolution_backward_63 = torch.ops.aten.convolution_backward.default(slice_41, getitem_242, primals_230, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 120, [True, True, False]);  slice_41 = getitem_242 = primals_230 = None
    getitem_625: "f32[8, 120, 14, 14]" = convolution_backward_63[0]
    getitem_626: "f32[120, 1, 3, 3]" = convolution_backward_63[1];  convolution_backward_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    cat_55: "f32[8, 480, 14, 14]" = torch.ops.aten.cat.default([getitem_625, getitem_622, getitem_619, getitem_616], 1);  getitem_625 = getitem_622 = getitem_619 = getitem_616 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_84: "f32[8, 480, 14, 14]" = torch.ops.aten.sigmoid.default(clone_27)
    full_20: "f32[8, 480, 14, 14]" = torch.ops.aten.full.default([8, 480, 14, 14], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_169: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(full_20, sigmoid_84);  full_20 = None
    mul_747: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(clone_27, sub_169);  clone_27 = sub_169 = None
    add_336: "f32[8, 480, 14, 14]" = torch.ops.aten.add.Scalar(mul_747, 1);  mul_747 = None
    mul_748: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_84, add_336);  sigmoid_84 = add_336 = None
    mul_749: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(cat_55, mul_748);  cat_55 = mul_748 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_484: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(squeeze_108, 0);  squeeze_108 = None
    unsqueeze_485: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_484, 2);  unsqueeze_484 = None
    unsqueeze_486: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_485, 3);  unsqueeze_485 = None
    sum_51: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_749, [0, 2, 3])
    sub_170: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(cat_25, unsqueeze_486)
    mul_750: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_749, sub_170);  sub_170 = None
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
    mul_755: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_109, primals_84);  primals_84 = None
    unsqueeze_493: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_755, 0);  mul_755 = None
    unsqueeze_494: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_493, 2);  unsqueeze_493 = None
    unsqueeze_495: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_494, 3);  unsqueeze_494 = None
    sub_171: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(cat_25, unsqueeze_486);  cat_25 = unsqueeze_486 = None
    mul_756: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_171, unsqueeze_492);  sub_171 = unsqueeze_492 = None
    sub_172: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(mul_749, mul_756);  mul_749 = mul_756 = None
    sub_173: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(sub_172, unsqueeze_489);  sub_172 = unsqueeze_489 = None
    mul_757: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_173, unsqueeze_495);  sub_173 = unsqueeze_495 = None
    mul_758: "f32[480]" = torch.ops.aten.mul.Tensor(sum_52, squeeze_109);  sum_52 = squeeze_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    slice_45: "f32[8, 240, 14, 14]" = torch.ops.aten.slice.Tensor(mul_757, 1, 0, 240)
    slice_46: "f32[8, 240, 14, 14]" = torch.ops.aten.slice.Tensor(mul_757, 1, 240, 480);  mul_757 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_backward_64 = torch.ops.aten.convolution_backward.default(slice_46, getitem_235, primals_229, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_46 = getitem_235 = primals_229 = None
    getitem_628: "f32[8, 80, 14, 14]" = convolution_backward_64[0]
    getitem_629: "f32[240, 80, 1, 1]" = convolution_backward_64[1];  convolution_backward_64 = None
    convolution_backward_65 = torch.ops.aten.convolution_backward.default(slice_45, getitem_234, primals_228, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_45 = getitem_234 = primals_228 = None
    getitem_631: "f32[8, 80, 14, 14]" = convolution_backward_65[0]
    getitem_632: "f32[240, 80, 1, 1]" = convolution_backward_65[1];  convolution_backward_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    cat_56: "f32[8, 160, 14, 14]" = torch.ops.aten.cat.default([getitem_631, getitem_628], 1);  getitem_631 = getitem_628 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    add_337: "f32[8, 160, 14, 14]" = torch.ops.aten.add.Tensor(add_332, cat_56);  add_332 = cat_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_496: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(squeeze_105, 0);  squeeze_105 = None
    unsqueeze_497: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_496, 2);  unsqueeze_496 = None
    unsqueeze_498: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_497, 3);  unsqueeze_497 = None
    sum_53: "f32[160]" = torch.ops.aten.sum.dim_IntList(add_337, [0, 2, 3])
    sub_174: "f32[8, 160, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_88, unsqueeze_498)
    mul_759: "f32[8, 160, 14, 14]" = torch.ops.aten.mul.Tensor(add_337, sub_174);  sub_174 = None
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
    mul_764: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_106, primals_82);  primals_82 = None
    unsqueeze_505: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_764, 0);  mul_764 = None
    unsqueeze_506: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_505, 2);  unsqueeze_505 = None
    unsqueeze_507: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_506, 3);  unsqueeze_506 = None
    sub_175: "f32[8, 160, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_88, unsqueeze_498);  convolution_88 = unsqueeze_498 = None
    mul_765: "f32[8, 160, 14, 14]" = torch.ops.aten.mul.Tensor(sub_175, unsqueeze_504);  sub_175 = unsqueeze_504 = None
    sub_176: "f32[8, 160, 14, 14]" = torch.ops.aten.sub.Tensor(add_337, mul_765);  add_337 = mul_765 = None
    sub_177: "f32[8, 160, 14, 14]" = torch.ops.aten.sub.Tensor(sub_176, unsqueeze_501);  sub_176 = unsqueeze_501 = None
    mul_766: "f32[8, 160, 14, 14]" = torch.ops.aten.mul.Tensor(sub_177, unsqueeze_507);  sub_177 = unsqueeze_507 = None
    mul_767: "f32[160]" = torch.ops.aten.mul.Tensor(sum_54, squeeze_106);  sum_54 = squeeze_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_66 = torch.ops.aten.convolution_backward.default(mul_766, mul_280, primals_227, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_766 = mul_280 = primals_227 = None
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
    convolution_backward_67 = torch.ops.aten.convolution_backward.default(mul_771, mul_279, primals_225, [624], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_771 = mul_279 = primals_225 = None
    getitem_637: "f32[8, 52, 1, 1]" = convolution_backward_67[0]
    getitem_638: "f32[624, 52, 1, 1]" = convolution_backward_67[1]
    getitem_639: "f32[624]" = convolution_backward_67[2];  convolution_backward_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_85: "f32[8, 52, 1, 1]" = torch.ops.aten.sigmoid.default(clone_26)
    full_21: "f32[8, 52, 1, 1]" = torch.ops.aten.full.default([8, 52, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_179: "f32[8, 52, 1, 1]" = torch.ops.aten.sub.Tensor(full_21, sigmoid_85);  full_21 = None
    mul_772: "f32[8, 52, 1, 1]" = torch.ops.aten.mul.Tensor(clone_26, sub_179);  clone_26 = sub_179 = None
    add_338: "f32[8, 52, 1, 1]" = torch.ops.aten.add.Scalar(mul_772, 1);  mul_772 = None
    mul_773: "f32[8, 52, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_85, add_338);  sigmoid_85 = add_338 = None
    mul_774: "f32[8, 52, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_637, mul_773);  getitem_637 = mul_773 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_68 = torch.ops.aten.convolution_backward.default(mul_774, mean_8, primals_223, [52], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_774 = mean_8 = primals_223 = None
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
    full_22: "f32[8, 624, 14, 14]" = torch.ops.aten.full.default([8, 624, 14, 14], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_180: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(full_22, sigmoid_86);  full_22 = None
    mul_775: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(clone_25, sub_180);  clone_25 = sub_180 = None
    add_340: "f32[8, 624, 14, 14]" = torch.ops.aten.add.Scalar(mul_775, 1);  mul_775 = None
    mul_776: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_86, add_340);  sigmoid_86 = add_340 = None
    mul_777: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(add_339, mul_776);  add_339 = mul_776 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_508: "f32[1, 624]" = torch.ops.aten.unsqueeze.default(squeeze_102, 0);  squeeze_102 = None
    unsqueeze_509: "f32[1, 624, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_508, 2);  unsqueeze_508 = None
    unsqueeze_510: "f32[1, 624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_509, 3);  unsqueeze_509 = None
    sum_56: "f32[624]" = torch.ops.aten.sum.dim_IntList(mul_777, [0, 2, 3])
    sub_181: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_85, unsqueeze_510)
    mul_778: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(mul_777, sub_181);  sub_181 = None
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
    mul_783: "f32[624]" = torch.ops.aten.mul.Tensor(squeeze_103, primals_80);  primals_80 = None
    unsqueeze_517: "f32[1, 624]" = torch.ops.aten.unsqueeze.default(mul_783, 0);  mul_783 = None
    unsqueeze_518: "f32[1, 624, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_517, 2);  unsqueeze_517 = None
    unsqueeze_519: "f32[1, 624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_518, 3);  unsqueeze_518 = None
    sub_182: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_85, unsqueeze_510);  convolution_85 = unsqueeze_510 = None
    mul_784: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(sub_182, unsqueeze_516);  sub_182 = unsqueeze_516 = None
    sub_183: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(mul_777, mul_784);  mul_777 = mul_784 = None
    sub_184: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(sub_183, unsqueeze_513);  sub_183 = unsqueeze_513 = None
    mul_785: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(sub_184, unsqueeze_519);  sub_184 = unsqueeze_519 = None
    mul_786: "f32[624]" = torch.ops.aten.mul.Tensor(sum_57, squeeze_103);  sum_57 = squeeze_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_69 = torch.ops.aten.convolution_backward.default(mul_785, mul_270, primals_222, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 624, [True, True, False]);  mul_785 = mul_270 = primals_222 = None
    getitem_643: "f32[8, 624, 14, 14]" = convolution_backward_69[0]
    getitem_644: "f32[624, 1, 3, 3]" = convolution_backward_69[1];  convolution_backward_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_87: "f32[8, 624, 14, 14]" = torch.ops.aten.sigmoid.default(clone_24)
    full_23: "f32[8, 624, 14, 14]" = torch.ops.aten.full.default([8, 624, 14, 14], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_185: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(full_23, sigmoid_87);  full_23 = None
    mul_787: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(clone_24, sub_185);  clone_24 = sub_185 = None
    add_341: "f32[8, 624, 14, 14]" = torch.ops.aten.add.Scalar(mul_787, 1);  mul_787 = None
    mul_788: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_87, add_341);  sigmoid_87 = add_341 = None
    mul_789: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_643, mul_788);  getitem_643 = mul_788 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_520: "f32[1, 624]" = torch.ops.aten.unsqueeze.default(squeeze_99, 0);  squeeze_99 = None
    unsqueeze_521: "f32[1, 624, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_520, 2);  unsqueeze_520 = None
    unsqueeze_522: "f32[1, 624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_521, 3);  unsqueeze_521 = None
    sum_58: "f32[624]" = torch.ops.aten.sum.dim_IntList(mul_789, [0, 2, 3])
    sub_186: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_84, unsqueeze_522)
    mul_790: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(mul_789, sub_186);  sub_186 = None
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
    mul_795: "f32[624]" = torch.ops.aten.mul.Tensor(squeeze_100, primals_78);  primals_78 = None
    unsqueeze_529: "f32[1, 624]" = torch.ops.aten.unsqueeze.default(mul_795, 0);  mul_795 = None
    unsqueeze_530: "f32[1, 624, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_529, 2);  unsqueeze_529 = None
    unsqueeze_531: "f32[1, 624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_530, 3);  unsqueeze_530 = None
    sub_187: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_84, unsqueeze_522);  convolution_84 = unsqueeze_522 = None
    mul_796: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(sub_187, unsqueeze_528);  sub_187 = unsqueeze_528 = None
    sub_188: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(mul_789, mul_796);  mul_789 = mul_796 = None
    sub_189: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(sub_188, unsqueeze_525);  sub_188 = unsqueeze_525 = None
    mul_797: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(sub_189, unsqueeze_531);  sub_189 = unsqueeze_531 = None
    mul_798: "f32[624]" = torch.ops.aten.mul.Tensor(sum_59, squeeze_100);  sum_59 = squeeze_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_70 = torch.ops.aten.convolution_backward.default(mul_797, add_172, primals_221, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_797 = add_172 = primals_221 = None
    getitem_646: "f32[8, 104, 14, 14]" = convolution_backward_70[0]
    getitem_647: "f32[624, 104, 1, 1]" = convolution_backward_70[1];  convolution_backward_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_532: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(squeeze_96, 0);  squeeze_96 = None
    unsqueeze_533: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_532, 2);  unsqueeze_532 = None
    unsqueeze_534: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_533, 3);  unsqueeze_533 = None
    sum_60: "f32[104]" = torch.ops.aten.sum.dim_IntList(getitem_646, [0, 2, 3])
    sub_190: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(cat_24, unsqueeze_534)
    mul_799: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_646, sub_190);  sub_190 = None
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
    mul_804: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_97, primals_76);  primals_76 = None
    unsqueeze_541: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_804, 0);  mul_804 = None
    unsqueeze_542: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_541, 2);  unsqueeze_541 = None
    unsqueeze_543: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_542, 3);  unsqueeze_542 = None
    sub_191: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(cat_24, unsqueeze_534);  cat_24 = unsqueeze_534 = None
    mul_805: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_191, unsqueeze_540);  sub_191 = unsqueeze_540 = None
    sub_192: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(getitem_646, mul_805);  mul_805 = None
    sub_193: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(sub_192, unsqueeze_537);  sub_192 = unsqueeze_537 = None
    mul_806: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_193, unsqueeze_543);  sub_193 = unsqueeze_543 = None
    mul_807: "f32[104]" = torch.ops.aten.mul.Tensor(sum_61, squeeze_97);  sum_61 = squeeze_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    slice_47: "f32[8, 52, 14, 14]" = torch.ops.aten.slice.Tensor(mul_806, 1, 0, 52)
    slice_48: "f32[8, 52, 14, 14]" = torch.ops.aten.slice.Tensor(mul_806, 1, 52, 104);  mul_806 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_backward_71 = torch.ops.aten.convolution_backward.default(slice_48, getitem_225, primals_220, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_48 = getitem_225 = primals_220 = None
    getitem_649: "f32[8, 312, 14, 14]" = convolution_backward_71[0]
    getitem_650: "f32[52, 312, 1, 1]" = convolution_backward_71[1];  convolution_backward_71 = None
    convolution_backward_72 = torch.ops.aten.convolution_backward.default(slice_47, getitem_224, primals_219, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_47 = getitem_224 = primals_219 = None
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
    convolution_backward_73 = torch.ops.aten.convolution_backward.default(mul_811, mul_254, primals_217, [624], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_811 = mul_254 = primals_217 = None
    getitem_655: "f32[8, 26, 1, 1]" = convolution_backward_73[0]
    getitem_656: "f32[624, 26, 1, 1]" = convolution_backward_73[1]
    getitem_657: "f32[624]" = convolution_backward_73[2];  convolution_backward_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_88: "f32[8, 26, 1, 1]" = torch.ops.aten.sigmoid.default(clone_23)
    full_24: "f32[8, 26, 1, 1]" = torch.ops.aten.full.default([8, 26, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_195: "f32[8, 26, 1, 1]" = torch.ops.aten.sub.Tensor(full_24, sigmoid_88);  full_24 = None
    mul_812: "f32[8, 26, 1, 1]" = torch.ops.aten.mul.Tensor(clone_23, sub_195);  clone_23 = sub_195 = None
    add_342: "f32[8, 26, 1, 1]" = torch.ops.aten.add.Scalar(mul_812, 1);  mul_812 = None
    mul_813: "f32[8, 26, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_88, add_342);  sigmoid_88 = add_342 = None
    mul_814: "f32[8, 26, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_655, mul_813);  getitem_655 = mul_813 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_74 = torch.ops.aten.convolution_backward.default(mul_814, mean_7, primals_215, [26], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_814 = mean_7 = primals_215 = None
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
    full_25: "f32[8, 624, 14, 14]" = torch.ops.aten.full.default([8, 624, 14, 14], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_196: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(full_25, sigmoid_89);  full_25 = None
    mul_815: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(clone_22, sub_196);  clone_22 = sub_196 = None
    add_344: "f32[8, 624, 14, 14]" = torch.ops.aten.add.Scalar(mul_815, 1);  mul_815 = None
    mul_816: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_89, add_344);  sigmoid_89 = add_344 = None
    mul_817: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(add_343, mul_816);  add_343 = mul_816 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_544: "f32[1, 624]" = torch.ops.aten.unsqueeze.default(squeeze_93, 0);  squeeze_93 = None
    unsqueeze_545: "f32[1, 624, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_544, 2);  unsqueeze_544 = None
    unsqueeze_546: "f32[1, 624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_545, 3);  unsqueeze_545 = None
    sum_63: "f32[624]" = torch.ops.aten.sum.dim_IntList(mul_817, [0, 2, 3])
    sub_197: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(cat_23, unsqueeze_546)
    mul_818: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(mul_817, sub_197);  sub_197 = None
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
    mul_823: "f32[624]" = torch.ops.aten.mul.Tensor(squeeze_94, primals_74);  primals_74 = None
    unsqueeze_553: "f32[1, 624]" = torch.ops.aten.unsqueeze.default(mul_823, 0);  mul_823 = None
    unsqueeze_554: "f32[1, 624, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_553, 2);  unsqueeze_553 = None
    unsqueeze_555: "f32[1, 624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_554, 3);  unsqueeze_554 = None
    sub_198: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(cat_23, unsqueeze_546);  cat_23 = unsqueeze_546 = None
    mul_824: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(sub_198, unsqueeze_552);  sub_198 = unsqueeze_552 = None
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
    convolution_backward_75 = torch.ops.aten.convolution_backward.default(slice_52, getitem_221, primals_214, [0], [1, 1], [4, 4], [1, 1], False, [0, 0], 156, [True, True, False]);  slice_52 = getitem_221 = primals_214 = None
    getitem_661: "f32[8, 156, 14, 14]" = convolution_backward_75[0]
    getitem_662: "f32[156, 1, 9, 9]" = convolution_backward_75[1];  convolution_backward_75 = None
    convolution_backward_76 = torch.ops.aten.convolution_backward.default(slice_51, getitem_216, primals_213, [0], [1, 1], [3, 3], [1, 1], False, [0, 0], 156, [True, True, False]);  slice_51 = getitem_216 = primals_213 = None
    getitem_664: "f32[8, 156, 14, 14]" = convolution_backward_76[0]
    getitem_665: "f32[156, 1, 7, 7]" = convolution_backward_76[1];  convolution_backward_76 = None
    convolution_backward_77 = torch.ops.aten.convolution_backward.default(slice_50, getitem_211, primals_212, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 156, [True, True, False]);  slice_50 = getitem_211 = primals_212 = None
    getitem_667: "f32[8, 156, 14, 14]" = convolution_backward_77[0]
    getitem_668: "f32[156, 1, 5, 5]" = convolution_backward_77[1];  convolution_backward_77 = None
    convolution_backward_78 = torch.ops.aten.convolution_backward.default(slice_49, getitem_206, primals_211, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 156, [True, True, False]);  slice_49 = getitem_206 = primals_211 = None
    getitem_670: "f32[8, 156, 14, 14]" = convolution_backward_78[0]
    getitem_671: "f32[156, 1, 3, 3]" = convolution_backward_78[1];  convolution_backward_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    cat_58: "f32[8, 624, 14, 14]" = torch.ops.aten.cat.default([getitem_670, getitem_667, getitem_664, getitem_661], 1);  getitem_670 = getitem_667 = getitem_664 = getitem_661 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_90: "f32[8, 624, 14, 14]" = torch.ops.aten.sigmoid.default(clone_21)
    full_26: "f32[8, 624, 14, 14]" = torch.ops.aten.full.default([8, 624, 14, 14], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_201: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(full_26, sigmoid_90);  full_26 = None
    mul_827: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(clone_21, sub_201);  clone_21 = sub_201 = None
    add_345: "f32[8, 624, 14, 14]" = torch.ops.aten.add.Scalar(mul_827, 1);  mul_827 = None
    mul_828: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_90, add_345);  sigmoid_90 = add_345 = None
    mul_829: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(cat_58, mul_828);  cat_58 = mul_828 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_556: "f32[1, 624]" = torch.ops.aten.unsqueeze.default(squeeze_90, 0);  squeeze_90 = None
    unsqueeze_557: "f32[1, 624, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_556, 2);  unsqueeze_556 = None
    unsqueeze_558: "f32[1, 624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_557, 3);  unsqueeze_557 = None
    sum_65: "f32[624]" = torch.ops.aten.sum.dim_IntList(mul_829, [0, 2, 3])
    sub_202: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(cat_22, unsqueeze_558)
    mul_830: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(mul_829, sub_202);  sub_202 = None
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
    mul_835: "f32[624]" = torch.ops.aten.mul.Tensor(squeeze_91, primals_72);  primals_72 = None
    unsqueeze_565: "f32[1, 624]" = torch.ops.aten.unsqueeze.default(mul_835, 0);  mul_835 = None
    unsqueeze_566: "f32[1, 624, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_565, 2);  unsqueeze_565 = None
    unsqueeze_567: "f32[1, 624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_566, 3);  unsqueeze_566 = None
    sub_203: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(cat_22, unsqueeze_558);  cat_22 = unsqueeze_558 = None
    mul_836: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(sub_203, unsqueeze_564);  sub_203 = unsqueeze_564 = None
    sub_204: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(mul_829, mul_836);  mul_829 = mul_836 = None
    sub_205: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(sub_204, unsqueeze_561);  sub_204 = unsqueeze_561 = None
    mul_837: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(sub_205, unsqueeze_567);  sub_205 = unsqueeze_567 = None
    mul_838: "f32[624]" = torch.ops.aten.mul.Tensor(sum_66, squeeze_91);  sum_66 = squeeze_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    slice_53: "f32[8, 312, 14, 14]" = torch.ops.aten.slice.Tensor(mul_837, 1, 0, 312)
    slice_54: "f32[8, 312, 14, 14]" = torch.ops.aten.slice.Tensor(mul_837, 1, 312, 624);  mul_837 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_backward_79 = torch.ops.aten.convolution_backward.default(slice_54, getitem_199, primals_210, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_54 = getitem_199 = primals_210 = None
    getitem_673: "f32[8, 52, 14, 14]" = convolution_backward_79[0]
    getitem_674: "f32[312, 52, 1, 1]" = convolution_backward_79[1];  convolution_backward_79 = None
    convolution_backward_80 = torch.ops.aten.convolution_backward.default(slice_53, getitem_198, primals_209, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_53 = getitem_198 = primals_209 = None
    getitem_676: "f32[8, 52, 14, 14]" = convolution_backward_80[0]
    getitem_677: "f32[312, 52, 1, 1]" = convolution_backward_80[1];  convolution_backward_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    cat_59: "f32[8, 104, 14, 14]" = torch.ops.aten.cat.default([getitem_676, getitem_673], 1);  getitem_676 = getitem_673 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    add_346: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(getitem_646, cat_59);  getitem_646 = cat_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_568: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(squeeze_87, 0);  squeeze_87 = None
    unsqueeze_569: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_568, 2);  unsqueeze_568 = None
    unsqueeze_570: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_569, 3);  unsqueeze_569 = None
    sum_67: "f32[104]" = torch.ops.aten.sum.dim_IntList(add_346, [0, 2, 3])
    sub_206: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(cat_21, unsqueeze_570)
    mul_839: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(add_346, sub_206);  sub_206 = None
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
    mul_844: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_88, primals_70);  primals_70 = None
    unsqueeze_577: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_844, 0);  mul_844 = None
    unsqueeze_578: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_577, 2);  unsqueeze_577 = None
    unsqueeze_579: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_578, 3);  unsqueeze_578 = None
    sub_207: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(cat_21, unsqueeze_570);  cat_21 = unsqueeze_570 = None
    mul_845: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_207, unsqueeze_576);  sub_207 = unsqueeze_576 = None
    sub_208: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(add_346, mul_845);  mul_845 = None
    sub_209: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(sub_208, unsqueeze_573);  sub_208 = unsqueeze_573 = None
    mul_846: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_209, unsqueeze_579);  sub_209 = unsqueeze_579 = None
    mul_847: "f32[104]" = torch.ops.aten.mul.Tensor(sum_68, squeeze_88);  sum_68 = squeeze_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    slice_55: "f32[8, 52, 14, 14]" = torch.ops.aten.slice.Tensor(mul_846, 1, 0, 52)
    slice_56: "f32[8, 52, 14, 14]" = torch.ops.aten.slice.Tensor(mul_846, 1, 52, 104);  mul_846 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_backward_81 = torch.ops.aten.convolution_backward.default(slice_56, getitem_195, primals_208, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_56 = getitem_195 = primals_208 = None
    getitem_679: "f32[8, 312, 14, 14]" = convolution_backward_81[0]
    getitem_680: "f32[52, 312, 1, 1]" = convolution_backward_81[1];  convolution_backward_81 = None
    convolution_backward_82 = torch.ops.aten.convolution_backward.default(slice_55, getitem_194, primals_207, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_55 = getitem_194 = primals_207 = None
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
    convolution_backward_83 = torch.ops.aten.convolution_backward.default(mul_851, mul_229, primals_205, [624], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_851 = mul_229 = primals_205 = None
    getitem_685: "f32[8, 26, 1, 1]" = convolution_backward_83[0]
    getitem_686: "f32[624, 26, 1, 1]" = convolution_backward_83[1]
    getitem_687: "f32[624]" = convolution_backward_83[2];  convolution_backward_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_91: "f32[8, 26, 1, 1]" = torch.ops.aten.sigmoid.default(clone_20)
    full_27: "f32[8, 26, 1, 1]" = torch.ops.aten.full.default([8, 26, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_211: "f32[8, 26, 1, 1]" = torch.ops.aten.sub.Tensor(full_27, sigmoid_91);  full_27 = None
    mul_852: "f32[8, 26, 1, 1]" = torch.ops.aten.mul.Tensor(clone_20, sub_211);  clone_20 = sub_211 = None
    add_347: "f32[8, 26, 1, 1]" = torch.ops.aten.add.Scalar(mul_852, 1);  mul_852 = None
    mul_853: "f32[8, 26, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_91, add_347);  sigmoid_91 = add_347 = None
    mul_854: "f32[8, 26, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_685, mul_853);  getitem_685 = mul_853 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_84 = torch.ops.aten.convolution_backward.default(mul_854, mean_6, primals_203, [26], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_854 = mean_6 = primals_203 = None
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
    full_28: "f32[8, 624, 14, 14]" = torch.ops.aten.full.default([8, 624, 14, 14], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_212: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(full_28, sigmoid_92);  full_28 = None
    mul_855: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(clone_19, sub_212);  clone_19 = sub_212 = None
    add_349: "f32[8, 624, 14, 14]" = torch.ops.aten.add.Scalar(mul_855, 1);  mul_855 = None
    mul_856: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_92, add_349);  sigmoid_92 = add_349 = None
    mul_857: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(add_348, mul_856);  add_348 = mul_856 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_580: "f32[1, 624]" = torch.ops.aten.unsqueeze.default(squeeze_84, 0);  squeeze_84 = None
    unsqueeze_581: "f32[1, 624, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_580, 2);  unsqueeze_580 = None
    unsqueeze_582: "f32[1, 624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_581, 3);  unsqueeze_581 = None
    sum_70: "f32[624]" = torch.ops.aten.sum.dim_IntList(mul_857, [0, 2, 3])
    sub_213: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(cat_20, unsqueeze_582)
    mul_858: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(mul_857, sub_213);  sub_213 = None
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
    mul_863: "f32[624]" = torch.ops.aten.mul.Tensor(squeeze_85, primals_68);  primals_68 = None
    unsqueeze_589: "f32[1, 624]" = torch.ops.aten.unsqueeze.default(mul_863, 0);  mul_863 = None
    unsqueeze_590: "f32[1, 624, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_589, 2);  unsqueeze_589 = None
    unsqueeze_591: "f32[1, 624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_590, 3);  unsqueeze_590 = None
    sub_214: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(cat_20, unsqueeze_582);  cat_20 = unsqueeze_582 = None
    mul_864: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(sub_214, unsqueeze_588);  sub_214 = unsqueeze_588 = None
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
    convolution_backward_85 = torch.ops.aten.convolution_backward.default(slice_60, getitem_191, primals_202, [0], [1, 1], [4, 4], [1, 1], False, [0, 0], 156, [True, True, False]);  slice_60 = getitem_191 = primals_202 = None
    getitem_691: "f32[8, 156, 14, 14]" = convolution_backward_85[0]
    getitem_692: "f32[156, 1, 9, 9]" = convolution_backward_85[1];  convolution_backward_85 = None
    convolution_backward_86 = torch.ops.aten.convolution_backward.default(slice_59, getitem_186, primals_201, [0], [1, 1], [3, 3], [1, 1], False, [0, 0], 156, [True, True, False]);  slice_59 = getitem_186 = primals_201 = None
    getitem_694: "f32[8, 156, 14, 14]" = convolution_backward_86[0]
    getitem_695: "f32[156, 1, 7, 7]" = convolution_backward_86[1];  convolution_backward_86 = None
    convolution_backward_87 = torch.ops.aten.convolution_backward.default(slice_58, getitem_181, primals_200, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 156, [True, True, False]);  slice_58 = getitem_181 = primals_200 = None
    getitem_697: "f32[8, 156, 14, 14]" = convolution_backward_87[0]
    getitem_698: "f32[156, 1, 5, 5]" = convolution_backward_87[1];  convolution_backward_87 = None
    convolution_backward_88 = torch.ops.aten.convolution_backward.default(slice_57, getitem_176, primals_199, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 156, [True, True, False]);  slice_57 = getitem_176 = primals_199 = None
    getitem_700: "f32[8, 156, 14, 14]" = convolution_backward_88[0]
    getitem_701: "f32[156, 1, 3, 3]" = convolution_backward_88[1];  convolution_backward_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    cat_61: "f32[8, 624, 14, 14]" = torch.ops.aten.cat.default([getitem_700, getitem_697, getitem_694, getitem_691], 1);  getitem_700 = getitem_697 = getitem_694 = getitem_691 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_93: "f32[8, 624, 14, 14]" = torch.ops.aten.sigmoid.default(clone_18)
    full_29: "f32[8, 624, 14, 14]" = torch.ops.aten.full.default([8, 624, 14, 14], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_217: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(full_29, sigmoid_93);  full_29 = None
    mul_867: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(clone_18, sub_217);  clone_18 = sub_217 = None
    add_350: "f32[8, 624, 14, 14]" = torch.ops.aten.add.Scalar(mul_867, 1);  mul_867 = None
    mul_868: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_93, add_350);  sigmoid_93 = add_350 = None
    mul_869: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(cat_61, mul_868);  cat_61 = mul_868 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_592: "f32[1, 624]" = torch.ops.aten.unsqueeze.default(squeeze_81, 0);  squeeze_81 = None
    unsqueeze_593: "f32[1, 624, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_592, 2);  unsqueeze_592 = None
    unsqueeze_594: "f32[1, 624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_593, 3);  unsqueeze_593 = None
    sum_72: "f32[624]" = torch.ops.aten.sum.dim_IntList(mul_869, [0, 2, 3])
    sub_218: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(cat_19, unsqueeze_594)
    mul_870: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(mul_869, sub_218);  sub_218 = None
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
    mul_875: "f32[624]" = torch.ops.aten.mul.Tensor(squeeze_82, primals_66);  primals_66 = None
    unsqueeze_601: "f32[1, 624]" = torch.ops.aten.unsqueeze.default(mul_875, 0);  mul_875 = None
    unsqueeze_602: "f32[1, 624, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_601, 2);  unsqueeze_601 = None
    unsqueeze_603: "f32[1, 624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_602, 3);  unsqueeze_602 = None
    sub_219: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(cat_19, unsqueeze_594);  cat_19 = unsqueeze_594 = None
    mul_876: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(sub_219, unsqueeze_600);  sub_219 = unsqueeze_600 = None
    sub_220: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(mul_869, mul_876);  mul_869 = mul_876 = None
    sub_221: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(sub_220, unsqueeze_597);  sub_220 = unsqueeze_597 = None
    mul_877: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(sub_221, unsqueeze_603);  sub_221 = unsqueeze_603 = None
    mul_878: "f32[624]" = torch.ops.aten.mul.Tensor(sum_73, squeeze_82);  sum_73 = squeeze_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    slice_61: "f32[8, 312, 14, 14]" = torch.ops.aten.slice.Tensor(mul_877, 1, 0, 312)
    slice_62: "f32[8, 312, 14, 14]" = torch.ops.aten.slice.Tensor(mul_877, 1, 312, 624);  mul_877 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_backward_89 = torch.ops.aten.convolution_backward.default(slice_62, getitem_169, primals_198, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_62 = getitem_169 = primals_198 = None
    getitem_703: "f32[8, 52, 14, 14]" = convolution_backward_89[0]
    getitem_704: "f32[312, 52, 1, 1]" = convolution_backward_89[1];  convolution_backward_89 = None
    convolution_backward_90 = torch.ops.aten.convolution_backward.default(slice_61, getitem_168, primals_197, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_61 = getitem_168 = primals_197 = None
    getitem_706: "f32[8, 52, 14, 14]" = convolution_backward_90[0]
    getitem_707: "f32[312, 52, 1, 1]" = convolution_backward_90[1];  convolution_backward_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    cat_62: "f32[8, 104, 14, 14]" = torch.ops.aten.cat.default([getitem_706, getitem_703], 1);  getitem_706 = getitem_703 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    add_351: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(add_346, cat_62);  add_346 = cat_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_604: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(squeeze_78, 0);  squeeze_78 = None
    unsqueeze_605: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_604, 2);  unsqueeze_604 = None
    unsqueeze_606: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_605, 3);  unsqueeze_605 = None
    sum_74: "f32[104]" = torch.ops.aten.sum.dim_IntList(add_351, [0, 2, 3])
    sub_222: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(cat_18, unsqueeze_606)
    mul_879: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(add_351, sub_222);  sub_222 = None
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
    mul_884: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_79, primals_64);  primals_64 = None
    unsqueeze_613: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_884, 0);  mul_884 = None
    unsqueeze_614: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_613, 2);  unsqueeze_613 = None
    unsqueeze_615: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_614, 3);  unsqueeze_614 = None
    sub_223: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(cat_18, unsqueeze_606);  cat_18 = unsqueeze_606 = None
    mul_885: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_223, unsqueeze_612);  sub_223 = unsqueeze_612 = None
    sub_224: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(add_351, mul_885);  mul_885 = None
    sub_225: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(sub_224, unsqueeze_609);  sub_224 = unsqueeze_609 = None
    mul_886: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_225, unsqueeze_615);  sub_225 = unsqueeze_615 = None
    mul_887: "f32[104]" = torch.ops.aten.mul.Tensor(sum_75, squeeze_79);  sum_75 = squeeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    slice_63: "f32[8, 52, 14, 14]" = torch.ops.aten.slice.Tensor(mul_886, 1, 0, 52)
    slice_64: "f32[8, 52, 14, 14]" = torch.ops.aten.slice.Tensor(mul_886, 1, 52, 104);  mul_886 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_backward_91 = torch.ops.aten.convolution_backward.default(slice_64, getitem_165, primals_196, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_64 = getitem_165 = primals_196 = None
    getitem_709: "f32[8, 312, 14, 14]" = convolution_backward_91[0]
    getitem_710: "f32[52, 312, 1, 1]" = convolution_backward_91[1];  convolution_backward_91 = None
    convolution_backward_92 = torch.ops.aten.convolution_backward.default(slice_63, getitem_164, primals_195, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_63 = getitem_164 = primals_195 = None
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
    convolution_backward_93 = torch.ops.aten.convolution_backward.default(mul_891, mul_204, primals_193, [624], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_891 = mul_204 = primals_193 = None
    getitem_715: "f32[8, 26, 1, 1]" = convolution_backward_93[0]
    getitem_716: "f32[624, 26, 1, 1]" = convolution_backward_93[1]
    getitem_717: "f32[624]" = convolution_backward_93[2];  convolution_backward_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_94: "f32[8, 26, 1, 1]" = torch.ops.aten.sigmoid.default(clone_17)
    full_30: "f32[8, 26, 1, 1]" = torch.ops.aten.full.default([8, 26, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_227: "f32[8, 26, 1, 1]" = torch.ops.aten.sub.Tensor(full_30, sigmoid_94);  full_30 = None
    mul_892: "f32[8, 26, 1, 1]" = torch.ops.aten.mul.Tensor(clone_17, sub_227);  clone_17 = sub_227 = None
    add_352: "f32[8, 26, 1, 1]" = torch.ops.aten.add.Scalar(mul_892, 1);  mul_892 = None
    mul_893: "f32[8, 26, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_94, add_352);  sigmoid_94 = add_352 = None
    mul_894: "f32[8, 26, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_715, mul_893);  getitem_715 = mul_893 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_94 = torch.ops.aten.convolution_backward.default(mul_894, mean_5, primals_191, [26], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_894 = mean_5 = primals_191 = None
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
    full_31: "f32[8, 624, 14, 14]" = torch.ops.aten.full.default([8, 624, 14, 14], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_228: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(full_31, sigmoid_95);  full_31 = None
    mul_895: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(clone_16, sub_228);  clone_16 = sub_228 = None
    add_354: "f32[8, 624, 14, 14]" = torch.ops.aten.add.Scalar(mul_895, 1);  mul_895 = None
    mul_896: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_95, add_354);  sigmoid_95 = add_354 = None
    mul_897: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(add_353, mul_896);  add_353 = mul_896 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_616: "f32[1, 624]" = torch.ops.aten.unsqueeze.default(squeeze_75, 0);  squeeze_75 = None
    unsqueeze_617: "f32[1, 624, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_616, 2);  unsqueeze_616 = None
    unsqueeze_618: "f32[1, 624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_617, 3);  unsqueeze_617 = None
    sum_77: "f32[624]" = torch.ops.aten.sum.dim_IntList(mul_897, [0, 2, 3])
    sub_229: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(cat_17, unsqueeze_618)
    mul_898: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(mul_897, sub_229);  sub_229 = None
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
    mul_903: "f32[624]" = torch.ops.aten.mul.Tensor(squeeze_76, primals_62);  primals_62 = None
    unsqueeze_625: "f32[1, 624]" = torch.ops.aten.unsqueeze.default(mul_903, 0);  mul_903 = None
    unsqueeze_626: "f32[1, 624, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_625, 2);  unsqueeze_625 = None
    unsqueeze_627: "f32[1, 624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_626, 3);  unsqueeze_626 = None
    sub_230: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(cat_17, unsqueeze_618);  cat_17 = unsqueeze_618 = None
    mul_904: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(sub_230, unsqueeze_624);  sub_230 = unsqueeze_624 = None
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
    convolution_backward_95 = torch.ops.aten.convolution_backward.default(slice_68, getitem_161, primals_190, [0], [1, 1], [4, 4], [1, 1], False, [0, 0], 156, [True, True, False]);  slice_68 = getitem_161 = primals_190 = None
    getitem_721: "f32[8, 156, 14, 14]" = convolution_backward_95[0]
    getitem_722: "f32[156, 1, 9, 9]" = convolution_backward_95[1];  convolution_backward_95 = None
    convolution_backward_96 = torch.ops.aten.convolution_backward.default(slice_67, getitem_156, primals_189, [0], [1, 1], [3, 3], [1, 1], False, [0, 0], 156, [True, True, False]);  slice_67 = getitem_156 = primals_189 = None
    getitem_724: "f32[8, 156, 14, 14]" = convolution_backward_96[0]
    getitem_725: "f32[156, 1, 7, 7]" = convolution_backward_96[1];  convolution_backward_96 = None
    convolution_backward_97 = torch.ops.aten.convolution_backward.default(slice_66, getitem_151, primals_188, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 156, [True, True, False]);  slice_66 = getitem_151 = primals_188 = None
    getitem_727: "f32[8, 156, 14, 14]" = convolution_backward_97[0]
    getitem_728: "f32[156, 1, 5, 5]" = convolution_backward_97[1];  convolution_backward_97 = None
    convolution_backward_98 = torch.ops.aten.convolution_backward.default(slice_65, getitem_146, primals_187, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 156, [True, True, False]);  slice_65 = getitem_146 = primals_187 = None
    getitem_730: "f32[8, 156, 14, 14]" = convolution_backward_98[0]
    getitem_731: "f32[156, 1, 3, 3]" = convolution_backward_98[1];  convolution_backward_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    cat_64: "f32[8, 624, 14, 14]" = torch.ops.aten.cat.default([getitem_730, getitem_727, getitem_724, getitem_721], 1);  getitem_730 = getitem_727 = getitem_724 = getitem_721 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_96: "f32[8, 624, 14, 14]" = torch.ops.aten.sigmoid.default(clone_15)
    full_32: "f32[8, 624, 14, 14]" = torch.ops.aten.full.default([8, 624, 14, 14], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_233: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(full_32, sigmoid_96);  full_32 = None
    mul_907: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(clone_15, sub_233);  clone_15 = sub_233 = None
    add_355: "f32[8, 624, 14, 14]" = torch.ops.aten.add.Scalar(mul_907, 1);  mul_907 = None
    mul_908: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_96, add_355);  sigmoid_96 = add_355 = None
    mul_909: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(cat_64, mul_908);  cat_64 = mul_908 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_628: "f32[1, 624]" = torch.ops.aten.unsqueeze.default(squeeze_72, 0);  squeeze_72 = None
    unsqueeze_629: "f32[1, 624, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_628, 2);  unsqueeze_628 = None
    unsqueeze_630: "f32[1, 624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_629, 3);  unsqueeze_629 = None
    sum_79: "f32[624]" = torch.ops.aten.sum.dim_IntList(mul_909, [0, 2, 3])
    sub_234: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(cat_16, unsqueeze_630)
    mul_910: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(mul_909, sub_234);  sub_234 = None
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
    mul_915: "f32[624]" = torch.ops.aten.mul.Tensor(squeeze_73, primals_60);  primals_60 = None
    unsqueeze_637: "f32[1, 624]" = torch.ops.aten.unsqueeze.default(mul_915, 0);  mul_915 = None
    unsqueeze_638: "f32[1, 624, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_637, 2);  unsqueeze_637 = None
    unsqueeze_639: "f32[1, 624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_638, 3);  unsqueeze_638 = None
    sub_235: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(cat_16, unsqueeze_630);  cat_16 = unsqueeze_630 = None
    mul_916: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(sub_235, unsqueeze_636);  sub_235 = unsqueeze_636 = None
    sub_236: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(mul_909, mul_916);  mul_909 = mul_916 = None
    sub_237: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(sub_236, unsqueeze_633);  sub_236 = unsqueeze_633 = None
    mul_917: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(sub_237, unsqueeze_639);  sub_237 = unsqueeze_639 = None
    mul_918: "f32[624]" = torch.ops.aten.mul.Tensor(sum_80, squeeze_73);  sum_80 = squeeze_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    slice_69: "f32[8, 312, 14, 14]" = torch.ops.aten.slice.Tensor(mul_917, 1, 0, 312)
    slice_70: "f32[8, 312, 14, 14]" = torch.ops.aten.slice.Tensor(mul_917, 1, 312, 624);  mul_917 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_backward_99 = torch.ops.aten.convolution_backward.default(slice_70, getitem_139, primals_186, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_70 = getitem_139 = primals_186 = None
    getitem_733: "f32[8, 52, 14, 14]" = convolution_backward_99[0]
    getitem_734: "f32[312, 52, 1, 1]" = convolution_backward_99[1];  convolution_backward_99 = None
    convolution_backward_100 = torch.ops.aten.convolution_backward.default(slice_69, getitem_138, primals_185, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_69 = getitem_138 = primals_185 = None
    getitem_736: "f32[8, 52, 14, 14]" = convolution_backward_100[0]
    getitem_737: "f32[312, 52, 1, 1]" = convolution_backward_100[1];  convolution_backward_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    cat_65: "f32[8, 104, 14, 14]" = torch.ops.aten.cat.default([getitem_736, getitem_733], 1);  getitem_736 = getitem_733 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    add_356: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(add_351, cat_65);  add_351 = cat_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_640: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(squeeze_69, 0);  squeeze_69 = None
    unsqueeze_641: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_640, 2);  unsqueeze_640 = None
    unsqueeze_642: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_641, 3);  unsqueeze_641 = None
    sum_81: "f32[104]" = torch.ops.aten.sum.dim_IntList(add_356, [0, 2, 3])
    sub_238: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_53, unsqueeze_642)
    mul_919: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(add_356, sub_238);  sub_238 = None
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
    mul_924: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_70, primals_58);  primals_58 = None
    unsqueeze_649: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(mul_924, 0);  mul_924 = None
    unsqueeze_650: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_649, 2);  unsqueeze_649 = None
    unsqueeze_651: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_650, 3);  unsqueeze_650 = None
    sub_239: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_53, unsqueeze_642);  convolution_53 = unsqueeze_642 = None
    mul_925: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_239, unsqueeze_648);  sub_239 = unsqueeze_648 = None
    sub_240: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(add_356, mul_925);  add_356 = mul_925 = None
    sub_241: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(sub_240, unsqueeze_645);  sub_240 = unsqueeze_645 = None
    mul_926: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_241, unsqueeze_651);  sub_241 = unsqueeze_651 = None
    mul_927: "f32[104]" = torch.ops.aten.mul.Tensor(sum_82, squeeze_70);  sum_82 = squeeze_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_101 = torch.ops.aten.convolution_backward.default(mul_926, mul_180, primals_184, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_926 = mul_180 = primals_184 = None
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
    convolution_backward_102 = torch.ops.aten.convolution_backward.default(mul_931, mul_179, primals_182, [336], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_931 = mul_179 = primals_182 = None
    getitem_742: "f32[8, 14, 1, 1]" = convolution_backward_102[0]
    getitem_743: "f32[336, 14, 1, 1]" = convolution_backward_102[1]
    getitem_744: "f32[336]" = convolution_backward_102[2];  convolution_backward_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_97: "f32[8, 14, 1, 1]" = torch.ops.aten.sigmoid.default(clone_14)
    full_33: "f32[8, 14, 1, 1]" = torch.ops.aten.full.default([8, 14, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_243: "f32[8, 14, 1, 1]" = torch.ops.aten.sub.Tensor(full_33, sigmoid_97);  full_33 = None
    mul_932: "f32[8, 14, 1, 1]" = torch.ops.aten.mul.Tensor(clone_14, sub_243);  clone_14 = sub_243 = None
    add_357: "f32[8, 14, 1, 1]" = torch.ops.aten.add.Scalar(mul_932, 1);  mul_932 = None
    mul_933: "f32[8, 14, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_97, add_357);  sigmoid_97 = add_357 = None
    mul_934: "f32[8, 14, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_742, mul_933);  getitem_742 = mul_933 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_103 = torch.ops.aten.convolution_backward.default(mul_934, mean_4, primals_180, [14], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_934 = mean_4 = primals_180 = None
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
    full_34: "f32[8, 336, 14, 14]" = torch.ops.aten.full.default([8, 336, 14, 14], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_244: "f32[8, 336, 14, 14]" = torch.ops.aten.sub.Tensor(full_34, sigmoid_98);  full_34 = None
    mul_935: "f32[8, 336, 14, 14]" = torch.ops.aten.mul.Tensor(clone_13, sub_244);  clone_13 = sub_244 = None
    add_359: "f32[8, 336, 14, 14]" = torch.ops.aten.add.Scalar(mul_935, 1);  mul_935 = None
    mul_936: "f32[8, 336, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_98, add_359);  sigmoid_98 = add_359 = None
    mul_937: "f32[8, 336, 14, 14]" = torch.ops.aten.mul.Tensor(add_358, mul_936);  add_358 = mul_936 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_652: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(squeeze_66, 0);  squeeze_66 = None
    unsqueeze_653: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_652, 2);  unsqueeze_652 = None
    unsqueeze_654: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_653, 3);  unsqueeze_653 = None
    sum_84: "f32[336]" = torch.ops.aten.sum.dim_IntList(mul_937, [0, 2, 3])
    sub_245: "f32[8, 336, 14, 14]" = torch.ops.aten.sub.Tensor(cat_15, unsqueeze_654)
    mul_938: "f32[8, 336, 14, 14]" = torch.ops.aten.mul.Tensor(mul_937, sub_245);  sub_245 = None
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
    mul_943: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_67, primals_56);  primals_56 = None
    unsqueeze_661: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(mul_943, 0);  mul_943 = None
    unsqueeze_662: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_661, 2);  unsqueeze_661 = None
    unsqueeze_663: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_662, 3);  unsqueeze_662 = None
    sub_246: "f32[8, 336, 14, 14]" = torch.ops.aten.sub.Tensor(cat_15, unsqueeze_654);  cat_15 = unsqueeze_654 = None
    mul_944: "f32[8, 336, 14, 14]" = torch.ops.aten.mul.Tensor(sub_246, unsqueeze_660);  sub_246 = unsqueeze_660 = None
    sub_247: "f32[8, 336, 14, 14]" = torch.ops.aten.sub.Tensor(mul_937, mul_944);  mul_937 = mul_944 = None
    sub_248: "f32[8, 336, 14, 14]" = torch.ops.aten.sub.Tensor(sub_247, unsqueeze_657);  sub_247 = unsqueeze_657 = None
    mul_945: "f32[8, 336, 14, 14]" = torch.ops.aten.mul.Tensor(sub_248, unsqueeze_663);  sub_248 = unsqueeze_663 = None
    mul_946: "f32[336]" = torch.ops.aten.mul.Tensor(sum_85, squeeze_67);  sum_85 = squeeze_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    slice_71: "f32[8, 112, 14, 14]" = torch.ops.aten.slice.Tensor(mul_945, 1, 0, 112)
    slice_72: "f32[8, 112, 14, 14]" = torch.ops.aten.slice.Tensor(mul_945, 1, 112, 224)
    slice_73: "f32[8, 112, 14, 14]" = torch.ops.aten.slice.Tensor(mul_945, 1, 224, 336);  mul_945 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_backward_104 = torch.ops.aten.convolution_backward.default(slice_73, constant_pad_nd_10, primals_55, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 112, [True, True, False]);  slice_73 = constant_pad_nd_10 = primals_55 = None
    getitem_748: "f32[8, 112, 33, 33]" = convolution_backward_104[0]
    getitem_749: "f32[112, 1, 7, 7]" = convolution_backward_104[1];  convolution_backward_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_19: "f32[8, 112, 28, 28]" = torch.ops.aten.constant_pad_nd.default(getitem_748, [-2, -3, -2, -3]);  getitem_748 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_backward_105 = torch.ops.aten.convolution_backward.default(slice_72, constant_pad_nd_9, primals_54, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 112, [True, True, False]);  slice_72 = constant_pad_nd_9 = primals_54 = None
    getitem_751: "f32[8, 112, 31, 31]" = convolution_backward_105[0]
    getitem_752: "f32[112, 1, 5, 5]" = convolution_backward_105[1];  convolution_backward_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_20: "f32[8, 112, 28, 28]" = torch.ops.aten.constant_pad_nd.default(getitem_751, [-1, -2, -1, -2]);  getitem_751 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_backward_106 = torch.ops.aten.convolution_backward.default(slice_71, constant_pad_nd_8, primals_53, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 112, [True, True, False]);  slice_71 = constant_pad_nd_8 = primals_53 = None
    getitem_754: "f32[8, 112, 29, 29]" = convolution_backward_106[0]
    getitem_755: "f32[112, 1, 3, 3]" = convolution_backward_106[1];  convolution_backward_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_21: "f32[8, 112, 28, 28]" = torch.ops.aten.constant_pad_nd.default(getitem_754, [0, -1, 0, -1]);  getitem_754 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    cat_66: "f32[8, 336, 28, 28]" = torch.ops.aten.cat.default([constant_pad_nd_21, constant_pad_nd_20, constant_pad_nd_19], 1);  constant_pad_nd_21 = constant_pad_nd_20 = constant_pad_nd_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_99: "f32[8, 336, 28, 28]" = torch.ops.aten.sigmoid.default(clone_12)
    full_35: "f32[8, 336, 28, 28]" = torch.ops.aten.full.default([8, 336, 28, 28], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_249: "f32[8, 336, 28, 28]" = torch.ops.aten.sub.Tensor(full_35, sigmoid_99);  full_35 = None
    mul_947: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(clone_12, sub_249);  clone_12 = sub_249 = None
    add_360: "f32[8, 336, 28, 28]" = torch.ops.aten.add.Scalar(mul_947, 1);  mul_947 = None
    mul_948: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(sigmoid_99, add_360);  sigmoid_99 = add_360 = None
    mul_949: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(cat_66, mul_948);  cat_66 = mul_948 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_664: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(squeeze_63, 0);  squeeze_63 = None
    unsqueeze_665: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_664, 2);  unsqueeze_664 = None
    unsqueeze_666: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_665, 3);  unsqueeze_665 = None
    sum_86: "f32[336]" = torch.ops.aten.sum.dim_IntList(mul_949, [0, 2, 3])
    sub_250: "f32[8, 336, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_47, unsqueeze_666)
    mul_950: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(mul_949, sub_250);  sub_250 = None
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
    mul_955: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_64, primals_51);  primals_51 = None
    unsqueeze_673: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(mul_955, 0);  mul_955 = None
    unsqueeze_674: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_673, 2);  unsqueeze_673 = None
    unsqueeze_675: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_674, 3);  unsqueeze_674 = None
    sub_251: "f32[8, 336, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_47, unsqueeze_666);  convolution_47 = unsqueeze_666 = None
    mul_956: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(sub_251, unsqueeze_672);  sub_251 = unsqueeze_672 = None
    sub_252: "f32[8, 336, 28, 28]" = torch.ops.aten.sub.Tensor(mul_949, mul_956);  mul_949 = mul_956 = None
    sub_253: "f32[8, 336, 28, 28]" = torch.ops.aten.sub.Tensor(sub_252, unsqueeze_669);  sub_252 = unsqueeze_669 = None
    mul_957: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(sub_253, unsqueeze_675);  sub_253 = unsqueeze_675 = None
    mul_958: "f32[336]" = torch.ops.aten.mul.Tensor(sum_87, squeeze_64);  sum_87 = squeeze_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_107 = torch.ops.aten.convolution_backward.default(mul_957, add_109, primals_179, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_957 = add_109 = primals_179 = None
    getitem_757: "f32[8, 56, 28, 28]" = convolution_backward_107[0]
    getitem_758: "f32[336, 56, 1, 1]" = convolution_backward_107[1];  convolution_backward_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_676: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(squeeze_60, 0);  squeeze_60 = None
    unsqueeze_677: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_676, 2);  unsqueeze_676 = None
    unsqueeze_678: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_677, 3);  unsqueeze_677 = None
    sum_88: "f32[56]" = torch.ops.aten.sum.dim_IntList(getitem_757, [0, 2, 3])
    sub_254: "f32[8, 56, 28, 28]" = torch.ops.aten.sub.Tensor(cat_14, unsqueeze_678)
    mul_959: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_757, sub_254);  sub_254 = None
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
    mul_964: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_61, primals_49);  primals_49 = None
    unsqueeze_685: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_964, 0);  mul_964 = None
    unsqueeze_686: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_685, 2);  unsqueeze_685 = None
    unsqueeze_687: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_686, 3);  unsqueeze_686 = None
    sub_255: "f32[8, 56, 28, 28]" = torch.ops.aten.sub.Tensor(cat_14, unsqueeze_678);  cat_14 = unsqueeze_678 = None
    mul_965: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(sub_255, unsqueeze_684);  sub_255 = unsqueeze_684 = None
    sub_256: "f32[8, 56, 28, 28]" = torch.ops.aten.sub.Tensor(getitem_757, mul_965);  mul_965 = None
    sub_257: "f32[8, 56, 28, 28]" = torch.ops.aten.sub.Tensor(sub_256, unsqueeze_681);  sub_256 = unsqueeze_681 = None
    mul_966: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(sub_257, unsqueeze_687);  sub_257 = unsqueeze_687 = None
    mul_967: "f32[56]" = torch.ops.aten.mul.Tensor(sum_89, squeeze_61);  sum_89 = squeeze_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    slice_74: "f32[8, 28, 28, 28]" = torch.ops.aten.slice.Tensor(mul_966, 1, 0, 28)
    slice_75: "f32[8, 28, 28, 28]" = torch.ops.aten.slice.Tensor(mul_966, 1, 28, 56);  mul_966 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_backward_108 = torch.ops.aten.convolution_backward.default(slice_75, getitem_117, primals_178, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_75 = getitem_117 = primals_178 = None
    getitem_760: "f32[8, 168, 28, 28]" = convolution_backward_108[0]
    getitem_761: "f32[28, 168, 1, 1]" = convolution_backward_108[1];  convolution_backward_108 = None
    convolution_backward_109 = torch.ops.aten.convolution_backward.default(slice_74, getitem_116, primals_177, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_74 = getitem_116 = primals_177 = None
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
    convolution_backward_110 = torch.ops.aten.convolution_backward.default(mul_971, mul_154, primals_175, [336], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_971 = mul_154 = primals_175 = None
    getitem_766: "f32[8, 28, 1, 1]" = convolution_backward_110[0]
    getitem_767: "f32[336, 28, 1, 1]" = convolution_backward_110[1]
    getitem_768: "f32[336]" = convolution_backward_110[2];  convolution_backward_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_100: "f32[8, 28, 1, 1]" = torch.ops.aten.sigmoid.default(clone_11)
    full_36: "f32[8, 28, 1, 1]" = torch.ops.aten.full.default([8, 28, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_259: "f32[8, 28, 1, 1]" = torch.ops.aten.sub.Tensor(full_36, sigmoid_100);  full_36 = None
    mul_972: "f32[8, 28, 1, 1]" = torch.ops.aten.mul.Tensor(clone_11, sub_259);  clone_11 = sub_259 = None
    add_361: "f32[8, 28, 1, 1]" = torch.ops.aten.add.Scalar(mul_972, 1);  mul_972 = None
    mul_973: "f32[8, 28, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_100, add_361);  sigmoid_100 = add_361 = None
    mul_974: "f32[8, 28, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_766, mul_973);  getitem_766 = mul_973 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_111 = torch.ops.aten.convolution_backward.default(mul_974, mean_3, primals_173, [28], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_974 = mean_3 = primals_173 = None
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
    full_37: "f32[8, 336, 28, 28]" = torch.ops.aten.full.default([8, 336, 28, 28], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_260: "f32[8, 336, 28, 28]" = torch.ops.aten.sub.Tensor(full_37, sigmoid_101);  full_37 = None
    mul_975: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(clone_10, sub_260);  clone_10 = sub_260 = None
    add_363: "f32[8, 336, 28, 28]" = torch.ops.aten.add.Scalar(mul_975, 1);  mul_975 = None
    mul_976: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(sigmoid_101, add_363);  sigmoid_101 = add_363 = None
    mul_977: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(add_362, mul_976);  add_362 = mul_976 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_688: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(squeeze_57, 0);  squeeze_57 = None
    unsqueeze_689: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_688, 2);  unsqueeze_688 = None
    unsqueeze_690: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_689, 3);  unsqueeze_689 = None
    sum_91: "f32[336]" = torch.ops.aten.sum.dim_IntList(mul_977, [0, 2, 3])
    sub_261: "f32[8, 336, 28, 28]" = torch.ops.aten.sub.Tensor(cat_13, unsqueeze_690)
    mul_978: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(mul_977, sub_261);  sub_261 = None
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
    mul_983: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_58, primals_47);  primals_47 = None
    unsqueeze_697: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(mul_983, 0);  mul_983 = None
    unsqueeze_698: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_697, 2);  unsqueeze_697 = None
    unsqueeze_699: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_698, 3);  unsqueeze_698 = None
    sub_262: "f32[8, 336, 28, 28]" = torch.ops.aten.sub.Tensor(cat_13, unsqueeze_690);  cat_13 = unsqueeze_690 = None
    mul_984: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(sub_262, unsqueeze_696);  sub_262 = unsqueeze_696 = None
    sub_263: "f32[8, 336, 28, 28]" = torch.ops.aten.sub.Tensor(mul_977, mul_984);  mul_977 = mul_984 = None
    sub_264: "f32[8, 336, 28, 28]" = torch.ops.aten.sub.Tensor(sub_263, unsqueeze_693);  sub_263 = unsqueeze_693 = None
    mul_985: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(sub_264, unsqueeze_699);  sub_264 = unsqueeze_699 = None
    mul_986: "f32[336]" = torch.ops.aten.mul.Tensor(sum_92, squeeze_58);  sum_92 = squeeze_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    slice_76: "f32[8, 168, 28, 28]" = torch.ops.aten.slice.Tensor(mul_985, 1, 0, 168)
    slice_77: "f32[8, 168, 28, 28]" = torch.ops.aten.slice.Tensor(mul_985, 1, 168, 336);  mul_985 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_backward_112 = torch.ops.aten.convolution_backward.default(slice_77, getitem_113, primals_172, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 168, [True, True, False]);  slice_77 = getitem_113 = primals_172 = None
    getitem_772: "f32[8, 168, 28, 28]" = convolution_backward_112[0]
    getitem_773: "f32[168, 1, 5, 5]" = convolution_backward_112[1];  convolution_backward_112 = None
    convolution_backward_113 = torch.ops.aten.convolution_backward.default(slice_76, getitem_110, primals_171, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 168, [True, True, False]);  slice_76 = getitem_110 = primals_171 = None
    getitem_775: "f32[8, 168, 28, 28]" = convolution_backward_113[0]
    getitem_776: "f32[168, 1, 3, 3]" = convolution_backward_113[1];  convolution_backward_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    cat_68: "f32[8, 336, 28, 28]" = torch.ops.aten.cat.default([getitem_775, getitem_772], 1);  getitem_775 = getitem_772 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_102: "f32[8, 336, 28, 28]" = torch.ops.aten.sigmoid.default(clone_9)
    full_38: "f32[8, 336, 28, 28]" = torch.ops.aten.full.default([8, 336, 28, 28], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_265: "f32[8, 336, 28, 28]" = torch.ops.aten.sub.Tensor(full_38, sigmoid_102);  full_38 = None
    mul_987: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(clone_9, sub_265);  clone_9 = sub_265 = None
    add_364: "f32[8, 336, 28, 28]" = torch.ops.aten.add.Scalar(mul_987, 1);  mul_987 = None
    mul_988: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(sigmoid_102, add_364);  sigmoid_102 = add_364 = None
    mul_989: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(cat_68, mul_988);  cat_68 = mul_988 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_700: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(squeeze_54, 0);  squeeze_54 = None
    unsqueeze_701: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_700, 2);  unsqueeze_700 = None
    unsqueeze_702: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_701, 3);  unsqueeze_701 = None
    sum_93: "f32[336]" = torch.ops.aten.sum.dim_IntList(mul_989, [0, 2, 3])
    sub_266: "f32[8, 336, 28, 28]" = torch.ops.aten.sub.Tensor(cat_12, unsqueeze_702)
    mul_990: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(mul_989, sub_266);  sub_266 = None
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
    mul_995: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_55, primals_45);  primals_45 = None
    unsqueeze_709: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(mul_995, 0);  mul_995 = None
    unsqueeze_710: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_709, 2);  unsqueeze_709 = None
    unsqueeze_711: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_710, 3);  unsqueeze_710 = None
    sub_267: "f32[8, 336, 28, 28]" = torch.ops.aten.sub.Tensor(cat_12, unsqueeze_702);  cat_12 = unsqueeze_702 = None
    mul_996: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(sub_267, unsqueeze_708);  sub_267 = unsqueeze_708 = None
    sub_268: "f32[8, 336, 28, 28]" = torch.ops.aten.sub.Tensor(mul_989, mul_996);  mul_989 = mul_996 = None
    sub_269: "f32[8, 336, 28, 28]" = torch.ops.aten.sub.Tensor(sub_268, unsqueeze_705);  sub_268 = unsqueeze_705 = None
    mul_997: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(sub_269, unsqueeze_711);  sub_269 = unsqueeze_711 = None
    mul_998: "f32[336]" = torch.ops.aten.mul.Tensor(sum_94, squeeze_55);  sum_94 = squeeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    slice_78: "f32[8, 168, 28, 28]" = torch.ops.aten.slice.Tensor(mul_997, 1, 0, 168)
    slice_79: "f32[8, 168, 28, 28]" = torch.ops.aten.slice.Tensor(mul_997, 1, 168, 336);  mul_997 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_backward_114 = torch.ops.aten.convolution_backward.default(slice_79, getitem_105, primals_170, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_79 = getitem_105 = primals_170 = None
    getitem_778: "f32[8, 28, 28, 28]" = convolution_backward_114[0]
    getitem_779: "f32[168, 28, 1, 1]" = convolution_backward_114[1];  convolution_backward_114 = None
    convolution_backward_115 = torch.ops.aten.convolution_backward.default(slice_78, getitem_104, primals_169, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_78 = getitem_104 = primals_169 = None
    getitem_781: "f32[8, 28, 28, 28]" = convolution_backward_115[0]
    getitem_782: "f32[168, 28, 1, 1]" = convolution_backward_115[1];  convolution_backward_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    cat_69: "f32[8, 56, 28, 28]" = torch.ops.aten.cat.default([getitem_781, getitem_778], 1);  getitem_781 = getitem_778 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    add_365: "f32[8, 56, 28, 28]" = torch.ops.aten.add.Tensor(getitem_757, cat_69);  getitem_757 = cat_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_712: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(squeeze_51, 0);  squeeze_51 = None
    unsqueeze_713: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_712, 2);  unsqueeze_712 = None
    unsqueeze_714: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_713, 3);  unsqueeze_713 = None
    sum_95: "f32[56]" = torch.ops.aten.sum.dim_IntList(add_365, [0, 2, 3])
    sub_270: "f32[8, 56, 28, 28]" = torch.ops.aten.sub.Tensor(cat_11, unsqueeze_714)
    mul_999: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(add_365, sub_270);  sub_270 = None
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
    mul_1004: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_52, primals_43);  primals_43 = None
    unsqueeze_721: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1004, 0);  mul_1004 = None
    unsqueeze_722: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_721, 2);  unsqueeze_721 = None
    unsqueeze_723: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_722, 3);  unsqueeze_722 = None
    sub_271: "f32[8, 56, 28, 28]" = torch.ops.aten.sub.Tensor(cat_11, unsqueeze_714);  cat_11 = unsqueeze_714 = None
    mul_1005: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(sub_271, unsqueeze_720);  sub_271 = unsqueeze_720 = None
    sub_272: "f32[8, 56, 28, 28]" = torch.ops.aten.sub.Tensor(add_365, mul_1005);  mul_1005 = None
    sub_273: "f32[8, 56, 28, 28]" = torch.ops.aten.sub.Tensor(sub_272, unsqueeze_717);  sub_272 = unsqueeze_717 = None
    mul_1006: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(sub_273, unsqueeze_723);  sub_273 = unsqueeze_723 = None
    mul_1007: "f32[56]" = torch.ops.aten.mul.Tensor(sum_96, squeeze_52);  sum_96 = squeeze_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    slice_80: "f32[8, 28, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1006, 1, 0, 28)
    slice_81: "f32[8, 28, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1006, 1, 28, 56);  mul_1006 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_backward_116 = torch.ops.aten.convolution_backward.default(slice_81, getitem_101, primals_168, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_81 = getitem_101 = primals_168 = None
    getitem_784: "f32[8, 168, 28, 28]" = convolution_backward_116[0]
    getitem_785: "f32[28, 168, 1, 1]" = convolution_backward_116[1];  convolution_backward_116 = None
    convolution_backward_117 = torch.ops.aten.convolution_backward.default(slice_80, getitem_100, primals_167, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_80 = getitem_100 = primals_167 = None
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
    convolution_backward_118 = torch.ops.aten.convolution_backward.default(mul_1011, mul_129, primals_165, [336], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_1011 = mul_129 = primals_165 = None
    getitem_790: "f32[8, 28, 1, 1]" = convolution_backward_118[0]
    getitem_791: "f32[336, 28, 1, 1]" = convolution_backward_118[1]
    getitem_792: "f32[336]" = convolution_backward_118[2];  convolution_backward_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_103: "f32[8, 28, 1, 1]" = torch.ops.aten.sigmoid.default(clone_8)
    full_39: "f32[8, 28, 1, 1]" = torch.ops.aten.full.default([8, 28, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_275: "f32[8, 28, 1, 1]" = torch.ops.aten.sub.Tensor(full_39, sigmoid_103);  full_39 = None
    mul_1012: "f32[8, 28, 1, 1]" = torch.ops.aten.mul.Tensor(clone_8, sub_275);  clone_8 = sub_275 = None
    add_366: "f32[8, 28, 1, 1]" = torch.ops.aten.add.Scalar(mul_1012, 1);  mul_1012 = None
    mul_1013: "f32[8, 28, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_103, add_366);  sigmoid_103 = add_366 = None
    mul_1014: "f32[8, 28, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_790, mul_1013);  getitem_790 = mul_1013 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_119 = torch.ops.aten.convolution_backward.default(mul_1014, mean_2, primals_163, [28], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_1014 = mean_2 = primals_163 = None
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
    full_40: "f32[8, 336, 28, 28]" = torch.ops.aten.full.default([8, 336, 28, 28], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_276: "f32[8, 336, 28, 28]" = torch.ops.aten.sub.Tensor(full_40, sigmoid_104);  full_40 = None
    mul_1015: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(clone_7, sub_276);  clone_7 = sub_276 = None
    add_368: "f32[8, 336, 28, 28]" = torch.ops.aten.add.Scalar(mul_1015, 1);  mul_1015 = None
    mul_1016: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(sigmoid_104, add_368);  sigmoid_104 = add_368 = None
    mul_1017: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(add_367, mul_1016);  add_367 = mul_1016 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_724: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(squeeze_48, 0);  squeeze_48 = None
    unsqueeze_725: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_724, 2);  unsqueeze_724 = None
    unsqueeze_726: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_725, 3);  unsqueeze_725 = None
    sum_98: "f32[336]" = torch.ops.aten.sum.dim_IntList(mul_1017, [0, 2, 3])
    sub_277: "f32[8, 336, 28, 28]" = torch.ops.aten.sub.Tensor(cat_10, unsqueeze_726)
    mul_1018: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(mul_1017, sub_277);  sub_277 = None
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
    mul_1023: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_49, primals_41);  primals_41 = None
    unsqueeze_733: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(mul_1023, 0);  mul_1023 = None
    unsqueeze_734: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_733, 2);  unsqueeze_733 = None
    unsqueeze_735: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_734, 3);  unsqueeze_734 = None
    sub_278: "f32[8, 336, 28, 28]" = torch.ops.aten.sub.Tensor(cat_10, unsqueeze_726);  cat_10 = unsqueeze_726 = None
    mul_1024: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(sub_278, unsqueeze_732);  sub_278 = unsqueeze_732 = None
    sub_279: "f32[8, 336, 28, 28]" = torch.ops.aten.sub.Tensor(mul_1017, mul_1024);  mul_1017 = mul_1024 = None
    sub_280: "f32[8, 336, 28, 28]" = torch.ops.aten.sub.Tensor(sub_279, unsqueeze_729);  sub_279 = unsqueeze_729 = None
    mul_1025: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(sub_280, unsqueeze_735);  sub_280 = unsqueeze_735 = None
    mul_1026: "f32[336]" = torch.ops.aten.mul.Tensor(sum_99, squeeze_49);  sum_99 = squeeze_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    slice_82: "f32[8, 168, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1025, 1, 0, 168)
    slice_83: "f32[8, 168, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1025, 1, 168, 336);  mul_1025 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_backward_120 = torch.ops.aten.convolution_backward.default(slice_83, getitem_97, primals_162, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 168, [True, True, False]);  slice_83 = getitem_97 = primals_162 = None
    getitem_796: "f32[8, 168, 28, 28]" = convolution_backward_120[0]
    getitem_797: "f32[168, 1, 5, 5]" = convolution_backward_120[1];  convolution_backward_120 = None
    convolution_backward_121 = torch.ops.aten.convolution_backward.default(slice_82, getitem_94, primals_161, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 168, [True, True, False]);  slice_82 = getitem_94 = primals_161 = None
    getitem_799: "f32[8, 168, 28, 28]" = convolution_backward_121[0]
    getitem_800: "f32[168, 1, 3, 3]" = convolution_backward_121[1];  convolution_backward_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    cat_71: "f32[8, 336, 28, 28]" = torch.ops.aten.cat.default([getitem_799, getitem_796], 1);  getitem_799 = getitem_796 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_105: "f32[8, 336, 28, 28]" = torch.ops.aten.sigmoid.default(clone_6)
    full_41: "f32[8, 336, 28, 28]" = torch.ops.aten.full.default([8, 336, 28, 28], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_281: "f32[8, 336, 28, 28]" = torch.ops.aten.sub.Tensor(full_41, sigmoid_105);  full_41 = None
    mul_1027: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(clone_6, sub_281);  clone_6 = sub_281 = None
    add_369: "f32[8, 336, 28, 28]" = torch.ops.aten.add.Scalar(mul_1027, 1);  mul_1027 = None
    mul_1028: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(sigmoid_105, add_369);  sigmoid_105 = add_369 = None
    mul_1029: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(cat_71, mul_1028);  cat_71 = mul_1028 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_736: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(squeeze_45, 0);  squeeze_45 = None
    unsqueeze_737: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_736, 2);  unsqueeze_736 = None
    unsqueeze_738: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_737, 3);  unsqueeze_737 = None
    sum_100: "f32[336]" = torch.ops.aten.sum.dim_IntList(mul_1029, [0, 2, 3])
    sub_282: "f32[8, 336, 28, 28]" = torch.ops.aten.sub.Tensor(cat_9, unsqueeze_738)
    mul_1030: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(mul_1029, sub_282);  sub_282 = None
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
    mul_1035: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_46, primals_39);  primals_39 = None
    unsqueeze_745: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(mul_1035, 0);  mul_1035 = None
    unsqueeze_746: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_745, 2);  unsqueeze_745 = None
    unsqueeze_747: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_746, 3);  unsqueeze_746 = None
    sub_283: "f32[8, 336, 28, 28]" = torch.ops.aten.sub.Tensor(cat_9, unsqueeze_738);  cat_9 = unsqueeze_738 = None
    mul_1036: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(sub_283, unsqueeze_744);  sub_283 = unsqueeze_744 = None
    sub_284: "f32[8, 336, 28, 28]" = torch.ops.aten.sub.Tensor(mul_1029, mul_1036);  mul_1029 = mul_1036 = None
    sub_285: "f32[8, 336, 28, 28]" = torch.ops.aten.sub.Tensor(sub_284, unsqueeze_741);  sub_284 = unsqueeze_741 = None
    mul_1037: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(sub_285, unsqueeze_747);  sub_285 = unsqueeze_747 = None
    mul_1038: "f32[336]" = torch.ops.aten.mul.Tensor(sum_101, squeeze_46);  sum_101 = squeeze_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    slice_84: "f32[8, 168, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1037, 1, 0, 168)
    slice_85: "f32[8, 168, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1037, 1, 168, 336);  mul_1037 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_backward_122 = torch.ops.aten.convolution_backward.default(slice_85, getitem_89, primals_160, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_85 = getitem_89 = primals_160 = None
    getitem_802: "f32[8, 28, 28, 28]" = convolution_backward_122[0]
    getitem_803: "f32[168, 28, 1, 1]" = convolution_backward_122[1];  convolution_backward_122 = None
    convolution_backward_123 = torch.ops.aten.convolution_backward.default(slice_84, getitem_88, primals_159, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_84 = getitem_88 = primals_159 = None
    getitem_805: "f32[8, 28, 28, 28]" = convolution_backward_123[0]
    getitem_806: "f32[168, 28, 1, 1]" = convolution_backward_123[1];  convolution_backward_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    cat_72: "f32[8, 56, 28, 28]" = torch.ops.aten.cat.default([getitem_805, getitem_802], 1);  getitem_805 = getitem_802 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    add_370: "f32[8, 56, 28, 28]" = torch.ops.aten.add.Tensor(add_365, cat_72);  add_365 = cat_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_748: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(squeeze_42, 0);  squeeze_42 = None
    unsqueeze_749: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_748, 2);  unsqueeze_748 = None
    unsqueeze_750: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_749, 3);  unsqueeze_749 = None
    sum_102: "f32[56]" = torch.ops.aten.sum.dim_IntList(add_370, [0, 2, 3])
    sub_286: "f32[8, 56, 28, 28]" = torch.ops.aten.sub.Tensor(cat_8, unsqueeze_750)
    mul_1039: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(add_370, sub_286);  sub_286 = None
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
    mul_1044: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_43, primals_37);  primals_37 = None
    unsqueeze_757: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1044, 0);  mul_1044 = None
    unsqueeze_758: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_757, 2);  unsqueeze_757 = None
    unsqueeze_759: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_758, 3);  unsqueeze_758 = None
    sub_287: "f32[8, 56, 28, 28]" = torch.ops.aten.sub.Tensor(cat_8, unsqueeze_750);  cat_8 = unsqueeze_750 = None
    mul_1045: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(sub_287, unsqueeze_756);  sub_287 = unsqueeze_756 = None
    sub_288: "f32[8, 56, 28, 28]" = torch.ops.aten.sub.Tensor(add_370, mul_1045);  mul_1045 = None
    sub_289: "f32[8, 56, 28, 28]" = torch.ops.aten.sub.Tensor(sub_288, unsqueeze_753);  sub_288 = unsqueeze_753 = None
    mul_1046: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(sub_289, unsqueeze_759);  sub_289 = unsqueeze_759 = None
    mul_1047: "f32[56]" = torch.ops.aten.mul.Tensor(sum_103, squeeze_43);  sum_103 = squeeze_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    slice_86: "f32[8, 28, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1046, 1, 0, 28)
    slice_87: "f32[8, 28, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1046, 1, 28, 56);  mul_1046 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_backward_124 = torch.ops.aten.convolution_backward.default(slice_87, getitem_85, primals_158, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_87 = getitem_85 = primals_158 = None
    getitem_808: "f32[8, 168, 28, 28]" = convolution_backward_124[0]
    getitem_809: "f32[28, 168, 1, 1]" = convolution_backward_124[1];  convolution_backward_124 = None
    convolution_backward_125 = torch.ops.aten.convolution_backward.default(slice_86, getitem_84, primals_157, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_86 = getitem_84 = primals_157 = None
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
    convolution_backward_126 = torch.ops.aten.convolution_backward.default(mul_1051, mul_104, primals_155, [336], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_1051 = mul_104 = primals_155 = None
    getitem_814: "f32[8, 28, 1, 1]" = convolution_backward_126[0]
    getitem_815: "f32[336, 28, 1, 1]" = convolution_backward_126[1]
    getitem_816: "f32[336]" = convolution_backward_126[2];  convolution_backward_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_106: "f32[8, 28, 1, 1]" = torch.ops.aten.sigmoid.default(clone_5)
    full_42: "f32[8, 28, 1, 1]" = torch.ops.aten.full.default([8, 28, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_291: "f32[8, 28, 1, 1]" = torch.ops.aten.sub.Tensor(full_42, sigmoid_106);  full_42 = None
    mul_1052: "f32[8, 28, 1, 1]" = torch.ops.aten.mul.Tensor(clone_5, sub_291);  clone_5 = sub_291 = None
    add_371: "f32[8, 28, 1, 1]" = torch.ops.aten.add.Scalar(mul_1052, 1);  mul_1052 = None
    mul_1053: "f32[8, 28, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_106, add_371);  sigmoid_106 = add_371 = None
    mul_1054: "f32[8, 28, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_814, mul_1053);  getitem_814 = mul_1053 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_127 = torch.ops.aten.convolution_backward.default(mul_1054, mean_1, primals_153, [28], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_1054 = mean_1 = primals_153 = None
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
    full_43: "f32[8, 336, 28, 28]" = torch.ops.aten.full.default([8, 336, 28, 28], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_292: "f32[8, 336, 28, 28]" = torch.ops.aten.sub.Tensor(full_43, sigmoid_107);  full_43 = None
    mul_1055: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(clone_4, sub_292);  clone_4 = sub_292 = None
    add_373: "f32[8, 336, 28, 28]" = torch.ops.aten.add.Scalar(mul_1055, 1);  mul_1055 = None
    mul_1056: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(sigmoid_107, add_373);  sigmoid_107 = add_373 = None
    mul_1057: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(add_372, mul_1056);  add_372 = mul_1056 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_760: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(squeeze_39, 0);  squeeze_39 = None
    unsqueeze_761: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_760, 2);  unsqueeze_760 = None
    unsqueeze_762: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_761, 3);  unsqueeze_761 = None
    sum_105: "f32[336]" = torch.ops.aten.sum.dim_IntList(mul_1057, [0, 2, 3])
    sub_293: "f32[8, 336, 28, 28]" = torch.ops.aten.sub.Tensor(cat_7, unsqueeze_762)
    mul_1058: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(mul_1057, sub_293);  sub_293 = None
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
    mul_1063: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_40, primals_35);  primals_35 = None
    unsqueeze_769: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(mul_1063, 0);  mul_1063 = None
    unsqueeze_770: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_769, 2);  unsqueeze_769 = None
    unsqueeze_771: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_770, 3);  unsqueeze_770 = None
    sub_294: "f32[8, 336, 28, 28]" = torch.ops.aten.sub.Tensor(cat_7, unsqueeze_762);  cat_7 = unsqueeze_762 = None
    mul_1064: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(sub_294, unsqueeze_768);  sub_294 = unsqueeze_768 = None
    sub_295: "f32[8, 336, 28, 28]" = torch.ops.aten.sub.Tensor(mul_1057, mul_1064);  mul_1057 = mul_1064 = None
    sub_296: "f32[8, 336, 28, 28]" = torch.ops.aten.sub.Tensor(sub_295, unsqueeze_765);  sub_295 = unsqueeze_765 = None
    mul_1065: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(sub_296, unsqueeze_771);  sub_296 = unsqueeze_771 = None
    mul_1066: "f32[336]" = torch.ops.aten.mul.Tensor(sum_106, squeeze_40);  sum_106 = squeeze_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    slice_88: "f32[8, 168, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1065, 1, 0, 168)
    slice_89: "f32[8, 168, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1065, 1, 168, 336);  mul_1065 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_backward_128 = torch.ops.aten.convolution_backward.default(slice_89, getitem_81, primals_152, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 168, [True, True, False]);  slice_89 = getitem_81 = primals_152 = None
    getitem_820: "f32[8, 168, 28, 28]" = convolution_backward_128[0]
    getitem_821: "f32[168, 1, 5, 5]" = convolution_backward_128[1];  convolution_backward_128 = None
    convolution_backward_129 = torch.ops.aten.convolution_backward.default(slice_88, getitem_78, primals_151, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 168, [True, True, False]);  slice_88 = getitem_78 = primals_151 = None
    getitem_823: "f32[8, 168, 28, 28]" = convolution_backward_129[0]
    getitem_824: "f32[168, 1, 3, 3]" = convolution_backward_129[1];  convolution_backward_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    cat_74: "f32[8, 336, 28, 28]" = torch.ops.aten.cat.default([getitem_823, getitem_820], 1);  getitem_823 = getitem_820 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_108: "f32[8, 336, 28, 28]" = torch.ops.aten.sigmoid.default(clone_3)
    full_44: "f32[8, 336, 28, 28]" = torch.ops.aten.full.default([8, 336, 28, 28], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_297: "f32[8, 336, 28, 28]" = torch.ops.aten.sub.Tensor(full_44, sigmoid_108);  full_44 = None
    mul_1067: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(clone_3, sub_297);  clone_3 = sub_297 = None
    add_374: "f32[8, 336, 28, 28]" = torch.ops.aten.add.Scalar(mul_1067, 1);  mul_1067 = None
    mul_1068: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(sigmoid_108, add_374);  sigmoid_108 = add_374 = None
    mul_1069: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(cat_74, mul_1068);  cat_74 = mul_1068 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_772: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(squeeze_36, 0);  squeeze_36 = None
    unsqueeze_773: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_772, 2);  unsqueeze_772 = None
    unsqueeze_774: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_773, 3);  unsqueeze_773 = None
    sum_107: "f32[336]" = torch.ops.aten.sum.dim_IntList(mul_1069, [0, 2, 3])
    sub_298: "f32[8, 336, 28, 28]" = torch.ops.aten.sub.Tensor(cat_6, unsqueeze_774)
    mul_1070: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(mul_1069, sub_298);  sub_298 = None
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
    mul_1075: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_37, primals_33);  primals_33 = None
    unsqueeze_781: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(mul_1075, 0);  mul_1075 = None
    unsqueeze_782: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_781, 2);  unsqueeze_781 = None
    unsqueeze_783: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_782, 3);  unsqueeze_782 = None
    sub_299: "f32[8, 336, 28, 28]" = torch.ops.aten.sub.Tensor(cat_6, unsqueeze_774);  cat_6 = unsqueeze_774 = None
    mul_1076: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(sub_299, unsqueeze_780);  sub_299 = unsqueeze_780 = None
    sub_300: "f32[8, 336, 28, 28]" = torch.ops.aten.sub.Tensor(mul_1069, mul_1076);  mul_1069 = mul_1076 = None
    sub_301: "f32[8, 336, 28, 28]" = torch.ops.aten.sub.Tensor(sub_300, unsqueeze_777);  sub_300 = unsqueeze_777 = None
    mul_1077: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(sub_301, unsqueeze_783);  sub_301 = unsqueeze_783 = None
    mul_1078: "f32[336]" = torch.ops.aten.mul.Tensor(sum_108, squeeze_37);  sum_108 = squeeze_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    slice_90: "f32[8, 168, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1077, 1, 0, 168)
    slice_91: "f32[8, 168, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1077, 1, 168, 336);  mul_1077 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_backward_130 = torch.ops.aten.convolution_backward.default(slice_91, getitem_73, primals_150, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_91 = getitem_73 = primals_150 = None
    getitem_826: "f32[8, 28, 28, 28]" = convolution_backward_130[0]
    getitem_827: "f32[168, 28, 1, 1]" = convolution_backward_130[1];  convolution_backward_130 = None
    convolution_backward_131 = torch.ops.aten.convolution_backward.default(slice_90, getitem_72, primals_149, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_90 = getitem_72 = primals_149 = None
    getitem_829: "f32[8, 28, 28, 28]" = convolution_backward_131[0]
    getitem_830: "f32[168, 28, 1, 1]" = convolution_backward_131[1];  convolution_backward_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    cat_75: "f32[8, 56, 28, 28]" = torch.ops.aten.cat.default([getitem_829, getitem_826], 1);  getitem_829 = getitem_826 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    add_375: "f32[8, 56, 28, 28]" = torch.ops.aten.add.Tensor(add_370, cat_75);  add_370 = cat_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_784: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(squeeze_33, 0);  squeeze_33 = None
    unsqueeze_785: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_784, 2);  unsqueeze_784 = None
    unsqueeze_786: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_785, 3);  unsqueeze_785 = None
    sum_109: "f32[56]" = torch.ops.aten.sum.dim_IntList(add_375, [0, 2, 3])
    sub_302: "f32[8, 56, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_786)
    mul_1079: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(add_375, sub_302);  sub_302 = None
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
    mul_1084: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_34, primals_31);  primals_31 = None
    unsqueeze_793: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_1084, 0);  mul_1084 = None
    unsqueeze_794: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_793, 2);  unsqueeze_793 = None
    unsqueeze_795: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_794, 3);  unsqueeze_794 = None
    sub_303: "f32[8, 56, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_786);  convolution_22 = unsqueeze_786 = None
    mul_1085: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(sub_303, unsqueeze_792);  sub_303 = unsqueeze_792 = None
    sub_304: "f32[8, 56, 28, 28]" = torch.ops.aten.sub.Tensor(add_375, mul_1085);  add_375 = mul_1085 = None
    sub_305: "f32[8, 56, 28, 28]" = torch.ops.aten.sub.Tensor(sub_304, unsqueeze_789);  sub_304 = unsqueeze_789 = None
    mul_1086: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(sub_305, unsqueeze_795);  sub_305 = unsqueeze_795 = None
    mul_1087: "f32[56]" = torch.ops.aten.mul.Tensor(sum_110, squeeze_34);  sum_110 = squeeze_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_132 = torch.ops.aten.convolution_backward.default(mul_1086, mul_80, primals_148, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1086 = mul_80 = primals_148 = None
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
    convolution_backward_133 = torch.ops.aten.convolution_backward.default(mul_1091, mul_79, primals_146, [240], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_1091 = mul_79 = primals_146 = None
    getitem_835: "f32[8, 20, 1, 1]" = convolution_backward_133[0]
    getitem_836: "f32[240, 20, 1, 1]" = convolution_backward_133[1]
    getitem_837: "f32[240]" = convolution_backward_133[2];  convolution_backward_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_109: "f32[8, 20, 1, 1]" = torch.ops.aten.sigmoid.default(clone_2)
    full_45: "f32[8, 20, 1, 1]" = torch.ops.aten.full.default([8, 20, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_307: "f32[8, 20, 1, 1]" = torch.ops.aten.sub.Tensor(full_45, sigmoid_109);  full_45 = None
    mul_1092: "f32[8, 20, 1, 1]" = torch.ops.aten.mul.Tensor(clone_2, sub_307);  clone_2 = sub_307 = None
    add_376: "f32[8, 20, 1, 1]" = torch.ops.aten.add.Scalar(mul_1092, 1);  mul_1092 = None
    mul_1093: "f32[8, 20, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_109, add_376);  sigmoid_109 = add_376 = None
    mul_1094: "f32[8, 20, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_835, mul_1093);  getitem_835 = mul_1093 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_134 = torch.ops.aten.convolution_backward.default(mul_1094, mean, primals_144, [20], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_1094 = mean = primals_144 = None
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
    full_46: "f32[8, 240, 28, 28]" = torch.ops.aten.full.default([8, 240, 28, 28], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_308: "f32[8, 240, 28, 28]" = torch.ops.aten.sub.Tensor(full_46, sigmoid_110);  full_46 = None
    mul_1095: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(clone_1, sub_308);  clone_1 = sub_308 = None
    add_378: "f32[8, 240, 28, 28]" = torch.ops.aten.add.Scalar(mul_1095, 1);  mul_1095 = None
    mul_1096: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(sigmoid_110, add_378);  sigmoid_110 = add_378 = None
    mul_1097: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(add_377, mul_1096);  add_377 = mul_1096 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_796: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(squeeze_30, 0);  squeeze_30 = None
    unsqueeze_797: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_796, 2);  unsqueeze_796 = None
    unsqueeze_798: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_797, 3);  unsqueeze_797 = None
    sum_112: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_1097, [0, 2, 3])
    sub_309: "f32[8, 240, 28, 28]" = torch.ops.aten.sub.Tensor(cat_5, unsqueeze_798)
    mul_1098: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(mul_1097, sub_309);  sub_309 = None
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
    mul_1103: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_31, primals_29);  primals_29 = None
    unsqueeze_805: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_1103, 0);  mul_1103 = None
    unsqueeze_806: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_805, 2);  unsqueeze_805 = None
    unsqueeze_807: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_806, 3);  unsqueeze_806 = None
    sub_310: "f32[8, 240, 28, 28]" = torch.ops.aten.sub.Tensor(cat_5, unsqueeze_798);  cat_5 = unsqueeze_798 = None
    mul_1104: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(sub_310, unsqueeze_804);  sub_310 = unsqueeze_804 = None
    sub_311: "f32[8, 240, 28, 28]" = torch.ops.aten.sub.Tensor(mul_1097, mul_1104);  mul_1097 = mul_1104 = None
    sub_312: "f32[8, 240, 28, 28]" = torch.ops.aten.sub.Tensor(sub_311, unsqueeze_801);  sub_311 = unsqueeze_801 = None
    mul_1105: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(sub_312, unsqueeze_807);  sub_312 = unsqueeze_807 = None
    mul_1106: "f32[240]" = torch.ops.aten.mul.Tensor(sum_113, squeeze_31);  sum_113 = squeeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    slice_92: "f32[8, 60, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1105, 1, 0, 60)
    slice_93: "f32[8, 60, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1105, 1, 60, 120)
    slice_94: "f32[8, 60, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1105, 1, 120, 180)
    slice_95: "f32[8, 60, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1105, 1, 180, 240);  mul_1105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_backward_135 = torch.ops.aten.convolution_backward.default(slice_95, constant_pad_nd_7, primals_28, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 60, [True, True, False]);  slice_95 = constant_pad_nd_7 = primals_28 = None
    getitem_841: "f32[8, 60, 63, 63]" = convolution_backward_135[0]
    getitem_842: "f32[60, 1, 9, 9]" = convolution_backward_135[1];  convolution_backward_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_22: "f32[8, 60, 56, 56]" = torch.ops.aten.constant_pad_nd.default(getitem_841, [-3, -4, -3, -4]);  getitem_841 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_backward_136 = torch.ops.aten.convolution_backward.default(slice_94, constant_pad_nd_6, primals_27, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 60, [True, True, False]);  slice_94 = constant_pad_nd_6 = primals_27 = None
    getitem_844: "f32[8, 60, 61, 61]" = convolution_backward_136[0]
    getitem_845: "f32[60, 1, 7, 7]" = convolution_backward_136[1];  convolution_backward_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_23: "f32[8, 60, 56, 56]" = torch.ops.aten.constant_pad_nd.default(getitem_844, [-2, -3, -2, -3]);  getitem_844 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_backward_137 = torch.ops.aten.convolution_backward.default(slice_93, constant_pad_nd_5, primals_26, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 60, [True, True, False]);  slice_93 = constant_pad_nd_5 = primals_26 = None
    getitem_847: "f32[8, 60, 59, 59]" = convolution_backward_137[0]
    getitem_848: "f32[60, 1, 5, 5]" = convolution_backward_137[1];  convolution_backward_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_24: "f32[8, 60, 56, 56]" = torch.ops.aten.constant_pad_nd.default(getitem_847, [-1, -2, -1, -2]);  getitem_847 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_backward_138 = torch.ops.aten.convolution_backward.default(slice_92, constant_pad_nd_4, primals_25, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 60, [True, True, False]);  slice_92 = constant_pad_nd_4 = primals_25 = None
    getitem_850: "f32[8, 60, 57, 57]" = convolution_backward_138[0]
    getitem_851: "f32[60, 1, 3, 3]" = convolution_backward_138[1];  convolution_backward_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_25: "f32[8, 60, 56, 56]" = torch.ops.aten.constant_pad_nd.default(getitem_850, [0, -1, 0, -1]);  getitem_850 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    cat_76: "f32[8, 240, 56, 56]" = torch.ops.aten.cat.default([constant_pad_nd_25, constant_pad_nd_24, constant_pad_nd_23, constant_pad_nd_22], 1);  constant_pad_nd_25 = constant_pad_nd_24 = constant_pad_nd_23 = constant_pad_nd_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_111: "f32[8, 240, 56, 56]" = torch.ops.aten.sigmoid.default(clone)
    full_47: "f32[8, 240, 56, 56]" = torch.ops.aten.full.default([8, 240, 56, 56], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_313: "f32[8, 240, 56, 56]" = torch.ops.aten.sub.Tensor(full_47, sigmoid_111);  full_47 = None
    mul_1107: "f32[8, 240, 56, 56]" = torch.ops.aten.mul.Tensor(clone, sub_313);  clone = sub_313 = None
    add_379: "f32[8, 240, 56, 56]" = torch.ops.aten.add.Scalar(mul_1107, 1);  mul_1107 = None
    mul_1108: "f32[8, 240, 56, 56]" = torch.ops.aten.mul.Tensor(sigmoid_111, add_379);  sigmoid_111 = add_379 = None
    mul_1109: "f32[8, 240, 56, 56]" = torch.ops.aten.mul.Tensor(cat_76, mul_1108);  cat_76 = mul_1108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_808: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(squeeze_27, 0);  squeeze_27 = None
    unsqueeze_809: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_808, 2);  unsqueeze_808 = None
    unsqueeze_810: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_809, 3);  unsqueeze_809 = None
    sum_114: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_1109, [0, 2, 3])
    sub_314: "f32[8, 240, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_810)
    mul_1110: "f32[8, 240, 56, 56]" = torch.ops.aten.mul.Tensor(mul_1109, sub_314);  sub_314 = None
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
    mul_1115: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_28, primals_23);  primals_23 = None
    unsqueeze_817: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_1115, 0);  mul_1115 = None
    unsqueeze_818: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_817, 2);  unsqueeze_817 = None
    unsqueeze_819: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_818, 3);  unsqueeze_818 = None
    sub_315: "f32[8, 240, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_810);  convolution_15 = unsqueeze_810 = None
    mul_1116: "f32[8, 240, 56, 56]" = torch.ops.aten.mul.Tensor(sub_315, unsqueeze_816);  sub_315 = unsqueeze_816 = None
    sub_316: "f32[8, 240, 56, 56]" = torch.ops.aten.sub.Tensor(mul_1109, mul_1116);  mul_1109 = mul_1116 = None
    sub_317: "f32[8, 240, 56, 56]" = torch.ops.aten.sub.Tensor(sub_316, unsqueeze_813);  sub_316 = unsqueeze_813 = None
    mul_1117: "f32[8, 240, 56, 56]" = torch.ops.aten.mul.Tensor(sub_317, unsqueeze_819);  sub_317 = unsqueeze_819 = None
    mul_1118: "f32[240]" = torch.ops.aten.mul.Tensor(sum_115, squeeze_28);  sum_115 = squeeze_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_139 = torch.ops.aten.convolution_backward.default(mul_1117, add_46, primals_143, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1117 = add_46 = primals_143 = None
    getitem_853: "f32[8, 40, 56, 56]" = convolution_backward_139[0]
    getitem_854: "f32[240, 40, 1, 1]" = convolution_backward_139[1];  convolution_backward_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_820: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(squeeze_24, 0);  squeeze_24 = None
    unsqueeze_821: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_820, 2);  unsqueeze_820 = None
    unsqueeze_822: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_821, 3);  unsqueeze_821 = None
    sum_116: "f32[40]" = torch.ops.aten.sum.dim_IntList(getitem_853, [0, 2, 3])
    sub_318: "f32[8, 40, 56, 56]" = torch.ops.aten.sub.Tensor(cat_4, unsqueeze_822)
    mul_1119: "f32[8, 40, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_853, sub_318);  sub_318 = None
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
    mul_1124: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_25, primals_21);  primals_21 = None
    unsqueeze_829: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_1124, 0);  mul_1124 = None
    unsqueeze_830: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_829, 2);  unsqueeze_829 = None
    unsqueeze_831: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_830, 3);  unsqueeze_830 = None
    sub_319: "f32[8, 40, 56, 56]" = torch.ops.aten.sub.Tensor(cat_4, unsqueeze_822);  cat_4 = unsqueeze_822 = None
    mul_1125: "f32[8, 40, 56, 56]" = torch.ops.aten.mul.Tensor(sub_319, unsqueeze_828);  sub_319 = unsqueeze_828 = None
    sub_320: "f32[8, 40, 56, 56]" = torch.ops.aten.sub.Tensor(getitem_853, mul_1125);  mul_1125 = None
    sub_321: "f32[8, 40, 56, 56]" = torch.ops.aten.sub.Tensor(sub_320, unsqueeze_825);  sub_320 = unsqueeze_825 = None
    mul_1126: "f32[8, 40, 56, 56]" = torch.ops.aten.mul.Tensor(sub_321, unsqueeze_831);  sub_321 = unsqueeze_831 = None
    mul_1127: "f32[40]" = torch.ops.aten.mul.Tensor(sum_117, squeeze_25);  sum_117 = squeeze_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    slice_96: "f32[8, 20, 56, 56]" = torch.ops.aten.slice.Tensor(mul_1126, 1, 0, 20)
    slice_97: "f32[8, 20, 56, 56]" = torch.ops.aten.slice.Tensor(mul_1126, 1, 20, 40);  mul_1126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_backward_140 = torch.ops.aten.convolution_backward.default(slice_97, getitem_43, primals_142, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_97 = getitem_43 = primals_142 = None
    getitem_856: "f32[8, 60, 56, 56]" = convolution_backward_140[0]
    getitem_857: "f32[20, 60, 1, 1]" = convolution_backward_140[1];  convolution_backward_140 = None
    convolution_backward_141 = torch.ops.aten.convolution_backward.default(slice_96, getitem_40, primals_141, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_96 = getitem_40 = primals_141 = None
    getitem_859: "f32[8, 60, 56, 56]" = convolution_backward_141[0]
    getitem_860: "f32[20, 60, 1, 1]" = convolution_backward_141[1];  convolution_backward_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    cat_77: "f32[8, 120, 56, 56]" = torch.ops.aten.cat.default([getitem_859, getitem_856], 1);  getitem_859 = getitem_856 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_43: "f32[8, 120, 56, 56]" = torch.ops.aten.alias.default(relu_5);  relu_5 = None
    alias_44: "f32[8, 120, 56, 56]" = torch.ops.aten.alias.default(alias_43);  alias_43 = None
    le_1: "b8[8, 120, 56, 56]" = torch.ops.aten.le.Scalar(alias_44, 0);  alias_44 = None
    scalar_tensor_1: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_1: "f32[8, 120, 56, 56]" = torch.ops.aten.where.self(le_1, scalar_tensor_1, cat_77);  le_1 = scalar_tensor_1 = cat_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_832: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(squeeze_21, 0);  squeeze_21 = None
    unsqueeze_833: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_832, 2);  unsqueeze_832 = None
    unsqueeze_834: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_833, 3);  unsqueeze_833 = None
    sum_118: "f32[120]" = torch.ops.aten.sum.dim_IntList(where_1, [0, 2, 3])
    sub_322: "f32[8, 120, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_834)
    mul_1128: "f32[8, 120, 56, 56]" = torch.ops.aten.mul.Tensor(where_1, sub_322);  sub_322 = None
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
    mul_1133: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_22, primals_19);  primals_19 = None
    unsqueeze_841: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1133, 0);  mul_1133 = None
    unsqueeze_842: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_841, 2);  unsqueeze_841 = None
    unsqueeze_843: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_842, 3);  unsqueeze_842 = None
    sub_323: "f32[8, 120, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_834);  convolution_12 = unsqueeze_834 = None
    mul_1134: "f32[8, 120, 56, 56]" = torch.ops.aten.mul.Tensor(sub_323, unsqueeze_840);  sub_323 = unsqueeze_840 = None
    sub_324: "f32[8, 120, 56, 56]" = torch.ops.aten.sub.Tensor(where_1, mul_1134);  where_1 = mul_1134 = None
    sub_325: "f32[8, 120, 56, 56]" = torch.ops.aten.sub.Tensor(sub_324, unsqueeze_837);  sub_324 = unsqueeze_837 = None
    mul_1135: "f32[8, 120, 56, 56]" = torch.ops.aten.mul.Tensor(sub_325, unsqueeze_843);  sub_325 = unsqueeze_843 = None
    mul_1136: "f32[120]" = torch.ops.aten.mul.Tensor(sum_119, squeeze_22);  sum_119 = squeeze_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_142 = torch.ops.aten.convolution_backward.default(mul_1135, relu_4, primals_140, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 120, [True, True, False]);  mul_1135 = primals_140 = None
    getitem_862: "f32[8, 120, 56, 56]" = convolution_backward_142[0]
    getitem_863: "f32[120, 1, 3, 3]" = convolution_backward_142[1];  convolution_backward_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_46: "f32[8, 120, 56, 56]" = torch.ops.aten.alias.default(relu_4);  relu_4 = None
    alias_47: "f32[8, 120, 56, 56]" = torch.ops.aten.alias.default(alias_46);  alias_46 = None
    le_2: "b8[8, 120, 56, 56]" = torch.ops.aten.le.Scalar(alias_47, 0);  alias_47 = None
    scalar_tensor_2: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_2: "f32[8, 120, 56, 56]" = torch.ops.aten.where.self(le_2, scalar_tensor_2, getitem_862);  le_2 = scalar_tensor_2 = getitem_862 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_844: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(squeeze_18, 0);  squeeze_18 = None
    unsqueeze_845: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_844, 2);  unsqueeze_844 = None
    unsqueeze_846: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_845, 3);  unsqueeze_845 = None
    sum_120: "f32[120]" = torch.ops.aten.sum.dim_IntList(where_2, [0, 2, 3])
    sub_326: "f32[8, 120, 56, 56]" = torch.ops.aten.sub.Tensor(cat_3, unsqueeze_846)
    mul_1137: "f32[8, 120, 56, 56]" = torch.ops.aten.mul.Tensor(where_2, sub_326);  sub_326 = None
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
    mul_1142: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_19, primals_17);  primals_17 = None
    unsqueeze_853: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1142, 0);  mul_1142 = None
    unsqueeze_854: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_853, 2);  unsqueeze_853 = None
    unsqueeze_855: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_854, 3);  unsqueeze_854 = None
    sub_327: "f32[8, 120, 56, 56]" = torch.ops.aten.sub.Tensor(cat_3, unsqueeze_846);  cat_3 = unsqueeze_846 = None
    mul_1143: "f32[8, 120, 56, 56]" = torch.ops.aten.mul.Tensor(sub_327, unsqueeze_852);  sub_327 = unsqueeze_852 = None
    sub_328: "f32[8, 120, 56, 56]" = torch.ops.aten.sub.Tensor(where_2, mul_1143);  where_2 = mul_1143 = None
    sub_329: "f32[8, 120, 56, 56]" = torch.ops.aten.sub.Tensor(sub_328, unsqueeze_849);  sub_328 = unsqueeze_849 = None
    mul_1144: "f32[8, 120, 56, 56]" = torch.ops.aten.mul.Tensor(sub_329, unsqueeze_855);  sub_329 = unsqueeze_855 = None
    mul_1145: "f32[120]" = torch.ops.aten.mul.Tensor(sum_121, squeeze_19);  sum_121 = squeeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    slice_98: "f32[8, 60, 56, 56]" = torch.ops.aten.slice.Tensor(mul_1144, 1, 0, 60)
    slice_99: "f32[8, 60, 56, 56]" = torch.ops.aten.slice.Tensor(mul_1144, 1, 60, 120);  mul_1144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_backward_143 = torch.ops.aten.convolution_backward.default(slice_99, getitem_33, primals_139, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_99 = getitem_33 = primals_139 = None
    getitem_865: "f32[8, 20, 56, 56]" = convolution_backward_143[0]
    getitem_866: "f32[60, 20, 1, 1]" = convolution_backward_143[1];  convolution_backward_143 = None
    convolution_backward_144 = torch.ops.aten.convolution_backward.default(slice_98, getitem_32, primals_138, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_98 = getitem_32 = primals_138 = None
    getitem_868: "f32[8, 20, 56, 56]" = convolution_backward_144[0]
    getitem_869: "f32[60, 20, 1, 1]" = convolution_backward_144[1];  convolution_backward_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    cat_78: "f32[8, 40, 56, 56]" = torch.ops.aten.cat.default([getitem_868, getitem_865], 1);  getitem_868 = getitem_865 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    add_380: "f32[8, 40, 56, 56]" = torch.ops.aten.add.Tensor(getitem_853, cat_78);  getitem_853 = cat_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_856: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(squeeze_15, 0);  squeeze_15 = None
    unsqueeze_857: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_856, 2);  unsqueeze_856 = None
    unsqueeze_858: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_857, 3);  unsqueeze_857 = None
    sum_122: "f32[40]" = torch.ops.aten.sum.dim_IntList(add_380, [0, 2, 3])
    sub_330: "f32[8, 40, 56, 56]" = torch.ops.aten.sub.Tensor(cat_2, unsqueeze_858)
    mul_1146: "f32[8, 40, 56, 56]" = torch.ops.aten.mul.Tensor(add_380, sub_330);  sub_330 = None
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
    mul_1151: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_16, primals_15);  primals_15 = None
    unsqueeze_865: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_1151, 0);  mul_1151 = None
    unsqueeze_866: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_865, 2);  unsqueeze_865 = None
    unsqueeze_867: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_866, 3);  unsqueeze_866 = None
    sub_331: "f32[8, 40, 56, 56]" = torch.ops.aten.sub.Tensor(cat_2, unsqueeze_858);  cat_2 = unsqueeze_858 = None
    mul_1152: "f32[8, 40, 56, 56]" = torch.ops.aten.mul.Tensor(sub_331, unsqueeze_864);  sub_331 = unsqueeze_864 = None
    sub_332: "f32[8, 40, 56, 56]" = torch.ops.aten.sub.Tensor(add_380, mul_1152);  add_380 = mul_1152 = None
    sub_333: "f32[8, 40, 56, 56]" = torch.ops.aten.sub.Tensor(sub_332, unsqueeze_861);  sub_332 = unsqueeze_861 = None
    mul_1153: "f32[8, 40, 56, 56]" = torch.ops.aten.mul.Tensor(sub_333, unsqueeze_867);  sub_333 = unsqueeze_867 = None
    mul_1154: "f32[40]" = torch.ops.aten.mul.Tensor(sum_123, squeeze_16);  sum_123 = squeeze_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    slice_100: "f32[8, 20, 56, 56]" = torch.ops.aten.slice.Tensor(mul_1153, 1, 0, 20)
    slice_101: "f32[8, 20, 56, 56]" = torch.ops.aten.slice.Tensor(mul_1153, 1, 20, 40);  mul_1153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_backward_145 = torch.ops.aten.convolution_backward.default(slice_101, getitem_29, primals_137, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_101 = getitem_29 = primals_137 = None
    getitem_871: "f32[8, 96, 56, 56]" = convolution_backward_145[0]
    getitem_872: "f32[20, 96, 1, 1]" = convolution_backward_145[1];  convolution_backward_145 = None
    convolution_backward_146 = torch.ops.aten.convolution_backward.default(slice_100, getitem_26, primals_136, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_100 = getitem_26 = primals_136 = None
    getitem_874: "f32[8, 96, 56, 56]" = convolution_backward_146[0]
    getitem_875: "f32[20, 96, 1, 1]" = convolution_backward_146[1];  convolution_backward_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    cat_79: "f32[8, 192, 56, 56]" = torch.ops.aten.cat.default([getitem_874, getitem_871], 1);  getitem_874 = getitem_871 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_49: "f32[8, 192, 56, 56]" = torch.ops.aten.alias.default(relu_3);  relu_3 = None
    alias_50: "f32[8, 192, 56, 56]" = torch.ops.aten.alias.default(alias_49);  alias_49 = None
    le_3: "b8[8, 192, 56, 56]" = torch.ops.aten.le.Scalar(alias_50, 0);  alias_50 = None
    scalar_tensor_3: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_3: "f32[8, 192, 56, 56]" = torch.ops.aten.where.self(le_3, scalar_tensor_3, cat_79);  le_3 = scalar_tensor_3 = cat_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_868: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_12, 0);  squeeze_12 = None
    unsqueeze_869: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_868, 2);  unsqueeze_868 = None
    unsqueeze_870: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_869, 3);  unsqueeze_869 = None
    sum_124: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_3, [0, 2, 3])
    sub_334: "f32[8, 192, 56, 56]" = torch.ops.aten.sub.Tensor(cat_1, unsqueeze_870)
    mul_1155: "f32[8, 192, 56, 56]" = torch.ops.aten.mul.Tensor(where_3, sub_334);  sub_334 = None
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
    mul_1160: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_13, primals_13);  primals_13 = None
    unsqueeze_877: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_1160, 0);  mul_1160 = None
    unsqueeze_878: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_877, 2);  unsqueeze_877 = None
    unsqueeze_879: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_878, 3);  unsqueeze_878 = None
    sub_335: "f32[8, 192, 56, 56]" = torch.ops.aten.sub.Tensor(cat_1, unsqueeze_870);  cat_1 = unsqueeze_870 = None
    mul_1161: "f32[8, 192, 56, 56]" = torch.ops.aten.mul.Tensor(sub_335, unsqueeze_876);  sub_335 = unsqueeze_876 = None
    sub_336: "f32[8, 192, 56, 56]" = torch.ops.aten.sub.Tensor(where_3, mul_1161);  where_3 = mul_1161 = None
    sub_337: "f32[8, 192, 56, 56]" = torch.ops.aten.sub.Tensor(sub_336, unsqueeze_873);  sub_336 = unsqueeze_873 = None
    mul_1162: "f32[8, 192, 56, 56]" = torch.ops.aten.mul.Tensor(sub_337, unsqueeze_879);  sub_337 = unsqueeze_879 = None
    mul_1163: "f32[192]" = torch.ops.aten.mul.Tensor(sum_125, squeeze_13);  sum_125 = squeeze_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    slice_102: "f32[8, 64, 56, 56]" = torch.ops.aten.slice.Tensor(mul_1162, 1, 0, 64)
    slice_103: "f32[8, 64, 56, 56]" = torch.ops.aten.slice.Tensor(mul_1162, 1, 64, 128)
    slice_104: "f32[8, 64, 56, 56]" = torch.ops.aten.slice.Tensor(mul_1162, 1, 128, 192);  mul_1162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_backward_147 = torch.ops.aten.convolution_backward.default(slice_104, constant_pad_nd_3, primals_12, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 64, [True, True, False]);  slice_104 = constant_pad_nd_3 = primals_12 = None
    getitem_877: "f32[8, 64, 117, 117]" = convolution_backward_147[0]
    getitem_878: "f32[64, 1, 7, 7]" = convolution_backward_147[1];  convolution_backward_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_26: "f32[8, 64, 112, 112]" = torch.ops.aten.constant_pad_nd.default(getitem_877, [-2, -3, -2, -3]);  getitem_877 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_backward_148 = torch.ops.aten.convolution_backward.default(slice_103, constant_pad_nd_2, primals_11, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 64, [True, True, False]);  slice_103 = constant_pad_nd_2 = primals_11 = None
    getitem_880: "f32[8, 64, 115, 115]" = convolution_backward_148[0]
    getitem_881: "f32[64, 1, 5, 5]" = convolution_backward_148[1];  convolution_backward_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_27: "f32[8, 64, 112, 112]" = torch.ops.aten.constant_pad_nd.default(getitem_880, [-1, -2, -1, -2]);  getitem_880 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_backward_149 = torch.ops.aten.convolution_backward.default(slice_102, constant_pad_nd_1, primals_10, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 64, [True, True, False]);  slice_102 = constant_pad_nd_1 = primals_10 = None
    getitem_883: "f32[8, 64, 113, 113]" = convolution_backward_149[0]
    getitem_884: "f32[64, 1, 3, 3]" = convolution_backward_149[1];  convolution_backward_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd_28: "f32[8, 64, 112, 112]" = torch.ops.aten.constant_pad_nd.default(getitem_883, [0, -1, 0, -1]);  getitem_883 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    cat_80: "f32[8, 192, 112, 112]" = torch.ops.aten.cat.default([constant_pad_nd_28, constant_pad_nd_27, constant_pad_nd_26], 1);  constant_pad_nd_28 = constant_pad_nd_27 = constant_pad_nd_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_52: "f32[8, 192, 112, 112]" = torch.ops.aten.alias.default(relu_2);  relu_2 = None
    alias_53: "f32[8, 192, 112, 112]" = torch.ops.aten.alias.default(alias_52);  alias_52 = None
    le_4: "b8[8, 192, 112, 112]" = torch.ops.aten.le.Scalar(alias_53, 0);  alias_53 = None
    scalar_tensor_4: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_4: "f32[8, 192, 112, 112]" = torch.ops.aten.where.self(le_4, scalar_tensor_4, cat_80);  le_4 = scalar_tensor_4 = cat_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_880: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_9, 0);  squeeze_9 = None
    unsqueeze_881: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_880, 2);  unsqueeze_880 = None
    unsqueeze_882: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_881, 3);  unsqueeze_881 = None
    sum_126: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_4, [0, 2, 3])
    sub_338: "f32[8, 192, 112, 112]" = torch.ops.aten.sub.Tensor(cat, unsqueeze_882)
    mul_1164: "f32[8, 192, 112, 112]" = torch.ops.aten.mul.Tensor(where_4, sub_338);  sub_338 = None
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
    mul_1169: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_10, primals_8);  primals_8 = None
    unsqueeze_889: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_1169, 0);  mul_1169 = None
    unsqueeze_890: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_889, 2);  unsqueeze_889 = None
    unsqueeze_891: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_890, 3);  unsqueeze_890 = None
    sub_339: "f32[8, 192, 112, 112]" = torch.ops.aten.sub.Tensor(cat, unsqueeze_882);  cat = unsqueeze_882 = None
    mul_1170: "f32[8, 192, 112, 112]" = torch.ops.aten.mul.Tensor(sub_339, unsqueeze_888);  sub_339 = unsqueeze_888 = None
    sub_340: "f32[8, 192, 112, 112]" = torch.ops.aten.sub.Tensor(where_4, mul_1170);  where_4 = mul_1170 = None
    sub_341: "f32[8, 192, 112, 112]" = torch.ops.aten.sub.Tensor(sub_340, unsqueeze_885);  sub_340 = unsqueeze_885 = None
    mul_1171: "f32[8, 192, 112, 112]" = torch.ops.aten.mul.Tensor(sub_341, unsqueeze_891);  sub_341 = unsqueeze_891 = None
    mul_1172: "f32[192]" = torch.ops.aten.mul.Tensor(sum_127, squeeze_10);  sum_127 = squeeze_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    slice_105: "f32[8, 96, 112, 112]" = torch.ops.aten.slice.Tensor(mul_1171, 1, 0, 96)
    slice_106: "f32[8, 96, 112, 112]" = torch.ops.aten.slice.Tensor(mul_1171, 1, 96, 192);  mul_1171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_backward_150 = torch.ops.aten.convolution_backward.default(slice_106, getitem_7, primals_135, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_106 = getitem_7 = primals_135 = None
    getitem_886: "f32[8, 16, 112, 112]" = convolution_backward_150[0]
    getitem_887: "f32[96, 16, 1, 1]" = convolution_backward_150[1];  convolution_backward_150 = None
    convolution_backward_151 = torch.ops.aten.convolution_backward.default(slice_105, getitem_6, primals_134, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_105 = getitem_6 = primals_134 = None
    getitem_889: "f32[8, 16, 112, 112]" = convolution_backward_151[0]
    getitem_890: "f32[96, 16, 1, 1]" = convolution_backward_151[1];  convolution_backward_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    cat_81: "f32[8, 32, 112, 112]" = torch.ops.aten.cat.default([getitem_889, getitem_886], 1);  getitem_889 = getitem_886 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_892: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze_6, 0);  squeeze_6 = None
    unsqueeze_893: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_892, 2);  unsqueeze_892 = None
    unsqueeze_894: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_893, 3);  unsqueeze_893 = None
    sum_128: "f32[32]" = torch.ops.aten.sum.dim_IntList(cat_81, [0, 2, 3])
    sub_342: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_894)
    mul_1173: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(cat_81, sub_342);  sub_342 = None
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
    mul_1178: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_7, primals_6);  primals_6 = None
    unsqueeze_901: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1178, 0);  mul_1178 = None
    unsqueeze_902: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_901, 2);  unsqueeze_901 = None
    unsqueeze_903: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_902, 3);  unsqueeze_902 = None
    sub_343: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_894);  convolution_2 = unsqueeze_894 = None
    mul_1179: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_343, unsqueeze_900);  sub_343 = unsqueeze_900 = None
    sub_344: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(cat_81, mul_1179);  mul_1179 = None
    sub_345: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(sub_344, unsqueeze_897);  sub_344 = unsqueeze_897 = None
    mul_1180: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_345, unsqueeze_903);  sub_345 = unsqueeze_903 = None
    mul_1181: "f32[32]" = torch.ops.aten.mul.Tensor(sum_129, squeeze_7);  sum_129 = squeeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:126, code: x = self.conv_pw(x)
    convolution_backward_152 = torch.ops.aten.convolution_backward.default(mul_1180, relu_1, primals_133, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1180 = primals_133 = None
    getitem_892: "f32[8, 32, 112, 112]" = convolution_backward_152[0]
    getitem_893: "f32[32, 32, 1, 1]" = convolution_backward_152[1];  convolution_backward_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_55: "f32[8, 32, 112, 112]" = torch.ops.aten.alias.default(relu_1);  relu_1 = None
    alias_56: "f32[8, 32, 112, 112]" = torch.ops.aten.alias.default(alias_55);  alias_55 = None
    le_5: "b8[8, 32, 112, 112]" = torch.ops.aten.le.Scalar(alias_56, 0);  alias_56 = None
    scalar_tensor_5: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_5: "f32[8, 32, 112, 112]" = torch.ops.aten.where.self(le_5, scalar_tensor_5, getitem_892);  le_5 = scalar_tensor_5 = getitem_892 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_904: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze_3, 0);  squeeze_3 = None
    unsqueeze_905: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_904, 2);  unsqueeze_904 = None
    unsqueeze_906: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_905, 3);  unsqueeze_905 = None
    sum_130: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_5, [0, 2, 3])
    sub_346: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_906)
    mul_1182: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(where_5, sub_346);  sub_346 = None
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
    mul_1187: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_4, primals_4);  primals_4 = None
    unsqueeze_913: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1187, 0);  mul_1187 = None
    unsqueeze_914: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_913, 2);  unsqueeze_913 = None
    unsqueeze_915: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_914, 3);  unsqueeze_914 = None
    sub_347: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_906);  convolution_1 = unsqueeze_906 = None
    mul_1188: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_347, unsqueeze_912);  sub_347 = unsqueeze_912 = None
    sub_348: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(where_5, mul_1188);  where_5 = mul_1188 = None
    sub_349: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(sub_348, unsqueeze_909);  sub_348 = unsqueeze_909 = None
    mul_1189: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_349, unsqueeze_915);  sub_349 = unsqueeze_915 = None
    mul_1190: "f32[32]" = torch.ops.aten.mul.Tensor(sum_131, squeeze_4);  sum_131 = squeeze_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:123, code: x = self.conv_dw(x)
    convolution_backward_153 = torch.ops.aten.convolution_backward.default(mul_1189, relu, primals_132, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_1189 = primals_132 = None
    getitem_895: "f32[8, 32, 112, 112]" = convolution_backward_153[0]
    getitem_896: "f32[32, 1, 3, 3]" = convolution_backward_153[1];  convolution_backward_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:123, code: x = self.conv_dw(x)
    add_381: "f32[8, 32, 112, 112]" = torch.ops.aten.add.Tensor(cat_81, getitem_895);  cat_81 = getitem_895 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_58: "f32[8, 32, 112, 112]" = torch.ops.aten.alias.default(relu);  relu = None
    alias_59: "f32[8, 32, 112, 112]" = torch.ops.aten.alias.default(alias_58);  alias_58 = None
    le_6: "b8[8, 32, 112, 112]" = torch.ops.aten.le.Scalar(alias_59, 0);  alias_59 = None
    scalar_tensor_6: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_6: "f32[8, 32, 112, 112]" = torch.ops.aten.where.self(le_6, scalar_tensor_6, add_381);  le_6 = scalar_tensor_6 = add_381 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_916: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
    unsqueeze_917: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_916, 2);  unsqueeze_916 = None
    unsqueeze_918: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_917, 3);  unsqueeze_917 = None
    sum_132: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_6, [0, 2, 3])
    sub_350: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_918)
    mul_1191: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(where_6, sub_350);  sub_350 = None
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
    mul_1196: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_1, primals_2);  primals_2 = None
    unsqueeze_925: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1196, 0);  mul_1196 = None
    unsqueeze_926: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_925, 2);  unsqueeze_925 = None
    unsqueeze_927: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_926, 3);  unsqueeze_926 = None
    sub_351: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_918);  convolution = unsqueeze_918 = None
    mul_1197: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_351, unsqueeze_924);  sub_351 = unsqueeze_924 = None
    sub_352: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(where_6, mul_1197);  where_6 = mul_1197 = None
    sub_353: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(sub_352, unsqueeze_921);  sub_352 = unsqueeze_921 = None
    mul_1198: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_353, unsqueeze_927);  sub_353 = unsqueeze_927 = None
    mul_1199: "f32[32]" = torch.ops.aten.mul.Tensor(sum_133, squeeze_1);  sum_133 = squeeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_backward_154 = torch.ops.aten.convolution_backward.default(mul_1198, constant_pad_nd, primals_1, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_1198 = constant_pad_nd = primals_1 = None
    getitem_899: "f32[32, 3, 3, 3]" = convolution_backward_154[1];  convolution_backward_154 = None
    
    # No stacktrace found for following nodes
    copy_: "i64[]" = torch.ops.aten.copy_.default(primals_306, add);  primals_306 = add = None
    copy__1: "f32[32]" = torch.ops.aten.copy_.default(primals_307, add_2);  primals_307 = add_2 = None
    copy__2: "f32[32]" = torch.ops.aten.copy_.default(primals_308, add_3);  primals_308 = add_3 = None
    copy__3: "i64[]" = torch.ops.aten.copy_.default(primals_309, add_5);  primals_309 = add_5 = None
    copy__4: "f32[32]" = torch.ops.aten.copy_.default(primals_310, add_7);  primals_310 = add_7 = None
    copy__5: "f32[32]" = torch.ops.aten.copy_.default(primals_311, add_8);  primals_311 = add_8 = None
    copy__6: "i64[]" = torch.ops.aten.copy_.default(primals_312, add_10);  primals_312 = add_10 = None
    copy__7: "f32[32]" = torch.ops.aten.copy_.default(primals_313, add_12);  primals_313 = add_12 = None
    copy__8: "f32[32]" = torch.ops.aten.copy_.default(primals_314, add_13);  primals_314 = add_13 = None
    copy__9: "i64[]" = torch.ops.aten.copy_.default(primals_315, add_16);  primals_315 = add_16 = None
    copy__10: "f32[192]" = torch.ops.aten.copy_.default(primals_316, add_18);  primals_316 = add_18 = None
    copy__11: "f32[192]" = torch.ops.aten.copy_.default(primals_317, add_19);  primals_317 = add_19 = None
    copy__12: "i64[]" = torch.ops.aten.copy_.default(primals_318, add_21);  primals_318 = add_21 = None
    copy__13: "f32[192]" = torch.ops.aten.copy_.default(primals_319, add_23);  primals_319 = add_23 = None
    copy__14: "f32[192]" = torch.ops.aten.copy_.default(primals_320, add_24);  primals_320 = add_24 = None
    copy__15: "i64[]" = torch.ops.aten.copy_.default(primals_321, add_26);  primals_321 = add_26 = None
    copy__16: "f32[40]" = torch.ops.aten.copy_.default(primals_322, add_28);  primals_322 = add_28 = None
    copy__17: "f32[40]" = torch.ops.aten.copy_.default(primals_323, add_29);  primals_323 = add_29 = None
    copy__18: "i64[]" = torch.ops.aten.copy_.default(primals_324, add_31);  primals_324 = add_31 = None
    copy__19: "f32[120]" = torch.ops.aten.copy_.default(primals_325, add_33);  primals_325 = add_33 = None
    copy__20: "f32[120]" = torch.ops.aten.copy_.default(primals_326, add_34);  primals_326 = add_34 = None
    copy__21: "i64[]" = torch.ops.aten.copy_.default(primals_327, add_36);  primals_327 = add_36 = None
    copy__22: "f32[120]" = torch.ops.aten.copy_.default(primals_328, add_38);  primals_328 = add_38 = None
    copy__23: "f32[120]" = torch.ops.aten.copy_.default(primals_329, add_39);  primals_329 = add_39 = None
    copy__24: "i64[]" = torch.ops.aten.copy_.default(primals_330, add_41);  primals_330 = add_41 = None
    copy__25: "f32[40]" = torch.ops.aten.copy_.default(primals_331, add_43);  primals_331 = add_43 = None
    copy__26: "f32[40]" = torch.ops.aten.copy_.default(primals_332, add_44);  primals_332 = add_44 = None
    copy__27: "i64[]" = torch.ops.aten.copy_.default(primals_333, add_47);  primals_333 = add_47 = None
    copy__28: "f32[240]" = torch.ops.aten.copy_.default(primals_334, add_49);  primals_334 = add_49 = None
    copy__29: "f32[240]" = torch.ops.aten.copy_.default(primals_335, add_50);  primals_335 = add_50 = None
    copy__30: "i64[]" = torch.ops.aten.copy_.default(primals_336, add_52);  primals_336 = add_52 = None
    copy__31: "f32[240]" = torch.ops.aten.copy_.default(primals_337, add_54);  primals_337 = add_54 = None
    copy__32: "f32[240]" = torch.ops.aten.copy_.default(primals_338, add_55);  primals_338 = add_55 = None
    copy__33: "i64[]" = torch.ops.aten.copy_.default(primals_339, add_57);  primals_339 = add_57 = None
    copy__34: "f32[56]" = torch.ops.aten.copy_.default(primals_340, add_59);  primals_340 = add_59 = None
    copy__35: "f32[56]" = torch.ops.aten.copy_.default(primals_341, add_60);  primals_341 = add_60 = None
    copy__36: "i64[]" = torch.ops.aten.copy_.default(primals_342, add_62);  primals_342 = add_62 = None
    copy__37: "f32[336]" = torch.ops.aten.copy_.default(primals_343, add_64);  primals_343 = add_64 = None
    copy__38: "f32[336]" = torch.ops.aten.copy_.default(primals_344, add_65);  primals_344 = add_65 = None
    copy__39: "i64[]" = torch.ops.aten.copy_.default(primals_345, add_67);  primals_345 = add_67 = None
    copy__40: "f32[336]" = torch.ops.aten.copy_.default(primals_346, add_69);  primals_346 = add_69 = None
    copy__41: "f32[336]" = torch.ops.aten.copy_.default(primals_347, add_70);  primals_347 = add_70 = None
    copy__42: "i64[]" = torch.ops.aten.copy_.default(primals_348, add_72);  primals_348 = add_72 = None
    copy__43: "f32[56]" = torch.ops.aten.copy_.default(primals_349, add_74);  primals_349 = add_74 = None
    copy__44: "f32[56]" = torch.ops.aten.copy_.default(primals_350, add_75);  primals_350 = add_75 = None
    copy__45: "i64[]" = torch.ops.aten.copy_.default(primals_351, add_78);  primals_351 = add_78 = None
    copy__46: "f32[336]" = torch.ops.aten.copy_.default(primals_352, add_80);  primals_352 = add_80 = None
    copy__47: "f32[336]" = torch.ops.aten.copy_.default(primals_353, add_81);  primals_353 = add_81 = None
    copy__48: "i64[]" = torch.ops.aten.copy_.default(primals_354, add_83);  primals_354 = add_83 = None
    copy__49: "f32[336]" = torch.ops.aten.copy_.default(primals_355, add_85);  primals_355 = add_85 = None
    copy__50: "f32[336]" = torch.ops.aten.copy_.default(primals_356, add_86);  primals_356 = add_86 = None
    copy__51: "i64[]" = torch.ops.aten.copy_.default(primals_357, add_88);  primals_357 = add_88 = None
    copy__52: "f32[56]" = torch.ops.aten.copy_.default(primals_358, add_90);  primals_358 = add_90 = None
    copy__53: "f32[56]" = torch.ops.aten.copy_.default(primals_359, add_91);  primals_359 = add_91 = None
    copy__54: "i64[]" = torch.ops.aten.copy_.default(primals_360, add_94);  primals_360 = add_94 = None
    copy__55: "f32[336]" = torch.ops.aten.copy_.default(primals_361, add_96);  primals_361 = add_96 = None
    copy__56: "f32[336]" = torch.ops.aten.copy_.default(primals_362, add_97);  primals_362 = add_97 = None
    copy__57: "i64[]" = torch.ops.aten.copy_.default(primals_363, add_99);  primals_363 = add_99 = None
    copy__58: "f32[336]" = torch.ops.aten.copy_.default(primals_364, add_101);  primals_364 = add_101 = None
    copy__59: "f32[336]" = torch.ops.aten.copy_.default(primals_365, add_102);  primals_365 = add_102 = None
    copy__60: "i64[]" = torch.ops.aten.copy_.default(primals_366, add_104);  primals_366 = add_104 = None
    copy__61: "f32[56]" = torch.ops.aten.copy_.default(primals_367, add_106);  primals_367 = add_106 = None
    copy__62: "f32[56]" = torch.ops.aten.copy_.default(primals_368, add_107);  primals_368 = add_107 = None
    copy__63: "i64[]" = torch.ops.aten.copy_.default(primals_369, add_110);  primals_369 = add_110 = None
    copy__64: "f32[336]" = torch.ops.aten.copy_.default(primals_370, add_112);  primals_370 = add_112 = None
    copy__65: "f32[336]" = torch.ops.aten.copy_.default(primals_371, add_113);  primals_371 = add_113 = None
    copy__66: "i64[]" = torch.ops.aten.copy_.default(primals_372, add_115);  primals_372 = add_115 = None
    copy__67: "f32[336]" = torch.ops.aten.copy_.default(primals_373, add_117);  primals_373 = add_117 = None
    copy__68: "f32[336]" = torch.ops.aten.copy_.default(primals_374, add_118);  primals_374 = add_118 = None
    copy__69: "i64[]" = torch.ops.aten.copy_.default(primals_375, add_120);  primals_375 = add_120 = None
    copy__70: "f32[104]" = torch.ops.aten.copy_.default(primals_376, add_122);  primals_376 = add_122 = None
    copy__71: "f32[104]" = torch.ops.aten.copy_.default(primals_377, add_123);  primals_377 = add_123 = None
    copy__72: "i64[]" = torch.ops.aten.copy_.default(primals_378, add_125);  primals_378 = add_125 = None
    copy__73: "f32[624]" = torch.ops.aten.copy_.default(primals_379, add_127);  primals_379 = add_127 = None
    copy__74: "f32[624]" = torch.ops.aten.copy_.default(primals_380, add_128);  primals_380 = add_128 = None
    copy__75: "i64[]" = torch.ops.aten.copy_.default(primals_381, add_130);  primals_381 = add_130 = None
    copy__76: "f32[624]" = torch.ops.aten.copy_.default(primals_382, add_132);  primals_382 = add_132 = None
    copy__77: "f32[624]" = torch.ops.aten.copy_.default(primals_383, add_133);  primals_383 = add_133 = None
    copy__78: "i64[]" = torch.ops.aten.copy_.default(primals_384, add_135);  primals_384 = add_135 = None
    copy__79: "f32[104]" = torch.ops.aten.copy_.default(primals_385, add_137);  primals_385 = add_137 = None
    copy__80: "f32[104]" = torch.ops.aten.copy_.default(primals_386, add_138);  primals_386 = add_138 = None
    copy__81: "i64[]" = torch.ops.aten.copy_.default(primals_387, add_141);  primals_387 = add_141 = None
    copy__82: "f32[624]" = torch.ops.aten.copy_.default(primals_388, add_143);  primals_388 = add_143 = None
    copy__83: "f32[624]" = torch.ops.aten.copy_.default(primals_389, add_144);  primals_389 = add_144 = None
    copy__84: "i64[]" = torch.ops.aten.copy_.default(primals_390, add_146);  primals_390 = add_146 = None
    copy__85: "f32[624]" = torch.ops.aten.copy_.default(primals_391, add_148);  primals_391 = add_148 = None
    copy__86: "f32[624]" = torch.ops.aten.copy_.default(primals_392, add_149);  primals_392 = add_149 = None
    copy__87: "i64[]" = torch.ops.aten.copy_.default(primals_393, add_151);  primals_393 = add_151 = None
    copy__88: "f32[104]" = torch.ops.aten.copy_.default(primals_394, add_153);  primals_394 = add_153 = None
    copy__89: "f32[104]" = torch.ops.aten.copy_.default(primals_395, add_154);  primals_395 = add_154 = None
    copy__90: "i64[]" = torch.ops.aten.copy_.default(primals_396, add_157);  primals_396 = add_157 = None
    copy__91: "f32[624]" = torch.ops.aten.copy_.default(primals_397, add_159);  primals_397 = add_159 = None
    copy__92: "f32[624]" = torch.ops.aten.copy_.default(primals_398, add_160);  primals_398 = add_160 = None
    copy__93: "i64[]" = torch.ops.aten.copy_.default(primals_399, add_162);  primals_399 = add_162 = None
    copy__94: "f32[624]" = torch.ops.aten.copy_.default(primals_400, add_164);  primals_400 = add_164 = None
    copy__95: "f32[624]" = torch.ops.aten.copy_.default(primals_401, add_165);  primals_401 = add_165 = None
    copy__96: "i64[]" = torch.ops.aten.copy_.default(primals_402, add_167);  primals_402 = add_167 = None
    copy__97: "f32[104]" = torch.ops.aten.copy_.default(primals_403, add_169);  primals_403 = add_169 = None
    copy__98: "f32[104]" = torch.ops.aten.copy_.default(primals_404, add_170);  primals_404 = add_170 = None
    copy__99: "i64[]" = torch.ops.aten.copy_.default(primals_405, add_173);  primals_405 = add_173 = None
    copy__100: "f32[624]" = torch.ops.aten.copy_.default(primals_406, add_175);  primals_406 = add_175 = None
    copy__101: "f32[624]" = torch.ops.aten.copy_.default(primals_407, add_176);  primals_407 = add_176 = None
    copy__102: "i64[]" = torch.ops.aten.copy_.default(primals_408, add_178);  primals_408 = add_178 = None
    copy__103: "f32[624]" = torch.ops.aten.copy_.default(primals_409, add_180);  primals_409 = add_180 = None
    copy__104: "f32[624]" = torch.ops.aten.copy_.default(primals_410, add_181);  primals_410 = add_181 = None
    copy__105: "i64[]" = torch.ops.aten.copy_.default(primals_411, add_183);  primals_411 = add_183 = None
    copy__106: "f32[160]" = torch.ops.aten.copy_.default(primals_412, add_185);  primals_412 = add_185 = None
    copy__107: "f32[160]" = torch.ops.aten.copy_.default(primals_413, add_186);  primals_413 = add_186 = None
    copy__108: "i64[]" = torch.ops.aten.copy_.default(primals_414, add_188);  primals_414 = add_188 = None
    copy__109: "f32[480]" = torch.ops.aten.copy_.default(primals_415, add_190);  primals_415 = add_190 = None
    copy__110: "f32[480]" = torch.ops.aten.copy_.default(primals_416, add_191);  primals_416 = add_191 = None
    copy__111: "i64[]" = torch.ops.aten.copy_.default(primals_417, add_193);  primals_417 = add_193 = None
    copy__112: "f32[480]" = torch.ops.aten.copy_.default(primals_418, add_195);  primals_418 = add_195 = None
    copy__113: "f32[480]" = torch.ops.aten.copy_.default(primals_419, add_196);  primals_419 = add_196 = None
    copy__114: "i64[]" = torch.ops.aten.copy_.default(primals_420, add_198);  primals_420 = add_198 = None
    copy__115: "f32[160]" = torch.ops.aten.copy_.default(primals_421, add_200);  primals_421 = add_200 = None
    copy__116: "f32[160]" = torch.ops.aten.copy_.default(primals_422, add_201);  primals_422 = add_201 = None
    copy__117: "i64[]" = torch.ops.aten.copy_.default(primals_423, add_204);  primals_423 = add_204 = None
    copy__118: "f32[480]" = torch.ops.aten.copy_.default(primals_424, add_206);  primals_424 = add_206 = None
    copy__119: "f32[480]" = torch.ops.aten.copy_.default(primals_425, add_207);  primals_425 = add_207 = None
    copy__120: "i64[]" = torch.ops.aten.copy_.default(primals_426, add_209);  primals_426 = add_209 = None
    copy__121: "f32[480]" = torch.ops.aten.copy_.default(primals_427, add_211);  primals_427 = add_211 = None
    copy__122: "f32[480]" = torch.ops.aten.copy_.default(primals_428, add_212);  primals_428 = add_212 = None
    copy__123: "i64[]" = torch.ops.aten.copy_.default(primals_429, add_214);  primals_429 = add_214 = None
    copy__124: "f32[160]" = torch.ops.aten.copy_.default(primals_430, add_216);  primals_430 = add_216 = None
    copy__125: "f32[160]" = torch.ops.aten.copy_.default(primals_431, add_217);  primals_431 = add_217 = None
    copy__126: "i64[]" = torch.ops.aten.copy_.default(primals_432, add_220);  primals_432 = add_220 = None
    copy__127: "f32[480]" = torch.ops.aten.copy_.default(primals_433, add_222);  primals_433 = add_222 = None
    copy__128: "f32[480]" = torch.ops.aten.copy_.default(primals_434, add_223);  primals_434 = add_223 = None
    copy__129: "i64[]" = torch.ops.aten.copy_.default(primals_435, add_225);  primals_435 = add_225 = None
    copy__130: "f32[480]" = torch.ops.aten.copy_.default(primals_436, add_227);  primals_436 = add_227 = None
    copy__131: "f32[480]" = torch.ops.aten.copy_.default(primals_437, add_228);  primals_437 = add_228 = None
    copy__132: "i64[]" = torch.ops.aten.copy_.default(primals_438, add_230);  primals_438 = add_230 = None
    copy__133: "f32[160]" = torch.ops.aten.copy_.default(primals_439, add_232);  primals_439 = add_232 = None
    copy__134: "f32[160]" = torch.ops.aten.copy_.default(primals_440, add_233);  primals_440 = add_233 = None
    copy__135: "i64[]" = torch.ops.aten.copy_.default(primals_441, add_236);  primals_441 = add_236 = None
    copy__136: "f32[960]" = torch.ops.aten.copy_.default(primals_442, add_238);  primals_442 = add_238 = None
    copy__137: "f32[960]" = torch.ops.aten.copy_.default(primals_443, add_239);  primals_443 = add_239 = None
    copy__138: "i64[]" = torch.ops.aten.copy_.default(primals_444, add_241);  primals_444 = add_241 = None
    copy__139: "f32[960]" = torch.ops.aten.copy_.default(primals_445, add_243);  primals_445 = add_243 = None
    copy__140: "f32[960]" = torch.ops.aten.copy_.default(primals_446, add_244);  primals_446 = add_244 = None
    copy__141: "i64[]" = torch.ops.aten.copy_.default(primals_447, add_246);  primals_447 = add_246 = None
    copy__142: "f32[264]" = torch.ops.aten.copy_.default(primals_448, add_248);  primals_448 = add_248 = None
    copy__143: "f32[264]" = torch.ops.aten.copy_.default(primals_449, add_249);  primals_449 = add_249 = None
    copy__144: "i64[]" = torch.ops.aten.copy_.default(primals_450, add_251);  primals_450 = add_251 = None
    copy__145: "f32[1584]" = torch.ops.aten.copy_.default(primals_451, add_253);  primals_451 = add_253 = None
    copy__146: "f32[1584]" = torch.ops.aten.copy_.default(primals_452, add_254);  primals_452 = add_254 = None
    copy__147: "i64[]" = torch.ops.aten.copy_.default(primals_453, add_256);  primals_453 = add_256 = None
    copy__148: "f32[1584]" = torch.ops.aten.copy_.default(primals_454, add_258);  primals_454 = add_258 = None
    copy__149: "f32[1584]" = torch.ops.aten.copy_.default(primals_455, add_259);  primals_455 = add_259 = None
    copy__150: "i64[]" = torch.ops.aten.copy_.default(primals_456, add_261);  primals_456 = add_261 = None
    copy__151: "f32[264]" = torch.ops.aten.copy_.default(primals_457, add_263);  primals_457 = add_263 = None
    copy__152: "f32[264]" = torch.ops.aten.copy_.default(primals_458, add_264);  primals_458 = add_264 = None
    copy__153: "i64[]" = torch.ops.aten.copy_.default(primals_459, add_267);  primals_459 = add_267 = None
    copy__154: "f32[1584]" = torch.ops.aten.copy_.default(primals_460, add_269);  primals_460 = add_269 = None
    copy__155: "f32[1584]" = torch.ops.aten.copy_.default(primals_461, add_270);  primals_461 = add_270 = None
    copy__156: "i64[]" = torch.ops.aten.copy_.default(primals_462, add_272);  primals_462 = add_272 = None
    copy__157: "f32[1584]" = torch.ops.aten.copy_.default(primals_463, add_274);  primals_463 = add_274 = None
    copy__158: "f32[1584]" = torch.ops.aten.copy_.default(primals_464, add_275);  primals_464 = add_275 = None
    copy__159: "i64[]" = torch.ops.aten.copy_.default(primals_465, add_277);  primals_465 = add_277 = None
    copy__160: "f32[264]" = torch.ops.aten.copy_.default(primals_466, add_279);  primals_466 = add_279 = None
    copy__161: "f32[264]" = torch.ops.aten.copy_.default(primals_467, add_280);  primals_467 = add_280 = None
    copy__162: "i64[]" = torch.ops.aten.copy_.default(primals_468, add_283);  primals_468 = add_283 = None
    copy__163: "f32[1584]" = torch.ops.aten.copy_.default(primals_469, add_285);  primals_469 = add_285 = None
    copy__164: "f32[1584]" = torch.ops.aten.copy_.default(primals_470, add_286);  primals_470 = add_286 = None
    copy__165: "i64[]" = torch.ops.aten.copy_.default(primals_471, add_288);  primals_471 = add_288 = None
    copy__166: "f32[1584]" = torch.ops.aten.copy_.default(primals_472, add_290);  primals_472 = add_290 = None
    copy__167: "f32[1584]" = torch.ops.aten.copy_.default(primals_473, add_291);  primals_473 = add_291 = None
    copy__168: "i64[]" = torch.ops.aten.copy_.default(primals_474, add_293);  primals_474 = add_293 = None
    copy__169: "f32[264]" = torch.ops.aten.copy_.default(primals_475, add_295);  primals_475 = add_295 = None
    copy__170: "f32[264]" = torch.ops.aten.copy_.default(primals_476, add_296);  primals_476 = add_296 = None
    copy__171: "i64[]" = torch.ops.aten.copy_.default(primals_477, add_299);  primals_477 = add_299 = None
    copy__172: "f32[1536]" = torch.ops.aten.copy_.default(primals_478, add_301);  primals_478 = add_301 = None
    copy__173: "f32[1536]" = torch.ops.aten.copy_.default(primals_479, add_302);  primals_479 = add_302 = None
    return pytree.tree_unflatten([addmm, getitem_899, mul_1199, sum_132, mul_1190, sum_130, mul_1181, sum_128, mul_1172, sum_126, getitem_884, getitem_881, getitem_878, mul_1163, sum_124, mul_1154, sum_122, mul_1145, sum_120, mul_1136, sum_118, mul_1127, sum_116, mul_1118, sum_114, getitem_851, getitem_848, getitem_845, getitem_842, mul_1106, sum_112, mul_1087, sum_109, mul_1078, sum_107, mul_1066, sum_105, mul_1047, sum_102, mul_1038, sum_100, mul_1026, sum_98, mul_1007, sum_95, mul_998, sum_93, mul_986, sum_91, mul_967, sum_88, mul_958, sum_86, getitem_755, getitem_752, getitem_749, mul_946, sum_84, mul_927, sum_81, mul_918, sum_79, mul_906, sum_77, mul_887, sum_74, mul_878, sum_72, mul_866, sum_70, mul_847, sum_67, mul_838, sum_65, mul_826, sum_63, mul_807, sum_60, mul_798, sum_58, mul_786, sum_56, mul_767, sum_53, mul_758, sum_51, mul_746, sum_49, mul_727, sum_46, mul_718, sum_44, mul_706, sum_42, mul_687, sum_39, mul_678, sum_37, mul_666, sum_35, mul_647, sum_32, mul_638, sum_30, getitem_539, getitem_536, getitem_533, getitem_530, mul_626, sum_28, mul_607, sum_25, mul_598, sum_23, mul_586, sum_21, mul_567, sum_18, mul_558, sum_16, mul_546, sum_14, mul_527, sum_11, mul_518, sum_9, mul_506, sum_7, mul_487, sum_4, mul_478, sum_2, getitem_896, getitem_893, getitem_890, getitem_887, getitem_875, getitem_872, getitem_869, getitem_866, getitem_863, getitem_860, getitem_857, getitem_854, getitem_839, getitem_840, getitem_836, getitem_837, getitem_833, getitem_830, getitem_827, getitem_824, getitem_821, getitem_818, getitem_819, getitem_815, getitem_816, getitem_812, getitem_809, getitem_806, getitem_803, getitem_800, getitem_797, getitem_794, getitem_795, getitem_791, getitem_792, getitem_788, getitem_785, getitem_782, getitem_779, getitem_776, getitem_773, getitem_770, getitem_771, getitem_767, getitem_768, getitem_764, getitem_761, getitem_758, getitem_746, getitem_747, getitem_743, getitem_744, getitem_740, getitem_737, getitem_734, getitem_731, getitem_728, getitem_725, getitem_722, getitem_719, getitem_720, getitem_716, getitem_717, getitem_713, getitem_710, getitem_707, getitem_704, getitem_701, getitem_698, getitem_695, getitem_692, getitem_689, getitem_690, getitem_686, getitem_687, getitem_683, getitem_680, getitem_677, getitem_674, getitem_671, getitem_668, getitem_665, getitem_662, getitem_659, getitem_660, getitem_656, getitem_657, getitem_653, getitem_650, getitem_647, getitem_644, getitem_641, getitem_642, getitem_638, getitem_639, getitem_635, getitem_632, getitem_629, getitem_626, getitem_623, getitem_620, getitem_617, getitem_614, getitem_615, getitem_611, getitem_612, getitem_608, getitem_605, getitem_602, getitem_599, getitem_596, getitem_593, getitem_590, getitem_587, getitem_584, getitem_585, getitem_581, getitem_582, getitem_578, getitem_575, getitem_572, getitem_569, getitem_566, getitem_563, getitem_560, getitem_557, getitem_554, getitem_555, getitem_551, getitem_552, getitem_548, getitem_545, getitem_542, getitem_527, getitem_528, getitem_524, getitem_525, getitem_521, getitem_518, getitem_515, getitem_512, getitem_509, getitem_506, getitem_503, getitem_504, getitem_500, getitem_501, getitem_497, getitem_494, getitem_491, getitem_488, getitem_485, getitem_482, getitem_479, getitem_476, getitem_477, getitem_473, getitem_474, getitem_470, getitem_467, getitem_464, getitem_461, getitem_458, getitem_455, getitem_452, getitem_449, getitem_450, getitem_446, getitem_447, getitem_443, getitem_440, getitem_437, permute_4, view_1, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None], self._out_spec)
    