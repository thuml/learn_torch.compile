from __future__ import annotations



def forward(self, primals, tangents):
    primals_1: "f32[64, 3, 7, 7]"; primals_2: "f32[64]"; primals_3: "f32[64]"; primals_4: "f32[128, 64, 1, 1]"; primals_5: "f32[128]"; primals_6: "f32[128]"; primals_7: "f32[32, 4, 3, 3]"; primals_8: "f32[32]"; primals_9: "f32[32]"; primals_10: "f32[32, 4, 3, 3]"; primals_11: "f32[32]"; primals_12: "f32[32]"; primals_13: "f32[32, 4, 3, 3]"; primals_14: "f32[32]"; primals_15: "f32[32]"; primals_16: "f32[256, 128, 1, 1]"; primals_17: "f32[256]"; primals_18: "f32[256]"; primals_19: "f32[256, 64, 1, 1]"; primals_20: "f32[256]"; primals_21: "f32[256]"; primals_22: "f32[128, 256, 1, 1]"; primals_23: "f32[128]"; primals_24: "f32[128]"; primals_25: "f32[32, 4, 3, 3]"; primals_26: "f32[32]"; primals_27: "f32[32]"; primals_28: "f32[32, 4, 3, 3]"; primals_29: "f32[32]"; primals_30: "f32[32]"; primals_31: "f32[32, 4, 3, 3]"; primals_32: "f32[32]"; primals_33: "f32[32]"; primals_34: "f32[256, 128, 1, 1]"; primals_35: "f32[256]"; primals_36: "f32[256]"; primals_37: "f32[128, 256, 1, 1]"; primals_38: "f32[128]"; primals_39: "f32[128]"; primals_40: "f32[32, 4, 3, 3]"; primals_41: "f32[32]"; primals_42: "f32[32]"; primals_43: "f32[32, 4, 3, 3]"; primals_44: "f32[32]"; primals_45: "f32[32]"; primals_46: "f32[32, 4, 3, 3]"; primals_47: "f32[32]"; primals_48: "f32[32]"; primals_49: "f32[256, 128, 1, 1]"; primals_50: "f32[256]"; primals_51: "f32[256]"; primals_52: "f32[256, 256, 1, 1]"; primals_53: "f32[256]"; primals_54: "f32[256]"; primals_55: "f32[64, 8, 3, 3]"; primals_56: "f32[64]"; primals_57: "f32[64]"; primals_58: "f32[64, 8, 3, 3]"; primals_59: "f32[64]"; primals_60: "f32[64]"; primals_61: "f32[64, 8, 3, 3]"; primals_62: "f32[64]"; primals_63: "f32[64]"; primals_64: "f32[512, 256, 1, 1]"; primals_65: "f32[512]"; primals_66: "f32[512]"; primals_67: "f32[512, 256, 1, 1]"; primals_68: "f32[512]"; primals_69: "f32[512]"; primals_70: "f32[256, 512, 1, 1]"; primals_71: "f32[256]"; primals_72: "f32[256]"; primals_73: "f32[64, 8, 3, 3]"; primals_74: "f32[64]"; primals_75: "f32[64]"; primals_76: "f32[64, 8, 3, 3]"; primals_77: "f32[64]"; primals_78: "f32[64]"; primals_79: "f32[64, 8, 3, 3]"; primals_80: "f32[64]"; primals_81: "f32[64]"; primals_82: "f32[512, 256, 1, 1]"; primals_83: "f32[512]"; primals_84: "f32[512]"; primals_85: "f32[256, 512, 1, 1]"; primals_86: "f32[256]"; primals_87: "f32[256]"; primals_88: "f32[64, 8, 3, 3]"; primals_89: "f32[64]"; primals_90: "f32[64]"; primals_91: "f32[64, 8, 3, 3]"; primals_92: "f32[64]"; primals_93: "f32[64]"; primals_94: "f32[64, 8, 3, 3]"; primals_95: "f32[64]"; primals_96: "f32[64]"; primals_97: "f32[512, 256, 1, 1]"; primals_98: "f32[512]"; primals_99: "f32[512]"; primals_100: "f32[256, 512, 1, 1]"; primals_101: "f32[256]"; primals_102: "f32[256]"; primals_103: "f32[64, 8, 3, 3]"; primals_104: "f32[64]"; primals_105: "f32[64]"; primals_106: "f32[64, 8, 3, 3]"; primals_107: "f32[64]"; primals_108: "f32[64]"; primals_109: "f32[64, 8, 3, 3]"; primals_110: "f32[64]"; primals_111: "f32[64]"; primals_112: "f32[512, 256, 1, 1]"; primals_113: "f32[512]"; primals_114: "f32[512]"; primals_115: "f32[512, 512, 1, 1]"; primals_116: "f32[512]"; primals_117: "f32[512]"; primals_118: "f32[128, 16, 3, 3]"; primals_119: "f32[128]"; primals_120: "f32[128]"; primals_121: "f32[128, 16, 3, 3]"; primals_122: "f32[128]"; primals_123: "f32[128]"; primals_124: "f32[128, 16, 3, 3]"; primals_125: "f32[128]"; primals_126: "f32[128]"; primals_127: "f32[1024, 512, 1, 1]"; primals_128: "f32[1024]"; primals_129: "f32[1024]"; primals_130: "f32[1024, 512, 1, 1]"; primals_131: "f32[1024]"; primals_132: "f32[1024]"; primals_133: "f32[512, 1024, 1, 1]"; primals_134: "f32[512]"; primals_135: "f32[512]"; primals_136: "f32[128, 16, 3, 3]"; primals_137: "f32[128]"; primals_138: "f32[128]"; primals_139: "f32[128, 16, 3, 3]"; primals_140: "f32[128]"; primals_141: "f32[128]"; primals_142: "f32[128, 16, 3, 3]"; primals_143: "f32[128]"; primals_144: "f32[128]"; primals_145: "f32[1024, 512, 1, 1]"; primals_146: "f32[1024]"; primals_147: "f32[1024]"; primals_148: "f32[512, 1024, 1, 1]"; primals_149: "f32[512]"; primals_150: "f32[512]"; primals_151: "f32[128, 16, 3, 3]"; primals_152: "f32[128]"; primals_153: "f32[128]"; primals_154: "f32[128, 16, 3, 3]"; primals_155: "f32[128]"; primals_156: "f32[128]"; primals_157: "f32[128, 16, 3, 3]"; primals_158: "f32[128]"; primals_159: "f32[128]"; primals_160: "f32[1024, 512, 1, 1]"; primals_161: "f32[1024]"; primals_162: "f32[1024]"; primals_163: "f32[512, 1024, 1, 1]"; primals_164: "f32[512]"; primals_165: "f32[512]"; primals_166: "f32[128, 16, 3, 3]"; primals_167: "f32[128]"; primals_168: "f32[128]"; primals_169: "f32[128, 16, 3, 3]"; primals_170: "f32[128]"; primals_171: "f32[128]"; primals_172: "f32[128, 16, 3, 3]"; primals_173: "f32[128]"; primals_174: "f32[128]"; primals_175: "f32[1024, 512, 1, 1]"; primals_176: "f32[1024]"; primals_177: "f32[1024]"; primals_178: "f32[512, 1024, 1, 1]"; primals_179: "f32[512]"; primals_180: "f32[512]"; primals_181: "f32[128, 16, 3, 3]"; primals_182: "f32[128]"; primals_183: "f32[128]"; primals_184: "f32[128, 16, 3, 3]"; primals_185: "f32[128]"; primals_186: "f32[128]"; primals_187: "f32[128, 16, 3, 3]"; primals_188: "f32[128]"; primals_189: "f32[128]"; primals_190: "f32[1024, 512, 1, 1]"; primals_191: "f32[1024]"; primals_192: "f32[1024]"; primals_193: "f32[512, 1024, 1, 1]"; primals_194: "f32[512]"; primals_195: "f32[512]"; primals_196: "f32[128, 16, 3, 3]"; primals_197: "f32[128]"; primals_198: "f32[128]"; primals_199: "f32[128, 16, 3, 3]"; primals_200: "f32[128]"; primals_201: "f32[128]"; primals_202: "f32[128, 16, 3, 3]"; primals_203: "f32[128]"; primals_204: "f32[128]"; primals_205: "f32[1024, 512, 1, 1]"; primals_206: "f32[1024]"; primals_207: "f32[1024]"; primals_208: "f32[1024, 1024, 1, 1]"; primals_209: "f32[1024]"; primals_210: "f32[1024]"; primals_211: "f32[256, 32, 3, 3]"; primals_212: "f32[256]"; primals_213: "f32[256]"; primals_214: "f32[256, 32, 3, 3]"; primals_215: "f32[256]"; primals_216: "f32[256]"; primals_217: "f32[256, 32, 3, 3]"; primals_218: "f32[256]"; primals_219: "f32[256]"; primals_220: "f32[2048, 1024, 1, 1]"; primals_221: "f32[2048]"; primals_222: "f32[2048]"; primals_223: "f32[2048, 1024, 1, 1]"; primals_224: "f32[2048]"; primals_225: "f32[2048]"; primals_226: "f32[1024, 2048, 1, 1]"; primals_227: "f32[1024]"; primals_228: "f32[1024]"; primals_229: "f32[256, 32, 3, 3]"; primals_230: "f32[256]"; primals_231: "f32[256]"; primals_232: "f32[256, 32, 3, 3]"; primals_233: "f32[256]"; primals_234: "f32[256]"; primals_235: "f32[256, 32, 3, 3]"; primals_236: "f32[256]"; primals_237: "f32[256]"; primals_238: "f32[2048, 1024, 1, 1]"; primals_239: "f32[2048]"; primals_240: "f32[2048]"; primals_241: "f32[1024, 2048, 1, 1]"; primals_242: "f32[1024]"; primals_243: "f32[1024]"; primals_244: "f32[256, 32, 3, 3]"; primals_245: "f32[256]"; primals_246: "f32[256]"; primals_247: "f32[256, 32, 3, 3]"; primals_248: "f32[256]"; primals_249: "f32[256]"; primals_250: "f32[256, 32, 3, 3]"; primals_251: "f32[256]"; primals_252: "f32[256]"; primals_253: "f32[2048, 1024, 1, 1]"; primals_254: "f32[2048]"; primals_255: "f32[2048]"; primals_256: "f32[1000, 2048]"; primals_257: "f32[1000]"; primals_258: "f32[64]"; primals_259: "f32[64]"; primals_260: "i64[]"; primals_261: "f32[128]"; primals_262: "f32[128]"; primals_263: "i64[]"; primals_264: "f32[32]"; primals_265: "f32[32]"; primals_266: "i64[]"; primals_267: "f32[32]"; primals_268: "f32[32]"; primals_269: "i64[]"; primals_270: "f32[32]"; primals_271: "f32[32]"; primals_272: "i64[]"; primals_273: "f32[256]"; primals_274: "f32[256]"; primals_275: "i64[]"; primals_276: "f32[256]"; primals_277: "f32[256]"; primals_278: "i64[]"; primals_279: "f32[128]"; primals_280: "f32[128]"; primals_281: "i64[]"; primals_282: "f32[32]"; primals_283: "f32[32]"; primals_284: "i64[]"; primals_285: "f32[32]"; primals_286: "f32[32]"; primals_287: "i64[]"; primals_288: "f32[32]"; primals_289: "f32[32]"; primals_290: "i64[]"; primals_291: "f32[256]"; primals_292: "f32[256]"; primals_293: "i64[]"; primals_294: "f32[128]"; primals_295: "f32[128]"; primals_296: "i64[]"; primals_297: "f32[32]"; primals_298: "f32[32]"; primals_299: "i64[]"; primals_300: "f32[32]"; primals_301: "f32[32]"; primals_302: "i64[]"; primals_303: "f32[32]"; primals_304: "f32[32]"; primals_305: "i64[]"; primals_306: "f32[256]"; primals_307: "f32[256]"; primals_308: "i64[]"; primals_309: "f32[256]"; primals_310: "f32[256]"; primals_311: "i64[]"; primals_312: "f32[64]"; primals_313: "f32[64]"; primals_314: "i64[]"; primals_315: "f32[64]"; primals_316: "f32[64]"; primals_317: "i64[]"; primals_318: "f32[64]"; primals_319: "f32[64]"; primals_320: "i64[]"; primals_321: "f32[512]"; primals_322: "f32[512]"; primals_323: "i64[]"; primals_324: "f32[512]"; primals_325: "f32[512]"; primals_326: "i64[]"; primals_327: "f32[256]"; primals_328: "f32[256]"; primals_329: "i64[]"; primals_330: "f32[64]"; primals_331: "f32[64]"; primals_332: "i64[]"; primals_333: "f32[64]"; primals_334: "f32[64]"; primals_335: "i64[]"; primals_336: "f32[64]"; primals_337: "f32[64]"; primals_338: "i64[]"; primals_339: "f32[512]"; primals_340: "f32[512]"; primals_341: "i64[]"; primals_342: "f32[256]"; primals_343: "f32[256]"; primals_344: "i64[]"; primals_345: "f32[64]"; primals_346: "f32[64]"; primals_347: "i64[]"; primals_348: "f32[64]"; primals_349: "f32[64]"; primals_350: "i64[]"; primals_351: "f32[64]"; primals_352: "f32[64]"; primals_353: "i64[]"; primals_354: "f32[512]"; primals_355: "f32[512]"; primals_356: "i64[]"; primals_357: "f32[256]"; primals_358: "f32[256]"; primals_359: "i64[]"; primals_360: "f32[64]"; primals_361: "f32[64]"; primals_362: "i64[]"; primals_363: "f32[64]"; primals_364: "f32[64]"; primals_365: "i64[]"; primals_366: "f32[64]"; primals_367: "f32[64]"; primals_368: "i64[]"; primals_369: "f32[512]"; primals_370: "f32[512]"; primals_371: "i64[]"; primals_372: "f32[512]"; primals_373: "f32[512]"; primals_374: "i64[]"; primals_375: "f32[128]"; primals_376: "f32[128]"; primals_377: "i64[]"; primals_378: "f32[128]"; primals_379: "f32[128]"; primals_380: "i64[]"; primals_381: "f32[128]"; primals_382: "f32[128]"; primals_383: "i64[]"; primals_384: "f32[1024]"; primals_385: "f32[1024]"; primals_386: "i64[]"; primals_387: "f32[1024]"; primals_388: "f32[1024]"; primals_389: "i64[]"; primals_390: "f32[512]"; primals_391: "f32[512]"; primals_392: "i64[]"; primals_393: "f32[128]"; primals_394: "f32[128]"; primals_395: "i64[]"; primals_396: "f32[128]"; primals_397: "f32[128]"; primals_398: "i64[]"; primals_399: "f32[128]"; primals_400: "f32[128]"; primals_401: "i64[]"; primals_402: "f32[1024]"; primals_403: "f32[1024]"; primals_404: "i64[]"; primals_405: "f32[512]"; primals_406: "f32[512]"; primals_407: "i64[]"; primals_408: "f32[128]"; primals_409: "f32[128]"; primals_410: "i64[]"; primals_411: "f32[128]"; primals_412: "f32[128]"; primals_413: "i64[]"; primals_414: "f32[128]"; primals_415: "f32[128]"; primals_416: "i64[]"; primals_417: "f32[1024]"; primals_418: "f32[1024]"; primals_419: "i64[]"; primals_420: "f32[512]"; primals_421: "f32[512]"; primals_422: "i64[]"; primals_423: "f32[128]"; primals_424: "f32[128]"; primals_425: "i64[]"; primals_426: "f32[128]"; primals_427: "f32[128]"; primals_428: "i64[]"; primals_429: "f32[128]"; primals_430: "f32[128]"; primals_431: "i64[]"; primals_432: "f32[1024]"; primals_433: "f32[1024]"; primals_434: "i64[]"; primals_435: "f32[512]"; primals_436: "f32[512]"; primals_437: "i64[]"; primals_438: "f32[128]"; primals_439: "f32[128]"; primals_440: "i64[]"; primals_441: "f32[128]"; primals_442: "f32[128]"; primals_443: "i64[]"; primals_444: "f32[128]"; primals_445: "f32[128]"; primals_446: "i64[]"; primals_447: "f32[1024]"; primals_448: "f32[1024]"; primals_449: "i64[]"; primals_450: "f32[512]"; primals_451: "f32[512]"; primals_452: "i64[]"; primals_453: "f32[128]"; primals_454: "f32[128]"; primals_455: "i64[]"; primals_456: "f32[128]"; primals_457: "f32[128]"; primals_458: "i64[]"; primals_459: "f32[128]"; primals_460: "f32[128]"; primals_461: "i64[]"; primals_462: "f32[1024]"; primals_463: "f32[1024]"; primals_464: "i64[]"; primals_465: "f32[1024]"; primals_466: "f32[1024]"; primals_467: "i64[]"; primals_468: "f32[256]"; primals_469: "f32[256]"; primals_470: "i64[]"; primals_471: "f32[256]"; primals_472: "f32[256]"; primals_473: "i64[]"; primals_474: "f32[256]"; primals_475: "f32[256]"; primals_476: "i64[]"; primals_477: "f32[2048]"; primals_478: "f32[2048]"; primals_479: "i64[]"; primals_480: "f32[2048]"; primals_481: "f32[2048]"; primals_482: "i64[]"; primals_483: "f32[1024]"; primals_484: "f32[1024]"; primals_485: "i64[]"; primals_486: "f32[256]"; primals_487: "f32[256]"; primals_488: "i64[]"; primals_489: "f32[256]"; primals_490: "f32[256]"; primals_491: "i64[]"; primals_492: "f32[256]"; primals_493: "f32[256]"; primals_494: "i64[]"; primals_495: "f32[2048]"; primals_496: "f32[2048]"; primals_497: "i64[]"; primals_498: "f32[1024]"; primals_499: "f32[1024]"; primals_500: "i64[]"; primals_501: "f32[256]"; primals_502: "f32[256]"; primals_503: "i64[]"; primals_504: "f32[256]"; primals_505: "f32[256]"; primals_506: "i64[]"; primals_507: "f32[256]"; primals_508: "f32[256]"; primals_509: "i64[]"; primals_510: "f32[2048]"; primals_511: "f32[2048]"; primals_512: "i64[]"; primals_513: "f32[8, 3, 224, 224]"; tangents_1: "f32[8, 1000]"; 

    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, tangents_1, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:520, code: x = self.conv1(x)
    convolution: "f32[8, 64, 112, 112]" = torch.ops.aten.convolution.default(primals_513, primals_1, None, [2, 2], [3, 3], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:521, code: x = self.bn1(x)
    add: "i64[]" = torch.ops.aten.add.Tensor(primals_260, 1)
    var_mean = torch.ops.aten.var_mean.correction(convolution, [0, 2, 3], correction = 0, keepdim = True)
    getitem: "f32[1, 64, 1, 1]" = var_mean[0]
    getitem_1: "f32[1, 64, 1, 1]" = var_mean[1];  var_mean = None
    add_1: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05)
    rsqrt: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
    sub: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, getitem_1)
    mul: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
    squeeze: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
    squeeze_1: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt, [0, 2, 3]);  rsqrt = None
    mul_1: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze, 0.1)
    mul_2: "f32[64]" = torch.ops.aten.mul.Tensor(primals_258, 0.9)
    add_2: "f32[64]" = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
    squeeze_2: "f32[64]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
    mul_3: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_2, 1.00000996502277);  squeeze_2 = None
    mul_4: "f32[64]" = torch.ops.aten.mul.Tensor(mul_3, 0.1);  mul_3 = None
    mul_5: "f32[64]" = torch.ops.aten.mul.Tensor(primals_259, 0.9)
    add_3: "f32[64]" = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
    unsqueeze: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_2, -1)
    unsqueeze_1: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
    mul_6: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
    unsqueeze_2: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_3, -1);  primals_3 = None
    unsqueeze_3: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
    add_4: "f32[8, 64, 112, 112]" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_3);  mul_6 = unsqueeze_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:522, code: x = self.act1(x)
    relu: "f32[8, 64, 112, 112]" = torch.ops.aten.relu.default(add_4);  add_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:523, code: x = self.maxpool(x)
    max_pool2d_with_indices = torch.ops.aten.max_pool2d_with_indices.default(relu, [3, 3], [2, 2], [1, 1])
    getitem_2: "f32[8, 64, 56, 56]" = max_pool2d_with_indices[0]
    getitem_3: "i64[8, 64, 56, 56]" = max_pool2d_with_indices[1];  max_pool2d_with_indices = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_1: "f32[8, 128, 56, 56]" = torch.ops.aten.convolution.default(getitem_2, primals_4, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    add_5: "i64[]" = torch.ops.aten.add.Tensor(primals_263, 1)
    var_mean_1 = torch.ops.aten.var_mean.correction(convolution_1, [0, 2, 3], correction = 0, keepdim = True)
    getitem_4: "f32[1, 128, 1, 1]" = var_mean_1[0]
    getitem_5: "f32[1, 128, 1, 1]" = var_mean_1[1];  var_mean_1 = None
    add_6: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05)
    rsqrt_1: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
    sub_1: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_1, getitem_5)
    mul_7: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
    squeeze_3: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2, 3]);  getitem_5 = None
    squeeze_4: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_1, [0, 2, 3]);  rsqrt_1 = None
    mul_8: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_3, 0.1)
    mul_9: "f32[128]" = torch.ops.aten.mul.Tensor(primals_261, 0.9)
    add_7: "f32[128]" = torch.ops.aten.add.Tensor(mul_8, mul_9);  mul_8 = mul_9 = None
    squeeze_5: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_4, [0, 2, 3]);  getitem_4 = None
    mul_10: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_5, 1.0000398612827361);  squeeze_5 = None
    mul_11: "f32[128]" = torch.ops.aten.mul.Tensor(mul_10, 0.1);  mul_10 = None
    mul_12: "f32[128]" = torch.ops.aten.mul.Tensor(primals_262, 0.9)
    add_8: "f32[128]" = torch.ops.aten.add.Tensor(mul_11, mul_12);  mul_11 = mul_12 = None
    unsqueeze_4: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_5, -1)
    unsqueeze_5: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
    mul_13: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_5);  mul_7 = unsqueeze_5 = None
    unsqueeze_6: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_6, -1);  primals_6 = None
    unsqueeze_7: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
    add_9: "f32[8, 128, 56, 56]" = torch.ops.aten.add.Tensor(mul_13, unsqueeze_7);  mul_13 = unsqueeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_1: "f32[8, 128, 56, 56]" = torch.ops.aten.relu.default(add_9);  add_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_1 = torch.ops.aten.split_with_sizes.default(relu_1, [32, 32, 32, 32], 1)
    getitem_10: "f32[8, 32, 56, 56]" = split_with_sizes_1[0];  split_with_sizes_1 = None
    convolution_2: "f32[8, 32, 56, 56]" = torch.ops.aten.convolution.default(getitem_10, primals_7, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_10: "i64[]" = torch.ops.aten.add.Tensor(primals_266, 1)
    var_mean_2 = torch.ops.aten.var_mean.correction(convolution_2, [0, 2, 3], correction = 0, keepdim = True)
    getitem_14: "f32[1, 32, 1, 1]" = var_mean_2[0]
    getitem_15: "f32[1, 32, 1, 1]" = var_mean_2[1];  var_mean_2 = None
    add_11: "f32[1, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05)
    rsqrt_2: "f32[1, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
    sub_2: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_2, getitem_15)
    mul_14: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
    squeeze_6: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_15, [0, 2, 3]);  getitem_15 = None
    squeeze_7: "f32[32]" = torch.ops.aten.squeeze.dims(rsqrt_2, [0, 2, 3]);  rsqrt_2 = None
    mul_15: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_6, 0.1)
    mul_16: "f32[32]" = torch.ops.aten.mul.Tensor(primals_264, 0.9)
    add_12: "f32[32]" = torch.ops.aten.add.Tensor(mul_15, mul_16);  mul_15 = mul_16 = None
    squeeze_8: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_14, [0, 2, 3]);  getitem_14 = None
    mul_17: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_8, 1.0000398612827361);  squeeze_8 = None
    mul_18: "f32[32]" = torch.ops.aten.mul.Tensor(mul_17, 0.1);  mul_17 = None
    mul_19: "f32[32]" = torch.ops.aten.mul.Tensor(primals_265, 0.9)
    add_13: "f32[32]" = torch.ops.aten.add.Tensor(mul_18, mul_19);  mul_18 = mul_19 = None
    unsqueeze_8: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_8, -1)
    unsqueeze_9: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
    mul_20: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(mul_14, unsqueeze_9);  mul_14 = unsqueeze_9 = None
    unsqueeze_10: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_9, -1);  primals_9 = None
    unsqueeze_11: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
    add_14: "f32[8, 32, 56, 56]" = torch.ops.aten.add.Tensor(mul_20, unsqueeze_11);  mul_20 = unsqueeze_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_2: "f32[8, 32, 56, 56]" = torch.ops.aten.relu.default(add_14);  add_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_2 = torch.ops.aten.split_with_sizes.default(relu_1, [32, 32, 32, 32], 1)
    getitem_17: "f32[8, 32, 56, 56]" = split_with_sizes_2[1];  split_with_sizes_2 = None
    convolution_3: "f32[8, 32, 56, 56]" = torch.ops.aten.convolution.default(getitem_17, primals_10, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_15: "i64[]" = torch.ops.aten.add.Tensor(primals_269, 1)
    var_mean_3 = torch.ops.aten.var_mean.correction(convolution_3, [0, 2, 3], correction = 0, keepdim = True)
    getitem_20: "f32[1, 32, 1, 1]" = var_mean_3[0]
    getitem_21: "f32[1, 32, 1, 1]" = var_mean_3[1];  var_mean_3 = None
    add_16: "f32[1, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05)
    rsqrt_3: "f32[1, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
    sub_3: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_3, getitem_21)
    mul_21: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
    squeeze_9: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_21, [0, 2, 3]);  getitem_21 = None
    squeeze_10: "f32[32]" = torch.ops.aten.squeeze.dims(rsqrt_3, [0, 2, 3]);  rsqrt_3 = None
    mul_22: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_9, 0.1)
    mul_23: "f32[32]" = torch.ops.aten.mul.Tensor(primals_267, 0.9)
    add_17: "f32[32]" = torch.ops.aten.add.Tensor(mul_22, mul_23);  mul_22 = mul_23 = None
    squeeze_11: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_20, [0, 2, 3]);  getitem_20 = None
    mul_24: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_11, 1.0000398612827361);  squeeze_11 = None
    mul_25: "f32[32]" = torch.ops.aten.mul.Tensor(mul_24, 0.1);  mul_24 = None
    mul_26: "f32[32]" = torch.ops.aten.mul.Tensor(primals_268, 0.9)
    add_18: "f32[32]" = torch.ops.aten.add.Tensor(mul_25, mul_26);  mul_25 = mul_26 = None
    unsqueeze_12: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_11, -1)
    unsqueeze_13: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
    mul_27: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(mul_21, unsqueeze_13);  mul_21 = unsqueeze_13 = None
    unsqueeze_14: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_12, -1);  primals_12 = None
    unsqueeze_15: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
    add_19: "f32[8, 32, 56, 56]" = torch.ops.aten.add.Tensor(mul_27, unsqueeze_15);  mul_27 = unsqueeze_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_3: "f32[8, 32, 56, 56]" = torch.ops.aten.relu.default(add_19);  add_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_3 = torch.ops.aten.split_with_sizes.default(relu_1, [32, 32, 32, 32], 1)
    getitem_24: "f32[8, 32, 56, 56]" = split_with_sizes_3[2];  split_with_sizes_3 = None
    convolution_4: "f32[8, 32, 56, 56]" = torch.ops.aten.convolution.default(getitem_24, primals_13, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_20: "i64[]" = torch.ops.aten.add.Tensor(primals_272, 1)
    var_mean_4 = torch.ops.aten.var_mean.correction(convolution_4, [0, 2, 3], correction = 0, keepdim = True)
    getitem_26: "f32[1, 32, 1, 1]" = var_mean_4[0]
    getitem_27: "f32[1, 32, 1, 1]" = var_mean_4[1];  var_mean_4 = None
    add_21: "f32[1, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-05)
    rsqrt_4: "f32[1, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
    sub_4: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_4, getitem_27)
    mul_28: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
    squeeze_12: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_27, [0, 2, 3]);  getitem_27 = None
    squeeze_13: "f32[32]" = torch.ops.aten.squeeze.dims(rsqrt_4, [0, 2, 3]);  rsqrt_4 = None
    mul_29: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_12, 0.1)
    mul_30: "f32[32]" = torch.ops.aten.mul.Tensor(primals_270, 0.9)
    add_22: "f32[32]" = torch.ops.aten.add.Tensor(mul_29, mul_30);  mul_29 = mul_30 = None
    squeeze_14: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_26, [0, 2, 3]);  getitem_26 = None
    mul_31: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_14, 1.0000398612827361);  squeeze_14 = None
    mul_32: "f32[32]" = torch.ops.aten.mul.Tensor(mul_31, 0.1);  mul_31 = None
    mul_33: "f32[32]" = torch.ops.aten.mul.Tensor(primals_271, 0.9)
    add_23: "f32[32]" = torch.ops.aten.add.Tensor(mul_32, mul_33);  mul_32 = mul_33 = None
    unsqueeze_16: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_14, -1)
    unsqueeze_17: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
    mul_34: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(mul_28, unsqueeze_17);  mul_28 = unsqueeze_17 = None
    unsqueeze_18: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_15, -1);  primals_15 = None
    unsqueeze_19: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
    add_24: "f32[8, 32, 56, 56]" = torch.ops.aten.add.Tensor(mul_34, unsqueeze_19);  mul_34 = unsqueeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_4: "f32[8, 32, 56, 56]" = torch.ops.aten.relu.default(add_24);  add_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:99, code: spo.append(self.pool(spx[-1]))
    split_with_sizes_4 = torch.ops.aten.split_with_sizes.default(relu_1, [32, 32, 32, 32], 1)
    getitem_31: "f32[8, 32, 56, 56]" = split_with_sizes_4[3];  split_with_sizes_4 = None
    avg_pool2d: "f32[8, 32, 56, 56]" = torch.ops.aten.avg_pool2d.default(getitem_31, [3, 3], [1, 1], [1, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    cat: "f32[8, 128, 56, 56]" = torch.ops.aten.cat.default([relu_2, relu_3, relu_4, avg_pool2d], 1);  avg_pool2d = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_5: "f32[8, 256, 56, 56]" = torch.ops.aten.convolution.default(cat, primals_16, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    add_25: "i64[]" = torch.ops.aten.add.Tensor(primals_275, 1)
    var_mean_5 = torch.ops.aten.var_mean.correction(convolution_5, [0, 2, 3], correction = 0, keepdim = True)
    getitem_32: "f32[1, 256, 1, 1]" = var_mean_5[0]
    getitem_33: "f32[1, 256, 1, 1]" = var_mean_5[1];  var_mean_5 = None
    add_26: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-05)
    rsqrt_5: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
    sub_5: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_5, getitem_33)
    mul_35: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
    squeeze_15: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_33, [0, 2, 3]);  getitem_33 = None
    squeeze_16: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_5, [0, 2, 3]);  rsqrt_5 = None
    mul_36: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_15, 0.1)
    mul_37: "f32[256]" = torch.ops.aten.mul.Tensor(primals_273, 0.9)
    add_27: "f32[256]" = torch.ops.aten.add.Tensor(mul_36, mul_37);  mul_36 = mul_37 = None
    squeeze_17: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_32, [0, 2, 3]);  getitem_32 = None
    mul_38: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_17, 1.0000398612827361);  squeeze_17 = None
    mul_39: "f32[256]" = torch.ops.aten.mul.Tensor(mul_38, 0.1);  mul_38 = None
    mul_40: "f32[256]" = torch.ops.aten.mul.Tensor(primals_274, 0.9)
    add_28: "f32[256]" = torch.ops.aten.add.Tensor(mul_39, mul_40);  mul_39 = mul_40 = None
    unsqueeze_20: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_17, -1)
    unsqueeze_21: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
    mul_41: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(mul_35, unsqueeze_21);  mul_35 = unsqueeze_21 = None
    unsqueeze_22: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_18, -1);  primals_18 = None
    unsqueeze_23: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
    add_29: "f32[8, 256, 56, 56]" = torch.ops.aten.add.Tensor(mul_41, unsqueeze_23);  mul_41 = unsqueeze_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:111, code: shortcut = self.downsample(x)
    convolution_6: "f32[8, 256, 56, 56]" = torch.ops.aten.convolution.default(getitem_2, primals_19, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_30: "i64[]" = torch.ops.aten.add.Tensor(primals_278, 1)
    var_mean_6 = torch.ops.aten.var_mean.correction(convolution_6, [0, 2, 3], correction = 0, keepdim = True)
    getitem_34: "f32[1, 256, 1, 1]" = var_mean_6[0]
    getitem_35: "f32[1, 256, 1, 1]" = var_mean_6[1];  var_mean_6 = None
    add_31: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-05)
    rsqrt_6: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
    sub_6: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_6, getitem_35)
    mul_42: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
    squeeze_18: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_35, [0, 2, 3]);  getitem_35 = None
    squeeze_19: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_6, [0, 2, 3]);  rsqrt_6 = None
    mul_43: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_18, 0.1)
    mul_44: "f32[256]" = torch.ops.aten.mul.Tensor(primals_276, 0.9)
    add_32: "f32[256]" = torch.ops.aten.add.Tensor(mul_43, mul_44);  mul_43 = mul_44 = None
    squeeze_20: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_34, [0, 2, 3]);  getitem_34 = None
    mul_45: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_20, 1.0000398612827361);  squeeze_20 = None
    mul_46: "f32[256]" = torch.ops.aten.mul.Tensor(mul_45, 0.1);  mul_45 = None
    mul_47: "f32[256]" = torch.ops.aten.mul.Tensor(primals_277, 0.9)
    add_33: "f32[256]" = torch.ops.aten.add.Tensor(mul_46, mul_47);  mul_46 = mul_47 = None
    unsqueeze_24: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_20, -1)
    unsqueeze_25: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
    mul_48: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(mul_42, unsqueeze_25);  mul_42 = unsqueeze_25 = None
    unsqueeze_26: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_21, -1);  primals_21 = None
    unsqueeze_27: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
    add_34: "f32[8, 256, 56, 56]" = torch.ops.aten.add.Tensor(mul_48, unsqueeze_27);  mul_48 = unsqueeze_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_35: "f32[8, 256, 56, 56]" = torch.ops.aten.add.Tensor(add_29, add_34);  add_29 = add_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_5: "f32[8, 256, 56, 56]" = torch.ops.aten.relu.default(add_35);  add_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_7: "f32[8, 128, 56, 56]" = torch.ops.aten.convolution.default(relu_5, primals_22, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    add_36: "i64[]" = torch.ops.aten.add.Tensor(primals_281, 1)
    var_mean_7 = torch.ops.aten.var_mean.correction(convolution_7, [0, 2, 3], correction = 0, keepdim = True)
    getitem_36: "f32[1, 128, 1, 1]" = var_mean_7[0]
    getitem_37: "f32[1, 128, 1, 1]" = var_mean_7[1];  var_mean_7 = None
    add_37: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-05)
    rsqrt_7: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
    sub_7: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_7, getitem_37)
    mul_49: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
    squeeze_21: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_37, [0, 2, 3]);  getitem_37 = None
    squeeze_22: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_7, [0, 2, 3]);  rsqrt_7 = None
    mul_50: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_21, 0.1)
    mul_51: "f32[128]" = torch.ops.aten.mul.Tensor(primals_279, 0.9)
    add_38: "f32[128]" = torch.ops.aten.add.Tensor(mul_50, mul_51);  mul_50 = mul_51 = None
    squeeze_23: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_36, [0, 2, 3]);  getitem_36 = None
    mul_52: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_23, 1.0000398612827361);  squeeze_23 = None
    mul_53: "f32[128]" = torch.ops.aten.mul.Tensor(mul_52, 0.1);  mul_52 = None
    mul_54: "f32[128]" = torch.ops.aten.mul.Tensor(primals_280, 0.9)
    add_39: "f32[128]" = torch.ops.aten.add.Tensor(mul_53, mul_54);  mul_53 = mul_54 = None
    unsqueeze_28: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_23, -1)
    unsqueeze_29: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
    mul_55: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(mul_49, unsqueeze_29);  mul_49 = unsqueeze_29 = None
    unsqueeze_30: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_24, -1);  primals_24 = None
    unsqueeze_31: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
    add_40: "f32[8, 128, 56, 56]" = torch.ops.aten.add.Tensor(mul_55, unsqueeze_31);  mul_55 = unsqueeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_6: "f32[8, 128, 56, 56]" = torch.ops.aten.relu.default(add_40);  add_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_6 = torch.ops.aten.split_with_sizes.default(relu_6, [32, 32, 32, 32], 1)
    getitem_42: "f32[8, 32, 56, 56]" = split_with_sizes_6[0];  split_with_sizes_6 = None
    convolution_8: "f32[8, 32, 56, 56]" = torch.ops.aten.convolution.default(getitem_42, primals_25, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_41: "i64[]" = torch.ops.aten.add.Tensor(primals_284, 1)
    var_mean_8 = torch.ops.aten.var_mean.correction(convolution_8, [0, 2, 3], correction = 0, keepdim = True)
    getitem_46: "f32[1, 32, 1, 1]" = var_mean_8[0]
    getitem_47: "f32[1, 32, 1, 1]" = var_mean_8[1];  var_mean_8 = None
    add_42: "f32[1, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-05)
    rsqrt_8: "f32[1, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
    sub_8: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_8, getitem_47)
    mul_56: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
    squeeze_24: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_47, [0, 2, 3]);  getitem_47 = None
    squeeze_25: "f32[32]" = torch.ops.aten.squeeze.dims(rsqrt_8, [0, 2, 3]);  rsqrt_8 = None
    mul_57: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_24, 0.1)
    mul_58: "f32[32]" = torch.ops.aten.mul.Tensor(primals_282, 0.9)
    add_43: "f32[32]" = torch.ops.aten.add.Tensor(mul_57, mul_58);  mul_57 = mul_58 = None
    squeeze_26: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_46, [0, 2, 3]);  getitem_46 = None
    mul_59: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_26, 1.0000398612827361);  squeeze_26 = None
    mul_60: "f32[32]" = torch.ops.aten.mul.Tensor(mul_59, 0.1);  mul_59 = None
    mul_61: "f32[32]" = torch.ops.aten.mul.Tensor(primals_283, 0.9)
    add_44: "f32[32]" = torch.ops.aten.add.Tensor(mul_60, mul_61);  mul_60 = mul_61 = None
    unsqueeze_32: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_26, -1)
    unsqueeze_33: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
    mul_62: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(mul_56, unsqueeze_33);  mul_56 = unsqueeze_33 = None
    unsqueeze_34: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_27, -1);  primals_27 = None
    unsqueeze_35: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
    add_45: "f32[8, 32, 56, 56]" = torch.ops.aten.add.Tensor(mul_62, unsqueeze_35);  mul_62 = unsqueeze_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_7: "f32[8, 32, 56, 56]" = torch.ops.aten.relu.default(add_45);  add_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_7 = torch.ops.aten.split_with_sizes.default(relu_6, [32, 32, 32, 32], 1)
    getitem_49: "f32[8, 32, 56, 56]" = split_with_sizes_7[1];  split_with_sizes_7 = None
    add_46: "f32[8, 32, 56, 56]" = torch.ops.aten.add.Tensor(relu_7, getitem_49);  getitem_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_9: "f32[8, 32, 56, 56]" = torch.ops.aten.convolution.default(add_46, primals_28, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_47: "i64[]" = torch.ops.aten.add.Tensor(primals_287, 1)
    var_mean_9 = torch.ops.aten.var_mean.correction(convolution_9, [0, 2, 3], correction = 0, keepdim = True)
    getitem_52: "f32[1, 32, 1, 1]" = var_mean_9[0]
    getitem_53: "f32[1, 32, 1, 1]" = var_mean_9[1];  var_mean_9 = None
    add_48: "f32[1, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-05)
    rsqrt_9: "f32[1, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_48);  add_48 = None
    sub_9: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_9, getitem_53)
    mul_63: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
    squeeze_27: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_53, [0, 2, 3]);  getitem_53 = None
    squeeze_28: "f32[32]" = torch.ops.aten.squeeze.dims(rsqrt_9, [0, 2, 3]);  rsqrt_9 = None
    mul_64: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_27, 0.1)
    mul_65: "f32[32]" = torch.ops.aten.mul.Tensor(primals_285, 0.9)
    add_49: "f32[32]" = torch.ops.aten.add.Tensor(mul_64, mul_65);  mul_64 = mul_65 = None
    squeeze_29: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_52, [0, 2, 3]);  getitem_52 = None
    mul_66: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_29, 1.0000398612827361);  squeeze_29 = None
    mul_67: "f32[32]" = torch.ops.aten.mul.Tensor(mul_66, 0.1);  mul_66 = None
    mul_68: "f32[32]" = torch.ops.aten.mul.Tensor(primals_286, 0.9)
    add_50: "f32[32]" = torch.ops.aten.add.Tensor(mul_67, mul_68);  mul_67 = mul_68 = None
    unsqueeze_36: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_29, -1)
    unsqueeze_37: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
    mul_69: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(mul_63, unsqueeze_37);  mul_63 = unsqueeze_37 = None
    unsqueeze_38: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_30, -1);  primals_30 = None
    unsqueeze_39: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
    add_51: "f32[8, 32, 56, 56]" = torch.ops.aten.add.Tensor(mul_69, unsqueeze_39);  mul_69 = unsqueeze_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_8: "f32[8, 32, 56, 56]" = torch.ops.aten.relu.default(add_51);  add_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_8 = torch.ops.aten.split_with_sizes.default(relu_6, [32, 32, 32, 32], 1)
    getitem_56: "f32[8, 32, 56, 56]" = split_with_sizes_8[2];  split_with_sizes_8 = None
    add_52: "f32[8, 32, 56, 56]" = torch.ops.aten.add.Tensor(relu_8, getitem_56);  getitem_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_10: "f32[8, 32, 56, 56]" = torch.ops.aten.convolution.default(add_52, primals_31, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_53: "i64[]" = torch.ops.aten.add.Tensor(primals_290, 1)
    var_mean_10 = torch.ops.aten.var_mean.correction(convolution_10, [0, 2, 3], correction = 0, keepdim = True)
    getitem_58: "f32[1, 32, 1, 1]" = var_mean_10[0]
    getitem_59: "f32[1, 32, 1, 1]" = var_mean_10[1];  var_mean_10 = None
    add_54: "f32[1, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_58, 1e-05)
    rsqrt_10: "f32[1, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_54);  add_54 = None
    sub_10: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_10, getitem_59)
    mul_70: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
    squeeze_30: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_59, [0, 2, 3]);  getitem_59 = None
    squeeze_31: "f32[32]" = torch.ops.aten.squeeze.dims(rsqrt_10, [0, 2, 3]);  rsqrt_10 = None
    mul_71: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_30, 0.1)
    mul_72: "f32[32]" = torch.ops.aten.mul.Tensor(primals_288, 0.9)
    add_55: "f32[32]" = torch.ops.aten.add.Tensor(mul_71, mul_72);  mul_71 = mul_72 = None
    squeeze_32: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_58, [0, 2, 3]);  getitem_58 = None
    mul_73: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_32, 1.0000398612827361);  squeeze_32 = None
    mul_74: "f32[32]" = torch.ops.aten.mul.Tensor(mul_73, 0.1);  mul_73 = None
    mul_75: "f32[32]" = torch.ops.aten.mul.Tensor(primals_289, 0.9)
    add_56: "f32[32]" = torch.ops.aten.add.Tensor(mul_74, mul_75);  mul_74 = mul_75 = None
    unsqueeze_40: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_32, -1)
    unsqueeze_41: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
    mul_76: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(mul_70, unsqueeze_41);  mul_70 = unsqueeze_41 = None
    unsqueeze_42: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_33, -1);  primals_33 = None
    unsqueeze_43: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
    add_57: "f32[8, 32, 56, 56]" = torch.ops.aten.add.Tensor(mul_76, unsqueeze_43);  mul_76 = unsqueeze_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_9: "f32[8, 32, 56, 56]" = torch.ops.aten.relu.default(add_57);  add_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    split_with_sizes_9 = torch.ops.aten.split_with_sizes.default(relu_6, [32, 32, 32, 32], 1)
    getitem_63: "f32[8, 32, 56, 56]" = split_with_sizes_9[3];  split_with_sizes_9 = None
    cat_1: "f32[8, 128, 56, 56]" = torch.ops.aten.cat.default([relu_7, relu_8, relu_9, getitem_63], 1);  getitem_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_11: "f32[8, 256, 56, 56]" = torch.ops.aten.convolution.default(cat_1, primals_34, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    add_58: "i64[]" = torch.ops.aten.add.Tensor(primals_293, 1)
    var_mean_11 = torch.ops.aten.var_mean.correction(convolution_11, [0, 2, 3], correction = 0, keepdim = True)
    getitem_64: "f32[1, 256, 1, 1]" = var_mean_11[0]
    getitem_65: "f32[1, 256, 1, 1]" = var_mean_11[1];  var_mean_11 = None
    add_59: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-05)
    rsqrt_11: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_59);  add_59 = None
    sub_11: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_11, getitem_65)
    mul_77: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
    squeeze_33: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_65, [0, 2, 3]);  getitem_65 = None
    squeeze_34: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_11, [0, 2, 3]);  rsqrt_11 = None
    mul_78: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_33, 0.1)
    mul_79: "f32[256]" = torch.ops.aten.mul.Tensor(primals_291, 0.9)
    add_60: "f32[256]" = torch.ops.aten.add.Tensor(mul_78, mul_79);  mul_78 = mul_79 = None
    squeeze_35: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_64, [0, 2, 3]);  getitem_64 = None
    mul_80: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_35, 1.0000398612827361);  squeeze_35 = None
    mul_81: "f32[256]" = torch.ops.aten.mul.Tensor(mul_80, 0.1);  mul_80 = None
    mul_82: "f32[256]" = torch.ops.aten.mul.Tensor(primals_292, 0.9)
    add_61: "f32[256]" = torch.ops.aten.add.Tensor(mul_81, mul_82);  mul_81 = mul_82 = None
    unsqueeze_44: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_35, -1)
    unsqueeze_45: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
    mul_83: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(mul_77, unsqueeze_45);  mul_77 = unsqueeze_45 = None
    unsqueeze_46: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_36, -1);  primals_36 = None
    unsqueeze_47: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
    add_62: "f32[8, 256, 56, 56]" = torch.ops.aten.add.Tensor(mul_83, unsqueeze_47);  mul_83 = unsqueeze_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_63: "f32[8, 256, 56, 56]" = torch.ops.aten.add.Tensor(add_62, relu_5);  add_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_10: "f32[8, 256, 56, 56]" = torch.ops.aten.relu.default(add_63);  add_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_12: "f32[8, 128, 56, 56]" = torch.ops.aten.convolution.default(relu_10, primals_37, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    add_64: "i64[]" = torch.ops.aten.add.Tensor(primals_296, 1)
    var_mean_12 = torch.ops.aten.var_mean.correction(convolution_12, [0, 2, 3], correction = 0, keepdim = True)
    getitem_66: "f32[1, 128, 1, 1]" = var_mean_12[0]
    getitem_67: "f32[1, 128, 1, 1]" = var_mean_12[1];  var_mean_12 = None
    add_65: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-05)
    rsqrt_12: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_65);  add_65 = None
    sub_12: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_12, getitem_67)
    mul_84: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
    squeeze_36: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_67, [0, 2, 3]);  getitem_67 = None
    squeeze_37: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_12, [0, 2, 3]);  rsqrt_12 = None
    mul_85: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_36, 0.1)
    mul_86: "f32[128]" = torch.ops.aten.mul.Tensor(primals_294, 0.9)
    add_66: "f32[128]" = torch.ops.aten.add.Tensor(mul_85, mul_86);  mul_85 = mul_86 = None
    squeeze_38: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_66, [0, 2, 3]);  getitem_66 = None
    mul_87: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_38, 1.0000398612827361);  squeeze_38 = None
    mul_88: "f32[128]" = torch.ops.aten.mul.Tensor(mul_87, 0.1);  mul_87 = None
    mul_89: "f32[128]" = torch.ops.aten.mul.Tensor(primals_295, 0.9)
    add_67: "f32[128]" = torch.ops.aten.add.Tensor(mul_88, mul_89);  mul_88 = mul_89 = None
    unsqueeze_48: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_38, -1)
    unsqueeze_49: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
    mul_90: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(mul_84, unsqueeze_49);  mul_84 = unsqueeze_49 = None
    unsqueeze_50: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_39, -1);  primals_39 = None
    unsqueeze_51: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
    add_68: "f32[8, 128, 56, 56]" = torch.ops.aten.add.Tensor(mul_90, unsqueeze_51);  mul_90 = unsqueeze_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_11: "f32[8, 128, 56, 56]" = torch.ops.aten.relu.default(add_68);  add_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_11 = torch.ops.aten.split_with_sizes.default(relu_11, [32, 32, 32, 32], 1)
    getitem_72: "f32[8, 32, 56, 56]" = split_with_sizes_11[0];  split_with_sizes_11 = None
    convolution_13: "f32[8, 32, 56, 56]" = torch.ops.aten.convolution.default(getitem_72, primals_40, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_69: "i64[]" = torch.ops.aten.add.Tensor(primals_299, 1)
    var_mean_13 = torch.ops.aten.var_mean.correction(convolution_13, [0, 2, 3], correction = 0, keepdim = True)
    getitem_76: "f32[1, 32, 1, 1]" = var_mean_13[0]
    getitem_77: "f32[1, 32, 1, 1]" = var_mean_13[1];  var_mean_13 = None
    add_70: "f32[1, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-05)
    rsqrt_13: "f32[1, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_70);  add_70 = None
    sub_13: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_13, getitem_77)
    mul_91: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
    squeeze_39: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_77, [0, 2, 3]);  getitem_77 = None
    squeeze_40: "f32[32]" = torch.ops.aten.squeeze.dims(rsqrt_13, [0, 2, 3]);  rsqrt_13 = None
    mul_92: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_39, 0.1)
    mul_93: "f32[32]" = torch.ops.aten.mul.Tensor(primals_297, 0.9)
    add_71: "f32[32]" = torch.ops.aten.add.Tensor(mul_92, mul_93);  mul_92 = mul_93 = None
    squeeze_41: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_76, [0, 2, 3]);  getitem_76 = None
    mul_94: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_41, 1.0000398612827361);  squeeze_41 = None
    mul_95: "f32[32]" = torch.ops.aten.mul.Tensor(mul_94, 0.1);  mul_94 = None
    mul_96: "f32[32]" = torch.ops.aten.mul.Tensor(primals_298, 0.9)
    add_72: "f32[32]" = torch.ops.aten.add.Tensor(mul_95, mul_96);  mul_95 = mul_96 = None
    unsqueeze_52: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_41, -1)
    unsqueeze_53: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
    mul_97: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(mul_91, unsqueeze_53);  mul_91 = unsqueeze_53 = None
    unsqueeze_54: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_42, -1);  primals_42 = None
    unsqueeze_55: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
    add_73: "f32[8, 32, 56, 56]" = torch.ops.aten.add.Tensor(mul_97, unsqueeze_55);  mul_97 = unsqueeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_12: "f32[8, 32, 56, 56]" = torch.ops.aten.relu.default(add_73);  add_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_12 = torch.ops.aten.split_with_sizes.default(relu_11, [32, 32, 32, 32], 1)
    getitem_79: "f32[8, 32, 56, 56]" = split_with_sizes_12[1];  split_with_sizes_12 = None
    add_74: "f32[8, 32, 56, 56]" = torch.ops.aten.add.Tensor(relu_12, getitem_79);  getitem_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_14: "f32[8, 32, 56, 56]" = torch.ops.aten.convolution.default(add_74, primals_43, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_75: "i64[]" = torch.ops.aten.add.Tensor(primals_302, 1)
    var_mean_14 = torch.ops.aten.var_mean.correction(convolution_14, [0, 2, 3], correction = 0, keepdim = True)
    getitem_82: "f32[1, 32, 1, 1]" = var_mean_14[0]
    getitem_83: "f32[1, 32, 1, 1]" = var_mean_14[1];  var_mean_14 = None
    add_76: "f32[1, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_82, 1e-05)
    rsqrt_14: "f32[1, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_76);  add_76 = None
    sub_14: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_14, getitem_83)
    mul_98: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
    squeeze_42: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_83, [0, 2, 3]);  getitem_83 = None
    squeeze_43: "f32[32]" = torch.ops.aten.squeeze.dims(rsqrt_14, [0, 2, 3]);  rsqrt_14 = None
    mul_99: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_42, 0.1)
    mul_100: "f32[32]" = torch.ops.aten.mul.Tensor(primals_300, 0.9)
    add_77: "f32[32]" = torch.ops.aten.add.Tensor(mul_99, mul_100);  mul_99 = mul_100 = None
    squeeze_44: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_82, [0, 2, 3]);  getitem_82 = None
    mul_101: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_44, 1.0000398612827361);  squeeze_44 = None
    mul_102: "f32[32]" = torch.ops.aten.mul.Tensor(mul_101, 0.1);  mul_101 = None
    mul_103: "f32[32]" = torch.ops.aten.mul.Tensor(primals_301, 0.9)
    add_78: "f32[32]" = torch.ops.aten.add.Tensor(mul_102, mul_103);  mul_102 = mul_103 = None
    unsqueeze_56: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_44, -1)
    unsqueeze_57: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
    mul_104: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(mul_98, unsqueeze_57);  mul_98 = unsqueeze_57 = None
    unsqueeze_58: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_45, -1);  primals_45 = None
    unsqueeze_59: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
    add_79: "f32[8, 32, 56, 56]" = torch.ops.aten.add.Tensor(mul_104, unsqueeze_59);  mul_104 = unsqueeze_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_13: "f32[8, 32, 56, 56]" = torch.ops.aten.relu.default(add_79);  add_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_13 = torch.ops.aten.split_with_sizes.default(relu_11, [32, 32, 32, 32], 1)
    getitem_86: "f32[8, 32, 56, 56]" = split_with_sizes_13[2];  split_with_sizes_13 = None
    add_80: "f32[8, 32, 56, 56]" = torch.ops.aten.add.Tensor(relu_13, getitem_86);  getitem_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_15: "f32[8, 32, 56, 56]" = torch.ops.aten.convolution.default(add_80, primals_46, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_81: "i64[]" = torch.ops.aten.add.Tensor(primals_305, 1)
    var_mean_15 = torch.ops.aten.var_mean.correction(convolution_15, [0, 2, 3], correction = 0, keepdim = True)
    getitem_88: "f32[1, 32, 1, 1]" = var_mean_15[0]
    getitem_89: "f32[1, 32, 1, 1]" = var_mean_15[1];  var_mean_15 = None
    add_82: "f32[1, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-05)
    rsqrt_15: "f32[1, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
    sub_15: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_15, getitem_89)
    mul_105: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = None
    squeeze_45: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_89, [0, 2, 3]);  getitem_89 = None
    squeeze_46: "f32[32]" = torch.ops.aten.squeeze.dims(rsqrt_15, [0, 2, 3]);  rsqrt_15 = None
    mul_106: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_45, 0.1)
    mul_107: "f32[32]" = torch.ops.aten.mul.Tensor(primals_303, 0.9)
    add_83: "f32[32]" = torch.ops.aten.add.Tensor(mul_106, mul_107);  mul_106 = mul_107 = None
    squeeze_47: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_88, [0, 2, 3]);  getitem_88 = None
    mul_108: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_47, 1.0000398612827361);  squeeze_47 = None
    mul_109: "f32[32]" = torch.ops.aten.mul.Tensor(mul_108, 0.1);  mul_108 = None
    mul_110: "f32[32]" = torch.ops.aten.mul.Tensor(primals_304, 0.9)
    add_84: "f32[32]" = torch.ops.aten.add.Tensor(mul_109, mul_110);  mul_109 = mul_110 = None
    unsqueeze_60: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_47, -1)
    unsqueeze_61: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
    mul_111: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(mul_105, unsqueeze_61);  mul_105 = unsqueeze_61 = None
    unsqueeze_62: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_48, -1);  primals_48 = None
    unsqueeze_63: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
    add_85: "f32[8, 32, 56, 56]" = torch.ops.aten.add.Tensor(mul_111, unsqueeze_63);  mul_111 = unsqueeze_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_14: "f32[8, 32, 56, 56]" = torch.ops.aten.relu.default(add_85);  add_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    split_with_sizes_14 = torch.ops.aten.split_with_sizes.default(relu_11, [32, 32, 32, 32], 1)
    getitem_93: "f32[8, 32, 56, 56]" = split_with_sizes_14[3];  split_with_sizes_14 = None
    cat_2: "f32[8, 128, 56, 56]" = torch.ops.aten.cat.default([relu_12, relu_13, relu_14, getitem_93], 1);  getitem_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_16: "f32[8, 256, 56, 56]" = torch.ops.aten.convolution.default(cat_2, primals_49, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    add_86: "i64[]" = torch.ops.aten.add.Tensor(primals_308, 1)
    var_mean_16 = torch.ops.aten.var_mean.correction(convolution_16, [0, 2, 3], correction = 0, keepdim = True)
    getitem_94: "f32[1, 256, 1, 1]" = var_mean_16[0]
    getitem_95: "f32[1, 256, 1, 1]" = var_mean_16[1];  var_mean_16 = None
    add_87: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_94, 1e-05)
    rsqrt_16: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_87);  add_87 = None
    sub_16: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_16, getitem_95)
    mul_112: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
    squeeze_48: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_95, [0, 2, 3]);  getitem_95 = None
    squeeze_49: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_16, [0, 2, 3]);  rsqrt_16 = None
    mul_113: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_48, 0.1)
    mul_114: "f32[256]" = torch.ops.aten.mul.Tensor(primals_306, 0.9)
    add_88: "f32[256]" = torch.ops.aten.add.Tensor(mul_113, mul_114);  mul_113 = mul_114 = None
    squeeze_50: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_94, [0, 2, 3]);  getitem_94 = None
    mul_115: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_50, 1.0000398612827361);  squeeze_50 = None
    mul_116: "f32[256]" = torch.ops.aten.mul.Tensor(mul_115, 0.1);  mul_115 = None
    mul_117: "f32[256]" = torch.ops.aten.mul.Tensor(primals_307, 0.9)
    add_89: "f32[256]" = torch.ops.aten.add.Tensor(mul_116, mul_117);  mul_116 = mul_117 = None
    unsqueeze_64: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_50, -1)
    unsqueeze_65: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
    mul_118: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(mul_112, unsqueeze_65);  mul_112 = unsqueeze_65 = None
    unsqueeze_66: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_51, -1);  primals_51 = None
    unsqueeze_67: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
    add_90: "f32[8, 256, 56, 56]" = torch.ops.aten.add.Tensor(mul_118, unsqueeze_67);  mul_118 = unsqueeze_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_91: "f32[8, 256, 56, 56]" = torch.ops.aten.add.Tensor(add_90, relu_10);  add_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_15: "f32[8, 256, 56, 56]" = torch.ops.aten.relu.default(add_91);  add_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_17: "f32[8, 256, 56, 56]" = torch.ops.aten.convolution.default(relu_15, primals_52, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    add_92: "i64[]" = torch.ops.aten.add.Tensor(primals_311, 1)
    var_mean_17 = torch.ops.aten.var_mean.correction(convolution_17, [0, 2, 3], correction = 0, keepdim = True)
    getitem_96: "f32[1, 256, 1, 1]" = var_mean_17[0]
    getitem_97: "f32[1, 256, 1, 1]" = var_mean_17[1];  var_mean_17 = None
    add_93: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_96, 1e-05)
    rsqrt_17: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_93);  add_93 = None
    sub_17: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_17, getitem_97)
    mul_119: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = None
    squeeze_51: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_97, [0, 2, 3]);  getitem_97 = None
    squeeze_52: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_17, [0, 2, 3]);  rsqrt_17 = None
    mul_120: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_51, 0.1)
    mul_121: "f32[256]" = torch.ops.aten.mul.Tensor(primals_309, 0.9)
    add_94: "f32[256]" = torch.ops.aten.add.Tensor(mul_120, mul_121);  mul_120 = mul_121 = None
    squeeze_53: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_96, [0, 2, 3]);  getitem_96 = None
    mul_122: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_53, 1.0000398612827361);  squeeze_53 = None
    mul_123: "f32[256]" = torch.ops.aten.mul.Tensor(mul_122, 0.1);  mul_122 = None
    mul_124: "f32[256]" = torch.ops.aten.mul.Tensor(primals_310, 0.9)
    add_95: "f32[256]" = torch.ops.aten.add.Tensor(mul_123, mul_124);  mul_123 = mul_124 = None
    unsqueeze_68: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_53, -1)
    unsqueeze_69: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
    mul_125: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(mul_119, unsqueeze_69);  mul_119 = unsqueeze_69 = None
    unsqueeze_70: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_54, -1);  primals_54 = None
    unsqueeze_71: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
    add_96: "f32[8, 256, 56, 56]" = torch.ops.aten.add.Tensor(mul_125, unsqueeze_71);  mul_125 = unsqueeze_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_16: "f32[8, 256, 56, 56]" = torch.ops.aten.relu.default(add_96);  add_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_16 = torch.ops.aten.split_with_sizes.default(relu_16, [64, 64, 64, 64], 1)
    getitem_102: "f32[8, 64, 56, 56]" = split_with_sizes_16[0];  split_with_sizes_16 = None
    convolution_18: "f32[8, 64, 28, 28]" = torch.ops.aten.convolution.default(getitem_102, primals_55, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_97: "i64[]" = torch.ops.aten.add.Tensor(primals_314, 1)
    var_mean_18 = torch.ops.aten.var_mean.correction(convolution_18, [0, 2, 3], correction = 0, keepdim = True)
    getitem_106: "f32[1, 64, 1, 1]" = var_mean_18[0]
    getitem_107: "f32[1, 64, 1, 1]" = var_mean_18[1];  var_mean_18 = None
    add_98: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_106, 1e-05)
    rsqrt_18: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
    sub_18: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_18, getitem_107)
    mul_126: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = None
    squeeze_54: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_107, [0, 2, 3]);  getitem_107 = None
    squeeze_55: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_18, [0, 2, 3]);  rsqrt_18 = None
    mul_127: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_54, 0.1)
    mul_128: "f32[64]" = torch.ops.aten.mul.Tensor(primals_312, 0.9)
    add_99: "f32[64]" = torch.ops.aten.add.Tensor(mul_127, mul_128);  mul_127 = mul_128 = None
    squeeze_56: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_106, [0, 2, 3]);  getitem_106 = None
    mul_129: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_56, 1.0001594642002871);  squeeze_56 = None
    mul_130: "f32[64]" = torch.ops.aten.mul.Tensor(mul_129, 0.1);  mul_129 = None
    mul_131: "f32[64]" = torch.ops.aten.mul.Tensor(primals_313, 0.9)
    add_100: "f32[64]" = torch.ops.aten.add.Tensor(mul_130, mul_131);  mul_130 = mul_131 = None
    unsqueeze_72: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_56, -1)
    unsqueeze_73: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
    mul_132: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(mul_126, unsqueeze_73);  mul_126 = unsqueeze_73 = None
    unsqueeze_74: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_57, -1);  primals_57 = None
    unsqueeze_75: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
    add_101: "f32[8, 64, 28, 28]" = torch.ops.aten.add.Tensor(mul_132, unsqueeze_75);  mul_132 = unsqueeze_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_17: "f32[8, 64, 28, 28]" = torch.ops.aten.relu.default(add_101);  add_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_17 = torch.ops.aten.split_with_sizes.default(relu_16, [64, 64, 64, 64], 1)
    getitem_109: "f32[8, 64, 56, 56]" = split_with_sizes_17[1];  split_with_sizes_17 = None
    convolution_19: "f32[8, 64, 28, 28]" = torch.ops.aten.convolution.default(getitem_109, primals_58, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_102: "i64[]" = torch.ops.aten.add.Tensor(primals_317, 1)
    var_mean_19 = torch.ops.aten.var_mean.correction(convolution_19, [0, 2, 3], correction = 0, keepdim = True)
    getitem_112: "f32[1, 64, 1, 1]" = var_mean_19[0]
    getitem_113: "f32[1, 64, 1, 1]" = var_mean_19[1];  var_mean_19 = None
    add_103: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_112, 1e-05)
    rsqrt_19: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_103);  add_103 = None
    sub_19: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_19, getitem_113)
    mul_133: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = None
    squeeze_57: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_113, [0, 2, 3]);  getitem_113 = None
    squeeze_58: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_19, [0, 2, 3]);  rsqrt_19 = None
    mul_134: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_57, 0.1)
    mul_135: "f32[64]" = torch.ops.aten.mul.Tensor(primals_315, 0.9)
    add_104: "f32[64]" = torch.ops.aten.add.Tensor(mul_134, mul_135);  mul_134 = mul_135 = None
    squeeze_59: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_112, [0, 2, 3]);  getitem_112 = None
    mul_136: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_59, 1.0001594642002871);  squeeze_59 = None
    mul_137: "f32[64]" = torch.ops.aten.mul.Tensor(mul_136, 0.1);  mul_136 = None
    mul_138: "f32[64]" = torch.ops.aten.mul.Tensor(primals_316, 0.9)
    add_105: "f32[64]" = torch.ops.aten.add.Tensor(mul_137, mul_138);  mul_137 = mul_138 = None
    unsqueeze_76: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_59, -1)
    unsqueeze_77: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
    mul_139: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(mul_133, unsqueeze_77);  mul_133 = unsqueeze_77 = None
    unsqueeze_78: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_60, -1);  primals_60 = None
    unsqueeze_79: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
    add_106: "f32[8, 64, 28, 28]" = torch.ops.aten.add.Tensor(mul_139, unsqueeze_79);  mul_139 = unsqueeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_18: "f32[8, 64, 28, 28]" = torch.ops.aten.relu.default(add_106);  add_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_18 = torch.ops.aten.split_with_sizes.default(relu_16, [64, 64, 64, 64], 1)
    getitem_116: "f32[8, 64, 56, 56]" = split_with_sizes_18[2];  split_with_sizes_18 = None
    convolution_20: "f32[8, 64, 28, 28]" = torch.ops.aten.convolution.default(getitem_116, primals_61, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_107: "i64[]" = torch.ops.aten.add.Tensor(primals_320, 1)
    var_mean_20 = torch.ops.aten.var_mean.correction(convolution_20, [0, 2, 3], correction = 0, keepdim = True)
    getitem_118: "f32[1, 64, 1, 1]" = var_mean_20[0]
    getitem_119: "f32[1, 64, 1, 1]" = var_mean_20[1];  var_mean_20 = None
    add_108: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_118, 1e-05)
    rsqrt_20: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_108);  add_108 = None
    sub_20: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_20, getitem_119)
    mul_140: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = None
    squeeze_60: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_119, [0, 2, 3]);  getitem_119 = None
    squeeze_61: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_20, [0, 2, 3]);  rsqrt_20 = None
    mul_141: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_60, 0.1)
    mul_142: "f32[64]" = torch.ops.aten.mul.Tensor(primals_318, 0.9)
    add_109: "f32[64]" = torch.ops.aten.add.Tensor(mul_141, mul_142);  mul_141 = mul_142 = None
    squeeze_62: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_118, [0, 2, 3]);  getitem_118 = None
    mul_143: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_62, 1.0001594642002871);  squeeze_62 = None
    mul_144: "f32[64]" = torch.ops.aten.mul.Tensor(mul_143, 0.1);  mul_143 = None
    mul_145: "f32[64]" = torch.ops.aten.mul.Tensor(primals_319, 0.9)
    add_110: "f32[64]" = torch.ops.aten.add.Tensor(mul_144, mul_145);  mul_144 = mul_145 = None
    unsqueeze_80: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_62, -1)
    unsqueeze_81: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
    mul_146: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(mul_140, unsqueeze_81);  mul_140 = unsqueeze_81 = None
    unsqueeze_82: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_63, -1);  primals_63 = None
    unsqueeze_83: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
    add_111: "f32[8, 64, 28, 28]" = torch.ops.aten.add.Tensor(mul_146, unsqueeze_83);  mul_146 = unsqueeze_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_19: "f32[8, 64, 28, 28]" = torch.ops.aten.relu.default(add_111);  add_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:99, code: spo.append(self.pool(spx[-1]))
    split_with_sizes_19 = torch.ops.aten.split_with_sizes.default(relu_16, [64, 64, 64, 64], 1)
    getitem_123: "f32[8, 64, 56, 56]" = split_with_sizes_19[3];  split_with_sizes_19 = None
    avg_pool2d_1: "f32[8, 64, 28, 28]" = torch.ops.aten.avg_pool2d.default(getitem_123, [3, 3], [2, 2], [1, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    cat_3: "f32[8, 256, 28, 28]" = torch.ops.aten.cat.default([relu_17, relu_18, relu_19, avg_pool2d_1], 1);  avg_pool2d_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_21: "f32[8, 512, 28, 28]" = torch.ops.aten.convolution.default(cat_3, primals_64, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    add_112: "i64[]" = torch.ops.aten.add.Tensor(primals_323, 1)
    var_mean_21 = torch.ops.aten.var_mean.correction(convolution_21, [0, 2, 3], correction = 0, keepdim = True)
    getitem_124: "f32[1, 512, 1, 1]" = var_mean_21[0]
    getitem_125: "f32[1, 512, 1, 1]" = var_mean_21[1];  var_mean_21 = None
    add_113: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_124, 1e-05)
    rsqrt_21: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_113);  add_113 = None
    sub_21: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_21, getitem_125)
    mul_147: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = None
    squeeze_63: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_125, [0, 2, 3]);  getitem_125 = None
    squeeze_64: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_21, [0, 2, 3]);  rsqrt_21 = None
    mul_148: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_63, 0.1)
    mul_149: "f32[512]" = torch.ops.aten.mul.Tensor(primals_321, 0.9)
    add_114: "f32[512]" = torch.ops.aten.add.Tensor(mul_148, mul_149);  mul_148 = mul_149 = None
    squeeze_65: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_124, [0, 2, 3]);  getitem_124 = None
    mul_150: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_65, 1.0001594642002871);  squeeze_65 = None
    mul_151: "f32[512]" = torch.ops.aten.mul.Tensor(mul_150, 0.1);  mul_150 = None
    mul_152: "f32[512]" = torch.ops.aten.mul.Tensor(primals_322, 0.9)
    add_115: "f32[512]" = torch.ops.aten.add.Tensor(mul_151, mul_152);  mul_151 = mul_152 = None
    unsqueeze_84: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_65, -1)
    unsqueeze_85: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
    mul_153: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_147, unsqueeze_85);  mul_147 = unsqueeze_85 = None
    unsqueeze_86: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_66, -1);  primals_66 = None
    unsqueeze_87: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
    add_116: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_153, unsqueeze_87);  mul_153 = unsqueeze_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:111, code: shortcut = self.downsample(x)
    convolution_22: "f32[8, 512, 28, 28]" = torch.ops.aten.convolution.default(relu_15, primals_67, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
    add_117: "i64[]" = torch.ops.aten.add.Tensor(primals_326, 1)
    var_mean_22 = torch.ops.aten.var_mean.correction(convolution_22, [0, 2, 3], correction = 0, keepdim = True)
    getitem_126: "f32[1, 512, 1, 1]" = var_mean_22[0]
    getitem_127: "f32[1, 512, 1, 1]" = var_mean_22[1];  var_mean_22 = None
    add_118: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_126, 1e-05)
    rsqrt_22: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_118);  add_118 = None
    sub_22: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_22, getitem_127)
    mul_154: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = None
    squeeze_66: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_127, [0, 2, 3]);  getitem_127 = None
    squeeze_67: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_22, [0, 2, 3]);  rsqrt_22 = None
    mul_155: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_66, 0.1)
    mul_156: "f32[512]" = torch.ops.aten.mul.Tensor(primals_324, 0.9)
    add_119: "f32[512]" = torch.ops.aten.add.Tensor(mul_155, mul_156);  mul_155 = mul_156 = None
    squeeze_68: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_126, [0, 2, 3]);  getitem_126 = None
    mul_157: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_68, 1.0001594642002871);  squeeze_68 = None
    mul_158: "f32[512]" = torch.ops.aten.mul.Tensor(mul_157, 0.1);  mul_157 = None
    mul_159: "f32[512]" = torch.ops.aten.mul.Tensor(primals_325, 0.9)
    add_120: "f32[512]" = torch.ops.aten.add.Tensor(mul_158, mul_159);  mul_158 = mul_159 = None
    unsqueeze_88: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_68, -1)
    unsqueeze_89: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
    mul_160: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_154, unsqueeze_89);  mul_154 = unsqueeze_89 = None
    unsqueeze_90: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_69, -1);  primals_69 = None
    unsqueeze_91: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
    add_121: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_160, unsqueeze_91);  mul_160 = unsqueeze_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_122: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(add_116, add_121);  add_116 = add_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_20: "f32[8, 512, 28, 28]" = torch.ops.aten.relu.default(add_122);  add_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_23: "f32[8, 256, 28, 28]" = torch.ops.aten.convolution.default(relu_20, primals_70, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    add_123: "i64[]" = torch.ops.aten.add.Tensor(primals_329, 1)
    var_mean_23 = torch.ops.aten.var_mean.correction(convolution_23, [0, 2, 3], correction = 0, keepdim = True)
    getitem_128: "f32[1, 256, 1, 1]" = var_mean_23[0]
    getitem_129: "f32[1, 256, 1, 1]" = var_mean_23[1];  var_mean_23 = None
    add_124: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_128, 1e-05)
    rsqrt_23: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_124);  add_124 = None
    sub_23: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_23, getitem_129)
    mul_161: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = None
    squeeze_69: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_129, [0, 2, 3]);  getitem_129 = None
    squeeze_70: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_23, [0, 2, 3]);  rsqrt_23 = None
    mul_162: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_69, 0.1)
    mul_163: "f32[256]" = torch.ops.aten.mul.Tensor(primals_327, 0.9)
    add_125: "f32[256]" = torch.ops.aten.add.Tensor(mul_162, mul_163);  mul_162 = mul_163 = None
    squeeze_71: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_128, [0, 2, 3]);  getitem_128 = None
    mul_164: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_71, 1.0001594642002871);  squeeze_71 = None
    mul_165: "f32[256]" = torch.ops.aten.mul.Tensor(mul_164, 0.1);  mul_164 = None
    mul_166: "f32[256]" = torch.ops.aten.mul.Tensor(primals_328, 0.9)
    add_126: "f32[256]" = torch.ops.aten.add.Tensor(mul_165, mul_166);  mul_165 = mul_166 = None
    unsqueeze_92: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_71, -1)
    unsqueeze_93: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
    mul_167: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(mul_161, unsqueeze_93);  mul_161 = unsqueeze_93 = None
    unsqueeze_94: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_72, -1);  primals_72 = None
    unsqueeze_95: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
    add_127: "f32[8, 256, 28, 28]" = torch.ops.aten.add.Tensor(mul_167, unsqueeze_95);  mul_167 = unsqueeze_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_21: "f32[8, 256, 28, 28]" = torch.ops.aten.relu.default(add_127);  add_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_21 = torch.ops.aten.split_with_sizes.default(relu_21, [64, 64, 64, 64], 1)
    getitem_134: "f32[8, 64, 28, 28]" = split_with_sizes_21[0];  split_with_sizes_21 = None
    convolution_24: "f32[8, 64, 28, 28]" = torch.ops.aten.convolution.default(getitem_134, primals_73, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_128: "i64[]" = torch.ops.aten.add.Tensor(primals_332, 1)
    var_mean_24 = torch.ops.aten.var_mean.correction(convolution_24, [0, 2, 3], correction = 0, keepdim = True)
    getitem_138: "f32[1, 64, 1, 1]" = var_mean_24[0]
    getitem_139: "f32[1, 64, 1, 1]" = var_mean_24[1];  var_mean_24 = None
    add_129: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_138, 1e-05)
    rsqrt_24: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_129);  add_129 = None
    sub_24: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_24, getitem_139)
    mul_168: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = None
    squeeze_72: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_139, [0, 2, 3]);  getitem_139 = None
    squeeze_73: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_24, [0, 2, 3]);  rsqrt_24 = None
    mul_169: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_72, 0.1)
    mul_170: "f32[64]" = torch.ops.aten.mul.Tensor(primals_330, 0.9)
    add_130: "f32[64]" = torch.ops.aten.add.Tensor(mul_169, mul_170);  mul_169 = mul_170 = None
    squeeze_74: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_138, [0, 2, 3]);  getitem_138 = None
    mul_171: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_74, 1.0001594642002871);  squeeze_74 = None
    mul_172: "f32[64]" = torch.ops.aten.mul.Tensor(mul_171, 0.1);  mul_171 = None
    mul_173: "f32[64]" = torch.ops.aten.mul.Tensor(primals_331, 0.9)
    add_131: "f32[64]" = torch.ops.aten.add.Tensor(mul_172, mul_173);  mul_172 = mul_173 = None
    unsqueeze_96: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_74, -1)
    unsqueeze_97: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
    mul_174: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(mul_168, unsqueeze_97);  mul_168 = unsqueeze_97 = None
    unsqueeze_98: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_75, -1);  primals_75 = None
    unsqueeze_99: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
    add_132: "f32[8, 64, 28, 28]" = torch.ops.aten.add.Tensor(mul_174, unsqueeze_99);  mul_174 = unsqueeze_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_22: "f32[8, 64, 28, 28]" = torch.ops.aten.relu.default(add_132);  add_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_22 = torch.ops.aten.split_with_sizes.default(relu_21, [64, 64, 64, 64], 1)
    getitem_141: "f32[8, 64, 28, 28]" = split_with_sizes_22[1];  split_with_sizes_22 = None
    add_133: "f32[8, 64, 28, 28]" = torch.ops.aten.add.Tensor(relu_22, getitem_141);  getitem_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_25: "f32[8, 64, 28, 28]" = torch.ops.aten.convolution.default(add_133, primals_76, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_134: "i64[]" = torch.ops.aten.add.Tensor(primals_335, 1)
    var_mean_25 = torch.ops.aten.var_mean.correction(convolution_25, [0, 2, 3], correction = 0, keepdim = True)
    getitem_144: "f32[1, 64, 1, 1]" = var_mean_25[0]
    getitem_145: "f32[1, 64, 1, 1]" = var_mean_25[1];  var_mean_25 = None
    add_135: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_144, 1e-05)
    rsqrt_25: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_135);  add_135 = None
    sub_25: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_25, getitem_145)
    mul_175: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = None
    squeeze_75: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_145, [0, 2, 3]);  getitem_145 = None
    squeeze_76: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_25, [0, 2, 3]);  rsqrt_25 = None
    mul_176: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_75, 0.1)
    mul_177: "f32[64]" = torch.ops.aten.mul.Tensor(primals_333, 0.9)
    add_136: "f32[64]" = torch.ops.aten.add.Tensor(mul_176, mul_177);  mul_176 = mul_177 = None
    squeeze_77: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_144, [0, 2, 3]);  getitem_144 = None
    mul_178: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_77, 1.0001594642002871);  squeeze_77 = None
    mul_179: "f32[64]" = torch.ops.aten.mul.Tensor(mul_178, 0.1);  mul_178 = None
    mul_180: "f32[64]" = torch.ops.aten.mul.Tensor(primals_334, 0.9)
    add_137: "f32[64]" = torch.ops.aten.add.Tensor(mul_179, mul_180);  mul_179 = mul_180 = None
    unsqueeze_100: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_77, -1)
    unsqueeze_101: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
    mul_181: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(mul_175, unsqueeze_101);  mul_175 = unsqueeze_101 = None
    unsqueeze_102: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_78, -1);  primals_78 = None
    unsqueeze_103: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
    add_138: "f32[8, 64, 28, 28]" = torch.ops.aten.add.Tensor(mul_181, unsqueeze_103);  mul_181 = unsqueeze_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_23: "f32[8, 64, 28, 28]" = torch.ops.aten.relu.default(add_138);  add_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_23 = torch.ops.aten.split_with_sizes.default(relu_21, [64, 64, 64, 64], 1)
    getitem_148: "f32[8, 64, 28, 28]" = split_with_sizes_23[2];  split_with_sizes_23 = None
    add_139: "f32[8, 64, 28, 28]" = torch.ops.aten.add.Tensor(relu_23, getitem_148);  getitem_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_26: "f32[8, 64, 28, 28]" = torch.ops.aten.convolution.default(add_139, primals_79, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_140: "i64[]" = torch.ops.aten.add.Tensor(primals_338, 1)
    var_mean_26 = torch.ops.aten.var_mean.correction(convolution_26, [0, 2, 3], correction = 0, keepdim = True)
    getitem_150: "f32[1, 64, 1, 1]" = var_mean_26[0]
    getitem_151: "f32[1, 64, 1, 1]" = var_mean_26[1];  var_mean_26 = None
    add_141: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_150, 1e-05)
    rsqrt_26: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_141);  add_141 = None
    sub_26: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_26, getitem_151)
    mul_182: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_26);  sub_26 = None
    squeeze_78: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_151, [0, 2, 3]);  getitem_151 = None
    squeeze_79: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_26, [0, 2, 3]);  rsqrt_26 = None
    mul_183: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_78, 0.1)
    mul_184: "f32[64]" = torch.ops.aten.mul.Tensor(primals_336, 0.9)
    add_142: "f32[64]" = torch.ops.aten.add.Tensor(mul_183, mul_184);  mul_183 = mul_184 = None
    squeeze_80: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_150, [0, 2, 3]);  getitem_150 = None
    mul_185: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_80, 1.0001594642002871);  squeeze_80 = None
    mul_186: "f32[64]" = torch.ops.aten.mul.Tensor(mul_185, 0.1);  mul_185 = None
    mul_187: "f32[64]" = torch.ops.aten.mul.Tensor(primals_337, 0.9)
    add_143: "f32[64]" = torch.ops.aten.add.Tensor(mul_186, mul_187);  mul_186 = mul_187 = None
    unsqueeze_104: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_80, -1)
    unsqueeze_105: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
    mul_188: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(mul_182, unsqueeze_105);  mul_182 = unsqueeze_105 = None
    unsqueeze_106: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_81, -1);  primals_81 = None
    unsqueeze_107: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
    add_144: "f32[8, 64, 28, 28]" = torch.ops.aten.add.Tensor(mul_188, unsqueeze_107);  mul_188 = unsqueeze_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_24: "f32[8, 64, 28, 28]" = torch.ops.aten.relu.default(add_144);  add_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    split_with_sizes_24 = torch.ops.aten.split_with_sizes.default(relu_21, [64, 64, 64, 64], 1)
    getitem_155: "f32[8, 64, 28, 28]" = split_with_sizes_24[3];  split_with_sizes_24 = None
    cat_4: "f32[8, 256, 28, 28]" = torch.ops.aten.cat.default([relu_22, relu_23, relu_24, getitem_155], 1);  getitem_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_27: "f32[8, 512, 28, 28]" = torch.ops.aten.convolution.default(cat_4, primals_82, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    add_145: "i64[]" = torch.ops.aten.add.Tensor(primals_341, 1)
    var_mean_27 = torch.ops.aten.var_mean.correction(convolution_27, [0, 2, 3], correction = 0, keepdim = True)
    getitem_156: "f32[1, 512, 1, 1]" = var_mean_27[0]
    getitem_157: "f32[1, 512, 1, 1]" = var_mean_27[1];  var_mean_27 = None
    add_146: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_156, 1e-05)
    rsqrt_27: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_146);  add_146 = None
    sub_27: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_27, getitem_157)
    mul_189: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_27);  sub_27 = None
    squeeze_81: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_157, [0, 2, 3]);  getitem_157 = None
    squeeze_82: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_27, [0, 2, 3]);  rsqrt_27 = None
    mul_190: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_81, 0.1)
    mul_191: "f32[512]" = torch.ops.aten.mul.Tensor(primals_339, 0.9)
    add_147: "f32[512]" = torch.ops.aten.add.Tensor(mul_190, mul_191);  mul_190 = mul_191 = None
    squeeze_83: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_156, [0, 2, 3]);  getitem_156 = None
    mul_192: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_83, 1.0001594642002871);  squeeze_83 = None
    mul_193: "f32[512]" = torch.ops.aten.mul.Tensor(mul_192, 0.1);  mul_192 = None
    mul_194: "f32[512]" = torch.ops.aten.mul.Tensor(primals_340, 0.9)
    add_148: "f32[512]" = torch.ops.aten.add.Tensor(mul_193, mul_194);  mul_193 = mul_194 = None
    unsqueeze_108: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_83, -1)
    unsqueeze_109: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
    mul_195: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_189, unsqueeze_109);  mul_189 = unsqueeze_109 = None
    unsqueeze_110: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_84, -1);  primals_84 = None
    unsqueeze_111: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
    add_149: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_195, unsqueeze_111);  mul_195 = unsqueeze_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_150: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(add_149, relu_20);  add_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_25: "f32[8, 512, 28, 28]" = torch.ops.aten.relu.default(add_150);  add_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_28: "f32[8, 256, 28, 28]" = torch.ops.aten.convolution.default(relu_25, primals_85, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    add_151: "i64[]" = torch.ops.aten.add.Tensor(primals_344, 1)
    var_mean_28 = torch.ops.aten.var_mean.correction(convolution_28, [0, 2, 3], correction = 0, keepdim = True)
    getitem_158: "f32[1, 256, 1, 1]" = var_mean_28[0]
    getitem_159: "f32[1, 256, 1, 1]" = var_mean_28[1];  var_mean_28 = None
    add_152: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_158, 1e-05)
    rsqrt_28: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_152);  add_152 = None
    sub_28: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_28, getitem_159)
    mul_196: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_28);  sub_28 = None
    squeeze_84: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_159, [0, 2, 3]);  getitem_159 = None
    squeeze_85: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_28, [0, 2, 3]);  rsqrt_28 = None
    mul_197: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_84, 0.1)
    mul_198: "f32[256]" = torch.ops.aten.mul.Tensor(primals_342, 0.9)
    add_153: "f32[256]" = torch.ops.aten.add.Tensor(mul_197, mul_198);  mul_197 = mul_198 = None
    squeeze_86: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_158, [0, 2, 3]);  getitem_158 = None
    mul_199: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_86, 1.0001594642002871);  squeeze_86 = None
    mul_200: "f32[256]" = torch.ops.aten.mul.Tensor(mul_199, 0.1);  mul_199 = None
    mul_201: "f32[256]" = torch.ops.aten.mul.Tensor(primals_343, 0.9)
    add_154: "f32[256]" = torch.ops.aten.add.Tensor(mul_200, mul_201);  mul_200 = mul_201 = None
    unsqueeze_112: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_86, -1)
    unsqueeze_113: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
    mul_202: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(mul_196, unsqueeze_113);  mul_196 = unsqueeze_113 = None
    unsqueeze_114: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_87, -1);  primals_87 = None
    unsqueeze_115: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
    add_155: "f32[8, 256, 28, 28]" = torch.ops.aten.add.Tensor(mul_202, unsqueeze_115);  mul_202 = unsqueeze_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_26: "f32[8, 256, 28, 28]" = torch.ops.aten.relu.default(add_155);  add_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_26 = torch.ops.aten.split_with_sizes.default(relu_26, [64, 64, 64, 64], 1)
    getitem_164: "f32[8, 64, 28, 28]" = split_with_sizes_26[0];  split_with_sizes_26 = None
    convolution_29: "f32[8, 64, 28, 28]" = torch.ops.aten.convolution.default(getitem_164, primals_88, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_156: "i64[]" = torch.ops.aten.add.Tensor(primals_347, 1)
    var_mean_29 = torch.ops.aten.var_mean.correction(convolution_29, [0, 2, 3], correction = 0, keepdim = True)
    getitem_168: "f32[1, 64, 1, 1]" = var_mean_29[0]
    getitem_169: "f32[1, 64, 1, 1]" = var_mean_29[1];  var_mean_29 = None
    add_157: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_168, 1e-05)
    rsqrt_29: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_157);  add_157 = None
    sub_29: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_29, getitem_169)
    mul_203: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_29);  sub_29 = None
    squeeze_87: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_169, [0, 2, 3]);  getitem_169 = None
    squeeze_88: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_29, [0, 2, 3]);  rsqrt_29 = None
    mul_204: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_87, 0.1)
    mul_205: "f32[64]" = torch.ops.aten.mul.Tensor(primals_345, 0.9)
    add_158: "f32[64]" = torch.ops.aten.add.Tensor(mul_204, mul_205);  mul_204 = mul_205 = None
    squeeze_89: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_168, [0, 2, 3]);  getitem_168 = None
    mul_206: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_89, 1.0001594642002871);  squeeze_89 = None
    mul_207: "f32[64]" = torch.ops.aten.mul.Tensor(mul_206, 0.1);  mul_206 = None
    mul_208: "f32[64]" = torch.ops.aten.mul.Tensor(primals_346, 0.9)
    add_159: "f32[64]" = torch.ops.aten.add.Tensor(mul_207, mul_208);  mul_207 = mul_208 = None
    unsqueeze_116: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_89, -1)
    unsqueeze_117: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
    mul_209: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(mul_203, unsqueeze_117);  mul_203 = unsqueeze_117 = None
    unsqueeze_118: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_90, -1);  primals_90 = None
    unsqueeze_119: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
    add_160: "f32[8, 64, 28, 28]" = torch.ops.aten.add.Tensor(mul_209, unsqueeze_119);  mul_209 = unsqueeze_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_27: "f32[8, 64, 28, 28]" = torch.ops.aten.relu.default(add_160);  add_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_27 = torch.ops.aten.split_with_sizes.default(relu_26, [64, 64, 64, 64], 1)
    getitem_171: "f32[8, 64, 28, 28]" = split_with_sizes_27[1];  split_with_sizes_27 = None
    add_161: "f32[8, 64, 28, 28]" = torch.ops.aten.add.Tensor(relu_27, getitem_171);  getitem_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_30: "f32[8, 64, 28, 28]" = torch.ops.aten.convolution.default(add_161, primals_91, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_162: "i64[]" = torch.ops.aten.add.Tensor(primals_350, 1)
    var_mean_30 = torch.ops.aten.var_mean.correction(convolution_30, [0, 2, 3], correction = 0, keepdim = True)
    getitem_174: "f32[1, 64, 1, 1]" = var_mean_30[0]
    getitem_175: "f32[1, 64, 1, 1]" = var_mean_30[1];  var_mean_30 = None
    add_163: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_174, 1e-05)
    rsqrt_30: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_163);  add_163 = None
    sub_30: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_30, getitem_175)
    mul_210: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_30);  sub_30 = None
    squeeze_90: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_175, [0, 2, 3]);  getitem_175 = None
    squeeze_91: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_30, [0, 2, 3]);  rsqrt_30 = None
    mul_211: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_90, 0.1)
    mul_212: "f32[64]" = torch.ops.aten.mul.Tensor(primals_348, 0.9)
    add_164: "f32[64]" = torch.ops.aten.add.Tensor(mul_211, mul_212);  mul_211 = mul_212 = None
    squeeze_92: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_174, [0, 2, 3]);  getitem_174 = None
    mul_213: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_92, 1.0001594642002871);  squeeze_92 = None
    mul_214: "f32[64]" = torch.ops.aten.mul.Tensor(mul_213, 0.1);  mul_213 = None
    mul_215: "f32[64]" = torch.ops.aten.mul.Tensor(primals_349, 0.9)
    add_165: "f32[64]" = torch.ops.aten.add.Tensor(mul_214, mul_215);  mul_214 = mul_215 = None
    unsqueeze_120: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_92, -1)
    unsqueeze_121: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
    mul_216: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(mul_210, unsqueeze_121);  mul_210 = unsqueeze_121 = None
    unsqueeze_122: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_93, -1);  primals_93 = None
    unsqueeze_123: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
    add_166: "f32[8, 64, 28, 28]" = torch.ops.aten.add.Tensor(mul_216, unsqueeze_123);  mul_216 = unsqueeze_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_28: "f32[8, 64, 28, 28]" = torch.ops.aten.relu.default(add_166);  add_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_28 = torch.ops.aten.split_with_sizes.default(relu_26, [64, 64, 64, 64], 1)
    getitem_178: "f32[8, 64, 28, 28]" = split_with_sizes_28[2];  split_with_sizes_28 = None
    add_167: "f32[8, 64, 28, 28]" = torch.ops.aten.add.Tensor(relu_28, getitem_178);  getitem_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_31: "f32[8, 64, 28, 28]" = torch.ops.aten.convolution.default(add_167, primals_94, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_168: "i64[]" = torch.ops.aten.add.Tensor(primals_353, 1)
    var_mean_31 = torch.ops.aten.var_mean.correction(convolution_31, [0, 2, 3], correction = 0, keepdim = True)
    getitem_180: "f32[1, 64, 1, 1]" = var_mean_31[0]
    getitem_181: "f32[1, 64, 1, 1]" = var_mean_31[1];  var_mean_31 = None
    add_169: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_180, 1e-05)
    rsqrt_31: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_169);  add_169 = None
    sub_31: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_31, getitem_181)
    mul_217: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_31);  sub_31 = None
    squeeze_93: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_181, [0, 2, 3]);  getitem_181 = None
    squeeze_94: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_31, [0, 2, 3]);  rsqrt_31 = None
    mul_218: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_93, 0.1)
    mul_219: "f32[64]" = torch.ops.aten.mul.Tensor(primals_351, 0.9)
    add_170: "f32[64]" = torch.ops.aten.add.Tensor(mul_218, mul_219);  mul_218 = mul_219 = None
    squeeze_95: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_180, [0, 2, 3]);  getitem_180 = None
    mul_220: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_95, 1.0001594642002871);  squeeze_95 = None
    mul_221: "f32[64]" = torch.ops.aten.mul.Tensor(mul_220, 0.1);  mul_220 = None
    mul_222: "f32[64]" = torch.ops.aten.mul.Tensor(primals_352, 0.9)
    add_171: "f32[64]" = torch.ops.aten.add.Tensor(mul_221, mul_222);  mul_221 = mul_222 = None
    unsqueeze_124: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_95, -1)
    unsqueeze_125: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
    mul_223: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(mul_217, unsqueeze_125);  mul_217 = unsqueeze_125 = None
    unsqueeze_126: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_96, -1);  primals_96 = None
    unsqueeze_127: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
    add_172: "f32[8, 64, 28, 28]" = torch.ops.aten.add.Tensor(mul_223, unsqueeze_127);  mul_223 = unsqueeze_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_29: "f32[8, 64, 28, 28]" = torch.ops.aten.relu.default(add_172);  add_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    split_with_sizes_29 = torch.ops.aten.split_with_sizes.default(relu_26, [64, 64, 64, 64], 1)
    getitem_185: "f32[8, 64, 28, 28]" = split_with_sizes_29[3];  split_with_sizes_29 = None
    cat_5: "f32[8, 256, 28, 28]" = torch.ops.aten.cat.default([relu_27, relu_28, relu_29, getitem_185], 1);  getitem_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_32: "f32[8, 512, 28, 28]" = torch.ops.aten.convolution.default(cat_5, primals_97, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    add_173: "i64[]" = torch.ops.aten.add.Tensor(primals_356, 1)
    var_mean_32 = torch.ops.aten.var_mean.correction(convolution_32, [0, 2, 3], correction = 0, keepdim = True)
    getitem_186: "f32[1, 512, 1, 1]" = var_mean_32[0]
    getitem_187: "f32[1, 512, 1, 1]" = var_mean_32[1];  var_mean_32 = None
    add_174: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_186, 1e-05)
    rsqrt_32: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_174);  add_174 = None
    sub_32: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_32, getitem_187)
    mul_224: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_32);  sub_32 = None
    squeeze_96: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_187, [0, 2, 3]);  getitem_187 = None
    squeeze_97: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_32, [0, 2, 3]);  rsqrt_32 = None
    mul_225: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_96, 0.1)
    mul_226: "f32[512]" = torch.ops.aten.mul.Tensor(primals_354, 0.9)
    add_175: "f32[512]" = torch.ops.aten.add.Tensor(mul_225, mul_226);  mul_225 = mul_226 = None
    squeeze_98: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_186, [0, 2, 3]);  getitem_186 = None
    mul_227: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_98, 1.0001594642002871);  squeeze_98 = None
    mul_228: "f32[512]" = torch.ops.aten.mul.Tensor(mul_227, 0.1);  mul_227 = None
    mul_229: "f32[512]" = torch.ops.aten.mul.Tensor(primals_355, 0.9)
    add_176: "f32[512]" = torch.ops.aten.add.Tensor(mul_228, mul_229);  mul_228 = mul_229 = None
    unsqueeze_128: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_98, -1)
    unsqueeze_129: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_128, -1);  unsqueeze_128 = None
    mul_230: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_224, unsqueeze_129);  mul_224 = unsqueeze_129 = None
    unsqueeze_130: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_99, -1);  primals_99 = None
    unsqueeze_131: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_130, -1);  unsqueeze_130 = None
    add_177: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_230, unsqueeze_131);  mul_230 = unsqueeze_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_178: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(add_177, relu_25);  add_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_30: "f32[8, 512, 28, 28]" = torch.ops.aten.relu.default(add_178);  add_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_33: "f32[8, 256, 28, 28]" = torch.ops.aten.convolution.default(relu_30, primals_100, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    add_179: "i64[]" = torch.ops.aten.add.Tensor(primals_359, 1)
    var_mean_33 = torch.ops.aten.var_mean.correction(convolution_33, [0, 2, 3], correction = 0, keepdim = True)
    getitem_188: "f32[1, 256, 1, 1]" = var_mean_33[0]
    getitem_189: "f32[1, 256, 1, 1]" = var_mean_33[1];  var_mean_33 = None
    add_180: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_188, 1e-05)
    rsqrt_33: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_180);  add_180 = None
    sub_33: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_33, getitem_189)
    mul_231: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_33);  sub_33 = None
    squeeze_99: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_189, [0, 2, 3]);  getitem_189 = None
    squeeze_100: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_33, [0, 2, 3]);  rsqrt_33 = None
    mul_232: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_99, 0.1)
    mul_233: "f32[256]" = torch.ops.aten.mul.Tensor(primals_357, 0.9)
    add_181: "f32[256]" = torch.ops.aten.add.Tensor(mul_232, mul_233);  mul_232 = mul_233 = None
    squeeze_101: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_188, [0, 2, 3]);  getitem_188 = None
    mul_234: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_101, 1.0001594642002871);  squeeze_101 = None
    mul_235: "f32[256]" = torch.ops.aten.mul.Tensor(mul_234, 0.1);  mul_234 = None
    mul_236: "f32[256]" = torch.ops.aten.mul.Tensor(primals_358, 0.9)
    add_182: "f32[256]" = torch.ops.aten.add.Tensor(mul_235, mul_236);  mul_235 = mul_236 = None
    unsqueeze_132: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_101, -1)
    unsqueeze_133: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_132, -1);  unsqueeze_132 = None
    mul_237: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(mul_231, unsqueeze_133);  mul_231 = unsqueeze_133 = None
    unsqueeze_134: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_102, -1);  primals_102 = None
    unsqueeze_135: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_134, -1);  unsqueeze_134 = None
    add_183: "f32[8, 256, 28, 28]" = torch.ops.aten.add.Tensor(mul_237, unsqueeze_135);  mul_237 = unsqueeze_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_31: "f32[8, 256, 28, 28]" = torch.ops.aten.relu.default(add_183);  add_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_31 = torch.ops.aten.split_with_sizes.default(relu_31, [64, 64, 64, 64], 1)
    getitem_194: "f32[8, 64, 28, 28]" = split_with_sizes_31[0];  split_with_sizes_31 = None
    convolution_34: "f32[8, 64, 28, 28]" = torch.ops.aten.convolution.default(getitem_194, primals_103, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_184: "i64[]" = torch.ops.aten.add.Tensor(primals_362, 1)
    var_mean_34 = torch.ops.aten.var_mean.correction(convolution_34, [0, 2, 3], correction = 0, keepdim = True)
    getitem_198: "f32[1, 64, 1, 1]" = var_mean_34[0]
    getitem_199: "f32[1, 64, 1, 1]" = var_mean_34[1];  var_mean_34 = None
    add_185: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_198, 1e-05)
    rsqrt_34: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_185);  add_185 = None
    sub_34: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_34, getitem_199)
    mul_238: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_34);  sub_34 = None
    squeeze_102: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_199, [0, 2, 3]);  getitem_199 = None
    squeeze_103: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_34, [0, 2, 3]);  rsqrt_34 = None
    mul_239: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_102, 0.1)
    mul_240: "f32[64]" = torch.ops.aten.mul.Tensor(primals_360, 0.9)
    add_186: "f32[64]" = torch.ops.aten.add.Tensor(mul_239, mul_240);  mul_239 = mul_240 = None
    squeeze_104: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_198, [0, 2, 3]);  getitem_198 = None
    mul_241: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_104, 1.0001594642002871);  squeeze_104 = None
    mul_242: "f32[64]" = torch.ops.aten.mul.Tensor(mul_241, 0.1);  mul_241 = None
    mul_243: "f32[64]" = torch.ops.aten.mul.Tensor(primals_361, 0.9)
    add_187: "f32[64]" = torch.ops.aten.add.Tensor(mul_242, mul_243);  mul_242 = mul_243 = None
    unsqueeze_136: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_104, -1)
    unsqueeze_137: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_136, -1);  unsqueeze_136 = None
    mul_244: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(mul_238, unsqueeze_137);  mul_238 = unsqueeze_137 = None
    unsqueeze_138: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_105, -1);  primals_105 = None
    unsqueeze_139: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_138, -1);  unsqueeze_138 = None
    add_188: "f32[8, 64, 28, 28]" = torch.ops.aten.add.Tensor(mul_244, unsqueeze_139);  mul_244 = unsqueeze_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_32: "f32[8, 64, 28, 28]" = torch.ops.aten.relu.default(add_188);  add_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_32 = torch.ops.aten.split_with_sizes.default(relu_31, [64, 64, 64, 64], 1)
    getitem_201: "f32[8, 64, 28, 28]" = split_with_sizes_32[1];  split_with_sizes_32 = None
    add_189: "f32[8, 64, 28, 28]" = torch.ops.aten.add.Tensor(relu_32, getitem_201);  getitem_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_35: "f32[8, 64, 28, 28]" = torch.ops.aten.convolution.default(add_189, primals_106, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_190: "i64[]" = torch.ops.aten.add.Tensor(primals_365, 1)
    var_mean_35 = torch.ops.aten.var_mean.correction(convolution_35, [0, 2, 3], correction = 0, keepdim = True)
    getitem_204: "f32[1, 64, 1, 1]" = var_mean_35[0]
    getitem_205: "f32[1, 64, 1, 1]" = var_mean_35[1];  var_mean_35 = None
    add_191: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_204, 1e-05)
    rsqrt_35: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_191);  add_191 = None
    sub_35: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_35, getitem_205)
    mul_245: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_35);  sub_35 = None
    squeeze_105: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_205, [0, 2, 3]);  getitem_205 = None
    squeeze_106: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_35, [0, 2, 3]);  rsqrt_35 = None
    mul_246: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_105, 0.1)
    mul_247: "f32[64]" = torch.ops.aten.mul.Tensor(primals_363, 0.9)
    add_192: "f32[64]" = torch.ops.aten.add.Tensor(mul_246, mul_247);  mul_246 = mul_247 = None
    squeeze_107: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_204, [0, 2, 3]);  getitem_204 = None
    mul_248: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_107, 1.0001594642002871);  squeeze_107 = None
    mul_249: "f32[64]" = torch.ops.aten.mul.Tensor(mul_248, 0.1);  mul_248 = None
    mul_250: "f32[64]" = torch.ops.aten.mul.Tensor(primals_364, 0.9)
    add_193: "f32[64]" = torch.ops.aten.add.Tensor(mul_249, mul_250);  mul_249 = mul_250 = None
    unsqueeze_140: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_107, -1)
    unsqueeze_141: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_140, -1);  unsqueeze_140 = None
    mul_251: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(mul_245, unsqueeze_141);  mul_245 = unsqueeze_141 = None
    unsqueeze_142: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_108, -1);  primals_108 = None
    unsqueeze_143: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_142, -1);  unsqueeze_142 = None
    add_194: "f32[8, 64, 28, 28]" = torch.ops.aten.add.Tensor(mul_251, unsqueeze_143);  mul_251 = unsqueeze_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_33: "f32[8, 64, 28, 28]" = torch.ops.aten.relu.default(add_194);  add_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_33 = torch.ops.aten.split_with_sizes.default(relu_31, [64, 64, 64, 64], 1)
    getitem_208: "f32[8, 64, 28, 28]" = split_with_sizes_33[2];  split_with_sizes_33 = None
    add_195: "f32[8, 64, 28, 28]" = torch.ops.aten.add.Tensor(relu_33, getitem_208);  getitem_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_36: "f32[8, 64, 28, 28]" = torch.ops.aten.convolution.default(add_195, primals_109, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_196: "i64[]" = torch.ops.aten.add.Tensor(primals_368, 1)
    var_mean_36 = torch.ops.aten.var_mean.correction(convolution_36, [0, 2, 3], correction = 0, keepdim = True)
    getitem_210: "f32[1, 64, 1, 1]" = var_mean_36[0]
    getitem_211: "f32[1, 64, 1, 1]" = var_mean_36[1];  var_mean_36 = None
    add_197: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_210, 1e-05)
    rsqrt_36: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_197);  add_197 = None
    sub_36: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_36, getitem_211)
    mul_252: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_36);  sub_36 = None
    squeeze_108: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_211, [0, 2, 3]);  getitem_211 = None
    squeeze_109: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_36, [0, 2, 3]);  rsqrt_36 = None
    mul_253: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_108, 0.1)
    mul_254: "f32[64]" = torch.ops.aten.mul.Tensor(primals_366, 0.9)
    add_198: "f32[64]" = torch.ops.aten.add.Tensor(mul_253, mul_254);  mul_253 = mul_254 = None
    squeeze_110: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_210, [0, 2, 3]);  getitem_210 = None
    mul_255: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_110, 1.0001594642002871);  squeeze_110 = None
    mul_256: "f32[64]" = torch.ops.aten.mul.Tensor(mul_255, 0.1);  mul_255 = None
    mul_257: "f32[64]" = torch.ops.aten.mul.Tensor(primals_367, 0.9)
    add_199: "f32[64]" = torch.ops.aten.add.Tensor(mul_256, mul_257);  mul_256 = mul_257 = None
    unsqueeze_144: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_110, -1)
    unsqueeze_145: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_144, -1);  unsqueeze_144 = None
    mul_258: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(mul_252, unsqueeze_145);  mul_252 = unsqueeze_145 = None
    unsqueeze_146: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_111, -1);  primals_111 = None
    unsqueeze_147: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_146, -1);  unsqueeze_146 = None
    add_200: "f32[8, 64, 28, 28]" = torch.ops.aten.add.Tensor(mul_258, unsqueeze_147);  mul_258 = unsqueeze_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_34: "f32[8, 64, 28, 28]" = torch.ops.aten.relu.default(add_200);  add_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    split_with_sizes_34 = torch.ops.aten.split_with_sizes.default(relu_31, [64, 64, 64, 64], 1)
    getitem_215: "f32[8, 64, 28, 28]" = split_with_sizes_34[3];  split_with_sizes_34 = None
    cat_6: "f32[8, 256, 28, 28]" = torch.ops.aten.cat.default([relu_32, relu_33, relu_34, getitem_215], 1);  getitem_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_37: "f32[8, 512, 28, 28]" = torch.ops.aten.convolution.default(cat_6, primals_112, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    add_201: "i64[]" = torch.ops.aten.add.Tensor(primals_371, 1)
    var_mean_37 = torch.ops.aten.var_mean.correction(convolution_37, [0, 2, 3], correction = 0, keepdim = True)
    getitem_216: "f32[1, 512, 1, 1]" = var_mean_37[0]
    getitem_217: "f32[1, 512, 1, 1]" = var_mean_37[1];  var_mean_37 = None
    add_202: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_216, 1e-05)
    rsqrt_37: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_202);  add_202 = None
    sub_37: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_37, getitem_217)
    mul_259: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_37);  sub_37 = None
    squeeze_111: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_217, [0, 2, 3]);  getitem_217 = None
    squeeze_112: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_37, [0, 2, 3]);  rsqrt_37 = None
    mul_260: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_111, 0.1)
    mul_261: "f32[512]" = torch.ops.aten.mul.Tensor(primals_369, 0.9)
    add_203: "f32[512]" = torch.ops.aten.add.Tensor(mul_260, mul_261);  mul_260 = mul_261 = None
    squeeze_113: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_216, [0, 2, 3]);  getitem_216 = None
    mul_262: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_113, 1.0001594642002871);  squeeze_113 = None
    mul_263: "f32[512]" = torch.ops.aten.mul.Tensor(mul_262, 0.1);  mul_262 = None
    mul_264: "f32[512]" = torch.ops.aten.mul.Tensor(primals_370, 0.9)
    add_204: "f32[512]" = torch.ops.aten.add.Tensor(mul_263, mul_264);  mul_263 = mul_264 = None
    unsqueeze_148: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_113, -1)
    unsqueeze_149: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_148, -1);  unsqueeze_148 = None
    mul_265: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_259, unsqueeze_149);  mul_259 = unsqueeze_149 = None
    unsqueeze_150: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_114, -1);  primals_114 = None
    unsqueeze_151: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_150, -1);  unsqueeze_150 = None
    add_205: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_265, unsqueeze_151);  mul_265 = unsqueeze_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_206: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(add_205, relu_30);  add_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_35: "f32[8, 512, 28, 28]" = torch.ops.aten.relu.default(add_206);  add_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_38: "f32[8, 512, 28, 28]" = torch.ops.aten.convolution.default(relu_35, primals_115, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    add_207: "i64[]" = torch.ops.aten.add.Tensor(primals_374, 1)
    var_mean_38 = torch.ops.aten.var_mean.correction(convolution_38, [0, 2, 3], correction = 0, keepdim = True)
    getitem_218: "f32[1, 512, 1, 1]" = var_mean_38[0]
    getitem_219: "f32[1, 512, 1, 1]" = var_mean_38[1];  var_mean_38 = None
    add_208: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_218, 1e-05)
    rsqrt_38: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_208);  add_208 = None
    sub_38: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_38, getitem_219)
    mul_266: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_38);  sub_38 = None
    squeeze_114: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_219, [0, 2, 3]);  getitem_219 = None
    squeeze_115: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_38, [0, 2, 3]);  rsqrt_38 = None
    mul_267: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_114, 0.1)
    mul_268: "f32[512]" = torch.ops.aten.mul.Tensor(primals_372, 0.9)
    add_209: "f32[512]" = torch.ops.aten.add.Tensor(mul_267, mul_268);  mul_267 = mul_268 = None
    squeeze_116: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_218, [0, 2, 3]);  getitem_218 = None
    mul_269: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_116, 1.0001594642002871);  squeeze_116 = None
    mul_270: "f32[512]" = torch.ops.aten.mul.Tensor(mul_269, 0.1);  mul_269 = None
    mul_271: "f32[512]" = torch.ops.aten.mul.Tensor(primals_373, 0.9)
    add_210: "f32[512]" = torch.ops.aten.add.Tensor(mul_270, mul_271);  mul_270 = mul_271 = None
    unsqueeze_152: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_116, -1)
    unsqueeze_153: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_152, -1);  unsqueeze_152 = None
    mul_272: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_266, unsqueeze_153);  mul_266 = unsqueeze_153 = None
    unsqueeze_154: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_117, -1);  primals_117 = None
    unsqueeze_155: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_154, -1);  unsqueeze_154 = None
    add_211: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_272, unsqueeze_155);  mul_272 = unsqueeze_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_36: "f32[8, 512, 28, 28]" = torch.ops.aten.relu.default(add_211);  add_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_36 = torch.ops.aten.split_with_sizes.default(relu_36, [128, 128, 128, 128], 1)
    getitem_224: "f32[8, 128, 28, 28]" = split_with_sizes_36[0];  split_with_sizes_36 = None
    convolution_39: "f32[8, 128, 14, 14]" = torch.ops.aten.convolution.default(getitem_224, primals_118, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_212: "i64[]" = torch.ops.aten.add.Tensor(primals_377, 1)
    var_mean_39 = torch.ops.aten.var_mean.correction(convolution_39, [0, 2, 3], correction = 0, keepdim = True)
    getitem_228: "f32[1, 128, 1, 1]" = var_mean_39[0]
    getitem_229: "f32[1, 128, 1, 1]" = var_mean_39[1];  var_mean_39 = None
    add_213: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_228, 1e-05)
    rsqrt_39: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_213);  add_213 = None
    sub_39: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_39, getitem_229)
    mul_273: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_39);  sub_39 = None
    squeeze_117: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_229, [0, 2, 3]);  getitem_229 = None
    squeeze_118: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_39, [0, 2, 3]);  rsqrt_39 = None
    mul_274: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_117, 0.1)
    mul_275: "f32[128]" = torch.ops.aten.mul.Tensor(primals_375, 0.9)
    add_214: "f32[128]" = torch.ops.aten.add.Tensor(mul_274, mul_275);  mul_274 = mul_275 = None
    squeeze_119: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_228, [0, 2, 3]);  getitem_228 = None
    mul_276: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_119, 1.0006381620931717);  squeeze_119 = None
    mul_277: "f32[128]" = torch.ops.aten.mul.Tensor(mul_276, 0.1);  mul_276 = None
    mul_278: "f32[128]" = torch.ops.aten.mul.Tensor(primals_376, 0.9)
    add_215: "f32[128]" = torch.ops.aten.add.Tensor(mul_277, mul_278);  mul_277 = mul_278 = None
    unsqueeze_156: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_119, -1)
    unsqueeze_157: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_156, -1);  unsqueeze_156 = None
    mul_279: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(mul_273, unsqueeze_157);  mul_273 = unsqueeze_157 = None
    unsqueeze_158: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_120, -1);  primals_120 = None
    unsqueeze_159: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_158, -1);  unsqueeze_158 = None
    add_216: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(mul_279, unsqueeze_159);  mul_279 = unsqueeze_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_37: "f32[8, 128, 14, 14]" = torch.ops.aten.relu.default(add_216);  add_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_37 = torch.ops.aten.split_with_sizes.default(relu_36, [128, 128, 128, 128], 1)
    getitem_231: "f32[8, 128, 28, 28]" = split_with_sizes_37[1];  split_with_sizes_37 = None
    convolution_40: "f32[8, 128, 14, 14]" = torch.ops.aten.convolution.default(getitem_231, primals_121, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_217: "i64[]" = torch.ops.aten.add.Tensor(primals_380, 1)
    var_mean_40 = torch.ops.aten.var_mean.correction(convolution_40, [0, 2, 3], correction = 0, keepdim = True)
    getitem_234: "f32[1, 128, 1, 1]" = var_mean_40[0]
    getitem_235: "f32[1, 128, 1, 1]" = var_mean_40[1];  var_mean_40 = None
    add_218: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_234, 1e-05)
    rsqrt_40: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_218);  add_218 = None
    sub_40: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_40, getitem_235)
    mul_280: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_40);  sub_40 = None
    squeeze_120: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_235, [0, 2, 3]);  getitem_235 = None
    squeeze_121: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_40, [0, 2, 3]);  rsqrt_40 = None
    mul_281: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_120, 0.1)
    mul_282: "f32[128]" = torch.ops.aten.mul.Tensor(primals_378, 0.9)
    add_219: "f32[128]" = torch.ops.aten.add.Tensor(mul_281, mul_282);  mul_281 = mul_282 = None
    squeeze_122: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_234, [0, 2, 3]);  getitem_234 = None
    mul_283: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_122, 1.0006381620931717);  squeeze_122 = None
    mul_284: "f32[128]" = torch.ops.aten.mul.Tensor(mul_283, 0.1);  mul_283 = None
    mul_285: "f32[128]" = torch.ops.aten.mul.Tensor(primals_379, 0.9)
    add_220: "f32[128]" = torch.ops.aten.add.Tensor(mul_284, mul_285);  mul_284 = mul_285 = None
    unsqueeze_160: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_122, -1)
    unsqueeze_161: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_160, -1);  unsqueeze_160 = None
    mul_286: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(mul_280, unsqueeze_161);  mul_280 = unsqueeze_161 = None
    unsqueeze_162: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_123, -1);  primals_123 = None
    unsqueeze_163: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_162, -1);  unsqueeze_162 = None
    add_221: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(mul_286, unsqueeze_163);  mul_286 = unsqueeze_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_38: "f32[8, 128, 14, 14]" = torch.ops.aten.relu.default(add_221);  add_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_38 = torch.ops.aten.split_with_sizes.default(relu_36, [128, 128, 128, 128], 1)
    getitem_238: "f32[8, 128, 28, 28]" = split_with_sizes_38[2];  split_with_sizes_38 = None
    convolution_41: "f32[8, 128, 14, 14]" = torch.ops.aten.convolution.default(getitem_238, primals_124, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_222: "i64[]" = torch.ops.aten.add.Tensor(primals_383, 1)
    var_mean_41 = torch.ops.aten.var_mean.correction(convolution_41, [0, 2, 3], correction = 0, keepdim = True)
    getitem_240: "f32[1, 128, 1, 1]" = var_mean_41[0]
    getitem_241: "f32[1, 128, 1, 1]" = var_mean_41[1];  var_mean_41 = None
    add_223: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_240, 1e-05)
    rsqrt_41: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_223);  add_223 = None
    sub_41: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_41, getitem_241)
    mul_287: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_41);  sub_41 = None
    squeeze_123: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_241, [0, 2, 3]);  getitem_241 = None
    squeeze_124: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_41, [0, 2, 3]);  rsqrt_41 = None
    mul_288: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_123, 0.1)
    mul_289: "f32[128]" = torch.ops.aten.mul.Tensor(primals_381, 0.9)
    add_224: "f32[128]" = torch.ops.aten.add.Tensor(mul_288, mul_289);  mul_288 = mul_289 = None
    squeeze_125: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_240, [0, 2, 3]);  getitem_240 = None
    mul_290: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_125, 1.0006381620931717);  squeeze_125 = None
    mul_291: "f32[128]" = torch.ops.aten.mul.Tensor(mul_290, 0.1);  mul_290 = None
    mul_292: "f32[128]" = torch.ops.aten.mul.Tensor(primals_382, 0.9)
    add_225: "f32[128]" = torch.ops.aten.add.Tensor(mul_291, mul_292);  mul_291 = mul_292 = None
    unsqueeze_164: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_125, -1)
    unsqueeze_165: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_164, -1);  unsqueeze_164 = None
    mul_293: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(mul_287, unsqueeze_165);  mul_287 = unsqueeze_165 = None
    unsqueeze_166: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_126, -1);  primals_126 = None
    unsqueeze_167: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_166, -1);  unsqueeze_166 = None
    add_226: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(mul_293, unsqueeze_167);  mul_293 = unsqueeze_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_39: "f32[8, 128, 14, 14]" = torch.ops.aten.relu.default(add_226);  add_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:99, code: spo.append(self.pool(spx[-1]))
    split_with_sizes_39 = torch.ops.aten.split_with_sizes.default(relu_36, [128, 128, 128, 128], 1)
    getitem_245: "f32[8, 128, 28, 28]" = split_with_sizes_39[3];  split_with_sizes_39 = None
    avg_pool2d_2: "f32[8, 128, 14, 14]" = torch.ops.aten.avg_pool2d.default(getitem_245, [3, 3], [2, 2], [1, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    cat_7: "f32[8, 512, 14, 14]" = torch.ops.aten.cat.default([relu_37, relu_38, relu_39, avg_pool2d_2], 1);  avg_pool2d_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_42: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(cat_7, primals_127, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    add_227: "i64[]" = torch.ops.aten.add.Tensor(primals_386, 1)
    var_mean_42 = torch.ops.aten.var_mean.correction(convolution_42, [0, 2, 3], correction = 0, keepdim = True)
    getitem_246: "f32[1, 1024, 1, 1]" = var_mean_42[0]
    getitem_247: "f32[1, 1024, 1, 1]" = var_mean_42[1];  var_mean_42 = None
    add_228: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_246, 1e-05)
    rsqrt_42: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_228);  add_228 = None
    sub_42: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_42, getitem_247)
    mul_294: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_42);  sub_42 = None
    squeeze_126: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_247, [0, 2, 3]);  getitem_247 = None
    squeeze_127: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_42, [0, 2, 3]);  rsqrt_42 = None
    mul_295: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_126, 0.1)
    mul_296: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_384, 0.9)
    add_229: "f32[1024]" = torch.ops.aten.add.Tensor(mul_295, mul_296);  mul_295 = mul_296 = None
    squeeze_128: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_246, [0, 2, 3]);  getitem_246 = None
    mul_297: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_128, 1.0006381620931717);  squeeze_128 = None
    mul_298: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_297, 0.1);  mul_297 = None
    mul_299: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_385, 0.9)
    add_230: "f32[1024]" = torch.ops.aten.add.Tensor(mul_298, mul_299);  mul_298 = mul_299 = None
    unsqueeze_168: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_128, -1)
    unsqueeze_169: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_168, -1);  unsqueeze_168 = None
    mul_300: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_294, unsqueeze_169);  mul_294 = unsqueeze_169 = None
    unsqueeze_170: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_129, -1);  primals_129 = None
    unsqueeze_171: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_170, -1);  unsqueeze_170 = None
    add_231: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_300, unsqueeze_171);  mul_300 = unsqueeze_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:111, code: shortcut = self.downsample(x)
    convolution_43: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_35, primals_130, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
    add_232: "i64[]" = torch.ops.aten.add.Tensor(primals_389, 1)
    var_mean_43 = torch.ops.aten.var_mean.correction(convolution_43, [0, 2, 3], correction = 0, keepdim = True)
    getitem_248: "f32[1, 1024, 1, 1]" = var_mean_43[0]
    getitem_249: "f32[1, 1024, 1, 1]" = var_mean_43[1];  var_mean_43 = None
    add_233: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_248, 1e-05)
    rsqrt_43: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_233);  add_233 = None
    sub_43: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_43, getitem_249)
    mul_301: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_43);  sub_43 = None
    squeeze_129: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_249, [0, 2, 3]);  getitem_249 = None
    squeeze_130: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_43, [0, 2, 3]);  rsqrt_43 = None
    mul_302: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_129, 0.1)
    mul_303: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_387, 0.9)
    add_234: "f32[1024]" = torch.ops.aten.add.Tensor(mul_302, mul_303);  mul_302 = mul_303 = None
    squeeze_131: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_248, [0, 2, 3]);  getitem_248 = None
    mul_304: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_131, 1.0006381620931717);  squeeze_131 = None
    mul_305: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_304, 0.1);  mul_304 = None
    mul_306: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_388, 0.9)
    add_235: "f32[1024]" = torch.ops.aten.add.Tensor(mul_305, mul_306);  mul_305 = mul_306 = None
    unsqueeze_172: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_131, -1)
    unsqueeze_173: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_172, -1);  unsqueeze_172 = None
    mul_307: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_301, unsqueeze_173);  mul_301 = unsqueeze_173 = None
    unsqueeze_174: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_132, -1);  primals_132 = None
    unsqueeze_175: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_174, -1);  unsqueeze_174 = None
    add_236: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_307, unsqueeze_175);  mul_307 = unsqueeze_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_237: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_231, add_236);  add_231 = add_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_40: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_237);  add_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_44: "f32[8, 512, 14, 14]" = torch.ops.aten.convolution.default(relu_40, primals_133, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    add_238: "i64[]" = torch.ops.aten.add.Tensor(primals_392, 1)
    var_mean_44 = torch.ops.aten.var_mean.correction(convolution_44, [0, 2, 3], correction = 0, keepdim = True)
    getitem_250: "f32[1, 512, 1, 1]" = var_mean_44[0]
    getitem_251: "f32[1, 512, 1, 1]" = var_mean_44[1];  var_mean_44 = None
    add_239: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_250, 1e-05)
    rsqrt_44: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_239);  add_239 = None
    sub_44: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_44, getitem_251)
    mul_308: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_44);  sub_44 = None
    squeeze_132: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_251, [0, 2, 3]);  getitem_251 = None
    squeeze_133: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_44, [0, 2, 3]);  rsqrt_44 = None
    mul_309: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_132, 0.1)
    mul_310: "f32[512]" = torch.ops.aten.mul.Tensor(primals_390, 0.9)
    add_240: "f32[512]" = torch.ops.aten.add.Tensor(mul_309, mul_310);  mul_309 = mul_310 = None
    squeeze_134: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_250, [0, 2, 3]);  getitem_250 = None
    mul_311: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_134, 1.0006381620931717);  squeeze_134 = None
    mul_312: "f32[512]" = torch.ops.aten.mul.Tensor(mul_311, 0.1);  mul_311 = None
    mul_313: "f32[512]" = torch.ops.aten.mul.Tensor(primals_391, 0.9)
    add_241: "f32[512]" = torch.ops.aten.add.Tensor(mul_312, mul_313);  mul_312 = mul_313 = None
    unsqueeze_176: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_134, -1)
    unsqueeze_177: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_176, -1);  unsqueeze_176 = None
    mul_314: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(mul_308, unsqueeze_177);  mul_308 = unsqueeze_177 = None
    unsqueeze_178: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_135, -1);  primals_135 = None
    unsqueeze_179: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_178, -1);  unsqueeze_178 = None
    add_242: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_314, unsqueeze_179);  mul_314 = unsqueeze_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_41: "f32[8, 512, 14, 14]" = torch.ops.aten.relu.default(add_242);  add_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_41 = torch.ops.aten.split_with_sizes.default(relu_41, [128, 128, 128, 128], 1)
    getitem_256: "f32[8, 128, 14, 14]" = split_with_sizes_41[0];  split_with_sizes_41 = None
    convolution_45: "f32[8, 128, 14, 14]" = torch.ops.aten.convolution.default(getitem_256, primals_136, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_243: "i64[]" = torch.ops.aten.add.Tensor(primals_395, 1)
    var_mean_45 = torch.ops.aten.var_mean.correction(convolution_45, [0, 2, 3], correction = 0, keepdim = True)
    getitem_260: "f32[1, 128, 1, 1]" = var_mean_45[0]
    getitem_261: "f32[1, 128, 1, 1]" = var_mean_45[1];  var_mean_45 = None
    add_244: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_260, 1e-05)
    rsqrt_45: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_244);  add_244 = None
    sub_45: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_45, getitem_261)
    mul_315: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_45);  sub_45 = None
    squeeze_135: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_261, [0, 2, 3]);  getitem_261 = None
    squeeze_136: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_45, [0, 2, 3]);  rsqrt_45 = None
    mul_316: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_135, 0.1)
    mul_317: "f32[128]" = torch.ops.aten.mul.Tensor(primals_393, 0.9)
    add_245: "f32[128]" = torch.ops.aten.add.Tensor(mul_316, mul_317);  mul_316 = mul_317 = None
    squeeze_137: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_260, [0, 2, 3]);  getitem_260 = None
    mul_318: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_137, 1.0006381620931717);  squeeze_137 = None
    mul_319: "f32[128]" = torch.ops.aten.mul.Tensor(mul_318, 0.1);  mul_318 = None
    mul_320: "f32[128]" = torch.ops.aten.mul.Tensor(primals_394, 0.9)
    add_246: "f32[128]" = torch.ops.aten.add.Tensor(mul_319, mul_320);  mul_319 = mul_320 = None
    unsqueeze_180: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_137, -1)
    unsqueeze_181: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_180, -1);  unsqueeze_180 = None
    mul_321: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(mul_315, unsqueeze_181);  mul_315 = unsqueeze_181 = None
    unsqueeze_182: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_138, -1);  primals_138 = None
    unsqueeze_183: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_182, -1);  unsqueeze_182 = None
    add_247: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(mul_321, unsqueeze_183);  mul_321 = unsqueeze_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_42: "f32[8, 128, 14, 14]" = torch.ops.aten.relu.default(add_247);  add_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_42 = torch.ops.aten.split_with_sizes.default(relu_41, [128, 128, 128, 128], 1)
    getitem_263: "f32[8, 128, 14, 14]" = split_with_sizes_42[1];  split_with_sizes_42 = None
    add_248: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(relu_42, getitem_263);  getitem_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_46: "f32[8, 128, 14, 14]" = torch.ops.aten.convolution.default(add_248, primals_139, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_249: "i64[]" = torch.ops.aten.add.Tensor(primals_398, 1)
    var_mean_46 = torch.ops.aten.var_mean.correction(convolution_46, [0, 2, 3], correction = 0, keepdim = True)
    getitem_266: "f32[1, 128, 1, 1]" = var_mean_46[0]
    getitem_267: "f32[1, 128, 1, 1]" = var_mean_46[1];  var_mean_46 = None
    add_250: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_266, 1e-05)
    rsqrt_46: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_250);  add_250 = None
    sub_46: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_46, getitem_267)
    mul_322: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_46);  sub_46 = None
    squeeze_138: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_267, [0, 2, 3]);  getitem_267 = None
    squeeze_139: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_46, [0, 2, 3]);  rsqrt_46 = None
    mul_323: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_138, 0.1)
    mul_324: "f32[128]" = torch.ops.aten.mul.Tensor(primals_396, 0.9)
    add_251: "f32[128]" = torch.ops.aten.add.Tensor(mul_323, mul_324);  mul_323 = mul_324 = None
    squeeze_140: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_266, [0, 2, 3]);  getitem_266 = None
    mul_325: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_140, 1.0006381620931717);  squeeze_140 = None
    mul_326: "f32[128]" = torch.ops.aten.mul.Tensor(mul_325, 0.1);  mul_325 = None
    mul_327: "f32[128]" = torch.ops.aten.mul.Tensor(primals_397, 0.9)
    add_252: "f32[128]" = torch.ops.aten.add.Tensor(mul_326, mul_327);  mul_326 = mul_327 = None
    unsqueeze_184: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_140, -1)
    unsqueeze_185: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_184, -1);  unsqueeze_184 = None
    mul_328: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(mul_322, unsqueeze_185);  mul_322 = unsqueeze_185 = None
    unsqueeze_186: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_141, -1);  primals_141 = None
    unsqueeze_187: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_186, -1);  unsqueeze_186 = None
    add_253: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(mul_328, unsqueeze_187);  mul_328 = unsqueeze_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_43: "f32[8, 128, 14, 14]" = torch.ops.aten.relu.default(add_253);  add_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_43 = torch.ops.aten.split_with_sizes.default(relu_41, [128, 128, 128, 128], 1)
    getitem_270: "f32[8, 128, 14, 14]" = split_with_sizes_43[2];  split_with_sizes_43 = None
    add_254: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(relu_43, getitem_270);  getitem_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_47: "f32[8, 128, 14, 14]" = torch.ops.aten.convolution.default(add_254, primals_142, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_255: "i64[]" = torch.ops.aten.add.Tensor(primals_401, 1)
    var_mean_47 = torch.ops.aten.var_mean.correction(convolution_47, [0, 2, 3], correction = 0, keepdim = True)
    getitem_272: "f32[1, 128, 1, 1]" = var_mean_47[0]
    getitem_273: "f32[1, 128, 1, 1]" = var_mean_47[1];  var_mean_47 = None
    add_256: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_272, 1e-05)
    rsqrt_47: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_256);  add_256 = None
    sub_47: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_47, getitem_273)
    mul_329: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_47);  sub_47 = None
    squeeze_141: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_273, [0, 2, 3]);  getitem_273 = None
    squeeze_142: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_47, [0, 2, 3]);  rsqrt_47 = None
    mul_330: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_141, 0.1)
    mul_331: "f32[128]" = torch.ops.aten.mul.Tensor(primals_399, 0.9)
    add_257: "f32[128]" = torch.ops.aten.add.Tensor(mul_330, mul_331);  mul_330 = mul_331 = None
    squeeze_143: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_272, [0, 2, 3]);  getitem_272 = None
    mul_332: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_143, 1.0006381620931717);  squeeze_143 = None
    mul_333: "f32[128]" = torch.ops.aten.mul.Tensor(mul_332, 0.1);  mul_332 = None
    mul_334: "f32[128]" = torch.ops.aten.mul.Tensor(primals_400, 0.9)
    add_258: "f32[128]" = torch.ops.aten.add.Tensor(mul_333, mul_334);  mul_333 = mul_334 = None
    unsqueeze_188: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_143, -1)
    unsqueeze_189: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_188, -1);  unsqueeze_188 = None
    mul_335: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(mul_329, unsqueeze_189);  mul_329 = unsqueeze_189 = None
    unsqueeze_190: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_144, -1);  primals_144 = None
    unsqueeze_191: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_190, -1);  unsqueeze_190 = None
    add_259: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(mul_335, unsqueeze_191);  mul_335 = unsqueeze_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_44: "f32[8, 128, 14, 14]" = torch.ops.aten.relu.default(add_259);  add_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    split_with_sizes_44 = torch.ops.aten.split_with_sizes.default(relu_41, [128, 128, 128, 128], 1)
    getitem_277: "f32[8, 128, 14, 14]" = split_with_sizes_44[3];  split_with_sizes_44 = None
    cat_8: "f32[8, 512, 14, 14]" = torch.ops.aten.cat.default([relu_42, relu_43, relu_44, getitem_277], 1);  getitem_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_48: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(cat_8, primals_145, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    add_260: "i64[]" = torch.ops.aten.add.Tensor(primals_404, 1)
    var_mean_48 = torch.ops.aten.var_mean.correction(convolution_48, [0, 2, 3], correction = 0, keepdim = True)
    getitem_278: "f32[1, 1024, 1, 1]" = var_mean_48[0]
    getitem_279: "f32[1, 1024, 1, 1]" = var_mean_48[1];  var_mean_48 = None
    add_261: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_278, 1e-05)
    rsqrt_48: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_261);  add_261 = None
    sub_48: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_48, getitem_279)
    mul_336: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_48);  sub_48 = None
    squeeze_144: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_279, [0, 2, 3]);  getitem_279 = None
    squeeze_145: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_48, [0, 2, 3]);  rsqrt_48 = None
    mul_337: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_144, 0.1)
    mul_338: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_402, 0.9)
    add_262: "f32[1024]" = torch.ops.aten.add.Tensor(mul_337, mul_338);  mul_337 = mul_338 = None
    squeeze_146: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_278, [0, 2, 3]);  getitem_278 = None
    mul_339: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_146, 1.0006381620931717);  squeeze_146 = None
    mul_340: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_339, 0.1);  mul_339 = None
    mul_341: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_403, 0.9)
    add_263: "f32[1024]" = torch.ops.aten.add.Tensor(mul_340, mul_341);  mul_340 = mul_341 = None
    unsqueeze_192: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_146, -1)
    unsqueeze_193: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_192, -1);  unsqueeze_192 = None
    mul_342: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_336, unsqueeze_193);  mul_336 = unsqueeze_193 = None
    unsqueeze_194: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_147, -1);  primals_147 = None
    unsqueeze_195: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_194, -1);  unsqueeze_194 = None
    add_264: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_342, unsqueeze_195);  mul_342 = unsqueeze_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_265: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_264, relu_40);  add_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_45: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_265);  add_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_49: "f32[8, 512, 14, 14]" = torch.ops.aten.convolution.default(relu_45, primals_148, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    add_266: "i64[]" = torch.ops.aten.add.Tensor(primals_407, 1)
    var_mean_49 = torch.ops.aten.var_mean.correction(convolution_49, [0, 2, 3], correction = 0, keepdim = True)
    getitem_280: "f32[1, 512, 1, 1]" = var_mean_49[0]
    getitem_281: "f32[1, 512, 1, 1]" = var_mean_49[1];  var_mean_49 = None
    add_267: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_280, 1e-05)
    rsqrt_49: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_267);  add_267 = None
    sub_49: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_49, getitem_281)
    mul_343: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_49);  sub_49 = None
    squeeze_147: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_281, [0, 2, 3]);  getitem_281 = None
    squeeze_148: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_49, [0, 2, 3]);  rsqrt_49 = None
    mul_344: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_147, 0.1)
    mul_345: "f32[512]" = torch.ops.aten.mul.Tensor(primals_405, 0.9)
    add_268: "f32[512]" = torch.ops.aten.add.Tensor(mul_344, mul_345);  mul_344 = mul_345 = None
    squeeze_149: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_280, [0, 2, 3]);  getitem_280 = None
    mul_346: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_149, 1.0006381620931717);  squeeze_149 = None
    mul_347: "f32[512]" = torch.ops.aten.mul.Tensor(mul_346, 0.1);  mul_346 = None
    mul_348: "f32[512]" = torch.ops.aten.mul.Tensor(primals_406, 0.9)
    add_269: "f32[512]" = torch.ops.aten.add.Tensor(mul_347, mul_348);  mul_347 = mul_348 = None
    unsqueeze_196: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_149, -1)
    unsqueeze_197: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_196, -1);  unsqueeze_196 = None
    mul_349: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(mul_343, unsqueeze_197);  mul_343 = unsqueeze_197 = None
    unsqueeze_198: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_150, -1);  primals_150 = None
    unsqueeze_199: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_198, -1);  unsqueeze_198 = None
    add_270: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_349, unsqueeze_199);  mul_349 = unsqueeze_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_46: "f32[8, 512, 14, 14]" = torch.ops.aten.relu.default(add_270);  add_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_46 = torch.ops.aten.split_with_sizes.default(relu_46, [128, 128, 128, 128], 1)
    getitem_286: "f32[8, 128, 14, 14]" = split_with_sizes_46[0];  split_with_sizes_46 = None
    convolution_50: "f32[8, 128, 14, 14]" = torch.ops.aten.convolution.default(getitem_286, primals_151, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_271: "i64[]" = torch.ops.aten.add.Tensor(primals_410, 1)
    var_mean_50 = torch.ops.aten.var_mean.correction(convolution_50, [0, 2, 3], correction = 0, keepdim = True)
    getitem_290: "f32[1, 128, 1, 1]" = var_mean_50[0]
    getitem_291: "f32[1, 128, 1, 1]" = var_mean_50[1];  var_mean_50 = None
    add_272: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_290, 1e-05)
    rsqrt_50: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_272);  add_272 = None
    sub_50: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_50, getitem_291)
    mul_350: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_50);  sub_50 = None
    squeeze_150: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_291, [0, 2, 3]);  getitem_291 = None
    squeeze_151: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_50, [0, 2, 3]);  rsqrt_50 = None
    mul_351: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_150, 0.1)
    mul_352: "f32[128]" = torch.ops.aten.mul.Tensor(primals_408, 0.9)
    add_273: "f32[128]" = torch.ops.aten.add.Tensor(mul_351, mul_352);  mul_351 = mul_352 = None
    squeeze_152: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_290, [0, 2, 3]);  getitem_290 = None
    mul_353: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_152, 1.0006381620931717);  squeeze_152 = None
    mul_354: "f32[128]" = torch.ops.aten.mul.Tensor(mul_353, 0.1);  mul_353 = None
    mul_355: "f32[128]" = torch.ops.aten.mul.Tensor(primals_409, 0.9)
    add_274: "f32[128]" = torch.ops.aten.add.Tensor(mul_354, mul_355);  mul_354 = mul_355 = None
    unsqueeze_200: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_152, -1)
    unsqueeze_201: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_200, -1);  unsqueeze_200 = None
    mul_356: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(mul_350, unsqueeze_201);  mul_350 = unsqueeze_201 = None
    unsqueeze_202: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_153, -1);  primals_153 = None
    unsqueeze_203: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_202, -1);  unsqueeze_202 = None
    add_275: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(mul_356, unsqueeze_203);  mul_356 = unsqueeze_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_47: "f32[8, 128, 14, 14]" = torch.ops.aten.relu.default(add_275);  add_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_47 = torch.ops.aten.split_with_sizes.default(relu_46, [128, 128, 128, 128], 1)
    getitem_293: "f32[8, 128, 14, 14]" = split_with_sizes_47[1];  split_with_sizes_47 = None
    add_276: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(relu_47, getitem_293);  getitem_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_51: "f32[8, 128, 14, 14]" = torch.ops.aten.convolution.default(add_276, primals_154, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_277: "i64[]" = torch.ops.aten.add.Tensor(primals_413, 1)
    var_mean_51 = torch.ops.aten.var_mean.correction(convolution_51, [0, 2, 3], correction = 0, keepdim = True)
    getitem_296: "f32[1, 128, 1, 1]" = var_mean_51[0]
    getitem_297: "f32[1, 128, 1, 1]" = var_mean_51[1];  var_mean_51 = None
    add_278: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_296, 1e-05)
    rsqrt_51: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_278);  add_278 = None
    sub_51: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_51, getitem_297)
    mul_357: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_51);  sub_51 = None
    squeeze_153: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_297, [0, 2, 3]);  getitem_297 = None
    squeeze_154: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_51, [0, 2, 3]);  rsqrt_51 = None
    mul_358: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_153, 0.1)
    mul_359: "f32[128]" = torch.ops.aten.mul.Tensor(primals_411, 0.9)
    add_279: "f32[128]" = torch.ops.aten.add.Tensor(mul_358, mul_359);  mul_358 = mul_359 = None
    squeeze_155: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_296, [0, 2, 3]);  getitem_296 = None
    mul_360: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_155, 1.0006381620931717);  squeeze_155 = None
    mul_361: "f32[128]" = torch.ops.aten.mul.Tensor(mul_360, 0.1);  mul_360 = None
    mul_362: "f32[128]" = torch.ops.aten.mul.Tensor(primals_412, 0.9)
    add_280: "f32[128]" = torch.ops.aten.add.Tensor(mul_361, mul_362);  mul_361 = mul_362 = None
    unsqueeze_204: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_155, -1)
    unsqueeze_205: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_204, -1);  unsqueeze_204 = None
    mul_363: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(mul_357, unsqueeze_205);  mul_357 = unsqueeze_205 = None
    unsqueeze_206: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_156, -1);  primals_156 = None
    unsqueeze_207: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_206, -1);  unsqueeze_206 = None
    add_281: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(mul_363, unsqueeze_207);  mul_363 = unsqueeze_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_48: "f32[8, 128, 14, 14]" = torch.ops.aten.relu.default(add_281);  add_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_48 = torch.ops.aten.split_with_sizes.default(relu_46, [128, 128, 128, 128], 1)
    getitem_300: "f32[8, 128, 14, 14]" = split_with_sizes_48[2];  split_with_sizes_48 = None
    add_282: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(relu_48, getitem_300);  getitem_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_52: "f32[8, 128, 14, 14]" = torch.ops.aten.convolution.default(add_282, primals_157, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_283: "i64[]" = torch.ops.aten.add.Tensor(primals_416, 1)
    var_mean_52 = torch.ops.aten.var_mean.correction(convolution_52, [0, 2, 3], correction = 0, keepdim = True)
    getitem_302: "f32[1, 128, 1, 1]" = var_mean_52[0]
    getitem_303: "f32[1, 128, 1, 1]" = var_mean_52[1];  var_mean_52 = None
    add_284: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_302, 1e-05)
    rsqrt_52: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_284);  add_284 = None
    sub_52: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_52, getitem_303)
    mul_364: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_52);  sub_52 = None
    squeeze_156: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_303, [0, 2, 3]);  getitem_303 = None
    squeeze_157: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_52, [0, 2, 3]);  rsqrt_52 = None
    mul_365: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_156, 0.1)
    mul_366: "f32[128]" = torch.ops.aten.mul.Tensor(primals_414, 0.9)
    add_285: "f32[128]" = torch.ops.aten.add.Tensor(mul_365, mul_366);  mul_365 = mul_366 = None
    squeeze_158: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_302, [0, 2, 3]);  getitem_302 = None
    mul_367: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_158, 1.0006381620931717);  squeeze_158 = None
    mul_368: "f32[128]" = torch.ops.aten.mul.Tensor(mul_367, 0.1);  mul_367 = None
    mul_369: "f32[128]" = torch.ops.aten.mul.Tensor(primals_415, 0.9)
    add_286: "f32[128]" = torch.ops.aten.add.Tensor(mul_368, mul_369);  mul_368 = mul_369 = None
    unsqueeze_208: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_158, -1)
    unsqueeze_209: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_208, -1);  unsqueeze_208 = None
    mul_370: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(mul_364, unsqueeze_209);  mul_364 = unsqueeze_209 = None
    unsqueeze_210: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_159, -1);  primals_159 = None
    unsqueeze_211: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_210, -1);  unsqueeze_210 = None
    add_287: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(mul_370, unsqueeze_211);  mul_370 = unsqueeze_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_49: "f32[8, 128, 14, 14]" = torch.ops.aten.relu.default(add_287);  add_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    split_with_sizes_49 = torch.ops.aten.split_with_sizes.default(relu_46, [128, 128, 128, 128], 1)
    getitem_307: "f32[8, 128, 14, 14]" = split_with_sizes_49[3];  split_with_sizes_49 = None
    cat_9: "f32[8, 512, 14, 14]" = torch.ops.aten.cat.default([relu_47, relu_48, relu_49, getitem_307], 1);  getitem_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_53: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(cat_9, primals_160, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    add_288: "i64[]" = torch.ops.aten.add.Tensor(primals_419, 1)
    var_mean_53 = torch.ops.aten.var_mean.correction(convolution_53, [0, 2, 3], correction = 0, keepdim = True)
    getitem_308: "f32[1, 1024, 1, 1]" = var_mean_53[0]
    getitem_309: "f32[1, 1024, 1, 1]" = var_mean_53[1];  var_mean_53 = None
    add_289: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_308, 1e-05)
    rsqrt_53: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_289);  add_289 = None
    sub_53: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_53, getitem_309)
    mul_371: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_53);  sub_53 = None
    squeeze_159: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_309, [0, 2, 3]);  getitem_309 = None
    squeeze_160: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_53, [0, 2, 3]);  rsqrt_53 = None
    mul_372: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_159, 0.1)
    mul_373: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_417, 0.9)
    add_290: "f32[1024]" = torch.ops.aten.add.Tensor(mul_372, mul_373);  mul_372 = mul_373 = None
    squeeze_161: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_308, [0, 2, 3]);  getitem_308 = None
    mul_374: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_161, 1.0006381620931717);  squeeze_161 = None
    mul_375: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_374, 0.1);  mul_374 = None
    mul_376: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_418, 0.9)
    add_291: "f32[1024]" = torch.ops.aten.add.Tensor(mul_375, mul_376);  mul_375 = mul_376 = None
    unsqueeze_212: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_161, -1)
    unsqueeze_213: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_212, -1);  unsqueeze_212 = None
    mul_377: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_371, unsqueeze_213);  mul_371 = unsqueeze_213 = None
    unsqueeze_214: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_162, -1);  primals_162 = None
    unsqueeze_215: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_214, -1);  unsqueeze_214 = None
    add_292: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_377, unsqueeze_215);  mul_377 = unsqueeze_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_293: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_292, relu_45);  add_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_50: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_293);  add_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_54: "f32[8, 512, 14, 14]" = torch.ops.aten.convolution.default(relu_50, primals_163, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    add_294: "i64[]" = torch.ops.aten.add.Tensor(primals_422, 1)
    var_mean_54 = torch.ops.aten.var_mean.correction(convolution_54, [0, 2, 3], correction = 0, keepdim = True)
    getitem_310: "f32[1, 512, 1, 1]" = var_mean_54[0]
    getitem_311: "f32[1, 512, 1, 1]" = var_mean_54[1];  var_mean_54 = None
    add_295: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_310, 1e-05)
    rsqrt_54: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_295);  add_295 = None
    sub_54: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_54, getitem_311)
    mul_378: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_54);  sub_54 = None
    squeeze_162: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_311, [0, 2, 3]);  getitem_311 = None
    squeeze_163: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_54, [0, 2, 3]);  rsqrt_54 = None
    mul_379: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_162, 0.1)
    mul_380: "f32[512]" = torch.ops.aten.mul.Tensor(primals_420, 0.9)
    add_296: "f32[512]" = torch.ops.aten.add.Tensor(mul_379, mul_380);  mul_379 = mul_380 = None
    squeeze_164: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_310, [0, 2, 3]);  getitem_310 = None
    mul_381: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_164, 1.0006381620931717);  squeeze_164 = None
    mul_382: "f32[512]" = torch.ops.aten.mul.Tensor(mul_381, 0.1);  mul_381 = None
    mul_383: "f32[512]" = torch.ops.aten.mul.Tensor(primals_421, 0.9)
    add_297: "f32[512]" = torch.ops.aten.add.Tensor(mul_382, mul_383);  mul_382 = mul_383 = None
    unsqueeze_216: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_164, -1)
    unsqueeze_217: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_216, -1);  unsqueeze_216 = None
    mul_384: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(mul_378, unsqueeze_217);  mul_378 = unsqueeze_217 = None
    unsqueeze_218: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_165, -1);  primals_165 = None
    unsqueeze_219: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_218, -1);  unsqueeze_218 = None
    add_298: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_384, unsqueeze_219);  mul_384 = unsqueeze_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_51: "f32[8, 512, 14, 14]" = torch.ops.aten.relu.default(add_298);  add_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_51 = torch.ops.aten.split_with_sizes.default(relu_51, [128, 128, 128, 128], 1)
    getitem_316: "f32[8, 128, 14, 14]" = split_with_sizes_51[0];  split_with_sizes_51 = None
    convolution_55: "f32[8, 128, 14, 14]" = torch.ops.aten.convolution.default(getitem_316, primals_166, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_299: "i64[]" = torch.ops.aten.add.Tensor(primals_425, 1)
    var_mean_55 = torch.ops.aten.var_mean.correction(convolution_55, [0, 2, 3], correction = 0, keepdim = True)
    getitem_320: "f32[1, 128, 1, 1]" = var_mean_55[0]
    getitem_321: "f32[1, 128, 1, 1]" = var_mean_55[1];  var_mean_55 = None
    add_300: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_320, 1e-05)
    rsqrt_55: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_300);  add_300 = None
    sub_55: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_55, getitem_321)
    mul_385: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_55, rsqrt_55);  sub_55 = None
    squeeze_165: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_321, [0, 2, 3]);  getitem_321 = None
    squeeze_166: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_55, [0, 2, 3]);  rsqrt_55 = None
    mul_386: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_165, 0.1)
    mul_387: "f32[128]" = torch.ops.aten.mul.Tensor(primals_423, 0.9)
    add_301: "f32[128]" = torch.ops.aten.add.Tensor(mul_386, mul_387);  mul_386 = mul_387 = None
    squeeze_167: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_320, [0, 2, 3]);  getitem_320 = None
    mul_388: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_167, 1.0006381620931717);  squeeze_167 = None
    mul_389: "f32[128]" = torch.ops.aten.mul.Tensor(mul_388, 0.1);  mul_388 = None
    mul_390: "f32[128]" = torch.ops.aten.mul.Tensor(primals_424, 0.9)
    add_302: "f32[128]" = torch.ops.aten.add.Tensor(mul_389, mul_390);  mul_389 = mul_390 = None
    unsqueeze_220: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_167, -1)
    unsqueeze_221: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_220, -1);  unsqueeze_220 = None
    mul_391: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(mul_385, unsqueeze_221);  mul_385 = unsqueeze_221 = None
    unsqueeze_222: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_168, -1);  primals_168 = None
    unsqueeze_223: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_222, -1);  unsqueeze_222 = None
    add_303: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(mul_391, unsqueeze_223);  mul_391 = unsqueeze_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_52: "f32[8, 128, 14, 14]" = torch.ops.aten.relu.default(add_303);  add_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_52 = torch.ops.aten.split_with_sizes.default(relu_51, [128, 128, 128, 128], 1)
    getitem_323: "f32[8, 128, 14, 14]" = split_with_sizes_52[1];  split_with_sizes_52 = None
    add_304: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(relu_52, getitem_323);  getitem_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_56: "f32[8, 128, 14, 14]" = torch.ops.aten.convolution.default(add_304, primals_169, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_305: "i64[]" = torch.ops.aten.add.Tensor(primals_428, 1)
    var_mean_56 = torch.ops.aten.var_mean.correction(convolution_56, [0, 2, 3], correction = 0, keepdim = True)
    getitem_326: "f32[1, 128, 1, 1]" = var_mean_56[0]
    getitem_327: "f32[1, 128, 1, 1]" = var_mean_56[1];  var_mean_56 = None
    add_306: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_326, 1e-05)
    rsqrt_56: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_306);  add_306 = None
    sub_56: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_56, getitem_327)
    mul_392: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_56);  sub_56 = None
    squeeze_168: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_327, [0, 2, 3]);  getitem_327 = None
    squeeze_169: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_56, [0, 2, 3]);  rsqrt_56 = None
    mul_393: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_168, 0.1)
    mul_394: "f32[128]" = torch.ops.aten.mul.Tensor(primals_426, 0.9)
    add_307: "f32[128]" = torch.ops.aten.add.Tensor(mul_393, mul_394);  mul_393 = mul_394 = None
    squeeze_170: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_326, [0, 2, 3]);  getitem_326 = None
    mul_395: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_170, 1.0006381620931717);  squeeze_170 = None
    mul_396: "f32[128]" = torch.ops.aten.mul.Tensor(mul_395, 0.1);  mul_395 = None
    mul_397: "f32[128]" = torch.ops.aten.mul.Tensor(primals_427, 0.9)
    add_308: "f32[128]" = torch.ops.aten.add.Tensor(mul_396, mul_397);  mul_396 = mul_397 = None
    unsqueeze_224: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_170, -1)
    unsqueeze_225: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_224, -1);  unsqueeze_224 = None
    mul_398: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(mul_392, unsqueeze_225);  mul_392 = unsqueeze_225 = None
    unsqueeze_226: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_171, -1);  primals_171 = None
    unsqueeze_227: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_226, -1);  unsqueeze_226 = None
    add_309: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(mul_398, unsqueeze_227);  mul_398 = unsqueeze_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_53: "f32[8, 128, 14, 14]" = torch.ops.aten.relu.default(add_309);  add_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_53 = torch.ops.aten.split_with_sizes.default(relu_51, [128, 128, 128, 128], 1)
    getitem_330: "f32[8, 128, 14, 14]" = split_with_sizes_53[2];  split_with_sizes_53 = None
    add_310: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(relu_53, getitem_330);  getitem_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_57: "f32[8, 128, 14, 14]" = torch.ops.aten.convolution.default(add_310, primals_172, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_311: "i64[]" = torch.ops.aten.add.Tensor(primals_431, 1)
    var_mean_57 = torch.ops.aten.var_mean.correction(convolution_57, [0, 2, 3], correction = 0, keepdim = True)
    getitem_332: "f32[1, 128, 1, 1]" = var_mean_57[0]
    getitem_333: "f32[1, 128, 1, 1]" = var_mean_57[1];  var_mean_57 = None
    add_312: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_332, 1e-05)
    rsqrt_57: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_312);  add_312 = None
    sub_57: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_57, getitem_333)
    mul_399: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_57, rsqrt_57);  sub_57 = None
    squeeze_171: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_333, [0, 2, 3]);  getitem_333 = None
    squeeze_172: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_57, [0, 2, 3]);  rsqrt_57 = None
    mul_400: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_171, 0.1)
    mul_401: "f32[128]" = torch.ops.aten.mul.Tensor(primals_429, 0.9)
    add_313: "f32[128]" = torch.ops.aten.add.Tensor(mul_400, mul_401);  mul_400 = mul_401 = None
    squeeze_173: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_332, [0, 2, 3]);  getitem_332 = None
    mul_402: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_173, 1.0006381620931717);  squeeze_173 = None
    mul_403: "f32[128]" = torch.ops.aten.mul.Tensor(mul_402, 0.1);  mul_402 = None
    mul_404: "f32[128]" = torch.ops.aten.mul.Tensor(primals_430, 0.9)
    add_314: "f32[128]" = torch.ops.aten.add.Tensor(mul_403, mul_404);  mul_403 = mul_404 = None
    unsqueeze_228: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_173, -1)
    unsqueeze_229: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_228, -1);  unsqueeze_228 = None
    mul_405: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(mul_399, unsqueeze_229);  mul_399 = unsqueeze_229 = None
    unsqueeze_230: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_174, -1);  primals_174 = None
    unsqueeze_231: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_230, -1);  unsqueeze_230 = None
    add_315: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(mul_405, unsqueeze_231);  mul_405 = unsqueeze_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_54: "f32[8, 128, 14, 14]" = torch.ops.aten.relu.default(add_315);  add_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    split_with_sizes_54 = torch.ops.aten.split_with_sizes.default(relu_51, [128, 128, 128, 128], 1)
    getitem_337: "f32[8, 128, 14, 14]" = split_with_sizes_54[3];  split_with_sizes_54 = None
    cat_10: "f32[8, 512, 14, 14]" = torch.ops.aten.cat.default([relu_52, relu_53, relu_54, getitem_337], 1);  getitem_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_58: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(cat_10, primals_175, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    add_316: "i64[]" = torch.ops.aten.add.Tensor(primals_434, 1)
    var_mean_58 = torch.ops.aten.var_mean.correction(convolution_58, [0, 2, 3], correction = 0, keepdim = True)
    getitem_338: "f32[1, 1024, 1, 1]" = var_mean_58[0]
    getitem_339: "f32[1, 1024, 1, 1]" = var_mean_58[1];  var_mean_58 = None
    add_317: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_338, 1e-05)
    rsqrt_58: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_317);  add_317 = None
    sub_58: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_58, getitem_339)
    mul_406: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_58, rsqrt_58);  sub_58 = None
    squeeze_174: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_339, [0, 2, 3]);  getitem_339 = None
    squeeze_175: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_58, [0, 2, 3]);  rsqrt_58 = None
    mul_407: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_174, 0.1)
    mul_408: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_432, 0.9)
    add_318: "f32[1024]" = torch.ops.aten.add.Tensor(mul_407, mul_408);  mul_407 = mul_408 = None
    squeeze_176: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_338, [0, 2, 3]);  getitem_338 = None
    mul_409: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_176, 1.0006381620931717);  squeeze_176 = None
    mul_410: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_409, 0.1);  mul_409 = None
    mul_411: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_433, 0.9)
    add_319: "f32[1024]" = torch.ops.aten.add.Tensor(mul_410, mul_411);  mul_410 = mul_411 = None
    unsqueeze_232: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_176, -1)
    unsqueeze_233: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_232, -1);  unsqueeze_232 = None
    mul_412: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_406, unsqueeze_233);  mul_406 = unsqueeze_233 = None
    unsqueeze_234: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_177, -1);  primals_177 = None
    unsqueeze_235: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_234, -1);  unsqueeze_234 = None
    add_320: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_412, unsqueeze_235);  mul_412 = unsqueeze_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_321: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_320, relu_50);  add_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_55: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_321);  add_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_59: "f32[8, 512, 14, 14]" = torch.ops.aten.convolution.default(relu_55, primals_178, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    add_322: "i64[]" = torch.ops.aten.add.Tensor(primals_437, 1)
    var_mean_59 = torch.ops.aten.var_mean.correction(convolution_59, [0, 2, 3], correction = 0, keepdim = True)
    getitem_340: "f32[1, 512, 1, 1]" = var_mean_59[0]
    getitem_341: "f32[1, 512, 1, 1]" = var_mean_59[1];  var_mean_59 = None
    add_323: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_340, 1e-05)
    rsqrt_59: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_323);  add_323 = None
    sub_59: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_59, getitem_341)
    mul_413: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_59);  sub_59 = None
    squeeze_177: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_341, [0, 2, 3]);  getitem_341 = None
    squeeze_178: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_59, [0, 2, 3]);  rsqrt_59 = None
    mul_414: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_177, 0.1)
    mul_415: "f32[512]" = torch.ops.aten.mul.Tensor(primals_435, 0.9)
    add_324: "f32[512]" = torch.ops.aten.add.Tensor(mul_414, mul_415);  mul_414 = mul_415 = None
    squeeze_179: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_340, [0, 2, 3]);  getitem_340 = None
    mul_416: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_179, 1.0006381620931717);  squeeze_179 = None
    mul_417: "f32[512]" = torch.ops.aten.mul.Tensor(mul_416, 0.1);  mul_416 = None
    mul_418: "f32[512]" = torch.ops.aten.mul.Tensor(primals_436, 0.9)
    add_325: "f32[512]" = torch.ops.aten.add.Tensor(mul_417, mul_418);  mul_417 = mul_418 = None
    unsqueeze_236: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_179, -1)
    unsqueeze_237: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_236, -1);  unsqueeze_236 = None
    mul_419: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(mul_413, unsqueeze_237);  mul_413 = unsqueeze_237 = None
    unsqueeze_238: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_180, -1);  primals_180 = None
    unsqueeze_239: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_238, -1);  unsqueeze_238 = None
    add_326: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_419, unsqueeze_239);  mul_419 = unsqueeze_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_56: "f32[8, 512, 14, 14]" = torch.ops.aten.relu.default(add_326);  add_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_56 = torch.ops.aten.split_with_sizes.default(relu_56, [128, 128, 128, 128], 1)
    getitem_346: "f32[8, 128, 14, 14]" = split_with_sizes_56[0];  split_with_sizes_56 = None
    convolution_60: "f32[8, 128, 14, 14]" = torch.ops.aten.convolution.default(getitem_346, primals_181, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_327: "i64[]" = torch.ops.aten.add.Tensor(primals_440, 1)
    var_mean_60 = torch.ops.aten.var_mean.correction(convolution_60, [0, 2, 3], correction = 0, keepdim = True)
    getitem_350: "f32[1, 128, 1, 1]" = var_mean_60[0]
    getitem_351: "f32[1, 128, 1, 1]" = var_mean_60[1];  var_mean_60 = None
    add_328: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_350, 1e-05)
    rsqrt_60: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_328);  add_328 = None
    sub_60: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_60, getitem_351)
    mul_420: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_60, rsqrt_60);  sub_60 = None
    squeeze_180: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_351, [0, 2, 3]);  getitem_351 = None
    squeeze_181: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_60, [0, 2, 3]);  rsqrt_60 = None
    mul_421: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_180, 0.1)
    mul_422: "f32[128]" = torch.ops.aten.mul.Tensor(primals_438, 0.9)
    add_329: "f32[128]" = torch.ops.aten.add.Tensor(mul_421, mul_422);  mul_421 = mul_422 = None
    squeeze_182: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_350, [0, 2, 3]);  getitem_350 = None
    mul_423: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_182, 1.0006381620931717);  squeeze_182 = None
    mul_424: "f32[128]" = torch.ops.aten.mul.Tensor(mul_423, 0.1);  mul_423 = None
    mul_425: "f32[128]" = torch.ops.aten.mul.Tensor(primals_439, 0.9)
    add_330: "f32[128]" = torch.ops.aten.add.Tensor(mul_424, mul_425);  mul_424 = mul_425 = None
    unsqueeze_240: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_182, -1)
    unsqueeze_241: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_240, -1);  unsqueeze_240 = None
    mul_426: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(mul_420, unsqueeze_241);  mul_420 = unsqueeze_241 = None
    unsqueeze_242: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_183, -1);  primals_183 = None
    unsqueeze_243: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_242, -1);  unsqueeze_242 = None
    add_331: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(mul_426, unsqueeze_243);  mul_426 = unsqueeze_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_57: "f32[8, 128, 14, 14]" = torch.ops.aten.relu.default(add_331);  add_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_57 = torch.ops.aten.split_with_sizes.default(relu_56, [128, 128, 128, 128], 1)
    getitem_353: "f32[8, 128, 14, 14]" = split_with_sizes_57[1];  split_with_sizes_57 = None
    add_332: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(relu_57, getitem_353);  getitem_353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_61: "f32[8, 128, 14, 14]" = torch.ops.aten.convolution.default(add_332, primals_184, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_333: "i64[]" = torch.ops.aten.add.Tensor(primals_443, 1)
    var_mean_61 = torch.ops.aten.var_mean.correction(convolution_61, [0, 2, 3], correction = 0, keepdim = True)
    getitem_356: "f32[1, 128, 1, 1]" = var_mean_61[0]
    getitem_357: "f32[1, 128, 1, 1]" = var_mean_61[1];  var_mean_61 = None
    add_334: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_356, 1e-05)
    rsqrt_61: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_334);  add_334 = None
    sub_61: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_61, getitem_357)
    mul_427: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_61, rsqrt_61);  sub_61 = None
    squeeze_183: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_357, [0, 2, 3]);  getitem_357 = None
    squeeze_184: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_61, [0, 2, 3]);  rsqrt_61 = None
    mul_428: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_183, 0.1)
    mul_429: "f32[128]" = torch.ops.aten.mul.Tensor(primals_441, 0.9)
    add_335: "f32[128]" = torch.ops.aten.add.Tensor(mul_428, mul_429);  mul_428 = mul_429 = None
    squeeze_185: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_356, [0, 2, 3]);  getitem_356 = None
    mul_430: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_185, 1.0006381620931717);  squeeze_185 = None
    mul_431: "f32[128]" = torch.ops.aten.mul.Tensor(mul_430, 0.1);  mul_430 = None
    mul_432: "f32[128]" = torch.ops.aten.mul.Tensor(primals_442, 0.9)
    add_336: "f32[128]" = torch.ops.aten.add.Tensor(mul_431, mul_432);  mul_431 = mul_432 = None
    unsqueeze_244: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_185, -1)
    unsqueeze_245: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_244, -1);  unsqueeze_244 = None
    mul_433: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(mul_427, unsqueeze_245);  mul_427 = unsqueeze_245 = None
    unsqueeze_246: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_186, -1);  primals_186 = None
    unsqueeze_247: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_246, -1);  unsqueeze_246 = None
    add_337: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(mul_433, unsqueeze_247);  mul_433 = unsqueeze_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_58: "f32[8, 128, 14, 14]" = torch.ops.aten.relu.default(add_337);  add_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_58 = torch.ops.aten.split_with_sizes.default(relu_56, [128, 128, 128, 128], 1)
    getitem_360: "f32[8, 128, 14, 14]" = split_with_sizes_58[2];  split_with_sizes_58 = None
    add_338: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(relu_58, getitem_360);  getitem_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_62: "f32[8, 128, 14, 14]" = torch.ops.aten.convolution.default(add_338, primals_187, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_339: "i64[]" = torch.ops.aten.add.Tensor(primals_446, 1)
    var_mean_62 = torch.ops.aten.var_mean.correction(convolution_62, [0, 2, 3], correction = 0, keepdim = True)
    getitem_362: "f32[1, 128, 1, 1]" = var_mean_62[0]
    getitem_363: "f32[1, 128, 1, 1]" = var_mean_62[1];  var_mean_62 = None
    add_340: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_362, 1e-05)
    rsqrt_62: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_340);  add_340 = None
    sub_62: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_62, getitem_363)
    mul_434: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_62);  sub_62 = None
    squeeze_186: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_363, [0, 2, 3]);  getitem_363 = None
    squeeze_187: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_62, [0, 2, 3]);  rsqrt_62 = None
    mul_435: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_186, 0.1)
    mul_436: "f32[128]" = torch.ops.aten.mul.Tensor(primals_444, 0.9)
    add_341: "f32[128]" = torch.ops.aten.add.Tensor(mul_435, mul_436);  mul_435 = mul_436 = None
    squeeze_188: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_362, [0, 2, 3]);  getitem_362 = None
    mul_437: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_188, 1.0006381620931717);  squeeze_188 = None
    mul_438: "f32[128]" = torch.ops.aten.mul.Tensor(mul_437, 0.1);  mul_437 = None
    mul_439: "f32[128]" = torch.ops.aten.mul.Tensor(primals_445, 0.9)
    add_342: "f32[128]" = torch.ops.aten.add.Tensor(mul_438, mul_439);  mul_438 = mul_439 = None
    unsqueeze_248: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_188, -1)
    unsqueeze_249: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_248, -1);  unsqueeze_248 = None
    mul_440: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(mul_434, unsqueeze_249);  mul_434 = unsqueeze_249 = None
    unsqueeze_250: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_189, -1);  primals_189 = None
    unsqueeze_251: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_250, -1);  unsqueeze_250 = None
    add_343: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(mul_440, unsqueeze_251);  mul_440 = unsqueeze_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_59: "f32[8, 128, 14, 14]" = torch.ops.aten.relu.default(add_343);  add_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    split_with_sizes_59 = torch.ops.aten.split_with_sizes.default(relu_56, [128, 128, 128, 128], 1)
    getitem_367: "f32[8, 128, 14, 14]" = split_with_sizes_59[3];  split_with_sizes_59 = None
    cat_11: "f32[8, 512, 14, 14]" = torch.ops.aten.cat.default([relu_57, relu_58, relu_59, getitem_367], 1);  getitem_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_63: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(cat_11, primals_190, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    add_344: "i64[]" = torch.ops.aten.add.Tensor(primals_449, 1)
    var_mean_63 = torch.ops.aten.var_mean.correction(convolution_63, [0, 2, 3], correction = 0, keepdim = True)
    getitem_368: "f32[1, 1024, 1, 1]" = var_mean_63[0]
    getitem_369: "f32[1, 1024, 1, 1]" = var_mean_63[1];  var_mean_63 = None
    add_345: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_368, 1e-05)
    rsqrt_63: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_345);  add_345 = None
    sub_63: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_63, getitem_369)
    mul_441: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_63, rsqrt_63);  sub_63 = None
    squeeze_189: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_369, [0, 2, 3]);  getitem_369 = None
    squeeze_190: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_63, [0, 2, 3]);  rsqrt_63 = None
    mul_442: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_189, 0.1)
    mul_443: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_447, 0.9)
    add_346: "f32[1024]" = torch.ops.aten.add.Tensor(mul_442, mul_443);  mul_442 = mul_443 = None
    squeeze_191: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_368, [0, 2, 3]);  getitem_368 = None
    mul_444: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_191, 1.0006381620931717);  squeeze_191 = None
    mul_445: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_444, 0.1);  mul_444 = None
    mul_446: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_448, 0.9)
    add_347: "f32[1024]" = torch.ops.aten.add.Tensor(mul_445, mul_446);  mul_445 = mul_446 = None
    unsqueeze_252: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_191, -1)
    unsqueeze_253: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_252, -1);  unsqueeze_252 = None
    mul_447: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_441, unsqueeze_253);  mul_441 = unsqueeze_253 = None
    unsqueeze_254: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_192, -1);  primals_192 = None
    unsqueeze_255: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_254, -1);  unsqueeze_254 = None
    add_348: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_447, unsqueeze_255);  mul_447 = unsqueeze_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_349: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_348, relu_55);  add_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_60: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_349);  add_349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_64: "f32[8, 512, 14, 14]" = torch.ops.aten.convolution.default(relu_60, primals_193, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    add_350: "i64[]" = torch.ops.aten.add.Tensor(primals_452, 1)
    var_mean_64 = torch.ops.aten.var_mean.correction(convolution_64, [0, 2, 3], correction = 0, keepdim = True)
    getitem_370: "f32[1, 512, 1, 1]" = var_mean_64[0]
    getitem_371: "f32[1, 512, 1, 1]" = var_mean_64[1];  var_mean_64 = None
    add_351: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_370, 1e-05)
    rsqrt_64: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_351);  add_351 = None
    sub_64: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_64, getitem_371)
    mul_448: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_64, rsqrt_64);  sub_64 = None
    squeeze_192: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_371, [0, 2, 3]);  getitem_371 = None
    squeeze_193: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_64, [0, 2, 3]);  rsqrt_64 = None
    mul_449: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_192, 0.1)
    mul_450: "f32[512]" = torch.ops.aten.mul.Tensor(primals_450, 0.9)
    add_352: "f32[512]" = torch.ops.aten.add.Tensor(mul_449, mul_450);  mul_449 = mul_450 = None
    squeeze_194: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_370, [0, 2, 3]);  getitem_370 = None
    mul_451: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_194, 1.0006381620931717);  squeeze_194 = None
    mul_452: "f32[512]" = torch.ops.aten.mul.Tensor(mul_451, 0.1);  mul_451 = None
    mul_453: "f32[512]" = torch.ops.aten.mul.Tensor(primals_451, 0.9)
    add_353: "f32[512]" = torch.ops.aten.add.Tensor(mul_452, mul_453);  mul_452 = mul_453 = None
    unsqueeze_256: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_194, -1)
    unsqueeze_257: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_256, -1);  unsqueeze_256 = None
    mul_454: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(mul_448, unsqueeze_257);  mul_448 = unsqueeze_257 = None
    unsqueeze_258: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_195, -1);  primals_195 = None
    unsqueeze_259: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_258, -1);  unsqueeze_258 = None
    add_354: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_454, unsqueeze_259);  mul_454 = unsqueeze_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_61: "f32[8, 512, 14, 14]" = torch.ops.aten.relu.default(add_354);  add_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_61 = torch.ops.aten.split_with_sizes.default(relu_61, [128, 128, 128, 128], 1)
    getitem_376: "f32[8, 128, 14, 14]" = split_with_sizes_61[0];  split_with_sizes_61 = None
    convolution_65: "f32[8, 128, 14, 14]" = torch.ops.aten.convolution.default(getitem_376, primals_196, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_355: "i64[]" = torch.ops.aten.add.Tensor(primals_455, 1)
    var_mean_65 = torch.ops.aten.var_mean.correction(convolution_65, [0, 2, 3], correction = 0, keepdim = True)
    getitem_380: "f32[1, 128, 1, 1]" = var_mean_65[0]
    getitem_381: "f32[1, 128, 1, 1]" = var_mean_65[1];  var_mean_65 = None
    add_356: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_380, 1e-05)
    rsqrt_65: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_356);  add_356 = None
    sub_65: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_65, getitem_381)
    mul_455: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_65, rsqrt_65);  sub_65 = None
    squeeze_195: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_381, [0, 2, 3]);  getitem_381 = None
    squeeze_196: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_65, [0, 2, 3]);  rsqrt_65 = None
    mul_456: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_195, 0.1)
    mul_457: "f32[128]" = torch.ops.aten.mul.Tensor(primals_453, 0.9)
    add_357: "f32[128]" = torch.ops.aten.add.Tensor(mul_456, mul_457);  mul_456 = mul_457 = None
    squeeze_197: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_380, [0, 2, 3]);  getitem_380 = None
    mul_458: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_197, 1.0006381620931717);  squeeze_197 = None
    mul_459: "f32[128]" = torch.ops.aten.mul.Tensor(mul_458, 0.1);  mul_458 = None
    mul_460: "f32[128]" = torch.ops.aten.mul.Tensor(primals_454, 0.9)
    add_358: "f32[128]" = torch.ops.aten.add.Tensor(mul_459, mul_460);  mul_459 = mul_460 = None
    unsqueeze_260: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_197, -1)
    unsqueeze_261: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_260, -1);  unsqueeze_260 = None
    mul_461: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(mul_455, unsqueeze_261);  mul_455 = unsqueeze_261 = None
    unsqueeze_262: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_198, -1);  primals_198 = None
    unsqueeze_263: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_262, -1);  unsqueeze_262 = None
    add_359: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(mul_461, unsqueeze_263);  mul_461 = unsqueeze_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_62: "f32[8, 128, 14, 14]" = torch.ops.aten.relu.default(add_359);  add_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_62 = torch.ops.aten.split_with_sizes.default(relu_61, [128, 128, 128, 128], 1)
    getitem_383: "f32[8, 128, 14, 14]" = split_with_sizes_62[1];  split_with_sizes_62 = None
    add_360: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(relu_62, getitem_383);  getitem_383 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_66: "f32[8, 128, 14, 14]" = torch.ops.aten.convolution.default(add_360, primals_199, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_361: "i64[]" = torch.ops.aten.add.Tensor(primals_458, 1)
    var_mean_66 = torch.ops.aten.var_mean.correction(convolution_66, [0, 2, 3], correction = 0, keepdim = True)
    getitem_386: "f32[1, 128, 1, 1]" = var_mean_66[0]
    getitem_387: "f32[1, 128, 1, 1]" = var_mean_66[1];  var_mean_66 = None
    add_362: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_386, 1e-05)
    rsqrt_66: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_362);  add_362 = None
    sub_66: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_66, getitem_387)
    mul_462: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_66, rsqrt_66);  sub_66 = None
    squeeze_198: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_387, [0, 2, 3]);  getitem_387 = None
    squeeze_199: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_66, [0, 2, 3]);  rsqrt_66 = None
    mul_463: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_198, 0.1)
    mul_464: "f32[128]" = torch.ops.aten.mul.Tensor(primals_456, 0.9)
    add_363: "f32[128]" = torch.ops.aten.add.Tensor(mul_463, mul_464);  mul_463 = mul_464 = None
    squeeze_200: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_386, [0, 2, 3]);  getitem_386 = None
    mul_465: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_200, 1.0006381620931717);  squeeze_200 = None
    mul_466: "f32[128]" = torch.ops.aten.mul.Tensor(mul_465, 0.1);  mul_465 = None
    mul_467: "f32[128]" = torch.ops.aten.mul.Tensor(primals_457, 0.9)
    add_364: "f32[128]" = torch.ops.aten.add.Tensor(mul_466, mul_467);  mul_466 = mul_467 = None
    unsqueeze_264: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_200, -1)
    unsqueeze_265: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_264, -1);  unsqueeze_264 = None
    mul_468: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(mul_462, unsqueeze_265);  mul_462 = unsqueeze_265 = None
    unsqueeze_266: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_201, -1);  primals_201 = None
    unsqueeze_267: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_266, -1);  unsqueeze_266 = None
    add_365: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(mul_468, unsqueeze_267);  mul_468 = unsqueeze_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_63: "f32[8, 128, 14, 14]" = torch.ops.aten.relu.default(add_365);  add_365 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_63 = torch.ops.aten.split_with_sizes.default(relu_61, [128, 128, 128, 128], 1)
    getitem_390: "f32[8, 128, 14, 14]" = split_with_sizes_63[2];  split_with_sizes_63 = None
    add_366: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(relu_63, getitem_390);  getitem_390 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_67: "f32[8, 128, 14, 14]" = torch.ops.aten.convolution.default(add_366, primals_202, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_367: "i64[]" = torch.ops.aten.add.Tensor(primals_461, 1)
    var_mean_67 = torch.ops.aten.var_mean.correction(convolution_67, [0, 2, 3], correction = 0, keepdim = True)
    getitem_392: "f32[1, 128, 1, 1]" = var_mean_67[0]
    getitem_393: "f32[1, 128, 1, 1]" = var_mean_67[1];  var_mean_67 = None
    add_368: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_392, 1e-05)
    rsqrt_67: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_368);  add_368 = None
    sub_67: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_67, getitem_393)
    mul_469: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_67, rsqrt_67);  sub_67 = None
    squeeze_201: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_393, [0, 2, 3]);  getitem_393 = None
    squeeze_202: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_67, [0, 2, 3]);  rsqrt_67 = None
    mul_470: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_201, 0.1)
    mul_471: "f32[128]" = torch.ops.aten.mul.Tensor(primals_459, 0.9)
    add_369: "f32[128]" = torch.ops.aten.add.Tensor(mul_470, mul_471);  mul_470 = mul_471 = None
    squeeze_203: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_392, [0, 2, 3]);  getitem_392 = None
    mul_472: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_203, 1.0006381620931717);  squeeze_203 = None
    mul_473: "f32[128]" = torch.ops.aten.mul.Tensor(mul_472, 0.1);  mul_472 = None
    mul_474: "f32[128]" = torch.ops.aten.mul.Tensor(primals_460, 0.9)
    add_370: "f32[128]" = torch.ops.aten.add.Tensor(mul_473, mul_474);  mul_473 = mul_474 = None
    unsqueeze_268: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_203, -1)
    unsqueeze_269: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_268, -1);  unsqueeze_268 = None
    mul_475: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(mul_469, unsqueeze_269);  mul_469 = unsqueeze_269 = None
    unsqueeze_270: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_204, -1);  primals_204 = None
    unsqueeze_271: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_270, -1);  unsqueeze_270 = None
    add_371: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(mul_475, unsqueeze_271);  mul_475 = unsqueeze_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_64: "f32[8, 128, 14, 14]" = torch.ops.aten.relu.default(add_371);  add_371 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    split_with_sizes_64 = torch.ops.aten.split_with_sizes.default(relu_61, [128, 128, 128, 128], 1)
    getitem_397: "f32[8, 128, 14, 14]" = split_with_sizes_64[3];  split_with_sizes_64 = None
    cat_12: "f32[8, 512, 14, 14]" = torch.ops.aten.cat.default([relu_62, relu_63, relu_64, getitem_397], 1);  getitem_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_68: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(cat_12, primals_205, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    add_372: "i64[]" = torch.ops.aten.add.Tensor(primals_464, 1)
    var_mean_68 = torch.ops.aten.var_mean.correction(convolution_68, [0, 2, 3], correction = 0, keepdim = True)
    getitem_398: "f32[1, 1024, 1, 1]" = var_mean_68[0]
    getitem_399: "f32[1, 1024, 1, 1]" = var_mean_68[1];  var_mean_68 = None
    add_373: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_398, 1e-05)
    rsqrt_68: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_373);  add_373 = None
    sub_68: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_68, getitem_399)
    mul_476: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_68, rsqrt_68);  sub_68 = None
    squeeze_204: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_399, [0, 2, 3]);  getitem_399 = None
    squeeze_205: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_68, [0, 2, 3]);  rsqrt_68 = None
    mul_477: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_204, 0.1)
    mul_478: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_462, 0.9)
    add_374: "f32[1024]" = torch.ops.aten.add.Tensor(mul_477, mul_478);  mul_477 = mul_478 = None
    squeeze_206: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_398, [0, 2, 3]);  getitem_398 = None
    mul_479: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_206, 1.0006381620931717);  squeeze_206 = None
    mul_480: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_479, 0.1);  mul_479 = None
    mul_481: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_463, 0.9)
    add_375: "f32[1024]" = torch.ops.aten.add.Tensor(mul_480, mul_481);  mul_480 = mul_481 = None
    unsqueeze_272: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_206, -1)
    unsqueeze_273: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_272, -1);  unsqueeze_272 = None
    mul_482: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_476, unsqueeze_273);  mul_476 = unsqueeze_273 = None
    unsqueeze_274: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_207, -1);  primals_207 = None
    unsqueeze_275: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_274, -1);  unsqueeze_274 = None
    add_376: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_482, unsqueeze_275);  mul_482 = unsqueeze_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_377: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_376, relu_60);  add_376 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_65: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_377);  add_377 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_69: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_65, primals_208, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    add_378: "i64[]" = torch.ops.aten.add.Tensor(primals_467, 1)
    var_mean_69 = torch.ops.aten.var_mean.correction(convolution_69, [0, 2, 3], correction = 0, keepdim = True)
    getitem_400: "f32[1, 1024, 1, 1]" = var_mean_69[0]
    getitem_401: "f32[1, 1024, 1, 1]" = var_mean_69[1];  var_mean_69 = None
    add_379: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_400, 1e-05)
    rsqrt_69: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_379);  add_379 = None
    sub_69: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_69, getitem_401)
    mul_483: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_69, rsqrt_69);  sub_69 = None
    squeeze_207: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_401, [0, 2, 3]);  getitem_401 = None
    squeeze_208: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_69, [0, 2, 3]);  rsqrt_69 = None
    mul_484: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_207, 0.1)
    mul_485: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_465, 0.9)
    add_380: "f32[1024]" = torch.ops.aten.add.Tensor(mul_484, mul_485);  mul_484 = mul_485 = None
    squeeze_209: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_400, [0, 2, 3]);  getitem_400 = None
    mul_486: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_209, 1.0006381620931717);  squeeze_209 = None
    mul_487: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_486, 0.1);  mul_486 = None
    mul_488: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_466, 0.9)
    add_381: "f32[1024]" = torch.ops.aten.add.Tensor(mul_487, mul_488);  mul_487 = mul_488 = None
    unsqueeze_276: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_209, -1)
    unsqueeze_277: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_276, -1);  unsqueeze_276 = None
    mul_489: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_483, unsqueeze_277);  mul_483 = unsqueeze_277 = None
    unsqueeze_278: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_210, -1);  primals_210 = None
    unsqueeze_279: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_278, -1);  unsqueeze_278 = None
    add_382: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_489, unsqueeze_279);  mul_489 = unsqueeze_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_66: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_382);  add_382 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_66 = torch.ops.aten.split_with_sizes.default(relu_66, [256, 256, 256, 256], 1)
    getitem_406: "f32[8, 256, 14, 14]" = split_with_sizes_66[0];  split_with_sizes_66 = None
    convolution_70: "f32[8, 256, 7, 7]" = torch.ops.aten.convolution.default(getitem_406, primals_211, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_383: "i64[]" = torch.ops.aten.add.Tensor(primals_470, 1)
    var_mean_70 = torch.ops.aten.var_mean.correction(convolution_70, [0, 2, 3], correction = 0, keepdim = True)
    getitem_410: "f32[1, 256, 1, 1]" = var_mean_70[0]
    getitem_411: "f32[1, 256, 1, 1]" = var_mean_70[1];  var_mean_70 = None
    add_384: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_410, 1e-05)
    rsqrt_70: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_384);  add_384 = None
    sub_70: "f32[8, 256, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_70, getitem_411)
    mul_490: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(sub_70, rsqrt_70);  sub_70 = None
    squeeze_210: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_411, [0, 2, 3]);  getitem_411 = None
    squeeze_211: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_70, [0, 2, 3]);  rsqrt_70 = None
    mul_491: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_210, 0.1)
    mul_492: "f32[256]" = torch.ops.aten.mul.Tensor(primals_468, 0.9)
    add_385: "f32[256]" = torch.ops.aten.add.Tensor(mul_491, mul_492);  mul_491 = mul_492 = None
    squeeze_212: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_410, [0, 2, 3]);  getitem_410 = None
    mul_493: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_212, 1.0025575447570332);  squeeze_212 = None
    mul_494: "f32[256]" = torch.ops.aten.mul.Tensor(mul_493, 0.1);  mul_493 = None
    mul_495: "f32[256]" = torch.ops.aten.mul.Tensor(primals_469, 0.9)
    add_386: "f32[256]" = torch.ops.aten.add.Tensor(mul_494, mul_495);  mul_494 = mul_495 = None
    unsqueeze_280: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_212, -1)
    unsqueeze_281: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_280, -1);  unsqueeze_280 = None
    mul_496: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(mul_490, unsqueeze_281);  mul_490 = unsqueeze_281 = None
    unsqueeze_282: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_213, -1);  primals_213 = None
    unsqueeze_283: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_282, -1);  unsqueeze_282 = None
    add_387: "f32[8, 256, 7, 7]" = torch.ops.aten.add.Tensor(mul_496, unsqueeze_283);  mul_496 = unsqueeze_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_67: "f32[8, 256, 7, 7]" = torch.ops.aten.relu.default(add_387);  add_387 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_67 = torch.ops.aten.split_with_sizes.default(relu_66, [256, 256, 256, 256], 1)
    getitem_413: "f32[8, 256, 14, 14]" = split_with_sizes_67[1];  split_with_sizes_67 = None
    convolution_71: "f32[8, 256, 7, 7]" = torch.ops.aten.convolution.default(getitem_413, primals_214, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_388: "i64[]" = torch.ops.aten.add.Tensor(primals_473, 1)
    var_mean_71 = torch.ops.aten.var_mean.correction(convolution_71, [0, 2, 3], correction = 0, keepdim = True)
    getitem_416: "f32[1, 256, 1, 1]" = var_mean_71[0]
    getitem_417: "f32[1, 256, 1, 1]" = var_mean_71[1];  var_mean_71 = None
    add_389: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_416, 1e-05)
    rsqrt_71: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_389);  add_389 = None
    sub_71: "f32[8, 256, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_71, getitem_417)
    mul_497: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(sub_71, rsqrt_71);  sub_71 = None
    squeeze_213: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_417, [0, 2, 3]);  getitem_417 = None
    squeeze_214: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_71, [0, 2, 3]);  rsqrt_71 = None
    mul_498: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_213, 0.1)
    mul_499: "f32[256]" = torch.ops.aten.mul.Tensor(primals_471, 0.9)
    add_390: "f32[256]" = torch.ops.aten.add.Tensor(mul_498, mul_499);  mul_498 = mul_499 = None
    squeeze_215: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_416, [0, 2, 3]);  getitem_416 = None
    mul_500: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_215, 1.0025575447570332);  squeeze_215 = None
    mul_501: "f32[256]" = torch.ops.aten.mul.Tensor(mul_500, 0.1);  mul_500 = None
    mul_502: "f32[256]" = torch.ops.aten.mul.Tensor(primals_472, 0.9)
    add_391: "f32[256]" = torch.ops.aten.add.Tensor(mul_501, mul_502);  mul_501 = mul_502 = None
    unsqueeze_284: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_215, -1)
    unsqueeze_285: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_284, -1);  unsqueeze_284 = None
    mul_503: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(mul_497, unsqueeze_285);  mul_497 = unsqueeze_285 = None
    unsqueeze_286: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_216, -1);  primals_216 = None
    unsqueeze_287: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_286, -1);  unsqueeze_286 = None
    add_392: "f32[8, 256, 7, 7]" = torch.ops.aten.add.Tensor(mul_503, unsqueeze_287);  mul_503 = unsqueeze_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_68: "f32[8, 256, 7, 7]" = torch.ops.aten.relu.default(add_392);  add_392 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_68 = torch.ops.aten.split_with_sizes.default(relu_66, [256, 256, 256, 256], 1)
    getitem_420: "f32[8, 256, 14, 14]" = split_with_sizes_68[2];  split_with_sizes_68 = None
    convolution_72: "f32[8, 256, 7, 7]" = torch.ops.aten.convolution.default(getitem_420, primals_217, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_393: "i64[]" = torch.ops.aten.add.Tensor(primals_476, 1)
    var_mean_72 = torch.ops.aten.var_mean.correction(convolution_72, [0, 2, 3], correction = 0, keepdim = True)
    getitem_422: "f32[1, 256, 1, 1]" = var_mean_72[0]
    getitem_423: "f32[1, 256, 1, 1]" = var_mean_72[1];  var_mean_72 = None
    add_394: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_422, 1e-05)
    rsqrt_72: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_394);  add_394 = None
    sub_72: "f32[8, 256, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_72, getitem_423)
    mul_504: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(sub_72, rsqrt_72);  sub_72 = None
    squeeze_216: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_423, [0, 2, 3]);  getitem_423 = None
    squeeze_217: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_72, [0, 2, 3]);  rsqrt_72 = None
    mul_505: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_216, 0.1)
    mul_506: "f32[256]" = torch.ops.aten.mul.Tensor(primals_474, 0.9)
    add_395: "f32[256]" = torch.ops.aten.add.Tensor(mul_505, mul_506);  mul_505 = mul_506 = None
    squeeze_218: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_422, [0, 2, 3]);  getitem_422 = None
    mul_507: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_218, 1.0025575447570332);  squeeze_218 = None
    mul_508: "f32[256]" = torch.ops.aten.mul.Tensor(mul_507, 0.1);  mul_507 = None
    mul_509: "f32[256]" = torch.ops.aten.mul.Tensor(primals_475, 0.9)
    add_396: "f32[256]" = torch.ops.aten.add.Tensor(mul_508, mul_509);  mul_508 = mul_509 = None
    unsqueeze_288: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_218, -1)
    unsqueeze_289: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_288, -1);  unsqueeze_288 = None
    mul_510: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(mul_504, unsqueeze_289);  mul_504 = unsqueeze_289 = None
    unsqueeze_290: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_219, -1);  primals_219 = None
    unsqueeze_291: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_290, -1);  unsqueeze_290 = None
    add_397: "f32[8, 256, 7, 7]" = torch.ops.aten.add.Tensor(mul_510, unsqueeze_291);  mul_510 = unsqueeze_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_69: "f32[8, 256, 7, 7]" = torch.ops.aten.relu.default(add_397);  add_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:99, code: spo.append(self.pool(spx[-1]))
    split_with_sizes_69 = torch.ops.aten.split_with_sizes.default(relu_66, [256, 256, 256, 256], 1)
    getitem_427: "f32[8, 256, 14, 14]" = split_with_sizes_69[3];  split_with_sizes_69 = None
    avg_pool2d_3: "f32[8, 256, 7, 7]" = torch.ops.aten.avg_pool2d.default(getitem_427, [3, 3], [2, 2], [1, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    cat_13: "f32[8, 1024, 7, 7]" = torch.ops.aten.cat.default([relu_67, relu_68, relu_69, avg_pool2d_3], 1);  avg_pool2d_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_73: "f32[8, 2048, 7, 7]" = torch.ops.aten.convolution.default(cat_13, primals_220, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    add_398: "i64[]" = torch.ops.aten.add.Tensor(primals_479, 1)
    var_mean_73 = torch.ops.aten.var_mean.correction(convolution_73, [0, 2, 3], correction = 0, keepdim = True)
    getitem_428: "f32[1, 2048, 1, 1]" = var_mean_73[0]
    getitem_429: "f32[1, 2048, 1, 1]" = var_mean_73[1];  var_mean_73 = None
    add_399: "f32[1, 2048, 1, 1]" = torch.ops.aten.add.Tensor(getitem_428, 1e-05)
    rsqrt_73: "f32[1, 2048, 1, 1]" = torch.ops.aten.rsqrt.default(add_399);  add_399 = None
    sub_73: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_73, getitem_429)
    mul_511: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_73, rsqrt_73);  sub_73 = None
    squeeze_219: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_429, [0, 2, 3]);  getitem_429 = None
    squeeze_220: "f32[2048]" = torch.ops.aten.squeeze.dims(rsqrt_73, [0, 2, 3]);  rsqrt_73 = None
    mul_512: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_219, 0.1)
    mul_513: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_477, 0.9)
    add_400: "f32[2048]" = torch.ops.aten.add.Tensor(mul_512, mul_513);  mul_512 = mul_513 = None
    squeeze_221: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_428, [0, 2, 3]);  getitem_428 = None
    mul_514: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_221, 1.0025575447570332);  squeeze_221 = None
    mul_515: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_514, 0.1);  mul_514 = None
    mul_516: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_478, 0.9)
    add_401: "f32[2048]" = torch.ops.aten.add.Tensor(mul_515, mul_516);  mul_515 = mul_516 = None
    unsqueeze_292: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_221, -1)
    unsqueeze_293: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_292, -1);  unsqueeze_292 = None
    mul_517: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(mul_511, unsqueeze_293);  mul_511 = unsqueeze_293 = None
    unsqueeze_294: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_222, -1);  primals_222 = None
    unsqueeze_295: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_294, -1);  unsqueeze_294 = None
    add_402: "f32[8, 2048, 7, 7]" = torch.ops.aten.add.Tensor(mul_517, unsqueeze_295);  mul_517 = unsqueeze_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:111, code: shortcut = self.downsample(x)
    convolution_74: "f32[8, 2048, 7, 7]" = torch.ops.aten.convolution.default(relu_65, primals_223, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
    add_403: "i64[]" = torch.ops.aten.add.Tensor(primals_482, 1)
    var_mean_74 = torch.ops.aten.var_mean.correction(convolution_74, [0, 2, 3], correction = 0, keepdim = True)
    getitem_430: "f32[1, 2048, 1, 1]" = var_mean_74[0]
    getitem_431: "f32[1, 2048, 1, 1]" = var_mean_74[1];  var_mean_74 = None
    add_404: "f32[1, 2048, 1, 1]" = torch.ops.aten.add.Tensor(getitem_430, 1e-05)
    rsqrt_74: "f32[1, 2048, 1, 1]" = torch.ops.aten.rsqrt.default(add_404);  add_404 = None
    sub_74: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_74, getitem_431)
    mul_518: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_74, rsqrt_74);  sub_74 = None
    squeeze_222: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_431, [0, 2, 3]);  getitem_431 = None
    squeeze_223: "f32[2048]" = torch.ops.aten.squeeze.dims(rsqrt_74, [0, 2, 3]);  rsqrt_74 = None
    mul_519: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_222, 0.1)
    mul_520: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_480, 0.9)
    add_405: "f32[2048]" = torch.ops.aten.add.Tensor(mul_519, mul_520);  mul_519 = mul_520 = None
    squeeze_224: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_430, [0, 2, 3]);  getitem_430 = None
    mul_521: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_224, 1.0025575447570332);  squeeze_224 = None
    mul_522: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_521, 0.1);  mul_521 = None
    mul_523: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_481, 0.9)
    add_406: "f32[2048]" = torch.ops.aten.add.Tensor(mul_522, mul_523);  mul_522 = mul_523 = None
    unsqueeze_296: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_224, -1)
    unsqueeze_297: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_296, -1);  unsqueeze_296 = None
    mul_524: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(mul_518, unsqueeze_297);  mul_518 = unsqueeze_297 = None
    unsqueeze_298: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_225, -1);  primals_225 = None
    unsqueeze_299: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_298, -1);  unsqueeze_298 = None
    add_407: "f32[8, 2048, 7, 7]" = torch.ops.aten.add.Tensor(mul_524, unsqueeze_299);  mul_524 = unsqueeze_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_408: "f32[8, 2048, 7, 7]" = torch.ops.aten.add.Tensor(add_402, add_407);  add_402 = add_407 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_70: "f32[8, 2048, 7, 7]" = torch.ops.aten.relu.default(add_408);  add_408 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_75: "f32[8, 1024, 7, 7]" = torch.ops.aten.convolution.default(relu_70, primals_226, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    add_409: "i64[]" = torch.ops.aten.add.Tensor(primals_485, 1)
    var_mean_75 = torch.ops.aten.var_mean.correction(convolution_75, [0, 2, 3], correction = 0, keepdim = True)
    getitem_432: "f32[1, 1024, 1, 1]" = var_mean_75[0]
    getitem_433: "f32[1, 1024, 1, 1]" = var_mean_75[1];  var_mean_75 = None
    add_410: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_432, 1e-05)
    rsqrt_75: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_410);  add_410 = None
    sub_75: "f32[8, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_75, getitem_433)
    mul_525: "f32[8, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(sub_75, rsqrt_75);  sub_75 = None
    squeeze_225: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_433, [0, 2, 3]);  getitem_433 = None
    squeeze_226: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_75, [0, 2, 3]);  rsqrt_75 = None
    mul_526: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_225, 0.1)
    mul_527: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_483, 0.9)
    add_411: "f32[1024]" = torch.ops.aten.add.Tensor(mul_526, mul_527);  mul_526 = mul_527 = None
    squeeze_227: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_432, [0, 2, 3]);  getitem_432 = None
    mul_528: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_227, 1.0025575447570332);  squeeze_227 = None
    mul_529: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_528, 0.1);  mul_528 = None
    mul_530: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_484, 0.9)
    add_412: "f32[1024]" = torch.ops.aten.add.Tensor(mul_529, mul_530);  mul_529 = mul_530 = None
    unsqueeze_300: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_227, -1)
    unsqueeze_301: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_300, -1);  unsqueeze_300 = None
    mul_531: "f32[8, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(mul_525, unsqueeze_301);  mul_525 = unsqueeze_301 = None
    unsqueeze_302: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_228, -1);  primals_228 = None
    unsqueeze_303: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_302, -1);  unsqueeze_302 = None
    add_413: "f32[8, 1024, 7, 7]" = torch.ops.aten.add.Tensor(mul_531, unsqueeze_303);  mul_531 = unsqueeze_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_71: "f32[8, 1024, 7, 7]" = torch.ops.aten.relu.default(add_413);  add_413 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_71 = torch.ops.aten.split_with_sizes.default(relu_71, [256, 256, 256, 256], 1)
    getitem_438: "f32[8, 256, 7, 7]" = split_with_sizes_71[0];  split_with_sizes_71 = None
    convolution_76: "f32[8, 256, 7, 7]" = torch.ops.aten.convolution.default(getitem_438, primals_229, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_414: "i64[]" = torch.ops.aten.add.Tensor(primals_488, 1)
    var_mean_76 = torch.ops.aten.var_mean.correction(convolution_76, [0, 2, 3], correction = 0, keepdim = True)
    getitem_442: "f32[1, 256, 1, 1]" = var_mean_76[0]
    getitem_443: "f32[1, 256, 1, 1]" = var_mean_76[1];  var_mean_76 = None
    add_415: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_442, 1e-05)
    rsqrt_76: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_415);  add_415 = None
    sub_76: "f32[8, 256, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_76, getitem_443)
    mul_532: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(sub_76, rsqrt_76);  sub_76 = None
    squeeze_228: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_443, [0, 2, 3]);  getitem_443 = None
    squeeze_229: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_76, [0, 2, 3]);  rsqrt_76 = None
    mul_533: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_228, 0.1)
    mul_534: "f32[256]" = torch.ops.aten.mul.Tensor(primals_486, 0.9)
    add_416: "f32[256]" = torch.ops.aten.add.Tensor(mul_533, mul_534);  mul_533 = mul_534 = None
    squeeze_230: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_442, [0, 2, 3]);  getitem_442 = None
    mul_535: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_230, 1.0025575447570332);  squeeze_230 = None
    mul_536: "f32[256]" = torch.ops.aten.mul.Tensor(mul_535, 0.1);  mul_535 = None
    mul_537: "f32[256]" = torch.ops.aten.mul.Tensor(primals_487, 0.9)
    add_417: "f32[256]" = torch.ops.aten.add.Tensor(mul_536, mul_537);  mul_536 = mul_537 = None
    unsqueeze_304: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_230, -1)
    unsqueeze_305: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_304, -1);  unsqueeze_304 = None
    mul_538: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(mul_532, unsqueeze_305);  mul_532 = unsqueeze_305 = None
    unsqueeze_306: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_231, -1);  primals_231 = None
    unsqueeze_307: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_306, -1);  unsqueeze_306 = None
    add_418: "f32[8, 256, 7, 7]" = torch.ops.aten.add.Tensor(mul_538, unsqueeze_307);  mul_538 = unsqueeze_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_72: "f32[8, 256, 7, 7]" = torch.ops.aten.relu.default(add_418);  add_418 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_72 = torch.ops.aten.split_with_sizes.default(relu_71, [256, 256, 256, 256], 1)
    getitem_445: "f32[8, 256, 7, 7]" = split_with_sizes_72[1];  split_with_sizes_72 = None
    add_419: "f32[8, 256, 7, 7]" = torch.ops.aten.add.Tensor(relu_72, getitem_445);  getitem_445 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_77: "f32[8, 256, 7, 7]" = torch.ops.aten.convolution.default(add_419, primals_232, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_420: "i64[]" = torch.ops.aten.add.Tensor(primals_491, 1)
    var_mean_77 = torch.ops.aten.var_mean.correction(convolution_77, [0, 2, 3], correction = 0, keepdim = True)
    getitem_448: "f32[1, 256, 1, 1]" = var_mean_77[0]
    getitem_449: "f32[1, 256, 1, 1]" = var_mean_77[1];  var_mean_77 = None
    add_421: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_448, 1e-05)
    rsqrt_77: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_421);  add_421 = None
    sub_77: "f32[8, 256, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_77, getitem_449)
    mul_539: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(sub_77, rsqrt_77);  sub_77 = None
    squeeze_231: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_449, [0, 2, 3]);  getitem_449 = None
    squeeze_232: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_77, [0, 2, 3]);  rsqrt_77 = None
    mul_540: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_231, 0.1)
    mul_541: "f32[256]" = torch.ops.aten.mul.Tensor(primals_489, 0.9)
    add_422: "f32[256]" = torch.ops.aten.add.Tensor(mul_540, mul_541);  mul_540 = mul_541 = None
    squeeze_233: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_448, [0, 2, 3]);  getitem_448 = None
    mul_542: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_233, 1.0025575447570332);  squeeze_233 = None
    mul_543: "f32[256]" = torch.ops.aten.mul.Tensor(mul_542, 0.1);  mul_542 = None
    mul_544: "f32[256]" = torch.ops.aten.mul.Tensor(primals_490, 0.9)
    add_423: "f32[256]" = torch.ops.aten.add.Tensor(mul_543, mul_544);  mul_543 = mul_544 = None
    unsqueeze_308: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_233, -1)
    unsqueeze_309: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_308, -1);  unsqueeze_308 = None
    mul_545: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(mul_539, unsqueeze_309);  mul_539 = unsqueeze_309 = None
    unsqueeze_310: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_234, -1);  primals_234 = None
    unsqueeze_311: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_310, -1);  unsqueeze_310 = None
    add_424: "f32[8, 256, 7, 7]" = torch.ops.aten.add.Tensor(mul_545, unsqueeze_311);  mul_545 = unsqueeze_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_73: "f32[8, 256, 7, 7]" = torch.ops.aten.relu.default(add_424);  add_424 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_73 = torch.ops.aten.split_with_sizes.default(relu_71, [256, 256, 256, 256], 1)
    getitem_452: "f32[8, 256, 7, 7]" = split_with_sizes_73[2];  split_with_sizes_73 = None
    add_425: "f32[8, 256, 7, 7]" = torch.ops.aten.add.Tensor(relu_73, getitem_452);  getitem_452 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_78: "f32[8, 256, 7, 7]" = torch.ops.aten.convolution.default(add_425, primals_235, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_426: "i64[]" = torch.ops.aten.add.Tensor(primals_494, 1)
    var_mean_78 = torch.ops.aten.var_mean.correction(convolution_78, [0, 2, 3], correction = 0, keepdim = True)
    getitem_454: "f32[1, 256, 1, 1]" = var_mean_78[0]
    getitem_455: "f32[1, 256, 1, 1]" = var_mean_78[1];  var_mean_78 = None
    add_427: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_454, 1e-05)
    rsqrt_78: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_427);  add_427 = None
    sub_78: "f32[8, 256, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_78, getitem_455)
    mul_546: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(sub_78, rsqrt_78);  sub_78 = None
    squeeze_234: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_455, [0, 2, 3]);  getitem_455 = None
    squeeze_235: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_78, [0, 2, 3]);  rsqrt_78 = None
    mul_547: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_234, 0.1)
    mul_548: "f32[256]" = torch.ops.aten.mul.Tensor(primals_492, 0.9)
    add_428: "f32[256]" = torch.ops.aten.add.Tensor(mul_547, mul_548);  mul_547 = mul_548 = None
    squeeze_236: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_454, [0, 2, 3]);  getitem_454 = None
    mul_549: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_236, 1.0025575447570332);  squeeze_236 = None
    mul_550: "f32[256]" = torch.ops.aten.mul.Tensor(mul_549, 0.1);  mul_549 = None
    mul_551: "f32[256]" = torch.ops.aten.mul.Tensor(primals_493, 0.9)
    add_429: "f32[256]" = torch.ops.aten.add.Tensor(mul_550, mul_551);  mul_550 = mul_551 = None
    unsqueeze_312: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_236, -1)
    unsqueeze_313: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_312, -1);  unsqueeze_312 = None
    mul_552: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(mul_546, unsqueeze_313);  mul_546 = unsqueeze_313 = None
    unsqueeze_314: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_237, -1);  primals_237 = None
    unsqueeze_315: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_314, -1);  unsqueeze_314 = None
    add_430: "f32[8, 256, 7, 7]" = torch.ops.aten.add.Tensor(mul_552, unsqueeze_315);  mul_552 = unsqueeze_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_74: "f32[8, 256, 7, 7]" = torch.ops.aten.relu.default(add_430);  add_430 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    split_with_sizes_74 = torch.ops.aten.split_with_sizes.default(relu_71, [256, 256, 256, 256], 1)
    getitem_459: "f32[8, 256, 7, 7]" = split_with_sizes_74[3];  split_with_sizes_74 = None
    cat_14: "f32[8, 1024, 7, 7]" = torch.ops.aten.cat.default([relu_72, relu_73, relu_74, getitem_459], 1);  getitem_459 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_79: "f32[8, 2048, 7, 7]" = torch.ops.aten.convolution.default(cat_14, primals_238, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    add_431: "i64[]" = torch.ops.aten.add.Tensor(primals_497, 1)
    var_mean_79 = torch.ops.aten.var_mean.correction(convolution_79, [0, 2, 3], correction = 0, keepdim = True)
    getitem_460: "f32[1, 2048, 1, 1]" = var_mean_79[0]
    getitem_461: "f32[1, 2048, 1, 1]" = var_mean_79[1];  var_mean_79 = None
    add_432: "f32[1, 2048, 1, 1]" = torch.ops.aten.add.Tensor(getitem_460, 1e-05)
    rsqrt_79: "f32[1, 2048, 1, 1]" = torch.ops.aten.rsqrt.default(add_432);  add_432 = None
    sub_79: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_79, getitem_461)
    mul_553: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_79, rsqrt_79);  sub_79 = None
    squeeze_237: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_461, [0, 2, 3]);  getitem_461 = None
    squeeze_238: "f32[2048]" = torch.ops.aten.squeeze.dims(rsqrt_79, [0, 2, 3]);  rsqrt_79 = None
    mul_554: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_237, 0.1)
    mul_555: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_495, 0.9)
    add_433: "f32[2048]" = torch.ops.aten.add.Tensor(mul_554, mul_555);  mul_554 = mul_555 = None
    squeeze_239: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_460, [0, 2, 3]);  getitem_460 = None
    mul_556: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_239, 1.0025575447570332);  squeeze_239 = None
    mul_557: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_556, 0.1);  mul_556 = None
    mul_558: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_496, 0.9)
    add_434: "f32[2048]" = torch.ops.aten.add.Tensor(mul_557, mul_558);  mul_557 = mul_558 = None
    unsqueeze_316: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_239, -1)
    unsqueeze_317: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_316, -1);  unsqueeze_316 = None
    mul_559: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(mul_553, unsqueeze_317);  mul_553 = unsqueeze_317 = None
    unsqueeze_318: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_240, -1);  primals_240 = None
    unsqueeze_319: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_318, -1);  unsqueeze_318 = None
    add_435: "f32[8, 2048, 7, 7]" = torch.ops.aten.add.Tensor(mul_559, unsqueeze_319);  mul_559 = unsqueeze_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_436: "f32[8, 2048, 7, 7]" = torch.ops.aten.add.Tensor(add_435, relu_70);  add_435 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_75: "f32[8, 2048, 7, 7]" = torch.ops.aten.relu.default(add_436);  add_436 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_80: "f32[8, 1024, 7, 7]" = torch.ops.aten.convolution.default(relu_75, primals_241, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    add_437: "i64[]" = torch.ops.aten.add.Tensor(primals_500, 1)
    var_mean_80 = torch.ops.aten.var_mean.correction(convolution_80, [0, 2, 3], correction = 0, keepdim = True)
    getitem_462: "f32[1, 1024, 1, 1]" = var_mean_80[0]
    getitem_463: "f32[1, 1024, 1, 1]" = var_mean_80[1];  var_mean_80 = None
    add_438: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_462, 1e-05)
    rsqrt_80: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_438);  add_438 = None
    sub_80: "f32[8, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_80, getitem_463)
    mul_560: "f32[8, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(sub_80, rsqrt_80);  sub_80 = None
    squeeze_240: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_463, [0, 2, 3]);  getitem_463 = None
    squeeze_241: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_80, [0, 2, 3]);  rsqrt_80 = None
    mul_561: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_240, 0.1)
    mul_562: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_498, 0.9)
    add_439: "f32[1024]" = torch.ops.aten.add.Tensor(mul_561, mul_562);  mul_561 = mul_562 = None
    squeeze_242: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_462, [0, 2, 3]);  getitem_462 = None
    mul_563: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_242, 1.0025575447570332);  squeeze_242 = None
    mul_564: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_563, 0.1);  mul_563 = None
    mul_565: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_499, 0.9)
    add_440: "f32[1024]" = torch.ops.aten.add.Tensor(mul_564, mul_565);  mul_564 = mul_565 = None
    unsqueeze_320: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_242, -1)
    unsqueeze_321: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_320, -1);  unsqueeze_320 = None
    mul_566: "f32[8, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(mul_560, unsqueeze_321);  mul_560 = unsqueeze_321 = None
    unsqueeze_322: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_243, -1);  primals_243 = None
    unsqueeze_323: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_322, -1);  unsqueeze_322 = None
    add_441: "f32[8, 1024, 7, 7]" = torch.ops.aten.add.Tensor(mul_566, unsqueeze_323);  mul_566 = unsqueeze_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_76: "f32[8, 1024, 7, 7]" = torch.ops.aten.relu.default(add_441);  add_441 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_76 = torch.ops.aten.split_with_sizes.default(relu_76, [256, 256, 256, 256], 1)
    getitem_468: "f32[8, 256, 7, 7]" = split_with_sizes_76[0];  split_with_sizes_76 = None
    convolution_81: "f32[8, 256, 7, 7]" = torch.ops.aten.convolution.default(getitem_468, primals_244, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_442: "i64[]" = torch.ops.aten.add.Tensor(primals_503, 1)
    var_mean_81 = torch.ops.aten.var_mean.correction(convolution_81, [0, 2, 3], correction = 0, keepdim = True)
    getitem_472: "f32[1, 256, 1, 1]" = var_mean_81[0]
    getitem_473: "f32[1, 256, 1, 1]" = var_mean_81[1];  var_mean_81 = None
    add_443: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_472, 1e-05)
    rsqrt_81: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_443);  add_443 = None
    sub_81: "f32[8, 256, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_81, getitem_473)
    mul_567: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(sub_81, rsqrt_81);  sub_81 = None
    squeeze_243: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_473, [0, 2, 3]);  getitem_473 = None
    squeeze_244: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_81, [0, 2, 3]);  rsqrt_81 = None
    mul_568: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_243, 0.1)
    mul_569: "f32[256]" = torch.ops.aten.mul.Tensor(primals_501, 0.9)
    add_444: "f32[256]" = torch.ops.aten.add.Tensor(mul_568, mul_569);  mul_568 = mul_569 = None
    squeeze_245: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_472, [0, 2, 3]);  getitem_472 = None
    mul_570: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_245, 1.0025575447570332);  squeeze_245 = None
    mul_571: "f32[256]" = torch.ops.aten.mul.Tensor(mul_570, 0.1);  mul_570 = None
    mul_572: "f32[256]" = torch.ops.aten.mul.Tensor(primals_502, 0.9)
    add_445: "f32[256]" = torch.ops.aten.add.Tensor(mul_571, mul_572);  mul_571 = mul_572 = None
    unsqueeze_324: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_245, -1)
    unsqueeze_325: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_324, -1);  unsqueeze_324 = None
    mul_573: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(mul_567, unsqueeze_325);  mul_567 = unsqueeze_325 = None
    unsqueeze_326: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_246, -1);  primals_246 = None
    unsqueeze_327: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_326, -1);  unsqueeze_326 = None
    add_446: "f32[8, 256, 7, 7]" = torch.ops.aten.add.Tensor(mul_573, unsqueeze_327);  mul_573 = unsqueeze_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_77: "f32[8, 256, 7, 7]" = torch.ops.aten.relu.default(add_446);  add_446 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_77 = torch.ops.aten.split_with_sizes.default(relu_76, [256, 256, 256, 256], 1)
    getitem_475: "f32[8, 256, 7, 7]" = split_with_sizes_77[1];  split_with_sizes_77 = None
    add_447: "f32[8, 256, 7, 7]" = torch.ops.aten.add.Tensor(relu_77, getitem_475);  getitem_475 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_82: "f32[8, 256, 7, 7]" = torch.ops.aten.convolution.default(add_447, primals_247, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_448: "i64[]" = torch.ops.aten.add.Tensor(primals_506, 1)
    var_mean_82 = torch.ops.aten.var_mean.correction(convolution_82, [0, 2, 3], correction = 0, keepdim = True)
    getitem_478: "f32[1, 256, 1, 1]" = var_mean_82[0]
    getitem_479: "f32[1, 256, 1, 1]" = var_mean_82[1];  var_mean_82 = None
    add_449: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_478, 1e-05)
    rsqrt_82: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_449);  add_449 = None
    sub_82: "f32[8, 256, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_82, getitem_479)
    mul_574: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(sub_82, rsqrt_82);  sub_82 = None
    squeeze_246: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_479, [0, 2, 3]);  getitem_479 = None
    squeeze_247: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_82, [0, 2, 3]);  rsqrt_82 = None
    mul_575: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_246, 0.1)
    mul_576: "f32[256]" = torch.ops.aten.mul.Tensor(primals_504, 0.9)
    add_450: "f32[256]" = torch.ops.aten.add.Tensor(mul_575, mul_576);  mul_575 = mul_576 = None
    squeeze_248: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_478, [0, 2, 3]);  getitem_478 = None
    mul_577: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_248, 1.0025575447570332);  squeeze_248 = None
    mul_578: "f32[256]" = torch.ops.aten.mul.Tensor(mul_577, 0.1);  mul_577 = None
    mul_579: "f32[256]" = torch.ops.aten.mul.Tensor(primals_505, 0.9)
    add_451: "f32[256]" = torch.ops.aten.add.Tensor(mul_578, mul_579);  mul_578 = mul_579 = None
    unsqueeze_328: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_248, -1)
    unsqueeze_329: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_328, -1);  unsqueeze_328 = None
    mul_580: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(mul_574, unsqueeze_329);  mul_574 = unsqueeze_329 = None
    unsqueeze_330: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_249, -1);  primals_249 = None
    unsqueeze_331: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_330, -1);  unsqueeze_330 = None
    add_452: "f32[8, 256, 7, 7]" = torch.ops.aten.add.Tensor(mul_580, unsqueeze_331);  mul_580 = unsqueeze_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_78: "f32[8, 256, 7, 7]" = torch.ops.aten.relu.default(add_452);  add_452 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    split_with_sizes_78 = torch.ops.aten.split_with_sizes.default(relu_76, [256, 256, 256, 256], 1)
    getitem_482: "f32[8, 256, 7, 7]" = split_with_sizes_78[2];  split_with_sizes_78 = None
    add_453: "f32[8, 256, 7, 7]" = torch.ops.aten.add.Tensor(relu_78, getitem_482);  getitem_482 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_83: "f32[8, 256, 7, 7]" = torch.ops.aten.convolution.default(add_453, primals_250, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_454: "i64[]" = torch.ops.aten.add.Tensor(primals_509, 1)
    var_mean_83 = torch.ops.aten.var_mean.correction(convolution_83, [0, 2, 3], correction = 0, keepdim = True)
    getitem_484: "f32[1, 256, 1, 1]" = var_mean_83[0]
    getitem_485: "f32[1, 256, 1, 1]" = var_mean_83[1];  var_mean_83 = None
    add_455: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_484, 1e-05)
    rsqrt_83: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_455);  add_455 = None
    sub_83: "f32[8, 256, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_83, getitem_485)
    mul_581: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(sub_83, rsqrt_83);  sub_83 = None
    squeeze_249: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_485, [0, 2, 3]);  getitem_485 = None
    squeeze_250: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_83, [0, 2, 3]);  rsqrt_83 = None
    mul_582: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_249, 0.1)
    mul_583: "f32[256]" = torch.ops.aten.mul.Tensor(primals_507, 0.9)
    add_456: "f32[256]" = torch.ops.aten.add.Tensor(mul_582, mul_583);  mul_582 = mul_583 = None
    squeeze_251: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_484, [0, 2, 3]);  getitem_484 = None
    mul_584: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_251, 1.0025575447570332);  squeeze_251 = None
    mul_585: "f32[256]" = torch.ops.aten.mul.Tensor(mul_584, 0.1);  mul_584 = None
    mul_586: "f32[256]" = torch.ops.aten.mul.Tensor(primals_508, 0.9)
    add_457: "f32[256]" = torch.ops.aten.add.Tensor(mul_585, mul_586);  mul_585 = mul_586 = None
    unsqueeze_332: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_251, -1)
    unsqueeze_333: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_332, -1);  unsqueeze_332 = None
    mul_587: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(mul_581, unsqueeze_333);  mul_581 = unsqueeze_333 = None
    unsqueeze_334: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_252, -1);  primals_252 = None
    unsqueeze_335: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_334, -1);  unsqueeze_334 = None
    add_458: "f32[8, 256, 7, 7]" = torch.ops.aten.add.Tensor(mul_587, unsqueeze_335);  mul_587 = unsqueeze_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_79: "f32[8, 256, 7, 7]" = torch.ops.aten.relu.default(add_458);  add_458 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    split_with_sizes_79 = torch.ops.aten.split_with_sizes.default(relu_76, [256, 256, 256, 256], 1)
    getitem_489: "f32[8, 256, 7, 7]" = split_with_sizes_79[3];  split_with_sizes_79 = None
    cat_15: "f32[8, 1024, 7, 7]" = torch.ops.aten.cat.default([relu_77, relu_78, relu_79, getitem_489], 1);  getitem_489 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_84: "f32[8, 2048, 7, 7]" = torch.ops.aten.convolution.default(cat_15, primals_253, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    add_459: "i64[]" = torch.ops.aten.add.Tensor(primals_512, 1)
    var_mean_84 = torch.ops.aten.var_mean.correction(convolution_84, [0, 2, 3], correction = 0, keepdim = True)
    getitem_490: "f32[1, 2048, 1, 1]" = var_mean_84[0]
    getitem_491: "f32[1, 2048, 1, 1]" = var_mean_84[1];  var_mean_84 = None
    add_460: "f32[1, 2048, 1, 1]" = torch.ops.aten.add.Tensor(getitem_490, 1e-05)
    rsqrt_84: "f32[1, 2048, 1, 1]" = torch.ops.aten.rsqrt.default(add_460);  add_460 = None
    sub_84: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_84, getitem_491)
    mul_588: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_84, rsqrt_84);  sub_84 = None
    squeeze_252: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_491, [0, 2, 3]);  getitem_491 = None
    squeeze_253: "f32[2048]" = torch.ops.aten.squeeze.dims(rsqrt_84, [0, 2, 3]);  rsqrt_84 = None
    mul_589: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_252, 0.1)
    mul_590: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_510, 0.9)
    add_461: "f32[2048]" = torch.ops.aten.add.Tensor(mul_589, mul_590);  mul_589 = mul_590 = None
    squeeze_254: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_490, [0, 2, 3]);  getitem_490 = None
    mul_591: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_254, 1.0025575447570332);  squeeze_254 = None
    mul_592: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_591, 0.1);  mul_591 = None
    mul_593: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_511, 0.9)
    add_462: "f32[2048]" = torch.ops.aten.add.Tensor(mul_592, mul_593);  mul_592 = mul_593 = None
    unsqueeze_336: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_254, -1)
    unsqueeze_337: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_336, -1);  unsqueeze_336 = None
    mul_594: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(mul_588, unsqueeze_337);  mul_588 = unsqueeze_337 = None
    unsqueeze_338: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_255, -1);  primals_255 = None
    unsqueeze_339: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_338, -1);  unsqueeze_338 = None
    add_463: "f32[8, 2048, 7, 7]" = torch.ops.aten.add.Tensor(mul_594, unsqueeze_339);  mul_594 = unsqueeze_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_464: "f32[8, 2048, 7, 7]" = torch.ops.aten.add.Tensor(add_463, relu_75);  add_463 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_80: "f32[8, 2048, 7, 7]" = torch.ops.aten.relu.default(add_464);  add_464 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    mean: "f32[8, 2048, 1, 1]" = torch.ops.aten.mean.dim(relu_80, [-1, -2], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view: "f32[8, 2048]" = torch.ops.aten.view.default(mean, [8, 2048]);  mean = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:538, code: return x if pre_logits else self.fc(x)
    permute: "f32[2048, 1000]" = torch.ops.aten.permute.default(primals_256, [1, 0]);  primals_256 = None
    addmm: "f32[8, 1000]" = torch.ops.aten.addmm.default(primals_257, view, permute);  primals_257 = None
    permute_1: "f32[1000, 2048]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
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
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    alias_82: "f32[8, 2048, 7, 7]" = torch.ops.aten.alias.default(relu_80);  relu_80 = None
    alias_83: "f32[8, 2048, 7, 7]" = torch.ops.aten.alias.default(alias_82);  alias_82 = None
    le: "b8[8, 2048, 7, 7]" = torch.ops.aten.le.Scalar(alias_83, 0);  alias_83 = None
    scalar_tensor: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where: "f32[8, 2048, 7, 7]" = torch.ops.aten.where.self(le, scalar_tensor, div);  le = scalar_tensor = div = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    unsqueeze_340: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_252, 0);  squeeze_252 = None
    unsqueeze_341: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_340, 2);  unsqueeze_340 = None
    unsqueeze_342: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_341, 3);  unsqueeze_341 = None
    sum_2: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where, [0, 2, 3])
    sub_85: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_84, unsqueeze_342)
    mul_595: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(where, sub_85);  sub_85 = None
    sum_3: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_595, [0, 2, 3]);  mul_595 = None
    mul_596: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_2, 0.002551020408163265)
    unsqueeze_343: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_596, 0);  mul_596 = None
    unsqueeze_344: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_343, 2);  unsqueeze_343 = None
    unsqueeze_345: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_344, 3);  unsqueeze_344 = None
    mul_597: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_3, 0.002551020408163265)
    mul_598: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_253, squeeze_253)
    mul_599: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_597, mul_598);  mul_597 = mul_598 = None
    unsqueeze_346: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_599, 0);  mul_599 = None
    unsqueeze_347: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_346, 2);  unsqueeze_346 = None
    unsqueeze_348: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_347, 3);  unsqueeze_347 = None
    mul_600: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_253, primals_254);  primals_254 = None
    unsqueeze_349: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_600, 0);  mul_600 = None
    unsqueeze_350: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_349, 2);  unsqueeze_349 = None
    unsqueeze_351: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_350, 3);  unsqueeze_350 = None
    sub_86: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_84, unsqueeze_342);  convolution_84 = unsqueeze_342 = None
    mul_601: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_86, unsqueeze_348);  sub_86 = unsqueeze_348 = None
    sub_87: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(where, mul_601);  mul_601 = None
    sub_88: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(sub_87, unsqueeze_345);  sub_87 = unsqueeze_345 = None
    mul_602: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_88, unsqueeze_351);  sub_88 = unsqueeze_351 = None
    mul_603: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_3, squeeze_253);  sum_3 = squeeze_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_backward = torch.ops.aten.convolution_backward.default(mul_602, cat_15, primals_253, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_602 = cat_15 = primals_253 = None
    getitem_492: "f32[8, 1024, 7, 7]" = convolution_backward[0]
    getitem_493: "f32[2048, 1024, 1, 1]" = convolution_backward[1];  convolution_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    slice_1: "f32[8, 256, 7, 7]" = torch.ops.aten.slice.Tensor(getitem_492, 1, 0, 256)
    slice_2: "f32[8, 256, 7, 7]" = torch.ops.aten.slice.Tensor(getitem_492, 1, 256, 512)
    slice_3: "f32[8, 256, 7, 7]" = torch.ops.aten.slice.Tensor(getitem_492, 1, 512, 768)
    slice_4: "f32[8, 256, 7, 7]" = torch.ops.aten.slice.Tensor(getitem_492, 1, 768, 1024);  getitem_492 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_85: "f32[8, 256, 7, 7]" = torch.ops.aten.alias.default(relu_79);  relu_79 = None
    alias_86: "f32[8, 256, 7, 7]" = torch.ops.aten.alias.default(alias_85);  alias_85 = None
    le_1: "b8[8, 256, 7, 7]" = torch.ops.aten.le.Scalar(alias_86, 0);  alias_86 = None
    scalar_tensor_1: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_1: "f32[8, 256, 7, 7]" = torch.ops.aten.where.self(le_1, scalar_tensor_1, slice_3);  le_1 = scalar_tensor_1 = slice_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_352: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_249, 0);  squeeze_249 = None
    unsqueeze_353: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_352, 2);  unsqueeze_352 = None
    unsqueeze_354: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_353, 3);  unsqueeze_353 = None
    sum_4: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_1, [0, 2, 3])
    sub_89: "f32[8, 256, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_83, unsqueeze_354)
    mul_604: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(where_1, sub_89);  sub_89 = None
    sum_5: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_604, [0, 2, 3]);  mul_604 = None
    mul_605: "f32[256]" = torch.ops.aten.mul.Tensor(sum_4, 0.002551020408163265)
    unsqueeze_355: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_605, 0);  mul_605 = None
    unsqueeze_356: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_355, 2);  unsqueeze_355 = None
    unsqueeze_357: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_356, 3);  unsqueeze_356 = None
    mul_606: "f32[256]" = torch.ops.aten.mul.Tensor(sum_5, 0.002551020408163265)
    mul_607: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_250, squeeze_250)
    mul_608: "f32[256]" = torch.ops.aten.mul.Tensor(mul_606, mul_607);  mul_606 = mul_607 = None
    unsqueeze_358: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_608, 0);  mul_608 = None
    unsqueeze_359: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_358, 2);  unsqueeze_358 = None
    unsqueeze_360: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_359, 3);  unsqueeze_359 = None
    mul_609: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_250, primals_251);  primals_251 = None
    unsqueeze_361: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_609, 0);  mul_609 = None
    unsqueeze_362: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_361, 2);  unsqueeze_361 = None
    unsqueeze_363: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_362, 3);  unsqueeze_362 = None
    sub_90: "f32[8, 256, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_83, unsqueeze_354);  convolution_83 = unsqueeze_354 = None
    mul_610: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(sub_90, unsqueeze_360);  sub_90 = unsqueeze_360 = None
    sub_91: "f32[8, 256, 7, 7]" = torch.ops.aten.sub.Tensor(where_1, mul_610);  where_1 = mul_610 = None
    sub_92: "f32[8, 256, 7, 7]" = torch.ops.aten.sub.Tensor(sub_91, unsqueeze_357);  sub_91 = unsqueeze_357 = None
    mul_611: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(sub_92, unsqueeze_363);  sub_92 = unsqueeze_363 = None
    mul_612: "f32[256]" = torch.ops.aten.mul.Tensor(sum_5, squeeze_250);  sum_5 = squeeze_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(mul_611, add_453, primals_250, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_611 = add_453 = primals_250 = None
    getitem_495: "f32[8, 256, 7, 7]" = convolution_backward_1[0]
    getitem_496: "f32[256, 32, 3, 3]" = convolution_backward_1[1];  convolution_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_465: "f32[8, 256, 7, 7]" = torch.ops.aten.add.Tensor(slice_2, getitem_495);  slice_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_88: "f32[8, 256, 7, 7]" = torch.ops.aten.alias.default(relu_78);  relu_78 = None
    alias_89: "f32[8, 256, 7, 7]" = torch.ops.aten.alias.default(alias_88);  alias_88 = None
    le_2: "b8[8, 256, 7, 7]" = torch.ops.aten.le.Scalar(alias_89, 0);  alias_89 = None
    scalar_tensor_2: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_2: "f32[8, 256, 7, 7]" = torch.ops.aten.where.self(le_2, scalar_tensor_2, add_465);  le_2 = scalar_tensor_2 = add_465 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_364: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_246, 0);  squeeze_246 = None
    unsqueeze_365: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_364, 2);  unsqueeze_364 = None
    unsqueeze_366: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_365, 3);  unsqueeze_365 = None
    sum_6: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_2, [0, 2, 3])
    sub_93: "f32[8, 256, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_82, unsqueeze_366)
    mul_613: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(where_2, sub_93);  sub_93 = None
    sum_7: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_613, [0, 2, 3]);  mul_613 = None
    mul_614: "f32[256]" = torch.ops.aten.mul.Tensor(sum_6, 0.002551020408163265)
    unsqueeze_367: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_614, 0);  mul_614 = None
    unsqueeze_368: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_367, 2);  unsqueeze_367 = None
    unsqueeze_369: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_368, 3);  unsqueeze_368 = None
    mul_615: "f32[256]" = torch.ops.aten.mul.Tensor(sum_7, 0.002551020408163265)
    mul_616: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_247, squeeze_247)
    mul_617: "f32[256]" = torch.ops.aten.mul.Tensor(mul_615, mul_616);  mul_615 = mul_616 = None
    unsqueeze_370: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_617, 0);  mul_617 = None
    unsqueeze_371: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_370, 2);  unsqueeze_370 = None
    unsqueeze_372: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_371, 3);  unsqueeze_371 = None
    mul_618: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_247, primals_248);  primals_248 = None
    unsqueeze_373: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_618, 0);  mul_618 = None
    unsqueeze_374: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_373, 2);  unsqueeze_373 = None
    unsqueeze_375: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_374, 3);  unsqueeze_374 = None
    sub_94: "f32[8, 256, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_82, unsqueeze_366);  convolution_82 = unsqueeze_366 = None
    mul_619: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(sub_94, unsqueeze_372);  sub_94 = unsqueeze_372 = None
    sub_95: "f32[8, 256, 7, 7]" = torch.ops.aten.sub.Tensor(where_2, mul_619);  where_2 = mul_619 = None
    sub_96: "f32[8, 256, 7, 7]" = torch.ops.aten.sub.Tensor(sub_95, unsqueeze_369);  sub_95 = unsqueeze_369 = None
    mul_620: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(sub_96, unsqueeze_375);  sub_96 = unsqueeze_375 = None
    mul_621: "f32[256]" = torch.ops.aten.mul.Tensor(sum_7, squeeze_247);  sum_7 = squeeze_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(mul_620, add_447, primals_247, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_620 = add_447 = primals_247 = None
    getitem_498: "f32[8, 256, 7, 7]" = convolution_backward_2[0]
    getitem_499: "f32[256, 32, 3, 3]" = convolution_backward_2[1];  convolution_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_466: "f32[8, 256, 7, 7]" = torch.ops.aten.add.Tensor(slice_1, getitem_498);  slice_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_91: "f32[8, 256, 7, 7]" = torch.ops.aten.alias.default(relu_77);  relu_77 = None
    alias_92: "f32[8, 256, 7, 7]" = torch.ops.aten.alias.default(alias_91);  alias_91 = None
    le_3: "b8[8, 256, 7, 7]" = torch.ops.aten.le.Scalar(alias_92, 0);  alias_92 = None
    scalar_tensor_3: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_3: "f32[8, 256, 7, 7]" = torch.ops.aten.where.self(le_3, scalar_tensor_3, add_466);  le_3 = scalar_tensor_3 = add_466 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_376: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_243, 0);  squeeze_243 = None
    unsqueeze_377: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_376, 2);  unsqueeze_376 = None
    unsqueeze_378: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_377, 3);  unsqueeze_377 = None
    sum_8: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_3, [0, 2, 3])
    sub_97: "f32[8, 256, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_81, unsqueeze_378)
    mul_622: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(where_3, sub_97);  sub_97 = None
    sum_9: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_622, [0, 2, 3]);  mul_622 = None
    mul_623: "f32[256]" = torch.ops.aten.mul.Tensor(sum_8, 0.002551020408163265)
    unsqueeze_379: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_623, 0);  mul_623 = None
    unsqueeze_380: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_379, 2);  unsqueeze_379 = None
    unsqueeze_381: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_380, 3);  unsqueeze_380 = None
    mul_624: "f32[256]" = torch.ops.aten.mul.Tensor(sum_9, 0.002551020408163265)
    mul_625: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_244, squeeze_244)
    mul_626: "f32[256]" = torch.ops.aten.mul.Tensor(mul_624, mul_625);  mul_624 = mul_625 = None
    unsqueeze_382: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_626, 0);  mul_626 = None
    unsqueeze_383: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_382, 2);  unsqueeze_382 = None
    unsqueeze_384: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_383, 3);  unsqueeze_383 = None
    mul_627: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_244, primals_245);  primals_245 = None
    unsqueeze_385: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_627, 0);  mul_627 = None
    unsqueeze_386: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_385, 2);  unsqueeze_385 = None
    unsqueeze_387: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_386, 3);  unsqueeze_386 = None
    sub_98: "f32[8, 256, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_81, unsqueeze_378);  convolution_81 = unsqueeze_378 = None
    mul_628: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(sub_98, unsqueeze_384);  sub_98 = unsqueeze_384 = None
    sub_99: "f32[8, 256, 7, 7]" = torch.ops.aten.sub.Tensor(where_3, mul_628);  where_3 = mul_628 = None
    sub_100: "f32[8, 256, 7, 7]" = torch.ops.aten.sub.Tensor(sub_99, unsqueeze_381);  sub_99 = unsqueeze_381 = None
    mul_629: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(sub_100, unsqueeze_387);  sub_100 = unsqueeze_387 = None
    mul_630: "f32[256]" = torch.ops.aten.mul.Tensor(sum_9, squeeze_244);  sum_9 = squeeze_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_3 = torch.ops.aten.convolution_backward.default(mul_629, getitem_468, primals_244, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_629 = getitem_468 = primals_244 = None
    getitem_501: "f32[8, 256, 7, 7]" = convolution_backward_3[0]
    getitem_502: "f32[256, 32, 3, 3]" = convolution_backward_3[1];  convolution_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    cat_16: "f32[8, 1024, 7, 7]" = torch.ops.aten.cat.default([getitem_501, getitem_498, getitem_495, slice_4], 1);  getitem_501 = getitem_498 = getitem_495 = slice_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    alias_94: "f32[8, 1024, 7, 7]" = torch.ops.aten.alias.default(relu_76);  relu_76 = None
    alias_95: "f32[8, 1024, 7, 7]" = torch.ops.aten.alias.default(alias_94);  alias_94 = None
    le_4: "b8[8, 1024, 7, 7]" = torch.ops.aten.le.Scalar(alias_95, 0);  alias_95 = None
    scalar_tensor_4: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_4: "f32[8, 1024, 7, 7]" = torch.ops.aten.where.self(le_4, scalar_tensor_4, cat_16);  le_4 = scalar_tensor_4 = cat_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    unsqueeze_388: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_240, 0);  squeeze_240 = None
    unsqueeze_389: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_388, 2);  unsqueeze_388 = None
    unsqueeze_390: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_389, 3);  unsqueeze_389 = None
    sum_10: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_4, [0, 2, 3])
    sub_101: "f32[8, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_80, unsqueeze_390)
    mul_631: "f32[8, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(where_4, sub_101);  sub_101 = None
    sum_11: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_631, [0, 2, 3]);  mul_631 = None
    mul_632: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_10, 0.002551020408163265)
    unsqueeze_391: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_632, 0);  mul_632 = None
    unsqueeze_392: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_391, 2);  unsqueeze_391 = None
    unsqueeze_393: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_392, 3);  unsqueeze_392 = None
    mul_633: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_11, 0.002551020408163265)
    mul_634: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_241, squeeze_241)
    mul_635: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_633, mul_634);  mul_633 = mul_634 = None
    unsqueeze_394: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_635, 0);  mul_635 = None
    unsqueeze_395: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_394, 2);  unsqueeze_394 = None
    unsqueeze_396: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_395, 3);  unsqueeze_395 = None
    mul_636: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_241, primals_242);  primals_242 = None
    unsqueeze_397: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_636, 0);  mul_636 = None
    unsqueeze_398: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_397, 2);  unsqueeze_397 = None
    unsqueeze_399: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_398, 3);  unsqueeze_398 = None
    sub_102: "f32[8, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_80, unsqueeze_390);  convolution_80 = unsqueeze_390 = None
    mul_637: "f32[8, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(sub_102, unsqueeze_396);  sub_102 = unsqueeze_396 = None
    sub_103: "f32[8, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(where_4, mul_637);  where_4 = mul_637 = None
    sub_104: "f32[8, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(sub_103, unsqueeze_393);  sub_103 = unsqueeze_393 = None
    mul_638: "f32[8, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(sub_104, unsqueeze_399);  sub_104 = unsqueeze_399 = None
    mul_639: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_11, squeeze_241);  sum_11 = squeeze_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_backward_4 = torch.ops.aten.convolution_backward.default(mul_638, relu_75, primals_241, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_638 = primals_241 = None
    getitem_504: "f32[8, 2048, 7, 7]" = convolution_backward_4[0]
    getitem_505: "f32[1024, 2048, 1, 1]" = convolution_backward_4[1];  convolution_backward_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    add_467: "f32[8, 2048, 7, 7]" = torch.ops.aten.add.Tensor(where, getitem_504);  where = getitem_504 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    alias_97: "f32[8, 2048, 7, 7]" = torch.ops.aten.alias.default(relu_75);  relu_75 = None
    alias_98: "f32[8, 2048, 7, 7]" = torch.ops.aten.alias.default(alias_97);  alias_97 = None
    le_5: "b8[8, 2048, 7, 7]" = torch.ops.aten.le.Scalar(alias_98, 0);  alias_98 = None
    scalar_tensor_5: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_5: "f32[8, 2048, 7, 7]" = torch.ops.aten.where.self(le_5, scalar_tensor_5, add_467);  le_5 = scalar_tensor_5 = add_467 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    unsqueeze_400: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_237, 0);  squeeze_237 = None
    unsqueeze_401: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_400, 2);  unsqueeze_400 = None
    unsqueeze_402: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_401, 3);  unsqueeze_401 = None
    sum_12: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_5, [0, 2, 3])
    sub_105: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_79, unsqueeze_402)
    mul_640: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(where_5, sub_105);  sub_105 = None
    sum_13: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_640, [0, 2, 3]);  mul_640 = None
    mul_641: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_12, 0.002551020408163265)
    unsqueeze_403: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_641, 0);  mul_641 = None
    unsqueeze_404: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_403, 2);  unsqueeze_403 = None
    unsqueeze_405: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_404, 3);  unsqueeze_404 = None
    mul_642: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_13, 0.002551020408163265)
    mul_643: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_238, squeeze_238)
    mul_644: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_642, mul_643);  mul_642 = mul_643 = None
    unsqueeze_406: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_644, 0);  mul_644 = None
    unsqueeze_407: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_406, 2);  unsqueeze_406 = None
    unsqueeze_408: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_407, 3);  unsqueeze_407 = None
    mul_645: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_238, primals_239);  primals_239 = None
    unsqueeze_409: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_645, 0);  mul_645 = None
    unsqueeze_410: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_409, 2);  unsqueeze_409 = None
    unsqueeze_411: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_410, 3);  unsqueeze_410 = None
    sub_106: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_79, unsqueeze_402);  convolution_79 = unsqueeze_402 = None
    mul_646: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_106, unsqueeze_408);  sub_106 = unsqueeze_408 = None
    sub_107: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(where_5, mul_646);  mul_646 = None
    sub_108: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(sub_107, unsqueeze_405);  sub_107 = unsqueeze_405 = None
    mul_647: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_108, unsqueeze_411);  sub_108 = unsqueeze_411 = None
    mul_648: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_13, squeeze_238);  sum_13 = squeeze_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_backward_5 = torch.ops.aten.convolution_backward.default(mul_647, cat_14, primals_238, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_647 = cat_14 = primals_238 = None
    getitem_507: "f32[8, 1024, 7, 7]" = convolution_backward_5[0]
    getitem_508: "f32[2048, 1024, 1, 1]" = convolution_backward_5[1];  convolution_backward_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    slice_5: "f32[8, 256, 7, 7]" = torch.ops.aten.slice.Tensor(getitem_507, 1, 0, 256)
    slice_6: "f32[8, 256, 7, 7]" = torch.ops.aten.slice.Tensor(getitem_507, 1, 256, 512)
    slice_7: "f32[8, 256, 7, 7]" = torch.ops.aten.slice.Tensor(getitem_507, 1, 512, 768)
    slice_8: "f32[8, 256, 7, 7]" = torch.ops.aten.slice.Tensor(getitem_507, 1, 768, 1024);  getitem_507 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_100: "f32[8, 256, 7, 7]" = torch.ops.aten.alias.default(relu_74);  relu_74 = None
    alias_101: "f32[8, 256, 7, 7]" = torch.ops.aten.alias.default(alias_100);  alias_100 = None
    le_6: "b8[8, 256, 7, 7]" = torch.ops.aten.le.Scalar(alias_101, 0);  alias_101 = None
    scalar_tensor_6: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_6: "f32[8, 256, 7, 7]" = torch.ops.aten.where.self(le_6, scalar_tensor_6, slice_7);  le_6 = scalar_tensor_6 = slice_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_412: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_234, 0);  squeeze_234 = None
    unsqueeze_413: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_412, 2);  unsqueeze_412 = None
    unsqueeze_414: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_413, 3);  unsqueeze_413 = None
    sum_14: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_6, [0, 2, 3])
    sub_109: "f32[8, 256, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_78, unsqueeze_414)
    mul_649: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(where_6, sub_109);  sub_109 = None
    sum_15: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_649, [0, 2, 3]);  mul_649 = None
    mul_650: "f32[256]" = torch.ops.aten.mul.Tensor(sum_14, 0.002551020408163265)
    unsqueeze_415: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_650, 0);  mul_650 = None
    unsqueeze_416: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_415, 2);  unsqueeze_415 = None
    unsqueeze_417: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_416, 3);  unsqueeze_416 = None
    mul_651: "f32[256]" = torch.ops.aten.mul.Tensor(sum_15, 0.002551020408163265)
    mul_652: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_235, squeeze_235)
    mul_653: "f32[256]" = torch.ops.aten.mul.Tensor(mul_651, mul_652);  mul_651 = mul_652 = None
    unsqueeze_418: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_653, 0);  mul_653 = None
    unsqueeze_419: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_418, 2);  unsqueeze_418 = None
    unsqueeze_420: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_419, 3);  unsqueeze_419 = None
    mul_654: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_235, primals_236);  primals_236 = None
    unsqueeze_421: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_654, 0);  mul_654 = None
    unsqueeze_422: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_421, 2);  unsqueeze_421 = None
    unsqueeze_423: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_422, 3);  unsqueeze_422 = None
    sub_110: "f32[8, 256, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_78, unsqueeze_414);  convolution_78 = unsqueeze_414 = None
    mul_655: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(sub_110, unsqueeze_420);  sub_110 = unsqueeze_420 = None
    sub_111: "f32[8, 256, 7, 7]" = torch.ops.aten.sub.Tensor(where_6, mul_655);  where_6 = mul_655 = None
    sub_112: "f32[8, 256, 7, 7]" = torch.ops.aten.sub.Tensor(sub_111, unsqueeze_417);  sub_111 = unsqueeze_417 = None
    mul_656: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(sub_112, unsqueeze_423);  sub_112 = unsqueeze_423 = None
    mul_657: "f32[256]" = torch.ops.aten.mul.Tensor(sum_15, squeeze_235);  sum_15 = squeeze_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_6 = torch.ops.aten.convolution_backward.default(mul_656, add_425, primals_235, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_656 = add_425 = primals_235 = None
    getitem_510: "f32[8, 256, 7, 7]" = convolution_backward_6[0]
    getitem_511: "f32[256, 32, 3, 3]" = convolution_backward_6[1];  convolution_backward_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_468: "f32[8, 256, 7, 7]" = torch.ops.aten.add.Tensor(slice_6, getitem_510);  slice_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_103: "f32[8, 256, 7, 7]" = torch.ops.aten.alias.default(relu_73);  relu_73 = None
    alias_104: "f32[8, 256, 7, 7]" = torch.ops.aten.alias.default(alias_103);  alias_103 = None
    le_7: "b8[8, 256, 7, 7]" = torch.ops.aten.le.Scalar(alias_104, 0);  alias_104 = None
    scalar_tensor_7: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_7: "f32[8, 256, 7, 7]" = torch.ops.aten.where.self(le_7, scalar_tensor_7, add_468);  le_7 = scalar_tensor_7 = add_468 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_424: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_231, 0);  squeeze_231 = None
    unsqueeze_425: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_424, 2);  unsqueeze_424 = None
    unsqueeze_426: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_425, 3);  unsqueeze_425 = None
    sum_16: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_7, [0, 2, 3])
    sub_113: "f32[8, 256, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_77, unsqueeze_426)
    mul_658: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(where_7, sub_113);  sub_113 = None
    sum_17: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_658, [0, 2, 3]);  mul_658 = None
    mul_659: "f32[256]" = torch.ops.aten.mul.Tensor(sum_16, 0.002551020408163265)
    unsqueeze_427: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_659, 0);  mul_659 = None
    unsqueeze_428: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_427, 2);  unsqueeze_427 = None
    unsqueeze_429: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_428, 3);  unsqueeze_428 = None
    mul_660: "f32[256]" = torch.ops.aten.mul.Tensor(sum_17, 0.002551020408163265)
    mul_661: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_232, squeeze_232)
    mul_662: "f32[256]" = torch.ops.aten.mul.Tensor(mul_660, mul_661);  mul_660 = mul_661 = None
    unsqueeze_430: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_662, 0);  mul_662 = None
    unsqueeze_431: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_430, 2);  unsqueeze_430 = None
    unsqueeze_432: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_431, 3);  unsqueeze_431 = None
    mul_663: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_232, primals_233);  primals_233 = None
    unsqueeze_433: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_663, 0);  mul_663 = None
    unsqueeze_434: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_433, 2);  unsqueeze_433 = None
    unsqueeze_435: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_434, 3);  unsqueeze_434 = None
    sub_114: "f32[8, 256, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_77, unsqueeze_426);  convolution_77 = unsqueeze_426 = None
    mul_664: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(sub_114, unsqueeze_432);  sub_114 = unsqueeze_432 = None
    sub_115: "f32[8, 256, 7, 7]" = torch.ops.aten.sub.Tensor(where_7, mul_664);  where_7 = mul_664 = None
    sub_116: "f32[8, 256, 7, 7]" = torch.ops.aten.sub.Tensor(sub_115, unsqueeze_429);  sub_115 = unsqueeze_429 = None
    mul_665: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(sub_116, unsqueeze_435);  sub_116 = unsqueeze_435 = None
    mul_666: "f32[256]" = torch.ops.aten.mul.Tensor(sum_17, squeeze_232);  sum_17 = squeeze_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_7 = torch.ops.aten.convolution_backward.default(mul_665, add_419, primals_232, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_665 = add_419 = primals_232 = None
    getitem_513: "f32[8, 256, 7, 7]" = convolution_backward_7[0]
    getitem_514: "f32[256, 32, 3, 3]" = convolution_backward_7[1];  convolution_backward_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_469: "f32[8, 256, 7, 7]" = torch.ops.aten.add.Tensor(slice_5, getitem_513);  slice_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_106: "f32[8, 256, 7, 7]" = torch.ops.aten.alias.default(relu_72);  relu_72 = None
    alias_107: "f32[8, 256, 7, 7]" = torch.ops.aten.alias.default(alias_106);  alias_106 = None
    le_8: "b8[8, 256, 7, 7]" = torch.ops.aten.le.Scalar(alias_107, 0);  alias_107 = None
    scalar_tensor_8: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_8: "f32[8, 256, 7, 7]" = torch.ops.aten.where.self(le_8, scalar_tensor_8, add_469);  le_8 = scalar_tensor_8 = add_469 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_436: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_228, 0);  squeeze_228 = None
    unsqueeze_437: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_436, 2);  unsqueeze_436 = None
    unsqueeze_438: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_437, 3);  unsqueeze_437 = None
    sum_18: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_8, [0, 2, 3])
    sub_117: "f32[8, 256, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_76, unsqueeze_438)
    mul_667: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(where_8, sub_117);  sub_117 = None
    sum_19: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_667, [0, 2, 3]);  mul_667 = None
    mul_668: "f32[256]" = torch.ops.aten.mul.Tensor(sum_18, 0.002551020408163265)
    unsqueeze_439: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_668, 0);  mul_668 = None
    unsqueeze_440: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_439, 2);  unsqueeze_439 = None
    unsqueeze_441: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_440, 3);  unsqueeze_440 = None
    mul_669: "f32[256]" = torch.ops.aten.mul.Tensor(sum_19, 0.002551020408163265)
    mul_670: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_229, squeeze_229)
    mul_671: "f32[256]" = torch.ops.aten.mul.Tensor(mul_669, mul_670);  mul_669 = mul_670 = None
    unsqueeze_442: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_671, 0);  mul_671 = None
    unsqueeze_443: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_442, 2);  unsqueeze_442 = None
    unsqueeze_444: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_443, 3);  unsqueeze_443 = None
    mul_672: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_229, primals_230);  primals_230 = None
    unsqueeze_445: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_672, 0);  mul_672 = None
    unsqueeze_446: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_445, 2);  unsqueeze_445 = None
    unsqueeze_447: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_446, 3);  unsqueeze_446 = None
    sub_118: "f32[8, 256, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_76, unsqueeze_438);  convolution_76 = unsqueeze_438 = None
    mul_673: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(sub_118, unsqueeze_444);  sub_118 = unsqueeze_444 = None
    sub_119: "f32[8, 256, 7, 7]" = torch.ops.aten.sub.Tensor(where_8, mul_673);  where_8 = mul_673 = None
    sub_120: "f32[8, 256, 7, 7]" = torch.ops.aten.sub.Tensor(sub_119, unsqueeze_441);  sub_119 = unsqueeze_441 = None
    mul_674: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(sub_120, unsqueeze_447);  sub_120 = unsqueeze_447 = None
    mul_675: "f32[256]" = torch.ops.aten.mul.Tensor(sum_19, squeeze_229);  sum_19 = squeeze_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_8 = torch.ops.aten.convolution_backward.default(mul_674, getitem_438, primals_229, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_674 = getitem_438 = primals_229 = None
    getitem_516: "f32[8, 256, 7, 7]" = convolution_backward_8[0]
    getitem_517: "f32[256, 32, 3, 3]" = convolution_backward_8[1];  convolution_backward_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    cat_17: "f32[8, 1024, 7, 7]" = torch.ops.aten.cat.default([getitem_516, getitem_513, getitem_510, slice_8], 1);  getitem_516 = getitem_513 = getitem_510 = slice_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    alias_109: "f32[8, 1024, 7, 7]" = torch.ops.aten.alias.default(relu_71);  relu_71 = None
    alias_110: "f32[8, 1024, 7, 7]" = torch.ops.aten.alias.default(alias_109);  alias_109 = None
    le_9: "b8[8, 1024, 7, 7]" = torch.ops.aten.le.Scalar(alias_110, 0);  alias_110 = None
    scalar_tensor_9: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_9: "f32[8, 1024, 7, 7]" = torch.ops.aten.where.self(le_9, scalar_tensor_9, cat_17);  le_9 = scalar_tensor_9 = cat_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    unsqueeze_448: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_225, 0);  squeeze_225 = None
    unsqueeze_449: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_448, 2);  unsqueeze_448 = None
    unsqueeze_450: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_449, 3);  unsqueeze_449 = None
    sum_20: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_9, [0, 2, 3])
    sub_121: "f32[8, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_75, unsqueeze_450)
    mul_676: "f32[8, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(where_9, sub_121);  sub_121 = None
    sum_21: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_676, [0, 2, 3]);  mul_676 = None
    mul_677: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_20, 0.002551020408163265)
    unsqueeze_451: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_677, 0);  mul_677 = None
    unsqueeze_452: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_451, 2);  unsqueeze_451 = None
    unsqueeze_453: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_452, 3);  unsqueeze_452 = None
    mul_678: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_21, 0.002551020408163265)
    mul_679: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_226, squeeze_226)
    mul_680: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_678, mul_679);  mul_678 = mul_679 = None
    unsqueeze_454: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_680, 0);  mul_680 = None
    unsqueeze_455: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_454, 2);  unsqueeze_454 = None
    unsqueeze_456: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_455, 3);  unsqueeze_455 = None
    mul_681: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_226, primals_227);  primals_227 = None
    unsqueeze_457: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_681, 0);  mul_681 = None
    unsqueeze_458: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_457, 2);  unsqueeze_457 = None
    unsqueeze_459: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_458, 3);  unsqueeze_458 = None
    sub_122: "f32[8, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_75, unsqueeze_450);  convolution_75 = unsqueeze_450 = None
    mul_682: "f32[8, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(sub_122, unsqueeze_456);  sub_122 = unsqueeze_456 = None
    sub_123: "f32[8, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(where_9, mul_682);  where_9 = mul_682 = None
    sub_124: "f32[8, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(sub_123, unsqueeze_453);  sub_123 = unsqueeze_453 = None
    mul_683: "f32[8, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(sub_124, unsqueeze_459);  sub_124 = unsqueeze_459 = None
    mul_684: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_21, squeeze_226);  sum_21 = squeeze_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_backward_9 = torch.ops.aten.convolution_backward.default(mul_683, relu_70, primals_226, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_683 = primals_226 = None
    getitem_519: "f32[8, 2048, 7, 7]" = convolution_backward_9[0]
    getitem_520: "f32[1024, 2048, 1, 1]" = convolution_backward_9[1];  convolution_backward_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    add_470: "f32[8, 2048, 7, 7]" = torch.ops.aten.add.Tensor(where_5, getitem_519);  where_5 = getitem_519 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    alias_112: "f32[8, 2048, 7, 7]" = torch.ops.aten.alias.default(relu_70);  relu_70 = None
    alias_113: "f32[8, 2048, 7, 7]" = torch.ops.aten.alias.default(alias_112);  alias_112 = None
    le_10: "b8[8, 2048, 7, 7]" = torch.ops.aten.le.Scalar(alias_113, 0);  alias_113 = None
    scalar_tensor_10: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_10: "f32[8, 2048, 7, 7]" = torch.ops.aten.where.self(le_10, scalar_tensor_10, add_470);  le_10 = scalar_tensor_10 = add_470 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:111, code: shortcut = self.downsample(x)
    unsqueeze_460: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_222, 0);  squeeze_222 = None
    unsqueeze_461: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_460, 2);  unsqueeze_460 = None
    unsqueeze_462: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_461, 3);  unsqueeze_461 = None
    sum_22: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_10, [0, 2, 3])
    sub_125: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_74, unsqueeze_462)
    mul_685: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(where_10, sub_125);  sub_125 = None
    sum_23: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_685, [0, 2, 3]);  mul_685 = None
    mul_686: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_22, 0.002551020408163265)
    unsqueeze_463: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_686, 0);  mul_686 = None
    unsqueeze_464: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_463, 2);  unsqueeze_463 = None
    unsqueeze_465: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_464, 3);  unsqueeze_464 = None
    mul_687: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_23, 0.002551020408163265)
    mul_688: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_223, squeeze_223)
    mul_689: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_687, mul_688);  mul_687 = mul_688 = None
    unsqueeze_466: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_689, 0);  mul_689 = None
    unsqueeze_467: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_466, 2);  unsqueeze_466 = None
    unsqueeze_468: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_467, 3);  unsqueeze_467 = None
    mul_690: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_223, primals_224);  primals_224 = None
    unsqueeze_469: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_690, 0);  mul_690 = None
    unsqueeze_470: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_469, 2);  unsqueeze_469 = None
    unsqueeze_471: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_470, 3);  unsqueeze_470 = None
    sub_126: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_74, unsqueeze_462);  convolution_74 = unsqueeze_462 = None
    mul_691: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_126, unsqueeze_468);  sub_126 = unsqueeze_468 = None
    sub_127: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(where_10, mul_691);  mul_691 = None
    sub_128: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(sub_127, unsqueeze_465);  sub_127 = unsqueeze_465 = None
    mul_692: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_128, unsqueeze_471);  sub_128 = unsqueeze_471 = None
    mul_693: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_23, squeeze_223);  sum_23 = squeeze_223 = None
    convolution_backward_10 = torch.ops.aten.convolution_backward.default(mul_692, relu_65, primals_223, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_692 = primals_223 = None
    getitem_522: "f32[8, 1024, 14, 14]" = convolution_backward_10[0]
    getitem_523: "f32[2048, 1024, 1, 1]" = convolution_backward_10[1];  convolution_backward_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    unsqueeze_472: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_219, 0);  squeeze_219 = None
    unsqueeze_473: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_472, 2);  unsqueeze_472 = None
    unsqueeze_474: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_473, 3);  unsqueeze_473 = None
    sum_24: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_10, [0, 2, 3])
    sub_129: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_73, unsqueeze_474)
    mul_694: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(where_10, sub_129);  sub_129 = None
    sum_25: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_694, [0, 2, 3]);  mul_694 = None
    mul_695: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_24, 0.002551020408163265)
    unsqueeze_475: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_695, 0);  mul_695 = None
    unsqueeze_476: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_475, 2);  unsqueeze_475 = None
    unsqueeze_477: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_476, 3);  unsqueeze_476 = None
    mul_696: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_25, 0.002551020408163265)
    mul_697: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_220, squeeze_220)
    mul_698: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_696, mul_697);  mul_696 = mul_697 = None
    unsqueeze_478: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_698, 0);  mul_698 = None
    unsqueeze_479: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_478, 2);  unsqueeze_478 = None
    unsqueeze_480: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_479, 3);  unsqueeze_479 = None
    mul_699: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_220, primals_221);  primals_221 = None
    unsqueeze_481: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_699, 0);  mul_699 = None
    unsqueeze_482: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_481, 2);  unsqueeze_481 = None
    unsqueeze_483: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_482, 3);  unsqueeze_482 = None
    sub_130: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_73, unsqueeze_474);  convolution_73 = unsqueeze_474 = None
    mul_700: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_130, unsqueeze_480);  sub_130 = unsqueeze_480 = None
    sub_131: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(where_10, mul_700);  where_10 = mul_700 = None
    sub_132: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(sub_131, unsqueeze_477);  sub_131 = unsqueeze_477 = None
    mul_701: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_132, unsqueeze_483);  sub_132 = unsqueeze_483 = None
    mul_702: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_25, squeeze_220);  sum_25 = squeeze_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_backward_11 = torch.ops.aten.convolution_backward.default(mul_701, cat_13, primals_220, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_701 = cat_13 = primals_220 = None
    getitem_525: "f32[8, 1024, 7, 7]" = convolution_backward_11[0]
    getitem_526: "f32[2048, 1024, 1, 1]" = convolution_backward_11[1];  convolution_backward_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    slice_9: "f32[8, 256, 7, 7]" = torch.ops.aten.slice.Tensor(getitem_525, 1, 0, 256)
    slice_10: "f32[8, 256, 7, 7]" = torch.ops.aten.slice.Tensor(getitem_525, 1, 256, 512)
    slice_11: "f32[8, 256, 7, 7]" = torch.ops.aten.slice.Tensor(getitem_525, 1, 512, 768)
    slice_12: "f32[8, 256, 7, 7]" = torch.ops.aten.slice.Tensor(getitem_525, 1, 768, 1024);  getitem_525 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:99, code: spo.append(self.pool(spx[-1]))
    avg_pool2d_backward: "f32[8, 256, 14, 14]" = torch.ops.aten.avg_pool2d_backward.default(slice_12, getitem_427, [3, 3], [2, 2], [1, 1], False, True, None);  slice_12 = getitem_427 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_115: "f32[8, 256, 7, 7]" = torch.ops.aten.alias.default(relu_69);  relu_69 = None
    alias_116: "f32[8, 256, 7, 7]" = torch.ops.aten.alias.default(alias_115);  alias_115 = None
    le_11: "b8[8, 256, 7, 7]" = torch.ops.aten.le.Scalar(alias_116, 0);  alias_116 = None
    scalar_tensor_11: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_11: "f32[8, 256, 7, 7]" = torch.ops.aten.where.self(le_11, scalar_tensor_11, slice_11);  le_11 = scalar_tensor_11 = slice_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_484: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_216, 0);  squeeze_216 = None
    unsqueeze_485: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_484, 2);  unsqueeze_484 = None
    unsqueeze_486: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_485, 3);  unsqueeze_485 = None
    sum_26: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_11, [0, 2, 3])
    sub_133: "f32[8, 256, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_72, unsqueeze_486)
    mul_703: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(where_11, sub_133);  sub_133 = None
    sum_27: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_703, [0, 2, 3]);  mul_703 = None
    mul_704: "f32[256]" = torch.ops.aten.mul.Tensor(sum_26, 0.002551020408163265)
    unsqueeze_487: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_704, 0);  mul_704 = None
    unsqueeze_488: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_487, 2);  unsqueeze_487 = None
    unsqueeze_489: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_488, 3);  unsqueeze_488 = None
    mul_705: "f32[256]" = torch.ops.aten.mul.Tensor(sum_27, 0.002551020408163265)
    mul_706: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_217, squeeze_217)
    mul_707: "f32[256]" = torch.ops.aten.mul.Tensor(mul_705, mul_706);  mul_705 = mul_706 = None
    unsqueeze_490: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_707, 0);  mul_707 = None
    unsqueeze_491: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_490, 2);  unsqueeze_490 = None
    unsqueeze_492: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_491, 3);  unsqueeze_491 = None
    mul_708: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_217, primals_218);  primals_218 = None
    unsqueeze_493: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_708, 0);  mul_708 = None
    unsqueeze_494: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_493, 2);  unsqueeze_493 = None
    unsqueeze_495: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_494, 3);  unsqueeze_494 = None
    sub_134: "f32[8, 256, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_72, unsqueeze_486);  convolution_72 = unsqueeze_486 = None
    mul_709: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(sub_134, unsqueeze_492);  sub_134 = unsqueeze_492 = None
    sub_135: "f32[8, 256, 7, 7]" = torch.ops.aten.sub.Tensor(where_11, mul_709);  where_11 = mul_709 = None
    sub_136: "f32[8, 256, 7, 7]" = torch.ops.aten.sub.Tensor(sub_135, unsqueeze_489);  sub_135 = unsqueeze_489 = None
    mul_710: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(sub_136, unsqueeze_495);  sub_136 = unsqueeze_495 = None
    mul_711: "f32[256]" = torch.ops.aten.mul.Tensor(sum_27, squeeze_217);  sum_27 = squeeze_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_12 = torch.ops.aten.convolution_backward.default(mul_710, getitem_420, primals_217, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_710 = getitem_420 = primals_217 = None
    getitem_528: "f32[8, 256, 14, 14]" = convolution_backward_12[0]
    getitem_529: "f32[256, 32, 3, 3]" = convolution_backward_12[1];  convolution_backward_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_118: "f32[8, 256, 7, 7]" = torch.ops.aten.alias.default(relu_68);  relu_68 = None
    alias_119: "f32[8, 256, 7, 7]" = torch.ops.aten.alias.default(alias_118);  alias_118 = None
    le_12: "b8[8, 256, 7, 7]" = torch.ops.aten.le.Scalar(alias_119, 0);  alias_119 = None
    scalar_tensor_12: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_12: "f32[8, 256, 7, 7]" = torch.ops.aten.where.self(le_12, scalar_tensor_12, slice_10);  le_12 = scalar_tensor_12 = slice_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_496: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_213, 0);  squeeze_213 = None
    unsqueeze_497: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_496, 2);  unsqueeze_496 = None
    unsqueeze_498: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_497, 3);  unsqueeze_497 = None
    sum_28: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_12, [0, 2, 3])
    sub_137: "f32[8, 256, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_71, unsqueeze_498)
    mul_712: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(where_12, sub_137);  sub_137 = None
    sum_29: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_712, [0, 2, 3]);  mul_712 = None
    mul_713: "f32[256]" = torch.ops.aten.mul.Tensor(sum_28, 0.002551020408163265)
    unsqueeze_499: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_713, 0);  mul_713 = None
    unsqueeze_500: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_499, 2);  unsqueeze_499 = None
    unsqueeze_501: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_500, 3);  unsqueeze_500 = None
    mul_714: "f32[256]" = torch.ops.aten.mul.Tensor(sum_29, 0.002551020408163265)
    mul_715: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_214, squeeze_214)
    mul_716: "f32[256]" = torch.ops.aten.mul.Tensor(mul_714, mul_715);  mul_714 = mul_715 = None
    unsqueeze_502: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_716, 0);  mul_716 = None
    unsqueeze_503: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_502, 2);  unsqueeze_502 = None
    unsqueeze_504: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_503, 3);  unsqueeze_503 = None
    mul_717: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_214, primals_215);  primals_215 = None
    unsqueeze_505: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_717, 0);  mul_717 = None
    unsqueeze_506: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_505, 2);  unsqueeze_505 = None
    unsqueeze_507: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_506, 3);  unsqueeze_506 = None
    sub_138: "f32[8, 256, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_71, unsqueeze_498);  convolution_71 = unsqueeze_498 = None
    mul_718: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(sub_138, unsqueeze_504);  sub_138 = unsqueeze_504 = None
    sub_139: "f32[8, 256, 7, 7]" = torch.ops.aten.sub.Tensor(where_12, mul_718);  where_12 = mul_718 = None
    sub_140: "f32[8, 256, 7, 7]" = torch.ops.aten.sub.Tensor(sub_139, unsqueeze_501);  sub_139 = unsqueeze_501 = None
    mul_719: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(sub_140, unsqueeze_507);  sub_140 = unsqueeze_507 = None
    mul_720: "f32[256]" = torch.ops.aten.mul.Tensor(sum_29, squeeze_214);  sum_29 = squeeze_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_13 = torch.ops.aten.convolution_backward.default(mul_719, getitem_413, primals_214, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_719 = getitem_413 = primals_214 = None
    getitem_531: "f32[8, 256, 14, 14]" = convolution_backward_13[0]
    getitem_532: "f32[256, 32, 3, 3]" = convolution_backward_13[1];  convolution_backward_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_121: "f32[8, 256, 7, 7]" = torch.ops.aten.alias.default(relu_67);  relu_67 = None
    alias_122: "f32[8, 256, 7, 7]" = torch.ops.aten.alias.default(alias_121);  alias_121 = None
    le_13: "b8[8, 256, 7, 7]" = torch.ops.aten.le.Scalar(alias_122, 0);  alias_122 = None
    scalar_tensor_13: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_13: "f32[8, 256, 7, 7]" = torch.ops.aten.where.self(le_13, scalar_tensor_13, slice_9);  le_13 = scalar_tensor_13 = slice_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_508: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_210, 0);  squeeze_210 = None
    unsqueeze_509: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_508, 2);  unsqueeze_508 = None
    unsqueeze_510: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_509, 3);  unsqueeze_509 = None
    sum_30: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_13, [0, 2, 3])
    sub_141: "f32[8, 256, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_70, unsqueeze_510)
    mul_721: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(where_13, sub_141);  sub_141 = None
    sum_31: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_721, [0, 2, 3]);  mul_721 = None
    mul_722: "f32[256]" = torch.ops.aten.mul.Tensor(sum_30, 0.002551020408163265)
    unsqueeze_511: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_722, 0);  mul_722 = None
    unsqueeze_512: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_511, 2);  unsqueeze_511 = None
    unsqueeze_513: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_512, 3);  unsqueeze_512 = None
    mul_723: "f32[256]" = torch.ops.aten.mul.Tensor(sum_31, 0.002551020408163265)
    mul_724: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_211, squeeze_211)
    mul_725: "f32[256]" = torch.ops.aten.mul.Tensor(mul_723, mul_724);  mul_723 = mul_724 = None
    unsqueeze_514: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_725, 0);  mul_725 = None
    unsqueeze_515: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_514, 2);  unsqueeze_514 = None
    unsqueeze_516: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_515, 3);  unsqueeze_515 = None
    mul_726: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_211, primals_212);  primals_212 = None
    unsqueeze_517: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_726, 0);  mul_726 = None
    unsqueeze_518: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_517, 2);  unsqueeze_517 = None
    unsqueeze_519: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_518, 3);  unsqueeze_518 = None
    sub_142: "f32[8, 256, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_70, unsqueeze_510);  convolution_70 = unsqueeze_510 = None
    mul_727: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(sub_142, unsqueeze_516);  sub_142 = unsqueeze_516 = None
    sub_143: "f32[8, 256, 7, 7]" = torch.ops.aten.sub.Tensor(where_13, mul_727);  where_13 = mul_727 = None
    sub_144: "f32[8, 256, 7, 7]" = torch.ops.aten.sub.Tensor(sub_143, unsqueeze_513);  sub_143 = unsqueeze_513 = None
    mul_728: "f32[8, 256, 7, 7]" = torch.ops.aten.mul.Tensor(sub_144, unsqueeze_519);  sub_144 = unsqueeze_519 = None
    mul_729: "f32[256]" = torch.ops.aten.mul.Tensor(sum_31, squeeze_211);  sum_31 = squeeze_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_14 = torch.ops.aten.convolution_backward.default(mul_728, getitem_406, primals_211, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_728 = getitem_406 = primals_211 = None
    getitem_534: "f32[8, 256, 14, 14]" = convolution_backward_14[0]
    getitem_535: "f32[256, 32, 3, 3]" = convolution_backward_14[1];  convolution_backward_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    cat_18: "f32[8, 1024, 14, 14]" = torch.ops.aten.cat.default([getitem_534, getitem_531, getitem_528, avg_pool2d_backward], 1);  getitem_534 = getitem_531 = getitem_528 = avg_pool2d_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    alias_124: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(relu_66);  relu_66 = None
    alias_125: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(alias_124);  alias_124 = None
    le_14: "b8[8, 1024, 14, 14]" = torch.ops.aten.le.Scalar(alias_125, 0);  alias_125 = None
    scalar_tensor_14: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_14: "f32[8, 1024, 14, 14]" = torch.ops.aten.where.self(le_14, scalar_tensor_14, cat_18);  le_14 = scalar_tensor_14 = cat_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    unsqueeze_520: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_207, 0);  squeeze_207 = None
    unsqueeze_521: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_520, 2);  unsqueeze_520 = None
    unsqueeze_522: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_521, 3);  unsqueeze_521 = None
    sum_32: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_14, [0, 2, 3])
    sub_145: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_69, unsqueeze_522)
    mul_730: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_14, sub_145);  sub_145 = None
    sum_33: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_730, [0, 2, 3]);  mul_730 = None
    mul_731: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_32, 0.0006377551020408163)
    unsqueeze_523: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_731, 0);  mul_731 = None
    unsqueeze_524: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_523, 2);  unsqueeze_523 = None
    unsqueeze_525: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_524, 3);  unsqueeze_524 = None
    mul_732: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_33, 0.0006377551020408163)
    mul_733: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_208, squeeze_208)
    mul_734: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_732, mul_733);  mul_732 = mul_733 = None
    unsqueeze_526: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_734, 0);  mul_734 = None
    unsqueeze_527: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_526, 2);  unsqueeze_526 = None
    unsqueeze_528: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_527, 3);  unsqueeze_527 = None
    mul_735: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_208, primals_209);  primals_209 = None
    unsqueeze_529: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_735, 0);  mul_735 = None
    unsqueeze_530: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_529, 2);  unsqueeze_529 = None
    unsqueeze_531: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_530, 3);  unsqueeze_530 = None
    sub_146: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_69, unsqueeze_522);  convolution_69 = unsqueeze_522 = None
    mul_736: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_146, unsqueeze_528);  sub_146 = unsqueeze_528 = None
    sub_147: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_14, mul_736);  where_14 = mul_736 = None
    sub_148: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_147, unsqueeze_525);  sub_147 = unsqueeze_525 = None
    mul_737: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_148, unsqueeze_531);  sub_148 = unsqueeze_531 = None
    mul_738: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_33, squeeze_208);  sum_33 = squeeze_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_backward_15 = torch.ops.aten.convolution_backward.default(mul_737, relu_65, primals_208, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_737 = primals_208 = None
    getitem_537: "f32[8, 1024, 14, 14]" = convolution_backward_15[0]
    getitem_538: "f32[1024, 1024, 1, 1]" = convolution_backward_15[1];  convolution_backward_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    add_471: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(getitem_522, getitem_537);  getitem_522 = getitem_537 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    alias_127: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(relu_65);  relu_65 = None
    alias_128: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(alias_127);  alias_127 = None
    le_15: "b8[8, 1024, 14, 14]" = torch.ops.aten.le.Scalar(alias_128, 0);  alias_128 = None
    scalar_tensor_15: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_15: "f32[8, 1024, 14, 14]" = torch.ops.aten.where.self(le_15, scalar_tensor_15, add_471);  le_15 = scalar_tensor_15 = add_471 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    unsqueeze_532: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_204, 0);  squeeze_204 = None
    unsqueeze_533: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_532, 2);  unsqueeze_532 = None
    unsqueeze_534: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_533, 3);  unsqueeze_533 = None
    sum_34: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_15, [0, 2, 3])
    sub_149: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_68, unsqueeze_534)
    mul_739: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_15, sub_149);  sub_149 = None
    sum_35: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_739, [0, 2, 3]);  mul_739 = None
    mul_740: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_34, 0.0006377551020408163)
    unsqueeze_535: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_740, 0);  mul_740 = None
    unsqueeze_536: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_535, 2);  unsqueeze_535 = None
    unsqueeze_537: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_536, 3);  unsqueeze_536 = None
    mul_741: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_35, 0.0006377551020408163)
    mul_742: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_205, squeeze_205)
    mul_743: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_741, mul_742);  mul_741 = mul_742 = None
    unsqueeze_538: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_743, 0);  mul_743 = None
    unsqueeze_539: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_538, 2);  unsqueeze_538 = None
    unsqueeze_540: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_539, 3);  unsqueeze_539 = None
    mul_744: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_205, primals_206);  primals_206 = None
    unsqueeze_541: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_744, 0);  mul_744 = None
    unsqueeze_542: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_541, 2);  unsqueeze_541 = None
    unsqueeze_543: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_542, 3);  unsqueeze_542 = None
    sub_150: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_68, unsqueeze_534);  convolution_68 = unsqueeze_534 = None
    mul_745: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_150, unsqueeze_540);  sub_150 = unsqueeze_540 = None
    sub_151: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_15, mul_745);  mul_745 = None
    sub_152: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_151, unsqueeze_537);  sub_151 = unsqueeze_537 = None
    mul_746: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_152, unsqueeze_543);  sub_152 = unsqueeze_543 = None
    mul_747: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_35, squeeze_205);  sum_35 = squeeze_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_backward_16 = torch.ops.aten.convolution_backward.default(mul_746, cat_12, primals_205, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_746 = cat_12 = primals_205 = None
    getitem_540: "f32[8, 512, 14, 14]" = convolution_backward_16[0]
    getitem_541: "f32[1024, 512, 1, 1]" = convolution_backward_16[1];  convolution_backward_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    slice_13: "f32[8, 128, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_540, 1, 0, 128)
    slice_14: "f32[8, 128, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_540, 1, 128, 256)
    slice_15: "f32[8, 128, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_540, 1, 256, 384)
    slice_16: "f32[8, 128, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_540, 1, 384, 512);  getitem_540 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_130: "f32[8, 128, 14, 14]" = torch.ops.aten.alias.default(relu_64);  relu_64 = None
    alias_131: "f32[8, 128, 14, 14]" = torch.ops.aten.alias.default(alias_130);  alias_130 = None
    le_16: "b8[8, 128, 14, 14]" = torch.ops.aten.le.Scalar(alias_131, 0);  alias_131 = None
    scalar_tensor_16: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_16: "f32[8, 128, 14, 14]" = torch.ops.aten.where.self(le_16, scalar_tensor_16, slice_15);  le_16 = scalar_tensor_16 = slice_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_544: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_201, 0);  squeeze_201 = None
    unsqueeze_545: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_544, 2);  unsqueeze_544 = None
    unsqueeze_546: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_545, 3);  unsqueeze_545 = None
    sum_36: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_16, [0, 2, 3])
    sub_153: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_67, unsqueeze_546)
    mul_748: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_16, sub_153);  sub_153 = None
    sum_37: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_748, [0, 2, 3]);  mul_748 = None
    mul_749: "f32[128]" = torch.ops.aten.mul.Tensor(sum_36, 0.0006377551020408163)
    unsqueeze_547: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_749, 0);  mul_749 = None
    unsqueeze_548: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_547, 2);  unsqueeze_547 = None
    unsqueeze_549: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_548, 3);  unsqueeze_548 = None
    mul_750: "f32[128]" = torch.ops.aten.mul.Tensor(sum_37, 0.0006377551020408163)
    mul_751: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_202, squeeze_202)
    mul_752: "f32[128]" = torch.ops.aten.mul.Tensor(mul_750, mul_751);  mul_750 = mul_751 = None
    unsqueeze_550: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_752, 0);  mul_752 = None
    unsqueeze_551: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_550, 2);  unsqueeze_550 = None
    unsqueeze_552: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_551, 3);  unsqueeze_551 = None
    mul_753: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_202, primals_203);  primals_203 = None
    unsqueeze_553: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_753, 0);  mul_753 = None
    unsqueeze_554: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_553, 2);  unsqueeze_553 = None
    unsqueeze_555: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_554, 3);  unsqueeze_554 = None
    sub_154: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_67, unsqueeze_546);  convolution_67 = unsqueeze_546 = None
    mul_754: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_154, unsqueeze_552);  sub_154 = unsqueeze_552 = None
    sub_155: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(where_16, mul_754);  where_16 = mul_754 = None
    sub_156: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(sub_155, unsqueeze_549);  sub_155 = unsqueeze_549 = None
    mul_755: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_156, unsqueeze_555);  sub_156 = unsqueeze_555 = None
    mul_756: "f32[128]" = torch.ops.aten.mul.Tensor(sum_37, squeeze_202);  sum_37 = squeeze_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_17 = torch.ops.aten.convolution_backward.default(mul_755, add_366, primals_202, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_755 = add_366 = primals_202 = None
    getitem_543: "f32[8, 128, 14, 14]" = convolution_backward_17[0]
    getitem_544: "f32[128, 16, 3, 3]" = convolution_backward_17[1];  convolution_backward_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_472: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(slice_14, getitem_543);  slice_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_133: "f32[8, 128, 14, 14]" = torch.ops.aten.alias.default(relu_63);  relu_63 = None
    alias_134: "f32[8, 128, 14, 14]" = torch.ops.aten.alias.default(alias_133);  alias_133 = None
    le_17: "b8[8, 128, 14, 14]" = torch.ops.aten.le.Scalar(alias_134, 0);  alias_134 = None
    scalar_tensor_17: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_17: "f32[8, 128, 14, 14]" = torch.ops.aten.where.self(le_17, scalar_tensor_17, add_472);  le_17 = scalar_tensor_17 = add_472 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_556: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_198, 0);  squeeze_198 = None
    unsqueeze_557: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_556, 2);  unsqueeze_556 = None
    unsqueeze_558: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_557, 3);  unsqueeze_557 = None
    sum_38: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_17, [0, 2, 3])
    sub_157: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_66, unsqueeze_558)
    mul_757: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_17, sub_157);  sub_157 = None
    sum_39: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_757, [0, 2, 3]);  mul_757 = None
    mul_758: "f32[128]" = torch.ops.aten.mul.Tensor(sum_38, 0.0006377551020408163)
    unsqueeze_559: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_758, 0);  mul_758 = None
    unsqueeze_560: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_559, 2);  unsqueeze_559 = None
    unsqueeze_561: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_560, 3);  unsqueeze_560 = None
    mul_759: "f32[128]" = torch.ops.aten.mul.Tensor(sum_39, 0.0006377551020408163)
    mul_760: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_199, squeeze_199)
    mul_761: "f32[128]" = torch.ops.aten.mul.Tensor(mul_759, mul_760);  mul_759 = mul_760 = None
    unsqueeze_562: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_761, 0);  mul_761 = None
    unsqueeze_563: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_562, 2);  unsqueeze_562 = None
    unsqueeze_564: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_563, 3);  unsqueeze_563 = None
    mul_762: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_199, primals_200);  primals_200 = None
    unsqueeze_565: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_762, 0);  mul_762 = None
    unsqueeze_566: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_565, 2);  unsqueeze_565 = None
    unsqueeze_567: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_566, 3);  unsqueeze_566 = None
    sub_158: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_66, unsqueeze_558);  convolution_66 = unsqueeze_558 = None
    mul_763: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_158, unsqueeze_564);  sub_158 = unsqueeze_564 = None
    sub_159: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(where_17, mul_763);  where_17 = mul_763 = None
    sub_160: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(sub_159, unsqueeze_561);  sub_159 = unsqueeze_561 = None
    mul_764: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_160, unsqueeze_567);  sub_160 = unsqueeze_567 = None
    mul_765: "f32[128]" = torch.ops.aten.mul.Tensor(sum_39, squeeze_199);  sum_39 = squeeze_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_18 = torch.ops.aten.convolution_backward.default(mul_764, add_360, primals_199, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_764 = add_360 = primals_199 = None
    getitem_546: "f32[8, 128, 14, 14]" = convolution_backward_18[0]
    getitem_547: "f32[128, 16, 3, 3]" = convolution_backward_18[1];  convolution_backward_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_473: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(slice_13, getitem_546);  slice_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_136: "f32[8, 128, 14, 14]" = torch.ops.aten.alias.default(relu_62);  relu_62 = None
    alias_137: "f32[8, 128, 14, 14]" = torch.ops.aten.alias.default(alias_136);  alias_136 = None
    le_18: "b8[8, 128, 14, 14]" = torch.ops.aten.le.Scalar(alias_137, 0);  alias_137 = None
    scalar_tensor_18: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_18: "f32[8, 128, 14, 14]" = torch.ops.aten.where.self(le_18, scalar_tensor_18, add_473);  le_18 = scalar_tensor_18 = add_473 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_568: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_195, 0);  squeeze_195 = None
    unsqueeze_569: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_568, 2);  unsqueeze_568 = None
    unsqueeze_570: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_569, 3);  unsqueeze_569 = None
    sum_40: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_18, [0, 2, 3])
    sub_161: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_65, unsqueeze_570)
    mul_766: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_18, sub_161);  sub_161 = None
    sum_41: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_766, [0, 2, 3]);  mul_766 = None
    mul_767: "f32[128]" = torch.ops.aten.mul.Tensor(sum_40, 0.0006377551020408163)
    unsqueeze_571: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_767, 0);  mul_767 = None
    unsqueeze_572: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_571, 2);  unsqueeze_571 = None
    unsqueeze_573: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_572, 3);  unsqueeze_572 = None
    mul_768: "f32[128]" = torch.ops.aten.mul.Tensor(sum_41, 0.0006377551020408163)
    mul_769: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_196, squeeze_196)
    mul_770: "f32[128]" = torch.ops.aten.mul.Tensor(mul_768, mul_769);  mul_768 = mul_769 = None
    unsqueeze_574: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_770, 0);  mul_770 = None
    unsqueeze_575: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_574, 2);  unsqueeze_574 = None
    unsqueeze_576: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_575, 3);  unsqueeze_575 = None
    mul_771: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_196, primals_197);  primals_197 = None
    unsqueeze_577: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_771, 0);  mul_771 = None
    unsqueeze_578: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_577, 2);  unsqueeze_577 = None
    unsqueeze_579: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_578, 3);  unsqueeze_578 = None
    sub_162: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_65, unsqueeze_570);  convolution_65 = unsqueeze_570 = None
    mul_772: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_162, unsqueeze_576);  sub_162 = unsqueeze_576 = None
    sub_163: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(where_18, mul_772);  where_18 = mul_772 = None
    sub_164: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(sub_163, unsqueeze_573);  sub_163 = unsqueeze_573 = None
    mul_773: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_164, unsqueeze_579);  sub_164 = unsqueeze_579 = None
    mul_774: "f32[128]" = torch.ops.aten.mul.Tensor(sum_41, squeeze_196);  sum_41 = squeeze_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_19 = torch.ops.aten.convolution_backward.default(mul_773, getitem_376, primals_196, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_773 = getitem_376 = primals_196 = None
    getitem_549: "f32[8, 128, 14, 14]" = convolution_backward_19[0]
    getitem_550: "f32[128, 16, 3, 3]" = convolution_backward_19[1];  convolution_backward_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    cat_19: "f32[8, 512, 14, 14]" = torch.ops.aten.cat.default([getitem_549, getitem_546, getitem_543, slice_16], 1);  getitem_549 = getitem_546 = getitem_543 = slice_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    alias_139: "f32[8, 512, 14, 14]" = torch.ops.aten.alias.default(relu_61);  relu_61 = None
    alias_140: "f32[8, 512, 14, 14]" = torch.ops.aten.alias.default(alias_139);  alias_139 = None
    le_19: "b8[8, 512, 14, 14]" = torch.ops.aten.le.Scalar(alias_140, 0);  alias_140 = None
    scalar_tensor_19: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_19: "f32[8, 512, 14, 14]" = torch.ops.aten.where.self(le_19, scalar_tensor_19, cat_19);  le_19 = scalar_tensor_19 = cat_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    unsqueeze_580: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_192, 0);  squeeze_192 = None
    unsqueeze_581: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_580, 2);  unsqueeze_580 = None
    unsqueeze_582: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_581, 3);  unsqueeze_581 = None
    sum_42: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_19, [0, 2, 3])
    sub_165: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_64, unsqueeze_582)
    mul_775: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(where_19, sub_165);  sub_165 = None
    sum_43: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_775, [0, 2, 3]);  mul_775 = None
    mul_776: "f32[512]" = torch.ops.aten.mul.Tensor(sum_42, 0.0006377551020408163)
    unsqueeze_583: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_776, 0);  mul_776 = None
    unsqueeze_584: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_583, 2);  unsqueeze_583 = None
    unsqueeze_585: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_584, 3);  unsqueeze_584 = None
    mul_777: "f32[512]" = torch.ops.aten.mul.Tensor(sum_43, 0.0006377551020408163)
    mul_778: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_193, squeeze_193)
    mul_779: "f32[512]" = torch.ops.aten.mul.Tensor(mul_777, mul_778);  mul_777 = mul_778 = None
    unsqueeze_586: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_779, 0);  mul_779 = None
    unsqueeze_587: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_586, 2);  unsqueeze_586 = None
    unsqueeze_588: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_587, 3);  unsqueeze_587 = None
    mul_780: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_193, primals_194);  primals_194 = None
    unsqueeze_589: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_780, 0);  mul_780 = None
    unsqueeze_590: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_589, 2);  unsqueeze_589 = None
    unsqueeze_591: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_590, 3);  unsqueeze_590 = None
    sub_166: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_64, unsqueeze_582);  convolution_64 = unsqueeze_582 = None
    mul_781: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_166, unsqueeze_588);  sub_166 = unsqueeze_588 = None
    sub_167: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(where_19, mul_781);  where_19 = mul_781 = None
    sub_168: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(sub_167, unsqueeze_585);  sub_167 = unsqueeze_585 = None
    mul_782: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_168, unsqueeze_591);  sub_168 = unsqueeze_591 = None
    mul_783: "f32[512]" = torch.ops.aten.mul.Tensor(sum_43, squeeze_193);  sum_43 = squeeze_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_backward_20 = torch.ops.aten.convolution_backward.default(mul_782, relu_60, primals_193, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_782 = primals_193 = None
    getitem_552: "f32[8, 1024, 14, 14]" = convolution_backward_20[0]
    getitem_553: "f32[512, 1024, 1, 1]" = convolution_backward_20[1];  convolution_backward_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    add_474: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(where_15, getitem_552);  where_15 = getitem_552 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    alias_142: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(relu_60);  relu_60 = None
    alias_143: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(alias_142);  alias_142 = None
    le_20: "b8[8, 1024, 14, 14]" = torch.ops.aten.le.Scalar(alias_143, 0);  alias_143 = None
    scalar_tensor_20: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_20: "f32[8, 1024, 14, 14]" = torch.ops.aten.where.self(le_20, scalar_tensor_20, add_474);  le_20 = scalar_tensor_20 = add_474 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    unsqueeze_592: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_189, 0);  squeeze_189 = None
    unsqueeze_593: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_592, 2);  unsqueeze_592 = None
    unsqueeze_594: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_593, 3);  unsqueeze_593 = None
    sum_44: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_20, [0, 2, 3])
    sub_169: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_63, unsqueeze_594)
    mul_784: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_20, sub_169);  sub_169 = None
    sum_45: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_784, [0, 2, 3]);  mul_784 = None
    mul_785: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_44, 0.0006377551020408163)
    unsqueeze_595: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_785, 0);  mul_785 = None
    unsqueeze_596: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_595, 2);  unsqueeze_595 = None
    unsqueeze_597: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_596, 3);  unsqueeze_596 = None
    mul_786: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_45, 0.0006377551020408163)
    mul_787: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_190, squeeze_190)
    mul_788: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_786, mul_787);  mul_786 = mul_787 = None
    unsqueeze_598: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_788, 0);  mul_788 = None
    unsqueeze_599: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_598, 2);  unsqueeze_598 = None
    unsqueeze_600: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_599, 3);  unsqueeze_599 = None
    mul_789: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_190, primals_191);  primals_191 = None
    unsqueeze_601: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_789, 0);  mul_789 = None
    unsqueeze_602: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_601, 2);  unsqueeze_601 = None
    unsqueeze_603: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_602, 3);  unsqueeze_602 = None
    sub_170: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_63, unsqueeze_594);  convolution_63 = unsqueeze_594 = None
    mul_790: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_170, unsqueeze_600);  sub_170 = unsqueeze_600 = None
    sub_171: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_20, mul_790);  mul_790 = None
    sub_172: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_171, unsqueeze_597);  sub_171 = unsqueeze_597 = None
    mul_791: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_172, unsqueeze_603);  sub_172 = unsqueeze_603 = None
    mul_792: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_45, squeeze_190);  sum_45 = squeeze_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_backward_21 = torch.ops.aten.convolution_backward.default(mul_791, cat_11, primals_190, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_791 = cat_11 = primals_190 = None
    getitem_555: "f32[8, 512, 14, 14]" = convolution_backward_21[0]
    getitem_556: "f32[1024, 512, 1, 1]" = convolution_backward_21[1];  convolution_backward_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    slice_17: "f32[8, 128, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_555, 1, 0, 128)
    slice_18: "f32[8, 128, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_555, 1, 128, 256)
    slice_19: "f32[8, 128, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_555, 1, 256, 384)
    slice_20: "f32[8, 128, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_555, 1, 384, 512);  getitem_555 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_145: "f32[8, 128, 14, 14]" = torch.ops.aten.alias.default(relu_59);  relu_59 = None
    alias_146: "f32[8, 128, 14, 14]" = torch.ops.aten.alias.default(alias_145);  alias_145 = None
    le_21: "b8[8, 128, 14, 14]" = torch.ops.aten.le.Scalar(alias_146, 0);  alias_146 = None
    scalar_tensor_21: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_21: "f32[8, 128, 14, 14]" = torch.ops.aten.where.self(le_21, scalar_tensor_21, slice_19);  le_21 = scalar_tensor_21 = slice_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_604: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_186, 0);  squeeze_186 = None
    unsqueeze_605: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_604, 2);  unsqueeze_604 = None
    unsqueeze_606: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_605, 3);  unsqueeze_605 = None
    sum_46: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_21, [0, 2, 3])
    sub_173: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_62, unsqueeze_606)
    mul_793: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_21, sub_173);  sub_173 = None
    sum_47: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_793, [0, 2, 3]);  mul_793 = None
    mul_794: "f32[128]" = torch.ops.aten.mul.Tensor(sum_46, 0.0006377551020408163)
    unsqueeze_607: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_794, 0);  mul_794 = None
    unsqueeze_608: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_607, 2);  unsqueeze_607 = None
    unsqueeze_609: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_608, 3);  unsqueeze_608 = None
    mul_795: "f32[128]" = torch.ops.aten.mul.Tensor(sum_47, 0.0006377551020408163)
    mul_796: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_187, squeeze_187)
    mul_797: "f32[128]" = torch.ops.aten.mul.Tensor(mul_795, mul_796);  mul_795 = mul_796 = None
    unsqueeze_610: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_797, 0);  mul_797 = None
    unsqueeze_611: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_610, 2);  unsqueeze_610 = None
    unsqueeze_612: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_611, 3);  unsqueeze_611 = None
    mul_798: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_187, primals_188);  primals_188 = None
    unsqueeze_613: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_798, 0);  mul_798 = None
    unsqueeze_614: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_613, 2);  unsqueeze_613 = None
    unsqueeze_615: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_614, 3);  unsqueeze_614 = None
    sub_174: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_62, unsqueeze_606);  convolution_62 = unsqueeze_606 = None
    mul_799: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_174, unsqueeze_612);  sub_174 = unsqueeze_612 = None
    sub_175: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(where_21, mul_799);  where_21 = mul_799 = None
    sub_176: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(sub_175, unsqueeze_609);  sub_175 = unsqueeze_609 = None
    mul_800: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_176, unsqueeze_615);  sub_176 = unsqueeze_615 = None
    mul_801: "f32[128]" = torch.ops.aten.mul.Tensor(sum_47, squeeze_187);  sum_47 = squeeze_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_22 = torch.ops.aten.convolution_backward.default(mul_800, add_338, primals_187, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_800 = add_338 = primals_187 = None
    getitem_558: "f32[8, 128, 14, 14]" = convolution_backward_22[0]
    getitem_559: "f32[128, 16, 3, 3]" = convolution_backward_22[1];  convolution_backward_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_475: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(slice_18, getitem_558);  slice_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_148: "f32[8, 128, 14, 14]" = torch.ops.aten.alias.default(relu_58);  relu_58 = None
    alias_149: "f32[8, 128, 14, 14]" = torch.ops.aten.alias.default(alias_148);  alias_148 = None
    le_22: "b8[8, 128, 14, 14]" = torch.ops.aten.le.Scalar(alias_149, 0);  alias_149 = None
    scalar_tensor_22: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_22: "f32[8, 128, 14, 14]" = torch.ops.aten.where.self(le_22, scalar_tensor_22, add_475);  le_22 = scalar_tensor_22 = add_475 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_616: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_183, 0);  squeeze_183 = None
    unsqueeze_617: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_616, 2);  unsqueeze_616 = None
    unsqueeze_618: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_617, 3);  unsqueeze_617 = None
    sum_48: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_22, [0, 2, 3])
    sub_177: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_61, unsqueeze_618)
    mul_802: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_22, sub_177);  sub_177 = None
    sum_49: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_802, [0, 2, 3]);  mul_802 = None
    mul_803: "f32[128]" = torch.ops.aten.mul.Tensor(sum_48, 0.0006377551020408163)
    unsqueeze_619: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_803, 0);  mul_803 = None
    unsqueeze_620: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_619, 2);  unsqueeze_619 = None
    unsqueeze_621: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_620, 3);  unsqueeze_620 = None
    mul_804: "f32[128]" = torch.ops.aten.mul.Tensor(sum_49, 0.0006377551020408163)
    mul_805: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_184, squeeze_184)
    mul_806: "f32[128]" = torch.ops.aten.mul.Tensor(mul_804, mul_805);  mul_804 = mul_805 = None
    unsqueeze_622: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_806, 0);  mul_806 = None
    unsqueeze_623: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_622, 2);  unsqueeze_622 = None
    unsqueeze_624: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_623, 3);  unsqueeze_623 = None
    mul_807: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_184, primals_185);  primals_185 = None
    unsqueeze_625: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_807, 0);  mul_807 = None
    unsqueeze_626: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_625, 2);  unsqueeze_625 = None
    unsqueeze_627: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_626, 3);  unsqueeze_626 = None
    sub_178: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_61, unsqueeze_618);  convolution_61 = unsqueeze_618 = None
    mul_808: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_178, unsqueeze_624);  sub_178 = unsqueeze_624 = None
    sub_179: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(where_22, mul_808);  where_22 = mul_808 = None
    sub_180: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(sub_179, unsqueeze_621);  sub_179 = unsqueeze_621 = None
    mul_809: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_180, unsqueeze_627);  sub_180 = unsqueeze_627 = None
    mul_810: "f32[128]" = torch.ops.aten.mul.Tensor(sum_49, squeeze_184);  sum_49 = squeeze_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_23 = torch.ops.aten.convolution_backward.default(mul_809, add_332, primals_184, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_809 = add_332 = primals_184 = None
    getitem_561: "f32[8, 128, 14, 14]" = convolution_backward_23[0]
    getitem_562: "f32[128, 16, 3, 3]" = convolution_backward_23[1];  convolution_backward_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_476: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(slice_17, getitem_561);  slice_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_151: "f32[8, 128, 14, 14]" = torch.ops.aten.alias.default(relu_57);  relu_57 = None
    alias_152: "f32[8, 128, 14, 14]" = torch.ops.aten.alias.default(alias_151);  alias_151 = None
    le_23: "b8[8, 128, 14, 14]" = torch.ops.aten.le.Scalar(alias_152, 0);  alias_152 = None
    scalar_tensor_23: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_23: "f32[8, 128, 14, 14]" = torch.ops.aten.where.self(le_23, scalar_tensor_23, add_476);  le_23 = scalar_tensor_23 = add_476 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_628: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_180, 0);  squeeze_180 = None
    unsqueeze_629: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_628, 2);  unsqueeze_628 = None
    unsqueeze_630: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_629, 3);  unsqueeze_629 = None
    sum_50: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_23, [0, 2, 3])
    sub_181: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_60, unsqueeze_630)
    mul_811: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_23, sub_181);  sub_181 = None
    sum_51: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_811, [0, 2, 3]);  mul_811 = None
    mul_812: "f32[128]" = torch.ops.aten.mul.Tensor(sum_50, 0.0006377551020408163)
    unsqueeze_631: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_812, 0);  mul_812 = None
    unsqueeze_632: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_631, 2);  unsqueeze_631 = None
    unsqueeze_633: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_632, 3);  unsqueeze_632 = None
    mul_813: "f32[128]" = torch.ops.aten.mul.Tensor(sum_51, 0.0006377551020408163)
    mul_814: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_181, squeeze_181)
    mul_815: "f32[128]" = torch.ops.aten.mul.Tensor(mul_813, mul_814);  mul_813 = mul_814 = None
    unsqueeze_634: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_815, 0);  mul_815 = None
    unsqueeze_635: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_634, 2);  unsqueeze_634 = None
    unsqueeze_636: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_635, 3);  unsqueeze_635 = None
    mul_816: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_181, primals_182);  primals_182 = None
    unsqueeze_637: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_816, 0);  mul_816 = None
    unsqueeze_638: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_637, 2);  unsqueeze_637 = None
    unsqueeze_639: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_638, 3);  unsqueeze_638 = None
    sub_182: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_60, unsqueeze_630);  convolution_60 = unsqueeze_630 = None
    mul_817: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_182, unsqueeze_636);  sub_182 = unsqueeze_636 = None
    sub_183: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(where_23, mul_817);  where_23 = mul_817 = None
    sub_184: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(sub_183, unsqueeze_633);  sub_183 = unsqueeze_633 = None
    mul_818: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_184, unsqueeze_639);  sub_184 = unsqueeze_639 = None
    mul_819: "f32[128]" = torch.ops.aten.mul.Tensor(sum_51, squeeze_181);  sum_51 = squeeze_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_24 = torch.ops.aten.convolution_backward.default(mul_818, getitem_346, primals_181, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_818 = getitem_346 = primals_181 = None
    getitem_564: "f32[8, 128, 14, 14]" = convolution_backward_24[0]
    getitem_565: "f32[128, 16, 3, 3]" = convolution_backward_24[1];  convolution_backward_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    cat_20: "f32[8, 512, 14, 14]" = torch.ops.aten.cat.default([getitem_564, getitem_561, getitem_558, slice_20], 1);  getitem_564 = getitem_561 = getitem_558 = slice_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    alias_154: "f32[8, 512, 14, 14]" = torch.ops.aten.alias.default(relu_56);  relu_56 = None
    alias_155: "f32[8, 512, 14, 14]" = torch.ops.aten.alias.default(alias_154);  alias_154 = None
    le_24: "b8[8, 512, 14, 14]" = torch.ops.aten.le.Scalar(alias_155, 0);  alias_155 = None
    scalar_tensor_24: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_24: "f32[8, 512, 14, 14]" = torch.ops.aten.where.self(le_24, scalar_tensor_24, cat_20);  le_24 = scalar_tensor_24 = cat_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    unsqueeze_640: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_177, 0);  squeeze_177 = None
    unsqueeze_641: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_640, 2);  unsqueeze_640 = None
    unsqueeze_642: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_641, 3);  unsqueeze_641 = None
    sum_52: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_24, [0, 2, 3])
    sub_185: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_59, unsqueeze_642)
    mul_820: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(where_24, sub_185);  sub_185 = None
    sum_53: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_820, [0, 2, 3]);  mul_820 = None
    mul_821: "f32[512]" = torch.ops.aten.mul.Tensor(sum_52, 0.0006377551020408163)
    unsqueeze_643: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_821, 0);  mul_821 = None
    unsqueeze_644: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_643, 2);  unsqueeze_643 = None
    unsqueeze_645: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_644, 3);  unsqueeze_644 = None
    mul_822: "f32[512]" = torch.ops.aten.mul.Tensor(sum_53, 0.0006377551020408163)
    mul_823: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_178, squeeze_178)
    mul_824: "f32[512]" = torch.ops.aten.mul.Tensor(mul_822, mul_823);  mul_822 = mul_823 = None
    unsqueeze_646: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_824, 0);  mul_824 = None
    unsqueeze_647: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_646, 2);  unsqueeze_646 = None
    unsqueeze_648: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_647, 3);  unsqueeze_647 = None
    mul_825: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_178, primals_179);  primals_179 = None
    unsqueeze_649: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_825, 0);  mul_825 = None
    unsqueeze_650: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_649, 2);  unsqueeze_649 = None
    unsqueeze_651: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_650, 3);  unsqueeze_650 = None
    sub_186: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_59, unsqueeze_642);  convolution_59 = unsqueeze_642 = None
    mul_826: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_186, unsqueeze_648);  sub_186 = unsqueeze_648 = None
    sub_187: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(where_24, mul_826);  where_24 = mul_826 = None
    sub_188: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(sub_187, unsqueeze_645);  sub_187 = unsqueeze_645 = None
    mul_827: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_188, unsqueeze_651);  sub_188 = unsqueeze_651 = None
    mul_828: "f32[512]" = torch.ops.aten.mul.Tensor(sum_53, squeeze_178);  sum_53 = squeeze_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_backward_25 = torch.ops.aten.convolution_backward.default(mul_827, relu_55, primals_178, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_827 = primals_178 = None
    getitem_567: "f32[8, 1024, 14, 14]" = convolution_backward_25[0]
    getitem_568: "f32[512, 1024, 1, 1]" = convolution_backward_25[1];  convolution_backward_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    add_477: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(where_20, getitem_567);  where_20 = getitem_567 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    alias_157: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(relu_55);  relu_55 = None
    alias_158: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(alias_157);  alias_157 = None
    le_25: "b8[8, 1024, 14, 14]" = torch.ops.aten.le.Scalar(alias_158, 0);  alias_158 = None
    scalar_tensor_25: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_25: "f32[8, 1024, 14, 14]" = torch.ops.aten.where.self(le_25, scalar_tensor_25, add_477);  le_25 = scalar_tensor_25 = add_477 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    unsqueeze_652: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_174, 0);  squeeze_174 = None
    unsqueeze_653: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_652, 2);  unsqueeze_652 = None
    unsqueeze_654: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_653, 3);  unsqueeze_653 = None
    sum_54: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_25, [0, 2, 3])
    sub_189: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_58, unsqueeze_654)
    mul_829: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_25, sub_189);  sub_189 = None
    sum_55: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_829, [0, 2, 3]);  mul_829 = None
    mul_830: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_54, 0.0006377551020408163)
    unsqueeze_655: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_830, 0);  mul_830 = None
    unsqueeze_656: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_655, 2);  unsqueeze_655 = None
    unsqueeze_657: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_656, 3);  unsqueeze_656 = None
    mul_831: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_55, 0.0006377551020408163)
    mul_832: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_175, squeeze_175)
    mul_833: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_831, mul_832);  mul_831 = mul_832 = None
    unsqueeze_658: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_833, 0);  mul_833 = None
    unsqueeze_659: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_658, 2);  unsqueeze_658 = None
    unsqueeze_660: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_659, 3);  unsqueeze_659 = None
    mul_834: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_175, primals_176);  primals_176 = None
    unsqueeze_661: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_834, 0);  mul_834 = None
    unsqueeze_662: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_661, 2);  unsqueeze_661 = None
    unsqueeze_663: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_662, 3);  unsqueeze_662 = None
    sub_190: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_58, unsqueeze_654);  convolution_58 = unsqueeze_654 = None
    mul_835: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_190, unsqueeze_660);  sub_190 = unsqueeze_660 = None
    sub_191: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_25, mul_835);  mul_835 = None
    sub_192: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_191, unsqueeze_657);  sub_191 = unsqueeze_657 = None
    mul_836: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_192, unsqueeze_663);  sub_192 = unsqueeze_663 = None
    mul_837: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_55, squeeze_175);  sum_55 = squeeze_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_backward_26 = torch.ops.aten.convolution_backward.default(mul_836, cat_10, primals_175, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_836 = cat_10 = primals_175 = None
    getitem_570: "f32[8, 512, 14, 14]" = convolution_backward_26[0]
    getitem_571: "f32[1024, 512, 1, 1]" = convolution_backward_26[1];  convolution_backward_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    slice_21: "f32[8, 128, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_570, 1, 0, 128)
    slice_22: "f32[8, 128, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_570, 1, 128, 256)
    slice_23: "f32[8, 128, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_570, 1, 256, 384)
    slice_24: "f32[8, 128, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_570, 1, 384, 512);  getitem_570 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_160: "f32[8, 128, 14, 14]" = torch.ops.aten.alias.default(relu_54);  relu_54 = None
    alias_161: "f32[8, 128, 14, 14]" = torch.ops.aten.alias.default(alias_160);  alias_160 = None
    le_26: "b8[8, 128, 14, 14]" = torch.ops.aten.le.Scalar(alias_161, 0);  alias_161 = None
    scalar_tensor_26: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_26: "f32[8, 128, 14, 14]" = torch.ops.aten.where.self(le_26, scalar_tensor_26, slice_23);  le_26 = scalar_tensor_26 = slice_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_664: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_171, 0);  squeeze_171 = None
    unsqueeze_665: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_664, 2);  unsqueeze_664 = None
    unsqueeze_666: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_665, 3);  unsqueeze_665 = None
    sum_56: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_26, [0, 2, 3])
    sub_193: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_57, unsqueeze_666)
    mul_838: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_26, sub_193);  sub_193 = None
    sum_57: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_838, [0, 2, 3]);  mul_838 = None
    mul_839: "f32[128]" = torch.ops.aten.mul.Tensor(sum_56, 0.0006377551020408163)
    unsqueeze_667: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_839, 0);  mul_839 = None
    unsqueeze_668: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_667, 2);  unsqueeze_667 = None
    unsqueeze_669: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_668, 3);  unsqueeze_668 = None
    mul_840: "f32[128]" = torch.ops.aten.mul.Tensor(sum_57, 0.0006377551020408163)
    mul_841: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_172, squeeze_172)
    mul_842: "f32[128]" = torch.ops.aten.mul.Tensor(mul_840, mul_841);  mul_840 = mul_841 = None
    unsqueeze_670: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_842, 0);  mul_842 = None
    unsqueeze_671: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_670, 2);  unsqueeze_670 = None
    unsqueeze_672: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_671, 3);  unsqueeze_671 = None
    mul_843: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_172, primals_173);  primals_173 = None
    unsqueeze_673: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_843, 0);  mul_843 = None
    unsqueeze_674: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_673, 2);  unsqueeze_673 = None
    unsqueeze_675: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_674, 3);  unsqueeze_674 = None
    sub_194: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_57, unsqueeze_666);  convolution_57 = unsqueeze_666 = None
    mul_844: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_194, unsqueeze_672);  sub_194 = unsqueeze_672 = None
    sub_195: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(where_26, mul_844);  where_26 = mul_844 = None
    sub_196: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(sub_195, unsqueeze_669);  sub_195 = unsqueeze_669 = None
    mul_845: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_196, unsqueeze_675);  sub_196 = unsqueeze_675 = None
    mul_846: "f32[128]" = torch.ops.aten.mul.Tensor(sum_57, squeeze_172);  sum_57 = squeeze_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_27 = torch.ops.aten.convolution_backward.default(mul_845, add_310, primals_172, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_845 = add_310 = primals_172 = None
    getitem_573: "f32[8, 128, 14, 14]" = convolution_backward_27[0]
    getitem_574: "f32[128, 16, 3, 3]" = convolution_backward_27[1];  convolution_backward_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_478: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(slice_22, getitem_573);  slice_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_163: "f32[8, 128, 14, 14]" = torch.ops.aten.alias.default(relu_53);  relu_53 = None
    alias_164: "f32[8, 128, 14, 14]" = torch.ops.aten.alias.default(alias_163);  alias_163 = None
    le_27: "b8[8, 128, 14, 14]" = torch.ops.aten.le.Scalar(alias_164, 0);  alias_164 = None
    scalar_tensor_27: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_27: "f32[8, 128, 14, 14]" = torch.ops.aten.where.self(le_27, scalar_tensor_27, add_478);  le_27 = scalar_tensor_27 = add_478 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_676: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_168, 0);  squeeze_168 = None
    unsqueeze_677: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_676, 2);  unsqueeze_676 = None
    unsqueeze_678: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_677, 3);  unsqueeze_677 = None
    sum_58: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_27, [0, 2, 3])
    sub_197: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_56, unsqueeze_678)
    mul_847: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_27, sub_197);  sub_197 = None
    sum_59: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_847, [0, 2, 3]);  mul_847 = None
    mul_848: "f32[128]" = torch.ops.aten.mul.Tensor(sum_58, 0.0006377551020408163)
    unsqueeze_679: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_848, 0);  mul_848 = None
    unsqueeze_680: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_679, 2);  unsqueeze_679 = None
    unsqueeze_681: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_680, 3);  unsqueeze_680 = None
    mul_849: "f32[128]" = torch.ops.aten.mul.Tensor(sum_59, 0.0006377551020408163)
    mul_850: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_169, squeeze_169)
    mul_851: "f32[128]" = torch.ops.aten.mul.Tensor(mul_849, mul_850);  mul_849 = mul_850 = None
    unsqueeze_682: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_851, 0);  mul_851 = None
    unsqueeze_683: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_682, 2);  unsqueeze_682 = None
    unsqueeze_684: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_683, 3);  unsqueeze_683 = None
    mul_852: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_169, primals_170);  primals_170 = None
    unsqueeze_685: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_852, 0);  mul_852 = None
    unsqueeze_686: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_685, 2);  unsqueeze_685 = None
    unsqueeze_687: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_686, 3);  unsqueeze_686 = None
    sub_198: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_56, unsqueeze_678);  convolution_56 = unsqueeze_678 = None
    mul_853: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_198, unsqueeze_684);  sub_198 = unsqueeze_684 = None
    sub_199: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(where_27, mul_853);  where_27 = mul_853 = None
    sub_200: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(sub_199, unsqueeze_681);  sub_199 = unsqueeze_681 = None
    mul_854: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_200, unsqueeze_687);  sub_200 = unsqueeze_687 = None
    mul_855: "f32[128]" = torch.ops.aten.mul.Tensor(sum_59, squeeze_169);  sum_59 = squeeze_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_28 = torch.ops.aten.convolution_backward.default(mul_854, add_304, primals_169, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_854 = add_304 = primals_169 = None
    getitem_576: "f32[8, 128, 14, 14]" = convolution_backward_28[0]
    getitem_577: "f32[128, 16, 3, 3]" = convolution_backward_28[1];  convolution_backward_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_479: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(slice_21, getitem_576);  slice_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_166: "f32[8, 128, 14, 14]" = torch.ops.aten.alias.default(relu_52);  relu_52 = None
    alias_167: "f32[8, 128, 14, 14]" = torch.ops.aten.alias.default(alias_166);  alias_166 = None
    le_28: "b8[8, 128, 14, 14]" = torch.ops.aten.le.Scalar(alias_167, 0);  alias_167 = None
    scalar_tensor_28: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_28: "f32[8, 128, 14, 14]" = torch.ops.aten.where.self(le_28, scalar_tensor_28, add_479);  le_28 = scalar_tensor_28 = add_479 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_688: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_165, 0);  squeeze_165 = None
    unsqueeze_689: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_688, 2);  unsqueeze_688 = None
    unsqueeze_690: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_689, 3);  unsqueeze_689 = None
    sum_60: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_28, [0, 2, 3])
    sub_201: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_55, unsqueeze_690)
    mul_856: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_28, sub_201);  sub_201 = None
    sum_61: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_856, [0, 2, 3]);  mul_856 = None
    mul_857: "f32[128]" = torch.ops.aten.mul.Tensor(sum_60, 0.0006377551020408163)
    unsqueeze_691: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_857, 0);  mul_857 = None
    unsqueeze_692: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_691, 2);  unsqueeze_691 = None
    unsqueeze_693: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_692, 3);  unsqueeze_692 = None
    mul_858: "f32[128]" = torch.ops.aten.mul.Tensor(sum_61, 0.0006377551020408163)
    mul_859: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_166, squeeze_166)
    mul_860: "f32[128]" = torch.ops.aten.mul.Tensor(mul_858, mul_859);  mul_858 = mul_859 = None
    unsqueeze_694: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_860, 0);  mul_860 = None
    unsqueeze_695: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_694, 2);  unsqueeze_694 = None
    unsqueeze_696: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_695, 3);  unsqueeze_695 = None
    mul_861: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_166, primals_167);  primals_167 = None
    unsqueeze_697: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_861, 0);  mul_861 = None
    unsqueeze_698: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_697, 2);  unsqueeze_697 = None
    unsqueeze_699: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_698, 3);  unsqueeze_698 = None
    sub_202: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_55, unsqueeze_690);  convolution_55 = unsqueeze_690 = None
    mul_862: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_202, unsqueeze_696);  sub_202 = unsqueeze_696 = None
    sub_203: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(where_28, mul_862);  where_28 = mul_862 = None
    sub_204: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(sub_203, unsqueeze_693);  sub_203 = unsqueeze_693 = None
    mul_863: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_204, unsqueeze_699);  sub_204 = unsqueeze_699 = None
    mul_864: "f32[128]" = torch.ops.aten.mul.Tensor(sum_61, squeeze_166);  sum_61 = squeeze_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_29 = torch.ops.aten.convolution_backward.default(mul_863, getitem_316, primals_166, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_863 = getitem_316 = primals_166 = None
    getitem_579: "f32[8, 128, 14, 14]" = convolution_backward_29[0]
    getitem_580: "f32[128, 16, 3, 3]" = convolution_backward_29[1];  convolution_backward_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    cat_21: "f32[8, 512, 14, 14]" = torch.ops.aten.cat.default([getitem_579, getitem_576, getitem_573, slice_24], 1);  getitem_579 = getitem_576 = getitem_573 = slice_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    alias_169: "f32[8, 512, 14, 14]" = torch.ops.aten.alias.default(relu_51);  relu_51 = None
    alias_170: "f32[8, 512, 14, 14]" = torch.ops.aten.alias.default(alias_169);  alias_169 = None
    le_29: "b8[8, 512, 14, 14]" = torch.ops.aten.le.Scalar(alias_170, 0);  alias_170 = None
    scalar_tensor_29: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_29: "f32[8, 512, 14, 14]" = torch.ops.aten.where.self(le_29, scalar_tensor_29, cat_21);  le_29 = scalar_tensor_29 = cat_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    unsqueeze_700: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_162, 0);  squeeze_162 = None
    unsqueeze_701: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_700, 2);  unsqueeze_700 = None
    unsqueeze_702: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_701, 3);  unsqueeze_701 = None
    sum_62: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_29, [0, 2, 3])
    sub_205: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_54, unsqueeze_702)
    mul_865: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(where_29, sub_205);  sub_205 = None
    sum_63: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_865, [0, 2, 3]);  mul_865 = None
    mul_866: "f32[512]" = torch.ops.aten.mul.Tensor(sum_62, 0.0006377551020408163)
    unsqueeze_703: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_866, 0);  mul_866 = None
    unsqueeze_704: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_703, 2);  unsqueeze_703 = None
    unsqueeze_705: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_704, 3);  unsqueeze_704 = None
    mul_867: "f32[512]" = torch.ops.aten.mul.Tensor(sum_63, 0.0006377551020408163)
    mul_868: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_163, squeeze_163)
    mul_869: "f32[512]" = torch.ops.aten.mul.Tensor(mul_867, mul_868);  mul_867 = mul_868 = None
    unsqueeze_706: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_869, 0);  mul_869 = None
    unsqueeze_707: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_706, 2);  unsqueeze_706 = None
    unsqueeze_708: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_707, 3);  unsqueeze_707 = None
    mul_870: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_163, primals_164);  primals_164 = None
    unsqueeze_709: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_870, 0);  mul_870 = None
    unsqueeze_710: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_709, 2);  unsqueeze_709 = None
    unsqueeze_711: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_710, 3);  unsqueeze_710 = None
    sub_206: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_54, unsqueeze_702);  convolution_54 = unsqueeze_702 = None
    mul_871: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_206, unsqueeze_708);  sub_206 = unsqueeze_708 = None
    sub_207: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(where_29, mul_871);  where_29 = mul_871 = None
    sub_208: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(sub_207, unsqueeze_705);  sub_207 = unsqueeze_705 = None
    mul_872: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_208, unsqueeze_711);  sub_208 = unsqueeze_711 = None
    mul_873: "f32[512]" = torch.ops.aten.mul.Tensor(sum_63, squeeze_163);  sum_63 = squeeze_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_backward_30 = torch.ops.aten.convolution_backward.default(mul_872, relu_50, primals_163, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_872 = primals_163 = None
    getitem_582: "f32[8, 1024, 14, 14]" = convolution_backward_30[0]
    getitem_583: "f32[512, 1024, 1, 1]" = convolution_backward_30[1];  convolution_backward_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    add_480: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(where_25, getitem_582);  where_25 = getitem_582 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    alias_172: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(relu_50);  relu_50 = None
    alias_173: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(alias_172);  alias_172 = None
    le_30: "b8[8, 1024, 14, 14]" = torch.ops.aten.le.Scalar(alias_173, 0);  alias_173 = None
    scalar_tensor_30: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_30: "f32[8, 1024, 14, 14]" = torch.ops.aten.where.self(le_30, scalar_tensor_30, add_480);  le_30 = scalar_tensor_30 = add_480 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    unsqueeze_712: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_159, 0);  squeeze_159 = None
    unsqueeze_713: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_712, 2);  unsqueeze_712 = None
    unsqueeze_714: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_713, 3);  unsqueeze_713 = None
    sum_64: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_30, [0, 2, 3])
    sub_209: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_53, unsqueeze_714)
    mul_874: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_30, sub_209);  sub_209 = None
    sum_65: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_874, [0, 2, 3]);  mul_874 = None
    mul_875: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_64, 0.0006377551020408163)
    unsqueeze_715: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_875, 0);  mul_875 = None
    unsqueeze_716: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_715, 2);  unsqueeze_715 = None
    unsqueeze_717: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_716, 3);  unsqueeze_716 = None
    mul_876: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_65, 0.0006377551020408163)
    mul_877: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_160, squeeze_160)
    mul_878: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_876, mul_877);  mul_876 = mul_877 = None
    unsqueeze_718: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_878, 0);  mul_878 = None
    unsqueeze_719: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_718, 2);  unsqueeze_718 = None
    unsqueeze_720: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_719, 3);  unsqueeze_719 = None
    mul_879: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_160, primals_161);  primals_161 = None
    unsqueeze_721: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_879, 0);  mul_879 = None
    unsqueeze_722: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_721, 2);  unsqueeze_721 = None
    unsqueeze_723: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_722, 3);  unsqueeze_722 = None
    sub_210: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_53, unsqueeze_714);  convolution_53 = unsqueeze_714 = None
    mul_880: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_210, unsqueeze_720);  sub_210 = unsqueeze_720 = None
    sub_211: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_30, mul_880);  mul_880 = None
    sub_212: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_211, unsqueeze_717);  sub_211 = unsqueeze_717 = None
    mul_881: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_212, unsqueeze_723);  sub_212 = unsqueeze_723 = None
    mul_882: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_65, squeeze_160);  sum_65 = squeeze_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_backward_31 = torch.ops.aten.convolution_backward.default(mul_881, cat_9, primals_160, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_881 = cat_9 = primals_160 = None
    getitem_585: "f32[8, 512, 14, 14]" = convolution_backward_31[0]
    getitem_586: "f32[1024, 512, 1, 1]" = convolution_backward_31[1];  convolution_backward_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    slice_25: "f32[8, 128, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_585, 1, 0, 128)
    slice_26: "f32[8, 128, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_585, 1, 128, 256)
    slice_27: "f32[8, 128, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_585, 1, 256, 384)
    slice_28: "f32[8, 128, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_585, 1, 384, 512);  getitem_585 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_175: "f32[8, 128, 14, 14]" = torch.ops.aten.alias.default(relu_49);  relu_49 = None
    alias_176: "f32[8, 128, 14, 14]" = torch.ops.aten.alias.default(alias_175);  alias_175 = None
    le_31: "b8[8, 128, 14, 14]" = torch.ops.aten.le.Scalar(alias_176, 0);  alias_176 = None
    scalar_tensor_31: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_31: "f32[8, 128, 14, 14]" = torch.ops.aten.where.self(le_31, scalar_tensor_31, slice_27);  le_31 = scalar_tensor_31 = slice_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_724: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_156, 0);  squeeze_156 = None
    unsqueeze_725: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_724, 2);  unsqueeze_724 = None
    unsqueeze_726: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_725, 3);  unsqueeze_725 = None
    sum_66: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_31, [0, 2, 3])
    sub_213: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_52, unsqueeze_726)
    mul_883: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_31, sub_213);  sub_213 = None
    sum_67: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_883, [0, 2, 3]);  mul_883 = None
    mul_884: "f32[128]" = torch.ops.aten.mul.Tensor(sum_66, 0.0006377551020408163)
    unsqueeze_727: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_884, 0);  mul_884 = None
    unsqueeze_728: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_727, 2);  unsqueeze_727 = None
    unsqueeze_729: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_728, 3);  unsqueeze_728 = None
    mul_885: "f32[128]" = torch.ops.aten.mul.Tensor(sum_67, 0.0006377551020408163)
    mul_886: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_157, squeeze_157)
    mul_887: "f32[128]" = torch.ops.aten.mul.Tensor(mul_885, mul_886);  mul_885 = mul_886 = None
    unsqueeze_730: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_887, 0);  mul_887 = None
    unsqueeze_731: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_730, 2);  unsqueeze_730 = None
    unsqueeze_732: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_731, 3);  unsqueeze_731 = None
    mul_888: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_157, primals_158);  primals_158 = None
    unsqueeze_733: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_888, 0);  mul_888 = None
    unsqueeze_734: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_733, 2);  unsqueeze_733 = None
    unsqueeze_735: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_734, 3);  unsqueeze_734 = None
    sub_214: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_52, unsqueeze_726);  convolution_52 = unsqueeze_726 = None
    mul_889: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_214, unsqueeze_732);  sub_214 = unsqueeze_732 = None
    sub_215: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(where_31, mul_889);  where_31 = mul_889 = None
    sub_216: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(sub_215, unsqueeze_729);  sub_215 = unsqueeze_729 = None
    mul_890: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_216, unsqueeze_735);  sub_216 = unsqueeze_735 = None
    mul_891: "f32[128]" = torch.ops.aten.mul.Tensor(sum_67, squeeze_157);  sum_67 = squeeze_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_32 = torch.ops.aten.convolution_backward.default(mul_890, add_282, primals_157, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_890 = add_282 = primals_157 = None
    getitem_588: "f32[8, 128, 14, 14]" = convolution_backward_32[0]
    getitem_589: "f32[128, 16, 3, 3]" = convolution_backward_32[1];  convolution_backward_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_481: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(slice_26, getitem_588);  slice_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_178: "f32[8, 128, 14, 14]" = torch.ops.aten.alias.default(relu_48);  relu_48 = None
    alias_179: "f32[8, 128, 14, 14]" = torch.ops.aten.alias.default(alias_178);  alias_178 = None
    le_32: "b8[8, 128, 14, 14]" = torch.ops.aten.le.Scalar(alias_179, 0);  alias_179 = None
    scalar_tensor_32: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_32: "f32[8, 128, 14, 14]" = torch.ops.aten.where.self(le_32, scalar_tensor_32, add_481);  le_32 = scalar_tensor_32 = add_481 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_736: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_153, 0);  squeeze_153 = None
    unsqueeze_737: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_736, 2);  unsqueeze_736 = None
    unsqueeze_738: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_737, 3);  unsqueeze_737 = None
    sum_68: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_32, [0, 2, 3])
    sub_217: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_738)
    mul_892: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_32, sub_217);  sub_217 = None
    sum_69: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_892, [0, 2, 3]);  mul_892 = None
    mul_893: "f32[128]" = torch.ops.aten.mul.Tensor(sum_68, 0.0006377551020408163)
    unsqueeze_739: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_893, 0);  mul_893 = None
    unsqueeze_740: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_739, 2);  unsqueeze_739 = None
    unsqueeze_741: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_740, 3);  unsqueeze_740 = None
    mul_894: "f32[128]" = torch.ops.aten.mul.Tensor(sum_69, 0.0006377551020408163)
    mul_895: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_154, squeeze_154)
    mul_896: "f32[128]" = torch.ops.aten.mul.Tensor(mul_894, mul_895);  mul_894 = mul_895 = None
    unsqueeze_742: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_896, 0);  mul_896 = None
    unsqueeze_743: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_742, 2);  unsqueeze_742 = None
    unsqueeze_744: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_743, 3);  unsqueeze_743 = None
    mul_897: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_154, primals_155);  primals_155 = None
    unsqueeze_745: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_897, 0);  mul_897 = None
    unsqueeze_746: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_745, 2);  unsqueeze_745 = None
    unsqueeze_747: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_746, 3);  unsqueeze_746 = None
    sub_218: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_738);  convolution_51 = unsqueeze_738 = None
    mul_898: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_218, unsqueeze_744);  sub_218 = unsqueeze_744 = None
    sub_219: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(where_32, mul_898);  where_32 = mul_898 = None
    sub_220: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(sub_219, unsqueeze_741);  sub_219 = unsqueeze_741 = None
    mul_899: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_220, unsqueeze_747);  sub_220 = unsqueeze_747 = None
    mul_900: "f32[128]" = torch.ops.aten.mul.Tensor(sum_69, squeeze_154);  sum_69 = squeeze_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_33 = torch.ops.aten.convolution_backward.default(mul_899, add_276, primals_154, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_899 = add_276 = primals_154 = None
    getitem_591: "f32[8, 128, 14, 14]" = convolution_backward_33[0]
    getitem_592: "f32[128, 16, 3, 3]" = convolution_backward_33[1];  convolution_backward_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_482: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(slice_25, getitem_591);  slice_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_181: "f32[8, 128, 14, 14]" = torch.ops.aten.alias.default(relu_47);  relu_47 = None
    alias_182: "f32[8, 128, 14, 14]" = torch.ops.aten.alias.default(alias_181);  alias_181 = None
    le_33: "b8[8, 128, 14, 14]" = torch.ops.aten.le.Scalar(alias_182, 0);  alias_182 = None
    scalar_tensor_33: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_33: "f32[8, 128, 14, 14]" = torch.ops.aten.where.self(le_33, scalar_tensor_33, add_482);  le_33 = scalar_tensor_33 = add_482 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_748: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_150, 0);  squeeze_150 = None
    unsqueeze_749: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_748, 2);  unsqueeze_748 = None
    unsqueeze_750: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_749, 3);  unsqueeze_749 = None
    sum_70: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_33, [0, 2, 3])
    sub_221: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_750)
    mul_901: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_33, sub_221);  sub_221 = None
    sum_71: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_901, [0, 2, 3]);  mul_901 = None
    mul_902: "f32[128]" = torch.ops.aten.mul.Tensor(sum_70, 0.0006377551020408163)
    unsqueeze_751: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_902, 0);  mul_902 = None
    unsqueeze_752: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_751, 2);  unsqueeze_751 = None
    unsqueeze_753: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_752, 3);  unsqueeze_752 = None
    mul_903: "f32[128]" = torch.ops.aten.mul.Tensor(sum_71, 0.0006377551020408163)
    mul_904: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_151, squeeze_151)
    mul_905: "f32[128]" = torch.ops.aten.mul.Tensor(mul_903, mul_904);  mul_903 = mul_904 = None
    unsqueeze_754: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_905, 0);  mul_905 = None
    unsqueeze_755: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_754, 2);  unsqueeze_754 = None
    unsqueeze_756: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_755, 3);  unsqueeze_755 = None
    mul_906: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_151, primals_152);  primals_152 = None
    unsqueeze_757: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_906, 0);  mul_906 = None
    unsqueeze_758: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_757, 2);  unsqueeze_757 = None
    unsqueeze_759: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_758, 3);  unsqueeze_758 = None
    sub_222: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_750);  convolution_50 = unsqueeze_750 = None
    mul_907: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_222, unsqueeze_756);  sub_222 = unsqueeze_756 = None
    sub_223: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(where_33, mul_907);  where_33 = mul_907 = None
    sub_224: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(sub_223, unsqueeze_753);  sub_223 = unsqueeze_753 = None
    mul_908: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_224, unsqueeze_759);  sub_224 = unsqueeze_759 = None
    mul_909: "f32[128]" = torch.ops.aten.mul.Tensor(sum_71, squeeze_151);  sum_71 = squeeze_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_34 = torch.ops.aten.convolution_backward.default(mul_908, getitem_286, primals_151, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_908 = getitem_286 = primals_151 = None
    getitem_594: "f32[8, 128, 14, 14]" = convolution_backward_34[0]
    getitem_595: "f32[128, 16, 3, 3]" = convolution_backward_34[1];  convolution_backward_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    cat_22: "f32[8, 512, 14, 14]" = torch.ops.aten.cat.default([getitem_594, getitem_591, getitem_588, slice_28], 1);  getitem_594 = getitem_591 = getitem_588 = slice_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    alias_184: "f32[8, 512, 14, 14]" = torch.ops.aten.alias.default(relu_46);  relu_46 = None
    alias_185: "f32[8, 512, 14, 14]" = torch.ops.aten.alias.default(alias_184);  alias_184 = None
    le_34: "b8[8, 512, 14, 14]" = torch.ops.aten.le.Scalar(alias_185, 0);  alias_185 = None
    scalar_tensor_34: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_34: "f32[8, 512, 14, 14]" = torch.ops.aten.where.self(le_34, scalar_tensor_34, cat_22);  le_34 = scalar_tensor_34 = cat_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    unsqueeze_760: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_147, 0);  squeeze_147 = None
    unsqueeze_761: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_760, 2);  unsqueeze_760 = None
    unsqueeze_762: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_761, 3);  unsqueeze_761 = None
    sum_72: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_34, [0, 2, 3])
    sub_225: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_49, unsqueeze_762)
    mul_910: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(where_34, sub_225);  sub_225 = None
    sum_73: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_910, [0, 2, 3]);  mul_910 = None
    mul_911: "f32[512]" = torch.ops.aten.mul.Tensor(sum_72, 0.0006377551020408163)
    unsqueeze_763: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_911, 0);  mul_911 = None
    unsqueeze_764: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_763, 2);  unsqueeze_763 = None
    unsqueeze_765: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_764, 3);  unsqueeze_764 = None
    mul_912: "f32[512]" = torch.ops.aten.mul.Tensor(sum_73, 0.0006377551020408163)
    mul_913: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_148, squeeze_148)
    mul_914: "f32[512]" = torch.ops.aten.mul.Tensor(mul_912, mul_913);  mul_912 = mul_913 = None
    unsqueeze_766: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_914, 0);  mul_914 = None
    unsqueeze_767: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_766, 2);  unsqueeze_766 = None
    unsqueeze_768: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_767, 3);  unsqueeze_767 = None
    mul_915: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_148, primals_149);  primals_149 = None
    unsqueeze_769: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_915, 0);  mul_915 = None
    unsqueeze_770: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_769, 2);  unsqueeze_769 = None
    unsqueeze_771: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_770, 3);  unsqueeze_770 = None
    sub_226: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_49, unsqueeze_762);  convolution_49 = unsqueeze_762 = None
    mul_916: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_226, unsqueeze_768);  sub_226 = unsqueeze_768 = None
    sub_227: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(where_34, mul_916);  where_34 = mul_916 = None
    sub_228: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(sub_227, unsqueeze_765);  sub_227 = unsqueeze_765 = None
    mul_917: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_228, unsqueeze_771);  sub_228 = unsqueeze_771 = None
    mul_918: "f32[512]" = torch.ops.aten.mul.Tensor(sum_73, squeeze_148);  sum_73 = squeeze_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_backward_35 = torch.ops.aten.convolution_backward.default(mul_917, relu_45, primals_148, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_917 = primals_148 = None
    getitem_597: "f32[8, 1024, 14, 14]" = convolution_backward_35[0]
    getitem_598: "f32[512, 1024, 1, 1]" = convolution_backward_35[1];  convolution_backward_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    add_483: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(where_30, getitem_597);  where_30 = getitem_597 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    alias_187: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(relu_45);  relu_45 = None
    alias_188: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(alias_187);  alias_187 = None
    le_35: "b8[8, 1024, 14, 14]" = torch.ops.aten.le.Scalar(alias_188, 0);  alias_188 = None
    scalar_tensor_35: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_35: "f32[8, 1024, 14, 14]" = torch.ops.aten.where.self(le_35, scalar_tensor_35, add_483);  le_35 = scalar_tensor_35 = add_483 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    unsqueeze_772: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_144, 0);  squeeze_144 = None
    unsqueeze_773: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_772, 2);  unsqueeze_772 = None
    unsqueeze_774: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_773, 3);  unsqueeze_773 = None
    sum_74: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_35, [0, 2, 3])
    sub_229: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_48, unsqueeze_774)
    mul_919: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_35, sub_229);  sub_229 = None
    sum_75: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_919, [0, 2, 3]);  mul_919 = None
    mul_920: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_74, 0.0006377551020408163)
    unsqueeze_775: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_920, 0);  mul_920 = None
    unsqueeze_776: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_775, 2);  unsqueeze_775 = None
    unsqueeze_777: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_776, 3);  unsqueeze_776 = None
    mul_921: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_75, 0.0006377551020408163)
    mul_922: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_145, squeeze_145)
    mul_923: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_921, mul_922);  mul_921 = mul_922 = None
    unsqueeze_778: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_923, 0);  mul_923 = None
    unsqueeze_779: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_778, 2);  unsqueeze_778 = None
    unsqueeze_780: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_779, 3);  unsqueeze_779 = None
    mul_924: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_145, primals_146);  primals_146 = None
    unsqueeze_781: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_924, 0);  mul_924 = None
    unsqueeze_782: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_781, 2);  unsqueeze_781 = None
    unsqueeze_783: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_782, 3);  unsqueeze_782 = None
    sub_230: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_48, unsqueeze_774);  convolution_48 = unsqueeze_774 = None
    mul_925: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_230, unsqueeze_780);  sub_230 = unsqueeze_780 = None
    sub_231: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_35, mul_925);  mul_925 = None
    sub_232: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_231, unsqueeze_777);  sub_231 = unsqueeze_777 = None
    mul_926: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_232, unsqueeze_783);  sub_232 = unsqueeze_783 = None
    mul_927: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_75, squeeze_145);  sum_75 = squeeze_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_backward_36 = torch.ops.aten.convolution_backward.default(mul_926, cat_8, primals_145, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_926 = cat_8 = primals_145 = None
    getitem_600: "f32[8, 512, 14, 14]" = convolution_backward_36[0]
    getitem_601: "f32[1024, 512, 1, 1]" = convolution_backward_36[1];  convolution_backward_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    slice_29: "f32[8, 128, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_600, 1, 0, 128)
    slice_30: "f32[8, 128, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_600, 1, 128, 256)
    slice_31: "f32[8, 128, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_600, 1, 256, 384)
    slice_32: "f32[8, 128, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_600, 1, 384, 512);  getitem_600 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_190: "f32[8, 128, 14, 14]" = torch.ops.aten.alias.default(relu_44);  relu_44 = None
    alias_191: "f32[8, 128, 14, 14]" = torch.ops.aten.alias.default(alias_190);  alias_190 = None
    le_36: "b8[8, 128, 14, 14]" = torch.ops.aten.le.Scalar(alias_191, 0);  alias_191 = None
    scalar_tensor_36: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_36: "f32[8, 128, 14, 14]" = torch.ops.aten.where.self(le_36, scalar_tensor_36, slice_31);  le_36 = scalar_tensor_36 = slice_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_784: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_141, 0);  squeeze_141 = None
    unsqueeze_785: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_784, 2);  unsqueeze_784 = None
    unsqueeze_786: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_785, 3);  unsqueeze_785 = None
    sum_76: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_36, [0, 2, 3])
    sub_233: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_47, unsqueeze_786)
    mul_928: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_36, sub_233);  sub_233 = None
    sum_77: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_928, [0, 2, 3]);  mul_928 = None
    mul_929: "f32[128]" = torch.ops.aten.mul.Tensor(sum_76, 0.0006377551020408163)
    unsqueeze_787: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_929, 0);  mul_929 = None
    unsqueeze_788: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_787, 2);  unsqueeze_787 = None
    unsqueeze_789: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_788, 3);  unsqueeze_788 = None
    mul_930: "f32[128]" = torch.ops.aten.mul.Tensor(sum_77, 0.0006377551020408163)
    mul_931: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_142, squeeze_142)
    mul_932: "f32[128]" = torch.ops.aten.mul.Tensor(mul_930, mul_931);  mul_930 = mul_931 = None
    unsqueeze_790: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_932, 0);  mul_932 = None
    unsqueeze_791: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_790, 2);  unsqueeze_790 = None
    unsqueeze_792: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_791, 3);  unsqueeze_791 = None
    mul_933: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_142, primals_143);  primals_143 = None
    unsqueeze_793: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_933, 0);  mul_933 = None
    unsqueeze_794: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_793, 2);  unsqueeze_793 = None
    unsqueeze_795: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_794, 3);  unsqueeze_794 = None
    sub_234: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_47, unsqueeze_786);  convolution_47 = unsqueeze_786 = None
    mul_934: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_234, unsqueeze_792);  sub_234 = unsqueeze_792 = None
    sub_235: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(where_36, mul_934);  where_36 = mul_934 = None
    sub_236: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(sub_235, unsqueeze_789);  sub_235 = unsqueeze_789 = None
    mul_935: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_236, unsqueeze_795);  sub_236 = unsqueeze_795 = None
    mul_936: "f32[128]" = torch.ops.aten.mul.Tensor(sum_77, squeeze_142);  sum_77 = squeeze_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_37 = torch.ops.aten.convolution_backward.default(mul_935, add_254, primals_142, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_935 = add_254 = primals_142 = None
    getitem_603: "f32[8, 128, 14, 14]" = convolution_backward_37[0]
    getitem_604: "f32[128, 16, 3, 3]" = convolution_backward_37[1];  convolution_backward_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_484: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(slice_30, getitem_603);  slice_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_193: "f32[8, 128, 14, 14]" = torch.ops.aten.alias.default(relu_43);  relu_43 = None
    alias_194: "f32[8, 128, 14, 14]" = torch.ops.aten.alias.default(alias_193);  alias_193 = None
    le_37: "b8[8, 128, 14, 14]" = torch.ops.aten.le.Scalar(alias_194, 0);  alias_194 = None
    scalar_tensor_37: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_37: "f32[8, 128, 14, 14]" = torch.ops.aten.where.self(le_37, scalar_tensor_37, add_484);  le_37 = scalar_tensor_37 = add_484 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_796: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_138, 0);  squeeze_138 = None
    unsqueeze_797: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_796, 2);  unsqueeze_796 = None
    unsqueeze_798: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_797, 3);  unsqueeze_797 = None
    sum_78: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_37, [0, 2, 3])
    sub_237: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_798)
    mul_937: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_37, sub_237);  sub_237 = None
    sum_79: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_937, [0, 2, 3]);  mul_937 = None
    mul_938: "f32[128]" = torch.ops.aten.mul.Tensor(sum_78, 0.0006377551020408163)
    unsqueeze_799: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_938, 0);  mul_938 = None
    unsqueeze_800: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_799, 2);  unsqueeze_799 = None
    unsqueeze_801: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_800, 3);  unsqueeze_800 = None
    mul_939: "f32[128]" = torch.ops.aten.mul.Tensor(sum_79, 0.0006377551020408163)
    mul_940: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_139, squeeze_139)
    mul_941: "f32[128]" = torch.ops.aten.mul.Tensor(mul_939, mul_940);  mul_939 = mul_940 = None
    unsqueeze_802: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_941, 0);  mul_941 = None
    unsqueeze_803: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_802, 2);  unsqueeze_802 = None
    unsqueeze_804: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_803, 3);  unsqueeze_803 = None
    mul_942: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_139, primals_140);  primals_140 = None
    unsqueeze_805: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_942, 0);  mul_942 = None
    unsqueeze_806: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_805, 2);  unsqueeze_805 = None
    unsqueeze_807: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_806, 3);  unsqueeze_806 = None
    sub_238: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_798);  convolution_46 = unsqueeze_798 = None
    mul_943: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_238, unsqueeze_804);  sub_238 = unsqueeze_804 = None
    sub_239: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(where_37, mul_943);  where_37 = mul_943 = None
    sub_240: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(sub_239, unsqueeze_801);  sub_239 = unsqueeze_801 = None
    mul_944: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_240, unsqueeze_807);  sub_240 = unsqueeze_807 = None
    mul_945: "f32[128]" = torch.ops.aten.mul.Tensor(sum_79, squeeze_139);  sum_79 = squeeze_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_38 = torch.ops.aten.convolution_backward.default(mul_944, add_248, primals_139, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_944 = add_248 = primals_139 = None
    getitem_606: "f32[8, 128, 14, 14]" = convolution_backward_38[0]
    getitem_607: "f32[128, 16, 3, 3]" = convolution_backward_38[1];  convolution_backward_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_485: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(slice_29, getitem_606);  slice_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_196: "f32[8, 128, 14, 14]" = torch.ops.aten.alias.default(relu_42);  relu_42 = None
    alias_197: "f32[8, 128, 14, 14]" = torch.ops.aten.alias.default(alias_196);  alias_196 = None
    le_38: "b8[8, 128, 14, 14]" = torch.ops.aten.le.Scalar(alias_197, 0);  alias_197 = None
    scalar_tensor_38: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_38: "f32[8, 128, 14, 14]" = torch.ops.aten.where.self(le_38, scalar_tensor_38, add_485);  le_38 = scalar_tensor_38 = add_485 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_808: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_135, 0);  squeeze_135 = None
    unsqueeze_809: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_808, 2);  unsqueeze_808 = None
    unsqueeze_810: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_809, 3);  unsqueeze_809 = None
    sum_80: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_38, [0, 2, 3])
    sub_241: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_810)
    mul_946: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_38, sub_241);  sub_241 = None
    sum_81: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_946, [0, 2, 3]);  mul_946 = None
    mul_947: "f32[128]" = torch.ops.aten.mul.Tensor(sum_80, 0.0006377551020408163)
    unsqueeze_811: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_947, 0);  mul_947 = None
    unsqueeze_812: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_811, 2);  unsqueeze_811 = None
    unsqueeze_813: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_812, 3);  unsqueeze_812 = None
    mul_948: "f32[128]" = torch.ops.aten.mul.Tensor(sum_81, 0.0006377551020408163)
    mul_949: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_136, squeeze_136)
    mul_950: "f32[128]" = torch.ops.aten.mul.Tensor(mul_948, mul_949);  mul_948 = mul_949 = None
    unsqueeze_814: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_950, 0);  mul_950 = None
    unsqueeze_815: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_814, 2);  unsqueeze_814 = None
    unsqueeze_816: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_815, 3);  unsqueeze_815 = None
    mul_951: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_136, primals_137);  primals_137 = None
    unsqueeze_817: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_951, 0);  mul_951 = None
    unsqueeze_818: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_817, 2);  unsqueeze_817 = None
    unsqueeze_819: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_818, 3);  unsqueeze_818 = None
    sub_242: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_810);  convolution_45 = unsqueeze_810 = None
    mul_952: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_242, unsqueeze_816);  sub_242 = unsqueeze_816 = None
    sub_243: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(where_38, mul_952);  where_38 = mul_952 = None
    sub_244: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(sub_243, unsqueeze_813);  sub_243 = unsqueeze_813 = None
    mul_953: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_244, unsqueeze_819);  sub_244 = unsqueeze_819 = None
    mul_954: "f32[128]" = torch.ops.aten.mul.Tensor(sum_81, squeeze_136);  sum_81 = squeeze_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_39 = torch.ops.aten.convolution_backward.default(mul_953, getitem_256, primals_136, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_953 = getitem_256 = primals_136 = None
    getitem_609: "f32[8, 128, 14, 14]" = convolution_backward_39[0]
    getitem_610: "f32[128, 16, 3, 3]" = convolution_backward_39[1];  convolution_backward_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    cat_23: "f32[8, 512, 14, 14]" = torch.ops.aten.cat.default([getitem_609, getitem_606, getitem_603, slice_32], 1);  getitem_609 = getitem_606 = getitem_603 = slice_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    alias_199: "f32[8, 512, 14, 14]" = torch.ops.aten.alias.default(relu_41);  relu_41 = None
    alias_200: "f32[8, 512, 14, 14]" = torch.ops.aten.alias.default(alias_199);  alias_199 = None
    le_39: "b8[8, 512, 14, 14]" = torch.ops.aten.le.Scalar(alias_200, 0);  alias_200 = None
    scalar_tensor_39: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_39: "f32[8, 512, 14, 14]" = torch.ops.aten.where.self(le_39, scalar_tensor_39, cat_23);  le_39 = scalar_tensor_39 = cat_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    unsqueeze_820: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_132, 0);  squeeze_132 = None
    unsqueeze_821: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_820, 2);  unsqueeze_820 = None
    unsqueeze_822: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_821, 3);  unsqueeze_821 = None
    sum_82: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_39, [0, 2, 3])
    sub_245: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_44, unsqueeze_822)
    mul_955: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(where_39, sub_245);  sub_245 = None
    sum_83: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_955, [0, 2, 3]);  mul_955 = None
    mul_956: "f32[512]" = torch.ops.aten.mul.Tensor(sum_82, 0.0006377551020408163)
    unsqueeze_823: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_956, 0);  mul_956 = None
    unsqueeze_824: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_823, 2);  unsqueeze_823 = None
    unsqueeze_825: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_824, 3);  unsqueeze_824 = None
    mul_957: "f32[512]" = torch.ops.aten.mul.Tensor(sum_83, 0.0006377551020408163)
    mul_958: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_133, squeeze_133)
    mul_959: "f32[512]" = torch.ops.aten.mul.Tensor(mul_957, mul_958);  mul_957 = mul_958 = None
    unsqueeze_826: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_959, 0);  mul_959 = None
    unsqueeze_827: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_826, 2);  unsqueeze_826 = None
    unsqueeze_828: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_827, 3);  unsqueeze_827 = None
    mul_960: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_133, primals_134);  primals_134 = None
    unsqueeze_829: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_960, 0);  mul_960 = None
    unsqueeze_830: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_829, 2);  unsqueeze_829 = None
    unsqueeze_831: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_830, 3);  unsqueeze_830 = None
    sub_246: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_44, unsqueeze_822);  convolution_44 = unsqueeze_822 = None
    mul_961: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_246, unsqueeze_828);  sub_246 = unsqueeze_828 = None
    sub_247: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(where_39, mul_961);  where_39 = mul_961 = None
    sub_248: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(sub_247, unsqueeze_825);  sub_247 = unsqueeze_825 = None
    mul_962: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_248, unsqueeze_831);  sub_248 = unsqueeze_831 = None
    mul_963: "f32[512]" = torch.ops.aten.mul.Tensor(sum_83, squeeze_133);  sum_83 = squeeze_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_backward_40 = torch.ops.aten.convolution_backward.default(mul_962, relu_40, primals_133, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_962 = primals_133 = None
    getitem_612: "f32[8, 1024, 14, 14]" = convolution_backward_40[0]
    getitem_613: "f32[512, 1024, 1, 1]" = convolution_backward_40[1];  convolution_backward_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    add_486: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(where_35, getitem_612);  where_35 = getitem_612 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    alias_202: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(relu_40);  relu_40 = None
    alias_203: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(alias_202);  alias_202 = None
    le_40: "b8[8, 1024, 14, 14]" = torch.ops.aten.le.Scalar(alias_203, 0);  alias_203 = None
    scalar_tensor_40: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_40: "f32[8, 1024, 14, 14]" = torch.ops.aten.where.self(le_40, scalar_tensor_40, add_486);  le_40 = scalar_tensor_40 = add_486 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:111, code: shortcut = self.downsample(x)
    unsqueeze_832: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_129, 0);  squeeze_129 = None
    unsqueeze_833: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_832, 2);  unsqueeze_832 = None
    unsqueeze_834: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_833, 3);  unsqueeze_833 = None
    sum_84: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_40, [0, 2, 3])
    sub_249: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_43, unsqueeze_834)
    mul_964: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_40, sub_249);  sub_249 = None
    sum_85: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_964, [0, 2, 3]);  mul_964 = None
    mul_965: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_84, 0.0006377551020408163)
    unsqueeze_835: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_965, 0);  mul_965 = None
    unsqueeze_836: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_835, 2);  unsqueeze_835 = None
    unsqueeze_837: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_836, 3);  unsqueeze_836 = None
    mul_966: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_85, 0.0006377551020408163)
    mul_967: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_130, squeeze_130)
    mul_968: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_966, mul_967);  mul_966 = mul_967 = None
    unsqueeze_838: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_968, 0);  mul_968 = None
    unsqueeze_839: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_838, 2);  unsqueeze_838 = None
    unsqueeze_840: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_839, 3);  unsqueeze_839 = None
    mul_969: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_130, primals_131);  primals_131 = None
    unsqueeze_841: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_969, 0);  mul_969 = None
    unsqueeze_842: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_841, 2);  unsqueeze_841 = None
    unsqueeze_843: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_842, 3);  unsqueeze_842 = None
    sub_250: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_43, unsqueeze_834);  convolution_43 = unsqueeze_834 = None
    mul_970: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_250, unsqueeze_840);  sub_250 = unsqueeze_840 = None
    sub_251: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_40, mul_970);  mul_970 = None
    sub_252: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_251, unsqueeze_837);  sub_251 = unsqueeze_837 = None
    mul_971: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_252, unsqueeze_843);  sub_252 = unsqueeze_843 = None
    mul_972: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_85, squeeze_130);  sum_85 = squeeze_130 = None
    convolution_backward_41 = torch.ops.aten.convolution_backward.default(mul_971, relu_35, primals_130, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_971 = primals_130 = None
    getitem_615: "f32[8, 512, 28, 28]" = convolution_backward_41[0]
    getitem_616: "f32[1024, 512, 1, 1]" = convolution_backward_41[1];  convolution_backward_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    unsqueeze_844: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_126, 0);  squeeze_126 = None
    unsqueeze_845: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_844, 2);  unsqueeze_844 = None
    unsqueeze_846: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_845, 3);  unsqueeze_845 = None
    sum_86: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_40, [0, 2, 3])
    sub_253: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_42, unsqueeze_846)
    mul_973: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_40, sub_253);  sub_253 = None
    sum_87: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_973, [0, 2, 3]);  mul_973 = None
    mul_974: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_86, 0.0006377551020408163)
    unsqueeze_847: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_974, 0);  mul_974 = None
    unsqueeze_848: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_847, 2);  unsqueeze_847 = None
    unsqueeze_849: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_848, 3);  unsqueeze_848 = None
    mul_975: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_87, 0.0006377551020408163)
    mul_976: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_127, squeeze_127)
    mul_977: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_975, mul_976);  mul_975 = mul_976 = None
    unsqueeze_850: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_977, 0);  mul_977 = None
    unsqueeze_851: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_850, 2);  unsqueeze_850 = None
    unsqueeze_852: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_851, 3);  unsqueeze_851 = None
    mul_978: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_127, primals_128);  primals_128 = None
    unsqueeze_853: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_978, 0);  mul_978 = None
    unsqueeze_854: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_853, 2);  unsqueeze_853 = None
    unsqueeze_855: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_854, 3);  unsqueeze_854 = None
    sub_254: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_42, unsqueeze_846);  convolution_42 = unsqueeze_846 = None
    mul_979: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_254, unsqueeze_852);  sub_254 = unsqueeze_852 = None
    sub_255: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_40, mul_979);  where_40 = mul_979 = None
    sub_256: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_255, unsqueeze_849);  sub_255 = unsqueeze_849 = None
    mul_980: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_256, unsqueeze_855);  sub_256 = unsqueeze_855 = None
    mul_981: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_87, squeeze_127);  sum_87 = squeeze_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_backward_42 = torch.ops.aten.convolution_backward.default(mul_980, cat_7, primals_127, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_980 = cat_7 = primals_127 = None
    getitem_618: "f32[8, 512, 14, 14]" = convolution_backward_42[0]
    getitem_619: "f32[1024, 512, 1, 1]" = convolution_backward_42[1];  convolution_backward_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    slice_33: "f32[8, 128, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_618, 1, 0, 128)
    slice_34: "f32[8, 128, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_618, 1, 128, 256)
    slice_35: "f32[8, 128, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_618, 1, 256, 384)
    slice_36: "f32[8, 128, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_618, 1, 384, 512);  getitem_618 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:99, code: spo.append(self.pool(spx[-1]))
    avg_pool2d_backward_1: "f32[8, 128, 28, 28]" = torch.ops.aten.avg_pool2d_backward.default(slice_36, getitem_245, [3, 3], [2, 2], [1, 1], False, True, None);  slice_36 = getitem_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_205: "f32[8, 128, 14, 14]" = torch.ops.aten.alias.default(relu_39);  relu_39 = None
    alias_206: "f32[8, 128, 14, 14]" = torch.ops.aten.alias.default(alias_205);  alias_205 = None
    le_41: "b8[8, 128, 14, 14]" = torch.ops.aten.le.Scalar(alias_206, 0);  alias_206 = None
    scalar_tensor_41: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_41: "f32[8, 128, 14, 14]" = torch.ops.aten.where.self(le_41, scalar_tensor_41, slice_35);  le_41 = scalar_tensor_41 = slice_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_856: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_123, 0);  squeeze_123 = None
    unsqueeze_857: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_856, 2);  unsqueeze_856 = None
    unsqueeze_858: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_857, 3);  unsqueeze_857 = None
    sum_88: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_41, [0, 2, 3])
    sub_257: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_858)
    mul_982: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_41, sub_257);  sub_257 = None
    sum_89: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_982, [0, 2, 3]);  mul_982 = None
    mul_983: "f32[128]" = torch.ops.aten.mul.Tensor(sum_88, 0.0006377551020408163)
    unsqueeze_859: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_983, 0);  mul_983 = None
    unsqueeze_860: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_859, 2);  unsqueeze_859 = None
    unsqueeze_861: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_860, 3);  unsqueeze_860 = None
    mul_984: "f32[128]" = torch.ops.aten.mul.Tensor(sum_89, 0.0006377551020408163)
    mul_985: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_124, squeeze_124)
    mul_986: "f32[128]" = torch.ops.aten.mul.Tensor(mul_984, mul_985);  mul_984 = mul_985 = None
    unsqueeze_862: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_986, 0);  mul_986 = None
    unsqueeze_863: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_862, 2);  unsqueeze_862 = None
    unsqueeze_864: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_863, 3);  unsqueeze_863 = None
    mul_987: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_124, primals_125);  primals_125 = None
    unsqueeze_865: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_987, 0);  mul_987 = None
    unsqueeze_866: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_865, 2);  unsqueeze_865 = None
    unsqueeze_867: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_866, 3);  unsqueeze_866 = None
    sub_258: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_858);  convolution_41 = unsqueeze_858 = None
    mul_988: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_258, unsqueeze_864);  sub_258 = unsqueeze_864 = None
    sub_259: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(where_41, mul_988);  where_41 = mul_988 = None
    sub_260: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(sub_259, unsqueeze_861);  sub_259 = unsqueeze_861 = None
    mul_989: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_260, unsqueeze_867);  sub_260 = unsqueeze_867 = None
    mul_990: "f32[128]" = torch.ops.aten.mul.Tensor(sum_89, squeeze_124);  sum_89 = squeeze_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_43 = torch.ops.aten.convolution_backward.default(mul_989, getitem_238, primals_124, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_989 = getitem_238 = primals_124 = None
    getitem_621: "f32[8, 128, 28, 28]" = convolution_backward_43[0]
    getitem_622: "f32[128, 16, 3, 3]" = convolution_backward_43[1];  convolution_backward_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_208: "f32[8, 128, 14, 14]" = torch.ops.aten.alias.default(relu_38);  relu_38 = None
    alias_209: "f32[8, 128, 14, 14]" = torch.ops.aten.alias.default(alias_208);  alias_208 = None
    le_42: "b8[8, 128, 14, 14]" = torch.ops.aten.le.Scalar(alias_209, 0);  alias_209 = None
    scalar_tensor_42: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_42: "f32[8, 128, 14, 14]" = torch.ops.aten.where.self(le_42, scalar_tensor_42, slice_34);  le_42 = scalar_tensor_42 = slice_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_868: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_120, 0);  squeeze_120 = None
    unsqueeze_869: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_868, 2);  unsqueeze_868 = None
    unsqueeze_870: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_869, 3);  unsqueeze_869 = None
    sum_90: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_42, [0, 2, 3])
    sub_261: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_870)
    mul_991: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_42, sub_261);  sub_261 = None
    sum_91: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_991, [0, 2, 3]);  mul_991 = None
    mul_992: "f32[128]" = torch.ops.aten.mul.Tensor(sum_90, 0.0006377551020408163)
    unsqueeze_871: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_992, 0);  mul_992 = None
    unsqueeze_872: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_871, 2);  unsqueeze_871 = None
    unsqueeze_873: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_872, 3);  unsqueeze_872 = None
    mul_993: "f32[128]" = torch.ops.aten.mul.Tensor(sum_91, 0.0006377551020408163)
    mul_994: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_121, squeeze_121)
    mul_995: "f32[128]" = torch.ops.aten.mul.Tensor(mul_993, mul_994);  mul_993 = mul_994 = None
    unsqueeze_874: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_995, 0);  mul_995 = None
    unsqueeze_875: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_874, 2);  unsqueeze_874 = None
    unsqueeze_876: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_875, 3);  unsqueeze_875 = None
    mul_996: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_121, primals_122);  primals_122 = None
    unsqueeze_877: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_996, 0);  mul_996 = None
    unsqueeze_878: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_877, 2);  unsqueeze_877 = None
    unsqueeze_879: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_878, 3);  unsqueeze_878 = None
    sub_262: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_870);  convolution_40 = unsqueeze_870 = None
    mul_997: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_262, unsqueeze_876);  sub_262 = unsqueeze_876 = None
    sub_263: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(where_42, mul_997);  where_42 = mul_997 = None
    sub_264: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(sub_263, unsqueeze_873);  sub_263 = unsqueeze_873 = None
    mul_998: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_264, unsqueeze_879);  sub_264 = unsqueeze_879 = None
    mul_999: "f32[128]" = torch.ops.aten.mul.Tensor(sum_91, squeeze_121);  sum_91 = squeeze_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_44 = torch.ops.aten.convolution_backward.default(mul_998, getitem_231, primals_121, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_998 = getitem_231 = primals_121 = None
    getitem_624: "f32[8, 128, 28, 28]" = convolution_backward_44[0]
    getitem_625: "f32[128, 16, 3, 3]" = convolution_backward_44[1];  convolution_backward_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_211: "f32[8, 128, 14, 14]" = torch.ops.aten.alias.default(relu_37);  relu_37 = None
    alias_212: "f32[8, 128, 14, 14]" = torch.ops.aten.alias.default(alias_211);  alias_211 = None
    le_43: "b8[8, 128, 14, 14]" = torch.ops.aten.le.Scalar(alias_212, 0);  alias_212 = None
    scalar_tensor_43: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_43: "f32[8, 128, 14, 14]" = torch.ops.aten.where.self(le_43, scalar_tensor_43, slice_33);  le_43 = scalar_tensor_43 = slice_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_880: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_117, 0);  squeeze_117 = None
    unsqueeze_881: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_880, 2);  unsqueeze_880 = None
    unsqueeze_882: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_881, 3);  unsqueeze_881 = None
    sum_92: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_43, [0, 2, 3])
    sub_265: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_39, unsqueeze_882)
    mul_1000: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(where_43, sub_265);  sub_265 = None
    sum_93: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1000, [0, 2, 3]);  mul_1000 = None
    mul_1001: "f32[128]" = torch.ops.aten.mul.Tensor(sum_92, 0.0006377551020408163)
    unsqueeze_883: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1001, 0);  mul_1001 = None
    unsqueeze_884: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_883, 2);  unsqueeze_883 = None
    unsqueeze_885: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_884, 3);  unsqueeze_884 = None
    mul_1002: "f32[128]" = torch.ops.aten.mul.Tensor(sum_93, 0.0006377551020408163)
    mul_1003: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_118, squeeze_118)
    mul_1004: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1002, mul_1003);  mul_1002 = mul_1003 = None
    unsqueeze_886: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1004, 0);  mul_1004 = None
    unsqueeze_887: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_886, 2);  unsqueeze_886 = None
    unsqueeze_888: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_887, 3);  unsqueeze_887 = None
    mul_1005: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_118, primals_119);  primals_119 = None
    unsqueeze_889: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1005, 0);  mul_1005 = None
    unsqueeze_890: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_889, 2);  unsqueeze_889 = None
    unsqueeze_891: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_890, 3);  unsqueeze_890 = None
    sub_266: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_39, unsqueeze_882);  convolution_39 = unsqueeze_882 = None
    mul_1006: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_266, unsqueeze_888);  sub_266 = unsqueeze_888 = None
    sub_267: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(where_43, mul_1006);  where_43 = mul_1006 = None
    sub_268: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(sub_267, unsqueeze_885);  sub_267 = unsqueeze_885 = None
    mul_1007: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_268, unsqueeze_891);  sub_268 = unsqueeze_891 = None
    mul_1008: "f32[128]" = torch.ops.aten.mul.Tensor(sum_93, squeeze_118);  sum_93 = squeeze_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_45 = torch.ops.aten.convolution_backward.default(mul_1007, getitem_224, primals_118, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_1007 = getitem_224 = primals_118 = None
    getitem_627: "f32[8, 128, 28, 28]" = convolution_backward_45[0]
    getitem_628: "f32[128, 16, 3, 3]" = convolution_backward_45[1];  convolution_backward_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    cat_24: "f32[8, 512, 28, 28]" = torch.ops.aten.cat.default([getitem_627, getitem_624, getitem_621, avg_pool2d_backward_1], 1);  getitem_627 = getitem_624 = getitem_621 = avg_pool2d_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    alias_214: "f32[8, 512, 28, 28]" = torch.ops.aten.alias.default(relu_36);  relu_36 = None
    alias_215: "f32[8, 512, 28, 28]" = torch.ops.aten.alias.default(alias_214);  alias_214 = None
    le_44: "b8[8, 512, 28, 28]" = torch.ops.aten.le.Scalar(alias_215, 0);  alias_215 = None
    scalar_tensor_44: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_44: "f32[8, 512, 28, 28]" = torch.ops.aten.where.self(le_44, scalar_tensor_44, cat_24);  le_44 = scalar_tensor_44 = cat_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    unsqueeze_892: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_114, 0);  squeeze_114 = None
    unsqueeze_893: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_892, 2);  unsqueeze_892 = None
    unsqueeze_894: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_893, 3);  unsqueeze_893 = None
    sum_94: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_44, [0, 2, 3])
    sub_269: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_38, unsqueeze_894)
    mul_1009: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(where_44, sub_269);  sub_269 = None
    sum_95: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1009, [0, 2, 3]);  mul_1009 = None
    mul_1010: "f32[512]" = torch.ops.aten.mul.Tensor(sum_94, 0.00015943877551020407)
    unsqueeze_895: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1010, 0);  mul_1010 = None
    unsqueeze_896: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_895, 2);  unsqueeze_895 = None
    unsqueeze_897: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_896, 3);  unsqueeze_896 = None
    mul_1011: "f32[512]" = torch.ops.aten.mul.Tensor(sum_95, 0.00015943877551020407)
    mul_1012: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_115, squeeze_115)
    mul_1013: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1011, mul_1012);  mul_1011 = mul_1012 = None
    unsqueeze_898: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1013, 0);  mul_1013 = None
    unsqueeze_899: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_898, 2);  unsqueeze_898 = None
    unsqueeze_900: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_899, 3);  unsqueeze_899 = None
    mul_1014: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_115, primals_116);  primals_116 = None
    unsqueeze_901: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1014, 0);  mul_1014 = None
    unsqueeze_902: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_901, 2);  unsqueeze_901 = None
    unsqueeze_903: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_902, 3);  unsqueeze_902 = None
    sub_270: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_38, unsqueeze_894);  convolution_38 = unsqueeze_894 = None
    mul_1015: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_270, unsqueeze_900);  sub_270 = unsqueeze_900 = None
    sub_271: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(where_44, mul_1015);  where_44 = mul_1015 = None
    sub_272: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(sub_271, unsqueeze_897);  sub_271 = unsqueeze_897 = None
    mul_1016: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_272, unsqueeze_903);  sub_272 = unsqueeze_903 = None
    mul_1017: "f32[512]" = torch.ops.aten.mul.Tensor(sum_95, squeeze_115);  sum_95 = squeeze_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_backward_46 = torch.ops.aten.convolution_backward.default(mul_1016, relu_35, primals_115, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1016 = primals_115 = None
    getitem_630: "f32[8, 512, 28, 28]" = convolution_backward_46[0]
    getitem_631: "f32[512, 512, 1, 1]" = convolution_backward_46[1];  convolution_backward_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    add_487: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(getitem_615, getitem_630);  getitem_615 = getitem_630 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    alias_217: "f32[8, 512, 28, 28]" = torch.ops.aten.alias.default(relu_35);  relu_35 = None
    alias_218: "f32[8, 512, 28, 28]" = torch.ops.aten.alias.default(alias_217);  alias_217 = None
    le_45: "b8[8, 512, 28, 28]" = torch.ops.aten.le.Scalar(alias_218, 0);  alias_218 = None
    scalar_tensor_45: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_45: "f32[8, 512, 28, 28]" = torch.ops.aten.where.self(le_45, scalar_tensor_45, add_487);  le_45 = scalar_tensor_45 = add_487 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    unsqueeze_904: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_111, 0);  squeeze_111 = None
    unsqueeze_905: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_904, 2);  unsqueeze_904 = None
    unsqueeze_906: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_905, 3);  unsqueeze_905 = None
    sum_96: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_45, [0, 2, 3])
    sub_273: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_37, unsqueeze_906)
    mul_1018: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(where_45, sub_273);  sub_273 = None
    sum_97: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1018, [0, 2, 3]);  mul_1018 = None
    mul_1019: "f32[512]" = torch.ops.aten.mul.Tensor(sum_96, 0.00015943877551020407)
    unsqueeze_907: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1019, 0);  mul_1019 = None
    unsqueeze_908: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_907, 2);  unsqueeze_907 = None
    unsqueeze_909: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_908, 3);  unsqueeze_908 = None
    mul_1020: "f32[512]" = torch.ops.aten.mul.Tensor(sum_97, 0.00015943877551020407)
    mul_1021: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_112, squeeze_112)
    mul_1022: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1020, mul_1021);  mul_1020 = mul_1021 = None
    unsqueeze_910: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1022, 0);  mul_1022 = None
    unsqueeze_911: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_910, 2);  unsqueeze_910 = None
    unsqueeze_912: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_911, 3);  unsqueeze_911 = None
    mul_1023: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_112, primals_113);  primals_113 = None
    unsqueeze_913: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1023, 0);  mul_1023 = None
    unsqueeze_914: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_913, 2);  unsqueeze_913 = None
    unsqueeze_915: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_914, 3);  unsqueeze_914 = None
    sub_274: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_37, unsqueeze_906);  convolution_37 = unsqueeze_906 = None
    mul_1024: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_274, unsqueeze_912);  sub_274 = unsqueeze_912 = None
    sub_275: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(where_45, mul_1024);  mul_1024 = None
    sub_276: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(sub_275, unsqueeze_909);  sub_275 = unsqueeze_909 = None
    mul_1025: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_276, unsqueeze_915);  sub_276 = unsqueeze_915 = None
    mul_1026: "f32[512]" = torch.ops.aten.mul.Tensor(sum_97, squeeze_112);  sum_97 = squeeze_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_backward_47 = torch.ops.aten.convolution_backward.default(mul_1025, cat_6, primals_112, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1025 = cat_6 = primals_112 = None
    getitem_633: "f32[8, 256, 28, 28]" = convolution_backward_47[0]
    getitem_634: "f32[512, 256, 1, 1]" = convolution_backward_47[1];  convolution_backward_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    slice_37: "f32[8, 64, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_633, 1, 0, 64)
    slice_38: "f32[8, 64, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_633, 1, 64, 128)
    slice_39: "f32[8, 64, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_633, 1, 128, 192)
    slice_40: "f32[8, 64, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_633, 1, 192, 256);  getitem_633 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_220: "f32[8, 64, 28, 28]" = torch.ops.aten.alias.default(relu_34);  relu_34 = None
    alias_221: "f32[8, 64, 28, 28]" = torch.ops.aten.alias.default(alias_220);  alias_220 = None
    le_46: "b8[8, 64, 28, 28]" = torch.ops.aten.le.Scalar(alias_221, 0);  alias_221 = None
    scalar_tensor_46: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_46: "f32[8, 64, 28, 28]" = torch.ops.aten.where.self(le_46, scalar_tensor_46, slice_39);  le_46 = scalar_tensor_46 = slice_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_916: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_108, 0);  squeeze_108 = None
    unsqueeze_917: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_916, 2);  unsqueeze_916 = None
    unsqueeze_918: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_917, 3);  unsqueeze_917 = None
    sum_98: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_46, [0, 2, 3])
    sub_277: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_918)
    mul_1027: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(where_46, sub_277);  sub_277 = None
    sum_99: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1027, [0, 2, 3]);  mul_1027 = None
    mul_1028: "f32[64]" = torch.ops.aten.mul.Tensor(sum_98, 0.00015943877551020407)
    unsqueeze_919: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1028, 0);  mul_1028 = None
    unsqueeze_920: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_919, 2);  unsqueeze_919 = None
    unsqueeze_921: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_920, 3);  unsqueeze_920 = None
    mul_1029: "f32[64]" = torch.ops.aten.mul.Tensor(sum_99, 0.00015943877551020407)
    mul_1030: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_109, squeeze_109)
    mul_1031: "f32[64]" = torch.ops.aten.mul.Tensor(mul_1029, mul_1030);  mul_1029 = mul_1030 = None
    unsqueeze_922: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1031, 0);  mul_1031 = None
    unsqueeze_923: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_922, 2);  unsqueeze_922 = None
    unsqueeze_924: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_923, 3);  unsqueeze_923 = None
    mul_1032: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_109, primals_110);  primals_110 = None
    unsqueeze_925: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1032, 0);  mul_1032 = None
    unsqueeze_926: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_925, 2);  unsqueeze_925 = None
    unsqueeze_927: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_926, 3);  unsqueeze_926 = None
    sub_278: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_918);  convolution_36 = unsqueeze_918 = None
    mul_1033: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(sub_278, unsqueeze_924);  sub_278 = unsqueeze_924 = None
    sub_279: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(where_46, mul_1033);  where_46 = mul_1033 = None
    sub_280: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(sub_279, unsqueeze_921);  sub_279 = unsqueeze_921 = None
    mul_1034: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(sub_280, unsqueeze_927);  sub_280 = unsqueeze_927 = None
    mul_1035: "f32[64]" = torch.ops.aten.mul.Tensor(sum_99, squeeze_109);  sum_99 = squeeze_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_48 = torch.ops.aten.convolution_backward.default(mul_1034, add_195, primals_109, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_1034 = add_195 = primals_109 = None
    getitem_636: "f32[8, 64, 28, 28]" = convolution_backward_48[0]
    getitem_637: "f32[64, 8, 3, 3]" = convolution_backward_48[1];  convolution_backward_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_488: "f32[8, 64, 28, 28]" = torch.ops.aten.add.Tensor(slice_38, getitem_636);  slice_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_223: "f32[8, 64, 28, 28]" = torch.ops.aten.alias.default(relu_33);  relu_33 = None
    alias_224: "f32[8, 64, 28, 28]" = torch.ops.aten.alias.default(alias_223);  alias_223 = None
    le_47: "b8[8, 64, 28, 28]" = torch.ops.aten.le.Scalar(alias_224, 0);  alias_224 = None
    scalar_tensor_47: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_47: "f32[8, 64, 28, 28]" = torch.ops.aten.where.self(le_47, scalar_tensor_47, add_488);  le_47 = scalar_tensor_47 = add_488 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_928: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_105, 0);  squeeze_105 = None
    unsqueeze_929: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_928, 2);  unsqueeze_928 = None
    unsqueeze_930: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_929, 3);  unsqueeze_929 = None
    sum_100: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_47, [0, 2, 3])
    sub_281: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_930)
    mul_1036: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(where_47, sub_281);  sub_281 = None
    sum_101: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1036, [0, 2, 3]);  mul_1036 = None
    mul_1037: "f32[64]" = torch.ops.aten.mul.Tensor(sum_100, 0.00015943877551020407)
    unsqueeze_931: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1037, 0);  mul_1037 = None
    unsqueeze_932: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_931, 2);  unsqueeze_931 = None
    unsqueeze_933: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_932, 3);  unsqueeze_932 = None
    mul_1038: "f32[64]" = torch.ops.aten.mul.Tensor(sum_101, 0.00015943877551020407)
    mul_1039: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_106, squeeze_106)
    mul_1040: "f32[64]" = torch.ops.aten.mul.Tensor(mul_1038, mul_1039);  mul_1038 = mul_1039 = None
    unsqueeze_934: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1040, 0);  mul_1040 = None
    unsqueeze_935: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_934, 2);  unsqueeze_934 = None
    unsqueeze_936: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_935, 3);  unsqueeze_935 = None
    mul_1041: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_106, primals_107);  primals_107 = None
    unsqueeze_937: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1041, 0);  mul_1041 = None
    unsqueeze_938: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_937, 2);  unsqueeze_937 = None
    unsqueeze_939: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_938, 3);  unsqueeze_938 = None
    sub_282: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_930);  convolution_35 = unsqueeze_930 = None
    mul_1042: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(sub_282, unsqueeze_936);  sub_282 = unsqueeze_936 = None
    sub_283: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(where_47, mul_1042);  where_47 = mul_1042 = None
    sub_284: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(sub_283, unsqueeze_933);  sub_283 = unsqueeze_933 = None
    mul_1043: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(sub_284, unsqueeze_939);  sub_284 = unsqueeze_939 = None
    mul_1044: "f32[64]" = torch.ops.aten.mul.Tensor(sum_101, squeeze_106);  sum_101 = squeeze_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_49 = torch.ops.aten.convolution_backward.default(mul_1043, add_189, primals_106, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_1043 = add_189 = primals_106 = None
    getitem_639: "f32[8, 64, 28, 28]" = convolution_backward_49[0]
    getitem_640: "f32[64, 8, 3, 3]" = convolution_backward_49[1];  convolution_backward_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_489: "f32[8, 64, 28, 28]" = torch.ops.aten.add.Tensor(slice_37, getitem_639);  slice_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_226: "f32[8, 64, 28, 28]" = torch.ops.aten.alias.default(relu_32);  relu_32 = None
    alias_227: "f32[8, 64, 28, 28]" = torch.ops.aten.alias.default(alias_226);  alias_226 = None
    le_48: "b8[8, 64, 28, 28]" = torch.ops.aten.le.Scalar(alias_227, 0);  alias_227 = None
    scalar_tensor_48: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_48: "f32[8, 64, 28, 28]" = torch.ops.aten.where.self(le_48, scalar_tensor_48, add_489);  le_48 = scalar_tensor_48 = add_489 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_940: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_102, 0);  squeeze_102 = None
    unsqueeze_941: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_940, 2);  unsqueeze_940 = None
    unsqueeze_942: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_941, 3);  unsqueeze_941 = None
    sum_102: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_48, [0, 2, 3])
    sub_285: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_942)
    mul_1045: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(where_48, sub_285);  sub_285 = None
    sum_103: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1045, [0, 2, 3]);  mul_1045 = None
    mul_1046: "f32[64]" = torch.ops.aten.mul.Tensor(sum_102, 0.00015943877551020407)
    unsqueeze_943: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1046, 0);  mul_1046 = None
    unsqueeze_944: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_943, 2);  unsqueeze_943 = None
    unsqueeze_945: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_944, 3);  unsqueeze_944 = None
    mul_1047: "f32[64]" = torch.ops.aten.mul.Tensor(sum_103, 0.00015943877551020407)
    mul_1048: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_103, squeeze_103)
    mul_1049: "f32[64]" = torch.ops.aten.mul.Tensor(mul_1047, mul_1048);  mul_1047 = mul_1048 = None
    unsqueeze_946: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1049, 0);  mul_1049 = None
    unsqueeze_947: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_946, 2);  unsqueeze_946 = None
    unsqueeze_948: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_947, 3);  unsqueeze_947 = None
    mul_1050: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_103, primals_104);  primals_104 = None
    unsqueeze_949: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1050, 0);  mul_1050 = None
    unsqueeze_950: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_949, 2);  unsqueeze_949 = None
    unsqueeze_951: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_950, 3);  unsqueeze_950 = None
    sub_286: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_942);  convolution_34 = unsqueeze_942 = None
    mul_1051: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(sub_286, unsqueeze_948);  sub_286 = unsqueeze_948 = None
    sub_287: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(where_48, mul_1051);  where_48 = mul_1051 = None
    sub_288: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(sub_287, unsqueeze_945);  sub_287 = unsqueeze_945 = None
    mul_1052: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(sub_288, unsqueeze_951);  sub_288 = unsqueeze_951 = None
    mul_1053: "f32[64]" = torch.ops.aten.mul.Tensor(sum_103, squeeze_103);  sum_103 = squeeze_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_50 = torch.ops.aten.convolution_backward.default(mul_1052, getitem_194, primals_103, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_1052 = getitem_194 = primals_103 = None
    getitem_642: "f32[8, 64, 28, 28]" = convolution_backward_50[0]
    getitem_643: "f32[64, 8, 3, 3]" = convolution_backward_50[1];  convolution_backward_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    cat_25: "f32[8, 256, 28, 28]" = torch.ops.aten.cat.default([getitem_642, getitem_639, getitem_636, slice_40], 1);  getitem_642 = getitem_639 = getitem_636 = slice_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    alias_229: "f32[8, 256, 28, 28]" = torch.ops.aten.alias.default(relu_31);  relu_31 = None
    alias_230: "f32[8, 256, 28, 28]" = torch.ops.aten.alias.default(alias_229);  alias_229 = None
    le_49: "b8[8, 256, 28, 28]" = torch.ops.aten.le.Scalar(alias_230, 0);  alias_230 = None
    scalar_tensor_49: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_49: "f32[8, 256, 28, 28]" = torch.ops.aten.where.self(le_49, scalar_tensor_49, cat_25);  le_49 = scalar_tensor_49 = cat_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    unsqueeze_952: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_99, 0);  squeeze_99 = None
    unsqueeze_953: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_952, 2);  unsqueeze_952 = None
    unsqueeze_954: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_953, 3);  unsqueeze_953 = None
    sum_104: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_49, [0, 2, 3])
    sub_289: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_954)
    mul_1054: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(where_49, sub_289);  sub_289 = None
    sum_105: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1054, [0, 2, 3]);  mul_1054 = None
    mul_1055: "f32[256]" = torch.ops.aten.mul.Tensor(sum_104, 0.00015943877551020407)
    unsqueeze_955: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1055, 0);  mul_1055 = None
    unsqueeze_956: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_955, 2);  unsqueeze_955 = None
    unsqueeze_957: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_956, 3);  unsqueeze_956 = None
    mul_1056: "f32[256]" = torch.ops.aten.mul.Tensor(sum_105, 0.00015943877551020407)
    mul_1057: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_100, squeeze_100)
    mul_1058: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1056, mul_1057);  mul_1056 = mul_1057 = None
    unsqueeze_958: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1058, 0);  mul_1058 = None
    unsqueeze_959: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_958, 2);  unsqueeze_958 = None
    unsqueeze_960: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_959, 3);  unsqueeze_959 = None
    mul_1059: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_100, primals_101);  primals_101 = None
    unsqueeze_961: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1059, 0);  mul_1059 = None
    unsqueeze_962: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_961, 2);  unsqueeze_961 = None
    unsqueeze_963: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_962, 3);  unsqueeze_962 = None
    sub_290: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_954);  convolution_33 = unsqueeze_954 = None
    mul_1060: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_290, unsqueeze_960);  sub_290 = unsqueeze_960 = None
    sub_291: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(where_49, mul_1060);  where_49 = mul_1060 = None
    sub_292: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(sub_291, unsqueeze_957);  sub_291 = unsqueeze_957 = None
    mul_1061: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_292, unsqueeze_963);  sub_292 = unsqueeze_963 = None
    mul_1062: "f32[256]" = torch.ops.aten.mul.Tensor(sum_105, squeeze_100);  sum_105 = squeeze_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_backward_51 = torch.ops.aten.convolution_backward.default(mul_1061, relu_30, primals_100, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1061 = primals_100 = None
    getitem_645: "f32[8, 512, 28, 28]" = convolution_backward_51[0]
    getitem_646: "f32[256, 512, 1, 1]" = convolution_backward_51[1];  convolution_backward_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    add_490: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(where_45, getitem_645);  where_45 = getitem_645 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    alias_232: "f32[8, 512, 28, 28]" = torch.ops.aten.alias.default(relu_30);  relu_30 = None
    alias_233: "f32[8, 512, 28, 28]" = torch.ops.aten.alias.default(alias_232);  alias_232 = None
    le_50: "b8[8, 512, 28, 28]" = torch.ops.aten.le.Scalar(alias_233, 0);  alias_233 = None
    scalar_tensor_50: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_50: "f32[8, 512, 28, 28]" = torch.ops.aten.where.self(le_50, scalar_tensor_50, add_490);  le_50 = scalar_tensor_50 = add_490 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    unsqueeze_964: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_96, 0);  squeeze_96 = None
    unsqueeze_965: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_964, 2);  unsqueeze_964 = None
    unsqueeze_966: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_965, 3);  unsqueeze_965 = None
    sum_106: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_50, [0, 2, 3])
    sub_293: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_966)
    mul_1063: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(where_50, sub_293);  sub_293 = None
    sum_107: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1063, [0, 2, 3]);  mul_1063 = None
    mul_1064: "f32[512]" = torch.ops.aten.mul.Tensor(sum_106, 0.00015943877551020407)
    unsqueeze_967: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1064, 0);  mul_1064 = None
    unsqueeze_968: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_967, 2);  unsqueeze_967 = None
    unsqueeze_969: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_968, 3);  unsqueeze_968 = None
    mul_1065: "f32[512]" = torch.ops.aten.mul.Tensor(sum_107, 0.00015943877551020407)
    mul_1066: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_97, squeeze_97)
    mul_1067: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1065, mul_1066);  mul_1065 = mul_1066 = None
    unsqueeze_970: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1067, 0);  mul_1067 = None
    unsqueeze_971: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_970, 2);  unsqueeze_970 = None
    unsqueeze_972: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_971, 3);  unsqueeze_971 = None
    mul_1068: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_97, primals_98);  primals_98 = None
    unsqueeze_973: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1068, 0);  mul_1068 = None
    unsqueeze_974: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_973, 2);  unsqueeze_973 = None
    unsqueeze_975: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_974, 3);  unsqueeze_974 = None
    sub_294: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_966);  convolution_32 = unsqueeze_966 = None
    mul_1069: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_294, unsqueeze_972);  sub_294 = unsqueeze_972 = None
    sub_295: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(where_50, mul_1069);  mul_1069 = None
    sub_296: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(sub_295, unsqueeze_969);  sub_295 = unsqueeze_969 = None
    mul_1070: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_296, unsqueeze_975);  sub_296 = unsqueeze_975 = None
    mul_1071: "f32[512]" = torch.ops.aten.mul.Tensor(sum_107, squeeze_97);  sum_107 = squeeze_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_backward_52 = torch.ops.aten.convolution_backward.default(mul_1070, cat_5, primals_97, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1070 = cat_5 = primals_97 = None
    getitem_648: "f32[8, 256, 28, 28]" = convolution_backward_52[0]
    getitem_649: "f32[512, 256, 1, 1]" = convolution_backward_52[1];  convolution_backward_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    slice_41: "f32[8, 64, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_648, 1, 0, 64)
    slice_42: "f32[8, 64, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_648, 1, 64, 128)
    slice_43: "f32[8, 64, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_648, 1, 128, 192)
    slice_44: "f32[8, 64, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_648, 1, 192, 256);  getitem_648 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_235: "f32[8, 64, 28, 28]" = torch.ops.aten.alias.default(relu_29);  relu_29 = None
    alias_236: "f32[8, 64, 28, 28]" = torch.ops.aten.alias.default(alias_235);  alias_235 = None
    le_51: "b8[8, 64, 28, 28]" = torch.ops.aten.le.Scalar(alias_236, 0);  alias_236 = None
    scalar_tensor_51: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_51: "f32[8, 64, 28, 28]" = torch.ops.aten.where.self(le_51, scalar_tensor_51, slice_43);  le_51 = scalar_tensor_51 = slice_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_976: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_93, 0);  squeeze_93 = None
    unsqueeze_977: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_976, 2);  unsqueeze_976 = None
    unsqueeze_978: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_977, 3);  unsqueeze_977 = None
    sum_108: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_51, [0, 2, 3])
    sub_297: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_31, unsqueeze_978)
    mul_1072: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(where_51, sub_297);  sub_297 = None
    sum_109: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1072, [0, 2, 3]);  mul_1072 = None
    mul_1073: "f32[64]" = torch.ops.aten.mul.Tensor(sum_108, 0.00015943877551020407)
    unsqueeze_979: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1073, 0);  mul_1073 = None
    unsqueeze_980: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_979, 2);  unsqueeze_979 = None
    unsqueeze_981: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_980, 3);  unsqueeze_980 = None
    mul_1074: "f32[64]" = torch.ops.aten.mul.Tensor(sum_109, 0.00015943877551020407)
    mul_1075: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_94, squeeze_94)
    mul_1076: "f32[64]" = torch.ops.aten.mul.Tensor(mul_1074, mul_1075);  mul_1074 = mul_1075 = None
    unsqueeze_982: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1076, 0);  mul_1076 = None
    unsqueeze_983: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_982, 2);  unsqueeze_982 = None
    unsqueeze_984: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_983, 3);  unsqueeze_983 = None
    mul_1077: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_94, primals_95);  primals_95 = None
    unsqueeze_985: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1077, 0);  mul_1077 = None
    unsqueeze_986: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_985, 2);  unsqueeze_985 = None
    unsqueeze_987: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_986, 3);  unsqueeze_986 = None
    sub_298: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_31, unsqueeze_978);  convolution_31 = unsqueeze_978 = None
    mul_1078: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(sub_298, unsqueeze_984);  sub_298 = unsqueeze_984 = None
    sub_299: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(where_51, mul_1078);  where_51 = mul_1078 = None
    sub_300: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(sub_299, unsqueeze_981);  sub_299 = unsqueeze_981 = None
    mul_1079: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(sub_300, unsqueeze_987);  sub_300 = unsqueeze_987 = None
    mul_1080: "f32[64]" = torch.ops.aten.mul.Tensor(sum_109, squeeze_94);  sum_109 = squeeze_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_53 = torch.ops.aten.convolution_backward.default(mul_1079, add_167, primals_94, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_1079 = add_167 = primals_94 = None
    getitem_651: "f32[8, 64, 28, 28]" = convolution_backward_53[0]
    getitem_652: "f32[64, 8, 3, 3]" = convolution_backward_53[1];  convolution_backward_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_491: "f32[8, 64, 28, 28]" = torch.ops.aten.add.Tensor(slice_42, getitem_651);  slice_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_238: "f32[8, 64, 28, 28]" = torch.ops.aten.alias.default(relu_28);  relu_28 = None
    alias_239: "f32[8, 64, 28, 28]" = torch.ops.aten.alias.default(alias_238);  alias_238 = None
    le_52: "b8[8, 64, 28, 28]" = torch.ops.aten.le.Scalar(alias_239, 0);  alias_239 = None
    scalar_tensor_52: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_52: "f32[8, 64, 28, 28]" = torch.ops.aten.where.self(le_52, scalar_tensor_52, add_491);  le_52 = scalar_tensor_52 = add_491 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_988: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_90, 0);  squeeze_90 = None
    unsqueeze_989: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_988, 2);  unsqueeze_988 = None
    unsqueeze_990: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_989, 3);  unsqueeze_989 = None
    sum_110: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_52, [0, 2, 3])
    sub_301: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_990)
    mul_1081: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(where_52, sub_301);  sub_301 = None
    sum_111: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1081, [0, 2, 3]);  mul_1081 = None
    mul_1082: "f32[64]" = torch.ops.aten.mul.Tensor(sum_110, 0.00015943877551020407)
    unsqueeze_991: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1082, 0);  mul_1082 = None
    unsqueeze_992: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_991, 2);  unsqueeze_991 = None
    unsqueeze_993: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_992, 3);  unsqueeze_992 = None
    mul_1083: "f32[64]" = torch.ops.aten.mul.Tensor(sum_111, 0.00015943877551020407)
    mul_1084: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_91, squeeze_91)
    mul_1085: "f32[64]" = torch.ops.aten.mul.Tensor(mul_1083, mul_1084);  mul_1083 = mul_1084 = None
    unsqueeze_994: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1085, 0);  mul_1085 = None
    unsqueeze_995: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_994, 2);  unsqueeze_994 = None
    unsqueeze_996: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_995, 3);  unsqueeze_995 = None
    mul_1086: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_91, primals_92);  primals_92 = None
    unsqueeze_997: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1086, 0);  mul_1086 = None
    unsqueeze_998: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_997, 2);  unsqueeze_997 = None
    unsqueeze_999: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_998, 3);  unsqueeze_998 = None
    sub_302: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_990);  convolution_30 = unsqueeze_990 = None
    mul_1087: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(sub_302, unsqueeze_996);  sub_302 = unsqueeze_996 = None
    sub_303: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(where_52, mul_1087);  where_52 = mul_1087 = None
    sub_304: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(sub_303, unsqueeze_993);  sub_303 = unsqueeze_993 = None
    mul_1088: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(sub_304, unsqueeze_999);  sub_304 = unsqueeze_999 = None
    mul_1089: "f32[64]" = torch.ops.aten.mul.Tensor(sum_111, squeeze_91);  sum_111 = squeeze_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_54 = torch.ops.aten.convolution_backward.default(mul_1088, add_161, primals_91, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_1088 = add_161 = primals_91 = None
    getitem_654: "f32[8, 64, 28, 28]" = convolution_backward_54[0]
    getitem_655: "f32[64, 8, 3, 3]" = convolution_backward_54[1];  convolution_backward_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_492: "f32[8, 64, 28, 28]" = torch.ops.aten.add.Tensor(slice_41, getitem_654);  slice_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_241: "f32[8, 64, 28, 28]" = torch.ops.aten.alias.default(relu_27);  relu_27 = None
    alias_242: "f32[8, 64, 28, 28]" = torch.ops.aten.alias.default(alias_241);  alias_241 = None
    le_53: "b8[8, 64, 28, 28]" = torch.ops.aten.le.Scalar(alias_242, 0);  alias_242 = None
    scalar_tensor_53: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_53: "f32[8, 64, 28, 28]" = torch.ops.aten.where.self(le_53, scalar_tensor_53, add_492);  le_53 = scalar_tensor_53 = add_492 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_1000: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_87, 0);  squeeze_87 = None
    unsqueeze_1001: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1000, 2);  unsqueeze_1000 = None
    unsqueeze_1002: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1001, 3);  unsqueeze_1001 = None
    sum_112: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_53, [0, 2, 3])
    sub_305: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_1002)
    mul_1090: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(where_53, sub_305);  sub_305 = None
    sum_113: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1090, [0, 2, 3]);  mul_1090 = None
    mul_1091: "f32[64]" = torch.ops.aten.mul.Tensor(sum_112, 0.00015943877551020407)
    unsqueeze_1003: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1091, 0);  mul_1091 = None
    unsqueeze_1004: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1003, 2);  unsqueeze_1003 = None
    unsqueeze_1005: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1004, 3);  unsqueeze_1004 = None
    mul_1092: "f32[64]" = torch.ops.aten.mul.Tensor(sum_113, 0.00015943877551020407)
    mul_1093: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_88, squeeze_88)
    mul_1094: "f32[64]" = torch.ops.aten.mul.Tensor(mul_1092, mul_1093);  mul_1092 = mul_1093 = None
    unsqueeze_1006: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1094, 0);  mul_1094 = None
    unsqueeze_1007: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1006, 2);  unsqueeze_1006 = None
    unsqueeze_1008: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1007, 3);  unsqueeze_1007 = None
    mul_1095: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_88, primals_89);  primals_89 = None
    unsqueeze_1009: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1095, 0);  mul_1095 = None
    unsqueeze_1010: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1009, 2);  unsqueeze_1009 = None
    unsqueeze_1011: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1010, 3);  unsqueeze_1010 = None
    sub_306: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_1002);  convolution_29 = unsqueeze_1002 = None
    mul_1096: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(sub_306, unsqueeze_1008);  sub_306 = unsqueeze_1008 = None
    sub_307: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(where_53, mul_1096);  where_53 = mul_1096 = None
    sub_308: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(sub_307, unsqueeze_1005);  sub_307 = unsqueeze_1005 = None
    mul_1097: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(sub_308, unsqueeze_1011);  sub_308 = unsqueeze_1011 = None
    mul_1098: "f32[64]" = torch.ops.aten.mul.Tensor(sum_113, squeeze_88);  sum_113 = squeeze_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_55 = torch.ops.aten.convolution_backward.default(mul_1097, getitem_164, primals_88, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_1097 = getitem_164 = primals_88 = None
    getitem_657: "f32[8, 64, 28, 28]" = convolution_backward_55[0]
    getitem_658: "f32[64, 8, 3, 3]" = convolution_backward_55[1];  convolution_backward_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    cat_26: "f32[8, 256, 28, 28]" = torch.ops.aten.cat.default([getitem_657, getitem_654, getitem_651, slice_44], 1);  getitem_657 = getitem_654 = getitem_651 = slice_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    alias_244: "f32[8, 256, 28, 28]" = torch.ops.aten.alias.default(relu_26);  relu_26 = None
    alias_245: "f32[8, 256, 28, 28]" = torch.ops.aten.alias.default(alias_244);  alias_244 = None
    le_54: "b8[8, 256, 28, 28]" = torch.ops.aten.le.Scalar(alias_245, 0);  alias_245 = None
    scalar_tensor_54: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_54: "f32[8, 256, 28, 28]" = torch.ops.aten.where.self(le_54, scalar_tensor_54, cat_26);  le_54 = scalar_tensor_54 = cat_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    unsqueeze_1012: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_84, 0);  squeeze_84 = None
    unsqueeze_1013: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1012, 2);  unsqueeze_1012 = None
    unsqueeze_1014: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1013, 3);  unsqueeze_1013 = None
    sum_114: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_54, [0, 2, 3])
    sub_309: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_1014)
    mul_1099: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(where_54, sub_309);  sub_309 = None
    sum_115: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1099, [0, 2, 3]);  mul_1099 = None
    mul_1100: "f32[256]" = torch.ops.aten.mul.Tensor(sum_114, 0.00015943877551020407)
    unsqueeze_1015: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1100, 0);  mul_1100 = None
    unsqueeze_1016: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1015, 2);  unsqueeze_1015 = None
    unsqueeze_1017: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1016, 3);  unsqueeze_1016 = None
    mul_1101: "f32[256]" = torch.ops.aten.mul.Tensor(sum_115, 0.00015943877551020407)
    mul_1102: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_85, squeeze_85)
    mul_1103: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1101, mul_1102);  mul_1101 = mul_1102 = None
    unsqueeze_1018: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1103, 0);  mul_1103 = None
    unsqueeze_1019: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1018, 2);  unsqueeze_1018 = None
    unsqueeze_1020: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1019, 3);  unsqueeze_1019 = None
    mul_1104: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_85, primals_86);  primals_86 = None
    unsqueeze_1021: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1104, 0);  mul_1104 = None
    unsqueeze_1022: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1021, 2);  unsqueeze_1021 = None
    unsqueeze_1023: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1022, 3);  unsqueeze_1022 = None
    sub_310: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_1014);  convolution_28 = unsqueeze_1014 = None
    mul_1105: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_310, unsqueeze_1020);  sub_310 = unsqueeze_1020 = None
    sub_311: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(where_54, mul_1105);  where_54 = mul_1105 = None
    sub_312: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(sub_311, unsqueeze_1017);  sub_311 = unsqueeze_1017 = None
    mul_1106: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_312, unsqueeze_1023);  sub_312 = unsqueeze_1023 = None
    mul_1107: "f32[256]" = torch.ops.aten.mul.Tensor(sum_115, squeeze_85);  sum_115 = squeeze_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_backward_56 = torch.ops.aten.convolution_backward.default(mul_1106, relu_25, primals_85, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1106 = primals_85 = None
    getitem_660: "f32[8, 512, 28, 28]" = convolution_backward_56[0]
    getitem_661: "f32[256, 512, 1, 1]" = convolution_backward_56[1];  convolution_backward_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    add_493: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(where_50, getitem_660);  where_50 = getitem_660 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    alias_247: "f32[8, 512, 28, 28]" = torch.ops.aten.alias.default(relu_25);  relu_25 = None
    alias_248: "f32[8, 512, 28, 28]" = torch.ops.aten.alias.default(alias_247);  alias_247 = None
    le_55: "b8[8, 512, 28, 28]" = torch.ops.aten.le.Scalar(alias_248, 0);  alias_248 = None
    scalar_tensor_55: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_55: "f32[8, 512, 28, 28]" = torch.ops.aten.where.self(le_55, scalar_tensor_55, add_493);  le_55 = scalar_tensor_55 = add_493 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    unsqueeze_1024: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_81, 0);  squeeze_81 = None
    unsqueeze_1025: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1024, 2);  unsqueeze_1024 = None
    unsqueeze_1026: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1025, 3);  unsqueeze_1025 = None
    sum_116: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_55, [0, 2, 3])
    sub_313: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_1026)
    mul_1108: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(where_55, sub_313);  sub_313 = None
    sum_117: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1108, [0, 2, 3]);  mul_1108 = None
    mul_1109: "f32[512]" = torch.ops.aten.mul.Tensor(sum_116, 0.00015943877551020407)
    unsqueeze_1027: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1109, 0);  mul_1109 = None
    unsqueeze_1028: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1027, 2);  unsqueeze_1027 = None
    unsqueeze_1029: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1028, 3);  unsqueeze_1028 = None
    mul_1110: "f32[512]" = torch.ops.aten.mul.Tensor(sum_117, 0.00015943877551020407)
    mul_1111: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_82, squeeze_82)
    mul_1112: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1110, mul_1111);  mul_1110 = mul_1111 = None
    unsqueeze_1030: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1112, 0);  mul_1112 = None
    unsqueeze_1031: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1030, 2);  unsqueeze_1030 = None
    unsqueeze_1032: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1031, 3);  unsqueeze_1031 = None
    mul_1113: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_82, primals_83);  primals_83 = None
    unsqueeze_1033: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1113, 0);  mul_1113 = None
    unsqueeze_1034: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1033, 2);  unsqueeze_1033 = None
    unsqueeze_1035: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1034, 3);  unsqueeze_1034 = None
    sub_314: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_1026);  convolution_27 = unsqueeze_1026 = None
    mul_1114: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_314, unsqueeze_1032);  sub_314 = unsqueeze_1032 = None
    sub_315: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(where_55, mul_1114);  mul_1114 = None
    sub_316: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(sub_315, unsqueeze_1029);  sub_315 = unsqueeze_1029 = None
    mul_1115: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_316, unsqueeze_1035);  sub_316 = unsqueeze_1035 = None
    mul_1116: "f32[512]" = torch.ops.aten.mul.Tensor(sum_117, squeeze_82);  sum_117 = squeeze_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_backward_57 = torch.ops.aten.convolution_backward.default(mul_1115, cat_4, primals_82, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1115 = cat_4 = primals_82 = None
    getitem_663: "f32[8, 256, 28, 28]" = convolution_backward_57[0]
    getitem_664: "f32[512, 256, 1, 1]" = convolution_backward_57[1];  convolution_backward_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    slice_45: "f32[8, 64, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_663, 1, 0, 64)
    slice_46: "f32[8, 64, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_663, 1, 64, 128)
    slice_47: "f32[8, 64, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_663, 1, 128, 192)
    slice_48: "f32[8, 64, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_663, 1, 192, 256);  getitem_663 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_250: "f32[8, 64, 28, 28]" = torch.ops.aten.alias.default(relu_24);  relu_24 = None
    alias_251: "f32[8, 64, 28, 28]" = torch.ops.aten.alias.default(alias_250);  alias_250 = None
    le_56: "b8[8, 64, 28, 28]" = torch.ops.aten.le.Scalar(alias_251, 0);  alias_251 = None
    scalar_tensor_56: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_56: "f32[8, 64, 28, 28]" = torch.ops.aten.where.self(le_56, scalar_tensor_56, slice_47);  le_56 = scalar_tensor_56 = slice_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_1036: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_78, 0);  squeeze_78 = None
    unsqueeze_1037: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1036, 2);  unsqueeze_1036 = None
    unsqueeze_1038: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1037, 3);  unsqueeze_1037 = None
    sum_118: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_56, [0, 2, 3])
    sub_317: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_1038)
    mul_1117: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(where_56, sub_317);  sub_317 = None
    sum_119: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1117, [0, 2, 3]);  mul_1117 = None
    mul_1118: "f32[64]" = torch.ops.aten.mul.Tensor(sum_118, 0.00015943877551020407)
    unsqueeze_1039: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1118, 0);  mul_1118 = None
    unsqueeze_1040: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1039, 2);  unsqueeze_1039 = None
    unsqueeze_1041: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1040, 3);  unsqueeze_1040 = None
    mul_1119: "f32[64]" = torch.ops.aten.mul.Tensor(sum_119, 0.00015943877551020407)
    mul_1120: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_79, squeeze_79)
    mul_1121: "f32[64]" = torch.ops.aten.mul.Tensor(mul_1119, mul_1120);  mul_1119 = mul_1120 = None
    unsqueeze_1042: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1121, 0);  mul_1121 = None
    unsqueeze_1043: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1042, 2);  unsqueeze_1042 = None
    unsqueeze_1044: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1043, 3);  unsqueeze_1043 = None
    mul_1122: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_79, primals_80);  primals_80 = None
    unsqueeze_1045: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1122, 0);  mul_1122 = None
    unsqueeze_1046: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1045, 2);  unsqueeze_1045 = None
    unsqueeze_1047: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1046, 3);  unsqueeze_1046 = None
    sub_318: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_1038);  convolution_26 = unsqueeze_1038 = None
    mul_1123: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(sub_318, unsqueeze_1044);  sub_318 = unsqueeze_1044 = None
    sub_319: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(where_56, mul_1123);  where_56 = mul_1123 = None
    sub_320: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(sub_319, unsqueeze_1041);  sub_319 = unsqueeze_1041 = None
    mul_1124: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(sub_320, unsqueeze_1047);  sub_320 = unsqueeze_1047 = None
    mul_1125: "f32[64]" = torch.ops.aten.mul.Tensor(sum_119, squeeze_79);  sum_119 = squeeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_58 = torch.ops.aten.convolution_backward.default(mul_1124, add_139, primals_79, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_1124 = add_139 = primals_79 = None
    getitem_666: "f32[8, 64, 28, 28]" = convolution_backward_58[0]
    getitem_667: "f32[64, 8, 3, 3]" = convolution_backward_58[1];  convolution_backward_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_494: "f32[8, 64, 28, 28]" = torch.ops.aten.add.Tensor(slice_46, getitem_666);  slice_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_253: "f32[8, 64, 28, 28]" = torch.ops.aten.alias.default(relu_23);  relu_23 = None
    alias_254: "f32[8, 64, 28, 28]" = torch.ops.aten.alias.default(alias_253);  alias_253 = None
    le_57: "b8[8, 64, 28, 28]" = torch.ops.aten.le.Scalar(alias_254, 0);  alias_254 = None
    scalar_tensor_57: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_57: "f32[8, 64, 28, 28]" = torch.ops.aten.where.self(le_57, scalar_tensor_57, add_494);  le_57 = scalar_tensor_57 = add_494 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_1048: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_75, 0);  squeeze_75 = None
    unsqueeze_1049: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1048, 2);  unsqueeze_1048 = None
    unsqueeze_1050: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1049, 3);  unsqueeze_1049 = None
    sum_120: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_57, [0, 2, 3])
    sub_321: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_1050)
    mul_1126: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(where_57, sub_321);  sub_321 = None
    sum_121: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1126, [0, 2, 3]);  mul_1126 = None
    mul_1127: "f32[64]" = torch.ops.aten.mul.Tensor(sum_120, 0.00015943877551020407)
    unsqueeze_1051: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1127, 0);  mul_1127 = None
    unsqueeze_1052: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1051, 2);  unsqueeze_1051 = None
    unsqueeze_1053: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1052, 3);  unsqueeze_1052 = None
    mul_1128: "f32[64]" = torch.ops.aten.mul.Tensor(sum_121, 0.00015943877551020407)
    mul_1129: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_76, squeeze_76)
    mul_1130: "f32[64]" = torch.ops.aten.mul.Tensor(mul_1128, mul_1129);  mul_1128 = mul_1129 = None
    unsqueeze_1054: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1130, 0);  mul_1130 = None
    unsqueeze_1055: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1054, 2);  unsqueeze_1054 = None
    unsqueeze_1056: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1055, 3);  unsqueeze_1055 = None
    mul_1131: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_76, primals_77);  primals_77 = None
    unsqueeze_1057: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1131, 0);  mul_1131 = None
    unsqueeze_1058: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1057, 2);  unsqueeze_1057 = None
    unsqueeze_1059: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1058, 3);  unsqueeze_1058 = None
    sub_322: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_1050);  convolution_25 = unsqueeze_1050 = None
    mul_1132: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(sub_322, unsqueeze_1056);  sub_322 = unsqueeze_1056 = None
    sub_323: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(where_57, mul_1132);  where_57 = mul_1132 = None
    sub_324: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(sub_323, unsqueeze_1053);  sub_323 = unsqueeze_1053 = None
    mul_1133: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(sub_324, unsqueeze_1059);  sub_324 = unsqueeze_1059 = None
    mul_1134: "f32[64]" = torch.ops.aten.mul.Tensor(sum_121, squeeze_76);  sum_121 = squeeze_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_59 = torch.ops.aten.convolution_backward.default(mul_1133, add_133, primals_76, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_1133 = add_133 = primals_76 = None
    getitem_669: "f32[8, 64, 28, 28]" = convolution_backward_59[0]
    getitem_670: "f32[64, 8, 3, 3]" = convolution_backward_59[1];  convolution_backward_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_495: "f32[8, 64, 28, 28]" = torch.ops.aten.add.Tensor(slice_45, getitem_669);  slice_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_256: "f32[8, 64, 28, 28]" = torch.ops.aten.alias.default(relu_22);  relu_22 = None
    alias_257: "f32[8, 64, 28, 28]" = torch.ops.aten.alias.default(alias_256);  alias_256 = None
    le_58: "b8[8, 64, 28, 28]" = torch.ops.aten.le.Scalar(alias_257, 0);  alias_257 = None
    scalar_tensor_58: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_58: "f32[8, 64, 28, 28]" = torch.ops.aten.where.self(le_58, scalar_tensor_58, add_495);  le_58 = scalar_tensor_58 = add_495 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_1060: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_72, 0);  squeeze_72 = None
    unsqueeze_1061: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1060, 2);  unsqueeze_1060 = None
    unsqueeze_1062: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1061, 3);  unsqueeze_1061 = None
    sum_122: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_58, [0, 2, 3])
    sub_325: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_1062)
    mul_1135: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(where_58, sub_325);  sub_325 = None
    sum_123: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1135, [0, 2, 3]);  mul_1135 = None
    mul_1136: "f32[64]" = torch.ops.aten.mul.Tensor(sum_122, 0.00015943877551020407)
    unsqueeze_1063: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1136, 0);  mul_1136 = None
    unsqueeze_1064: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1063, 2);  unsqueeze_1063 = None
    unsqueeze_1065: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1064, 3);  unsqueeze_1064 = None
    mul_1137: "f32[64]" = torch.ops.aten.mul.Tensor(sum_123, 0.00015943877551020407)
    mul_1138: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_73, squeeze_73)
    mul_1139: "f32[64]" = torch.ops.aten.mul.Tensor(mul_1137, mul_1138);  mul_1137 = mul_1138 = None
    unsqueeze_1066: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1139, 0);  mul_1139 = None
    unsqueeze_1067: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1066, 2);  unsqueeze_1066 = None
    unsqueeze_1068: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1067, 3);  unsqueeze_1067 = None
    mul_1140: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_73, primals_74);  primals_74 = None
    unsqueeze_1069: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1140, 0);  mul_1140 = None
    unsqueeze_1070: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1069, 2);  unsqueeze_1069 = None
    unsqueeze_1071: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1070, 3);  unsqueeze_1070 = None
    sub_326: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_1062);  convolution_24 = unsqueeze_1062 = None
    mul_1141: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(sub_326, unsqueeze_1068);  sub_326 = unsqueeze_1068 = None
    sub_327: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(where_58, mul_1141);  where_58 = mul_1141 = None
    sub_328: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(sub_327, unsqueeze_1065);  sub_327 = unsqueeze_1065 = None
    mul_1142: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(sub_328, unsqueeze_1071);  sub_328 = unsqueeze_1071 = None
    mul_1143: "f32[64]" = torch.ops.aten.mul.Tensor(sum_123, squeeze_73);  sum_123 = squeeze_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_60 = torch.ops.aten.convolution_backward.default(mul_1142, getitem_134, primals_73, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_1142 = getitem_134 = primals_73 = None
    getitem_672: "f32[8, 64, 28, 28]" = convolution_backward_60[0]
    getitem_673: "f32[64, 8, 3, 3]" = convolution_backward_60[1];  convolution_backward_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    cat_27: "f32[8, 256, 28, 28]" = torch.ops.aten.cat.default([getitem_672, getitem_669, getitem_666, slice_48], 1);  getitem_672 = getitem_669 = getitem_666 = slice_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    alias_259: "f32[8, 256, 28, 28]" = torch.ops.aten.alias.default(relu_21);  relu_21 = None
    alias_260: "f32[8, 256, 28, 28]" = torch.ops.aten.alias.default(alias_259);  alias_259 = None
    le_59: "b8[8, 256, 28, 28]" = torch.ops.aten.le.Scalar(alias_260, 0);  alias_260 = None
    scalar_tensor_59: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_59: "f32[8, 256, 28, 28]" = torch.ops.aten.where.self(le_59, scalar_tensor_59, cat_27);  le_59 = scalar_tensor_59 = cat_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    unsqueeze_1072: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_69, 0);  squeeze_69 = None
    unsqueeze_1073: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1072, 2);  unsqueeze_1072 = None
    unsqueeze_1074: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1073, 3);  unsqueeze_1073 = None
    sum_124: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_59, [0, 2, 3])
    sub_329: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_1074)
    mul_1144: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(where_59, sub_329);  sub_329 = None
    sum_125: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1144, [0, 2, 3]);  mul_1144 = None
    mul_1145: "f32[256]" = torch.ops.aten.mul.Tensor(sum_124, 0.00015943877551020407)
    unsqueeze_1075: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1145, 0);  mul_1145 = None
    unsqueeze_1076: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1075, 2);  unsqueeze_1075 = None
    unsqueeze_1077: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1076, 3);  unsqueeze_1076 = None
    mul_1146: "f32[256]" = torch.ops.aten.mul.Tensor(sum_125, 0.00015943877551020407)
    mul_1147: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_70, squeeze_70)
    mul_1148: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1146, mul_1147);  mul_1146 = mul_1147 = None
    unsqueeze_1078: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1148, 0);  mul_1148 = None
    unsqueeze_1079: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1078, 2);  unsqueeze_1078 = None
    unsqueeze_1080: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1079, 3);  unsqueeze_1079 = None
    mul_1149: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_70, primals_71);  primals_71 = None
    unsqueeze_1081: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1149, 0);  mul_1149 = None
    unsqueeze_1082: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1081, 2);  unsqueeze_1081 = None
    unsqueeze_1083: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1082, 3);  unsqueeze_1082 = None
    sub_330: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_1074);  convolution_23 = unsqueeze_1074 = None
    mul_1150: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_330, unsqueeze_1080);  sub_330 = unsqueeze_1080 = None
    sub_331: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(where_59, mul_1150);  where_59 = mul_1150 = None
    sub_332: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(sub_331, unsqueeze_1077);  sub_331 = unsqueeze_1077 = None
    mul_1151: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_332, unsqueeze_1083);  sub_332 = unsqueeze_1083 = None
    mul_1152: "f32[256]" = torch.ops.aten.mul.Tensor(sum_125, squeeze_70);  sum_125 = squeeze_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_backward_61 = torch.ops.aten.convolution_backward.default(mul_1151, relu_20, primals_70, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1151 = primals_70 = None
    getitem_675: "f32[8, 512, 28, 28]" = convolution_backward_61[0]
    getitem_676: "f32[256, 512, 1, 1]" = convolution_backward_61[1];  convolution_backward_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    add_496: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(where_55, getitem_675);  where_55 = getitem_675 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    alias_262: "f32[8, 512, 28, 28]" = torch.ops.aten.alias.default(relu_20);  relu_20 = None
    alias_263: "f32[8, 512, 28, 28]" = torch.ops.aten.alias.default(alias_262);  alias_262 = None
    le_60: "b8[8, 512, 28, 28]" = torch.ops.aten.le.Scalar(alias_263, 0);  alias_263 = None
    scalar_tensor_60: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_60: "f32[8, 512, 28, 28]" = torch.ops.aten.where.self(le_60, scalar_tensor_60, add_496);  le_60 = scalar_tensor_60 = add_496 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:111, code: shortcut = self.downsample(x)
    unsqueeze_1084: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_66, 0);  squeeze_66 = None
    unsqueeze_1085: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1084, 2);  unsqueeze_1084 = None
    unsqueeze_1086: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1085, 3);  unsqueeze_1085 = None
    sum_126: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_60, [0, 2, 3])
    sub_333: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_1086)
    mul_1153: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(where_60, sub_333);  sub_333 = None
    sum_127: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1153, [0, 2, 3]);  mul_1153 = None
    mul_1154: "f32[512]" = torch.ops.aten.mul.Tensor(sum_126, 0.00015943877551020407)
    unsqueeze_1087: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1154, 0);  mul_1154 = None
    unsqueeze_1088: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1087, 2);  unsqueeze_1087 = None
    unsqueeze_1089: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1088, 3);  unsqueeze_1088 = None
    mul_1155: "f32[512]" = torch.ops.aten.mul.Tensor(sum_127, 0.00015943877551020407)
    mul_1156: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_67, squeeze_67)
    mul_1157: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1155, mul_1156);  mul_1155 = mul_1156 = None
    unsqueeze_1090: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1157, 0);  mul_1157 = None
    unsqueeze_1091: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1090, 2);  unsqueeze_1090 = None
    unsqueeze_1092: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1091, 3);  unsqueeze_1091 = None
    mul_1158: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_67, primals_68);  primals_68 = None
    unsqueeze_1093: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1158, 0);  mul_1158 = None
    unsqueeze_1094: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1093, 2);  unsqueeze_1093 = None
    unsqueeze_1095: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1094, 3);  unsqueeze_1094 = None
    sub_334: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_1086);  convolution_22 = unsqueeze_1086 = None
    mul_1159: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_334, unsqueeze_1092);  sub_334 = unsqueeze_1092 = None
    sub_335: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(where_60, mul_1159);  mul_1159 = None
    sub_336: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(sub_335, unsqueeze_1089);  sub_335 = unsqueeze_1089 = None
    mul_1160: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_336, unsqueeze_1095);  sub_336 = unsqueeze_1095 = None
    mul_1161: "f32[512]" = torch.ops.aten.mul.Tensor(sum_127, squeeze_67);  sum_127 = squeeze_67 = None
    convolution_backward_62 = torch.ops.aten.convolution_backward.default(mul_1160, relu_15, primals_67, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1160 = primals_67 = None
    getitem_678: "f32[8, 256, 56, 56]" = convolution_backward_62[0]
    getitem_679: "f32[512, 256, 1, 1]" = convolution_backward_62[1];  convolution_backward_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    unsqueeze_1096: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_63, 0);  squeeze_63 = None
    unsqueeze_1097: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1096, 2);  unsqueeze_1096 = None
    unsqueeze_1098: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1097, 3);  unsqueeze_1097 = None
    sum_128: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_60, [0, 2, 3])
    sub_337: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_1098)
    mul_1162: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(where_60, sub_337);  sub_337 = None
    sum_129: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1162, [0, 2, 3]);  mul_1162 = None
    mul_1163: "f32[512]" = torch.ops.aten.mul.Tensor(sum_128, 0.00015943877551020407)
    unsqueeze_1099: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1163, 0);  mul_1163 = None
    unsqueeze_1100: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1099, 2);  unsqueeze_1099 = None
    unsqueeze_1101: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1100, 3);  unsqueeze_1100 = None
    mul_1164: "f32[512]" = torch.ops.aten.mul.Tensor(sum_129, 0.00015943877551020407)
    mul_1165: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_64, squeeze_64)
    mul_1166: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1164, mul_1165);  mul_1164 = mul_1165 = None
    unsqueeze_1102: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1166, 0);  mul_1166 = None
    unsqueeze_1103: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1102, 2);  unsqueeze_1102 = None
    unsqueeze_1104: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1103, 3);  unsqueeze_1103 = None
    mul_1167: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_64, primals_65);  primals_65 = None
    unsqueeze_1105: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1167, 0);  mul_1167 = None
    unsqueeze_1106: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1105, 2);  unsqueeze_1105 = None
    unsqueeze_1107: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1106, 3);  unsqueeze_1106 = None
    sub_338: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_1098);  convolution_21 = unsqueeze_1098 = None
    mul_1168: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_338, unsqueeze_1104);  sub_338 = unsqueeze_1104 = None
    sub_339: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(where_60, mul_1168);  where_60 = mul_1168 = None
    sub_340: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(sub_339, unsqueeze_1101);  sub_339 = unsqueeze_1101 = None
    mul_1169: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_340, unsqueeze_1107);  sub_340 = unsqueeze_1107 = None
    mul_1170: "f32[512]" = torch.ops.aten.mul.Tensor(sum_129, squeeze_64);  sum_129 = squeeze_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_backward_63 = torch.ops.aten.convolution_backward.default(mul_1169, cat_3, primals_64, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1169 = cat_3 = primals_64 = None
    getitem_681: "f32[8, 256, 28, 28]" = convolution_backward_63[0]
    getitem_682: "f32[512, 256, 1, 1]" = convolution_backward_63[1];  convolution_backward_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    slice_49: "f32[8, 64, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_681, 1, 0, 64)
    slice_50: "f32[8, 64, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_681, 1, 64, 128)
    slice_51: "f32[8, 64, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_681, 1, 128, 192)
    slice_52: "f32[8, 64, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_681, 1, 192, 256);  getitem_681 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:99, code: spo.append(self.pool(spx[-1]))
    avg_pool2d_backward_2: "f32[8, 64, 56, 56]" = torch.ops.aten.avg_pool2d_backward.default(slice_52, getitem_123, [3, 3], [2, 2], [1, 1], False, True, None);  slice_52 = getitem_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_265: "f32[8, 64, 28, 28]" = torch.ops.aten.alias.default(relu_19);  relu_19 = None
    alias_266: "f32[8, 64, 28, 28]" = torch.ops.aten.alias.default(alias_265);  alias_265 = None
    le_61: "b8[8, 64, 28, 28]" = torch.ops.aten.le.Scalar(alias_266, 0);  alias_266 = None
    scalar_tensor_61: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_61: "f32[8, 64, 28, 28]" = torch.ops.aten.where.self(le_61, scalar_tensor_61, slice_51);  le_61 = scalar_tensor_61 = slice_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_1108: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_60, 0);  squeeze_60 = None
    unsqueeze_1109: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1108, 2);  unsqueeze_1108 = None
    unsqueeze_1110: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1109, 3);  unsqueeze_1109 = None
    sum_130: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_61, [0, 2, 3])
    sub_341: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_1110)
    mul_1171: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(where_61, sub_341);  sub_341 = None
    sum_131: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1171, [0, 2, 3]);  mul_1171 = None
    mul_1172: "f32[64]" = torch.ops.aten.mul.Tensor(sum_130, 0.00015943877551020407)
    unsqueeze_1111: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1172, 0);  mul_1172 = None
    unsqueeze_1112: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1111, 2);  unsqueeze_1111 = None
    unsqueeze_1113: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1112, 3);  unsqueeze_1112 = None
    mul_1173: "f32[64]" = torch.ops.aten.mul.Tensor(sum_131, 0.00015943877551020407)
    mul_1174: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_61, squeeze_61)
    mul_1175: "f32[64]" = torch.ops.aten.mul.Tensor(mul_1173, mul_1174);  mul_1173 = mul_1174 = None
    unsqueeze_1114: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1175, 0);  mul_1175 = None
    unsqueeze_1115: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1114, 2);  unsqueeze_1114 = None
    unsqueeze_1116: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1115, 3);  unsqueeze_1115 = None
    mul_1176: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_61, primals_62);  primals_62 = None
    unsqueeze_1117: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1176, 0);  mul_1176 = None
    unsqueeze_1118: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1117, 2);  unsqueeze_1117 = None
    unsqueeze_1119: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1118, 3);  unsqueeze_1118 = None
    sub_342: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_1110);  convolution_20 = unsqueeze_1110 = None
    mul_1177: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(sub_342, unsqueeze_1116);  sub_342 = unsqueeze_1116 = None
    sub_343: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(where_61, mul_1177);  where_61 = mul_1177 = None
    sub_344: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(sub_343, unsqueeze_1113);  sub_343 = unsqueeze_1113 = None
    mul_1178: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(sub_344, unsqueeze_1119);  sub_344 = unsqueeze_1119 = None
    mul_1179: "f32[64]" = torch.ops.aten.mul.Tensor(sum_131, squeeze_61);  sum_131 = squeeze_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_64 = torch.ops.aten.convolution_backward.default(mul_1178, getitem_116, primals_61, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_1178 = getitem_116 = primals_61 = None
    getitem_684: "f32[8, 64, 56, 56]" = convolution_backward_64[0]
    getitem_685: "f32[64, 8, 3, 3]" = convolution_backward_64[1];  convolution_backward_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_268: "f32[8, 64, 28, 28]" = torch.ops.aten.alias.default(relu_18);  relu_18 = None
    alias_269: "f32[8, 64, 28, 28]" = torch.ops.aten.alias.default(alias_268);  alias_268 = None
    le_62: "b8[8, 64, 28, 28]" = torch.ops.aten.le.Scalar(alias_269, 0);  alias_269 = None
    scalar_tensor_62: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_62: "f32[8, 64, 28, 28]" = torch.ops.aten.where.self(le_62, scalar_tensor_62, slice_50);  le_62 = scalar_tensor_62 = slice_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_1120: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_57, 0);  squeeze_57 = None
    unsqueeze_1121: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1120, 2);  unsqueeze_1120 = None
    unsqueeze_1122: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1121, 3);  unsqueeze_1121 = None
    sum_132: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_62, [0, 2, 3])
    sub_345: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_1122)
    mul_1180: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(where_62, sub_345);  sub_345 = None
    sum_133: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1180, [0, 2, 3]);  mul_1180 = None
    mul_1181: "f32[64]" = torch.ops.aten.mul.Tensor(sum_132, 0.00015943877551020407)
    unsqueeze_1123: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1181, 0);  mul_1181 = None
    unsqueeze_1124: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1123, 2);  unsqueeze_1123 = None
    unsqueeze_1125: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1124, 3);  unsqueeze_1124 = None
    mul_1182: "f32[64]" = torch.ops.aten.mul.Tensor(sum_133, 0.00015943877551020407)
    mul_1183: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_58, squeeze_58)
    mul_1184: "f32[64]" = torch.ops.aten.mul.Tensor(mul_1182, mul_1183);  mul_1182 = mul_1183 = None
    unsqueeze_1126: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1184, 0);  mul_1184 = None
    unsqueeze_1127: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1126, 2);  unsqueeze_1126 = None
    unsqueeze_1128: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1127, 3);  unsqueeze_1127 = None
    mul_1185: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_58, primals_59);  primals_59 = None
    unsqueeze_1129: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1185, 0);  mul_1185 = None
    unsqueeze_1130: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1129, 2);  unsqueeze_1129 = None
    unsqueeze_1131: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1130, 3);  unsqueeze_1130 = None
    sub_346: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_1122);  convolution_19 = unsqueeze_1122 = None
    mul_1186: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(sub_346, unsqueeze_1128);  sub_346 = unsqueeze_1128 = None
    sub_347: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(where_62, mul_1186);  where_62 = mul_1186 = None
    sub_348: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(sub_347, unsqueeze_1125);  sub_347 = unsqueeze_1125 = None
    mul_1187: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(sub_348, unsqueeze_1131);  sub_348 = unsqueeze_1131 = None
    mul_1188: "f32[64]" = torch.ops.aten.mul.Tensor(sum_133, squeeze_58);  sum_133 = squeeze_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_65 = torch.ops.aten.convolution_backward.default(mul_1187, getitem_109, primals_58, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_1187 = getitem_109 = primals_58 = None
    getitem_687: "f32[8, 64, 56, 56]" = convolution_backward_65[0]
    getitem_688: "f32[64, 8, 3, 3]" = convolution_backward_65[1];  convolution_backward_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_271: "f32[8, 64, 28, 28]" = torch.ops.aten.alias.default(relu_17);  relu_17 = None
    alias_272: "f32[8, 64, 28, 28]" = torch.ops.aten.alias.default(alias_271);  alias_271 = None
    le_63: "b8[8, 64, 28, 28]" = torch.ops.aten.le.Scalar(alias_272, 0);  alias_272 = None
    scalar_tensor_63: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_63: "f32[8, 64, 28, 28]" = torch.ops.aten.where.self(le_63, scalar_tensor_63, slice_49);  le_63 = scalar_tensor_63 = slice_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_1132: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_54, 0);  squeeze_54 = None
    unsqueeze_1133: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1132, 2);  unsqueeze_1132 = None
    unsqueeze_1134: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1133, 3);  unsqueeze_1133 = None
    sum_134: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_63, [0, 2, 3])
    sub_349: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_1134)
    mul_1189: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(where_63, sub_349);  sub_349 = None
    sum_135: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1189, [0, 2, 3]);  mul_1189 = None
    mul_1190: "f32[64]" = torch.ops.aten.mul.Tensor(sum_134, 0.00015943877551020407)
    unsqueeze_1135: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1190, 0);  mul_1190 = None
    unsqueeze_1136: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1135, 2);  unsqueeze_1135 = None
    unsqueeze_1137: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1136, 3);  unsqueeze_1136 = None
    mul_1191: "f32[64]" = torch.ops.aten.mul.Tensor(sum_135, 0.00015943877551020407)
    mul_1192: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_55, squeeze_55)
    mul_1193: "f32[64]" = torch.ops.aten.mul.Tensor(mul_1191, mul_1192);  mul_1191 = mul_1192 = None
    unsqueeze_1138: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1193, 0);  mul_1193 = None
    unsqueeze_1139: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1138, 2);  unsqueeze_1138 = None
    unsqueeze_1140: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1139, 3);  unsqueeze_1139 = None
    mul_1194: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_55, primals_56);  primals_56 = None
    unsqueeze_1141: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1194, 0);  mul_1194 = None
    unsqueeze_1142: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1141, 2);  unsqueeze_1141 = None
    unsqueeze_1143: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1142, 3);  unsqueeze_1142 = None
    sub_350: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_1134);  convolution_18 = unsqueeze_1134 = None
    mul_1195: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(sub_350, unsqueeze_1140);  sub_350 = unsqueeze_1140 = None
    sub_351: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(where_63, mul_1195);  where_63 = mul_1195 = None
    sub_352: "f32[8, 64, 28, 28]" = torch.ops.aten.sub.Tensor(sub_351, unsqueeze_1137);  sub_351 = unsqueeze_1137 = None
    mul_1196: "f32[8, 64, 28, 28]" = torch.ops.aten.mul.Tensor(sub_352, unsqueeze_1143);  sub_352 = unsqueeze_1143 = None
    mul_1197: "f32[64]" = torch.ops.aten.mul.Tensor(sum_135, squeeze_55);  sum_135 = squeeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_66 = torch.ops.aten.convolution_backward.default(mul_1196, getitem_102, primals_55, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_1196 = getitem_102 = primals_55 = None
    getitem_690: "f32[8, 64, 56, 56]" = convolution_backward_66[0]
    getitem_691: "f32[64, 8, 3, 3]" = convolution_backward_66[1];  convolution_backward_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    cat_28: "f32[8, 256, 56, 56]" = torch.ops.aten.cat.default([getitem_690, getitem_687, getitem_684, avg_pool2d_backward_2], 1);  getitem_690 = getitem_687 = getitem_684 = avg_pool2d_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    alias_274: "f32[8, 256, 56, 56]" = torch.ops.aten.alias.default(relu_16);  relu_16 = None
    alias_275: "f32[8, 256, 56, 56]" = torch.ops.aten.alias.default(alias_274);  alias_274 = None
    le_64: "b8[8, 256, 56, 56]" = torch.ops.aten.le.Scalar(alias_275, 0);  alias_275 = None
    scalar_tensor_64: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_64: "f32[8, 256, 56, 56]" = torch.ops.aten.where.self(le_64, scalar_tensor_64, cat_28);  le_64 = scalar_tensor_64 = cat_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    unsqueeze_1144: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_51, 0);  squeeze_51 = None
    unsqueeze_1145: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1144, 2);  unsqueeze_1144 = None
    unsqueeze_1146: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1145, 3);  unsqueeze_1145 = None
    sum_136: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_64, [0, 2, 3])
    sub_353: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_1146)
    mul_1198: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(where_64, sub_353);  sub_353 = None
    sum_137: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1198, [0, 2, 3]);  mul_1198 = None
    mul_1199: "f32[256]" = torch.ops.aten.mul.Tensor(sum_136, 3.985969387755102e-05)
    unsqueeze_1147: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1199, 0);  mul_1199 = None
    unsqueeze_1148: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1147, 2);  unsqueeze_1147 = None
    unsqueeze_1149: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1148, 3);  unsqueeze_1148 = None
    mul_1200: "f32[256]" = torch.ops.aten.mul.Tensor(sum_137, 3.985969387755102e-05)
    mul_1201: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_52, squeeze_52)
    mul_1202: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1200, mul_1201);  mul_1200 = mul_1201 = None
    unsqueeze_1150: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1202, 0);  mul_1202 = None
    unsqueeze_1151: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1150, 2);  unsqueeze_1150 = None
    unsqueeze_1152: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1151, 3);  unsqueeze_1151 = None
    mul_1203: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_52, primals_53);  primals_53 = None
    unsqueeze_1153: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1203, 0);  mul_1203 = None
    unsqueeze_1154: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1153, 2);  unsqueeze_1153 = None
    unsqueeze_1155: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1154, 3);  unsqueeze_1154 = None
    sub_354: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_1146);  convolution_17 = unsqueeze_1146 = None
    mul_1204: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_354, unsqueeze_1152);  sub_354 = unsqueeze_1152 = None
    sub_355: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(where_64, mul_1204);  where_64 = mul_1204 = None
    sub_356: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(sub_355, unsqueeze_1149);  sub_355 = unsqueeze_1149 = None
    mul_1205: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_356, unsqueeze_1155);  sub_356 = unsqueeze_1155 = None
    mul_1206: "f32[256]" = torch.ops.aten.mul.Tensor(sum_137, squeeze_52);  sum_137 = squeeze_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_backward_67 = torch.ops.aten.convolution_backward.default(mul_1205, relu_15, primals_52, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1205 = primals_52 = None
    getitem_693: "f32[8, 256, 56, 56]" = convolution_backward_67[0]
    getitem_694: "f32[256, 256, 1, 1]" = convolution_backward_67[1];  convolution_backward_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    add_497: "f32[8, 256, 56, 56]" = torch.ops.aten.add.Tensor(getitem_678, getitem_693);  getitem_678 = getitem_693 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    alias_277: "f32[8, 256, 56, 56]" = torch.ops.aten.alias.default(relu_15);  relu_15 = None
    alias_278: "f32[8, 256, 56, 56]" = torch.ops.aten.alias.default(alias_277);  alias_277 = None
    le_65: "b8[8, 256, 56, 56]" = torch.ops.aten.le.Scalar(alias_278, 0);  alias_278 = None
    scalar_tensor_65: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_65: "f32[8, 256, 56, 56]" = torch.ops.aten.where.self(le_65, scalar_tensor_65, add_497);  le_65 = scalar_tensor_65 = add_497 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    unsqueeze_1156: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_48, 0);  squeeze_48 = None
    unsqueeze_1157: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1156, 2);  unsqueeze_1156 = None
    unsqueeze_1158: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1157, 3);  unsqueeze_1157 = None
    sum_138: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_65, [0, 2, 3])
    sub_357: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_1158)
    mul_1207: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(where_65, sub_357);  sub_357 = None
    sum_139: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1207, [0, 2, 3]);  mul_1207 = None
    mul_1208: "f32[256]" = torch.ops.aten.mul.Tensor(sum_138, 3.985969387755102e-05)
    unsqueeze_1159: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1208, 0);  mul_1208 = None
    unsqueeze_1160: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1159, 2);  unsqueeze_1159 = None
    unsqueeze_1161: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1160, 3);  unsqueeze_1160 = None
    mul_1209: "f32[256]" = torch.ops.aten.mul.Tensor(sum_139, 3.985969387755102e-05)
    mul_1210: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_49, squeeze_49)
    mul_1211: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1209, mul_1210);  mul_1209 = mul_1210 = None
    unsqueeze_1162: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1211, 0);  mul_1211 = None
    unsqueeze_1163: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1162, 2);  unsqueeze_1162 = None
    unsqueeze_1164: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1163, 3);  unsqueeze_1163 = None
    mul_1212: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_49, primals_50);  primals_50 = None
    unsqueeze_1165: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1212, 0);  mul_1212 = None
    unsqueeze_1166: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1165, 2);  unsqueeze_1165 = None
    unsqueeze_1167: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1166, 3);  unsqueeze_1166 = None
    sub_358: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_1158);  convolution_16 = unsqueeze_1158 = None
    mul_1213: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_358, unsqueeze_1164);  sub_358 = unsqueeze_1164 = None
    sub_359: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(where_65, mul_1213);  mul_1213 = None
    sub_360: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(sub_359, unsqueeze_1161);  sub_359 = unsqueeze_1161 = None
    mul_1214: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_360, unsqueeze_1167);  sub_360 = unsqueeze_1167 = None
    mul_1215: "f32[256]" = torch.ops.aten.mul.Tensor(sum_139, squeeze_49);  sum_139 = squeeze_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_backward_68 = torch.ops.aten.convolution_backward.default(mul_1214, cat_2, primals_49, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1214 = cat_2 = primals_49 = None
    getitem_696: "f32[8, 128, 56, 56]" = convolution_backward_68[0]
    getitem_697: "f32[256, 128, 1, 1]" = convolution_backward_68[1];  convolution_backward_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    slice_53: "f32[8, 32, 56, 56]" = torch.ops.aten.slice.Tensor(getitem_696, 1, 0, 32)
    slice_54: "f32[8, 32, 56, 56]" = torch.ops.aten.slice.Tensor(getitem_696, 1, 32, 64)
    slice_55: "f32[8, 32, 56, 56]" = torch.ops.aten.slice.Tensor(getitem_696, 1, 64, 96)
    slice_56: "f32[8, 32, 56, 56]" = torch.ops.aten.slice.Tensor(getitem_696, 1, 96, 128);  getitem_696 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_280: "f32[8, 32, 56, 56]" = torch.ops.aten.alias.default(relu_14);  relu_14 = None
    alias_281: "f32[8, 32, 56, 56]" = torch.ops.aten.alias.default(alias_280);  alias_280 = None
    le_66: "b8[8, 32, 56, 56]" = torch.ops.aten.le.Scalar(alias_281, 0);  alias_281 = None
    scalar_tensor_66: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_66: "f32[8, 32, 56, 56]" = torch.ops.aten.where.self(le_66, scalar_tensor_66, slice_55);  le_66 = scalar_tensor_66 = slice_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_1168: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze_45, 0);  squeeze_45 = None
    unsqueeze_1169: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1168, 2);  unsqueeze_1168 = None
    unsqueeze_1170: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1169, 3);  unsqueeze_1169 = None
    sum_140: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_66, [0, 2, 3])
    sub_361: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_1170)
    mul_1216: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(where_66, sub_361);  sub_361 = None
    sum_141: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_1216, [0, 2, 3]);  mul_1216 = None
    mul_1217: "f32[32]" = torch.ops.aten.mul.Tensor(sum_140, 3.985969387755102e-05)
    unsqueeze_1171: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1217, 0);  mul_1217 = None
    unsqueeze_1172: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1171, 2);  unsqueeze_1171 = None
    unsqueeze_1173: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1172, 3);  unsqueeze_1172 = None
    mul_1218: "f32[32]" = torch.ops.aten.mul.Tensor(sum_141, 3.985969387755102e-05)
    mul_1219: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_46, squeeze_46)
    mul_1220: "f32[32]" = torch.ops.aten.mul.Tensor(mul_1218, mul_1219);  mul_1218 = mul_1219 = None
    unsqueeze_1174: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1220, 0);  mul_1220 = None
    unsqueeze_1175: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1174, 2);  unsqueeze_1174 = None
    unsqueeze_1176: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1175, 3);  unsqueeze_1175 = None
    mul_1221: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_46, primals_47);  primals_47 = None
    unsqueeze_1177: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1221, 0);  mul_1221 = None
    unsqueeze_1178: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1177, 2);  unsqueeze_1177 = None
    unsqueeze_1179: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1178, 3);  unsqueeze_1178 = None
    sub_362: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_1170);  convolution_15 = unsqueeze_1170 = None
    mul_1222: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(sub_362, unsqueeze_1176);  sub_362 = unsqueeze_1176 = None
    sub_363: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(where_66, mul_1222);  where_66 = mul_1222 = None
    sub_364: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(sub_363, unsqueeze_1173);  sub_363 = unsqueeze_1173 = None
    mul_1223: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(sub_364, unsqueeze_1179);  sub_364 = unsqueeze_1179 = None
    mul_1224: "f32[32]" = torch.ops.aten.mul.Tensor(sum_141, squeeze_46);  sum_141 = squeeze_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_69 = torch.ops.aten.convolution_backward.default(mul_1223, add_80, primals_46, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_1223 = add_80 = primals_46 = None
    getitem_699: "f32[8, 32, 56, 56]" = convolution_backward_69[0]
    getitem_700: "f32[32, 4, 3, 3]" = convolution_backward_69[1];  convolution_backward_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_498: "f32[8, 32, 56, 56]" = torch.ops.aten.add.Tensor(slice_54, getitem_699);  slice_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_283: "f32[8, 32, 56, 56]" = torch.ops.aten.alias.default(relu_13);  relu_13 = None
    alias_284: "f32[8, 32, 56, 56]" = torch.ops.aten.alias.default(alias_283);  alias_283 = None
    le_67: "b8[8, 32, 56, 56]" = torch.ops.aten.le.Scalar(alias_284, 0);  alias_284 = None
    scalar_tensor_67: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_67: "f32[8, 32, 56, 56]" = torch.ops.aten.where.self(le_67, scalar_tensor_67, add_498);  le_67 = scalar_tensor_67 = add_498 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_1180: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze_42, 0);  squeeze_42 = None
    unsqueeze_1181: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1180, 2);  unsqueeze_1180 = None
    unsqueeze_1182: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1181, 3);  unsqueeze_1181 = None
    sum_142: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_67, [0, 2, 3])
    sub_365: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_1182)
    mul_1225: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(where_67, sub_365);  sub_365 = None
    sum_143: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_1225, [0, 2, 3]);  mul_1225 = None
    mul_1226: "f32[32]" = torch.ops.aten.mul.Tensor(sum_142, 3.985969387755102e-05)
    unsqueeze_1183: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1226, 0);  mul_1226 = None
    unsqueeze_1184: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1183, 2);  unsqueeze_1183 = None
    unsqueeze_1185: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1184, 3);  unsqueeze_1184 = None
    mul_1227: "f32[32]" = torch.ops.aten.mul.Tensor(sum_143, 3.985969387755102e-05)
    mul_1228: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
    mul_1229: "f32[32]" = torch.ops.aten.mul.Tensor(mul_1227, mul_1228);  mul_1227 = mul_1228 = None
    unsqueeze_1186: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1229, 0);  mul_1229 = None
    unsqueeze_1187: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1186, 2);  unsqueeze_1186 = None
    unsqueeze_1188: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1187, 3);  unsqueeze_1187 = None
    mul_1230: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_43, primals_44);  primals_44 = None
    unsqueeze_1189: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1230, 0);  mul_1230 = None
    unsqueeze_1190: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1189, 2);  unsqueeze_1189 = None
    unsqueeze_1191: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1190, 3);  unsqueeze_1190 = None
    sub_366: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_1182);  convolution_14 = unsqueeze_1182 = None
    mul_1231: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(sub_366, unsqueeze_1188);  sub_366 = unsqueeze_1188 = None
    sub_367: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(where_67, mul_1231);  where_67 = mul_1231 = None
    sub_368: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(sub_367, unsqueeze_1185);  sub_367 = unsqueeze_1185 = None
    mul_1232: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(sub_368, unsqueeze_1191);  sub_368 = unsqueeze_1191 = None
    mul_1233: "f32[32]" = torch.ops.aten.mul.Tensor(sum_143, squeeze_43);  sum_143 = squeeze_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_70 = torch.ops.aten.convolution_backward.default(mul_1232, add_74, primals_43, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_1232 = add_74 = primals_43 = None
    getitem_702: "f32[8, 32, 56, 56]" = convolution_backward_70[0]
    getitem_703: "f32[32, 4, 3, 3]" = convolution_backward_70[1];  convolution_backward_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_499: "f32[8, 32, 56, 56]" = torch.ops.aten.add.Tensor(slice_53, getitem_702);  slice_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_286: "f32[8, 32, 56, 56]" = torch.ops.aten.alias.default(relu_12);  relu_12 = None
    alias_287: "f32[8, 32, 56, 56]" = torch.ops.aten.alias.default(alias_286);  alias_286 = None
    le_68: "b8[8, 32, 56, 56]" = torch.ops.aten.le.Scalar(alias_287, 0);  alias_287 = None
    scalar_tensor_68: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_68: "f32[8, 32, 56, 56]" = torch.ops.aten.where.self(le_68, scalar_tensor_68, add_499);  le_68 = scalar_tensor_68 = add_499 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_1192: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze_39, 0);  squeeze_39 = None
    unsqueeze_1193: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1192, 2);  unsqueeze_1192 = None
    unsqueeze_1194: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1193, 3);  unsqueeze_1193 = None
    sum_144: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_68, [0, 2, 3])
    sub_369: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_1194)
    mul_1234: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(where_68, sub_369);  sub_369 = None
    sum_145: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_1234, [0, 2, 3]);  mul_1234 = None
    mul_1235: "f32[32]" = torch.ops.aten.mul.Tensor(sum_144, 3.985969387755102e-05)
    unsqueeze_1195: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1235, 0);  mul_1235 = None
    unsqueeze_1196: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1195, 2);  unsqueeze_1195 = None
    unsqueeze_1197: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1196, 3);  unsqueeze_1196 = None
    mul_1236: "f32[32]" = torch.ops.aten.mul.Tensor(sum_145, 3.985969387755102e-05)
    mul_1237: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_40, squeeze_40)
    mul_1238: "f32[32]" = torch.ops.aten.mul.Tensor(mul_1236, mul_1237);  mul_1236 = mul_1237 = None
    unsqueeze_1198: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1238, 0);  mul_1238 = None
    unsqueeze_1199: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1198, 2);  unsqueeze_1198 = None
    unsqueeze_1200: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1199, 3);  unsqueeze_1199 = None
    mul_1239: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_40, primals_41);  primals_41 = None
    unsqueeze_1201: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1239, 0);  mul_1239 = None
    unsqueeze_1202: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1201, 2);  unsqueeze_1201 = None
    unsqueeze_1203: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1202, 3);  unsqueeze_1202 = None
    sub_370: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_1194);  convolution_13 = unsqueeze_1194 = None
    mul_1240: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(sub_370, unsqueeze_1200);  sub_370 = unsqueeze_1200 = None
    sub_371: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(where_68, mul_1240);  where_68 = mul_1240 = None
    sub_372: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(sub_371, unsqueeze_1197);  sub_371 = unsqueeze_1197 = None
    mul_1241: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(sub_372, unsqueeze_1203);  sub_372 = unsqueeze_1203 = None
    mul_1242: "f32[32]" = torch.ops.aten.mul.Tensor(sum_145, squeeze_40);  sum_145 = squeeze_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_71 = torch.ops.aten.convolution_backward.default(mul_1241, getitem_72, primals_40, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_1241 = getitem_72 = primals_40 = None
    getitem_705: "f32[8, 32, 56, 56]" = convolution_backward_71[0]
    getitem_706: "f32[32, 4, 3, 3]" = convolution_backward_71[1];  convolution_backward_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    cat_29: "f32[8, 128, 56, 56]" = torch.ops.aten.cat.default([getitem_705, getitem_702, getitem_699, slice_56], 1);  getitem_705 = getitem_702 = getitem_699 = slice_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    alias_289: "f32[8, 128, 56, 56]" = torch.ops.aten.alias.default(relu_11);  relu_11 = None
    alias_290: "f32[8, 128, 56, 56]" = torch.ops.aten.alias.default(alias_289);  alias_289 = None
    le_69: "b8[8, 128, 56, 56]" = torch.ops.aten.le.Scalar(alias_290, 0);  alias_290 = None
    scalar_tensor_69: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_69: "f32[8, 128, 56, 56]" = torch.ops.aten.where.self(le_69, scalar_tensor_69, cat_29);  le_69 = scalar_tensor_69 = cat_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    unsqueeze_1204: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_36, 0);  squeeze_36 = None
    unsqueeze_1205: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1204, 2);  unsqueeze_1204 = None
    unsqueeze_1206: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1205, 3);  unsqueeze_1205 = None
    sum_146: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_69, [0, 2, 3])
    sub_373: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_1206)
    mul_1243: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(where_69, sub_373);  sub_373 = None
    sum_147: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1243, [0, 2, 3]);  mul_1243 = None
    mul_1244: "f32[128]" = torch.ops.aten.mul.Tensor(sum_146, 3.985969387755102e-05)
    unsqueeze_1207: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1244, 0);  mul_1244 = None
    unsqueeze_1208: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1207, 2);  unsqueeze_1207 = None
    unsqueeze_1209: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1208, 3);  unsqueeze_1208 = None
    mul_1245: "f32[128]" = torch.ops.aten.mul.Tensor(sum_147, 3.985969387755102e-05)
    mul_1246: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
    mul_1247: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1245, mul_1246);  mul_1245 = mul_1246 = None
    unsqueeze_1210: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1247, 0);  mul_1247 = None
    unsqueeze_1211: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1210, 2);  unsqueeze_1210 = None
    unsqueeze_1212: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1211, 3);  unsqueeze_1211 = None
    mul_1248: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_37, primals_38);  primals_38 = None
    unsqueeze_1213: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1248, 0);  mul_1248 = None
    unsqueeze_1214: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1213, 2);  unsqueeze_1213 = None
    unsqueeze_1215: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1214, 3);  unsqueeze_1214 = None
    sub_374: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_1206);  convolution_12 = unsqueeze_1206 = None
    mul_1249: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_374, unsqueeze_1212);  sub_374 = unsqueeze_1212 = None
    sub_375: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(where_69, mul_1249);  where_69 = mul_1249 = None
    sub_376: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(sub_375, unsqueeze_1209);  sub_375 = unsqueeze_1209 = None
    mul_1250: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_376, unsqueeze_1215);  sub_376 = unsqueeze_1215 = None
    mul_1251: "f32[128]" = torch.ops.aten.mul.Tensor(sum_147, squeeze_37);  sum_147 = squeeze_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_backward_72 = torch.ops.aten.convolution_backward.default(mul_1250, relu_10, primals_37, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1250 = primals_37 = None
    getitem_708: "f32[8, 256, 56, 56]" = convolution_backward_72[0]
    getitem_709: "f32[128, 256, 1, 1]" = convolution_backward_72[1];  convolution_backward_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    add_500: "f32[8, 256, 56, 56]" = torch.ops.aten.add.Tensor(where_65, getitem_708);  where_65 = getitem_708 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    alias_292: "f32[8, 256, 56, 56]" = torch.ops.aten.alias.default(relu_10);  relu_10 = None
    alias_293: "f32[8, 256, 56, 56]" = torch.ops.aten.alias.default(alias_292);  alias_292 = None
    le_70: "b8[8, 256, 56, 56]" = torch.ops.aten.le.Scalar(alias_293, 0);  alias_293 = None
    scalar_tensor_70: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_70: "f32[8, 256, 56, 56]" = torch.ops.aten.where.self(le_70, scalar_tensor_70, add_500);  le_70 = scalar_tensor_70 = add_500 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    unsqueeze_1216: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_33, 0);  squeeze_33 = None
    unsqueeze_1217: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1216, 2);  unsqueeze_1216 = None
    unsqueeze_1218: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1217, 3);  unsqueeze_1217 = None
    sum_148: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_70, [0, 2, 3])
    sub_377: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_1218)
    mul_1252: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(where_70, sub_377);  sub_377 = None
    sum_149: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1252, [0, 2, 3]);  mul_1252 = None
    mul_1253: "f32[256]" = torch.ops.aten.mul.Tensor(sum_148, 3.985969387755102e-05)
    unsqueeze_1219: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1253, 0);  mul_1253 = None
    unsqueeze_1220: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1219, 2);  unsqueeze_1219 = None
    unsqueeze_1221: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1220, 3);  unsqueeze_1220 = None
    mul_1254: "f32[256]" = torch.ops.aten.mul.Tensor(sum_149, 3.985969387755102e-05)
    mul_1255: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_34, squeeze_34)
    mul_1256: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1254, mul_1255);  mul_1254 = mul_1255 = None
    unsqueeze_1222: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1256, 0);  mul_1256 = None
    unsqueeze_1223: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1222, 2);  unsqueeze_1222 = None
    unsqueeze_1224: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1223, 3);  unsqueeze_1223 = None
    mul_1257: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_34, primals_35);  primals_35 = None
    unsqueeze_1225: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1257, 0);  mul_1257 = None
    unsqueeze_1226: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1225, 2);  unsqueeze_1225 = None
    unsqueeze_1227: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1226, 3);  unsqueeze_1226 = None
    sub_378: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_1218);  convolution_11 = unsqueeze_1218 = None
    mul_1258: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_378, unsqueeze_1224);  sub_378 = unsqueeze_1224 = None
    sub_379: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(where_70, mul_1258);  mul_1258 = None
    sub_380: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(sub_379, unsqueeze_1221);  sub_379 = unsqueeze_1221 = None
    mul_1259: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_380, unsqueeze_1227);  sub_380 = unsqueeze_1227 = None
    mul_1260: "f32[256]" = torch.ops.aten.mul.Tensor(sum_149, squeeze_34);  sum_149 = squeeze_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_backward_73 = torch.ops.aten.convolution_backward.default(mul_1259, cat_1, primals_34, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1259 = cat_1 = primals_34 = None
    getitem_711: "f32[8, 128, 56, 56]" = convolution_backward_73[0]
    getitem_712: "f32[256, 128, 1, 1]" = convolution_backward_73[1];  convolution_backward_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    slice_57: "f32[8, 32, 56, 56]" = torch.ops.aten.slice.Tensor(getitem_711, 1, 0, 32)
    slice_58: "f32[8, 32, 56, 56]" = torch.ops.aten.slice.Tensor(getitem_711, 1, 32, 64)
    slice_59: "f32[8, 32, 56, 56]" = torch.ops.aten.slice.Tensor(getitem_711, 1, 64, 96)
    slice_60: "f32[8, 32, 56, 56]" = torch.ops.aten.slice.Tensor(getitem_711, 1, 96, 128);  getitem_711 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_295: "f32[8, 32, 56, 56]" = torch.ops.aten.alias.default(relu_9);  relu_9 = None
    alias_296: "f32[8, 32, 56, 56]" = torch.ops.aten.alias.default(alias_295);  alias_295 = None
    le_71: "b8[8, 32, 56, 56]" = torch.ops.aten.le.Scalar(alias_296, 0);  alias_296 = None
    scalar_tensor_71: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_71: "f32[8, 32, 56, 56]" = torch.ops.aten.where.self(le_71, scalar_tensor_71, slice_59);  le_71 = scalar_tensor_71 = slice_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_1228: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze_30, 0);  squeeze_30 = None
    unsqueeze_1229: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1228, 2);  unsqueeze_1228 = None
    unsqueeze_1230: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1229, 3);  unsqueeze_1229 = None
    sum_150: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_71, [0, 2, 3])
    sub_381: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_1230)
    mul_1261: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(where_71, sub_381);  sub_381 = None
    sum_151: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_1261, [0, 2, 3]);  mul_1261 = None
    mul_1262: "f32[32]" = torch.ops.aten.mul.Tensor(sum_150, 3.985969387755102e-05)
    unsqueeze_1231: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1262, 0);  mul_1262 = None
    unsqueeze_1232: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1231, 2);  unsqueeze_1231 = None
    unsqueeze_1233: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1232, 3);  unsqueeze_1232 = None
    mul_1263: "f32[32]" = torch.ops.aten.mul.Tensor(sum_151, 3.985969387755102e-05)
    mul_1264: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
    mul_1265: "f32[32]" = torch.ops.aten.mul.Tensor(mul_1263, mul_1264);  mul_1263 = mul_1264 = None
    unsqueeze_1234: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1265, 0);  mul_1265 = None
    unsqueeze_1235: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1234, 2);  unsqueeze_1234 = None
    unsqueeze_1236: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1235, 3);  unsqueeze_1235 = None
    mul_1266: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_31, primals_32);  primals_32 = None
    unsqueeze_1237: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1266, 0);  mul_1266 = None
    unsqueeze_1238: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1237, 2);  unsqueeze_1237 = None
    unsqueeze_1239: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1238, 3);  unsqueeze_1238 = None
    sub_382: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_1230);  convolution_10 = unsqueeze_1230 = None
    mul_1267: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(sub_382, unsqueeze_1236);  sub_382 = unsqueeze_1236 = None
    sub_383: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(where_71, mul_1267);  where_71 = mul_1267 = None
    sub_384: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(sub_383, unsqueeze_1233);  sub_383 = unsqueeze_1233 = None
    mul_1268: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(sub_384, unsqueeze_1239);  sub_384 = unsqueeze_1239 = None
    mul_1269: "f32[32]" = torch.ops.aten.mul.Tensor(sum_151, squeeze_31);  sum_151 = squeeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_74 = torch.ops.aten.convolution_backward.default(mul_1268, add_52, primals_31, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_1268 = add_52 = primals_31 = None
    getitem_714: "f32[8, 32, 56, 56]" = convolution_backward_74[0]
    getitem_715: "f32[32, 4, 3, 3]" = convolution_backward_74[1];  convolution_backward_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_501: "f32[8, 32, 56, 56]" = torch.ops.aten.add.Tensor(slice_58, getitem_714);  slice_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_298: "f32[8, 32, 56, 56]" = torch.ops.aten.alias.default(relu_8);  relu_8 = None
    alias_299: "f32[8, 32, 56, 56]" = torch.ops.aten.alias.default(alias_298);  alias_298 = None
    le_72: "b8[8, 32, 56, 56]" = torch.ops.aten.le.Scalar(alias_299, 0);  alias_299 = None
    scalar_tensor_72: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_72: "f32[8, 32, 56, 56]" = torch.ops.aten.where.self(le_72, scalar_tensor_72, add_501);  le_72 = scalar_tensor_72 = add_501 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_1240: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze_27, 0);  squeeze_27 = None
    unsqueeze_1241: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1240, 2);  unsqueeze_1240 = None
    unsqueeze_1242: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1241, 3);  unsqueeze_1241 = None
    sum_152: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_72, [0, 2, 3])
    sub_385: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_1242)
    mul_1270: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(where_72, sub_385);  sub_385 = None
    sum_153: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_1270, [0, 2, 3]);  mul_1270 = None
    mul_1271: "f32[32]" = torch.ops.aten.mul.Tensor(sum_152, 3.985969387755102e-05)
    unsqueeze_1243: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1271, 0);  mul_1271 = None
    unsqueeze_1244: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1243, 2);  unsqueeze_1243 = None
    unsqueeze_1245: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1244, 3);  unsqueeze_1244 = None
    mul_1272: "f32[32]" = torch.ops.aten.mul.Tensor(sum_153, 3.985969387755102e-05)
    mul_1273: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_28, squeeze_28)
    mul_1274: "f32[32]" = torch.ops.aten.mul.Tensor(mul_1272, mul_1273);  mul_1272 = mul_1273 = None
    unsqueeze_1246: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1274, 0);  mul_1274 = None
    unsqueeze_1247: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1246, 2);  unsqueeze_1246 = None
    unsqueeze_1248: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1247, 3);  unsqueeze_1247 = None
    mul_1275: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_28, primals_29);  primals_29 = None
    unsqueeze_1249: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1275, 0);  mul_1275 = None
    unsqueeze_1250: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1249, 2);  unsqueeze_1249 = None
    unsqueeze_1251: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1250, 3);  unsqueeze_1250 = None
    sub_386: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_1242);  convolution_9 = unsqueeze_1242 = None
    mul_1276: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(sub_386, unsqueeze_1248);  sub_386 = unsqueeze_1248 = None
    sub_387: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(where_72, mul_1276);  where_72 = mul_1276 = None
    sub_388: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(sub_387, unsqueeze_1245);  sub_387 = unsqueeze_1245 = None
    mul_1277: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(sub_388, unsqueeze_1251);  sub_388 = unsqueeze_1251 = None
    mul_1278: "f32[32]" = torch.ops.aten.mul.Tensor(sum_153, squeeze_28);  sum_153 = squeeze_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_75 = torch.ops.aten.convolution_backward.default(mul_1277, add_46, primals_28, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_1277 = add_46 = primals_28 = None
    getitem_717: "f32[8, 32, 56, 56]" = convolution_backward_75[0]
    getitem_718: "f32[32, 4, 3, 3]" = convolution_backward_75[1];  convolution_backward_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    add_502: "f32[8, 32, 56, 56]" = torch.ops.aten.add.Tensor(slice_57, getitem_717);  slice_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_301: "f32[8, 32, 56, 56]" = torch.ops.aten.alias.default(relu_7);  relu_7 = None
    alias_302: "f32[8, 32, 56, 56]" = torch.ops.aten.alias.default(alias_301);  alias_301 = None
    le_73: "b8[8, 32, 56, 56]" = torch.ops.aten.le.Scalar(alias_302, 0);  alias_302 = None
    scalar_tensor_73: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_73: "f32[8, 32, 56, 56]" = torch.ops.aten.where.self(le_73, scalar_tensor_73, add_502);  le_73 = scalar_tensor_73 = add_502 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_1252: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze_24, 0);  squeeze_24 = None
    unsqueeze_1253: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1252, 2);  unsqueeze_1252 = None
    unsqueeze_1254: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1253, 3);  unsqueeze_1253 = None
    sum_154: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_73, [0, 2, 3])
    sub_389: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_1254)
    mul_1279: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(where_73, sub_389);  sub_389 = None
    sum_155: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_1279, [0, 2, 3]);  mul_1279 = None
    mul_1280: "f32[32]" = torch.ops.aten.mul.Tensor(sum_154, 3.985969387755102e-05)
    unsqueeze_1255: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1280, 0);  mul_1280 = None
    unsqueeze_1256: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1255, 2);  unsqueeze_1255 = None
    unsqueeze_1257: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1256, 3);  unsqueeze_1256 = None
    mul_1281: "f32[32]" = torch.ops.aten.mul.Tensor(sum_155, 3.985969387755102e-05)
    mul_1282: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
    mul_1283: "f32[32]" = torch.ops.aten.mul.Tensor(mul_1281, mul_1282);  mul_1281 = mul_1282 = None
    unsqueeze_1258: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1283, 0);  mul_1283 = None
    unsqueeze_1259: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1258, 2);  unsqueeze_1258 = None
    unsqueeze_1260: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1259, 3);  unsqueeze_1259 = None
    mul_1284: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_25, primals_26);  primals_26 = None
    unsqueeze_1261: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1284, 0);  mul_1284 = None
    unsqueeze_1262: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1261, 2);  unsqueeze_1261 = None
    unsqueeze_1263: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1262, 3);  unsqueeze_1262 = None
    sub_390: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_1254);  convolution_8 = unsqueeze_1254 = None
    mul_1285: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(sub_390, unsqueeze_1260);  sub_390 = unsqueeze_1260 = None
    sub_391: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(where_73, mul_1285);  where_73 = mul_1285 = None
    sub_392: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(sub_391, unsqueeze_1257);  sub_391 = unsqueeze_1257 = None
    mul_1286: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(sub_392, unsqueeze_1263);  sub_392 = unsqueeze_1263 = None
    mul_1287: "f32[32]" = torch.ops.aten.mul.Tensor(sum_155, squeeze_25);  sum_155 = squeeze_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_76 = torch.ops.aten.convolution_backward.default(mul_1286, getitem_42, primals_25, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_1286 = getitem_42 = primals_25 = None
    getitem_720: "f32[8, 32, 56, 56]" = convolution_backward_76[0]
    getitem_721: "f32[32, 4, 3, 3]" = convolution_backward_76[1];  convolution_backward_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    cat_30: "f32[8, 128, 56, 56]" = torch.ops.aten.cat.default([getitem_720, getitem_717, getitem_714, slice_60], 1);  getitem_720 = getitem_717 = getitem_714 = slice_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    alias_304: "f32[8, 128, 56, 56]" = torch.ops.aten.alias.default(relu_6);  relu_6 = None
    alias_305: "f32[8, 128, 56, 56]" = torch.ops.aten.alias.default(alias_304);  alias_304 = None
    le_74: "b8[8, 128, 56, 56]" = torch.ops.aten.le.Scalar(alias_305, 0);  alias_305 = None
    scalar_tensor_74: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_74: "f32[8, 128, 56, 56]" = torch.ops.aten.where.self(le_74, scalar_tensor_74, cat_30);  le_74 = scalar_tensor_74 = cat_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    unsqueeze_1264: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_21, 0);  squeeze_21 = None
    unsqueeze_1265: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1264, 2);  unsqueeze_1264 = None
    unsqueeze_1266: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1265, 3);  unsqueeze_1265 = None
    sum_156: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_74, [0, 2, 3])
    sub_393: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_1266)
    mul_1288: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(where_74, sub_393);  sub_393 = None
    sum_157: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1288, [0, 2, 3]);  mul_1288 = None
    mul_1289: "f32[128]" = torch.ops.aten.mul.Tensor(sum_156, 3.985969387755102e-05)
    unsqueeze_1267: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1289, 0);  mul_1289 = None
    unsqueeze_1268: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1267, 2);  unsqueeze_1267 = None
    unsqueeze_1269: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1268, 3);  unsqueeze_1268 = None
    mul_1290: "f32[128]" = torch.ops.aten.mul.Tensor(sum_157, 3.985969387755102e-05)
    mul_1291: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_22, squeeze_22)
    mul_1292: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1290, mul_1291);  mul_1290 = mul_1291 = None
    unsqueeze_1270: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1292, 0);  mul_1292 = None
    unsqueeze_1271: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1270, 2);  unsqueeze_1270 = None
    unsqueeze_1272: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1271, 3);  unsqueeze_1271 = None
    mul_1293: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_22, primals_23);  primals_23 = None
    unsqueeze_1273: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1293, 0);  mul_1293 = None
    unsqueeze_1274: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1273, 2);  unsqueeze_1273 = None
    unsqueeze_1275: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1274, 3);  unsqueeze_1274 = None
    sub_394: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_1266);  convolution_7 = unsqueeze_1266 = None
    mul_1294: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_394, unsqueeze_1272);  sub_394 = unsqueeze_1272 = None
    sub_395: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(where_74, mul_1294);  where_74 = mul_1294 = None
    sub_396: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(sub_395, unsqueeze_1269);  sub_395 = unsqueeze_1269 = None
    mul_1295: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_396, unsqueeze_1275);  sub_396 = unsqueeze_1275 = None
    mul_1296: "f32[128]" = torch.ops.aten.mul.Tensor(sum_157, squeeze_22);  sum_157 = squeeze_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_backward_77 = torch.ops.aten.convolution_backward.default(mul_1295, relu_5, primals_22, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1295 = primals_22 = None
    getitem_723: "f32[8, 256, 56, 56]" = convolution_backward_77[0]
    getitem_724: "f32[128, 256, 1, 1]" = convolution_backward_77[1];  convolution_backward_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    add_503: "f32[8, 256, 56, 56]" = torch.ops.aten.add.Tensor(where_70, getitem_723);  where_70 = getitem_723 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    alias_307: "f32[8, 256, 56, 56]" = torch.ops.aten.alias.default(relu_5);  relu_5 = None
    alias_308: "f32[8, 256, 56, 56]" = torch.ops.aten.alias.default(alias_307);  alias_307 = None
    le_75: "b8[8, 256, 56, 56]" = torch.ops.aten.le.Scalar(alias_308, 0);  alias_308 = None
    scalar_tensor_75: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_75: "f32[8, 256, 56, 56]" = torch.ops.aten.where.self(le_75, scalar_tensor_75, add_503);  le_75 = scalar_tensor_75 = add_503 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:111, code: shortcut = self.downsample(x)
    unsqueeze_1276: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_18, 0);  squeeze_18 = None
    unsqueeze_1277: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1276, 2);  unsqueeze_1276 = None
    unsqueeze_1278: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1277, 3);  unsqueeze_1277 = None
    sum_158: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_75, [0, 2, 3])
    sub_397: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_1278)
    mul_1297: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(where_75, sub_397);  sub_397 = None
    sum_159: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1297, [0, 2, 3]);  mul_1297 = None
    mul_1298: "f32[256]" = torch.ops.aten.mul.Tensor(sum_158, 3.985969387755102e-05)
    unsqueeze_1279: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1298, 0);  mul_1298 = None
    unsqueeze_1280: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1279, 2);  unsqueeze_1279 = None
    unsqueeze_1281: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1280, 3);  unsqueeze_1280 = None
    mul_1299: "f32[256]" = torch.ops.aten.mul.Tensor(sum_159, 3.985969387755102e-05)
    mul_1300: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_19, squeeze_19)
    mul_1301: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1299, mul_1300);  mul_1299 = mul_1300 = None
    unsqueeze_1282: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1301, 0);  mul_1301 = None
    unsqueeze_1283: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1282, 2);  unsqueeze_1282 = None
    unsqueeze_1284: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1283, 3);  unsqueeze_1283 = None
    mul_1302: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_19, primals_20);  primals_20 = None
    unsqueeze_1285: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1302, 0);  mul_1302 = None
    unsqueeze_1286: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1285, 2);  unsqueeze_1285 = None
    unsqueeze_1287: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1286, 3);  unsqueeze_1286 = None
    sub_398: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_1278);  convolution_6 = unsqueeze_1278 = None
    mul_1303: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_398, unsqueeze_1284);  sub_398 = unsqueeze_1284 = None
    sub_399: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(where_75, mul_1303);  mul_1303 = None
    sub_400: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(sub_399, unsqueeze_1281);  sub_399 = unsqueeze_1281 = None
    mul_1304: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_400, unsqueeze_1287);  sub_400 = unsqueeze_1287 = None
    mul_1305: "f32[256]" = torch.ops.aten.mul.Tensor(sum_159, squeeze_19);  sum_159 = squeeze_19 = None
    convolution_backward_78 = torch.ops.aten.convolution_backward.default(mul_1304, getitem_2, primals_19, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1304 = primals_19 = None
    getitem_726: "f32[8, 64, 56, 56]" = convolution_backward_78[0]
    getitem_727: "f32[256, 64, 1, 1]" = convolution_backward_78[1];  convolution_backward_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    unsqueeze_1288: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_15, 0);  squeeze_15 = None
    unsqueeze_1289: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1288, 2);  unsqueeze_1288 = None
    unsqueeze_1290: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1289, 3);  unsqueeze_1289 = None
    sum_160: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_75, [0, 2, 3])
    sub_401: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_1290)
    mul_1306: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(where_75, sub_401);  sub_401 = None
    sum_161: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1306, [0, 2, 3]);  mul_1306 = None
    mul_1307: "f32[256]" = torch.ops.aten.mul.Tensor(sum_160, 3.985969387755102e-05)
    unsqueeze_1291: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1307, 0);  mul_1307 = None
    unsqueeze_1292: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1291, 2);  unsqueeze_1291 = None
    unsqueeze_1293: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1292, 3);  unsqueeze_1292 = None
    mul_1308: "f32[256]" = torch.ops.aten.mul.Tensor(sum_161, 3.985969387755102e-05)
    mul_1309: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_16, squeeze_16)
    mul_1310: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1308, mul_1309);  mul_1308 = mul_1309 = None
    unsqueeze_1294: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1310, 0);  mul_1310 = None
    unsqueeze_1295: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1294, 2);  unsqueeze_1294 = None
    unsqueeze_1296: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1295, 3);  unsqueeze_1295 = None
    mul_1311: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_16, primals_17);  primals_17 = None
    unsqueeze_1297: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1311, 0);  mul_1311 = None
    unsqueeze_1298: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1297, 2);  unsqueeze_1297 = None
    unsqueeze_1299: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1298, 3);  unsqueeze_1298 = None
    sub_402: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_1290);  convolution_5 = unsqueeze_1290 = None
    mul_1312: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_402, unsqueeze_1296);  sub_402 = unsqueeze_1296 = None
    sub_403: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(where_75, mul_1312);  where_75 = mul_1312 = None
    sub_404: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(sub_403, unsqueeze_1293);  sub_403 = unsqueeze_1293 = None
    mul_1313: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_404, unsqueeze_1299);  sub_404 = unsqueeze_1299 = None
    mul_1314: "f32[256]" = torch.ops.aten.mul.Tensor(sum_161, squeeze_16);  sum_161 = squeeze_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_backward_79 = torch.ops.aten.convolution_backward.default(mul_1313, cat, primals_16, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1313 = cat = primals_16 = None
    getitem_729: "f32[8, 128, 56, 56]" = convolution_backward_79[0]
    getitem_730: "f32[256, 128, 1, 1]" = convolution_backward_79[1];  convolution_backward_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    slice_61: "f32[8, 32, 56, 56]" = torch.ops.aten.slice.Tensor(getitem_729, 1, 0, 32)
    slice_62: "f32[8, 32, 56, 56]" = torch.ops.aten.slice.Tensor(getitem_729, 1, 32, 64)
    slice_63: "f32[8, 32, 56, 56]" = torch.ops.aten.slice.Tensor(getitem_729, 1, 64, 96)
    slice_64: "f32[8, 32, 56, 56]" = torch.ops.aten.slice.Tensor(getitem_729, 1, 96, 128);  getitem_729 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:99, code: spo.append(self.pool(spx[-1]))
    avg_pool2d_backward_3: "f32[8, 32, 56, 56]" = torch.ops.aten.avg_pool2d_backward.default(slice_64, getitem_31, [3, 3], [1, 1], [1, 1], False, True, None);  slice_64 = getitem_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_310: "f32[8, 32, 56, 56]" = torch.ops.aten.alias.default(relu_4);  relu_4 = None
    alias_311: "f32[8, 32, 56, 56]" = torch.ops.aten.alias.default(alias_310);  alias_310 = None
    le_76: "b8[8, 32, 56, 56]" = torch.ops.aten.le.Scalar(alias_311, 0);  alias_311 = None
    scalar_tensor_76: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_76: "f32[8, 32, 56, 56]" = torch.ops.aten.where.self(le_76, scalar_tensor_76, slice_63);  le_76 = scalar_tensor_76 = slice_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_1300: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze_12, 0);  squeeze_12 = None
    unsqueeze_1301: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1300, 2);  unsqueeze_1300 = None
    unsqueeze_1302: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1301, 3);  unsqueeze_1301 = None
    sum_162: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_76, [0, 2, 3])
    sub_405: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_1302)
    mul_1315: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(where_76, sub_405);  sub_405 = None
    sum_163: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_1315, [0, 2, 3]);  mul_1315 = None
    mul_1316: "f32[32]" = torch.ops.aten.mul.Tensor(sum_162, 3.985969387755102e-05)
    unsqueeze_1303: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1316, 0);  mul_1316 = None
    unsqueeze_1304: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1303, 2);  unsqueeze_1303 = None
    unsqueeze_1305: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1304, 3);  unsqueeze_1304 = None
    mul_1317: "f32[32]" = torch.ops.aten.mul.Tensor(sum_163, 3.985969387755102e-05)
    mul_1318: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
    mul_1319: "f32[32]" = torch.ops.aten.mul.Tensor(mul_1317, mul_1318);  mul_1317 = mul_1318 = None
    unsqueeze_1306: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1319, 0);  mul_1319 = None
    unsqueeze_1307: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1306, 2);  unsqueeze_1306 = None
    unsqueeze_1308: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1307, 3);  unsqueeze_1307 = None
    mul_1320: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_13, primals_14);  primals_14 = None
    unsqueeze_1309: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1320, 0);  mul_1320 = None
    unsqueeze_1310: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1309, 2);  unsqueeze_1309 = None
    unsqueeze_1311: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1310, 3);  unsqueeze_1310 = None
    sub_406: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_1302);  convolution_4 = unsqueeze_1302 = None
    mul_1321: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(sub_406, unsqueeze_1308);  sub_406 = unsqueeze_1308 = None
    sub_407: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(where_76, mul_1321);  where_76 = mul_1321 = None
    sub_408: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(sub_407, unsqueeze_1305);  sub_407 = unsqueeze_1305 = None
    mul_1322: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(sub_408, unsqueeze_1311);  sub_408 = unsqueeze_1311 = None
    mul_1323: "f32[32]" = torch.ops.aten.mul.Tensor(sum_163, squeeze_13);  sum_163 = squeeze_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_80 = torch.ops.aten.convolution_backward.default(mul_1322, getitem_24, primals_13, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_1322 = getitem_24 = primals_13 = None
    getitem_732: "f32[8, 32, 56, 56]" = convolution_backward_80[0]
    getitem_733: "f32[32, 4, 3, 3]" = convolution_backward_80[1];  convolution_backward_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_313: "f32[8, 32, 56, 56]" = torch.ops.aten.alias.default(relu_3);  relu_3 = None
    alias_314: "f32[8, 32, 56, 56]" = torch.ops.aten.alias.default(alias_313);  alias_313 = None
    le_77: "b8[8, 32, 56, 56]" = torch.ops.aten.le.Scalar(alias_314, 0);  alias_314 = None
    scalar_tensor_77: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_77: "f32[8, 32, 56, 56]" = torch.ops.aten.where.self(le_77, scalar_tensor_77, slice_62);  le_77 = scalar_tensor_77 = slice_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_1312: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze_9, 0);  squeeze_9 = None
    unsqueeze_1313: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1312, 2);  unsqueeze_1312 = None
    unsqueeze_1314: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1313, 3);  unsqueeze_1313 = None
    sum_164: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_77, [0, 2, 3])
    sub_409: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_1314)
    mul_1324: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(where_77, sub_409);  sub_409 = None
    sum_165: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_1324, [0, 2, 3]);  mul_1324 = None
    mul_1325: "f32[32]" = torch.ops.aten.mul.Tensor(sum_164, 3.985969387755102e-05)
    unsqueeze_1315: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1325, 0);  mul_1325 = None
    unsqueeze_1316: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1315, 2);  unsqueeze_1315 = None
    unsqueeze_1317: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1316, 3);  unsqueeze_1316 = None
    mul_1326: "f32[32]" = torch.ops.aten.mul.Tensor(sum_165, 3.985969387755102e-05)
    mul_1327: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_10, squeeze_10)
    mul_1328: "f32[32]" = torch.ops.aten.mul.Tensor(mul_1326, mul_1327);  mul_1326 = mul_1327 = None
    unsqueeze_1318: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1328, 0);  mul_1328 = None
    unsqueeze_1319: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1318, 2);  unsqueeze_1318 = None
    unsqueeze_1320: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1319, 3);  unsqueeze_1319 = None
    mul_1329: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_10, primals_11);  primals_11 = None
    unsqueeze_1321: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1329, 0);  mul_1329 = None
    unsqueeze_1322: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1321, 2);  unsqueeze_1321 = None
    unsqueeze_1323: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1322, 3);  unsqueeze_1322 = None
    sub_410: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_1314);  convolution_3 = unsqueeze_1314 = None
    mul_1330: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(sub_410, unsqueeze_1320);  sub_410 = unsqueeze_1320 = None
    sub_411: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(where_77, mul_1330);  where_77 = mul_1330 = None
    sub_412: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(sub_411, unsqueeze_1317);  sub_411 = unsqueeze_1317 = None
    mul_1331: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(sub_412, unsqueeze_1323);  sub_412 = unsqueeze_1323 = None
    mul_1332: "f32[32]" = torch.ops.aten.mul.Tensor(sum_165, squeeze_10);  sum_165 = squeeze_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_81 = torch.ops.aten.convolution_backward.default(mul_1331, getitem_17, primals_10, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_1331 = getitem_17 = primals_10 = None
    getitem_735: "f32[8, 32, 56, 56]" = convolution_backward_81[0]
    getitem_736: "f32[32, 4, 3, 3]" = convolution_backward_81[1];  convolution_backward_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_316: "f32[8, 32, 56, 56]" = torch.ops.aten.alias.default(relu_2);  relu_2 = None
    alias_317: "f32[8, 32, 56, 56]" = torch.ops.aten.alias.default(alias_316);  alias_316 = None
    le_78: "b8[8, 32, 56, 56]" = torch.ops.aten.le.Scalar(alias_317, 0);  alias_317 = None
    scalar_tensor_78: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_78: "f32[8, 32, 56, 56]" = torch.ops.aten.where.self(le_78, scalar_tensor_78, slice_61);  le_78 = scalar_tensor_78 = slice_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_1324: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze_6, 0);  squeeze_6 = None
    unsqueeze_1325: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1324, 2);  unsqueeze_1324 = None
    unsqueeze_1326: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1325, 3);  unsqueeze_1325 = None
    sum_166: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_78, [0, 2, 3])
    sub_413: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_1326)
    mul_1333: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(where_78, sub_413);  sub_413 = None
    sum_167: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_1333, [0, 2, 3]);  mul_1333 = None
    mul_1334: "f32[32]" = torch.ops.aten.mul.Tensor(sum_166, 3.985969387755102e-05)
    unsqueeze_1327: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1334, 0);  mul_1334 = None
    unsqueeze_1328: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1327, 2);  unsqueeze_1327 = None
    unsqueeze_1329: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1328, 3);  unsqueeze_1328 = None
    mul_1335: "f32[32]" = torch.ops.aten.mul.Tensor(sum_167, 3.985969387755102e-05)
    mul_1336: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
    mul_1337: "f32[32]" = torch.ops.aten.mul.Tensor(mul_1335, mul_1336);  mul_1335 = mul_1336 = None
    unsqueeze_1330: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1337, 0);  mul_1337 = None
    unsqueeze_1331: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1330, 2);  unsqueeze_1330 = None
    unsqueeze_1332: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1331, 3);  unsqueeze_1331 = None
    mul_1338: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_7, primals_8);  primals_8 = None
    unsqueeze_1333: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1338, 0);  mul_1338 = None
    unsqueeze_1334: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1333, 2);  unsqueeze_1333 = None
    unsqueeze_1335: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1334, 3);  unsqueeze_1334 = None
    sub_414: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_1326);  convolution_2 = unsqueeze_1326 = None
    mul_1339: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(sub_414, unsqueeze_1332);  sub_414 = unsqueeze_1332 = None
    sub_415: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(where_78, mul_1339);  where_78 = mul_1339 = None
    sub_416: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(sub_415, unsqueeze_1329);  sub_415 = unsqueeze_1329 = None
    mul_1340: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(sub_416, unsqueeze_1335);  sub_416 = unsqueeze_1335 = None
    mul_1341: "f32[32]" = torch.ops.aten.mul.Tensor(sum_167, squeeze_7);  sum_167 = squeeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_backward_82 = torch.ops.aten.convolution_backward.default(mul_1340, getitem_10, primals_7, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_1340 = getitem_10 = primals_7 = None
    getitem_738: "f32[8, 32, 56, 56]" = convolution_backward_82[0]
    getitem_739: "f32[32, 4, 3, 3]" = convolution_backward_82[1];  convolution_backward_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:85, code: spx = torch.split(out, self.width, 1)
    cat_31: "f32[8, 128, 56, 56]" = torch.ops.aten.cat.default([getitem_738, getitem_735, getitem_732, avg_pool2d_backward_3], 1);  getitem_738 = getitem_735 = getitem_732 = avg_pool2d_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    alias_319: "f32[8, 128, 56, 56]" = torch.ops.aten.alias.default(relu_1);  relu_1 = None
    alias_320: "f32[8, 128, 56, 56]" = torch.ops.aten.alias.default(alias_319);  alias_319 = None
    le_79: "b8[8, 128, 56, 56]" = torch.ops.aten.le.Scalar(alias_320, 0);  alias_320 = None
    scalar_tensor_79: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_79: "f32[8, 128, 56, 56]" = torch.ops.aten.where.self(le_79, scalar_tensor_79, cat_31);  le_79 = scalar_tensor_79 = cat_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    unsqueeze_1336: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_3, 0);  squeeze_3 = None
    unsqueeze_1337: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1336, 2);  unsqueeze_1336 = None
    unsqueeze_1338: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1337, 3);  unsqueeze_1337 = None
    sum_168: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_79, [0, 2, 3])
    sub_417: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_1338)
    mul_1342: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(where_79, sub_417);  sub_417 = None
    sum_169: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1342, [0, 2, 3]);  mul_1342 = None
    mul_1343: "f32[128]" = torch.ops.aten.mul.Tensor(sum_168, 3.985969387755102e-05)
    unsqueeze_1339: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1343, 0);  mul_1343 = None
    unsqueeze_1340: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1339, 2);  unsqueeze_1339 = None
    unsqueeze_1341: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1340, 3);  unsqueeze_1340 = None
    mul_1344: "f32[128]" = torch.ops.aten.mul.Tensor(sum_169, 3.985969387755102e-05)
    mul_1345: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
    mul_1346: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1344, mul_1345);  mul_1344 = mul_1345 = None
    unsqueeze_1342: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1346, 0);  mul_1346 = None
    unsqueeze_1343: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1342, 2);  unsqueeze_1342 = None
    unsqueeze_1344: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1343, 3);  unsqueeze_1343 = None
    mul_1347: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_4, primals_5);  primals_5 = None
    unsqueeze_1345: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1347, 0);  mul_1347 = None
    unsqueeze_1346: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1345, 2);  unsqueeze_1345 = None
    unsqueeze_1347: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1346, 3);  unsqueeze_1346 = None
    sub_418: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_1338);  convolution_1 = unsqueeze_1338 = None
    mul_1348: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_418, unsqueeze_1344);  sub_418 = unsqueeze_1344 = None
    sub_419: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(where_79, mul_1348);  where_79 = mul_1348 = None
    sub_420: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(sub_419, unsqueeze_1341);  sub_419 = unsqueeze_1341 = None
    mul_1349: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_420, unsqueeze_1347);  sub_420 = unsqueeze_1347 = None
    mul_1350: "f32[128]" = torch.ops.aten.mul.Tensor(sum_169, squeeze_4);  sum_169 = squeeze_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_backward_83 = torch.ops.aten.convolution_backward.default(mul_1349, getitem_2, primals_4, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1349 = getitem_2 = primals_4 = None
    getitem_741: "f32[8, 64, 56, 56]" = convolution_backward_83[0]
    getitem_742: "f32[128, 64, 1, 1]" = convolution_backward_83[1];  convolution_backward_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    add_504: "f32[8, 64, 56, 56]" = torch.ops.aten.add.Tensor(getitem_726, getitem_741);  getitem_726 = getitem_741 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:523, code: x = self.maxpool(x)
    max_pool2d_with_indices_backward: "f32[8, 64, 112, 112]" = torch.ops.aten.max_pool2d_with_indices_backward.default(add_504, relu, [3, 3], [2, 2], [1, 1], [1, 1], False, getitem_3);  add_504 = getitem_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:522, code: x = self.act1(x)
    alias_322: "f32[8, 64, 112, 112]" = torch.ops.aten.alias.default(relu);  relu = None
    alias_323: "f32[8, 64, 112, 112]" = torch.ops.aten.alias.default(alias_322);  alias_322 = None
    le_80: "b8[8, 64, 112, 112]" = torch.ops.aten.le.Scalar(alias_323, 0);  alias_323 = None
    scalar_tensor_80: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_80: "f32[8, 64, 112, 112]" = torch.ops.aten.where.self(le_80, scalar_tensor_80, max_pool2d_with_indices_backward);  le_80 = scalar_tensor_80 = max_pool2d_with_indices_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:521, code: x = self.bn1(x)
    unsqueeze_1348: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
    unsqueeze_1349: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1348, 2);  unsqueeze_1348 = None
    unsqueeze_1350: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1349, 3);  unsqueeze_1349 = None
    sum_170: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_80, [0, 2, 3])
    sub_421: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_1350)
    mul_1351: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(where_80, sub_421);  sub_421 = None
    sum_171: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1351, [0, 2, 3]);  mul_1351 = None
    mul_1352: "f32[64]" = torch.ops.aten.mul.Tensor(sum_170, 9.964923469387754e-06)
    unsqueeze_1351: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1352, 0);  mul_1352 = None
    unsqueeze_1352: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1351, 2);  unsqueeze_1351 = None
    unsqueeze_1353: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1352, 3);  unsqueeze_1352 = None
    mul_1353: "f32[64]" = torch.ops.aten.mul.Tensor(sum_171, 9.964923469387754e-06)
    mul_1354: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
    mul_1355: "f32[64]" = torch.ops.aten.mul.Tensor(mul_1353, mul_1354);  mul_1353 = mul_1354 = None
    unsqueeze_1354: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1355, 0);  mul_1355 = None
    unsqueeze_1355: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1354, 2);  unsqueeze_1354 = None
    unsqueeze_1356: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1355, 3);  unsqueeze_1355 = None
    mul_1356: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_1, primals_2);  primals_2 = None
    unsqueeze_1357: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1356, 0);  mul_1356 = None
    unsqueeze_1358: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1357, 2);  unsqueeze_1357 = None
    unsqueeze_1359: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1358, 3);  unsqueeze_1358 = None
    sub_422: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_1350);  convolution = unsqueeze_1350 = None
    mul_1357: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_422, unsqueeze_1356);  sub_422 = unsqueeze_1356 = None
    sub_423: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(where_80, mul_1357);  where_80 = mul_1357 = None
    sub_424: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(sub_423, unsqueeze_1353);  sub_423 = unsqueeze_1353 = None
    mul_1358: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_424, unsqueeze_1359);  sub_424 = unsqueeze_1359 = None
    mul_1359: "f32[64]" = torch.ops.aten.mul.Tensor(sum_171, squeeze_1);  sum_171 = squeeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:520, code: x = self.conv1(x)
    convolution_backward_84 = torch.ops.aten.convolution_backward.default(mul_1358, primals_513, primals_1, [0], [2, 2], [3, 3], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_1358 = primals_513 = primals_1 = None
    getitem_745: "f32[64, 3, 7, 7]" = convolution_backward_84[1];  convolution_backward_84 = None
    
    # No stacktrace found for following nodes
    copy_: "f32[64]" = torch.ops.aten.copy_.default(primals_258, add_2);  primals_258 = add_2 = None
    copy__1: "f32[64]" = torch.ops.aten.copy_.default(primals_259, add_3);  primals_259 = add_3 = None
    copy__2: "i64[]" = torch.ops.aten.copy_.default(primals_260, add);  primals_260 = add = None
    copy__3: "f32[128]" = torch.ops.aten.copy_.default(primals_261, add_7);  primals_261 = add_7 = None
    copy__4: "f32[128]" = torch.ops.aten.copy_.default(primals_262, add_8);  primals_262 = add_8 = None
    copy__5: "i64[]" = torch.ops.aten.copy_.default(primals_263, add_5);  primals_263 = add_5 = None
    copy__6: "f32[32]" = torch.ops.aten.copy_.default(primals_264, add_12);  primals_264 = add_12 = None
    copy__7: "f32[32]" = torch.ops.aten.copy_.default(primals_265, add_13);  primals_265 = add_13 = None
    copy__8: "i64[]" = torch.ops.aten.copy_.default(primals_266, add_10);  primals_266 = add_10 = None
    copy__9: "f32[32]" = torch.ops.aten.copy_.default(primals_267, add_17);  primals_267 = add_17 = None
    copy__10: "f32[32]" = torch.ops.aten.copy_.default(primals_268, add_18);  primals_268 = add_18 = None
    copy__11: "i64[]" = torch.ops.aten.copy_.default(primals_269, add_15);  primals_269 = add_15 = None
    copy__12: "f32[32]" = torch.ops.aten.copy_.default(primals_270, add_22);  primals_270 = add_22 = None
    copy__13: "f32[32]" = torch.ops.aten.copy_.default(primals_271, add_23);  primals_271 = add_23 = None
    copy__14: "i64[]" = torch.ops.aten.copy_.default(primals_272, add_20);  primals_272 = add_20 = None
    copy__15: "f32[256]" = torch.ops.aten.copy_.default(primals_273, add_27);  primals_273 = add_27 = None
    copy__16: "f32[256]" = torch.ops.aten.copy_.default(primals_274, add_28);  primals_274 = add_28 = None
    copy__17: "i64[]" = torch.ops.aten.copy_.default(primals_275, add_25);  primals_275 = add_25 = None
    copy__18: "f32[256]" = torch.ops.aten.copy_.default(primals_276, add_32);  primals_276 = add_32 = None
    copy__19: "f32[256]" = torch.ops.aten.copy_.default(primals_277, add_33);  primals_277 = add_33 = None
    copy__20: "i64[]" = torch.ops.aten.copy_.default(primals_278, add_30);  primals_278 = add_30 = None
    copy__21: "f32[128]" = torch.ops.aten.copy_.default(primals_279, add_38);  primals_279 = add_38 = None
    copy__22: "f32[128]" = torch.ops.aten.copy_.default(primals_280, add_39);  primals_280 = add_39 = None
    copy__23: "i64[]" = torch.ops.aten.copy_.default(primals_281, add_36);  primals_281 = add_36 = None
    copy__24: "f32[32]" = torch.ops.aten.copy_.default(primals_282, add_43);  primals_282 = add_43 = None
    copy__25: "f32[32]" = torch.ops.aten.copy_.default(primals_283, add_44);  primals_283 = add_44 = None
    copy__26: "i64[]" = torch.ops.aten.copy_.default(primals_284, add_41);  primals_284 = add_41 = None
    copy__27: "f32[32]" = torch.ops.aten.copy_.default(primals_285, add_49);  primals_285 = add_49 = None
    copy__28: "f32[32]" = torch.ops.aten.copy_.default(primals_286, add_50);  primals_286 = add_50 = None
    copy__29: "i64[]" = torch.ops.aten.copy_.default(primals_287, add_47);  primals_287 = add_47 = None
    copy__30: "f32[32]" = torch.ops.aten.copy_.default(primals_288, add_55);  primals_288 = add_55 = None
    copy__31: "f32[32]" = torch.ops.aten.copy_.default(primals_289, add_56);  primals_289 = add_56 = None
    copy__32: "i64[]" = torch.ops.aten.copy_.default(primals_290, add_53);  primals_290 = add_53 = None
    copy__33: "f32[256]" = torch.ops.aten.copy_.default(primals_291, add_60);  primals_291 = add_60 = None
    copy__34: "f32[256]" = torch.ops.aten.copy_.default(primals_292, add_61);  primals_292 = add_61 = None
    copy__35: "i64[]" = torch.ops.aten.copy_.default(primals_293, add_58);  primals_293 = add_58 = None
    copy__36: "f32[128]" = torch.ops.aten.copy_.default(primals_294, add_66);  primals_294 = add_66 = None
    copy__37: "f32[128]" = torch.ops.aten.copy_.default(primals_295, add_67);  primals_295 = add_67 = None
    copy__38: "i64[]" = torch.ops.aten.copy_.default(primals_296, add_64);  primals_296 = add_64 = None
    copy__39: "f32[32]" = torch.ops.aten.copy_.default(primals_297, add_71);  primals_297 = add_71 = None
    copy__40: "f32[32]" = torch.ops.aten.copy_.default(primals_298, add_72);  primals_298 = add_72 = None
    copy__41: "i64[]" = torch.ops.aten.copy_.default(primals_299, add_69);  primals_299 = add_69 = None
    copy__42: "f32[32]" = torch.ops.aten.copy_.default(primals_300, add_77);  primals_300 = add_77 = None
    copy__43: "f32[32]" = torch.ops.aten.copy_.default(primals_301, add_78);  primals_301 = add_78 = None
    copy__44: "i64[]" = torch.ops.aten.copy_.default(primals_302, add_75);  primals_302 = add_75 = None
    copy__45: "f32[32]" = torch.ops.aten.copy_.default(primals_303, add_83);  primals_303 = add_83 = None
    copy__46: "f32[32]" = torch.ops.aten.copy_.default(primals_304, add_84);  primals_304 = add_84 = None
    copy__47: "i64[]" = torch.ops.aten.copy_.default(primals_305, add_81);  primals_305 = add_81 = None
    copy__48: "f32[256]" = torch.ops.aten.copy_.default(primals_306, add_88);  primals_306 = add_88 = None
    copy__49: "f32[256]" = torch.ops.aten.copy_.default(primals_307, add_89);  primals_307 = add_89 = None
    copy__50: "i64[]" = torch.ops.aten.copy_.default(primals_308, add_86);  primals_308 = add_86 = None
    copy__51: "f32[256]" = torch.ops.aten.copy_.default(primals_309, add_94);  primals_309 = add_94 = None
    copy__52: "f32[256]" = torch.ops.aten.copy_.default(primals_310, add_95);  primals_310 = add_95 = None
    copy__53: "i64[]" = torch.ops.aten.copy_.default(primals_311, add_92);  primals_311 = add_92 = None
    copy__54: "f32[64]" = torch.ops.aten.copy_.default(primals_312, add_99);  primals_312 = add_99 = None
    copy__55: "f32[64]" = torch.ops.aten.copy_.default(primals_313, add_100);  primals_313 = add_100 = None
    copy__56: "i64[]" = torch.ops.aten.copy_.default(primals_314, add_97);  primals_314 = add_97 = None
    copy__57: "f32[64]" = torch.ops.aten.copy_.default(primals_315, add_104);  primals_315 = add_104 = None
    copy__58: "f32[64]" = torch.ops.aten.copy_.default(primals_316, add_105);  primals_316 = add_105 = None
    copy__59: "i64[]" = torch.ops.aten.copy_.default(primals_317, add_102);  primals_317 = add_102 = None
    copy__60: "f32[64]" = torch.ops.aten.copy_.default(primals_318, add_109);  primals_318 = add_109 = None
    copy__61: "f32[64]" = torch.ops.aten.copy_.default(primals_319, add_110);  primals_319 = add_110 = None
    copy__62: "i64[]" = torch.ops.aten.copy_.default(primals_320, add_107);  primals_320 = add_107 = None
    copy__63: "f32[512]" = torch.ops.aten.copy_.default(primals_321, add_114);  primals_321 = add_114 = None
    copy__64: "f32[512]" = torch.ops.aten.copy_.default(primals_322, add_115);  primals_322 = add_115 = None
    copy__65: "i64[]" = torch.ops.aten.copy_.default(primals_323, add_112);  primals_323 = add_112 = None
    copy__66: "f32[512]" = torch.ops.aten.copy_.default(primals_324, add_119);  primals_324 = add_119 = None
    copy__67: "f32[512]" = torch.ops.aten.copy_.default(primals_325, add_120);  primals_325 = add_120 = None
    copy__68: "i64[]" = torch.ops.aten.copy_.default(primals_326, add_117);  primals_326 = add_117 = None
    copy__69: "f32[256]" = torch.ops.aten.copy_.default(primals_327, add_125);  primals_327 = add_125 = None
    copy__70: "f32[256]" = torch.ops.aten.copy_.default(primals_328, add_126);  primals_328 = add_126 = None
    copy__71: "i64[]" = torch.ops.aten.copy_.default(primals_329, add_123);  primals_329 = add_123 = None
    copy__72: "f32[64]" = torch.ops.aten.copy_.default(primals_330, add_130);  primals_330 = add_130 = None
    copy__73: "f32[64]" = torch.ops.aten.copy_.default(primals_331, add_131);  primals_331 = add_131 = None
    copy__74: "i64[]" = torch.ops.aten.copy_.default(primals_332, add_128);  primals_332 = add_128 = None
    copy__75: "f32[64]" = torch.ops.aten.copy_.default(primals_333, add_136);  primals_333 = add_136 = None
    copy__76: "f32[64]" = torch.ops.aten.copy_.default(primals_334, add_137);  primals_334 = add_137 = None
    copy__77: "i64[]" = torch.ops.aten.copy_.default(primals_335, add_134);  primals_335 = add_134 = None
    copy__78: "f32[64]" = torch.ops.aten.copy_.default(primals_336, add_142);  primals_336 = add_142 = None
    copy__79: "f32[64]" = torch.ops.aten.copy_.default(primals_337, add_143);  primals_337 = add_143 = None
    copy__80: "i64[]" = torch.ops.aten.copy_.default(primals_338, add_140);  primals_338 = add_140 = None
    copy__81: "f32[512]" = torch.ops.aten.copy_.default(primals_339, add_147);  primals_339 = add_147 = None
    copy__82: "f32[512]" = torch.ops.aten.copy_.default(primals_340, add_148);  primals_340 = add_148 = None
    copy__83: "i64[]" = torch.ops.aten.copy_.default(primals_341, add_145);  primals_341 = add_145 = None
    copy__84: "f32[256]" = torch.ops.aten.copy_.default(primals_342, add_153);  primals_342 = add_153 = None
    copy__85: "f32[256]" = torch.ops.aten.copy_.default(primals_343, add_154);  primals_343 = add_154 = None
    copy__86: "i64[]" = torch.ops.aten.copy_.default(primals_344, add_151);  primals_344 = add_151 = None
    copy__87: "f32[64]" = torch.ops.aten.copy_.default(primals_345, add_158);  primals_345 = add_158 = None
    copy__88: "f32[64]" = torch.ops.aten.copy_.default(primals_346, add_159);  primals_346 = add_159 = None
    copy__89: "i64[]" = torch.ops.aten.copy_.default(primals_347, add_156);  primals_347 = add_156 = None
    copy__90: "f32[64]" = torch.ops.aten.copy_.default(primals_348, add_164);  primals_348 = add_164 = None
    copy__91: "f32[64]" = torch.ops.aten.copy_.default(primals_349, add_165);  primals_349 = add_165 = None
    copy__92: "i64[]" = torch.ops.aten.copy_.default(primals_350, add_162);  primals_350 = add_162 = None
    copy__93: "f32[64]" = torch.ops.aten.copy_.default(primals_351, add_170);  primals_351 = add_170 = None
    copy__94: "f32[64]" = torch.ops.aten.copy_.default(primals_352, add_171);  primals_352 = add_171 = None
    copy__95: "i64[]" = torch.ops.aten.copy_.default(primals_353, add_168);  primals_353 = add_168 = None
    copy__96: "f32[512]" = torch.ops.aten.copy_.default(primals_354, add_175);  primals_354 = add_175 = None
    copy__97: "f32[512]" = torch.ops.aten.copy_.default(primals_355, add_176);  primals_355 = add_176 = None
    copy__98: "i64[]" = torch.ops.aten.copy_.default(primals_356, add_173);  primals_356 = add_173 = None
    copy__99: "f32[256]" = torch.ops.aten.copy_.default(primals_357, add_181);  primals_357 = add_181 = None
    copy__100: "f32[256]" = torch.ops.aten.copy_.default(primals_358, add_182);  primals_358 = add_182 = None
    copy__101: "i64[]" = torch.ops.aten.copy_.default(primals_359, add_179);  primals_359 = add_179 = None
    copy__102: "f32[64]" = torch.ops.aten.copy_.default(primals_360, add_186);  primals_360 = add_186 = None
    copy__103: "f32[64]" = torch.ops.aten.copy_.default(primals_361, add_187);  primals_361 = add_187 = None
    copy__104: "i64[]" = torch.ops.aten.copy_.default(primals_362, add_184);  primals_362 = add_184 = None
    copy__105: "f32[64]" = torch.ops.aten.copy_.default(primals_363, add_192);  primals_363 = add_192 = None
    copy__106: "f32[64]" = torch.ops.aten.copy_.default(primals_364, add_193);  primals_364 = add_193 = None
    copy__107: "i64[]" = torch.ops.aten.copy_.default(primals_365, add_190);  primals_365 = add_190 = None
    copy__108: "f32[64]" = torch.ops.aten.copy_.default(primals_366, add_198);  primals_366 = add_198 = None
    copy__109: "f32[64]" = torch.ops.aten.copy_.default(primals_367, add_199);  primals_367 = add_199 = None
    copy__110: "i64[]" = torch.ops.aten.copy_.default(primals_368, add_196);  primals_368 = add_196 = None
    copy__111: "f32[512]" = torch.ops.aten.copy_.default(primals_369, add_203);  primals_369 = add_203 = None
    copy__112: "f32[512]" = torch.ops.aten.copy_.default(primals_370, add_204);  primals_370 = add_204 = None
    copy__113: "i64[]" = torch.ops.aten.copy_.default(primals_371, add_201);  primals_371 = add_201 = None
    copy__114: "f32[512]" = torch.ops.aten.copy_.default(primals_372, add_209);  primals_372 = add_209 = None
    copy__115: "f32[512]" = torch.ops.aten.copy_.default(primals_373, add_210);  primals_373 = add_210 = None
    copy__116: "i64[]" = torch.ops.aten.copy_.default(primals_374, add_207);  primals_374 = add_207 = None
    copy__117: "f32[128]" = torch.ops.aten.copy_.default(primals_375, add_214);  primals_375 = add_214 = None
    copy__118: "f32[128]" = torch.ops.aten.copy_.default(primals_376, add_215);  primals_376 = add_215 = None
    copy__119: "i64[]" = torch.ops.aten.copy_.default(primals_377, add_212);  primals_377 = add_212 = None
    copy__120: "f32[128]" = torch.ops.aten.copy_.default(primals_378, add_219);  primals_378 = add_219 = None
    copy__121: "f32[128]" = torch.ops.aten.copy_.default(primals_379, add_220);  primals_379 = add_220 = None
    copy__122: "i64[]" = torch.ops.aten.copy_.default(primals_380, add_217);  primals_380 = add_217 = None
    copy__123: "f32[128]" = torch.ops.aten.copy_.default(primals_381, add_224);  primals_381 = add_224 = None
    copy__124: "f32[128]" = torch.ops.aten.copy_.default(primals_382, add_225);  primals_382 = add_225 = None
    copy__125: "i64[]" = torch.ops.aten.copy_.default(primals_383, add_222);  primals_383 = add_222 = None
    copy__126: "f32[1024]" = torch.ops.aten.copy_.default(primals_384, add_229);  primals_384 = add_229 = None
    copy__127: "f32[1024]" = torch.ops.aten.copy_.default(primals_385, add_230);  primals_385 = add_230 = None
    copy__128: "i64[]" = torch.ops.aten.copy_.default(primals_386, add_227);  primals_386 = add_227 = None
    copy__129: "f32[1024]" = torch.ops.aten.copy_.default(primals_387, add_234);  primals_387 = add_234 = None
    copy__130: "f32[1024]" = torch.ops.aten.copy_.default(primals_388, add_235);  primals_388 = add_235 = None
    copy__131: "i64[]" = torch.ops.aten.copy_.default(primals_389, add_232);  primals_389 = add_232 = None
    copy__132: "f32[512]" = torch.ops.aten.copy_.default(primals_390, add_240);  primals_390 = add_240 = None
    copy__133: "f32[512]" = torch.ops.aten.copy_.default(primals_391, add_241);  primals_391 = add_241 = None
    copy__134: "i64[]" = torch.ops.aten.copy_.default(primals_392, add_238);  primals_392 = add_238 = None
    copy__135: "f32[128]" = torch.ops.aten.copy_.default(primals_393, add_245);  primals_393 = add_245 = None
    copy__136: "f32[128]" = torch.ops.aten.copy_.default(primals_394, add_246);  primals_394 = add_246 = None
    copy__137: "i64[]" = torch.ops.aten.copy_.default(primals_395, add_243);  primals_395 = add_243 = None
    copy__138: "f32[128]" = torch.ops.aten.copy_.default(primals_396, add_251);  primals_396 = add_251 = None
    copy__139: "f32[128]" = torch.ops.aten.copy_.default(primals_397, add_252);  primals_397 = add_252 = None
    copy__140: "i64[]" = torch.ops.aten.copy_.default(primals_398, add_249);  primals_398 = add_249 = None
    copy__141: "f32[128]" = torch.ops.aten.copy_.default(primals_399, add_257);  primals_399 = add_257 = None
    copy__142: "f32[128]" = torch.ops.aten.copy_.default(primals_400, add_258);  primals_400 = add_258 = None
    copy__143: "i64[]" = torch.ops.aten.copy_.default(primals_401, add_255);  primals_401 = add_255 = None
    copy__144: "f32[1024]" = torch.ops.aten.copy_.default(primals_402, add_262);  primals_402 = add_262 = None
    copy__145: "f32[1024]" = torch.ops.aten.copy_.default(primals_403, add_263);  primals_403 = add_263 = None
    copy__146: "i64[]" = torch.ops.aten.copy_.default(primals_404, add_260);  primals_404 = add_260 = None
    copy__147: "f32[512]" = torch.ops.aten.copy_.default(primals_405, add_268);  primals_405 = add_268 = None
    copy__148: "f32[512]" = torch.ops.aten.copy_.default(primals_406, add_269);  primals_406 = add_269 = None
    copy__149: "i64[]" = torch.ops.aten.copy_.default(primals_407, add_266);  primals_407 = add_266 = None
    copy__150: "f32[128]" = torch.ops.aten.copy_.default(primals_408, add_273);  primals_408 = add_273 = None
    copy__151: "f32[128]" = torch.ops.aten.copy_.default(primals_409, add_274);  primals_409 = add_274 = None
    copy__152: "i64[]" = torch.ops.aten.copy_.default(primals_410, add_271);  primals_410 = add_271 = None
    copy__153: "f32[128]" = torch.ops.aten.copy_.default(primals_411, add_279);  primals_411 = add_279 = None
    copy__154: "f32[128]" = torch.ops.aten.copy_.default(primals_412, add_280);  primals_412 = add_280 = None
    copy__155: "i64[]" = torch.ops.aten.copy_.default(primals_413, add_277);  primals_413 = add_277 = None
    copy__156: "f32[128]" = torch.ops.aten.copy_.default(primals_414, add_285);  primals_414 = add_285 = None
    copy__157: "f32[128]" = torch.ops.aten.copy_.default(primals_415, add_286);  primals_415 = add_286 = None
    copy__158: "i64[]" = torch.ops.aten.copy_.default(primals_416, add_283);  primals_416 = add_283 = None
    copy__159: "f32[1024]" = torch.ops.aten.copy_.default(primals_417, add_290);  primals_417 = add_290 = None
    copy__160: "f32[1024]" = torch.ops.aten.copy_.default(primals_418, add_291);  primals_418 = add_291 = None
    copy__161: "i64[]" = torch.ops.aten.copy_.default(primals_419, add_288);  primals_419 = add_288 = None
    copy__162: "f32[512]" = torch.ops.aten.copy_.default(primals_420, add_296);  primals_420 = add_296 = None
    copy__163: "f32[512]" = torch.ops.aten.copy_.default(primals_421, add_297);  primals_421 = add_297 = None
    copy__164: "i64[]" = torch.ops.aten.copy_.default(primals_422, add_294);  primals_422 = add_294 = None
    copy__165: "f32[128]" = torch.ops.aten.copy_.default(primals_423, add_301);  primals_423 = add_301 = None
    copy__166: "f32[128]" = torch.ops.aten.copy_.default(primals_424, add_302);  primals_424 = add_302 = None
    copy__167: "i64[]" = torch.ops.aten.copy_.default(primals_425, add_299);  primals_425 = add_299 = None
    copy__168: "f32[128]" = torch.ops.aten.copy_.default(primals_426, add_307);  primals_426 = add_307 = None
    copy__169: "f32[128]" = torch.ops.aten.copy_.default(primals_427, add_308);  primals_427 = add_308 = None
    copy__170: "i64[]" = torch.ops.aten.copy_.default(primals_428, add_305);  primals_428 = add_305 = None
    copy__171: "f32[128]" = torch.ops.aten.copy_.default(primals_429, add_313);  primals_429 = add_313 = None
    copy__172: "f32[128]" = torch.ops.aten.copy_.default(primals_430, add_314);  primals_430 = add_314 = None
    copy__173: "i64[]" = torch.ops.aten.copy_.default(primals_431, add_311);  primals_431 = add_311 = None
    copy__174: "f32[1024]" = torch.ops.aten.copy_.default(primals_432, add_318);  primals_432 = add_318 = None
    copy__175: "f32[1024]" = torch.ops.aten.copy_.default(primals_433, add_319);  primals_433 = add_319 = None
    copy__176: "i64[]" = torch.ops.aten.copy_.default(primals_434, add_316);  primals_434 = add_316 = None
    copy__177: "f32[512]" = torch.ops.aten.copy_.default(primals_435, add_324);  primals_435 = add_324 = None
    copy__178: "f32[512]" = torch.ops.aten.copy_.default(primals_436, add_325);  primals_436 = add_325 = None
    copy__179: "i64[]" = torch.ops.aten.copy_.default(primals_437, add_322);  primals_437 = add_322 = None
    copy__180: "f32[128]" = torch.ops.aten.copy_.default(primals_438, add_329);  primals_438 = add_329 = None
    copy__181: "f32[128]" = torch.ops.aten.copy_.default(primals_439, add_330);  primals_439 = add_330 = None
    copy__182: "i64[]" = torch.ops.aten.copy_.default(primals_440, add_327);  primals_440 = add_327 = None
    copy__183: "f32[128]" = torch.ops.aten.copy_.default(primals_441, add_335);  primals_441 = add_335 = None
    copy__184: "f32[128]" = torch.ops.aten.copy_.default(primals_442, add_336);  primals_442 = add_336 = None
    copy__185: "i64[]" = torch.ops.aten.copy_.default(primals_443, add_333);  primals_443 = add_333 = None
    copy__186: "f32[128]" = torch.ops.aten.copy_.default(primals_444, add_341);  primals_444 = add_341 = None
    copy__187: "f32[128]" = torch.ops.aten.copy_.default(primals_445, add_342);  primals_445 = add_342 = None
    copy__188: "i64[]" = torch.ops.aten.copy_.default(primals_446, add_339);  primals_446 = add_339 = None
    copy__189: "f32[1024]" = torch.ops.aten.copy_.default(primals_447, add_346);  primals_447 = add_346 = None
    copy__190: "f32[1024]" = torch.ops.aten.copy_.default(primals_448, add_347);  primals_448 = add_347 = None
    copy__191: "i64[]" = torch.ops.aten.copy_.default(primals_449, add_344);  primals_449 = add_344 = None
    copy__192: "f32[512]" = torch.ops.aten.copy_.default(primals_450, add_352);  primals_450 = add_352 = None
    copy__193: "f32[512]" = torch.ops.aten.copy_.default(primals_451, add_353);  primals_451 = add_353 = None
    copy__194: "i64[]" = torch.ops.aten.copy_.default(primals_452, add_350);  primals_452 = add_350 = None
    copy__195: "f32[128]" = torch.ops.aten.copy_.default(primals_453, add_357);  primals_453 = add_357 = None
    copy__196: "f32[128]" = torch.ops.aten.copy_.default(primals_454, add_358);  primals_454 = add_358 = None
    copy__197: "i64[]" = torch.ops.aten.copy_.default(primals_455, add_355);  primals_455 = add_355 = None
    copy__198: "f32[128]" = torch.ops.aten.copy_.default(primals_456, add_363);  primals_456 = add_363 = None
    copy__199: "f32[128]" = torch.ops.aten.copy_.default(primals_457, add_364);  primals_457 = add_364 = None
    copy__200: "i64[]" = torch.ops.aten.copy_.default(primals_458, add_361);  primals_458 = add_361 = None
    copy__201: "f32[128]" = torch.ops.aten.copy_.default(primals_459, add_369);  primals_459 = add_369 = None
    copy__202: "f32[128]" = torch.ops.aten.copy_.default(primals_460, add_370);  primals_460 = add_370 = None
    copy__203: "i64[]" = torch.ops.aten.copy_.default(primals_461, add_367);  primals_461 = add_367 = None
    copy__204: "f32[1024]" = torch.ops.aten.copy_.default(primals_462, add_374);  primals_462 = add_374 = None
    copy__205: "f32[1024]" = torch.ops.aten.copy_.default(primals_463, add_375);  primals_463 = add_375 = None
    copy__206: "i64[]" = torch.ops.aten.copy_.default(primals_464, add_372);  primals_464 = add_372 = None
    copy__207: "f32[1024]" = torch.ops.aten.copy_.default(primals_465, add_380);  primals_465 = add_380 = None
    copy__208: "f32[1024]" = torch.ops.aten.copy_.default(primals_466, add_381);  primals_466 = add_381 = None
    copy__209: "i64[]" = torch.ops.aten.copy_.default(primals_467, add_378);  primals_467 = add_378 = None
    copy__210: "f32[256]" = torch.ops.aten.copy_.default(primals_468, add_385);  primals_468 = add_385 = None
    copy__211: "f32[256]" = torch.ops.aten.copy_.default(primals_469, add_386);  primals_469 = add_386 = None
    copy__212: "i64[]" = torch.ops.aten.copy_.default(primals_470, add_383);  primals_470 = add_383 = None
    copy__213: "f32[256]" = torch.ops.aten.copy_.default(primals_471, add_390);  primals_471 = add_390 = None
    copy__214: "f32[256]" = torch.ops.aten.copy_.default(primals_472, add_391);  primals_472 = add_391 = None
    copy__215: "i64[]" = torch.ops.aten.copy_.default(primals_473, add_388);  primals_473 = add_388 = None
    copy__216: "f32[256]" = torch.ops.aten.copy_.default(primals_474, add_395);  primals_474 = add_395 = None
    copy__217: "f32[256]" = torch.ops.aten.copy_.default(primals_475, add_396);  primals_475 = add_396 = None
    copy__218: "i64[]" = torch.ops.aten.copy_.default(primals_476, add_393);  primals_476 = add_393 = None
    copy__219: "f32[2048]" = torch.ops.aten.copy_.default(primals_477, add_400);  primals_477 = add_400 = None
    copy__220: "f32[2048]" = torch.ops.aten.copy_.default(primals_478, add_401);  primals_478 = add_401 = None
    copy__221: "i64[]" = torch.ops.aten.copy_.default(primals_479, add_398);  primals_479 = add_398 = None
    copy__222: "f32[2048]" = torch.ops.aten.copy_.default(primals_480, add_405);  primals_480 = add_405 = None
    copy__223: "f32[2048]" = torch.ops.aten.copy_.default(primals_481, add_406);  primals_481 = add_406 = None
    copy__224: "i64[]" = torch.ops.aten.copy_.default(primals_482, add_403);  primals_482 = add_403 = None
    copy__225: "f32[1024]" = torch.ops.aten.copy_.default(primals_483, add_411);  primals_483 = add_411 = None
    copy__226: "f32[1024]" = torch.ops.aten.copy_.default(primals_484, add_412);  primals_484 = add_412 = None
    copy__227: "i64[]" = torch.ops.aten.copy_.default(primals_485, add_409);  primals_485 = add_409 = None
    copy__228: "f32[256]" = torch.ops.aten.copy_.default(primals_486, add_416);  primals_486 = add_416 = None
    copy__229: "f32[256]" = torch.ops.aten.copy_.default(primals_487, add_417);  primals_487 = add_417 = None
    copy__230: "i64[]" = torch.ops.aten.copy_.default(primals_488, add_414);  primals_488 = add_414 = None
    copy__231: "f32[256]" = torch.ops.aten.copy_.default(primals_489, add_422);  primals_489 = add_422 = None
    copy__232: "f32[256]" = torch.ops.aten.copy_.default(primals_490, add_423);  primals_490 = add_423 = None
    copy__233: "i64[]" = torch.ops.aten.copy_.default(primals_491, add_420);  primals_491 = add_420 = None
    copy__234: "f32[256]" = torch.ops.aten.copy_.default(primals_492, add_428);  primals_492 = add_428 = None
    copy__235: "f32[256]" = torch.ops.aten.copy_.default(primals_493, add_429);  primals_493 = add_429 = None
    copy__236: "i64[]" = torch.ops.aten.copy_.default(primals_494, add_426);  primals_494 = add_426 = None
    copy__237: "f32[2048]" = torch.ops.aten.copy_.default(primals_495, add_433);  primals_495 = add_433 = None
    copy__238: "f32[2048]" = torch.ops.aten.copy_.default(primals_496, add_434);  primals_496 = add_434 = None
    copy__239: "i64[]" = torch.ops.aten.copy_.default(primals_497, add_431);  primals_497 = add_431 = None
    copy__240: "f32[1024]" = torch.ops.aten.copy_.default(primals_498, add_439);  primals_498 = add_439 = None
    copy__241: "f32[1024]" = torch.ops.aten.copy_.default(primals_499, add_440);  primals_499 = add_440 = None
    copy__242: "i64[]" = torch.ops.aten.copy_.default(primals_500, add_437);  primals_500 = add_437 = None
    copy__243: "f32[256]" = torch.ops.aten.copy_.default(primals_501, add_444);  primals_501 = add_444 = None
    copy__244: "f32[256]" = torch.ops.aten.copy_.default(primals_502, add_445);  primals_502 = add_445 = None
    copy__245: "i64[]" = torch.ops.aten.copy_.default(primals_503, add_442);  primals_503 = add_442 = None
    copy__246: "f32[256]" = torch.ops.aten.copy_.default(primals_504, add_450);  primals_504 = add_450 = None
    copy__247: "f32[256]" = torch.ops.aten.copy_.default(primals_505, add_451);  primals_505 = add_451 = None
    copy__248: "i64[]" = torch.ops.aten.copy_.default(primals_506, add_448);  primals_506 = add_448 = None
    copy__249: "f32[256]" = torch.ops.aten.copy_.default(primals_507, add_456);  primals_507 = add_456 = None
    copy__250: "f32[256]" = torch.ops.aten.copy_.default(primals_508, add_457);  primals_508 = add_457 = None
    copy__251: "i64[]" = torch.ops.aten.copy_.default(primals_509, add_454);  primals_509 = add_454 = None
    copy__252: "f32[2048]" = torch.ops.aten.copy_.default(primals_510, add_461);  primals_510 = add_461 = None
    copy__253: "f32[2048]" = torch.ops.aten.copy_.default(primals_511, add_462);  primals_511 = add_462 = None
    copy__254: "i64[]" = torch.ops.aten.copy_.default(primals_512, add_459);  primals_512 = add_459 = None
    return pytree.tree_unflatten([addmm, getitem_745, mul_1359, sum_170, getitem_742, mul_1350, sum_168, getitem_739, mul_1341, sum_166, getitem_736, mul_1332, sum_164, getitem_733, mul_1323, sum_162, getitem_730, mul_1314, sum_160, getitem_727, mul_1305, sum_158, getitem_724, mul_1296, sum_156, getitem_721, mul_1287, sum_154, getitem_718, mul_1278, sum_152, getitem_715, mul_1269, sum_150, getitem_712, mul_1260, sum_148, getitem_709, mul_1251, sum_146, getitem_706, mul_1242, sum_144, getitem_703, mul_1233, sum_142, getitem_700, mul_1224, sum_140, getitem_697, mul_1215, sum_138, getitem_694, mul_1206, sum_136, getitem_691, mul_1197, sum_134, getitem_688, mul_1188, sum_132, getitem_685, mul_1179, sum_130, getitem_682, mul_1170, sum_128, getitem_679, mul_1161, sum_126, getitem_676, mul_1152, sum_124, getitem_673, mul_1143, sum_122, getitem_670, mul_1134, sum_120, getitem_667, mul_1125, sum_118, getitem_664, mul_1116, sum_116, getitem_661, mul_1107, sum_114, getitem_658, mul_1098, sum_112, getitem_655, mul_1089, sum_110, getitem_652, mul_1080, sum_108, getitem_649, mul_1071, sum_106, getitem_646, mul_1062, sum_104, getitem_643, mul_1053, sum_102, getitem_640, mul_1044, sum_100, getitem_637, mul_1035, sum_98, getitem_634, mul_1026, sum_96, getitem_631, mul_1017, sum_94, getitem_628, mul_1008, sum_92, getitem_625, mul_999, sum_90, getitem_622, mul_990, sum_88, getitem_619, mul_981, sum_86, getitem_616, mul_972, sum_84, getitem_613, mul_963, sum_82, getitem_610, mul_954, sum_80, getitem_607, mul_945, sum_78, getitem_604, mul_936, sum_76, getitem_601, mul_927, sum_74, getitem_598, mul_918, sum_72, getitem_595, mul_909, sum_70, getitem_592, mul_900, sum_68, getitem_589, mul_891, sum_66, getitem_586, mul_882, sum_64, getitem_583, mul_873, sum_62, getitem_580, mul_864, sum_60, getitem_577, mul_855, sum_58, getitem_574, mul_846, sum_56, getitem_571, mul_837, sum_54, getitem_568, mul_828, sum_52, getitem_565, mul_819, sum_50, getitem_562, mul_810, sum_48, getitem_559, mul_801, sum_46, getitem_556, mul_792, sum_44, getitem_553, mul_783, sum_42, getitem_550, mul_774, sum_40, getitem_547, mul_765, sum_38, getitem_544, mul_756, sum_36, getitem_541, mul_747, sum_34, getitem_538, mul_738, sum_32, getitem_535, mul_729, sum_30, getitem_532, mul_720, sum_28, getitem_529, mul_711, sum_26, getitem_526, mul_702, sum_24, getitem_523, mul_693, sum_22, getitem_520, mul_684, sum_20, getitem_517, mul_675, sum_18, getitem_514, mul_666, sum_16, getitem_511, mul_657, sum_14, getitem_508, mul_648, sum_12, getitem_505, mul_639, sum_10, getitem_502, mul_630, sum_8, getitem_499, mul_621, sum_6, getitem_496, mul_612, sum_4, getitem_493, mul_603, sum_2, permute_4, view_1, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None], self._out_spec)
    